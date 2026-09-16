# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Unit tests for the fine-tuned Qwen prediction tool."""

import json
from typing import Any, Dict, List, Optional, get_args
from unittest.mock import MagicMock, patch

import openai
import pytest
import requests

import packages.valory.customs.finetuned_prediction.finetuned_prediction as module
from packages.valory.customs.finetuned_prediction.finetuned_prediction import (
    MODEL_BY_TOOL,
    SERVED_MODEL_FINE_TUNED_CALIBRATED,
    TOOL_BASE,
    TOOL_FINE_TUNED,
    TOOL_FINE_TUNED_CALIBRATED,
    VLLM_ENDPOINT,
    build_forecaster_prompt,
    build_messages,
    canonical_prediction,
    gather_sources,
    parse_p_yes,
    parse_prompt,
    resolve_model,
    run,
    with_key_rotation,
)

MODULE_PATH = "packages.valory.customs.finetuned_prediction.finetuned_prediction"
ENDPOINT = "http://vllm:8000/v1"
WELL_FORMED = (
    "<think>weighing base rates and the sources</think>\n"
    '{"p_yes": 0.73, "p_no": 0.27, "confidence": 0.8, "info_utility": 0.9}'
)
# The served checkpoints emit a BARE closing tag -- the chat template supplies
# the opening one, so it is absent from the completion. The reasoning that
# precedes it contains a DRAFT probability that must not be picked.
BARE_CLOSE = (
    'Draft: {"p_yes": 0.9, "p_no": 0.1} but on reflection lower.\n'
    "</think>\n"
    '{"p_yes": 0.3, "p_no": 0.7, "confidence": 0.6, "info_utility": 0.5}'
)
# Same completion with an UPPERCASE tag. The tags come from the chat template,
# a deployment artifact this tool does not control, so a template emitting
# <THINK> must not silently turn the think guards into no-ops.
BARE_CLOSE_UPPER = BARE_CLOSE.replace("</think>", "</THINK>")
# A budget cut that lands while the answer is being written, where the object
# under construction has already opened AND closed a nested one. The brace that
# closes the nested object is the only "}" after the opener, so asking whether
# ANY closing brace follows reads the cut as complete and delivers the earlier
# draft. Reviewer-reproduced.
CUT_AFTER_NESTED_CLOSE = (
    "</think>\n" '{"p_yes": 0.25, "p_no": 0.75}\n' '{"p_yes": 0.4, "meta": {"a": 1}'
)
# The mirror image: a complete answer followed by an unclosed brace in PROSE.
# It never closes, but it never began an object either, so reading it as a cut
# turns a delivered forecast into a typed error. Reviewer-reproduced.
STRAY_BRACE_AFTER_ANSWER = (
    "</think>\n" '{"p_yes": 0.3, "p_no": 0.7}\n' "Note: see {source for details"
)
# Two closing tags: a draft sits between them and the object after the last one
# is unparseable. Stripping only to the FIRST tag leaves the draft in the
# candidate pool and delivers 0.82 as the answer.
TWO_CLOSING_TAGS = (
    "</think>\n"
    'still weighing {"p_yes": 0.82, "p_no": 0.18}\n'
    "</think>\n"
    '{"p_yes": }'
)


class FakeKeyChain:
    """Minimal stand-in for the task-execution KeyChain object."""

    def __init__(self, keys: Dict[str, str]):
        """Initialise with a service->key mapping."""
        self._keys = dict(keys)
        self.rotated: List[str] = []

    def __getitem__(self, service: str) -> str:
        """Return the key for `service`, raising KeyError when absent."""
        return self._keys[service]  # raises KeyError when absent, like the real one

    def max_retries(self) -> Dict[str, int]:
        """Return one retry per configured service."""
        return {service: 1 for service in self._keys}

    def rotate(self, service: str) -> None:
        """Record a rotation request for `service`."""
        self.rotated.append(service)


# ---------------------------------------------------------------------------
# parse_p_yes / canonical_prediction — vendored-parser parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (WELL_FORMED, 0.73),  # think block stripped
        (BARE_CLOSE, 0.3),  # bare </think>: answer, not the mid-reasoning draft
        ('{"p_yes": 0.4, "p_no": 0.6}', 0.4),  # bare JSON, no think
        ("<think>no json here</think> nothing", None),  # no JSON object
        ('{"p_no": 0.6}', None),  # missing p_yes
        ('{"p_yes": "high"}', None),  # non-numeric p_yes
        ('{"p_yes": null}', None),  # null p_yes -- pins the TypeError arm
        ('{"p_yes": 1.5}', None),  # out of [0, 1]
        ("", None),  # empty
    ],
    ids=[
        "think_block",
        "bare_close_tag",
        "bare_json",
        "no_json",
        "missing",
        "non_numeric",
        "null_p_yes",
        "out_of_range",
        "empty",
    ],
)
def test_parse_p_yes(completion: str, expected: Optional[float]) -> None:
    """parse_p_yes extracts p_yes and rejects malformed / out-of-range outputs."""
    assert parse_p_yes(completion) == expected


def test_canonical_prediction_normalises_schema() -> None:
    """A well-formed completion yields the four-field delivery JSON."""
    result = canonical_prediction(WELL_FORMED)
    assert result is not None
    obj = json.loads(result)
    assert obj == {"p_yes": 0.73, "p_no": 0.27, "confidence": 0.8, "info_utility": 0.9}


def test_canonical_prediction_derives_p_no_and_defaults() -> None:
    # confidence/info_utility absent -> defaulted; p_no derived from p_yes.
    """p_no is derived from p_yes; confidence / info_utility default when absent."""
    result = canonical_prediction('{"p_yes": 0.25}')
    assert result is not None
    obj = json.loads(result)
    assert obj["p_yes"] == 0.25
    assert obj["p_no"] == 0.75
    assert obj["confidence"] == 0.5
    assert obj["info_utility"] == 0.5


def test_canonical_prediction_returns_none_on_malformed() -> None:
    """Unparseable or missing completions yield None."""
    assert canonical_prediction("<think>oops</think> not json") is None
    assert canonical_prediction(None) is None


@pytest.mark.parametrize(
    "completion",
    ['{"p_yes": 1.5}', '{"p_yes": -0.2}', '</think>\n{"p_yes": 1.5, "p_no": -0.5}'],
    ids=["above_one", "below_zero", "after_think_block"],
)
def test_canonical_prediction_rejects_an_out_of_range_p_yes(completion: str) -> None:
    """An out-of-range p_yes yields None, never a derived p_no outside [0, 1]."""
    # canonical_prediction range-checks its own object rather than deferring to
    # parse_p_yes. Without that check the delivery is built anyway: p_yes 1.5
    # gives p_no -0.5, i.e. a NEGATIVE probability answered on-chain.
    assert canonical_prediction(completion) is None


def test_a_revised_forecast_wins_over_the_earlier_one() -> None:
    """With two usable objects after the strip, the LAST is the answer."""
    # A model that revises its estimate emits both. Walking forward would
    # deliver the superseded one.
    completion = (
        '</think>\n{"p_yes": 0.25, "p_no": 0.75}\n'
        'On reflection:\n{"p_yes": 0.9, "p_no": 0.1}'
    )
    assert json.loads(canonical_prediction(completion) or "{}")["p_yes"] == 0.9


def test_a_cut_mid_object_after_the_think_block_is_not_a_draft() -> None:
    """A budget cut while writing the answer must not deliver an earlier draft."""
    # A max_tokens cut lands mid-object and leaves no closing brace. Anything
    # complete before it is a draft, so there is no answer to recover.
    completion = '</think>\n{"p_yes": 0.25, "p_no": 0.75}\n{"p_yes": '
    assert canonical_prediction(completion) is None


def test_a_closed_but_invalid_trailing_object_still_falls_back() -> None:
    """A complete-but-unparseable trailing object is junk, not a cut."""
    # It closes, so the completion was not cut: the earlier forecast stands.
    completion = '</think>\n{"p_yes": 0.25, "p_no": 0.75}\n{"p_yes": }'
    assert json.loads(canonical_prediction(completion) or "{}")["p_yes"] == 0.25


def test_bare_closing_tag_discards_the_reasoning_draft() -> None:
    """A draft written before a bare </think> must not win over the answer."""
    assert json.loads(canonical_prediction(BARE_CLOSE) or "{}")["p_yes"] == 0.3


def test_an_uppercase_bare_closing_tag_discards_the_reasoning_draft() -> None:
    """An uppercase </THINK> must strip the reasoning just like the lower one."""
    assert json.loads(canonical_prediction(BARE_CLOSE_UPPER) or "{}")["p_yes"] == 0.3
    # The draft parses and the post-tag answer does not: only a case-insensitive
    # strip keeps the draft out of the candidate pool.
    draft_only = 'Draft: {"p_yes": 0.9, "p_no": 0.1}\n</THINK>\n{"p_yes": }'
    assert canonical_prediction(draft_only) is None


def test_an_uppercase_closing_tag_still_ends_the_reasoning_block() -> None:
    """A lowercase opener closed by </THINK> is complete, not truncated."""
    # THINK_CLOSE_RE decides truncation. Case-sensitive, it misses </THINK> and
    # discards a COMPLETE completion as if the token budget had run out.
    completion = (
        '<think>draft {"p_yes": 0.9, "p_no": 0.1}</THINK>\n'
        '{"p_yes": 0.3, "p_no": 0.7, "confidence": 0.6, "info_utility": 0.5}'
    )
    assert json.loads(canonical_prediction(completion) or "{}")["p_yes"] == 0.3


def test_a_reasoning_draft_is_never_delivered_as_the_answer() -> None:
    """An unparseable answer must yield None, never a draft from the reasoning."""
    # The draft parses; the post-</think> answer does not. Only discarding the
    # reasoning outright keeps the draft out of the candidate pool.
    completion = 'Draft: {"p_yes": 0.9, "p_no": 0.1}\n</think>\n{"p_yes": }'
    assert canonical_prediction(completion) is None


def test_malformed_final_object_falls_back_to_an_earlier_valid_one() -> None:
    """Walking from the end must not strand a valid earlier object."""
    completion = (
        "</think>\n"
        '{"p_yes": 0.25, "p_no": 0.75, "confidence": 0.5, "info_utility": 0.5}\n'
        '{"p_yes": }'
    )
    assert json.loads(canonical_prediction(completion) or "{}")["p_yes"] == 0.25


def test_a_stray_brace_does_not_swallow_the_answer_object() -> None:
    """Candidate matching must not span an unclosed brace into the answer."""
    completion = '</think>\nNote {see below\n{"p_yes": 0.31, "p_no": 0.69}'
    assert json.loads(canonical_prediction(completion) or "{}")["p_yes"] == 0.31


def test_completion_without_any_closing_tag_is_left_intact() -> None:
    """No closing tag must not cause the whole completion to be stripped."""
    parsed = json.loads(canonical_prediction('{"p_yes": 0.42, "p_no": 0.58}') or "{}")
    assert parsed["p_yes"] == 0.42


def test_a_trailing_non_prediction_object_does_not_shadow_the_answer() -> None:
    """An object with no p_yes after the answer is skipped, not taken as it."""
    completion = (
        "</think>\n"
        '{"p_yes": 0.3, "p_no": 0.7, "confidence": 0.6, "info_utility": 0.5}\n'
        '{"ok": true}'
    )
    assert json.loads(canonical_prediction(completion) or "{}")["p_yes"] == 0.3


def test_a_truncated_think_block_never_delivers_a_draft() -> None:
    """An opener with no closer is a cut-off completion, not an answer."""
    # Real on the truncated path (max_tokens exhausted mid-reasoning): every
    # probability present is a working estimate, so there is nothing to deliver.
    completion = '<think>maybe {"p_yes": 0.9, "p_no": 0.1} still thinking'
    assert canonical_prediction(completion) is None
    assert parse_p_yes(completion) is None


def test_an_uppercase_truncated_think_block_never_delivers_a_draft() -> None:
    """An uppercase <THINK> opener with no closer is truncated all the same."""
    # THINK_OPEN_RE decides truncation. Case-sensitive, it misses <THINK> and
    # the draft probability is harvested and delivered as the final answer.
    completion = '<THINK>maybe {"p_yes": 0.9, "p_no": 0.1} still thinking'
    assert canonical_prediction(completion) is None
    assert parse_p_yes(completion) is None


def test_a_nested_object_in_the_answer_still_parses() -> None:
    """An answer carrying a nested object must not be lost to a null delivery."""
    completion = (
        "</think>\n"
        '{"p_yes": 0.44, "p_no": 0.56, "meta": {"draft": 0.9}, "confidence": 0.7}'
    )
    parsed = json.loads(canonical_prediction(completion) or "{}")
    assert parsed["p_yes"] == 0.44
    assert parsed["confidence"] == 0.7


def test_a_brace_inside_a_string_value_does_not_cut_the_answer_short() -> None:
    """A '}' inside a string value must not truncate the answer object."""
    completion = (
        "</think>\n"
        '{"p_yes": 0.61, "p_no": 0.39, "note": "resolves if } appears", '
        '"info_utility": 0.8}'
    )
    parsed = json.loads(canonical_prediction(completion) or "{}")
    assert parsed["p_yes"] == 0.61
    assert parsed["info_utility"] == 0.8


def test_a_cut_whose_nested_object_closes_is_still_a_cut() -> None:
    """A cut mid-object counts even when a nested object inside it closed."""
    # The nested "}" is the only closing brace after the opener. Asking whether
    # ANY "}" follows therefore reads this cut as a complete object and
    # delivers the 0.25 written one object earlier -- a draft.
    assert canonical_prediction(CUT_AFTER_NESTED_CLOSE) is None


def test_a_stray_brace_in_prose_after_the_answer_is_not_a_cut() -> None:
    """An unclosed brace that never began an object must not void the answer."""
    # "{source" is prose, not a truncated object: a JSON object opens with a
    # quoted key. Calling it a cut turns a delivered forecast into an error.
    parsed = json.loads(canonical_prediction(STRAY_BRACE_AFTER_ANSWER) or "{}")
    assert parsed["p_yes"] == 0.3


def test_a_completion_ending_on_the_opening_brace_is_a_cut() -> None:
    """A cut landing ON the brace must not deliver an earlier draft."""
    # The tail after "{" is empty here, which the first version read as prose.
    assert canonical_prediction('{"p_yes": 0.9, "p_no": 0.1}\n<answer>\n{') is None


def test_a_completion_ending_on_brace_plus_whitespace_is_a_cut() -> None:
    """Whitespace after the opening brace is still a cut, not prose."""
    assert canonical_prediction('{"p_yes": 0.9, "p_no": 0.1}\n<answer>\n{\n ') is None


def test_a_completion_cut_at_max_tokens_is_raised_and_not_retried() -> None:
    """finish_reason 'length' fails fast instead of handing a draft to the parser."""
    # The draft is complete and the cut lands in prose, so canonical_prediction
    # alone would deliver 0.3 as the answer.
    completion = '</think>\nDraft {"p_yes": 0.3, "p_no": 0.7} looks low given the ne'
    assert canonical_prediction(completion) is not None
    choice = MagicMock(finish_reason="length")
    choice.message.content = completion
    with (
        patch(f"{MODULE_PATH}.openai.OpenAI") as mock_openai,
        patch(f"{MODULE_PATH}.time.sleep") as mock_sleep,
    ):
        create = mock_openai.return_value.chat.completions.create
        create.return_value = MagicMock(
            choices=[choice], usage=MagicMock(prompt_tokens=10, completion_tokens=5)
        )
        client = module.VLLMClient(api_key="k", base_url=ENDPOINT)
        with pytest.raises(module.TruncatedCompletionError):
            module.generate_prediction_with_retry(
                client=client, model="m", messages=[], temperature=0.0, max_tokens=64
            )
    assert create.call_count == 1
    mock_sleep.assert_not_called()


def test_retry_exhaustion_keeps_the_last_cause() -> None:
    """After the last attempt the raised error names the underlying failure."""
    client = MagicMock()
    client.completions.side_effect = ValueError("vllm endpoint unreachable")
    with patch(f"{MODULE_PATH}.time.sleep"):
        with pytest.raises(RuntimeError, match="vllm endpoint unreachable") as excinfo:
            module.generate_prediction_with_retry(
                client=client, model="m", messages=[], temperature=0.0, max_tokens=64
            )
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert client.completions.call_count == module.COMPLETION_RETRIES


def test_the_reasoning_strip_runs_to_the_last_closing_tag() -> None:
    """Two closing tags: everything before the LAST one is reasoning."""
    # Stripping only to the first tag leaves the mid-reasoning 0.82 as a
    # candidate, and the unparseable final object hands the delivery to it.
    assert canonical_prediction(TWO_CLOSING_TAGS) is None


# ---------------------------------------------------------------------------
# build_messages — mech-parity framing, optional sources
# ---------------------------------------------------------------------------


def test_build_messages_is_single_user_message_no_system() -> None:
    # to_chat_format parity: one user message, NO system message.
    """build_messages wraps content in a single user message, no system message."""
    assert build_messages("CONTENT") == [{"role": "user", "content": "CONTENT"}]


def test_build_forecaster_prompt_fills_background_template() -> None:
    """The <background> template is filled with question, date, and sources."""
    out = build_forecaster_prompt("Will X happen?", "05/06/2026", "SOURCE BLOCK")
    assert "<background>" in out and "</background>" in out
    assert "SOURCE BLOCK" in out
    assert "05/06/2026" in out
    # The question appears at the Question: header AND the trailing recall echo.
    assert out.count("Will X happen?") == 2
    # Literal JSON braces from the template survive sentinel substitution.
    assert "{" in out


# ---------------------------------------------------------------------------
# with_key_rotation — framework contract + generic rotation
# ---------------------------------------------------------------------------


def test_key_rotation_appends_api_keys_on_success() -> None:
    """A successful tool call returns its result with api_keys appended."""
    keychain = FakeKeyChain({"finetuned": "EMPTY"})

    @with_key_rotation
    def tool(**kwargs: Any) -> tuple[str, str, None, None, dict[str, str]]:
        return "result", "prompt", None, None, {"k": "v"}

    out = tool(api_keys=keychain)
    assert out == ("result", "prompt", None, None, {"k": "v"}, keychain)


def test_key_rotation_converts_exception_to_error_tuple() -> None:
    """A raising tool call is converted into a typed error-null result."""
    keychain = FakeKeyChain({"finetuned": "EMPTY"})

    @with_key_rotation
    def tool(**kwargs: Any) -> None:
        raise RuntimeError("boom")

    out = tool(api_keys=keychain)
    assert out[1:] == ("", None, None, None, keychain)
    parsed = json.loads(out[0])
    assert parsed["p_yes"] is None
    assert parsed["p_no"] is None
    assert parsed["confidence"] == 0.0
    assert parsed["info_utility"] == 0.0
    assert parsed["error"] == "boom"
    assert parsed["error_type"] == "RuntimeError"


def _rate_limit_error(message: str) -> openai.RateLimitError:
    """Build an openai.RateLimitError without touching the network."""
    return openai.RateLimitError(
        message, response=MagicMock(status_code=429, headers={}), body={}
    )


def test_rate_limit_rotates_every_configured_service_then_retries() -> None:
    """A 429 with retries left rotates every service and runs the tool again."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    attempts = 0

    @with_key_rotation
    def tool(**kwargs: Any) -> tuple[str, str, None, None, dict[str, str]]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _rate_limit_error("429 Too Many Requests")
        return "result", "prompt", None, None, {"k": "v"}

    out = tool(api_keys=keychain)
    assert attempts == 2
    assert out == ("result", "prompt", None, None, {"k": "v"}, keychain)
    assert sorted(keychain.rotated) == ["finetuned", "serperapi"]


def test_rate_limit_exhaustion_returns_the_typed_error_null() -> None:
    """Exhausted keys deliver the typed error null, not an escaping exception."""
    # Python does not cascade sibling `except` clauses, so a `raise` from the
    # RateLimitError branch would bypass the `except Exception` branch that
    # builds the null and reach the mech as a raw exception.
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    attempts = 0

    @with_key_rotation
    def tool(**kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        raise _rate_limit_error("429 Too Many Requests")

    out = tool(api_keys=keychain)
    # Every configured service was rotated once before the keys ran out.
    assert attempts == 2
    assert sorted(keychain.rotated) == ["finetuned", "serperapi"]
    assert out[1:] == ("", None, None, None, keychain)
    parsed = json.loads(out[0])
    assert parsed["p_yes"] is None
    assert parsed["p_no"] is None
    assert parsed["confidence"] == 0.0
    assert parsed["info_utility"] == 0.0
    assert parsed["error_type"] == "RateLimitError"
    assert "429" in parsed["error"]


# ---------------------------------------------------------------------------
# resolve_model — tool (mode) → fixed vLLM served name
# ---------------------------------------------------------------------------


def test_each_mode_resolves_to_its_served_model() -> None:
    """Each tool name resolves to its own distinct served-model name."""
    assert resolve_model(TOOL_BASE) == MODEL_BY_TOOL[TOOL_BASE]
    assert resolve_model(TOOL_FINE_TUNED) == MODEL_BY_TOOL[TOOL_FINE_TUNED]
    assert (
        resolve_model(TOOL_FINE_TUNED_CALIBRATED)
        == MODEL_BY_TOOL[TOOL_FINE_TUNED_CALIBRATED]
    )
    # The three modes are pairwise distinct served names.
    assert (
        len(
            {
                resolve_model(TOOL_BASE),
                resolve_model(TOOL_FINE_TUNED),
                resolve_model(TOOL_FINE_TUNED_CALIBRATED),
            }
        )
        == 3
    )


def test_calibrated_mode_targets_the_calibrated_served_name() -> None:
    """The calibrated tool requests ft-serve's virtual calibrated served name."""
    assert (
        resolve_model(TOOL_FINE_TUNED_CALIBRATED) == SERVED_MODEL_FINE_TUNED_CALIBRATED
    )
    assert SERVED_MODEL_FINE_TUNED_CALIBRATED == "qwen-14b-fine-tuned-calibrated"


# ---------------------------------------------------------------------------
# run() — end to end with mocked inference
# ---------------------------------------------------------------------------


def test_run_rejects_unknown_tool() -> None:
    """An unsupported tool name surfaces a 'not supported' error result."""
    out = run(tool="not-a-tool", prompt="q", api_keys=FakeKeyChain({"finetuned": "x"}))
    # with_key_rotation converts the ValueError into an error result tuple.
    assert "not supported" in out[0]


def _bare_prompt(question: str) -> str:
    return (
        f'With the given question "{question}" and the `yes` option represented '
        "by `Yes` and the `no` option represented by `No`, what are the "
        "respective probabilities of `p_yes` and `p_no` occurring?"
    )


def test_run_fine_tuned_mode_calls_its_model_and_returns_canonical_json() -> None:
    """Fine-tuned mode calls the fine-tuned model and returns canonical JSON."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC") as gather,
    ):
        gen.return_value = (WELL_FORMED, None)
        out = run(
            tool=TOOL_FINE_TUNED,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )

    result, completion, tx, _callback, used_params, returned_keys = out
    assert json.loads(result)["p_yes"] == 0.73
    assert completion == WELL_FORMED
    assert tx is None
    assert returned_keys is keychain
    # Fine-tuned mode calls the fine-tuned served model.
    assert used_params["tool"] == TOOL_FINE_TUNED
    assert used_params["model"] == MODEL_BY_TOOL[TOOL_FINE_TUNED]
    assert gen.call_args.kwargs["model"] == MODEL_BY_TOOL[TOOL_FINE_TUNED]

    # The trader-template path records its tier and is never scan-truncated.
    assert used_params["parse_tier"] == "template"
    assert used_params["scan_truncated"] is False

    # The question is extracted from the bare prompt and web-searched, then
    # embedded in the <background> forecaster prompt as a single user message.
    gather.assert_called_once_with("Will X happen?", "serp-key")
    sent = gen.call_args.kwargs["messages"]
    assert len(sent) == 1 and sent[0]["role"] == "user"
    user_content = sent[0]["content"]
    assert "<background>" in user_content and "</background>" in user_content
    assert "SRC" in user_content
    assert "Will X happen?" in user_content


def test_run_base_mode_calls_the_base_served_model() -> None:
    """Base mode calls the base served model."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = (WELL_FORMED, None)
        run(
            tool=TOOL_BASE,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    assert gen.call_args.kwargs["model"] == MODEL_BY_TOOL[TOOL_BASE]


def test_run_ignores_requester_supplied_model() -> None:
    # The served model is fixed per mode; a `model` in the request must NOT
    # change which model the tool calls (no untrusted model input).
    """A requester-supplied `model` is ignored; the per-mode model is used."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = (WELL_FORMED, None)
        run(
            tool=TOOL_BASE,
            model="attacker-chosen-model",
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    assert gen.call_args.kwargs["model"] == MODEL_BY_TOOL[TOOL_BASE]


def test_run_uses_default_endpoint() -> None:
    """Without an override, run uses the default VLLM_ENDPOINT."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager") as mgr,
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = (WELL_FORMED, None)
        run(tool=TOOL_BASE, prompt=_bare_prompt("Will X happen?"), api_keys=keychain)
    # VLLMClientManager(api_key, endpoint) — endpoint is the 2nd positional arg.
    assert mgr.call_args.args[1] == VLLM_ENDPOINT


def test_run_ignores_requester_supplied_endpoint() -> None:
    """A request-supplied endpoint is ignored; the constant is always used."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager") as mgr,
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = (WELL_FORMED, None)
        run(
            tool=TOOL_BASE,
            vllm_endpoint="http://attacker/v1",
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    assert mgr.call_args.args[1] == VLLM_ENDPOINT


def test_run_uses_keychain_endpoint_override() -> None:
    """A `finetuned_endpoint` KeyChain entry overrides the default base_url."""
    override = "http://vllm.internal:8000/v1"
    keychain = FakeKeyChain(
        {"finetuned": "EMPTY", "finetuned_endpoint": override, "serperapi": "serp-key"}
    )
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager") as mgr,
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = (WELL_FORMED, None)
        run(tool=TOOL_BASE, prompt=_bare_prompt("Will X happen?"), api_keys=keychain)
    # VLLMClientManager(api_key, endpoint) — endpoint is the 2nd positional arg.
    assert mgr.call_args.args[1] == override


def test_run_raises_on_unparseable_completion() -> None:
    """An unparseable completion surfaces a 'parseable p_yes' error result."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = ("<think>only reasoning, no json</think>", None)
        out = run(
            tool=TOOL_BASE,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    # with_key_rotation converts the ValueError into an error result tuple.
    assert "parseable p_yes" in out[0]


def test_run_delivers_the_error_not_a_negative_probability() -> None:
    """An out-of-range p_yes is delivered as the error, not as p_no < 0."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = ('</think>\n{"p_yes": 1.5, "p_no": -0.5}', None)
        out = run(
            tool=TOOL_BASE,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    # with_key_rotation converts the ValueError into a typed error null.
    parsed = json.loads(out[0])
    assert parsed["p_yes"] is None
    assert parsed["p_no"] is None
    assert parsed["error_type"] == "ValueError"
    assert "parseable p_yes" in parsed["error"]


def test_run_delivers_the_extracted_object_not_the_raw_completion() -> None:
    """run() delivers the extracted forecast, not the completion around it."""
    # Wiring check: the completion is not valid JSON on its own (it carries the
    # closing tag and trailing prose), so an unwired delivery fails json.loads.
    # The fixture also pins the fix -- the trailing "{source" is prose, and the
    # old cut heuristic turned this delivery into an error result.
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = (STRAY_BRACE_AFTER_ANSWER, None)
        out = run(
            tool=TOOL_BASE,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    result, completion = out[0], out[1]
    assert completion == STRAY_BRACE_AFTER_ANSWER
    assert result != completion
    assert json.loads(result) == {
        "p_yes": 0.3,
        "p_no": 0.7,
        "confidence": 0.5,
        "info_utility": 0.5,
    }


def test_run_delivers_the_error_when_the_answer_was_cut_off() -> None:
    """A completion cut mid-answer is delivered as the typed error null."""
    # The draft written one object earlier must never reach the requester as
    # the final answer.
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = (CUT_AFTER_NESTED_CLOSE, None)
        out = run(
            tool=TOOL_BASE,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    parsed = json.loads(out[0])
    assert parsed["p_yes"] is None
    assert parsed["error_type"] == "ValueError"
    assert "parseable p_yes" in parsed["error"]


def test_run_delivery_rate_zero_returns_max_cost() -> None:
    """A zero delivery rate returns the counter callback's max cost."""
    counter = MagicMock(return_value=1.23)
    out = run(
        tool=TOOL_BASE,
        prompt="q",
        delivery_rate=0,
        counter_callback=counter,
        api_keys=FakeKeyChain({"finetuned": "EMPTY"}),
    )
    # max-cost path returns the float straight through the decorator.
    assert out == 1.23
    counter.assert_called_once()


def test_vllm_client_passes_base_url() -> None:
    """Test that the VLLMClient builds the OpenAI client with the given base_url."""
    with patch("openai.OpenAI") as MockOpenAI:
        module.VLLMClient(api_key="EMPTY", base_url=ENDPOINT)
        MockOpenAI.assert_called_once_with(api_key="EMPTY", base_url=ENDPOINT)


# ---------------------------------------------------------------------------
# gather_sources -- no web context (OOD for the model) yields None / an error
# ---------------------------------------------------------------------------


def _serper_response(payload: dict) -> MagicMock:
    """Build a fake Serper response whose .json() returns `payload`."""
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


def test_gather_sources_formats_results() -> None:
    """A normal Serper response is formatted into the <background> body."""
    payload = {"organic": [{"position": 1, "title": "T", "link": "L", "snippet": "S"}]}
    with patch(
        f"{MODULE_PATH}.fetch_additional_sources",
        return_value=_serper_response(payload),
    ):
        out = gather_sources("Will X happen?", "serp-key")
    assert out is not None
    assert "Organic Results" in out and "T" in out


def test_gather_sources_raises_on_serper_request_failure() -> None:
    """A Serper request error becomes an explanatory failure (not empty context)."""
    with patch(
        f"{MODULE_PATH}.fetch_additional_sources", side_effect=Exception("down")
    ):
        with pytest.raises(RuntimeError, match="request failed"):
            gather_sources("Will X happen?", "serp-key")


def test_gather_sources_surfaces_a_non_2xx_status_instead_of_a_json_crash() -> None:
    """A 4xx/5xx Serper body raises with its status, never reaching .json()."""
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.HTTPError(
        "403 Client Error: Forbidden for url: https://google.serper.dev/search"
    )
    # A credit/auth error body is HTML, so .json() would raise an opaque
    # JSONDecodeError that hides the status the operator needs.
    resp.json.side_effect = AssertionError("json() called on an error body")
    with patch(f"{MODULE_PATH}.fetch_additional_sources", return_value=resp):
        with pytest.raises(RuntimeError, match="403 Client Error"):
            gather_sources("Will X happen?", "serp-key")


def test_gather_sources_returns_none_on_zero_results() -> None:
    """Zero usable results returns None so run() can deliver the flagged null."""
    payload: dict[str, list] = {"organic": [], "peopleAlsoAsk": []}
    with patch(
        f"{MODULE_PATH}.fetch_additional_sources",
        return_value=_serper_response(payload),
    ):
        assert gather_sources("Will X happen?", "serp-key") is None


# ---------------------------------------------------------------------------
# parse_prompt + empty-retrieval guard (issue #455 port)
# ---------------------------------------------------------------------------


def test_parse_prompt_trader_template_parity() -> None:
    """Trader-template path: the extracted title serves as BOTH values."""
    question, query, tier = parse_prompt(_bare_prompt("Will X happen by 2026?"))
    assert tier == "template"
    # LLM-input parity with the old extract_question behavior: the question
    # fed to the LLM and the search query are both the extracted title.
    assert question == "Will X happen by 2026?"
    assert query == question


def test_parse_prompt_free_text_clause_derivation() -> None:
    """A boilerplate lead-in anchors the market question as the search query."""
    prompt = (
        "Please predict the following market: Will Alexander Isak permanently "
        "transfer to Liverpool FC before September 2, 2025? Resolution source: "
        "official club announcements or BBC Sport."
    )
    question, query, tier = parse_prompt(prompt)
    assert tier == "clause"
    # The LLM question is the WHOLE prompt; only the search query is derived.
    assert question == prompt
    assert query.startswith("Will Alexander Isak")
    assert query.endswith("2025?")


@pytest.mark.parametrize("degenerate", ["", "   ", "???", '"""'])
def test_degenerate_prompt_short_circuits_before_search(degenerate: str) -> None:
    """Unsearchable prompts return the flagged null with ZERO search calls."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.fetch_additional_sources") as fetch,
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
    ):
        out = run(tool=TOOL_BASE, prompt=degenerate, api_keys=keychain)
    fetch.assert_not_called()
    gen.assert_not_called()
    result, _, tx, _, used_params, _ = out
    assert json.loads(result) == {
        "p_yes": 0.5,
        "p_no": 0.5,
        "confidence": 0.0,
        "info_utility": 0.0,
    }
    assert tx is None
    assert used_params["empty_retrieval"] is True
    assert used_params["null_reason"] == "empty query"
    assert used_params["scan_truncated"] is False


def test_both_empty_retrieval_returns_flagged_null_live_search() -> None:
    """Organic AND peopleAlsoAsk both empty -> flagged null, 'live search'."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    payload: dict[str, list] = {"organic": [], "peopleAlsoAsk": []}
    with (
        patch(
            f"{MODULE_PATH}.fetch_additional_sources",
            return_value=_serper_response(payload),
        ),
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
    ):
        out = run(
            tool=TOOL_BASE,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    gen.assert_not_called()
    result, _, _, _, used_params, _ = out
    assert json.loads(result)["p_yes"] == 0.5
    assert used_params["empty_retrieval"] is True
    assert used_params["null_reason"] == "live search"
    assert used_params["parse_tier"] == "template"


def test_organic_empty_but_people_also_ask_present_still_calls_the_llm() -> None:
    """The guard needs BOTH lists empty; peopleAlsoAsk alone keeps the LLM path."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    payload = {"organic": [], "peopleAlsoAsk": [{"question": "Q?", "snippet": "A."}]}
    with (
        patch(
            f"{MODULE_PATH}.fetch_additional_sources",
            return_value=_serper_response(payload),
        ),
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
    ):
        gen.return_value = (WELL_FORMED, None)
        out = run(
            tool=TOOL_BASE,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    gen.assert_called_once()
    result, _, _, _, used_params, _ = out
    assert json.loads(result)["p_yes"] == 0.73
    assert "empty_retrieval" not in used_params
    # The peopleAlsoAsk block is the evidence the forecaster actually receives.
    assert "People Also Ask" in gen.call_args.kwargs["messages"][0]["content"]


def test_template_past_scan_window_not_marked_truncated() -> None:
    """A trader-template prompt longer than the window is NOT scan_truncated."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    scan_chars = module._MAX_SCAN_CHARS
    prompt = _bare_prompt("Will X happen?") + " filler" * (scan_chars // 3)
    assert len(prompt) > scan_chars
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = (WELL_FORMED, None)
        out = run(tool=TOOL_BASE, prompt=prompt, api_keys=keychain)
    used_params = out[4]
    assert used_params["parse_tier"] == "template"
    assert used_params["scan_truncated"] is False


def test_raw_tier_past_scan_window_is_marked_truncated() -> None:
    """A question-free prompt past the window is raw tier AND scan_truncated."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    prompt = "no question words at all here. " * (module._MAX_SCAN_CHARS // 10)
    assert len(prompt) > module._MAX_SCAN_CHARS
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = (WELL_FORMED, None)
        out = run(tool=TOOL_BASE, prompt=prompt, api_keys=keychain)
    used_params = out[4]
    assert used_params["parse_tier"] == "raw"
    assert used_params["scan_truncated"] is True


def test_clause_tier_past_scan_window_is_marked_truncated() -> None:
    """A clause-tier pick on a longer-than-window prompt is still marked."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    prompt = "Will the ECB cut rates at its next meeting? " + "filler " * (
        module._MAX_SCAN_CHARS // 3
    )
    assert len(prompt) > module._MAX_SCAN_CHARS
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"),
    ):
        gen.return_value = (WELL_FORMED, None)
        out = run(tool=TOOL_BASE, prompt=prompt, api_keys=keychain)
    used_params = out[4]
    assert used_params["parse_tier"] == "clause"
    assert used_params["scan_truncated"] is True


def test_free_text_search_uses_derived_query_not_full_prompt() -> None:
    """The Serper call gets the derived clause; the LLM gets the whole prompt."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    prompt = (
        "Please predict the following market: Will Alexander Isak permanently "
        "transfer to Liverpool FC before September 2, 2025? Resolution source: "
        "official club announcements or BBC Sport."
    )
    payload = {"organic": [{"position": 1, "title": "T", "link": "L", "snippet": "S"}]}
    with (
        patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen,
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(
            f"{MODULE_PATH}.fetch_additional_sources",
            return_value=_serper_response(payload),
        ) as fetch,
    ):
        gen.return_value = (WELL_FORMED, None)
        out = run(tool=TOOL_BASE, prompt=prompt, api_keys=keychain)
    query = fetch.call_args.args[0]
    assert query.startswith("Will Alexander Isak")
    assert len(query) < len(prompt)
    # The full prompt (resolution criteria included) reaches the forecaster
    # template, not the compressed search query.
    user_content = gen.call_args.kwargs["messages"][0]["content"]
    assert "official club announcements or BBC Sport" in user_content
    assert out[4]["parse_tier"] == "clause"


def test_malformed_serper_body_is_typed_error_not_flagged_null() -> None:
    """organic: null surfaces as the shape ValueError, not a flagged null."""
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with (
        patch(f"{MODULE_PATH}.VLLMClientManager"),
        patch(
            f"{MODULE_PATH}.fetch_additional_sources",
            return_value=_serper_response({"organic": None, "peopleAlsoAsk": []}),
        ),
    ):
        out = run(
            tool=TOOL_BASE,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    # with_key_rotation converts the ValueError into a typed error null.
    parsed = json.loads(out[0])
    assert "malformed 'organic'" in parsed["error"]
    assert parsed["p_yes"] is None
    assert parsed["p_no"] is None
    assert parsed["confidence"] == 0.0
    assert parsed["info_utility"] == 0.0
    assert parsed["error_type"] == "ValueError"


# ---------------------------------------------------------------------------
# Shared annotations -- the tier/reason strings reach used_params verbatim
# ---------------------------------------------------------------------------


def test_parse_prompt_only_ever_returns_a_declared_tier() -> None:
    """Every tier parse_prompt can emit is a member of ParsedPrompt's Literal."""
    # Reads the tier set off ParsedPrompt, which every copy in the fleet
    # declares identically -- not off a module-level alias this tool alone
    # would carry.
    declared = set(get_args(module.ParsedPrompt.__annotations__["tier"]))
    prompts = [
        _bare_prompt("Will X happen?"),  # template
        "Some preamble. Will the club sign a striker? More text.",  # clause
        "no question words at all here",  # raw
    ]
    tiers = {parse_prompt(prompt).tier for prompt in prompts}
    assert tiers == declared
