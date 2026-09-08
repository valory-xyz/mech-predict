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
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

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
        ('{"p_yes": 0.4, "p_no": 0.6}', 0.4),  # bare JSON, no think
        ("<think>no json here</think> nothing", None),  # no JSON object
        ('{"p_no": 0.6}', None),  # missing p_yes
        ('{"p_yes": "high"}', None),  # non-numeric p_yes
        ('{"p_yes": 1.5}', None),  # out of [0, 1]
        ("", None),  # empty
    ],
    ids=[
        "think_block",
        "bare_json",
        "no_json",
        "missing",
        "non_numeric",
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
    """A raising tool call is converted into an error result tuple."""
    keychain = FakeKeyChain({"finetuned": "EMPTY"})

    @with_key_rotation
    def tool(**kwargs: Any) -> None:
        raise RuntimeError("boom")

    out = tool(api_keys=keychain)
    assert out == ("boom", "", None, None, None, keychain)


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
    # with_key_rotation converts the ValueError into an error result tuple.
    assert "malformed 'organic'" in out[0]
