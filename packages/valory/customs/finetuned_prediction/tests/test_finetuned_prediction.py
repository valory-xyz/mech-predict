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
    TOOL_BASE,
    TOOL_FINE_TUNED,
    VLLM_ENDPOINT,
    build_forecaster_prompt,
    build_messages,
    canonical_prediction,
    parse_p_yes,
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
        self._keys = dict(keys)
        self.rotated: List[str] = []

    def __getitem__(self, service: str) -> str:
        return self._keys[service]  # raises KeyError when absent, like the real one

    def max_retries(self) -> Dict[str, int]:
        return {service: 1 for service in self._keys}

    def rotate(self, service: str) -> None:
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
    assert parse_p_yes(completion) == expected


def test_canonical_prediction_normalises_schema() -> None:
    result = canonical_prediction(WELL_FORMED)
    assert result is not None
    obj = json.loads(result)
    assert obj == {"p_yes": 0.73, "p_no": 0.27, "confidence": 0.8, "info_utility": 0.9}


def test_canonical_prediction_derives_p_no_and_defaults() -> None:
    # confidence/info_utility absent -> defaulted; p_no derived from p_yes.
    result = canonical_prediction('{"p_yes": 0.25}')
    assert result is not None
    obj = json.loads(result)
    assert obj["p_yes"] == 0.25
    assert obj["p_no"] == 0.75
    assert obj["confidence"] == 0.5
    assert obj["info_utility"] == 0.5


def test_canonical_prediction_returns_none_on_malformed() -> None:
    assert canonical_prediction("<think>oops</think> not json") is None
    assert canonical_prediction(None) is None


# ---------------------------------------------------------------------------
# build_messages — mech-parity framing, optional sources
# ---------------------------------------------------------------------------


def test_build_messages_is_single_user_message_no_system() -> None:
    # to_chat_format parity: one user message, NO system message.
    assert build_messages("CONTENT") == [{"role": "user", "content": "CONTENT"}]


def test_build_forecaster_prompt_fills_background_template() -> None:
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
    keychain = FakeKeyChain({"finetuned": "EMPTY"})

    @with_key_rotation
    def tool(**kwargs: Any):
        return "result", "prompt", None, None, {"k": "v"}

    out = tool(api_keys=keychain)
    assert out == ("result", "prompt", None, None, {"k": "v"}, keychain)


def test_key_rotation_converts_exception_to_error_tuple() -> None:
    keychain = FakeKeyChain({"finetuned": "EMPTY"})

    @with_key_rotation
    def tool(**kwargs: Any):
        raise RuntimeError("boom")

    out = tool(api_keys=keychain)
    assert out == ("boom", "", None, None, None, keychain)


# ---------------------------------------------------------------------------
# resolve_model — tool (mode) → fixed vLLM served name
# ---------------------------------------------------------------------------


def test_each_mode_resolves_to_its_served_model() -> None:
    assert resolve_model(TOOL_BASE) == MODEL_BY_TOOL[TOOL_BASE]
    assert resolve_model(TOOL_FINE_TUNED) == MODEL_BY_TOOL[TOOL_FINE_TUNED]
    assert MODEL_BY_TOOL[TOOL_BASE] != MODEL_BY_TOOL[TOOL_FINE_TUNED]


# ---------------------------------------------------------------------------
# run() — end to end with mocked inference
# ---------------------------------------------------------------------------


def test_run_rejects_unknown_tool() -> None:
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
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen, patch(
        f"{MODULE_PATH}.VLLMClientManager"
    ), patch(f"{MODULE_PATH}.gather_sources", return_value="SRC") as gather:
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
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen, patch(
        f"{MODULE_PATH}.VLLMClientManager"
    ), patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"):
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
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen, patch(
        f"{MODULE_PATH}.VLLMClientManager"
    ), patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"):
        gen.return_value = (WELL_FORMED, None)
        run(
            tool=TOOL_BASE,
            model="attacker-chosen-model",
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    assert gen.call_args.kwargs["model"] == MODEL_BY_TOOL[TOOL_BASE]


def test_run_uses_default_endpoint() -> None:
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen, patch(
        f"{MODULE_PATH}.VLLMClientManager"
    ) as mgr, patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"):
        gen.return_value = (WELL_FORMED, None)
        run(tool=TOOL_BASE, prompt=_bare_prompt("Will X happen?"), api_keys=keychain)
    # VLLMClientManager(api_key, endpoint) — endpoint is the 2nd positional arg.
    assert mgr.call_args.args[1] == VLLM_ENDPOINT


def test_run_endpoint_kwarg_overrides_default() -> None:
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    custom = "http://other-box:9000/v1"
    with patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen, patch(
        f"{MODULE_PATH}.VLLMClientManager"
    ) as mgr, patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"):
        gen.return_value = (WELL_FORMED, None)
        run(
            tool=TOOL_BASE,
            vllm_endpoint=custom,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    assert mgr.call_args.args[1] == custom


def test_run_raises_on_unparseable_completion() -> None:
    keychain = FakeKeyChain({"finetuned": "EMPTY", "serperapi": "serp-key"})
    with patch(f"{MODULE_PATH}.generate_prediction_with_retry") as gen, patch(
        f"{MODULE_PATH}.VLLMClientManager"
    ), patch(f"{MODULE_PATH}.gather_sources", return_value="SRC"):
        gen.return_value = ("<think>only reasoning, no json</think>", None)
        out = run(
            tool=TOOL_BASE,
            prompt=_bare_prompt("Will X happen?"),
            api_keys=keychain,
        )
    # with_key_rotation converts the ValueError into an error result tuple.
    assert "parseable p_yes" in out[0]


def test_run_delivery_rate_zero_returns_max_cost() -> None:
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
    with patch("openai.OpenAI") as MockOpenAI:
        module.VLLMClient(api_key="EMPTY", base_url=ENDPOINT)
        MockOpenAI.assert_called_once_with(api_key="EMPTY", base_url=ENDPOINT)
