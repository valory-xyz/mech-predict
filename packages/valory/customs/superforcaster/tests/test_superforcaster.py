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

"""Unit tests for superforcaster: thread-safe client, structured outputs, source_content."""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import packages.valory.customs.superforcaster.superforcaster as module
from packages.valory.customs.superforcaster.superforcaster import (
    OpenAIClientManager,
    PredictionResult,
    _MAX_SCAN_CHARS,
    _MAX_SEARCH_QUERY_LEN,
    _parse_completion,
    parse_prompt,
    run,
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_openai_client(self) -> None:
        """__enter__ returns a fresh openai.OpenAI client, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(
            "packages.valory.customs.superforcaster.superforcaster.OpenAI"
        ) as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance

            with mgr as client:
                assert client is mock_instance
                MockOpenAI.assert_called_once_with(api_key="sk-test")

            mock_instance.close.assert_called_once()

    def test_no_global_client_variable(self) -> None:
        """The module must not define a module-level 'client' variable."""
        source = Path(module.__file__).read_text(encoding="utf-8")
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("client:") or stripped.startswith("client ="):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client' variable found at line {i}: {line}"
                    )

    def test_parse_completion_requires_client_param(self) -> None:
        """_parse_completion requires client as first param."""
        params = list(inspect.signature(_parse_completion).parameters)
        assert params[0] == "client"


SF_MODULE = "packages.valory.customs.superforcaster.superforcaster"

FAKE_SERPER_RESPONSE = {
    "searchParameters": {"q": "test query", "type": "search"},
    "organic": [
        {
            "title": "Test Result",
            "link": "http://example.com/result",
            "snippet": "Test snippet content",
            "position": 1,
        },
    ],
    "peopleAlsoAsk": [
        {"question": "What is test?", "snippet": "A test answer."},
    ],
}

FAKE_PREDICTION = PredictionResult(
    facts="Fact 1. Fact 2.",
    reasons_no="No 1 (strength 6). No 2 (strength 4).",
    reasons_yes="Yes 1 (strength 5). Yes 2 (strength 3).",
    aggregation="Base rate 0.5. Tentative: 0.5.",
    reflection="Passes evidence bar.",
    p_yes=0.5,
    p_no=0.5,
    confidence=0.5,
    info_utility=0.5,
)

PREDICTION_PROMPT = (
    'With the given question "Will X happen?" '
    "and the `yes` option represented by `Yes` and the `no` option represented by `No`, "
    "what are the respective probabilities of `p_yes` and `p_no` occurring?"
)


def _make_mock_api_keys(return_source_content: str = "false") -> MagicMock:
    """Create a mock KeyChain-like api_keys object."""
    services = {
        "openai": ["sk-test"],
        "serperapi": ["serper-test"],
        "return_source_content": [return_source_content],
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key][0]
    mock.get = lambda key, default="": services.get(key, [default])[0]
    return mock


def _mock_parse_response() -> MagicMock:
    """Build a fake response object matching the beta.chat.completions.parse shape."""
    return MagicMock(
        choices=[MagicMock(message=MagicMock(parsed=FAKE_PREDICTION, refusal=None))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )


class TestStructuredOutputContract:
    """Verify the tool uses OpenAI Structured Outputs and returns only the 4 on-chain fields."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_uses_beta_parse_with_prediction_result_schema(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """run() calls client.beta.chat.completions.parse with response_format=PredictionResult."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        mock_client.beta.chat.completions.parse.assert_called_once()
        kwargs = mock_client.beta.chat.completions.parse.call_args.kwargs
        assert kwargs["response_format"] is PredictionResult
        assert kwargs["model"] == "gpt-4.1-2025-04-14"
        mock_client.chat.completions.create.assert_not_called()

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_returns_only_four_mech_fields_on_chain(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """The on-chain result JSON contains exactly p_yes/p_no/confidence/info_utility."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        on_chain_json = result[0]
        parsed = json.loads(on_chain_json)
        assert set(parsed.keys()) == {"p_yes", "p_no", "confidence", "info_utility"}
        assert "facts" not in parsed
        assert "reasons_no" not in parsed
        assert "aggregation" not in parsed
        assert "reflection" not in parsed
        assert parsed["p_yes"] == 0.5
        assert parsed["p_no"] == 0.5


class TestPredictionResultValidator:
    """Pydantic sum validator rejects malformed probabilities."""

    def test_valid_sum_accepted(self) -> None:
        """p_yes + p_no = 1.0 is accepted."""
        obj = PredictionResult(
            facts="f",
            reasons_no="n",
            reasons_yes="y",
            aggregation="a",
            reflection="r",
            p_yes=0.7,
            p_no=0.3,
            confidence=0.6,
            info_utility=0.5,
        )
        assert obj.p_yes == 0.7

    def test_sum_violation_rejected(self) -> None:
        """p_yes + p_no outside 0.01 tolerance raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PredictionResult(
                facts="f",
                reasons_no="n",
                reasons_yes="y",
                aggregation="a",
                reflection="r",
                p_yes=0.7,
                p_no=0.2,
                confidence=0.6,
                info_utility=0.5,
            )


class TestSuperforcasterSourceContent:
    """Verify superforcaster captures and replays source_content correctly."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_live_capture_wraps_serper_json(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Live run wraps Serper response in {'serper_response': ...}."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
        )

        used_params = result[4]
        assert "source_content" in used_params
        assert "mode" in used_params["source_content"]
        assert "serper_response" in used_params["source_content"]
        assert used_params["source_content"]["serper_response"] == FAKE_SERPER_RESPONSE

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_replay_with_serper_response_format(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Replay with {'serper_response': ...} uses organic and peopleAlsoAsk."""
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        source_content = {"serper_response": FAKE_SERPER_RESPONSE}
        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        # Verify the prompt contains the organic result
        prediction_prompt = result[1]
        assert "Test Result" in prediction_prompt
        assert "Test snippet content" in prediction_prompt
        assert "What is test?" in prediction_prompt

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_flag_off_no_source_content(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """When return_source_content is false, source_content is not in used_params."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        used_params = result[4]
        assert "source_content" not in used_params


FREE_TEXT_PROMPT = "Will Alexander Isak join Liverpool before September 2 2025?"
LONG_FREE_TEXT_PROMPT = (
    "Please predict the following market: Will Alexander Isak permanently transfer "
    "to Liverpool FC before the end of the summer 2025 transfer window (September 2, "
    "2025 23:59 UTC)? Resolution source: official club announcements or BBC Sport. "
    "The market resolves YES if a permanent transfer (not a loan) is confirmed by "
    "the resolution source before the deadline."
)

EMPTY_SERPER_RESPONSE: dict = {"organic": [], "peopleAlsoAsk": []}


class TestParsePromptContract:
    """parse_prompt() -> (question_for_llm, search_query, tier) -- issue #455."""

    def test_trader_template_uses_extracted_question_for_both(self) -> None:
        """Trader-template path: the bare question serves as both values."""
        question, query, tier = parse_prompt(PREDICTION_PROMPT)
        assert question == "Will X happen?"
        assert query == question
        assert tier == "template"

    def test_free_text_llm_gets_full_prompt(self) -> None:
        """Free-text input: the LLM question is the whole prompt."""
        question, _, _ = parse_prompt(FREE_TEXT_PROMPT)
        assert question == FREE_TEXT_PROMPT
        question, _, _ = parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert question == LONG_FREE_TEXT_PROMPT

    def test_boilerplate_prefix_is_dropped_from_query(self) -> None:
        """The query anchors at the market question, dropping instruction text."""
        _, query, tier = parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert query.startswith("Will Alexander Isak")
        assert query.endswith("?")
        assert len(query) <= _MAX_SEARCH_QUERY_LEN
        assert tier == "clause"


class TestEmptyRetrievalGuard:
    """Degenerate prompts short-circuit; zero-hit retrieval returns a flagged null."""

    @pytest.mark.parametrize("degenerate", ["", "   ", "???", '"""'])
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_degenerate_prompt_short_circuits_before_serper(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock, degenerate: str
    ) -> None:
        """Prompts with no searchable content never reach Serper at all."""
        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=degenerate,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        mock_fetch.assert_not_called()
        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.5
        assert parsed["confidence"] == 0.0
        used_params = result[4]
        assert used_params["empty_retrieval"] is True
        assert used_params["null_reason"] == "empty query"
        assert used_params["scan_truncated"] is False

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_both_empty_live_retrieval_returns_flagged_null(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Organic AND peopleAlsoAsk empty -> flagged null, no LLM call."""
        mock_fetch.return_value = MagicMock(json=lambda: EMPTY_SERPER_RESPONSE)
        mock_client = MagicMock()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        mock_client.beta.chat.completions.parse.assert_not_called()
        assert json.loads(result[0])["p_yes"] == 0.5
        used_params = result[4]
        assert used_params["empty_retrieval"] is True
        assert used_params["null_reason"] == "live search"
        assert used_params["parse_tier"] == "clause"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_both_empty_cached_replay_returns_flagged_null(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """An empty cached capture takes the guard too, without a Serper call."""
        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
            source_content={"serper_response": EMPTY_SERPER_RESPONSE},
        )
        mock_fetch.assert_not_called()
        assert json.loads(result[0])["p_yes"] == 0.5
        used_params = result[4]
        assert used_params["empty_retrieval"] is True
        assert used_params["null_reason"] == "cached replay"


class TestScanWindowObservability:
    """parse_tier / scan_truncated reach used_params on every path."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_long_template_prompt_is_not_marked_truncated(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Template past the window is NOT flagged: the match precedes the scan."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        prompt = PREDICTION_PROMPT + " filler" * (_MAX_SCAN_CHARS // 3)
        assert len(prompt) > _MAX_SCAN_CHARS
        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        used_params = result[4]
        assert used_params["parse_tier"] == "template"
        assert used_params["scan_truncated"] is False

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_trader_request_sends_extracted_question_to_serper(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """LLM-input parity pin: the trader path is byte-identical to before."""
        serper_resp = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_fetch.return_value = serper_resp
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        # the HTTP-error guard must actually run on the happy path
        serper_resp.raise_for_status.assert_called_once()
        query_sent = mock_fetch.call_args[0][0]
        assert query_sent == "Will X happen?"
        # and the LLM prompt carries the bare question, not the full template
        llm_prompt = mock_client.beta.chat.completions.parse.call_args[1]["messages"][
            1
        ]["content"]
        assert "Will X happen?" in llm_prompt
        assert "`yes` option represented" not in llm_prompt

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_success_used_params_carry_parse_tier(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A free-text success records parse_tier=clause, not truncated."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _mock_parse_response()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster",
            model="gpt-4.1-2025-04-14",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        used_params = result[4]
        assert used_params["parse_tier"] == "clause"
        assert used_params["scan_truncated"] is False
