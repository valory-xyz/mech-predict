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

"""Unit tests for superforcaster: thread-safe client, offline tiktoken, and source_content."""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import packages.valory.customs.superforcaster_polymarket_v2.superforcaster_polymarket_v2 as module
from packages.valory.customs.superforcaster_polymarket_v2.superforcaster_polymarket_v2 import (
    OpenAIClientManager,
    generate_prediction_with_retry,
    parse_prompt,
    run,
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh OpenAIClient, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(
            "packages.valory.customs.superforcaster_polymarket_v2.superforcaster_polymarket_v2.OpenAIClient"
        ) as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance

            with mgr as client:
                assert client is mock_instance
                MockClient.assert_called_once_with(api_key="sk-test")

            mock_instance.client.close.assert_called_once()

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

    def test_generate_prediction_requires_client_param(self) -> None:
        """generate_prediction_with_retry requires client as first param."""
        params = list(inspect.signature(generate_prediction_with_retry).parameters)
        assert params[0] == "client"


SF_MODULE = (
    "packages.valory.customs.superforcaster_polymarket_v2.superforcaster_polymarket_v2"
)

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

PREDICTION_JSON = json.dumps(
    {"p_yes": 0.5, "p_no": 0.5, "confidence": 0.5, "info_utility": 0.5}
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


def _install_mock_client(mock_client_mgr: MagicMock) -> MagicMock:
    """Wire OpenAIClientManager to return a client whose completions() yields PREDICTION_JSON.

    The wrapper's call path is `OpenAIClient.completions(...)`, so the response
    must be configured on `mock_client.completions.return_value` — not on
    `mock_client.chat.completions.create` (which is the raw OpenAI SDK path the
    wrapper hides).

    :param mock_client_mgr: the patched OpenAIClientManager mock.
    :return: the inner mock_client wired into the manager's __enter__.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = PREDICTION_JSON
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_client.completions.return_value = mock_response
    mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)
    return mock_client


class TestSuperforcasterSourceContent:
    """Verify superforcaster captures and replays source_content correctly."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_live_capture_wraps_serper_json(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Live run wraps Serper response in {'serper_response': ...}."""
        mock_serper = MagicMock()
        mock_serper.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_serper
        _install_mock_client(mock_client_mgr)

        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
        )

        assert result[0] == PREDICTION_JSON
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
        _install_mock_client(mock_client_mgr)

        source_content = {"serper_response": FAKE_SERPER_RESPONSE}
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        assert result[0] == PREDICTION_JSON
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
        mock_serper = MagicMock()
        mock_serper.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_serper
        _install_mock_client(mock_client_mgr)

        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        assert result[0] == PREDICTION_JSON
        used_params = result[4]
        assert "source_content" not in used_params


EMPTY_SERPER_RESPONSE: dict = {"organic": [], "peopleAlsoAsk": []}

# Free-text format prompts: the advertised contract (issue #455)
FREE_TEXT_PROMPT = "Will Alexander Isak join Liverpool before September 2 2025?"
LONG_FREE_TEXT_PROMPT = (
    "Please predict the following market: Will Alexander Isak permanently transfer "
    "to Liverpool FC before the end of the summer 2025 transfer window (September 2, "
    "2025 23:59 UTC)? Resolution source: official club announcements or BBC Sport. "
    "The market resolves YES if a permanent transfer (not a loan) is confirmed by "
    "the resolution source before the deadline."
)


class TestParsePromptPort:
    """parse_prompt() -> (question_for_llm, search_query, tier) (issue #455 port)."""

    def test_trader_template_uses_extracted_question_for_both(self) -> None:
        """Trader-template path: the bare question serves as both values."""
        question, query, tier = parse_prompt(PREDICTION_PROMPT)
        assert question == "Will X happen?"
        assert query == question
        assert tier == "template"

    def test_free_text_llm_gets_full_prompt(self) -> None:
        """Free-text input: the LLM question is the whole prompt."""
        question, _, tier = parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert question == LONG_FREE_TEXT_PROMPT
        assert tier == "clause"

    def test_boilerplate_prefix_is_dropped_from_query(self) -> None:
        """The query anchors at the market question, dropping instruction text."""
        _, query, _ = parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert query.startswith("Will Alexander Isak")
        assert query.endswith("?")
        assert len(query) <= module._MAX_SEARCH_QUERY_LEN

    def test_double_quotes_are_stripped_from_query_only(self) -> None:
        """Quoted spans become exact-match Serper terms; drop them from the query."""
        prompt = (
            'Will any candle have a final "High" price >= 82000 in the window? '
            "Resolution source: Binance."
        )
        question, query, _ = parse_prompt(prompt)
        assert '"' not in query
        assert "High" in query
        assert '"High"' in question  # the LLM still sees the exact wording

    def test_no_question_clause_truncates(self) -> None:
        """A prompt with no question clause falls back to the capped prompt."""
        no_q = "x" * 300
        question, query, tier = parse_prompt(no_q)
        assert question == no_q
        assert tier == "raw"
        assert len(query) == module._MAX_SEARCH_QUERY_LEN


class TestIssue455Guards:
    """Short-circuit, empty-retrieval flagged nulls, and parity (issue #455)."""

    @pytest.mark.parametrize("degenerate", ["", "   ", "???", '"""'])
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_degenerate_prompt_short_circuits_before_search(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock, degenerate: str
    ) -> None:
        """Prompts with no searchable content never reach Serper or the LLM."""
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=degenerate,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        mock_fetch.assert_not_called()
        mock_client.completions.assert_not_called()
        assert json.loads(result[0]) == {
            "p_yes": 0.5,
            "p_no": 0.5,
            "confidence": 0.0,
            "info_utility": 0.0,
        }
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "empty query"
        assert result[4]["scan_truncated"] is False

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_zero_hit_live_search_returns_flagged_null(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Organic AND peopleAlsoAsk both empty -> flagged null, reason 'live search'."""
        mock_fetch.return_value = MagicMock(json=lambda: EMPTY_SERPER_RESPONSE)
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        mock_fetch.assert_called_once()
        mock_client.completions.assert_not_called()
        assert json.loads(result[0])["p_yes"] == 0.5
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "live search"
        assert result[4]["parse_tier"] == "clause"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_empty_cached_replay_returns_flagged_null(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """An empty cached capture replays to the same flagged null."""
        _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
            source_content={"serper_response": EMPTY_SERPER_RESPONSE},
        )
        assert json.loads(result[0])["confidence"] == 0.0
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "cached replay"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_trader_template_parity_end_to_end(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """LLM-input parity: the template path feeds the extracted question to both sinks."""
        mock_serper = MagicMock()
        mock_serper.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_serper
        _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        mock_fetch.assert_called_once_with("Will X happen?", "serper-test")
        assert "Question:\nWill X happen?" in result[1]
        assert result[4]["parse_tier"] == "template"
        assert result[4]["scan_truncated"] is False

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_long_template_prompt_is_not_marked_truncated(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Template past the scan window is NOT flagged: the match precedes the scan."""
        mock_serper = MagicMock()
        mock_serper.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_serper
        _install_mock_client(mock_client_mgr)
        prompt = PREDICTION_PROMPT + " filler" * (module._MAX_SCAN_CHARS // 3)
        assert len(prompt) > module._MAX_SCAN_CHARS
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert result[4]["parse_tier"] == "template"
        assert result[4]["scan_truncated"] is False

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_free_text_search_uses_derived_query_llm_gets_full_prompt(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Serper gets the short derived query; the LLM still sees the whole prompt."""
        mock_serper = MagicMock()
        mock_serper.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_serper
        _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        sent_query = mock_fetch.call_args[0][0]
        assert sent_query.startswith("Will Alexander Isak")
        assert len(sent_query) <= module._MAX_SEARCH_QUERY_LEN
        # criteria text the derived query drops must still reach the LLM
        assert "official club announcements or BBC Sport" in result[1]
        assert result[4]["parse_tier"] == "clause"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_malformed_serper_body_raises_typed_error(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """organic: null is a broken integration -> typed error, not a flagged null."""
        mock_fetch.return_value = MagicMock(
            json=lambda: {"organic": None, "peopleAlsoAsk": []}
        )
        _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        # v2's key-rotation wrapper surfaces the error string as result[0]
        assert "'organic'" in result[0]
        assert result[4] is None
