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

"""Unit tests for superforcaster-polymarket-v3's free-text-input contract (issue #455)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.customs.superforcaster_polymarket_v3.superforcaster_polymarket_v3 import (
    _MAX_SCAN_CHARS,
    _MAX_SEARCH_QUERY_LEN,
    parse_prompt,
    run,
)

V3_MODULE = (
    "packages.valory.customs.superforcaster_polymarket_v3."
    "superforcaster_polymarket_v3"
)

FAKE_SERPER_RESPONSE = {
    "organic": [{"title": "T", "link": "https://example.test", "snippet": "S"}],
    "peopleAlsoAsk": [{"question": "Q?", "snippet": "A."}],
}

EMPTY_SERPER_RESPONSE: dict = {"organic": [], "peopleAlsoAsk": []}

PREDICTION_JSON = json.dumps(
    {"p_yes": 0.6, "p_no": 0.4, "confidence": 0.8, "info_utility": 0.6}
)

# Trader-template format prompt (regression: previous callers must still work)
TRADER_PROMPT = (
    'Given the question "Will X happen?" and the `yes` answer criterion, ...'
)
# Free-text format prompt: the advertised contract (issue #455)
FREE_TEXT_PROMPT = "Will Alexander Isak join Liverpool before September 2 2025?"
# Long free-text prompt that would return empty Serper results if passed raw
LONG_FREE_TEXT_PROMPT = (
    "Please predict the following market: Will Alexander Isak permanently transfer "
    "to Liverpool FC before the end of the summer 2025 transfer window (September 2, "
    "2025 23:59 UTC)? Resolution source: official club announcements or BBC Sport. "
    "The market resolves YES if a permanent transfer (not a loan) is confirmed by "
    "the resolution source before the deadline."
)


def _make_mock_api_keys() -> MagicMock:
    """Create a mock KeyChain-like api_keys object with both provider keys."""
    services = {
        "openai": "sk-test",
        "anthropic": "sk-ant-test",
        "serperapi": "serper-test",
        "return_source_content": "false",
        "source_content_mode": "cleaned",
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key]
    mock.get = lambda key, default="": services.get(key, default)
    mock.max_retries = lambda: {"openai": 0, "openrouter": 0, "anthropic": 0}
    return mock


def _install_mock_client(mock_client_mgr: MagicMock) -> MagicMock:
    """Wire LLMClientManager to a client whose completions() yields PREDICTION_JSON.

    :param mock_client_mgr: the patched LLMClientManager mock.
    :return: the inner mock client wired into the manager's __enter__.
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


class TestParsePrompt:
    """parse_prompt() -> (question_for_llm, search_query, tier)."""

    def test_trader_template_uses_extracted_question_for_both(self) -> None:
        """Trader-template path: the bare question serves as both values."""
        question, query, tier = parse_prompt(TRADER_PROMPT)
        assert question == "Will X happen?"
        assert query == question
        assert tier == "template"

    def test_free_text_llm_gets_full_prompt_query_is_the_clause(self) -> None:
        """Free-text input: whole prompt to the LLM, question clause to Serper."""
        question, query, tier = parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert question == LONG_FREE_TEXT_PROMPT
        # "Please predict the following market: " is gone; the clause survives
        # whole, including the deadline and the trailing '?'.
        assert query.startswith("Will Alexander Isak")
        assert query.endswith("?")
        assert len(query) <= _MAX_SEARCH_QUERY_LEN
        assert tier == "clause"

    def test_boilerplate_lead_in_does_not_anchor_the_query(self) -> None:
        """Boilerplate lead-in must not anchor; the market question wins."""
        prompt = (
            "You are being asked to provide a probability estimate for a "
            "prediction market question. Please respond with a JSON object. "
            "Question: Will Bitcoin reach $150,000 or higher on any major "
            "exchange by December 31, 2026? Resolution source: TradingView."
        )
        _, query, tier = parse_prompt(prompt)
        assert query.startswith("Will Bitcoin reach")
        assert query.endswith("2026?")
        assert tier == "clause"

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

    def test_tier_is_reported(self) -> None:
        """The tier tags template / clause / raw explicitly."""
        assert parse_prompt(TRADER_PROMPT)[2] == "template"
        assert parse_prompt(FREE_TEXT_PROMPT)[2] == "clause"
        assert parse_prompt("no question mark here at all")[2] == "raw"

    def test_scan_window_bounds_candidate_search(self) -> None:
        """A clause past the scan window is not found; the LLM still gets all."""
        prompt = "x" * (3 * _MAX_SCAN_CHARS) + " Will X happen by 2027?"
        question, _, tier = parse_prompt(prompt)
        assert tier == "raw"
        assert question == prompt


class TestEmptyRetrievalGuard:
    """v3 returns a flagged null prediction on empty retrieval (issue #455)."""

    @pytest.mark.parametrize(
        "degenerate",
        [
            "",
            "   ",
            "???",
            '"""',
            "\u201c\u201d\u2018\u2019",
        ],
    )
    @patch(f"{V3_MODULE}.LLMClientManager")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_degenerate_prompt_short_circuits_before_serper(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock, degenerate: str
    ) -> None:
        """Prompts with no searchable content never reach Serper at all."""
        mock_fetch.return_value = MagicMock(json=lambda: EMPTY_SERPER_RESPONSE)
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt=degenerate,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        mock_fetch.assert_not_called()
        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.5 and parsed["p_no"] == 0.5
        assert parsed["confidence"] == 0.0 and parsed["info_utility"] == 0.0
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "empty query"
        assert result[4]["scan_truncated"] is False

    @patch(f"{V3_MODULE}.LLMClientManager")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_both_empty_live_search_returns_flagged_null(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Empty organic AND peopleAlsoAsk yields the flagged null, no LLM call."""
        mock_fetch.return_value = MagicMock(json=lambda: EMPTY_SERPER_RESPONSE)
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.5 and parsed["confidence"] == 0.0
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "live search"
        assert result[4]["parse_tier"] == "clause"
        mock_client.completions.assert_not_called()

    @patch(f"{V3_MODULE}.LLMClientManager")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_cached_replay_both_empty_returns_flagged_null(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """An empty cached source_content yields the flagged null, no fetch."""
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
            source_content={
                "mode": "cleaned",
                "serper_response": EMPTY_SERPER_RESPONSE,
            },
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.5 and parsed["confidence"] == 0.0
        assert result[4]["null_reason"] == "cached replay"
        mock_fetch.assert_not_called()

    @patch(f"{V3_MODULE}.LLMClientManager")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_reshaped_serper_body_is_an_error_not_a_flagged_null(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A 200 body without the organic key surfaces as an error, not 0.5."""
        mock_fetch.return_value = MagicMock(json=lambda: {"message": "quota exceeded"})
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        # v3's decorator wraps unexpected exceptions as a stringified error
        # tuple; the shape ValueError must be visible there, not a 0.5 null.
        assert "organic" in result[0]
        assert result[4] is None

    @patch(f"{V3_MODULE}.LLMClientManager")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_organic_empty_but_misc_present_still_calls_llm(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """The guard needs BOTH lists empty; PAA alone keeps the LLM path."""
        mock_fetch.return_value = MagicMock(
            json=lambda: {
                "organic": [],
                "peopleAlsoAsk": [{"question": "Q?", "snippet": "A."}],
            }
        )
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        mock_client.completions.assert_called_once()
        assert json.loads(result[0])["p_yes"] == 0.6


class TestRunWiring:
    """run() feeds parse_prompt's outputs to the right consumers."""

    @patch(f"{V3_MODULE}.LLMClientManager")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_trader_request_sends_extracted_question_to_serper(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """LLM-input parity: a trader request searches the bare question."""
        serper_resp = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_fetch.return_value = serper_resp
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt=TRADER_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        # the HTTP-error guard must actually run on the happy path (a MagicMock
        # would silently absorb its removal otherwise)
        serper_resp.raise_for_status.assert_called_once()
        query_sent = mock_fetch.call_args[0][0]
        assert query_sent == "Will X happen?"
        # and the LLM prompt carries the bare question, not the full template
        llm_prompt = mock_client.completions.call_args.kwargs["messages"][1]["content"]
        assert "Will X happen?" in llm_prompt
        assert "`yes` answer criterion" not in llm_prompt
        assert result[4]["parse_tier"] == "template"
        assert result[4]["scan_truncated"] is False

    @patch(f"{V3_MODULE}.LLMClientManager")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_free_text_serper_gets_short_query_llm_gets_full_prompt(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Serper gets the derived clause; the LLM sees the whole prompt."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        query_sent = mock_fetch.call_args[0][0]
        assert len(query_sent) <= _MAX_SEARCH_QUERY_LEN
        assert query_sent != LONG_FREE_TEXT_PROMPT
        # criteria text that the derived query drops must still reach the LLM
        llm_prompt = mock_client.completions.call_args.kwargs["messages"][1]["content"]
        assert "official club announcements or BBC Sport" in llm_prompt
        assert result[1] == llm_prompt
        assert result[4]["parse_tier"] == "clause"
        assert result[4]["scan_truncated"] is False

    @patch(f"{V3_MODULE}.LLMClientManager")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_long_template_prompt_is_not_marked_truncated(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Template past the window is NOT flagged: question precedes the scan."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        _install_mock_client(mock_client_mgr)
        prompt = TRADER_PROMPT + " filler" * (_MAX_SCAN_CHARS // 3)
        assert len(prompt) > _MAX_SCAN_CHARS
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert result[4]["parse_tier"] == "template"
        assert result[4]["scan_truncated"] is False

    @patch(f"{V3_MODULE}.LLMClientManager")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_scan_truncation_is_observable(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Raw tier from an exhausted scan window is marked, not silent."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        _install_mock_client(mock_client_mgr)
        # the only '?' sits past the scan window -> raw tier via truncation
        prompt = "word " * (_MAX_SCAN_CHARS // 4) + "Will it happen by 2027?"
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert result[4]["parse_tier"] == "raw"
        assert result[4]["scan_truncated"] is True
