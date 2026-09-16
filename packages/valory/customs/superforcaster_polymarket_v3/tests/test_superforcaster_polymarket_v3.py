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
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.customs.superforcaster_polymarket_v3.superforcaster_polymarket_v3 import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_SETTINGS,
    _MAX_SCAN_CHARS,
    _MAX_SEARCH_QUERY_LEN,
    extract_prediction,
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

    def test_default_max_tokens_admits_a_full_free_text_completion(self) -> None:
        """The default cap clears an observed free-text completion."""
        # Free-text prompts elicit the evidence block before the verdict.
        # Observed production completions ran to 1016 tokens, where a 500
        # cap truncated before any JSON was emitted at all.
        # Pins the VALUE, not a floor: a wiring test alone leaves anything in
        # [2048, 4095] invisible to CI. The Anthropic branch is already pinned
        # exactly, so this brings the OpenAI branch in line.
        assert DEFAULT_OPENAI_SETTINGS["max_tokens"] == 4096


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
    def test_degenerate_query_with_cached_content_still_predicts(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A degenerate query must not veto a non-empty cached replay."""
        # The empty-query gate exists to skip a doomed Serper call, so it
        # belongs on the live-search branch ONLY. Hoisted above the cached
        # branch it would flag a null here even though the sources needed to
        # answer are already in hand and no network call is at stake.
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt="???",
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
            source_content={
                "mode": "cleaned",
                "serper_response": FAKE_SERPER_RESPONSE,
            },
        )
        mock_fetch.assert_not_called()
        mock_client.completions.assert_called_once()
        assert json.loads(result[0])["p_yes"] == 0.6
        assert "empty_retrieval" not in result[4]
        assert "null_reason" not in result[4]
        assert result[4]["parse_tier"] == "raw"

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
        # the shape ValueError must surface as the TYPED error null: asserting
        # only on the message would also pass for a bare stringified return.
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["p_no"] is None
        assert parsed["confidence"] == 0.0
        assert parsed["info_utility"] == 0.0
        assert parsed["error_type"] == "ValueError"
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
    def test_truncation_with_in_window_clause_is_marked(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A clause-tier pick on a longer-than-window prompt is still marked."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        _install_mock_client(mock_client_mgr)
        # the chosen clause is in-window, but the real market question sits
        # past it -- the flag must be set on the clause tier, not only on raw
        prompt = (
            "Can I clarify the resolution source by 2025? "
            + "filler " * (_MAX_SCAN_CHARS // 6)
            + "Will the ECB cut rates at the next meeting?"
        )
        assert len(prompt) > _MAX_SCAN_CHARS
        result = run(
            tool="superforcaster-polymarket-v3",
            model="claude-fable-5",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert result[4]["parse_tier"] == "clause"
        assert result[4]["scan_truncated"] is True

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


class TestMaxTokensWiring:
    """The default max_tokens must reach the provider SDK, not just exist."""

    @patch(f"{V3_MODULE}.openai.OpenAI")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_default_max_tokens_reaches_the_openai_sdk_call(
        self, mock_fetch: MagicMock, mock_openai: MagicMock
    ) -> None:
        """run() forwards the default cap all the way into chat.completions.create."""
        # The constant assertion in TestParsePrompt pins the VALUE; this pins
        # its PATH. Only the real LLMClientManager / LLMClient are exercised
        # here (the openai SDK constructor is the single patch point), so a
        # regression that drops `max_tokens=` anywhere between run() and the
        # SDK call fails this test while the constant stays at 4096.
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        sdk_response = MagicMock()
        sdk_response.choices = [MagicMock(message=MagicMock(content=PREDICTION_JSON))]
        sdk_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        create = mock_openai.return_value.chat.completions.create
        create.return_value = sdk_response

        result = run(
            tool="superforcaster-polymarket-v3",
            model=DEFAULT_OPENAI_MODEL,
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        assert json.loads(result[0])["p_yes"] == 0.6
        create_kwargs = create.call_args.kwargs
        assert create_kwargs["max_tokens"] == DEFAULT_OPENAI_SETTINGS["max_tokens"]
        # and the same value is what the delivery reports as used
        assert result[4]["max_tokens"] == DEFAULT_OPENAI_SETTINGS["max_tokens"]


FORECAST = {"p_yes": 0.62, "p_no": 0.38, "confidence": 0.8, "info_utility": 0.6}
FORECAST_JSON = json.dumps(FORECAST)

# What the model actually returns on a free-text prompt: the seven-step
# scaffold the prompt asks for, a lower draft written mid-reasoning, and the
# real forecast last. Delivered verbatim it is not parseable JSON.
SCAFFOLD_COMPLETION = (
    "<facts>\n"
    "* No transfer has been announced as of today.\n"
    "</facts>\n"
    "<thinking>\n"
    'An early read put it lower: {"p_yes": 0.30, "p_no": 0.70, '
    '"confidence": 0.4, "info_utility": 0.3}\n'
    "Later reporting firmed the story up, so revise upward.\n"
    "</thinking>\n"
    "<answer>\n" + FORECAST_JSON + "\n"
    "</answer>\n"
)

# A max_tokens cut that lands while the answer object is being written.
CUT_COMPLETION = '<answer>\n{"p_yes": 0.62, "p_no"'

REFUSAL = "I cannot help with that request."

# A max_tokens cut that lands in prose AFTER a complete draft object. Nothing is
# left unclosed, so no text heuristic can tell the draft from an answer.
DRAFT_THEN_CUT_IN_PROSE = (
    '<facts>x</facts>\n<thinking>\nDraft estimate {"p_yes": 0.35, "p_no": 0.65} '
    "seems too low, let me reconsider given the news that changes the probabi"
)


def _openai_sdk_response(content: str) -> MagicMock:
    """Build a mock ``chat.completions.create`` response carrying *content*.

    :param content: the text the SDK reports as the message content.
    :return: a MagicMock shaped like an openai ChatCompletion.
    """
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return resp


def _anthropic_sdk_response(text: str) -> MagicMock:
    """Build a mock ``messages.create`` response carrying *text*.

    :param text: the text the single TextBlock returns.
    :return: a MagicMock shaped like an anthropic Message.
    """
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    resp = MagicMock()
    resp.content = [text_block]
    resp.stop_reason = "end_turn"
    resp.usage = MagicMock(input_tokens=10, output_tokens=5)
    return resp


class TestExtractPrediction:
    """Shape-by-shape coverage of the forecast extractor."""

    def test_bare_json_passes_through(self) -> None:
        """A completion that is already the forecast object survives unchanged."""
        assert json.loads(extract_prediction(FORECAST_JSON) or "") == FORECAST

    def test_reasoning_scaffold_yields_the_answer_object(self) -> None:
        """The forecast is sliced out of the facts/thinking/answer block."""
        assert json.loads(extract_prediction(SCAFFOLD_COMPLETION) or "") == FORECAST

    def test_draft_before_the_answer_does_not_shadow_it(self) -> None:
        """A tentative object written mid-reasoning loses to the final one."""
        content = '{"p_yes": 0.30}\nrevised after new reporting\n' + FORECAST_JSON
        assert json.loads(extract_prediction(content) or "") == FORECAST

    def test_trailing_non_forecast_object_is_skipped(self) -> None:
        """An object with no p_yes after the forecast is not mistaken for it."""
        content = FORECAST_JSON + '\nSources used: {"organic": 3, "misc": 1}'
        assert json.loads(extract_prediction(content) or "") == FORECAST

    def test_cut_mid_object_returns_none(self) -> None:
        """A completion cut while the answer was being written yields None."""
        assert extract_prediction(CUT_COMPLETION) is None

    def test_cut_whose_inner_object_closes_returns_none(self) -> None:
        """A nested object closing inside the cut does not fake a complete answer."""
        content = '{"p_yes": 0.30}\n{"meta": {"sources": 3}, "p_yes": 0.6'
        assert extract_prediction(content) is None

    def test_stray_brace_in_prose_is_not_a_cut(self) -> None:
        """An unclosed brace in prose must not suppress a delivered forecast."""
        content = "See {source for details\n" + FORECAST_JSON
        assert json.loads(extract_prediction(content) or "") == FORECAST

    def test_braces_and_escaped_quotes_inside_string_values(self) -> None:
        """A brace or an escaped quote inside a value does not end the object."""
        forecast: Dict[str, Any] = dict(FORECAST)
        forecast["rationale"] = 'resolves if } appears and "confirmed" is said'
        content = "<answer>\n" + json.dumps(forecast) + "\n</answer>"
        assert json.loads(extract_prediction(content) or "") == forecast

    def test_out_of_range_p_yes_is_skipped(self) -> None:
        """A p_yes outside [0, 1] is not a forecast; the valid earlier one wins."""
        content = FORECAST_JSON + '\ncorrection: {"p_yes": 1.4, "p_no": -0.4}'
        assert json.loads(extract_prediction(content) or "") == FORECAST

    def test_null_p_yes_is_skipped(self) -> None:
        """A null p_yes is not coercible, so the valid earlier object wins."""
        content = FORECAST_JSON + '\n{"p_yes": null, "p_no": null}'
        assert json.loads(extract_prediction(content) or "") == FORECAST

    def test_no_candidate_returns_none(self) -> None:
        """With no forecast object at all the raw completion is not handed on."""
        assert extract_prediction("I cannot answer this question.") is None

    def test_empty_content_returns_none(self) -> None:
        """Empty and None completions yield None, not an empty delivery."""
        assert extract_prediction(None) is None
        assert extract_prediction("") is None

    def test_a_completion_ending_on_the_opening_brace_is_a_cut(self) -> None:
        """A cut landing ON the brace must not deliver an earlier draft."""
        # tail is empty here, which the first version read as prose. On
        # pretty-printed JSON a cut after "{" is one of the likelier stops.
        content = '{"p_yes": 0.9, "p_no": 0.1}\n<answer>\n{'
        assert extract_prediction(content) is None

    def test_a_completion_ending_on_brace_plus_whitespace_is_a_cut(self) -> None:
        """Whitespace after the opening brace is still a cut, not prose."""
        content = '{"p_yes": 0.9, "p_no": 0.1}\n<answer>\n{\n '
        assert extract_prediction(content) is None

    def test_a_sole_out_of_range_forecast_is_not_delivered(self) -> None:
        """When the only object is out of range there is nothing to deliver."""
        # Returning the content here would put p_yes 1.7 on-chain as a normal
        # forecast; the caller's guard turns None into a retry instead.
        assert extract_prediction('{"p_yes": 1.7, "p_no": -0.7}') is None


class TestDeliveredPredictionIsParseable:
    """run() must deliver the forecast object on BOTH provider branches.

    These patch the provider SDK constructor only, so the real
    LLMClientManager / LLMClient / completions() path runs end to end.
    A helper-level test alone would stay green if the extractor were
    unwired from the return paths, which is the failure these pin.
    """

    @patch(f"{V3_MODULE}.openai.OpenAI")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_openai_branch_delivers_the_forecast_object(
        self, mock_fetch: MagicMock, mock_openai: MagicMock
    ) -> None:
        """run() on the OpenAI model delivers parseable JSON, not the scaffold."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        create = mock_openai.return_value.chat.completions.create
        create.return_value = _openai_sdk_response(SCAFFOLD_COMPLETION)

        result = run(
            tool="superforcaster-polymarket-v3",
            model=DEFAULT_OPENAI_MODEL,
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        assert json.loads(result[0]) == FORECAST
        assert "<thinking>" not in result[0]

    @patch(f"{V3_MODULE}.Anthropic")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_anthropic_branch_delivers_the_forecast_object(
        self, mock_fetch: MagicMock, mock_anthropic: MagicMock
    ) -> None:
        """run() on the claude model delivers parseable JSON, not the scaffold."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        create = mock_anthropic.return_value.messages.create
        create.return_value = _anthropic_sdk_response(SCAFFOLD_COMPLETION)

        result = run(
            tool="superforcaster-polymarket-v3",
            model=DEFAULT_ANTHROPIC_MODEL,
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        assert json.loads(result[0]) == FORECAST
        assert "<thinking>" not in result[0]

    @patch(f"{V3_MODULE}.time.sleep")
    @patch(f"{V3_MODULE}.openai.OpenAI")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_cut_completion_is_not_delivered(
        self, mock_fetch: MagicMock, mock_openai: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """A cut answer reaches the retry loop instead of being delivered partial."""
        # The OpenAI branch has no stop_reason guard, so the extractor's None
        # is the only thing between a truncated object and the caller.
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        create = mock_openai.return_value.chat.completions.create
        create.return_value = _openai_sdk_response(CUT_COMPLETION)

        result = run(
            tool="superforcaster-polymarket-v3",
            model=DEFAULT_OPENAI_MODEL,
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        delivered = json.loads(result[0])
        assert delivered["p_yes"] is None
        assert "<answer>" not in result[0]
        assert create.call_count == 3

    @patch(f"{V3_MODULE}.time.sleep")
    @patch(f"{V3_MODULE}.openai.OpenAI")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_openai_cut_at_max_tokens_is_not_delivered_or_retried(
        self, mock_fetch: MagicMock, mock_openai: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """finish_reason 'length' fails fast, like stop_reason on the Anthropic branch."""
        # The draft is complete and the cut lands in prose, so the extractor
        # alone would deliver 0.35 as the forecast.
        assert extract_prediction(DRAFT_THEN_CUT_IN_PROSE) is not None
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        create = mock_openai.return_value.chat.completions.create
        create.return_value = _openai_sdk_response(DRAFT_THEN_CUT_IN_PROSE)
        create.return_value.choices[0].finish_reason = "length"

        result = run(
            tool="superforcaster-polymarket-v3",
            model=DEFAULT_OPENAI_MODEL,
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        delivered = json.loads(result[0])
        assert delivered["p_yes"] is None
        assert delivered["error_type"] == "TruncatedCompletionError"
        assert "0.35" not in result[0]
        assert create.call_count == 1

    @patch(f"{V3_MODULE}.time.sleep")
    @patch(f"{V3_MODULE}.openai.OpenAI")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_prose_completion_is_not_delivered(
        self, mock_fetch: MagicMock, mock_openai: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """A completion with no forecast object reaches the retry loop, not the caller."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        create = mock_openai.return_value.chat.completions.create
        create.return_value = _openai_sdk_response(REFUSAL)

        result = run(
            tool="superforcaster-polymarket-v3",
            model=DEFAULT_OPENAI_MODEL,
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        delivered = json.loads(result[0])
        assert delivered["p_yes"] is None
        assert "cannot help" not in result[0]
        assert create.call_count == 3

    @patch(f"{V3_MODULE}.time.sleep")
    @patch(f"{V3_MODULE}.Anthropic")
    @patch(f"{V3_MODULE}.fetch_additional_sources")
    def test_anthropic_cut_at_max_tokens_is_not_retried(
        self, mock_fetch: MagicMock, mock_anthropic: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """stop_reason 'max_tokens' fails fast with the same error type as OpenAI's cut."""
        # Fail-fast keys on the exception class, so a truncation raised as a
        # plain ValueError would be retried three times with the same budget.
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        create = mock_anthropic.return_value.messages.create
        create.return_value = _anthropic_sdk_response(DRAFT_THEN_CUT_IN_PROSE)
        create.return_value.stop_reason = "max_tokens"

        result = run(
            tool="superforcaster-polymarket-v3",
            model=DEFAULT_ANTHROPIC_MODEL,
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        delivered = json.loads(result[0])
        assert delivered["p_yes"] is None
        assert delivered["error_type"] == "TruncatedCompletionError"
        assert "0.35" not in result[0]
        assert create.call_count == 1
