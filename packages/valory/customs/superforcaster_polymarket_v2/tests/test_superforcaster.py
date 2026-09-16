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

import openai
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

    def test_default_max_tokens_admits_a_full_free_text_completion(self) -> None:
        """The default cap clears an observed free-text completion."""
        # Free-text prompts elicit the evidence block before the verdict.
        # Observed production completions ran to 1016 tokens, where a 500
        # cap truncated before any JSON was emitted at all.
        assert module.DEFAULT_OPENAI_SETTINGS["max_tokens"] >= 2048


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
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_organic_empty_but_misc_present_still_calls_llm(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """The guard needs BOTH lists empty; peopleAlsoAsk alone keeps the LLM path."""
        mock_fetch.return_value = MagicMock(
            json=lambda: {
                "organic": [],
                "peopleAlsoAsk": [{"question": "Q?", "snippet": "A."}],
            }
        )
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        mock_client.completions.assert_called_once()
        assert result[0] == PREDICTION_JSON
        assert "Q?" in result[1]

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
    def test_cached_replay_organic_empty_but_misc_present_still_calls_llm(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Replay guard needs BOTH lists empty; cached peopleAlsoAsk alone still predicts."""
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
            source_content={
                "serper_response": {
                    "organic": [],
                    "peopleAlsoAsk": [
                        {"question": "Cached Q?", "snippet": "Cached A."}
                    ],
                }
            },
        )
        mock_fetch.assert_not_called()
        mock_client.completions.assert_called_once()
        assert result[0] == PREDICTION_JSON
        assert "Cached Q?" in result[1]
        assert "empty_retrieval" not in result[4]

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_degenerate_prompt_with_cached_content_still_predicts(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """The empty-query short-circuit is live-path only: a cached capture is still replayed."""
        # The unsearchable query only means no Serper call can be made; with a
        # non-empty capture there is nothing to search for in the first place,
        # so the run must reach the LLM instead of short-circuiting to a null.
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt="???",
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )
        mock_fetch.assert_not_called()
        mock_client.completions.assert_called_once()
        assert result[0] == PREDICTION_JSON
        assert "Test snippet content" in result[1]
        assert "empty_retrieval" not in result[4]
        assert result[4]["parse_tier"] == "raw"

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
    def test_raw_tier_past_window_is_marked_truncated(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Question-free prompt past the window: raw tier AND truncated."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        _install_mock_client(mock_client_mgr)
        prompt = "no question words at all here. " * (module._MAX_SCAN_CHARS // 10)
        assert len(prompt) > module._MAX_SCAN_CHARS
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert result[4]["parse_tier"] == "raw"
        assert result[4]["scan_truncated"] is True

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_clause_tier_past_window_is_marked_truncated(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A clause-tier pick on a longer-than-window prompt is still marked."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        _install_mock_client(mock_client_mgr)
        prompt = "Will the ECB cut rates at its next meeting? " + "filler " * (
            module._MAX_SCAN_CHARS // 3
        )
        assert len(prompt) > module._MAX_SCAN_CHARS
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert result[4]["parse_tier"] == "clause"
        assert result[4]["scan_truncated"] is True

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
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["p_no"] is None
        assert parsed["confidence"] == 0.0
        assert parsed["info_utility"] == 0.0
        assert parsed["error_type"] == "ValueError"
        assert result[4] is None


def _rate_limit_error(message: str = "fleet-wide 429") -> openai.RateLimitError:
    """Build a real openai.RateLimitError so type(e).__name__ is the real name.

    Skips the real constructor (which requires a live httpx.Response) and sets
    up just the attributes the decorator under test touches -- the repo-wide
    pattern, so the suite does not need httpx on the type-check path.

    :param message: the ``str(exc)`` payload.
    :return: a RateLimitError instance usable as a raise target in tests.
    """
    err: openai.RateLimitError = openai.RateLimitError.__new__(  # type: ignore[call-overload]
        openai.RateLimitError
    )
    Exception.__init__(err, message)
    err.message = message  # type: ignore[attr-defined]
    return err


def _make_throttled_api_keys(openai_retries: int, openrouter_retries: int) -> MagicMock:
    """Create a KeyChain-like mock whose max_retries() returns real ints."""
    mock = _make_mock_api_keys()
    mock.max_retries.return_value = {
        "openai": openai_retries,
        "openrouter": openrouter_retries,
    }
    return mock


class TestExtractPrediction:
    """The delivery must be the forecast object, never the reasoning block."""

    def test_bare_json_is_returned_unchanged(self) -> None:
        """The trader-template shape already emits JSON and must not change."""
        bare = '{"p_yes": 0.19, "p_no": 0.81, "confidence": 0.8, "info_utility": 0.7}'
        assert json.loads(module.extract_prediction(bare) or "")["p_yes"] == 0.19

    def test_reasoning_scaffold_yields_only_the_forecast(self) -> None:
        """A free-text completion delivers the trailing object, not the block."""
        # The shape a free-text prompt actually produces: the prompt asks for
        # the seven-step scaffold AND for JSON only, and the model does both.
        completion = (
            "<facts>\n- BTC must print at or above 150000.\n</facts>\n"
            "<no>\n1. Requires an unprecedented rally. (Strength: 8)\n</no>\n"
            "<thinking>\nWeighing base rates against the sources.\n</thinking>\n"
            "<tentative>\n0.08\n</tentative>\n"
            "<answer>\n*0.06*\n</answer>\n"
            '{"p_yes": 0.06, "p_no": 0.94, "confidence": 0.7, "info_utility": 0.6}'
        )
        out = module.extract_prediction(completion) or ""
        parsed = json.loads(out)
        assert parsed == {
            "p_yes": 0.06,
            "p_no": 0.94,
            "confidence": 0.7,
            "info_utility": 0.6,
        }
        assert "<facts>" not in out

    def test_a_tentative_value_does_not_shadow_the_answer(self) -> None:
        """An earlier object in the reasoning must not win over the final one."""
        completion = (
            'Draft so far: {"p_yes": 0.9, "p_no": 0.1} but revising down.\n'
            '{"p_yes": 0.06, "p_no": 0.94, "confidence": 0.7, "info_utility": 0.6}'
        )
        assert json.loads(module.extract_prediction(completion) or "")["p_yes"] == 0.06

    def test_trailing_non_forecast_object_is_skipped(self) -> None:
        """An object with no usable p_yes must not shadow the forecast."""
        completion = (
            '{"p_yes": 0.06, "p_no": 0.94, "confidence": 0.7, "info_utility": 0.6}\n'
            '{"note": "sources retrieved"}'
        )
        assert json.loads(module.extract_prediction(completion) or "")["p_yes"] == 0.06

    def test_a_completion_with_no_forecast_is_left_untouched(self) -> None:
        """With nothing to extract the caller's error path must still see it."""
        assert module.extract_prediction("no json at all") == "no json at all"
        assert module.extract_prediction(None) is None

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_run_delivers_parseable_json_for_a_reasoning_completion(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """run() delivers the forecast object, not the raw reasoning block."""
        # Pins the WIRING, not just the helper: without extract_prediction on
        # the completion path this delivers the whole scaffold and the
        # requester's json.loads raises.
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_client = _install_mock_client(mock_client_mgr)
        mock_client.completions.return_value.content = (
            "<facts>\n- BTC must print at or above 150000.\n</facts>\n"
            "<thinking>\nWeighing base rates.\n</thinking>\n"
            "<answer>\n*0.06*\n</answer>\n"
            '{"p_yes": 0.06, "p_no": 0.94, "confidence": 0.7, "info_utility": 0.6}'
        )
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4.1-2025-04-14",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert json.loads(result[0])["p_yes"] == 0.06
        assert "<facts>" not in result[0]


class TestExtractPredictionCutAndRange:
    """A max_tokens cut must not deliver a draft, and p_yes must be in range."""

    def test_a_cut_mid_object_delivers_nothing(self) -> None:
        """The reviewer counterexample: a cut after a draft must not deliver it."""
        completion = '{"p_yes": 0.25, "p_no": 0.75}\n{"p_yes": '
        assert module.extract_prediction(completion) is None

    def test_a_cut_whose_nested_object_closes_is_still_a_cut(self) -> None:
        """An inner object closing must not be read as the outer one closing."""
        completion = (
            '{"p_yes": 0.25, "p_no": 0.75}\n'
            '{"p_yes": 0.31, "meta": {"src": "a"}, "p_no": '
        )
        assert module.extract_prediction(completion) is None

    def test_a_stray_brace_in_prose_is_not_a_cut(self) -> None:
        """An unclosed brace that never began an object must not block delivery."""
        completion = (
            "See {source for details on the resolution criteria.\n"
            '{"p_yes": 0.42, "p_no": 0.58, "confidence": 0.7, "info_utility": 0.6}'
        )
        assert json.loads(module.extract_prediction(completion) or "")["p_yes"] == 0.42

    def test_braces_and_escaped_quotes_inside_strings_are_ignored(self) -> None:
        """A brace or escaped quote inside a value must not end the object."""
        completion = (
            '{"p_yes": 0.42, "p_no": 0.58, '
            '"note": "resolves if } appears in the \\"title\\""}'
        )
        parsed = json.loads(module.extract_prediction(completion) or "")
        assert parsed["p_yes"] == 0.42
        assert parsed["note"] == 'resolves if } appears in the "title"'

    def test_a_brace_inside_a_pending_string_does_not_hide_a_cut(self) -> None:
        """A cut object whose pending string holds a brace is still a cut."""
        completion = (
            '{"p_yes": 0.25, "p_no": 0.75}\n'
            '{"p_yes": 0.31, "note": "resolves if } appears'
        )
        assert module.extract_prediction(completion) is None

    def test_an_escaped_quote_does_not_hide_a_cut(self) -> None:
        """An escaped quote must not be read as the end of a string value."""
        completion = (
            '{"p_yes": 0.25, "p_no": 0.75}\n'
            '{"note": "he wrote \\"} done\\" here", "p_yes": '
        )
        assert module.extract_prediction(completion) is None

    def test_an_out_of_range_p_yes_is_not_delivered(self) -> None:
        """p_yes outside [0, 1] is not a probability and must be skipped."""
        completion = (
            '{"p_yes": 0.06, "p_no": 0.94, "confidence": 0.7, "info_utility": 0.6}\n'
            '{"p_yes": 1.7, "p_no": -0.7}'
        )
        assert json.loads(module.extract_prediction(completion) or "")["p_yes"] == 0.06

    def test_a_sole_out_of_range_object_is_not_delivered(self) -> None:
        """When the only object is out of range there is nothing to deliver."""
        # Was: returned the content unchanged, which put p_yes 1.7 on-chain as
        # a normal forecast. The range check stopped a bad object beating a
        # good one but not a bad object being the only one.
        assert module.extract_prediction('{"p_yes": 1.7, "p_no": -0.7}') is None

    def test_a_null_p_yes_is_skipped(self) -> None:
        """A JSON null p_yes must not be coerced into a forecast."""
        completion = (
            '{"p_yes": 0.06, "p_no": 0.94, "confidence": 0.7, "info_utility": 0.6}\n'
            '{"p_yes": null, "p_no": null}'
        )
        assert json.loads(module.extract_prediction(completion) or "")["p_yes"] == 0.06

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_run_does_not_deliver_a_draft_when_the_completion_is_cut(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """run() delivers nothing when max_tokens cut the answer mid-object."""
        # Pins the WIRING on the delivery path: unwired this returns the raw
        # completion, and with the pre-cut-detection extractor it returns the
        # 0.25 draft as if it were the forecast.
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_client = _install_mock_client(mock_client_mgr)
        mock_client.completions.return_value.content = (
            '{"p_yes": 0.25, "p_no": 0.75}\n{"p_yes": '
        )
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4.1-2025-04-14",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        # A typed error null, not a bare None: None would be delivered verbatim
        # on-chain with no exception, so no retry and no typed-null branch.
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["error_type"] == "Exception"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_run_does_not_deliver_an_out_of_range_forecast(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """run() skips a trailing out-of-range object and delivers the forecast."""
        # Pins the WIRING too: unwired this returns both objects as raw text,
        # which the requester's json.loads rejects.
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_client = _install_mock_client(mock_client_mgr)
        mock_client.completions.return_value.content = (
            '{"p_yes": 0.06, "p_no": 0.94, "confidence": 0.7, "info_utility": 0.6}\n'
            '{"p_yes": 1.7, "p_no": -0.7}'
        )
        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4.1-2025-04-14",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert json.loads(result[0])["p_yes"] == 0.06


class TestRateLimitExhaustionNull:
    """with_key_rotation must not let a rate-limit exhaustion escape as an exception."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_exhausted_keys_return_typed_null_not_exception(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Every key exhausted -> typed error JSON returned, no exception escapes."""
        mock_client_mgr.side_effect = _rate_limit_error()
        api_keys = _make_throttled_api_keys(0, 0)

        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=api_keys,
            counter_callback=None,
        )

        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["p_no"] is None
        assert parsed["confidence"] == 0.0
        assert parsed["info_utility"] == 0.0
        assert parsed["error_type"] == "RateLimitError"
        assert "fleet-wide 429" in parsed["error"]
        api_keys.rotate.assert_not_called()

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_exhausted_keys_return_full_six_tuple(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """The exhaustion null is the same 6-tuple shape as a normal delivery."""
        mock_client_mgr.side_effect = _rate_limit_error()
        api_keys = _make_throttled_api_keys(0, 0)

        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=api_keys,
            counter_callback=None,
        )

        assert isinstance(result, tuple)
        assert len(result) == 6
        assert result[1] == ""
        assert result[2] is None
        assert result[3] is None
        assert result[4] is None
        assert result[5] is api_keys

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_keys_are_rotated_before_exhaustion(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Retries left -> rotate both providers, then fall back to the typed null."""
        mock_client_mgr.side_effect = _rate_limit_error()
        api_keys = _make_throttled_api_keys(2, 2)

        result = run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=api_keys,
            counter_callback=None,
        )

        rotated = [call.args[0] for call in api_keys.rotate.call_args_list]
        assert rotated == ["openai", "openrouter", "openai", "openrouter"]
        assert json.loads(result[0])["error_type"] == "RateLimitError"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_generic_failure_branch_matches_the_shared_helper(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """The except-Exception branch delivers exactly the helper's tuple."""
        # Comparing the helper against itself does not prove the branch uses
        # it -- the branch could drift to a divergent inline shape and stay
        # green. This drives the real branch and compares field for field.
        api_keys = _make_mock_api_keys()
        boom = ValueError("boom")
        mock_client_mgr.side_effect = boom

        result = module.run(
            tool="superforcaster-polymarket-v2",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=api_keys,
            counter_callback=None,
        )

        expected = module._null_prediction_response(boom, api_keys)
        assert json.loads(result[0]) == json.loads(expected[0])
        assert result[1:] == expected[1:]

    def test_null_prediction_response_is_shared_by_both_failure_paths(self) -> None:
        """The permanent-failure branch and the rate-limit branch build the same shape."""
        api_keys = MagicMock()
        rate_limited = module._null_prediction_response(_rate_limit_error(), api_keys)
        permanent = module._null_prediction_response(ValueError("boom"), api_keys)

        assert json.loads(rate_limited[0]).keys() == json.loads(permanent[0]).keys()
        assert json.loads(rate_limited[0])["error_type"] == "RateLimitError"
        assert json.loads(permanent[0])["error_type"] == "ValueError"
        assert rate_limited[1:] == permanent[1:] == ("", None, None, None, api_keys)
