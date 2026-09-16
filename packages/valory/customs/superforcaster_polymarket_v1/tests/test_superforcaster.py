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

import packages.valory.customs.superforcaster_polymarket_v1.superforcaster_polymarket_v1 as module
from packages.valory.customs.superforcaster_polymarket_v1.superforcaster_polymarket_v1 import (
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
            "packages.valory.customs.superforcaster_polymarket_v1.superforcaster_polymarket_v1.OpenAIClient"
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
    "packages.valory.customs.superforcaster_polymarket_v1.superforcaster_polymarket_v1"
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


def _install_mock_client(
    mock_client_mgr: MagicMock, content: str = PREDICTION_JSON
) -> MagicMock:
    """Wire OpenAIClientManager to return a client whose completions() yields PREDICTION_JSON.

    The wrapper's call path is `OpenAIClient.completions(...)`, so the response
    must be configured on `mock_client.completions.return_value` — not on
    `mock_client.chat.completions.create` (which is the raw OpenAI SDK path the
    wrapper hides).

    :param mock_client_mgr: the patched OpenAIClientManager mock.
    :param content: the raw completion the mocked model returns.
    :return: the inner mock_client wired into the manager's __enter__.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = content
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
            tool="superforcaster-polymarket-v1",
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
            tool="superforcaster-polymarket-v1",
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
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        assert result[0] == PREDICTION_JSON
        used_params = result[4]
        assert "source_content" not in used_params


EMPTY_SERPER_RESPONSE: dict = {"organic": [], "peopleAlsoAsk": []}

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


class TestParsePromptContract:
    """parse_prompt() -> (question_for_llm, search_query, tier)."""

    def test_trader_template_parity(self) -> None:
        """Trader-template path: the bare question serves as both values."""
        question, query, tier = parse_prompt(PREDICTION_PROMPT)
        assert tier == "template"
        assert question == "Will X happen?"
        assert query == question

    def test_free_text_clause_derivation(self) -> None:
        """Boilerplate lead-in anchors the market question; LLM sees everything."""
        question, query, tier = parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert tier == "clause"
        assert question == LONG_FREE_TEXT_PROMPT
        # "Please predict the following market: " is gone; the clause survives
        # whole, including the deadline and the trailing '?'.
        assert query.startswith("Will Alexander Isak")
        assert query.endswith("?")
        assert len(query) <= module._MAX_SEARCH_QUERY_LEN


class TestIssue455Guards:
    """Empty-query short-circuit and both-empty-retrieval flagged nulls."""

    @pytest.mark.parametrize("degenerate", ["", "   ", "???", '"""'])
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_degenerate_prompt_short_circuits_before_serper(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock, degenerate: str
    ) -> None:
        """Prompts with no searchable content never reach Serper at all."""
        _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
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

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_both_empty_live_retrieval_returns_flagged_null(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A genuine zero-hit returns the flagged null, LLM never called."""
        mock_fetch.return_value = MagicMock(json=lambda: EMPTY_SERPER_RESPONSE)
        mock_client = _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
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

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_cached_replay_both_empty_returns_flagged_null(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """An empty cached source_content returns the flagged null too."""
        _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
            source_content={"serper_response": EMPTY_SERPER_RESPONSE},
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.5 and parsed["confidence"] == 0.0
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "cached replay"
        mock_fetch.assert_not_called()

    @pytest.mark.parametrize(
        "body",
        [
            {"organic": {"not": "a list"}, "peopleAlsoAsk": []},
            {"organic": "reshaped", "peopleAlsoAsk": []},
            {"organic": [{"title": "T"}], "peopleAlsoAsk": None},
            {"organic": [{"title": "T"}], "peopleAlsoAsk": "nope"},
        ],
    )
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_malformed_serper_shapes_are_typed_errors(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock, body: dict
    ) -> None:
        """Both shape checks covered: non-list organic AND non-list peopleAlsoAsk."""
        mock_fetch.return_value = MagicMock(json=lambda: body)
        _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v1",
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
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        used_params = result[4]
        assert used_params["parse_tier"] == "raw"
        assert used_params["scan_truncated"] is True

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_reshaped_serper_body_is_an_error_not_a_flagged_null(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A 200 body without the organic key surfaces as an error, not 0.5."""
        mock_fetch.return_value = MagicMock(json=lambda: {"message": "quota exceeded"})
        _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        # A broken integration is a typed error null, never 0.5/0.5.
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["p_no"] is None
        assert parsed["confidence"] == 0.0
        assert parsed["info_utility"] == 0.0
        assert parsed["error_type"] == "ValueError"
        assert parsed["error"].startswith("live search:")
        assert "organic" in result[0]

    def test_default_max_tokens_admits_a_full_free_text_completion(self) -> None:
        """The default cap clears an observed free-text completion."""
        # Free-text prompts elicit the evidence block before the verdict.
        # Observed production completions ran to 1016 tokens, where a 500
        # cap truncated before any JSON was emitted at all.
        assert module.DEFAULT_OPENAI_SETTINGS["max_tokens"] >= 2048


class TestIssue455RunWiring:
    """run() feeds the LLM the parsed question and Serper the derived query."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_trader_template_llm_and_serper_parity(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Trader path parity: the bare extracted question feeds both sinks."""
        serper_resp = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_fetch.return_value = serper_resp
        _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert mock_fetch.call_args[0][0] == "Will X happen?"
        # the HTTP-error guard must actually run on the happy path
        serper_resp.raise_for_status.assert_called_once()
        # the LLM prompt carries the bare question, not the full template
        assert "Will X happen?" in result[1]
        assert "`yes` option" not in result[1]
        assert result[4]["parse_tier"] == "template"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_free_text_serper_gets_short_query_llm_gets_full_prompt(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Free text: Serper gets the derived query, the LLM the whole prompt."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        _install_mock_client(mock_client_mgr)
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        query_sent = mock_fetch.call_args[0][0]
        assert len(query_sent) <= module._MAX_SEARCH_QUERY_LEN
        assert query_sent != LONG_FREE_TEXT_PROMPT
        # criteria text the derived query drops must still reach the LLM
        assert "official club announcements or BBC Sport" in result[1]
        assert result[4]["parse_tier"] == "clause"
        assert result[4]["scan_truncated"] is False

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_long_template_prompt_is_not_marked_truncated(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Template past the window is NOT flagged: it precedes the scan."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        _install_mock_client(mock_client_mgr)
        prompt = PREDICTION_PROMPT + " filler" * (module._MAX_SCAN_CHARS // 3)
        assert len(prompt) > module._MAX_SCAN_CHARS
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert result[4]["parse_tier"] == "template"
        assert result[4]["scan_truncated"] is False


FORECAST = {"p_yes": 0.19, "p_no": 0.81, "confidence": 0.7, "info_utility": 0.6}

SCAFFOLD_COMPLETION = (
    "<facts>\n- Isak handed in a transfer request on 2025-08-01.\n</facts>\n"
    "<no>\n- Newcastle publicly refused the bid (8/10).\n</no>\n"
    "<yes>\n- The player is pushing for the move (6/10).\n</yes>\n"
    "<thinking>\nThe deadline leaves little room for a new bid.\n</thinking>\n"
    "<tentative>\n0.22\n</tentative>\n"
    "<thinking>\nThe criterion needs a permanent transfer, not a loan.\n</thinking>\n"
    "<answer>\n*0.19*\n</answer>\n" + json.dumps(FORECAST)
)


class TestExtractPrediction:
    """extract_prediction() slices the forecast out of a scaffolded completion."""

    def test_bare_json_is_returned_unchanged(self) -> None:
        """A trader-template completion is already JSON and survives intact."""
        bare = json.dumps(FORECAST)
        assert json.loads(module.extract_prediction(bare) or "") == FORECAST

    def test_reasoning_scaffold_yields_only_the_forecast(self) -> None:
        """The seven-step block is dropped; only the JSON object is delivered."""
        out = module.extract_prediction(SCAFFOLD_COMPLETION) or ""
        assert json.loads(out) == FORECAST
        assert "<facts>" not in out
        assert "<answer>" not in out

    def test_a_draft_before_the_answer_does_not_shadow_it(self) -> None:
        """A drafted object earlier in the text loses to the final one."""
        completion = "Draft: " + json.dumps(
            {"p_yes": 0.8, "p_no": 0.2}
        ) + "\n" "On reflection that is too high.\n" + json.dumps(FORECAST)
        assert json.loads(module.extract_prediction(completion) or "") == FORECAST

    def test_trailing_non_forecast_object_is_skipped(self) -> None:
        """An object after the forecast with no p_yes is passed over."""
        completion = json.dumps(FORECAST) + '\nSources used: {"count": 3}'
        assert json.loads(module.extract_prediction(completion) or "") == FORECAST

    def test_a_cut_mid_object_returns_none(self) -> None:
        """A max_tokens cut while the answer is being written yields None."""
        completion = SCAFFOLD_COMPLETION[: -len(json.dumps(FORECAST))] + (
            '{"p_yes": 0.19, "p_no": 0.8'
        )
        assert module.extract_prediction(completion) is None

    def test_a_cut_whose_nested_object_closes_still_returns_none(self) -> None:
        """A closing brace belonging to a nested object is not the answer's."""
        completion = 'Answer:\n{"meta": {"model": "gpt-4o"}, "p_yes": 0.19'
        assert module.extract_prediction(completion) is None

    def test_a_draft_before_a_cut_is_not_delivered(self) -> None:
        """A tentative object written before a cut is a draft, not a forecast."""
        completion = json.dumps({"p_yes": 0.8, "p_no": 0.2}) + '\n{"p_yes": 0.19'
        assert module.extract_prediction(completion) is None

    def test_a_stray_brace_in_prose_is_not_a_cut(self) -> None:
        """An unclosed brace that does not open an object leaves the forecast."""
        completion = (
            "See the {source for details on the resolution criterion.\n"
            + json.dumps(FORECAST)
        )
        assert json.loads(module.extract_prediction(completion) or "") == FORECAST

    def test_balanced_non_json_braces_are_stepped_over(self) -> None:
        """Balanced braces that are not valid JSON do not stop the scan."""
        completion = "Use {curly braces} sparingly.\n" + json.dumps(FORECAST)
        assert json.loads(module.extract_prediction(completion) or "") == FORECAST

    def test_a_brace_inside_a_string_value_does_not_close_the_object(self) -> None:
        """A } inside a string value is skipped while scanning for the close."""
        payload = dict(FORECAST, note="resolves if } appears in the filing")
        completion = "<answer>\n*0.19*\n</answer>\n" + json.dumps(payload)
        assert json.loads(module.extract_prediction(completion) or "") == payload

    def test_escaped_quotes_inside_a_string_value_survive(self) -> None:
        """An escaped quote does not end the string the scanner is inside."""
        payload = dict(FORECAST, note='the club said "maybe" about the } bid')
        completion = "<thinking>\nWeighing the quote.\n</thinking>\n" + json.dumps(
            payload
        )
        assert json.loads(module.extract_prediction(completion) or "") == payload

    def test_out_of_range_p_yes_is_skipped(self) -> None:
        """A p_yes outside 0..1 is not a probability and loses to a valid one."""
        completion = json.dumps(FORECAST) + "\n" + json.dumps({"p_yes": 1.4})
        assert json.loads(module.extract_prediction(completion) or "") == FORECAST

    def test_null_p_yes_is_skipped(self) -> None:
        """A null p_yes carries no forecast and loses to a valid one."""
        completion = json.dumps(FORECAST) + "\n" + json.dumps({"p_yes": None})
        assert json.loads(module.extract_prediction(completion) or "") == FORECAST

    def test_non_numeric_p_yes_is_skipped(self) -> None:
        """A p_yes that will not coerce to a float loses to a valid one."""
        completion = json.dumps(FORECAST) + "\n" + json.dumps({"p_yes": "high"})
        assert json.loads(module.extract_prediction(completion) or "") == FORECAST

    def test_numeric_string_p_yes_is_accepted(self) -> None:
        """A p_yes quoted as a numeric string still counts as a forecast."""
        payload = dict(FORECAST, p_yes="0.19")
        out = module.extract_prediction(json.dumps(payload)) or ""
        assert float(json.loads(out)["p_yes"]) == 0.19

    def test_a_completion_with_no_forecast_returns_none(self) -> None:
        """Prose with no object at all is not handed on as the forecast."""
        # Returning the text put a result on-chain that json.loads rejects; the
        # caller's None guard turns this into a retry and then a typed null.
        assert module.extract_prediction("no json at all") is None

    def test_empty_and_missing_content_return_none(self) -> None:
        """An empty completion is None too, so the `is None` guard catches it."""
        assert module.extract_prediction(None) is None
        assert module.extract_prediction("") is None

    def test_a_cut_after_a_brace_in_a_string_value_returns_none(self) -> None:
        """A } inside a string must not make a truncated object look complete."""
        completion = '{"note": "resolves if } appears", "p_yes": 0.19'
        assert module.extract_prediction(completion) is None

    def test_a_cut_after_an_escaped_quote_returns_none(self) -> None:
        """An escaped quote must not end the string and expose a later }."""
        completion = '{"note": "he said \\"maybe}\\" indeed", "p_yes": 0.19'
        assert module.extract_prediction(completion) is None

    def test_a_completion_ending_on_the_opening_brace_is_a_cut(self) -> None:
        """A cut landing ON the brace must not deliver an earlier draft."""
        # tail is empty here, which the first version read as prose. On
        # pretty-printed JSON a cut after "{" is one of the likelier stops.
        content = '{"p_yes": 0.9, "p_no": 0.1}\n<answer>\n{'
        assert module.extract_prediction(content) is None

    def test_a_completion_ending_on_brace_plus_whitespace_is_a_cut(self) -> None:
        """Whitespace after the opening brace is still a cut, not prose."""
        content = '{"p_yes": 0.9, "p_no": 0.1}\n<answer>\n{\n '
        assert module.extract_prediction(content) is None

    def test_a_sole_out_of_range_forecast_is_not_delivered(self) -> None:
        """When the only object is out of range there is nothing to deliver."""
        # Returning the content here would put p_yes 1.7 on-chain as a normal
        # forecast; the caller's guard turns None into a retry instead.
        assert module.extract_prediction('{"p_yes": 1.7, "p_no": -0.7}') is None


REFUSAL = "I cannot help with that request."

# A max_tokens cut that lands in prose AFTER a complete draft object. Nothing is
# left unclosed, so no text heuristic can tell the draft from an answer.
DRAFT_THEN_CUT_IN_PROSE = (
    '<facts>x</facts>\n<thinking>\nDraft estimate {"p_yes": 0.35, "p_no": 0.65} '
    "seems too low, let me reconsider given the news that changes the probabi"
)


def _sdk_response(content: str, finish_reason: str = "stop") -> MagicMock:
    """Build a mock ``chat.completions.create`` response.

    :param content: the message content the SDK reports.
    :param finish_reason: why the provider stopped generating.
    :return: a MagicMock shaped like an openai ChatCompletion.
    """
    choice = MagicMock(finish_reason=finish_reason)
    choice.message.content = content
    return MagicMock(
        choices=[choice], usage=MagicMock(prompt_tokens=10, completion_tokens=5)
    )


class TestExtractPredictionRunWiring:
    """The extractor sits on run()'s delivery path, not only in a helper."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_run_delivers_parseable_json_for_a_scaffolded_completion(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """Unwiring the extractor makes run() deliver the scaffold verbatim."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        _install_mock_client(mock_client_mgr, content=SCAFFOLD_COMPLETION)
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert json.loads(result[0]) == FORECAST
        assert "<facts>" not in result[0]

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_run_does_not_deliver_a_draft_from_a_cut_completion(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A completion cut mid-answer yields a typed error null, not a draft."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        cut = json.dumps({"p_yes": 0.8, "p_no": 0.2}) + '\n{"p_yes": 0.19'
        _install_mock_client(mock_client_mgr, content=cut)
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        # Not a bare None: that would be delivered verbatim as the on-chain
        # result with no exception, so neither the retry loop nor
        # with_key_rotation's typed-null branch would run.
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["error_type"] == "Exception"
        assert "0.8" not in result[0]

    @patch(f"{SF_MODULE}.time.sleep", return_value=None)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_run_does_not_deliver_prose_as_the_result(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_sleep: MagicMock,
    ) -> None:
        """A completion with no forecast object is a typed error null, not text."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        _install_mock_client(mock_client_mgr, content=REFUSAL)
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert "cannot help" not in result[0]

    @patch(f"{SF_MODULE}.time.sleep", return_value=None)
    @patch(f"{SF_MODULE}.openai.OpenAI")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_run_does_not_deliver_a_draft_when_max_tokens_cut_the_prose(
        self,
        mock_fetch: MagicMock,
        mock_openai: MagicMock,
        _mock_sleep: MagicMock,
    ) -> None:
        """A cut after a complete draft is a typed null, raised on the first call."""
        # The extractor alone would deliver the 0.35 draft: only the provider's
        # finish_reason tells this cut apart from an answer.
        assert module.extract_prediction(DRAFT_THEN_CUT_IN_PROSE) is not None
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        create = mock_openai.return_value.chat.completions.create
        create.return_value = _sdk_response(DRAFT_THEN_CUT_IN_PROSE, "length")
        result = run(
            tool="superforcaster-polymarket-v1",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["error_type"] == "TruncatedCompletionError"
        assert "0.35" not in result[0]
        # The same budget cuts a retry the same way, so it is not retried.
        assert create.call_count == 1
