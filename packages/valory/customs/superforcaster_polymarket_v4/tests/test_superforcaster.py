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

"""Unit tests for superforcaster: thread-safe client, offline tiktoken, source_content, market prior."""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import packages.valory.customs.superforcaster_polymarket_v4.superforcaster_polymarket_v4 as module
from packages.valory.customs.superforcaster_polymarket_v4.superforcaster_polymarket_v4 import (
    OpenAIClientManager,
    _extract_market_prob,
    format_market_prior,
    generate_prediction_with_retry,
    run,
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh OpenAIClient, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(
            "packages.valory.customs.superforcaster_polymarket_v4.superforcaster_polymarket_v4.OpenAIClient"
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
    "packages.valory.customs.superforcaster_polymarket_v4.superforcaster_polymarket_v4"
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
            tool="superforcaster-polymarket-v4",
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
            tool="superforcaster-polymarket-v4",
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
            tool="superforcaster-polymarket-v4",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        assert result[0] == PREDICTION_JSON
        used_params = result[4]
        assert "source_content" not in used_params


class TestFormatMarketPrior:
    """Verify the market-prior anchor block rendering and validation."""

    def test_valid_decimal_renders_anchor_with_number(self) -> None:
        """A valid price yields the anchor instruction containing that decimal."""
        block = format_market_prior(0.62)
        assert "0.62" in block
        assert "market-implied probability" in block
        # carries its own trailing blank line so it slots into the template
        assert block.endswith("\n\n")

    def test_string_numeric_is_accepted(self) -> None:
        """A numeric string (as it may arrive over the wire) is parsed."""
        assert "0.3" in format_market_prior("0.3")

    @pytest.mark.parametrize("bad", [None, "", "n/a", -0.1, 1.5])
    def test_absent_or_out_of_range_yields_empty(self, bad: object) -> None:
        """Missing, non-numeric, or out-of-[0,1] prices collapse to no anchor."""
        assert format_market_prior(bad) == ""

    def test_rounds_to_four_decimals(self) -> None:
        """Long decimals are rounded so the prompt stays clean."""
        assert "0.6667" in format_market_prior(0.6666666)


class TestExtractMarketProb:
    """Verify market-prob extraction from kwargs and request_context."""

    def test_top_level_key(self) -> None:
        """A top-level market_prob is found."""
        assert _extract_market_prob({"market_prob": 0.42}) == 0.42

    def test_request_context_key(self) -> None:
        """A price nested in request_context is found."""
        assert _extract_market_prob({"request_context": {"current_prob": 0.7}}) == 0.7

    def test_top_level_takes_precedence(self) -> None:
        """Top-level kwargs win over request_context for the same concept."""
        kwargs = {
            "market_prob": 0.1,
            "request_context": {"market_prob_at_prediction": 0.9},
        }
        assert _extract_market_prob(kwargs) == 0.1

    def test_absent_returns_none(self) -> None:
        """No recognised key anywhere returns None."""
        assert _extract_market_prob({"foo": "bar"}) is None


class TestMarketPriorInPrompt:
    """End-to-end: the rendered anchor block reaches the formatted prompt."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_price_present_injects_anchor(self, mock_client_mgr: MagicMock) -> None:
        """A supplied market_prob places the anchor sentence into the prompt."""
        _install_mock_client(mock_client_mgr)

        result = run(
            tool="superforcaster-polymarket-v4",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
            market_prob=0.62,
        )

        prediction_prompt = result[1]
        assert (
            "market-implied probability that the answer is YES is 0.62"
            in prediction_prompt
        )

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_price_absent_no_anchor(self, mock_client_mgr: MagicMock) -> None:
        """Without a price the prompt has no anchor block (degrades to v1)."""
        _install_mock_client(mock_client_mgr)

        result = run(
            tool="superforcaster-polymarket-v4",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )

        prediction_prompt = result[1]
        assert "market-implied probability" not in prediction_prompt
