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


# ---------------------------------------------------------------------------
# v3-specific tests — verify the dual-SDK dispatch added in this variant.
# The tests above re-validate v1's surface (matching v2's pattern of
# inheriting tests from v1); v3 introduces real new code (LLMClientManager
# dispatch by model name), so it gets its own coverage here.
# ---------------------------------------------------------------------------

import packages.valory.customs.superforcaster_polymarket_v3.superforcaster_polymarket_v3 as v3_module  # noqa: E402
from packages.valory.customs.superforcaster_polymarket_v3.superforcaster_polymarket_v3 import (  # noqa: E402
    ALLOWED_MODELS,
    ALLOWED_TOOLS,
    DEFAULT_ANTHROPIC_MODEL,
    LLMClient,
    LLMClientManager,
    _provider_for,
)


def _make_v3_api_keys() -> MagicMock:
    """KeyChain-like mock carrying both openai + anthropic keys."""
    services = {
        "openai": "sk-test",
        "anthropic": "sk-ant-test",
        "serperapi": "serper-test",
        "return_source_content": "false",
        "source_content_mode": "cleaned",
        "openrouter": "",
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key]
    mock.get = lambda key, default="": services.get(key, default)
    mock.max_retries = lambda: {"openai": 0, "openrouter": 0, "anthropic": 0}
    return mock


class TestV3LLMClientManager:
    """Verify v3's dual-SDK dispatch picks the right provider by model name."""

    def test_provider_for_claude_model(self) -> None:
        """'claude' substring routes to the anthropic provider."""
        assert _provider_for("claude-fable-5") == "anthropic"
        assert _provider_for("claude-3-5-sonnet") == "anthropic"

    def test_provider_for_gpt_model(self) -> None:
        """Non-claude model strings fall back to the openai provider."""
        assert _provider_for("gpt-4.1-2025-04-14") == "openai"
        assert _provider_for("gpt-4o") == "openai"

    def test_manager_constructs_anthropic_client_for_claude(self) -> None:
        """LLMClientManager.__enter__ instantiates Anthropic when model is claude-*."""
        with patch(f"{v3_module.__name__}.Anthropic") as MockAnthropic:
            mock_instance = MagicMock()
            MockAnthropic.return_value = mock_instance
            keys = _make_v3_api_keys()
            mgr = LLMClientManager(keys, model="claude-fable-5")
            with mgr as client:
                assert mgr.provider == "anthropic"
                assert isinstance(client, LLMClient)
                MockAnthropic.assert_called_once_with(api_key="sk-ant-test")
            mock_instance.close.assert_called_once()

    def test_manager_constructs_openai_client_for_gpt(self) -> None:
        """LLMClientManager.__enter__ instantiates OpenAI when model is non-claude."""
        with patch(f"{v3_module.__name__}.openai.OpenAI") as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance
            keys = _make_v3_api_keys()
            mgr = LLMClientManager(keys, model="gpt-4.1-2025-04-14")
            with mgr as client:
                assert mgr.provider == "openai"
                assert isinstance(client, LLMClient)
                MockOpenAI.assert_called_once_with(api_key="sk-test")
            mock_instance.close.assert_called_once()

    def test_default_anthropic_model_is_fable_5(self) -> None:
        """The new default for v3 is claude-fable-5."""
        assert DEFAULT_ANTHROPIC_MODEL == "claude-fable-5"
        assert "claude-fable-5" in ALLOWED_MODELS

    def test_allowed_tools_targets_v3(self) -> None:
        """The wire-name for this variant is superforcaster-polymarket-v3."""
        assert ALLOWED_TOOLS == ["superforcaster-polymarket-v3"]
