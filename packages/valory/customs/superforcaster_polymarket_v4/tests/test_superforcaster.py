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

"""Unit tests for superforcaster_polymarket v4.

v4 is a one-axis model swap of v2 (gpt-4.1 -> gpt-5.4). The first half of
this file re-validates v2's shared surface (client manager, source_content
capture/replay) against v4's own module. The classes after the
``# v4-specific tests`` divider pin the reasoning-model adaptations that v4
introduces: max_completion_tokens + reasoning_effort wiring, the
finish_reason truncation guard, the empty-content guard, the tiktoken
fallback for gpt-5.4, and the ALLOWED_MODELS enforcement.
"""

import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import packages.valory.customs.superforcaster_polymarket_v4.superforcaster_polymarket_v4 as module
from packages.valory.customs.superforcaster_polymarket_v4.superforcaster_polymarket_v4 import (
    ALLOWED_MODELS,
    ALLOWED_REASONING_EFFORTS,
    ALLOWED_TOOLS,
    DEFAULT_OPENAI_MODEL,
    OpenAIClient,
    OpenAIClientManager,
    OpenAIResponse,
    count_tokens,
    generate_prediction_with_retry,
    run,
)

SF_MODULE = (
    "packages.valory.customs.superforcaster_polymarket_v4.superforcaster_polymarket_v4"
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh OpenAIClient, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(f"{SF_MODULE}.OpenAIClient") as MockClient:
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
            model="gpt-5.4",
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
            model="gpt-5.4",
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
            model="gpt-5.4",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        assert result[0] == PREDICTION_JSON
        used_params = result[4]
        assert "source_content" not in used_params


# ---------------------------------------------------------------------------
# v4-specific tests — verify the gpt-5.4 reasoning-model adaptations.
# ---------------------------------------------------------------------------


def _make_openai_chat_response(
    content: Any,
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 7,
    completion_tokens: int = 13,
) -> MagicMock:
    """Build a mock ``chat.completions.create`` response.

    Shapes it like the real ``ChatCompletion``: ``choices[0].message.content``,
    ``choices[0].finish_reason`` and a ``usage`` block.

    :param content: the visible message content (``None`` for an empty
        reasoning-only response).
    :param finish_reason: ``"stop"`` (normal) or ``"length"`` (truncated).
    :param prompt_tokens: prompt token count for the usage block.
    :param completion_tokens: completion token count for the usage block.
    :return: a MagicMock shaped like ``openai.types.ChatCompletion``.
    """
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return resp


def _client_with_chat_response(resp: Any) -> OpenAIClient:
    """Build a v4 OpenAIClient with a mocked openai.OpenAI backing."""
    with patch(f"{SF_MODULE}.openai.OpenAI") as MockOpenAI:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = resp
        MockOpenAI.return_value = mock_instance
        client = OpenAIClient(api_key="sk-test")
    return client


class TestV4Constants:
    """Pin v4's wire-name, model allow-list and reasoning-effort menu."""

    def test_default_model_is_gpt_5_4(self) -> None:
        """The new default for v4 is gpt-5.4 and it's the only allowed model."""
        assert DEFAULT_OPENAI_MODEL == "gpt-5.4"
        assert ALLOWED_MODELS == ["gpt-5.4"]

    def test_allowed_tools_targets_v4(self) -> None:
        """The wire-name for this variant is superforcaster-polymarket-v4."""
        assert ALLOWED_TOOLS == ["superforcaster-polymarket-v4"]

    def test_reasoning_effort_menu(self) -> None:
        """ALLOWED_REASONING_EFFORTS matches the gpt-5.4 effort levels."""
        assert ALLOWED_REASONING_EFFORTS == [
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
        ]


class TestV4CountTokensFallback:
    """tiktoken 0.12.0 has no gpt-5.4 mapping; count_tokens must not crash."""

    def test_count_tokens_falls_back_for_gpt_5_4(self) -> None:
        """``count_tokens(text, "gpt-5.4")`` falls back to o200k_base, not KeyError."""
        # Sanity: tiktoken genuinely lacks the gpt-5.4 mapping (otherwise this
        # test would pass without exercising the fallback branch).
        import tiktoken

        with pytest.raises(KeyError):
            tiktoken.encoding_for_model("gpt-5.4")

        n = count_tokens("hello world", "gpt-5.4")
        assert n == len(tiktoken.get_encoding("o200k_base").encode("hello world"))
        assert n > 0

    def test_count_tokens_uses_native_mapping_when_present(self) -> None:
        """A model tiktoken knows still uses its native encoding."""
        import tiktoken

        n = count_tokens("hello world", "gpt-4o")
        assert n == len(tiktoken.encoding_for_model("gpt-4o").encode("hello world"))


class TestV4OpenAICompletions:
    """Pin the gpt-5.4 reasoning-model call contract in OpenAIClient.completions."""

    def test_sends_reasoning_params_not_legacy_params(self) -> None:
        """completions() sends max_completion_tokens + reasoning_effort, NOT temperature/max_tokens/top_p."""
        client = _client_with_chat_response(_make_openai_chat_response(PREDICTION_JSON))
        client.completions(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "U1"}],
            reasoning_effort="medium",
            max_completion_tokens=16000,
        )
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["max_completion_tokens"] == 16000
        assert kwargs["reasoning_effort"] == "medium"
        assert kwargs["model"] == "gpt-5.4"
        # Reasoning models reject these — they must never be forwarded.
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "max_tokens" not in kwargs

    def test_truncation_raises(self) -> None:
        """finish_reason == 'length' raises so the retry loop engages."""
        client = _client_with_chat_response(
            _make_openai_chat_response('{"p_yes": 0.5', finish_reason="length")
        )
        with pytest.raises(ValueError, match="Response truncated"):
            client.completions(
                model="gpt-5.4",
                messages=[{"role": "user", "content": "U1"}],
                max_completion_tokens=10,
            )

    def test_returns_content_and_usage_on_success(self) -> None:
        """A normal response carries content + usage through unchanged."""
        client = _client_with_chat_response(
            _make_openai_chat_response(
                PREDICTION_JSON, prompt_tokens=111, completion_tokens=222
            )
        )
        result = client.completions(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "U1"}],
        )
        assert isinstance(result, OpenAIResponse)
        assert result.content == PREDICTION_JSON
        assert result.usage.prompt_tokens == 111
        assert result.usage.completion_tokens == 222


class TestV4GeneratePredictionWithRetry:
    """The empty-content guard added for reasoning models."""

    def test_empty_content_engages_retry_then_fails(self) -> None:
        """``content is None`` raises inside the loop; exhausting retries raises a clean error."""
        mock_client = MagicMock()
        # client.completions() returns an OpenAIResponse-shaped object whose
        # .content is None (the reasoning-only / empty-output case).
        empty = MagicMock()
        empty.content = None
        mock_client.completions.return_value = empty

        with patch(f"{SF_MODULE}.time.sleep"):
            with pytest.raises(Exception, match="Failed to generate prediction"):
                generate_prediction_with_retry(
                    client=mock_client,
                    model="gpt-5.4",
                    messages=[{"role": "user", "content": "U1"}],
                    reasoning_effort="medium",
                    max_completion_tokens=16000,
                    retries=2,
                    delay=0,
                    counter_callback=None,
                )
        # Both attempts ran (the None content didn't short-circuit as success).
        assert mock_client.completions.call_count == 2

    def test_threads_reasoning_params_into_completions(self) -> None:
        """reasoning_effort + max_completion_tokens reach the client unchanged."""
        mock_client = MagicMock()
        ok = MagicMock()
        ok.content = PREDICTION_JSON
        ok.usage.prompt_tokens = 10
        ok.usage.completion_tokens = 5
        mock_client.completions.return_value = ok
        content, _ = generate_prediction_with_retry(
            client=mock_client,
            model="gpt-5.4",
            messages=[{"role": "user", "content": "U1"}],
            reasoning_effort="high",
            max_completion_tokens=12345,
            retries=1,
            delay=0,
            counter_callback=None,
        )
        assert content == PREDICTION_JSON
        kwargs = mock_client.completions.call_args.kwargs
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["max_completion_tokens"] == 12345


class TestV4RunEndToEnd:
    """End-to-end ``run()`` smoke tests on v4 with gpt-5.4."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_run_records_reasoning_params_and_default_effort(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """run() defaults to reasoning_effort='medium' and records the reasoning params."""
        mock_serper = MagicMock()
        mock_serper.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_serper
        mock_client = _install_mock_client(mock_client_mgr)

        result = run(
            tool="superforcaster-polymarket-v4",
            model="gpt-5.4",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        assert result[0] == PREDICTION_JSON
        used_params = result[4]
        assert used_params["model"] == "gpt-5.4"
        assert used_params["reasoning_effort"] == "medium"
        assert used_params["max_completion_tokens"] == 16000
        # The default effort + budget were forwarded to the client.
        call_kwargs = mock_client.completions.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "medium"
        assert call_kwargs["max_completion_tokens"] == 16000

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_run_honors_reasoning_effort_override(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A caller-supplied reasoning_effort overrides the default."""
        mock_serper = MagicMock()
        mock_serper.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_serper
        mock_client = _install_mock_client(mock_client_mgr)

        run(
            tool="superforcaster-polymarket-v4",
            model="gpt-5.4",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            reasoning_effort="high",
        )

        assert mock_client.completions.call_args.kwargs["reasoning_effort"] == "high"

    def test_run_with_unknown_model_returns_allowed_models_error(self) -> None:
        """A model outside ALLOWED_MODELS produces a clean error tuple, not an SDK error.

        ``@with_key_rotation`` catches the inner ``ValueError`` and wraps it
        into the standard mech response tuple, so we assert on the first
        element rather than ``pytest.raises``.
        """
        result = run(
            tool="superforcaster-polymarket-v4",
            model="gpt-4.1-2025-04-14",
            prompt="P",
            api_keys=_make_mock_api_keys("false"),
            delivery_rate=10000,
        )
        assert "ALLOWED_MODELS" in result[0]
        assert "gpt-4.1-2025-04-14" in result[0]

    def test_run_with_invalid_reasoning_effort_returns_error(self) -> None:
        """An out-of-menu reasoning_effort is rejected with a clean error tuple."""
        result = run(
            tool="superforcaster-polymarket-v4",
            model="gpt-5.4",
            prompt="P",
            api_keys=_make_mock_api_keys("false"),
            delivery_rate=10000,
            reasoning_effort="ultra",
        )
        assert "reasoning_effort" in result[0]
        assert "ultra" in result[0]
