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

"""Unit tests for superforcaster_polymarket v3.

The first half of this file (``TestOpenAIClientManager``,
``TestSuperforcasterSourceContent``) imports from v1 and intentionally
re-validates v1's surface — v3 is a one-axis swap of v1, so any
regression in the shared code surfaces here. v3-only behaviour
(``LLMClientManager`` dispatch by model name, the Anthropic branch of
``LLMClient.completions``, the key-rotation decorator's anthropic
branch, the v3 ``run()`` end-to-end) is exercised by the classes after
the ``# v3-specific tests`` divider.
"""

import inspect
import json
from pathlib import Path
from typing import Any
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


def _make_anthropic_text_response(
    text: str,
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 7,
    output_tokens: int = 13,
) -> MagicMock:
    """Build a mock anthropic ``Messages.create`` response.

    Shapes the response like the real ``anthropic.types.Message``: a
    ``content`` list of blocks (where each has a ``type`` and either
    ``text`` or thinking content), ``stop_reason``, and ``usage`` with
    ``input_tokens`` / ``output_tokens``.

    :param text: the JSON string the TextBlock returns.
    :param stop_reason: ``"end_turn"`` (normal), ``"max_tokens"`` (trip
        the truncation guard), ``"refusal"``, etc.
    :param input_tokens: input token count for the usage block.
    :param output_tokens: output token count for the usage block.
    :return: a MagicMock shaped like ``anthropic.types.Message``.
    """
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response = MagicMock()
    response.content = [text_block]
    response.stop_reason = stop_reason
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


def _make_anthropic_error(cls: type, message: str = "simulated") -> Exception:
    """Build an anthropic APIStatusError-family exception for tests.

    Skips the real constructor (which requires a live httpx.Response) and
    sets up just the attributes the decorator under test touches.

    :param cls: the anthropic exception subclass to instantiate.
    :param message: the `str(exc)` payload.
    :return: an instance of `cls` usable as a raise target in tests.
    """
    err: Exception = cls.__new__(cls)  # type: ignore[call-overload]
    Exception.__init__(err, message)
    err.message = message  # type: ignore[attr-defined]
    return err


class TestV3LLMClientAnthropicCompletions:
    """Coverage for the Anthropic branch of ``LLMClient.completions``.

    The class above (``TestV3LLMClientManager``) covers dispatch wiring
    only. These tests pin the Anthropic-side contract end-to-end:

    - ``system`` messages are extracted out of the list and joined.
    - ``temperature=0`` is NOT forwarded (fable-5 returns 400 on it).
    - ``max_tokens`` defaults to 4096 on the Anthropic branch (not v1's
      OpenAI default of 500 — that truncated almost every fable-5 call).
    - ``stop_reason == "max_tokens"`` raises ValueError so the caller's
      retry loop engages instead of returning truncated JSON silently.
    - No ``TextBlock`` raises ValueError for the same reason.
    - ``ThinkingBlock`` is skipped — only the first ``TextBlock`` parses.
    - ``counter_callback`` receives the Anthropic ``input_tokens`` /
      ``output_tokens`` attribute names, NOT OpenAI's
      ``prompt_tokens`` / ``completion_tokens``.
    """

    @staticmethod
    def _client_with_anthropic_response(resp: Any) -> LLMClient:
        """Build a v3 LLMClient with a mocked anthropic.Anthropic backing."""
        with patch(f"{v3_module.__name__}.Anthropic") as MockAnthropic:
            mock_instance = MagicMock()
            mock_instance.messages.create.return_value = resp
            MockAnthropic.return_value = mock_instance
            client = LLMClient(api_keys=_make_v3_api_keys(), model="claude-fable-5")
        return client

    def test_init_raises_keyerror_when_keychain_lacks_anthropic_key(self) -> None:
        """A v1-era keychain (no anthropic key) raises KeyError in LLMClient.__init__.

        Pins the construction-time failure mode flagged on PR #340: the real
        ``KeyError: 'anthropic'`` for a deployment that hasn't been
        provisioned with an anthropic key fires here, BEFORE any rotation
        path runs. The decorator's bare ``except Exception`` then wraps it
        into a result tuple whose first element is the stringified KeyError.
        """

        class _KeychainMissingAnthropic:
            """KeyChain stand-in shaped like v1's (no ``anthropic`` entry)."""

            def __getitem__(self, key: str) -> str:
                if key == "anthropic":
                    raise KeyError(key)
                return "sk-test"

        with pytest.raises(KeyError, match="anthropic"):
            LLMClient(
                api_keys=_KeychainMissingAnthropic(),
                model="claude-fable-5",
            )

    def test_extracts_system_messages_out(self) -> None:
        """``system`` entries are joined and passed via ``system=``, not ``messages=``."""
        client = self._client_with_anthropic_response(
            _make_anthropic_text_response('{"p_yes": 0.5}')
        )
        client.completions(
            model="claude-fable-5",
            messages=[
                {"role": "system", "content": "SYS-A"},
                {"role": "user", "content": "U1"},
                {"role": "system", "content": "SYS-B"},
            ],
        )
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["system"] == "SYS-A\n\nSYS-B"
        assert kwargs["messages"] == [{"role": "user", "content": "U1"}]

    def test_temperature_zero_not_forwarded(self) -> None:
        """``temperature=0`` is dropped (fable-5 rejects it with HTTP 400)."""
        client = self._client_with_anthropic_response(
            _make_anthropic_text_response('{"p_yes": 0.5}')
        )
        client.completions(
            model="claude-fable-5",
            messages=[{"role": "user", "content": "U1"}],
            temperature=0,
        )
        kwargs = client.client.messages.create.call_args.kwargs
        assert "temperature" not in kwargs

    def test_default_max_tokens_is_4096_on_anthropic_branch(self) -> None:
        """Default ``max_tokens`` is 4096 on the Anthropic branch, not v1's 500."""
        client = self._client_with_anthropic_response(
            _make_anthropic_text_response('{"p_yes": 0.5}')
        )
        client.completions(
            model="claude-fable-5",
            messages=[{"role": "user", "content": "U1"}],
            # max_tokens not supplied → expect 4096 default.
        )
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 4096

    def test_max_tokens_truncation_raises(self) -> None:
        """``stop_reason == "max_tokens"`` raises so the retry loop engages."""
        client = self._client_with_anthropic_response(
            _make_anthropic_text_response('{"p_yes": 0.5', stop_reason="max_tokens")
        )
        with pytest.raises(ValueError, match="Response truncated"):
            client.completions(
                model="claude-fable-5",
                messages=[{"role": "user", "content": "U1"}],
                max_tokens=10,
            )

    def test_no_text_block_raises(self) -> None:
        """A response with no ``TextBlock`` raises so retry engages."""
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "internal monologue only"
        resp = MagicMock()
        resp.content = [thinking_block]
        resp.stop_reason = "end_turn"
        resp.usage = MagicMock(input_tokens=5, output_tokens=10)
        client = self._client_with_anthropic_response(resp)
        with pytest.raises(ValueError, match="no text block"):
            client.completions(
                model="claude-fable-5",
                messages=[{"role": "user", "content": "U1"}],
            )

    def test_thinking_block_is_skipped(self) -> None:
        """A ThinkingBlock-then-TextBlock response returns the TextBlock text."""
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "internal monologue"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '{"p_yes": 0.7, "p_no": 0.3}'
        resp = MagicMock()
        resp.content = [thinking_block, text_block]
        resp.stop_reason = "end_turn"
        resp.usage = MagicMock(input_tokens=5, output_tokens=10)
        client = self._client_with_anthropic_response(resp)
        result = client.completions(
            model="claude-fable-5",
            messages=[{"role": "user", "content": "U1"}],
        )
        assert result is not None
        assert result.content == '{"p_yes": 0.7, "p_no": 0.3}'

    def test_usage_uses_anthropic_token_names(self) -> None:
        """``usage.prompt_tokens``/``completion_tokens`` come from Anthropic's ``input_tokens``/``output_tokens``."""
        client = self._client_with_anthropic_response(
            _make_anthropic_text_response(
                '{"p_yes": 0.5}', input_tokens=111, output_tokens=222
            )
        )
        result = client.completions(
            model="claude-fable-5",
            messages=[{"role": "user", "content": "U1"}],
        )
        assert result is not None
        assert result.usage.prompt_tokens == 111
        assert result.usage.completion_tokens == 222


class TestV3WithKeyRotationAnthropic:
    """Anthropic-side coverage for v3's ``@with_key_rotation`` decorator.

    The existing OpenAI-only rotation tests on the v1-inherited test
    surface don't exercise the v3 dispatch. These pin:

    - ``anthropic.RateLimitError`` rotates ONLY the anthropic pool (not
      openai/openrouter), regressing the cross-pool waste bug.
    - An ``openai.RateLimitError`` doesn't burn anthropic budget.
    - Anthropic-pool exhausted re-raises so the framework marks the
      task failed.
    - Older keychains without an ``anthropic`` entry don't crash on
      dict lookup; the call re-raises cleanly.
    """

    def test_rotates_anthropic_pool_on_rate_limit(self) -> None:
        """``anthropic.RateLimitError`` rotates ONLY the anthropic key."""
        keys = _make_v3_api_keys()
        keys.max_retries = lambda: {
            "openai": 5,
            "openrouter": 5,
            "anthropic": 1,
        }
        keys.rotate = MagicMock()
        call_count = {"n": 0}

        @v3_module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _make_anthropic_error(
                    v3_module.anthropic.RateLimitError, "anthropic-burst"
                )
            return "ok", "", None, None, None

        result = fake(api_keys=keys)
        assert call_count["n"] == 2
        rotated_services = [c.args[0] for c in keys.rotate.call_args_list]
        assert rotated_services == ["anthropic"]
        assert result[-1] is keys

    def test_openai_error_does_not_burn_anthropic_budget(self) -> None:
        """Cross-pool isolation — an openai error leaves anthropic budget intact."""
        keys = _make_v3_api_keys()
        keys.max_retries = lambda: {
            "openai": 1,
            "openrouter": 1,
            "anthropic": 1,
        }
        keys.rotate = MagicMock()
        call_count = {"n": 0}

        @v3_module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First failure: openai-side rate limit. anthropic budget
                # must still be 1 after this.
                raise _make_anthropic_error(
                    v3_module.openai.RateLimitError, "openai-burst"
                )
            if call_count["n"] == 2:
                # Second failure: anthropic-side. Must be able to rotate
                # the anthropic key because the budget wasn't burned.
                raise _make_anthropic_error(
                    v3_module.anthropic.RateLimitError, "anthropic-burst"
                )
            return "ok", "", None, None, None

        result = fake(api_keys=keys)
        assert call_count["n"] == 3
        rotated_services = [c.args[0] for c in keys.rotate.call_args_list]
        assert rotated_services == ["openai", "openrouter", "anthropic"]
        assert result[-1] is keys

    def test_anthropic_pool_exhausted_returns_error_tuple(self) -> None:
        """When the anthropic pool is exhausted, the error is wrapped — same as bare except."""
        keys = _make_v3_api_keys()
        keys.max_retries = lambda: {
            "openai": 5,
            "openrouter": 5,
            "anthropic": 0,
        }

        @v3_module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            raise _make_anthropic_error(
                v3_module.anthropic.RateLimitError, "anthropic-burned"
            )

        result = fake(api_keys=keys)
        assert "anthropic-burned" in result[0]
        assert result[1:] == ("", None, None, None, keys)

    def test_missing_anthropic_in_retries_left_does_not_crash_rotation(self) -> None:
        """Older ``max_retries()`` without ``anthropic`` doesn't crash the rotation lookup.

        Scoped to the rotation path's ``retries_left`` lookup ONLY. The
        production-path ``KeyError: 'anthropic'`` from a v1-era keychain
        actually fires earlier in ``LLMClient.__init__`` (where
        ``self.api_keys["anthropic"]`` runs before any rotation) — pinned
        by ``test_init_raises_keyerror_when_keychain_lacks_anthropic_key``
        below. This test just confirms the ``setdefault`` guard in the
        rotation decorator does its job and the wrapped error tuple still
        carries the original message.
        """
        keys = _make_v3_api_keys()
        keys.max_retries = lambda: {"openai": 5, "openrouter": 5}

        @v3_module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            raise _make_anthropic_error(
                v3_module.anthropic.RateLimitError, "anthropic-burst"
            )

        result = fake(api_keys=keys)
        assert "anthropic-burst" in result[0]
        assert result[1:] == ("", None, None, None, keys)


class TestV3RunEndToEnd:
    """End-to-end ``run()`` smoke tests on v3 with ``claude-fable-5``.

    The v1-inherited tests above never call v3's ``run()`` with the new
    default model, so a wrong-API-key wiring bug would ship undetected.
    These tests patch ``LLMClientManager`` to a static prediction-JSON
    return and verify ``run()`` produces a valid prediction with both
    ``model="claude-fable-5"`` and the ALLOWED_MODELS enforcement.
    """

    PREDICTION_JSON = (
        '{"p_yes": 0.6, "p_no": 0.4, "confidence": 0.8, "info_utility": 0.6}'
    )

    def test_run_with_claude_fable_5_returns_valid_prediction(self) -> None:
        """``run(model="claude-fable-5", …)`` produces a valid prediction JSON."""
        prompt = (
            'With the given question "Will X happen?" and the `yes` option '
            "represented by `Yes` and the `no` option represented by `No`, "
            "what are the respective probabilities of `p_yes` and `p_no`?"
        )

        mock_response = MagicMock()
        mock_response.content = self.PREDICTION_JSON
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        with (
            patch(f"{v3_module.__name__}.LLMClientManager") as MockManager,
            patch(f"{v3_module.__name__}.fetch_additional_sources") as MockFetchSources,
        ):
            mock_llm_client = MagicMock()
            mock_llm_client.completions.return_value = mock_response
            MockManager.return_value.__enter__.return_value = mock_llm_client
            MockManager.return_value.__exit__.return_value = None
            MockFetchSources.return_value = MagicMock(
                json=MagicMock(return_value={"organic": []})
            )

            result = v3_module.run(
                tool="superforcaster-polymarket-v3",
                model="claude-fable-5",
                prompt=prompt,
                api_keys=_make_v3_api_keys(),
                delivery_rate=10000,
            )

        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.6
        assert parsed["confidence"] == 0.8
        # Verify the dispatcher actually selected the anthropic branch
        # (constructed LLMClientManager with model="claude-fable-5").
        construct_call = MockManager.call_args
        assert construct_call.args[1] == "claude-fable-5"
        # Regression for the 4096-default bug: run() must forward
        # max_tokens=4096 on the Anthropic branch (NOT the OpenAI default
        # of 500 that v1's settings hard-coded). Without this guard the
        # ``or 4096`` fallback in LLMClient.completions() resolves to
        # ``500 or 4096 -> 500`` and fable-5 truncates almost every call.
        completions_call = mock_llm_client.completions.call_args
        assert completions_call.kwargs.get("max_tokens") == 4096

    def test_run_with_unknown_model_returns_allowed_models_error(self) -> None:
        """``model`` outside ``ALLOWED_MODELS`` produces a clean error tuple, not an SDK error.

        ``@with_key_rotation`` catches the inner ``ValueError`` and wraps it
        into the standard mech response tuple, so we assert on the first
        element rather than ``pytest.raises``. Without the ALLOWED_MODELS
        guard the failure would surface deep in the SDK with an opaque
        error string like ``model: claude-typo-3 not found``.
        """
        result = v3_module.run(
            tool="superforcaster-polymarket-v3",
            model="claude-typo-3",
            prompt="P",
            api_keys=_make_v3_api_keys(),
            delivery_rate=10000,
        )
        assert "ALLOWED_MODELS" in result[0]
        assert "claude-typo-3" in result[0]
