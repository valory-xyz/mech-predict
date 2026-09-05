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

"""Unit tests for prediction_request_reasoning: thread-safe client, offline tiktoken, and source_content."""

import inspect
import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

import packages.napthaai.customs.prediction_request_reasoning_v1.prediction_request_reasoning_v1 as module
from packages.napthaai.customs.prediction_request_reasoning_v1.prediction_request_reasoning_v1 import (
    ExtendedDocument,
    LLMClientManager,
    count_tokens,
    do_reasoning_with_retry,
    extract_texts,
    fetch_additional_information,
    get_urls_from_queries_serper,
    multi_queries,
    multi_questions_response,
    parse_prompt,
    run,
)


class TestLLMClientManager:
    """Verify LLMClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_tuple(self) -> None:
        """__enter__ returns a (client, client_embedding) tuple."""
        mock_keys = {"openai": "sk-test"}
        mgr = LLMClientManager(
            api_keys=mock_keys, model="gpt-4.1-2025-04-14", embedding_provider="openai"
        )
        with patch(
            "packages.napthaai.customs.prediction_request_reasoning_v1.prediction_request_reasoning_v1.LLMClient"
        ) as MockClient:
            mock_llm = MagicMock(name="llm")
            mock_embed = MagicMock(name="embed")
            MockClient.side_effect = [mock_llm, mock_embed]

            with mgr as (llm_client, embedding_client):
                assert llm_client is mock_llm
                assert embedding_client is mock_embed

    def test_no_global_client_variable(self) -> None:
        """The module must not define module-level client variables."""
        source = Path(module.__file__).read_text(encoding="utf-8")
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("client:") or stripped.startswith("client ="):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client' variable found at line {i}: {line}"
                    )
            if stripped.startswith("client_embedding:") or stripped.startswith(
                "client_embedding ="
            ):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client_embedding' variable found at line {i}: {line}"
                    )


class TestFunctionsAcceptClient:
    """Verify refactored functions accept client as an explicit parameter."""

    def test_count_tokens_without_client_uses_tiktoken(self) -> None:
        """count_tokens falls back to tiktoken when client is None."""
        token_count = count_tokens("hello world", "gpt-4o-2024-08-06")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_claude_without_client_uses_fallback(self) -> None:
        """count_tokens for Claude models without client uses cl100k_base fallback."""
        token_count = count_tokens("hello world", "claude-sonnet-4-6")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_multi_questions_response_requires_client_param(self) -> None:
        """multi_questions_response requires client as first param."""
        params = list(inspect.signature(multi_questions_response).parameters)
        assert params[0] == "client"

    def test_do_reasoning_requires_client_param(self) -> None:
        """do_reasoning_with_retry requires client as first param."""
        params = list(inspect.signature(do_reasoning_with_retry).parameters)
        assert params[0] == "client"

    def test_fetch_additional_information_requires_client_param(self) -> None:
        """fetch_additional_information requires client as first param."""
        params = list(inspect.signature(fetch_additional_information).parameters)
        assert params[0] == "client"


REASONING_MODULE = "packages.napthaai.customs.prediction_request_reasoning_v1.prediction_request_reasoning_v1"


def _make_html_future(url: str, html: str) -> tuple:
    """Create a (future, url) pair with a fake HTML response."""
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.text = html
    response.content = b"<html>"
    future: Future = Future()
    future.set_result(response)
    return (future, url)


def _make_pdf_future(url: str) -> tuple:
    """Create a (future, url) pair with a fake PDF response."""
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.text = ""
    response.content = b"%PDF-1.4 fake content"
    future: Future = Future()
    future.set_result(response)
    return (future, url)


class TestExtractTextsCapture:
    """Verify extract_texts captures raw source content correctly."""

    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_cleaned_mode_stores_extracted_text(self, mock_batches: MagicMock) -> None:
        """In cleaned mode (default), extracted text is stored instead of raw HTML."""
        html = "<html><body>Hello world</body></html>"
        mock_batches.return_value = [[_make_html_future("http://example.com", html)]]

        _, raw_sc = extract_texts(["http://example.com"])

        assert raw_sc["mode"] == "cleaned"
        assert "http://example.com" in raw_sc["pages"]
        assert raw_sc["pages"]["http://example.com"] != html
        assert "Hello world" in raw_sc["pages"]["http://example.com"]
        assert not raw_sc["pdfs"]

    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_raw_mode_stores_html(self, mock_batches: MagicMock) -> None:
        """In raw mode, raw HTML is stored."""
        html = "<html><body>Hello world</body></html>"
        mock_batches.return_value = [[_make_html_future("http://example.com", html)]]

        _, raw_sc = extract_texts(["http://example.com"], source_content_mode="raw")

        assert raw_sc["mode"] == "raw"
        assert raw_sc["pages"]["http://example.com"] == html

    @patch(f"{REASONING_MODULE}.extract_text_from_pdf")
    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_pdf_captured(
        self, mock_batches: MagicMock, mock_pdf_extract: MagicMock
    ) -> None:
        """PDF responses are stored in raw_source_content['pdfs']."""
        mock_batches.return_value = [[_make_pdf_future("http://example.com/doc.pdf")]]
        mock_pdf_extract.return_value = ExtendedDocument(
            text="pdf content", url="http://example.com/doc.pdf"
        )

        _, raw_sc = extract_texts(["http://example.com/doc.pdf"])

        assert "http://example.com/doc.pdf" in raw_sc["pdfs"]
        assert raw_sc["pdfs"]["http://example.com/doc.pdf"] == "pdf content"
        assert not raw_sc["pages"]

    @patch(f"{REASONING_MODULE}.extract_text_from_pdf")
    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_failed_pdf_stores_empty_string(
        self, mock_batches: MagicMock, mock_pdf_extract: MagicMock
    ) -> None:
        """When extract_text_from_pdf returns None, empty string is stored."""
        mock_batches.return_value = [[_make_pdf_future("http://example.com/doc.pdf")]]
        mock_pdf_extract.return_value = None

        _, raw_sc = extract_texts(["http://example.com/doc.pdf"])

        assert raw_sc["pdfs"]["http://example.com/doc.pdf"] == ""

    @patch(f"{REASONING_MODULE}.extract_text_from_pdf")
    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_mixed_html_and_pdf(
        self, mock_batches: MagicMock, mock_pdf_extract: MagicMock
    ) -> None:
        """Both HTML and PDF are captured in their respective keys."""
        html = "<html><body>page</body></html>"
        mock_batches.return_value = [
            [
                _make_html_future("http://example.com", html),
                _make_pdf_future("http://example.com/doc.pdf"),
            ]
        ]
        mock_pdf_extract.return_value = ExtendedDocument(
            text="pdf text", url="http://example.com/doc.pdf"
        )

        _, raw_sc = extract_texts(["http://example.com", "http://example.com/doc.pdf"])

        assert "http://example.com" in raw_sc["pages"]
        assert "http://example.com/doc.pdf" in raw_sc["pdfs"]

    @patch(f"{REASONING_MODULE}.process_in_batches")
    def test_non_200_not_captured(self, mock_batches: MagicMock) -> None:
        """Non-200 responses are not stored in raw_source_content."""
        response = MagicMock(spec=requests.Response)
        response.status_code = 404
        future: Future = Future()
        future.set_result(response)
        mock_batches.return_value = [[(future, "http://example.com")]]

        _, raw_sc = extract_texts(["http://example.com"])

        assert not raw_sc["pages"]
        assert not raw_sc["pdfs"]


class TestFetchReplayPath:
    """Verify fetch_additional_information replays from structured source_content."""

    @patch(f"{REASONING_MODULE}.reciprocal_rank_refusion")
    @patch(f"{REASONING_MODULE}.find_similar_chunks")
    @patch(f"{REASONING_MODULE}.get_embeddings")
    @patch(f"{REASONING_MODULE}.multi_questions_response")
    @patch(f"{REASONING_MODULE}.multi_queries")
    def test_cleaned_mode_uses_text_directly(
        self,
        mock_queries: MagicMock,
        mock_questions: MagicMock,
        mock_embeddings: MagicMock,
        mock_similar: MagicMock,
        mock_refusion: MagicMock,
    ) -> None:
        """In cleaned mode, cached text is used directly without re-extraction."""
        source_content = {
            "mode": "cleaned",
            "pages": {
                "http://example.com": "test content here",
            },
            "pdfs": {},
        }
        mock_queries.return_value = (["test query"], None)
        mock_questions.return_value = (["question 1"], None)
        doc = ExtendedDocument(text="test content here", url="http://example.com")
        mock_embeddings.return_value = [doc]
        mock_similar.return_value = [doc]
        mock_refusion.return_value = [doc]

        result, raw_sc, _, _ = fetch_additional_information(
            client=MagicMock(),
            client_embedding=MagicMock(),
            prompt="test",
            search_query="test",
            model="gpt-4.1-2025-04-14",
            google_api_key=None,
            google_engine_id=None,
            serper_api_key=None,
            search_provider="google",
            source_content=source_content,
        )

        assert raw_sc is source_content
        assert "test content here" in result
        assert "http://example.com" in result

    @patch(f"{REASONING_MODULE}.reciprocal_rank_refusion")
    @patch(f"{REASONING_MODULE}.find_similar_chunks")
    @patch(f"{REASONING_MODULE}.get_embeddings")
    @patch(f"{REASONING_MODULE}.multi_questions_response")
    @patch(f"{REASONING_MODULE}.multi_queries")
    def test_raw_mode_re_extracts(
        self,
        mock_queries: MagicMock,
        mock_questions: MagicMock,
        mock_embeddings: MagicMock,
        mock_similar: MagicMock,
        mock_refusion: MagicMock,
    ) -> None:
        """In raw mode, HTML is re-extracted via extract_text."""
        source_content = {
            "mode": "raw",
            "pages": {
                "http://example.com": "<html><body>test content here</body></html>",
            },
            "pdfs": {},
        }
        mock_queries.return_value = (["test query"], None)
        mock_questions.return_value = (["question 1"], None)
        doc = ExtendedDocument(text="test content", url="http://example.com")
        mock_embeddings.return_value = [doc]
        mock_similar.return_value = [doc]
        mock_refusion.return_value = [doc]

        result, raw_sc, _, _ = fetch_additional_information(
            client=MagicMock(),
            client_embedding=MagicMock(),
            prompt="test",
            search_query="test",
            model="gpt-4.1-2025-04-14",
            google_api_key=None,
            google_engine_id=None,
            serper_api_key=None,
            search_provider="google",
            source_content=source_content,
        )

        assert raw_sc is source_content
        assert "http://example.com" in result

    @patch(f"{REASONING_MODULE}.multi_queries")
    def test_empty_source_content_returns_empty_information(
        self, mock_queries: MagicMock
    ) -> None:
        """Empty source_content yields an empty information block (flagged-null path)."""
        source_content: dict = {"pages": {}, "pdfs": {}}
        mock_queries.return_value = (["test query"], None)

        result, raw_sc, _, _ = fetch_additional_information(
            client=MagicMock(),
            client_embedding=MagicMock(),
            prompt="test",
            search_query="test",
            model="gpt-4.1-2025-04-14",
            google_api_key=None,
            google_engine_id=None,
            serper_api_key=None,
            search_provider="google",
            source_content=source_content,
        )

        assert result == ""
        assert raw_sc is source_content


def _make_mock_api_keys(return_source_content: str = "false") -> MagicMock:
    """Create a mock api_keys object (KeyChain-like) for run()."""
    services = {
        "openai": "sk-test",
        "google_api_key": None,
        "google_engine_id": None,
        "serperapi": None,
        "search_provider": "google",
        "return_source_content": return_source_content,
    }
    mock_keys = MagicMock()
    mock_keys.__getitem__ = MagicMock(side_effect=lambda k: services[k])
    mock_keys.get = MagicMock(
        side_effect=lambda k, default=None: services.get(k, default)
    )
    mock_keys.max_retries = MagicMock(
        return_value={"openai": 0, "anthropic": 0, "google_api_key": 0, "openrouter": 0}
    )
    return mock_keys


class TestRunFlagBehavior:
    """Verify return_source_content flag controls source_content in used_params."""

    @patch(
        f"{REASONING_MODULE}.parser_prediction_response", return_value='{"p_yes": 0.5}'
    )
    @patch(f"{REASONING_MODULE}.do_reasoning_with_retry")
    @patch(f"{REASONING_MODULE}.fetch_additional_information")
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    def test_flag_on_includes_source_content(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_reasoning: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """When return_source_content is 'true', used_params contains source_content."""
        mock_llm = MagicMock()
        mock_embed = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=(mock_llm, mock_embed))
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)

        mock_fetch.return_value = (
            "additional info",
            {"pages": {"http://x.com": "<html/>"}},
            ["query1"],
            None,
        )
        mock_reasoning.return_value = ("reasoning result", None)

        mock_llm.completions.return_value = MagicMock(
            content="<p_yes>0.5</p_yes>",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=_make_mock_api_keys("true"),
        )

        used_params = result[4]
        assert "source_content" in used_params

    @patch(
        f"{REASONING_MODULE}.parser_prediction_response", return_value='{"p_yes": 0.5}'
    )
    @patch(f"{REASONING_MODULE}.do_reasoning_with_retry")
    @patch(f"{REASONING_MODULE}.fetch_additional_information")
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    def test_flag_off_excludes_source_content(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_reasoning: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """When return_source_content is 'false', used_params omits source_content."""
        mock_llm = MagicMock()
        mock_embed = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=(mock_llm, mock_embed))
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)

        mock_fetch.return_value = (
            "additional info",
            {"pages": {}},
            ["query1"],
            None,
        )
        mock_reasoning.return_value = ("reasoning result", None)

        mock_llm.completions.return_value = MagicMock(
            content="<p_yes>0.5</p_yes>",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=_make_mock_api_keys("false"),
        )

        used_params = result[4]
        assert "source_content" not in used_params


def _make_anthropic_text_response(
    text: str, *, input_tokens: int = 7, output_tokens: int = 13
) -> MagicMock:
    """Build a mock anthropic ``messages.create`` response (content block + usage)."""
    text_block = MagicMock()
    text_block.text = text
    response = MagicMock()
    response.content = [text_block]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


def _make_anthropic_error(cls: type, message: str = "simulated") -> Exception:
    """Build an anthropic error instance without a live ``httpx.Response``."""
    err: Exception = cls.__new__(cls)  # type: ignore[call-overload]
    Exception.__init__(err, message)
    err.message = message  # type: ignore[attr-defined]
    return err


def _anthropic_client(resp: MagicMock) -> Any:
    """Construct an ``LLMClient`` on the anthropic branch with a mocked backing client."""
    with patch("anthropic.Anthropic") as MockAnthropic:
        instance = MagicMock()
        instance.messages.create.return_value = resp
        MockAnthropic.return_value = instance
        client = module.LLMClient(
            api_keys={"anthropic": "sk-ant"}, llm_provider="anthropic"
        )
    return client


class TestLLMClientAnthropicCompletions:
    """Cover the Anthropic branch of ``LLMClient.completions``.

    The wider suites mock ``LLMClientManager`` wholesale, so the
    Anthropic-side mapping under the 0.109.1 bump is otherwise unexercised:
    system-prompt extraction, ``content[0].text``, and the
    ``input_tokens``/``output_tokens`` -> ``prompt_tokens``/``completion_tokens``
    rename that feeds billing.
    """

    def test_system_message_extracted_to_system_kwarg(self) -> None:
        """``system`` entries are passed via ``system=`` and dropped from ``messages=``."""
        client = _anthropic_client(_make_anthropic_text_response('{"p_yes": 0.5}'))
        client.completions(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "U1"},
            ],
        )
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["system"] == "SYS"
        assert kwargs["messages"] == [{"role": "user", "content": "U1"}]

    def test_content_and_usage_mapped_from_anthropic_names(self) -> None:
        """``content[0].text`` and Anthropic token names map onto ``LLMResponse``."""
        client = _anthropic_client(
            _make_anthropic_text_response(
                '{"p_yes": 0.5}', input_tokens=111, output_tokens=222
            )
        )
        result = client.completions(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "U1"},
            ],
        )
        assert result is not None
        assert result.content == '{"p_yes": 0.5}'
        assert result.usage.prompt_tokens == 111
        assert result.usage.completion_tokens == 222

    def test_caller_messages_list_not_mutated(self) -> None:
        """The system-prompt strip works on a copy, leaving the caller's list intact.

        Regression for the retry-path bug: stripping ``messages`` in place
        meant a re-call ran without the system prompt.
        """
        client = _anthropic_client(_make_anthropic_text_response('{"p_yes": 0.5}'))
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U1"},
        ]
        client.completions(model="claude-sonnet-4-6", messages=messages)
        assert messages == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U1"},
        ]

    def test_missing_system_message_uses_default_prompt(self) -> None:
        """With no ``system`` entry, the default ``SYSTEM_PROMPT`` is used instead of crashing.

        Regression for the unbound-``system_prompt`` path: a user-only
        message list previously raised ``UnboundLocalError`` that the broad
        ``except`` shipped on-chain as the prediction string.
        """
        client = _anthropic_client(_make_anthropic_text_response('{"p_yes": 0.5}'))
        client.completions(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "U1"}],
        )
        kwargs = client.client.messages.create.call_args.kwargs
        assert kwargs["system"] == module.SYSTEM_PROMPT


class TestWithKeyRotationAnthropic:
    """Cover the ``anthropic.RateLimitError`` branch of ``with_key_rotation``."""

    @staticmethod
    def _keys(anthropic_budget: int) -> MagicMock:
        """Build an api_keys mock with the given anthropic retry budget."""
        keys = MagicMock()
        keys.max_retries = lambda: {
            "openai": 5,
            "openrouter": 5,
            "anthropic": anthropic_budget,
        }
        keys.rotate = MagicMock()
        return keys

    def test_rate_limit_rotates_anthropic_pool_only(self) -> None:
        """An ``anthropic.RateLimitError`` rotates ONLY the anthropic key, then retries."""
        keys = self._keys(anthropic_budget=1)
        calls = {"n": 0}

        @module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            calls["n"] += 1
            if calls["n"] == 1:
                raise _make_anthropic_error(module.anthropic.RateLimitError, "burst")
            return "ok", "", None, None, None

        result = fake(api_keys=keys)
        assert calls["n"] == 2
        assert [c.args[0] for c in keys.rotate.call_args_list] == ["anthropic"]
        assert result[-1] is keys

    def test_anthropic_pool_exhausted_reraises(self) -> None:
        """When the anthropic pool is exhausted, the error re-raises so the task fails."""
        keys = self._keys(anthropic_budget=0)

        @module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            raise _make_anthropic_error(module.anthropic.RateLimitError, "burned")

        with pytest.raises(module.anthropic.RateLimitError, match="burned"):
            fake(api_keys=keys)


class TestCountTokensAnthropic:
    """Cover the with-client Anthropic ``count_tokens`` path (previously untested in the napthaai forks)."""

    def test_with_client_uses_anthropic_tokenizer(self) -> None:
        """With an anthropic client, the Anthropic ``count_tokens`` result is returned."""
        mock_client = MagicMock()
        mock_client.llm_provider = "anthropic"
        mock_client.client.messages.count_tokens.return_value = SimpleNamespace(
            input_tokens=42
        )
        result = count_tokens("hello world", "claude-sonnet-4-6", client=mock_client)
        assert result == 42
        mock_client.client.messages.count_tokens.assert_called_once()

    def test_anthropic_tokenizer_error_falls_back(self) -> None:
        """A network error from the Anthropic tokenizer falls back instead of raising."""
        mock_client = MagicMock()
        mock_client.llm_provider = "anthropic"
        mock_client.client.messages.count_tokens.side_effect = _make_anthropic_error(
            module.anthropic.APIConnectionError, "net down"
        )
        result = count_tokens("hello world", "claude-sonnet-4-6", client=mock_client)
        assert isinstance(result, int)
        assert result > 0


# ---------------------------------------------------------------------------
# Free-text input contract (issue #455): parse_prompt + flagged-null guards.
# ---------------------------------------------------------------------------

# Trader-template format prompt (regression: previous callers must still work)
TRADER_PROMPT = (
    'Given the question "Will X happen?" and the `yes` answer criterion, ...'
)
# Free-text format prompt: the advertised contract (issue #455)
FREE_TEXT_PROMPT = "Will Alexander Isak join Liverpool before September 2 2025?"
# Long free-text prompt that would return empty search results if passed raw
LONG_FREE_TEXT_PROMPT = (
    "Please predict the following market: Will Alexander Isak permanently transfer "
    "to Liverpool FC before the end of the summer 2025 transfer window (September 2, "
    "2025 23:59 UTC)? Resolution source: official club announcements or BBC Sport. "
    "The market resolves YES if a permanent transfer (not a loan) is confirmed by "
    "the resolution source before the deadline."
)


def _make_serper_api_keys() -> MagicMock:
    """Create a mock api_keys object routed to the (patched) Serper provider."""
    services = {
        "openai": "sk-test",
        "google_api_key": None,
        "google_engine_id": None,
        "serperapi": "serper-test",
        "search_provider": "serper",
        "return_source_content": "false",
    }
    mock_keys = MagicMock()
    mock_keys.__getitem__ = MagicMock(side_effect=lambda k: services[k])
    mock_keys.get = MagicMock(
        side_effect=lambda k, default=None: services.get(k, default)
    )
    mock_keys.max_retries = MagicMock(
        return_value={"openai": 0, "anthropic": 0, "google_api_key": 0, "openrouter": 0}
    )
    return mock_keys


def _mock_client_manager(mock_mgr: MagicMock) -> MagicMock:
    """Wire an LLMClientManager mock to yield (llm, embedding) client mocks."""
    mock_llm = MagicMock()
    mock_mgr.return_value.__enter__ = MagicMock(return_value=(mock_llm, MagicMock()))
    mock_mgr.return_value.__exit__ = MagicMock(return_value=False)
    return mock_llm


class TestParsePromptContract:
    """parse_prompt(): trader-template parity + free-text clause derivation."""

    def test_trader_template_uses_extracted_question_for_both(self) -> None:
        """Trader-template path: the bare question serves as both values."""
        question, query, tier = parse_prompt(TRADER_PROMPT)
        assert tier == "template"
        assert question == "Will X happen?"
        assert query == question

    def test_free_text_llm_gets_full_prompt(self) -> None:
        """Free-text input: the LLM question is the whole prompt."""
        question, _, _ = parse_prompt(FREE_TEXT_PROMPT)
        assert question == FREE_TEXT_PROMPT
        question, _, _ = parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert question == LONG_FREE_TEXT_PROMPT

    def test_boilerplate_prefix_is_dropped_from_query(self) -> None:
        """The query anchors at the market question, dropping instruction text."""
        _, query, tier = parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert tier == "clause"
        assert query.startswith("Will Alexander Isak")
        assert query.endswith("?")
        assert (
            len(query) <= module._MAX_SEARCH_QUERY_LEN
        )  # pylint: disable=protected-access

    def test_tier_is_reported(self) -> None:
        """The tier tags template / clause / raw explicitly."""
        assert parse_prompt(TRADER_PROMPT)[2] == "template"
        assert parse_prompt(FREE_TEXT_PROMPT)[2] == "clause"
        assert parse_prompt("no question mark here at all")[2] == "raw"


class TestDegenerateShortCircuit:
    """Degenerate prompts return the flagged null with ZERO search calls."""

    @pytest.mark.parametrize("degenerate", ["", "   ", "???", '"""'])
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    @patch(f"{REASONING_MODULE}.get_urls_from_queries")
    @patch(f"{REASONING_MODULE}.get_urls_from_queries_serper")
    def test_degenerate_prompt_short_circuits(
        self,
        mock_serper: MagicMock,
        mock_google: MagicMock,
        mock_mgr: MagicMock,
        degenerate: str,
    ) -> None:
        """Prompts with no searchable content never reach a search provider."""
        _mock_client_manager(mock_mgr)
        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt=degenerate,
            api_keys=_make_mock_api_keys(),
        )
        mock_serper.assert_not_called()
        mock_google.assert_not_called()
        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.5 and parsed["p_no"] == 0.5
        assert parsed["confidence"] == 0.0 and parsed["info_utility"] == 0.0
        used_params = result[4]
        assert used_params["empty_retrieval"] is True
        assert used_params["null_reason"] == "empty query"
        assert used_params["scan_truncated"] is False


class TestEmptyRetrievalFlaggedNull:
    """Empty retrieval yields a parseable flagged null, not an error string."""

    @patch(f"{REASONING_MODULE}.LLMClientManager")
    @patch(f"{REASONING_MODULE}.get_urls_from_queries_serper", return_value=[])
    def test_zero_urls_returns_flagged_null_live_search(
        self, mock_serper: MagicMock, mock_mgr: MagicMock
    ) -> None:
        """A live search with no usable documents records null_reason='live search'."""
        _mock_client_manager(mock_mgr)
        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_serper_api_keys(),
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.5 and parsed["confidence"] == 0.0
        used_params = result[4]
        assert used_params["empty_retrieval"] is True
        assert used_params["null_reason"] == "live search"
        assert used_params["parse_tier"] == "clause"

    @patch(f"{REASONING_MODULE}.LLMClientManager")
    def test_empty_cached_replay_returns_flagged_null(
        self, mock_mgr: MagicMock
    ) -> None:
        """An empty cached capture records null_reason='cached replay'."""
        _mock_client_manager(mock_mgr)
        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            source_content={"pages": {}, "pdfs": {}},
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.5 and parsed["confidence"] == 0.0
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "cached replay"


class TestQueryLeakFix:
    """The compact query, not the raw prompt, reaches the search engine."""

    def test_multi_queries_appends_search_query_not_prompt(self) -> None:
        """The direct-search append carries search_query; the LLM sees the prompt."""
        client = MagicMock()
        client.completions.return_value = MagicMock(
            content="<queries>alpha\nbeta</queries>",
            usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        )
        search_query = "Will Alexander Isak permanently transfer to Liverpool FC?"
        queries, _ = multi_queries(
            client=client,
            prompt=LONG_FREE_TEXT_PROMPT,
            search_query=search_query,
            model="gpt-4.1-2025-04-14",
            num_queries=2,
        )
        assert queries[-1] == search_query
        assert LONG_FREE_TEXT_PROMPT not in queries
        sent = client.completions.call_args.kwargs["messages"][1]["content"]
        assert LONG_FREE_TEXT_PROMPT in sent

    @patch(
        f"{REASONING_MODULE}.parser_prediction_response", return_value='{"p_yes": 0.5}'
    )
    @patch(f"{REASONING_MODULE}.do_reasoning_with_retry")
    @patch(f"{REASONING_MODULE}.fetch_additional_information")
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    def test_run_feeds_question_to_llm_and_query_to_fetch(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_reasoning: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """Free text: the LLM slots get the whole prompt, the search the clause."""
        mock_llm = _mock_client_manager(mock_mgr)
        mock_fetch.return_value = ("additional info", {"pages": {}}, ["q"], None)
        mock_reasoning.return_value = ("reasoning result", None)
        mock_llm.completions.return_value = MagicMock(
            content="<p_yes>0.5</p_yes>",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
        )
        fetch_kwargs = mock_fetch.call_args.kwargs
        assert fetch_kwargs["prompt"] == LONG_FREE_TEXT_PROMPT
        assert fetch_kwargs["search_query"].startswith("Will Alexander Isak")
        assert fetch_kwargs["search_query"] != LONG_FREE_TEXT_PROMPT
        # LLM-input parity: criteria the query drops still reach the LLM prompts
        assert "official club announcements or BBC Sport" in result[1]
        assert result[4]["parse_tier"] == "clause"
        assert result[4]["scan_truncated"] is False

    @patch(
        f"{REASONING_MODULE}.parser_prediction_response", return_value='{"p_yes": 0.5}'
    )
    @patch(f"{REASONING_MODULE}.do_reasoning_with_retry")
    @patch(f"{REASONING_MODULE}.fetch_additional_information")
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    def test_trader_template_run_parity(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_reasoning: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """Trader template: LLM question and search query stay the bare title."""
        mock_llm = _mock_client_manager(mock_mgr)
        mock_fetch.return_value = ("additional info", {"pages": {}}, ["q"], None)
        mock_reasoning.return_value = ("reasoning result", None)
        mock_llm.completions.return_value = MagicMock(
            content="<p_yes>0.5</p_yes>",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt=TRADER_PROMPT,
            api_keys=_make_mock_api_keys(),
        )
        fetch_kwargs = mock_fetch.call_args.kwargs
        assert fetch_kwargs["prompt"] == "Will X happen?"
        assert fetch_kwargs["search_query"] == "Will X happen?"
        # exactly the old extract_question behavior: bare question, no template
        assert "Will X happen?" in result[1]
        assert "`yes` answer criterion" not in result[1]
        assert result[4]["parse_tier"] == "template"

    @patch(
        f"{REASONING_MODULE}.parser_prediction_response", return_value='{"p_yes": 0.5}'
    )
    @patch(f"{REASONING_MODULE}.do_reasoning_with_retry")
    @patch(f"{REASONING_MODULE}.fetch_additional_information")
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    def test_long_template_prompt_not_marked_truncated(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_reasoning: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """Template past the window is NOT flagged: question precedes the scan."""
        mock_llm = _mock_client_manager(mock_mgr)
        mock_fetch.return_value = ("additional info", {"pages": {}}, ["q"], None)
        mock_reasoning.return_value = ("reasoning result", None)
        mock_llm.completions.return_value = MagicMock(
            content="<p_yes>0.5</p_yes>",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        prompt = TRADER_PROMPT + " filler" * (
            module._MAX_SCAN_CHARS // 3
        )  # pylint: disable=protected-access
        assert len(prompt) > module._MAX_SCAN_CHARS  # pylint: disable=protected-access
        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
        )
        assert result[4]["parse_tier"] == "template"
        assert result[4]["scan_truncated"] is False


class TestSerperShapeGuard:
    """Serper bodies are validated with the typed shape helper."""

    @patch(f"{REASONING_MODULE}.requests.request")
    def test_malformed_serper_body_is_skipped_not_crashed(
        self, mock_request: MagicMock
    ) -> None:
        """A 200 body without the organic key is skipped for that query."""
        mock_request.return_value = MagicMock(
            status_code=200, json=lambda: {"message": "quota exceeded"}
        )
        urls = get_urls_from_queries_serper(["q1"], api_key="k", num=3)
        assert urls == []  # pylint: disable=use-implicit-booleaness-not-comparison

    @patch(f"{REASONING_MODULE}.requests.request")
    def test_valid_serper_body_yields_links(self, mock_request: MagicMock) -> None:
        """A well-formed organic list still yields its links."""
        mock_request.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "organic": [
                    {"title": "T", "link": "https://example.test", "snippet": "S"}
                ]
            },
        )
        urls = get_urls_from_queries_serper(["q1"], api_key="k", num=3)
        assert urls == ["https://example.test"]
