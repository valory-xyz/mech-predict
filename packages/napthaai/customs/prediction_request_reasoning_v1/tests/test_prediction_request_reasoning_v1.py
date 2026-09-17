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
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

import packages.napthaai.customs.prediction_request_reasoning_v1.prediction_request_reasoning_v1 as module
from packages.napthaai.customs.prediction_request_reasoning_v1.prediction_request_reasoning_v1 import (
    ExtendedDocument,
    LLMClientManager,
    count_tokens,
    do_reasoning_with_retry,
    extract_prediction,
    extract_texts,
    fetch_additional_information,
    get_urls_from_queries_serper,
    multi_queries,
    multi_questions_response,
    parse_prompt,
    parser_prediction_response,
    run,
)

# Aliases for module-private caps: one disable each here, so call sites stay
# clean and the suppression cannot drift under formatter line-wrapping.
_QUERY_CAP = module._MAX_SEARCH_QUERY_LEN  # pylint: disable=protected-access
_SCAN_CAP = module._MAX_SCAN_CHARS  # pylint: disable=protected-access


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


def _make_http_error(status_code: int, message: str) -> Exception:
    """Build a googleapiclient HttpError carrying the given status code."""
    resp = SimpleNamespace(status=status_code, reason=message)
    return module.googleapiclient.errors.HttpError(
        resp=resp, content=message.encode("utf-8")
    )


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


class TestLLMClientFinishReason:
    """LLMClient.completions() carries the provider's stop reason, normalised."""

    def test_openai_finish_reason_is_carried(self) -> None:
        """The value the OpenAI SDK reports reaches the response unchanged."""
        choice = MagicMock(finish_reason="length")
        choice.message.content = "x"
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = MagicMock(
                choices=[choice], usage=MagicMock(prompt_tokens=1, completion_tokens=1)
            )
            client = module.LLMClient(api_keys={"openai": "sk"}, llm_provider="openai")
        response = client.completions(model="gpt-4.1-2025-04-14", messages=[])
        assert response is not None
        assert response.finish_reason == "length"

    def test_anthropic_max_tokens_is_normalised_to_length(self) -> None:
        """Anthropic's max_tokens stop maps onto OpenAI's 'length'."""
        resp = _make_anthropic_text_response('{"p_yes": 0.5}')
        resp.stop_reason = "max_tokens"
        client = _anthropic_client(resp)
        response = client.completions(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "U1"}]
        )
        assert response is not None
        assert response.finish_reason == "length"

    def test_anthropic_normal_stop_is_not_a_cut(self) -> None:
        """A normal Anthropic stop is carried as-is and never reads as a cut."""
        resp = _make_anthropic_text_response('{"p_yes": 0.5}')
        resp.stop_reason = "end_turn"
        client = _anthropic_client(resp)
        response = client.completions(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "U1"}]
        )
        assert response is not None
        assert response.finish_reason == "end_turn"


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

    def test_anthropic_pool_exhausted_returns_typed_null(self) -> None:
        """An exhausted anthropic pool delivers the typed null, it does not raise."""
        keys = self._keys(anthropic_budget=0)

        @module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            raise _make_anthropic_error(module.anthropic.RateLimitError, "burned")

        result = fake(api_keys=keys)
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None and parsed["p_no"] is None
        assert parsed["error_type"] == "RateLimitError"
        assert parsed["error"] == "burned"
        assert result[-1] is keys

    def test_openai_pool_exhausted_returns_typed_null(self) -> None:
        """An exhausted openai/openrouter pool delivers the typed null too."""
        keys = self._keys(anthropic_budget=5)
        keys.max_retries = lambda: {"openai": 0, "openrouter": 0, "anthropic": 5}

        @module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            raise _make_anthropic_error(module.openai.RateLimitError, "openai burned")

        result = fake(api_keys=keys)
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["error_type"] == "RateLimitError"
        assert keys.rotate.call_count == 0

    def test_google_rate_limit_exhausted_returns_typed_null(self) -> None:
        """An exhausted google pool delivers the typed null instead of raising."""
        keys = self._keys(anthropic_budget=5)
        keys.max_retries = lambda: {
            "openai": 5,
            "openrouter": 5,
            "anthropic": 5,
            "google_api_key": 0,
        }
        err = _make_http_error(module.GOOGLE_RATE_LIMIT_EXCEEDED_CODE, "quota")

        @module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            raise err

        result = fake(api_keys=keys)
        assert json.loads(result[0])["p_yes"] is None

    def test_non_rate_limit_google_error_returns_typed_null(self) -> None:
        """A non-429 google error is a permanent failure, delivered as the typed null."""
        keys = self._keys(anthropic_budget=5)
        err = _make_http_error(403, "forbidden")

        @module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            raise err

        result = fake(api_keys=keys)
        assert json.loads(result[0])["p_yes"] is None
        assert keys.rotate.call_count == 0


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
        assert len(query) <= _QUERY_CAP

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


class TestDegenerateQueryWithCachedContent:
    """A degenerate query must NOT skip a supplied cached capture."""

    @patch(
        f"{REASONING_MODULE}.parser_prediction_response",
        return_value='{"p_yes": 0.7, "p_no": 0.3}',
    )
    @patch(f"{REASONING_MODULE}.do_reasoning_with_retry")
    @patch(f"{REASONING_MODULE}.fetch_additional_information")
    @patch(f"{REASONING_MODULE}.get_urls_from_queries")
    @patch(f"{REASONING_MODULE}.get_urls_from_queries_serper")
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    def test_degenerate_query_with_cached_pages_still_predicts(
        self,
        mock_mgr: MagicMock,
        mock_serper: MagicMock,
        mock_google: MagicMock,
        mock_fetch: MagicMock,
        mock_reasoning: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """Non-empty cached pages outrank the empty-query short circuit."""
        mock_llm = _mock_client_manager(mock_mgr)
        cached = {"pages": {"http://u.example": "cached text"}, "pdfs": {}}
        mock_fetch.return_value = ("cached text", cached, ["q"], None)
        mock_reasoning.return_value = ("reasoning result", None)
        mock_llm.completions.return_value = MagicMock(
            content="<p_yes>0.7</p_yes>",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt="???",
            api_keys=_make_mock_api_keys(),
            source_content=cached,
        )

        # The cached capture must actually be consumed, and no live search
        # may fire to replace it.
        mock_fetch.assert_called_once()
        assert mock_fetch.call_args.kwargs["source_content"] == cached
        mock_serper.assert_not_called()
        mock_google.assert_not_called()
        used_params = result[4]
        assert "empty_retrieval" not in used_params
        assert "null_reason" not in used_params
        assert json.loads(result[0])["p_yes"] == 0.7

    @patch(f"{REASONING_MODULE}.get_urls_from_queries")
    @patch(f"{REASONING_MODULE}.get_urls_from_queries_serper")
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    def test_degenerate_query_with_empty_cache_reports_cached_replay(
        self,
        mock_mgr: MagicMock,
        mock_serper: MagicMock,
        mock_google: MagicMock,
    ) -> None:
        """A degenerate query over an empty capture is a cached replay, not an empty query."""
        _mock_client_manager(mock_mgr)
        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt="???",
            api_keys=_make_mock_api_keys(),
            source_content={"pages": {}, "pdfs": {}},
        )
        mock_serper.assert_not_called()
        mock_google.assert_not_called()
        used_params = result[4]
        assert used_params["empty_retrieval"] is True
        assert used_params["null_reason"] == "cached replay"


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

    def test_multi_queries_dedups_the_appended_search_query(self) -> None:
        """A brainstormed query equal to search_query is not searched twice."""
        search_query = "Will Isak transfer to Liverpool?"
        client = MagicMock()
        client.completions.return_value = MagicMock(
            content=f"<queries>alpha\n{search_query}</queries>",
            usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        )
        queries, _ = multi_queries(
            client=client,
            prompt=LONG_FREE_TEXT_PROMPT,
            search_query=search_query,
            model="gpt-4.1-2025-04-14",
            num_queries=2,
        )
        assert queries == ["alpha", search_query]

    def test_multi_queries_dedup_ignores_case(self) -> None:
        """Dedup compares case-insensitively, so a recased duplicate is dropped."""
        client = MagicMock()
        client.completions.return_value = MagicMock(
            content="<queries>  Will Isak Transfer?  \nalpha</queries>",
            usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        )
        queries, _ = multi_queries(
            client=client,
            prompt=LONG_FREE_TEXT_PROMPT,
            search_query="will isak transfer?",
            model="gpt-4.1-2025-04-14",
            num_queries=2,
        )
        assert queries == ["Will Isak Transfer?", "alpha"]

    def test_multi_queries_keeps_distinct_queries_in_order(self) -> None:
        """Dedup leaves an all-distinct query list untouched."""
        client = MagicMock()
        client.completions.return_value = MagicMock(
            content="<queries>alpha\nbeta</queries>",
            usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        )
        queries, _ = multi_queries(
            client=client,
            prompt=LONG_FREE_TEXT_PROMPT,
            search_query="gamma",
            model="gpt-4.1-2025-04-14",
            num_queries=2,
        )
        assert queries == ["alpha", "beta", "gamma"]

    @pytest.mark.parametrize(
        "func", [multi_queries, fetch_additional_information], ids=["multi", "fetch"]
    )
    def test_search_query_is_a_required_parameter(self, func: Any) -> None:
        """search_query carries no default, so a caller cannot silently omit it."""
        param = inspect.signature(func).parameters["search_query"]
        assert param.default is inspect.Parameter.empty
        assert param.annotation is str

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
        prompt = TRADER_PROMPT + " filler" * (_SCAN_CAP // 3)
        assert len(prompt) > _SCAN_CAP
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
    def test_malformed_serper_body_raises_typed_error(
        self, mock_request: MagicMock
    ) -> None:
        """A missing/malformed organic key raises instead of being swallowed."""
        mock_request.return_value = MagicMock(
            status_code=200, json=lambda: {"message": "quota exceeded"}
        )
        with pytest.raises(ValueError, match="organic"):
            get_urls_from_queries_serper(["q1"], api_key="k", num=3)

    @patch(f"{REASONING_MODULE}.requests.request")
    def test_malformed_people_also_ask_raises_typed_error(
        self, mock_request: MagicMock
    ) -> None:
        """The second shape check fires too: a non-list peopleAlsoAsk raises."""
        mock_request.return_value = MagicMock(
            status_code=200,
            json=lambda: {"organic": [{"link": "https://a.test"}], "peopleAlsoAsk": {}},
        )
        with pytest.raises(ValueError, match="peopleAlsoAsk"):
            get_urls_from_queries_serper(["q1"], api_key="k", num=3)

    @patch(f"{REASONING_MODULE}.requests.request")
    def test_empty_organic_with_people_also_ask_yields_no_urls(
        self, mock_request: MagicMock
    ) -> None:
        """Boundary: PAA-only is a well-formed zero-hit, not a shape error."""
        mock_request.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "organic": [],
                "peopleAlsoAsk": [{"question": "Q?", "snippet": "A."}],
            },
        )
        urls = get_urls_from_queries_serper(["q1"], api_key="k", num=3)
        assert urls == []  # pylint: disable=use-implicit-booleaness-not-comparison

    @patch(f"{REASONING_MODULE}.requests.request")
    def test_transport_failure_is_still_swallowed_per_query(
        self, mock_request: MagicMock
    ) -> None:
        """The carve-out is shape-only: a per-query transport error still skips."""
        mock_request.side_effect = requests.RequestException("connection reset")
        urls = get_urls_from_queries_serper(["q1"], api_key="k", num=3)
        assert urls == []  # pylint: disable=use-implicit-booleaness-not-comparison

    @patch(f"{REASONING_MODULE}.requests.request")
    def test_a_systemic_http_status_is_not_swallowed(
        self, mock_request: MagicMock
    ) -> None:
        """A 401/403/429 fails every query alike, so it must surface."""
        # HTTPError subclasses RequestException: without a dedicated arm it
        # lands in the transport bucket, urls ends empty and the delivery is a
        # flagged null indistinguishable on-chain from a genuine zero-hit.
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        mock_request.return_value = resp
        with pytest.raises(requests.HTTPError):
            get_urls_from_queries_serper(["q1", "q2"], "key", num=3)

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


class TestSerperShapeErrorDelivery:
    """A broken Serper integration is delivered as the TYPED error null."""

    @pytest.mark.parametrize(
        "body",
        [
            {"message": "quota exceeded"},
            {"organic": None, "peopleAlsoAsk": []},
            {"organic": {"not": "a list"}, "peopleAlsoAsk": []},
            {"organic": "reshaped", "peopleAlsoAsk": []},
            {"organic": [{"link": "https://a.test"}], "peopleAlsoAsk": "nope"},
        ],
    )
    @patch(f"{REASONING_MODULE}.LLMClientManager")
    @patch(f"{REASONING_MODULE}.requests.request")
    def test_broken_serper_body_delivers_error_null_not_flagged_null(
        self, mock_request: MagicMock, mock_mgr: MagicMock, body: dict
    ) -> None:
        """Reshaped bodies deliver p_yes None + error_type, not the 0.5 null."""
        _mock_client_manager(mock_mgr)
        mock_request.return_value = MagicMock(status_code=200, json=lambda: body)
        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_serper_api_keys(),
        )
        parsed = json.loads(result[0])
        # error null, NOT the 0.5 flagged null a genuine zero-hit delivers
        assert parsed["p_yes"] is None and parsed["p_no"] is None
        assert parsed["confidence"] == 0.0 and parsed["info_utility"] == 0.0
        assert parsed["error"] and parsed["error_type"] == "ValueError"

    @patch(f"{REASONING_MODULE}.LLMClientManager")
    @patch(f"{REASONING_MODULE}.requests.request")
    def test_genuine_zero_hit_stays_the_flagged_null(
        self, mock_request: MagicMock, mock_mgr: MagicMock
    ) -> None:
        """Control: a well-formed zero-hit keeps the 0.5 flagged-null delivery."""
        _mock_client_manager(mock_mgr)
        mock_request.return_value = MagicMock(
            status_code=200, json=lambda: {"organic": [], "peopleAlsoAsk": []}
        )
        result = run(
            tool="prediction-request-reasoning-v1",
            model="gpt-4.1-2025-04-14",
            prompt=FREE_TEXT_PROMPT,
            api_keys=_make_serper_api_keys(),
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.5 and parsed["confidence"] == 0.0
        assert "error_type" not in parsed
        assert result[4]["null_reason"] == "live search"


class TestScanTruncationObservable:
    """A scan window that did not cover the prompt is marked, not silent."""

    @staticmethod
    def _run_free_text(prompt: str) -> tuple:
        """Run the tool on a free-text prompt with fetch + LLM mocked."""
        with (
            patch(f"{REASONING_MODULE}.LLMClientManager") as mock_mgr,
            patch(f"{REASONING_MODULE}.fetch_additional_information") as mock_fetch,
            patch(f"{REASONING_MODULE}.do_reasoning_with_retry") as mock_reasoning,
            patch(
                f"{REASONING_MODULE}.parser_prediction_response",
                return_value='{"p_yes": 0.5}',
            ),
        ):
            mock_llm = _mock_client_manager(mock_mgr)
            mock_fetch.return_value = ("additional info", {"pages": {}}, ["q"], None)
            mock_reasoning.return_value = ("reasoning result", None)
            mock_llm.completions.return_value = MagicMock(
                content="<p_yes>0.5</p_yes>",
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            return run(
                tool="prediction-request-reasoning-v1",
                model="gpt-4.1-2025-04-14",
                prompt=prompt,
                api_keys=_make_mock_api_keys(),
            )

    def test_raw_tier_past_window_is_marked_truncated(self) -> None:
        """No clause inside the window: raw tier AND scan_truncated True."""
        prompt = "word " * (_SCAN_CAP // 4) + "Will it happen by 2027?"
        assert len(prompt) > _SCAN_CAP
        result = self._run_free_text(prompt)
        assert result[4]["parse_tier"] == "raw"
        assert result[4]["scan_truncated"] is True

    def test_clause_tier_past_window_is_marked_truncated(self) -> None:
        """A clause inside the window still flags what sat past it."""
        prompt = FREE_TEXT_PROMPT + " filler" * (_SCAN_CAP // 3)
        assert len(prompt) > _SCAN_CAP
        result = self._run_free_text(prompt)
        assert result[4]["parse_tier"] == "clause"
        assert result[4]["scan_truncated"] is True


# ---------------------------------------------------------------------------
# Free-text answer extraction: the model replies in JSON, not the tag form.
# ---------------------------------------------------------------------------

FORECAST_JSON = '{"p_yes": 0.83, "p_no": 0.17, "confidence": 0.8, "info_utility": 0.7}'
TAG_RESPONSE = (
    "<p_yes>0.6</p_yes><p_no>0.4</p_no>"
    "<info_utility>0.5</info_utility><confidence>0.7</confidence>"
)


class TestExtractPrediction:
    """extract_prediction(): the completion shapes a free-text prompt produces."""

    def test_bare_json_object_passes_through(self) -> None:
        """A completion that is only the forecast object comes back as that object."""
        assert json.loads(extract_prediction(FORECAST_JSON) or "") == json.loads(
            FORECAST_JSON
        )

    def test_reasoning_scaffold_yields_only_the_object(self) -> None:
        """Prose around the answer is dropped, leaving the forecast object."""
        completion = (
            "Step 1: the transfer was confirmed by the club.\n"
            "Step 2: no competing reports.\n"
            f"Final answer:\n{FORECAST_JSON}\n"
        )
        extracted = extract_prediction(completion)
        assert json.loads(extracted or "")["p_yes"] == 0.83
        assert "Step 1" not in (extracted or "")

    def test_draft_before_the_answer_loses_to_the_final_one(self) -> None:
        """When two forecasts appear, the last one is delivered."""
        completion = (
            'Draft: {"p_yes": 0.2, "p_no": 0.8}\n'
            f"On reflection, the final answer is {FORECAST_JSON}"
        )
        assert json.loads(extract_prediction(completion) or "")["p_yes"] == 0.83

    def test_trailing_non_forecast_object_is_skipped(self) -> None:
        """An object without p_yes after the answer does not displace the forecast."""
        completion = f'{FORECAST_JSON}\nSources: {{"urls": ["http://x.com"]}}'
        assert json.loads(extract_prediction(completion) or "")["p_yes"] == 0.83

    def test_cut_mid_object_returns_none(self) -> None:
        """A max_tokens cut while the answer is being written yields no forecast."""
        completion = 'Reasoning done.\n{"p_yes": 0.83, "p_no": 0.1'
        assert extract_prediction(completion) is None

    def test_cut_whose_inner_object_closed_returns_none(self) -> None:
        """A nested object closing inside the cut answer does not close the answer."""
        completion = '{"meta": {"model": "gpt"}, "p_yes": 0.83, "p_no"'
        assert extract_prediction(completion) is None

    def test_earlier_forecast_before_a_cut_is_not_delivered(self) -> None:
        """A complete forecast followed by a cut object is a draft, so None."""
        completion = f'{FORECAST_JSON}\nRevised: {{"p_yes": 0.5'
        assert extract_prediction(completion) is None

    def test_stray_brace_in_prose_is_not_a_cut(self) -> None:
        """An unclosed brace that never began an object leaves the forecast delivered."""
        completion = f"{FORECAST_JSON}\nSee {{source for details"
        assert json.loads(extract_prediction(completion) or "")["p_yes"] == 0.83

    def test_brace_inside_a_string_value_does_not_close_the_object(self) -> None:
        """A brace inside a value is part of the string, not the object terminator."""
        completion = '{"p_yes": 0.3, "p_no": 0.7, "reason": "resolves if } appears"}'
        parsed = json.loads(extract_prediction(completion) or "")
        assert parsed["p_yes"] == 0.3 and parsed["reason"] == "resolves if } appears"

    def test_escaped_quote_inside_a_string_value_is_handled(self) -> None:
        """An escaped quote does not end the string, so the object still parses."""
        completion = (
            '{"p_yes": 0.3, "p_no": 0.7, '
            '"reason": "the \\"final\\" report shows } by June"}'
        )
        assert json.loads(extract_prediction(completion) or "")["p_yes"] == 0.3

    def test_out_of_range_p_yes_is_skipped_for_the_valid_one(self) -> None:
        """A p_yes above 1 is not a probability, so an earlier valid forecast wins."""
        completion = f'{FORECAST_JSON}\nScaled: {{"p_yes": 83, "p_no": 17}}'
        assert json.loads(extract_prediction(completion) or "")["p_yes"] == 0.83

    def test_a_lone_out_of_range_p_yes_is_not_delivered(self) -> None:
        """A sole out-of-range forecast yields None rather than being delivered."""
        assert extract_prediction('{"p_yes": 83, "p_no": 17}') is None

    def test_null_p_yes_is_not_a_forecast(self) -> None:
        """A sole null p_yes yields None, not the unusable object."""
        assert extract_prediction('{"p_yes": null, "p_no": null}') is None

    def test_prose_without_json_returns_none(self) -> None:
        """A completion with no object at all is not handed on as text."""
        assert extract_prediction("I cannot estimate this probability.") is None

    @pytest.mark.parametrize("content", [None, ""])
    def test_empty_content_returns_none(self, content: Any) -> None:
        """Empty or missing content yields None, never an empty delivery."""
        assert extract_prediction(content) is None

    def test_a_completion_ending_on_the_opening_brace_is_a_cut(self) -> None:
        """A cut landing ON the brace must not deliver an earlier draft."""
        # The tail after "{" is empty here, which the first version read as
        # prose. On pretty-printed JSON a cut after "{" is a likely stop.
        content = '{"p_yes": 0.9, "p_no": 0.1}\n<answer>\n{'
        assert extract_prediction(content) is None

    def test_a_completion_ending_on_brace_plus_whitespace_is_a_cut(self) -> None:
        """Whitespace after the opening brace is still a cut, not prose."""
        content = '{"p_yes": 0.9, "p_no": 0.1}\n<answer>\n{\n '
        assert extract_prediction(content) is None


class TestParserPredictionResponse:
    """parser_prediction_response(): tag parity, JSON fallback, honest errors."""

    def test_tag_form_still_parses_to_the_four_floats(self) -> None:
        """The advertised tag form is unaffected by the fallback."""
        parsed = json.loads(parser_prediction_response(TAG_RESPONSE))
        assert parsed == {
            "p_yes": 0.6,
            "p_no": 0.4,
            "info_utility": 0.5,
            "confidence": 0.7,
        }

    def test_json_answer_is_delivered_as_the_forecast_object(self) -> None:
        """A tagless JSON answer is extracted instead of crashing."""
        completion = f"Reasoning about the market.\n{FORECAST_JSON}"
        parsed = json.loads(parser_prediction_response(completion))
        assert parsed["p_yes"] == 0.83 and parsed["confidence"] == 0.8

    def test_tagless_prose_raises_value_error(self) -> None:
        """A tagless answer with no forecast object is a ValueError, not an IndexError."""
        with pytest.raises(ValueError, match="No forecast object"):
            parser_prediction_response("I cannot estimate this probability.")

    def test_cut_json_answer_raises_value_error(self) -> None:
        """A cut-off JSON answer is rejected rather than delivered as a draft."""
        with pytest.raises(ValueError, match="No forecast object"):
            parser_prediction_response('{"p_yes": 0.83, "p_no"')

    def test_missing_later_tag_reports_the_real_cause(self) -> None:
        """A response missing a later tag names the cause instead of an unbound value."""
        with pytest.raises(ValueError, match="Error for p_no: IndexError"):
            parser_prediction_response("<p_yes>0.6</p_yes>")

    def test_unparseable_tag_value_reports_the_real_cause(self) -> None:
        """A non-numeric tag value names the float failure, not an unbound value."""
        with pytest.raises(ValueError, match="Error for p_yes: ValueError"):
            parser_prediction_response("<p_yes>maybe</p_yes>")


# A max_tokens cut that lands in prose AFTER a complete draft object. Nothing is
# left unclosed, so no text heuristic can tell the draft from an answer.
DRAFT_THEN_CUT_IN_PROSE = (
    '<facts>x</facts>\n<thinking>\nDraft estimate {"p_yes": 0.35, "p_no": 0.65} '
    "seems too low, let me reconsider given the news that changes the probabi"
)


class TestFreeTextAnswerDelivery:
    """run(): the extractor sits on the path that returns the completion."""

    @staticmethod
    def _run_with_completion(
        content: Optional[str], finish_reason: str = "stop"
    ) -> tuple:
        """Run the tool end to end with the prediction model returning `content`."""
        with (
            patch(f"{REASONING_MODULE}.LLMClientManager") as mock_mgr,
            patch(f"{REASONING_MODULE}.fetch_additional_information") as mock_fetch,
            patch(f"{REASONING_MODULE}.do_reasoning_with_retry") as mock_reasoning,
        ):
            mock_llm = _mock_client_manager(mock_mgr)
            mock_fetch.return_value = ("additional info", {"pages": {}}, ["q1"], None)
            mock_reasoning.return_value = ("reasoning result", None)
            mock_llm.completions.return_value = MagicMock(
                content=content,
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
                finish_reason=finish_reason,
            )
            return run(
                tool="prediction-request-reasoning-v1",
                model="gpt-4.1-2025-04-14",
                prompt=FREE_TEXT_PROMPT,
                api_keys=_make_mock_api_keys(),
            )

    def test_run_delivers_the_extracted_forecast_object(self) -> None:
        """The delivery is the forecast object, not the whole reasoning block."""
        completion = (
            "Step 1: the club confirmed the transfer.\n"
            f"Final answer:\n{FORECAST_JSON}"
        )
        result = self._run_with_completion(completion)
        parsed = json.loads(result[0])
        assert parsed["p_yes"] == 0.83 and parsed["p_no"] == 0.17
        assert "Step 1" not in result[0]

    def test_run_delivers_a_parseable_null_when_the_answer_was_cut(self) -> None:
        """A cut answer is delivered as the typed null, never as truncated text."""
        result = self._run_with_completion('Reasoning done.\n{"p_yes": 0.83, "p_no"')
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["error_type"] == "ValueError"

    def test_a_completion_cut_at_max_tokens_delivers_the_typed_null(self) -> None:
        """A cut after a complete draft is the typed null, not the draft."""
        # The draft is complete and the cut lands in prose, so the parser alone
        # would deliver 0.35 as the forecast.
        assert (
            json.loads(parser_prediction_response(DRAFT_THEN_CUT_IN_PROSE))["p_yes"]
            == 0.35
        )
        result = self._run_with_completion(
            DRAFT_THEN_CUT_IN_PROSE, finish_reason="length"
        )
        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert payload["error_type"] == "TruncatedCompletionError"
        assert "0.35" not in result[0]

    def test_a_missing_completion_delivers_the_typed_null(self) -> None:
        """No content (a refusal) is the typed null, not a plain-text result."""
        result = self._run_with_completion(None)
        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert payload["error_type"] == "ValueError"


def test_tag_form_out_of_range_p_yes_raises_value_error() -> None:
    """The tag form holds the same [0, 1] bar as the JSON fallback."""
    completion = (
        "<p_yes>1.7</p_yes><p_no>-0.7</p_no>"
        "<confidence>0.5</confidence><info_utility>0.5</info_utility>"
    )
    with pytest.raises(ValueError, match="not a probability"):
        parser_prediction_response(completion)


def test_tag_form_boundary_p_yes_is_accepted() -> None:
    """p_yes of exactly 1.0 is a probability and is delivered."""
    completion = (
        "<p_yes>1.0</p_yes><p_no>0.0</p_no>"
        "<confidence>0.5</confidence><info_utility>0.5</info_utility>"
    )
    assert json.loads(parser_prediction_response(completion))["p_yes"] == 1.0


def test_json_form_lower_boundary_p_yes_is_accepted() -> None:
    """The JSON path keeps a p_yes of exactly 0.0 as well."""
    content = '{"p_yes": 0.0, "p_no": 1.0, "confidence": 0.5, "info_utility": 0.5}'
    delivered = extract_prediction(content)
    assert delivered is not None
    assert json.loads(delivered)["p_yes"] == 0.0


def test_tag_form_lower_boundary_p_yes_is_accepted() -> None:
    """p_yes of exactly 0.0 is a probability too: the guard excludes neither bound."""
    completion = (
        "<p_yes>0.0</p_yes><p_no>1.0</p_no>"
        "<confidence>0.5</confidence><info_utility>0.5</info_utility>"
    )
    assert json.loads(parser_prediction_response(completion))["p_yes"] == 0.0


def test_max_cost_request_returns_the_float() -> None:
    """A delivery_rate of 0 asks for a cost estimate, not a forecast.

    with_key_rotation used to append api_keys unconditionally, so the float
    hit `float + tuple` and the wrapper delivered a typed TypeError null.
    """
    keys = MagicMock()
    keys.max_retries = lambda: {"openai": 1, "anthropic": 1, "openrouter": 1}
    result = run(
        prompt="Will it rain tomorrow?",
        tool="prediction-request-reasoning-v1",
        model="gpt-4.1-2025-04-14",
        api_keys=keys,
        delivery_rate=0,
        counter_callback=lambda **kwargs: 12345.0,
    )
    assert result == 12345.0


def test_run_with_an_anthropic_max_tokens_cut_delivers_the_typed_null() -> None:
    """A real Anthropic response shape stopped at max_tokens ends as the typed null."""
    resp = _make_anthropic_text_response(DRAFT_THEN_CUT_IN_PROSE)
    resp.stop_reason = "max_tokens"
    api_keys = _make_mock_api_keys()
    base_getitem = api_keys.__getitem__.side_effect
    api_keys.__getitem__.side_effect = lambda k: (
        "sk-ant" if k == "anthropic" else base_getitem(k)
    )
    with (
        patch("anthropic.Anthropic") as mock_anthropic,
        patch("openai.OpenAI"),
        patch(f"{REASONING_MODULE}.fetch_additional_information") as mock_fetch,
        patch(f"{REASONING_MODULE}.do_reasoning_with_retry") as mock_reasoning,
    ):
        create = mock_anthropic.return_value.messages.create
        create.return_value = resp
        mock_fetch.return_value = ("additional info", {"pages": {}}, ["q1"], None)
        mock_reasoning.return_value = ("reasoning result", None)
        result = run(
            tool="prediction-request-reasoning-v1",
            model="claude-sonnet-4-6",
            prompt=FREE_TEXT_PROMPT,
            api_keys=api_keys,
        )
    payload = json.loads(result[0])
    assert payload["p_yes"] is None
    assert payload["error_type"] == "TruncatedCompletionError"
    assert create.call_count == 1
