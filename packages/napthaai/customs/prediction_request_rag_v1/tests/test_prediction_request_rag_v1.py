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

"""Unit tests for prediction_request_rag: thread-safe client, offline tiktoken, and source_content."""

import inspect
import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import requests

import packages.napthaai.customs.prediction_request_rag_v1.prediction_request_rag_v1 as module
from packages.napthaai.customs.prediction_request_rag_v1.prediction_request_rag_v1 import (
    ExtendedDocument,
    LLMClientManager,
    count_tokens,
    extract_texts,
    fetch_additional_information,
    multi_queries,
    run,
)

# Aliases for module-private caps: one disable each here, so call sites stay
# clean and the suppression cannot drift under formatter line-wrapping.
_QUERY_CAP = module._MAX_SEARCH_QUERY_LEN  # pylint: disable=protected-access
_SCAN_CAP = module._MAX_SCAN_CHARS  # pylint: disable=protected-access
_SHAPE_SERPER = module._shape_serper_sources  # pylint: disable=protected-access


class TestLLMClientManager:
    """Verify LLMClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_tuple(self) -> None:
        """__enter__ returns a (client, client_embedding) tuple."""
        mock_keys = {"openai": "sk-test"}
        mgr = LLMClientManager(
            api_keys=mock_keys, model="gpt-4o-2024-08-06", embedding_provider="openai"
        )
        with patch(
            "packages.napthaai.customs.prediction_request_rag_v1.prediction_request_rag_v1.LLMClient"
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

    def test_multi_queries_requires_client_param(self) -> None:
        """multi_queries requires client as first param."""
        params = list(inspect.signature(multi_queries).parameters)
        assert params[0] == "client"

    def test_fetch_additional_information_requires_client_param(self) -> None:
        """fetch_additional_information requires client as first param."""
        params = list(inspect.signature(fetch_additional_information).parameters)
        assert params[0] == "client"

    def test_multi_queries_search_query_is_required(self) -> None:
        """multi_queries takes search_query as a required param right after prompt."""
        params = inspect.signature(multi_queries).parameters
        assert list(params)[:3] == ["client", "prompt", "search_query"]
        assert params["search_query"].default is inspect.Parameter.empty

    def test_fetch_additional_information_search_query_is_required(self) -> None:
        """fetch_additional_information takes search_query as a required param."""
        params = inspect.signature(fetch_additional_information).parameters
        assert list(params)[:4] == [
            "client",
            "client_embedding",
            "prompt",
            "search_query",
        ]
        assert params["search_query"].default is inspect.Parameter.empty


RAG_MODULE = (
    "packages.napthaai.customs.prediction_request_rag_v1.prediction_request_rag_v1"
)


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
    """Verify extract_texts captures source content correctly."""

    @patch(f"{RAG_MODULE}.process_in_batches")
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

    @patch(f"{RAG_MODULE}.process_in_batches")
    def test_raw_mode_stores_html(self, mock_batches: MagicMock) -> None:
        """In raw mode, raw HTML is stored."""
        html = "<html><body>Hello world</body></html>"
        mock_batches.return_value = [[_make_html_future("http://example.com", html)]]

        _, raw_sc = extract_texts(["http://example.com"], source_content_mode="raw")

        assert raw_sc["mode"] == "raw"
        assert raw_sc["pages"]["http://example.com"] == html

    @patch(f"{RAG_MODULE}.extract_text_from_pdf")
    @patch(f"{RAG_MODULE}.process_in_batches")
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

    @patch(f"{RAG_MODULE}.extract_text_from_pdf")
    @patch(f"{RAG_MODULE}.process_in_batches")
    def test_failed_pdf_stores_empty_string(
        self, mock_batches: MagicMock, mock_pdf_extract: MagicMock
    ) -> None:
        """When extract_text_from_pdf returns None, empty string is stored."""
        mock_batches.return_value = [[_make_pdf_future("http://example.com/doc.pdf")]]
        mock_pdf_extract.return_value = None

        _, raw_sc = extract_texts(["http://example.com/doc.pdf"])

        assert raw_sc["pdfs"]["http://example.com/doc.pdf"] == ""

    @patch(f"{RAG_MODULE}.extract_text_from_pdf")
    @patch(f"{RAG_MODULE}.process_in_batches")
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

    @patch(f"{RAG_MODULE}.process_in_batches")
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

    @patch(f"{RAG_MODULE}.find_similar_chunks")
    @patch(f"{RAG_MODULE}.get_embeddings")
    @patch(f"{RAG_MODULE}.multi_queries")
    def test_cleaned_mode_uses_text_directly(
        self,
        mock_queries: MagicMock,
        mock_embeddings: MagicMock,
        mock_similar: MagicMock,
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
        mock_embeddings.return_value = [
            ExtendedDocument(text="test content here", url="http://example.com")
        ]
        mock_similar.return_value = [
            ExtendedDocument(text="test content here", url="http://example.com")
        ]

        result, raw_sc, _ = fetch_additional_information(
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

    @patch(f"{RAG_MODULE}.find_similar_chunks")
    @patch(f"{RAG_MODULE}.get_embeddings")
    @patch(f"{RAG_MODULE}.multi_queries")
    def test_raw_mode_re_extracts(
        self,
        mock_queries: MagicMock,
        mock_embeddings: MagicMock,
        mock_similar: MagicMock,
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
        mock_embeddings.return_value = [
            ExtendedDocument(text="test content", url="http://example.com")
        ]
        mock_similar.return_value = [
            ExtendedDocument(text="test content", url="http://example.com")
        ]

        result, raw_sc, _ = fetch_additional_information(
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

    @patch(f"{RAG_MODULE}.find_similar_chunks")
    @patch(f"{RAG_MODULE}.get_embeddings")
    @patch(f"{RAG_MODULE}.multi_queries")
    def test_pdfs_replayed(
        self,
        mock_queries: MagicMock,
        mock_embeddings: MagicMock,
        mock_similar: MagicMock,
    ) -> None:
        """Pdfs in source_content are loaded as ExtendedDocuments."""
        source_content = {
            "pages": {},
            "pdfs": {
                "http://example.com/doc.pdf": "pdf extracted text for testing",
            },
        }
        mock_queries.return_value = (["test query"], None)
        mock_embeddings.return_value = [
            ExtendedDocument(
                text="pdf extracted text for testing",
                url="http://example.com/doc.pdf",
            )
        ]
        mock_similar.return_value = [
            ExtendedDocument(
                text="pdf extracted text for testing",
                url="http://example.com/doc.pdf",
            )
        ]

        result, _, _ = fetch_additional_information(
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

        assert "pdf extracted text for testing" in result

    @patch(f"{RAG_MODULE}.multi_queries")
    def test_empty_source_content_raises(self, mock_queries: MagicMock) -> None:
        """Empty source_content raises ValueError (no valid documents)."""
        source_content: dict = {"pages": {}, "pdfs": {}}
        mock_queries.return_value = (["test query"], None)

        with pytest.raises(ValueError, match="No valid documents"):
            fetch_additional_information(
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


def _make_mock_api_keys(
    return_source_content: str = "false", **overrides: Any
) -> MagicMock:
    """Create a mock api_keys object (KeyChain-like) for run()."""
    services = {
        "openai": "sk-test",
        "google_api_key": None,
        "google_engine_id": None,
        "serperapi": None,
        "search_provider": "google",
        "return_source_content": return_source_content,
    }
    services.update(overrides)
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

    @patch(f"{RAG_MODULE}.parser_prediction_response", return_value='{"p_yes": 0.5}')
    @patch(f"{RAG_MODULE}.fetch_additional_information")
    @patch(f"{RAG_MODULE}.LLMClientManager")
    def test_flag_on_includes_source_content(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
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
            None,
        )

        mock_llm.completions.return_value = MagicMock(
            content="<p_yes>0.5</p_yes><p_no>0.5</p_no><confidence>0.5</confidence><info_utility>0.5</info_utility>",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        result = run(
            tool="prediction-request-rag-v1",
            model="gpt-4.1-2025-04-14",
            prompt="test",
            api_keys=_make_mock_api_keys("true"),
        )

        used_params = result[4]
        assert "source_content" in used_params

    @patch(f"{RAG_MODULE}.parser_prediction_response", return_value='{"p_yes": 0.5}')
    @patch(f"{RAG_MODULE}.fetch_additional_information")
    @patch(f"{RAG_MODULE}.LLMClientManager")
    def test_flag_off_excludes_source_content(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """When return_source_content is 'false', used_params omits source_content."""
        mock_llm = MagicMock()
        mock_embed = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=(mock_llm, mock_embed))
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)

        mock_fetch.return_value = ("additional info", {"pages": {}}, None)

        mock_llm.completions.return_value = MagicMock(
            content="<p_yes>0.5</p_yes><p_no>0.5</p_no><confidence>0.5</confidence><info_utility>0.5</info_utility>",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        result = run(
            tool="prediction-request-rag-v1",
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

    def test_anthropic_pool_exhausted_returns_typed_null(self) -> None:
        """An exhausted anthropic pool delivers the typed null, not a raised error."""
        keys = self._keys(anthropic_budget=0)

        @module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            raise _make_anthropic_error(module.anthropic.RateLimitError, "burned")

        result = fake(api_keys=keys)
        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert payload["error_type"] == "RateLimitError"
        assert payload["error"] == "burned"
        assert result[-1] is keys


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
# issue-455 free-text input contract, ported from superforcaster-polymarket-v4:
# parse_prompt tiers, the no-alphanumeric short-circuit, the empty-retrieval
# flagged null, and the typed Serper shape errors.
# ---------------------------------------------------------------------------

# Trader-template format prompt (regression: previous callers must still work)
TRADER_PROMPT = (
    'Given the question "Will X happen?" and the `yes` answer criterion, ...'
)
# Free-text prompt that would return degraded Serper results if passed raw
LONG_FREE_TEXT_PROMPT = (
    "Please predict the following market: Will Alexander Isak permanently transfer "
    "to Liverpool FC before the end of the summer 2025 transfer window (September 2, "
    "2025 23:59 UTC)? Resolution source: official club announcements or BBC Sport. "
    "The market resolves YES if a permanent transfer (not a loan) is confirmed by "
    "the resolution source before the deadline."
)


# A real Serper peopleAlsoAsk entry carries a link, not just question/snippet.
# The link is what makes the "misc is not a retrieval source" assertions
# discriminating: a harvester that reached into misc would return this URL,
# so a fixture without it cannot tell the two behaviours apart.
PAA_WITH_LINK = [
    {
        "question": "Q?",
        "snippet": "A.",
        "title": "PAA answer",
        "link": "http://paa.example.com/answer",
    },
]


def _mock_client_manager(mock_mgr: MagicMock) -> tuple:
    """Configure a mocked LLMClientManager and return its (llm, embed) pair."""
    mock_llm = MagicMock()
    mock_embed = MagicMock()
    mock_mgr.return_value.__enter__ = MagicMock(return_value=(mock_llm, mock_embed))
    mock_mgr.return_value.__exit__ = MagicMock(return_value=False)
    return mock_llm, mock_embed


VALID_TAGGED_COMPLETION = (
    "<p_yes>0.5</p_yes><p_no>0.5</p_no><confidence>0.5</confidence>"
    "<info_utility>0.5</info_utility>"
)


class TestParsePromptContract:
    """parse_prompt() -> (question_for_llm, search_query, tier)."""

    def test_trader_template_uses_extracted_question_for_both(self) -> None:
        """Trader-template path: the bare question serves as both values."""
        question, query, tier = module.parse_prompt(TRADER_PROMPT)
        assert question == "Will X happen?"
        assert query == question
        assert tier == "template"

    def test_free_text_llm_gets_full_prompt(self) -> None:
        """Free-text input: the LLM question is the whole prompt."""
        question, _, tier = module.parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert question == LONG_FREE_TEXT_PROMPT
        assert tier == "clause"

    def test_boilerplate_prefix_is_dropped_from_query(self) -> None:
        """The query anchors at the market question, dropping the lead-in."""
        _, query, _ = module.parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert query.startswith("Will Alexander Isak")
        assert query.endswith("?")
        assert len(query) <= _QUERY_CAP


class TestDegenerateShortCircuit:
    """Prompts with nothing searchable never reach the brainstorm or search."""

    @pytest.mark.parametrize("degenerate", ["", "   ", "???", '"""'])
    @patch(f"{RAG_MODULE}.get_urls_from_queries_serper")
    @patch(f"{RAG_MODULE}.get_urls_from_queries")
    @patch(f"{RAG_MODULE}.multi_queries")
    @patch(f"{RAG_MODULE}.LLMClientManager")
    def test_degenerate_prompt_short_circuits_with_zero_search_calls(
        self,
        mock_mgr: MagicMock,
        mock_queries: MagicMock,
        mock_google: MagicMock,
        mock_serper: MagicMock,
        degenerate: str,
    ) -> None:
        """Degenerate prompts return the flagged null before any network call."""
        _mock_client_manager(mock_mgr)
        result = run(
            tool="prediction-request-rag-v1",
            model="gpt-4.1-2025-04-14",
            prompt=degenerate,
            api_keys=_make_mock_api_keys(),
        )
        mock_queries.assert_not_called()
        mock_google.assert_not_called()
        mock_serper.assert_not_called()
        assert json.loads(result[0])["p_yes"] == 0.5
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "empty query"
        assert result[4]["scan_truncated"] is False

    @patch(f"{RAG_MODULE}.get_urls_from_queries_serper")
    @patch(f"{RAG_MODULE}.get_urls_from_queries")
    @patch(f"{RAG_MODULE}.find_similar_chunks")
    @patch(f"{RAG_MODULE}.get_embeddings")
    @patch(f"{RAG_MODULE}.multi_queries", return_value=(["market question"], None))
    @patch(f"{RAG_MODULE}.LLMClientManager")
    def test_replay_with_degenerate_query_uses_the_cached_capture(
        self,
        mock_mgr: MagicMock,
        mock_queries: MagicMock,
        mock_embeddings: MagicMock,
        mock_similar: MagicMock,
        mock_google: MagicMock,
        mock_serper: MagicMock,
    ) -> None:
        """A replay carries its own documents, so the short-circuit must not fire."""
        # The derived query is never used on the replay path: gating the
        # short-circuit on source_content is what keeps a captured-document
        # replay from being answered with the "empty query" null.
        mock_llm, _ = _mock_client_manager(mock_mgr)
        cached_doc = ExtendedDocument(
            text="cached evidence here", url="http://cached.example.com"
        )
        mock_embeddings.return_value = [cached_doc]
        mock_similar.return_value = [cached_doc]
        mock_llm.completions.return_value = MagicMock(
            content=VALID_TAGGED_COMPLETION,
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        source_content = {
            "mode": "cleaned",
            "pages": {"http://cached.example.com": "cached evidence here"},
            "pdfs": {},
        }

        result = run(
            tool="prediction-request-rag-v1",
            model="gpt-4.1-2025-04-14",
            prompt="???",
            api_keys=_make_mock_api_keys("true"),
            source_content=source_content,
        )

        mock_google.assert_not_called()
        mock_serper.assert_not_called()
        assert "empty_retrieval" not in result[4]
        assert result[4]["source_content"] is source_content
        assert "cached evidence here" in result[1]
        assert json.loads(result[0])["p_yes"] == 0.5


class TestEmptyRetrievalFlaggedNull:
    """Empty retrieval converges on the flagged null, not an error string."""

    @patch(f"{RAG_MODULE}.multi_queries", return_value=(["market question"], None))
    @patch(f"{RAG_MODULE}.LLMClientManager")
    def test_zero_hit_null_reason_is_live_search(
        self, mock_mgr: MagicMock, mock_queries: MagicMock
    ) -> None:
        """A genuine zero-hit records null_reason='live search'."""
        _mock_client_manager(mock_mgr)
        serper_resp = MagicMock()
        serper_resp.raise_for_status.return_value = None
        serper_resp.json.return_value = {"organic": [], "peopleAlsoAsk": []}
        with patch(f"{RAG_MODULE}.requests.request", return_value=serper_resp):
            result = run(
                tool="prediction-request-rag-v1",
                model="gpt-4.1-2025-04-14",
                prompt=LONG_FREE_TEXT_PROMPT,
                api_keys=_make_mock_api_keys(
                    search_provider="serper", serperapi="serper-test"
                ),
            )
        assert json.loads(result[0])["p_yes"] == 0.5
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "live search"
        assert result[4]["parse_tier"] == "clause"

    @patch(f"{RAG_MODULE}.multi_queries", return_value=(["market question"], None))
    @patch(f"{RAG_MODULE}.LLMClientManager")
    def test_empty_cached_replay_null_reason_is_cached_replay(
        self, mock_mgr: MagicMock, mock_queries: MagicMock
    ) -> None:
        """An empty cached capture on replay records null_reason='cached replay'."""
        _mock_client_manager(mock_mgr)
        result = run(
            tool="prediction-request-rag-v1",
            model="gpt-4.1-2025-04-14",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(),
            source_content={"pages": {}, "pdfs": {}},
        )
        assert json.loads(result[0])["p_yes"] == 0.5
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "cached replay"

    def test_malformed_serper_body_raises_typed_error(self) -> None:
        """A missing/malformed organic key raises instead of being swallowed."""
        serper_resp = MagicMock()
        serper_resp.raise_for_status.return_value = None
        serper_resp.json.return_value = {"organic": None}
        with patch(f"{RAG_MODULE}.requests.request", return_value=serper_resp):
            with pytest.raises(ValueError, match="organic"):
                module.get_urls_from_queries_serper(["q"], api_key="k", num=5)

    def test_empty_serper_body_returns_no_urls(self) -> None:
        """A well-formed zero-hit body yields no URLs without raising."""
        serper_resp = MagicMock()
        serper_resp.raise_for_status.return_value = None
        serper_resp.json.return_value = {"organic": [], "peopleAlsoAsk": []}
        with patch(f"{RAG_MODULE}.requests.request", return_value=serper_resp):
            assert not module.get_urls_from_queries_serper(["q"], api_key="k", num=5)

    def test_misc_only_body_is_well_formed_but_yields_no_urls(self) -> None:
        """Organic empty with peopleAlsoAsk populated: no raise, and no URL."""
        serper_resp = MagicMock()
        serper_resp.raise_for_status.return_value = None
        # The PAA entry carries a link: if misc were ever harvested, that link
        # would come back here and this assertion would fail.
        serper_resp.json.return_value = {
            "organic": [],
            "peopleAlsoAsk": PAA_WITH_LINK,
        }
        with patch(f"{RAG_MODULE}.requests.request", return_value=serper_resp):
            assert not module.get_urls_from_queries_serper(["q"], api_key="k", num=5)

    def test_misc_links_are_not_harvested_alongside_organic(self) -> None:
        """With both present, only the organic link becomes a source URL."""
        serper_resp = MagicMock()
        serper_resp.raise_for_status.return_value = None
        serper_resp.json.return_value = {
            "organic": [{"title": "T", "link": "http://organic.example.com/a"}],
            "peopleAlsoAsk": PAA_WITH_LINK,
        }
        with patch(f"{RAG_MODULE}.requests.request", return_value=serper_resp):
            urls = module.get_urls_from_queries_serper(["q"], api_key="k", num=5)
        assert urls == ["http://organic.example.com/a"]

    def test_shaper_returns_misc_that_the_harvester_drops(self) -> None:
        """The shaper hands misc back; dropping it is the call site's choice."""
        body = {
            "organic": [{"title": "T", "link": "http://organic.example.com/a"}],
            "peopleAlsoAsk": PAA_WITH_LINK,
        }
        organic, misc = _SHAPE_SERPER(body, "live search")
        assert organic == body["organic"]
        assert misc == PAA_WITH_LINK

    @patch(f"{RAG_MODULE}.multi_queries", return_value=(["market question"], None))
    @patch(f"{RAG_MODULE}.LLMClientManager")
    def test_misc_only_body_converges_on_the_flagged_null(
        self, mock_mgr: MagicMock, mock_queries: MagicMock
    ) -> None:
        """This tool retrieves from organic links only, so PAA alone is a zero-hit."""
        _mock_client_manager(mock_mgr)
        serper_resp = MagicMock()
        serper_resp.raise_for_status.return_value = None
        serper_resp.json.return_value = {
            "organic": [],
            "peopleAlsoAsk": PAA_WITH_LINK,
        }
        empty_capture = {"mode": "cleaned", "pages": {}, "pdfs": {}}
        with (
            patch(f"{RAG_MODULE}.requests.request", return_value=serper_resp),
            patch(
                f"{RAG_MODULE}.extract_texts", return_value=([], empty_capture)
            ) as mock_extract,
        ):
            result = run(
                tool="prediction-request-rag-v1",
                model="gpt-4.1-2025-04-14",
                prompt=LONG_FREE_TEXT_PROMPT,
                api_keys=_make_mock_api_keys(
                    search_provider="serper", serperapi="serper-test"
                ),
            )
        # Stubbing the fetcher keeps this offline AND discriminating: harvesting
        # the linked PAA entry would hand extract_texts a URL to retrieve from.
        assert mock_extract.call_args.kwargs["urls"] == []
        assert json.loads(result[0])["p_yes"] == 0.5
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "live search"


class TestRunTypedErrorNull:
    """A non-EmptyRetrieval failure reaches the requester as a parseable null."""

    @patch(f"{RAG_MODULE}.multi_queries", return_value=(["market question"], None))
    @patch(f"{RAG_MODULE}.LLMClientManager")
    def test_malformed_serper_body_is_a_typed_error_null(
        self, mock_mgr: MagicMock, mock_queries: MagicMock
    ) -> None:
        """A reshaped Serper body lands as error_type='ValueError', not a raw string."""
        _mock_client_manager(mock_mgr)
        serper_resp = MagicMock()
        serper_resp.raise_for_status.return_value = None
        serper_resp.json.return_value = {"organic": None, "peopleAlsoAsk": []}
        with patch(f"{RAG_MODULE}.requests.request", return_value=serper_resp):
            result = run(
                tool="prediction-request-rag-v1",
                model="gpt-4.1-2025-04-14",
                prompt=LONG_FREE_TEXT_PROMPT,
                api_keys=_make_mock_api_keys(
                    search_provider="serper", serperapi="serper-test"
                ),
            )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["p_no"] is None
        assert parsed["confidence"] == 0.0
        assert parsed["info_utility"] == 0.0
        assert parsed["error_type"] == "ValueError"

    @patch(f"{RAG_MODULE}.LLMClientManager")
    def test_invalid_source_content_mode_is_a_typed_error_null(
        self, mock_mgr: MagicMock
    ) -> None:
        """An unsupported source_content_mode uses the same typed-null shape."""
        _mock_client_manager(mock_mgr)
        result = run(
            tool="prediction-request-rag-v1",
            model="gpt-4.1-2025-04-14",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys(source_content_mode="bogus"),
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["error_type"] == "ValueError"


class TestRunParityAndParseMetadata:
    """run() wiring: LLM-input parity on the template path + parse metadata."""

    @staticmethod
    def _run_with_fetch_mock(prompt: str) -> tuple:
        """Run the tool with fetch + LLM mocked; return (result, fetch kwargs)."""
        with (
            patch(f"{RAG_MODULE}.LLMClientManager") as mock_mgr,
            patch(f"{RAG_MODULE}.fetch_additional_information") as mock_fetch,
        ):
            mock_llm, _ = _mock_client_manager(mock_mgr)
            mock_fetch.return_value = ("additional info", {"pages": {}}, None)
            mock_llm.completions.return_value = MagicMock(
                content=VALID_TAGGED_COMPLETION,
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            result = run(
                tool="prediction-request-rag-v1",
                model="gpt-4.1-2025-04-14",
                prompt=prompt,
                api_keys=_make_mock_api_keys(),
            )
            return result, mock_fetch.call_args.kwargs

    def test_trader_template_feeds_extracted_question_everywhere(self) -> None:
        """LLM-input parity: template path is byte-identical to extract_question."""
        result, fetch_kwargs = self._run_with_fetch_mock(TRADER_PROMPT)
        assert fetch_kwargs["prompt"] == "Will X happen?"
        assert fetch_kwargs["search_query"] == "Will X happen?"
        assert "Will X happen?" in result[1]
        assert result[4]["parse_tier"] == "template"

    def test_long_template_prompt_is_not_marked_truncated(self) -> None:
        """Template past the scan window is NOT flagged as truncated."""
        prompt = TRADER_PROMPT + " filler" * (_SCAN_CAP // 3)
        assert len(prompt) > _SCAN_CAP
        result, _ = self._run_with_fetch_mock(prompt)
        assert result[4]["parse_tier"] == "template"
        assert result[4]["scan_truncated"] is False

    def test_raw_tier_past_window_is_marked_truncated(self) -> None:
        """Question-free prompt past the window: raw tier AND truncated."""
        prompt = "no question words at all here. " * (_SCAN_CAP // 10)
        assert len(prompt) > _SCAN_CAP
        result, _ = self._run_with_fetch_mock(prompt)
        assert result[4]["parse_tier"] == "raw"
        assert result[4]["scan_truncated"] is True

    def test_clause_tier_past_window_is_marked_truncated(self) -> None:
        """A clause-tier pick on a longer-than-window prompt is still marked."""
        prompt = "Will the ECB cut rates at its next meeting? " + "filler " * (
            _SCAN_CAP // 3
        )
        assert len(prompt) > _SCAN_CAP
        result, _ = self._run_with_fetch_mock(prompt)
        assert result[4]["parse_tier"] == "clause"
        assert result[4]["scan_truncated"] is True

    def test_free_text_llm_receives_full_prompt_and_short_query(self) -> None:
        """Free text: the LLM sees the whole prompt; search gets the clause."""
        result, fetch_kwargs = self._run_with_fetch_mock(LONG_FREE_TEXT_PROMPT)
        assert "official club announcements or BBC Sport" in result[1]
        assert fetch_kwargs["prompt"] == LONG_FREE_TEXT_PROMPT
        assert fetch_kwargs["search_query"].startswith("Will Alexander Isak")


class TestSearchQueryPlumbing:
    """The compressed query replaces the raw prompt at the direct-search site."""

    def test_multi_queries_appends_search_query_not_prompt(self) -> None:
        """The direct query appended to the brainstormed ones is search_query."""
        client = MagicMock()
        client.completions.return_value = MagicMock(
            content="<queries>\nquery one\nquery two\n</queries>",
            usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        )
        queries, _ = multi_queries(
            client=client,
            prompt="LONG PROMPT",
            search_query="short q",
            model="gpt-4.1-2025-04-14",
            num_queries=2,
        )
        assert queries[-1] == "short q"
        assert "LONG PROMPT" not in queries
        # The other half of the split: the brainstorming LLM still sees the
        # whole prompt, and never the compressed query.
        user_messages = [
            message["content"]
            for message in client.completions.call_args.kwargs["messages"]
            if message["role"] == "user"
        ]
        assert len(user_messages) == 1
        assert "LONG PROMPT" in user_messages[0]
        assert "short q" not in user_messages[0]

    def test_multi_queries_dedups_repeated_search_query(self) -> None:
        """A brainstormed query equal to search_query is not searched twice."""
        client = MagicMock()
        client.completions.return_value = MagicMock(
            content="<queries>\nshort q\nquery two\n</queries>",
            usage=MagicMock(prompt_tokens=1, completion_tokens=1),
        )
        queries, _ = multi_queries(
            client=client,
            prompt="LONG PROMPT",
            search_query="short q",
            model="gpt-4.1-2025-04-14",
            num_queries=2,
        )
        assert queries == ["short q", "query two"]

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

    @patch(f"{RAG_MODULE}.get_urls_from_queries_serper", return_value=[])
    @patch(f"{RAG_MODULE}.multi_queries", side_effect=RuntimeError("boom"))
    def test_brainstorm_failure_falls_back_to_search_query(
        self, mock_queries: MagicMock, mock_serper: MagicMock
    ) -> None:
        """When the brainstorm fails, the fallback query is search_query."""
        with pytest.raises(module.EmptyRetrievalError):
            fetch_additional_information(
                client=MagicMock(),
                client_embedding=MagicMock(),
                prompt=LONG_FREE_TEXT_PROMPT,
                search_query="short q",
                model="gpt-4.1-2025-04-14",
                google_api_key=None,
                google_engine_id=None,
                serper_api_key="k",
                search_provider="serper",
            )
        mock_serper.assert_called_once()
        assert mock_serper.call_args.kwargs["queries"] == ["short q"]


def _make_bare_error(cls: type, message: str = "simulated") -> Exception:
    """Build an SDK error instance without a live transport response."""
    err: Exception = cls.__new__(cls)  # type: ignore[call-overload]
    Exception.__init__(err, message)
    return err


def _make_google_http_error(status: int, message: str = "simulated") -> Exception:
    """Build a googleapiclient HttpError carrying the given status code."""
    err = _make_bare_error(module.googleapiclient.errors.HttpError, message)
    err.resp = SimpleNamespace(  # type: ignore[attr-defined]
        status=status, reason=message
    )
    err.content = message.encode()  # type: ignore[attr-defined]
    err.uri = None  # type: ignore[attr-defined]
    err.error_details = ""  # type: ignore[attr-defined]
    err.reason = message  # type: ignore[attr-defined]
    return err


class TestWithKeyRotationNullContract:
    """No failure branch of with_key_rotation may let an exception escape."""

    @staticmethod
    def _keys() -> MagicMock:
        """Build an api_keys mock with every retry budget exhausted."""
        keys = MagicMock()
        keys.max_retries = lambda: {
            "openai": 0,
            "openrouter": 0,
            "anthropic": 0,
            "google_api_key": 0,
        }
        keys.rotate = MagicMock()
        return keys

    @staticmethod
    def _deliver(exc: Exception, keys: MagicMock) -> tuple:
        """Wrap a function that always raises exc and call it."""

        @module.with_key_rotation
        def fake(api_keys: Any) -> tuple:  # pylint: disable=unused-argument
            raise exc

        return fake(api_keys=keys)

    def test_openai_pool_exhausted_returns_typed_null(self) -> None:
        """An exhausted openai/openrouter pool delivers the typed null."""
        keys = self._keys()
        result = self._deliver(
            _make_bare_error(module.openai.RateLimitError, "burned"), keys
        )
        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert payload["p_no"] is None
        assert payload["error_type"] == "RateLimitError"
        assert result[-1] is keys

    def test_google_pool_exhausted_returns_typed_null(self) -> None:
        """An exhausted google key pool on a 429 delivers the typed null."""
        keys = self._keys()
        result = self._deliver(_make_google_http_error(429, "quota"), keys)
        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert payload["error_type"] == "HttpError"
        assert result[-1] is keys

    def test_google_non_rate_limit_error_returns_typed_null(self) -> None:
        """A non-429 google failure delivers the typed null instead of escaping."""
        keys = self._keys()
        result = self._deliver(_make_google_http_error(403, "forbidden"), keys)
        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert payload["error_type"] == "HttpError"
        assert result[-1] is keys


# Completion shapes the extractor has to survive. The forecast object is the
# one with an in-range p_yes; everything else in the string is noise.
FORECAST = {"p_yes": 0.62, "p_no": 0.38, "confidence": 0.7, "info_utility": 0.6}
FORECAST_JSON = json.dumps(FORECAST)
SCAFFOLD_COMPLETION = (
    "<facts>Isak has not been transferred.</facts>\n"
    "<thinking>Base rate for a deadline-day move is low.</thinking>\n"
    "<answer>\n" + FORECAST_JSON + "\n</answer>"
)


class TestExtractPrediction:
    """The shared extractor: which object of a completion is the forecast."""

    def test_bare_json_passes_through(self) -> None:
        """A completion that is only the forecast object round-trips."""
        assert json.loads(module.extract_prediction(FORECAST_JSON) or "") == FORECAST

    def test_reasoning_scaffold_yields_only_the_forecast(self) -> None:
        """The scaffold's prose is dropped and the answer object kept."""
        extracted = module.extract_prediction(SCAFFOLD_COMPLETION)
        assert json.loads(extracted or "") == FORECAST
        assert "<thinking>" not in (extracted or "")

    def test_draft_before_the_answer_loses_to_the_final_object(self) -> None:
        """A mid-reasoning draft cannot shadow the final forecast."""
        content = (
            'Draft: {"p_yes": 0.2, "p_no": 0.8} -- revising upward.\n' + FORECAST_JSON
        )
        assert json.loads(module.extract_prediction(content) or "") == FORECAST

    def test_trailing_non_forecast_object_is_skipped(self) -> None:
        """An object after the forecast with no p_yes is not delivered."""
        content = FORECAST_JSON + '\n{"sources": ["bbc.co.uk"]}'
        assert json.loads(module.extract_prediction(content) or "") == FORECAST

    def test_cut_mid_object_returns_none(self) -> None:
        """A max_tokens cut while writing the answer yields no forecast."""
        content = 'Reasoning done.\n{"p_yes": 0.62, "p_no": 0.3'
        assert module.extract_prediction(content) is None

    def test_cut_after_an_inner_object_closed_returns_none(self) -> None:
        """A cut whose nested object closed is still a cut, not a forecast."""
        content = '{"p_yes": 0.62, "meta": {"model": "x"}, "p_no": 0.3'
        assert module.extract_prediction(content) is None

    def test_draft_before_a_cut_is_not_delivered(self) -> None:
        """A complete draft followed by a cut answer is not the forecast."""
        content = '{"p_yes": 0.2, "p_no": 0.8}\nFinal answer:\n{"p_yes": 0.6'
        assert module.extract_prediction(content) is None

    def test_stray_brace_in_prose_is_not_a_cut(self) -> None:
        """An unclosed brace in prose must not suppress the forecast."""
        content = FORECAST_JSON + "\nSee {source for details"
        assert json.loads(module.extract_prediction(content) or "") == FORECAST

    def test_braces_and_escaped_quotes_inside_values_survive(self) -> None:
        """Braces and escaped quotes inside a string value do not break the scan."""
        payload: Dict[str, Any] = dict(FORECAST)
        payload["info"] = 'resolves if } appears in the "final" text'
        content = "Answer:\n" + json.dumps(payload)
        assert json.loads(module.extract_prediction(content) or "") == payload

    def test_out_of_range_p_yes_is_not_a_forecast(self) -> None:
        """An out-of-range p_yes is skipped, not delivered as a forecast."""
        # The earlier fixture was byte-identical to json.dumps of its own
        # parse, so passthrough and extraction were indistinguishable and the
        # mutant survived. Pair it with a valid object so the two differ.
        # The out-of-range object must be LAST: the walk runs in reverse, so
        # a valid object after it would be found first and the skip would
        # never be exercised.
        completion = (
            '{"p_yes": 0.3, "p_no": 0.7, "confidence": 0.6, "info_utility": 0.5}\n'
            '{"p_yes": 1.7, "p_no": -0.7}'
        )
        assert json.loads(module.extract_prediction(completion) or "")["p_yes"] == 0.3

    def test_null_p_yes_is_not_a_forecast(self) -> None:
        """A sole null p_yes yields None, not the unusable object."""
        # Previously this returned the content unchanged, which delivered the
        # unusable object on-chain when it was the ONLY candidate. Objects were
        # present and none carried a usable p_yes, so there is nothing to
        # deliver and the caller's guard turns it into a retry.
        content = '{"p_yes": null, "p_no": null}'
        assert module.extract_prediction(content) is None

    def test_empty_content_is_returned_unchanged(self) -> None:
        """Empty or missing content is passed straight back."""
        assert module.extract_prediction("") == ""
        assert module.extract_prediction(None) is None

    @patch(f"{RAG_MODULE}.requests.request")
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
            module.get_urls_from_queries_serper(["q1", "q2"], "key", num=3)


class TestParserPredictionResponse:
    """parser_prediction_response: tag form, JSON fallback, and error surfacing."""

    def test_tag_form_still_parsed(self) -> None:
        """The documented tag form keeps producing the four-field JSON."""
        parsed = json.loads(module.parser_prediction_response(VALID_TAGGED_COMPLETION))
        assert parsed == {
            "p_yes": 0.5,
            "p_no": 0.5,
            "info_utility": 0.5,
            "confidence": 0.5,
        }

    def test_json_answer_falls_back_to_the_extractor(self) -> None:
        """A JSON answer with no tags is parsed by the extractor, not rejected."""
        parsed = json.loads(module.parser_prediction_response(SCAFFOLD_COMPLETION))
        assert parsed == FORECAST

    def test_missing_tag_raises_value_error_naming_the_tag(self) -> None:
        """A half-written tag block raises ValueError carrying the real error."""
        broken = "<p_yes>0.5</p_yes><p_no>0.5</p_no>"
        with pytest.raises(ValueError) as excinfo:
            module.parser_prediction_response(broken)
        assert "info_utility" in str(excinfo.value)
        assert "IndexError" in str(excinfo.value)
        assert not isinstance(excinfo.value, UnboundLocalError)

    def test_non_numeric_tag_value_raises_value_error(self) -> None:
        """A non-numeric tag value reports the float failure, not an unbound name."""
        broken = (
            "<p_yes>very likely</p_yes><p_no>0.5</p_no>"
            "<confidence>0.5</confidence><info_utility>0.5</info_utility>"
        )
        with pytest.raises(ValueError) as excinfo:
            module.parser_prediction_response(broken)
        assert "p_yes" in str(excinfo.value)
        assert "UnboundLocalError" not in str(excinfo.value)

    def test_prose_without_a_forecast_raises_value_error(self) -> None:
        """A completion with neither tags nor a forecast object is an error."""
        with pytest.raises(ValueError, match="No forecast object"):
            module.parser_prediction_response("I cannot answer that question.")


def _run_with_completion(content: str, prompt: str = "Will X happen?") -> tuple:
    """Run the tool end to end with fetch + the LLM mocked to deliver content."""
    with (
        patch(f"{RAG_MODULE}.LLMClientManager") as mock_mgr,
        patch(f"{RAG_MODULE}.fetch_additional_information") as mock_fetch,
    ):
        mock_llm, _ = _mock_client_manager(mock_mgr)
        mock_fetch.return_value = ("additional info", {"pages": {}}, None)
        mock_llm.completions.return_value = MagicMock(
            content=content,
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        return run(
            tool="prediction-request-rag-v1",
            model="gpt-4.1-2025-04-14",
            prompt=prompt,
            api_keys=_make_mock_api_keys(),
        )


class TestRunDeliversTheForecastObject:
    """run() wiring: the extractor must sit on the delivery path, not beside it."""

    def test_free_text_json_completion_is_delivered_as_the_forecast(self) -> None:
        """A JSON answer to a free-text prompt is delivered as the forecast object."""
        result = _run_with_completion(SCAFFOLD_COMPLETION, LONG_FREE_TEXT_PROMPT)
        assert json.loads(result[0]) == FORECAST
        assert "<thinking>" not in result[0]

    def test_bare_json_completion_is_delivered_unwrapped(self) -> None:
        """A bare JSON completion reaches the caller as a parseable forecast."""
        result = _run_with_completion(FORECAST_JSON)
        assert json.loads(result[0])["p_yes"] == 0.62

    def test_tagged_completion_delivery_is_unchanged(self) -> None:
        """The tag-form path still delivers the four-field JSON."""
        result = _run_with_completion(VALID_TAGGED_COMPLETION)
        assert json.loads(result[0]) == {
            "p_yes": 0.5,
            "p_no": 0.5,
            "info_utility": 0.5,
            "confidence": 0.5,
        }

    def test_truncated_completion_delivers_the_typed_error_null(self) -> None:
        """A cut-off answer becomes the typed null, never an escaping exception."""
        result = _run_with_completion('Reasoning.\n{"p_yes": 0.62, "p_no": 0.3')
        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert payload["error_type"] == "ValueError"
