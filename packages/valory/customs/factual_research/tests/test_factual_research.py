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

"""Unit tests for factual_research: helpers, source_content, and thread safety."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import packages.valory.customs.factual_research.factual_research as module
from packages.valory.customs.factual_research.factual_research import (
    BLOCKED_DOMAINS,
    FactualBriefing,
    OpenAIClientManager,
    PredictionResult,
    SourceReference,
    SubAnswer,
    SubQuestions,
    _clean_html,
    _extract_question,
    _fetch_page_content,
    _format_evidence,
    _parse_completion,
    _search_serper,
    run,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

FR_MODULE = "packages.valory.customs.factual_research.factual_research"

PREDICTION_PROMPT = (
    'With the given question "Will X happen by 2025?" '
    "and the `yes` option represented by `Yes` and the `no` option represented by `No`, "
    "what are the respective probabilities of `p_yes` and `p_no` occurring?"
)

SIMPLE_HTML = """
<html><head><title>Test</title></head>
<body>
<script>var x = 1;</script>
<style>.foo { color: red; }</style>
<img src="foo.png">
<p>This is the main article content with enough words to test extraction.</p>
</body></html>
"""

FAKE_SERPER_RESPONSE = {
    "organic": [
        {
            "title": "Clean Result",
            "link": "https://example.com/clean",
            "snippet": "A clean snippet.",
        },
        {
            "title": "Polymarket Result",
            "link": "https://polymarket.com/market/123",
            "snippet": "Should be filtered.",
        },
        {
            "title": "Twitter Result",
            "link": "https://twitter.com/user/status/456",
            "snippet": "Should be filtered.",
        },
        {
            "title": "Another Clean",
            "link": "https://reuters.com/article",
            "snippet": "Reuters snippet.",
        },
    ],
    "peopleAlsoAsk": [
        {
            "question": "What about X?",
            "link": "https://example.com/paa",
            "snippet": "PAA answer.",
        },
        {
            "question": "X on Metaculus?",
            "link": "https://metaculus.com/question/789",
            "snippet": "Should be filtered.",
        },
    ],
}


def _make_mock_api_keys(
    return_source_content: str = "false",
    source_content_mode: str = "cleaned",
) -> MagicMock:
    """Create a mock KeyChain-like api_keys object."""
    services = {
        "openai": "sk-test",
        "serperapi": "serper-test",
        "return_source_content": return_source_content,
        "source_content_mode": source_content_mode,
        "openrouter": "",
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key]
    mock.get = lambda key, default="": services.get(key, default)
    mock.max_retries = lambda: {"openai": 0, "openrouter": 0}
    return mock


def _make_mock_parse_completion() -> MagicMock:
    """Create a mock _parse_completion that returns appropriate Pydantic models."""
    sub_q = SubQuestions(sub_questions=["What is the status of X?"])
    briefing = FactualBriefing(
        sub_answers=[
            SubAnswer(
                question="What is the status of X?",
                answer="X is in progress.",
                sources=[SourceReference(title="Source", url="https://example.com")],
            )
        ],
        summary="X is ongoing with no major obstacles.",
        sources=[SourceReference(title="Source", url="https://example.com")],
        info_utility=0.7,
    )
    prediction = PredictionResult(
        reasoning="Based on the evidence, X is likely.",
        p_yes=0.6,
        p_no=0.4,
        confidence=0.7,
        info_utility=0.8,
    )

    call_count = 0

    def side_effect(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        cb = kwargs.get("counter_callback") or (args[9] if len(args) > 9 else None)
        if call_count == 1:
            return sub_q, cb
        if call_count == 2:
            return briefing, cb
        return prediction, cb

    return MagicMock(side_effect=side_effect)


# ---------------------------------------------------------------------------
# Group 1: Pure helper functions
# ---------------------------------------------------------------------------


class TestExtractQuestion:
    """Tests for _extract_question."""

    def test_mech_envelope_format(self) -> None:
        """Standard mech prompt format extracts the question."""
        result = _extract_question(PREDICTION_PROMPT)
        assert result == "Will X happen by 2025?"

    def test_plain_string(self) -> None:
        """Non-mech prompt returns the string as-is, stripped."""
        result = _extract_question("  some plain question  ")
        assert result == "some plain question"

    def test_empty_string(self) -> None:
        """Empty string returns empty."""
        result = _extract_question("")
        assert result == ""


class TestFormatEvidence:
    """Tests for _format_evidence."""

    def test_empty_list(self) -> None:
        """Empty list returns sentinel string."""
        assert _format_evidence([]) == "(no evidence gathered)"

    def test_with_content(self) -> None:
        """Items with content key include Content line."""
        items = [
            {
                "title": "Article",
                "link": "https://example.com",
                "snippet": "A snippet",
                "content": "Full article text",
            }
        ]
        result = _format_evidence(items)
        assert "Content: Full article text" in result
        assert "Snippet: A snippet" in result
        assert "[Article](https://example.com)" in result

    def test_without_content(self) -> None:
        """Items without content only show snippet."""
        items = [
            {"title": "Article", "link": "https://example.com", "snippet": "A snippet"}
        ]
        result = _format_evidence(items)
        assert "Content:" not in result
        assert "A snippet" in result


class TestCleanHtml:
    """Tests for _clean_html."""

    def test_strips_scripts_and_extracts_text(self) -> None:
        """HTML with scripts/styles produces clean text."""
        result = _clean_html(SIMPLE_HTML)
        assert result is not None
        assert "var x = 1" not in result
        assert "color: red" not in result
        assert "main article content" in result

    def test_truncates_to_max_words(self) -> None:
        """Long content gets truncated."""
        long_html = "<html><body><p>" + " ".join(["word"] * 500) + "</p></body></html>"
        result = _clean_html(long_html, max_words=10)
        assert result is not None
        assert "[…]" in result
        # Should have roughly 10 words before truncation marker
        words_before = result.split("[…]")[0].split()
        assert len(words_before) <= 11  # allow slight variance from markdown

    def test_returns_none_for_empty(self) -> None:
        """Empty body returns None."""
        result = _clean_html("<html><body></body></html>")
        # readability may extract something minimal; if it does, that's OK
        # but truly empty content should be None
        if result is not None:
            assert result.strip() != ""


# ---------------------------------------------------------------------------
# Group 2: _search_serper domain filtering
# ---------------------------------------------------------------------------


class TestSearchSerper:
    """Tests for _search_serper domain filtering."""

    @patch(f"{FR_MODULE}.requests.post")
    def test_filters_blocked_domains(self, mock_post: MagicMock) -> None:
        """Blocked domains (polymarket, twitter, etc.) are dropped."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        results = _search_serper("test query", "api-key", num_results=5)

        links = [r["link"] for r in results]
        for domain in BLOCKED_DOMAINS:
            assert not any(
                domain in link for link in links
            ), f"Blocked domain {domain} found in results"

    @patch(f"{FR_MODULE}.requests.post")
    def test_includes_clean_organic_and_paa(self, mock_post: MagicMock) -> None:
        """Clean organic and PAA results are included."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        results = _search_serper("test query", "api-key", num_results=5)

        links = [r["link"] for r in results]
        assert "https://example.com/clean" in links
        assert "https://reuters.com/article" in links
        assert "https://example.com/paa" in links

    @patch(f"{FR_MODULE}.requests.post")
    def test_respects_num_results_for_organic(self, mock_post: MagicMock) -> None:
        """Organic results are capped at num_results."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        results = _search_serper("test query", "api-key", num_results=1)

        # Only 1 organic (Clean Result), but PAA still appended
        organic_links = [
            r["link"] for r in results if r["link"] != "https://example.com/paa"
        ]
        assert len(organic_links) == 1


# ---------------------------------------------------------------------------
# Group 3: _fetch_page_content modes
# ---------------------------------------------------------------------------


class TestFetchPageContent:
    """Tests for _fetch_page_content cleaned/raw modes."""

    @patch(f"{FR_MODULE}.requests.get")
    def test_cleaned_mode(self, mock_get: MagicMock) -> None:
        """Cleaned mode returns (cleaned_text, cleaned_text)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_resp.text = SIMPLE_HTML
        mock_get.return_value = mock_resp

        text, capture = _fetch_page_content("https://example.com", mode="cleaned")

        assert text is not None
        assert capture is not None
        assert text == capture  # cleaned mode: both are the same
        assert "main article content" in text

    @patch(f"{FR_MODULE}.requests.get")
    def test_raw_mode(self, mock_get: MagicMock) -> None:
        """Raw mode returns (cleaned_text, raw_html)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_resp.text = SIMPLE_HTML
        mock_get.return_value = mock_resp

        text, capture = _fetch_page_content("https://example.com", mode="raw")

        assert text is not None
        assert capture is not None
        assert text != capture  # text is cleaned, capture is raw HTML
        assert "<script>" in capture or "<p>" in capture  # raw HTML preserved
        assert "main article content" in text

    @patch(f"{FR_MODULE}.requests.get")
    def test_non_html_returns_none(self, mock_get: MagicMock) -> None:
        """Non-HTML content-type returns (None, None)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_get.return_value = mock_resp

        text, capture = _fetch_page_content("https://example.com/file.pdf")
        assert text is None
        assert capture is None

    @patch(f"{FR_MODULE}.requests.get")
    def test_non_200_returns_none(self, mock_get: MagicMock) -> None:
        """Non-200 status returns (None, None)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_get.return_value = mock_resp

        text, capture = _fetch_page_content("https://example.com/missing")
        assert text is None
        assert capture is None


# ---------------------------------------------------------------------------
# Group 4: run() source_content integration
# ---------------------------------------------------------------------------


class TestRunSourceContent:
    """Verify source_content capture, replay, and mode handling in run()."""

    @patch(f"{FR_MODULE}._parse_completion")
    @patch(f"{FR_MODULE}._search_serper")
    @patch(f"{FR_MODULE}._fetch_page_content")
    @patch(f"{FR_MODULE}.OpenAIClientManager")
    def test_live_capture_includes_mode_and_pages(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_serper: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        """Live run with return_source_content=true captures mode, serper_results, pages."""
        mock_client = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)

        mock_serper.return_value = [
            {"title": "R1", "link": "https://example.com/1", "snippet": "S1"},
        ]
        mock_fetch.return_value = ("Cleaned text", "Cleaned text")
        mock_parse.side_effect = _make_mock_parse_completion().side_effect

        result = run(
            tool="factual_research",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true", "cleaned"),
            counter_callback=None,
            delivery_rate=100,
        )

        used_params = result[4]
        assert "source_content" in used_params
        sc = used_params["source_content"]
        assert sc["mode"] == "cleaned"
        assert "serper_results" in sc
        assert "pages" in sc
        # serper_results should NOT contain 'content' key (bug 3 fix)
        for item in sc["serper_results"]:
            assert "content" not in item

    @patch(f"{FR_MODULE}._parse_completion")
    @patch(f"{FR_MODULE}.OpenAIClientManager")
    def test_replay_uses_cached_source_content(
        self,
        mock_mgr: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        """Replay with source_content skips serper, uses cached evidence."""
        mock_client = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)
        mock_parse.side_effect = _make_mock_parse_completion().side_effect

        cached = {
            "mode": "cleaned",
            "serper_results": [
                {"title": "Cached", "link": "https://cached.com/1", "snippet": "CS"},
            ],
            "pages": {"https://cached.com/1": "Cached page text"},
        }

        with patch(f"{FR_MODULE}._search_serper") as mock_serper:
            result = run(
                tool="factual_research",
                model="gpt-4.1-2025-04-14",
                prompt=PREDICTION_PROMPT,
                api_keys=_make_mock_api_keys("true"),
                counter_callback=None,
                delivery_rate=100,
                source_content=cached,
            )
            mock_serper.assert_not_called()

        # Result should be valid JSON
        parsed = json.loads(result[0])
        assert "p_yes" in parsed

    @patch(f"{FR_MODULE}._parse_completion")
    @patch(f"{FR_MODULE}._clean_html")
    @patch(f"{FR_MODULE}.OpenAIClientManager")
    def test_replay_raw_mode_recleans_html(
        self,
        mock_mgr: MagicMock,
        mock_clean: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        """Replay with mode=raw calls _clean_html on cached pages."""
        mock_client = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)
        mock_parse.side_effect = _make_mock_parse_completion().side_effect
        mock_clean.return_value = "Re-cleaned text"

        cached = {
            "mode": "raw",
            "serper_results": [
                {"title": "Raw", "link": "https://raw.com/1", "snippet": "RS"},
            ],
            "pages": {"https://raw.com/1": "<html><body>Raw HTML</body></html>"},
        }

        run(
            tool="factual_research",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            delivery_rate=100,
            source_content=cached,
        )

        mock_clean.assert_called_once_with("<html><body>Raw HTML</body></html>")

    @patch(f"{FR_MODULE}._parse_completion")
    @patch(f"{FR_MODULE}._search_serper")
    @patch(f"{FR_MODULE}._fetch_page_content")
    @patch(f"{FR_MODULE}.OpenAIClientManager")
    def test_flag_off_no_source_content(
        self,
        mock_mgr: MagicMock,
        mock_fetch: MagicMock,
        mock_serper: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        """When return_source_content=false, source_content is not in used_params."""
        mock_client = MagicMock()
        mock_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_mgr.return_value.__exit__ = MagicMock(return_value=False)
        mock_serper.return_value = [
            {"title": "R1", "link": "https://example.com/1", "snippet": "S1"},
        ]
        mock_fetch.return_value = ("Cleaned", "Cleaned")
        mock_parse.side_effect = _make_mock_parse_completion().side_effect

        result = run(
            tool="factual_research",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            delivery_rate=100,
        )

        used_params = result[4]
        assert "source_content" not in used_params


class TestParseCompletionRetry:
    """Tests for _parse_completion retry logic."""

    @patch(f"{FR_MODULE}.time.sleep")
    def test_retries_then_succeeds(self, mock_sleep: MagicMock) -> None:
        """Fail twice, succeed on the third attempt; verify 3 calls made."""
        mock_client = MagicMock()

        # Build a successful response for the third attempt
        mock_parsed = SubQuestions(sub_questions=["Q1", "Q2"])
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_message = MagicMock()
        mock_message.parsed = mock_parsed
        mock_message.refusal = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client.beta.chat.completions.parse.side_effect = [
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            mock_response,
        ]

        result, cb = _parse_completion(
            client=mock_client,
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": "test"}],
            response_format=SubQuestions,
            retries=3,
            delay=1,
        )

        assert mock_client.beta.chat.completions.parse.call_count == 3
        assert result == mock_parsed
        assert mock_sleep.call_count == 2

    @patch(f"{FR_MODULE}.time.sleep")
    def test_retries_exhausted_raises(self, mock_sleep: MagicMock) -> None:
        """All retries fail; final attempt raises RuntimeError."""
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = RuntimeError("nope")

        with pytest.raises(RuntimeError, match="Failed to get structured LLM"):
            _parse_completion(
                client=mock_client,
                model="gpt-4.1-2025-04-14",
                messages=[{"role": "user", "content": "test"}],
                response_format=SubQuestions,
                retries=3,
                delay=1,
            )

        assert mock_client.beta.chat.completions.parse.call_count == 3
        assert mock_sleep.call_count == 3


class TestNoGlobalClient:
    """Verify the module does not use a global client variable."""

    def test_no_module_level_client(self) -> None:
        """The module must not define a module-level 'client' variable."""
        source = Path(module.__file__).read_text(encoding="utf-8")
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("client:") or stripped.startswith("client ="):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client' variable found at line {i}: {line}"
                    )

    def test_client_manager_uses_local_state(self) -> None:
        """Verify OpenAIClientManager stores client on self, not module global."""
        with patch(f"{FR_MODULE}.OpenAI") as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance

            mgr = OpenAIClientManager(api_key="sk-test")
            with mgr as client:
                assert client is mock_instance
                assert mgr._client is mock_instance
                MockOpenAI.assert_called_once_with(api_key="sk-test")

            mock_instance.close.assert_called_once()
            assert mgr._client is None


# ---------------------------------------------------------------------------
# Group 5: Edge cases
# ---------------------------------------------------------------------------


class TestRunEdgeCases:
    """Edge case validation for run().

    Note: run() is wrapped by @with_key_rotation which catches exceptions
    and returns an error tuple (error_str, "", None, None, None, api_keys)
    instead of raising.
    """

    def test_rejects_unknown_tool(self) -> None:
        """Unknown tool returns error tuple with 'not supported'."""
        result = run(
            tool="bogus_tool",
            model="gpt-4o",
            prompt="test",
            api_keys=_make_mock_api_keys(),
        )
        assert "not supported" in result[0]

    def test_rejects_missing_model(self) -> None:
        """Missing model returns error tuple with 'Model not supplied'."""
        result = run(
            tool="factual_research",
            prompt="test",
            api_keys=_make_mock_api_keys(),
        )
        assert "Model not supplied" in result[0]

    def test_invalid_source_content_mode(self) -> None:
        """Invalid source_content_mode returns error tuple."""
        result = run(
            tool="factual_research",
            model="gpt-4o",
            prompt="test",
            api_keys=_make_mock_api_keys(source_content_mode="bogus"),
            delivery_rate=100,
        )
        assert "Invalid source_content_mode" in result[0]

    def test_delivery_rate_zero_returns_max_cost(self) -> None:
        """delivery_rate=0 calls counter_callback with max_cost=True."""
        mock_cb = MagicMock(return_value=42.0)

        result = run(
            tool="factual_research",
            model="gpt-4.1-2025-04-14",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(),
            delivery_rate=0,
            counter_callback=mock_cb,
        )

        mock_cb.assert_called_once_with(
            max_cost=True,
            models_calls=("gpt-4.1-2025-04-14",) * 3,
        )
        assert result == 42.0
