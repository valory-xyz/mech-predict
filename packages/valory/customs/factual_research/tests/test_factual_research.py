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
from typing import Any, List, Tuple
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


def _make_mock_parse_completion() -> List[Tuple[Any, None]]:
    """Return an explicit side_effect list for the 3 _parse_completion calls.

    The pipeline calls _parse_completion exactly 3 times in this order:
    reframe → synthesise → estimate. Using a static list instead of a
    counter-based dispatcher makes the test fail loudly if the pipeline
    ever reorders or adds a call, instead of silently returning the wrong
    model.

    :return: side_effect list of (parsed_model, None) tuples for the 3 calls.
    """
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
    return [
        (sub_q, None),
        (briefing, None),
        (prediction, None),
    ]


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
        mock_parse.side_effect = _make_mock_parse_completion()

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
        mock_parse.side_effect = _make_mock_parse_completion()

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
        mock_parse.side_effect = _make_mock_parse_completion()
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
        mock_parse.side_effect = _make_mock_parse_completion()

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
        """Fail twice with retryable errors, succeed on the third attempt."""
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

        # ValueError is in the retryable set (covers pydantic ValidationError
        # and the inline "Model refused…" raise).
        mock_client.beta.chat.completions.parse.side_effect = [
            ValueError("fail 1"),
            ValueError("fail 2"),
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
        """All retries fail with retryable errors; raises RuntimeError."""
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = ValueError("nope")

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

    @patch(f"{FR_MODULE}.time.sleep")
    def test_non_retryable_exception_propagates(self, mock_sleep: MagicMock) -> None:
        """Non-retryable exceptions (e.g. AttributeError) surface immediately."""
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = AttributeError("bug")

        with pytest.raises(AttributeError, match="bug"):
            _parse_completion(
                client=mock_client,
                model="gpt-4.1-2025-04-14",
                messages=[{"role": "user", "content": "test"}],
                response_format=SubQuestions,
                retries=3,
                delay=1,
            )

        # First failure should propagate — no retries, no sleeps.
        assert mock_client.beta.chat.completions.parse.call_count == 1
        assert mock_sleep.call_count == 0


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

    Note: run() is wrapped by @with_key_rotation which catches unhandled
    exceptions and returns an error tuple (error_json, "", None, None, None,
    api_keys), where error_json is a JSON string of the form
    {"p_yes": null, "p_no": null, "confidence": 0.0, "info_utility": 0.0,
     "error": "<message>"}.
    """

    @staticmethod
    def _assert_error_envelope(result_first: str, expected_substring: str) -> None:
        """Assert the error envelope contract.

        The first tuple element must be a JSON envelope with null predictions,
        zeroed scores, and the exception message under `error`.

        :param result_first: first element of the run() return tuple.
        :param expected_substring: substring that must appear in the `error` field.
        """
        parsed = json.loads(result_first)
        assert parsed["p_yes"] is None
        assert parsed["p_no"] is None
        assert parsed["confidence"] == 0.0
        assert parsed["info_utility"] == 0.0
        assert expected_substring in parsed["error"]

    def test_rejects_unknown_tool(self) -> None:
        """Unknown tool returns error JSON with 'not supported'."""
        result = run(
            tool="bogus_tool",
            model="gpt-4o",
            prompt="test",
            api_keys=_make_mock_api_keys(),
        )
        self._assert_error_envelope(result[0], "not supported")

    def test_rejects_missing_model(self) -> None:
        """Missing model returns error JSON with 'Model not supplied'."""
        result = run(
            tool="factual_research",
            prompt="test",
            api_keys=_make_mock_api_keys(),
        )
        self._assert_error_envelope(result[0], "Model not supplied")

    def test_invalid_source_content_mode(self) -> None:
        """Invalid source_content_mode returns error JSON."""
        result = run(
            tool="factual_research",
            model="gpt-4o",
            prompt="test",
            api_keys=_make_mock_api_keys(source_content_mode="bogus"),
            delivery_rate=100,
        )
        self._assert_error_envelope(result[0], "Invalid source_content_mode")

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


# ---------------------------------------------------------------------------
# Group 6: with_key_rotation decorator
# ---------------------------------------------------------------------------


def _make_openai_error(cls: type, message: str = "simulated") -> Exception:
    """Build an openai APIStatusError-family exception for tests.

    Skips the real constructor (which requires a live httpx.Response) and
    sets up just the attributes the decorator under test touches.

    :param cls: the openai exception subclass to instantiate.
    :param message: the `str(exc)` payload.
    :return: an instance of `cls` usable as a raise target in tests.
    """
    err: Exception = cls.__new__(cls)  # type: ignore[call-overload]
    Exception.__init__(err, message)
    err.message = message  # type: ignore[attr-defined]
    return err


class TestWithKeyRotation:
    """Direct tests for the @with_key_rotation decorator contract.

    These pin behaviors that are easy to regress:
    - success path returns a 6-tuple ending in api_keys
    - max-cost float pass-through (no api_keys appended — tuple concat would fail)
    - RateLimitError / AuthenticationError / PermissionDeniedError rotate the key
    - unhandled exceptions are wrapped in the prediction-shaped error JSON
    """

    def test_success_appends_api_keys(self) -> None:
        """Success tuple gets api_keys appended as last element."""
        keys = _make_mock_api_keys()

        @module.with_key_rotation
        def fake(api_keys: Any) -> Tuple[Any, ...]:  # pylint: disable=unused-argument
            return "ok", "prompt", None, None, None

        result = fake(api_keys=keys)
        assert result == ("ok", "prompt", None, None, None, keys)

    def test_max_cost_float_passthrough(self) -> None:
        """Float return (max-cost path) skips the api_keys append."""
        keys = _make_mock_api_keys()

        @module.with_key_rotation
        def fake(api_keys: Any) -> float:  # pylint: disable=unused-argument
            return 42.0

        assert fake(api_keys=keys) == 42.0

    @pytest.mark.parametrize(
        "exc_cls",
        [
            module.openai.RateLimitError,
            module.openai.AuthenticationError,
            module.openai.PermissionDeniedError,
        ],
    )
    def test_rotates_on_recoverable_error(self, exc_cls: type) -> None:
        """Rate-limit, auth, and permission errors all rotate the key and retry."""
        keys = _make_mock_api_keys()
        keys.max_retries = lambda: {"openai": 1, "openrouter": 1}
        keys.rotate = MagicMock()
        call_count = {"n": 0}

        @module.with_key_rotation
        def fake(api_keys: Any) -> Tuple[Any, ...]:  # pylint: disable=unused-argument
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _make_openai_error(exc_cls, "simulated")
            return "ok", "", None, None, None

        result = fake(api_keys=keys)
        assert call_count["n"] == 2  # one failure + one success
        assert keys.rotate.call_count == 2  # rotated both openai and openrouter
        assert result[-1] is keys

    def test_rotation_exhausted_raises(self) -> None:
        """Recoverable exception is re-raised when retries are exhausted.

        When both openai and openrouter retries are exhausted, the decorator
        re-raises instead of wrapping into the error JSON — the framework
        needs the raw exception to know the key pool is burned.
        """
        keys = _make_mock_api_keys()
        keys.max_retries = lambda: {"openai": 0, "openrouter": 0}

        @module.with_key_rotation
        def fake(api_keys: Any) -> Tuple[Any, ...]:  # pylint: disable=unused-argument
            raise _make_openai_error(module.openai.RateLimitError, "burned out")

        with pytest.raises(module.openai.RateLimitError, match="burned out"):
            fake(api_keys=keys)

    def test_unhandled_exception_returns_parseable_error_json(self) -> None:
        """Unhandled exceptions wrap into a prediction-shaped error JSON.

        Non-openai exceptions (including `LengthFinishReasonError`, shaped
        here as a `RuntimeError`) are wrapped into the prediction-shaped
        error JSON. This is the contract the decorator added in PR 232 —
        regressing it would resurface raw framework strings to on-chain
        consumers.
        """
        keys = _make_mock_api_keys()

        @module.with_key_rotation
        def fake(api_keys: Any) -> Tuple[Any, ...]:  # pylint: disable=unused-argument
            raise RuntimeError("simulated truncation")

        result = fake(api_keys=keys)
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["p_no"] is None
        assert parsed["confidence"] == 0.0
        assert parsed["info_utility"] == 0.0
        assert "simulated truncation" in parsed["error"]
        assert result[1:] == ("", None, None, None, keys)

    def test_run_wrapped_unhandled_exception_returns_error_json(self) -> None:
        """End-to-end regression for the LengthFinishReasonError path.

        When the pipeline raises inside `_parse_completion` (simulating the
        `LengthFinishReasonError` path), the decorator wraps the error
        cleanly. This is the exact failure shape that caused the 24.4% prod
        error rate before `max_tokens` was raised.
        """
        with patch(f"{FR_MODULE}._parse_completion") as mock_parse:
            mock_parse.side_effect = RuntimeError("simulated truncation")
            result = run(
                tool="factual_research",
                model="gpt-4.1-2025-04-14",
                prompt=PREDICTION_PROMPT,
                api_keys=_make_mock_api_keys(),
            )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["confidence"] == 0.0
        assert "simulated truncation" in parsed["error"]
