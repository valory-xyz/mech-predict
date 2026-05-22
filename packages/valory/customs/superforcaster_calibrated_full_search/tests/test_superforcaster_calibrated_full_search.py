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

"""Unit tests for superforcaster_calibrated_full_search.

Covers the v0.18.1-base shape (Structured Outputs via Pydantic) plus the
new evidence-gathering layer (page scrape, capture/replay, fallbacks,
evidence-block token cap).
"""

import inspect
import json
from typing import Optional
from unittest.mock import MagicMock, patch

import openai
import pytest
import requests
from pydantic import ValidationError

from packages.valory.customs.superforcaster_calibrated_full_search.superforcaster_calibrated_full_search import (
    MAX_EVIDENCE_TOKENS,
    OpenAIClientManager,
    PredictionResult,
    _cap_evidence_block,
    _fetch_page_content,
    _parse_completion,
    _scrape_pages,
    count_tokens,
    fetch_additional_sources,
    run,
)

SFC_MODULE = (
    "packages.valory.customs.superforcaster_calibrated_full_search"
    ".superforcaster_calibrated_full_search"
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
        {
            "title": "Second Result",
            "link": "http://example.com/second",
            "snippet": "Second snippet",
            "position": 2,
        },
    ],
    "peopleAlsoAsk": [
        {"question": "What is test?", "snippet": "A test answer."},
    ],
}

FAKE_PAGE_CONTENT = "Extracted main article body about the test topic."
FAKE_FETCH_RESULTS = {
    "http://example.com/result": (FAKE_PAGE_CONTENT, FAKE_PAGE_CONTENT),
    "http://example.com/second": ("Second page body.", "Second page body."),
}

# Real HTML that readability + markdownify extract into non-empty article text.
_HTML_PAGE = (
    "<html><head><title>Fed decision</title></head><body><article>"
    "<h1>Federal Reserve holds rates</h1>"
    "<p>The Federal Reserve held interest rates steady on Wednesday, citing "
    "persistent inflation concerns and a resilient labor market. Officials "
    "signaled they expect two more cuts before the end of the year.</p>"
    "<p>The decision was widely expected by economists surveyed beforehand. "
    "Markets moved modestly higher following the announcement as investors "
    "digested the updated projections.</p>"
    "</article></body></html>"
)


def _fake_fetch(
    url: str, mode: str = "cleaned", **_: object
) -> tuple[Optional[str], Optional[str]]:
    """Stand-in for _fetch_page_content that never touches the network."""
    return FAKE_FETCH_RESULTS.get(url, (None, None))


PREDICTION_PROMPT = (
    'With the given question "Will X happen?" '
    "and the `yes` option represented by `Yes` and the `no` option represented by `No`, "
    "what are the respective probabilities of `p_yes` and `p_no` occurring?"
)


def _make_mock_api_keys(
    return_source_content: str = "false", source_content_mode: str = "cleaned"
) -> MagicMock:
    """Create a mock KeyChain-like api_keys object."""
    services = {
        "openai": ["sk-test"],
        "serperapi": ["serper-test"],
        "return_source_content": [return_source_content],
        "source_content_mode": [source_content_mode],
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key][0]
    mock.get = lambda key, default="": services.get(key, [default])[0]
    return mock


def _make_prediction_stub() -> PredictionResult:
    """Build a valid PredictionResult instance for use as a parse() return."""
    return PredictionResult(
        facts="some facts",
        reasons_no="reasons no",
        reasons_yes="reasons yes",
        aggregation="aggregation block",
        reflection="reflection block",
        p_yes=0.5,
        p_no=0.5,
        confidence=0.5,
        info_utility=0.5,
    )


def _stub_openai(mock_client_mgr: MagicMock) -> MagicMock:
    """Wire OpenAIClientManager to a client whose parse() returns a PredictionResult."""
    mock_client = MagicMock()
    parsed_msg = MagicMock(parsed=_make_prediction_stub(), refusal=None)
    mock_client.beta.chat.completions.parse.return_value = MagicMock(
        choices=[MagicMock(message=parsed_msg)],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )
    mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)
    return mock_client


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_creates_and_closes_client(self) -> None:
        """__enter__ returns a fresh OpenAI client, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(f"{SFC_MODULE}.OpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            with mgr as client:
                assert client is mock_instance
                mock_openai.assert_called_once_with(api_key="sk-test")

            mock_instance.close.assert_called_once()


class TestParseCompletionRequiresClientParam:
    """Sanity: structured-output helper takes client as first positional."""

    def test_first_param_is_client(self) -> None:
        """_parse_completion's first param is `client`."""
        params = list(inspect.signature(_parse_completion).parameters)
        assert params[0] == "client"


class TestSourceContentCaptureReplay:
    """Verify capture and replay of source_content with pages."""

    @patch(f"{SFC_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SFC_MODULE}.OpenAIClientManager")
    @patch(f"{SFC_MODULE}.fetch_additional_sources")
    def test_live_capture_includes_serper_and_pages(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
    ) -> None:
        """Live run captures Serper response AND scraped page texts."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        _stub_openai(mock_client_mgr)

        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
        )

        captured = result[4]["source_content"]
        assert captured["mode"] == "cleaned"
        assert captured["serper_response"] == FAKE_SERPER_RESPONSE
        assert captured["pages"] == {
            "http://example.com/result": FAKE_PAGE_CONTENT,
            "http://example.com/second": "Second page body.",
        }
        # Scraped page text reaches the prediction prompt under "Content:"
        prediction_prompt = result[1]
        assert FAKE_PAGE_CONTENT in prediction_prompt
        assert "**Content:**" in prediction_prompt

    @patch(f"{SFC_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SFC_MODULE}.OpenAIClientManager")
    @patch(f"{SFC_MODULE}.fetch_additional_sources")
    def test_live_scrape_failure_falls_back_to_snippet(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        mock_page_fetch: MagicMock,
    ) -> None:
        """Scrape returning (None, None) for every URL is non-fatal."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        mock_page_fetch.side_effect = lambda *a, **kw: (None, None)
        _stub_openai(mock_client_mgr)

        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
        )

        captured = result[4]["source_content"]
        assert captured["pages"] == {}
        prediction_prompt = result[1]
        assert "Test snippet content" in prediction_prompt
        assert "**Content:**" not in prediction_prompt

    @patch(f"{SFC_MODULE}.OpenAIClientManager")
    def test_replay_with_pages_hydrates_content_into_prompt(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Replay format with `pages` injects cached content into the prompt."""
        _stub_openai(mock_client_mgr)

        source_content = {
            "mode": "cleaned",
            "serper_response": FAKE_SERPER_RESPONSE,
            "pages": {
                "http://example.com/result": "Cached cleaned article text.",
            },
        }
        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        prediction_prompt = result[1]
        assert "Cached cleaned article text." in prediction_prompt
        assert "Test snippet content" in prediction_prompt

    @patch(f"{SFC_MODULE}.OpenAIClientManager")
    def test_replay_raw_mode_runs_clean_html_on_cached_html(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """mode='raw' replay re-extracts cleaned text from cached HTML."""
        _stub_openai(mock_client_mgr)

        source_content = {
            "mode": "raw",
            "serper_response": FAKE_SERPER_RESPONSE,
            "pages": {"http://example.com/result": _HTML_PAGE},
        }
        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        prediction_prompt = result[1]
        assert "Federal Reserve" in prediction_prompt
        assert "<html>" not in prediction_prompt

    @patch(f"{SFC_MODULE}.OpenAIClientManager")
    def test_replay_legacy_format_without_pages_still_works(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Replay against captures from before evidence-gathering works cleanly."""
        _stub_openai(mock_client_mgr)

        source_content = {"serper_response": FAKE_SERPER_RESPONSE}
        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        prediction_prompt = result[1]
        assert "Test Result" in prediction_prompt
        assert "Test snippet content" in prediction_prompt
        assert "What is test?" in prediction_prompt
        assert "**Content:**" not in prediction_prompt

    @patch(f"{SFC_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SFC_MODULE}.OpenAIClientManager")
    @patch(f"{SFC_MODULE}.fetch_additional_sources")
    def test_flag_off_no_source_content(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
    ) -> None:
        """When return_source_content is false, source_content is not in used_params."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        _stub_openai(mock_client_mgr)

        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        used_params = result[4]
        assert "source_content" not in used_params

    @patch(f"{SFC_MODULE}.OpenAIClientManager")
    def test_result_serialises_only_mech_protocol_fields(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """On-chain result string includes only p_yes/p_no/confidence/info_utility."""
        _stub_openai(mock_client_mgr)

        source_content = {"serper_response": FAKE_SERPER_RESPONSE}
        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content=source_content,
        )

        parsed = json.loads(result[0])
        assert set(parsed.keys()) == {"p_yes", "p_no", "confidence", "info_utility"}
        assert parsed["p_yes"] == 0.5
        assert parsed["p_no"] == 0.5


class TestScrapePages:
    """Unit-level coverage for the scrape helper that runs in-process."""

    @patch(f"{SFC_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    def test_scrape_pages_attaches_content_and_captures(
        self, _mock_page_fetch: MagicMock
    ) -> None:
        """Successful scrapes mutate items and return the capture dict."""
        organic = [
            {"link": "http://example.com/result", "title": "T1", "snippet": "s1"},
            {"link": "http://example.com/second", "title": "T2", "snippet": "s2"},
        ]
        captured = _scrape_pages(organic, mode="cleaned")
        assert organic[0]["content"] == FAKE_PAGE_CONTENT
        assert organic[1]["content"] == "Second page body."
        assert captured == {
            "http://example.com/result": FAKE_PAGE_CONTENT,
            "http://example.com/second": "Second page body.",
        }

    @patch(
        f"{SFC_MODULE}._fetch_page_content",
        side_effect=lambda *a, **kw: (None, None),
    )
    def test_scrape_pages_failure_returns_empty(
        self, _mock_page_fetch: MagicMock
    ) -> None:
        """When every fetch fails, items are untouched and capture is empty."""
        organic = [{"link": "http://example.com/x", "title": "T", "snippet": "s"}]
        captured = _scrape_pages(organic, mode="cleaned")
        assert "content" not in organic[0]
        assert captured == {}


class TestEvidenceBlockCap:
    """The cap drops trailing organic items until the rendered block fits."""

    def test_small_evidence_unchanged(self) -> None:
        """Below-budget evidence is returned without truncation marker."""
        organic = [
            {"title": "T", "link": "http://x", "snippet": "s", "position": 1},
        ]
        rendered = _cap_evidence_block(organic, [], model="gpt-4.1")
        assert "[… evidence truncated …]" not in rendered
        assert "T" in rendered

    def test_oversize_evidence_is_trimmed_with_marker(self) -> None:
        """When over budget, trailing items are dropped and a marker is appended."""
        # Build an oversized evidence block: 5 items, each with a giant content
        # payload so the cap MUST drop some.
        huge = "lorem ipsum " * 800  # ~1600 tokens per item
        organic = [
            {
                "title": f"T{i}",
                "link": f"http://x/{i}",
                "snippet": f"s{i}",
                "position": i + 1,
                "content": huge,
            }
            for i in range(5)
        ]
        rendered = _cap_evidence_block(organic, [], model="gpt-4.1")
        assert "[… evidence truncated …]" in rendered
        assert count_tokens(rendered, "gpt-4.1") <= MAX_EVIDENCE_TOKENS + 100
        # Trailing items are dropped, leading (most-relevant) kept: a
        # leading-drop mutation would keep T4 and drop T0, failing this.
        assert "T0" in rendered
        assert "T4" not in rendered

    def test_paa_only_overflow_returns_without_loop(self) -> None:
        """With no organic items the cap returns as-is (no marker, no infinite loop)."""
        huge_paa = [
            {"question": "lorem ipsum " * 800, "link": "http://x", "snippet": "s"}
        ]
        rendered = _cap_evidence_block([], huge_paa, model="gpt-4.1")
        assert "[… evidence truncated …]" not in rendered
        assert "lorem ipsum" in rendered


class TestPredictionResultValidator:
    """The p_yes + p_no ≈ 1 model validator rejects inconsistent pairs."""

    def test_mismatched_p_yes_p_no_raises(self) -> None:
        """|p_yes + p_no - 1| > 0.01 raises a validation error."""
        with pytest.raises(ValidationError, match="p_yes \\+ p_no must equal 1"):
            PredictionResult(
                facts="f",
                reasons_no="n",
                reasons_yes="y",
                aggregation="a",
                reflection="r",
                p_yes=0.3,
                p_no=0.3,  # sums to 0.6, not 1
                confidence=0.5,
                info_utility=0.5,
            )

    def test_valid_pair_accepted(self) -> None:
        """A consistent p_yes + p_no = 1 pair constructs fine."""
        result = PredictionResult(
            facts="f",
            reasons_no="n",
            reasons_yes="y",
            aggregation="a",
            reflection="r",
            p_yes=0.7,
            p_no=0.3,
            confidence=0.5,
            info_utility=0.5,
        )
        assert result.p_yes == 0.7


class TestFetchPageContent:
    """Direct coverage of _fetch_page_content's four early-return paths."""

    @staticmethod
    def _resp(
        status: int = 200,
        content_type: str = "text/html; charset=utf-8",
        text: str = "",
    ) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.headers = {"Content-Type": content_type}
        resp.text = text
        return resp

    @patch(f"{SFC_MODULE}.requests.get")
    def test_happy_path_cleaned(self, mock_get: MagicMock) -> None:
        """200 + HTML → (cleaned_text, cleaned_text) in cleaned mode."""
        mock_get.return_value = self._resp(text=_HTML_PAGE)
        text, capture = _fetch_page_content("http://x", mode="cleaned")
        assert text is not None and "Federal Reserve" in text
        assert capture == text

    @patch(f"{SFC_MODULE}.requests.get")
    def test_happy_path_raw_stores_html(self, mock_get: MagicMock) -> None:
        """200 + HTML → capture is the raw HTML in raw mode."""
        mock_get.return_value = self._resp(text=_HTML_PAGE)
        text, capture = _fetch_page_content("http://x", mode="raw")
        assert text is not None and "Federal Reserve" in text
        assert capture == _HTML_PAGE

    @patch(f"{SFC_MODULE}.requests.get")
    def test_non_200_returns_none(self, mock_get: MagicMock) -> None:
        """A 404 yields (None, None)."""
        mock_get.return_value = self._resp(status=404, text=_HTML_PAGE)
        assert _fetch_page_content("http://x") == (None, None)

    @patch(f"{SFC_MODULE}.requests.get")
    def test_non_html_content_type_returns_none(self, mock_get: MagicMock) -> None:
        """A non-HTML content-type (JSON) yields (None, None)."""
        mock_get.return_value = self._resp(
            content_type="application/json", text='{"a": 1}'
        )
        assert _fetch_page_content("http://x") == (None, None)

    @patch(f"{SFC_MODULE}.requests.get")
    def test_request_exception_returns_none(self, mock_get: MagicMock) -> None:
        """A network exception is swallowed → (None, None)."""
        mock_get.side_effect = requests.Timeout("slow")
        assert _fetch_page_content("http://x") == (None, None)

    @patch(f"{SFC_MODULE}._clean_html", return_value=None)
    @patch(f"{SFC_MODULE}.requests.get")
    def test_unextractable_html_returns_none(
        self, mock_get: MagicMock, _mock_clean: MagicMock
    ) -> None:
        """200 + HTML but readability extracts nothing → (None, None)."""
        mock_get.return_value = self._resp(text="<html></html>")
        assert _fetch_page_content("http://x") == (None, None)


def _rate_limit_error() -> openai.RateLimitError:
    """Build a RateLimitError with a stub response (no network, no httpx dep)."""
    return openai.RateLimitError(
        "rate limited", response=MagicMock(status_code=429), body=None
    )


class TestKeyRotationAndErrors:
    """Rate-limit propagation + parseable error-JSON catch-all."""

    def test_parse_completion_propagates_rate_limit_error(self) -> None:
        """Rate-limit errors are NOT caught in _parse_completion (decorator rotates)."""
        client = MagicMock()
        client.beta.chat.completions.parse.side_effect = _rate_limit_error()

        with pytest.raises(openai.RateLimitError):
            _parse_completion(
                client=client,
                model="gpt-4o",
                messages=[],
                response_format=PredictionResult,
            )

    def test_parse_completion_retries_value_error_then_raises_runtime(self) -> None:
        """Non-rate-limit transient errors retry in place, then RuntimeError."""
        client = MagicMock()
        client.beta.chat.completions.parse.side_effect = ValueError("transient")

        with pytest.raises(RuntimeError, match="Failed to get structured LLM"):
            _parse_completion(
                client=client,
                model="gpt-4o",
                messages=[],
                response_format=PredictionResult,
                retries=2,
                delay=0,
            )
        assert client.beta.chat.completions.parse.call_count == 2

    @patch(f"{SFC_MODULE}.OpenAIClientManager")
    @patch(f"{SFC_MODULE}.fetch_additional_sources")
    def test_unexpected_error_returns_parseable_error_json(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """An unexpected exception yields {p_yes:None,...,error:...}, not a raw string."""
        _stub_openai(mock_client_mgr)
        mock_fetch.side_effect = RuntimeError("boom")

        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        payload = json.loads(result[0])  # must be valid JSON, not a bare str
        assert payload["p_yes"] is None
        assert payload["p_no"] is None
        assert payload["error"] == "boom"

    @patch(f"{SFC_MODULE}.requests.request")
    def test_fetch_additional_sources_passes_timeout(
        self, mock_request: MagicMock
    ) -> None:
        """The Serper request forwards timeout=30 (fleet standard)."""
        fetch_additional_sources("question?", "serper-key")
        _, kwargs = mock_request.call_args
        assert kwargs["timeout"] == 30

    @patch(f"{SFC_MODULE}.OpenAIClientManager")
    @patch(f"{SFC_MODULE}.fetch_additional_sources")
    def test_serper_http_error_surfaces_as_error_json(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A 4xx/5xx Serper response raises via raise_for_status → error JSON."""
        _stub_openai(mock_client_mgr)
        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many")
        mock_fetch.return_value = bad_response

        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        bad_response.raise_for_status.assert_called_once()
        bad_response.json.assert_not_called()  # never reached on HTTP error
        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert "429" in payload["error"]

    def test_invalid_source_content_mode_returns_error_json(self) -> None:
        """A bad source_content_mode yields error JSON (not a silent string)."""
        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(source_content_mode="bogus"),
            counter_callback=None,
        )

        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert "Invalid source_content_mode" in payload["error"]

    def test_max_cost_returns_float_not_wrapped_tuple(self) -> None:
        """delivery_rate=0 returns the float max_cost untouched (float guard)."""
        result = run(
            tool="superforcaster_calibrated_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=lambda **_: 0.0123,
            delivery_rate=0,
        )
        assert result == 0.0123
