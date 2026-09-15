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

"""Unit tests for superforcaster_full_search_olas_predict_r1_14b: page scrape, capture/replay, fallbacks."""

import inspect
import json
from pathlib import Path
from typing import Any, Optional, Tuple
from unittest.mock import MagicMock, patch

import openai
import pytest
import requests
import yaml

import packages.valory.customs.superforcaster_full_search_olas_predict_r1_14b.superforcaster_full_search_olas_predict_r1_14b as module
from packages.valory.customs.superforcaster_full_search_olas_predict_r1_14b.superforcaster_full_search_olas_predict_r1_14b import (
    OpenAIClientManager,
    OpenAIResponse,
    Usage,
    canonical_prediction,
    fetch_additional_sources,
    generate_prediction_with_retry,
    parse_prompt,
    run,
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh OpenAIClient, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test", base_url="http://vllm:8000/v1")
        with patch(
            "packages.valory.customs.superforcaster_full_search_olas_predict_r1_14b.superforcaster_full_search_olas_predict_r1_14b.OpenAIClient"
        ) as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance

            with mgr as client:
                assert client is mock_instance
                # the base_url must reach the client: without it the OpenAI SDK
                # would silently talk to api.openai.com instead of the vLLM
                # server, and the served model name would not resolve there
                MockClient.assert_called_once_with(
                    api_key="sk-test",
                    base_url="http://vllm:8000/v1",
                )

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


SF_MODULE = "packages.valory.customs.superforcaster_full_search_olas_predict_r1_14b.superforcaster_full_search_olas_predict_r1_14b"

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

# (cleaned_text, capture_payload) tuples -- matches _fetch_page_content's return
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


PREDICTION_JSON = json.dumps(
    {"p_yes": 0.5, "p_no": 0.5, "confidence": 0.5, "info_utility": 0.5}
)

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
        # the forecasting endpoint is authenticated; run() requires the key
        "vllm_server_api_key": ["r1-14b-test-key"],
        # the vLLM endpoint URL is passed via the KeyChain
        "vllm_server_url": ["https://vllm.example/v1"],
        "return_source_content": [return_source_content],
        "source_content_mode": [source_content_mode],
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key][0]
    mock.get = lambda key, default="": services.get(key, [default])[0]
    return mock


def _stub_openai(mock_client_mgr: MagicMock) -> MagicMock:
    """Wire OpenAIClientManager to a stub returning PREDICTION_JSON."""
    # The non-calibrated path calls the OpenAIClient.completions(...) wrapper
    # (not chat.completions.create directly), so the stub must set
    # completions.return_value to a real OpenAIResponse -- otherwise result[0]
    # is an auto-MagicMock and JSON-shape assertions are vacuous.
    mock_client = MagicMock()
    mock_client.completions.return_value = OpenAIResponse(
        content=PREDICTION_JSON,
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )
    mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)
    return mock_client


class TestSuperforcasterSourceContent:
    """Verify superforcaster_full_search_olas_predict_r1_14b captures and replays source_content correctly."""

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
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
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
        )

        captured = result[4]["source_content"]
        assert captured["mode"] == "cleaned"
        assert captured["serper_response"] == FAKE_SERPER_RESPONSE
        # Both organic URLs were scraped -- both in pages capture
        assert captured["pages"] == {
            "http://example.com/result": FAKE_PAGE_CONTENT,
            "http://example.com/second": "Second page body.",
        }
        # Scraped page text reaches the prediction prompt under "Content:"
        prediction_prompt = result[1]
        assert FAKE_PAGE_CONTENT in prediction_prompt
        assert "**Content:**" in prediction_prompt
        # result[0] is the LLM completion content (via the OpenAIClient
        # wrapper), not an auto-MagicMock -- so this JSON assertion is real.
        assert json.loads(result[0]) == json.loads(PREDICTION_JSON)

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
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
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
        )

        captured = result[4]["source_content"]
        assert captured["pages"] == {}
        prediction_prompt = result[1]
        # Still has Serper-tier evidence; no Content line was rendered
        assert "Test snippet content" in prediction_prompt
        assert "**Content:**" not in prediction_prompt

    @patch(f"{SF_MODULE}.OpenAIClientManager")
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
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        prediction_prompt = result[1]
        assert "Cached cleaned article text." in prediction_prompt
        assert "Test snippet content" in prediction_prompt  # snippet preserved

    @patch(f"{SF_MODULE}.OpenAIClientManager")
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
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        prediction_prompt = result[1]
        # raw HTML was run back through _clean_html -- extracted article text
        assert "Federal Reserve" in prediction_prompt
        assert "<html>" not in prediction_prompt  # raw markup not dumped verbatim

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_replay_legacy_format_without_pages_still_works(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Captures produced before evidence-gathering replay cleanly."""
        _stub_openai(mock_client_mgr)

        # Old format: no `pages` key, no `mode` key.
        source_content = {"serper_response": FAKE_SERPER_RESPONSE}
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
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
        # No Content line because there were no cached pages
        assert "**Content:**" not in prediction_prompt

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
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
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        used_params = result[4]
        assert "source_content" not in used_params


class TestScrapePages:
    """Unit-level coverage for the scrape helper that runs in-process."""

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    def test_scrape_pages_attaches_content_and_captures(
        self, _mock_page_fetch: MagicMock
    ) -> None:
        """Successful scrapes mutate items and return the capture dict."""
        from packages.valory.customs.superforcaster_full_search_olas_predict_r1_14b.superforcaster_full_search_olas_predict_r1_14b import (
            _scrape_pages,
        )

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
        f"{SF_MODULE}._fetch_page_content",
        side_effect=lambda *a, **kw: (None, None),
    )
    def test_scrape_pages_failure_returns_empty(
        self, _mock_page_fetch: MagicMock
    ) -> None:
        """When every fetch fails, items are untouched and capture is empty."""
        from packages.valory.customs.superforcaster_full_search_olas_predict_r1_14b.superforcaster_full_search_olas_predict_r1_14b import (
            _scrape_pages,
        )

        organic = [{"link": "http://example.com/x", "title": "T", "snippet": "s"}]
        captured = _scrape_pages(organic, mode="cleaned")
        assert "content" not in organic[0]
        assert captured == {}

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    def test_scrape_pages_mixed_success(self, _mock_page_fetch: MagicMock) -> None:
        """One URL succeeds, one fails: only the success gets content + capture."""
        from packages.valory.customs.superforcaster_full_search_olas_predict_r1_14b.superforcaster_full_search_olas_predict_r1_14b import (
            _scrape_pages,
        )

        organic = [
            {"link": "http://example.com/result", "title": "T1", "snippet": "s1"},
            {"link": "http://example.com/unknown", "title": "T2", "snippet": "s2"},
        ]
        captured = _scrape_pages(organic, mode="cleaned")
        # success -- content attached + in capture; failure -- neither (exercises
        # the `if text:` / `if capture:` guards).
        assert organic[0]["content"] == FAKE_PAGE_CONTENT
        assert "content" not in organic[1]
        assert captured == {"http://example.com/result": FAKE_PAGE_CONTENT}


class TestEvidenceBlockCap:
    """The cap drops trailing organic items until the rendered block fits."""

    def test_small_evidence_unchanged(self) -> None:
        """Below-budget evidence is returned without truncation marker."""
        from packages.valory.customs.superforcaster_full_search_olas_predict_r1_14b.superforcaster_full_search_olas_predict_r1_14b import (
            _cap_evidence_block,
        )

        organic = [
            {"title": "T", "link": "http://x", "snippet": "s", "position": 1},
        ]
        rendered = _cap_evidence_block(organic, [], model="gpt-4.1").rendered
        assert "[… evidence truncated …]" not in rendered
        assert "T" in rendered

    def test_oversize_evidence_is_trimmed_with_marker(self) -> None:
        """When over budget, trailing items are dropped and a marker is appended."""
        from packages.valory.customs.superforcaster_full_search_olas_predict_r1_14b.superforcaster_full_search_olas_predict_r1_14b import (
            MAX_EVIDENCE_TOKENS,
            _cap_evidence_block,
            count_tokens,
        )

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
        rendered = _cap_evidence_block(organic, [], model="gpt-4.1").rendered
        assert "[… evidence truncated …]" in rendered
        assert count_tokens(rendered, "gpt-4.1") <= MAX_EVIDENCE_TOKENS + 100
        # Trailing items are dropped, leading (most-relevant) kept: a
        # leading-drop mutation would keep T4 and drop T0, failing this.
        assert "T0" in rendered
        assert "T4" not in rendered

    def test_paa_only_overflow_is_trimmed_not_returned_whole(self) -> None:
        """A peopleAlsoAsk-only response must still be trimmed."""
        # This previously asserted the opposite and pinned a real hole: the
        # parent's `or not organic_data` early return was correct there because
        # it never trimmed peopleAlsoAsk, but this file trims it FIRST, so the
        # clause skipped BOTH loops. Measured before the fix: 21304 tokens
        # returned against a 3000 budget, on an 8192 window.
        # There is no infinite loop -- `while misc and ...: misc.pop()`
        # terminates when misc empties.
        page = " ".join(["token"] * module._MAX_PAGE_WORDS)
        huge_paa = [{"question": f"q{i}?", "snippet": page} for i in range(40)]
        capped = module._cap_evidence_block([], huge_paa, "olas-predict-r1-14b", 3000)
        assert module.budget_tokens(capped.rendered, "olas-predict-r1-14b") <= 3000
        assert capped.misc_kept < len(huge_paa), "peopleAlsoAsk was not trimmed"
        assert capped.organic_kept == 0


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

    @patch(f"{SF_MODULE}.requests.get")
    def test_happy_path_cleaned(self, mock_get: MagicMock) -> None:
        """200 + HTML returns (cleaned_text, cleaned_text) in cleaned mode."""
        mock_get.return_value = self._resp(text=_HTML_PAGE)
        text, capture = module._fetch_page_content("http://x", mode="cleaned")
        assert text is not None and "Federal Reserve" in text
        assert capture == text  # cleaned mode stores the cleaned text

    @patch(f"{SF_MODULE}.requests.get")
    def test_happy_path_raw_stores_html(self, mock_get: MagicMock) -> None:
        """200 + HTML returns raw HTML as the capture in raw mode."""
        mock_get.return_value = self._resp(text=_HTML_PAGE)
        text, capture = module._fetch_page_content("http://x", mode="raw")
        assert text is not None and "Federal Reserve" in text
        assert capture == _HTML_PAGE  # raw mode stores the raw html

    @patch(f"{SF_MODULE}.requests.get")
    def test_non_200_returns_none(self, mock_get: MagicMock) -> None:
        """A 404 yields (None, None)."""
        mock_get.return_value = self._resp(status=404, text=_HTML_PAGE)
        assert module._fetch_page_content("http://x") == (None, None)

    @patch(f"{SF_MODULE}.requests.get")
    def test_non_html_content_type_returns_none(self, mock_get: MagicMock) -> None:
        """A non-HTML content-type (JSON) yields (None, None)."""
        mock_get.return_value = self._resp(
            content_type="application/json", text='{"a": 1}'
        )
        assert module._fetch_page_content("http://x") == (None, None)

    @patch(f"{SF_MODULE}.requests.get")
    def test_request_exception_returns_none(self, mock_get: MagicMock) -> None:
        """A network exception is swallowed and returns (None, None)."""
        mock_get.side_effect = requests.Timeout("slow")
        assert module._fetch_page_content("http://x") == (None, None)

    @patch(f"{SF_MODULE}._clean_html", return_value=None)
    @patch(f"{SF_MODULE}.requests.get")
    def test_unextractable_html_returns_none(
        self, mock_get: MagicMock, _mock_clean: MagicMock
    ) -> None:
        """Unextractable HTML returns (None, None)."""
        mock_get.return_value = self._resp(text="<html></html>")
        assert module._fetch_page_content("http://x") == (None, None)


class TestErrorHandling:
    """with_key_rotation's catch-all returns parseable null-prediction JSON."""

    @patch(f"{SF_MODULE}.time.sleep", return_value=None)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_rate_limit_is_retried_at_the_completion(
        self, mock_client_mgr: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """A vLLM 429 retries the completion, not the whole pipeline."""
        # Re-raising to `with_key_rotation` would re-run search and page
        # scraping for a single-key endpoint that has nothing to rotate to, and
        # then propagate the 429 anyway. The parent retries the completion.
        mock_client = _stub_openai(mock_client_mgr)
        rate_limit = openai.RateLimitError(
            "429 Too Many Requests",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        mock_client.completions.side_effect = [
            rate_limit,
            OpenAIResponse(content=PREDICTION_JSON, usage=Usage()),
        ]
        keys = _make_mock_api_keys("false")
        keys.max_retries = lambda: {"vllm_server_api_key": 1}

        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            prompt=PREDICTION_PROMPT,
            api_keys=keys,
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )

        assert json.loads(result[0]) == json.loads(PREDICTION_JSON)
        assert mock_client.completions.call_count == 2
        keys.rotate.assert_not_called()

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_unexpected_error_returns_parseable_error_json(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """An unexpected exception yields {p_yes:None,...,error:...}, not a raw string."""
        _stub_openai(mock_client_mgr)
        mock_fetch.side_effect = RuntimeError("boom")

        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        payload = json.loads(result[0])  # must be valid JSON, not a bare str
        assert payload["p_yes"] is None
        assert payload["p_no"] is None
        assert payload["error"] == "boom"

    @patch(f"{SF_MODULE}.time.sleep", return_value=None)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_null_content_surfaces_as_error_json(
        self, mock_client_mgr: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """An LLM refusal (content=None) becomes error JSON, not a None prediction."""
        mock_client = _stub_openai(mock_client_mgr)
        mock_client.completions.return_value = OpenAIResponse(content=None)

        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )

        payload = json.loads(result[0])  # not None -- no downstream json.loads(None)
        assert payload["p_yes"] is None
        assert "content" in payload["error"].lower()


class TestSerperRequest:
    """Serper call carries a timeout and surfaces HTTP errors."""

    @patch(f"{SF_MODULE}.requests.request")
    def test_fetch_additional_sources_passes_timeout(
        self, mock_request: MagicMock
    ) -> None:
        """The Serper request forwards timeout=30 (fleet standard)."""
        fetch_additional_sources("question?", "serper-key")
        _, kwargs = mock_request.call_args
        assert kwargs["timeout"] == 30

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_serper_http_error_surfaces_as_error_json(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A 4xx/5xx Serper response becomes an error JSON result."""
        _stub_openai(mock_client_mgr)
        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many")
        mock_fetch.return_value = bad_response

        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
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


class TestSourceContentModeValidation:
    """An invalid source_content_mode surfaces as a recognisable error."""

    def test_invalid_mode_returns_error_json(self) -> None:
        """A bad mode yields error JSON (not a silent string) via the catch-all."""
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(source_content_mode="bogus"),
            counter_callback=None,
        )

        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert "Invalid source_content_mode" in payload["error"]


class TestMaxCostPath:
    """delivery_rate=0 returns the float max_cost untouched (float guard)."""

    def test_max_cost_returns_float_not_wrapped_tuple(self) -> None:
        """Without the isinstance(result, float) guard this raises TypeError."""
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=lambda **_: 0.0123,
            delivery_rate=0,
        )
        assert result == 0.0123


# Free-text format prompt with instruction boilerplate: the advertised input
# contract (issue #455) that the old extract_question collapsed into a
# prompt-shaped Serper query.
LONG_FREE_TEXT_PROMPT = (
    "Please predict the following market: Will Alexander Isak permanently "
    "transfer to Liverpool FC before the end of the summer 2025 transfer "
    "window (September 2, 2025 23:59 UTC)? Resolution source: official club "
    "announcements or BBC Sport. The market resolves YES if a permanent "
    "transfer (not a loan) is confirmed by the resolution source before the "
    "deadline."
)

EMPTY_SERPER_RESPONSE: dict = {"organic": [], "peopleAlsoAsk": []}


class TestIssue455ParsePrompt:
    """parse_prompt() -> (question_for_llm, search_query, tier)."""

    def test_trader_template_uses_extracted_question_for_both(self) -> None:
        """Trader-template parity: the bare question serves as both values."""
        question, query, tier = parse_prompt(PREDICTION_PROMPT)
        assert tier == "template"
        assert question == "Will X happen?"
        assert query == question

    def test_free_text_llm_gets_full_prompt(self) -> None:
        """Free-text input: the LLM question is the whole prompt."""
        question, _, tier = parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert tier == "clause"
        assert question == LONG_FREE_TEXT_PROMPT

    def test_boilerplate_prefix_is_dropped_from_query(self) -> None:
        """The query anchors at the market question, dropping instruction text."""
        _, query, _ = parse_prompt(LONG_FREE_TEXT_PROMPT)
        assert query.startswith("Will Alexander Isak")
        assert query.endswith("?")
        assert len(query) <= module._MAX_SEARCH_QUERY_LEN


class TestIssue455EmptyRetrievalGuard:
    """Degenerate prompts and zero-hit retrieval return the flagged null."""

    @pytest.mark.parametrize("degenerate", ["", "   ", "???", '"""'])
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_degenerate_prompt_short_circuits_before_serper(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock, degenerate: str
    ) -> None:
        """Prompts with no searchable content never reach Serper at all."""
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=degenerate,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )
        mock_fetch.assert_not_called()
        assert json.loads(result[0])["p_yes"] == 0.5
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "empty query"
        assert result[4]["scan_truncated"] is False

    @patch(f"{SF_MODULE}._fetch_page_content")
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_zero_hit_live_returns_flagged_null_before_scrape(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        mock_page_fetch: MagicMock,
    ) -> None:
        """Both-empty live retrieval -> flagged null; no page is scraped."""
        mock_response = MagicMock()
        mock_response.json.return_value = EMPTY_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )
        mock_page_fetch.assert_not_called()
        assert json.loads(result[0])["p_yes"] == 0.5
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "live search"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_empty_cached_replay_returns_flagged_null(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Both-empty cached retrieval -> flagged null with the replay reason."""
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": EMPTY_SERPER_RESPONSE},
        )
        assert json.loads(result[0])["p_yes"] == 0.5
        assert result[4]["empty_retrieval"] is True
        assert result[4]["null_reason"] == "cached replay"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_malformed_serper_body_is_error_not_flagged_null(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """organic: null is a broken integration -> typed error, not a null forecast."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic": None, "peopleAlsoAsk": []}
        mock_fetch.return_value = mock_response
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )
        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert "organic" in payload["error"]


class TestIssue455RunWiring:
    """run() feeds parsed.query to Serper and parsed.question to the LLM."""

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_trader_request_sends_extracted_question_to_serper(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
    ) -> None:
        """Template path parity: Serper gets the bare question (old behavior)."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        _stub_openai(mock_client_mgr)
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )
        mock_fetch.assert_called_once_with("Will X happen?", "serper-test")
        assert result[4]["parse_tier"] == "template"
        assert result[4]["scan_truncated"] is False

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_free_text_serper_gets_query_llm_gets_full_prompt(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
    ) -> None:
        """Free-text path: short query to Serper, whole prompt to the LLM."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        _stub_openai(mock_client_mgr)
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )
        sent_query = mock_fetch.call_args[0][0]
        assert sent_query.startswith("Will Alexander Isak")
        assert len(sent_query) <= module._MAX_SEARCH_QUERY_LEN
        # criteria text that the derived query drops must still reach the LLM
        assert "official club announcements or BBC Sport" in result[1]
        assert result[4]["parse_tier"] == "clause"

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_raw_tier_past_window_is_marked_truncated(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
    ) -> None:
        """Question-free prompt past the window: raw tier AND truncated."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        _stub_openai(mock_client_mgr)
        prompt = "no question words at all here. " * (module._MAX_SCAN_CHARS // 10)
        assert len(prompt) > module._MAX_SCAN_CHARS
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=prompt,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )
        assert result[4]["parse_tier"] == "raw"
        assert result[4]["scan_truncated"] is True

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_clause_tier_past_window_is_marked_truncated(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
    ) -> None:
        """A clause-tier pick on a longer-than-window prompt is still marked."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        _stub_openai(mock_client_mgr)
        prompt = "Will the ECB cut rates at its next meeting? " + "filler " * (
            module._MAX_SCAN_CHARS // 3
        )
        assert len(prompt) > module._MAX_SCAN_CHARS
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=prompt,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )
        assert result[4]["parse_tier"] == "clause"
        assert result[4]["scan_truncated"] is True

    @pytest.mark.parametrize(
        "body",
        [
            {"peopleAlsoAsk": []},
            {"organic": {"not": "a list"}, "peopleAlsoAsk": []},
            {"organic": "reshaped", "peopleAlsoAsk": []},
            {"organic": [{"title": "T"}], "peopleAlsoAsk": None},
            {"organic": [{"title": "T"}], "peopleAlsoAsk": "nope"},
        ],
    )
    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_malformed_serper_body_is_a_typed_error_null(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
        body: dict,
    ) -> None:
        """Malformed Serper body -> typed error null, whole field set pinned."""
        mock_response = MagicMock()
        mock_response.json.return_value = body
        mock_fetch.return_value = mock_response
        _stub_openai(mock_client_mgr)
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["p_no"] is None
        assert parsed["confidence"] == 0.0
        assert parsed["info_utility"] == 0.0
        assert parsed["error_type"] == "ValueError"

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_cached_replay_malformed_body_is_a_typed_error_null(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
    ) -> None:
        """Cached-replay branch applies the same shape check as the live one."""
        _stub_openai(mock_client_mgr)
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=LONG_FREE_TEXT_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": {"message": "quota exceeded"}},
        )
        mock_fetch.assert_not_called()
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["p_no"] is None
        assert parsed["confidence"] == 0.0
        assert parsed["info_utility"] == 0.0
        assert parsed["error_type"] == "ValueError"

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_long_template_prompt_is_not_marked_truncated(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
    ) -> None:
        """Template past the window is NOT flagged: question precedes the scan."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        _stub_openai(mock_client_mgr)
        prompt = PREDICTION_PROMPT + " filler" * (module._MAX_SCAN_CHARS // 3)
        assert len(prompt) > module._MAX_SCAN_CHARS
        result = run(
            tool="superforcaster_full_search_olas_predict_r1_14b_omen",
            model="gpt-4o",
            prompt=prompt,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )
        assert result[4]["parse_tier"] == "template"
        assert result[4]["scan_truncated"] is False


class TestOlasPredictWiring:
    """The deltas from the superforcaster_full_search parent."""

    def test_both_platform_wire_names_are_allowed(self) -> None:
        """One package serves an Omen and a Polymarket name; others are rejected."""
        assert module.ALLOWED_TOOLS == [
            "superforcaster_full_search_olas_predict_r1_14b_omen",
            "superforcaster_full_search_olas_predict_r1_14b_polymarket",
        ]

    def test_unknown_tool_name_is_rejected(self) -> None:
        """A wire name we do not serve must not silently run the model.

        `run` is wrapped by `with_key_rotation`, which converts exceptions into
        an error response rather than propagating them, so the rejection shows
        up as error JSON and not as a raised ValueError.
        """
        result = module.run(
            tool="superforcaster_full_search",
            prompt=PREDICTION_PROMPT,
            model="olas-predict-r1-14b",
            api_keys=_make_mock_api_keys(),
        )
        # result is (error_json, prompt, ..., api_keys); the KeyChain mock at
        # the tail is not JSON-serialisable, so assert on the payload itself
        assert "not supported" in result[0]

    def test_completion_budget_leaves_room_for_the_prompt(self) -> None:
        """max_tokens is below the fleet's 4096 because the window is 8k."""
        # #470 raised the fleet to 4096 so free-text completions are not cut off
        # before the JSON; the same reasoning applies to a think block. But the
        # fleet targets GPT-4.1, where the prompt is unconstrained. Here the
        # prompt and the completion share 8192, and at 4096 the evidence block
        # alone overruns what is left. Observed completions: 394-523 tokens.
        max_tokens = module.DEFAULT_MODEL_SETTINGS["max_tokens"]
        assert max_tokens == 2048
        assert max_tokens > 523 * 2, "must clear the longest observed completion"

    def test_worst_case_request_fits_the_context_window(self) -> None:
        """Full evidence + peopleAlsoAsk + a capped question must still fit."""
        page = " ".join(["token"] * module._MAX_PAGE_WORDS)
        organic = [
            {
                "title": f"t{i}",
                "link": f"https://e.com/{i}",
                "snippet": page,
                "date": "x",
            }
            for i in range(module.MAX_SOURCES)
        ]
        misc = [{"question": f"q{i}?", "snippet": page} for i in range(8)]
        question = module._truncate_to_tokens(
            " ".join(["word"] * 5000),
            module._MAX_QUESTION_TOKENS,
            "olas-predict-r1-14b",
        )
        max_tokens = module.DEFAULT_MODEL_SETTINGS["max_tokens"]
        budget = module._evidence_budget(
            question, "14/09/2026", "olas-predict-r1-14b", max_tokens
        )
        sources = module._cap_evidence_block(
            organic, misc, "olas-predict-r1-14b", budget
        ).rendered
        prompt = module.PREDICTION_PROMPT.format(
            question=question, today="14/09/2026", sources=sources
        )
        total = module.count_tokens(prompt, "olas-predict-r1-14b") + max_tokens
        assert (
            total <= module.MODEL_CONTEXT_WINDOW
        ), f"{total} tokens exceeds the {module.MODEL_CONTEXT_WINDOW} window"

    def test_long_free_text_question_is_capped(self) -> None:
        """Pearl sends the user's message verbatim, and it lands twice."""
        long_q = " ".join(["word"] * 5000)
        capped = module._truncate_to_tokens(
            long_q, module._MAX_QUESTION_TOKENS, "olas-predict-r1-14b"
        )
        assert (
            module.count_tokens(capped, "olas-predict-r1-14b")
            <= module._MAX_QUESTION_TOKENS
        )
        # a short question is returned untouched
        short = "Will it rain tomorrow?"
        assert (
            module._truncate_to_tokens(
                short, module._MAX_QUESTION_TOKENS, "olas-predict-r1-14b"
            )
            == short
        )

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_endpoint_comes_from_keychain(self, mock_client_mgr: MagicMock) -> None:
        """The vLLM base URL comes from the KeyChain (olas_predict_r1_14b_endpoint).

        The mech forwards the endpoint via the KeyChain, mirroring the
        finetuned_prediction pattern. The mock KeyChain has the endpoint
        service, so a successful run proves the URL reached the client from
        the KeyChain.

        :param mock_client_mgr: Mocked OpenAIClientManager.
        """
        _stub_openai(mock_client_mgr)
        result = run(
            tool=module.TOOL_OMEN,
            prompt=PREDICTION_PROMPT,
            model="olas-predict-r1-14b",
            api_keys=_make_mock_api_keys(),
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )
        assert json.loads(result[0])["p_yes"] == 0.5
        mock_client_mgr.assert_called_once_with(
            "r1-14b-test-key", "https://vllm.example/v1"
        )

    def test_missing_api_key_fails_loudly(self) -> None:
        """An authenticated endpoint rejects missing keys.

        A placeholder key would produce an opaque downstream 401 instead.
        """
        keys = MagicMock()
        services = {"serperapi": "serper-test"}
        keys.__getitem__ = lambda self, key: services[key]
        keys.get = lambda key, default="": services.get(key, default)
        keys.max_retries = lambda: {"openai": 1, "openrouter": 1}
        result = module.run(
            tool=module.TOOL_OMEN,
            prompt=PREDICTION_PROMPT,
            model="olas-predict-r1-14b",
            api_keys=keys,
        )
        assert "No API key for the forecasting endpoint" in result[0]

    def test_absent_endpoint_fails_loudly(self) -> None:
        """A missing endpoint is a config error, not a silent localhost fallback.

        The endpoint is a required deployment input; without it the tool must
        surface a clear error instead of guessing a host.
        """
        keys = MagicMock()
        services = {
            "openai": "sk-test",
            "serperapi": "serper-test",
            "vllm_server_api_key": "r1-14b-test-key",
            # deliberately omit vllm_server_url
            "return_source_content": "false",
            "source_content_mode": "cleaned",
        }
        keys.__getitem__ = lambda self, key: services[key]
        keys.get = lambda key, default="": services.get(key, default)
        keys.max_retries = lambda: {"openai": 1, "openrouter": 1}
        result = module.run(
            tool=module.TOOL_OMEN,
            prompt=PREDICTION_PROMPT,
            model="olas-predict-r1-14b",
            api_keys=keys,
        )
        assert "No endpoint for the forecasting service" in result[0]

    def test_base_url_reaches_the_openai_sdk(self) -> None:
        """Without base_url the SDK would silently talk to api.openai.com."""
        with patch(
            "packages.valory.customs.superforcaster_full_search_olas_predict_r1_14b."
            "superforcaster_full_search_olas_predict_r1_14b.openai.OpenAI"
        ) as mock_openai:
            module.OpenAIClient(api_key="sk-test", base_url="http://vllm:8000/v1")
        mock_openai.assert_called_once_with(
            api_key="sk-test", base_url="http://vllm:8000/v1"
        )

    def test_served_model_is_resolved_from_the_tool(self) -> None:
        """Both wire names resolve to the one checkpoint this endpoint serves."""
        assert module.resolve_model(module.TOOL_OMEN) == "olas-predict-r1-14b"
        assert module.resolve_model(module.TOOL_POLYMARKET) == "olas-predict-r1-14b"
        assert set(module.MODEL_BY_TOOL) == set(module.ALLOWED_TOOLS)
        # component.yaml's default_model must agree: the mech shows it in the
        # tool metadata, and a drift there misreports what is being served.
        component = yaml.safe_load(
            (Path(module.__file__).parent / "component.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert component["params"]["default_model"] == "olas-predict-r1-14b"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_requester_supplied_model_is_ignored(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """The requester-controlled `model` kwarg is not honoured."""
        # `task_data.get("model", params.default_model)` lets a request
        # override the component default, and the benchmark tournament passes
        # its own `--model` default. Either would reach a vLLM that serves one
        # checkpoint.
        mock_client = _stub_openai(mock_client_mgr)
        run(
            tool=module.TOOL_OMEN,
            prompt=PREDICTION_PROMPT,
            model="gpt-4.1-2025-04-14",
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )
        assert (
            mock_client.completions.call_args.kwargs["model"] == "olas-predict-r1-14b"
        )

    def test_model_kwarg_is_not_required(self) -> None:
        """Pearl and the tournament may omit `model` entirely."""
        with patch(f"{SF_MODULE}.OpenAIClientManager") as mock_client_mgr:
            _stub_openai(mock_client_mgr)
            result = run(
                tool=module.TOOL_POLYMARKET,
                prompt=PREDICTION_PROMPT,
                api_keys=_make_mock_api_keys("false"),
                counter_callback=None,
                source_content={"serper_response": FAKE_SERPER_RESPONSE},
            )
        assert json.loads(result[0]) == json.loads(PREDICTION_JSON)

    def test_reasoning_completion_is_normalized_to_delivery_json(self) -> None:
        """Reasoning prose and fenced JSON do not leak into the delivery."""
        completion = (
            "Reasoning about the evidence.\n</think>.\n```json\n"
            '{"p_yes": 0.2, "p_no": 0.8, "confidence": 0.7, '
            '"info_utility": 0.5}\n```'
        )
        assert json.loads(canonical_prediction(completion) or "{}") == {
            "p_yes": 0.2,
            "p_no": 0.8,
            "confidence": 0.7,
            "info_utility": 0.5,
        }

    def test_draft_before_think_close_loses_to_the_final_answer(self) -> None:
        """The exact regression: a draft p_yes inside the reasoning must lose."""
        # Pre-fix this delivered 0.9 -- the draft written while reasoning --
        # because the paired-tag strip did not match a BARE closer and the
        # first-match JSON pick took whatever came earliest.
        completion = (
            'Reasoning. A first pass would be {"p_yes": 0.9, "p_no": 0.1} but '
            "the evidence points lower, so I will revise.\n</think>\n"
            '{"p_yes": 0.3, "p_no": 0.7, "confidence": 0.6, "info_utility": 0.5}'
        )
        assert json.loads(canonical_prediction(completion) or "{}")["p_yes"] == 0.3

    def test_people_also_ask_is_trimmed_before_scraped_pages(self) -> None:
        """A large peopleAlsoAsk block must not evict the better evidence."""
        page = " ".join(["token"] * module._MAX_PAGE_WORDS)
        organic = [
            {
                "title": f"Page {i}",
                "link": f"https://e.com/{i}",
                "snippet": page,
                "date": "x",
            }
            for i in range(module.MAX_SOURCES)
        ]
        paa = [{"question": f"q{i}?", "snippet": page} for i in range(30)]
        # A budget that cannot hold both: the scraped pages must be what survives.
        rendered = module._cap_evidence_block(
            organic, paa, "olas-predict-r1-14b", 3000
        ).rendered
        assert rendered.count("**Title:**") == module.MAX_SOURCES, "pages evicted"
        assert "**Question:**" not in rendered, "peopleAlsoAsk should go first"

    def test_requester_max_tokens_cannot_starve_the_evidence(self) -> None:
        """`max_tokens` is requester-controlled, so it is clamped like `model`."""
        # Unclamped, a value near the window drives the evidence budget to zero
        # and the tool forecasts on an empty <background> while still returning
        # a normal-looking four-field answer.
        assert module.MODEL_CONTEXT_WINDOW - module.MIN_PROMPT_BUDGET < 8192
        clamped = min(8192, module.MODEL_CONTEXT_WINDOW - module.MIN_PROMPT_BUDGET)
        budget = module._evidence_budget(
            "Will it rain?", "14/09/2026", "olas-predict-r1-14b", clamped
        )
        assert budget > 0, "clamp must leave room for evidence"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_run_clamps_an_oversized_requester_max_tokens(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """The clamp must be exercised through run(), not recomputed here."""
        # Recomputing min(...) in the test would pass with the production clamp
        # deleted, changed to max, or off by one -- which is the regression it
        # exists to catch. Assert on what reaches the endpoint instead.
        mock_client = _stub_openai(mock_client_mgr)
        run(
            tool=module.TOOL_OMEN,
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            max_tokens=8000,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )
        sent = mock_client.completions.call_args.kwargs["max_tokens"]
        assert sent == module.MODEL_CONTEXT_WINDOW - module.MIN_PROMPT_BUDGET
        assert sent < 8000

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_run_rejects_a_null_or_nonpositive_max_tokens(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """An explicit null or 0 must not raise or reach the endpoint."""
        # `kwargs.get("max_tokens", DEFAULT)` returns None when the key is
        # present with a null value, and int(None) raises TypeError; 0 would be
        # rejected by the endpoint after three retries.
        for bad in (None, 0, -5):
            mock_client = _stub_openai(mock_client_mgr)
            result = run(
                tool=module.TOOL_OMEN,
                prompt=PREDICTION_PROMPT,
                api_keys=_make_mock_api_keys("false"),
                counter_callback=None,
                max_tokens=bad,
                source_content={"serper_response": FAKE_SERPER_RESPONSE},
            )
            assert "error_type" not in result[0], f"max_tokens={bad!r} errored"
            # All three fall back to the DEFAULT, not to 1: a 1-token budget is
            # a request that cannot produce a parseable answer, so clamping a
            # negative upward to 1 would trade a loud failure for a silent one.
            assert (
                mock_client.completions.call_args.kwargs["max_tokens"]
                == module.DEFAULT_MODEL_SETTINGS["max_tokens"]
            ), f"max_tokens={bad!r} did not fall back to the default"

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_run_flags_a_budget_starved_prompt_without_calling_the_model(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Evidence trimmed to nothing must surface, not reach the endpoint."""
        mock_client = _stub_openai(mock_client_mgr)
        page = " ".join(["token"] * module._MAX_PAGE_WORDS)
        bulky = {
            "serper_response": {
                "organic": [
                    {
                        "title": f"P{i}",
                        "link": f"https://e.com/{i}",
                        "snippet": page,
                        "date": "x",
                    }
                    for i in range(module.MAX_SOURCES)
                ],
                "peopleAlsoAsk": [],
            }
        }
        # A completion budget that leaves the prompt almost nothing.
        result = run(
            tool=module.TOOL_OMEN,
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            max_tokens=module.MODEL_CONTEXT_WINDOW - module.MIN_PROMPT_BUDGET,
            source_content=bulky,
        )
        used = result[4] or {}
        if used.get("null_reason") == "evidence budget exhausted":
            assert not mock_client.completions.called, "doomed prompt was sent"
            assert used["empty_retrieval"] is True
        else:
            # Budget held: then the counts must still be reported.
            assert "sources_used" in used and "sources_dropped" in used

    def test_nested_json_in_a_valid_forecast_is_not_discarded(self) -> None:
        """One extra nested key must not cost the whole forecast."""
        # A `[^{}]*` character class cannot match an object containing an
        # object, so this returned None and the delivery became a null -- which
        # on a dashboard is indistinguishable from a real model failure.
        completion = (
            '</think>\n{"p_yes": 0.8, "p_no": 0.2, "confidence": 0.9, '
            '"info_utility": 0.7, "meta": {"a": 1}}'
        )
        assert json.loads(canonical_prediction(completion) or "{}")["p_yes"] == 0.8

    def test_think_tags_are_matched_case_insensitively(self) -> None:
        """The tags come from the chat template, which we do not control."""
        upper = '<THINK> draft {"p_yes": 0.9} still thinking'
        assert canonical_prediction(upper) is None, "uppercase guard bypassed"
        closed = (
            'draft {"p_yes": 0.9}</THINK>{"p_yes": 0.3, "p_no": 0.7, '
            '"confidence": 0.6, "info_utility": 0.5}'
        )
        assert json.loads(canonical_prediction(closed) or "{}")["p_yes"] == 0.3

    def test_clamp_max_tokens_handles_every_rejected_shape(self) -> None:
        """Extracted so each rejection path is testable on its own."""
        ceiling = module.MODEL_CONTEXT_WINDOW - module.MIN_PROMPT_BUDGET
        default = module.DEFAULT_MODEL_SETTINGS["max_tokens"]
        # bool is checked before int: isinstance(True, int) is True and
        # int(True) is 1, a budget that cannot produce a parseable answer.
        rejected: Tuple[Any, ...] = (None, 0, -500, True, False, "2048", [], 3.0e-9)
        for raw in rejected:
            assert module._clamp_max_tokens(raw, ceiling) == min(default, ceiling), raw
        assert module._clamp_max_tokens(8000, ceiling) == ceiling
        assert module._clamp_max_tokens(512, ceiling) == 512

    def test_think_tags_match_mixed_case_too(self) -> None:
        """Not just fully-uppercase: a `.upper()` reimplementation would slip."""
        assert canonical_prediction('<Think> draft {"p_yes": 0.9} still') is None
        assert canonical_prediction('<ThInK> draft {"p_yes": 0.9} still') is None
        closed = (
            'draft {"p_yes": 0.9}</ThInk>{"p_yes": 0.3, "p_no": 0.7, '
            '"confidence": 0.6, "info_utility": 0.5}'
        )
        assert json.loads(canonical_prediction(closed) or "{}")["p_yes"] == 0.3

    def test_question_cap_boundary(self) -> None:
        """An off-by-one in the gate would not show up in the coarse tests."""
        model = "olas-predict-r1-14b"
        limit = module._MAX_QUESTION_TOKENS
        short = "word " * 10
        assert module._truncate_to_tokens(short, limit, model) == short
        long_q = "word " * 5000
        capped = module._truncate_to_tokens(long_q, limit, model)
        assert module.budget_tokens(capped, model) <= limit
        assert len(capped) < len(long_q)

    def test_capped_evidence_partial_trim_is_not_empty(self) -> None:
        """Some survivors must read as not-empty, not just all-or-nothing."""
        page = " ".join(["token"] * module._MAX_PAGE_WORDS)
        organic = [
            {
                "title": f"P{i}",
                "link": f"https://e.com/{i}",
                "snippet": page,
                "date": "x",
            }
            for i in range(module.MAX_SOURCES)
        ]
        partial = module._cap_evidence_block(organic, [], "olas-predict-r1-14b", 1500)
        assert 0 < partial.organic_kept < module.MAX_SOURCES
        assert partial.is_empty is False

    def test_source_counts_are_present_on_the_null_paths_too(self) -> None:
        """Consumers index these unconditionally, so every path must carry them."""
        keys = ("sources_used", "sources_dropped")
        result = module._flagged_null_result(
            model="m",
            temperature=0,
            max_tokens=2048,
            captured_source_content=None,
            return_source_content=False,
            counter_callback=None,
            context="live search",
            tier="template",
            sources_dropped=7,
        )
        used = result[4] or {}
        for k in keys:
            assert k in used, f"{k} missing from a flagged null"
        assert used["sources_used"] == 0 and used["sources_dropped"] == 7

    def test_capped_evidence_reports_what_survived(self) -> None:
        """The cap reports counts, so nothing has to string-match the render."""
        # Previously this was inferred by looking for "**Title:**" in the
        # rendered block -- a template reword would have silently flipped it.
        page = " ".join(["token"] * module._MAX_PAGE_WORDS)
        organic = [
            {
                "title": f"P{i}",
                "link": f"https://e.com/{i}",
                "snippet": page,
                "date": "x",
            }
            for i in range(module.MAX_SOURCES)
        ]
        fits = module._cap_evidence_block(organic, [], "olas-predict-r1-14b", 99999)
        assert fits.organic_kept == module.MAX_SOURCES and not fits.is_empty
        starved = module._cap_evidence_block(organic, [], "olas-predict-r1-14b", 10)
        assert starved.organic_kept == 0 and starved.misc_kept == 0
        assert starved.is_empty

    def test_rate_limit_exhaustion_preserves_the_type_for_rotation(self) -> None:
        """429 exhaustion must re-raise RateLimitError, not RuntimeError."""
        # with_key_rotation dispatches on the exact type, so wrapping it in a
        # RuntimeError silently disables key rotation for the one case it exists
        # to handle.
        rate_limit = openai.RateLimitError(
            "429", response=MagicMock(status_code=429, headers={}), body={}
        )
        client = MagicMock()
        client.completions.side_effect = rate_limit
        with patch(f"{SF_MODULE}.time.sleep", return_value=None):
            with pytest.raises(openai.RateLimitError):
                generate_prediction_with_retry(
                    client=client,
                    model="olas-predict-r1-14b",
                    messages=[],
                    temperature=0,
                    max_tokens=2048,
                    retries=2,
                    delay=0,
                )

    def test_prompt_matches_the_parent_byte_for_byte(self) -> None:
        """The lineage claim depends on this staying true, so pin it."""
        # tool_lineage.json calls this a byte-identical copy of the parent's
        # prompt, and the out-of-time evaluation measured the parent's exact
        # tokens. Nothing else in the repo checks it.
        from packages.valory.customs.superforcaster_full_search.superforcaster_full_search import (  # noqa: E501
            PREDICTION_PROMPT as PARENT_PROMPT,
        )

        assert module.PREDICTION_PROMPT == PARENT_PROMPT

    def test_budget_tokens_is_conservative_about_the_tokeniser(self) -> None:
        """Tiktoken has no Qwen encoding, so the raw count under-reads."""
        # Measured against the endpoint's own prompt_tokens: numeric/URL-heavy
        # evidence tokenises 1.158x denser in Qwen than in o200k_base, which on
        # a full prompt is ~740 tokens -- far past CONTEXT_SAFETY_MARGIN.
        text = "BTC/USD closed at $63,412.77 on 2025-09-18 (+2.3%). " * 40
        raw = module.count_tokens(text, "olas-predict-r1-14b")
        scaled = module.budget_tokens(text, "olas-predict-r1-14b")
        assert scaled > raw
        assert module.TOKENIZER_SAFETY_FACTOR >= 1.158, "below the worst measured shape"

    def test_unterminated_think_block_is_rejected(self) -> None:
        """An opener with no closer means the completion was cut off."""
        # Everything present is then a DRAFT written while reasoning. Harvesting
        # one would deliver a working estimate as the final answer -- the exact
        # failure the think strip exists to prevent. More likely at max_tokens
        # 2048 than at the fleet's 4096, so it is a real path, not a curiosity.
        assert (
            canonical_prediction(
                '<think> I estimate {"p_yes": 0.77} tentatively, but need to check'
            )
            is None
        )
        assert (
            canonical_prediction(
                '<think> draft {"p_yes": 0.1} then revise {"p_yes": 0.95} still'
            )
            is None
        )

    def test_bare_closing_think_tag_still_parses(self) -> None:
        """The guard must not break the shape the endpoint actually emits."""
        # The chat template supplies the opener, so completions carry only the
        # closer. Verified against the live endpoint: '<think>' never appears.
        completion = (
            'reasoning with a draft {"p_yes": 0.9}\n</think>\n'
            '{"p_yes": 0.3, "p_no": 0.7, "confidence": 0.6, "info_utility": 0.5}'
        )
        assert json.loads(canonical_prediction(completion) or "{}")["p_yes"] == 0.3

    def test_unparseable_reasoning_completion_is_rejected(self) -> None:
        """A completion without prediction JSON does not produce a delivery."""
        assert canonical_prediction("Reasoning only.") is None
