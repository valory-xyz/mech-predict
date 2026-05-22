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

"""Unit tests for superforcaster_full_search: page scrape, capture/replay, fallbacks."""

import inspect
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

import packages.valory.customs.superforcaster_full_search.superforcaster_full_search as module
from packages.valory.customs.superforcaster_full_search.superforcaster_full_search import (
    OpenAIClientManager,
    OpenAIResponse,
    Usage,
    fetch_additional_sources,
    generate_prediction_with_retry,
    run,
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh OpenAIClient, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(
            "packages.valory.customs.superforcaster_full_search.superforcaster_full_search.OpenAIClient"
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
    "packages.valory.customs.superforcaster_full_search.superforcaster_full_search"
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

# (cleaned_text, capture_payload) tuples — matches _fetch_page_content's return
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
    # completions.return_value to a real OpenAIResponse — otherwise result[0]
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
    """Verify superforcaster_full_search captures and replays source_content correctly."""

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
            tool="superforcaster_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
        )

        captured = result[4]["source_content"]
        assert captured["mode"] == "cleaned"
        assert captured["serper_response"] == FAKE_SERPER_RESPONSE
        # Both organic URLs were scraped → both in pages capture
        assert captured["pages"] == {
            "http://example.com/result": FAKE_PAGE_CONTENT,
            "http://example.com/second": "Second page body.",
        }
        # Scraped page text reaches the prediction prompt under "Content:"
        prediction_prompt = result[1]
        assert FAKE_PAGE_CONTENT in prediction_prompt
        assert "**Content:**" in prediction_prompt
        # result[0] is the LLM completion content (via the OpenAIClient
        # wrapper), not an auto-MagicMock — so this JSON assertion is real.
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
            tool="superforcaster_full_search",
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
            tool="superforcaster_full_search",
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
            tool="superforcaster_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        prediction_prompt = result[1]
        # raw HTML was run back through _clean_html → extracted article text
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
            tool="superforcaster_full_search",
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
            tool="superforcaster_full_search",
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
        from packages.valory.customs.superforcaster_full_search.superforcaster_full_search import (
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
        from packages.valory.customs.superforcaster_full_search.superforcaster_full_search import (
            _scrape_pages,
        )

        organic = [{"link": "http://example.com/x", "title": "T", "snippet": "s"}]
        captured = _scrape_pages(organic, mode="cleaned")
        assert "content" not in organic[0]
        assert captured == {}

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    def test_scrape_pages_mixed_success(self, _mock_page_fetch: MagicMock) -> None:
        """One URL succeeds, one fails: only the success gets content + capture."""
        from packages.valory.customs.superforcaster_full_search.superforcaster_full_search import (
            _scrape_pages,
        )

        organic = [
            {"link": "http://example.com/result", "title": "T1", "snippet": "s1"},
            {"link": "http://example.com/unknown", "title": "T2", "snippet": "s2"},
        ]
        captured = _scrape_pages(organic, mode="cleaned")
        # success → content attached + in capture; failure → neither (exercises
        # the `if text:` / `if capture:` guards).
        assert organic[0]["content"] == FAKE_PAGE_CONTENT
        assert "content" not in organic[1]
        assert captured == {"http://example.com/result": FAKE_PAGE_CONTENT}


class TestEvidenceBlockCap:
    """The cap drops trailing organic items until the rendered block fits."""

    def test_small_evidence_unchanged(self) -> None:
        """Below-budget evidence is returned without truncation marker."""
        from packages.valory.customs.superforcaster_full_search.superforcaster_full_search import (
            _cap_evidence_block,
        )

        organic = [
            {"title": "T", "link": "http://x", "snippet": "s", "position": 1},
        ]
        rendered = _cap_evidence_block(organic, [], model="gpt-4.1")
        assert "[… evidence truncated …]" not in rendered
        assert "T" in rendered

    def test_oversize_evidence_is_trimmed_with_marker(self) -> None:
        """When over budget, trailing items are dropped and a marker is appended."""
        from packages.valory.customs.superforcaster_full_search.superforcaster_full_search import (
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
        rendered = _cap_evidence_block(organic, [], model="gpt-4.1")
        assert "[… evidence truncated …]" in rendered
        assert count_tokens(rendered, "gpt-4.1") <= MAX_EVIDENCE_TOKENS + 100
        # Trailing items are dropped, leading (most-relevant) kept: a
        # leading-drop mutation would keep T4 and drop T0, failing this.
        assert "T0" in rendered
        assert "T4" not in rendered

    def test_paa_only_overflow_returns_without_loop(self) -> None:
        """With no organic items the cap returns as-is (no marker, no infinite loop)."""
        from packages.valory.customs.superforcaster_full_search.superforcaster_full_search import (
            _cap_evidence_block,
        )

        huge_paa = [
            {"question": "lorem ipsum " * 800, "link": "http://x", "snippet": "s"}
        ]
        rendered = _cap_evidence_block([], huge_paa, model="gpt-4.1")
        # organic is empty → early return, no trailing-drop marker added
        assert "[… evidence truncated …]" not in rendered
        assert "lorem ipsum" in rendered


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
        """200 + HTML → (cleaned_text, cleaned_text) in cleaned mode."""
        mock_get.return_value = self._resp(text=_HTML_PAGE)
        text, capture = module._fetch_page_content("http://x", mode="cleaned")
        assert text is not None and "Federal Reserve" in text
        assert capture == text  # cleaned mode stores the cleaned text

    @patch(f"{SF_MODULE}.requests.get")
    def test_happy_path_raw_stores_html(self, mock_get: MagicMock) -> None:
        """200 + HTML → capture is the raw HTML in raw mode."""
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
        """A network exception is swallowed → (None, None)."""
        mock_get.side_effect = requests.Timeout("slow")
        assert module._fetch_page_content("http://x") == (None, None)

    @patch(f"{SF_MODULE}._clean_html", return_value=None)
    @patch(f"{SF_MODULE}.requests.get")
    def test_unextractable_html_returns_none(
        self, mock_get: MagicMock, _mock_clean: MagicMock
    ) -> None:
        """200 + HTML but readability extracts nothing → (None, None)."""
        mock_get.return_value = self._resp(text="<html></html>")
        assert module._fetch_page_content("http://x") == (None, None)


class TestErrorHandling:
    """with_key_rotation's catch-all returns parseable null-prediction JSON."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_unexpected_error_returns_parseable_error_json(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """An unexpected exception yields {p_yes:None,...,error:...}, not a raw string."""
        _stub_openai(mock_client_mgr)
        mock_fetch.side_effect = RuntimeError("boom")

        result = run(
            tool="superforcaster_full_search",
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
            tool="superforcaster_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )

        payload = json.loads(result[0])  # not None → no downstream json.loads(None)
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
        """A 4xx/5xx Serper response raises via raise_for_status → error JSON."""
        _stub_openai(mock_client_mgr)
        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many")
        mock_fetch.return_value = bad_response

        result = run(
            tool="superforcaster_full_search",
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
            tool="superforcaster_full_search",
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
            tool="superforcaster_full_search",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=lambda **_: 0.0123,
            delivery_rate=0,
        )
        assert result == 0.0123
