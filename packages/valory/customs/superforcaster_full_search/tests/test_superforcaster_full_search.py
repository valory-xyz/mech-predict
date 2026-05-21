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

import packages.valory.customs.superforcaster_full_search.superforcaster_full_search as module
from packages.valory.customs.superforcaster_full_search.superforcaster_full_search import (
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


def _stub_openai(mock_client_mgr: MagicMock) -> MagicMock:
    """Wire OpenAIClientManager to a stub that returns PREDICTION_JSON once."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=PREDICTION_JSON))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
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
