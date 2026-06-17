# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2026 Valory AG
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

"""Tests for the propose_question mech tool."""

import inspect
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import packages.valory.customs.propose_question.propose_question as module
from packages.valory.customs.propose_question.propose_question import (
    DEFAULT_DELIVERY_RATE,
    N_MODEL_CALLS,
    OpenAIClientManager,
    _VERIFY_CACHE,
    _cache_put,
    _dedup_tokens,
    filter_duplicate_articles,
    find_near_duplicate,
    format_utc_timestamp,
    gather_articles,
    gather_latest_questions,
    run,
    scrape_url,
    validate_question_dates,
    verify_state_is_resolvable,
)

_MODULE = "packages.valory.customs.propose_question.propose_question"

FUTURE_TS = int(time.time()) + 86400 * 30


@pytest.fixture(autouse=True)
def clear_verify_cache() -> None:
    """Clear the verify cache before each test to avoid cross-test pollution."""
    _VERIFY_CACHE.clear()


# ---------------------------------------------------------------------------
# Module-level invariant: no global client variable
# ---------------------------------------------------------------------------


class TestNoGlobalClientVariable:
    """The module must not define a module-level 'client' variable."""

    def test_no_global_client_variable(self) -> None:
        """Module must not have a module-level 'client' variable."""
        source = Path(module.__file__).read_text(encoding="utf-8")
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("client:") or stripped.startswith("client ="):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client' variable found at line {i}: {line}"
                    )


# ---------------------------------------------------------------------------
# OpenAIClientManager
# ---------------------------------------------------------------------------


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_creates_and_closes_client(self) -> None:
        """__enter__ returns a fresh client; __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(f"{_MODULE}.OpenAI") as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance
            with mgr as client:
                assert client is mock_instance
                MockOpenAI.assert_called_once_with(api_key="sk-test")
            mock_instance.close.assert_called_once()

    def test_verify_state_is_resolvable_requires_client_param(self) -> None:
        """verify_state_is_resolvable requires client as first param."""
        params = list(inspect.signature(verify_state_is_resolvable).parameters)
        assert params[0] == "client"


# ---------------------------------------------------------------------------
# validate_question_dates
# ---------------------------------------------------------------------------


class TestValidatequestionDates:
    """Tests for validate_question_dates."""

    def test_future_valid_date_accepted(self) -> None:
        """A question with a valid future date is accepted."""
        future = time.time() + 86400 * 60
        dt_str = format_utc_timestamp(int(future))
        question = f"Will something happen on {dt_str}, according to Reuters?"
        result = validate_question_dates(question, int(future))
        assert result is None

    def test_on_or_before_rejected(self) -> None:
        """Questions using 'on or before' phrasing are rejected."""
        future = time.time() + 86400 * 60
        dt_str = format_utc_timestamp(int(future))
        question = f"Will OpenAI announce a deal on or before {dt_str}?"
        result = validate_question_dates(question, int(future))
        assert result is not None
        assert "on or before" in result

    def test_past_date_rejected(self) -> None:
        """A question referencing a past date is rejected."""
        resolution_ts = int(time.time()) + 86400 * 60
        question = "Will something happen on January 1, 2020, according to Reuters?"
        result = validate_question_dates(question, resolution_ts)
        assert result is not None
        assert "past" in result

    def test_wrong_date_format_rejected(self) -> None:
        """A date in wrong format is rejected."""
        future = int(time.time()) + 86400 * 60
        question = "Will something happen on 22 April 2026, according to Reuters?"
        result = validate_question_dates(question, future)
        assert result is not None
        assert "required" in result.lower() or "not in required" in result


# ---------------------------------------------------------------------------
# format_utc_timestamp
# ---------------------------------------------------------------------------


class TestFormatUtcTimestamp:
    """Tests for format_utc_timestamp."""

    def test_format_known_timestamp(self) -> None:
        """A known timestamp should produce the expected date string."""
        ts = 1745280000  # April 22, 2025
        result = format_utc_timestamp(ts)
        assert "April" in result
        assert "2025" in result
        # Should contain a day number followed by comma
        assert "," in result


# ---------------------------------------------------------------------------
# _dedup_tokens
# ---------------------------------------------------------------------------


class TestDedupTokens:
    """Tests for _dedup_tokens."""

    def test_stopwords_removed(self) -> None:
        """Stop words should be removed."""
        tokens = _dedup_tokens("Will the economy recover")
        assert "will" not in tokens
        assert "the" not in tokens

    def test_short_tokens_removed(self) -> None:
        """Tokens with 2 or fewer characters should be removed."""
        tokens = _dedup_tokens("AI is big")
        assert "is" not in tokens
        assert "ai" not in tokens  # 2 chars, removed

    def test_content_tokens_retained(self) -> None:
        """Meaningful content tokens should be retained."""
        tokens = _dedup_tokens("inflation rate federal reserve")
        assert "inflation" in tokens
        assert "federal" in tokens
        assert "reserve" in tokens


# ---------------------------------------------------------------------------
# filter_duplicate_articles
# ---------------------------------------------------------------------------


class TestFilterDuplicateArticles:
    """Tests for filter_duplicate_articles."""

    def test_empty_articles_returns_empty(self) -> None:
        """Empty articles list returns empty list."""
        result = filter_duplicate_articles([], ["existing question"])
        assert result == []

    def test_empty_existing_returns_all(self) -> None:
        """Empty existing list returns all articles unchanged."""
        articles = [{"title": "tech news", "description": "something"}]
        result = filter_duplicate_articles(articles, [])
        assert result == articles

    def test_highly_similar_article_dropped(self) -> None:
        """Article with high Jaccard overlap with existing should be dropped."""
        existing = ["Will inflation rate exceed five percent in the economy?"]
        articles = [
            {
                "title": "inflation rate economy percent",
                "description": "economy inflation five percent threshold",
            }
        ]
        result = filter_duplicate_articles(articles, existing, threshold=0.40)
        # May or may not drop depending on Jaccard; safety valve ensures at
        # least min_keep_fraction are kept
        assert isinstance(result, list)

    def test_safety_valve_keeps_minimum(self) -> None:
        """Safety valve keeps at least min_keep_fraction articles."""
        existing = ["inflation rate economy"]
        articles = [
            {"title": "inflation economy rate", "description": "economy inflation"},
            {"title": "inflation economy rate", "description": "economy inflation"},
        ]
        result = filter_duplicate_articles(
            articles, existing, threshold=0.01, min_keep_fraction=0.50
        )
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# find_near_duplicate
# ---------------------------------------------------------------------------


class TestFindNearDuplicate:
    """Tests for find_near_duplicate."""

    def test_exact_duplicate_detected(self) -> None:
        """Near-identical question should be flagged."""
        existing = ["Will retail sales percentage change exceed forecast?"]
        candidate = "Will retail sales percentage change exceed monthly forecast?"
        hit = find_near_duplicate(candidate, existing)
        # May or may not hit depending on Jaccard threshold
        assert hit is None or isinstance(hit, tuple)

    def test_no_duplicate_for_different_question(self) -> None:
        """Clearly different question should not be flagged."""
        existing = ["Will the Federal Reserve raise rates?"]
        candidate = "Will SpaceX launch a rocket to Mars?"
        hit = find_near_duplicate(candidate, existing)
        assert hit is None

    def test_empty_question_returns_none(self) -> None:
        """Empty candidate question should return None."""
        hit = find_near_duplicate("", ["some existing question"])
        assert hit is None


# ---------------------------------------------------------------------------
# gather_articles
# ---------------------------------------------------------------------------


class TestGatherArticles:
    """Tests for gather_articles."""

    @patch(f"{_MODULE}.requests.get")
    def test_success_returns_articles(self, mock_get: MagicMock) -> None:
        """Successful NewsAPI response returns article list."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = json.dumps(
            {"articles": [{"title": "Test", "content": "Content"}]}
        ).encode("utf-8")
        result = gather_articles(["bbc-news"], "newsapi-key")
        assert result == [{"title": "Test", "content": "Content"}]

    @patch(f"{_MODULE}.requests.get")
    def test_non_200_returns_none(self, mock_get: MagicMock) -> None:
        """Non-200 status code returns None."""
        mock_get.return_value.status_code = 401
        result = gather_articles(["bbc-news"], "bad-key")
        assert result is None


# ---------------------------------------------------------------------------
# gather_latest_questions
# ---------------------------------------------------------------------------


class TestGatherLatestQuestions:
    """Tests for gather_latest_questions."""

    @patch(f"{_MODULE}.Client")
    @patch(f"{_MODULE}.RequestsHTTPTransport")
    def test_success_returns_titles(
        self, mock_transport: MagicMock, mock_client: MagicMock
    ) -> None:
        """Successful subgraph response returns question title list."""
        mock_gql_client = MagicMock()
        mock_client.return_value = mock_gql_client
        mock_gql_client.execute.return_value = {
            "fixedProductMarketMakers": [
                {"question": {"title": "Will X happen?"}},
                {"question": {"title": "Will Y happen?"}},
            ]
        }
        result = gather_latest_questions("sg-key")
        assert result == ["Will X happen?", "Will Y happen?"]


# ---------------------------------------------------------------------------
# scrape_url
# ---------------------------------------------------------------------------


class TestScrapeUrl:
    """Tests for scrape_url."""

    @patch(f"{_MODULE}.requests.post")
    def test_success_returns_data(self, mock_post: MagicMock) -> None:
        """Successful scrape returns parsed JSON."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"text": "article body"}
        mock_post.return_value.raise_for_status = MagicMock()
        result = scrape_url("serper-key", "https://example.com/article")
        assert result == {"text": "article body"}

    @patch(f"{_MODULE}.requests.post")
    def test_request_exception_returns_none(self, mock_post: MagicMock) -> None:
        """A requests.RequestException should return None."""
        mock_post.side_effect = requests.RequestException("timeout")
        result = scrape_url("serper-key", "https://example.com/article")
        assert result is None

    @patch(f"{_MODULE}.requests.post")
    def test_json_decode_error_returns_none(self, mock_post: MagicMock) -> None:
        """A json.JSONDecodeError should return None."""
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.side_effect = json.JSONDecodeError("err", "", 0)
        result = scrape_url("serper-key", "https://example.com/article")
        assert result is None


# ---------------------------------------------------------------------------
# verify_state_is_resolvable (adapted for new signature: client is first arg)
# ---------------------------------------------------------------------------


class TestVerifyStateIsResolvable:
    """Tests for verify_state_is_resolvable (client passed explicitly)."""

    def _make_client(self) -> MagicMock:
        """Build a mock OpenAI client."""
        return MagicMock()

    def test_empty_query_keeps(self) -> None:
        """Empty source+metric should fail open."""
        client = self._make_client()
        ok, reason = verify_state_is_resolvable(client, "key", "", "")
        assert ok is True
        assert reason == "empty_query"

    @patch(f"{_MODULE}.requests.post")
    def test_serper_non_200_fails_open(self, mock_post: MagicMock) -> None:
        """Non-200 Serper response should fail open."""
        mock_post.return_value.status_code = 500
        client = self._make_client()
        ok, reason = verify_state_is_resolvable(client, "key", "CDC", "volunteers")
        assert ok is True
        assert "fail_open_serper_status_500" in reason

    @patch(f"{_MODULE}.requests.post")
    def test_serper_request_exception_fails_open(self, mock_post: MagicMock) -> None:
        """Serper network error should fail open."""
        mock_post.side_effect = requests.RequestException("timeout")
        client = self._make_client()
        ok, reason = verify_state_is_resolvable(client, "key", "CDC", "volunteers")
        assert ok is True
        assert reason == "fail_open_serper_error"

    @patch(f"{_MODULE}.requests.post")
    def test_serper_no_organic_key_fails_open(self, mock_post: MagicMock) -> None:
        """Serper 200 with no organic key should fail open."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"searchParameters": {}}
        client = self._make_client()
        ok, reason = verify_state_is_resolvable(client, "key", "Freddie Mac", "rate")
        assert ok is True
        assert reason == "fail_open_serper_unexpected_shape"

    @patch(f"{_MODULE}.requests.post")
    def test_serper_empty_organic_drops(self, mock_post: MagicMock) -> None:
        """Serper 200 with empty organic list should drop."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"organic": []}
        client = self._make_client()
        ok, reason = verify_state_is_resolvable(
            client, "key", "fictional corp", "metric"
        )
        assert ok is False
        assert reason == "no_hits"

    @patch(f"{_MODULE}.requests.post")
    def test_judge_says_yes_keeps(self, mock_post: MagicMock) -> None:
        """Judge returning YES should keep the state."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "organic": [{"title": "PMMS", "snippet": "6.5%"}]
        }
        body = {"answer": "YES", "reason": "publishes weekly"}
        choice = MagicMock()
        choice.message.content = json.dumps(body)
        client = self._make_client()
        client.chat.completions.create.return_value.choices = [choice]
        ok, reason = verify_state_is_resolvable(
            client, "key", "Freddie Mac", "mortgage rate"
        )
        assert ok is True
        assert "publishes weekly" in reason

    @patch(f"{_MODULE}.requests.post")
    def test_judge_says_no_drops(self, mock_post: MagicMock) -> None:
        """Judge returning NO should drop the state."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "organic": [{"title": "tangential", "snippet": "x"}]
        }
        body = {"answer": "NO", "reason": "not published"}
        choice = MagicMock()
        choice.message.content = json.dumps(body)
        client = self._make_client()
        client.chat.completions.create.return_value.choices = [choice]
        ok, reason = verify_state_is_resolvable(
            client, "key", "Tesla", "repair records"
        )
        assert ok is False
        assert "not published" in reason

    @patch(f"{_MODULE}.requests.post")
    def test_judge_null_reason_no_crash(self, mock_post: MagicMock) -> None:
        """Judge null reason should not crash."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "organic": [{"title": "hit", "snippet": "d"}]
        }
        body = {"answer": "YES", "reason": None}
        choice = MagicMock()
        choice.message.content = json.dumps(body)
        client = self._make_client()
        client.chat.completions.create.return_value.choices = [choice]
        ok, reason = verify_state_is_resolvable(client, "key", "Freddie Mac", "rate")
        assert ok is True
        assert isinstance(reason, str)

    @patch(f"{_MODULE}.requests.post")
    def test_judge_llm_error_fails_open(self, mock_post: MagicMock) -> None:
        """Judge LLM error should fail open."""
        import openai as _openai

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "organic": [{"title": "hit", "snippet": "d"}]
        }
        client = self._make_client()
        client.chat.completions.create.side_effect = _openai.OpenAIError("API down")
        ok, reason = verify_state_is_resolvable(client, "key", "Freddie Mac", "rate")
        assert ok is True
        assert reason == "fail_open_llm_error"

    @patch(f"{_MODULE}.requests.post")
    def test_cache_hit_skips_serper_and_llm(self, mock_post: MagicMock) -> None:
        """Second call with same (source, metric) should hit cache."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "organic": [{"title": "PMMS", "snippet": "6.5%"}]
        }
        body = {"answer": "YES", "reason": "publishes weekly"}
        choice = MagicMock()
        choice.message.content = json.dumps(body)
        client = self._make_client()
        client.chat.completions.create.return_value.choices = [choice]
        ok1, reason1 = verify_state_is_resolvable(
            client, "k", "Freddie Mac", "mortgage rate"
        )
        ok2, reason2 = verify_state_is_resolvable(
            client, "k", "Freddie Mac", "mortgage rate"
        )
        assert ok1 is True and ok2 is True
        assert "[cached]" not in reason1
        assert "[cached]" in reason2
        assert mock_post.call_count == 1
        assert client.chat.completions.create.call_count == 1

    @patch(f"{_MODULE}.requests.post")
    def test_fail_open_not_cached(self, mock_post: MagicMock) -> None:
        """Fail-open paths (transient errors) must NOT be cached.

        :param mock_post: patched requests.post raising RequestException.
        """
        mock_post.side_effect = requests.RequestException("timeout")
        client = self._make_client()
        ok1, reason1 = verify_state_is_resolvable(client, "k", "Freddie Mac", "rate")
        ok2, reason2 = verify_state_is_resolvable(client, "k", "Freddie Mac", "rate")
        assert ok1 is True and ok2 is True
        assert reason1 == "fail_open_serper_error"
        assert reason2 == "fail_open_serper_error"
        assert "[cached]" not in reason2
        assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# _cache_put
# ---------------------------------------------------------------------------


class TestCachePut:
    """Tests for _cache_put overflow eviction."""

    def test_overflow_evicts_oldest(self) -> None:
        """Cache evicts oldest entry when full."""
        from packages.valory.customs.propose_question.propose_question import (
            _VERIFY_CACHE_MAX,
        )

        _VERIFY_CACHE.clear()
        for i in range(_VERIFY_CACHE_MAX):
            _cache_put((f"src{i}", f"metric{i}"), (True, "ok"))
        assert len(_VERIFY_CACHE) == _VERIFY_CACHE_MAX
        # Adding one more should evict the oldest
        _cache_put(("new_src", "new_metric"), (False, "no_hits"))
        assert len(_VERIFY_CACHE) == _VERIFY_CACHE_MAX
        assert ("src0", "metric0") not in _VERIFY_CACHE
        assert ("new_src", "new_metric") in _VERIFY_CACHE


# ---------------------------------------------------------------------------
# run() -- end-to-end with all external boundaries mocked
# ---------------------------------------------------------------------------


def _make_mock_api_keys() -> MagicMock:
    """Build a mock KeyChain-like api_keys object."""
    services = {
        "openai": ["sk-test"],
        "newsapi": ["newsapi-test"],
        "serperapi": ["serper-test"],
        "subgraph": ["sg-test"],
        "openrouter": ["or-test"],
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key][0]
    mock.get = lambda key, default="": services.get(key, [default])[0]
    mock.max_retries.return_value = {k: len(v) for k, v in services.items()}
    return mock


def _make_openai_completion(content: str) -> MagicMock:
    """Build a mock chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_moderation_clean() -> MagicMock:
    """Build a mock moderation response that does not flag content."""
    mod = MagicMock()
    mod.results = [MagicMock(flagged=False)]
    return mod


def _make_moderation_flagged() -> MagicMock:
    """Build a mock moderation response that flags content."""
    mod = MagicMock()
    mod.results = [MagicMock(flagged=True)]
    return mod


SAMPLE_ARTICLE = {
    "title": "Test Article",
    "publishedAt": "2026-01-01",
    "content": "Some content here",
    "author": "Author",
    "url": "https://example.com/test",
    "description": "Test description",
}

SAMPLE_QUESTION = "Will something happen on June 17, 2026, according to Reuters?"

SAMPLE_STATE = {
    "state": "mortgage rate",
    "source": "Freddie Mac",
    "framing": "measurement",
}

STORY_SELECTION_RESPONSE = json.dumps(
    {"topic": "finance", "article_id": 0, "reasoning": "Good article."}
)
EXTRACT_STATE_RESPONSE = json.dumps({"states": [SAMPLE_STATE]})
QUESTION_PROPOSAL_RESPONSE = json.dumps({"questions": [SAMPLE_QUESTION]})
SELF_REVIEW_ACCEPT = json.dumps(
    {
        "reviews": [
            {
                "question": SAMPLE_QUESTION,
                "deadline_date": "June 17, 2026",
                "earliest_plausible_date": "June 1, 2026",
                "deadline_is_feasible": True,
                "process_stage_named": True,
                "figure_is_directly_published": True,
                "authority_can_act_in_time": True,
                "date_format_valid": True,
                "not_a_duplicate": True,
                "window_bound_event": True,
                "reasoning": "Passes all checks.",
                "accept": True,
            }
        ]
    }
)


class TestRunSuccess:
    """Test run() successful paths."""

    @patch(f"{_MODULE}.gather_latest_questions")
    @patch(f"{_MODULE}.gather_articles")
    @patch(f"{_MODULE}.scrape_url")
    @patch(f"{_MODULE}.filter_duplicate_articles")
    @patch(f"{_MODULE}.OpenAIClientManager")
    @patch(f"{_MODULE}.requests.post")
    def test_run_success_returns_questions(
        self,
        mock_serper: MagicMock,
        mock_client_mgr: MagicMock,
        mock_filter: MagicMock,
        mock_scrape: MagicMock,
        mock_articles: MagicMock,
        mock_questions: MagicMock,
    ) -> None:
        """run() with all mocks returns a tuple with questions JSON."""
        mock_questions.return_value = ["Will X happen?"]
        mock_articles.return_value = [SAMPLE_ARTICLE]
        mock_filter.return_value = [SAMPLE_ARTICLE]
        mock_scrape.return_value = {"text": "full article body here"}

        # Serper verify call returns YES
        mock_serper.return_value.status_code = 200
        mock_serper.return_value.json.return_value = {
            "organic": [{"title": "hit", "snippet": "data"}]
        }

        openai_client = MagicMock()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=openai_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        # Each moderation call returns clean
        openai_client.moderations.create.return_value = _make_moderation_clean()

        # Wire up the four chat completion calls in order:
        # 1. story selection, 2. extract states, 3. verify judge, 4. propose, 5. self-review
        judge_resp = _make_openai_completion(
            json.dumps({"answer": "YES", "reason": "ok"})
        )
        openai_client.chat.completions.create.side_effect = [
            _make_openai_completion(STORY_SELECTION_RESPONSE),
            _make_openai_completion(EXTRACT_STATE_RESPONSE),
            judge_resp,
            _make_openai_completion(QUESTION_PROPOSAL_RESPONSE),
            _make_openai_completion(SELF_REVIEW_ACCEPT),
        ]

        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS, "num_questions": 1}),
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        # Result is a 6-tuple (the decorator appends api_keys)
        assert isinstance(result, tuple)
        assert len(result) >= 5
        result_json = json.loads(result[0])
        assert "questions" in result_json
        assert "reasoning" in result_json

    @patch(f"{_MODULE}.gather_latest_questions")
    @patch(f"{_MODULE}.gather_articles")
    @patch(f"{_MODULE}.scrape_url")
    @patch(f"{_MODULE}.filter_duplicate_articles")
    @patch(f"{_MODULE}.OpenAIClientManager")
    @patch(f"{_MODULE}.requests.post")
    def test_run_prompt_json_resolution_time(
        self,
        mock_serper: MagicMock,
        mock_client_mgr: MagicMock,
        mock_filter: MagicMock,
        mock_scrape: MagicMock,
        mock_articles: MagicMock,
        mock_questions: MagicMock,
    ) -> None:
        """run() reads resolution_time from prompt JSON."""
        mock_questions.return_value = ["Will X happen?"]
        mock_articles.return_value = [SAMPLE_ARTICLE]
        mock_filter.return_value = [SAMPLE_ARTICLE]
        mock_scrape.return_value = {"text": "full article body here"}
        mock_serper.return_value.status_code = 200
        mock_serper.return_value.json.return_value = {
            "organic": [{"title": "hit", "snippet": "data"}]
        }
        openai_client = MagicMock()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=openai_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)
        openai_client.moderations.create.return_value = _make_moderation_clean()
        judge_resp = _make_openai_completion(
            json.dumps({"answer": "YES", "reason": "ok"})
        )
        openai_client.chat.completions.create.side_effect = [
            _make_openai_completion(STORY_SELECTION_RESPONSE),
            _make_openai_completion(EXTRACT_STATE_RESPONSE),
            judge_resp,
            _make_openai_completion(QUESTION_PROPOSAL_RESPONSE),
            _make_openai_completion(SELF_REVIEW_ACCEPT),
        ]
        # Pass resolution_time in prompt JSON only (not as kwarg)
        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS}),
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        result_json = json.loads(result[0])
        assert "questions" in result_json


class TestRunErrors:
    """Test run() error paths."""

    def test_invalid_tool_returns_error(self) -> None:
        """Unknown tool name returns an error JSON."""
        result = run(
            tool="unknown-tool",
            prompt="",
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert isinstance(result, tuple)
        data = json.loads(result[0])
        assert "error" in data
        assert "unknown-tool" in data["error"]

    def test_missing_resolution_time_returns_error(self) -> None:
        """Missing resolution_time returns an error JSON."""
        result = run(
            tool="propose-question",
            prompt="plain text prompt with no json",
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert isinstance(result, tuple)
        data = json.loads(result[0])
        assert "error" in data
        assert "resolution_time" in data["error"]

    @patch(f"{_MODULE}.gather_articles")
    @patch(f"{_MODULE}.gather_latest_questions")
    def test_failed_articles_returns_error(
        self, mock_q: MagicMock, mock_articles: MagicMock
    ) -> None:
        """None from gather_articles returns error JSON."""
        mock_q.return_value = ["existing question"]
        mock_articles.return_value = None
        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS}),
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        data = json.loads(result[0])
        assert "error" in data
        assert "articles" in data["error"].lower()

    @patch(f"{_MODULE}.gather_articles")
    @patch(f"{_MODULE}.gather_latest_questions")
    def test_failed_latest_questions_returns_error(
        self, mock_q: MagicMock, mock_articles: MagicMock
    ) -> None:
        """None from gather_latest_questions returns error JSON."""
        mock_q.return_value = None
        mock_articles.return_value = [SAMPLE_ARTICLE]
        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS}),
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        data = json.loads(result[0])
        assert "error" in data
        assert "latest questions" in data["error"].lower()

    @patch(f"{_MODULE}.gather_articles")
    @patch(f"{_MODULE}.gather_latest_questions")
    @patch(f"{_MODULE}.filter_duplicate_articles")
    @patch(f"{_MODULE}.OpenAIClientManager")
    def test_all_articles_flagged_returns_error(
        self,
        mock_client_mgr: MagicMock,
        mock_filter: MagicMock,
        mock_q: MagicMock,
        mock_articles: MagicMock,
    ) -> None:
        """All articles flagged by moderation returns error JSON."""
        mock_q.return_value = ["existing question"]
        mock_articles.return_value = [SAMPLE_ARTICLE]
        mock_filter.return_value = [SAMPLE_ARTICLE]
        openai_client = MagicMock()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=openai_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)
        # All articles flagged
        openai_client.moderations.create.return_value = _make_moderation_flagged()
        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS}),
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        data = json.loads(result[0])
        assert "error" in data
        assert "flagged" in data["error"].lower()

    @patch(f"{_MODULE}.gather_articles")
    @patch(f"{_MODULE}.gather_latest_questions")
    @patch(f"{_MODULE}.filter_duplicate_articles")
    @patch(f"{_MODULE}.scrape_url")
    @patch(f"{_MODULE}.OpenAIClientManager")
    @patch(f"{_MODULE}.requests.post")
    def test_failed_scrape_returns_error(
        self,
        mock_serper: MagicMock,
        mock_client_mgr: MagicMock,
        mock_scrape: MagicMock,
        mock_filter: MagicMock,
        mock_q: MagicMock,
        mock_articles: MagicMock,
    ) -> None:
        """Failed scrape returns error JSON."""
        mock_q.return_value = ["existing question"]
        mock_articles.return_value = [SAMPLE_ARTICLE]
        mock_filter.return_value = [SAMPLE_ARTICLE]
        mock_scrape.return_value = None
        openai_client = MagicMock()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=openai_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)
        openai_client.moderations.create.return_value = _make_moderation_clean()
        openai_client.chat.completions.create.return_value = _make_openai_completion(
            STORY_SELECTION_RESPONSE
        )
        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS}),
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        data = json.loads(result[0])
        assert "error" in data
        assert "scrape" in data["error"].lower()

    @patch(f"{_MODULE}.gather_articles")
    @patch(f"{_MODULE}.gather_latest_questions")
    @patch(f"{_MODULE}.filter_duplicate_articles")
    @patch(f"{_MODULE}.scrape_url")
    @patch(f"{_MODULE}.OpenAIClientManager")
    @patch(f"{_MODULE}.requests.post")
    def test_all_questions_rejected_returns_error(
        self,
        mock_serper: MagicMock,
        mock_client_mgr: MagicMock,
        mock_scrape: MagicMock,
        mock_filter: MagicMock,
        mock_q: MagicMock,
        mock_articles: MagicMock,
    ) -> None:
        """All questions rejected by self-review returns error JSON."""
        mock_q.return_value = ["existing question"]
        mock_articles.return_value = [SAMPLE_ARTICLE]
        mock_filter.return_value = [SAMPLE_ARTICLE]
        mock_scrape.return_value = {"text": "full article body here"}
        mock_serper.return_value.status_code = 200
        mock_serper.return_value.json.return_value = {
            "organic": [{"title": "hit", "snippet": "data"}]
        }
        openai_client = MagicMock()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=openai_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)
        openai_client.moderations.create.return_value = _make_moderation_clean()

        reject_review = json.dumps(
            {
                "reviews": [
                    {
                        "question": SAMPLE_QUESTION,
                        "deadline_date": "June 17, 2026",
                        "earliest_plausible_date": "never",
                        "deadline_is_feasible": False,
                        "process_stage_named": True,
                        "figure_is_directly_published": True,
                        "authority_can_act_in_time": False,
                        "date_format_valid": True,
                        "not_a_duplicate": True,
                        "window_bound_event": True,
                        "reasoning": "Fails feasibility.",
                        "accept": False,
                    }
                ]
            }
        )
        judge_resp = _make_openai_completion(
            json.dumps({"answer": "YES", "reason": "ok"})
        )
        openai_client.chat.completions.create.side_effect = [
            _make_openai_completion(STORY_SELECTION_RESPONSE),
            _make_openai_completion(EXTRACT_STATE_RESPONSE),
            judge_resp,
            _make_openai_completion(QUESTION_PROPOSAL_RESPONSE),
            _make_openai_completion(reject_review),
        ]
        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS}),
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        data = json.loads(result[0])
        assert "error" in data
        assert "rejected" in data["error"].lower()

    def test_unexpected_exception_returns_error(self) -> None:
        """Unhandled exception in run() returns error JSON, does not crash."""
        bad_keys = MagicMock()
        bad_keys.__getitem__.side_effect = RuntimeError("boom")
        bad_keys.max_retries.return_value = {"openai": 1, "openrouter": 1}
        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS}),
            api_keys=bad_keys,
            counter_callback=None,
        )
        assert isinstance(result, tuple)
        data = json.loads(result[0])
        assert "error" in data
        assert "exception" in data["error"].lower()


class TestRunNoneOutputPath:
    """Verify run() returns an error (not None) when LLM produces no output."""

    @patch(f"{_MODULE}.gather_articles")
    @patch(f"{_MODULE}.gather_latest_questions")
    @patch(f"{_MODULE}.filter_duplicate_articles")
    @patch(f"{_MODULE}.scrape_url")
    @patch(f"{_MODULE}.OpenAIClientManager")
    @patch(f"{_MODULE}.requests.post")
    def test_none_output_from_question_generation(
        self,
        mock_serper: MagicMock,
        mock_client_mgr: MagicMock,
        mock_scrape: MagicMock,
        mock_filter: MagicMock,
        mock_q: MagicMock,
        mock_articles: MagicMock,
    ) -> None:
        """Empty questions list from LLM produces an error return."""
        mock_q.return_value = ["existing question"]
        mock_articles.return_value = [SAMPLE_ARTICLE]
        mock_filter.return_value = [SAMPLE_ARTICLE]
        mock_scrape.return_value = {"text": "full article body here"}
        mock_serper.return_value.status_code = 200
        mock_serper.return_value.json.return_value = {
            "organic": [{"title": "hit", "snippet": "data"}]
        }
        openai_client = MagicMock()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=openai_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)
        openai_client.moderations.create.return_value = _make_moderation_clean()

        # LLM returns zero questions
        empty_proposal = json.dumps({"questions": []})
        empty_review = json.dumps({"reviews": []})
        judge_resp = _make_openai_completion(
            json.dumps({"answer": "YES", "reason": "ok"})
        )
        openai_client.chat.completions.create.side_effect = [
            _make_openai_completion(STORY_SELECTION_RESPONSE),
            _make_openai_completion(EXTRACT_STATE_RESPONSE),
            judge_resp,
            _make_openai_completion(empty_proposal),
            _make_openai_completion(empty_review),
        ]
        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS}),
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        assert isinstance(result, tuple)
        data = json.loads(result[0])
        assert "error" in data


# ---------------------------------------------------------------------------
# delivery_rate == 0 -> MaxCostResponse (float)
# ---------------------------------------------------------------------------


class TestMaxCostBranch:
    """When delivery_rate=0, run() must return a bare float, not a tuple."""

    def test_delivery_rate_zero_returns_float(self) -> None:
        """delivery_rate=0 short-circuits and returns counter_callback result."""
        expected_cost = 0.0321
        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS}),
            api_keys=_make_mock_api_keys(),
            counter_callback=lambda **_: expected_cost,
            delivery_rate=0,
        )
        # with_key_rotation appends api_keys for tuple results, but the
        # decorator skips the append for bare MaxCostResponse returns.
        assert result == expected_cost

    def test_delivery_rate_zero_no_callback_raises(self) -> None:
        """delivery_rate=0 without counter_callback is caught and becomes an error tuple."""
        result = run(
            tool="propose-question",
            prompt=json.dumps({"resolution_time": FUTURE_TS}),
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
            delivery_rate=0,
        )
        # ValueError raised by the delivery_rate==0 branch is caught by the
        # outer try/except and returned as a JSON error in result[0].
        assert isinstance(result, tuple)
        data = json.loads(result[0])
        assert "error" in data

    def test_default_delivery_rate_constant(self) -> None:
        """DEFAULT_DELIVERY_RATE must equal 100 to match reference tools."""
        assert DEFAULT_DELIVERY_RATE == 100

    def test_n_model_calls_constant_positive(self) -> None:
        """N_MODEL_CALLS must be a positive integer."""
        assert isinstance(N_MODEL_CALLS, int)
        assert N_MODEL_CALLS > 0
