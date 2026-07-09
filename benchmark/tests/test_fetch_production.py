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
"""Tests for benchmark/datasets/fetch_production.py"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from benchmark.datasets.fetch_production import (
    DEDUP_LOOKBACK_DAYS,
    DELIVERS_SCHEMA_LEGACY,
    DELIVERS_SCHEMA_PARSED,
    PARSED_DELIVERY_GRACE_SECONDS,
    PENDING_MAX_AGE_DAYS,
    ResolvedMarkets,
    SUBGRAPH_UNHANDLED_TYPE,
    _compute_config_hash,
    _dedup_pending,
    _extract_question_title,
    _load_ids_from_file,
    _make_row_id,
    _match_and_build,
    _match_delivery,
    _migrate_legacy_log,
    _parse_request_context,
    _should_defer_unparsed,
    append_rows,
    build_row,
    classify_category,
    daily_log_path,
    detect_delivers_schema,
    extract_delivery_fields,
    load_existing_row_ids,
    load_fetch_state,
    parse_tool_response,
    refresh_unparsed_pending,
    save_fetch_state,
)

QUESTION_DATA_SEPARATOR = "\u241f"


# ---------------------------------------------------------------------------
# parse_tool_response
# ---------------------------------------------------------------------------


class TestParseToolResponse:
    """Tests for toolResponse JSON parsing."""

    def test_valid_json(self) -> None:
        """Test valid JSON response parsing."""
        resp = parse_tool_response('{"p_yes": 0.72, "p_no": 0.28, "confidence": 0.85}')
        assert resp["prediction_parse_status"] == "valid"
        assert resp["p_yes"] == 0.72
        assert resp["p_no"] == 0.28
        assert resp["confidence"] == 0.85

    def test_valid_json_with_newlines(self) -> None:
        """Test valid JSON with newline formatting."""
        resp = parse_tool_response(
            '{\n  "p_yes": 0.82,\n  "p_no": 0.18,\n  "confidence": 0.85\n}'
        )
        assert resp["prediction_parse_status"] == "valid"
        assert resp["p_yes"] == 0.82

    def test_incoherent_probabilities_rejected(self) -> None:
        """p_yes + p_no = 1.8, well outside tolerance."""
        resp = parse_tool_response('{"p_yes": 0.9, "p_no": 0.9}')
        assert resp["prediction_parse_status"] == "malformed"

    def test_within_tolerance_accepted(self) -> None:
        """p_yes + p_no = 1.02, within PROBABILITY_SUM_TOLERANCE (0.05)."""
        resp = parse_tool_response('{"p_yes": 0.72, "p_no": 0.30}')
        assert resp["prediction_parse_status"] == "valid"
        assert resp["p_yes"] == 0.72

    def test_null_response(self) -> None:
        """Test None response handling."""
        resp = parse_tool_response(None)
        assert resp["prediction_parse_status"] == "missing_fields"
        assert resp["p_yes"] is None

    def test_empty_string(self) -> None:
        """Test empty string handling."""
        resp = parse_tool_response("")
        assert resp["prediction_parse_status"] == "missing_fields"

    def test_ipfs_error(self) -> None:
        """Test IPFS error detection."""
        resp = parse_tool_response(
            "Request data could not be retrieved from IPFS "
            "(detail: Failed to download: bafyxyz)"
        )
        assert resp["prediction_parse_status"] == "error"

    def test_failed_in_valid_json_not_treated_as_error(self) -> None:
        """A valid JSON response containing 'failed' should still parse."""
        resp = parse_tool_response(
            '{"p_yes": 0.7, "p_no": 0.3, "confidence": 0.8, '
            '"reasoning": "The ceasefire talks have failed."}'
        )
        assert resp["prediction_parse_status"] == "valid"
        assert resp["p_yes"] == 0.7

    def test_malformed_missing_p_no(self) -> None:
        """Test malformed response with missing p_no."""
        resp = parse_tool_response('{"p_yes": 0.72}')
        assert resp["prediction_parse_status"] == "malformed"

    def test_malformed_out_of_range(self) -> None:
        """Test malformed response with out-of-range values."""
        resp = parse_tool_response('{"p_yes": 5.0, "p_no": -4.0}')
        assert resp["prediction_parse_status"] == "malformed"

    def test_regex_fallback(self) -> None:
        """Regex extraction works when JSON is wrapped in extra text."""
        resp = parse_tool_response('Here is the result: {"p_yes": 0.6, "p_no": 0.4}')
        assert resp["prediction_parse_status"] == "valid"
        assert resp["p_yes"] == 0.6

    def test_regex_also_validates_sum(self) -> None:
        """Regex path should also reject incoherent probabilities."""
        resp = parse_tool_response('Result: {"p_yes": 0.8, "p_no": 0.8}')
        assert resp["prediction_parse_status"] == "malformed"

    def test_unhandled_type_sentinel_is_malformed(self) -> None:
        """Test a malformed unhandled type sentinel.

        The subgraph's ``[unhandled type]`` sentinel means the payload was
        fetched but had no parseable ``result`` — malformed, not missing.
        """
        result = parse_tool_response(SUBGRAPH_UNHANDLED_TYPE)
        assert result["prediction_parse_status"] == "malformed"
        assert result["p_yes"] is None

    def test_unparseable_garbage(self) -> None:
        """Test unparseable garbage input."""
        resp = parse_tool_response("not json at all")
        assert resp["prediction_parse_status"] == "malformed"

    def test_no_confidence_field(self) -> None:
        """Test response without confidence field."""
        resp = parse_tool_response('{"p_yes": 0.5, "p_no": 0.5}')
        assert resp["prediction_parse_status"] == "valid"
        assert resp["confidence"] is None


# ---------------------------------------------------------------------------
# Neg-risk outcome decoding
# ---------------------------------------------------------------------------


class TestPolymarketOutcome:
    """Tests for Polymarket outcome decoding.

    winningIndex follows CLOB token order: 0 = Yes, 1 = No.
    The subgraph outcomes array is unreliable and ignored.
    """

    @staticmethod
    def _decode(winning_index: int) -> bool:
        """Replicate the outcome decoding logic from fetch_polymarket_resolved."""
        return winning_index == 0

    def test_yes(self) -> None:
        """Winning index 0 maps to Yes."""
        assert self._decode(0) is True

    def test_no(self) -> None:
        """Winning index 1 maps to No."""
        assert self._decode(1) is False


# ---------------------------------------------------------------------------
# _match_delivery
# ---------------------------------------------------------------------------


class TestMatchDelivery:
    """Tests for _match_delivery."""

    @staticmethod
    def _make_markets() -> ResolvedMarkets:
        markets = ResolvedMarkets()
        markets.add(
            "0xabc",
            "Will Bitcoin hit $100k by June?",
            {"outcome": True, "resolved_at_ts": 100},
        )
        markets.add(
            "0xdef",
            "Will the president win the next election?",
            {"outcome": False, "resolved_at_ts": 200},
        )
        return markets

    def test_match_by_market_id(self) -> None:
        """Test matching delivery by market ID."""
        markets = self._make_markets()
        delivery = {"market_id": "0xabc", "question_title": "totally different"}
        market, confidence = _match_delivery(delivery, markets)
        assert market is not None
        assert market["outcome"] is True
        assert confidence == 1.0

    def test_market_id_takes_priority_over_title(self) -> None:
        """Test market ID takes priority over title."""
        markets = self._make_markets()
        delivery = {
            "market_id": "0xabc",
            "question_title": "Will the president win the next election?",
        }
        # market_id 0xabc → outcome True, even though title matches 0xdef (False)
        market, _confidence = _match_delivery(delivery, markets)
        assert market is not None
        assert market["outcome"] is True

    def test_exact_title_match(self) -> None:
        """Test exact title match."""
        markets = self._make_markets()
        delivery = {
            "market_id": None,
            "question_title": "Will the president win the next election?",
        }
        market, confidence = _match_delivery(delivery, markets)
        assert market is not None
        assert market["outcome"] is False
        assert confidence == 1.0

    def test_prefix_match(self) -> None:
        """Test prefix title match."""
        markets = self._make_markets()
        delivery = {
            "market_id": None,
            "question_title": "Will Bitcoin hit $100k by June? More context here",
        }
        market, confidence = _match_delivery(delivery, markets)
        assert market is not None
        assert confidence == 0.8

    def test_short_prefix_rejected(self) -> None:
        """Prefix match requires min 20 chars."""
        markets = ResolvedMarkets()
        markets.add(None, "Will it", {"outcome": True, "resolved_at_ts": 100})
        delivery = {"market_id": None, "question_title": "Will it rain in London?"}
        market, _confidence = _match_delivery(delivery, markets)
        assert market is None

    def test_no_match(self) -> None:
        """Test no match returns None."""
        markets = self._make_markets()
        delivery = {"market_id": None, "question_title": "Completely unrelated"}
        market, confidence = _match_delivery(delivery, markets)
        assert market is None
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# fetch_polymarket_resolved (regression: questions-entity discovery)
# ---------------------------------------------------------------------------


class TestFetchPolymarketResolvedUsesQuestions:
    """Regression: fetch_polymarket_resolved must discover via questions.

    Uses the ``resolution_.blockTimestamp_gt`` server-side filter on the
    ``questions`` entity, not the legacy bets path. A question that has a
    resolution but no bet fixture must still be discovered.
    """

    def test_uses_questions_entity_and_skips_invalid_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Discovers via ``questions``, encodes outcome, skips malformed rows."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        captured: dict[str, Any] = {}

        def fake_paginated(
            url: str,
            query: str,
            entity_key: str,
            template_vars: dict[str, Any],
            **_kwargs: Any,
        ) -> list[dict[str, Any]]:
            captured["url"] = url
            captured["query"] = query
            captured["entity_key"] = entity_key
            captured["template_vars"] = template_vars
            return [
                {
                    "id": "0xqid_resolved",
                    "metadata": {
                        "title": "Will X happen by date Y?",
                        "outcomes": ["No", "Yes"],
                    },
                    "resolution": {"winningIndex": 0, "blockTimestamp": "1700000000"},
                },
                # Edge case: missing winningIndex — must be skipped.
                {
                    "id": "0xqid_partial",
                    "metadata": {"title": "Partial"},
                    "resolution": {"blockTimestamp": "1700000000"},
                },
                # Edge case: missing title — must be skipped.
                {
                    "id": "0xqid_no_title",
                    "metadata": {"title": ""},
                    "resolution": {"winningIndex": 1, "blockTimestamp": "1700000000"},
                },
                # Edge case: null resolution — must be skipped.
                {
                    "id": "0xqid_null_res",
                    "metadata": {"title": "Null resolution"},
                    "resolution": None,
                },
            ]

        monkeypatch.setattr(fp, "_paginated_fetch", fake_paginated)
        markets = fp.fetch_polymarket_resolved(resolved_after=1_600_000_000)

        assert captured["entity_key"] == "questions"
        assert "questions(" in captured["query"]
        assert "bets(" not in captured["query"]
        assert captured["template_vars"] == {"resolved_after": 1_600_000_000}

        assert "0xqid_resolved" in markets.by_id
        assert markets.by_id["0xqid_resolved"]["outcome"] is True
        assert markets.by_id["0xqid_resolved"]["resolved_at_ts"] == 1_700_000_000
        assert "0xqid_partial" not in markets.by_id
        assert "0xqid_no_title" not in markets.by_id
        assert "0xqid_null_res" not in markets.by_id

    def test_winning_index_one_maps_to_no(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """winningIndex=1 maps to outcome=False."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        def fake_paginated(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "id": "0xqid_no",
                    "metadata": {"title": "Will it not happen?"},
                    "resolution": {"winningIndex": 1, "blockTimestamp": "1700000000"},
                },
            ]

        monkeypatch.setattr(fp, "_paginated_fetch", fake_paginated)
        markets = fp.fetch_polymarket_resolved(resolved_after=0)
        assert markets.by_id["0xqid_no"]["outcome"] is False


# ---------------------------------------------------------------------------
# classify_category
# ---------------------------------------------------------------------------


class TestClassifyCategory:
    """Tests for classify_category."""

    @pytest.mark.parametrize(
        "question,expected",
        [
            ("Will Bitcoin hit $100k?", "finance"),
            ("Will ETH price rise?", "finance"),
            ("Will the president win the election?", "politics"),
            ("Will Tesla revenue grow?", "business"),
            ("Will NASA launch the rocket?", "science"),
            ("Will Apple release a new iPhone?", "technology"),
            ("Will GDP grow this quarter?", "business"),
            ("Will NATO expand?", "international"),
            ("Will the NBA finals be exciting?", "sports"),
            ("Will Netflix release a new series?", "entertainment"),
            ("Will the hurricane hit Florida?", "weather"),
        ],
    )
    def test_known_categories(self, question: str, expected: str) -> None:
        """Test known category classification."""
        assert classify_category(question) == expected

    def test_unknown_falls_back_to_other(self) -> None:
        """Test unknown question falls back to other."""
        assert classify_category("Will something random happen?") == "other"

    def test_word_boundary_prevents_substring_match(self) -> None:
        """'eth' should not match inside 'something'."""
        assert classify_category("Will something happen?") == "other"

    def test_case_insensitive(self) -> None:
        """Test case-insensitive classification."""
        assert classify_category("WILL BITCOIN HIT $100K?") == "finance"


class TestClassifyCategoryPlatformAware:
    """Platform-aware filter routes off-list categories to ``other``.

    When ``platform`` is provided, the keyword-classified category must be
    in that platform's upstream taxonomy (``OMEN_CATEGORIES`` for omen,
    ``POLYMARKET_ACTIVE_CATEGORIES`` for polymarket); otherwise the row
    drops to ``"other"`` so per-platform reports never advertise a
    category the platform doesn't actually trade.
    """

    def test_travel_question_omen_buckets_as_other(self) -> None:
        """``travel`` is in CATEGORY_KEYWORDS but NOT in OMEN_CATEGORIES.

        Market-creator never emits ``travel``; a keyword leak (e.g. an
        omen question mentioning a flight or vacation) must land in
        ``other`` instead.
        """
        assert (
            classify_category("Will the airline launch a new flight route?", "omen")
            == "other"
        )

    def test_curiosities_question_polymarket_buckets_as_other(self) -> None:
        """``curiosities`` is not in trader's POLYMARKET_CATEGORY_TAGS."""
        assert (
            classify_category("Will UFO sightings double this year?", "polymarket")
            == "other"
        )

    def test_legitimate_finance_passes_through_for_polymarket(self) -> None:
        """Categories already in the platform's allowed set pass unchanged."""
        assert (
            classify_category("Will Tesla stock close above 380?", "polymarket")
            == "business"
        )
        assert (
            classify_category(
                "Will the S&P 500 close above 7000 on Friday?", "polymarket"
            )
            == "finance"
        )

    def test_legitimate_omen_categories_pass_through(self) -> None:
        """Omen questions hitting OMEN_CATEGORIES keep their bucket."""
        assert (
            classify_category("Will the president win the election?", "omen")
            == "politics"
        )
        assert classify_category("Will the hurricane hit Florida?", "omen") == "weather"

    def test_no_platform_preserves_legacy_behavior(self) -> None:
        """``platform=None`` (default) keeps the historical, unfiltered behavior.

        Used by callers that don't yet know the platform. Backward-
        compatible — a row whose keyword matches ``travel`` still gets
        ``travel`` back.
        """
        assert (
            classify_category("Will the airline launch a new flight route?") == "travel"
        )

    def test_unknown_platform_disables_filter(self) -> None:
        """A platform key not in ``PLATFORM_ALLOWED_CATEGORIES`` disables the filter.

        Defensive: a future platform name shouldn't silently drop every
        row to ``other`` before its allowed set is registered.
        """
        assert (
            classify_category(
                "Will the airline launch a new flight route?", "future-chain"
            )
            == "travel"
        )

    def test_no_keyword_match_returns_other_regardless_of_platform(self) -> None:
        """A question with no keyword match is always ``other``."""
        assert classify_category("Will quux?", "omen") == "other"
        assert classify_category("Will quux?", "polymarket") == "other"


# ---------------------------------------------------------------------------
# _parse_request_context
# ---------------------------------------------------------------------------


class TestParseRequestContext:
    """Tests for _parse_request_context."""

    def test_schema_v2(self) -> None:
        """Test schema v2 request context parsing."""
        content = json.dumps(
            {
                "prompt": "...",
                "tool": "superforcaster",
                "schema_version": "2.0",
                "request_context": {
                    "market_id": "0xabc",
                    "type": "polymarket",
                    "market_prob": 0.65,
                    "market_liquidity_usd": 1234.56,
                    "market_close_at": "2026-04-01T00:00:00Z",
                },
            }
        )
        ctx = _parse_request_context(content)
        assert ctx["market_id"] == "0xabc"
        assert ctx["market_type"] == "polymarket"
        assert ctx["market_prob"] == 0.65

    def test_schema_v1_no_context(self) -> None:
        """Test schema v1 without request context."""
        content = json.dumps({"prompt": "...", "tool": "test", "nonce": "abc"})
        assert not _parse_request_context(content)

    def test_empty_string(self) -> None:
        """Test empty string handling."""
        assert not _parse_request_context("")

    def test_invalid_json(self) -> None:
        """Test invalid JSON input."""
        assert not _parse_request_context("not json")


# ---------------------------------------------------------------------------
# _extract_question_title
# ---------------------------------------------------------------------------


class TestExtractQuestionTitle:
    """Tests for _extract_question_title."""

    def test_simple(self) -> None:
        """Test simple question title extraction."""
        assert _extract_question_title("Will X happen?") == "Will X happen?"

    def test_with_separator(self) -> None:
        """Test question title with separator."""
        raw = f"Will X happen?{QUESTION_DATA_SEPARATOR}extra data"
        assert _extract_question_title(raw) == "Will X happen?"

    def test_empty(self) -> None:
        """Test empty string extraction."""
        assert _extract_question_title("") == ""

    def test_none(self) -> None:
        """Test None input extraction."""
        assert _extract_question_title(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_row
# ---------------------------------------------------------------------------


class TestBuildRow:
    """Tests for build_row."""

    def test_full_row(self) -> None:
        """Test building a full row with all fields."""
        # Use realistic timestamps: delivery at March 28, resolution at March 30
        delivery_ts = 1774900000  # ~2026-03-28
        request_ts = delivery_ts - 50  # 50 seconds earlier
        resolved_ts = delivery_ts + 2 * 86400  # 2 days later

        delivery = {
            "deliver_id": "0xabc",
            "timestamp": delivery_ts,
            "request_timestamp": request_ts,
            "model": "gpt-4.1",
            "tool_response": '{"p_yes": 0.8, "p_no": 0.2, "confidence": 0.9}',
            "tool": "superforcaster",
            "question_title": "Will Bitcoin hit $100k?",
            "market_id": "0xmarket",
            "market_prob": 0.65,
            "market_liquidity_usd": 1000.0,
            "market_close_at": "2026-04-01T00:00:00Z",
        }
        market = {"outcome": True, "resolved_at_ts": resolved_ts}
        row = build_row(delivery, market, 1.0, "omen")

        assert row["schema_version"] == "1.0"
        assert row["mode"] == "production_replay"
        assert row["platform"] == "omen"
        assert row["p_yes"] == 0.8
        assert row["final_outcome"] is True
        assert row["latency_s"] == 50
        assert row["prediction_lead_time_days"] == 2.0
        assert row["market_id"] == "0xmarket"
        assert row["market_prob_at_prediction"] == 0.65
        assert row["category"] == "finance"

    def test_missing_request_timestamp(self) -> None:
        """Test row with missing request timestamp."""
        delivery = {
            "deliver_id": "0xdef",
            "timestamp": 1000,
            "request_timestamp": None,
            "model": "gpt-4.1",
            "tool_response": '{"p_yes": 0.5, "p_no": 0.5}',
            "tool": "test-tool",
            "question_title": "Will something happen?",
            "market_id": None,
            "market_prob": None,
            "market_liquidity_usd": None,
            "market_close_at": None,
        }
        market = {"outcome": False, "resolved_at_ts": 2000}
        row = build_row(delivery, market, 1.0, "polymarket")
        assert row["latency_s"] is None
        assert row["requested_at"] is None

    def test_tool_hash_from_delivery_sets_version_and_config(self) -> None:
        """Test that the tool hash from delivery sets version and config.

        A delivery-level tool_hash (ParsedDelivery.toolHash) populates
        tool_version and config_hash without IPFS metadata.
        """
        delivery = {
            "deliver_id": "0xghi",
            "timestamp": 1000,
            "request_timestamp": None,
            "model": "gpt-4.1",
            "tool_response": '{"p_yes": 0.5, "p_no": 0.5}',
            "tool_hash": "bafytoolhash",
            "tool": "test-tool",
            "question_title": "Will something happen?",
            "market_id": None,
            "market_prob": None,
            "market_liquidity_usd": None,
            "market_close_at": None,
        }
        market = {"outcome": False, "resolved_at_ts": 2000}
        row = build_row(delivery, market, 1.0, "omen")
        assert row["tool_version"] == "bafytoolhash"
        assert row["config_hash"] == _compute_config_hash("bafytoolhash", "gpt-4.1")

    def test_tool_hash_takes_priority_over_ipfs_metadata(self) -> None:
        """The exact per-delivery hash wins over the sampled IPFS metadata."""
        delivery = {
            "deliver_id": "0xjkl",
            "timestamp": 1000,
            "request_timestamp": None,
            "model": "gpt-4.1",
            "tool_response": '{"p_yes": 0.5, "p_no": 0.5}',
            "tool_hash": "bafyexact",
            "tool": "test-tool",
            "question_title": "Will something happen?",
            "market_id": None,
            "market_prob": None,
            "market_liquidity_usd": None,
            "market_close_at": None,
        }
        market = {"outcome": False, "resolved_at_ts": 2000}
        row = build_row(
            delivery, market, 1.0, "omen", ipfs_metadata={"tool_hash": "bafysampled"}
        )
        assert row["tool_version"] == "bafyexact"


# ---------------------------------------------------------------------------
# IPFS metadata enrichment
# ---------------------------------------------------------------------------


class TestComputeConfigHash:
    """Tests for _compute_config_hash."""

    def test_deterministic(self) -> None:
        """Test config hash is deterministic."""
        h1 = _compute_config_hash("bafyabc", "gpt-4.1", 0.7, 4096)
        h2 = _compute_config_hash("bafyabc", "gpt-4.1", 0.7, 4096)
        assert h1 == h2

    def test_differs_on_model_change(self) -> None:
        """Test config hash differs on model change."""
        h1 = _compute_config_hash("bafyabc", "gpt-4.1")
        h2 = _compute_config_hash("bafyabc", "gpt-4o")
        assert h1 != h2

    def test_differs_on_tool_hash_change(self) -> None:
        """Test config hash differs on tool hash change."""
        h1 = _compute_config_hash("bafyabc", "gpt-4.1")
        h2 = _compute_config_hash("bafydef", "gpt-4.1")
        assert h1 != h2

    def test_none_inputs_returns_none(self) -> None:
        """Test None inputs return None."""
        assert _compute_config_hash(None, None) is None

    def test_zero_temperature_not_treated_as_none(self) -> None:
        """temperature=0.0 is valid and must differ from temperature=None."""
        h_zero = _compute_config_hash("bafyabc", "gpt-4.1", 0.0, 4096)
        h_none = _compute_config_hash("bafyabc", "gpt-4.1", None, 4096)
        assert h_zero != h_none

    def test_zero_max_tokens_not_treated_as_none(self) -> None:
        """max_tokens=0 is valid and must differ from max_tokens=None."""
        h_zero = _compute_config_hash("bafyabc", "gpt-4.1", 0.7, 0)
        h_none = _compute_config_hash("bafyabc", "gpt-4.1", 0.7, None)
        assert h_zero != h_none


class TestBuildRowWithMetadata:
    """Tests for build_row with IPFS metadata."""

    def _delivery(self) -> dict[str, Any]:
        return {
            "deliver_id": "0xabc",
            "timestamp": 1774900000,
            "request_timestamp": None,
            "model": "gpt-4.1",
            "tool_response": '{"p_yes": 0.8, "p_no": 0.2, "confidence": 0.9}',
            "tool": "superforcaster",
            "question_title": "Will BTC hit $100k?",
            "market_id": None,
            "market_prob": None,
            "market_liquidity_usd": None,
            "market_close_at": None,
        }

    def test_with_metadata(self) -> None:
        """Test build_row with IPFS metadata."""
        metadata = {
            "tool_hash": "bafyabc123",
            "params": {"temperature": 0.7, "max_tokens": 4096},
        }
        row = build_row(
            self._delivery(),
            {"outcome": True, "resolved_at_ts": 1774900000 + 86400},
            1.0,
            "omen",
            ipfs_metadata=metadata,
        )
        assert row["tool_version"] == "bafyabc123"
        assert row["config_hash"] is not None
        assert len(row["config_hash"]) == 12
        assert "prompt_template" not in row

    def test_without_metadata(self) -> None:
        """Test build_row without IPFS metadata."""
        row = build_row(
            self._delivery(),
            {"outcome": True, "resolved_at_ts": 1774900000 + 86400},
            1.0,
            "omen",
        )
        assert row["tool_version"] is None
        assert row["config_hash"] is None
        assert "prompt_template" not in row


# ---------------------------------------------------------------------------
# ResolvedMarkets
# ---------------------------------------------------------------------------


class TestResolvedMarkets:
    """Tests for ResolvedMarkets."""

    def test_len_counts_by_title(self) -> None:
        """Test len counts markets by title."""
        m = ResolvedMarkets()
        m.add("0x1", "Question A", {"outcome": True})
        m.add("0x2", "Question B", {"outcome": False})
        assert len(m) == 2

    def test_bool_true_when_populated(self) -> None:
        """Test bool is True when populated."""
        m = ResolvedMarkets()
        assert not m
        m.add(None, "Question A", {"outcome": True})
        assert m

    def test_add_with_id_only(self) -> None:
        """Test adding market with ID only."""
        m = ResolvedMarkets()
        m.add("0x1", "", {"outcome": True})
        # title is empty so by_title is empty, but by_id has an entry
        assert "0x1" in m.by_id
        assert len(m.by_title) == 0


# ---------------------------------------------------------------------------
# Incremental state & deduplication
# ---------------------------------------------------------------------------


class TestIncrementalState:
    """Tests for incremental state persistence."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Test state save and load round trip."""
        state_path = tmp_path / ".fetch_state.json"
        state = {
            "omen": {
                "last_delivery_timestamp": 12345,
                "last_resolved_timestamp": 12000,
                "last_run": "2026-03-31T00:00:00Z",
            }
        }
        save_fetch_state(state_path, state)
        loaded = load_fetch_state(state_path)
        assert loaded == state

    def test_missing_file(self, tmp_path: Path) -> None:
        """Test loading from missing file."""
        assert load_fetch_state(tmp_path / "nonexistent.json") == {}

    def test_corrupt_file(self, tmp_path: Path) -> None:
        """Test loading from corrupt file."""
        state_path = tmp_path / ".fetch_state.json"
        state_path.write_text("not json")
        assert load_fetch_state(state_path) == {}


class TestDeduplication:
    """Tests for row deduplication."""

    def test_load_ids_from_file(self, tmp_path: Path) -> None:
        """Test loading row IDs from file."""
        log_path = tmp_path / "log.jsonl"
        log_path.write_text(
            '{"row_id": "a"}\n' + '{"row_id": "b"}\n' + '{"row_id": "c"}\n'
        )
        ids = _load_ids_from_file(log_path)
        assert ids == {"a", "b", "c"}

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test loading from empty file."""
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("")
        assert _load_ids_from_file(log_path) == set()

    def test_missing_file(self, tmp_path: Path) -> None:
        """Test loading from missing file."""
        assert _load_ids_from_file(tmp_path / "nope.jsonl") == set()

    def test_row_id_deterministic(self) -> None:
        """Test row ID generation is deterministic."""
        id1 = _make_row_id("omen", "0xabc")
        id2 = _make_row_id("omen", "0xabc")
        id3 = _make_row_id("polymarket", "0xabc")
        assert id1 == id2
        assert id1 != id3


# ---------------------------------------------------------------------------
# Daily log rotation
# ---------------------------------------------------------------------------


class TestDailyLogPath:
    """Tests for daily_log_path helper."""

    def test_returns_dated_filename(self) -> None:
        """Test returns dated filename."""

        d = datetime(2026, 4, 6, 12, 0, 0, tzinfo=timezone.utc)
        result = daily_log_path(Path("/tmp/logs"), d)
        assert result == Path("/tmp/logs/production_log_2026_04_06.jsonl")

    def test_defaults_to_today(self) -> None:
        """Test defaults to today date."""

        today = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        result = daily_log_path(Path("/tmp/logs"))
        assert result.name == f"production_log_{today}.jsonl"


class TestDailyLogRotation:
    """Tests for daily log file writing and dedup scoping."""

    def test_writes_to_dated_file(self, tmp_path: Path) -> None:
        """Rows are written to today's dated log file."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        output = daily_log_path(logs_dir)
        rows = [{"row_id": "r1", "data": "x"}, {"row_id": "r2", "data": "y"}]
        append_rows(output, rows)
        assert output.exists()
        lines = [json.loads(ln) for ln in output.read_text().strip().split("\n")]
        assert len(lines) == 2
        assert {ln["row_id"] for ln in lines} == {"r1", "r2"}

    def test_same_day_appends(self, tmp_path: Path) -> None:
        """Two writes on the same day append to the same file."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        output = daily_log_path(logs_dir)
        append_rows(output, [{"row_id": "r1"}])
        append_rows(output, [{"row_id": "r2"}])
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_dedup_reads_todays_file_only(self, tmp_path: Path) -> None:
        """Normal dedup (state_loss=False) reads only today's file."""

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        # Write to yesterday's file
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        old_path = daily_log_path(logs_dir, yesterday)
        old_path.write_text('{"row_id": "old_row"}\n')
        # Write to today's file
        today_path = daily_log_path(logs_dir)
        today_path.write_text('{"row_id": "today_row"}\n')

        ids = load_existing_row_ids(logs_dir, state_loss=False)
        assert "today_row" in ids
        assert "old_row" not in ids

    def test_dedup_reads_7_days_on_state_loss(self, tmp_path: Path) -> None:
        """State-loss recovery reads the last DEDUP_LOOKBACK_DAYS files."""

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        now = datetime.now(timezone.utc)

        # Write files for days 0..8 (today through 8 days ago)
        for i in range(9):
            d = now - timedelta(days=i)
            p = daily_log_path(logs_dir, d)
            p.write_text(f'{{"row_id": "row_day_{i}"}}\n')

        ids = load_existing_row_ids(logs_dir, state_loss=True)
        # Days 0-6 (within DEDUP_LOOKBACK_DAYS=7) should be included
        for i in range(DEDUP_LOOKBACK_DAYS):
            assert f"row_day_{i}" in ids
        # Day 8 is outside the window
        assert "row_day_8" not in ids

    def test_dedup_empty_logs_dir(self, tmp_path: Path) -> None:
        """Empty logs dir returns empty set."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        assert load_existing_row_ids(logs_dir) == set()
        assert load_existing_row_ids(logs_dir, state_loss=True) == set()


class TestMigration:
    """Tests for _migrate_legacy_log."""

    def test_legacy_file_moved(self, tmp_path: Path) -> None:
        """Old production_log.jsonl is moved to logs/production_log_legacy.jsonl."""
        legacy = tmp_path / "production_log.jsonl"
        legacy.write_text('{"row_id": "r1"}\n{"row_id": "r2"}\n')
        logs_dir = tmp_path / "logs"

        _migrate_legacy_log(legacy, logs_dir)

        assert not legacy.exists()
        dest = logs_dir / "production_log_legacy.jsonl"
        assert dest.exists()
        assert dest.read_text() == '{"row_id": "r1"}\n{"row_id": "r2"}\n'

    def test_no_legacy_file_noop(self, tmp_path: Path) -> None:
        """No crash when legacy file doesn't exist."""
        logs_dir = tmp_path / "logs"
        _migrate_legacy_log(tmp_path / "production_log.jsonl", logs_dir)
        assert not logs_dir.exists()  # logs dir not created unnecessarily

    def test_legacy_empty_file_moved(self, tmp_path: Path) -> None:
        """Empty legacy file is still moved."""
        legacy = tmp_path / "production_log.jsonl"
        legacy.write_text("")
        logs_dir = tmp_path / "logs"

        _migrate_legacy_log(legacy, logs_dir)

        assert not legacy.exists()
        assert (logs_dir / "production_log_legacy.jsonl").exists()


# ---------------------------------------------------------------------------
# Pending deliveries (_match_and_build)
# ---------------------------------------------------------------------------


def _make_delivery(
    deliver_id: str = "0xabc",
    question_title: str = "Will Bitcoin hit $100k by June?",
    market_id: str | None = None,
    timestamp: int | None = None,
) -> dict[str, Any]:
    """Build a minimal delivery dict for testing."""
    return {
        "deliver_id": deliver_id,
        "timestamp": timestamp or int(time.time()),
        "request_timestamp": None,
        "model": "gpt-4.1",
        "tool_response": '{"p_yes": 0.7, "p_no": 0.3, "confidence": 0.8}',
        "tool": "superforcaster",
        "question_title": question_title,
        "market_id": market_id,
        "market_prob": None,
        "market_liquidity_usd": None,
        "market_close_at": None,
    }


class TestMatchAndBuild:
    """Tests for _match_and_build and pending delivery logic."""

    def test_matched_delivery_becomes_row(self) -> None:
        """Test matched delivery becomes a row."""
        markets = ResolvedMarkets()
        markets.add(
            "0xm1",
            "Will Bitcoin hit $100k by June?",
            {"outcome": True, "resolved_at_ts": 2000},
        )
        deliveries = [_make_delivery(deliver_id="0xd1")]

        rows, pending, _, _, _, _ = _match_and_build(deliveries, markets, set(), "omen")
        assert len(rows) == 1
        assert len(pending) == 0
        assert rows[0]["p_yes"] == 0.7

    def test_unmatched_delivery_goes_to_pending(self) -> None:
        """Test unmatched delivery goes to pending."""
        markets = ResolvedMarkets()  # empty — no resolved markets
        deliveries = [_make_delivery(deliver_id="0xd1")]

        rows, pending, _, _, _, _ = _match_and_build(deliveries, markets, set(), "omen")
        assert len(rows) == 0
        assert len(pending) == 1
        assert pending[0]["deliver_id"] == "0xd1"

    def test_pending_delivery_matched_on_retry(self) -> None:
        """Simulate pending delivery matched on retry.

        Delivery created in run 1 (unmatched),
        market resolves, run 2 retries and matches.
        """
        # Run 1: no resolved markets
        deliveries = [_make_delivery(deliver_id="0xd1")]
        _, pending, _, _, _, _ = _match_and_build(
            deliveries, ResolvedMarkets(), set(), "omen"
        )
        assert len(pending) == 1

        # Run 2: market resolved
        markets = ResolvedMarkets()
        markets.add(
            None,
            "Will Bitcoin hit $100k by June?",
            {"outcome": True, "resolved_at_ts": 2000},
        )
        rows, still_pending, _, _, _, _ = _match_and_build(
            pending, markets, set(), "omen"
        )
        assert len(rows) == 1
        assert len(still_pending) == 0

    def test_already_emitted_row_not_duplicated(self) -> None:
        """If a delivery was already emitted (row_id in existing_ids), skip it."""
        markets = ResolvedMarkets()
        markets.add(
            None,
            "Will Bitcoin hit $100k by June?",
            {"outcome": True, "resolved_at_ts": 2000},
        )
        delivery = _make_delivery(deliver_id="0xd1")
        existing = {_make_row_id("omen", "0xd1")}

        rows, pending, _, _, _, _ = _match_and_build(
            [delivery], markets, existing, "omen"
        )
        assert len(rows) == 0
        assert len(pending) == 0  # not pending either — already emitted

    def test_mixed_matched_and_unmatched(self) -> None:
        """Test mixed matched and unmatched deliveries."""
        markets = ResolvedMarkets()
        markets.add(
            None,
            "Will Bitcoin hit $100k by June?",
            {"outcome": True, "resolved_at_ts": 2000},
        )
        deliveries = [
            _make_delivery(
                deliver_id="0xd1", question_title="Will Bitcoin hit $100k by June?"
            ),
            _make_delivery(deliver_id="0xd2", question_title="Will ETH hit $5k?"),
            _make_delivery(
                deliver_id="0xd3", question_title="Will Bitcoin hit $100k by June?"
            ),
        ]

        rows, pending, _, _, _, _ = _match_and_build(deliveries, markets, set(), "omen")
        assert len(rows) == 2  # d1 and d3 match
        assert len(pending) == 1  # d2 unmatched


class TestPendingAgeCap:
    """Tests for the 90-day pending delivery pruning."""

    def test_recent_delivery_kept(self) -> None:
        """Delivery from today should not be pruned."""
        now = int(time.time())
        cutoff = now - (PENDING_MAX_AGE_DAYS * 86400)
        delivery = _make_delivery(timestamp=now)
        assert delivery["timestamp"] > cutoff

    def test_old_delivery_pruned(self) -> None:
        """Delivery older than PENDING_MAX_AGE_DAYS should be pruned."""
        now = int(time.time())
        cutoff = now - (PENDING_MAX_AGE_DAYS * 86400)
        old_ts = cutoff - 86400  # 1 day older than cutoff
        delivery = _make_delivery(timestamp=old_ts)
        assert delivery["timestamp"] <= cutoff


class TestPendingInState:  # pylint: disable=too-few-public-methods
    """Tests that pending deliveries round-trip through state file."""

    def test_pending_persisted_and_loaded(self, tmp_path: Path) -> None:
        """Test pending deliveries persisted and loaded."""
        state_path = tmp_path / ".fetch_state.json"
        pending = [_make_delivery(deliver_id="0xpending1")]
        state = {
            "omen": {
                "last_delivery_timestamp": 100,
                "last_resolved_timestamp": 200,
                "pending_deliveries": pending,
                "last_run": "2026-03-31T00:00:00Z",
            }
        }
        save_fetch_state(state_path, state)
        loaded = load_fetch_state(state_path)
        loaded_pending = loaded["omen"]["pending_deliveries"]
        assert len(loaded_pending) == 1
        assert loaded_pending[0]["deliver_id"] == "0xpending1"


class TestNoScoreFlag:
    """Tests for the --no-score flag that gates the inline scorer call."""

    def test_flag_sets_attribute_true(self) -> None:
        """--no-score on the CLI sets args.no_score to True."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets.fetch_production import _build_arg_parser

        args = _build_arg_parser().parse_args(["--no-score"])
        assert args.no_score is True

    def test_default_is_false(self) -> None:
        """Omitting --no-score keeps the fast path (inline scoring) enabled."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets.fetch_production import _build_arg_parser

        args = _build_arg_parser().parse_args([])
        assert args.no_score is False

    def _stub_main_deps(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        row: dict[str, Any],
    ) -> list[tuple]:
        """Stub out fetch/state deps for main() and capture scorer calls."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        calls: list[tuple] = []

        monkeypatch.setattr(fp, "_migrate_legacy_log", lambda *a, **kw: None)
        monkeypatch.setattr(fp, "load_fetch_state", lambda *a, **kw: {})
        monkeypatch.setattr(fp, "load_existing_row_ids", lambda *a, **kw: set())
        monkeypatch.setattr(fp, "fetch_omen_resolved", lambda *a, **kw: [])
        monkeypatch.setattr(fp, "fetch_polymarket_resolved", lambda *a, **kw: [])
        # Omen returns one row, polymarket returns none.
        response_queue: list[Any] = [([row], [], 0, 0), ([], [], 0, 0)]
        monkeypatch.setattr(
            fp, "process_platform", lambda *a, **kw: response_queue.pop(0)
        )
        monkeypatch.setattr(fp, "append_rows", lambda *a, **kw: 1)
        monkeypatch.setattr(fp, "_update_platform_state", lambda *a, **kw: None)
        monkeypatch.setattr(fp, "save_fetch_state", lambda *a, **kw: None)
        monkeypatch.setattr(
            fp,
            "_run_scorer_update",
            lambda rows, scores, history: calls.append((rows, scores, history)),
        )
        return calls

    def test_main_skips_scorer_update_with_no_score(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """main(--no-score) must not call _run_scorer_update even when rows exist."""
        # pylint: disable-next=import-outside-toplevel
        import sys

        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        calls = self._stub_main_deps(
            monkeypatch, tmp_path, {"row_id": "r1", "prediction_parse_status": "valid"}
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "fetch_production",
                "--no-score",
                "--logs-dir",
                str(tmp_path),
                "--state-file",
                str(tmp_path / "state.json"),
                "--scores",
                str(tmp_path / "scores.json"),
                "--history",
                str(tmp_path / "history.jsonl"),
            ],
        )
        fp.main()
        assert not calls

    def test_main_calls_scorer_update_by_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """main() without --no-score runs the fast-path inline scorer once."""
        # pylint: disable-next=import-outside-toplevel
        import sys

        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        calls = self._stub_main_deps(
            monkeypatch, tmp_path, {"row_id": "r1", "prediction_parse_status": "valid"}
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "fetch_production",
                "--logs-dir",
                str(tmp_path),
                "--state-file",
                str(tmp_path / "state.json"),
                "--scores",
                str(tmp_path / "scores.json"),
                "--history",
                str(tmp_path / "history.jsonl"),
            ],
        )
        fp.main()
        assert len(calls) == 1
        rows, _scores, _history = calls[0]
        assert rows == [{"row_id": "r1", "prediction_parse_status": "valid"}]


# ---------------------------------------------------------------------------
# parsedDelivery schema migration (autonolas-subgraph PR #113)
# ---------------------------------------------------------------------------


@pytest.fixture(name="clear_schema_cache", autouse=False)
def _clear_schema_cache_fixture() -> Any:
    """Reset the per-URL delivers-schema cache around a test."""
    # pylint: disable-next=import-outside-toplevel
    from benchmark.datasets import fetch_production as fp

    fp._DELIVERS_SCHEMA_CACHE.clear()  # pylint: disable=protected-access
    yield
    fp._DELIVERS_SCHEMA_CACHE.clear()  # pylint: disable=protected-access


class TestExtractDeliveryFields:
    """Tests for extract_delivery_fields (both schema shapes)."""

    @pytest.mark.parametrize(
        ("deliver", "schema", "expected"),
        [
            # New schema: fields relocate to the nested ParsedDelivery entity.
            (
                {
                    "id": "0x1",
                    "parsedDelivery": {
                        "response": '{"p_yes": 0.8, "p_no": 0.2}',
                        "model": "gpt-4.1",
                        "tool": "superforcaster",
                        "toolHash": "bafytool",
                    },
                },
                DELIVERS_SCHEMA_PARSED,
                {
                    "model": "gpt-4.1",
                    "tool_response": '{"p_yes": 0.8, "p_no": 0.2}',
                    "tool_hash": "bafytool",
                    "parsed_missing": False,
                },
            ),
            # New schema: sentinel model/toolHash normalize to None; the
            # sentinel response passes through for parse_tool_response.
            (
                {
                    "id": "0x2",
                    "parsedDelivery": {
                        "response": SUBGRAPH_UNHANDLED_TYPE,
                        "model": SUBGRAPH_UNHANDLED_TYPE,
                        "tool": SUBGRAPH_UNHANDLED_TYPE,
                        "toolHash": SUBGRAPH_UNHANDLED_TYPE,
                    },
                },
                DELIVERS_SCHEMA_PARSED,
                {
                    "model": None,
                    "tool_response": SUBGRAPH_UNHANDLED_TYPE,
                    "tool_hash": None,
                    "parsed_missing": False,
                },
            ),
            # New schema: ParsedDelivery not yet indexed (async FDS lag).
            (
                {"id": "0x3", "parsedDelivery": None},
                DELIVERS_SCHEMA_PARSED,
                {
                    "model": None,
                    "tool_response": None,
                    "tool_hash": None,
                    "parsed_missing": True,
                },
            ),
            # Legacy schema: flat fields, no tool_hash, never "missing".
            (
                {
                    "id": "0x4",
                    "model": "gpt-4.1",
                    "toolResponse": '{"p_yes": 0.1, "p_no": 0.9}',
                },
                DELIVERS_SCHEMA_LEGACY,
                {
                    "model": "gpt-4.1",
                    "tool_response": '{"p_yes": 0.1, "p_no": 0.9}',
                    "tool_hash": None,
                    "parsed_missing": False,
                },
            ),
            # Legacy schema: sentinel model normalizes to None here too.
            (
                {"id": "0x5", "model": SUBGRAPH_UNHANDLED_TYPE, "toolResponse": None},
                DELIVERS_SCHEMA_LEGACY,
                {
                    "model": None,
                    "tool_response": None,
                    "tool_hash": None,
                    "parsed_missing": False,
                },
            ),
        ],
        ids=[
            "parsed_full",
            "parsed_sentinels",
            "parsed_not_indexed",
            "legacy_full",
            "legacy_sentinel_model",
        ],
    )
    def test_extraction(
        self, deliver: dict[str, Any], schema: str, expected: dict[str, Any]
    ) -> None:
        """Both shapes map into the same internal delivery keys."""
        assert extract_delivery_fields(deliver, schema) == expected


@pytest.mark.usefixtures("clear_schema_cache")
class TestDetectDeliversSchema:
    """Tests for the per-endpoint schema probe with legacy fallback."""

    def test_parsed_schema_when_probe_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful parsedDelivery probe selects the new shape."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        monkeypatch.setattr(fp, "_post_graphql", lambda url, payload: {"delivers": []})
        assert detect_delivers_schema("http://gnosis") == DELIVERS_SCHEMA_PARSED

    def test_legacy_fallback_on_unknown_field_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unknown-field validation error falls back to the legacy shape."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        def fail(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError(
                "GraphQL errors from http://polygon: Type `Deliver` "
                "has no field `parsedDelivery`"
            )

        monkeypatch.setattr(fp, "_post_graphql", fail)
        assert detect_delivers_schema("http://polygon") == DELIVERS_SCHEMA_LEGACY

    def test_other_graphql_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-validation errors are not mistaken for a legacy schema."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        def fail(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("GraphQL errors from http://gnosis: store error")

        monkeypatch.setattr(fp, "_post_graphql", fail)
        with pytest.raises(RuntimeError, match="store error"):
            detect_delivers_schema("http://gnosis")

    def test_result_is_cached_per_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe runs once per endpoint per process."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        calls: list[str] = []

        def probe(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            calls.append(url)
            return {"delivers": []}

        monkeypatch.setattr(fp, "_post_graphql", probe)
        detect_delivers_schema("http://gnosis")
        detect_delivers_schema("http://gnosis")
        assert len(calls) == 1


@pytest.mark.usefixtures("clear_schema_cache")
class TestFetchDeliveriesSchemaShapes:
    """fetch_deliveries selects the query by detected schema."""

    @staticmethod
    def _raw_deliver(parsed_delivery: Any) -> dict[str, Any]:
        """Build a raw subgraph deliver dict with the given parsedDelivery."""
        return {
            "id": "0xdeliver",
            "blockTimestamp": "1700000100",
            "parsedDelivery": parsed_delivery,
            "request": {
                "id": "0xreq",
                "blockTimestamp": "1700000000",
                "parsedRequest": {
                    "questionTitle": "Will X happen?",
                    "tool": "superforcaster",
                    "content": "",
                },
            },
        }

    def test_parsed_schema_maps_nested_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """parsedDelivery.response/model/toolHash land on the delivery dict."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        captured: dict[str, Any] = {}

        def fake_paginated(
            url: str,
            query: str,
            entity_key: str,
            template_vars: dict[str, Any],
            **_kwargs: Any,
        ) -> list[dict[str, Any]]:
            captured["query"] = query
            return [
                self._raw_deliver(
                    {
                        "response": '{"p_yes": 0.7, "p_no": 0.3}',
                        "model": "gpt-4.1",
                        "tool": "superforcaster",
                        "toolHash": "bafytool",
                    }
                )
            ]

        monkeypatch.setattr(
            fp, "detect_delivers_schema", lambda url: DELIVERS_SCHEMA_PARSED
        )
        monkeypatch.setattr(fp, "_paginated_fetch", fake_paginated)

        deliveries, unparsed_cap = fp.fetch_deliveries("http://gnosis", 0)
        assert unparsed_cap is None
        assert "parsedDelivery" in captured["query"]
        assert "toolResponse" not in captured["query"]
        assert deliveries[0]["tool_response"] == '{"p_yes": 0.7, "p_no": 0.3}'
        assert deliveries[0]["model"] == "gpt-4.1"
        assert deliveries[0]["tool_hash"] == "bafytool"
        assert deliveries[0]["parsed_missing"] is False

    def test_legacy_schema_maps_flat_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The legacy query keeps working against the polygon subgraph."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        captured: dict[str, Any] = {}

        def fake_paginated(
            url: str,
            query: str,
            entity_key: str,
            template_vars: dict[str, Any],
            **_kwargs: Any,
        ) -> list[dict[str, Any]]:
            captured["query"] = query
            raw = self._raw_deliver(None)
            del raw["parsedDelivery"]
            raw["model"] = "gpt-4.1"
            raw["toolResponse"] = '{"p_yes": 0.2, "p_no": 0.8}'
            return [raw]

        monkeypatch.setattr(
            fp, "detect_delivers_schema", lambda url: DELIVERS_SCHEMA_LEGACY
        )
        monkeypatch.setattr(fp, "_paginated_fetch", fake_paginated)

        deliveries, _ = fp.fetch_deliveries("http://polygon", 0)
        assert "toolResponse" in captured["query"]
        assert "parsedDelivery" not in captured["query"]
        assert deliveries[0]["tool_response"] == '{"p_yes": 0.2, "p_no": 0.8}'
        assert deliveries[0]["tool_hash"] is None


class TestDeferUnparsedDeliveries:
    """Test that Deliveries without an indexed ParsedDelivery defer.

    Deliveries without an indexed ParsedDelivery defer instead of
    becoming permanently-invalid rows.
    """

    @staticmethod
    def _delivery(ts: int, parsed_missing: bool) -> dict[str, Any]:
        """Build a matched-able delivery dict."""
        return {
            "deliver_id": f"0x{ts}",
            "timestamp": ts,
            "request_timestamp": ts - 10,
            "model": None,
            "tool_response": None,
            "tool": "superforcaster",
            "question_title": "Will the parsed delivery arrive?",
            "market_id": "0xmarket",
            "market_prob": None,
            "market_liquidity_usd": None,
            "market_close_at": None,
            "parsed_missing": parsed_missing,
        }

    @staticmethod
    def _markets() -> ResolvedMarkets:
        """One resolved market matching the delivery by id."""
        markets = ResolvedMarkets()
        markets.add(
            "0xmarket",
            "will the parsed delivery arrive?",
            {"outcome": True, "resolved_at_ts": int(time.time())},
        )
        return markets

    @pytest.mark.parametrize(
        ("age_seconds", "parsed_missing", "expect_deferred"),
        [
            (3600, True, True),  # recent + unindexed -> wait for the FDS
            (PARSED_DELIVERY_GRACE_SECONDS + 3600, True, False),  # grace expired
            (3600, False, False),  # legacy null response -> build immediately
        ],
        ids=["recent_unindexed", "grace_expired", "legacy_null"],
    )
    def test_deferral(
        self, age_seconds: int, parsed_missing: bool, expect_deferred: bool
    ) -> None:
        """Matched-but-unindexed deliveries defer only within the grace period."""
        delivery = self._delivery(int(time.time()) - age_seconds, parsed_missing)
        rows, still_pending, *_ = _match_and_build(
            [delivery], self._markets(), set(), "omen"
        )
        if expect_deferred:
            assert not rows
            # The deferred entry snapshots the matched market so it can
            # still build once the market leaves the resolution cursor.
            assert len(still_pending) == 1
            snapshot = still_pending[0]["deferred_market"]
            assert snapshot["outcome"] is True
            assert snapshot["match_confidence"] == 1.0
        else:
            assert len(rows) == 1
            assert rows[0]["prediction_parse_status"] == "missing_fields"
            assert not still_pending

    def test_deferred_entry_builds_after_market_leaves_cursor(self) -> None:
        """Two-run strand scenario: deferred in run 1, market no longer
        fetchable in run 2 — the snapshot still produces a row."""
        now = int(time.time())
        delivery = self._delivery(now - 3600, True)

        # Run 1: matched but unindexed -> deferred with market snapshot.
        rows, still_pending, *_ = _match_and_build(
            [delivery], self._markets(), set(), "omen"
        )
        assert not rows
        deferred = still_pending[0]

        # Run 2: the refresh healed the parsed fields; the resolved-markets
        # cursor has advanced, so the market is gone from the fetch.
        healed = {
            **deferred,
            "tool_response": '{"p_yes": 0.7, "p_no": 0.3}',
            "parsed_missing": False,
        }
        rows, still_pending, *_ = _match_and_build(
            [healed], ResolvedMarkets(), set(), "omen"
        )
        assert not still_pending
        assert len(rows) == 1
        assert rows[0]["p_yes"] == 0.7
        assert rows[0]["final_outcome"] is True
        assert rows[0]["prediction_parse_status"] == "valid"

    def test_deferred_entry_expires_after_market_leaves_cursor(self) -> None:
        """A never-healed deferred entry still surfaces as missing_fields
        once grace expires, even without a fetchable market."""
        now = int(time.time())
        expired = {
            **self._delivery(now - PARSED_DELIVERY_GRACE_SECONDS - 3600, True),
            "deferred_market": {
                "outcome": False,
                "resolved_at_ts": now - PARSED_DELIVERY_GRACE_SECONDS,
                "match_confidence": 0.8,
            },
        }
        rows, still_pending, *_ = _match_and_build(
            [expired], ResolvedMarkets(), set(), "omen"
        )
        assert not still_pending
        assert len(rows) == 1
        assert rows[0]["prediction_parse_status"] == "missing_fields"
        assert rows[0]["final_outcome"] is False
        assert rows[0]["match_confidence"] == 0.8

    def test_deferred_entry_redefers_within_grace(self) -> None:
        """Still-unindexed, still-recent entries keep waiting (with their
        snapshot) when the market is no longer fetchable."""
        now = int(time.time())
        deferred = {
            **self._delivery(now - 3600, True),
            "deferred_market": {
                "outcome": True,
                "resolved_at_ts": now,
                "match_confidence": 1.0,
            },
        }
        rows, still_pending, *_ = _match_and_build(
            [deferred], ResolvedMarkets(), set(), "omen"
        )
        assert not rows
        assert len(still_pending) == 1
        assert still_pending[0]["deferred_market"]["outcome"] is True

    def test_should_defer_unparsed_explicit_now(self) -> None:
        """_should_defer_unparsed honors the injected clock."""
        delivery = self._delivery(1000, True)
        assert _should_defer_unparsed(delivery, now=1000 + 60) is True
        assert (
            _should_defer_unparsed(
                delivery, now=1000 + PARSED_DELIVERY_GRACE_SECONDS + 1
            )
            is False
        )


@pytest.mark.usefixtures("clear_schema_cache")
class TestRefreshUnparsedPending:
    """Pending entries missing a tool response get re-read from the subgraph."""

    @staticmethod
    def _pending(deliver_id: str, tool_response: Any) -> dict[str, Any]:
        """Build a minimal pending-store entry."""
        return {
            "deliver_id": deliver_id,
            "timestamp": 1000,
            "tool_response": tool_response,
            "model": None,
            "tool": "superforcaster",
            "question_title": "q",
            "parsed_missing": tool_response is None,
        }

    def test_refreshes_stale_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Null-response entries pick up the now-indexed ParsedDelivery."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        monkeypatch.setattr(
            fp, "detect_delivers_schema", lambda url: DELIVERS_SCHEMA_PARSED
        )

        def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            assert '"0xstale"' in payload["query"]
            return {
                "delivers": [
                    {
                        "id": "0xstale",
                        "parsedDelivery": {
                            "response": '{"p_yes": 0.6, "p_no": 0.4}',
                            "model": "gpt-4.1",
                            "tool": "superforcaster",
                            "toolHash": "bafytool",
                        },
                    }
                ]
            }

        monkeypatch.setattr(fp, "_post_graphql", fake_post)

        stale = self._pending("0xstale", None)
        fresh = self._pending("0xfresh", '{"p_yes": 0.9, "p_no": 0.1}')
        refreshed = refresh_unparsed_pending([stale, fresh], "http://gnosis")

        by_id = {d["deliver_id"]: d for d in refreshed}
        assert by_id["0xstale"]["tool_response"] == '{"p_yes": 0.6, "p_no": 0.4}'
        assert by_id["0xstale"]["tool_hash"] == "bafytool"
        assert by_id["0xstale"]["parsed_missing"] is False
        # Untouched entries are passed through unchanged (and not mutated).
        assert by_id["0xfresh"] is fresh
        assert stale["tool_response"] is None  # original dict not mutated

    def test_still_unindexed_entries_stay_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Entries whose ParsedDelivery is still absent are kept as-is."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        monkeypatch.setattr(
            fp, "detect_delivers_schema", lambda url: DELIVERS_SCHEMA_PARSED
        )
        monkeypatch.setattr(
            fp,
            "_post_graphql",
            lambda url, payload: {
                "delivers": [{"id": "0xstale", "parsedDelivery": None}]
            },
        )
        stale = self._pending("0xstale", None)
        refreshed = refresh_unparsed_pending([stale], "http://gnosis")
        assert refreshed == [stale]

    def test_noop_on_legacy_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legacy endpoints have nothing to refresh from."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        monkeypatch.setattr(
            fp, "detect_delivers_schema", lambda url: DELIVERS_SCHEMA_LEGACY
        )

        def boom(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("must not query on legacy schema")

        monkeypatch.setattr(fp, "_post_graphql", boom)
        pending = [self._pending("0xstale", None)]
        assert refresh_unparsed_pending(pending, "http://polygon") == pending

    def test_noop_without_stale_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No probe and no query when every entry already has a response."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        def boom(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("must not query without stale entries")

        monkeypatch.setattr(fp, "_post_graphql", boom)
        pending = [self._pending("0xfresh", '{"p_yes": 0.9, "p_no": 0.1}')]
        assert refresh_unparsed_pending(pending, "http://gnosis") == pending


class TestDedupPending:
    """Pending deliveries deduplicate by deliver_id, keeping the freshest."""

    def test_last_copy_wins(self) -> None:
        """Refetched copies of the same delivery replace stale snapshots."""
        stale = {"deliver_id": "0xdup", "tool_response": None}
        fresh = {"deliver_id": "0xdup", "tool_response": '{"p_yes": 0.5}'}
        other = {"deliver_id": "0xother", "tool_response": None}
        result = _dedup_pending([stale, other, fresh])
        assert len(result) == 2
        by_id = {d["deliver_id"]: d for d in result}
        assert by_id["0xdup"] is fresh
        assert by_id["0xother"] is other

    def test_deferred_market_snapshot_survives_refetch(self) -> None:
        """A refetched copy (no snapshot) must not clobber the deferred
        market snapshot — the market is no longer re-matchable."""
        snapshot = {"outcome": True, "resolved_at_ts": 123, "match_confidence": 1.0}
        deferred = {
            "deliver_id": "0xdup",
            "tool_response": None,
            "deferred_market": snapshot,
        }
        refetched = {
            "deliver_id": "0xdup",
            "tool_response": '{"p_yes": 0.5, "p_no": 0.5}',
        }
        result = _dedup_pending([deferred, refetched])
        assert len(result) == 1
        merged = result[0]
        # Fresh parsed fields win; the snapshot is carried over.
        assert merged["tool_response"] == '{"p_yes": 0.5, "p_no": 0.5}'
        assert merged["deferred_market"] == snapshot
        assert refetched.get("deferred_market") is None  # input not mutated


@pytest.mark.usefixtures("clear_schema_cache")
class TestUnparsedRequestCursorCap:
    """Delivers skipped for a lagging parsedRequest hold the delivery cursor."""

    @staticmethod
    def _raw_deliver(ts: int, parsed_request: Any) -> dict[str, Any]:
        """Build a raw subgraph deliver with the given parsedRequest."""
        return {
            "id": f"0x{ts}",
            "blockTimestamp": str(ts),
            "parsedDelivery": None,
            "request": {
                "id": "0xreq",
                "blockTimestamp": str(ts - 10),
                "parsedRequest": parsed_request,
            },
        }

    @pytest.mark.parametrize(
        ("age_seconds", "expect_cap"),
        [
            (3600, True),  # recent skip: FDS lag -> hold the cursor
            (PARSED_DELIVERY_GRACE_SECONDS + 3600, False),  # permanent skip
        ],
        ids=["recent_lag", "permanently_unparseable"],
    )
    def test_null_parsed_request_caps_cursor_within_grace(
        self, monkeypatch: pytest.MonkeyPatch, age_seconds: int, expect_cap: bool
    ) -> None:
        """Only grace-fresh null-parsedRequest delivers return a cursor cap."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        ts = int(time.time()) - age_seconds
        monkeypatch.setattr(
            fp, "detect_delivers_schema", lambda url: DELIVERS_SCHEMA_PARSED
        )
        monkeypatch.setattr(
            fp,
            "_paginated_fetch",
            lambda *a, **k: [self._raw_deliver(ts, None)],
        )
        deliveries, cap = fp.fetch_deliveries("http://gnosis", 0)
        assert deliveries == []
        assert cap == (ts - 1 if expect_cap else None)

    def test_cap_holds_platform_delivery_cursor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cap wins over a newer built row's delivery timestamp."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        now = int(time.time())
        matched = {
            "deliver_id": "0xmatched",
            "timestamp": now,  # newer than the lagging deliver
            "request_timestamp": now - 10,
            "model": "gpt-4.1",
            "tool_response": '{"p_yes": 0.5, "p_no": 0.5}',
            "tool": "superforcaster",
            "question_title": "will it match?",
            "market_id": "0xmarket",
            "market_prob": None,
            "market_liquidity_usd": None,
            "market_close_at": None,
            "parsed_missing": False,
        }
        markets = ResolvedMarkets()
        markets.add(
            "0xmarket", "will it match?", {"outcome": True, "resolved_at_ts": now}
        )
        lagging_cap = now - 600
        monkeypatch.setattr(
            fp, "fetch_deliveries", lambda url, ts: ([matched], lagging_cap)
        )
        monkeypatch.setattr(
            fp, "_enrich_rows_with_ipfs_metadata", lambda rows, url: None
        )

        rows, _, max_delivery_ts, _ = fp.process_platform(
            "omen", "http://gnosis", markets, 0, set(), []
        )
        assert len(rows) == 1
        assert max_delivery_ts == lagging_cap


class TestPendingSurvivesQuietRun:
    """The pending store must not be wiped by a run with no resolutions."""

    def test_pending_survives_run_without_resolved_markets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run with zero newly-resolved markets must carry the pending
        store forward, not overwrite it with just the new deliveries."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        old_pending = {
            "deliver_id": "0xold",
            # Recent enough to survive the PENDING_MAX_AGE_DAYS prune.
            "timestamp": int(time.time()) - 3600,
            "tool_response": '{"p_yes": 0.5, "p_no": 0.5}',  # no refresh needed
            "question_title": "old pending question",
            "market_id": None,
        }
        new_delivery = {
            "deliver_id": "0xnew",
            "timestamp": int(time.time()),
            "request_timestamp": None,
            "model": None,
            "tool_response": None,
            "tool": "superforcaster",
            "question_title": "new unmatched question",
            "market_id": None,
            "market_prob": None,
            "market_liquidity_usd": None,
            "market_close_at": None,
            "parsed_missing": False,
        }
        monkeypatch.setattr(
            fp, "fetch_deliveries", lambda url, ts: ([new_delivery], None)
        )

        rows, all_pending, _, _ = fp.process_platform(
            "omen",
            "http://gnosis",
            ResolvedMarkets(),  # nothing resolved this run
            0,
            set(),
            [old_pending],
        )
        assert not rows
        pending_ids = {d["deliver_id"] for d in all_pending}
        assert pending_ids == {"0xold", "0xnew"}


class TestEnrichmentSkipsPrefilled:
    """IPFS enrichment skips rows already versioned via ParsedDelivery."""

    def test_all_prefilled_rows_skip_subgraph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No subgraph or IPFS traffic when every row has tool_version."""
        # pylint: disable-next=import-outside-toplevel
        from benchmark.datasets import fetch_production as fp

        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("prefilled rows must not hit the subgraph")

        monkeypatch.setattr(fp, "_fetch_delivery_info", boom)
        rows = [{"deliver_id": "0x1", "tool_version": "bafyexact", "tool_name": "t"}]
        fp._enrich_rows_with_ipfs_metadata(  # pylint: disable=protected-access
            rows, "http://gnosis"
        )
        assert rows[0]["tool_version"] == "bafyexact"
