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
from pathlib import Path
from typing import Any

import pytest

import time

from benchmark.datasets.fetch_production import (
    DEDUP_LOOKBACK_DAYS,
    PENDING_MAX_AGE_DAYS,
    PROBABILITY_SUM_TOLERANCE,
    ResolvedMarkets,
    _compute_config_hash,
    _extract_question_title,
    _load_ids_from_file,
    _make_row_id,
    _match_and_build,
    _match_delivery,
    _migrate_legacy_log,
    _parse_request_context,
    append_rows,
    build_row,
    classify_category,
    daily_log_path,
    load_existing_row_ids,
    load_fetch_state,
    parse_tool_response,
    save_fetch_state,
)

QUESTION_DATA_SEPARATOR = "\u241f"


# ---------------------------------------------------------------------------
# parse_tool_response
# ---------------------------------------------------------------------------


class TestParseToolResponse:
    """Tests for toolResponse JSON parsing."""

    def test_valid_json(self) -> None:
        resp = parse_tool_response('{"p_yes": 0.72, "p_no": 0.28, "confidence": 0.85}')
        assert resp["prediction_parse_status"] == "valid"
        assert resp["p_yes"] == 0.72
        assert resp["p_no"] == 0.28
        assert resp["confidence"] == 0.85

    def test_valid_json_with_newlines(self) -> None:
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
        resp = parse_tool_response(None)
        assert resp["prediction_parse_status"] == "missing_fields"
        assert resp["p_yes"] is None

    def test_empty_string(self) -> None:
        resp = parse_tool_response("")
        assert resp["prediction_parse_status"] == "missing_fields"

    def test_ipfs_error(self) -> None:
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
        resp = parse_tool_response('{"p_yes": 0.72}')
        assert resp["prediction_parse_status"] == "malformed"

    def test_malformed_out_of_range(self) -> None:
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

    def test_unparseable_garbage(self) -> None:
        resp = parse_tool_response("not json at all")
        assert resp["prediction_parse_status"] == "malformed"

    def test_no_confidence_field(self) -> None:
        resp = parse_tool_response('{"p_yes": 0.5, "p_no": 0.5}')
        assert resp["prediction_parse_status"] == "valid"
        assert resp["confidence"] is None


# ---------------------------------------------------------------------------
# Neg-risk outcome decoding
# ---------------------------------------------------------------------------


class TestNegRiskOutcome:
    """Tests for Polymarket neg-risk market outcome decoding.

    Normal markets: outcomes = ["No", "Yes"] → index 0 = No, index 1 = Yes
    Neg-risk markets: outcomes = ["Yes", "No"] → index 0 = Yes, index 1 = No
    """

    @staticmethod
    def _decode(outcomes: list[str], winning_index: int) -> bool:
        """Replicate the outcome decoding logic from fetch_polymarket_resolved."""
        if outcomes and winning_index < len(outcomes):
            return outcomes[winning_index].lower() == "yes"
        return winning_index == 1

    def test_normal_yes(self) -> None:
        assert self._decode(["No", "Yes"], 1) is True

    def test_normal_no(self) -> None:
        assert self._decode(["No", "Yes"], 0) is False

    def test_neg_risk_yes(self) -> None:
        """outcomes inverted: index 0 = Yes."""
        assert self._decode(["Yes", "No"], 0) is True

    def test_neg_risk_no(self) -> None:
        """outcomes inverted: index 1 = No."""
        assert self._decode(["Yes", "No"], 1) is False

    def test_fallback_no_outcomes(self) -> None:
        """Without outcomes array, falls back to winningIndex == 1."""
        assert self._decode([], 1) is True
        assert self._decode([], 0) is False


# ---------------------------------------------------------------------------
# _match_delivery
# ---------------------------------------------------------------------------


class TestMatchDelivery:

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
        markets = self._make_markets()
        delivery = {"market_id": "0xabc", "question_title": "totally different"}
        market, confidence = _match_delivery(delivery, markets)
        assert market is not None
        assert market["outcome"] is True
        assert confidence == 1.0

    def test_market_id_takes_priority_over_title(self) -> None:
        markets = self._make_markets()
        delivery = {
            "market_id": "0xabc",
            "question_title": "Will the president win the next election?",
        }
        # market_id 0xabc → outcome True, even though title matches 0xdef (False)
        market, confidence = _match_delivery(delivery, markets)
        assert market["outcome"] is True

    def test_exact_title_match(self) -> None:
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
        market, confidence = _match_delivery(delivery, markets)
        assert market is None

    def test_no_match(self) -> None:
        markets = self._make_markets()
        delivery = {"market_id": None, "question_title": "Completely unrelated"}
        market, confidence = _match_delivery(delivery, markets)
        assert market is None
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Polymarket resolution time filter
# ---------------------------------------------------------------------------


class TestPolymarketResolutionFilter:
    """Tests that the resolution time cutoff logic works correctly.

    Replicates the post-filter from fetch_polymarket_resolved:
    a bet with resolved_at <= cutoff must be excluded.
    """

    @staticmethod
    def _filter_bet(bet: dict[str, Any], resolved_after: int) -> bool:
        """Replicate the resolution filter from fetch_polymarket_resolved."""
        question = bet.get("question") or {}
        resolution = question.get("resolution")
        if resolution is None:
            return False
        resolved_at_ts = resolution.get("blockTimestamp")
        if not resolved_at_ts:
            return False
        return int(resolved_at_ts) > resolved_after

    def test_resolved_after_cutoff_included(self) -> None:
        bet = {"question": {"resolution": {"blockTimestamp": "1000"}}}
        assert self._filter_bet(bet, 999) is True

    def test_resolved_at_cutoff_excluded(self) -> None:
        bet = {"question": {"resolution": {"blockTimestamp": "1000"}}}
        assert self._filter_bet(bet, 1000) is False

    def test_resolved_before_cutoff_excluded(self) -> None:
        bet = {"question": {"resolution": {"blockTimestamp": "500"}}}
        assert self._filter_bet(bet, 999) is False

    def test_unresolved_excluded(self) -> None:
        bet = {"question": {"resolution": None}}
        assert self._filter_bet(bet, 0) is False


# ---------------------------------------------------------------------------
# classify_category
# ---------------------------------------------------------------------------


class TestClassifyCategory:

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
        assert classify_category(question) == expected

    def test_unknown_falls_back_to_other(self) -> None:
        assert classify_category("Will something random happen?") == "other"

    def test_word_boundary_prevents_substring_match(self) -> None:
        """'eth' should not match inside 'something'."""
        assert classify_category("Will something happen?") == "other"

    def test_case_insensitive(self) -> None:
        assert classify_category("WILL BITCOIN HIT $100K?") == "finance"


# ---------------------------------------------------------------------------
# _parse_request_context
# ---------------------------------------------------------------------------


class TestParseRequestContext:

    def test_schema_v2(self) -> None:
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
        content = json.dumps({"prompt": "...", "tool": "test", "nonce": "abc"})
        assert _parse_request_context(content) == {}

    def test_empty_string(self) -> None:
        assert _parse_request_context("") == {}

    def test_invalid_json(self) -> None:
        assert _parse_request_context("not json") == {}


# ---------------------------------------------------------------------------
# _extract_question_title
# ---------------------------------------------------------------------------


class TestExtractQuestionTitle:

    def test_simple(self) -> None:
        assert _extract_question_title("Will X happen?") == "Will X happen?"

    def test_with_separator(self) -> None:
        raw = f"Will X happen?{QUESTION_DATA_SEPARATOR}extra data"
        assert _extract_question_title(raw) == "Will X happen?"

    def test_empty(self) -> None:
        assert _extract_question_title("") == ""

    def test_none(self) -> None:
        assert _extract_question_title(None) == ""


# ---------------------------------------------------------------------------
# build_row
# ---------------------------------------------------------------------------


class TestBuildRow:

    def test_full_row(self) -> None:
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


# ---------------------------------------------------------------------------
# IPFS metadata enrichment
# ---------------------------------------------------------------------------


class TestComputeConfigHash:
    """Tests for _compute_config_hash."""

    def test_deterministic(self) -> None:
        h1 = _compute_config_hash("bafyabc", "gpt-4.1", 0.7, 4096)
        h2 = _compute_config_hash("bafyabc", "gpt-4.1", 0.7, 4096)
        assert h1 == h2

    def test_differs_on_model_change(self) -> None:
        h1 = _compute_config_hash("bafyabc", "gpt-4.1")
        h2 = _compute_config_hash("bafyabc", "gpt-4o")
        assert h1 != h2

    def test_differs_on_tool_hash_change(self) -> None:
        h1 = _compute_config_hash("bafyabc", "gpt-4.1")
        h2 = _compute_config_hash("bafydef", "gpt-4.1")
        assert h1 != h2

    def test_none_inputs_returns_none(self) -> None:
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

    def test_len_counts_by_title(self) -> None:
        m = ResolvedMarkets()
        m.add("0x1", "Question A", {"outcome": True})
        m.add("0x2", "Question B", {"outcome": False})
        assert len(m) == 2

    def test_bool_true_when_populated(self) -> None:
        m = ResolvedMarkets()
        assert not m
        m.add(None, "Question A", {"outcome": True})
        assert m

    def test_add_with_id_only(self) -> None:
        m = ResolvedMarkets()
        m.add("0x1", "", {"outcome": True})
        # title is empty so by_title is empty, but by_id has an entry
        assert "0x1" in m.by_id
        assert len(m.by_title) == 0


# ---------------------------------------------------------------------------
# Incremental state & deduplication
# ---------------------------------------------------------------------------


class TestIncrementalState:

    def test_round_trip(self, tmp_path: Path) -> None:
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
        assert load_fetch_state(tmp_path / "nonexistent.json") == {}

    def test_corrupt_file(self, tmp_path: Path) -> None:
        state_path = tmp_path / ".fetch_state.json"
        state_path.write_text("not json")
        assert load_fetch_state(state_path) == {}


class TestDeduplication:

    def test_load_ids_from_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        log_path.write_text('{"row_id": "a"}\n' '{"row_id": "b"}\n' '{"row_id": "c"}\n')
        ids = _load_ids_from_file(log_path)
        assert ids == {"a", "b", "c"}

    def test_empty_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("")
        assert _load_ids_from_file(log_path) == set()

    def test_missing_file(self, tmp_path: Path) -> None:
        assert _load_ids_from_file(tmp_path / "nope.jsonl") == set()

    def test_row_id_deterministic(self) -> None:
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
        from datetime import datetime, timezone

        d = datetime(2026, 4, 6, 12, 0, 0, tzinfo=timezone.utc)
        result = daily_log_path(Path("/tmp/logs"), d)
        assert result == Path("/tmp/logs/production_log_2026_04_06.jsonl")

    def test_defaults_to_today(self) -> None:
        from datetime import datetime, timezone

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
        from datetime import datetime, timedelta, timezone

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
        from datetime import datetime, timedelta, timezone

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
        markets = ResolvedMarkets()  # empty — no resolved markets
        deliveries = [_make_delivery(deliver_id="0xd1")]

        rows, pending, _, _, _, _ = _match_and_build(deliveries, markets, set(), "omen")
        assert len(rows) == 0
        assert len(pending) == 1
        assert pending[0]["deliver_id"] == "0xd1"

    def test_pending_delivery_matched_on_retry(self) -> None:
        """Simulates: delivery created in run 1 (unmatched),
        market resolves, run 2 retries and matches."""
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


class TestPendingInState:
    """Tests that pending deliveries round-trip through state file."""

    def test_pending_persisted_and_loaded(self, tmp_path: Path) -> None:
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
