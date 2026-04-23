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
"""Tests for benchmark/scorer.py."""

import argparse
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from benchmark.scorer import (
    DISAGREE_THRESHOLD,
    LARGE_TRADE_THRESHOLD,
    LATENCY_RESERVOIR_SIZE,
    PLATFORMS,
    WORST_BEST_SIZE,
    _accumulate_group,
    _cli_legacy_full_recompute,
    _derive_group,
    _derive_platform_path,
    _derive_tournament_path,
    _empty_group,
    _is_edge_eligible,
    _partition_rows_by_platform,
    brier_score,
    classify_difficulty,
    classify_disagreement,
    classify_horizon,
    classify_liquidity,
    compute_calibration_regression,
    compute_ece,
    compute_group_stats,
    disagree_bucket,
    edge_score,
    group_by,
    group_by_horizon,
    group_by_month,
    load_history,
    log_loss_score,
    rebuild,
    score,
    score_period,
    score_period_split_by_platform,
    update,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    p_yes: float = 0.5,
    outcome: bool = True,
    status: str = "valid",
    tool: str = "test-tool",
    platform: str = "omen",
    category: str = "other",
    lead_days: float | None = 2.0,
    predicted_at: str = "2026-03-15T10:00:00Z",
    tool_version: str | None = None,
    config_hash: str | None = None,
    market_prob: float | None = None,
    market_liquidity: float | None = None,
    market_spread: float | None = None,
    row_id: str | None = None,
    tool_ipfs_hash: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Build a minimal production_log row for testing."""
    if row_id is None:
        row_id = f"test_{uuid.uuid4().hex[:12]}"
    row: dict[str, Any] = {
        "row_id": row_id,
        "prediction_parse_status": status,
        "p_yes": p_yes if status == "valid" else None,
        "p_no": (1 - p_yes) if status == "valid" else None,
        "final_outcome": outcome,
        "tool_name": tool,
        "platform": platform,
        "category": category,
        "prediction_lead_time_days": lead_days,
        "predicted_at": predicted_at,
        "tool_version": tool_version,
        "config_hash": config_hash,
        "market_prob_at_prediction": market_prob,
        "market_liquidity_at_prediction": market_liquidity,
        "market_spread_at_prediction": market_spread,
    }
    if tool_ipfs_hash is not None:
        row["tool_ipfs_hash"] = tool_ipfs_hash
    if mode is not None:
        row["mode"] = mode
    return row


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------


class TestBrierScore:
    """Tests for brier_score."""

    def test_perfect_prediction_yes(self) -> None:
        """Test perfect prediction for yes outcome."""
        assert brier_score(1.0, True) == 0.0

    def test_perfect_prediction_no(self) -> None:
        """Test perfect prediction for no outcome."""
        assert brier_score(0.0, False) == 0.0

    def test_worst_prediction_yes(self) -> None:
        """Test worst prediction for yes outcome."""
        assert brier_score(0.0, True) == 1.0

    def test_worst_prediction_no(self) -> None:
        """Test worst prediction for no outcome."""
        assert brier_score(1.0, False) == 1.0

    def test_random_guessing(self) -> None:
        """Test random guessing gives 0.25."""
        assert brier_score(0.5, True) == 0.25
        assert brier_score(0.5, False) == 0.25

    def test_real_example(self) -> None:
        """p_yes=0.13, outcome=True → (0.13 - 1)² = 0.7569"""
        result = brier_score(0.13, True)
        assert abs(result - 0.7569) < 1e-10


# ---------------------------------------------------------------------------
# compute_group_stats
# ---------------------------------------------------------------------------


class TestComputeGroupStats:
    """Tests for compute_group_stats."""

    def test_all_valid(self) -> None:
        """Test stats with all valid rows."""
        rows = [
            _row(p_yes=0.9, outcome=True),  # (0.9-1)²  = 0.01
            _row(p_yes=0.8, outcome=False),  # (0.8-0)²  = 0.64
            _row(p_yes=0.6, outcome=True),  # (0.6-1)²  = 0.16
        ]
        stats = compute_group_stats(rows)
        expected_brier = round((0.01 + 0.64 + 0.16) / 3, 4)
        assert stats["brier"] == expected_brier
        assert stats["reliability"] == 1.0
        assert stats["n"] == 3

    def test_mixed_valid_and_malformed(self) -> None:
        """Test stats with mixed valid and malformed rows."""
        rows = [
            _row(p_yes=0.5, outcome=True),
            _row(status="malformed"),
            _row(status="error"),
        ]
        stats = compute_group_stats(rows)
        assert stats["reliability"] == round(1 / 3, 4)
        assert stats["n"] == 3
        # Brier only from the 1 valid row: (0.5 - 1)² = 0.25
        assert stats["brier"] == 0.25

    def test_all_invalid(self) -> None:
        """Test stats with all invalid rows."""
        rows = [_row(status="malformed"), _row(status="error")]
        stats = compute_group_stats(rows)
        assert stats["brier"] is None
        assert stats["reliability"] == 0.0
        assert stats["n"] == 2

    def test_empty(self) -> None:
        """Test stats with empty input."""
        stats = compute_group_stats([])
        assert stats["brier"] is None
        assert stats["reliability"] is None
        assert stats["n"] == 0


# ---------------------------------------------------------------------------
# classify_horizon
# ---------------------------------------------------------------------------


class TestClassifyHorizon:
    """Tests for classify_horizon."""

    def test_short(self) -> None:
        """Test short horizon classification."""
        assert classify_horizon(0.5) == "short_lt_7d"
        assert classify_horizon(6.9) == "short_lt_7d"

    def test_medium(self) -> None:
        """Test medium horizon classification."""
        assert classify_horizon(7.0) == "medium_7_30d"
        assert classify_horizon(15.0) == "medium_7_30d"
        assert classify_horizon(30.0) == "medium_7_30d"

    def test_long(self) -> None:
        """Test long horizon classification."""
        assert classify_horizon(30.1) == "long_gt_30d"
        assert classify_horizon(90.0) == "long_gt_30d"

    def test_none(self) -> None:
        """Test None lead days returns unknown."""
        assert classify_horizon(None) == "unknown"


# ---------------------------------------------------------------------------
# group_by
# ---------------------------------------------------------------------------


class TestGroupBy:
    """Tests for group_by."""

    def test_groups_by_tool(self) -> None:
        """Test grouping rows by tool name."""
        rows = [
            _row(tool="tool-a"),
            _row(tool="tool-a"),
            _row(tool="tool-b"),
        ]
        groups = group_by(rows, "tool_name")
        assert len(groups["tool-a"]) == 2
        assert len(groups["tool-b"]) == 1

    def test_missing_key_goes_to_unknown(self) -> None:
        """Test missing key groups to unknown."""
        rows = [{"other_field": "x"}]
        groups = group_by(rows, "tool_name")
        assert "unknown" in groups


# ---------------------------------------------------------------------------
# group_by_horizon
# ---------------------------------------------------------------------------


class TestGroupByHorizon:  # pylint: disable=too-few-public-methods
    """Tests for group_by_horizon."""

    def test_buckets(self) -> None:
        """Test horizon bucketing."""
        rows = [
            _row(lead_days=3.0),  # short
            _row(lead_days=15.0),  # medium
            _row(lead_days=45.0),  # long
            _row(lead_days=None),  # unknown
        ]
        result = group_by_horizon(rows)
        assert "short_lt_7d" in result
        assert "medium_7_30d" in result
        assert "long_gt_30d" in result
        assert "unknown" in result
        assert result["short_lt_7d"]["n"] == 1
        assert result["medium_7_30d"]["n"] == 1
        assert result["long_gt_30d"]["n"] == 1


# ---------------------------------------------------------------------------
# group_by_month
# ---------------------------------------------------------------------------


class TestGroupByMonth:
    """Tests for group_by_month."""

    def test_monthly_trend(self) -> None:
        """Test monthly trend grouping."""
        rows = [
            _row(predicted_at="2026-01-10T10:00:00Z"),
            _row(predicted_at="2026-01-20T10:00:00Z"),
            _row(predicted_at="2026-02-05T10:00:00Z"),
        ]
        trend = group_by_month(rows)
        months = [t["month"] for t in trend]
        assert months == ["2026-01", "2026-02"]
        assert trend[0]["n"] == 2
        assert trend[1]["n"] == 1

    def test_null_predicted_at_excluded(self) -> None:
        """Test null predicted_at rows are excluded."""
        rows = [
            _row(predicted_at="2026-03-01T10:00:00Z"),
            _row(predicted_at=None),  # type: ignore[arg-type]
        ]
        trend = group_by_month(rows)
        assert len(trend) == 1
        assert trend[0]["n"] == 1


# ---------------------------------------------------------------------------
# score (full pipeline)
# ---------------------------------------------------------------------------


class TestScore:
    """Tests for score."""

    def test_full_scoring(self) -> None:
        """Test full scoring pipeline."""
        rows = [
            _row(
                p_yes=0.9,
                outcome=True,
                tool="tool-a",
                platform="omen",
                category="crypto",
            ),
            _row(
                p_yes=0.8,
                outcome=False,
                tool="tool-a",
                platform="polymarket",
                category="politics",
            ),
            _row(
                p_yes=0.5,
                outcome=True,
                tool="tool-b",
                platform="omen",
                category="crypto",
            ),
            _row(status="malformed", tool="tool-b", platform="omen", category="other"),
        ]
        result = score(rows)

        assert result["total_rows"] == 4
        assert result["valid_rows"] == 3

        # Overall
        assert result["overall"]["n"] == 4
        assert result["overall"]["reliability"] == 0.75  # 3/4

        # By tool
        assert "tool-a" in result["by_tool"]
        assert "tool-b" in result["by_tool"]
        assert result["by_tool"]["tool-a"]["n"] == 2
        assert result["by_tool"]["tool-b"]["n"] == 2

        # By platform
        assert result["by_platform"]["omen"]["n"] == 3
        assert result["by_platform"]["polymarket"]["n"] == 1

        # By category
        assert result["by_category"]["crypto"]["n"] == 2

        # By tool × category — fleet-per-category cross breakdown
        assert "tool-a | crypto" in result["by_tool_category"]
        assert "tool-a | politics" in result["by_tool_category"]
        assert "tool-b | crypto" in result["by_tool_category"]
        assert result["by_tool_category"]["tool-a | crypto"]["n"] == 1
        assert result["by_tool_category"]["tool-b | crypto"]["n"] == 1
        # Brier asserts confirm the RIGHT row landed in each cell — catches
        # mutations that swap the composite key fields or mis-route rows.
        # tool-a|crypto: p=0.9, outcome=True → (0.9-1)^2 = 0.01
        # tool-b|crypto: p=0.5, outcome=True → (0.5-1)^2 = 0.25
        # tool-a|politics: p=0.8, outcome=False → (0.8-0)^2 = 0.64
        assert result["by_tool_category"]["tool-a | crypto"]["brier"] == 0.01
        assert result["by_tool_category"]["tool-b | crypto"]["brier"] == 0.25
        assert result["by_tool_category"]["tool-a | politics"]["brier"] == 0.64

    def test_empty_input(self) -> None:
        """Test scoring with empty input."""
        result = score([])
        assert result["total_rows"] == 0
        assert result["valid_rows"] == 0
        assert result["overall"]["brier"] is None

    def test_hand_calculated_brier(self) -> None:
        """Verify overall Brier against manual calculation."""
        rows = [
            _row(p_yes=0.13, outcome=True),  # (0.13-1)² = 0.7569
            _row(p_yes=0.90, outcome=True),  # (0.90-1)² = 0.01
            _row(p_yes=0.80, outcome=False),  # (0.80-0)² = 0.64
            _row(p_yes=0.60, outcome=True),  # (0.60-1)² = 0.16
            _row(p_yes=0.30, outcome=False),  # (0.30-0)² = 0.09
        ]
        result = score(rows)
        expected = round((0.7569 + 0.01 + 0.64 + 0.16 + 0.09) / 5, 4)
        assert result["overall"]["brier"] == expected

    def test_output_has_all_keys(self) -> None:
        """Test output contains all expected keys."""
        result = score([_row()])
        expected_keys = [
            "generated_at",
            "total_rows",
            "valid_rows",
            "overall",
            "by_tool",
            "by_platform",
            "by_category",
            "by_horizon",
            "trend",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------


class TestIncrementalUpdate:
    """Tests for incremental update."""

    def test_empty_scores_initialized(self, tmp_path: Path) -> None:
        """First update with no existing scores creates accumulators."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        rows = [_row(p_yes=0.9, outcome=True)]

        result = update(rows, scores_path, history_path)

        assert result["overall"]["n"] == 1
        assert result["overall"]["valid_n"] == 1
        assert result["overall"]["brier"] is not None
        assert scores_path.exists()

    def test_merge_two_batches(self, tmp_path: Path) -> None:
        """Two sequential updates accumulate correctly."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        batch1 = [_row(p_yes=0.9, outcome=True), _row(p_yes=0.8, outcome=False)]
        update(batch1, scores_path, history_path)

        batch2 = [_row(p_yes=0.6, outcome=True)]
        result = update(batch2, scores_path, history_path)

        assert result["overall"]["n"] == 3
        assert result["overall"]["valid_n"] == 3

    def test_brier_matches_full_recompute(self, tmp_path: Path) -> None:
        """Incremental Brier matches computing from all rows at once."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        all_rows = [
            _row(p_yes=0.13, outcome=True),
            _row(p_yes=0.90, outcome=True),
            _row(p_yes=0.80, outcome=False),
            _row(p_yes=0.60, outcome=True),
            _row(p_yes=0.30, outcome=False),
        ]

        # Incremental: 2 batches
        update(all_rows[:2], scores_path, history_path)
        inc_result = update(all_rows[2:], scores_path, history_path)

        # Full recompute
        full_result = score(all_rows)

        assert inc_result["overall"]["brier"] == full_result["overall"]["brier"]
        assert (
            inc_result["overall"]["directional_accuracy"]
            == full_result["overall"]["directional_accuracy"]
        )
        assert inc_result["overall"]["n"] == full_result["overall"]["n"]

    def test_by_tool_accumulated(self, tmp_path: Path) -> None:
        """Per-tool breakdown accumulates across updates."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        update([_row(tool="tool-a"), _row(tool="tool-b")], scores_path, history_path)
        result = update([_row(tool="tool-a")], scores_path, history_path)

        assert result["by_tool"]["tool-a"]["n"] == 2
        assert result["by_tool"]["tool-b"]["n"] == 1

    def test_calibration_buckets_accumulate(self, tmp_path: Path) -> None:
        """Calibration buckets accumulate counts correctly."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(p_yes=0.75, outcome=True),
            _row(p_yes=0.75, outcome=False),
            _row(p_yes=0.15, outcome=False),
        ]
        update(rows[:2], scores_path, history_path)
        result = update(rows[2:], scores_path, history_path)

        cal = {b["bin"]: b for b in result["calibration"]}
        assert cal["0.7-0.8"]["n"] == 2
        assert cal["0.1-0.2"]["n"] == 1

    def test_parse_breakdown_accumulates(self, tmp_path: Path) -> None:
        """Parse status counters accumulate across updates."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(status="valid", tool="t1"),
            _row(status="malformed", tool="t1"),
            _row(status="valid", tool="t1"),
        ]
        update(rows[:1], scores_path, history_path)
        result = update(rows[1:], scores_path, history_path)

        assert result["parse_breakdown"]["t1"]["valid"] == 2
        assert result["parse_breakdown"]["t1"]["malformed"] == 1

    def test_latency_reservoir_bounded(self, tmp_path: Path) -> None:
        """Reservoir stays at max LATENCY_RESERVOIR_SIZE per tool."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            {**_row(tool="t1"), "latency_s": i}
            for i in range(LATENCY_RESERVOIR_SIZE + 50)
        ]
        result = update(rows, scores_path, history_path)

        reservoir = result["latency_reservoir"]["t1"]
        assert len(reservoir) == LATENCY_RESERVOIR_SIZE
        # Should be the last 200 values (deterministic last-N)
        assert reservoir[0] == 50
        assert reservoir[-1] == LATENCY_RESERVOIR_SIZE + 49

    def test_worst_10_maintained(self, tmp_path: Path) -> None:
        """Worst 10 list keeps the highest Brier scores, deduplicated."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        # Create 15 rows with unique questions and varying Brier scores
        rows = [
            {**_row(p_yes=0.5 + i * 0.03, outcome=False), "question_text": f"Q{i}?"}
            for i in range(15)
        ]
        result = update(rows, scores_path, history_path)

        assert len(result["worst_10"]) == WORST_BEST_SIZE
        # Worst should have highest Brier (highest p_yes when outcome=False)
        worst_briers = [w["brier"] for w in result["worst_10"]]
        assert worst_briers == sorted(worst_briers, reverse=True)

    def test_best_10_maintained(self, tmp_path: Path) -> None:
        """Best 10 list keeps the lowest Brier scores, deduplicated."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            {**_row(p_yes=0.5 + i * 0.03, outcome=True), "question_text": f"Q{i}?"}
            for i in range(15)
        ]
        result = update(rows, scores_path, history_path)

        assert len(result["best_10"]) == WORST_BEST_SIZE
        best_briers = [b["brier"] for b in result["best_10"]]
        assert best_briers == sorted(best_briers)

    def test_worst_10_deduplicates_by_question(self, tmp_path: Path) -> None:
        """Same question keeps only the worst Brier."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            {**_row(p_yes=0.9, outcome=False), "question_text": "Same Q?"},
            {**_row(p_yes=0.8, outcome=False), "question_text": "Same Q?"},
            {**_row(p_yes=0.7, outcome=False), "question_text": "Different Q?"},
        ]
        result = update(rows, scores_path, history_path)

        questions = [w["question_text"] for w in result["worst_10"]]
        assert questions.count("Same Q?") == 1
        same_q = [w for w in result["worst_10"] if w["question_text"] == "Same Q?"][0]
        assert same_q["brier"] == round(0.9**2, 4)  # worst of the two

    def test_mixed_valid_invalid(self, tmp_path: Path) -> None:
        """Malformed rows count toward n but not valid_n or Brier."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(p_yes=0.9, outcome=True),
            _row(status="malformed"),
            _row(status="error"),
        ]
        result = update(rows, scores_path, history_path)

        assert result["overall"]["n"] == 3
        assert result["overall"]["valid_n"] == 1
        assert result["overall"]["reliability"] == round(1 / 3, 4)


# ---------------------------------------------------------------------------
# Month rollover
# ---------------------------------------------------------------------------


class TestMonthRollover:
    """Tests for month rollover logic."""

    def test_new_month_creates_snapshot(self, tmp_path: Path) -> None:
        """When month changes, old month is snapshot to history."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        # First update in March
        with patch("benchmark.scorer.datetime") as mock_dt:
            mock_dt.now.return_value = type(
                "D",
                (),
                {
                    "strftime": lambda self, fmt: (
                        "2026-03" if fmt == "%Y-%m" else "2026-03-15T10:00:00Z"
                    ),
                },
            )()
            mock_dt.side_effect = lambda *a, **k: type(
                "D",
                (),
                {
                    "strftime": lambda self, fmt: (
                        "2026-03" if fmt == "%Y-%m" else "2026-03-15T10:00:00Z"
                    ),
                },
            )()
            update([_row()], scores_path, history_path)

        # Force current_month to March in the saved file
        data = json.loads(scores_path.read_text())
        data["current_month"] = "2026-03"
        scores_path.write_text(json.dumps(data))

        # Second update in April (real time)
        with patch("benchmark.scorer.datetime") as mock_dt:
            real_dt = datetime
            mock_dt.now.return_value = real_dt(2026, 4, 6, 12, tzinfo=timezone.utc)
            update([_row()], scores_path, history_path)

        history = load_history(history_path)
        assert len(history) == 1
        assert history[0]["month"] == "2026-03"

    def test_snapshot_contains_final_stats(self, tmp_path: Path) -> None:
        """Snapshot has correct n and by_tool from the completed month."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [_row(tool="t1"), _row(tool="t1"), _row(tool="t2")]
        update(rows, scores_path, history_path)

        # Force month to old value
        data = json.loads(scores_path.read_text())
        data["current_month"] = "2026-01"
        scores_path.write_text(json.dumps(data))

        # Trigger rollover
        update([_row()], scores_path, history_path)

        history = load_history(history_path)
        assert history[0]["overall"]["n"] == 3
        assert "t1" in history[0]["by_tool"]
        assert history[0]["by_tool"]["t1"]["n"] == 2

    def test_accumulators_reset_after_rollover(self, tmp_path: Path) -> None:
        """After rollover, scores.json starts fresh for new month."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        update([_row(), _row(), _row()], scores_path, history_path)

        # Force old month
        data = json.loads(scores_path.read_text())
        data["current_month"] = "2025-12"
        scores_path.write_text(json.dumps(data))

        # Trigger rollover with 1 new row
        result = update([_row()], scores_path, history_path)
        assert result["overall"]["n"] == 1  # only the new row

    def test_same_month_no_snapshot(self, tmp_path: Path) -> None:
        """No snapshot when month hasn't changed."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        update([_row()], scores_path, history_path)
        update([_row()], scores_path, history_path)

        assert not history_path.exists() or history_path.read_text().strip() == ""

    def test_load_history(self, tmp_path: Path) -> None:
        """load_history reads multiple monthly lines."""
        history_path = tmp_path / "history.jsonl"
        history_path.write_text(
            '{"month": "2026-01", "overall": {"n": 100}}\n'
            '{"month": "2026-02", "overall": {"n": 200}}\n'
        )
        entries = load_history(history_path)
        assert len(entries) == 2
        assert entries[0]["month"] == "2026-01"
        assert entries[1]["overall"]["n"] == 200

    def test_load_history_missing_file(self, tmp_path: Path) -> None:
        """load_history returns empty list for missing file."""
        assert not load_history(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------


class TestRebuild:
    """Tests for rebuild."""

    def test_rebuild_from_archive_files(self, tmp_path: Path) -> None:
        """Rebuild reads all log files and produces valid scores."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        # Write two daily files with rows from different months
        f1 = logs_dir / "production_log_2026_03_15.jsonl"
        f1.write_text(
            json.dumps(
                _row(p_yes=0.9, outcome=True, predicted_at="2026-03-15T10:00:00Z")
            )
            + "\n"
            + json.dumps(
                _row(p_yes=0.8, outcome=False, predicted_at="2026-03-16T10:00:00Z")
            )
            + "\n"
        )
        f2 = logs_dir / "production_log_2026_04_01.jsonl"
        f2.write_text(
            json.dumps(
                _row(p_yes=0.7, outcome=True, predicted_at="2026-04-01T10:00:00Z")
            )
            + "\n"
        )

        result = rebuild(logs_dir, scores_path, history_path)

        # March should be in history, April should be current
        history = load_history(history_path)
        assert len(history) == 1
        assert history[0]["month"] == "2026-03"
        assert history[0]["overall"]["n"] == 2

        # Current scores should have April's row
        assert result["overall"]["n"] == 1
        assert result["overall"]["brier"] is not None

    def test_rebuild_single_month(self, tmp_path: Path) -> None:
        """Rebuild with all rows in same month creates no history."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        f1 = logs_dir / "production_log_2026_04_01.jsonl"
        rows = [_row(predicted_at="2026-04-01T10:00:00Z") for _ in range(5)]
        f1.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        result = rebuild(logs_dir, scores_path, history_path)

        assert result["overall"]["n"] == 5
        assert not history_path.exists() or history_path.read_text().strip() == ""

    def test_rebuild_empty_dir(self, tmp_path: Path) -> None:
        """Rebuild with no log files produces empty scores."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        result = rebuild(logs_dir, scores_path, history_path)

        assert result["overall"]["n"] == 0
        assert result["overall"]["brier"] is None

    def test_rebuild_includes_legacy(self, tmp_path: Path) -> None:
        """Rebuild picks up production_log_legacy.jsonl."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        legacy = logs_dir / "production_log_legacy.jsonl"
        legacy.write_text(json.dumps(_row(predicted_at="2026-03-01T10:00:00Z")) + "\n")

        result = rebuild(logs_dir, scores_path, history_path)
        # Legacy file matched by production_log_*.jsonl glob
        assert result["overall"]["n"] == 1


# ---------------------------------------------------------------------------
# Tool version and config breakdowns
# ---------------------------------------------------------------------------


class TestToolVersionBreakdown:
    """Tests for tool version breakdown."""

    def test_groups_by_version(self, tmp_path: Path) -> None:
        """Test grouping by tool version."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(tool="t1", tool_version="v1"),
            _row(tool="t1", tool_version="v1"),
            _row(tool="t1", tool_version="v2"),
        ]
        result = update(rows, scores_path, history_path)

        assert "t1 | v1" in result["by_tool_version"]
        assert "t1 | v2" in result["by_tool_version"]
        assert result["by_tool_version"]["t1 | v1"]["n"] == 2
        assert result["by_tool_version"]["t1 | v2"]["n"] == 1

    def test_null_version_grouped_as_unknown(self, tmp_path: Path) -> None:
        """Test null version grouped as unknown."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [_row(tool="t1"), _row(tool="t1")]  # tool_version=None
        result = update(rows, scores_path, history_path)

        assert "t1 | unknown" in result["by_tool_version"]
        assert result["by_tool_version"]["t1 | unknown"]["n"] == 2


class TestTournamentVersionNormalization:
    """Tests that tool_ipfs_hash is treated as tool_version (tournament path)."""

    def test_ipfs_hash_populates_version_when_tool_version_missing(
        self, tmp_path: Path
    ) -> None:
        """Tournament rows store the hash in tool_ipfs_hash, not tool_version."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(tool="t1", tool_ipfs_hash="bafy_abc"),
            _row(tool="t1", tool_ipfs_hash="bafy_abc"),
        ]
        result = update(rows, scores_path, history_path)

        assert "t1 | bafy_abc" in result["by_tool_version"]
        assert result["by_tool_version"]["t1 | bafy_abc"]["n"] == 2

    def test_tool_version_takes_precedence_over_ipfs_hash(self, tmp_path: Path) -> None:
        """If both fields are set, tool_version wins (production semantics)."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [_row(tool="t1", tool_version="prod_v1", tool_ipfs_hash="bafy_other")]
        result = update(rows, scores_path, history_path)

        assert "t1 | prod_v1" in result["by_tool_version"]
        assert "t1 | bafy_other" not in result["by_tool_version"]


class TestToolVersionModeBreakdown:
    """Tests for the (tool, version, mode) aggregation dimension."""

    def test_splits_by_mode_when_hash_matches(self, tmp_path: Path) -> None:
        """Same tool + same hash, different modes, land in separate files.

        :param tmp_path: pytest tmpdir fixture.
        """
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        tournament_path = tmp_path / "scores_tournament.json"

        rows = [
            _row(tool="t1", tool_version="v1", mode="production_replay"),
            _row(tool="t1", tool_ipfs_hash="v1", mode="tournament"),
        ]
        prod_result = update(
            rows, scores_path, history_path, tournament_scores_path=tournament_path
        )
        tourn_result = json.loads(tournament_path.read_text())

        assert "t1 | v1 | production_replay" in prod_result["by_tool_version_mode"]
        assert (
            prod_result["by_tool_version_mode"]["t1 | v1 | production_replay"]["n"] == 1
        )
        assert "t1 | v1 | tournament" in tourn_result["by_tool_version_mode"]
        assert tourn_result["by_tool_version_mode"]["t1 | v1 | tournament"]["n"] == 1

    def test_defaults_mode_to_production_replay(self, tmp_path: Path) -> None:
        """Rows without a mode field default to production_replay."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [_row(tool="t1", tool_version="v1")]
        result = update(rows, scores_path, history_path)

        assert "t1 | v1 | production_replay" in result["by_tool_version_mode"]

    def test_distinct_versions_within_same_mode(self, tmp_path: Path) -> None:
        """Two hashes of the same tool in the same mode produce two cells."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(tool="t1", tool_version="v1", mode="production_replay"),
            _row(tool="t1", tool_version="v1", mode="production_replay"),
            _row(tool="t1", tool_version="v2", mode="production_replay"),
        ]
        result = update(rows, scores_path, history_path)

        assert result["by_tool_version_mode"]["t1 | v1 | production_replay"]["n"] == 2
        assert result["by_tool_version_mode"]["t1 | v2 | production_replay"]["n"] == 1


class TestConfigBreakdown:
    """Tests for config breakdown."""

    def test_groups_by_config(self, tmp_path: Path) -> None:
        """Test grouping by config hash."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(tool="t1", config_hash="abc123"),
            _row(tool="t1", config_hash="abc123"),
            _row(tool="t1", config_hash="def456"),
        ]
        result = update(rows, scores_path, history_path)

        assert "t1 | abc123" in result["by_config"]
        assert "t1 | def456" in result["by_config"]
        assert result["by_config"]["t1 | abc123"]["n"] == 2
        assert result["by_config"]["t1 | def456"]["n"] == 1

    def test_null_config_grouped_as_unknown(self, tmp_path: Path) -> None:
        """Test null config grouped as unknown."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [_row(tool="t1")]  # config_hash=None
        result = update(rows, scores_path, history_path)

        assert "t1 | unknown" in result["by_config"]


# ---------------------------------------------------------------------------
# edge_score
# ---------------------------------------------------------------------------


class TestEdgeScore:
    """Tests for edge_score."""

    def test_tool_beats_market(self) -> None:
        """Positive edge when tool is closer to outcome than market."""
        # Outcome=True, tool says 0.9, market says 0.6
        # market_brier = (0.6-1)^2 = 0.16, tool_brier = (0.9-1)^2 = 0.01
        edge = edge_score(0.9, 0.6, True)
        assert round(edge, 4) == 0.15

    def test_market_beats_tool(self) -> None:
        """Negative edge when market is closer to outcome than tool."""
        # Outcome=False, tool says 0.7, market says 0.4
        # market_brier = (0.4-0)^2 = 0.16, tool_brier = (0.7-0)^2 = 0.49
        edge = edge_score(0.7, 0.4, False)
        assert round(edge, 4) == -0.33

    def test_tool_equals_market(self) -> None:
        """Zero edge when tool and market agree."""
        edge = edge_score(0.6, 0.6, True)
        assert edge == 0.0

    def test_both_wrong_outcome_true(self) -> None:
        """Both predict low but outcome is True — less wrong tool wins."""
        # market=0.3, tool=0.4, outcome=True
        # market_brier = (0.3-1)^2 = 0.49, tool_brier = (0.4-1)^2 = 0.36
        edge = edge_score(0.4, 0.3, True)
        assert round(edge, 4) == 0.13

    def test_both_wrong_outcome_false(self) -> None:
        """Both predict high but outcome is False — less wrong tool wins."""
        # market=0.8, tool=0.6, outcome=False
        # market_brier = 0.64, tool_brier = 0.36
        edge = edge_score(0.6, 0.8, False)
        assert round(edge, 4) == 0.28


# ---------------------------------------------------------------------------
# _is_edge_eligible
# ---------------------------------------------------------------------------


class TestIsEdgeEligible:
    """Tests for _is_edge_eligible."""

    def test_eligible_row(self) -> None:
        """Row with all required fields is eligible."""
        row = _row(p_yes=0.7, outcome=True, market_prob=0.6)
        assert _is_edge_eligible(row)

    def test_missing_market_prob(self) -> None:
        """Row without market_prob is not eligible."""
        row = _row(p_yes=0.7, outcome=True)
        assert not _is_edge_eligible(row)

    def test_invalid_parse(self) -> None:
        """Malformed prediction is not eligible."""
        row = _row(status="malformed", market_prob=0.6)
        assert not _is_edge_eligible(row)

    def test_no_outcome(self) -> None:
        """Row without outcome is not eligible."""
        row = _row(p_yes=0.7, market_prob=0.6)
        row["final_outcome"] = None
        assert not _is_edge_eligible(row)


# ---------------------------------------------------------------------------
# classify_difficulty
# ---------------------------------------------------------------------------


class TestClassifyDifficulty:
    """Tests for classify_difficulty."""

    def test_hard(self) -> None:
        """Market near 50/50 is hard."""
        assert classify_difficulty(0.55) == "hard"
        assert classify_difficulty(0.45) == "hard"

    def test_medium(self) -> None:
        """Market between thresholds is medium."""
        assert classify_difficulty(0.75) == "medium"
        assert classify_difficulty(0.25) == "medium"

    def test_easy(self) -> None:
        """Market far from 50/50 is easy."""
        assert classify_difficulty(0.9) == "easy"
        assert classify_difficulty(0.1) == "easy"

    def test_none(self) -> None:
        """None returns unknown."""
        assert classify_difficulty(None) == "unknown"

    def test_boundary_hard_medium(self) -> None:
        """Exact boundary: |0.65-0.5|=0.15 is medium (>= lo)."""
        assert classify_difficulty(0.65) == "medium"
        assert classify_difficulty(0.35) == "medium"

    def test_boundary_medium_easy(self) -> None:
        """Exact boundary: |0.8-0.5|=0.3 is medium (<= hi)."""
        assert classify_difficulty(0.8) == "medium"
        assert classify_difficulty(0.2) == "medium"


# ---------------------------------------------------------------------------
# classify_liquidity
# ---------------------------------------------------------------------------


class TestClassifyLiquidity:
    """Tests for classify_liquidity."""

    def test_low(self) -> None:
        """Below threshold is low."""
        assert classify_liquidity(6.0) == "low"
        assert classify_liquidity(499.99) == "low"

    def test_medium(self) -> None:
        """Between thresholds is medium."""
        assert classify_liquidity(500.0) == "medium"
        assert classify_liquidity(3000.0) == "medium"

    def test_high(self) -> None:
        """Above threshold is high."""
        assert classify_liquidity(5001.0) == "high"

    def test_none(self) -> None:
        """None returns unknown."""
        assert classify_liquidity(None) == "unknown"

    def test_boundary(self) -> None:
        """Exact boundary: 5000 is medium (<= hi)."""
        assert classify_liquidity(5000.0) == "medium"


# ---------------------------------------------------------------------------
# Edge in compute_group_stats (batch path)
# ---------------------------------------------------------------------------


class TestEdgeInGroupStats:
    """Tests for edge metrics in compute_group_stats."""

    def test_edge_with_market_prob(self) -> None:
        """Edge is computed when market_prob is available."""
        rows = [
            _row(p_yes=0.9, outcome=True, market_prob=0.6),  # edge > 0
            _row(p_yes=0.3, outcome=False, market_prob=0.4),  # edge > 0
        ]
        result = compute_group_stats(rows)
        assert result["edge_n"] == 2
        assert result["edge"] is not None
        assert result["edge"] > 0
        assert result["edge_positive_rate"] == 1.0

    def test_edge_without_market_prob(self) -> None:
        """Edge is null when no rows have market_prob."""
        rows = [_row(p_yes=0.7, outcome=True)]
        result = compute_group_stats(rows)
        assert result["edge_n"] == 0
        assert result["edge"] is None
        assert result["edge_positive_rate"] is None

    def test_edge_mixed(self) -> None:
        """Edge computed only from rows that have market_prob."""
        rows = [
            _row(p_yes=0.9, outcome=True, market_prob=0.6),  # eligible
            _row(p_yes=0.7, outcome=True),  # not eligible
        ]
        result = compute_group_stats(rows)
        assert result["edge_n"] == 1
        assert result["valid_n"] == 2  # both valid for Brier
        assert result["edge"] is not None

    def test_brier_unchanged_by_edge(self) -> None:
        """Adding market_prob doesn't change Brier computation."""
        rows_without = [_row(p_yes=0.7, outcome=True)]
        rows_with = [_row(p_yes=0.7, outcome=True, market_prob=0.5)]
        stats_without = compute_group_stats(rows_without)
        stats_with = compute_group_stats(rows_with)
        assert stats_without["brier"] == stats_with["brier"]
        assert (
            stats_without["directional_accuracy"] == stats_with["directional_accuracy"]
        )


# ---------------------------------------------------------------------------
# Edge in incremental path
# ---------------------------------------------------------------------------


class TestEdgeIncremental:
    """Tests for edge metrics in incremental update path."""

    def test_incremental_edge(self, tmp_path: Path) -> None:
        """Edge accumulators work through update()."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(p_yes=0.9, outcome=True, market_prob=0.6),
            _row(p_yes=0.3, outcome=True, market_prob=0.7),
        ]
        result = update(rows, scores_path, history_path)
        assert result["overall"]["edge_n"] == 2
        assert result["overall"]["edge"] is not None

    def test_incremental_no_edge(self, tmp_path: Path) -> None:
        """No edge when no market_prob."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [_row(p_yes=0.7, outcome=True)]
        result = update(rows, scores_path, history_path)
        assert result["overall"]["edge_n"] == 0
        assert result["overall"]["edge"] is None

    def test_resume_with_old_scores(self, tmp_path: Path) -> None:
        """Old scores.json without edge fields loads gracefully."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        # Write old-format scores.json (no edge fields)
        old_scores = {
            "current_month": "2026-04",
            "generated_at": "2026-04-08T00:00:00Z",
            "overall": {
                "n": 10,
                "valid_n": 9,
                "brier_sum": 2.0,
                "correct_count": 6,
                "sharpness_sum": 1.5,
                "outcome_yes_count": 5,
                "brier": 0.22,
                "directional_accuracy": 0.67,
                "sharpness": 0.17,
                "reliability": 0.9,
                "decision_worthy": False,
            },
            "by_tool": {},
            "by_platform": {},
            "by_category": {},
            "by_horizon": {},
            "by_tool_platform": {},
            "by_tool_version": {},
            "by_config": {},
            "calibration": {},
            "parse_breakdown": {},
            "latency_reservoir": {},
            "worst_10": [],
            "best_10": [],
        }
        scores_path.write_text(json.dumps(old_scores))

        # Add a new row with market_prob
        rows = [_row(p_yes=0.8, outcome=True, market_prob=0.5)]
        result = update(rows, scores_path, history_path)

        # Old rows didn't have edge, new one does
        assert result["overall"]["edge_n"] == 1
        assert result["overall"]["n"] == 11

    def test_difficulty_and_liquidity_dimensions(self, tmp_path: Path) -> None:
        """by_difficulty and by_liquidity dimensions are populated."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(p_yes=0.7, outcome=True, market_prob=0.55, market_liquidity=6.0),
            _row(p_yes=0.8, outcome=True, market_prob=0.85, market_liquidity=1000.0),
        ]
        result = update(rows, scores_path, history_path)

        assert "hard" in result["by_difficulty"]
        assert "easy" in result["by_difficulty"]
        assert "low" in result["by_liquidity"]
        assert "medium" in result["by_liquidity"]


# ---------------------------------------------------------------------------
# Edge in batch score()
# ---------------------------------------------------------------------------


class TestEdgeBatchScore:
    """Tests for edge metrics in batch score()."""

    def test_score_includes_edge_eligibility(self) -> None:
        """score() output includes edge_eligibility section."""
        rows = [
            _row(p_yes=0.7, outcome=True, market_prob=0.5),
            _row(p_yes=0.6, outcome=False),
        ]
        result = score(rows)
        elig = result["edge_eligibility"]
        assert elig["n_total"] == 2
        assert elig["n_eligible"] == 1
        assert elig["n_excluded"] == 1
        reasons = elig["exclusion_reasons"]
        assert reasons["missing_market_prob"] == 1
        assert reasons["invalid_or_incomplete"] == 0

    def test_score_includes_difficulty_and_liquidity(self) -> None:
        """score() output includes by_difficulty and by_liquidity."""
        rows = [
            _row(p_yes=0.7, outcome=True, market_prob=0.55, market_liquidity=6.0),
        ]
        result = score(rows)
        assert "by_difficulty" in result
        assert "by_liquidity" in result
        assert "hard" in result["by_difficulty"]
        assert "low" in result["by_liquidity"]


# ---------------------------------------------------------------------------
# Row-ID deduplication
# ---------------------------------------------------------------------------


class TestUpdateDedup:
    """Tests for row_id deduplication in update()."""

    def test_duplicate_rows_not_double_counted(self, tmp_path: Path) -> None:
        """Same rows passed twice produce same result as single pass."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "dedup.json"

        rows = [
            _row(p_yes=0.7, outcome=True, row_id="r1"),
            _row(p_yes=0.3, outcome=False, row_id="r2"),
        ]

        # First pass
        result1 = update(rows, scores_path, history_path, dedup_path)
        assert result1["overall"]["n"] == 2

        # Second pass with same rows — should be skipped
        result2 = update(rows, scores_path, history_path, dedup_path)
        assert (
            result2["overall"]["n"] == 2
        ), f"Expected 2 after dedup, got {result2['overall']['n']}"
        assert result2["overall"]["brier"] == result1["overall"]["brier"]

    def test_new_rows_added_after_dedup(self, tmp_path: Path) -> None:
        """New rows are added, duplicates are skipped."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "dedup.json"

        batch1 = [_row(p_yes=0.7, outcome=True, row_id="r1")]
        batch2 = [
            _row(p_yes=0.7, outcome=True, row_id="r1"),  # duplicate
            _row(p_yes=0.4, outcome=False, row_id="r3"),  # new
        ]

        update(batch1, scores_path, history_path, dedup_path)
        result = update(batch2, scores_path, history_path, dedup_path)

        assert (
            result["overall"]["n"] == 2
        ), f"Expected 2 (r1 + r3), got {result['overall']['n']}"

    def test_scored_row_ids_persisted(self, tmp_path: Path) -> None:
        """scored_row_ids are saved to separate dedup file."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "dedup.json"

        rows = [_row(p_yes=0.7, outcome=True, row_id="r1")]
        update(rows, scores_path, history_path, dedup_path)

        # Check the dedup file (not scores.json)
        assert dedup_path.exists()
        saved_ids = json.loads(dedup_path.read_text())
        assert "r1" in saved_ids

        # scores.json should NOT contain scored_row_ids
        saved_scores = json.loads(scores_path.read_text())
        assert "scored_row_ids" not in saved_scores

    def test_rows_without_row_id_always_counted(self, tmp_path: Path) -> None:
        """Rows without row_id are always accumulated (no dedup possible)."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "dedup.json"

        row_no_id = _row(p_yes=0.7, outcome=True)
        row_no_id.pop("row_id")

        update([row_no_id], scores_path, history_path, dedup_path)
        result = update([row_no_id], scores_path, history_path, dedup_path)
        # Without row_id, can't dedup — both passes count
        assert result["overall"]["n"] == 2

    def test_rebuild_then_update_deduplicates(self, tmp_path: Path) -> None:
        """Rows scored during rebuild() are not double-counted by update()."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "dedup.json"
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        # Write rows spanning two months to a log file
        rows = [
            _row(
                p_yes=0.7,
                outcome=True,
                row_id="march_r1",
                predicted_at="2026-03-15T10:00:00Z",
            ),
            _row(
                p_yes=0.4,
                outcome=False,
                row_id="april_r1",
                predicted_at="2026-04-05T10:00:00Z",
            ),
        ]
        log_file = logs_dir / "production_log_test.jsonl"
        with open(log_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Rebuild from log files
        rebuild(
            logs_dir=logs_dir,
            scores_path=scores_path,
            history_path=history_path,
            dedup_path=dedup_path,
        )

        # Verify rebuild tracked both row_ids in the dedup file
        saved_ids = json.loads(dedup_path.read_text())
        assert "march_r1" in saved_ids
        assert "april_r1" in saved_ids

        # Now call update() with the same rows — should all be skipped
        result = update(rows, scores_path, history_path, dedup_path)
        # n should still be 1 (only april_r1 in current month accumulators)
        # because rebuild() only accumulates the last month into scores.json
        # but dedup file contains both months' IDs
        assert (
            result["overall"]["n"] == 1
        ), f"Expected 1 (only last month in accumulators), got {result['overall']['n']}"

    def test_dedup_survives_month_rollover(self, tmp_path: Path) -> None:
        """Dedup state persists across month rollover."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "dedup.json"

        rows = [_row(p_yes=0.7, outcome=True, row_id="r1")]

        # Score in "March"
        with patch("benchmark.scorer.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 15, tzinfo=timezone.utc)
            mock_dt.side_effect = datetime
            update(rows, scores_path, history_path, dedup_path)

        # Rollover to "April" — accumulators reset, but dedup file stays
        with patch("benchmark.scorer.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 1, tzinfo=timezone.utc)
            mock_dt.side_effect = datetime
            result = update(rows, scores_path, history_path, dedup_path)

        # Row should be skipped even after rollover
        assert (
            result["overall"]["n"] == 0
        ), f"Expected 0 (r1 deduped across month rollover), got {result['overall']['n']}"

    def test_rebuild_deduplicates_across_log_files(self, tmp_path: Path) -> None:
        """Same row_id in two log files is only counted once during rebuild."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "dedup.json"
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        row = _row(p_yes=0.7, outcome=True, row_id="dup1")

        # Write same row to two different log files
        for name in ["production_log_a.jsonl", "production_log_b.jsonl"]:
            with open(logs_dir / name, "w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        result = rebuild(
            logs_dir=logs_dir,
            scores_path=scores_path,
            history_path=history_path,
            dedup_path=dedup_path,
        )
        assert (
            result["overall"]["n"] == 1
        ), f"Expected 1 (deduped), got {result['overall']['n']}"

    def test_empty_rebuild_clears_dedup(self, tmp_path: Path) -> None:
        """Empty rebuild clears stale dedup state so rows aren't falsely skipped."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "dedup.json"
        empty_logs = tmp_path / "empty_logs"
        empty_logs.mkdir()

        # Seed dedup via update
        rows = [_row(p_yes=0.7, outcome=True, row_id="r1")]
        update(rows, scores_path, history_path, dedup_path)
        assert json.loads(dedup_path.read_text()) == ["r1"]

        # Rebuild on empty logs — should clear dedup
        rebuild(
            logs_dir=empty_logs,
            scores_path=scores_path,
            history_path=history_path,
            dedup_path=dedup_path,
        )
        assert json.loads(dedup_path.read_text()) == []

        # Now update with the same row — should be accepted (not stale-skipped)
        result = update(rows, scores_path, history_path, dedup_path)
        assert (
            result["overall"]["n"] == 1
        ), f"Expected 1 (not stale-skipped), got {result['overall']['n']}"

    def test_legacy_scored_row_ids_migrated_from_scores_json(
        self, tmp_path: Path
    ) -> None:
        """Legacy scored_row_ids in scores.json are migrated to dedup file."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "dedup.json"

        # Simulate a legacy scores.json that contains scored_row_ids
        rows = [_row(p_yes=0.7, outcome=True, row_id="legacy1")]
        update(rows, scores_path, history_path, dedup_path)

        # Inject legacy scored_row_ids into scores.json and remove dedup file
        data = json.loads(scores_path.read_text())
        data["scored_row_ids"] = ["legacy1"]
        scores_path.write_text(json.dumps(data))
        dedup_path.unlink()

        # Update with the same row — should be skipped via migration
        result = update(rows, scores_path, history_path, dedup_path)
        assert (
            result["overall"]["n"] == 1
        ), f"Expected 1 (resumed accumulators), got {result['overall']['n']}"

        # The dedup file should now contain the migrated ID
        dedup_ids = json.loads(dedup_path.read_text())
        assert "legacy1" in dedup_ids


# ---------------------------------------------------------------------------
# Directional accuracy and no-signal rate
# ---------------------------------------------------------------------------


class TestDirectionalAccuracy:
    """Tests for directional_accuracy and no_signal_rate."""

    def test_excludes_half_predictions(self) -> None:
        """p_yes=0.5 rows excluded from directional accuracy, counted as no-signal."""
        rows = [
            _row(p_yes=0.5, outcome=True),
            _row(p_yes=0.5, outcome=False),
            _row(p_yes=0.8, outcome=True),
        ]
        stats = compute_group_stats(rows)
        assert stats["directional_accuracy"] == 1.0
        assert stats["n_directional"] == 1
        assert stats["no_signal_count"] == 2
        assert stats["no_signal_rate"] == round(2 / 3, 4)

    def test_all_half_returns_none(self) -> None:
        """All predictions at 0.5 gives directional_accuracy=None."""
        rows = [_row(p_yes=0.5, outcome=True), _row(p_yes=0.5, outcome=False)]
        stats = compute_group_stats(rows)
        assert stats["directional_accuracy"] is None
        assert stats["n_directional"] == 0
        assert stats["no_signal_rate"] == 1.0

    def test_no_half_predictions(self) -> None:
        """No 0.5 predictions gives no_signal_rate=0."""
        rows = [
            _row(p_yes=0.7, outcome=True),
            _row(p_yes=0.3, outcome=False),
        ]
        stats = compute_group_stats(rows)
        assert stats["no_signal_rate"] == 0.0
        assert stats["no_signal_count"] == 0
        assert stats["n_directional"] == 2

    def test_batch_incremental_parity(self, tmp_path: Path) -> None:
        """Batch and incremental paths produce the same directional_accuracy."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(p_yes=0.5, outcome=True),
            _row(p_yes=0.5, outcome=False),
            _row(p_yes=0.8, outcome=True),
            _row(p_yes=0.3, outcome=False),
            _row(p_yes=0.6, outcome=True),
        ]
        batch = compute_group_stats(rows)
        inc = update(rows, scores_path, history_path)
        assert inc["overall"]["directional_accuracy"] == batch["directional_accuracy"]
        assert inc["overall"]["no_signal_rate"] == batch["no_signal_rate"]
        assert inc["overall"]["n_directional"] == batch["n_directional"]


# ---------------------------------------------------------------------------
# Log loss
# ---------------------------------------------------------------------------


class TestLogLoss:
    """Tests for log_loss_score."""

    def test_confident_correct(self) -> None:
        """p_yes=0.9, outcome=True → -log(0.9) ≈ 0.1054."""
        result = log_loss_score(0.9, True)
        assert abs(result - 0.10536) < 0.001

    def test_confident_wrong(self) -> None:
        """p_yes=0.9, outcome=False → -log(0.1) ≈ 2.3026."""
        result = log_loss_score(0.9, False)
        assert abs(result - 2.3026) < 0.001

    def test_coin_flip(self) -> None:
        """p_yes=0.5 → -log(0.5) ≈ 0.6931."""
        result = log_loss_score(0.5, True)
        assert abs(result - 0.6931) < 0.001

    def test_extreme_clamped(self) -> None:
        """p_yes=0.0 does not crash (clamped)."""
        result = log_loss_score(0.0, True)
        assert result > 30  # -log(1e-15) ≈ 34.5

    def test_in_batch_output(self) -> None:
        """log_loss appears in compute_group_stats output."""
        rows = [_row(p_yes=0.9, outcome=True), _row(p_yes=0.3, outcome=False)]
        stats = compute_group_stats(rows)
        assert stats["log_loss"] is not None
        assert stats["log_loss"] > 0

    def test_batch_incremental_parity(self, tmp_path: Path) -> None:
        """Log loss matches between batch and incremental."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(p_yes=0.9, outcome=True),
            _row(p_yes=0.3, outcome=False),
            _row(p_yes=0.6, outcome=True),
        ]
        batch = compute_group_stats(rows)
        inc = update(rows, scores_path, history_path)
        assert inc["overall"]["log_loss"] == batch["log_loss"]


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


class TestECE:
    """Tests for compute_ece."""

    def test_hand_calculated(self) -> None:
        """Hand-calculated ECE: 3 bins."""
        bins = [
            {"n": 50, "gap": 0.05},
            {"n": 30, "gap": 0.10},
            {"n": 20, "gap": 0.02},
        ]
        # ECE = (50*0.05 + 30*0.10 + 20*0.02) / 100 = 5.9/100 = 0.059
        ece = compute_ece(bins)
        assert ece == 0.059

    def test_perfect_calibration(self) -> None:
        """All gaps = 0 gives ECE = 0."""
        bins = [{"n": 25, "gap": 0.0}, {"n": 30, "gap": 0.0}]
        assert compute_ece(bins) == 0.0

    def test_empty_bins(self) -> None:
        """No populated bins gives None."""
        assert compute_ece([]) is None
        assert compute_ece([{"n": 0, "gap": 0.1}]) is None

    def test_small_bins_excluded(self) -> None:
        """Bins with < MIN_CALIBRATION_BIN_SIZE are excluded."""
        bins = [{"n": 5, "gap": 0.5}, {"n": 50, "gap": 0.1}]
        # Only the n=50 bin qualifies → ECE = 0.1
        assert compute_ece(bins) == 0.1

    def test_all_bins_below_min(self) -> None:
        """All bins below min size returns None."""
        bins = [{"n": 10, "gap": 0.1}, {"n": 15, "gap": 0.2}]
        assert compute_ece(bins) is None

    def test_negative_gap_absolute(self) -> None:
        """ECE uses absolute gap."""
        bins = [{"n": 25, "gap": -0.1}]
        assert compute_ece(bins) == 0.1

    def test_in_score_output(self) -> None:
        """ECE appears in score() output."""
        rows = [_row(p_yes=0.75, outcome=True) for _ in range(10)]
        result = score(rows)
        assert "ece" in result


# ---------------------------------------------------------------------------
# Calibration regression
# ---------------------------------------------------------------------------


class TestCalibrationRegression:
    """Tests for compute_calibration_regression (row-level logistic)."""

    def test_well_calibrated(self) -> None:
        """Well-calibrated predictions: slope ≈ 1.0, intercept ≈ 0.0."""
        # 40 rows: 20 at p_yes=0.2 (4 Yes), 20 at p_yes=0.8 (16 Yes)
        rows = (
            [_row(p_yes=0.2, outcome=True)] * 4
            + [_row(p_yes=0.2, outcome=False)] * 16
            + [_row(p_yes=0.8, outcome=True)] * 16
            + [_row(p_yes=0.8, outcome=False)] * 4
        )
        result = compute_calibration_regression(rows)
        slope = result["calibration_slope"]
        intercept = result["calibration_intercept"]
        assert slope is not None and abs(slope - 1.0) < 0.25
        assert intercept is not None and abs(intercept) < 0.25

    def test_overconfident(self) -> None:
        """Overconfident tool: slope < 1.0."""
        # Tool predicts 0.9 but only 60% are Yes → overconfident
        rows = (
            [_row(p_yes=0.1, outcome=False)] * 12
            + [_row(p_yes=0.1, outcome=True)] * 8
            + [_row(p_yes=0.9, outcome=True)] * 12
            + [_row(p_yes=0.9, outcome=False)] * 8
        )
        result = compute_calibration_regression(rows)
        slope = result["calibration_slope"]
        assert slope is not None and slope < 1.0

    def test_too_few_rows(self) -> None:
        """< MIN_CAL_REG_ROWS returns None for both."""
        rows = [_row(p_yes=0.5, outcome=True)] * 5
        result = compute_calibration_regression(rows)
        assert result["calibration_intercept"] is None
        assert result["calibration_slope"] is None

    def test_in_score_output(self) -> None:
        """Calibration regression appears in score() output."""
        rows = [_row(p_yes=0.15, outcome=False)] * 15 + [
            _row(p_yes=0.85, outcome=True)
        ] * 15
        result = score(rows)
        assert "calibration_intercept" in result
        assert "calibration_slope" in result


# ---------------------------------------------------------------------------
# _derive_group null schema parity
# ---------------------------------------------------------------------------


class TestDeriveGroupSchema:
    """Verify _derive_group produces the same keys as compute_group_stats."""

    def test_empty_group_has_all_keys(self) -> None:
        """_derive_group on empty accumulators has BSS/baseline/yes_rate."""
        result = _derive_group(_empty_group())
        for key in ("outcome_yes_rate", "baseline_brier", "brier_skill_score"):
            assert key in result, f"_derive_group missing '{key}' on n==0 path"
            assert result[key] is None

    def test_schema_matches_batch(self) -> None:
        """Keys from _derive_group match compute_group_stats for same data."""
        rows = [_row(p_yes=0.7, outcome=True), _row(p_yes=0.3, outcome=False)]
        batch_keys = set(compute_group_stats(rows).keys())
        # Simulate incremental
        group = _empty_group()
        for r in rows:
            _accumulate_group(group, r)
        inc_keys = set(_derive_group(group).keys())
        missing = batch_keys - inc_keys
        assert not missing, f"_derive_group missing keys: {missing}"


# ---------------------------------------------------------------------------
# score_period
# ---------------------------------------------------------------------------


class TestScorePeriod:
    """Tests for score_period — timestamp-based filtering."""

    def _ts(self, days_ago: int) -> str:
        """Return an ISO timestamp *days_ago* days in the past.

        :param days_ago: how many days before now.
        :return: ISO 8601 UTC timestamp string.
        """
        dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def test_filters_by_timestamp_not_file(self, tmp_path: Path) -> None:
        """days=1 includes rows from the last 24h regardless of file name."""
        logs = tmp_path / "logs"
        logs.mkdir()

        old_row = _row(p_yes=0.7, outcome=True, predicted_at=self._ts(5))
        new_row = _row(p_yes=0.3, outcome=False, predicted_at=self._ts(0))

        (logs / "production_log_2020_01_01.jsonl").write_text(
            json.dumps(old_row) + "\n" + json.dumps(new_row) + "\n"
        )

        result = score_period(logs, days=1)
        assert result["total_rows"] == 1
        assert result["overall"]["brier"] is not None

    def test_days7_includes_recent_week(self, tmp_path: Path) -> None:
        """days=7 includes rows from the last 7 calendar days."""
        logs = tmp_path / "logs"
        logs.mkdir()

        rows_data = [
            _row(p_yes=0.7, outcome=True, predicted_at=self._ts(10)),
            _row(p_yes=0.4, outcome=True, predicted_at=self._ts(3)),
            _row(p_yes=0.9, outcome=True, predicted_at=self._ts(0)),
        ]
        content = "\n".join(json.dumps(r) for r in rows_data) + "\n"
        (logs / "production_log_2020_01_01.jsonl").write_text(content)

        result = score_period(logs, days=7)
        assert result["total_rows"] == 2

    def test_reads_both_naming_conventions(self, tmp_path: Path) -> None:
        """Rows from both file naming conventions are included."""
        logs = tmp_path / "logs"
        logs.mkdir()

        row_a = _row(p_yes=0.6, outcome=True, predicted_at=self._ts(0))
        row_b = _row(p_yes=0.4, outcome=False, predicted_at=self._ts(0))

        (logs / "production_log_2020_01_01.jsonl").write_text(json.dumps(row_a) + "\n")
        (logs / "2020-01-02.jsonl").write_text(json.dumps(row_b) + "\n")

        result = score_period(logs, days=1)
        assert result["total_rows"] == 2

    def test_empty_dir(self, tmp_path: Path) -> None:
        """Empty log directory returns zero-row scores."""
        logs = tmp_path / "empty_logs"
        logs.mkdir()
        result = score_period(logs, days=7)
        assert result["total_rows"] == 0
        assert result["overall"]["brier"] is None


# ---------------------------------------------------------------------------
# Degenerate calibration regression
# ---------------------------------------------------------------------------


class TestCalibrationRegressionDegenerate:
    """Tests for edge cases in compute_calibration_regression."""

    def test_uniform_probability_returns_none(self) -> None:
        """All predictions at same p_yes → slope is unidentifiable."""
        rows = [_row(p_yes=0.5, outcome=True)] * 20 + [
            _row(p_yes=0.5, outcome=False)
        ] * 20
        result = compute_calibration_regression(rows)
        assert result["calibration_slope"] is None
        assert result["calibration_intercept"] is None

    def test_all_same_outcome_converges(self) -> None:
        """All outcomes True — optimizer should still return or return None."""
        rows = [_row(p_yes=0.3, outcome=True)] * 20 + [
            _row(p_yes=0.8, outcome=True)
        ] * 20
        result = compute_calibration_regression(rows)
        # May return values or None — either is acceptable, just no crash
        assert "calibration_slope" in result
        assert "calibration_intercept" in result


# ---------------------------------------------------------------------------
# classify_disagreement
# ---------------------------------------------------------------------------


class TestClassifyDisagreement:
    """Tests for classify_disagreement."""

    def test_tool_wins(self) -> None:
        """Tool closer to truth."""
        assert classify_disagreement(0.8, 0.3, True) == "tool_win"

    def test_market_wins(self) -> None:
        """Market closer to truth."""
        assert classify_disagreement(0.3, 0.8, True) == "market_win"

    def test_tie(self) -> None:
        """Same prediction → both equidistant from truth."""
        assert classify_disagreement(0.5, 0.5, True) == "tie"

    def test_symmetric_tie(self) -> None:
        """Identical p_yes and market_prob → tie regardless of outcome."""
        assert classify_disagreement(0.3, 0.3, True) == "tie"

    def test_outcome_false(self) -> None:
        """Tool closer when outcome is False."""
        assert classify_disagreement(0.2, 0.7, False) == "tool_win"

    def test_extreme_values(self) -> None:
        """Test with 0.0 and 1.0 predictions."""
        assert classify_disagreement(1.0, 0.0, True) == "tool_win"
        assert classify_disagreement(0.0, 1.0, True) == "market_win"
        assert classify_disagreement(0.0, 1.0, False) == "tool_win"


# ---------------------------------------------------------------------------
# disagree_bucket
# ---------------------------------------------------------------------------


class TestDisagreeBucket:
    """Tests for disagree_bucket."""

    def test_no_trade_zero_disagreement(self) -> None:
        """Identical predictions → no_trade."""
        assert disagree_bucket(0.5, 0.5) == "no_trade"

    def test_no_trade_at_threshold(self) -> None:
        """Exactly at DISAGREE_THRESHOLD → no_trade (<=)."""
        assert disagree_bucket(0.5, 0.5 + DISAGREE_THRESHOLD) == "no_trade"

    def test_small_trade_just_above(self) -> None:
        """Just above DISAGREE_THRESHOLD → small_trade."""
        assert disagree_bucket(0.5, 0.5 + DISAGREE_THRESHOLD + 0.001) == "small_trade"

    def test_small_trade_at_large_threshold(self) -> None:
        """Exactly at LARGE_TRADE_THRESHOLD → small_trade (<=)."""
        assert disagree_bucket(0.5, 0.5 + LARGE_TRADE_THRESHOLD) == "small_trade"

    def test_large_trade_above(self) -> None:
        """Above LARGE_TRADE_THRESHOLD → large_trade."""
        assert disagree_bucket(0.5, 0.5 + LARGE_TRADE_THRESHOLD + 0.01) == "large_trade"

    def test_large_trade_extreme(self) -> None:
        """Maximum disagreement."""
        assert disagree_bucket(0.0, 1.0) == "large_trade"

    def test_sign_doesnt_matter(self) -> None:
        """Negative disagreement (tool < market) uses absolute value."""
        assert disagree_bucket(0.3, 0.5) == "large_trade"  # |0.3-0.5|=0.2 > 0.10


# ---------------------------------------------------------------------------
# Diagnostic metrics in accumulator path
# ---------------------------------------------------------------------------


class TestDiagnosticAccumulators:
    """Test that diagnostic metrics accumulate correctly and derive properly."""

    def _make_edge_rows(self) -> list[dict[str, Any]]:
        """Build rows with known diagnostic metric outcomes.

        20 tool-win large-trade, 10 market-win large-trade,
        10 no-trade rows.

        :return: list of production log row dicts.
        """
        rows = []
        for i in range(20):
            rows.append(
                _row(p_yes=0.8, outcome=True, market_prob=0.3, row_id=f"diag_tw_{i}")
            )
        for i in range(10):
            rows.append(
                _row(p_yes=0.3, outcome=True, market_prob=0.8, row_id=f"diag_mw_{i}")
            )
        for i in range(10):
            rows.append(
                _row(p_yes=0.52, outcome=True, market_prob=0.50, row_id=f"diag_nt_{i}")
            )
        return rows

    def test_accumulate_group_counts(self) -> None:
        """Verify raw accumulator values after accumulating known rows."""
        group = _empty_group()
        for row in self._make_edge_rows():
            _accumulate_group(group, row)

        # 30 large_trade rows (20 tool_win + 10 market_win)
        assert group["n_large_trade"] == 30
        # 10 no_trade rows
        assert group["n_no_trade"] == 10
        assert group["n_small_trade"] == 0
        # disagree_n = 20 tool_wins + 10 market_wins (no ties)
        assert group["disagree_n"] == 30
        assert group["disagree_tool_win_count"] == 20
        # bias losses = 10 market_wins
        assert group["n_bias_losses"] == 10
        # bias_sum = 10 * (0.3 - 1.0) = -7.0
        assert abs(group["bias_sum"] - (-7.0)) < 1e-9

    def test_derive_conditional_accuracy(self) -> None:
        """Conditional accuracy = tool_wins / disagree_n."""
        group = _empty_group()
        for row in self._make_edge_rows():
            _accumulate_group(group, row)
        result = _derive_group(group)

        # 20 tool_wins / 30 disagree = 0.6667
        assert result["conditional_accuracy_rate"] == round(20 / 30, 4)

    def test_derive_disagreement_brier(self) -> None:
        """Brier per bucket matches hand-calculated values."""
        group = _empty_group()
        for row in self._make_edge_rows():
            _accumulate_group(group, row)
        result = _derive_group(group)

        assert result["brier_large_trade"] == round((20 * 0.04 + 10 * 0.49) / 30, 4)
        assert result["brier_no_trade"] is None
        assert result["n_no_trade"] == 10

    def test_derive_directional_bias(self) -> None:
        """Directional bias = mean(p_yes - outcome) for losses."""
        group = _empty_group()
        for row in self._make_edge_rows():
            _accumulate_group(group, row)
        result = _derive_group(group)

        # 10 losses, bias_sum = -7.0 → bias = -0.7
        # n=10 < MIN_SAMPLE_SIZE=30 → None
        assert result["directional_bias"] is None
        assert result["n_bias_losses"] == 10

    def test_derive_bias_with_enough_samples(self) -> None:
        """Directional bias computed when n_bias_losses >= MIN_SAMPLE_SIZE."""
        group = _empty_group()
        # Need >= 30 market_win rows
        for i in range(35):
            row = _row(p_yes=0.3, outcome=True, market_prob=0.8, row_id=f"bias_{i}")
            _accumulate_group(group, row)
        result = _derive_group(group)

        assert result["n_bias_losses"] == 35
        expected_bias = round((0.3 - 1.0), 4)
        assert result["directional_bias"] == expected_bias

    def test_no_edge_rows_all_none(self) -> None:
        """No edge-eligible rows → all diagnostic metrics are None."""
        group = _empty_group()
        for _ in range(50):
            _accumulate_group(group, _row(p_yes=0.7, outcome=True))
        result = _derive_group(group)

        assert result["conditional_accuracy_rate"] is None
        assert result["brier_no_trade"] is None
        assert result["brier_small_trade"] is None
        assert result["brier_large_trade"] is None
        assert result["directional_bias"] is None
        assert result["disagree_n"] == 0

    def test_all_ties_conditional_accuracy_none(self) -> None:
        """All ties → disagree_n=0 → conditional accuracy is None."""
        group = _empty_group()
        for i in range(50):
            # Same p_yes and market_prob → tie
            _accumulate_group(
                group, _row(p_yes=0.5, outcome=True, market_prob=0.5, row_id=f"tie_{i}")
            )
        result = _derive_group(group)

        assert result["disagree_n"] == 0
        assert result["conditional_accuracy_rate"] is None

    def test_batch_matches_incremental(self) -> None:
        """compute_group_stats matches _accumulate_group + _derive_group."""
        rows = self._make_edge_rows()
        batch = compute_group_stats(rows)

        group = _empty_group()
        for row in rows:
            _accumulate_group(group, row)
        incremental = _derive_group(group)

        for key in (
            "conditional_accuracy_rate",
            "brier_large_trade",
            "brier_small_trade",
            "brier_no_trade",
            "directional_bias",
            "disagree_n",
            "n_no_trade",
            "n_small_trade",
            "n_large_trade",
            "n_bias_losses",
        ):
            assert (
                batch[key] == incremental[key]
            ), f"{key}: batch={batch[key]} != incremental={incremental[key]}"


# ---------------------------------------------------------------------------
# Mode split: scores.json vs scores_tournament.json (BENCHMARK_MODE_SPLIT_SPEC)
# ---------------------------------------------------------------------------


class TestModeSplit:
    """Tests for production/tournament scoring split."""

    def test_partition_rows_by_mode(self) -> None:
        """_partition_rows_by_mode routes rows by the mode field; missing mode defaults to production."""
        # pylint: disable=import-outside-toplevel
        from benchmark.scorer import _partition_rows_by_mode

        rows = [
            _row(mode="tournament", row_id="t1"),
            _row(mode="production_replay", row_id="p1"),
            _row(mode=None, row_id="p2"),  # missing mode -> production
            _row(mode="tournament", row_id="t2"),
        ]
        prod, tourn = _partition_rows_by_mode(rows)
        prod_ids = {r["row_id"] for r in prod}
        tourn_ids = {r["row_id"] for r in tourn}
        assert prod_ids == {"p1", "p2"}
        assert tourn_ids == {"t1", "t2"}

    def test_unknown_mode_routes_to_production_with_warning(self, caplog: Any) -> None:
        """Unknown modes route to production but log a warning (once per mode).

        :param caplog: pytest log capture fixture.
        """
        # pylint: disable=import-outside-toplevel
        from benchmark import scorer as _scorer_mod
        from benchmark.scorer import _partition_rows_by_mode

        _scorer_mod._WARNED_UNKNOWN_MODES.clear()  # pylint: disable=protected-access
        rows = [
            _row(mode="shadow", row_id="s1"),
            _row(mode="shadow", row_id="s2"),
            _row(mode="staging", row_id="g1"),
        ]
        with caplog.at_level(logging.WARNING, logger="benchmark.scorer"):
            prod, tourn = _partition_rows_by_mode(rows)

        assert {r["row_id"] for r in prod} == {"s1", "s2", "g1"}
        assert not tourn
        # One warning per distinct unknown mode, not per row
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        modes_warned = {w.args[0] for w in warnings if w.args}
        assert modes_warned == {"shadow", "staging"}

    def test_derive_tournament_path(self) -> None:
        """_derive_tournament_path appends _tournament to the stem."""
        assert _derive_tournament_path(Path("/tmp/results/scores.json")) == Path(
            "/tmp/results/scores_tournament.json"
        )

    def test_update_writes_both_files_and_preserves_split(self, tmp_path: Path) -> None:
        """update() splits rows by mode and writes two score files; modes don't cross-contaminate."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        tournament_path = tmp_path / "scores_tournament.json"

        rows = [
            _row(mode="production_replay", tool="prod-tool", row_id="p1"),
            _row(mode="production_replay", tool="prod-tool", row_id="p2"),
            _row(mode="tournament", tool="tourn-tool", row_id="t1"),
        ]
        update(rows, scores_path, history_path, tournament_scores_path=tournament_path)

        prod = json.loads(scores_path.read_text())
        tourn = json.loads(tournament_path.read_text())

        assert "prod-tool" in prod["by_tool"]
        assert "tourn-tool" not in prod["by_tool"]
        assert "tourn-tool" in tourn["by_tool"]
        assert "prod-tool" not in tourn["by_tool"]

    def test_update_shares_dedup_across_modes(self, tmp_path: Path) -> None:
        """Both modes write to a single dedup file; re-feeding same rows is a no-op."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "ids.json"
        tournament_path = tmp_path / "scores_tournament.json"

        rows = [
            _row(mode="production_replay", row_id="p1"),
            _row(mode="tournament", row_id="t1"),
        ]
        update(
            rows,
            scores_path,
            history_path,
            dedup_path=dedup_path,
            tournament_scores_path=tournament_path,
        )
        ids_after_first = set(json.loads(dedup_path.read_text()))
        assert ids_after_first == {"p1", "t1"}

        # Second call with the same rows should skip both
        update(
            rows,
            scores_path,
            history_path,
            dedup_path=dedup_path,
            tournament_scores_path=tournament_path,
        )
        prod = json.loads(scores_path.read_text())
        tourn = json.loads(tournament_path.read_text())
        # Sample size unchanged after no-op second update
        assert prod["overall"]["n"] == 1
        assert tourn["overall"]["n"] == 1

    def test_tournament_accumulates_across_month_rollover(self, tmp_path: Path) -> None:
        """Tournament scores must not reset on calendar month rollover.

        Spec requires tournament cells to accumulate cumulatively so the
        CALLOUT_MIN_N threshold holds across month boundaries. Production
        still snapshots + resets; tournament only updates the month label.

        :param tmp_path: pytest tmpdir fixture.
        """
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        tournament_path = tmp_path / "scores_tournament.json"

        # Seed tournament scores with "previous month" rows
        with patch("benchmark.scorer.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 20, tzinfo=timezone.utc)
            mock_dt.side_effect = datetime
            update(
                [
                    _row(
                        row_id="prev_m_1",
                        predicted_at="2026-03-20T10:00:00Z",
                        mode="tournament",
                    )
                ],
                scores_path,
                history_path,
                tournament_scores_path=tournament_path,
            )

        seeded = json.loads(tournament_path.read_text())
        assert seeded["overall"]["n"] == 1

        # Now run again under "current month" — tournament must keep the
        # seeded accumulator, not reset to zero.
        with patch("benchmark.scorer.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 5, tzinfo=timezone.utc)
            mock_dt.side_effect = datetime
            update(
                [
                    _row(
                        row_id="curr_m_1",
                        predicted_at="2026-04-05T10:00:00Z",
                        mode="tournament",
                    )
                ],
                scores_path,
                history_path,
                tournament_scores_path=tournament_path,
            )

        final = json.loads(tournament_path.read_text())
        assert (
            final["overall"]["n"] == 2
        ), "tournament accumulator reset on month rollover"

    def test_rebuild_history_production_only(self, tmp_path: Path) -> None:
        """rebuild() emits monthly history snapshots only for production rows."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        tournament_path = tmp_path / "scores_tournament.json"

        # Two months of production + tournament rows each
        log_file = logs_dir / "production_log_2026_01.jsonl"
        rows = [
            _row(
                mode="production_replay",
                predicted_at="2026-01-15T10:00:00Z",
                row_id="p_jan",
            ),
            _row(
                mode="production_replay",
                predicted_at="2026-02-15T10:00:00Z",
                row_id="p_feb",
            ),
            _row(
                mode="tournament",
                predicted_at="2026-01-15T10:00:00Z",
                row_id="t_jan",
            ),
            _row(
                mode="tournament",
                predicted_at="2026-02-15T10:00:00Z",
                row_id="t_feb",
            ),
        ]
        log_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        rebuild(
            logs_dir=logs_dir,
            scores_path=scores_path,
            history_path=history_path,
            tournament_scores_path=tournament_path,
        )

        # History exists and contains at most production months (1 snapshot for Jan).
        assert history_path.exists()
        history_lines = [
            json.loads(ln) for ln in history_path.read_text().splitlines() if ln.strip()
        ]
        # One closed prior month (January); current month doesn't snapshot.
        # Verify the snapshot reflects production (n=1, not n=2 from pooled modes).
        assert len(history_lines) == 1
        assert history_lines[0]["overall"]["n"] == 1


# ---------------------------------------------------------------------------
# Per-platform partitioning — scorer emits scores_omen.json and
# scores_polymarket.json alongside the combined file so the daily report can
# render a platform-scoped view.
# ---------------------------------------------------------------------------


class TestDerivePlatformPath:
    """Tests for _derive_platform_path."""

    def test_basic(self) -> None:
        """scores.json -> scores_omen.json in the same directory."""
        p = _derive_platform_path(Path("/tmp/results/scores.json"), "omen")
        assert p == Path("/tmp/results/scores_omen.json")

    def test_tournament_stem(self) -> None:
        """Composes with tournament suffix: scores_tournament_omen.json."""
        p = _derive_platform_path(
            Path("/tmp/results/scores_tournament.json"), "polymarket"
        )
        assert p == Path("/tmp/results/scores_tournament_polymarket.json")

    def test_period_stem(self) -> None:
        """Works for period / rolling stems too."""
        p = _derive_platform_path(Path("/tmp/results/rolling_scores.json"), "omen")
        assert p == Path("/tmp/results/rolling_scores_omen.json")


class TestPartitionRowsByPlatform:
    """Tests for _partition_rows_by_platform."""

    def test_splits_by_platform(self) -> None:
        """Rows routed to omen/polymarket buckets by row['platform']."""
        rows = [
            _row(platform="omen", row_id="o1"),
            _row(platform="polymarket", row_id="p1"),
            _row(platform="omen", row_id="o2"),
        ]
        buckets = _partition_rows_by_platform(rows)
        assert set(buckets.keys()) == set(PLATFORMS)
        assert [r["row_id"] for r in buckets["omen"]] == ["o1", "o2"]
        assert [r["row_id"] for r in buckets["polymarket"]] == ["p1"]

    def test_unknown_platform_dropped(self) -> None:
        """Rows with unknown/missing platform stay out of per-platform buckets."""
        rows = [
            _row(platform="omen", row_id="o1"),
            _row(platform="unknown_chain", row_id="x1"),
            {"row_id": "n1"},  # no platform key at all
        ]
        buckets = _partition_rows_by_platform(rows)
        assert [r["row_id"] for r in buckets["omen"]] == ["o1"]
        assert buckets["polymarket"] == []

    def test_empty_input_still_returns_all_platform_keys(self) -> None:
        """Every PLATFORMS key must be present so callers can emit files."""
        buckets = _partition_rows_by_platform([])
        assert set(buckets.keys()) == set(PLATFORMS)
        assert all(v == [] for v in buckets.values())


class TestPerPlatformRebuild:
    """Rebuild emits per-platform scores files alongside the combined file."""

    def test_rebuild_emits_per_platform_files(self, tmp_path: Path) -> None:
        """Mixed omen + polymarket rows produce three independent scores files."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        log_file = logs_dir / "production_log_2026_04_01.jsonl"
        rows = [
            _row(platform="omen", predicted_at="2026-04-01T10:00:00Z") for _ in range(4)
        ] + [
            _row(platform="polymarket", predicted_at="2026-04-01T10:00:00Z")
            for _ in range(2)
        ]
        log_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        rebuild(logs_dir, scores_path, history_path)

        combined = json.loads(scores_path.read_text())
        omen = json.loads((tmp_path / "scores_omen.json").read_text())
        poly = json.loads((tmp_path / "scores_polymarket.json").read_text())

        assert combined["overall"]["n"] == 6
        assert omen["overall"]["n"] == 4
        assert poly["overall"]["n"] == 2

    def test_rebuild_per_platform_tournament_files(self, tmp_path: Path) -> None:
        """Tournament partition writes scores_tournament_<platform>.json."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        log_file = logs_dir / "production_log_2026_04_01.jsonl"
        rows = [
            _row(
                platform="omen",
                mode="tournament",
                predicted_at="2026-04-01T10:00:00Z",
                row_id="t_omen",
            ),
            _row(
                platform="polymarket",
                mode="tournament",
                predicted_at="2026-04-01T10:00:00Z",
                row_id="t_poly",
            ),
            _row(
                platform="omen",
                mode="production_replay",
                predicted_at="2026-04-01T10:00:00Z",
                row_id="p_omen",
            ),
        ]
        log_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        rebuild(logs_dir, scores_path, history_path)

        t_omen = json.loads((tmp_path / "scores_tournament_omen.json").read_text())
        t_poly = json.loads(
            (tmp_path / "scores_tournament_polymarket.json").read_text()
        )
        prod_omen = json.loads((tmp_path / "scores_omen.json").read_text())

        assert t_omen["overall"]["n"] == 1
        assert t_poly["overall"]["n"] == 1
        # Mode partition holds: tournament rows do not leak into prod file.
        assert prod_omen["overall"]["n"] == 1

    def test_rebuild_unknown_platform_in_combined_only(self, tmp_path: Path) -> None:
        """Unknown-platform rows stay in combined; drop from per-platform files."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        log_file = logs_dir / "production_log_2026_04_01.jsonl"
        rows = [
            _row(platform="omen", predicted_at="2026-04-01T10:00:00Z"),
            _row(platform="weird_chain", predicted_at="2026-04-01T10:00:00Z"),
        ]
        log_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        rebuild(logs_dir, scores_path, history_path)

        combined = json.loads(scores_path.read_text())
        omen = json.loads((tmp_path / "scores_omen.json").read_text())
        poly = json.loads((tmp_path / "scores_polymarket.json").read_text())

        assert combined["overall"]["n"] == 2
        assert omen["overall"]["n"] == 1
        assert poly["overall"]["n"] == 0

    def test_rebuild_empty_logs_emits_empty_per_platform_files(
        self, tmp_path: Path
    ) -> None:
        """Empty log dir produces empty-but-valid per-platform files."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rebuild(logs_dir, scores_path, history_path)

        for platform in PLATFORMS:
            plat_path = tmp_path / f"scores_{platform}.json"
            assert plat_path.exists()
            assert json.loads(plat_path.read_text())["overall"]["n"] == 0
            t_plat_path = tmp_path / f"scores_tournament_{platform}.json"
            assert t_plat_path.exists()
            assert json.loads(t_plat_path.read_text())["overall"]["n"] == 0

    def test_rebuild_empty_platform_partition_still_emits_file(
        self, tmp_path: Path
    ) -> None:
        """Only-omen rows produce a valid but-empty polymarket file."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        log_file = logs_dir / "production_log_2026_04_01.jsonl"
        log_file.write_text(
            json.dumps(_row(platform="omen", predicted_at="2026-04-01T10:00:00Z"))
            + "\n"
        )

        rebuild(logs_dir, scores_path, history_path)

        poly = json.loads((tmp_path / "scores_polymarket.json").read_text())
        assert poly["overall"]["n"] == 0
        assert poly["overall"]["brier"] is None


class TestPerPlatformUpdate:
    """update() emits per-platform accumulator files + incremental merge."""

    def test_update_emits_per_platform_files(self, tmp_path: Path) -> None:
        """Mixed rows via update() write three independent accumulator files."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(platform="omen", row_id="o1"),
            _row(platform="omen", row_id="o2"),
            _row(platform="polymarket", row_id="p1"),
        ]
        update(rows, scores_path, history_path)

        combined = json.loads(scores_path.read_text())
        omen = json.loads((tmp_path / "scores_omen.json").read_text())
        poly = json.loads((tmp_path / "scores_polymarket.json").read_text())

        assert combined["overall"]["n"] == 3
        assert omen["overall"]["n"] == 2
        assert poly["overall"]["n"] == 1

    def test_update_incremental_merge_per_platform(self, tmp_path: Path) -> None:
        """Two sequential update() batches merge into per-platform accumulators."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        update(
            [_row(platform="omen", row_id="o1")],
            scores_path,
            history_path,
        )
        update(
            [
                _row(platform="omen", row_id="o2"),
                _row(platform="polymarket", row_id="p1"),
            ],
            scores_path,
            history_path,
        )

        omen = json.loads((tmp_path / "scores_omen.json").read_text())
        poly = json.loads((tmp_path / "scores_polymarket.json").read_text())
        combined = json.loads(scores_path.read_text())

        assert omen["overall"]["n"] == 2
        assert poly["overall"]["n"] == 1
        assert combined["overall"]["n"] == 3

    def test_update_dedup_applies_to_per_platform(self, tmp_path: Path) -> None:
        """Re-feeding the same rows is a no-op on per-platform accumulators too."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"
        dedup_path = tmp_path / "ids.json"

        rows = [
            _row(platform="omen", row_id="o1"),
            _row(platform="polymarket", row_id="p1"),
        ]
        update(rows, scores_path, history_path, dedup_path=dedup_path)
        update(rows, scores_path, history_path, dedup_path=dedup_path)

        omen = json.loads((tmp_path / "scores_omen.json").read_text())
        poly = json.loads((tmp_path / "scores_polymarket.json").read_text())
        assert omen["overall"]["n"] == 1
        assert poly["overall"]["n"] == 1


class TestPerPlatformLegacyRecompute:
    """Legacy ``--input`` full-recompute path writes per-platform files too.

    This CLI mode isn't used by the daily workflow but must stay at parity
    with rebuild/update/period so local dev runs produce the same artifacts.
    """

    @staticmethod
    def _write_input(input_path: Path, rows: list[dict[str, Any]]) -> None:
        """Serialize ``rows`` to a jsonl file."""
        input_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def test_legacy_full_recompute_emits_per_platform_files(
        self, tmp_path: Path
    ) -> None:
        """Mixed-platform input jsonl -> combined + per-platform scores files."""
        input_path = tmp_path / "input.jsonl"
        self._write_input(
            input_path,
            [
                _row(platform="omen", row_id="o1"),
                _row(platform="omen", row_id="o2"),
                _row(platform="polymarket", row_id="p1"),
            ],
        )
        output_path = tmp_path / "scores.json"
        tournament_path = _derive_tournament_path(output_path)

        args = argparse.Namespace(input=input_path, output=output_path)
        _cli_legacy_full_recompute(args, tournament_path)

        combined = json.loads(output_path.read_text())
        omen = json.loads((tmp_path / "scores_omen.json").read_text())
        poly = json.loads((tmp_path / "scores_polymarket.json").read_text())

        assert combined["overall"]["n"] == 3
        assert omen["overall"]["n"] == 2
        assert poly["overall"]["n"] == 1

    def test_legacy_full_recompute_unknown_platform_in_combined_only(
        self, tmp_path: Path
    ) -> None:
        """Rows on an unrecognised platform stay out of per-platform outputs."""
        input_path = tmp_path / "input.jsonl"
        self._write_input(
            input_path,
            [
                _row(platform="omen", row_id="o1"),
                _row(platform="weird_chain", row_id="x1"),
            ],
        )
        output_path = tmp_path / "scores.json"
        tournament_path = _derive_tournament_path(output_path)

        args = argparse.Namespace(input=input_path, output=output_path)
        _cli_legacy_full_recompute(args, tournament_path)

        combined = json.loads(output_path.read_text())
        omen = json.loads((tmp_path / "scores_omen.json").read_text())
        poly = json.loads((tmp_path / "scores_polymarket.json").read_text())

        assert combined["overall"]["n"] == 2
        assert omen["overall"]["n"] == 1
        assert poly["overall"]["n"] == 0


class TestPerPlatformPeriod:
    """score_period_split_by_platform returns {all, omen, polymarket}."""

    @staticmethod
    def _populate_logs(logs_dir: Path, rows: list[dict[str, Any]]) -> str:
        """Write ``rows`` into a production_log file named for today."""
        logs_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        log_file = logs_dir / f"production_log_{stamp}.jsonl"
        log_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return stamp

    def test_period_by_platform_shape(self, tmp_path: Path) -> None:
        """Return dict keys cover combined + every PLATFORMS entry."""
        logs_dir = tmp_path / "logs"
        cutoff_date = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self._populate_logs(
            logs_dir,
            [
                _row(platform="omen", predicted_at=cutoff_date, row_id="o1"),
                _row(platform="polymarket", predicted_at=cutoff_date, row_id="p1"),
            ],
        )

        result = score_period_split_by_platform(logs_dir=logs_dir, days=7)

        assert set(result.keys()) == {"all", *PLATFORMS}
        prod_all, _ = result["all"]
        prod_omen, _ = result["omen"]
        prod_poly, _ = result["polymarket"]
        assert prod_all["overall"]["n"] == 2
        assert prod_omen["overall"]["n"] == 1
        assert prod_poly["overall"]["n"] == 1

    def test_period_by_platform_empty_window(self, tmp_path: Path) -> None:
        """Rows older than the window produce zero-count per-platform results."""
        logs_dir = tmp_path / "logs"
        old_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self._populate_logs(
            logs_dir,
            [_row(platform="omen", predicted_at=old_date, row_id="o_old")],
        )

        result = score_period_split_by_platform(logs_dir=logs_dir, days=7)

        for scope in ("all", *PLATFORMS):
            prod, _ = result[scope]
            assert prod["overall"]["n"] == 0

    def test_period_keeps_rows_with_fractional_seconds(self, tmp_path: Path) -> None:
        """Fractional-second timestamps inside the window aren't dropped."""
        logs_dir = tmp_path / "logs"
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%S.123Z"
        )
        self._populate_logs(
            logs_dir,
            [_row(platform="omen", predicted_at=recent, row_id="o_frac")],
        )

        result = score_period_split_by_platform(logs_dir=logs_dir, days=7)
        prod_all, _ = result["all"]
        prod_omen, _ = result["omen"]
        assert prod_all["overall"]["n"] == 1
        assert prod_omen["overall"]["n"] == 1

    def test_period_skips_unparseable_predicted_at(self, tmp_path: Path) -> None:
        """Rows whose predicted_at is missing or malformed are excluded."""
        logs_dir = tmp_path / "logs"
        self._populate_logs(
            logs_dir,
            [
                _row(platform="omen", predicted_at="", row_id="o_empty"),
                _row(platform="omen", predicted_at="not-a-date", row_id="o_bad"),
            ],
        )

        result = score_period_split_by_platform(logs_dir=logs_dir, days=7)
        prod_all, _ = result["all"]
        assert prod_all["overall"]["n"] == 0
