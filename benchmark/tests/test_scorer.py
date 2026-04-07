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

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from benchmark.scorer import (
    LATENCY_RESERVOIR_SIZE,
    WORST_BEST_SIZE,
    brier_score,
    classify_horizon,
    compute_group_stats,
    group_by,
    group_by_horizon,
    group_by_month,
    load_history,
    rebuild,
    score,
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
) -> dict[str, Any]:
    """Build a minimal production_log row for testing."""
    return {
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
    }


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
        assert inc_result["overall"]["accuracy"] == full_result["overall"]["accuracy"]
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
