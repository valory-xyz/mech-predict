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
    classify_difficulty,
    classify_horizon,
    classify_liquidity,
    compute_group_stats,
    edge_score,
    group_by,
    group_by_horizon,
    group_by_month,
    load_history,
    rebuild,
    score,
    update,
    _is_edge_eligible,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ROW_COUNTER = 0


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
) -> dict[str, Any]:
    """Build a minimal production_log row for testing."""
    global _ROW_COUNTER
    if row_id is None:
        _ROW_COUNTER += 1
        row_id = f"test_row_{_ROW_COUNTER}"
    return {
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
        """Near boundary: |0.8-0.5|≈0.3 falls to easy due to float precision."""
        # abs(0.8 - 0.5) = 0.30000000000000004 > 0.3 → easy
        assert classify_difficulty(0.8) == "easy"
        # 0.79 is clearly medium
        assert classify_difficulty(0.79) == "medium"


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
            _row(p_yes=0.9, outcome=True, market_prob=0.6),   # edge > 0
            _row(p_yes=0.3, outcome=False, market_prob=0.4),   # edge > 0
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
            _row(p_yes=0.7, outcome=True),                    # not eligible
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
        assert stats_without["accuracy"] == stats_with["accuracy"]


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
                "accuracy": 0.67,
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
        assert elig["exclusion_reasons"]["missing_market_prob"] == 1

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
# Dedup in update()
# ---------------------------------------------------------------------------


class TestUpdateDedup:
    """Tests for row_id deduplication in update()."""

    def test_duplicate_rows_not_double_counted(self, tmp_path: Path) -> None:
        """Same rows passed twice produce same result as single pass."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [
            _row(p_yes=0.7, outcome=True, row_id="r1"),
            _row(p_yes=0.3, outcome=False, row_id="r2"),
        ]

        # First pass
        result1 = update(rows, scores_path, history_path)
        assert result1["overall"]["n"] == 2

        # Second pass with same rows — should be skipped
        result2 = update(rows, scores_path, history_path)
        assert result2["overall"]["n"] == 2, (
            f"Expected 2 after dedup, got {result2['overall']['n']}"
        )
        assert result2["overall"]["brier"] == result1["overall"]["brier"]

    def test_new_rows_added_after_dedup(self, tmp_path: Path) -> None:
        """New rows are added, duplicates are skipped."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        batch1 = [_row(p_yes=0.7, outcome=True, row_id="r1")]
        batch2 = [
            _row(p_yes=0.7, outcome=True, row_id="r1"),   # duplicate
            _row(p_yes=0.4, outcome=False, row_id="r3"),   # new
        ]

        update(batch1, scores_path, history_path)
        result = update(batch2, scores_path, history_path)

        assert result["overall"]["n"] == 2, (
            f"Expected 2 (r1 + r3), got {result['overall']['n']}"
        )

    def test_scored_row_ids_persisted(self, tmp_path: Path) -> None:
        """scored_row_ids are saved to scores.json and loaded on resume."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        rows = [_row(p_yes=0.7, outcome=True, row_id="r1")]
        update(rows, scores_path, history_path)

        # Check the saved file
        saved = json.loads(scores_path.read_text())
        assert "scored_row_ids" in saved
        assert "r1" in saved["scored_row_ids"]

    def test_rows_without_row_id_always_counted(self, tmp_path: Path) -> None:
        """Rows without row_id are always accumulated (no dedup possible)."""
        scores_path = tmp_path / "scores.json"
        history_path = tmp_path / "history.jsonl"

        row_no_id = _row(p_yes=0.7, outcome=True, row_id="")
        row_no_id.pop("row_id")  # remove the key entirely

        update([row_no_id], scores_path, history_path)
        result = update([row_no_id], scores_path, history_path)
        # Without row_id, can't dedup — both passes count
        assert result["overall"]["n"] == 2
