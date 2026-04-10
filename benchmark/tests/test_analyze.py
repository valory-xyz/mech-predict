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
"""Tests for benchmark/analyze.py"""

from typing import Any

from benchmark.analyze import (
    generate_report,
    section_best_predictions,
    section_latency,
    section_overall,
    section_parse_breakdown,
    section_sample_size_warnings,
    section_trend,
    section_weak_spots,
    section_worst_predictions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scores(
    brier: float | None = 0.3,
    reliability: float | None = 0.95,
    total: int = 100,
    valid: int = 95,
    by_tool: dict | None = None,
    by_platform: dict | None = None,
    by_category: dict | None = None,
    worst_10: list | None = None,
    best_10: list | None = None,
    parse_breakdown: dict | None = None,
    latency_reservoir: dict | None = None,
) -> dict[str, Any]:
    """Build a minimal scores dict for testing."""
    return {
        "generated_at": "2026-03-31T06:00:00Z",
        "total_rows": total,
        "valid_rows": valid,
        "overall": {"brier": brier, "reliability": reliability, "n": total},
        "by_tool": by_tool or {},
        "by_platform": by_platform or {},
        "by_category": by_category or {},
        "by_horizon": {},
        "by_tool_platform": {},
        "calibration": [],
        "worst_10": worst_10 or [],
        "best_10": best_10 or [],
        "parse_breakdown": parse_breakdown or {},
        "latency_reservoir": latency_reservoir or {},
    }


# ---------------------------------------------------------------------------
# section_overall
# ---------------------------------------------------------------------------


class TestSectionOverall:
    """Tests for section_overall."""

    def test_normal(self) -> None:
        """Test normal scores with valid brier and reliability."""
        result = section_overall(_scores(brier=0.31, reliability=0.95))
        assert "0.31" in result
        assert "95%" in result

    def test_empty_dataset(self) -> None:
        """Test empty dataset with no predictions."""
        result = section_overall(
            _scores(brier=None, reliability=None, total=0, valid=0)
        )
        assert "No predictions to score" in result

    def test_all_invalid(self) -> None:
        """Test all invalid predictions."""
        result = section_overall(_scores(brier=None, reliability=0.0, total=5, valid=0))
        assert "N/A" in result  # Brier is N/A


# ---------------------------------------------------------------------------
# section_weak_spots
# ---------------------------------------------------------------------------


class TestSectionWeakSpots:
    """Tests for section_weak_spots."""

    def test_anti_predictive_label(self) -> None:
        """Brier > 0.5 should say 'anti-predictive'."""
        s = _scores(
            by_category={"social": {"brier": 0.81, "n": 100, "reliability": 0.9}}
        )
        result = section_weak_spots(s)
        assert "anti-predictive" in result

    def test_weak_performance_label(self) -> None:
        """Brier between 0.4 and 0.5 should say 'weak performance'."""
        s = _scores(by_category={"tech": {"brier": 0.45, "n": 100, "reliability": 0.9}})
        result = section_weak_spots(s)
        assert "weak performance" in result
        assert "anti-predictive" not in result

    def test_no_weak_spots(self) -> None:
        """Test no weak spots detected."""
        s = _scores(
            by_category={"crypto": {"brier": 0.2, "n": 100, "reliability": 0.9}}
        )
        result = section_weak_spots(s)
        assert "No weak spots" in result

    def test_threshold_boundary(self) -> None:
        """Brier exactly at threshold (0.40) should NOT be flagged."""
        s = _scores(by_tool={"test": {"brier": 0.40, "n": 50, "reliability": 1.0}})
        result = section_weak_spots(s)
        assert "No weak spots" in result


# ---------------------------------------------------------------------------
# section_trend
# ---------------------------------------------------------------------------


class TestSectionTrend:
    """Tests for section_trend."""

    def test_worsening_alert(self) -> None:
        """Test worsening trend triggers alert."""
        history = [
            {"month": "2026-01", "overall": {"brier": 0.20, "n": 50}},
            {"month": "2026-02", "overall": {"brier": 0.25, "n": 60}},
        ]
        result = section_trend(history)
        assert "Warning" in result

    def test_no_alert_when_stable(self) -> None:
        """Test stable trend produces no alert."""
        history = [
            {"month": "2026-01", "overall": {"brier": 0.20, "n": 50}},
            {"month": "2026-02", "overall": {"brier": 0.21, "n": 60}},
        ]
        result = section_trend(history)
        assert "Warning" not in result

    def test_empty_trend(self) -> None:
        """Test empty trend data."""
        result = section_trend([])
        assert "No trend data" in result

    def test_current_month_appended(self) -> None:
        """Current month from scores.json appears in trend."""
        scores = _scores(brier=0.28)
        scores["current_month"] = "2026-04"
        result = section_trend([], scores)
        assert "2026-04" in result
        assert "in progress" in result
        assert "No trend data" not in result

    def test_empty_history_with_scores(self) -> None:
        """First run: no history but current month still shows."""
        scores = _scores(brier=0.30)
        scores["current_month"] = "2026-04"
        result = section_trend([], scores)
        assert "2026-04" in result


# ---------------------------------------------------------------------------
# section_sample_size_warnings
# ---------------------------------------------------------------------------


class TestSectionSampleSizeWarnings:
    """Tests for section_sample_size_warnings."""

    def test_small_category_warned(self) -> None:
        """Test small category triggers warning."""
        s = _scores(by_category={"weather": {"brier": 0.3, "n": 4, "reliability": 1.0}})
        result = section_sample_size_warnings(s)
        assert "weather" in result
        assert "4 questions" in result

    def test_large_category_not_warned(self) -> None:
        """Test large category produces no warning."""
        s = _scores(
            by_category={"crypto": {"brier": 0.3, "n": 200, "reliability": 1.0}}
        )
        result = section_sample_size_warnings(s)
        assert "sufficient sample size" in result


# ---------------------------------------------------------------------------
# section_worst_predictions / section_best_predictions
# ---------------------------------------------------------------------------


class TestSectionWorstPredictions:
    """Tests for section_worst_predictions."""

    def test_renders_entries(self) -> None:
        """Test rendering worst prediction entries."""
        s = _scores(
            worst_10=[
                {
                    "question_text": "Will X happen?",
                    "tool_name": "tool-a",
                    "p_yes": 0.95,
                    "final_outcome": False,
                    "brier": 0.9025,
                    "platform": "omen",
                    "category": "finance",
                },
            ]
        )
        result = section_worst_predictions(s)
        assert "Will X happen?" in result
        assert "0.9025" in result
        assert "tool-a" in result

    def test_empty(self) -> None:
        """Test empty worst predictions."""
        result = section_worst_predictions(_scores())
        assert "No prediction data" in result


class TestSectionBestPredictions:
    """Tests for section_best_predictions."""

    def test_renders_entries(self) -> None:
        """Test rendering best prediction entries."""
        s = _scores(
            best_10=[
                {
                    "question_text": "Will Y happen?",
                    "tool_name": "tool-b",
                    "p_yes": 0.98,
                    "final_outcome": True,
                    "brier": 0.0004,
                    "platform": "polymarket",
                    "category": "politics",
                },
            ]
        )
        result = section_best_predictions(s)
        assert "Will Y happen?" in result
        assert "0.0004" in result

    def test_empty(self) -> None:
        """Test empty best predictions."""
        result = section_best_predictions(_scores())
        assert "No prediction data" in result


# ---------------------------------------------------------------------------
# Parse breakdown
# ---------------------------------------------------------------------------


class TestSectionParseBreakdown:
    """Tests for section_parse_breakdown."""

    def test_renders_table(self) -> None:
        """Test rendering parse breakdown table."""
        s = _scores(
            parse_breakdown={
                "tool-a": {"valid": 90, "malformed": 5, "error": 5},
            }
        )
        result = section_parse_breakdown(s)
        assert "tool-a" in result
        assert "90" in result

    def test_empty(self) -> None:
        """Test empty parse breakdown."""
        result = section_parse_breakdown(_scores())
        assert "No parse data" in result


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


class TestSectionLatency:
    """Tests for section_latency."""

    def test_renders_table(self) -> None:
        """Test rendering latency table."""
        s = _scores(
            latency_reservoir={
                "tool-a": [10, 12, 15, 20, 25, 30, 8, 11, 14, 18],
            }
        )
        result = section_latency(s)
        assert "tool-a" in result
        assert "Median" in result

    def test_empty(self) -> None:
        """Test empty latency data."""
        result = section_latency(_scores())
        assert "No latency data" in result


# ---------------------------------------------------------------------------
# Generate report (integration)
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for generate_report."""

    def test_has_all_sections(self) -> None:
        """Test report contains all expected sections."""
        s = _scores(
            by_tool={"tool-a": {"brier": 0.3, "n": 50, "reliability": 1.0}},
            by_platform={"omen": {"brier": 0.4, "n": 50, "reliability": 1.0}},
            by_category={"crypto": {"brier": 0.2, "n": 50, "reliability": 1.0}},
            worst_10=[
                {
                    "question_text": "Will X?",
                    "tool_name": "tool-a",
                    "p_yes": 0.9,
                    "final_outcome": True,
                    "brier": 0.01,
                    "platform": "omen",
                    "category": "crypto",
                }
            ],
            best_10=[
                {
                    "question_text": "Will Y?",
                    "tool_name": "tool-a",
                    "p_yes": 0.1,
                    "final_outcome": False,
                    "brier": 0.01,
                    "platform": "omen",
                    "category": "crypto",
                }
            ],
        )
        history = [{"month": "2026-03", "overall": {"brier": 0.3, "n": 50}}]
        report = generate_report(s, history)

        assert "# Benchmark Report" in report
        assert "## Overall" in report
        assert "## Tool Ranking" in report
        assert "## Platform Comparison" in report
        assert "## Weak Spots" in report
        assert "## Reliability Issues" in report
        assert "## Worst Predictions" in report
        assert "## Best Predictions" in report
        assert "## Trend" in report
        assert "## Sample Size Warnings" in report
        assert "## Diagnostic Edge Metrics" in report

    def test_empty_data_no_crash(self) -> None:
        """Test empty data does not crash."""
        s = _scores(brier=None, reliability=None, total=0, valid=0)
        report = generate_report(s, [])
        assert "# Benchmark Report" in report
        assert "No predictions to score" in report
