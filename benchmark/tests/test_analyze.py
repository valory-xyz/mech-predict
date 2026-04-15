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
    ACTIVE_CATEGORIES,
    OMEN_CATEGORIES,
    POLYMARKET_ACTIVE_CATEGORIES,
    _parse_tvm_key,
    generate_report,
    section_best_predictions,
    section_category,
    section_latency,
    section_overall,
    section_parse_breakdown,
    section_sample_size_warnings,
    section_tool_category,
    section_tool_version_breakdown,
    section_trend,
    section_version_deltas,
    section_weak_spots,
    section_worst_predictions,
)
from benchmark.scorer import MIN_SAMPLE_SIZE

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
    by_tool_category: dict | None = None,
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
        "by_tool_category": by_tool_category or {},
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
        s = _scores(
            by_category={"technology": {"brier": 0.45, "n": 100, "reliability": 0.9}}
        )
        result = section_weak_spots(s)
        assert "weak performance" in result
        assert "anti-predictive" not in result

    def test_no_weak_spots(self) -> None:
        """Test no weak spots detected."""
        s = _scores(
            by_category={"finance": {"brier": 0.2, "n": 100, "reliability": 0.9}}
        )
        result = section_weak_spots(s)
        assert "No weak spots" in result

    def test_threshold_boundary(self) -> None:
        """Brier exactly at threshold (0.40) should NOT be flagged."""
        s = _scores(by_tool={"test": {"brier": 0.40, "n": 50, "reliability": 1.0}})
        result = section_weak_spots(s)
        assert "No weak spots" in result

    def test_legacy_category_not_flagged(self) -> None:
        """Categories no longer emitted by either platform are skipped."""
        s = _scores(
            by_category={"travel": {"brier": 0.80, "n": 100, "reliability": 0.9}}
        )
        result = section_weak_spots(s)
        assert "travel" not in result.split("_Skipped", maxsplit=1)[0]
        assert "No weak spots detected" in result

    def test_legacy_category_footnote_listed(self) -> None:
        """Skipped legacy categories are surfaced in a footnote."""
        s = _scores(
            by_category={
                "travel": {"brier": 0.80, "n": 100, "reliability": 0.9},
                "crypto": {"brier": 0.75, "n": 50, "reliability": 0.9},
                "politics": {"brier": 0.45, "n": 100, "reliability": 0.9},
            }
        )
        result = section_weak_spots(s)
        assert "politics" in result
        assert "Skipped 2 legacy category label(s)" in result
        assert "crypto" in result
        assert "travel" in result


class TestActiveCategoriesInvariants:
    """Pin the contents of ACTIVE_CATEGORIES.

    Guards against accidental edits that silently shrink the filter set.
    The behavioural weak-spots tests would not catch the mutation "remove a
    single label from OMEN_CATEGORIES or POLYMARKET_ACTIVE_CATEGORIES" on
    their own.
    """

    def test_shared_categories_are_active(self) -> None:
        """Categories emitted by both platforms must pass the filter."""
        shared = {
            "business",
            "politics",
            "science",
            "technology",
            "health",
            "entertainment",
            "weather",
            "finance",
            "international",
        }
        assert shared.issubset(ACTIVE_CATEGORIES)
        assert shared.issubset(OMEN_CATEGORIES)
        assert shared.issubset(POLYMARKET_ACTIVE_CATEGORIES)

    def test_omen_only_categories_are_active(self) -> None:
        """Categories emitted only by Omen's market creator must pass."""
        omen_only = {"cryptocurrency", "sports", "sustainability", "pets"}
        assert omen_only.issubset(ACTIVE_CATEGORIES)
        assert omen_only.isdisjoint(POLYMARKET_ACTIVE_CATEGORIES)

    def test_removed_labels_are_not_active(self) -> None:
        """Labels removed from either upstream taxonomy must not be active."""
        removed = {"travel", "crypto", "tech", "other", "economics", "fashion"}
        assert removed.isdisjoint(ACTIVE_CATEGORIES)


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


class TestSectionCategory:
    """Tests for section_category (fleet-level category performance)."""

    def test_header_present(self) -> None:
        """Always emits the Category Performance header."""
        result = section_category(_scores(by_category={}))
        assert "## Category Performance" in result

    def test_empty_says_no_data(self) -> None:
        """Explicit 'no data' when dimension is empty — do not silently skip."""
        result = section_category(_scores(by_category={}))
        assert "No per-category data available" in result

    def test_sufficient_category_rendered_with_metrics(self) -> None:
        """Categories with n >= MIN_SAMPLE_SIZE render Brier/LogLoss/Edge/BSS."""
        s = _scores(
            by_category={
                "politics": {
                    "brier": 0.22,
                    "log_loss": 0.64,
                    "baseline_brier": 0.25,
                    "brier_skill_score": 0.12,
                    "edge": -0.04,
                    "edge_n": 80,
                    "outcome_yes_rate": 0.37,
                    "n": 100,
                }
            }
        )
        result = section_category(s)
        assert "**politics**" in result
        assert "Brier: 0.22" in result
        assert "LogLoss: 0.6400" in result
        assert "BSS: +0.1200" in result
        assert "edge: -0.0400 (n=80)" in result
        assert "yes rate: 37%" in result
        assert "n=100" in result

    def test_insufficient_data_flagged_not_skipped(self) -> None:
        """Render insufficient-data line for categories below MIN_SAMPLE_SIZE.

        Do not silently omit — reader must see the dimension was considered.
        """
        s = _scores(by_category={"crypto": {"brier": 0.3, "n": 5, "reliability": 1.0}})
        result = section_category(s)
        assert "**crypto**" in result
        assert "insufficient data" in result
        assert f"need {MIN_SAMPLE_SIZE}" in result
        # A missing `continue` in the insufficient branch would emit both
        # the insufficient line AND the metric line — catch that.
        assert result.count("**crypto**") == 1
        # The sufficient-path metric line uses "Brier: X, n=..." with a
        # comma; the insufficient line embeds Brier as "noisy Brier: X"
        # without a comma. Assert the sufficient-path format does NOT appear.
        assert "Brier: 0.3, " not in result
        # But the noisy-Brier hint IS present so the sort is traceable.
        assert "noisy Brier: 0.3" in result

    def test_sorted_by_brier_ascending(self) -> None:
        """Best (lowest Brier) category appears before worst."""
        s = _scores(
            by_category={
                "bad": {"brier": 0.45, "n": 100},
                "good": {"brier": 0.15, "n": 100},
            }
        )
        result = section_category(s)
        assert result.find("**good**") < result.find("**bad**")

    def test_none_brier_sorts_last(self) -> None:
        """Categories with Brier=None sort after populated rows."""
        s = _scores(
            by_category={
                "none_brier": {"brier": None, "n": 100},
                "good": {"brier": 0.15, "n": 100},
            }
        )
        result = section_category(s)
        assert result.find("**good**") < result.find("**none_brier**")

    def test_omits_optional_fields_when_missing(self) -> None:
        """Categories without BSS/edge/yes_rate still render a line with n."""
        s = _scores(by_category={"weather": {"brier": 0.2, "n": 50}})
        result = section_category(s)
        assert "**weather**" in result
        assert "Brier: 0.2" in result
        assert "n=50" in result

    def test_homogeneous_zero_yes_rate_flagged(self) -> None:
        """Categories with yes rate 0% are flagged as one-sided.

        Mirrors the base-rate guard in notify_slack.py so readers of the
        raw markdown see the same warning the Slack LLM gets — a low
        Brier on a homogeneous category reflects the base rate, not
        predictive skill.
        """
        s = _scores(
            by_category={
                "tech": {"brier": 0.05, "n": 180, "outcome_yes_rate": 0.0},
            }
        )
        result = section_category(s)
        assert "⚠ **tech**" in result
        assert "one-sided outcomes; Brier not meaningful here" in result

    def test_homogeneous_full_yes_rate_flagged(self) -> None:
        """Categories with yes rate 100% get the same one-sided flag."""
        s = _scores(
            by_category={
                "health": {"brier": 0.05, "n": 88, "outcome_yes_rate": 1.0},
            }
        )
        result = section_category(s)
        assert "⚠ **health**" in result
        assert "one-sided outcomes; Brier not meaningful here" in result

    def test_mixed_outcomes_not_flagged(self) -> None:
        """Non-homogeneous categories render without the ⚠ marker or tail.

        Near-homogeneous values (0.01, 0.99) are NOT one-sided — there
        are real mixed outcomes and Brier is still meaningful.
        """
        s = _scores(
            by_category={
                "business": {"brier": 0.15, "n": 986, "outcome_yes_rate": 0.10},
                "edge_case": {"brier": 0.01, "n": 100, "outcome_yes_rate": 0.01},
            }
        )
        result = section_category(s)
        assert "⚠" not in result
        assert "one-sided outcomes" not in result
        assert "**business**" in result
        assert "**edge_case**" in result

    def test_missing_yes_rate_not_flagged(self) -> None:
        """Categories without outcome_yes_rate are not flagged as homogeneous."""
        s = _scores(by_category={"weather": {"brier": 0.2, "n": 50}})
        result = section_category(s)
        assert "⚠" not in result
        assert "one-sided outcomes" not in result


class TestSectionToolCategory:
    """Tests for section_tool_category (fleet × category cross-breakdown)."""

    def test_empty_returns_no_data(self) -> None:
        """Empty dimension returns an explicit no-data message."""
        result = section_tool_category(_scores(by_tool_category={}))
        assert "## Tool × Category" in result
        assert "No cross-breakdown data" in result

    def test_sufficient_cell_rendered_in_table(self) -> None:
        """Cells with n >= MIN_SAMPLE_SIZE appear in the ranked table."""
        s = _scores(
            by_tool_category={
                "tool-a | politics": {
                    "brier": 0.19,
                    "brier_skill_score": 0.05,
                    "log_loss": 0.60,
                    "edge": -0.03,
                    "edge_n": 40,
                    "directional_accuracy": 0.76,
                    "sharpness": 0.12,
                    "n": 60,
                    "decision_worthy": True,
                }
            }
        )
        result = section_tool_category(s)
        assert (
            "| tool-a | politics | 0.1900 | +0.0500 | 0.6000 | -0.0300 | 40 | 76% | 0.1200 | 60 |"
            in result
        )

    def test_sparse_cell_listed_in_sparse_section_not_table(self) -> None:
        """Route sparse cells to the list-only path, never the ranking table.

        A bug flipping the gate would flip which section a cell lands in,
        so both assertions below together catch it.
        """
        s = _scores(
            by_tool_category={
                "tool-a | crypto": {
                    "brier": 0.10,
                    "n": 5,
                    "decision_worthy": False,
                }
            }
        )
        result = section_tool_category(s)
        # Not in the ranking table body (between header and sparse marker)
        pre_sparse, _, post_sparse = result.partition("below n=")
        assert "| tool-a | crypto | 0.1000" not in pre_sparse
        # But listed in the sparse section with insufficient-data marker
        assert "insufficient data (n=5)" in post_sparse

    def test_all_sparse_shows_placeholder_row(self) -> None:
        """Show an explicit placeholder row when no cell meets the threshold.

        Rendering an empty table instead would be confusing.
        """
        s = _scores(
            by_tool_category={
                "tool-a | x": {"brier": 0.1, "n": 2},
                "tool-b | y": {"brier": 0.2, "n": 3},
            }
        )
        result = section_tool_category(s)
        assert f"no cells with n ≥ {MIN_SAMPLE_SIZE}" in result
        assert "2 cell(s) below" in result

    def test_threshold_boundary(self) -> None:
        """A cell with exactly n = MIN_SAMPLE_SIZE is included (gate is >=)."""
        s = _scores(
            by_tool_category={
                "tool-a | politics": {
                    "brier": 0.2,
                    "n": MIN_SAMPLE_SIZE,
                    "decision_worthy": True,
                }
            }
        )
        result = section_tool_category(s)
        assert "| tool-a | politics | 0.2000" in result
        assert "below n=" not in result  # no sparse section

    def test_sparse_examples_capped_at_five(self) -> None:
        """Cap rendered sparse examples at 5 while reporting the true total.

        Keeps the table readable while still signaling how many cells were
        considered.
        """
        sparse_cells = {
            f"tool-{i} | cat-{i}": {"brier": 0.1 + i * 0.01, "n": 5} for i in range(7)
        }
        s = _scores(by_tool_category=sparse_cells)
        result = section_tool_category(s)
        assert "7 cell(s) below" in result  # true total
        rendered = sum(
            1
            for line in result.splitlines()
            if line.startswith("- **tool-") and "insufficient data" in line
        )
        assert rendered == 5

    def test_table_ranked_by_brier_ascending(self) -> None:
        """Best cell ranks above worst."""
        s = _scores(
            by_tool_category={
                "tool-a | good": {
                    "brier": 0.1,
                    "n": 50,
                    "decision_worthy": True,
                },
                "tool-b | bad": {
                    "brier": 0.4,
                    "n": 50,
                    "decision_worthy": True,
                },
            }
        )
        result = section_tool_category(s)
        assert result.find("tool-a | good") < result.find("tool-b | bad")


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
        assert "## Category Performance" in report
        assert "## Tool × Category" in report
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


# ---------------------------------------------------------------------------
# _parse_tvm_key
# ---------------------------------------------------------------------------


class TestParseTvmKey:
    """Tests for _parse_tvm_key."""

    def test_three_parts(self) -> None:
        """Standard tool | version | mode key splits cleanly."""
        assert _parse_tvm_key("tool-a | bafy_v1 | tournament") == (
            "tool-a",
            "bafy_v1",
            "tournament",
        )

    def test_pads_missing_mode(self) -> None:
        """Legacy two-part keys get unknown mode."""
        assert _parse_tvm_key("tool-a | bafy_v1") == (
            "tool-a",
            "bafy_v1",
            "unknown",
        )

    def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace around | is stripped."""
        tool, version, mode = _parse_tvm_key("tool-a|bafy_v1|tournament")
        assert (tool, version, mode) == ("tool-a", "bafy_v1", "tournament")


# ---------------------------------------------------------------------------
# Tool × version × mode breakdown
# ---------------------------------------------------------------------------


class TestSectionToolVersionBreakdown:
    """Tests for section_tool_version_breakdown."""

    def test_empty_returns_blank(self) -> None:
        """Missing/empty by_tool_version_mode returns empty string."""
        assert section_tool_version_breakdown({}) == ""
        assert section_tool_version_breakdown({"by_tool_version_mode": {}}) == ""

    def test_renders_table_with_high_n_no_warning(self) -> None:
        """Cells with n >= 30 render without ⚠ marker."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | bafy_v1 | production_replay": {
                    "n": 500,
                    "valid_n": 480,
                    "brier": 0.21,
                    "directional_accuracy": 0.7,
                    "brier_skill_score": 0.05,
                },
            }
        }
        result = section_tool_version_breakdown(scores)
        assert "tool-a" in result
        assert "`bafy_v1`" in result
        assert "production_replay" in result
        assert "500" in result and "0.2100" in result
        assert "⚠" not in result  # high-n row should not be flagged

    def test_low_n_row_gets_warning_marker(self) -> None:
        """Cells with n < 30 carry a ⚠ marker on the n cell and a footnote."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | bafy_v1 | tournament": {
                    "n": 9,
                    "valid_n": 9,
                    "brier": 0.18,
                    "directional_accuracy": 0.78,
                    "brier_skill_score": 0.10,
                },
            }
        }
        result = section_tool_version_breakdown(scores)
        assert "9 ⚠" in result  # per-row marker
        assert "n < 30" in result  # footnote warning

    def test_custom_title(self) -> None:
        """Title argument controls the section heading."""
        scores = {
            "by_tool_version_mode": {"t | v | m": {"n": 5, "valid_n": 5, "brier": 0.2}}
        }
        result = section_tool_version_breakdown(scores, title="Last 7 Days")
        assert result.startswith("## Last 7 Days")


# ---------------------------------------------------------------------------
# section_version_deltas
# ---------------------------------------------------------------------------


class TestSectionVersionDeltas:
    """Tests for section_version_deltas."""

    def test_empty_returns_blank(self) -> None:
        """Missing data returns empty string."""
        assert section_version_deltas({}) == ""

    def test_single_version_per_tool_returns_blank(self) -> None:
        """Tools with only one (version, mode) cell yield no delta section."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | v1 | production_replay": {"n": 100, "brier": 0.2},
            }
        }
        assert section_version_deltas(scores) == ""

    def test_renders_pairwise_delta_for_multi_version_tool(self) -> None:
        """Tool with two versions produces a delta row with direction."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | v1 | production_replay": {
                    "n": 100,
                    "valid_n": 100,
                    "brier": 0.30,
                    "directional_accuracy": 0.6,
                    "log_loss": 0.7,
                    "sharpness": 0.3,
                    "reliability": 1.0,
                },
                "tool-a | v2 | production_replay": {
                    "n": 100,
                    "valid_n": 100,
                    "brier": 0.20,
                    "directional_accuracy": 0.7,
                    "log_loss": 0.6,
                    "sharpness": 0.3,
                    "reliability": 1.0,
                },
            }
        }
        result = section_version_deltas(scores)
        assert "## Version Deltas" in result
        assert "### tool-a" in result
        assert "improved" in result  # v2 has lower Brier
        assert "-0.1000" in result

    def test_low_sample_marker_on_delta(self) -> None:
        """Delta where either side has n < 30 carries the ⚠ marker."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | v1 | production_replay": {
                    "n": 1000,
                    "valid_n": 1000,
                    "brier": 0.30,
                    "directional_accuracy": 0.6,
                    "log_loss": 0.7,
                    "sharpness": 0.3,
                    "reliability": 1.0,
                },
                "tool-a | v2 | tournament": {
                    "n": 9,
                    "valid_n": 9,
                    "brier": 0.20,
                    "directional_accuracy": 0.7,
                    "log_loss": 0.6,
                    "sharpness": 0.3,
                    "reliability": 1.0,
                },
            }
        }
        result = section_version_deltas(scores)
        assert "⚠" in result
        assert "n_b" in result  # column header preserved


# ---------------------------------------------------------------------------
# generate_report — include_tournament toggle
# ---------------------------------------------------------------------------


class TestGenerateReportTournamentToggle:
    """Tests for the include_tournament flag on generate_report."""

    def _scores_with_versions(self) -> dict[str, Any]:
        s = _scores(
            by_tool={"tool-a": {"brier": 0.3, "n": 50, "reliability": 1.0}},
        )
        s["by_tool_version_mode"] = {
            "tool-a | v1 | production_replay": {
                "n": 100,
                "valid_n": 100,
                "brier": 0.30,
                "directional_accuracy": 0.6,
                "log_loss": 0.7,
                "sharpness": 0.3,
                "reliability": 1.0,
            },
            "tool-a | v2 | tournament": {
                "n": 50,
                "valid_n": 50,
                "brier": 0.20,
                "directional_accuracy": 0.7,
                "log_loss": 0.6,
                "sharpness": 0.3,
                "reliability": 1.0,
            },
        }
        return s

    def test_off_by_default_omits_tournament_sections(self) -> None:
        """Default behavior: no tournament sections in the rendered report."""
        s = self._scores_with_versions()
        report = generate_report(s, [])
        assert "Tool × Version × Mode" not in report
        assert "Version Deltas" not in report

    def test_on_renders_cumulative_breakdown_and_deltas(self) -> None:
        """include_tournament=True renders the cumulative breakdown.

        Version Deltas is temporarily disabled pending rework.
        """
        s = self._scores_with_versions()
        report = generate_report(s, [], include_tournament=True)
        assert "Tool × Version × Mode (All-Time)" in report
        assert "## Version Deltas" not in report

    def test_rolling_scores_render_separate_section(self) -> None:
        """When rolling_scores has version cells, a 7d section appears too."""
        s = self._scores_with_versions()
        rolling = _scores()
        rolling["by_tool_version_mode"] = {
            "tool-a | v2 | tournament": {
                "n": 35,
                "valid_n": 35,
                "brier": 0.18,
                "directional_accuracy": 0.8,
                "brier_skill_score": 0.1,
            }
        }
        report = generate_report(s, [], rolling_scores=rolling, include_tournament=True)
        assert "Tool × Version × Mode (All-Time)" in report
        assert "Tool × Version × Mode (Last 7 Days)" in report


# ---------------------------------------------------------------------------
# Mode split: tournament sections and callouts (BENCHMARK_MODE_SPLIT_SPEC)
# ---------------------------------------------------------------------------


def _scores_with_tool(
    tool: str,
    brier: float,
    n: int,
    valid: int | None = None,
    baseline: float = 0.25,
) -> dict[str, Any]:
    """Build a production scores dict with one by_tool entry."""
    valid_n = n if valid is None else valid
    return {
        "generated_at": "2026-03-31T06:00:00Z",
        "total_rows": n,
        "valid_rows": valid_n,
        "overall": {"brier": brier, "reliability": 0.95, "n": n},
        "by_tool": {
            tool: {
                "brier": brier,
                "baseline_brier": baseline,
                "n": n,
                "valid_n": valid_n,
                "reliability": 0.95,
                "directional_accuracy": 0.7,
                "brier_skill_score": 0.0,
            }
        },
        "by_platform": {},
        "by_category": {},
        "by_horizon": {},
        "by_tool_platform": {},
        "by_tool_version_mode": {},
        "calibration": [],
        "worst_10": [],
        "best_10": [],
        "parse_breakdown": {},
        "latency_reservoir": {},
    }


def _tournament_scores_with_version(
    tool: str,
    version: str,
    brier: float,
    n: int,
) -> dict[str, Any]:
    """Build a tournament scores dict with one by_tool_version_mode cell."""
    s = _scores_with_tool(tool, brier, n)
    s["by_tool_version_mode"] = {
        f"{tool} | {version} | tournament": {
            "brier": brier,
            "n": n,
            "valid_n": n,
            "directional_accuracy": 0.75,
            "brier_skill_score": 0.1,
        }
    }
    return s


class TestTournamentCallouts:
    """Tests for section_tournament_callouts."""

    def test_empty_when_no_tournament_data(self) -> None:
        """Missing tournament scores returns empty string."""
        # pylint: disable=import-outside-toplevel
        from benchmark.analyze import section_tournament_callouts

        prod = _scores_with_tool("tool-a", 0.20, 1000)
        assert section_tournament_callouts(prod, None) == ""
        assert section_tournament_callouts(prod, {"total_rows": 0}) == ""

    def test_promotion_candidate_flagged(self) -> None:
        """Tournament Brier meaningfully lower than production triggers a promotion bullet."""
        # pylint: disable=import-outside-toplevel
        from benchmark.analyze import section_tournament_callouts

        prod = _scores_with_tool("tool-a", 0.20, 1000)
        tourn = _tournament_scores_with_version("tool-a", "v2", 0.10, 50)
        result = section_tournament_callouts(prod, tourn)
        assert "Promotion candidates:" in result
        assert "tool-a" in result
        assert "v2" in result
        assert "Tournament regressions:" not in result

    def test_tournament_regression_flagged(self) -> None:
        """Tournament Brier meaningfully higher than production triggers a regression bullet."""
        # pylint: disable=import-outside-toplevel
        from benchmark.analyze import section_tournament_callouts

        prod = _scores_with_tool("tool-a", 0.20, 1000)
        tourn = _tournament_scores_with_version("tool-a", "v2", 0.40, 50)
        result = section_tournament_callouts(prod, tourn)
        assert "Tournament regressions:" in result
        assert "Promotion candidates:" not in result

    def test_suppressed_below_min_n(self) -> None:
        """Tournament cells with n below CALLOUT_MIN_N are ignored."""
        # pylint: disable=import-outside-toplevel
        from benchmark.analyze import section_tournament_callouts

        prod = _scores_with_tool("tool-a", 0.20, 1000)
        tourn = _tournament_scores_with_version("tool-a", "v2", 0.05, 10)
        assert section_tournament_callouts(prod, tourn) == ""

    def test_suppressed_within_delta_band(self) -> None:
        """Tournament vs production deltas within CALLOUT_DELTA are not flagged."""
        # pylint: disable=import-outside-toplevel
        from benchmark.analyze import section_tournament_callouts

        prod = _scores_with_tool("tool-a", 0.20, 1000)
        tourn = _tournament_scores_with_version("tool-a", "v2", 0.21, 100)
        assert section_tournament_callouts(prod, tourn) == ""


class TestGenerateReportWithTournamentFiles:
    """Tests for generate_report dual-mode rendering."""

    def test_tournament_sections_omitted_when_file_absent(self) -> None:
        """No tournament inputs -> no '— Tournament' headings, no callouts."""
        prod = _scores_with_tool("tool-a", 0.20, 1000)
        report = generate_report(prod, [], include_tournament=True)
        assert "— Tournament" not in report
        assert "## Tournament Callouts" not in report

    def test_tournament_sections_rendered_when_data_present(self) -> None:
        """Tournament inputs with rows -> duplicated per-mode headings render."""
        prod = _scores_with_tool("tool-a", 0.20, 1000)
        tourn = _tournament_scores_with_version("tool-a", "v2", 0.18, 100)
        report = generate_report(
            prod,
            [],
            include_tournament=True,
            scores_tournament=tourn,
        )
        assert "## Overall — Tournament" in report
        assert "## Tool Ranking — Tournament" in report

    def test_empty_period_tournament_does_not_render_section(self) -> None:
        """Tournament period files with zero rows don't emit empty sections.

        scorer.score_period_split writes tournament period files on every
        run; days with zero tournament rows in the window must not add a
        dangling '## Since Last Report — Tournament' header.
        """
        prod = _scores_with_tool("tool-a", 0.20, 1000)
        all_time_tourn = _tournament_scores_with_version("tool-a", "v2", 0.18, 100)
        empty_period_tourn = {"total_rows": 0, "overall": {}}
        report = generate_report(
            prod,
            [],
            period_scores=None,
            rolling_scores=None,
            include_tournament=True,
            scores_tournament=all_time_tourn,
            period_scores_tournament=empty_period_tourn,
            rolling_scores_tournament=empty_period_tourn,
        )
        assert "## Since Last Report — Tournament" not in report
        assert "## Last 7 Days Rolling — Tournament" not in report

    def test_merged_tool_version_mode_includes_both_modes(self) -> None:
        """Tool × Version × Mode table shows both production and tournament cells."""
        prod = _scores_with_tool("tool-a", 0.20, 1000)
        prod["by_tool_version_mode"] = {
            "tool-a | v1 | production_replay": {
                "n": 500,
                "valid_n": 500,
                "brier": 0.20,
                "directional_accuracy": 0.7,
                "brier_skill_score": 0.0,
            }
        }
        tourn = _tournament_scores_with_version("tool-a", "v2", 0.18, 100)
        report = generate_report(
            prod,
            [],
            include_tournament=True,
            scores_tournament=tourn,
        )
        assert "v1" in report
        assert "v2" in report

    def test_callout_section_included_when_triggered(self) -> None:
        """Promotion-worthy tournament data causes the callouts section to render."""
        prod = _scores_with_tool("tool-a", 0.20, 1000)
        tourn = _tournament_scores_with_version("tool-a", "v2", 0.10, 50)
        report = generate_report(
            prod,
            [],
            include_tournament=True,
            scores_tournament=tourn,
        )
        assert "## Tournament Callouts" in report

    def test_callout_section_omitted_when_empty(self) -> None:
        """Tournament data present but within delta band -> no callout section."""
        prod = _scores_with_tool("tool-a", 0.20, 1000)
        tourn = _tournament_scores_with_version("tool-a", "v2", 0.21, 100)
        report = generate_report(
            prod,
            [],
            include_tournament=True,
            scores_tournament=tourn,
        )
        assert "## Tournament Callouts" not in report
