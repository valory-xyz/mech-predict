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
    SAMPLE_SIZE_WARNING,
    _parse_tvm_key,
    _sample_label,
    generate_report,
    section_best_predictions,
    section_latency,
    section_overall,
    section_parse_breakdown,
    section_period,
    section_sample_size_warnings,
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

    def test_no_signal_rate_small_value_renders_two_decimals(self) -> None:
        """No-signal rate formats with 2 decimal places.

        Previously rendered as ':.0%' which rounded tiny-but-nonzero
        rates like 0.00072 to '0%' even though the count shown next to
        it was clearly positive. Two decimals surface values in the
        0.01%-1% range where the no-signal rate typically lives.
        """
        s = _scores(brier=0.3, reliability=0.95)
        s["overall"]["no_signal_rate"] = 0.00072
        s["overall"]["no_signal_count"] = 142
        result = section_overall(s)
        assert "0.07%" in result
        assert "142 predictions at exactly 0.5" in result
        # Make sure the rendered value is not the stale '0%' form.
        assert "No-signal rate: 0%" not in result


# ---------------------------------------------------------------------------
# _sample_label
# ---------------------------------------------------------------------------


class TestSampleLabel:
    """Tests for _sample_label.

    Guards against the regression where tools with a large total n but
    zero valid parses were labelled 'low sample' (misleading: the real
    problem is that every row was malformed, not that the sample was
    too small).
    """

    def test_low_sample_below_gate(self) -> None:
        """Stats with decision_worthy=False render 'low sample'."""
        assert (
            _sample_label({"n": 5, "valid_n": 5, "decision_worthy": False})
            == " ⚠ low sample"
        )

    def test_all_malformed_large_n(self) -> None:
        """Large n with valid_n == 0 renders 'all malformed', not 'low sample'.

        The malformed branch short-circuits before ``decision_worthy``
        is consulted because the scorer sets ``decision_worthy=False``
        whenever valid_n is below the gate, which would otherwise hide
        the more specific pipeline-failure signal.
        """
        assert (
            _sample_label({"n": 55, "valid_n": 0, "decision_worthy": False})
            == " ⚠ all malformed"
        )

    def test_sufficient_returns_empty(self) -> None:
        """decision_worthy=True renders no label."""
        assert _sample_label({"n": 100, "valid_n": 80, "decision_worthy": True}) == ""

    def test_small_n_all_malformed_falls_through_to_low_sample(self) -> None:
        """Tiny n with valid_n==0 stays as 'low sample', not 'all malformed'.

        The 'all malformed' label only applies when there is enough
        volume to be confident the malformed-ness is systemic, not just
        a tiny tail. Below the reporting gate it stays 'low sample'.
        """
        assert (
            _sample_label({"n": 3, "valid_n": 0, "decision_worthy": False})
            == " ⚠ low sample"
        )

    def test_boundary_exact_min_sample_size(self) -> None:
        """Boundary cases at exactly MIN_SAMPLE_SIZE pin >= vs > choice.

        Guards against a mutation from ``n >= MIN_SAMPLE_SIZE`` to
        ``n > MIN_SAMPLE_SIZE`` going unnoticed on the malformed branch,
        and against the scorer's own ``valid_n >= MIN_SAMPLE_SIZE`` gate
        drifting on the low-sample branch. MIN_SAMPLE_SIZE is the
        reporting gate; exactly hitting it is supposed to count as
        sufficient.
        """
        # decision_worthy True → no label.
        assert (
            _sample_label(
                {
                    "n": MIN_SAMPLE_SIZE,
                    "valid_n": MIN_SAMPLE_SIZE,
                    "decision_worthy": True,
                }
            )
            == ""
        )
        # n exactly at the gate with zero valid → 'all malformed'
        # (n >= gate, not n > gate).
        assert (
            _sample_label(
                {
                    "n": MIN_SAMPLE_SIZE,
                    "valid_n": 0,
                    "decision_worthy": False,
                }
            )
            == " ⚠ all malformed"
        )
        # One below the gate → 'low sample'.
        assert (
            _sample_label(
                {
                    "n": MIN_SAMPLE_SIZE - 1,
                    "valid_n": MIN_SAMPLE_SIZE - 1,
                    "decision_worthy": False,
                }
            )
            == " ⚠ low sample"
        )

    def test_missing_keys_returns_empty(self) -> None:
        """Partial stats dict renders no label rather than a spurious one.

        Preserves the old ``decision_worthy is False`` behaviour: a
        dict missing the key produces ``""`` rather than silently
        landing in either of the warning branches. Protects callers
        that pass a projection of the full stats shape.
        """
        assert _sample_label({}) == ""
        assert _sample_label({"n": 5}) == ""
        assert _sample_label({"n": 100, "valid_n": 100}) == ""


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


class TestSectionSampleSizeWarnings:
    """Tests for section_sample_size_warnings."""

    def test_small_category_warned(self) -> None:
        """Small category triggers a warning that quotes the active gate.

        The ``(< N)`` suffix documents the threshold inline so a reader
        can see both the observed count and the gate it fell under
        without consulting the code. Guards against the suffix being
        dropped or hardcoded to a wrong value.
        """
        s = _scores(by_category={"weather": {"brier": 0.3, "n": 4, "reliability": 1.0}})
        result = section_sample_size_warnings(s)
        assert "weather" in result
        assert "4 questions" in result
        # Couple the assertion to SAMPLE_SIZE_WARNING so moving the
        # gate updates both the code and the test in lockstep and a
        # drifted-copy bug surfaces explicitly.
        assert f"(< {SAMPLE_SIZE_WARNING})" in result

    def test_large_category_not_warned(self) -> None:
        """Large category triggers the 'all categories sufficient' copy.

        The copy is deliberately explicit that the gate here is total
        category rows, not the narrower denominators used by subsections
        like directional bias; a reader should not treat this line as
        contradicting a per-subsection 'insufficient data' note.
        """
        s = _scores(
            by_category={"crypto": {"brier": 0.3, "n": 200, "reliability": 1.0}}
        )
        result = section_sample_size_warnings(s)
        assert "category reporting gate" in result
        assert "directional bias" in result

    def test_threshold_value_embedded_in_copy(self) -> None:
        """The 'at least N' copy should quote the active threshold.

        Interpolates ``SAMPLE_SIZE_WARNING`` so the test fails loudly
        when the constant moves but the user-facing copy silently
        drifts out of sync (or vice versa).
        """
        s = _scores(
            by_category={"crypto": {"brier": 0.3, "n": 200, "reliability": 1.0}}
        )
        result = section_sample_size_warnings(s)
        assert f"at least {SAMPLE_SIZE_WARNING}" in result


# ---------------------------------------------------------------------------
# section_period
# ---------------------------------------------------------------------------


class TestSectionPeriod:
    """Tests for section_period per-tool bullet rendering.

    Guards against the regression where 'Since Last Report' rendered
    tools with n=1 or n=8 alongside tools with n>=30 without any
    low-sample marker, making it trivial to read noise as signal.
    """

    @staticmethod
    def _period(by_tool: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Build a minimal period_scores dict for section_period.

        Auto-populates ``decision_worthy`` on each tool stats dict the
        same way the scorer does (``valid_n >= MIN_SAMPLE_SIZE``) so
        the fixtures match production-shape stats without each caller
        having to set it explicitly.

        :param by_tool: per-tool stats dicts keyed by tool name.
        :return: a minimal period_scores dict ready to pass to
            ``section_period``.
        """
        for stats in by_tool.values():
            stats.setdefault("decision_worthy", stats["valid_n"] >= MIN_SAMPLE_SIZE)
        total_n = sum(t["n"] for t in by_tool.values())
        total_valid = sum(t["valid_n"] for t in by_tool.values())
        return {
            "overall": {
                "n": total_n,
                "valid_n": total_valid,
                "brier": 0.2,
                "log_loss": 0.5,
            },
            "by_tool": by_tool,
        }

    @staticmethod
    def _alltime() -> dict[str, Any]:
        """Minimal all-time scores for delta comparison."""
        return {"overall": {"brier": 0.25, "log_loss": 0.6}, "by_tool": {}}

    def test_tiny_tool_flagged_as_low_sample(self) -> None:
        """Tool with n=1 in the period gets a low-sample marker.

        Also asserts the ``(n=…)`` count still renders next to the
        marker, so a regression that drops the count but keeps the
        marker (or vice versa) is caught.
        """
        period = self._period(
            {
                "prediction-online-sme": {
                    "n": 1,
                    "valid_n": 1,
                    "brier": 0.1875,
                },
            }
        )
        result = section_period(period, self._alltime(), "Since Last Report")
        assert "prediction-online-sme" in result
        assert "⚠ low sample" in result
        assert "(n=1)" in result

    def test_sufficient_tool_not_flagged(self) -> None:
        """Tool with valid_n above the gate gets no marker."""
        period = self._period(
            {
                "superforcaster": {
                    "n": 95,
                    "valid_n": 95,
                    "brier": 0.22,
                },
            }
        )
        result = section_period(period, self._alltime(), "Since Last Report")
        assert "superforcaster" in result
        assert "⚠" not in result

    def test_mixed_population_flags_only_small_ones(self) -> None:
        """Only tools below the gate should carry the marker."""
        period = self._period(
            {
                "superforcaster": {"n": 95, "valid_n": 95, "brier": 0.22},
                "prediction-online-sme": {"n": 1, "valid_n": 1, "brier": 0.19},
            }
        )
        result = section_period(period, self._alltime(), "Since Last Report")
        # Count markers: exactly one (on the small tool).
        assert result.count("⚠") == 1
        # The marker must be on the line that names the small tool,
        # not the line that names superforcaster.
        for line in result.splitlines():
            if "⚠" in line:
                assert "prediction-online-sme" in line
                assert "superforcaster" not in line

    def test_all_malformed_tool_gets_distinct_label(self) -> None:
        """Tool with n above gate but valid_n == 0 is 'all malformed'."""
        period = self._period(
            {"resolve-market-jury-v1": {"n": 55, "valid_n": 0, "brier": 0.42}}
        )
        result = section_period(period, self._alltime(), "Since Last Report")
        assert "⚠ all malformed" in result
        assert "⚠ low sample" not in result


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
