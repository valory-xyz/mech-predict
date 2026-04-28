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

import pytest

from benchmark.analyze import (
    ACTIVE_CATEGORIES,
    BRIER_RANDOM,
    OMEN_CATEGORIES,
    PLATFORM_LABELS,
    POLYMARKET_ACTIVE_CATEGORIES,
    ROLLING_WINDOW_DAYS,
    SAMPLE_SIZE_WARNING,
    _active_tools_for_platform,
    _always_majority,
    _da_lift,
    _delta_cell,
    _filter_by_active,
    _parse_tvm_key,
    _sample_label,
    generate_fleet_report,
    generate_report,
    section_category,
    section_category_platform,
    section_diagnostics_comparison,
    section_metric_reference,
    section_parse_breakdown,
    section_period,
    section_platform_comparison,
    section_platform_snapshot,
    section_reliability_comparison,
    section_sample_size_warnings,
    section_tool_category,
    section_tool_category_diagnostics,
    section_tool_category_platform,
    section_tool_comparison,
    section_tool_deployment_status,
    section_tool_version_breakdown,
    section_trend,
    section_version_deltas,
    section_weak_spots,
)
from benchmark.scorer import MIN_SAMPLE_SIZE


@pytest.fixture(autouse=True)
def _stub_release_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent any test from building the real release map via gh/git.

    Every analyze helper that touches CIDs calls release_map.get_release_map()
    when no explicit map is passed. In tests, redirect that call to an empty
    map so CIDs render as ``untagged@...`` and no subprocess is spawned.

    :param monkeypatch: pytest fixture for per-test attribute swapping.
    """
    # pylint: disable=import-outside-toplevel
    from benchmark import release_map

    empty_map = {
        "generated_at": "test",
        "tags_scanned": [],
        "cid_to_tag": {},
        "cid_to_package": {},
    }
    monkeypatch.setattr(
        release_map, "get_release_map", lambda force_rebuild=False: empty_map
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

    def test_omen_categories_exact_membership(self) -> None:
        """Pin the exact OMEN_CATEGORIES set.

        The subset / disjoint assertions elsewhere in this class cover
        well-known groupings but let a single-label add or drop slip
        through unnoticed. Asserting equality against the full expected
        set fails immediately on any membership mutation.
        """
        expected = frozenset(
            {
                "business",
                "cryptocurrency",
                "politics",
                "science",
                "technology",
                "trending",
                "social",
                "health",
                "sustainability",
                "internet",
                "food",
                "pets",
                "animals",
                "curiosities",
                "economy",
                "arts",
                "entertainment",
                "weather",
                "sports",
                "finance",
                "international",
            }
        )
        assert OMEN_CATEGORIES == expected

    def test_polymarket_active_categories_exact_membership(self) -> None:
        """Pin the exact POLYMARKET_ACTIVE_CATEGORIES set.

        Same rationale as the Omen equality test above: catches single-
        label mutations that subset/disjoint checks miss.
        """
        expected = frozenset(
            {
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
        )
        assert POLYMARKET_ACTIVE_CATEGORIES == expected


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
        """Cells with n >= MIN_SAMPLE_SIZE appear in the ranked table.

        Column order: Tool | Category | n | Reliability | Brier |
        Baseline Brier | BSS | DirAcc | Yes% | No% | Always-majority |
        DA lift.
        """
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
                    "valid_n": 55,
                    "reliability": 0.92,
                    "baseline_brier": 0.25,
                    "outcome_yes_rate": 0.40,
                    "decision_worthy": True,
                }
            }
        )
        result = section_tool_category(s)
        # Columns in order: Tool, Category, n, Reliability, Brier,
        # Baseline Brier, BSS, DirAcc, Yes%, No%, Always-majority, DA lift.
        # always_majority = max(0.4, 0.6) = 0.6; DA lift = 0.76 - 0.60 = 0.16.
        assert (
            "| tool-a | politics | 60 | 92% | 0.1900 | 0.2500 | +0.0500 | 76%"
            " | 40% | 60% | 60% | +0.1600 |" in result
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
        # Column order: Tool | Category | n | ... Brier is the 5th column.
        assert f"| tool-a | politics | {MIN_SAMPLE_SIZE} " in result
        assert "| 0.2000 |" in result
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
        """Tool with n above gate but valid_n == 0 is 'all malformed'.

        Uses ``brier=None`` because that is what the scorer emits when
        ``valid_n == 0`` (see scorer._derive_group). Any fixture that
        passes a numeric brier alongside valid_n=0 would be scorer-
        impossible and would let the test pass by traversing the
        wrong branch in ``section_period``.
        """
        period = self._period(
            {"resolve-market-jury-v1": {"n": 55, "valid_n": 0, "brier": None}}
        )
        result = section_period(period, self._alltime(), "Since Last Report")
        assert "resolve-market-jury-v1" in result
        assert "⚠ all malformed" in result
        assert "⚠ low sample" not in result
        # Brier must render as N/A for a scorer-impossible-otherwise group.
        assert "N/A" in result


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
        report = generate_report(
            s, history, platform="omen", rolling_scores=s, disabled_tools={}
        )

        assert "# Benchmark Report (Omenstrat) — " in report
        assert "## Metric References" in report
        assert f"## Platform Snapshot (Current {ROLLING_WINDOW_DAYS}d)" in report
        assert "## Platform Historical Comparison" in report
        assert "## Tool Historical Comparison" in report
        assert f"## Tool \u00d7 Category (Current {ROLLING_WINDOW_DAYS}d)" in report
        assert "## Tool \u00d7 Category Historical Comparison" in report
        assert "## Diagnostics Historical Comparison" in report
        assert "## Reliability & Parse Quality (Current vs All-Time)" in report
        # Per-platform reports drop Platform Comparison / Tool × Platform —
        # they'd be single-row tables with no signal.
        assert "## Platform Comparison" not in report
        assert "## Tool \u00d7 Platform" not in report
        assert "## Trend" in report
        assert "## Sample Size Warnings" in report
        # The reviewer's P1 restructure dropped overlapping-window and
        # single-scope point-in-time sections. Their data is now folded
        # into the three-window comparison tables.
        assert "## Since Last Report" not in report
        assert f"## Last {ROLLING_WINDOW_DAYS} Days" not in report
        assert "## Overall" not in report
        assert "## Worst Predictions" not in report
        assert "## Best Predictions" not in report
        assert "## Edge Over Market" not in report
        assert "## Calibration" not in report
        assert "## Latency" not in report

    def test_empty_data_no_crash(self) -> None:
        """Test empty data does not crash."""
        s = _scores(brier=None, reliability=None, total=0, valid=0)
        report = generate_report(s, [], platform="omen", disabled_tools={})
        assert "# Benchmark Report (Omenstrat) — " in report
        # With no rolling_scores, the snapshot banner explains the gap.
        assert (
            f"Scores for the last {ROLLING_WINDOW_DAYS} days are unavailable" in report
        )


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
        """Cells with n >= 30 render without ⚠ marker; CIDs become release-tag labels."""
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
        rm = {
            "tags_scanned": ["v1.0.0"],
            "cid_to_tag": {"bafy_v1": "v1.0.0"},
            "cid_to_package": {},
        }
        result = section_tool_version_breakdown(scores, release_map_data=rm)
        assert "tool-a" in result
        assert "`v1.0.0`" in result
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
        rm = {
            "tags_scanned": ["v1.0.0"],
            "cid_to_tag": {"bafy_v1": "v1.0.0"},
            "cid_to_package": {},
        }
        result = section_tool_version_breakdown(scores, release_map_data=rm)
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


def _rm(cid_to_tag: dict[str, str], tags: list[str]) -> dict[str, Any]:
    """Build a minimal release map for tests."""
    return {
        "generated_at": "2026-04-15T00:00:00Z",
        "tags_scanned": tags,
        "cid_to_tag": cid_to_tag,
        "cid_to_package": {},
    }


class TestSectionVersionDeltas:
    """Tests for section_version_deltas."""

    def test_empty_returns_blank(self) -> None:
        """Missing data returns empty string."""
        assert section_version_deltas({}, _rm({}, [])) == ""

    def test_single_version_per_tool_returns_blank(self) -> None:
        """Tools with only one (version, mode) cell yield no delta section."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | cidA | production_replay": {"n": 100, "brier": 0.2},
            }
        }
        assert section_version_deltas(scores, _rm({"cidA": "v1.0.0"}, ["v1.0.0"])) == ""

    def test_renders_prior_and_pooled_tables(self) -> None:
        """Tool with 3 versions renders both sub-tables with release-tag labels."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | cidA | production_replay": {
                    "n": 500,
                    "valid_n": 500,
                    "brier": 0.30,
                    "directional_accuracy": 0.6,
                },
                "tool-a | cidB | production_replay": {
                    "n": 500,
                    "valid_n": 500,
                    "brier": 0.25,
                    "directional_accuracy": 0.7,
                },
                "tool-a | cidC | production_replay": {
                    "n": 500,
                    "valid_n": 500,
                    "brier": 0.20,
                    "directional_accuracy": 0.7,
                },
            }
        }
        rm = _rm(
            {"cidA": "v1.0.0", "cidB": "v1.1.0", "cidC": "v1.2.0"},
            ["v1.0.0", "v1.1.0", "v1.2.0"],
        )
        result = section_version_deltas(scores, rm)
        assert "## Version Deltas" in result
        assert "### tool-a (production_replay)" in result
        assert "**vs prior version:**" in result
        assert "**vs previous pooled:**" in result
        # Release-tag labels in rows, not CIDs
        assert "v1.0.0" in result
        assert "v1.1.0" in result
        assert "v1.2.0" in result
        assert "cidA" not in result
        assert "improved" in result  # each step reduces Brier

    def test_within_mode_only(self) -> None:
        """Prod and tournament versions never appear in the same sub-table."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | cidA | production_replay": {
                    "n": 500,
                    "valid_n": 500,
                    "brier": 0.30,
                    "directional_accuracy": 0.6,
                },
                "tool-a | cidB | production_replay": {
                    "n": 500,
                    "valid_n": 500,
                    "brier": 0.20,
                    "directional_accuracy": 0.7,
                },
                "tool-a | cidB | tournament": {
                    "n": 500,
                    "valid_n": 500,
                    "brier": 0.15,
                    "directional_accuracy": 0.8,
                },
            }
        }
        rm = _rm({"cidA": "v1.0.0", "cidB": "v1.1.0"}, ["v1.0.0", "v1.1.0"])
        result = section_version_deltas(scores, rm)
        # Production sub-section renders (2 versions); tournament does not
        # (only 1 version). No cross-mode row either way.
        assert "### tool-a (production_replay)" in result
        assert "### tool-a (tournament)" not in result

    def test_low_sample_marker_on_delta(self) -> None:
        """Delta with min(n) < VERSION_DELTA_LOW_SAMPLE_STRICT carries ⚠."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | cidA | production_replay": {
                    "n": 1000,
                    "valid_n": 1000,
                    "brier": 0.30,
                    "directional_accuracy": 0.6,
                },
                "tool-a | cidB | production_replay": {
                    "n": 50,
                    "valid_n": 50,
                    "brier": 0.20,
                    "directional_accuracy": 0.7,
                },
            }
        }
        rm = _rm({"cidA": "v1.0.0", "cidB": "v1.1.0"}, ["v1.0.0", "v1.1.0"])
        result = section_version_deltas(scores, rm)
        assert "⚠" in result

    def test_pooling_is_n_weighted_mean(self) -> None:
        """pool(n=100 B=0.20, n=200 B=0.30) == B=0.2667."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | cidA | production_replay": {
                    "n": 100,
                    "valid_n": 100,
                    "brier": 0.20,
                    "directional_accuracy": 0.7,
                },
                "tool-a | cidB | production_replay": {
                    "n": 200,
                    "valid_n": 200,
                    "brier": 0.30,
                    "directional_accuracy": 0.6,
                },
                "tool-a | cidC | production_replay": {
                    "n": 300,
                    "valid_n": 300,
                    "brier": 0.2667,
                    "directional_accuracy": 0.65,
                },
            }
        }
        rm = _rm(
            {"cidA": "v1.0.0", "cidB": "v1.1.0", "cidC": "v1.2.0"},
            ["v1.0.0", "v1.1.0", "v1.2.0"],
        )
        result = section_version_deltas(scores, rm)
        # Pooled baseline for v1.2.0 is (0.20*100 + 0.30*200)/300 = 0.2667,
        # so the delta in the pooled table is +0.0000 (unchanged direction).
        assert "v1.0.0..v1.1.0" in result
        assert "unchanged" in result
        assert "flat" not in result  # renamed for repo-wide vocabulary consistency

    def test_no_low_sample_footer_when_no_rows_flagged(self) -> None:
        """The ⚠ legend is absent when every row's n clears the threshold."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | cidA | production_replay": {
                    "n": 500,
                    "valid_n": 500,
                    "brier": 0.30,
                    "directional_accuracy": 0.6,
                },
                "tool-a | cidB | production_replay": {
                    "n": 500,
                    "valid_n": 500,
                    "brier": 0.20,
                    "directional_accuracy": 0.7,
                },
            }
        }
        rm = _rm({"cidA": "v1.0.0", "cidB": "v1.1.0"}, ["v1.0.0", "v1.1.0"])
        result = section_version_deltas(scores, rm)
        assert "Rows marked with ⚠" not in result

    def test_low_sample_footer_when_rows_flagged(self) -> None:
        """The ⚠ legend is present when at least one row carries the marker."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | cidA | production_replay": {
                    "n": 1000,
                    "valid_n": 1000,
                    "brier": 0.30,
                    "directional_accuracy": 0.6,
                },
                "tool-a | cidB | production_replay": {
                    "n": 50,
                    "valid_n": 50,
                    "brier": 0.20,
                    "directional_accuracy": 0.7,
                },
            }
        }
        rm = _rm({"cidA": "v1.0.0", "cidB": "v1.1.0"}, ["v1.0.0", "v1.1.0"])
        result = section_version_deltas(scores, rm)
        assert "Rows marked with ⚠" in result

    def test_pool_label_collapses_when_start_equals_end(self) -> None:
        """With exactly 2 versions, pool baseline renders without a range."""
        scores = {
            "by_tool_version_mode": {
                "tool-a | cidA | production_replay": {
                    "n": 500,
                    "valid_n": 500,
                    "brier": 0.30,
                    "directional_accuracy": 0.6,
                },
                "tool-a | cidB | production_replay": {
                    "n": 500,
                    "valid_n": 500,
                    "brier": 0.20,
                    "directional_accuracy": 0.7,
                },
            }
        }
        rm = _rm({"cidA": "v1.0.0", "cidB": "v1.1.0"}, ["v1.0.0", "v1.1.0"])
        result = section_version_deltas(scores, rm)
        # The pool row's baseline is V_0 alone; no "v1.0.0..v1.0.0" degenerate range.
        assert "v1.0.0..v1.0.0" not in result
        # And the collapsed label does appear in the pooled sub-table.
        assert "| `v1.0.0` | `v1.1.0` |" in result


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
            "tool-a | v2 | production_replay": {
                "n": 100,
                "valid_n": 100,
                "brier": 0.25,
                "directional_accuracy": 0.7,
                "log_loss": 0.6,
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
        report = generate_report(s, [], platform="omen", disabled_tools={})
        assert "Tool × Version × Mode" not in report
        assert "Version Deltas" not in report

    def test_on_renders_cumulative_breakdown_and_deltas(self) -> None:
        """Tournament flag renders both the breakdown table and Version Deltas.

        Version Deltas emerges only when a tool has 2+ versions in one mode.
        """
        s = self._scores_with_versions()
        report = generate_report(
            s, [], platform="omen", include_tournament=True, disabled_tools={}
        )
        assert "Tool × Version × Mode (All-Time)" in report
        assert "## Version Deltas" in report

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
        report = generate_report(
            s,
            [],
            platform="omen",
            rolling_scores=rolling,
            include_tournament=True,
            disabled_tools={},
        )
        assert "Tool × Version × Mode (All-Time)" in report
        assert f"Tool × Version × Mode (Last {ROLLING_WINDOW_DAYS} Days)" in report


# ---------------------------------------------------------------------------
# Mode split: tournament sections and callouts (BENCHMARK_MODE_SPLIT_SPEC)
# ---------------------------------------------------------------------------


def _scores_with_tool(
    tool: str,
    brier: float,
    n: int,
    valid: int | None = None,
    baseline: float = 0.25,
    prod_cid: str = "cid_prod",
) -> dict[str, Any]:
    """Build a production scores dict with one by_tool entry and matching TVM cell."""
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
        "by_tool_version_mode": {
            f"{tool} | {prod_cid} | production_replay": {
                "brier": brier,
                "baseline_brier": baseline,
                "n": n,
                "valid_n": valid_n,
                "directional_accuracy": 0.7,
                "brier_skill_score": 0.0,
            }
        },
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

    def test_same_cid_on_both_sides_is_skipped(self) -> None:
        """After rollout, tournament and production share a CID; don't flag noise.

        ``_most_recent_prod_cid`` returns the latest-tagged prod CID.
        Once the candidate has rolled out, ``cand_cid == prod_cid`` and
        the loop compares two samples of the same version — sampling
        noise alone can exceed ``CALLOUT_DELTA`` at small n. The section
        must suppress this, otherwise the rollout fix promised by this
        PR is itself a regression.
        """
        # pylint: disable=import-outside-toplevel
        from benchmark.analyze import section_tournament_callouts

        shared_cid = "cid_v2"
        prod = _scores_with_tool("tool-a", 0.20, 1000, prod_cid=shared_cid)
        # Tournament cell uses the same CID; Brier diverges enough to
        # trigger a promotion bullet if the same-CID guard is missing.
        tourn = _tournament_scores_with_version("tool-a", shared_cid, 0.10, 50)
        assert section_tournament_callouts(prod, tourn) == ""


class TestGenerateReportWithTournamentFiles:
    """Tests for generate_report dual-mode rendering."""

    def test_tournament_sections_omitted_when_file_absent(self) -> None:
        """No tournament inputs -> no '— Tournament' headings, no callouts."""
        prod = _scores_with_tool("tool-a", 0.20, 1000)
        report = generate_report(prod, [], platform="omen", include_tournament=True)
        assert "— Tournament" not in report
        assert "## Tournament Callouts" not in report

    def test_tournament_sections_rendered_when_data_present(self) -> None:
        """Tournament inputs with rows -> Tool × Version × Mode covers both modes."""
        prod = _scores_with_tool("tool-a", 0.20, 1000)
        tourn = _tournament_scores_with_version("tool-a", "v2", 0.18, 100)
        report = generate_report(
            prod,
            [],
            platform="omen",
            include_tournament=True,
            scores_tournament=tourn,
            rolling_scores=prod,
            rolling_scores_tournament=tourn,
        )
        # Under the three-window restructure the per-mode headings live in
        # Tool × Version × Mode, not a duplicated " — Tournament" ranking.
        assert "## Overall" not in report
        assert "## Tool × Version × Mode (All-Time)" in report
        assert f"## Tool × Version × Mode (Last {ROLLING_WINDOW_DAYS} Days)" in report

    def test_empty_rolling_tournament_does_not_crash(self) -> None:
        """Rolling tournament input with zero rows renders cleanly.

        scorer.score_period_split writes tournament period files on every
        run; days with zero tournament rows in the window must not blow
        up the TVM merge.
        """
        prod = _scores_with_tool("tool-a", 0.20, 1000)
        all_time_tourn = _tournament_scores_with_version("tool-a", "v2", 0.18, 100)
        empty_rolling_tourn = {"total_rows": 0, "overall": {}}
        report = generate_report(
            prod,
            [],
            platform="omen",
            rolling_scores=None,
            include_tournament=True,
            scores_tournament=all_time_tourn,
            rolling_scores_tournament=empty_rolling_tourn,
        )
        assert "# Benchmark Report (Omenstrat)" in report

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
            platform="omen",
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
            platform="omen",
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
            platform="omen",
            include_tournament=True,
            scores_tournament=tourn,
        )
        assert "## Tournament Callouts" not in report


# ---------------------------------------------------------------------------
# Per-platform rendering — generate_report gates fleet-wide comparison
# sections when the scores are already partitioned to one platform.
# ---------------------------------------------------------------------------


class TestGenerateReportPerPlatform:
    """generate_report(platform=...) scopes the report to one deployment."""

    def test_header_uses_deployment_label_for_omen(self) -> None:
        """Omen scores render with 'Omenstrat' in the header."""
        s = _scores()
        report = generate_report(s, [], platform="omen", disabled_tools={})
        assert "# Benchmark Report (Omenstrat) — " in report

    def test_header_uses_deployment_label_for_polymarket(self) -> None:
        """Polymarket scores render with 'Polystrat' in the header."""
        s = _scores()
        report = generate_report(s, [], platform="polymarket", disabled_tools={})
        assert "# Benchmark Report (Polystrat) — " in report

    def test_platform_comparison_absent(self) -> None:
        """Platform Comparison section never renders in per-platform mode."""
        s = _scores()
        # Give by_platform some content so the fleet-wide path would render
        # the section in the old code path.
        s["by_platform"] = {
            "omen": {"brier": 0.2, "n": 100, "edge": 0.04, "edge_n": 80},
        }
        report = generate_report(s, [], platform="omen", disabled_tools={})
        assert "## Platform Comparison" not in report

    def test_tool_platform_section_absent(self) -> None:
        """Tool × Platform section never renders in per-platform mode."""
        s = _scores()
        report = generate_report(s, [], platform="omen", disabled_tools={})
        assert "## Tool \u00d7 Platform" not in report

    def test_rejects_unknown_platform(self) -> None:
        """Unknown platform raises ValueError with a helpful message."""
        s = _scores()
        with pytest.raises(ValueError, match="platform must be one of"):
            generate_report(s, [], platform="gnosis", disabled_tools={})

    def test_platform_labels_are_deployment_names(self) -> None:
        """Label map pairs scorer keys with the team's deployment names."""
        # Guards against a silent relabel — the Slack ask explicitly said
        # Omenstrat / Polystrat, so a rename to e.g. "Omen" would read wrong.
        assert PLATFORM_LABELS == {
            "omen": "Omenstrat",
            "polymarket": "Polystrat",
        }


class TestTrendSectionPlatformAnnotation:
    """section_trend warns when rendered inside a per-platform report.

    scores_history.jsonl is only populated by the combined prod accumulator,
    so the same monthly numbers appear in every per-platform report. The
    annotation prevents a reader from mistaking the data for platform-scoped.
    """

    def _history(self) -> list[dict[str, Any]]:
        return [{"month": "2026-03", "overall": {"brier": 0.2, "n": 100}}]

    def test_fleet_wide_note_renders_with_platform(self) -> None:
        """Platform-scoped render inserts the fleet-wide disclaimer."""
        rendered = section_trend(self._history(), None, platform="omen")
        assert "Aggregated across all platforms" in rendered

    def test_no_note_without_platform(self) -> None:
        """Fleet-wide render (no platform arg) stays quiet — it's correct there."""
        rendered = section_trend(self._history(), None)
        assert "Aggregated across all platforms" not in rendered

    def test_heading_is_fleet_wide_monthly(self) -> None:
        """Heading names the scope so it's unambiguous even out of context."""
        rendered = section_trend(self._history(), None)
        assert "## Trend (Fleet-wide, Monthly)" in rendered

    def test_per_platform_report_omits_in_progress_row(self) -> None:
        """generate_report does not append a current-month row in per-platform mode.

        The per-platform scores dict is all-time cumulative (history is never
        emitted for per-platform accumulators), so appending it beside
        fleet-wide per-month history would mix two different quantities. A
        platform-scoped report renders only the fleet-wide completed months.
        """
        prod = _scores_with_tool("tool-a", 0.20, 1000)
        prod["current_month"] = "2026-04"
        history = [{"month": "2026-03", "overall": {"brier": 0.2, "n": 100}}]
        report = generate_report(prod, history, platform="omen", disabled_tools={})
        assert "*(in progress)*" not in report
        assert "2026-04" not in report


class TestSectionToolDeploymentStatusInverted:
    """Phase 3: section renders active tools per deployment, scoped to platform."""

    def _scores_with_tools(self, *tools: str) -> dict[str, Any]:
        return {"by_tool": {t: {"brier": 0.2, "n": 100, "valid_n": 100} for t in tools}}

    def test_omen_platform_hides_polystrat_deployment(self) -> None:
        """Omenstrat report never mentions polystrat Pearl."""
        scores = self._scores_with_tools("tool-a", "tool-b")
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": [],
            "omenstrat QS": [],
            "polystrat Pearl": ["tool-a"],
        }
        rendered = section_tool_deployment_status(scores, disabled, platform="omen")
        assert "omenstrat Pearl" in rendered
        assert "omenstrat QS" in rendered
        assert "polystrat" not in rendered

    def test_polymarket_platform_hides_omenstrat_deployments(self) -> None:
        """Polystrat report never mentions omenstrat-anything."""
        scores = self._scores_with_tools("tool-a")
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["tool-a"],
            "omenstrat QS": ["tool-a"],
            "polystrat Pearl": [],
        }
        rendered = section_tool_deployment_status(
            scores, disabled, platform="polymarket"
        )
        assert "polystrat Pearl" in rendered
        assert "omenstrat" not in rendered

    def test_heading_carries_platform_label(self) -> None:
        """Section heading is scoped with the deployment label."""
        scores = self._scores_with_tools("tool-a")
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": [],
            "omenstrat QS": [],
            "polystrat Pearl": [],
        }
        rendered = section_tool_deployment_status(scores, disabled, platform="omen")
        assert rendered.startswith("## Tool Deployment Status (Omenstrat)")

    def test_active_tools_exclude_disabled(self) -> None:
        """Active tools are the benchmarked set with disabled tools removed."""
        scores = self._scores_with_tools("tool-a", "tool-b", "tool-c")
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["tool-b"],
            "omenstrat QS": [],
            "polystrat Pearl": [],
        }
        rendered = section_tool_deployment_status(scores, disabled, platform="omen")
        pearl_line = [
            line for line in rendered.splitlines() if "omenstrat Pearl" in line
        ][0]
        assert "`tool-a`" in pearl_line
        assert "`tool-c`" in pearl_line
        assert "`tool-b`" not in pearl_line
        qs_line = [line for line in rendered.splitlines() if "omenstrat QS" in line][0]
        for expected in ("`tool-a`", "`tool-b`", "`tool-c`"):
            assert expected in qs_line

    def test_normalizes_underscores_in_disabled_list(self) -> None:
        """Config files sometimes list prediction_request_X; treat as equivalent."""
        scores = self._scores_with_tools("prediction-request-reasoning")
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["prediction_request_reasoning"],
            "omenstrat QS": [],
            "polystrat Pearl": [],
        }
        rendered = section_tool_deployment_status(scores, disabled, platform="omen")
        pearl_line = [
            line for line in rendered.splitlines() if "omenstrat Pearl" in line
        ][0]
        assert "prediction-request-reasoning" not in pearl_line

    def test_failed_fetch_renders_unavailable_banner(self) -> None:
        """Deployment whose fetch returned None is not claimed as empty."""
        scores = self._scores_with_tools("tool-a")
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": None,
            "omenstrat QS": [],
            "polystrat Pearl": [],
        }
        rendered = section_tool_deployment_status(scores, disabled, platform="omen")
        assert "omenstrat Pearl" in rendered
        assert "⚠️ unavailable" in rendered

    def test_fleet_wide_mode_still_supported(self) -> None:
        """platform=None keeps the legacy fleet-wide render for ad-hoc callers."""
        scores = self._scores_with_tools("tool-a")
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": [],
            "omenstrat QS": [],
            "polystrat Pearl": [],
        }
        rendered = section_tool_deployment_status(scores, disabled, platform=None)
        for deployment in ("omenstrat Pearl", "omenstrat QS", "polystrat Pearl"):
            assert deployment in rendered
        assert "(Omenstrat)" not in rendered
        assert "(Polystrat)" not in rendered

    def test_empty_disabled_opts_out_of_section(self) -> None:
        """Existing contract: empty dict skips the section entirely."""
        scores = self._scores_with_tools("tool-a")
        assert section_tool_deployment_status(scores, {}, platform="omen") == ""


class TestToolCategoryPositionInReport:
    """Phase 3: Tool × Category is rendered before Tool × Version × Mode."""

    def test_tool_category_current_and_comparison_both_render(self) -> None:
        """Current-3d table and historical comparison both render, in that order."""
        scores = _scores_with_tool("tool-a", 0.20, 1000)
        report = generate_report(
            scores, [], platform="omen", rolling_scores=scores, disabled_tools={}
        )
        current_idx = report.index(
            f"## Tool \u00d7 Category (Current {ROLLING_WINDOW_DAYS}d)"
        )
        comparison_idx = report.index("## Tool \u00d7 Category Historical Comparison")
        assert current_idx < comparison_idx

    def test_tool_category_lands_before_tool_version_mode(self) -> None:
        """Ordering: comparison tables before the Tool × Version × Mode tables."""
        scores = _scores_with_tool("tool-a", 0.20, 1000)
        scores["by_tool_version_mode"] = {
            "tool-a | v1 | production_replay": {
                "n": 1000,
                "valid_n": 1000,
                "brier": 0.2,
            }
        }
        report = generate_report(
            scores,
            [],
            platform="omen",
            rolling_scores=scores,
            include_tournament=True,
            disabled_tools={},
        )
        tc_idx = report.index("## Tool \u00d7 Category Historical Comparison")
        tvm_idx = report.index("## Tool \u00d7 Version \u00d7 Mode (All-Time)")
        assert tc_idx < tvm_idx


class TestSectionMetricReference:
    """Tests for section_metric_reference."""

    def test_renders_heading_and_core_metrics(self) -> None:
        """Legend names every headline metric the report renders."""
        rendered = section_metric_reference()
        assert "## Metric References" in rendered
        for label in ("Brier", "Log Loss", "BSS", "Edge over market"):
            assert label in rendered

    def test_cites_rolling_window_from_constant(self) -> None:
        """Legend quotes ROLLING_WINDOW_DAYS so a one-line change updates both."""
        rendered = section_metric_reference()
        assert f"Current {ROLLING_WINDOW_DAYS}d" in rendered
        assert f"Prev {ROLLING_WINDOW_DAYS}d" in rendered

    def test_names_all_three_windows(self) -> None:
        """Legend explicitly names current, all-time, and prev-rolling windows."""
        rendered = section_metric_reference()
        assert f"Current {ROLLING_WINDOW_DAYS}d" in rendered
        assert "All-Time" in rendered
        assert f"Prev {ROLLING_WINDOW_DAYS}d" in rendered

    def test_documents_sample_size_guardrail(self) -> None:
        """Legend spells out the MIN_SAMPLE_SIZE delta-suppression rule."""
        rendered = section_metric_reference()
        assert f"n < {MIN_SAMPLE_SIZE}" in rendered
        assert "insufficient data" in rendered

    def test_cites_brier_random_baseline(self) -> None:
        """Coin-flip Brier anchor is sourced from BRIER_RANDOM."""
        rendered = section_metric_reference()
        assert f"coin-flip {BRIER_RANDOM}" in rendered


class TestGenerateReportLegendPlacement:
    """Regression tests for where the metric legend lands in the report body."""

    def test_legend_rendered_before_first_data_section(self) -> None:
        """Legend sits between the H1 title and the first data section."""
        scores = _scores()
        rolling = _scores_with_tool("tool-a", 0.20, 100)
        report = generate_report(
            scores,
            [],
            platform="omen",
            rolling_scores=rolling,
            disabled_tools={},
        )
        legend_idx = report.index("## Metric References")
        snapshot_idx = report.index("## Platform Snapshot")
        assert legend_idx < snapshot_idx
        header_end = report.index("\n")
        assert legend_idx > header_end

    def test_legend_rendered_exactly_once(self) -> None:
        """Legend is not duplicated across production + tournament branches."""
        scores = _scores()
        tourn = _scores_with_tool("tool-a", 0.20, 1000)
        report = generate_report(
            scores,
            [],
            platform="omen",
            include_tournament=True,
            scores_tournament=tourn,
            disabled_tools={},
        )
        assert report.count("## Metric References") == 1


class TestGenerateReportRollingBanner:
    """Missing rolling_scores must not silently fall back to all-time data."""

    def test_rolling_sections_absent_when_rolling_scores_missing(self) -> None:
        """No rolling_scores -> comparison sections omitted with an explicit banner.

        Falling back to all-time ``scores`` under a "(Current 3d)" heading
        would mislabel the window. Instead, a single banner explains the
        gap and every rolling/comparison section is skipped.
        """
        scores = _scores_with_tool("tool-a", 0.20, 1000)
        report = generate_report(scores, [], platform="omen", disabled_tools={})
        assert f"## Platform Snapshot (Current {ROLLING_WINDOW_DAYS}d)" in report
        assert (
            f"Scores for the last {ROLLING_WINDOW_DAYS} days are unavailable" in report
        )
        for heading in (
            "## Platform Historical Comparison",
            "## Tool Historical Comparison",
            f"## Tool × Category (Current {ROLLING_WINDOW_DAYS}d)",
            f"## Tool × Category Diagnostics (Current {ROLLING_WINDOW_DAYS}d)",
            "## Tool × Category Historical Comparison",
            "## Diagnostics Historical Comparison",
            "## Reliability & Parse Quality (Current vs All-Time)",
        ):
            assert heading not in report

    def test_tool_version_mode_rolling_section_carries_window_note(self) -> None:
        """Tool × Version × Mode (Last N Days) carries the n= annotation."""
        scores = _scores_with_tool("tool-a", 0.20, 1000)
        scores["by_tool_version_mode"] = {
            "tool-a | v1 | production_replay": {
                "n": 1000,
                "valid_n": 1000,
                "brier": 0.2,
            }
        }
        report = generate_report(
            scores,
            [],
            platform="omen",
            rolling_scores=scores,
            include_tournament=True,
            disabled_tools={},
        )
        heading_idx = report.index(
            f"## Tool × Version × Mode (Last {ROLLING_WINDOW_DAYS} Days)"
        )
        next_heading = report.find("##", heading_idx + 1)
        body = report[heading_idx : next_heading if next_heading > -1 else len(report)]
        assert (
            f"_n= values below are over the last {ROLLING_WINDOW_DAYS} days._" in body
        )


class TestGenerateReportStructure:
    """Seven-section report structure from the P1 restructure."""

    def test_sections_in_reviewer_order(self) -> None:
        """The new comparison sections render in the reviewer-specified order."""
        scores = _scores_with_tool("tool-a", 0.20, 100)
        rolling = _scores_with_tool("tool-a", 0.15, 60)
        report = generate_report(
            scores,
            [],
            platform="omen",
            rolling_scores=rolling,
            disabled_tools={},
        )
        ordered_headings = [
            f"## Platform Snapshot (Current {ROLLING_WINDOW_DAYS}d)",
            "## Platform Historical Comparison",
            "## Tool Historical Comparison",
            f"## Tool × Category (Current {ROLLING_WINDOW_DAYS}d)",
            f"## Tool × Category Diagnostics (Current {ROLLING_WINDOW_DAYS}d)",
            "## Tool × Category Historical Comparison",
            "## Diagnostics Historical Comparison",
            "## Reliability & Parse Quality (Current vs All-Time)",
        ]
        prev_idx = -1
        for heading in ordered_headings:
            idx = report.index(heading)
            assert idx > prev_idx, f"{heading} out of order"
            prev_idx = idx

    def test_since_last_report_not_in_report(self) -> None:
        """Overlapping-window 'Since Last Report' was dropped per reviewer."""
        scores = _scores_with_tool("tool-a", 0.20, 100)
        report = generate_report(
            scores,
            [],
            platform="omen",
            rolling_scores=scores,
            disabled_tools={},
        )
        assert "## Since Last Report" not in report


class TestSectionCategoryPlatform:
    """section_category_platform renders the category × platform cross cell table."""

    def _scores_with_cp(self) -> dict[str, Any]:
        return {
            "by_category_platform": {
                "crypto | omen": {
                    "brier": 0.22,
                    "brier_skill_score": 0.1,
                    "log_loss": 0.5,
                    "directional_accuracy": 0.6,
                    "n": 50,
                },
                "politics | polymarket": {
                    "brier": 0.31,
                    "brier_skill_score": None,
                    "log_loss": 0.62,
                    "directional_accuracy": 0.55,
                    "n": 10,
                },
            }
        }

    def test_sufficient_cells_render_in_table(self) -> None:
        """Cells with n >= MIN_SAMPLE_SIZE render inline in the main table."""
        rendered = section_category_platform(self._scores_with_cp())
        assert "| crypto | omen | 0.2200" in rendered

    def test_sparse_cells_moved_to_footnote(self) -> None:
        """Cells below MIN_SAMPLE_SIZE are moved out of the ranking."""
        rendered = section_category_platform(self._scores_with_cp())
        # politics | polymarket has n=10, below the gate
        assert "| politics | polymarket | 0.3100" not in rendered
        assert "insufficient data (n=10)" in rendered

    def test_empty_input(self) -> None:
        """Empty data renders a placeholder, never blows up."""
        assert "No cross-breakdown data available." in section_category_platform({})


class TestAlwaysMajorityAndDALift:
    """_always_majority / _da_lift helpers derive Maria Pia's requested fields.

    Both are trivial math but they're user-facing numbers, so lock them
    in with tests rather than trusting the diff review.
    """

    def test_always_majority_yes_heavy(self) -> None:
        """yes_rate=0.7 → majority outcome is yes at 70%."""

        assert _always_majority(0.7) == 0.7

    def test_always_majority_no_heavy(self) -> None:
        """yes_rate=0.3 → majority outcome is no at 70% (1 - 0.3)."""

        assert _always_majority(0.3) == pytest.approx(0.7)

    def test_always_majority_balanced(self) -> None:
        """yes_rate=0.5 → majority baseline is 0.5 (no class is majority)."""

        assert _always_majority(0.5) == 0.5

    def test_always_majority_none(self) -> None:
        """Missing yes_rate yields None, not a crash or 0.5 default."""

        assert _always_majority(None) is None

    def test_da_lift_positive_when_beating_majority(self) -> None:
        """Tool predicting above always-majority has positive lift."""

        # yes_rate=0.4 → majority=0.6. DirAcc=0.75 → lift=+0.15.
        assert _da_lift(0.75, 0.4) == pytest.approx(0.15)

    def test_da_lift_zero_when_equal_to_majority(self) -> None:
        """Tool matching always-majority has zero lift."""

        assert _da_lift(0.6, 0.4) == pytest.approx(0.0)

    def test_da_lift_negative_when_below_majority(self) -> None:
        """Tool worse than always-majority has negative lift."""

        # yes_rate=0.3 → majority=0.7. DirAcc=0.55 → lift=-0.15.
        assert _da_lift(0.55, 0.3) == pytest.approx(-0.15)

    def test_da_lift_none_when_inputs_missing(self) -> None:
        """Either missing input yields None, not arithmetic error."""

        assert _da_lift(None, 0.4) is None
        assert _da_lift(0.75, None) is None


class TestSectionToolCategoryDiagnostics:
    """section_tool_category_diagnostics renders edge / edge_n / log loss."""

    def test_sufficient_cell_renders(self) -> None:
        """Cells with n >= MIN_SAMPLE_SIZE render with all three diagnostics."""

        scores = {
            "by_tool_category": {
                "tool-a | crypto": {
                    "n": 60,
                    "edge": 0.03,
                    "edge_n": 50,
                    "log_loss": 0.48,
                }
            }
        }
        rendered = section_tool_category_diagnostics(scores)
        assert "## Tool × Category Diagnostics" in rendered
        assert "| tool-a | crypto | +0.0300 | 50 | 0.4800 | 60 |" in rendered

    def test_sparse_cells_dropped(self) -> None:
        """Cells below MIN_SAMPLE_SIZE don't render in the table body."""

        scores = {
            "by_tool_category": {
                "tool-a | crypto": {"n": 5, "edge": 0.1, "edge_n": 3, "log_loss": 0.5},
            }
        }
        rendered = section_tool_category_diagnostics(scores)
        assert "| tool-a | crypto | +0.1000" not in rendered
        assert f"no cells with n ≥ {MIN_SAMPLE_SIZE}" in rendered

    def test_missing_diagnostic_fields_render_na(self) -> None:
        """Cells with None edge / edge_n / log_loss render N/A rather than crash."""

        scores = {
            "by_tool_category": {
                "tool-a | crypto": {"n": 60}  # no edge / log_loss keys
            }
        }
        rendered = section_tool_category_diagnostics(scores)
        assert "N/A" in rendered
        assert "| tool-a | crypto |" in rendered

    def test_empty_input(self) -> None:
        """Empty data collapses to a placeholder message, not a crash."""

        rendered = section_tool_category_diagnostics({})
        assert "No cross-breakdown data available" in rendered


class TestSectionToolCategoryPlatform:
    """section_tool_category_platform renders the tri-dimensional slice."""

    def test_sufficient_cell_rendered(self) -> None:
        """Cell with n >= MIN_SAMPLE_SIZE appears in the table."""
        scores = {
            "by_tool_category_platform": {
                "tool-a | crypto | omen": {
                    "brier": 0.18,
                    "brier_skill_score": 0.12,
                    "log_loss": 0.48,
                    "edge": 0.03,
                    "directional_accuracy": 0.7,
                    "n": 50,
                },
            }
        }
        rendered = section_tool_category_platform(scores)
        assert "| tool-a | crypto | omen | 0.1800" in rendered

    def test_sparse_cells_omitted(self) -> None:
        """Cells below MIN_SAMPLE_SIZE are omitted with a count footnote."""
        scores = {
            "by_tool_category_platform": {
                "tool-a | crypto | omen": {"brier": 0.1, "n": 5},
                "tool-b | crypto | omen": {"brier": 0.2, "n": 7},
            }
        }
        rendered = section_tool_category_platform(scores)
        # Neither sparse row renders inline.
        assert "| tool-a | crypto | omen | 0.1000" not in rendered
        assert "2 cell(s) below" in rendered

    def test_empty_input(self) -> None:
        """Empty input renders a placeholder."""
        assert "No cross-breakdown data available." in section_tool_category_platform(
            {}
        )


class TestGenerateFleetReport:
    """generate_fleet_report renders the cross-platform view."""

    def _fleet_scores(self) -> dict[str, Any]:
        return {
            "generated_at": "2026-03-31T06:00:00Z",
            "overall": {"brier": 0.25, "n": 100},
            "by_platform": {
                "omen": {"brier": 0.22, "n": 60},
                "polymarket": {"brier": 0.28, "n": 40},
            },
            "by_category_platform": {
                "crypto | omen": {
                    "brier": 0.20,
                    "brier_skill_score": 0.1,
                    "log_loss": 0.5,
                    "directional_accuracy": 0.7,
                    "n": 50,
                },
            },
            "by_tool_category_platform": {
                "tool-a | crypto | omen": {
                    "brier": 0.18,
                    "brier_skill_score": 0.1,
                    "log_loss": 0.48,
                    "edge": 0.03,
                    "directional_accuracy": 0.7,
                    "n": 50,
                },
            },
        }

    def test_header_names_fleet_scope(self) -> None:
        """Fleet header makes the cross-platform scope unambiguous."""
        report = generate_fleet_report(self._fleet_scores(), [])
        assert "# Benchmark Report (Fleet, Cross-Platform)" in report

    def test_includes_cross_platform_sections(self) -> None:
        """Fleet report carries the two cross-platform breakdown headings."""
        report = generate_fleet_report(self._fleet_scores(), [])
        assert "## Category × Platform" in report
        assert "## Tool × Category × Platform" in report

    def test_skips_per_platform_deep_dives(self) -> None:
        """Fleet report does not duplicate per-platform deep dives."""
        report = generate_fleet_report(self._fleet_scores(), [])
        # No rolling window content, no deployment status, no weak spots —
        # those live in the per-platform reports.
        assert "Tool Deployment Status" not in report
        assert f"(Last {ROLLING_WINDOW_DAYS} Days)" not in report
        assert "Weak Spots" not in report

    def test_trend_renders_without_disclaimer(self) -> None:
        """Fleet scope matches Trend's fleet-wide semantics — no disclaimer."""
        history = [{"month": "2026-03", "overall": {"brier": 0.25, "n": 100}}]
        report = generate_fleet_report(self._fleet_scores(), history)
        assert "## Trend (Fleet-wide, Monthly)" in report
        assert "not scoped to this report" not in report

    def test_header_notes_all_time_scope(self) -> None:
        """Fleet report intro points readers to per-platform reports for deltas."""
        report = generate_fleet_report(self._fleet_scores(), [])
        assert "All metrics here are all-time" in report
        assert "report_omen.md" in report
        assert "report_polymarket.md" in report


# ---------------------------------------------------------------------------
# Three-window comparison helpers
# ---------------------------------------------------------------------------


class TestDeltaCell:
    """_delta_cell enforces the sample-size and direction-label contract."""

    def test_delta_suppressed_when_current_below_min(self) -> None:
        """Low current-window n yields insufficient data, not a signed number."""

        assert _delta_cell(0.2, 0.3, current_n=5, reference_n=1000) == (
            "insufficient data"
        )

    def test_delta_suppressed_when_reference_below_min(self) -> None:
        """Low reference-window n yields insufficient data."""

        assert _delta_cell(0.2, 0.3, current_n=1000, reference_n=5) == (
            "insufficient data"
        )

    def test_none_values_yield_na(self) -> None:
        """Missing values collapse to N/A rather than arithmetic error."""

        assert _delta_cell(None, 0.3, 1000, 1000) == "N/A"
        assert _delta_cell(0.2, None, 1000, 1000) == "N/A"

    def test_lower_is_better_direction(self) -> None:
        """For Brier-like metrics, negative delta = better."""

        rendered = _delta_cell(0.20, 0.25, 100, 100, lower_is_better=True)
        assert rendered.startswith("-0.0500")
        assert "better" in rendered

    def test_higher_is_better_direction(self) -> None:
        """For BSS / directional accuracy, positive delta = better."""

        rendered = _delta_cell(0.75, 0.70, 100, 100, lower_is_better=False)
        assert rendered.startswith("+0.0500")
        assert "better" in rendered

    def test_zero_delta_labeled_same(self) -> None:
        """Delta of exactly zero renders as same, not better/worse."""

        rendered = _delta_cell(0.20, 0.20, 100, 100)
        assert "same" in rendered


class TestSectionPlatformSnapshot:
    """section_platform_snapshot renders the current-window overall metrics."""

    def test_renders_all_snapshot_metrics(self) -> None:
        """Every reviewer-requested snapshot metric appears in the output."""

        scores = {
            "overall": {
                "n": 200,
                "valid_n": 190,
                "reliability": 0.95,
                "brier": 0.22,
                "baseline_brier": 0.25,
                "brier_skill_score": 0.12,
                "directional_accuracy": 0.70,
                "outcome_yes_rate": 0.55,
            }
        }
        rendered = section_platform_snapshot(scores)
        for label in (
            "n",
            "Reliability",
            "Brier",
            "Baseline Brier",
            "BSS",
            "Directional Accuracy",
            "Outcome Yes Rate",
            "Outcome No Rate",
            "Always-majority baseline",
            "DA lift",
        ):
            assert label in rendered
        # DA lift = 0.70 - max(0.55, 0.45) = +0.15
        assert "+0.1500" in rendered

    def test_empty_scores_renders_placeholder(self) -> None:
        """Zero rows collapses to a placeholder, not a crash."""

        assert "No rows scored" in section_platform_snapshot({"overall": {"n": 0}})


class TestSectionPlatformComparison:
    """section_platform_comparison threads the three windows correctly."""

    def _rolling(self) -> dict:
        return {
            "overall": {
                "n": 200,
                "valid_n": 190,
                "reliability": 0.95,
                "brier": 0.22,
                "baseline_brier": 0.25,
                "brier_skill_score": 0.12,
                "directional_accuracy": 0.70,
                "log_loss": 0.50,
            }
        }

    def _alltime(self) -> dict:
        return {
            "overall": {
                "n": 5000,
                "valid_n": 4800,
                "reliability": 0.93,
                "brier": 0.24,
                "baseline_brier": 0.26,
                "brier_skill_score": 0.08,
                "directional_accuracy": 0.68,
                "log_loss": 0.52,
            }
        }

    def _prev(self) -> dict:
        return {
            "overall": {
                "n": 200,
                "valid_n": 185,
                "reliability": 0.94,
                "brier": 0.23,
                "baseline_brier": 0.25,
                "brier_skill_score": 0.10,
                "directional_accuracy": 0.69,
                "log_loss": 0.51,
            }
        }

    def test_three_window_table_header(self) -> None:
        """Header names current, all-time, and prev windows with delta columns."""

        rendered = section_platform_comparison(
            self._rolling(), self._alltime(), self._prev()
        )
        assert f"Current {ROLLING_WINDOW_DAYS}d" in rendered
        assert "All-Time" in rendered
        assert f"Prev {ROLLING_WINDOW_DAYS}d" in rendered
        assert "Δ vs All-Time" in rendered
        assert f"Δ vs Prev {ROLLING_WINDOW_DAYS}d" in rendered

    def test_no_prev_window_when_prev_is_none(self) -> None:
        """Prev column renders 'no prev window' placeholder instead of a delta."""

        rendered = section_platform_comparison(self._rolling(), self._alltime(), None)
        assert "no prev window" in rendered

    def test_brier_row_renders_signed_delta(self) -> None:
        """Brier delta sign + direction word appears in the table body."""

        rendered = section_platform_comparison(
            self._rolling(), self._alltime(), self._prev()
        )
        # current=0.22, all-time=0.24 → delta=-0.02, better (Brier is
        # lower-is-better).
        assert "-0.0200 better" in rendered


class TestSectionToolComparison:
    """section_tool_comparison ranks tools and threads deltas correctly."""

    def _s(self, tools: dict[str, tuple[float | None, int]]) -> dict:
        return {
            "by_tool": {
                name: {
                    "brier": brier,
                    "valid_n": n,
                    "n": n,
                    "decision_worthy": n >= 30,
                }
                for name, (brier, n) in tools.items()
            }
        }

    def test_tool_row_cites_current_value_and_deltas(self) -> None:
        """Each row shows the current Brier, all-time delta, and prev delta."""

        rolling = self._s({"tool-a": (0.22, 100)})
        alltime = self._s({"tool-a": (0.25, 5000)})
        prev = self._s({"tool-a": (0.24, 100)})
        rendered = section_tool_comparison(rolling, alltime, prev)
        assert "**tool-a**" in rendered
        assert "0.2200 (n=100)" in rendered
        assert "better" in rendered

    def test_no_prev_window_placeholder(self) -> None:
        """None prev-rolling renders the placeholder instead of N/A or empty."""

        rolling = self._s({"tool-a": (0.22, 100)})
        alltime = self._s({"tool-a": (0.25, 5000)})
        rendered = section_tool_comparison(rolling, alltime, None)
        assert "no prev window" in rendered

    def test_empty_universe_renders_placeholder(self) -> None:
        """No tools anywhere collapses to a placeholder."""

        rendered = section_tool_comparison({}, {}, None)
        assert "No tool data available." in rendered


class TestSectionDiagnosticsComparisonPlaceholder:
    """Diagnostics table renders an explicit placeholder when all cells skip.

    Before the ``len(lines) == 4`` fix the table would emit a bare
    header + separator with no body rows when every metric cell hit
    the ``all three windows None`` continue — readers saw an empty
    table and had to guess whether that meant "no data" or "rendering
    bug".
    """

    def test_placeholder_rendered_when_every_cell_skips(self) -> None:
        """Tool with no edge / log_loss / ... keys triggers the placeholder."""
        # tool-a exists in by_tool so _tool_universe yields it, but
        # none of the five diagnostic metric keys are populated on any
        # window — every (tool, metric) pair hits the continue branch.
        bare_tool = {"by_tool": {"tool-a": {"n": 100, "valid_n": 90}}}
        rendered = section_diagnostics_comparison(bare_tool, bare_tool, bare_tool)
        assert "_(no diagnostic data)_" in rendered

    def test_placeholder_absent_when_any_metric_populated(self) -> None:
        """Placeholder only fires when no (tool, metric) cell renders."""
        with_edge = {
            "by_tool": {
                "tool-a": {
                    "n": 100,
                    "valid_n": 90,
                    "edge": 0.02,
                    "edge_n": 80,
                }
            }
        }
        rendered = section_diagnostics_comparison(with_edge, with_edge, with_edge)
        assert "_(no diagnostic data)_" not in rendered
        assert "**tool-a** | Edge" in rendered


class TestZeroRowPrevRollingRouting:
    """Zero-row prev_rolling_scores routes to the "no prev window" placeholder.

    A prev-scoring CI step that succeeds on an empty window writes
    ``{"total_rows": 0, "overall": {}}`` to disk. Pre-normalization,
    ``prev is not None`` was True in the comparison sections and the
    "no prev window" placeholder never fired — readers saw ``N/A (n=0)``
    cells that read as "we measured zero rows" instead of "no reference
    window available".
    """

    def _scores(self, tool: str, brier: float, n: int) -> dict:
        return _scores_with_tool(tool, brier, n)

    def test_empty_prev_rolling_routes_to_no_prev_window(self) -> None:
        """Zero total_rows on prev_rolling_scores renders the explicit placeholder."""
        scores = self._scores("tool-a", 0.22, 1000)
        rolling = self._scores("tool-a", 0.20, 100)
        empty_prev = {"total_rows": 0, "overall": {}}
        report = generate_report(
            scores,
            [],
            platform="omen",
            rolling_scores=rolling,
            prev_rolling_scores=empty_prev,
            disabled_tools={},
        )
        assert "no prev window" in report

    def test_missing_prev_rolling_and_empty_prev_rolling_render_the_same(
        self,
    ) -> None:
        """Zero-row prev and None prev emit the same user-facing copy."""
        scores = self._scores("tool-a", 0.22, 1000)
        rolling = self._scores("tool-a", 0.20, 100)
        empty_prev = {"total_rows": 0, "overall": {}}
        report_empty = generate_report(
            scores,
            [],
            platform="omen",
            rolling_scores=rolling,
            prev_rolling_scores=empty_prev,
            disabled_tools={},
        )
        report_none = generate_report(
            scores,
            [],
            platform="omen",
            rolling_scores=rolling,
            prev_rolling_scores=None,
            disabled_tools={},
        )
        # The date stamps in the header carry ``generated_at`` which is
        # identical across both calls for a single-scorer fixture, so the
        # full reports compare equal.
        assert report_empty == report_none

    def test_populated_prev_rolling_still_renders_deltas(self) -> None:
        """The zero-row guard does not accidentally swallow real prev data."""
        scores = self._scores("tool-a", 0.22, 1000)
        rolling = self._scores("tool-a", 0.20, 100)
        populated_prev = self._scores("tool-a", 0.21, 100)
        report = generate_report(
            scores,
            [],
            platform="omen",
            rolling_scores=rolling,
            prev_rolling_scores=populated_prev,
            disabled_tools={},
        )
        assert "no prev window" not in report

    def test_prev_value_and_delta_cells_agree_when_no_prev_window(self) -> None:
        """When prev is None, the prev-value cell and delta cell render identically.

        Regression test for a consistency bug where the delta cell
        said ``no prev window`` but the prev-value cell on the same row
        rendered ``N/A (n=0)`` — two different messages for the same
        state in neighboring columns of a single row.
        """
        scores = self._scores("tool-a", 0.22, 1000)
        rolling = self._scores("tool-a", 0.20, 100)
        report = generate_report(
            scores,
            [],
            platform="omen",
            rolling_scores=rolling,
            prev_rolling_scores=None,
            disabled_tools={},
        )
        # The value-cell on the prev column must carry "no prev window"
        # wherever a delta cell does; an "N/A (n=0)" leaking through
        # would mean a no-prev-window row is claiming it measured the
        # previous window and found zero rows.
        for line in report.splitlines():
            if line.startswith("|") and "no prev window" in line:
                assert "N/A (n=0)" not in line, line


class TestSectionReliabilityComparison:
    """section_reliability_comparison shows reliability and valid % deltas."""

    def test_reliability_regression_labeled_worse(self) -> None:
        """Tool whose reliability dropped renders with 'worse' direction."""

        rolling = {
            "by_tool": {"tool-a": {"reliability": 0.85, "n": 100}},
            "parse_breakdown": {"tool-a": {"valid": 80, "malformed": 20}},
        }
        alltime = {
            "by_tool": {"tool-a": {"reliability": 0.95, "n": 5000}},
            "parse_breakdown": {"tool-a": {"valid": 4750, "malformed": 250}},
        }
        rendered = section_reliability_comparison(rolling, alltime)
        assert "**tool-a**" in rendered
        assert "worse" in rendered

    def test_low_sample_suppresses_delta(self) -> None:
        """Low-n reliability delta renders insufficient data, not a signed %."""

        rolling = {
            "by_tool": {"tool-a": {"reliability": 0.85, "n": 5}},
            "parse_breakdown": {"tool-a": {"valid": 4, "malformed": 1}},
        }
        alltime = {
            "by_tool": {"tool-a": {"reliability": 0.95, "n": 5000}},
            "parse_breakdown": {"tool-a": {"valid": 4750, "malformed": 250}},
        }
        rendered = section_reliability_comparison(rolling, alltime)
        assert "insufficient data" in rendered


class TestActiveToolsForPlatform:
    """``_active_tools_for_platform`` builds the per-platform deployed set."""

    _SCORES: dict[str, dict[str, dict[str, Any]]] = {
        "by_tool": {"tool-a": {}, "tool-b": {}, "tool-c": {}}
    }

    def test_full_failure_returns_none(self) -> None:
        """All deployments None -> caller falls back to "show all" with a notice."""
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": None,
            "omenstrat QS": None,
            "polystrat Pearl": None,
        }
        assert _active_tools_for_platform(disabled, "omen", self._SCORES) is None

    def test_partial_failure_uses_what_is_available(self) -> None:
        """One omen deployment fails, the other still drives the active set."""
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["tool-a"],
            "omenstrat QS": None,
            "polystrat Pearl": None,
        }
        active = _active_tools_for_platform(disabled, "omen", self._SCORES)
        # tool-a is disabled on Pearl and QS is unknown; union of (benchmarked - disabled)
        # over successful deployments is {tool-b, tool-c}.
        assert active == frozenset({"tool-b", "tool-c"})

    def test_polystrat_failure_returns_none(self) -> None:
        """Polystrat has one deployment — its failure means full failure."""
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": [],
            "omenstrat QS": [],
            "polystrat Pearl": None,
        }
        assert _active_tools_for_platform(disabled, "polymarket", self._SCORES) is None

    def test_union_across_deployments(self) -> None:
        """A tool active on at least one deployment is in the active set."""
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["tool-a"],
            "omenstrat QS": ["tool-b"],
            "polystrat Pearl": None,
        }
        active = _active_tools_for_platform(disabled, "omen", self._SCORES)
        # Pearl excludes tool-a (so {tool-b, tool-c}); QS excludes tool-b (so
        # {tool-a, tool-c}); union = {tool-a, tool-b, tool-c}.
        assert active == frozenset({"tool-a", "tool-b", "tool-c"})

    def test_empty_disabled_returns_none(self) -> None:
        """Empty input is the test-only opt-out — caller skips both filter and warning."""
        assert _active_tools_for_platform({}, "omen", self._SCORES) is None
        assert _active_tools_for_platform(None, "omen", self._SCORES) is None

    def test_underscore_hyphen_normalization(self) -> None:
        """Disabled list with underscores excludes the hyphenated benchmark name."""
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["tool_a"],  # underscore variant of "tool-a"
            "omenstrat QS": ["tool_a"],
            "polystrat Pearl": None,
        }
        active = _active_tools_for_platform(disabled, "omen", self._SCORES)
        # Both deployments disable tool-a (underscore variant), so it should
        # NOT appear in the active set.
        assert active is not None
        assert "tool-a" not in active
        assert "tool-b" in active and "tool-c" in active


class TestFilterByActive:
    """``_filter_by_active`` drops non-deployed tools from ranked iterations."""

    def test_none_active_disables_filter(self) -> None:
        """``active_tools=None`` is the deployment-config-fallback path: no filter."""
        items: list[tuple[str, Any]] = [("tool-a", {}), ("tool-b", {})]
        assert _filter_by_active(items, None) == items

    def test_simple_key_filter(self) -> None:
        """Plain tool-name keys are filtered by membership."""
        items: list[tuple[str, Any]] = [("tool-a", {}), ("tool-b", {}), ("tool-c", {})]
        out = _filter_by_active(items, frozenset({"tool-a", "tool-c"}))
        assert [k for k, _ in out] == ["tool-a", "tool-c"]

    def test_composite_key_filter(self) -> None:
        """``by_tool_category`` keys filter on the tool half (before separator)."""
        items: list[tuple[str, Any]] = [
            ("tool-a | politics", {}),
            ("tool-b | politics", {}),
            ("tool-a | finance", {}),
        ]
        out = _filter_by_active(
            items, frozenset({"tool-a"}), composite_key_separator=" | "
        )
        assert [k for k, _ in out] == ["tool-a | politics", "tool-a | finance"]


class TestActiveToolsFilterInSections:
    """End-to-end: each comparison section drops tools not in active_tools."""

    def _scores(self, brier: float, n: int) -> dict[str, Any]:
        return {
            "valid_n": n,
            "n": n,
            "brier": brier,
            "reliability": 0.95,
            "directional_accuracy": 0.7,
        }

    def _by_tool(self) -> dict[str, dict[str, Any]]:
        return {
            "tool-a": self._scores(0.20, 200),
            "tool-b": self._scores(0.30, 200),
            "tool-historical": self._scores(0.15, 200),
        }

    def test_tool_comparison_drops_non_active(self) -> None:
        """Tool Historical Comparison hides tool-historical when filter is active."""
        rolling = {"by_tool": self._by_tool()}
        alltime = {"by_tool": self._by_tool()}
        rendered = section_tool_comparison(
            rolling, alltime, None, active_tools=frozenset({"tool-a", "tool-b"})
        )
        assert "**tool-a**" in rendered
        assert "**tool-b**" in rendered
        assert "**tool-historical**" not in rendered

    def test_tool_comparison_no_filter_when_none(self) -> None:
        """``active_tools=None`` preserves the legacy "all tools" rendering."""
        rolling = {"by_tool": self._by_tool()}
        alltime = {"by_tool": self._by_tool()}
        rendered = section_tool_comparison(rolling, alltime, None, active_tools=None)
        assert "**tool-historical**" in rendered

    def test_tool_category_drops_non_active_cells(self) -> None:
        """Tool × Category snapshot hides cells whose tool is not in active set."""
        scores = {
            "by_tool_category": {
                "tool-a | politics": self._scores(0.20, 200),
                "tool-historical | politics": self._scores(0.15, 200),
            }
        }
        rendered = section_tool_category(scores, active_tools=frozenset({"tool-a"}))
        assert "tool-a | politics" in rendered or (
            "tool-a" in rendered and "politics" in rendered
        )
        assert "tool-historical" not in rendered

    def test_diagnostics_comparison_drops_non_active(self) -> None:
        """Diagnostics Historical Comparison hides non-deployed tools."""
        rolling = {
            "by_tool": {
                "tool-a": {
                    "edge": 0.05,
                    "edge_n": 200,
                    "log_loss": 0.5,
                    "valid_n": 200,
                },
                "tool-historical": {
                    "edge": 0.10,
                    "edge_n": 200,
                    "log_loss": 0.4,
                    "valid_n": 200,
                },
            }
        }
        alltime = rolling
        rendered = section_diagnostics_comparison(
            rolling, alltime, None, active_tools=frozenset({"tool-a"})
        )
        assert "**tool-a**" in rendered
        assert "**tool-historical**" not in rendered

    def test_reliability_comparison_drops_non_active(self) -> None:
        """Reliability & Parse Quality table drops non-deployed tools."""
        rolling = {
            "by_tool": {
                "tool-a": {"reliability": 0.95, "n": 200},
                "tool-historical": {"reliability": 0.80, "n": 200},
            },
            "parse_breakdown": {
                "tool-a": {"valid": 190, "malformed": 10},
                "tool-historical": {"valid": 160, "malformed": 40},
            },
        }
        alltime = rolling
        rendered = section_reliability_comparison(
            rolling, alltime, active_tools=frozenset({"tool-a"})
        )
        assert "**tool-a**" in rendered
        assert "**tool-historical**" not in rendered


class TestGenerateReportDeploymentConfigUnavailable:
    """Full deployment-fetch failure renders a notice + falls back to all tools."""

    def test_notice_rendered_when_all_deployments_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the live fetch returns all-None, a ⚠ notice prefixes the report."""
        # pylint: disable=import-outside-toplevel
        from benchmark import analyze

        monkeypatch.setattr(
            analyze,
            "fetch_disabled_tools",
            lambda: {
                "omenstrat Pearl": None,
                "omenstrat QS": None,
                "polystrat Pearl": None,
            },
        )
        scores = {
            "by_tool": {"tool-a": {"brier": 0.20, "n": 100, "valid_n": 100}},
            "by_tool_category": {},
        }
        report = generate_report(scores, [], platform="omen")
        assert "Deployment config unavailable" in report

    def test_no_notice_for_explicit_test_optout(self) -> None:
        """``disabled_tools={}`` is the unit-test opt-out; no notice rendered."""
        scores = {
            "by_tool": {"tool-a": {"brier": 0.20, "n": 100, "valid_n": 100}},
            "by_tool_category": {},
        }
        report = generate_report(scores, [], platform="omen", disabled_tools={})
        assert "Deployment config unavailable" not in report
