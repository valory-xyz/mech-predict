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
"""Tests for benchmark/notify_slack.py — platform-scoped Slack summaries."""

from pathlib import Path

import pytest
from benchmark.analyze import PLATFORM_LABELS, ROLLING_WINDOW_DAYS
from benchmark.notify_slack import (
    _build_system_prompt,
    _compute_top_k,
    _count_eligible_tools,
    _infer_platform_label,
)
from benchmark.scorer import MIN_SAMPLE_SIZE


# A "headline" prompt used by structural tests that don't care about the
# ranking-block dispatch — eligible_count=5 puts the dispatcher in the
# Top-K + Worst-K branch with K = 2, exercising the most common shape.
def _default_prompt(label: str = "Omenstrat") -> str:
    return _build_system_prompt(label, eligible_count=5)


class TestBuildSystemPrompt:
    """_build_system_prompt threads the deployment label through the template."""

    def test_omenstrat_label_appears_in_prompt(self) -> None:
        """Omenstrat label renders into the Summary / Category / Actions sections."""
        prompt = _default_prompt("Omenstrat")
        assert "Omenstrat" in prompt
        assert "for the *Omenstrat* deployment" in prompt
        assert "for Omenstrat" in prompt

    def test_polystrat_label_appears_in_prompt(self) -> None:
        """Polystrat label renders symmetrically to Omenstrat."""
        prompt = _default_prompt("Polystrat")
        assert "Polystrat" in prompt
        assert "for the *Polystrat* deployment" in prompt

    def test_no_cross_platform_leakage(self) -> None:
        """Omenstrat prompt must not reference Polystrat and vice versa."""
        omen = _default_prompt("Omenstrat")
        assert "Polystrat" not in omen

        poly = _default_prompt("Polystrat")
        assert "Omenstrat" not in poly

    def test_template_no_longer_instructs_platform_comparison(self) -> None:
        """Single-platform summaries must not ask the LLM to 'list all platforms'.

        The legacy fleet-wide prompt had ``*Platform performance:*`` and
        ``*Edge by difficulty:* ... per platform`` blocks. Both are
        meaningless in per-platform mode and must not bleed into the
        per-platform template regardless of which ranking-block branch
        the dispatcher lands in.
        """
        for n in (0, 1, 3, 5, 10):
            prompt = _build_system_prompt("Omenstrat", eligible_count=n)
            assert "list all platforms" not in prompt
            assert "one line per platform" not in prompt
            assert "Platform × Difficulty" not in prompt
            assert "*Platform performance:*" not in prompt

    def test_template_still_carries_core_sections(self) -> None:
        """Core single-platform footer sections remain wired up."""
        prompt = _default_prompt()
        for heading in (
            "*Summary:*",
            "*Tool × Category:*",
            "*Tournament callouts:*",
            "*Diagnostics:*",
            "*Reliability:*",
            "*Recommended actions:*",
        ):
            assert heading in prompt, f"missing: {heading}"

    def test_prompt_references_rolling_window_days_constant(self) -> None:
        """Prompt cites the current ROLLING_WINDOW_DAYS value in its window labels."""
        prompt = _default_prompt()
        assert f"Current {ROLLING_WINDOW_DAYS}d" in prompt
        assert f"Prev {ROLLING_WINDOW_DAYS}d" in prompt

    def test_prompt_drops_alltime_scope_instructions(self) -> None:
        """Prompt no longer tells the LLM to cite all-time or cumulative figures."""
        prompt = _default_prompt()
        assert "Only mention all-time numbers for context" not in prompt
        # The prompt still refers to "All-Time" as a window label, but not
        # as a bolt-on scope that the LLM should opportunistically mix in.
        assert "deltas vs all-time" not in prompt

    def test_prompt_anchors_sections_to_comparison_heading_names(self) -> None:
        """Prompt points the LLM at the new three-window comparison headings."""
        prompt = _default_prompt()
        for heading in (
            "Platform Snapshot",
            "Platform Historical Comparison",
            "Tool Historical Comparison",
            f"Tool × Category (Current {ROLLING_WINDOW_DAYS}d)",
            "Tool × Category Historical Comparison",
            "Diagnostics Historical Comparison",
            "Reliability & Parse Quality",
        ):
            assert heading in prompt, f"missing: {heading}"

    def test_prompt_enforces_no_mixed_window_claims(self) -> None:
        """Every cited number must be paired with its window label."""
        prompt = _default_prompt()
        assert "Never mix windows" in prompt
        assert "insufficient data" in prompt
        assert "no prev window" in prompt

    def test_deployment_status_points_at_platform_scoped_section(self) -> None:
        """Deployment status bullet anchors to the per-platform section heading."""
        prompt = _default_prompt()
        assert '"Tool Deployment Status (Omenstrat)"' in prompt
        assert "count of active tools only" in prompt
        assert "do NOT enumerate the tool names" in prompt
        assert "`⚠️ unavailable`" in prompt

    def test_tool_category_prompt_lists_every_qualifying_cell(self) -> None:
        """Tool × Category bullet instructs the LLM to list every qualifying cell."""
        prompt = _default_prompt()
        assert "list every cell that clears the sample-size threshold" in prompt
        assert "insufficient tool × category data" in prompt


class TestInferPlatformLabel:
    """_infer_platform_label recovers the deployment label from the filename."""

    def test_omen_report(self) -> None:
        """report_omen.md -> Omenstrat."""
        assert _infer_platform_label(Path("/tmp/report_omen.md")) == "Omenstrat"

    def test_polymarket_report(self) -> None:
        """report_polymarket.md -> Polystrat."""
        assert _infer_platform_label(Path("/tmp/report_polymarket.md")) == "Polystrat"

    def test_unknown_stem_returns_none(self) -> None:
        """Unrecognised filenames get None so the caller can error explicitly."""
        assert _infer_platform_label(Path("/tmp/report.md")) is None
        assert _infer_platform_label(Path("/tmp/report_gnosis.md")) is None


class TestPromptRejectsUnformattedPlaceholder:
    """Guard against a missed ``{platform_label}`` replacement."""

    def test_build_raises_on_empty_label(self) -> None:
        """Empty label is rejected — would render "for the ** deployment"."""
        with pytest.raises(ValueError, match="platform_label"):
            _build_system_prompt("", eligible_count=5)

    def test_build_raises_on_unknown_label(self) -> None:
        """A label outside PLATFORM_LABELS is rejected before reaching the LLM."""
        with pytest.raises(ValueError, match="must be one of"):
            _build_system_prompt("Omenstrap", eligible_count=5)

    def test_labels_tracked_from_analyze(self) -> None:
        """Every ``benchmark.analyze.PLATFORM_LABELS`` value is accepted."""
        for label in PLATFORM_LABELS.values():
            _build_system_prompt(label, eligible_count=5)

    def test_no_unfilled_placeholder_in_rendered_prompt(self) -> None:
        """Rendered prompt has no surviving ``{platform_label}`` after dispatch."""
        for n in (0, 1, 3, 5, 10):
            prompt = _build_system_prompt("Omenstrat", eligible_count=n)
            assert "{platform_label}" not in prompt


class TestEligibilityBlock:
    """Header carries the ``MIN_SAMPLE_SIZE`` floor and ⚠ flag exclusion rule."""

    def test_min_sample_size_floor_named_in_eligibility_block(self) -> None:
        """Eligibility block cites the live ``MIN_SAMPLE_SIZE`` constant."""
        prompt = _default_prompt()
        assert "Eligibility for the tool ranking section" in prompt
        assert f"at least {MIN_SAMPLE_SIZE}" in prompt

    def test_eligibility_excludes_low_sample_and_malformed(self) -> None:
        """Eligibility block names both ⚠ flags so neither leaks into rankings."""
        prompt = _default_prompt()
        assert "⚠ low sample" in prompt
        assert "⚠ all malformed" in prompt


class TestComputeTopK:
    """``_compute_top_k`` keeps Top and Worst slices disjoint at every N.

    Constraint: Top K + Worst K rows must come from disjoint regions of
    the sorted eligible list, so ``2 * K < N``. The dispatcher returns
    ``0`` for ``N <= 2`` to switch to a combined "Tool performance"
    listing instead of a useless 1-vs-1 split.
    """

    @pytest.mark.parametrize(
        "eligible,expected_k",
        [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 1),
            (4, 1),
            (5, 2),
            (6, 2),
            (7, 3),
            (8, 3),
            (10, 3),
            (50, 3),
        ],
    )
    def test_k_satisfies_disjoint_constraint(
        self, eligible: int, expected_k: int
    ) -> None:
        """Returned K matches the table from the design doc."""
        assert _compute_top_k(eligible) == expected_k

    @pytest.mark.parametrize("eligible", list(range(0, 50)))
    def test_top_and_worst_are_always_disjoint(self, eligible: int) -> None:
        """For every N, ``2 * K < N`` (or K = 0 to disable the split)."""
        k = _compute_top_k(eligible)
        if k > 0:
            assert 2 * k < eligible, f"N={eligible}, K={k}: top+worst overlap"

    def test_capped_at_three_for_large_eligible_sets(self) -> None:
        """K never exceeds 3 — keeps the Slack message scannable."""
        assert _compute_top_k(100) == 3
        assert _compute_top_k(1000) == 3


class TestCountEligibleTools:
    """``_count_eligible_tools`` parses the markdown for ranking-block dispatch."""

    def test_counts_rows_above_floor_only(self) -> None:
        """Rows below ``MIN_SAMPLE_SIZE`` don't contribute."""
        report = (
            "## Tool Historical Comparison\n"
            "\n"
            "| Tool | Current 7d Brier | All-Time | Δ |\n"
            "|------|------------------|----------|---|\n"
            f"| **good-tool** | 0.1 (n={MIN_SAMPLE_SIZE}) | x | x |\n"
            f"| **below-floor** | 0.1 (n={MIN_SAMPLE_SIZE - 1}) | x | x |\n"
            "\n"
            "## Next Section\n"
        )
        assert _count_eligible_tools(report) == 1

    def test_drops_low_sample_flagged_rows(self) -> None:
        """Rows carrying ``⚠ low sample`` are excluded even if n is high."""
        report = (
            "## Tool Historical Comparison\n"
            "\n"
            "| Tool | Current 7d Brier | All-Time | Δ |\n"
            "|------|------------------|----------|---|\n"
            "| **good-tool** | 0.1 (n=100) | x | x |\n"
            "| **flagged-tool** ⚠ low sample | 0.0 (n=200) | x | x |\n"
            "\n"
            "## Next Section\n"
        )
        assert _count_eligible_tools(report) == 1

    def test_drops_all_malformed_flagged_rows(self) -> None:
        """``⚠ all malformed`` rows are excluded — same eligibility contract."""
        report = (
            "## Tool Historical Comparison\n"
            "\n"
            "| Tool | Current 7d Brier | All-Time | Δ |\n"
            "|------|------------------|----------|---|\n"
            "| **good-tool** | 0.1 (n=100) | x | x |\n"
            "| **broken-tool** ⚠ all malformed | N/A (n=200) | x | x |\n"
            "\n"
            "## Next Section\n"
        )
        assert _count_eligible_tools(report) == 1

    def test_returns_zero_when_section_absent(self) -> None:
        """Reports without a Tool Historical Comparison section count zero."""
        assert _count_eligible_tools("# Some other report\n") == 0

    def test_returns_zero_for_empty_table(self) -> None:
        """A section heading with no data rows counts zero."""
        report = (
            "## Tool Historical Comparison\n"
            "\n"
            "No tool data available.\n"
            "\n"
            "## Next Section\n"
        )
        assert _count_eligible_tools(report) == 0

    def test_counts_three_when_all_pass_floor(self) -> None:
        """Three rows above the floor and no flags -> three eligible."""
        report = (
            "## Tool Historical Comparison\n"
            "\n"
            "| Tool | Current 7d Brier | All-Time | Δ |\n"
            "|------|------------------|----------|---|\n"
            "| **a** | 0.1 (n=73) | x | x |\n"
            "| **b** | 0.2 (n=79) | x | x |\n"
            "| **c** | 0.3 (n=714) | x | x |\n"
            "\n"
            "## Next Section\n"
        )
        assert _count_eligible_tools(report) == 3

    def test_counts_when_section_is_last_in_report(self) -> None:
        r"""Block parser also terminates at end-of-report.

        Without the ``\Z`` anchor, the regex needs another ``^## ``
        heading to close the block. If a future analyze.py reorder ever
        lands Tool Historical Comparison as the final section, the
        helper would silently return 0 and every report would render
        the "no eligible tools" placeholder. Pin the contract.
        """
        report = (
            "## Tool Historical Comparison\n"
            "\n"
            "| Tool | Current 7d Brier | All-Time | Δ |\n"
            "|------|------------------|----------|---|\n"
            "| **a** | 0.1 (n=73) | x | x |\n"
            "| **b** | 0.2 (n=79) | x | x |\n"
        )
        assert _count_eligible_tools(report) == 2


class TestRankingBlockDispatch:
    """The prompt only ever exposes ONE section convention per request.

    This is what makes the dispatch deterministic: when N <= 2, the LLM
    sees ``*Tool performance:*`` and never sees Top/Worst. When N >= 3,
    the LLM sees Top/Worst with a specific K and never sees Tool
    performance. There is no "skip" instruction the LLM has to obey —
    the prohibited section is simply not in the prompt.
    """

    @pytest.mark.parametrize("eligible", [0, 1, 2])
    def test_small_n_uses_tool_performance_only(self, eligible: int) -> None:
        """N <= 2 -> Tool performance; Top/Worst absent from the prompt."""
        prompt = _build_system_prompt("Omenstrat", eligible_count=eligible)
        assert "*Tool performance:*" in prompt
        assert "*Top tools:*" not in prompt
        assert "*Worst tools:*" not in prompt

    @pytest.mark.parametrize(
        "eligible,top_k", [(3, 1), (4, 1), (5, 2), (6, 2), (7, 3), (10, 3)]
    )
    def test_large_n_uses_top_worst_only(self, eligible: int, top_k: int) -> None:
        """N >= 3 -> Top/Worst with the right K; Tool performance absent."""
        prompt = _build_system_prompt("Omenstrat", eligible_count=eligible)
        assert "*Tool performance:*" not in prompt
        assert "*Top tools:*" in prompt
        assert "*Worst tools:*" in prompt
        assert f"top {top_k} eligible rows" in prompt
        assert f"bottom {top_k} eligible rows" in prompt

    def test_zero_eligible_renders_explicit_placeholder(self) -> None:
        """Zero-eligible case has a deterministic placeholder so the LLM doesn't guess."""
        prompt = _build_system_prompt("Omenstrat", eligible_count=0)
        assert "no eligible tools" in prompt

    def test_one_eligible_lists_the_single_tool(self) -> None:
        """One eligible tool -> Tool performance with one bullet, no placeholder."""
        prompt = _build_system_prompt("Omenstrat", eligible_count=1)
        assert "list ALL eligible rows" in prompt
        assert "no eligible tools" not in prompt
