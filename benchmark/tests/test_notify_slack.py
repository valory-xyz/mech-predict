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
    SUMMARY_SYSTEM_PROMPT_TEMPLATE,
    _build_system_prompt,
    _infer_platform_label,
)


class TestBuildSystemPrompt:
    """_build_system_prompt threads the deployment label through the template."""

    def test_omenstrat_label_appears_in_prompt(self) -> None:
        """Omenstrat label renders into the Summary / Category / Actions sections."""
        prompt = _build_system_prompt("Omenstrat")
        assert "Omenstrat" in prompt
        # Spot-check the sentences that should carry the label so a future
        # template refactor doesn't silently drop deployment scoping.
        assert "for the *Omenstrat* deployment" in prompt
        assert "for Omenstrat" in prompt

    def test_polystrat_label_appears_in_prompt(self) -> None:
        """Polystrat label renders symmetrically to Omenstrat."""
        prompt = _build_system_prompt("Polystrat")
        assert "Polystrat" in prompt
        assert "for the *Polystrat* deployment" in prompt

    def test_no_cross_platform_leakage(self) -> None:
        """Omenstrat prompt must not reference Polystrat and vice versa."""
        omen = _build_system_prompt("Omenstrat")
        assert "Polystrat" not in omen

        poly = _build_system_prompt("Polystrat")
        assert "Omenstrat" not in poly

    def test_template_no_longer_instructs_platform_comparison(self) -> None:
        """Single-platform summaries must not ask the LLM to 'list all platforms'.

        The legacy fleet-wide prompt had ``*Platform performance:* list all
        platforms`` and ``*Edge by difficulty:* ... per platform`` blocks.
        Both are meaningless in per-platform mode and must not bleed into
        the new template.
        """
        # Check the raw template so the assertion is independent of any
        # particular platform substitution.
        assert "list all platforms" not in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "one line per platform" not in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "Platform × Difficulty" not in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        # "*Platform performance:*" was a dedicated bullet in the old prompt.
        assert "*Platform performance:*" not in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_template_still_carries_core_sections(self) -> None:
        """Core single-platform sections remain wired up after the refactor."""
        assert "*Summary:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Top tools:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Worst tools:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Tool × Category:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Tournament callouts:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Diagnostics:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Reliability:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Recommended actions:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_prompt_references_rolling_window_days_constant(self) -> None:
        """Prompt cites the current ROLLING_WINDOW_DAYS value in its window labels."""
        assert f"Current {ROLLING_WINDOW_DAYS}d" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert f"Prev {ROLLING_WINDOW_DAYS}d" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_prompt_drops_alltime_scope_instructions(self) -> None:
        """Prompt no longer tells the LLM to cite all-time or cumulative figures.

        Phase 2 drops the all-time point-in-time sections from the report, so
        the summary must not instruct the LLM to reference them.
        """
        assert "Only mention all-time numbers for context" not in (
            SUMMARY_SYSTEM_PROMPT_TEMPLATE
        )
        # The prompt still refers to "All-Time" as a window label, but not
        # as a bolt-on scope that the LLM should opportunistically mix in.
        assert "deltas vs all-time" not in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_prompt_anchors_sections_to_comparison_heading_names(self) -> None:
        """Prompt points the LLM at the new three-window comparison headings."""
        for heading in (
            "Platform Snapshot",
            "Platform Historical Comparison",
            "Tool Historical Comparison",
            f"Tool × Category (Current {ROLLING_WINDOW_DAYS}d)",
            "Tool × Category Historical Comparison",
            "Diagnostics Historical Comparison",
            "Reliability & Parse Quality",
        ):
            assert heading in SUMMARY_SYSTEM_PROMPT_TEMPLATE, f"missing: {heading}"

    def test_prompt_enforces_no_mixed_window_claims(self) -> None:
        """Every cited number must be paired with its window label."""
        assert "Never mix windows" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        # Guardrail wording for the `insufficient data` / `no prev window`
        # cells the comparison tables render when n is too small.
        assert "insufficient data" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "no prev window" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_deployment_status_points_at_platform_scoped_section(self) -> None:
        """Deployment status bullet anchors to the per-platform section heading.

        Phase 3 partitioned Tool Deployment Status in analyze.py, so the
        prompt no longer needs a lowercase-match filter — it simply names
        the "Tool Deployment Status ({platform_label})" heading and tells
        the LLM to summarize every deployment listed there.
        """
        assert (
            '"Tool Deployment Status ({platform_label})"'
            in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        )
        assert "count of active tools only" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "do NOT enumerate the tool names" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "`⚠️ unavailable`" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_tool_category_prompt_lists_every_qualifying_cell(self) -> None:
        """Tool × Category bullet instructs the LLM to list every qualifying cell.

        Per-platform tables are small enough that exhaustively listing
        cells that clear the sample-size threshold is preferable to
        picking an editorial subset.
        """
        assert (
            "list every cell that clears the sample-size threshold"
            in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        )
        assert "insufficient tool × category data" in SUMMARY_SYSTEM_PROMPT_TEMPLATE


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

    def test_template_contains_placeholder(self) -> None:
        """Template must include a ``{platform_label}`` placeholder."""
        # Without the placeholder, _build_system_prompt becomes a no-op and
        # the LLM loses deployment scoping silently.
        assert "{platform_label}" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_build_raises_on_empty_label(self) -> None:
        """Empty label is rejected — would render "for the ** deployment"."""
        with pytest.raises(ValueError, match="platform_label"):
            _build_system_prompt("")

    def test_build_raises_on_unknown_label(self) -> None:
        """A label outside PLATFORM_LABELS is rejected before reaching the LLM.

        Guards against a workflow-level typo like ``--platform-label Omenstrap``
        silently producing a deployment-mislabeled summary.
        """
        with pytest.raises(ValueError, match="must be one of"):
            _build_system_prompt("Omenstrap")

    def test_labels_tracked_from_analyze(self) -> None:
        """Every ``benchmark.analyze.PLATFORM_LABELS`` value is accepted.

        Reusing the same import surface means a rename in analyze.py
        (e.g. Omenstrat -> Omen Strat) doesn't drift the two modules out
        of sync.
        """
        for label in PLATFORM_LABELS.values():
            _build_system_prompt(label)


class TestTopToolsEligibilityFilter:
    """Top tools must apply the same low-sample / floor filter as Worst tools."""

    def test_min_sample_size_floor_named_in_eligibility_block(self) -> None:
        """Eligibility block cites the live ``MIN_SAMPLE_SIZE`` constant."""
        # pylint: disable=import-outside-toplevel
        from benchmark.scorer import MIN_SAMPLE_SIZE

        assert "Eligibility for Top tools" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert f"at least {MIN_SAMPLE_SIZE}" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_top_tools_lists_only_eligible_rows(self) -> None:
        """Top-tools instruction restricts ranking to the eligible set."""
        assert "top 3 eligible rows" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_worst_tools_lists_only_eligible_rows(self) -> None:
        """Worst-tools instruction restricts ranking to the eligible set."""
        assert "bottom 3 eligible rows" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_eligibility_excludes_low_sample_and_malformed(self) -> None:
        """Eligibility block names both ⚠ flags so neither leaks into rankings."""
        assert "⚠ low sample" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "⚠ all malformed" in SUMMARY_SYSTEM_PROMPT_TEMPLATE


class TestSingleToolFallback:
    """When only one tool is eligible, collapse to a single Tool performance bullet."""

    def test_single_tool_fallback_section_named(self) -> None:
        """Prompt names the ``*Tool performance:*`` fallback heading."""
        assert "*Tool performance:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_fallback_clause_mentions_skip_top_and_worst(self) -> None:
        """Single-tool path explicitly skips both Top and Worst sections."""
        assert "SINGLE-TOOL FALLBACK" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "skip both" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_zero_eligible_renders_explicit_placeholder(self) -> None:
        """Zero-eligible case has a deterministic placeholder so the LLM doesn't guess."""
        assert "no eligible tools" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
