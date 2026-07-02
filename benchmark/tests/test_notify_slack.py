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

import io
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from benchmark.analyze import PLATFORM_LABELS, ROLLING_WINDOW_DAYS
from benchmark.notify_slack import (
    _build_system_prompt,
    _count_eligible_tools,
    _infer_platform_label,
    _tables_to_monospace,
    post_to_slack,
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
        """The edge-led digest carries its five decision-first blocks."""
        prompt = _default_prompt()
        for heading in (
            "*Headline:*",
            "*Beats market? — edge over market",
            "*Promote watch — tournament candidates:*",
            "*Regressions:*",
            "*Action:*",
        ):
            assert heading in prompt, f"missing: {heading}"

    def test_headline_leads_with_edge_not_brier(self) -> None:
        """Headline pivots to beats-market (edge), keeping BSS distinct.

        The money-relevant verdict is edge over market; BSS (skill vs the
        base-rate predictor) must be present but explicitly NOT framed as
        a market comparison, so the two are never conflated.
        """
        prompt = _default_prompt()
        assert "beat the market" in prompt
        assert "Edge over market" in prompt
        # BSS is kept, but flagged as NOT the market baseline.
        assert "NOT the market" in prompt

    def test_tournament_callouts_coverage_preserved(self) -> None:
        """Promote-watch keeps the mandatory one-row-per-candidate rule."""
        prompt = _default_prompt()
        assert "## Tournament Callouts" in prompt
        assert "one table row per Callouts data row" in prompt
        assert "EXACTLY N table rows" in prompt

    def test_tabular_sections_request_markdown_tables(self) -> None:
        """Beats-market / Promote-watch / Regressions are emitted as tables.

        The LLM emits clean markdown pipe tables (its strength); the
        ``_tables_to_monospace`` pass aligns them for Slack. Pin that the
        prompt asks for markdown tables (not hand-aligned ASCII, which the
        LLM gets wrong on long tool names).
        """
        prompt = _default_prompt()
        assert "markdown table" in prompt
        assert "post-processor reformats those tables" in prompt
        # The headline + action stay mrkdwn prose, not tables.
        assert "*Headline:*" in prompt
        assert "*Action:*" in prompt

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
        """Prompt points the LLM at the report sections it actually reads.

        The edge-led digest sources the headline from Platform Snapshot /
        Platform Historical Comparison, the per-tool edge from Diagnostics
        Historical Comparison, eligibility from Tool Historical Comparison,
        and the worst category from Tool × Category. It no longer dumps
        Tool × Category Historical Comparison or Reliability & Parse
        Quality, so those headings are intentionally absent.
        """
        prompt = _default_prompt()
        for heading in (
            "Platform Snapshot",
            "Platform Historical Comparison",
            "Tool Historical Comparison",
            f"Tool × Category (Current {ROLLING_WINDOW_DAYS}d)",
            "Diagnostics Historical Comparison",
        ):
            assert heading in prompt, f"missing: {heading}"

    def test_prompt_enforces_no_mixed_window_claims(self) -> None:
        """Every cited number must be paired with its window label."""
        prompt = _default_prompt()
        assert "Never mix windows" in prompt
        assert "insufficient data" in prompt
        assert "no prev window" in prompt

    def test_regressions_section_sources_wow_deltas(self) -> None:
        """Regressions block draws from the Prev-window delta columns."""
        prompt = _default_prompt()
        assert "*Regressions:*" in prompt
        assert f"Δ vs Prev {ROLLING_WINDOW_DAYS}d" in prompt
        assert "Skip this section entirely when nothing regressed" in prompt


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
        assert "Eligibility for the per-tool edge table" in prompt
        assert f"at least {MIN_SAMPLE_SIZE}" in prompt

    def test_eligibility_excludes_low_sample_and_malformed(self) -> None:
        """Eligibility block names both ⚠ flags so neither leaks into rankings."""
        prompt = _default_prompt()
        assert "⚠ low sample" in prompt
        assert "⚠ all malformed" in prompt


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
    """The beats-market block exposes ONE convention per request.

    The dispatch is deterministic: with 0 eligible tools the LLM sees a
    single placeholder bullet and no list instruction; with 1+ it sees
    the edge-list instruction and never the placeholder. The prohibited
    branch is simply not in the prompt — there is no "skip" rule to obey.
    """

    def test_zero_eligible_renders_explicit_placeholder(self) -> None:
        """Zero eligible -> placeholder bullet, no edge-list instruction."""
        prompt = _build_system_prompt("Omenstrat", eligible_count=0)
        assert "no eligible tools" in prompt
        assert "For EVERY eligible tool" not in prompt

    @pytest.mark.parametrize("eligible", [1, 2, 3, 5, 10])
    def test_eligible_lists_every_tool_edge(self, eligible: int) -> None:
        """1+ eligible -> edge-list instruction, no placeholder."""
        prompt = _build_system_prompt("Omenstrat", eligible_count=eligible)
        assert "For EVERY eligible tool" in prompt
        assert "edge descending (best first)" in prompt
        assert "no eligible tools" not in prompt

    def test_no_legacy_top_worst_split_remains(self) -> None:
        """The old Brier Top/Worst convention is gone at every N."""
        for n in (0, 1, 3, 5, 10):
            prompt = _build_system_prompt("Omenstrat", eligible_count=n)
            assert "*Top tools:*" not in prompt
            assert "*Worst tools:*" not in prompt
            assert "*Tool performance:*" not in prompt


class TestTablesToMonospace:
    """``_tables_to_monospace`` aligns LLM markdown tables for Slack."""

    def test_converts_markdown_table_to_aligned_fence(self) -> None:
        """A pipe table becomes a backtick-fenced, space-aligned block."""
        text = (
            "*Beats market?*\n"
            "| tool | edge | verdict |\n"
            "|---|---|---|\n"
            "| superforcaster | -0.0128 | 🔴 loses |\n"
            "| factual_research | -0.0690 | 🔴 loses |"
        )
        out = _tables_to_monospace(text)
        lines = out.split("\n")
        # Bold header passes through untouched.
        assert lines[0] == "*Beats market?*"
        # Fenced.
        assert lines[1] == "```"
        assert lines[-1] == "```"
        # Separator row dropped.
        assert "---" not in out
        # No pipes remain (Slack would show them raw).
        assert "|" not in out
        # Columns aligned: the header cell `tool` is padded to the width
        # of the longest value (`factual_research`, 16 chars).
        assert "tool              edge" in out

    def test_non_table_text_passes_through(self) -> None:
        """Prose and bullets without pipe tables are returned unchanged."""
        text = "*Headline:* 🔴 loses to market.\n\n*Action:*\n- do a thing"
        assert _tables_to_monospace(text) == text

    def test_handles_multiple_tables(self) -> None:
        """Each markdown table in the text is fenced independently."""
        text = (
            "*A:*\n| x | y |\n|---|---|\n| 1 | 2 |\n\n"
            "*B:*\n| p | q |\n|---|---|\n| 3 | 4 |"
        )
        out = _tables_to_monospace(text)
        assert out.count("```") == 4  # two fenced blocks
        assert "|" not in out

    def test_ragged_rows_do_not_crash(self) -> None:
        """A row with fewer cells than the header is padded, not dropped."""
        text = "| a | b | c |\n|---|---|---|\n| 1 | 2 |"
        out = _tables_to_monospace(text)
        assert "```" in out
        assert "1" in out and "2" in out


class TestPostToSlack:
    """``post_to_slack`` must surface Slack's rejection reason, not swallow it."""

    _WEBHOOK = "https://hooks.slack.com/services/T000/B000/XXXX"

    def test_success_reads_response_without_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 200 response is consumed and no error is raised."""
        captured: dict[str, Any] = {}
        resp = MagicMock()
        resp.__enter__.return_value = resp  # `with urlopen(...) as r` yields resp
        resp.__exit__.return_value = False  # don't suppress exceptions
        resp.read.return_value = b"ok"

        def _fake_urlopen(req: Request, timeout: float) -> MagicMock:
            captured["data"] = req.data
            return resp

        monkeypatch.setattr("benchmark.notify_slack.urlopen", _fake_urlopen)
        post_to_slack(self._WEBHOOK, "hello")
        resp.read.assert_called_once()
        assert b"hello" in captured["data"]

    def test_http_error_surfaces_slack_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 400 must raise RuntimeError carrying Slack's plaintext reason."""

        def _fake_urlopen(req: Request, timeout: float) -> None:
            raise HTTPError(
                self._WEBHOOK,
                400,
                "Bad Request",
                Message(),
                io.BytesIO(b"invalid_payload"),
            )

        monkeypatch.setattr("benchmark.notify_slack.urlopen", _fake_urlopen)
        with pytest.raises(RuntimeError, match="invalid_payload"):
            post_to_slack(self._WEBHOOK, "hello")
