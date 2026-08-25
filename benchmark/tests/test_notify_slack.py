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
import logging
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from benchmark.analyze import PLATFORM_LABELS, ROLLING_WINDOW_DAYS
from benchmark.notify_slack import (
    _LEVEL_PREFIX_FORMAT,
    _build_system_prompt,
    _compute_top_k,
    _computed_tables_enabled,
    _configure_logging,
    _count_eligible_tools,
    _deployed_tools_for,
    _infer_platform_label,
    _main_window_label,
    _v1_heading,
    post_to_slack,
)
from benchmark.scoring_primitives import MIN_SAMPLE_SIZE


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

    def test_tool_category_prompt_caps_at_top_3_by_sample_size(self) -> None:
        """Tool × Category bullet caps at top 3 cells, selected by Current-window n."""
        prompt = _default_prompt()
        assert "list the top 3 cells by sample size" in prompt
        assert "n-descending order" in prompt
        assert "re-rank by n, not take the first 3 you see" in prompt
        assert "insufficient tool × category data" in prompt


class TestLogFormat:
    """The flywheel log format keeps WARNING distinguishable from INFO."""

    def test_configure_logging_passes_the_prefix_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The CALL SITE is pinned, not just the constant's value.

        Reverting `format=_LEVEL_PREFIX_FORMAT` at the call site while leaving
        the constant intact left the whole suite green, so the line that
        actually fixes the flywheel log was unprotected -- a refactor or merge
        dropping the argument would silently restore the unfilterable log.

        :param monkeypatch: pytest monkeypatch fixture.
        """
        captured: dict[str, Any] = {}

        def fake_basic_config(**kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
        _configure_logging()
        assert captured["format"] == _LEVEL_PREFIX_FORMAT
        assert captured["level"] == logging.INFO

    def test_level_is_recoverable_from_a_rendered_line(self) -> None:
        """A WARNING and an INFO with the same text render differently.

        The ROI freshness and malformed-payload guards report through
        log.warning. Under the previous bare "%(message)s" format they were
        byte-identical to the surrounding INFO chatter, so the log half of the
        guard was write-only: nothing could grep or filter for it.
        """
        formatter = logging.Formatter(_LEVEL_PREFIX_FORMAT)

        def render(level: int) -> str:
            record = logging.LogRecord(
                name="benchmark.roi_slack",
                level=level,
                pathname=__file__,
                lineno=1,
                msg="ROI results are 6.0 days old",
                args=(),
                exc_info=None,
            )
            return formatter.format(record)

        warning_line = render(logging.WARNING)
        info_line = render(logging.INFO)
        assert warning_line != info_line
        assert "WARNING" in warning_line
        # Greppable by level at line start -- the property an operator uses.
        assert warning_line.startswith("WARNING")


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


class TestMainWindowLabel:
    """Prompt label follows the flag so MTD data isn't described as All-Time."""

    def test_flag_off_uses_all_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legacy path keeps the All-Time label."""
        monkeypatch.delenv("USE_MECH_ANALYTICS_ROWS", raising=False)
        assert _main_window_label() == "All-Time"

    def test_flag_on_uses_mtd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Self-contained mode labels the main window as MTD."""
        monkeypatch.setenv("USE_MECH_ANALYTICS_ROWS", "true")
        assert _main_window_label() == "MTD"

    def test_prompt_renders_provided_label(self) -> None:
        """Whatever label is passed shows up in the prompt string."""
        prompt = _build_system_prompt(
            "Omenstrat", eligible_count=5, main_window_label="MTD"
        )
        assert "`MTD` (cumulative)" in prompt
        assert "Δ vs MTD" in prompt
        assert "All-Time" not in prompt

    def test_prompt_defaults_to_all_time(self) -> None:
        """The default keeps behavior identical for flag-off callers."""
        prompt = _build_system_prompt("Omenstrat", eligible_count=5)
        assert "All-Time" in prompt


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
        """Rendered prompt has no surviving placeholders after dispatch."""
        for n in (0, 1, 3, 5, 10):
            prompt = _build_system_prompt("Omenstrat", eligible_count=n)
            assert "{platform_label}" not in prompt
            # main_window_label was added when the flag-on path landed.
            # Regression guard exists precisely to catch a future refactor
            # that misses one of the two placeholders.
            assert "{main_window_label}" not in prompt


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


class TestComputedTablesFlag:
    """The computed tables are opt-in while the redesign is validated."""

    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset variable keeps the existing digest untouched."""
        monkeypatch.delenv("BENCHMARK_COMPUTED_TABLES", raising=False)
        assert _computed_tables_enabled() is False

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", " true "])
    def test_truthy_values_enable(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Common truthy spellings all turn the tables on.

        A strict == "true" check would silently no-op on the also-idiomatic "1".

        :param monkeypatch: pytest fixture.
        :param value: environment value under test.
        """
        monkeypatch.setenv("BENCHMARK_COMPUTED_TABLES", value)
        assert _computed_tables_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "off", ""])
    def test_falsy_values_stay_off(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Anything else leaves the digest exactly as it is today.

        :param monkeypatch: pytest fixture.
        :param value: environment value under test.
        """
        monkeypatch.setenv("BENCHMARK_COMPUTED_TABLES", value)
        assert _computed_tables_enabled() is False


class TestV1Heading:
    """The prose digest titles itself in the shared rule-framed style."""

    def test_badge_rule_and_date_from_the_report(self) -> None:
        """V1 badge, rules sized to the line, date read from the heading."""
        heading = _v1_heading(
            "Omenstrat", "# Benchmark Report (Omenstrat) \u2014 2026-08-17\n..."
        )
        rule, line, closing = heading.split("\n")
        assert rule == closing and set(rule) == {"\u2501"}
        assert line == "*OMENSTRAT  \u00b7  REPORT V1  \u00b7  2026-08-17*"
        assert len(rule) >= len(line) - 2

    def test_dateless_report_still_titles(self) -> None:
        """No date in the heading -> badge without a stamp, never a crash."""
        heading = _v1_heading("Polystrat", "no heading here")
        assert "*POLYSTRAT  \u00b7  REPORT V1*" in heading


class TestDeployedToolsForTriState:
    """[] means nothing deployed; None means the lookup failed."""

    def test_successful_empty_lookup_is_empty_not_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All deployments answered with zero tools -> [], never None.

        Collapsing them let a legitimately empty roster fall back to
        rendering every historically-scored tool.

        :param monkeypatch: pytest monkeypatch.
        """
        import benchmark.tool_usage as tool_usage

        monkeypatch.setattr(
            tool_usage, "fetch_valid_tools", lambda: {"polystrat Pearl": []}
        )
        assert _deployed_tools_for("polymarket", "") == []

    def test_failed_lookup_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deployment answering None poisons the platform to None.

        :param monkeypatch: pytest monkeypatch.
        """
        import benchmark.tool_usage as tool_usage

        monkeypatch.setattr(
            tool_usage, "fetch_valid_tools", lambda: {"polystrat Pearl": None}
        )
        assert _deployed_tools_for("polymarket", "") is None
