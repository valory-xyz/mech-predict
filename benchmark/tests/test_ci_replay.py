# -*- coding: utf-8 -*-
"""Tests for parse-reliability metrics and rendering in ci_replay."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.ci_replay import (
    PARSE_STATUS_BUCKETS,
    _compute_parse_reliability,
    _format_reliability_block,
    _load_filter_stats,
    compute_metrics,
    format_report,
)


def _row(status: str = "valid", p_yes: float | None = 0.6) -> dict:
    return {
        "platform": "omen",
        "tool_name": "superforcaster",
        "p_yes": p_yes,
        "p_no": None if p_yes is None else 1 - p_yes,
        "prediction_parse_status": status,
        "final_outcome": True,
    }


class TestParseReliability:
    """Bucket counts and rates from ``prediction_parse_status`` values."""

    def test_breakdown_keys_are_always_present(self) -> None:
        """All four buckets are present even when only 'valid' has entries."""
        rel = _compute_parse_reliability([_row("valid")])
        assert set(rel["breakdown"]) == set(PARSE_STATUS_BUCKETS)

    def test_mixed_statuses_counted(self) -> None:
        """Mixed statuses (valid, malformed, missing_fields) land in the right buckets."""
        rows = [
            _row("valid"),
            _row("valid"),
            _row("malformed", None),
            _row("missing_fields", None),
        ]
        rel = _compute_parse_reliability(rows)
        assert rel == {
            "total": 4,
            "valid": 2,
            "parse_rate": 0.5,
            "breakdown": {
                "valid": 2,
                "missing_fields": 1,
                "malformed": 1,
                "error": 0,
            },
        }

    def test_unknown_status_bucketed_as_error(self) -> None:
        """An unexpected status value must not lose the row."""
        rel = _compute_parse_reliability([_row("weird_new_status", None)])
        assert rel["breakdown"]["error"] == 1
        assert rel["total"] == 1

    def test_compute_metrics_embeds_parse_reliability(self) -> None:
        """compute_metrics must surface parse_reliability alongside Brier."""
        metrics = compute_metrics([_row("valid"), _row("malformed", None)])
        assert "parse_reliability" in metrics
        assert metrics["parse_reliability"]["parse_rate"] == 0.5


def _stats(
    *,
    accepted: int = 100,
    not_valid_parse: int = 0,
    wrong_tool: int = 0,
    no_deliver_id: int = 0,
    no_outcome: int = 0,
    older_than_cutoff: int = 0,
) -> dict:
    return {
        "accepted": accepted,
        "rejected": {
            "wrong_tool": wrong_tool,
            "no_deliver_id": no_deliver_id,
            "not_valid_parse": not_valid_parse,
            "no_outcome": no_outcome,
            "older_than_cutoff": older_than_cutoff,
        },
    }


class TestReliabilityBlock:
    """Markdown rendering of the Reliability section (candidate + pre-filter)."""

    def _metrics(self, rows: list[dict]) -> dict:
        return compute_metrics(rows)

    def test_green_happy_path_is_two_lines(self) -> None:
        """All-valid candidate + clean pre-filter → two bullets, no breakdown noise."""
        candidate = self._metrics([_row("valid")] * 5)
        lines = _format_reliability_block(candidate, [], _stats(accepted=5))
        text = "\n".join(lines)
        assert "Candidate parse rate: 5/5 (100.0%) ✅" in text
        assert (
            "Pre-filter (enrich): 5 accepted, 0 rejected, not_valid_parse=0 ✅" in text
        )
        # Breakdown is hidden when candidate is at 100%.
        assert "Breakdown:" not in text
        # Scoping is hidden when there are no rejections.
        assert "Scoping:" not in text
        assert "⚠️" not in text

    def test_candidate_drift_flags_and_shows_breakdown(self) -> None:
        """Candidate < 100% → ⚠️ + breakdown line exposes which buckets failed."""
        candidate = self._metrics([_row("valid")] * 7 + [_row("malformed", None)] * 3)
        lines = _format_reliability_block(candidate, [], _stats(accepted=10))
        text = "\n".join(lines)
        assert "Candidate parse rate: 7/10 (70.0%) ⚠️" in text
        assert "malformed=3" in text

    def test_prefilter_not_valid_parse_flagged(self) -> None:
        """not_valid_parse > 0 is the load-bearing invariant — must surface ⚠️."""
        candidate = self._metrics([_row("valid")] * 5)
        lines = _format_reliability_block(
            candidate, [], _stats(accepted=5, not_valid_parse=2)
        )
        text = "\n".join(lines)
        assert "not_valid_parse=2 ⚠️" in text

    def test_scoping_rejections_shown_but_not_flagged(self) -> None:
        """Scoping buckets (wrong_tool etc.) are informational, not warnings."""
        candidate = self._metrics([_row("valid")] * 5)
        lines = _format_reliability_block(
            candidate,
            [],
            _stats(accepted=5, wrong_tool=12, no_outcome=4, older_than_cutoff=7),
        )
        text = "\n".join(lines)
        assert "Scoping:" in text
        assert "wrong_tool=12" in text
        assert "no_outcome=4" in text
        assert "older_than_cutoff=7" in text
        # Scoping alone must not raise the invariant marker.
        assert "⚠️" not in text

    def test_baseline_prefilter_parse_rate_rendered_with_rejections(self) -> None:
        """Rejections > 0: render ``accepted/(accepted+not_valid_parse)`` as a percentage.

        Direct response to PR #231 review — post-filter baseline=100% is a
        tautology because enrich drops non-valid rows. The ratio before the
        filter is what tells reviewers how noisy production actually was.
        """
        candidate = self._metrics([_row("valid")] * 100)
        lines = _format_reliability_block(
            candidate, [], _stats(accepted=100, not_valid_parse=35)
        )
        text = "\n".join(lines)
        assert "Baseline pre-filter parse rate: 100/135 (74.1%)" in text

    def test_baseline_prefilter_parse_rate_at_100_when_no_parse_rejections(
        self,
    ) -> None:
        """Zero not_valid_parse rejections: rate is 100%, but still rendered.

        Keeping it rendered on the happy path surfaces the *fact* that this
        is now a reported metric, so a later regression (non-zero
        ``not_valid_parse``) isn't a surprise line appearing out of nowhere.
        """
        candidate = self._metrics([_row("valid")] * 50)
        lines = _format_reliability_block(
            candidate,
            [],
            _stats(accepted=50, wrong_tool=3, no_outcome=1),
        )
        text = "\n".join(lines)
        assert "Baseline pre-filter parse rate: 50/50 (100.0%)" in text

    def test_baseline_prefilter_parse_rate_omitted_when_no_sidecar(self) -> None:
        """Older pipelines (no filter_stats) don't render the baseline rate line."""
        candidate = self._metrics([_row("valid")] * 5)
        text = "\n".join(_format_reliability_block(candidate, [], None))
        assert "Baseline pre-filter parse rate" not in text

    def test_scoping_stays_nested_under_prefilter_not_parse_rate(self) -> None:
        """Scoping (breakdown of rejections) must remain adjacent to Pre-filter.

        Scoping renders as an indented sub-bullet; markdown nests it under
        the preceding top-level bullet. If Baseline pre-filter parse rate is
        inserted between Pre-filter and Scoping, the sub-bullet visually
        becomes a child of the wrong parent.
        """
        candidate = self._metrics([_row("valid")] * 50)
        lines = _format_reliability_block(
            candidate,
            [],
            _stats(accepted=50, wrong_tool=3, no_outcome=1),
        )
        text = "\n".join(lines)
        assert text.index("Scoping:") < text.index("Baseline pre-filter parse rate:")

    def test_prefilter_omitted_when_stats_none(self) -> None:
        """Older pipelines (no sidecar) render without the Pre-filter line."""
        candidate = self._metrics([_row("valid")] * 3)
        text = "\n".join(_format_reliability_block(candidate, [], None))
        assert "Pre-filter" not in text
        assert "Candidate parse rate" in text

    def test_failure_bodies_rendered_in_collapsed_details(self) -> None:
        """Failure bodies inline under a <details> so the PR comment stays tidy."""
        candidate = self._metrics([_row("valid"), _row("malformed", None)])
        failures = [
            {
                "row_id": "c2",
                "question_text": "Q2",
                "prediction_parse_status": "malformed",
                "raw_response": "<facts> leaked content here",
            }
        ]
        text = "\n".join(_format_reliability_block(candidate, failures, None))
        assert "<details>" in text
        assert "<facts> leaked content here" in text
        assert "malformed" in text

    def test_body_with_backticks_is_escaped(self) -> None:
        """Markdown code fences in the body can't break out of the outer ``` block."""
        candidate = self._metrics([_row("malformed", None)])
        failures = [
            {
                "row_id": "c1",
                "question_text": "Q",
                "prediction_parse_status": "malformed",
                "raw_response": 'some ```json{"p_yes":0.5}``` leak',
            }
        ]
        text = "\n".join(_format_reliability_block(candidate, failures, None))
        assert text.count("```") == 2


class TestLoadFilterStats:
    """Sidecar loading alongside candidate.jsonl."""

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        """Older pipelines without the sidecar return cleanly."""
        candidate = tmp_path / "candidate.jsonl"
        candidate.write_text("", encoding="utf-8")
        assert _load_filter_stats(candidate) is None

    def test_parses_sidecar_when_present(self, tmp_path: Path) -> None:
        """filter_stats.json in the same dir as candidate.jsonl is loaded."""
        candidate = tmp_path / "candidate.jsonl"
        candidate.write_text("", encoding="utf-8")
        stats = {"accepted": 5, "rejected": {"not_valid_parse": 1}}
        (tmp_path / "filter_stats.json").write_text(json.dumps(stats), encoding="utf-8")
        assert _load_filter_stats(candidate) == stats

    def test_returns_none_on_malformed_json(self, tmp_path: Path) -> None:
        """A corrupt sidecar must not crash the whole report."""
        candidate = tmp_path / "candidate.jsonl"
        candidate.write_text("", encoding="utf-8")
        (tmp_path / "filter_stats.json").write_text("{not json", encoding="utf-8")
        assert _load_filter_stats(candidate) is None


class TestFormatReportEndToEnd:
    """End-to-end ``format_report`` shape: table first, Reliability below."""

    def test_metrics_table_precedes_reliability_section(self) -> None:
        """Primary comparison (Brier etc.) must come before reliability checks."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(baseline, candidate, {"tool": "superforcaster"})
        assert "**Reliability**" in report
        assert "| Metric |" in report
        assert report.index("| Metric |") < report.index("**Reliability**")

    def test_report_renders_without_failures_or_filter_stats(self) -> None:
        """format_report works on its minimal arg set (older pipelines)."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(baseline, candidate, {"tool": "superforcaster"})
        assert "Candidate parse rate: 3/3 (100.0%) ✅" in report
        assert "Pre-filter" not in report
        assert "<details><summary>Candidate parse failures" not in report

    def test_report_embeds_failure_bodies_when_provided(self) -> None:
        """format_report threads the failure_rows arg into the reliability section."""
        baseline = compute_metrics([_row("valid")] * 2)
        candidate = compute_metrics([_row("valid"), _row("error", None)])
        failures = [
            {
                "row_id": "c2",
                "question_text": "Q2",
                "prediction_parse_status": "error",
                "raw_response": "Boom.",
            }
        ]
        report = format_report(
            baseline,
            candidate,
            {"tool": "superforcaster"},
            failure_rows=failures,
        )
        assert "Boom." in report
        assert "⚠️" in report

    def test_report_renders_prefilter_when_stats_provided(self) -> None:
        """A filter_stats dict adds the Pre-filter line to the reliability block."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(
            baseline,
            candidate,
            {"tool": "superforcaster"},
            filter_stats=_stats(accepted=3, wrong_tool=10, no_outcome=2),
        )
        assert "Pre-filter (enrich): 3 accepted, 12 rejected" in report
        assert "⚠️" not in report  # scoping only, invariant held

    def test_filter_regression_surfaces_warning_in_full_report(self) -> None:
        """not_valid_parse > 0 propagates the ⚠️ marker into the final report."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(
            baseline,
            candidate,
            {"tool": "superforcaster"},
            filter_stats=_stats(accepted=3, not_valid_parse=2),
        )
        assert "⚠️" in report
        assert "not_valid_parse=2 ⚠️" in report


class TestFormatReportFooter:
    """Footer must make multi-seed runs distinguishable and auditable.

    Reviewers will trigger ``/benchmark`` repeatedly with different seeds;
    without seed + trigger-comment attribution in the footer, the resulting
    PR comments are visually indistinguishable and the parameters that
    produced each one are invisible. Per PR #231 review (#233).
    """

    def _report(self, **meta: str) -> str:
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        full_meta: dict[str, str] = {"tool": "superforcaster", **meta}
        return format_report(baseline, candidate, full_meta)

    def test_footer_includes_seed_when_provided(self) -> None:
        """`meta["seed"]` must appear as `seed <N>` in the footer."""
        report = self._report(seed="1337")
        assert "seed 1337" in report.splitlines()[-1]

    def test_footer_omits_seed_when_absent(self) -> None:
        """Without a seed, no seed label — otherwise older callers emit 'seed None'."""
        report = self._report()
        assert "seed" not in report.splitlines()[-1]

    def test_footer_links_triggered_by_when_comment_url_provided(self) -> None:
        """trigger_comment_url turns the `@user` mention into a markdown link.

        Reviewers need to jump from a benchmark comment to the exact
        ``/benchmark`` request that produced it (params, author, time).
        """
        report = self._report(
            triggered_by="LOCKhart07",
            trigger_comment_url="https://github.com/valory-xyz/mech-predict/pull/231#issuecomment-4263150118",
        )
        expected = (
            "triggered by [@LOCKhart07]"
            "(https://github.com/valory-xyz/mech-predict/pull/231#issuecomment-4263150118)"
        )
        assert expected in report.splitlines()[-1]

    def test_footer_plain_triggered_by_without_comment_url(self) -> None:
        """No URL → plain `@user` mention (backwards compatible)."""
        report = self._report(triggered_by="LOCKhart07")
        footer = report.splitlines()[-1]
        assert "@LOCKhart07" in footer
        assert "[@LOCKhart07]" not in footer

    def test_footer_order_markets_seed_triggered_by(self) -> None:
        """Footer parts stay in the existing order: markets → seed → triggered-by."""
        report = self._report(
            seed="1337",
            triggered_by="LOCKhart07",
            trigger_comment_url="https://example.com/c",
        )
        footer = report.splitlines()[-1]
        assert footer.index("markets") < footer.index("seed 1337")
        assert footer.index("seed 1337") < footer.index("LOCKhart07")
