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
