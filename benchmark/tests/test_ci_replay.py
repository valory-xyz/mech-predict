# -*- coding: utf-8 -*-
"""Tests for parse-reliability metrics and rendering in ci_replay."""

from __future__ import annotations

from pathlib import Path

import json

from benchmark.ci_replay import (
    PARSE_STATUS_BUCKETS,
    _compute_parse_reliability,
    _format_prefilter_section,
    _format_reliability_section,
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


class TestReliabilitySection:
    """Markdown rendering of the reliability block + failure-body <details>."""

    def _metrics(self, rows: list[dict]) -> dict:
        return compute_metrics(rows)

    def test_section_flags_regression_when_candidate_drops(self) -> None:
        """A drop vs baseline carries the ⚠️ marker; zero-failure section omits <details>."""
        baseline = self._metrics([_row("valid")] * 10)
        candidate = self._metrics([_row("valid")] * 7 + [_row("malformed", None)] * 3)
        lines = _format_reliability_section(baseline, candidate, failure_rows=[])
        text = "\n".join(lines)
        assert "⚠️" in text
        assert "70.0%" in text
        assert "malformed=3" in text
        # No failure bodies passed in -> no <details> block
        assert "<details>" not in text

    def test_section_no_regression_gets_check_mark(self) -> None:
        """Equal parse rates render ✅ and no warning."""
        baseline = self._metrics([_row("valid")] * 5)
        candidate = self._metrics([_row("valid")] * 5)
        text = "\n".join(
            _format_reliability_section(baseline, candidate, failure_rows=[])
        )
        assert "✅" in text
        assert "⚠️" not in text

    def test_failure_bodies_rendered_in_collapsed_details(self) -> None:
        """Failure bodies inline under a <details> so the PR comment stays tidy."""
        baseline = self._metrics([_row("valid")] * 2)
        candidate = self._metrics([_row("valid"), _row("malformed", None)])
        failures = [
            {
                "row_id": "c2",
                "question_text": "Q2",
                "prediction_parse_status": "malformed",
                "raw_response": "<facts> leaked content here",
            }
        ]
        lines = _format_reliability_section(baseline, candidate, failures)
        text = "\n".join(lines)
        assert "<details>" in text
        assert "<facts> leaked content here" in text
        assert "malformed" in text

    def test_body_with_backticks_is_escaped(self) -> None:
        """Markdown code fences in the body can't break out of the outer ``` block."""
        baseline = self._metrics([_row("valid")])
        candidate = self._metrics([_row("malformed", None)])
        failures = [
            {
                "row_id": "c1",
                "question_text": "Q",
                "prediction_parse_status": "malformed",
                "raw_response": 'some ```json{"p_yes":0.5}``` leak',
            }
        ]
        lines = _format_reliability_section(baseline, candidate, failures)
        text = "\n".join(lines)
        # Exactly two literal ``` fences from the template (open + close of the body block).
        assert text.count("```") == 2


class TestFormatReportLoadsFailuresFromDisk:
    """End-to-end ``format_report`` wiring, with and without failure bodies."""

    def test_report_includes_reliability_even_with_no_failures(
        self, tmp_path: Path
    ) -> None:
        """format_report is callable without a failures file (None ≡ empty list)."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(baseline, candidate, {"tool": "superforcaster"})
        assert "Parse reliability" in report
        assert "<details>" not in report

    def test_report_embeds_failure_bodies_when_provided(self, tmp_path: Path) -> None:
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


class TestPrefilterSection:
    """Markdown rendering of the Pre-filter block (filter_stats sidecar)."""

    def _stats(
        self,
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

    def test_zero_not_valid_parse_renders_green(self) -> None:
        """With the filter invariant held, the block is informational only."""
        text = "\n".join(_format_prefilter_section(self._stats()))
        assert "✅ 0 rejected for not_valid_parse" in text
        assert "⚠️" not in text

    def test_nonzero_not_valid_parse_flagged(self) -> None:
        """A single leaked non-valid row is flagged so it can't hide."""
        text = "\n".join(_format_prefilter_section(self._stats(not_valid_parse=3)))
        assert "⚠️" in text
        assert "3 row(s) rejected for not_valid_parse" in text

    def test_scoping_rejections_listed_separately(self) -> None:
        """Wrong-tool / no-outcome / cutoff rejections are shown but not flagged."""
        text = "\n".join(
            _format_prefilter_section(
                self._stats(wrong_tool=12, no_outcome=4, older_than_cutoff=7)
            )
        )
        assert "wrong_tool=12" in text
        assert "no_outcome=4" in text
        assert "older_than_cutoff=7" in text
        # Scoping rejections must not escalate the invariant marker.
        assert "⚠️" not in text


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
        (tmp_path / "filter_stats.json").write_text(
            json.dumps(stats), encoding="utf-8"
        )
        assert _load_filter_stats(candidate) == stats

    def test_returns_none_on_malformed_json(self, tmp_path: Path) -> None:
        """A corrupt sidecar must not crash the whole report."""
        candidate = tmp_path / "candidate.jsonl"
        candidate.write_text("", encoding="utf-8")
        (tmp_path / "filter_stats.json").write_text("{not json", encoding="utf-8")
        assert _load_filter_stats(candidate) is None


class TestFormatReportWithFilterStats:
    """End-to-end: format_report wiring for the filter_stats arg."""

    def test_omits_prefilter_block_when_stats_missing(self) -> None:
        """Older datasets (no sidecar) render the original report unchanged."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(baseline, candidate, {"tool": "superforcaster"})
        assert "Pre-filter" not in report

    def test_renders_prefilter_block_when_stats_present(self) -> None:
        """A provided filter_stats dict drops a Pre-filter section above reliability."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        stats = {
            "accepted": 3,
            "rejected": {
                "wrong_tool": 10,
                "no_deliver_id": 0,
                "not_valid_parse": 0,
                "no_outcome": 2,
                "older_than_cutoff": 0,
            },
        }
        report = format_report(
            baseline,
            candidate,
            {"tool": "superforcaster"},
            filter_stats=stats,
        )
        assert "Pre-filter" in report
        # Ordering: Pre-filter block must precede Parse reliability block so a
        # filter regression is the first thing a reviewer sees.
        assert report.index("Pre-filter") < report.index("Parse reliability")

    def test_prefilter_regression_surfaces_in_full_report(self) -> None:
        """A not_valid_parse>0 count propagates the ⚠️ marker into the final report."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        stats = {
            "accepted": 3,
            "rejected": {
                "wrong_tool": 0,
                "no_deliver_id": 0,
                "not_valid_parse": 2,
                "no_outcome": 0,
                "older_than_cutoff": 0,
            },
        }
        report = format_report(
            baseline,
            candidate,
            {"tool": "superforcaster"},
            filter_stats=stats,
        )
        assert "⚠️" in report
        assert "2 row(s) rejected for not_valid_parse" in report
