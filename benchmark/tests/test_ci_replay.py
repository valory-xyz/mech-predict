# -*- coding: utf-8 -*-
"""Tests for parse-reliability metrics and rendering in ci_replay."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.ci_replay import (
    PARSE_STATUS_BUCKETS,
    _compute_parse_reliability,
    _format_reliability_section,
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
    def test_breakdown_keys_are_always_present(self) -> None:
        """All four buckets are present even when only 'valid' has entries."""
        rel = _compute_parse_reliability([_row("valid")])
        assert set(rel["breakdown"]) == set(PARSE_STATUS_BUCKETS)

    def test_mixed_statuses_counted(self) -> None:
        """valid + malformed + missing_fields all land in the right buckets."""
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
                "raw_response": "some ```json{\"p_yes\":0.5}``` leak",
            }
        ]
        lines = _format_reliability_section(baseline, candidate, failures)
        text = "\n".join(lines)
        # Exactly two literal ``` fences from the template (open + close of the body block).
        assert text.count("```") == 2


class TestFormatReportLoadsFailuresFromDisk:
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
