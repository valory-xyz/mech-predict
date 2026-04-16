# -*- coding: utf-8 -*-
"""Tests for prompt_replay filter + sidecar plumbing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.prompt_replay import _load_and_filter_rows, _log_replay_summary


def _row(
    *,
    tool: str = "superforcaster",
    deliver_id: str = "0xabc",
    status: str = "valid",
    outcome: Any = True,
    predicted_at: str | None = None,
) -> dict:
    r: dict = {
        "tool_name": tool,
        "deliver_id": deliver_id,
        "prediction_parse_status": status,
        "final_outcome": outcome,
    }
    if predicted_at is not None:
        r["predicted_at"] = predicted_at
    return r


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


class TestLoadAndFilterRows:
    """Rejection counters on `_load_and_filter_rows`."""

    def test_all_accepted_zero_rejections(self, tmp_path: Path) -> None:
        """Rows matching every filter land in rows; counters stay at zero."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(), _row()])
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert len(rows) == 2
        assert sum(rejected.values()) == 0

    def test_wrong_tool_counted(self, tmp_path: Path) -> None:
        """Rows with a different tool_name hit the wrong_tool bucket only."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(tool="other"), _row(tool="another")])
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert rows == []
        assert rejected["wrong_tool"] == 2
        assert rejected["not_valid_parse"] == 0

    def test_not_valid_parse_counted(self, tmp_path: Path) -> None:
        """The load-bearing bucket — a leaked non-valid row lands here."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(
            path,
            [
                _row(status="malformed"),
                _row(status="missing_fields"),
                _row(status="error"),
                _row(status="valid"),
            ],
        )
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert len(rows) == 1
        assert rejected["not_valid_parse"] == 3

    def test_no_outcome_counted(self, tmp_path: Path) -> None:
        """Rows without final_outcome hit no_outcome, not not_valid_parse."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(outcome=None)])
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert rows == []
        assert rejected["no_outcome"] == 1
        assert rejected["not_valid_parse"] == 0

    def test_no_deliver_id_counted(self, tmp_path: Path) -> None:
        """Rows without deliver_id hit no_deliver_id."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(deliver_id="")])
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert rows == []
        assert rejected["no_deliver_id"] == 1

    def test_filter_order_gives_first_failing_reason(self, tmp_path: Path) -> None:
        """A row failing multiple predicates counts in the first one only.

        Order: wrong_tool → no_deliver_id → not_valid_parse → no_outcome → cutoff.
        A row with wrong tool AND bad parse AND no outcome increments wrong_tool
        only — so the total rejection count matches row count exactly.
        """
        path = tmp_path / "log.jsonl"
        _write_jsonl(
            path,
            [_row(tool="other", status="malformed", outcome=None)],
        )
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert rows == []
        assert rejected["wrong_tool"] == 1
        assert sum(rejected.values()) == 1


class TestLogReplaySummaryFilterStats:
    """`_log_replay_summary` renders the Pre-filter block when stats are given."""

    def _sampled(self) -> list[dict]:
        return [
            {"p_yes": 0.7, "p_no": 0.3, "final_outcome": True, "tool_name": "x"},
            {"p_yes": 0.4, "p_no": 0.6, "final_outcome": False, "tool_name": "x"},
        ]

    def _write_candidate(self, path: Path) -> None:
        rows = [
            {"p_yes": 0.8, "p_no": 0.2, "final_outcome": True},
            {"p_yes": 0.3, "p_no": 0.7, "final_outcome": False},
        ]
        _write_jsonl(path, rows)

    def test_block_rendered_when_filter_stats_given(
        self, tmp_path: Path, caplog: Any
    ) -> None:
        """The Pre-filter line appears in the log when filter_stats is not None."""
        candidate = tmp_path / "candidate.jsonl"
        self._write_candidate(candidate)
        stats = {
            "accepted": 2,
            "rejected": {
                "wrong_tool": 4,
                "no_deliver_id": 0,
                "not_valid_parse": 1,
                "no_outcome": 2,
                "older_than_cutoff": 0,
            },
        }
        with caplog.at_level("INFO"):
            _log_replay_summary(
                sampled=self._sampled(),
                candidate_path=candidate,
                baseline_brier_sum=0.1,
                candidate_brier_sum=0.2,
                total=2,
                n_scored=2,
                baseline_path=candidate,
                status_counts={
                    "valid": 2,
                    "missing_fields": 0,
                    "malformed": 0,
                    "error": 0,
                },
                filter_stats=stats,
            )
        text = caplog.text
        assert "Pre-filter" in text
        assert "not_valid_parse=1" in text

    def test_block_omitted_when_stats_none(self, tmp_path: Path, caplog: Any) -> None:
        """When no sidecar is provided the summary logs exactly as before."""
        candidate = tmp_path / "candidate.jsonl"
        self._write_candidate(candidate)
        with caplog.at_level("INFO"):
            _log_replay_summary(
                sampled=self._sampled(),
                candidate_path=candidate,
                baseline_brier_sum=0.1,
                candidate_brier_sum=0.2,
                total=2,
                n_scored=2,
                baseline_path=candidate,
                status_counts={
                    "valid": 2,
                    "missing_fields": 0,
                    "malformed": 0,
                    "error": 0,
                },
            )
        assert "Pre-filter" not in caplog.text
