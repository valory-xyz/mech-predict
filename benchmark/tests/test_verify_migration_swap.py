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
"""Divergence-gate tests for scripts/verify_migration_swap.py.

The gate is the actual go/no-go for the migration. An earlier edit
could invert one threshold comparison, swap a numerator/denominator,
or leave a stray early ``return 0`` and the script would print "PASS"
on materially divergent data. These tests pin exit codes 0/2/3.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


def _mp_row(request_id: str, **overrides: Any) -> dict[str, Any]:
    """Synthetic mech-predict row with the fields the gate reads."""
    row = {
        "request_id": request_id,
        "deliver_id": request_id,
        "platform": "omen",
        "market_id": f"market-{request_id}",
        "final_outcome": True,
        "p_yes": 0.7,
        "tool_name": "superforcaster",
        "question_text": "will X?",
        "requested_at": "2026-07-01T00:00:00Z",
        "predicted_at": "2026-07-01T00:00:05Z",
    }
    row.update(overrides)
    return row


def _lake_row(request_id: str, **overrides: Any) -> dict[str, Any]:
    """Synthetic lake row with the fields the gate reads."""
    row = {
        "request_id": request_id,
        "platform": "omen",
        "market_id": f"market-{request_id}",
        "final_outcome": True,
        "p_yes": 0.7,
        "tool_name": "superforcaster",
        "question_text": "will X?",
        "requested_at": "2026-07-01T00:00:00Z",
    }
    row.update(overrides)
    return row


def _patch_pulls(
    monkeypatch: pytest.MonkeyPatch,
    mp_rows: list[dict[str, Any]],
    lake_rows: list[dict[str, Any]],
    deliver_to_request: dict[str, dict[str, str]] | None = None,
) -> None:
    """Replace both pull entry points so main() runs with synthetic data."""
    import sys

    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import verify_migration_swap as vms

    dtr = deliver_to_request if deliver_to_request is not None else {
        "omen": {r["deliver_id"]: r["request_id"] for r in mp_rows},
    }

    def _mp(_since_ts: int, _until_ts: int) -> tuple[list[dict[str, Any]], dict]:
        return mp_rows, dtr

    def _lake(_since: datetime, _until: datetime) -> list[dict[str, Any]]:
        return lake_rows

    monkeypatch.setattr(vms, "_pull_mech_predict_rows", _mp)
    monkeypatch.setattr(vms, "_pull_lake_rows", _lake)


def _base_argv() -> list[str]:
    """Argv covering a 24h window so _compute_window doesn't reject."""
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    until = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    return ["--since", since, "--until", until, "--min-rows", "5"]


class TestDivergenceGate:
    """Exit code contract: 0 pass, 2 thin, 3 divergent."""

    def test_perfect_match_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """100% overlap + no outcome mismatches passes the gate."""
        import sys
        scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import verify_migration_swap as vms

        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        _patch_pulls(monkeypatch, rows, lake)
        assert vms.main(_base_argv()) == 0

    def test_thin_side_trips_min_row_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Either side under --min-rows returns 2 (vacuous window)."""
        import sys
        scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import verify_migration_swap as vms

        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row("r0"), _lake_row("r1")]  # only 2, threshold 5
        _patch_pulls(monkeypatch, rows, lake)
        assert vms.main(_base_argv()) == 2

    def test_low_overlap_returns_three(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Overlap fraction below --min-overlap-fraction returns 3."""
        import sys
        scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import verify_migration_swap as vms

        # 10 mp rows, 10 lake rows, but only 2 overlap (20%).
        rows = [_mp_row(f"mp{i}") for i in range(8)] + [_mp_row("shared0"), _mp_row("shared1")]
        lake = [_lake_row(f"lake{i}") for i in range(8)] + [_lake_row("shared0"), _lake_row("shared1")]
        _patch_pulls(monkeypatch, rows, lake)
        # Default --min-overlap-fraction is 0.90; 0.2 fails.
        assert vms.main(_base_argv()) == 3

    def test_outcome_mismatches_return_three(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Outcome-mismatch fraction above --max-outcome-mismatch-fraction returns 3."""
        import sys
        scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import verify_migration_swap as vms

        # 10 rows fully overlapping, but 5 have disagreeing outcomes
        # (50% mismatch, default threshold is 0.02).
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = []
        for i in range(10):
            outcome = False if i < 5 else True  # first 5 disagree with mp
            lake.append(_lake_row(f"r{i}", final_outcome=outcome))
        _patch_pulls(monkeypatch, rows, lake)
        assert vms.main(_base_argv()) == 3

    def test_below_threshold_but_nonzero_mismatch_still_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tiny outcome-mismatch under the threshold does not fail the gate.

        Guard against a future edit that flips ``>`` to ``>=`` on the
        mismatch threshold and rejects the accepted-delta band Ojus
        called out in the migration plan.
        """
        import sys
        scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import verify_migration_swap as vms

        # 100 rows fully overlapping, 1 mismatch (1%, default max 2%).
        rows = [_mp_row(f"r{i}") for i in range(100)]
        lake = []
        for i in range(100):
            outcome = False if i == 0 else True
            lake.append(_lake_row(f"r{i}", final_outcome=outcome))
        _patch_pulls(monkeypatch, rows, lake)
        argv = _base_argv() + ["--min-rows", "50"]
        assert vms.main(argv) == 0
