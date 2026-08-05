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
on materially divergent data. These tests pin exit codes 0/2/3/4 and
the artifact contract.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from scripts import verify_migration_swap as vms


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
    marketplace_ids_by_platform: dict[str, set[str]] | None = None,
    lake_ids_unfiltered: set[str] | None = None,
) -> None:
    """Replace every pull entry point so main() runs with synthetic data.

    ``main()`` calls four pull functions now — the two original ones
    (``_pull_mech_predict_rows``, ``_pull_lake_rows``) plus the two
    coverage-check helpers (``_pull_marketplace_request_ids``,
    ``_pull_lake_request_ids_unfiltered``). Stubbing only two makes the
    tests attempt real outbound HTTP for the other two, which fails on
    ``MechAnalyticsError: MECH_ANALYTICS_URL is not set``.

    Defaults derive the coverage-side ids from ``mp_rows`` / ``lake_rows``
    so tests that only care about parity don't have to spell them out.
    ``marketplace_ids_by_platform`` and ``lake_ids_unfiltered`` are
    explicit knobs for the coverage-focused tests.
    """
    dtr = (
        deliver_to_request
        if deliver_to_request is not None
        else {
            "omen": {r["deliver_id"]: r["request_id"] for r in mp_rows},
        }
    )
    mp_ids = {r["request_id"] for r in mp_rows if r.get("request_id")}
    lake_ids = {r["request_id"] for r in lake_rows if r.get("request_id")}
    marketplace = (
        marketplace_ids_by_platform
        if marketplace_ids_by_platform is not None
        else {"omen": mp_ids, "polymarket": set()}
    )
    # Default: lake has at least everything mech-predict has (no coverage
    # gap) so pre-existing parity tests aren't secondarily blocked by the
    # coverage-gap exit code (4). Tests that specifically want to
    # exercise coverage-gap semantics pass their own set.
    lake_unfiltered = (
        lake_ids_unfiltered if lake_ids_unfiltered is not None else lake_ids | mp_ids
    )

    def _mp(_since_ts: int, _until_ts: int) -> tuple[list[dict[str, Any]], dict]:
        return mp_rows, dtr

    def _lake(_since: datetime, _until: datetime) -> list[dict[str, Any]]:
        return lake_rows

    def _marketplace_ids(
        _since_ts: int, _until_ts: int
    ) -> tuple[dict[str, set[str]], dict[str, int]]:
        # Second element = per-platform ``dropped_no_request_id``. Zero
        # here for the parity-focused tests that use the default; the
        # coverage-focused tests spell it out via monkeypatch.
        return marketplace, {p: 0 for p in marketplace}

    def _lake_ids(_since: datetime, _until: datetime) -> set[str]:
        return lake_unfiltered

    monkeypatch.setattr(vms, "_pull_mech_predict_rows", _mp)
    monkeypatch.setattr(vms, "_pull_lake_rows", _lake)
    monkeypatch.setattr(vms, "_pull_marketplace_request_ids", _marketplace_ids)
    monkeypatch.setattr(vms, "_pull_lake_request_ids_unfiltered", _lake_ids)


def _base_argv() -> list[str]:
    """Argv covering a 24h window so _compute_window doesn't reject.

    ``--min-marketplace-rows 1`` overrides the production default
    (10) — test fixtures have single-digit row counts on purpose
    and the coverage-side floor exists to catch a subgraph outage in
    production, not to constrain fixtures. Tests that specifically
    exercise the min-marketplace-rows gate pass their own value.
    """
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    until = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    return [
        "--since",
        since,
        "--until",
        until,
        "--min-rows",
        "5",
        "--min-marketplace-rows",
        "1",
    ]


class TestDivergenceGate:
    """Exit code contract: 0 pass, 2 thin, 3 divergent."""

    def test_perfect_match_returns_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """100% overlap + no outcome mismatches passes the gate."""
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        _patch_pulls(monkeypatch, rows, lake)
        assert vms.main(_base_argv()) == 0

    def test_thin_side_trips_min_row_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Either side under --min-rows returns 2 (vacuous window)."""
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row("r0"), _lake_row("r1")]
        _patch_pulls(monkeypatch, rows, lake)
        assert vms.main(_base_argv()) == 2

    def test_low_overlap_returns_three(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Overlap fraction below --min-overlap-fraction returns 3."""
        rows = [_mp_row(f"mp{i}") for i in range(8)] + [
            _mp_row("shared0"),
            _mp_row("shared1"),
        ]
        lake = [_lake_row(f"lake{i}") for i in range(8)] + [
            _lake_row("shared0"),
            _lake_row("shared1"),
        ]
        _patch_pulls(monkeypatch, rows, lake)
        assert vms.main(_base_argv()) == 3

    def test_outcome_mismatches_return_three(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Outcome-mismatch fraction above --max-outcome-mismatch-fraction returns 3."""
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = []
        for i in range(10):
            outcome = not i < 5
            lake.append(_lake_row(f"r{i}", final_outcome=outcome))
        _patch_pulls(monkeypatch, rows, lake)
        assert vms.main(_base_argv()) == 3

    def test_below_threshold_but_nonzero_mismatch_still_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tiny outcome-mismatch under the threshold does not fail the gate."""
        # Guard against a future edit that flips ``>`` to ``>=`` on the
        # mismatch threshold and rejects the accepted-delta band Ojus
        # called out in the migration plan.
        rows = [_mp_row(f"r{i}") for i in range(100)]
        lake = []
        for i in range(100):
            outcome = i != 0
            lake.append(_lake_row(f"r{i}", final_outcome=outcome))
        _patch_pulls(monkeypatch, rows, lake)
        argv = _base_argv() + ["--min-rows", "50"]
        assert vms.main(argv) == 0

    def test_overlap_denominator_is_set_size_not_row_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multi-delivery-per-request rows must not inflate the denominator."""
        # Numerator is a set size (|intersection of request_ids|). Denominator
        # must also be a set size: min(|mp_ids|, |lake_ids|). If a future edit
        # reverts to ``min(len(mp_rows), len(lake_rows))`` the fraction dips
        # below any sane threshold on multi-deliver windows and the gate
        # rejects a perfectly aligned dataset.
        # 6 request_ids on mp with 2 deliveries each (12 rows) and 8 unique
        # request_ids on lake (8 rows). Only the 6 mp ids are shared with
        # lake, so |intersection| = 6.
        # Row-count denom (bug) = min(12, 8) = 8 -> 6/8 = 0.75 -> gate fails.
        # Set-size denom (fix) = min(6, 8) = 6 -> 6/6 = 1.0 -> gate passes.
        rows: list[dict[str, Any]] = []
        for i in range(6):
            rid = f"r{i}"
            rows.append(_mp_row(rid, deliver_id=f"{rid}-d1"))
            rows.append(_mp_row(rid, deliver_id=f"{rid}-d2"))
        lake = [_lake_row(f"r{i}") for i in range(8)]
        dtr = {"omen": {r["deliver_id"]: r["request_id"] for r in rows}}
        _patch_pulls(monkeypatch, rows, lake, deliver_to_request=dtr)
        argv = _base_argv() + ["--min-rows", "5", "--min-overlap-fraction", "0.9"]
        assert vms.main(argv) == 0


class TestCoverageGate:
    """Exit code 4 covers three coverage failure modes, each blocking PASS.

    The coverage check runs before the parity min-row guard on purpose:
    a coverage gap and a thin parity window are correlated (a lake
    missing rows *is* a lake with fewer rows), so a coverage failure
    would get miscategorised as "window too thin" if parity ran first.
    Each of the three failure modes here must return 4, not 2.
    """

    def test_marketplace_id_missing_from_lake_returns_four(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single request_id in the marketplace but not the lake trips exit 4."""
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        marketplace_ids = {r["request_id"] for r in rows} | {"only-marketplace"}
        # Lake unfiltered lacks ``only-marketplace``.
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            marketplace_ids_by_platform={"omen": marketplace_ids, "polymarket": set()},
            lake_ids_unfiltered={r["request_id"] for r in lake},
        )
        assert vms.main(_base_argv()) == 4

    def test_dropped_no_request_id_returns_four(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A marketplace delivery dropped for missing request.id blocks PASS.

        Even with a perfect set-diff (every returned id is in the lake),
        a non-zero drop means one or more requests could be silently
        absent from the lake without the check being able to name them.
        The gate must fail.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        marketplace_ids = {r["request_id"] for r in rows}

        def _marketplace_ids(
            _since_ts: int, _until_ts: int
        ) -> tuple[dict[str, set[str]], dict[str, int]]:
            # One row dropped on the omen side, nothing missing from the
            # lake for what we DID resolve.
            return {"omen": marketplace_ids, "polymarket": set()}, {
                "omen": 1,
                "polymarket": 0,
            }

        def _lake_ids(_since: datetime, _until: datetime) -> set[str]:
            return marketplace_ids

        _patch_pulls(monkeypatch, rows, lake)
        monkeypatch.setattr(vms, "_pull_marketplace_request_ids", _marketplace_ids)
        monkeypatch.setattr(vms, "_pull_lake_request_ids_unfiltered", _lake_ids)
        assert vms.main(_base_argv()) == 4

    def test_vacuous_marketplace_side_returns_four(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both marketplace platforms returning zero rows must not PASS.

        ``marketplace_ids - lake_ids`` is trivially empty when the
        marketplace side is empty. Without the ``--min-marketplace-rows``
        floor a subgraph outage would silently PASS as "no missing IDs".
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            marketplace_ids_by_platform={"omen": set(), "polymarket": set()},
            lake_ids_unfiltered={r["request_id"] for r in lake},
        )
        # Base argv uses --min-marketplace-rows 1, override upward so a
        # zero-row total actually trips the floor.
        argv = [
            "--since",
            (datetime.now(timezone.utc) - timedelta(days=2))
            .isoformat()
            .replace("+00:00", "Z"),
            "--until",
            (datetime.now(timezone.utc) - timedelta(days=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "--min-rows",
            "5",
            "--min-marketplace-rows",
            "1",
        ]
        assert vms.main(argv) == 4


class TestArtifact:
    """--output-dir artifact contract: PASS runs write PASS + exit_code 0;
    exception in flight writes ERROR + non-zero exit_code (never a false PASS).
    """

    def test_pass_run_writes_pass_verdict_and_zero_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        _patch_pulls(monkeypatch, rows, lake)
        argv = _base_argv() + ["--output-dir", str(tmp_path)]

        assert vms.main(argv) == 0

        artifacts = sorted(tmp_path.glob("*.json"))
        assert len(artifacts) == 1
        data = json.loads(artifacts[0].read_text())
        assert data["verdict"] == "PASS"
        assert data["exit_code"] == 0
        assert data["coverage"]["total_missing"] == 0

    def test_exception_in_flight_writes_error_not_false_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A raised exception must not leave the artifact certifying PASS.

        Before the fix, ``exit_code`` initialised to 0 and only flipped
        on the explicit return paths, so any exception in the try body
        landed a JSON artifact with ``verdict: PASS`` on the mounted
        PVC. The K8s Job runner's non-zero exit told the truth, the
        report on the volume did not.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        _patch_pulls(monkeypatch, rows, lake)

        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("simulated subgraph outage")

        # Blow up on the first coverage-side pull after metadata is emitted.
        monkeypatch.setattr(vms, "_pull_marketplace_request_ids", _boom)
        argv = _base_argv() + ["--output-dir", str(tmp_path)]

        with pytest.raises(RuntimeError, match="simulated subgraph outage"):
            vms.main(argv)

        artifacts = sorted(tmp_path.glob("*.json"))
        assert len(artifacts) == 1
        data = json.loads(artifacts[0].read_text())
        assert data["verdict"] == "ERROR"
        assert data["exit_code"] != 0
