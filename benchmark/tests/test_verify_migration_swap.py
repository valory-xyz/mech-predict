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


@pytest.fixture(autouse=True)
def _isolate_predict_api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ``PREDICT_API_POSTGRES_URL`` from the process env for every test.

    ``main()`` reads this variable as a fallback for ``--predict-api-url``.
    If it leaks in from the ambient shell (CLAUDE.md's ``source .env``
    workflow, a docker-compose ``.env`` file, a CI job export), every
    test that constructs an argv without ``--predict-api-url`` would
    silently try to hit that DSN — attempting a real ``psycopg.connect``
    with a 30 s timeout AND changing the verdict path (raw vs adjusted)
    so pre-existing tests fail for reasons unrelated to what they
    assert. Autouse so no test has to remember, module-scoped so the
    coupling is impossible to reintroduce accidentally.

    :param monkeypatch: pytest fixture used to unset the env var
        for the duration of every test in this module.
    """
    monkeypatch.delenv("PREDICT_API_POSTGRES_URL", raising=False)


@pytest.fixture(autouse=True)
def _stub_undelivered_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``_pull_undelivered_pending`` to return empty sets by default.

    Every test that supplies ``--predict-api-url`` (or the
    ``PREDICT_API_POSTGRES_URL`` env var) also implicitly enables the
    undelivered-pending pull. Without this stub each such test would
    open a real ``psycopg.connect`` against the placeholder DSN
    ``postgresql://ignored`` and hang the suite on DNS resolution.
    Tests that specifically exercise the undelivered-pending semantics
    override this stub with their own fixture data.

    :param monkeypatch: pytest fixture used to swap the
        ``_pull_undelivered_pending`` symbol on the module under test.
    """

    def _empty(_url: str, _since: datetime, _until: datetime) -> dict[str, set[str]]:
        return {"omen": set(), "polymarket": set()}

    monkeypatch.setattr(vms, "_pull_undelivered_pending", _empty)


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

    :param monkeypatch: pytest fixture used to swap the four pull
        entry points on ``vms``.
    :param mp_rows: synthetic mech-predict rows.
    :param lake_rows: synthetic lake rows for the resolved parity pull.
    :param deliver_to_request: per-platform ``{deliver_id: request_id}``.
        Defaults to a 1:1 map on the omen platform derived from
        ``mp_rows``.
    :param marketplace_ids_by_platform: explicit override for the
        marketplace-side coverage IDs. Defaults to ``mp_rows`` request
        ids on the omen platform only — polymarket is omitted so the
        per-platform vacuous-sweep floor doesn't secondarily block
        parity-focused tests whose fixtures are single-platform.
    :param lake_ids_unfiltered: explicit override for the lake-side
        coverage IDs. Defaults to a union of ``lake_rows`` and
        ``mp_rows`` so parity tests don't secondarily fail the
        coverage gate.
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
        else {"omen": mp_ids}
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

    :return: argv list suitable for feeding to ``vms.main``.
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
        """A single request_id in the marketplace but not the lake trips exit 4.

        Both platforms are populated above the vacuous floor so the ONLY
        reason to return 4 is the missing_from_lake predicate. A previous
        version of this test seeded ``polymarket: set()`` which tripped the
        per-platform vacuous floor before the missing-from-lake check ever
        fired — the test asserted exit 4 for the wrong reason. Mutating
        ``if coverage.total_missing > 0: → if False:`` would silently keep
        the old test green while completely deleting the feature this PR
        adds. This test is written so that mutation now fails.

        :param monkeypatch: pytest fixture used to swap the four pull
            entry points.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        omen_ids = {r["request_id"] for r in rows} | {"only-marketplace"}
        # Populate polymarket too, all present in the lake, so only the
        # missing "only-marketplace" ID is the reason for a coverage gap.
        polymarket_ids = {f"poly{i}" for i in range(5)}
        lake_unfiltered = {r["request_id"] for r in lake} | polymarket_ids
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            marketplace_ids_by_platform={
                "omen": omen_ids,
                "polymarket": polymarket_ids,
            },
            lake_ids_unfiltered=lake_unfiltered,
        )
        assert vms.main(_base_argv()) == 4

    def test_dropped_no_request_id_above_tolerance_returns_four(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drop count above tolerance still blocks PASS.

        Even with a perfect set-diff (every returned id is in the lake),
        a large drop count signals subgraph schema drift or a broken
        ingest — one or more requests could be silently absent from the
        lake without the check being able to name them. Above the
        tolerance (``max(_DROPPED_NO_RID_ABSOLUTE_TOLERANCE, ...)``) the
        gate must fail. Below the tolerance a subgraph-side data quality
        artifact of a handful of stale drops is tolerated as a warning
        (see ``test_dropped_no_request_id_within_tolerance_warns``).

        Both platforms populated above the vacuous floor here for the same
        reason as the sibling test above — the drop counter is what has to
        trip, not the floor.

        :param monkeypatch: pytest fixture used to override the
            marketplace pull stub with a non-zero drop count.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        omen_ids = {r["request_id"] for r in rows}
        polymarket_ids = {f"poly{i}" for i in range(5)}
        # Absolute tolerance is 100; anything strictly above trips fail-close.
        tol = vms._DROPPED_NO_RID_ABSOLUTE_TOLERANCE  # pylint: disable=protected-access
        drops_above_tolerance = tol + 1

        def _marketplace_ids(
            _since_ts: int, _until_ts: int
        ) -> tuple[dict[str, set[str]], dict[str, int]]:
            return {"omen": omen_ids, "polymarket": polymarket_ids}, {
                "omen": drops_above_tolerance,
                "polymarket": 0,
            }

        def _lake_ids(_since: datetime, _until: datetime) -> set[str]:
            return omen_ids | polymarket_ids

        _patch_pulls(monkeypatch, rows, lake)
        monkeypatch.setattr(vms, "_pull_marketplace_request_ids", _marketplace_ids)
        monkeypatch.setattr(vms, "_pull_lake_request_ids_unfiltered", _lake_ids)
        assert vms.main(_base_argv()) == 4

    def test_dropped_no_request_id_within_tolerance_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A small drop count (subgraph data quality artifact) logs a warning and passes.

        The threshold is ``max(_DROPPED_NO_RID_ABSOLUTE_TOLERANCE,
        _DROPPED_NO_RID_RELATIVE_TOLERANCE * total_marketplace)``. A
        handful of stale drops from missing subgraph Request events is
        a data-quality issue we can't zero from our side, so it must
        not fail the run. Verifies the tolerance is applied AND a
        warning is emitted so operators can see the drop count in the
        logs.

        :param monkeypatch: pytest fixture used to swap the subgraph +
            lake helpers with in-memory stubs.
        :param caplog: pytest fixture capturing log records so the
            warning can be asserted.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        omen_ids = {r["request_id"] for r in rows}
        polymarket_ids = {f"poly{i}" for i in range(5)}
        tol = vms._DROPPED_NO_RID_ABSOLUTE_TOLERANCE  # pylint: disable=protected-access
        drops_within_tolerance = tol

        def _marketplace_ids(
            _since_ts: int, _until_ts: int
        ) -> tuple[dict[str, set[str]], dict[str, int]]:
            return {"omen": omen_ids, "polymarket": polymarket_ids}, {
                "omen": drops_within_tolerance,
                "polymarket": 0,
            }

        def _lake_ids(_since: datetime, _until: datetime) -> set[str]:
            return omen_ids | polymarket_ids

        _patch_pulls(monkeypatch, rows, lake)
        monkeypatch.setattr(vms, "_pull_marketplace_request_ids", _marketplace_ids)
        monkeypatch.setattr(vms, "_pull_lake_request_ids_unfiltered", _lake_ids)
        with caplog.at_level("WARNING"):
            assert vms.main(_base_argv()) == 0
        assert any(
            "within tolerance" in rec.message and "dropped" in rec.message
            for rec in caplog.records
        )

    def test_min_marketplace_rows_zero_still_traps_empty_platform(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator setting the flag to 0 must not turn off the zero-backstop.

        ``--min-marketplace-rows 0`` used to let a total marketplace
        outage on one platform PASS: the enforcement was strict
        ``count < min``, and ``count < 0`` never held. That would let
        an operator opt out of the check entirely with one env var,
        turning a subgraph outage into a green artifact. The floor is
        now ``max(1, min_marketplace_rows)`` so zero is always
        vacuous, regardless of the flag.

        :param monkeypatch: pytest fixture used to swap the four pull
            entry points.
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
            "0",
        ]
        assert vms.main(argv) == 4

    def test_coverage_gap_precedence_over_min_row_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When both min-rows (2) and coverage-gap (4) could fire, coverage wins.

        Sets up a state where the parity pull is thin enough to trip the
        min-row guard AND the marketplace side has a missing-from-lake
        request. If the order in ``main()`` is ever swapped (or the
        coverage check moved after the parity min-row check), this
        assertion flips to 2. That would produce artifacts labelled as
        "window too thin" for actual coverage gaps — misleading exactly
        the operator this gate exists for.

        :param monkeypatch: pytest fixture used to swap the four pull
            entry points.
        """
        # Two rows on each side (below --min-rows 5 in _base_argv). Both
        # platforms above the vacuous floor.
        rows = [_mp_row(f"r{i}") for i in range(2)]
        lake = [_lake_row(f"r{i}") for i in range(2)]
        omen_ids = {r["request_id"] for r in rows} | {"only-marketplace"}
        polymarket_ids = {f"poly{i}" for i in range(5)}
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            marketplace_ids_by_platform={
                "omen": omen_ids,
                "polymarket": polymarket_ids,
            },
            # Lake missing "only-marketplace" → coverage-gap condition met.
            lake_ids_unfiltered={r["request_id"] for r in lake} | polymarket_ids,
        )
        assert vms.main(_base_argv()) == 4

    def test_vacuous_marketplace_side_returns_four(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both marketplace platforms returning zero rows must not PASS.

        ``marketplace_ids - lake_ids`` is trivially empty when the
        marketplace side is empty. Without the ``--min-marketplace-rows``
        floor a subgraph outage would silently PASS as "no missing IDs".

        :param monkeypatch: pytest fixture used to stub the pulls with
            empty marketplace-side sets.
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

    def test_single_platform_vacuous_returns_four(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One populated platform can't mask an outage on the other.

        The floor is enforced per platform, not on the aggregate. If
        omen returns 12 IDs (above the floor of 10) but polymarket
        returns zero (well below), aggregate = 12 would silently PASS
        an aggregate check while polymarket's coverage is fully
        vacuous. This test would fail if the guard were ever collapsed
        back to a sum-across-platforms check.

        :param monkeypatch: pytest fixture used to stub the pulls with
            one populated and one empty platform.
        """
        rows = [_mp_row(f"r{i}") for i in range(12)]
        lake = [_lake_row(f"r{i}") for i in range(12)]
        omen_ids = {r["request_id"] for r in rows}
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            # omen well above the floor, polymarket empty.
            marketplace_ids_by_platform={
                "omen": omen_ids,
                "polymarket": set(),
            },
            # Lake covers every marketplace ID so ``missing_from_lake`` is
            # zero — this test isolates the per-platform floor from the
            # set-diff failure mode.
            lake_ids_unfiltered=omen_ids,
        )
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
            "10",
        ]
        assert vms.main(argv) == 4


class TestArtifact:
    """--output-dir artifact contract.

    PASS runs write ``verdict=PASS + exit_code=0``; an exception in
    flight writes ``verdict=ERROR + non-zero exit_code`` — never a
    false PASS on the mounted volume regardless of what killed the run.
    """

    def test_pass_run_writes_pass_verdict_and_zero_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A clean parity+coverage PASS lands the expected artifact.

        :param monkeypatch: pytest fixture used to swap the pull entries.
        :param tmp_path: pytest fixture giving a per-test writable dir.
        """
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

        :param monkeypatch: pytest fixture used to inject a raising
            stub into the first coverage pull.
        :param tmp_path: pytest fixture giving a per-test writable dir.
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


class TestKnownLossAdjustment:
    """Coverage-gate contract with the known-loss adjustment on.

    Predict-api marks structurally unrecoverable IPFS payloads in
    ``mech_migration_failures`` (parsed_request_null,
    parsed_delivery_null, unhandled_type_prompt). When the operator
    passes ``--predict-api-url`` / ``PREDICT_API_POSTGRES_URL`` the
    coverage check must subtract that set from the missing side AND
    from the marketplace denominator BEFORE the verdict fires,
    otherwise the check reports a coverage gap for a population the
    pipeline cannot ingest. Every test here mutates one thing about
    the production logic and confirms the assertion flips.
    """

    def test_known_loss_subtraction_flips_missing_to_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raw missing-from-lake set fully covered by known-loss must PASS.

        Without the subtraction the coverage-gap branch fires (exit 4).
        With the subtraction the adjusted missing set is empty and the
        gate reaches the parity path (which passes for identical rows).
        Mutating ``adjusted_missing_set = raw_missing_set - excluded_set``
        back to ``adjusted_missing_set = raw_missing_set`` must fail
        this test.

        :param monkeypatch: pytest fixture used to swap the four pull
            entries and the known-loss pull on the module under test.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        # Marketplace has 10 IDs the lake also has plus 3 IDs the lake
        # doesn't have. Predict-api marks all 3 of those as known-loss,
        # so the adjusted missing set is empty.
        marketplace = {
            "omen": {r["request_id"] for r in rows}
            | {"unparseable-a", "unparseable-b", "unparseable-c"},
            "polymarket": {f"poly{i}" for i in range(5)},
        }
        lake_ids = {r["request_id"] for r in lake} | {f"poly{i}" for i in range(5)}
        known_loss = {
            "omen": {"unparseable-a", "unparseable-b", "unparseable-c"},
            "polymarket": set(),
        }
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            marketplace_ids_by_platform=marketplace,
            lake_ids_unfiltered=lake_ids,
        )
        monkeypatch.setattr(vms, "_pull_known_loss_failures", lambda _url: known_loss)
        argv = _base_argv() + ["--predict-api-url", "postgresql://ignored"]
        assert vms.main(argv) == 0

    def test_raw_missing_from_lake_without_flag_still_fails_four(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without --predict-api-url the raw missing set still fails exit 4.

        Guard against the case where the subtraction accidentally
        applies unconditionally (e.g. a bug that treats ``None`` as
        an empty dict at the wrong layer). If ``known_loss_by_platform``
        is ``None`` the coverage gate must behave exactly as before
        the flag existed.

        :param monkeypatch: pytest fixture used to swap the four pull
            entries and to install a raising stub on the known-loss
            pull.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        marketplace = {
            "omen": {r["request_id"] for r in rows} | {"unparseable-a"},
            "polymarket": {f"poly{i}" for i in range(5)},
        }
        lake_ids = {r["request_id"] for r in lake} | {f"poly{i}" for i in range(5)}
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            marketplace_ids_by_platform=marketplace,
            lake_ids_unfiltered=lake_ids,
        )
        # Argv omits the predict-api URL, so the known-loss pull must
        # not be called. Assert that by monkeypatching it to a raiser.
        monkeypatch.setattr(
            vms,
            "_pull_known_loss_failures",
            lambda _url: (_ for _ in ()).throw(
                AssertionError("must not query known-loss without --predict-api-url")
            ),
        )
        assert vms.main(_base_argv()) == 4

    def test_known_loss_partial_leaves_residual_missing_and_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If only some of the missing IDs are known-loss, the rest fail exit 4.

        Guards against a bug where the subtraction accidentally clears
        the full missing set instead of the intersection with
        known-loss.

        :param monkeypatch: pytest fixture used to swap the four pull
            entries and the known-loss pull on the module under test.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        marketplace = {
            "omen": {r["request_id"] for r in rows} | {"unparseable-a", "genuine-gap"},
            "polymarket": {f"poly{i}" for i in range(5)},
        }
        lake_ids = {r["request_id"] for r in lake} | {f"poly{i}" for i in range(5)}
        known_loss = {"omen": {"unparseable-a"}, "polymarket": set()}
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            marketplace_ids_by_platform=marketplace,
            lake_ids_unfiltered=lake_ids,
        )
        monkeypatch.setattr(vms, "_pull_known_loss_failures", lambda _url: known_loss)
        argv = _base_argv() + ["--predict-api-url", "postgresql://ignored"]
        assert vms.main(argv) == 4

    def test_env_var_fallback_activates_known_loss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PREDICT_API_POSTGRES_URL alone is enough to activate the flag.

        Infra sets the DSN as a K8s secret env, not as a CLI flag —
        so the env-var fallback path must also enable the known-loss
        pull. Mutating ``args.predict_api_url or os.environ.get(...)``
        to only read ``args.predict_api_url`` must fail this test.

        :param monkeypatch: pytest fixture used to swap the pull
            entries, install a recording known-loss stub, and set
            the ``PREDICT_API_POSTGRES_URL`` env var.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        marketplace = {
            "omen": {r["request_id"] for r in rows} | {"unparseable-a"},
            "polymarket": {f"poly{i}" for i in range(5)},
        }
        lake_ids = {r["request_id"] for r in lake} | {f"poly{i}" for i in range(5)}
        known_loss = {"omen": {"unparseable-a"}, "polymarket": set()}
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            marketplace_ids_by_platform=marketplace,
            lake_ids_unfiltered=lake_ids,
        )
        called: dict[str, str] = {}

        def _fake_known_loss(url: str) -> dict[str, set[str]]:
            called["url"] = url
            return known_loss

        monkeypatch.setattr(vms, "_pull_known_loss_failures", _fake_known_loss)
        monkeypatch.setenv("PREDICT_API_POSTGRES_URL", "postgresql://from-env")

        assert vms.main(_base_argv()) == 0
        assert called == {"url": "postgresql://from-env"}

    def test_min_marketplace_rows_applies_to_adjusted_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The vacuous-sweep floor compares against the adjusted count.

        The direction that matters: raw is always >= adjusted, so
        checking raw counts against the floor risks a false PASS when
        the adjusted (ingestible) population is genuinely too thin
        to draw a conclusion from. The floor has to see the adjusted
        number to catch a vacuous adjusted sweep.

        Fixture: 15 raw omen IDs, 5 of which are known-loss ->
        adjusted count = 15 - 5 = 10. Floor is 12. Adjusted 10 < 12
        trips the floor and returns exit 4. If a future edit checks
        the raw count here (15 > 12) the assertion flips to 0 and
        the vacuous-adjusted-sweep case slips through.

        :param monkeypatch: pytest fixture used to swap the four pull
            entries and the known-loss pull on the module under test.
        """
        # Small rows list under --min-rows just to isolate the coverage
        # verdict from the parity gate — parity would trip on 10 rows
        # regardless, but coverage runs first.
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        # 15 raw omen IDs, 10 of which are known-loss → adjusted count 5
        raw_omen = {r["request_id"] for r in rows} | {
            f"unparseable-{i}" for i in range(5)
        }
        marketplace = {
            "omen": raw_omen,
            "polymarket": {f"poly{i}" for i in range(20)},
        }
        # Lake has everything ingestible (the 10 real omen IDs plus
        # polymarket) so ``missing_from_lake`` is only the known-loss
        # set, which the subtraction handles.
        lake_ids = {r["request_id"] for r in rows} | {f"poly{i}" for i in range(20)}
        known_loss = {
            "omen": {f"unparseable-{i}" for i in range(5)},
            "polymarket": set(),
        }
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            marketplace_ids_by_platform=marketplace,
            lake_ids_unfiltered=lake_ids,
        )
        monkeypatch.setattr(vms, "_pull_known_loss_failures", lambda _url: known_loss)
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
            "12",
            "--predict-api-url",
            "postgresql://ignored",
        ]
        # Raw omen = 15 (above floor 12), adjusted omen = 15 - 5 = 10
        # (below floor 12). Fails on the adjusted count.
        assert vms.main(argv) == 4

    def test_report_shape_contains_raw_and_adjusted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The JSON artifact records both raw and adjusted counts.

        A reviewer looking at a passed artifact still needs to see
        how much was excluded and what the raw picture looked like —
        otherwise the adjustment is invisible after the fact. The
        exact key names below are what infra's report indexer keys
        on; the ``coverage`` sub-object must carry every field.

        :param monkeypatch: pytest fixture used to swap the four pull
            entries and the known-loss pull on the module under test.
        :param tmp_path: pytest fixture giving a per-test writable dir
            that receives the ``--output-dir`` artifact.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        marketplace = {
            "omen": {r["request_id"] for r in rows}
            | {"unparseable-a", "unparseable-b"},
            "polymarket": {f"poly{i}" for i in range(5)},
        }
        lake_ids = {r["request_id"] for r in lake} | {f"poly{i}" for i in range(5)}
        known_loss = {
            "omen": {"unparseable-a", "unparseable-b"},
            "polymarket": set(),
        }
        _patch_pulls(
            monkeypatch,
            rows,
            lake,
            marketplace_ids_by_platform=marketplace,
            lake_ids_unfiltered=lake_ids,
        )
        monkeypatch.setattr(vms, "_pull_known_loss_failures", lambda _url: known_loss)
        argv = _base_argv() + [
            "--output-dir",
            str(tmp_path),
            "--predict-api-url",
            "postgresql://ignored",
        ]

        assert vms.main(argv) == 0

        artifacts = sorted(tmp_path.glob("*.json"))
        assert len(artifacts) == 1
        data = json.loads(artifacts[0].read_text())
        coverage = data["coverage"]
        assert coverage["known_loss_applied"] is True
        assert coverage["total_missing"] == 0
        assert coverage["total_excluded_known_loss"] == 2
        assert coverage["excluded_known_loss_by_platform"] == {
            "omen": 2,
            "polymarket": 0,
        }
        assert coverage["raw_marketplace_by_platform_count"] == {
            "omen": len(marketplace["omen"]),
            "polymarket": len(marketplace["polymarket"]),
        }
        assert coverage["raw_missing_from_lake_by_platform_count"] == {
            "omen": 2,
            "polymarket": 0,
        }
        # Adjusted counts also present — sanity check the arithmetic.
        assert coverage["marketplace_ids_by_platform_count"] == {
            "omen": len(marketplace["omen"]) - 2,
            "polymarket": len(marketplace["polymarket"]),
        }

    def test_report_shape_when_flag_missing_flags_raw_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Without the flag the artifact records ``known_loss_applied: False``.

        Downstream consumers need a machine-readable signal that this
        run's verdict is on RAW coverage, not adjusted — otherwise a
        historical dashboard mixes the two silently.

        :param monkeypatch: pytest fixture used to swap the four pull
            entries on the module under test.
        :param tmp_path: pytest fixture giving a per-test writable dir
            that receives the ``--output-dir`` artifact.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        _patch_pulls(monkeypatch, rows, lake)
        argv = _base_argv() + ["--output-dir", str(tmp_path)]

        assert vms.main(argv) == 0

        artifacts = sorted(tmp_path.glob("*.json"))
        data = json.loads(artifacts[0].read_text())
        coverage = data["coverage"]
        assert coverage["known_loss_applied"] is False
        assert coverage["total_excluded_known_loss"] == 0
        # Every platform is present in the excluded map with a zero count
        # regardless of the flag (the map keys track what was iterated,
        # not what was excluded). The flag/verdict signal is
        # ``known_loss_applied`` — that's what downstream indexers key on.
        assert coverage["excluded_known_loss_by_platform"] == {"omen": 0}


class TestReportCoverageCheckDirect:
    """Direct tests for :func:`_report_coverage_check`'s subtraction logic.

    ``main()``-level tests are integration-style. These pin the small
    unit that computes the subtraction so a future refactor that
    changes the flow through ``main()`` doesn't hide an arithmetic
    regression here.
    """

    def test_excluded_set_is_intersection_of_raw_missing_and_known_loss(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Excluded = raw_missing ∩ known_loss, not just known_loss.

        A known-loss ID that IS in the lake (e.g. a stale entry the
        indexer already retried and succeeded on) must not be counted
        as excluded — that would inflate the excluded number without
        actually shrinking the missing set. Mutating
        ``excluded_set = raw_missing_set & known_loss`` back to
        ``excluded_set = known_loss`` must fail this assertion.

        :param capsys: pytest fixture used to discard the coverage
            summary print so the test output stays clean.
        """
        marketplace = {"omen": {"a", "b", "c", "d"}}
        # Lake has "a" (present) and "d" (present). Missing = {b, c}.
        lake_ids = {"a", "d"}
        # Known-loss claims "c" (in missing → real exclude) and
        # "a" (in lake → must NOT count as excluded).
        known_loss = {"omen": {"a", "c"}}
        result = vms._report_coverage_check(  # pylint: disable=protected-access
            marketplace, {"omen": 0}, lake_ids, 1, known_loss_by_platform=known_loss
        )
        capsys.readouterr()  # discard printed output
        assert result.excluded_by_platform == {"omen": 1}
        assert result.total_excluded == 1
        assert result.missing_by_platform == {"omen": ["b"]}
        assert result.total_missing == 1
        assert result.marketplace_counts == {"omen": 3}

    def test_known_loss_none_yields_zero_exclusions_and_raw_verdict(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``known_loss_by_platform=None`` skips the adjustment entirely.

        ``known_loss_applied`` must be False and no exclusions should
        be recorded. The raw counts should equal the adjusted counts.

        :param capsys: pytest fixture used to discard the coverage
            summary print so the test output stays clean.
        """
        marketplace = {"omen": {"a", "b"}}
        lake_ids = {"a"}
        # pylint: disable-next=protected-access
        result = vms._report_coverage_check(marketplace, {"omen": 0}, lake_ids, 1)
        capsys.readouterr()
        assert result.known_loss_applied is False
        assert result.total_excluded == 0
        assert result.excluded_by_platform == {"omen": 0}
        assert result.missing_by_platform == {"omen": ["b"]}
        assert result.raw_missing_by_platform == {"omen": ["b"]}
        assert result.marketplace_counts == {"omen": 2}
        assert result.raw_marketplace_counts == {"omen": 2}

    def test_empty_known_loss_dict_still_marks_applied(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Passing ``{}`` is different from ``None`` — flag is on, zero excludes.

        An operator whose predict-api instance has an empty
        ``mech_migration_failures`` table (a happy but rare state)
        must still see ``known_loss_applied=True`` — otherwise the
        artifact conflates "we asked and got nothing" with "we didn't
        ask".

        :param capsys: pytest fixture used to discard the coverage
            summary print so the test output stays clean.
        """
        marketplace = {"omen": {"a", "b"}}
        lake_ids = {"a", "b"}
        result = vms._report_coverage_check(  # pylint: disable=protected-access
            marketplace, {"omen": 0}, lake_ids, 1, known_loss_by_platform={}
        )
        capsys.readouterr()
        assert result.known_loss_applied is True
        assert result.total_excluded == 0
        assert result.missing_by_platform == {"omen": []}

    def test_known_loss_pull_failure_writes_error_not_false_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A predict-api DB failure must land an ERROR artifact.

        Same class of guarantee as
        ``test_exception_in_flight_writes_error_not_false_pass`` but
        for the ``_pull_known_loss_failures`` call site: a psycopg
        connect timeout, a role-permission error, or a network hiccup
        must not certify PASS in the artifact just because raw
        coverage would have been zero missing. The sentinel discipline
        that keeps ``exit_code`` at 1 and ``verdict`` at ERROR until
        the explicit success path flips them has to hold here too.

        :param monkeypatch: pytest fixture used to swap the four pull
            entries and inject a raising known-loss stub.
        :param tmp_path: pytest fixture giving a per-test writable dir.
        """
        rows = [_mp_row(f"r{i}") for i in range(10)]
        lake = [_lake_row(f"r{i}") for i in range(10)]
        _patch_pulls(monkeypatch, rows, lake)

        def _boom(_url: str) -> dict[str, set[str]]:
            raise RuntimeError("simulated predict-api DB outage")

        monkeypatch.setattr(vms, "_pull_known_loss_failures", _boom)
        argv = _base_argv() + [
            "--output-dir",
            str(tmp_path),
            "--predict-api-url",
            "postgresql://ignored",
        ]

        with pytest.raises(RuntimeError, match="simulated predict-api DB outage"):
            vms.main(argv)

        artifacts = sorted(tmp_path.glob("*.json"))
        assert len(artifacts) == 1
        data = json.loads(artifacts[0].read_text())
        assert data["verdict"] == "ERROR"
        assert data["exit_code"] != 0
