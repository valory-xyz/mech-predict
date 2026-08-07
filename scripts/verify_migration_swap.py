#!/usr/bin/env python3
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
r"""Reconcile mech-predict's row universe against mech-analytics.

Two checks per window:

* **Coverage** — every request delivered in the window (per the
  marketplace subgraph's ``delivers`` collection) must be present in
  the mech-analytics lake. This proves ingest completeness ahead of
  the mech-predict cutover: switching to the lake cannot silently
  drop rows that today's production path can see. Note this is a
  delivery-derived denominator: a ``Request`` never answered (or
  answered more than the delivery-lag envelope after the window
  closed) is outside the set. That matches what today's mech-predict
  path consumes; it is not a claim about request-level completeness.
  Fails with exit code 4 if any delivered request is missing from
  the lake.
* **Parity** — on the resolved intersection, ``final_outcome``,
  ``market_id`` binding, and per-tool ``brier`` / ``directional_accuracy``
  must agree between the two sides. Fails with exit code 3 if the
  overlap fraction is under ``--min-overlap-fraction`` or the outcome
  mismatch rate exceeds ``--max-outcome-mismatch-fraction``.

Left side (both checks): the actual production path in mech-predict —
marketplace subgraph + IPFS via ``benchmark.datasets.fetch_production``.
Both platforms (Omen on Gnosis, Polymarket on Polygon).

Right side (both checks): mech-analytics's ``/v1/data/scored-rows``
endpoint via ``benchmark.mech_analytics_client``. Coverage uses the
unfiltered call; parity uses ``resolved=true`` to match production's
scorer path.

Fails loud with exit code 2 if either side returns under
``MINIMUM_ROWS_PER_SIDE`` so a vacuous match on empty data doesn't get
reported as parity.

Usage:
    # local, prints to stdout
    scripts/verify_migration_swap.py --days 7

    # K8s Job style — also writes <window>.md and <window>.json
    scripts/verify_migration_swap.py --since 2026-07-01T00:00:00Z \\
                                     --until 2026-07-08T00:00:00Z \\
                                     --output-dir /reports

Exit codes:
    0 — coverage complete, parity within thresholds
    2 — min-row guard tripped (window too thin to conclude anything)
    3 — parity divergence exceeds thresholds
    4 — coverage gap (at least one delivered request missing from lake)

Required env vars:
    MECH_ANALYTICS_URL             (lake-side endpoint)
    MECH_MARKETPLACE_GNOSIS_URL    (has default in fetch_production)
    MECH_MARKETPLACE_POLYGON_URL   (has default in fetch_production)

Optional env vars:
    PREDICT_API_POSTGRES_URL       (predict-api DSN). When set, the
                                   coverage check excludes rows in
                                   ``mech_migration_failures`` from the
                                   missing-from-lake set before computing
                                   the verdict. Falls back to the
                                   ``--predict-api-url`` CLI flag.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger("verify_migration_swap")


# --------------------------------------------------------------------------- #
# Guards / knobs                                                              #
# --------------------------------------------------------------------------- #

# Below this row count on either side the parity comparison is vacuous.
MINIMUM_ROWS_PER_SIDE = 100

# Applied per platform: any single platform with fewer than
# ``MINIMUM_MARKETPLACE_ROWS`` delivered requests in the window fails
# the coverage gate. An empty coverage side on one platform makes
# ``marketplace_ids - lake_ids`` trivially empty regardless of whether
# the lake actually has anything, so a healthy pull on the other
# platform must not mask a subgraph outage on this one. The floor
# distinguishes "healthy sweep, no gap" from "sweep returned nothing"
# on a per-platform basis.
MINIMUM_MARKETPLACE_ROWS = 10

# Comparing floats coming from independent code paths — allow small drift.
FLOAT_TOL = 1e-6

# Mapping from chain_id (as stored in predict-api's ``mech_migration_failures``
# table) to the platform label the coverage check keys on. Predict-api's
# indexer records the origin chain per failed row, so a single chain-agnostic
# query returns everything and this map partitions the result into the same
# per-platform buckets the marketplace side uses. Extending to a new chain
# means adding a row here AND a new ``(platform, marketplace_url)`` entry in
# the marketplace pull — otherwise the excluded set would silently drop
# failures for the new chain.
_CHAIN_ID_TO_PLATFORM: dict[int, str] = {
    100: "omen",  # Gnosis Chain
    137: "polymarket",  # Polygon
}

# Only these reasons are structurally unrecoverable — the IPFS payload
# itself is unparseable and no downstream ingestor can recover it. We
# subtract them from the coverage denominator because failing exit 4
# on a payload predict-api can never parse doesn't measure anything
# actionable.
#
# NOT included here on purpose: any transient reason predict-api may
# add later (``ipfs_timeout``, rate-limit, etc.) OR a payload type it
# later learns to parse. A regression that starts mis-tagging
# well-formed deliveries under one of those must NOT be silently
# subtracted from the denominator — that's exactly the coverage-gap
# class the pre-adjustment raw check was catching. If predict-api
# publishes a new structurally-unrecoverable reason, add it here
# explicitly.
_KNOWN_LOSS_REASONS: tuple[str, ...] = (
    "parsed_request_null",
    "parsed_delivery_null",
    "unhandled_type_prompt",
)

# Per-query statement timeout for the predict-api pull. The table is
# ~1.16M rows in prod, small in absolute terms, and the query has
# ``WHERE request_id IS NOT NULL AND error IN (...)`` on it — but a
# missing index, a replica lag spike, or a lock contention burst
# turns the documented "few seconds" into an unbounded hang with
# only K8s ``activeDeadlineSeconds`` to stop it. Bound it here so
# the failure mode surfaces as a psycopg timeout on the K8s Job's
# logs rather than a mysterious pod hang.
_KNOWN_LOSS_STATEMENT_TIMEOUT_MS = 60_000


# --------------------------------------------------------------------------- #
# Data pulls                                                                  #
# --------------------------------------------------------------------------- #


def _pull_mech_predict_rows(
    since_ts: int, until_ts: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Fetch mech-predict's row universe via the real production path.

    Returns a tuple of ``(rows, deliver_to_request_by_platform)`` where the
    second element maps ``platform -> {deliver_id: request_id}``. The
    mapping is populated by a follow-up subgraph query and used by the
    dedup-granularity check.

    :param since_ts: unix seconds, inclusive lower bound.
    :param until_ts: unix seconds, exclusive upper bound.
    :return: rows list plus per-platform deliver_id-to-request_id map.
    """
    # pylint: disable=import-outside-toplevel
    from benchmark.datasets import fetch_production as fp

    all_rows: list[dict[str, Any]] = []
    deliver_to_request: dict[str, dict[str, str]] = {}

    platforms = [
        ("omen", fp.MECH_MARKETPLACE_GNOSIS_URL, fp.fetch_omen_resolved),
        ("polymarket", fp.MECH_MARKETPLACE_POLYGON_URL, fp.fetch_polymarket_resolved),
    ]
    for platform, marketplace_url, fetch_resolved in platforms:
        log.info("pulling mech-predict rows: platform=%s", platform)
        # Delivery time IS bounded to the window here (parity's row set is
        # keyed on delivery, which both sides agree on). Resolution time is
        # NOT bounded: the lake's ``/v1/data/scored-rows`` endpoint only
        # supports filtering on ``requested_at``, not on resolution time.
        # Narrowing the mech-predict side on resolution time while the
        # lake side has no equivalent bound would deflate the overlap
        # near the trailing edge — markets that resolve shortly after
        # ``until`` would stay in the lake but disappear from
        # mech-predict, showing up as ``only_lake`` for reasons unrelated
        # to real migration divergence. Pay the enumeration cost for
        # correctness.
        resolved = fetch_resolved(resolved_after=since_ts)
        rows, _still_pending, _max_delivery_ts, _max_resolved_ts = fp.process_platform(
            platform=platform,
            marketplace_url=marketplace_url,
            resolved_markets=resolved,
            delivery_ts_gt=since_ts,
            existing_ids=set(),
            pending_deliveries=[],
            delivery_ts_lt=until_ts,
        )
        rows = [
            r
            for r in rows
            if (ts := _row_ts(r, "predicted_at")) is not None and ts < until_ts
        ]
        all_rows.extend(rows)

        deliver_ids = [r["deliver_id"] for r in rows]
        deliver_to_request[platform] = _fetch_deliver_to_request(
            marketplace_url, deliver_ids
        )

    for row in all_rows:
        d2r = deliver_to_request.get(row["platform"], {})
        row["request_id"] = d2r.get(row["deliver_id"])

    return all_rows, deliver_to_request


def _fetch_deliver_to_request(
    marketplace_url: str, deliver_ids: list[str]
) -> dict[str, str]:
    """Query the marketplace subgraph for ``{deliver_id: request_id}`` mapping.

    Needed because mech-predict rows key on ``deliver_id`` but the lake
    keys on ``request_id`` — the dedup-granularity divergence class is
    only measurable with this mapping in hand.

    :param marketplace_url: GraphQL endpoint.
    :param deliver_ids: deliver ids to look up.
    :return: dict mapping deliver_id to request_id (both as strings).
    """
    if not deliver_ids:
        return {}
    # pylint: disable=import-outside-toplevel
    from benchmark.datasets.subgraph import post_graphql

    query = """
      query DeliverToRequest($ids: [ID!]!) {
        delivers(where: { id_in: $ids }, first: 1000) {
          id
          request { id }
        }
      }
    """
    result: dict[str, str] = {}
    batch_size = 500
    for i in range(0, len(deliver_ids), batch_size):
        batch = deliver_ids[i : i + batch_size]
        data = post_graphql(
            marketplace_url, {"query": query, "variables": {"ids": batch}}
        )
        for d in data.get("delivers", []):
            req = d.get("request") or {}
            if req.get("id"):
                result[d["id"]] = req["id"]
    return result


def _pull_lake_rows(since: datetime, until: datetime) -> list[dict[str, Any]]:
    """Fetch mech-analytics's rows via ``/v1/data/scored-rows``.

    :param since: timezone-aware datetime, inclusive lower bound.
    :param until: timezone-aware datetime, exclusive upper bound.
    :return: list of scored rows.
    """
    # pylint: disable=import-outside-toplevel
    from benchmark.mech_analytics_client import iter_scored_rows

    # Match production's filter (scorer.rebuild_from_mech_analytics and
    # analyze._build_scores_from_mech_analytics both pass resolved=True).
    # Without it the gate compares against a population the report never
    # sees, and only_lake inflates with unresolved rows.
    log.info(
        "pulling lake rows since=%s until=%s", since.isoformat(), until.isoformat()
    )
    return list(iter_scored_rows(since=since, until=until, resolved=True))


def _pull_marketplace_request_ids(
    since_ts: int, until_ts: int
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Fetch every request delivered in the window, per platform.

    Coverage-check input: the ground truth of delivered mech traffic
    for the window, straight from the marketplace subgraph's
    ``delivers`` collection without any parsedRequest / tool filter.
    Compared against the unfiltered lake ID set below to prove no
    delivered request is missing from the lake. Requests that were
    never delivered — or delivered more than
    ``fp._DELIVERY_LAG_ENVELOPE_S`` after ``until_ts`` — are outside
    this set by construction; the coverage claim is about ingest
    completeness for the mech-predict path, which itself consumes
    deliveries.

    Also threads the per-platform "delivery had no request.id" drop
    count out of ``fetch_all_delivery_request_ids``. A non-zero drop
    means the coverage denominator shrank silently — those requests
    could be genuinely absent from the lake but the check has no way
    to name them. The caller blocks PASS on any non-zero drop.

    :param since_ts: unix seconds, inclusive lower bound.
    :param until_ts: unix seconds, exclusive upper bound.
    :return: tuple of (platform → request_ids set, platform → dropped_no_rid).
    """
    # pylint: disable=import-outside-toplevel
    from benchmark.datasets import fetch_production as fp

    platforms = [
        ("omen", fp.MECH_MARKETPLACE_GNOSIS_URL),
        ("polymarket", fp.MECH_MARKETPLACE_POLYGON_URL),
    ]
    ids_by_platform: dict[str, set[str]] = {}
    dropped_by_platform: dict[str, int] = {}
    for platform, marketplace_url in platforms:
        log.info(
            "pulling marketplace request_ids: platform=%s window=[%s, %s)",
            platform,
            since_ts,
            until_ts,
        )
        ids, dropped = fp.fetch_all_delivery_request_ids(
            marketplace_url,
            request_ts_gt=since_ts,
            request_ts_lt=until_ts,
        )
        ids_by_platform[platform] = ids
        dropped_by_platform[platform] = dropped
    return ids_by_platform, dropped_by_platform


def _pull_lake_request_ids_unfiltered(since: datetime, until: datetime) -> set[str]:
    """Fetch every lake request_id in the window regardless of resolution.

    Coverage-check counterpart to :func:`_pull_marketplace_request_ids`.
    Includes pending rows and off-chain rows, because the coverage
    question is "is this delivered request in the lake at all?", not
    "did mech-analytics finish scoring it?".

    :param since: timezone-aware datetime, inclusive lower bound.
    :param until: timezone-aware datetime, exclusive upper bound.
    :return: set of request_ids the lake has for the window.
    """
    # pylint: disable=import-outside-toplevel
    from benchmark.mech_analytics_client import iter_scored_rows

    log.info(
        "pulling lake request_ids (unfiltered) since=%s until=%s",
        since.isoformat(),
        until.isoformat(),
    )
    return {
        row["request_id"]
        for row in iter_scored_rows(since=since, until=until, resolved=None)
        if row.get("request_id")
    }


def _pull_known_loss_failures(predict_api_url: str) -> dict[str, set[str]]:
    """Fetch the set of request_ids predict-api can never ingest, per platform.

    Predict-api's IPFS ingestor writes a row to ``mech_migration_failures``
    every time it decodes an on-chain Deliver payload and the parsed
    ``request`` or ``delivery`` field comes back null (or the payload type
    is unhandled). These rows are structurally unrecoverable — the IPFS
    payload itself is unparseable, not a transient network issue — so
    mech-analytics's lake correctly never sees them. Without excluding
    this set, the coverage check reports the entire known-loss population
    as "missing from lake" and fails exit 4 for a condition the pipeline
    can never resolve.

    The query is chain-agnostic; the result is partitioned by
    ``_CHAIN_ID_TO_PLATFORM`` so the caller can subtract per-platform.
    Rows on chains not in the map are dropped (with a count logged) —
    a new chain should be added to the map so its failures are excluded
    on the correct platform bucket.

    No time window filter: whether a request_id is in the failures table
    is a permanent property of the payload, not a function of when
    predict-api attempted the fetch. Table size is bounded (~1.16M rows
    in prod as of 2026-08) and the whole thing loads in a few seconds.

    :param predict_api_url: postgres DSN for the predict-api database.
        Read-only access to ``mech_migration_failures`` is sufficient.
    :return: dict mapping platform name to the set of request_ids
        marked as known-loss on that platform's chain.
    :raises RuntimeError: if psycopg is not installed. The caller must
        catch or let it bubble; running without the exclusion is the
        opt-in default (skip when the CLI flag is unset).
    """
    try:
        # pylint: disable=import-outside-toplevel
        import psycopg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - env-specific
        raise RuntimeError(
            "psycopg is required to query mech_migration_failures. "
            "Install with ``uv sync`` (psycopg is a runtime dep of the "
            "mech-predict image) or omit --predict-api-url to skip the "
            "known-loss adjustment."
        ) from exc

    log.info(
        "pulling known-loss failures from predict-api (reasons=%s)",
        ", ".join(_KNOWN_LOSS_REASONS),
    )
    by_platform: dict[str, set[str]] = {
        p: set() for p in _CHAIN_ID_TO_PLATFORM.values()
    }
    unknown_chain_counts: dict[int, int] = {}
    # Setting statement_timeout at connect time via the DSN options
    # applies it to every statement on the session. Only bounding
    # connect_timeout leaves the actual SELECT unbounded — see
    # ``_KNOWN_LOSS_STATEMENT_TIMEOUT_MS`` above for why.
    connect_options = f"-c statement_timeout={_KNOWN_LOSS_STATEMENT_TIMEOUT_MS}"
    with psycopg.connect(
        predict_api_url,
        connect_timeout=30,
        options=connect_options,
    ) as conn:
        with conn.cursor() as cur:
            # The DB column is named ``error`` (per predict-api's schema:
            # ``id, chain_id, request_id, cid, error, attempted_at``). We
            # refer to the values as "reasons" in Python — matching how
            # predict-api's own logs and this repo's docstrings describe
            # them — but the SQL column is ``error``.
            cur.execute(
                "SELECT chain_id, request_id FROM mech_migration_failures "
                "WHERE request_id IS NOT NULL "
                "AND error = ANY(%(reasons)s)",
                {"reasons": list(_KNOWN_LOSS_REASONS)},
            )
            for chain_id, request_id in cur:
                platform = _CHAIN_ID_TO_PLATFORM.get(chain_id)
                if platform is None:
                    unknown_chain_counts[chain_id] = (
                        unknown_chain_counts.get(chain_id, 0) + 1
                    )
                    continue
                by_platform[platform].add(request_id)
    if unknown_chain_counts:
        log.warning(
            "known-loss pull: dropped %d row(s) on unmapped chain(s) %s "
            "(add to _CHAIN_ID_TO_PLATFORM if these should be excluded)",
            sum(unknown_chain_counts.values()),
            # ``chain_id`` may be NULL upstream and NULL vs int is not
            # orderable under Python 3's total ordering. Push None to
            # the end explicitly so a NULL-chain row can't crash the
            # warning path whose entire job is to safely flag it.
            sorted(
                unknown_chain_counts.items(),
                key=lambda kv: (kv[0] is None, kv[0]),
            ),
        )
    for platform, ids in by_platform.items():
        log.info("known-loss failures: platform=%s count=%d", platform, len(ids))
    return by_platform


# --------------------------------------------------------------------------- #
# Divergence-class reports                                                    #
# --------------------------------------------------------------------------- #


def _report_row_set_overlap(
    mp_rows: list[dict[str, Any]], lake_rows: list[dict[str, Any]]
) -> tuple[set[str], set[str], set[str]]:
    """Class 1: row-set overlap on ``request_id``."""
    mp_ids = {r["request_id"] for r in mp_rows if r.get("request_id")}
    lake_ids = {r["request_id"] for r in lake_rows if r.get("request_id")}
    intersection = mp_ids & lake_ids
    only_mp = mp_ids - lake_ids
    only_lake = lake_ids - mp_ids

    _section("1. Row-set overlap (request_id)")
    print(f"  mech-predict rows w/ request_id: {len(mp_ids)}")
    print(f"  lake rows w/ request_id:         {len(lake_ids)}")
    print(f"  intersection:                    {len(intersection)}")
    print(f"  mech-predict only:               {len(only_mp)}")
    print(f"  lake only:                       {len(only_lake)}")
    if only_mp:
        print(f"    example only-in-mp request_ids: {list(only_mp)[:5]}")
    if only_lake:
        print(f"    example only-in-lake request_ids: {list(only_lake)[:5]}")

    unmapped = sum(1 for r in mp_rows if not r.get("request_id"))
    if unmapped:
        print(
            f"  WARNING: {unmapped} mech-predict rows missing request_id "
            "(deliver_id -> request_id lookup did not resolve them)"
        )
    return intersection, only_mp, only_lake


def _report_final_outcome_diff(
    mp_rows: list[dict[str, Any]],
    lake_rows: list[dict[str, Any]],
    overlap: set[str],
) -> tuple[int, int]:
    """Class 2: ``final_outcome`` diff on rows resolved on both sides.

    :param mp_rows: mech-predict rows.
    :param lake_rows: lake / mech-analytics rows.
    :param overlap: request_id set present on both sides.
    :return: ``(resolved_both, outcome_mismatch)`` for gate threshold checks.
    """
    mp_by_id = {r["request_id"]: r for r in mp_rows if r.get("request_id")}
    lake_by_id = {r["request_id"]: r for r in lake_rows if r.get("request_id")}

    resolved_both = 0
    outcome_mismatch = 0
    resolved_mp_only = 0
    resolved_lake_only = 0
    unresolved_both = 0
    mismatch_samples: list[dict[str, Any]] = []

    for rid in overlap:
        mp_o = mp_by_id[rid].get("final_outcome")
        lake_o = lake_by_id[rid].get("final_outcome")
        if mp_o is None and lake_o is None:
            unresolved_both += 1
        elif mp_o is not None and lake_o is None:
            resolved_mp_only += 1
        elif mp_o is None and lake_o is not None:
            resolved_lake_only += 1
        else:
            resolved_both += 1
            if bool(mp_o) != bool(lake_o):
                outcome_mismatch += 1
                if len(mismatch_samples) < 5:
                    mismatch_samples.append(
                        {
                            "request_id": rid,
                            "mp_final_outcome": mp_o,
                            "lake_final_outcome": lake_o,
                            "market_id": mp_by_id[rid].get("market_id"),
                        }
                    )

    _section("2. final_outcome reconciliation on the overlap")
    print(f"  resolved on both sides:      {resolved_both}")
    print(f"    matching:                  {resolved_both - outcome_mismatch}")
    print(f"    mismatched:                {outcome_mismatch}")
    print(f"  resolved on mech-predict only: {resolved_mp_only}")
    print(f"  resolved on lake only:         {resolved_lake_only}")
    print(f"  unresolved on both:            {unresolved_both}")
    if mismatch_samples:
        print("    example mismatches (Reality.eth arbitration or similar):")
        for s in mismatch_samples:
            print(
                f"      request_id={s['request_id'][:16]}... "
                f"mp={s['mp_final_outcome']} lake={s['lake_final_outcome']} "
                f"market={s['market_id']}"
            )
    return resolved_both, outcome_mismatch


def _report_market_id_diff(
    mp_rows: list[dict[str, Any]],
    lake_rows: list[dict[str, Any]],
    overlap: set[str],
) -> None:
    """Class 3: ``market_id`` binding diff on rows in both sets."""
    mp_by_id = {r["request_id"]: r for r in mp_rows if r.get("request_id")}
    lake_by_id = {r["request_id"]: r for r in lake_rows if r.get("request_id")}

    same = 0
    diff = 0
    mp_only_market = 0
    lake_only_market = 0
    neither = 0
    diff_samples: list[dict[str, Any]] = []
    for rid in overlap:
        mp_m = mp_by_id[rid].get("market_id")
        lake_m = lake_by_id[rid].get("market_id")
        if mp_m is None and lake_m is None:
            neither += 1
        elif mp_m is None:
            lake_only_market += 1
        elif lake_m is None:
            mp_only_market += 1
        elif _normalize_market_id(mp_m) == _normalize_market_id(lake_m):
            same += 1
        else:
            diff += 1
            if len(diff_samples) < 5:
                diff_samples.append(
                    {
                        "request_id": rid,
                        "mp_market_id": mp_m,
                        "lake_market_id": lake_m,
                        "question": mp_by_id[rid].get("question_text", "")[:80],
                    }
                )

    _section(
        "3. market_id binding reconciliation on the overlap "
        "(mech-predict uses request_context.market_id then fuzzy prefix; "
        "lake ipfs_historical uses exact-normalized title match)"
    )
    print(f"  both have market_id, same value:      {same}")
    print(f"  both have market_id, different value: {diff}")
    print(f"  mech-predict has market_id, lake doesn't: {mp_only_market}")
    print(f"  lake has market_id, mech-predict doesn't: {lake_only_market}")
    print(f"  neither has market_id:                {neither}")
    if diff_samples:
        print("    example diffs (expected on backfilled tail):")
        for s in diff_samples:
            print(
                f"      request_id={s['request_id'][:16]}... "
                f"mp={s['mp_market_id']} lake={s['lake_market_id']} "
                f'q="{s["question"]}"'
            )


def _report_dedup_granularity(
    mp_rows: list[dict[str, Any]],
    deliver_to_request_by_platform: dict[str, dict[str, str]],
) -> None:
    """Class 4a: multi-deliver-per-request rate on mech-predict side.

    mech-predict rows key on ``(platform, deliver_id)``; the lake keys
    on ``request_id``. Non-1:1 relationships surface as diverging
    per-group denominators when the migrated code aggregates.
    """
    _section("4a. Dedup granularity: multi-deliver-per-request on mech-predict")
    total_delivers = 0
    total_requests = 0
    multi_deliver_requests = 0
    biggest = 0
    for platform, mapping in deliver_to_request_by_platform.items():
        per_request: Counter[str] = Counter()
        for _delid, reqid in mapping.items():
            per_request[reqid] += 1
        multi = {r: c for r, c in per_request.items() if c > 1}
        total_delivers += len(mapping)
        total_requests += len(per_request)
        multi_deliver_requests += len(multi)
        if multi:
            biggest = max(biggest, *multi.values())
        print(f"  {platform}:")
        print(f"    delivers seen:                 {len(mapping)}")
        print(f"    distinct request_ids:          {len(per_request)}")
        print(f"    request_ids with >1 deliver:   {len(multi)}")
        if multi:
            top = sorted(multi.items(), key=lambda kv: -kv[1])[:3]
            for rid, cnt in top:
                print(f"      request_id={rid[:16]}... deliver_count={cnt}")

    unused = sum(1 for r in mp_rows if not r.get("request_id"))
    if unused:
        print(
            f"  NOTE: {unused} mech-predict rows have no request_id mapping "
            "(deliver_id lookup miss); these can't be joined to the lake side."
        )


def _report_null_provenance(lake_rows: list[dict[str, Any]]) -> None:
    """Class 4b: NULL rate on backfill-provenance fields.

    ``market_prob_at_prediction`` / ``market_liquidity_usd`` /
    ``market_spread_at_prediction`` land NULL on rows sourced from
    predict-api's ``ipfs_historical`` backfill because
    ``request_context`` isn't on IPFS. Consumers rendering
    ``by_difficulty`` / ``by_liquidity`` / ``edge`` need this rate to
    scope the parity comparison.
    """
    _section("4b. NULL rate for market_prob / liquidity / spread on lake side")
    n = len(lake_rows)
    if n == 0:
        print("  no lake rows")
        return
    for field in (
        "market_prob_at_prediction",
        "market_liquidity_at_prediction",
        "market_spread_at_prediction",
    ):
        nulls = sum(1 for r in lake_rows if r.get(field) is None)
        pct = 100.0 * nulls / n
        print(f"  {field}: {nulls}/{n} NULL ({pct:.1f}%)")


# --------------------------------------------------------------------------- #
# Aggregate parity on the reconciled row set                                  #
# --------------------------------------------------------------------------- #


def _report_per_tool_aggregate(
    mp_rows: list[dict[str, Any]],
    lake_rows: list[dict[str, Any]],
    overlap: set[str],
) -> None:
    """Per-tool aggregate parity on the overlap.

    Compares brier / log_loss / directional_accuracy per tool on rows
    that (a) exist on both sides and (b) resolve on both sides with the
    same ``final_outcome`` (so the resolution-finality shift, class 1,
    doesn't confound the aggregate).
    """
    _section("5. Per-tool aggregate on the resolved-and-outcome-matching overlap")
    mp_by_id = {r["request_id"]: r for r in mp_rows if r.get("request_id")}
    lake_by_id = {r["request_id"]: r for r in lake_rows if r.get("request_id")}

    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rid in overlap:
        mp_r, lake_r = mp_by_id[rid], lake_by_id[rid]
        mp_o = mp_r.get("final_outcome")
        lake_o = lake_r.get("final_outcome")
        if mp_o is None or lake_o is None:
            continue
        if bool(mp_o) != bool(lake_o):
            continue
        matched.append((mp_r, lake_r))

    print(f"  rows in aggregate: {len(matched)}")
    if not matched:
        print("  no rows to aggregate")
        return

    by_tool_mp: dict[str, list[float]] = defaultdict(list)
    by_tool_lake: dict[str, list[float]] = defaultdict(list)
    dir_acc_mp: dict[str, list[int]] = defaultdict(list)
    dir_acc_lake: dict[str, list[int]] = defaultdict(list)
    for mp_r, lake_r in matched:
        tool = mp_r.get("tool_name") or lake_r.get("tool") or "unknown"
        mp_brier = _mp_row_brier(mp_r)
        lake_brier = lake_r.get("brier")
        if mp_brier is not None:
            by_tool_mp[tool].append(mp_brier)
        if lake_brier is not None:
            by_tool_lake[tool].append(lake_brier)
        mp_dc = _mp_row_dir_correct(mp_r)
        lake_dc = lake_r.get("directional_correct")
        if mp_dc is not None:
            dir_acc_mp[tool].append(1 if mp_dc else 0)
        if lake_dc is not None:
            dir_acc_lake[tool].append(1 if lake_dc else 0)

    tools = sorted(set(by_tool_mp) | set(by_tool_lake))
    print(f"  {'tool':45s}  n_mp  n_lake  brier_mp  brier_lake   Δ    da_mp  da_lake")
    for tool in tools:
        mp_b = by_tool_mp.get(tool, [])
        lake_b = by_tool_lake.get(tool, [])
        mp_d = dir_acc_mp.get(tool, [])
        lake_d = dir_acc_lake.get(tool, [])
        mp_b_mean = statistics.fmean(mp_b) if mp_b else float("nan")
        lake_b_mean = statistics.fmean(lake_b) if lake_b else float("nan")
        delta = mp_b_mean - lake_b_mean if mp_b and lake_b else float("nan")
        mp_d_mean = statistics.fmean(mp_d) if mp_d else float("nan")
        lake_d_mean = statistics.fmean(lake_d) if lake_d else float("nan")
        print(
            f"  {tool[:45]:45s}  {len(mp_b):4d}  {len(lake_b):6d}  "
            f"{mp_b_mean:.4f}    {lake_b_mean:.4f}   {delta:+.4f}  "
            f"{mp_d_mean:.3f}  {lake_d_mean:.3f}"
        )


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #


def _row_ts(row: dict[str, Any], field: str) -> int | None:
    """Parse an ISO timestamp field into unix seconds; None on missing/malformed.

    Callers must handle None explicitly — the previous sentinel of 0
    always passed `< until_ts` comparisons and kept malformed rows.

    :param row: dict row.
    :param field: field name to parse.
    :return: unix seconds, or None if the field is missing / unparseable.
    """
    # pylint: disable=import-outside-toplevel
    from benchmark.mech_analytics_client import _parse_iso

    parsed = _parse_iso(row.get(field))
    if parsed is None:
        return None
    return int(parsed.timestamp())


def _normalize_market_id(m: str) -> str:
    """Lowercase, strip ``0x`` prefix — both sides sometimes preserve casing."""
    if m is None:
        return ""
    s = str(m).lower()
    return s[2:] if s.startswith("0x") else s


def _mp_row_brier(row: dict[str, Any]) -> float | None:
    """Compute Brier locally so we don't require the caller to score.

    :param row: mech-predict row with p_yes + final_outcome.
    :return: Brier score, or None when either field is missing / non-numeric.
    """
    # pylint: disable=import-outside-toplevel
    from benchmark.scoring_primitives import brier_score

    p_yes = row.get("p_yes")
    outcome = row.get("final_outcome")
    if p_yes is None or outcome is None:
        return None
    try:
        return brier_score(float(p_yes), bool(outcome))
    except (TypeError, ValueError):
        return None


def _mp_row_dir_correct(row: dict[str, Any]) -> bool | None:
    """Directional correctness for a mech-predict row."""
    p_yes = row.get("p_yes")
    outcome = row.get("final_outcome")
    if p_yes is None or outcome is None:
        return None
    try:
        return (float(p_yes) >= 0.5) == bool(outcome)
    except (TypeError, ValueError):
        return None


def _section(title: str) -> None:
    print()
    print(f"=== {title} ===")


# --------------------------------------------------------------------------- #
# Coverage check                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _CoverageResult:
    """Structured outcome of the coverage check.

    Groups every fact the verdict decision needs plus everything the
    JSON artifact needs to persist, so ``main`` doesn't juggle five
    parallel dicts and the caller can't accidentally rely on stale
    fields.

    Adjusted-vs-raw distinction: when the caller supplies a known-loss
    set, ``missing_by_platform`` / ``total_missing`` and
    ``marketplace_counts`` / ``total_marketplace`` are the *adjusted*
    values (raw minus known-loss). The raw values are also kept in
    ``raw_missing_by_platform`` / ``raw_marketplace_counts`` and the
    excluded count in ``excluded_by_platform`` / ``total_excluded`` so
    the artifact and human report can show both. The verdict and the
    per-platform floor operate on the adjusted values so the gate
    measures what the pipeline can actually ingest rather than a
    structurally-unrecoverable known-loss population. The floor can
    still trip on the adjusted count if a real subgraph outage drives
    a platform below it — the adjustment only removes rows the
    pipeline could never process, not rows the pipeline should have.
    """

    total_missing: int
    missing_by_platform: dict[str, list[str]]
    total_dropped_no_rid: int
    dropped_by_platform: dict[str, int]
    marketplace_counts: dict[str, int]
    total_marketplace: int
    raw_missing_by_platform: dict[str, list[str]]
    raw_marketplace_counts: dict[str, int]
    excluded_by_platform: dict[str, int]
    total_excluded: int
    known_loss_applied: bool


def _report_coverage_check(
    marketplace_ids_by_platform: dict[str, set[str]],
    dropped_by_platform: dict[str, int],
    lake_ids: set[str],
    min_marketplace_rows: int,
    known_loss_by_platform: dict[str, set[str]] | None = None,
) -> _CoverageResult:
    """Coverage check: prove every delivered request is in the lake.

    For each platform, computes the set of request_ids the marketplace
    subgraph has a ``Deliver`` for in the window but the lake does
    not. Prints a per-platform breakdown and returns a
    :class:`_CoverageResult` with every fact the caller needs to
    compute the verdict and populate the artifact. Requests that were
    never answered — or answered after the widened delivery envelope
    — are not in the marketplace set by construction, so they cannot
    be reported missing here.

    Coverage failure modes the caller must gate on:

    * ``total_missing > 0`` — genuine gap, delivered requests absent
      from the lake.
    * ``total_dropped_no_rid > 0`` — the marketplace pull returned
      deliveries whose ``request.id`` was missing, so those requests
      never entered the coverage denominator and could be silently
      absent from the lake without being reported.
    * per-platform ``marketplace_count < min_marketplace_rows`` (checked
      by caller) — an empty or near-empty coverage sweep on any single
      platform is indistinguishable from complete coverage on the
      set-diff alone; the caller enforces the floor per platform so a
      subgraph outage on one platform can't be masked by a healthy pull
      on the other.

    :param marketplace_ids_by_platform: output of
        :func:`_pull_marketplace_request_ids` (first tuple element).
    :param dropped_by_platform: output of
        :func:`_pull_marketplace_request_ids` (second tuple element) —
        per-platform count of deliveries dropped because their
        ``request.id`` was missing.
    :param lake_ids: output of :func:`_pull_lake_request_ids_unfiltered`.
    :param min_marketplace_rows: floor from ``--min-marketplace-rows``,
        used here purely for the per-platform ``✗`` marker so an
        under-populated platform is called out in the report body even
        though the verdict itself is decided in the caller. Keeping the
        floor in one place (the failure-reason helper) is what
        determines PASS/FAIL; this is a display hint only. The floor is
        compared against the *adjusted* marketplace count so a
        known-loss-dominated platform doesn't fail the vacuous-sweep
        check for a population the pipeline can't ingest.
    :param known_loss_by_platform: per-platform set of request_ids
        predict-api marks as unparseable-upstream in
        ``mech_migration_failures``. When supplied, these IDs are
        removed from both the missing set and the marketplace
        denominator before the verdict is decided; the report shows
        both raw and adjusted numbers so a reviewer can see what got
        excluded. ``None`` (the default) skips the adjustment and the
        verdict is on the raw counts, matching the pre-flag behaviour.
    :return: :class:`_CoverageResult` with per-platform and aggregate stats.
    """
    _section("Coverage check (marketplace subgraph → mech-analytics lake)")
    known_loss = known_loss_by_platform or {}
    known_loss_applied = known_loss_by_platform is not None
    raw_missing_by_platform: dict[str, list[str]] = {}
    missing_by_platform: dict[str, list[str]] = {}
    raw_marketplace_counts: dict[str, int] = {}
    marketplace_counts: dict[str, int] = {}
    excluded_by_platform: dict[str, int] = {}
    total_missing = 0
    total_marketplace = 0
    total_excluded = 0
    total_raw_missing = 0
    total_raw_marketplace = 0
    for platform, marketplace_ids in marketplace_ids_by_platform.items():
        raw_missing_set = marketplace_ids - lake_ids
        platform_known_loss = known_loss.get(platform, set())
        excluded_set = raw_missing_set & platform_known_loss
        # Sanity flag: if predict-api returned a substantial known-loss
        # set for this platform but zero of them intersect the raw
        # missing set, that's near-certainly an ID-format mismatch
        # between the subgraph and predict-api rather than a genuine
        # zero. Without this warning the feature silently no-ops and
        # the run still fails exit 4 with no hint that the adjustment
        # didn't apply. Threshold 100 keeps small windows quiet.
        if (
            known_loss_applied
            and len(platform_known_loss) >= 100
            and len(raw_missing_set) > 0
            and len(excluded_set) == 0
        ):
            log.warning(
                "known-loss no-op on %s: predict-api returned %d IDs but 0 "
                "intersect the %d raw-missing set — likely an ID-format "
                "mismatch between the subgraph and predict-api (case, prefix, "
                "encoding). Adjustment did not apply for this platform.",
                platform,
                len(platform_known_loss),
                len(raw_missing_set),
            )
        adjusted_missing_set = raw_missing_set - excluded_set
        adjusted_marketplace_count = len(marketplace_ids) - len(excluded_set)
        raw_missing_by_platform[platform] = sorted(raw_missing_set)
        missing_by_platform[platform] = sorted(adjusted_missing_set)
        raw_marketplace_counts[platform] = len(marketplace_ids)
        marketplace_counts[platform] = adjusted_marketplace_count
        excluded_by_platform[platform] = len(excluded_set)
        total_missing += len(adjusted_missing_set)
        total_marketplace += adjusted_marketplace_count
        total_excluded += len(excluded_set)
        total_raw_missing += len(raw_missing_set)
        total_raw_marketplace += len(marketplace_ids)
        dropped = dropped_by_platform.get(platform, 0)
        vacuous = adjusted_marketplace_count < min_marketplace_rows
        marker = (
            "✓" if not adjusted_missing_set and dropped == 0 and not vacuous else "✗"
        )
        vacuous_note = f" (below floor {min_marketplace_rows})" if vacuous else ""
        if known_loss_applied and len(excluded_set) > 0:
            print(
                f"  {platform}: marketplace_raw={len(marketplace_ids):>7} "
                f"excluded_known_loss={len(excluded_set):>7}  "
                f"marketplace_adjusted={adjusted_marketplace_count:>7}"
                f"{vacuous_note}  "
                f"missing_from_lake={len(adjusted_missing_set):>5} "
                f"(raw {len(raw_missing_set):>5})  "
                f"dropped_no_request_id={dropped:>4}  {marker}"
            )
        else:
            print(
                f"  {platform}: marketplace={len(marketplace_ids):>7}"
                f"{vacuous_note}  "
                f"missing_from_lake={len(adjusted_missing_set):>5}  "
                f"dropped_no_request_id={dropped:>4}  {marker}"
            )
    total_dropped = sum(dropped_by_platform.values())
    print(f"  lake (all chains, incl. off-chain): {len(lake_ids):>7}")
    if known_loss_applied:
        print(
            f"  excluded as unparseable upstream:   {total_excluded:>7}  "
            f"(known-loss from predict-api mech_migration_failures)"
        )
        print(
            f"  adjusted total marketplace:         {total_marketplace:>7}  "
            f"(raw {total_raw_marketplace} - excluded {total_excluded})"
        )
        print(
            f"  adjusted total missing from lake:   {total_missing:>7}  "
            f"(raw {total_raw_missing} - excluded {total_excluded})  "
            f"{'✓' if total_missing == 0 else '✗'}"
        )
    else:
        print(
            f"  total missing from lake:            {total_missing:>7}  "
            f"{'✓' if total_missing == 0 else '✗'}"
        )
        print(
            "  (raw coverage — --predict-api-url not set, so the verdict is "
            "NOT adjusted for known-loss upstream. See --help.)"
        )
    print(
        f"  total dropped_no_request_id:        {total_dropped:>7}  "
        f"{'✓' if total_dropped == 0 else '✗'}"
    )
    if total_missing:
        print()
        print("  These delivered request_ids are not in the mech-analytics lake.")
        if known_loss_applied:
            print(
                "  Known-loss IDs (parsed_request_null, parsed_delivery_null, "
                "unhandled_type_prompt) have already been excluded. The IDs below "
                "are unexplained; investigate the lake ingest path."
            )
        else:
            print("  Check ``mech_migration_failures`` on the predict-api DB for each")
            print(
                "  ID to see if it's a known-loss backfill failure "
                "(parsed_request_null, unhandled_type_prompt, etc.). Re-run with "
                "--predict-api-url to exclude those automatically."
            )
        print()
        for platform, missing in missing_by_platform.items():
            if not missing:
                continue
            print(f"  {platform} — first 20 of {len(missing)}:")
            for rid in missing[:20]:
                print(f"    {rid}")
            print("    (full list in the .json artifact)")
    if total_dropped:
        print()
        print(
            f"  {total_dropped} marketplace delivery/deliveries had a missing "
            f"``request.id`` and were dropped from the coverage denominator. "
            f"Each dropped row is a request that could genuinely be absent "
            f"from the lake but that the check cannot name. Investigate "
            f"the marketplace subgraph schema."
        )
    return _CoverageResult(
        total_missing=total_missing,
        missing_by_platform=missing_by_platform,
        total_dropped_no_rid=total_dropped,
        dropped_by_platform=dropped_by_platform,
        marketplace_counts=marketplace_counts,
        total_marketplace=total_marketplace,
        raw_missing_by_platform=raw_missing_by_platform,
        raw_marketplace_counts=raw_marketplace_counts,
        excluded_by_platform=excluded_by_platform,
        total_excluded=total_excluded,
        known_loss_applied=known_loss_applied,
    )


# --------------------------------------------------------------------------- #
# Report artifact writers                                                     #
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    """Return the current git commit SHA, or ``unknown`` on failure.

    Reads from the ``GIT_SHA`` env var first, then falls back to
    invoking ``git rev-parse``. The env-var branch is what the shipped
    K8s image uses: the Dockerfile has no ``git`` binary and no
    ``.git`` directory, so the ``git`` branch would always fail there
    and every artifact would attribute the run to ``unknown``. The
    publish workflow bakes ``GITHUB_SHA`` into the image at build
    time via ``--build-arg GIT_SHA=...`` so the artifact can point
    at the exact commit that produced it. Local dev runs use the
    ``git`` branch.

    :return: 40-char SHA hex string, or the literal ``unknown``.
    """
    from_env = os.environ.get("GIT_SHA", "").strip()
    if from_env:
        return from_env
    try:
        return (
            subprocess.check_output(  # nosec B603 B607
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parent,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.warning(
            "_git_sha: neither GIT_SHA env var nor git binary available; "
            "run provenance will be attributed to 'unknown'"
        )
        return "unknown"


def _print_metadata(
    since: datetime, until: datetime, mech_analytics_url: str
) -> dict[str, Any]:
    """Print the run's provenance header and return the same as a dict.

    Every artifact this script writes needs to be dated and attributable
    so a reviewer months later can tell whether a report is still
    current — which mech-analytics version it verified, which script
    commit produced it, when it ran.

    :param since: window lower bound (inclusive, timezone-aware).
    :param until: window upper bound (exclusive, timezone-aware).
    :param mech_analytics_url: the ``MECH_ANALYTICS_URL`` this run hit
        (or ``<unset>`` when the env var was missing).
    :return: the same metadata as a dict, for embedding in the
        ``.json`` artifact.
    """
    metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_git_sha": _git_sha(),
        "mech_analytics_url": mech_analytics_url,
        "window": {
            "since": since.isoformat(),
            "until": until.isoformat(),
        },
    }
    print("=== Run metadata ===")
    for line in (
        f"  run_at (UTC):        {metadata['run_at_utc']}",
        f"  script git SHA:      {metadata['script_git_sha']}",
        f"  mech-analytics URL:  {metadata['mech_analytics_url']}",
        f"  window:              {since.isoformat()} → {until.isoformat()}",
    ):
        print(line)
    return metadata


def _write_report_files(
    output_dir: Path,
    window_slug: str,
    human_report: str,
    json_data: dict[str, Any],
) -> None:
    """Write the human report and structured JSON to ``output_dir``.

    A K8s Job mounts a PVC or emptyDir at ``output_dir``; a reviewer or
    a follow-up step can then read the ``.md`` for the human summary
    and the ``.json`` for full ID lists and machine consumption.

    :param output_dir: directory to write into; created if missing.
    :param window_slug: filename stem, typically ``<since>_to_<until>``.
    :param human_report: full stdout capture from the run.
    :param json_data: structured summary (metadata + coverage + parity).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{window_slug}.md"
    json_path = output_dir / f"{window_slug}.json"
    # Encoding is pinned to UTF-8: the report body contains the Unicode
    # arrow (``→``) in the coverage-check section header, which the
    # Windows default codec (cp1252) can't encode. K8s images run
    # Linux so this only shows up in the CI matrix's Windows job, but
    # the fix is portable and doesn't cost anything on Linux/macOS.
    md_path.write_text(human_report, encoding="utf-8")
    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {md_path}")
    print(f"wrote {json_path}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--days", type=int, default=7, help="window size in days (default 7)"
    )
    p.add_argument("--since", type=str, help="ISO 8601 lower bound (with tz)")
    p.add_argument("--until", type=str, help="ISO 8601 upper bound (with tz)")
    p.add_argument(
        "--min-rows",
        type=int,
        default=MINIMUM_ROWS_PER_SIDE,
        help=f"min rows per side to accept comparison (default {MINIMUM_ROWS_PER_SIDE})",
    )
    p.add_argument(
        "--min-marketplace-rows",
        type=int,
        default=MINIMUM_MARKETPLACE_ROWS,
        help=(
            "min marketplace deliveries per platform for the coverage "
            "check to be non-vacuous. Applied to each platform "
            "independently — an empty coverage sweep on one platform "
            "makes an empty set-diff and would otherwise let a subgraph "
            "outage on that platform hide behind a healthy pull on the "
            f"other. Default {MINIMUM_MARKETPLACE_ROWS}."
        ),
    )
    p.add_argument(
        "--min-overlap-fraction",
        type=float,
        default=0.90,
        help="min |intersection| / min(|mp|, |lake|) to accept parity (default 0.90)",
    )
    p.add_argument(
        "--max-outcome-mismatch-fraction",
        type=float,
        default=0.02,
        help="max outcome_mismatch / resolved_both to accept parity (default 0.02)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "if set, write <window>.md and <window>.json report artifacts to "
            "this directory in addition to printing to stdout. "
            "Intended for K8s Job runs that publish reports to a PVC."
        ),
    )
    p.add_argument(
        "--predict-api-url",
        type=str,
        default=None,
        help=(
            "postgres DSN for predict-api. When set, the coverage check "
            "queries ``mech_migration_failures`` and excludes rows whose "
            "reason is in a hardcoded structurally-unrecoverable set "
            f"({', '.join(_KNOWN_LOSS_REASONS)}) from the "
            "missing-from-lake set BEFORE the verdict is computed. Any "
            "other reason predict-api may record in that table (e.g. a "
            "transient network reason or a payload type it later learns "
            "to parse) is NOT excluded, so a coverage regression that "
            "starts mis-tagging deliveries under a new reason still "
            "fails exit 4. Without this flag the verdict is raw "
            "coverage — every known-loss row inflates the 'missing' "
            "count and can fail exit 4 for a population the pipeline is "
            "structurally unable to ingest. Falls back to the "
            "``PREDICT_API_POSTGRES_URL`` env var."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _compute_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.since or args.until:
        if not (args.since and args.until):
            sys.exit("--since and --until must be provided together")
        since = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
        until = datetime.fromisoformat(args.until.replace("Z", "+00:00"))
    else:
        # Skip the last 24h — mech-analytics's Omen finality gate would
        # otherwise produce a fake count divergence at the trailing edge.
        until = datetime.now(timezone.utc) - timedelta(hours=24)
        since = until - timedelta(days=args.days)
    if since.tzinfo is None or until.tzinfo is None:
        sys.exit("--since / --until must include timezone (Z or +HH:MM)")
    if until <= since:
        sys.exit("--until must be strictly after --since")
    return since, until


class _Tee:
    """Duplicate every ``print`` to the real stdout and a StringIO buffer.

    Lets ``main`` emit its report live to the console (for local runs
    and K8s pod logs) while also capturing the full text for the
    ``--output-dir`` report artifact.
    """

    def __init__(self, buffer: io.StringIO) -> None:
        """Store the buffer and snapshot the original stdout stream.

        :param buffer: in-memory buffer that receives every write.
        """
        self._buffer = buffer
        # ``sys.__stdout__`` is typed ``Optional[TextIO]`` (it's None when
        # the interpreter runs with stdout detached, e.g. under some
        # embedded contexts). This script only runs as an ordinary CLI so
        # the field is always populated. Raise (not assert) so ``-O`` /
        # ``PYTHONOPTIMIZE`` cannot strip the guard and cause a downstream
        # ``AttributeError`` on the first write.
        if sys.__stdout__ is None:
            raise RuntimeError(
                "sys.__stdout__ unavailable; _Tee requires an attached stdout stream"
            )
        self._stdout = sys.__stdout__

    def write(self, s: str) -> int:
        """Write ``s`` to both the original stdout and the capture buffer.

        :param s: the text ``print`` handed us.
        :return: number of characters the buffer accepted (satisfies the
            file-like contract ``print`` relies on).
        """
        self._stdout.write(s)
        return self._buffer.write(s)

    def flush(self) -> None:
        """Flush the underlying stdout so line-buffered pods see output.

        The buffer is in-memory so it has no flush semantics.
        """
        self._stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Run coverage + parity checks and emit a report to stdout (+ file).

    :param argv: optional CLI argument override; defaults to ``sys.argv``.
    :return: exit code — 0 pass; 2 min-row guard trip; 3 parity
        divergence above thresholds; 4 coverage gap
        (delivered requests missing from the lake, or drops in the
        marketplace pull, or a below-floor coverage sweep on any
        single platform).
    """
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    since, until = _compute_window(args)

    # Tee stdout so the human-readable report can also be written to
    # ``--output-dir`` at the end without a second pass.
    report_buffer = io.StringIO()
    sys.stdout = _Tee(report_buffer)  # type: ignore[assignment]
    started_at = time.monotonic()
    # Sentinel: any non-zero exit + non-PASS verdict. Only the explicit
    # success path flips these to 0 / "PASS". If the try body raises
    # before reaching the verdict block, the ``finally`` writes the
    # artifact certifying "ERROR" — matches what the pod's non-zero
    # exit already tells the K8s Job runner, but now the persisted
    # report on the PVC agrees instead of falsely certifying PASS.
    exit_code = 1
    verdict = "ERROR"
    metadata: dict[str, Any] = {}
    coverage_summary: dict[str, Any] = {}
    parity_summary: dict[str, Any] = {}
    try:
        mech_analytics_url = os.environ.get("MECH_ANALYTICS_URL", "<unset>")
        metadata = _print_metadata(since, until, mech_analytics_url)

        since_ts = int(since.timestamp())
        until_ts = int(until.timestamp())

        # --- Coverage check (fast-fail: if the lake is missing on-chain
        # requests, parity on the overlap doesn't rescue that. Return
        # exit 4 *before* the parity pull so a coverage gap can't be
        # miscategorised by the min-row guard.).
        marketplace_ids_by_platform, dropped_by_platform = (
            _pull_marketplace_request_ids(since_ts, until_ts)
        )
        lake_ids_unfiltered = _pull_lake_request_ids_unfiltered(since, until)
        # Optional known-loss adjustment: pull the set of request_ids
        # predict-api's IPFS ingestor has permanently rejected and
        # subtract them from the missing set before the verdict is
        # computed. Off by default so the script still runs standalone
        # without a predict-api DB URL; the caller opts in via the CLI
        # flag or the ``PREDICT_API_POSTGRES_URL`` env var.
        predict_api_url = args.predict_api_url or os.environ.get(
            "PREDICT_API_POSTGRES_URL"
        )
        known_loss_by_platform: dict[str, set[str]] | None = None
        if predict_api_url:
            known_loss_by_platform = _pull_known_loss_failures(predict_api_url)
        else:
            log.warning(
                "known-loss adjustment SKIPPED — --predict-api-url not set. "
                "Coverage verdict will be RAW; a known-loss population "
                "(parsed_request_null etc.) will be counted as missing."
            )
        coverage = _report_coverage_check(
            marketplace_ids_by_platform,
            dropped_by_platform,
            lake_ids_unfiltered,
            args.min_marketplace_rows,
            known_loss_by_platform=known_loss_by_platform,
        )
        coverage_summary = {
            "marketplace_ids_by_platform_count": coverage.marketplace_counts,
            "total_marketplace": coverage.total_marketplace,
            "lake_ids_unfiltered_count": len(lake_ids_unfiltered),
            "missing_from_lake_by_platform": coverage.missing_by_platform,
            "total_missing": coverage.total_missing,
            "dropped_no_request_id_by_platform": coverage.dropped_by_platform,
            "total_dropped_no_request_id": coverage.total_dropped_no_rid,
            "known_loss_applied": coverage.known_loss_applied,
            "excluded_known_loss_by_platform": coverage.excluded_by_platform,
            "total_excluded_known_loss": coverage.total_excluded,
            "raw_marketplace_by_platform_count": coverage.raw_marketplace_counts,
            "raw_missing_from_lake_by_platform_count": {
                p: len(v) for p, v in coverage.raw_missing_by_platform.items()
            },
        }

        # Gate on the coverage side BEFORE hitting parity: a coverage
        # gap or a vacuous sweep or a schema-drift drop is worth its
        # own dedicated exit code, and the min-row guard on the parity
        # pull below would otherwise swallow the same situation as a
        # generic "window too thin".
        coverage_failure = _coverage_failure_reason(coverage, args.min_marketplace_rows)
        if coverage_failure is not None:
            print()
            print(f"FAIL (coverage): {coverage_failure}")
            exit_code = 4
            verdict = "FAIL"
            return exit_code

        # --- Parity check on the resolved intersection.
        mp_rows, deliver_to_request_by_platform = _pull_mech_predict_rows(
            since_ts, until_ts
        )
        lake_rows = _pull_lake_rows(since, until)

        if len(mp_rows) < args.min_rows or len(lake_rows) < args.min_rows:
            print()
            print(
                f"FAIL: min-row guard tripped "
                f"(mech-predict={len(mp_rows)}, lake={len(lake_rows)}, "
                f"threshold={args.min_rows})"
            )
            print(
                "Refusing to report parity on a vacuous window. "
                "Extend the window, check the endpoints, or lower --min-rows."
            )
            exit_code = 2
            verdict = "FAIL"
            return exit_code

        print(f"mech-predict rows: {len(mp_rows)}   lake rows: {len(lake_rows)}")

        intersection, only_mp, only_lake = _report_row_set_overlap(mp_rows, lake_rows)
        resolved_both, outcome_mismatch = _report_final_outcome_diff(
            mp_rows, lake_rows, intersection
        )
        _report_market_id_diff(mp_rows, lake_rows, intersection)
        _report_dedup_granularity(mp_rows, deliver_to_request_by_platform)
        _report_null_provenance(lake_rows)
        _report_per_tool_aggregate(mp_rows, lake_rows, intersection)

        # --- Verdict.
        _section("Verdict")
        mp_ids_size = len(intersection) + len(only_mp)
        lake_ids_size = len(intersection) + len(only_lake)
        overlap_denom = min(mp_ids_size, lake_ids_size)
        overlap_fraction = len(intersection) / overlap_denom if overlap_denom else 0.0
        mismatch_fraction = outcome_mismatch / resolved_both if resolved_both else 0.0
        per_platform_counts = ", ".join(
            f"{platform}={count}"
            for platform, count in sorted(coverage.marketplace_counts.items())
        )
        print(
            f"  coverage: missing_from_lake={coverage.total_missing} "
            f"dropped_no_request_id={coverage.total_dropped_no_rid} "
            f"marketplace_per_platform=[{per_platform_counts}] "
            f"total_marketplace={coverage.total_marketplace} (thresholds "
            f"missing==0, dropped==0, "
            f"per-platform marketplace>={args.min_marketplace_rows})"
        )
        print(
            f"  overlap:  {len(intersection)} / {overlap_denom} = "
            f"{overlap_fraction:.4f}  (threshold >= {args.min_overlap_fraction})"
        )
        print(
            f"  outcome:  {outcome_mismatch} / {resolved_both} = "
            f"{mismatch_fraction:.4f}  (threshold <= {args.max_outcome_mismatch_fraction})"
        )
        parity_summary = {
            "mp_rows": len(mp_rows),
            "lake_rows_resolved": len(lake_rows),
            "intersection": len(intersection),
            "only_mp": len(only_mp),
            "only_lake": len(only_lake),
            "outcome_mismatch": outcome_mismatch,
            "resolved_both": resolved_both,
            "overlap_fraction": overlap_fraction,
            "mismatch_fraction": mismatch_fraction,
        }

        failures: list[str] = []
        if overlap_fraction < args.min_overlap_fraction:
            failures.append(
                f"overlap_fraction {overlap_fraction:.4f} < "
                f"min_overlap_fraction {args.min_overlap_fraction}"
            )
        if mismatch_fraction > args.max_outcome_mismatch_fraction:
            failures.append(
                f"outcome_mismatch_fraction {mismatch_fraction:.4f} > "
                f"max_outcome_mismatch_fraction {args.max_outcome_mismatch_fraction}"
            )
        if failures:
            print()
            print("FAIL (parity): divergence exceeds thresholds:")
            for reason in failures:
                print(f"  - {reason}")
            exit_code = 3
            verdict = "FAIL"
            return exit_code
        print()
        print("PASS: coverage complete, parity within thresholds")
        exit_code = 0
        verdict = "PASS"
        return exit_code
    finally:
        elapsed_s = time.monotonic() - started_at
        print(
            f"\nwall time: {elapsed_s:.1f}s  verdict: {verdict}  exit_code: {exit_code}"
        )
        sys.stdout = sys.__stdout__  # type: ignore[assignment]
        if args.output_dir:
            window_slug = f"{since.date().isoformat()}_to_{until.date().isoformat()}"
            json_data = {
                "metadata": {**metadata, "wall_time_s": round(elapsed_s, 1)},
                "coverage": coverage_summary,
                "parity": parity_summary,
                "verdict": verdict,
                "exit_code": exit_code,
            }
            _write_report_files(
                Path(args.output_dir),
                window_slug,
                report_buffer.getvalue(),
                json_data,
            )


def _coverage_failure_reason(
    coverage: "_CoverageResult", min_marketplace_rows: int
) -> str | None:
    """Return a human-readable failure reason for the coverage check, or None.

    Three ways coverage can fail, in priority order:

    1. Any single platform has ``marketplace_count <
       max(1, min_marketplace_rows)`` — that platform's sweep is
       vacuous. ``marketplace_ids - lake_ids`` is trivially empty when
       a platform's marketplace side is empty, so without this
       per-platform floor a subgraph outage on ONE platform would
       silently PASS as "no missing IDs" (the healthy platform's rows
       would carry the aggregate above the floor). The ``max(1, ...)``
       is an unconditional zero-backstop: an operator lowering
       ``--min-marketplace-rows`` to ``0`` is opting into a weaker
       check, not out of the check — a total outage on a platform
       must still fail closed.
    2. ``total_dropped_no_rid > 0`` — deliveries came back without a
       resolvable ``request.id`` and were dropped from the coverage
       denominator. Each dropped row is a request that could be absent
       from the lake but that we can never name, so a "no missing IDs"
       PASS would be misleading.
    3. ``total_missing > 0`` — a genuine coverage gap.

    :param coverage: the result of :func:`_report_coverage_check`.
        Every count read here (``marketplace_counts``,
        ``total_missing``) is the *adjusted* value when
        ``--predict-api-url`` is set — known-loss IDs have already been
        removed from both the missing set and the marketplace
        denominator, so the vacuous floor and the missing-from-lake gate
        both operate on what's ingestible, not on the raw counts.
    :param min_marketplace_rows: floor from ``--min-marketplace-rows``,
        enforced per platform (not on the aggregate). Effective floor
        is ``max(1, min_marketplace_rows)`` — see item 1.
    :return: human-readable failure reason, or ``None`` on success.
    """
    effective_floor = max(1, min_marketplace_rows)
    vacuous_platforms = sorted(
        (platform, count)
        for platform, count in coverage.marketplace_counts.items()
        if count < effective_floor
    )
    if vacuous_platforms:
        details = ", ".join(
            f"{platform}={count}" for platform, count in vacuous_platforms
        )
        return (
            f"vacuous coverage sweep on {len(vacuous_platforms)} platform(s) "
            f"({details}) — each below effective floor {effective_floor} "
            f"(from --min-marketplace-rows={min_marketplace_rows}, promoted "
            f"to at least 1 so a total outage always fails closed). An "
            f"empty coverage side for even one platform makes "
            f"``marketplace_ids - lake_ids`` trivially empty for that "
            f"platform and would otherwise report PASS. Check the "
            f"marketplace subgraph for the affected platform(s)."
        )
    if coverage.total_dropped_no_rid > 0:
        return (
            f"{coverage.total_dropped_no_rid} marketplace delivery/"
            f"deliveries dropped from the coverage denominator because "
            f"``request.id`` was missing. Each dropped row is a request "
            f"the check can't name — cannot confirm coverage."
        )
    if coverage.total_missing > 0:
        return (
            f"{coverage.total_missing} delivered request_ids are "
            f"missing from the lake."
        )
    return None


if __name__ == "__main__":
    sys.exit(main())
