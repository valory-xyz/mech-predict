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

* **Coverage** — every on-chain request_id the marketplace subgraph
  knows about must be present in the mech-analytics lake. This proves
  ingest completeness ahead of the mech-predict cutover: switching to
  the lake cannot silently drop rows that today's production path can
  see. Fails with exit code 4 if any on-chain request is missing.
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
    4 — coverage gap (at least one on-chain request missing from lake)

Required env vars:
    MECH_ANALYTICS_URL             (lake-side endpoint)
    MECH_MARKETPLACE_GNOSIS_URL    (has default in fetch_production)
    MECH_MARKETPLACE_POLYGON_URL   (has default in fetch_production)
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger("verify_migration_swap")


# --------------------------------------------------------------------------- #
# Guards / knobs                                                              #
# --------------------------------------------------------------------------- #

# Below this row count on either side the comparison is vacuous.
MINIMUM_ROWS_PER_SIDE = 100

# Comparing floats coming from independent code paths — allow small drift.
FLOAT_TOL = 1e-6


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
        # Bound both queries to the window so enumeration doesn't fan out from
        # `since` to `now` for old windows (the source of hours-long runs
        # against 6-month-old windows).
        resolved = fetch_resolved(resolved_after=since_ts, resolved_before=until_ts)
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


def _pull_marketplace_request_ids(since_ts: int, until_ts: int) -> dict[str, set[str]]:
    """Fetch every on-chain request_id delivered in the window, per platform.

    Coverage-check input: the ground truth of on-chain mech traffic
    for the window, straight from the marketplace subgraph without any
    parsedRequest / tool filter. Compared against the unfiltered lake
    ID set below to prove no on-chain request is missing from the
    lake.

    :param since_ts: unix seconds, inclusive lower bound.
    :param until_ts: unix seconds, exclusive upper bound.
    :return: platform name → set of on-chain request_ids for that platform.
    """
    # pylint: disable=import-outside-toplevel
    from benchmark.datasets import fetch_production as fp

    platforms = [
        ("omen", fp.MECH_MARKETPLACE_GNOSIS_URL),
        ("polymarket", fp.MECH_MARKETPLACE_POLYGON_URL),
    ]
    ids_by_platform: dict[str, set[str]] = {}
    for platform, marketplace_url in platforms:
        log.info(
            "pulling marketplace request_ids: platform=%s window=[%s, %s)",
            platform,
            since_ts,
            until_ts,
        )
        ids_by_platform[platform] = fp.fetch_all_delivery_request_ids(
            marketplace_url, since_ts, timestamp_lt=until_ts
        )
    return ids_by_platform


def _pull_lake_request_ids_unfiltered(since: datetime, until: datetime) -> set[str]:
    """Fetch every lake request_id in the window regardless of resolution.

    Coverage-check counterpart to :func:`_pull_marketplace_request_ids`.
    Includes pending rows and off-chain rows, because the coverage
    question is "is this on-chain request in the lake at all?", not
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


def _report_coverage_check(
    marketplace_ids_by_platform: dict[str, set[str]],
    lake_ids: set[str],
) -> tuple[int, dict[str, list[str]]]:
    """Coverage check: prove every on-chain request is in the lake.

    For each platform, computes the set of request_ids the marketplace
    subgraph knows about but the lake does not. Prints a per-platform
    breakdown and returns the total gap size plus the per-platform ID
    lists so the caller can put the full lists in the JSON artifact
    and act on the gate.

    :param marketplace_ids_by_platform: output of
        :func:`_pull_marketplace_request_ids`.
    :param lake_ids: output of :func:`_pull_lake_request_ids_unfiltered`.
    :return: tuple of (total_missing_count, per-platform missing lists).
    """
    _section("Coverage check (marketplace subgraph → mech-analytics lake)")
    missing_by_platform: dict[str, list[str]] = {}
    total_missing = 0
    for platform, marketplace_ids in marketplace_ids_by_platform.items():
        missing = sorted(marketplace_ids - lake_ids)
        missing_by_platform[platform] = missing
        total_missing += len(missing)
        marker = "✓" if not missing else "✗"
        print(
            f"  {platform}: marketplace={len(marketplace_ids):>7}  "
            f"missing_from_lake={len(missing):>5}  {marker}"
        )
    print(f"  lake (all chains, incl. off-chain): {len(lake_ids):>7}")
    print(
        f"  total missing from lake:            {total_missing:>7}  "
        f"{'✓' if total_missing == 0 else '✗'}"
    )
    if total_missing:
        print()
        print("  These on-chain request_ids are not in the mech-analytics lake.")
        print("  Check ``mech_migration_failures`` on the predict-api DB for each")
        print("  ID to see if it's a known-loss backfill failure")
        print("  (parsed_request_null, unhandled_type_prompt, etc.).")
        print()
        for platform, missing in missing_by_platform.items():
            if not missing:
                continue
            print(f"  {platform} — first 20 of {len(missing)}:")
            for rid in missing[:20]:
                print(f"    {rid}")
            print("    (full list in the .json artifact)")
    return total_missing, missing_by_platform


# --------------------------------------------------------------------------- #
# Report artifact writers                                                     #
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    """Return the current git commit SHA, or ``unknown`` if git isn't
    available (e.g. running inside a stripped-down container).
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parent,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _print_metadata(
    since: datetime, until: datetime, mech_analytics_url: str
) -> dict[str, Any]:
    """Print the run's provenance header and return the same fields as a
    dict for the JSON artifact.

    Every artifact this script writes needs to be dated and attributable
    so a reviewer months later can tell whether a report is still
    current — which mech-analytics version it verified, which script
    commit produced it, when it ran.
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
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{window_slug}.md"
    json_path = output_dir / f"{window_slug}.json"
    md_path.write_text(human_report)
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
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
        self._buffer = buffer
        self._stdout = sys.__stdout__

    def write(self, s: str) -> int:
        self._stdout.write(s)
        return self._buffer.write(s)

    def flush(self) -> None:
        self._stdout.flush()


def main(
    argv: Sequence[str] | None = None,
) -> int:  # pylint: disable=too-many-locals,too-many-statements
    """Run coverage + parity checks and emit a report to stdout (+ file).

    :param argv: optional CLI argument override; defaults to ``sys.argv``.
    :return: exit code — 0 pass; 2 min-row guard trip; 3 parity
        divergence above thresholds; 4 coverage gap
        (on-chain requests missing from the lake).
    """
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
    exit_code = 0
    metadata: dict[str, Any] = {}
    coverage_summary: dict[str, Any] = {}
    parity_summary: dict[str, Any] = {}
    try:
        mech_analytics_url = os.environ.get("MECH_ANALYTICS_URL", "<unset>")
        metadata = _print_metadata(since, until, mech_analytics_url)

        since_ts = int(since.timestamp())
        until_ts = int(until.timestamp())

        # --- Coverage check (fast-fail: if the lake is missing on-chain
        # requests, parity on the overlap doesn't rescue that).
        marketplace_ids_by_platform = _pull_marketplace_request_ids(since_ts, until_ts)
        lake_ids_unfiltered = _pull_lake_request_ids_unfiltered(since, until)
        total_missing, missing_by_platform = _report_coverage_check(
            marketplace_ids_by_platform, lake_ids_unfiltered
        )
        coverage_summary = {
            "marketplace_ids_by_platform_count": {
                p: len(v) for p, v in marketplace_ids_by_platform.items()
            },
            "lake_ids_unfiltered_count": len(lake_ids_unfiltered),
            "missing_from_lake_by_platform": missing_by_platform,
            "total_missing": total_missing,
        }

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
        print(f"  coverage: missing_from_lake={total_missing} " f"(threshold == 0)")
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

        # Coverage failure takes precedence — an incomplete lake means
        # even a perfect parity number is describing an incomplete
        # picture. Reviewer needs to know the gap first.
        if total_missing > 0:
            print()
            print(
                f"FAIL (coverage): {total_missing} on-chain request_ids "
                f"are missing from the lake."
            )
            exit_code = 4
            return exit_code

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
            return exit_code
        print()
        print("PASS: coverage complete, parity within thresholds")
        return exit_code
    finally:
        elapsed_s = time.monotonic() - started_at
        print(f"\nwall time: {elapsed_s:.1f}s")
        sys.stdout = sys.__stdout__  # type: ignore[assignment]
        if args.output_dir:
            window_slug = f"{since.date().isoformat()}_to_{until.date().isoformat()}"
            json_data = {
                "metadata": {**metadata, "wall_time_s": round(elapsed_s, 1)},
                "coverage": coverage_summary,
                "parity": parity_summary,
                "verdict": "PASS" if exit_code == 0 else "FAIL",
                "exit_code": exit_code,
            }
            _write_report_files(
                Path(args.output_dir),
                window_slug,
                report_buffer.getvalue(),
                json_data,
            )


if __name__ == "__main__":
    sys.exit(main())
