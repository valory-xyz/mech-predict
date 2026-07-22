#!/usr/bin/env python3
r"""Reconcile mech-predict's row universe against mech-analytics.

Runs the four divergence-class checks documented in
``mech-analytics/docs/testing_mech_predict.md`` for a chosen window,
plus a per-tool aggregate parity comparison on the reconciled row set.

Left side: the actual production path in mech-predict — marketplace
subgraph + IPFS via ``benchmark.datasets.fetch_production``. Both
platforms (Omen on Gnosis, Polymarket on Polygon), multi-day, comparing
field values not id membership.

Right side: mech-analytics's ``/v1/data/scored-rows`` endpoint via
``benchmark.mech_analytics_client``.

Reports:
    1. Row-set overlap on ``request_id``: intersection, mech-predict
       only, lake only.
    2. ``final_outcome`` diff on rows resolved on both sides.
    3. ``market_id`` diff on rows present in both sets.
    4. Multi-deliver-per-request rate on mech-predict + NULL rate
       for ``market_prob_at_prediction`` / ``market_liquidity_usd`` /
       ``market_spread_at_prediction`` on the lake side.
    5. Per-tool aggregate on the overlap (brier, log_loss,
       directional_accuracy).

Fails loud if either side returns under ``MINIMUM_ROWS_PER_SIDE`` so a
vacuous match on empty data doesn't get reported as parity.

Usage:
    scripts/verify_migration_swap.py --days 7
    scripts/verify_migration_swap.py --since 2026-07-01T00:00:00Z \\
                                     --until 2026-07-08T00:00:00Z

Required env vars:
    MECH_ANALYTICS_URL             (lake-side endpoint)
    MECH_MARKETPLACE_GNOSIS_URL    (has default in fetch_production)
    MECH_MARKETPLACE_POLYGON_URL   (has default in fetch_production)
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
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
        resolved = fetch_resolved(resolved_after=since_ts)
        rows, _still_pending, _max_delivery_ts, _max_resolved_ts = fp.process_platform(
            platform=platform,
            marketplace_url=marketplace_url,
            resolved_markets=resolved,
            delivery_ts_gt=since_ts,
            existing_ids=set(),
            pending_deliveries=[],
        )
        rows = [r for r in rows if _row_ts(r, "predicted_at") < until_ts]
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

    log.info(
        "pulling lake rows since=%s until=%s", since.isoformat(), until.isoformat()
    )
    return list(iter_scored_rows(since=since, until=until))


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
) -> None:
    """Class 2: ``final_outcome`` diff on rows resolved on both sides."""
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
        lake_o = _lake_outcome_to_bool(lake_by_id[rid].get("resolved_outcome"))
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
                            "lake_resolved_outcome": lake_by_id[rid].get(
                                "resolved_outcome"
                            ),
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
                f"mp={s['mp_final_outcome']} lake={s['lake_resolved_outcome']} "
                f"market={s['market_id']}"
            )


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
        "market_liquidity_usd",
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
        lake_o = _lake_outcome_to_bool(lake_r.get("resolved_outcome"))
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


def _row_ts(row: dict[str, Any], field: str) -> int:
    """Parse an ISO timestamp field from a row into unix seconds.

    Returns 0 on missing/malformed so the callsite's ``< until_ts``
    filter drops the row without raising.
    """
    val = row.get(field)
    if not val:
        return 0
    try:
        return int(datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp())
    except (TypeError, ValueError):
        return 0


def _lake_outcome_to_bool(v: Any) -> bool | None:
    if v is None:
        return None
    try:
        return bool(int(round(float(v))))
    except (TypeError, ValueError):
        return None


def _normalize_market_id(m: str) -> str:
    """Lowercase, strip ``0x`` prefix — both sides sometimes preserve casing."""
    if m is None:
        return ""
    s = str(m).lower()
    return s[2:] if s.startswith("0x") else s


def _mp_row_brier(row: dict[str, Any]) -> float | None:
    """Compute Brier locally so we don't require the caller to score."""
    p_yes = row.get("p_yes")
    outcome = row.get("final_outcome")
    if p_yes is None or outcome is None:
        return None
    try:
        return (float(p_yes) - (1.0 if outcome else 0.0)) ** 2
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run all reconciliation checks and print the report to stdout.

    :param argv: optional CLI argument override; defaults to ``sys.argv``.
    :return: 0 on success, 2 if the min-row guard tripped.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    since, until = _compute_window(args)
    print(f"window: {since.isoformat()} to {until.isoformat()} (excl. trailing 24h)")

    since_ts = int(since.timestamp())
    until_ts = int(until.timestamp())

    mp_rows, deliver_to_request_by_platform = _pull_mech_predict_rows(
        since_ts, until_ts
    )
    lake_rows = _pull_lake_rows(since, until)

    # Min-row guard: fail loud on either side coming back thin.
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
        return 2

    print(f"mech-predict rows: {len(mp_rows)}   " f"lake rows: {len(lake_rows)}")

    intersection, _only_mp, _only_lake = _report_row_set_overlap(mp_rows, lake_rows)
    _report_final_outcome_diff(mp_rows, lake_rows, intersection)
    _report_market_id_diff(mp_rows, lake_rows, intersection)
    _report_dedup_granularity(mp_rows, deliver_to_request_by_platform)
    _report_null_provenance(lake_rows)
    _report_per_tool_aggregate(mp_rows, lake_rows, intersection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
