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
"""
Self-healing backfill for production-log rows with missing tool responses.

Delivery rows are occasionally written to the daily shards before their
response is available upstream (indexer lag or transient subgraph
failures): the fetch reads the prediction from the marketplace subgraph,
so such rows land with ``p_yes=null`` and
``prediction_parse_status="missing_fields"``. The fetch is append-only
(row_id dedup), so those rows are never revisited and the gap would be
permanent. This module re-queries such rows and repairs them in place
once the data exists, making the pipeline self-healing.

Complementary to ``fetch_production.refresh_unparsed_pending``, which
refreshes deliveries that have NOT been written to shards yet (the
pending store); this module repairs rows ALREADY written to shards.

Design:
  1. Scan all shards in the logs directory for rows with
     ``prediction_parse_status == "missing_fields"`` and a ``deliver_id``
     (any platform -- the repair is platform-generic).
  2. Batch-requery each platform's marketplace subgraph for those deliver
     ids, using the query shape the endpoint supports (nested
     ParsedDelivery vs legacy flat fields, probed once per endpoint via
     ``fetch_production.detect_delivers_schema``).
  3. Rows whose response is now available are re-parsed with the same
     parser the fetcher uses; rows that parse to a valid prediction are
     updated in place (p_yes, p_no, parse status, confidence when parsed).
     Rows still missing or still unparseable stay untouched.
  4. Each modified shard is rewritten atomically (tmp file + os.replace).

Idempotent: repaired rows no longer carry ``missing_fields``, so a second
run finds nothing to repair; running while the upstream data is still
missing is a no-op; healthy data is never touched. Exit code is always 0
-- repairing nothing is success -- so the daily flywheel can run it
unconditionally.

Usage:
    python -m benchmark.datasets.backfill_responses
    python -m benchmark.datasets.backfill_responses --logs-dir benchmark/datasets/logs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from benchmark.datasets.fetch_production import (
    DELIVERS_PARSED_BY_IDS_QUERY,
    DELIVERS_SCHEMA_PARSED,
    HTTP_TIMEOUT,
    LOGS_DIR,
    MECH_MARKETPLACE_GNOSIS_URL,
    MECH_MARKETPLACE_POLYGON_URL,
    detect_delivers_schema,
    extract_delivery_fields,
    parse_tool_response,
)
from benchmark.datasets.subgraph import post_graphql

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Batch size for the delivers-by-ids requery. Deliver ids are long hex
# strings, so keep the query body well below subgraph proxy limits.
BACKFILL_BATCH_SIZE = 100

# Only rows written with this parse status are repair candidates: it is
# exactly what fetch_production records when the tool response is
# null/empty. "malformed" and "error" rows carried a real response and
# are not recoverable by requerying.
CANDIDATE_PARSE_STATUS = "missing_fields"

# Platform (as recorded in each row) -> marketplace subgraph endpoint.
# The URLs come from fetch_production and therefore respect the same
# env-var overrides (MECH_MARKETPLACE_GNOSIS_URL / _POLYGON_URL).
PLATFORM_MARKETPLACE_URLS: dict[str, str] = {
    "omen": MECH_MARKETPLACE_GNOSIS_URL,
    "polymarket": MECH_MARKETPLACE_POLYGON_URL,
}

# ---------------------------------------------------------------------------
# GraphQL query
# ---------------------------------------------------------------------------

# Legacy-schema counterpart of fetch_production.DELIVERS_PARSED_BY_IDS_QUERY:
# re-read the flat Deliver fields for specific deliveries on endpoints that
# do not expose the nested ParsedDelivery entity. The field shape mirrors
# fetch_production.DELIVERS_QUERY_LEGACY so the result feeds
# extract_delivery_fields unchanged.
DELIVERS_LEGACY_BY_IDS_QUERY = """
{
  delivers(
    first: %(first)s
    where: { id_in: [%(ids)s] }
  ) {
    id
    model
    toolResponse
  }
}
"""

# ---------------------------------------------------------------------------
# Subgraph helpers
# ---------------------------------------------------------------------------


def _post_graphql(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Post a GraphQL query and return the JSON response data.

    Thin wrapper over the shared :func:`benchmark.datasets.subgraph.post_graphql`
    retry helper, kept as a module-level seam for tests.

    :param url: subgraph endpoint URL.
    :param payload: GraphQL request body (``{"query": ...}``).
    :return: the ``data`` object from the GraphQL response.
    """
    return post_graphql(url, payload, timeout=HTTP_TIMEOUT)


def fetch_tool_responses(
    marketplace_url: str,
    deliver_ids: list[str],
    batch_size: int = BACKFILL_BATCH_SIZE,
) -> dict[str, Optional[str]]:
    """Requery the marketplace subgraph for tool responses by deliver id.

    The endpoint's delivers shape (nested ParsedDelivery vs legacy flat
    fields) is probed once via :func:`detect_delivers_schema`, and every
    returned deliver is mapped through :func:`extract_delivery_fields`, so
    a future schema change only needs handling in ``fetch_production``.
    Deliveries whose parsed payload has not been indexed upstream yet are
    omitted from the result.

    Queries in batches. A failed probe or batch is logged and skipped (its
    rows simply stay unrepaired until the next run) so a subgraph failure
    can never fail the backfill.

    :param marketplace_url: subgraph endpoint URL.
    :param deliver_ids: deliver ids to look up.
    :param batch_size: max ids per GraphQL request.
    :return: mapping of deliver_id to tool response (possibly None).
    """
    responses: dict[str, Optional[str]] = {}
    try:
        schema = detect_delivers_schema(marketplace_url)
    except Exception as e:  # pylint: disable=broad-except
        log.warning(
            "Delivers schema probe failed against %s: %s -- skipping",
            marketplace_url,
            e,
        )
        return responses
    query_template = (
        DELIVERS_PARSED_BY_IDS_QUERY
        if schema == DELIVERS_SCHEMA_PARSED
        else DELIVERS_LEGACY_BY_IDS_QUERY
    )
    for i in range(0, len(deliver_ids), batch_size):
        batch = deliver_ids[i : i + batch_size]
        ids_str = ", ".join(f'"{did}"' for did in batch)
        query = query_template % {"first": len(batch), "ids": ids_str}
        try:
            data = _post_graphql(marketplace_url, {"query": query})
        except Exception as e:  # pylint: disable=broad-except
            log.warning(
                "Backfill batch of %d ids failed against %s: %s",
                len(batch),
                marketplace_url,
                e,
            )
            continue
        for deliver in data.get("delivers", []):
            fields = extract_delivery_fields(deliver, schema)
            if fields["parsed_missing"]:
                # Parsed payload not indexed upstream yet -- leave the row
                # a candidate for a future run.
                continue
            responses[deliver["id"]] = fields["tool_response"]
    return responses


# ---------------------------------------------------------------------------
# Shard scanning & repair
# ---------------------------------------------------------------------------


def _load_shard(path: Path) -> list[dict[str, Any]]:
    """Load all rows from one JSONL shard, skipping blank lines.

    :param path: path to the shard file.
    :return: list of row dicts in file order.
    """
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_candidate(row: dict[str, Any]) -> bool:
    """Return whether a row is a repair candidate.

    :param row: production log row.
    :return: True when the row lost its response and carries a deliver_id.
    """
    return bool(
        row.get("prediction_parse_status") == CANDIDATE_PARSE_STATUS
        and row.get("deliver_id")
    )


def repair_row(row: dict[str, Any], tool_response: Optional[str]) -> bool:
    """Re-parse a requeried toolResponse and repair the row in place.

    Only a response that parses to a *valid* prediction mutates the row;
    still-null or still-unparseable responses leave it untouched, so the
    row stays a candidate for future runs.

    :param row: production log row (mutated in place on success).
    :param tool_response: requeried toolResponse, possibly still None.
    :return: True when the row was repaired.
    """
    if not tool_response:
        return False
    parsed = parse_tool_response(tool_response)
    if parsed["prediction_parse_status"] != "valid":
        return False
    row["p_yes"] = parsed["p_yes"]
    row["p_no"] = parsed["p_no"]
    row["prediction_parse_status"] = parsed["prediction_parse_status"]
    if parsed["confidence"] is not None:
        row["confidence"] = parsed["confidence"]
    return True


def _rewrite_shard_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically rewrite a shard: temp file + os.replace.

    Mirrors the atomic-rewrite pattern used by
    ``score_tournament._update_predictions_file`` so a crash mid-write
    can never corrupt a shard.

    :param path: shard path to replace.
    :param rows: full row list to write, in order.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp:
            for row in rows:
                tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, str(path))
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def backfill(
    logs_dir: Path,
    batch_size: int = BACKFILL_BATCH_SIZE,
) -> dict[str, Any]:
    """Scan all shards, requery lost responses, and repair rows in place.

    :param logs_dir: directory containing the daily JSONL shards.
    :param batch_size: max deliver ids per GraphQL request.
    :return: summary dict with ``shards_scanned``, ``candidates``,
        ``repaired``, and a per-platform breakdown.
    """
    summary: dict[str, Any] = {
        "shards_scanned": 0,
        "candidates": 0,
        "repaired": 0,
        "platforms": {},
    }
    if not logs_dir.is_dir():
        log.info("Logs dir %s does not exist -- nothing to backfill", logs_dir)
        return summary

    # Pass 1: scan shards, collect candidate deliver ids per platform.
    shards: list[tuple[Path, list[dict[str, Any]]]] = []
    ids_by_platform: dict[str, list[str]] = {}
    for path in sorted(logs_dir.glob("*.jsonl")):
        rows = _load_shard(path)
        shards.append((path, rows))
        summary["shards_scanned"] += 1
        for row in rows:
            if not _is_candidate(row):
                continue
            summary["candidates"] += 1
            platform = row.get("platform") or "unknown"
            plat_stats = summary["platforms"].setdefault(
                platform, {"candidates": 0, "repaired": 0}
            )
            plat_stats["candidates"] += 1
            if platform in PLATFORM_MARKETPLACE_URLS:
                ids_by_platform.setdefault(platform, []).append(row["deliver_id"])

    unknown = set(summary["platforms"]) - set(PLATFORM_MARKETPLACE_URLS)
    if unknown:
        log.warning("No marketplace URL for platform(s) %s -- skipped", sorted(unknown))

    # Pass 2: batch-requery each platform's marketplace subgraph.
    responses: dict[str, dict[str, Optional[str]]] = {}
    for platform, deliver_ids in ids_by_platform.items():
        unique_ids = sorted(set(deliver_ids))
        log.info(
            "%s: requerying %d deliver ids in batches of %d",
            platform,
            len(unique_ids),
            batch_size,
        )
        responses[platform] = fetch_tool_responses(
            PLATFORM_MARKETPLACE_URLS[platform], unique_ids, batch_size
        )

    # Pass 3: repair rows in place; rewrite only shards that changed.
    for path, rows in shards:
        shard_repaired = 0
        for row in rows:
            if not _is_candidate(row):
                continue
            platform = row.get("platform") or "unknown"
            tool_response = responses.get(platform, {}).get(row["deliver_id"])
            if repair_row(row, tool_response):
                shard_repaired += 1
                summary["platforms"][platform]["repaired"] += 1
        if shard_repaired:
            _rewrite_shard_atomic(path, rows)
            summary["repaired"] += shard_repaired
            log.info("Repaired %d rows in %s", shard_repaired, path.name)

    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _log_summary(summary: dict[str, Any]) -> None:
    """Log the backfill summary.

    :param summary: summary dict from :func:`backfill`.
    """
    log.info(
        "Backfill summary: %d shards scanned, %d candidate rows, %d repaired",
        summary["shards_scanned"],
        summary["candidates"],
        summary["repaired"],
    )
    for platform, stats in sorted(summary["platforms"].items()):
        log.info(
            "  %s: %d candidates, %d repaired",
            platform,
            stats["candidates"],
            stats["repaired"],
        )


def _emit_repaired_output(repaired: int) -> None:
    """Emit the repaired count for workflow consumption.

    Prints a ``repaired=<n>`` line to stdout and appends the same line to
    ``$GITHUB_OUTPUT`` when that env var is set, so the flywheel step can
    gate the force-rebuild path on it.

    :param repaired: number of rows repaired this run.
    """
    line = f"repaired={repaired}"
    print(line)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the backfill.

    :return: configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Repair production-log rows whose toolResponse was lost.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help="Directory containing daily production log shards",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BACKFILL_BATCH_SIZE,
        help="Max deliver ids per subgraph request",
    )
    return parser


def main() -> None:
    """CLI entry point. Always exits 0: repairing nothing is success."""
    args = _build_arg_parser().parse_args()
    repaired = 0
    try:
        summary = backfill(args.logs_dir, args.batch_size)
        repaired = summary["repaired"]
        _log_summary(summary)
    except Exception:
        log.exception("Backfill failed (non-fatal: the flywheel must not block)")
    _emit_repaired_output(repaired)


if __name__ == "__main__":
    main()
