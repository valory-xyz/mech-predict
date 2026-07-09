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
  1. Scan all shards in the logs directory (bounded by
     ``--max-shard-age-days`` so legacy/unhealable shards are not
     re-globbed forever as ``logs/`` grows) for rows with
     ``prediction_parse_status == "missing_fields"`` and a ``deliver_id``
     (any platform -- the repair is platform-generic). Malformed JSONL
     lines are skipped with a warning (a bad line -- exactly the failure
     class this module heals -- must never abort the scan) and an
     unreadable shard is isolated so it cannot sink the run.
  2. Batch-requery each platform's marketplace subgraph for those deliver
     ids, using the query shape the endpoint supports (nested
     ParsedDelivery vs legacy flat fields, probed once per endpoint via
     ``fetch_production.detect_delivers_schema``).
  3. Rows whose response is now available are re-parsed with the same
     parser the fetcher uses; rows that parse to a valid prediction are
     updated in place (p_yes, p_no, parse status, confidence when parsed,
     plus model/tool_version/config_hash when the row lacked them, so a
     repaired row buckets correctly rather than under "unknown"). Rows
     still missing or still unparseable stay untouched.
  4. Each modified shard is rewritten atomically (tmp file + os.replace),
     each in its own try/except so a rewrite failure on one shard cannot
     discard the repairs already persisted to earlier shards; the
     ``repaired`` count is accumulated as each shard succeeds.

Failure isolation: ``backfill`` never raises -- a failed schema probe,
subgraph batch, malformed line, unreadable shard, or shard rewrite is
logged and skipped, and the partial summary (with the count of repairs
already persisted) is RETURNED, not unwound. When any subgraph batch/probe
or shard rewrite failed, the summary carries ``backfill_error=True`` so the
caller (and CI) can distinguish "nothing to repair" from "crashed partway"
and force a rebuild of the already-repaired rows.

Idempotent: repaired rows no longer carry ``missing_fields``, so a second
run finds nothing to repair; running while the upstream data is still
missing is a no-op; healthy data is never touched. Exit code: backfill
failures (subgraph/parsing/IO errors) never raise; argument-parse errors
and a fatal ``$GITHUB_OUTPUT`` write are the only non-zero exits -- so the
daily flywheel can run it effectively unconditionally.

Usage:
    python -m benchmark.datasets.backfill_responses
    python -m benchmark.datasets.backfill_responses --logs-dir benchmark/datasets/logs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from benchmark.datasets.fetch_production import (
    DELIVERS_PARSED_BY_IDS_QUERY,
    DELIVERS_SCHEMA_PARSED,
    HTTP_TIMEOUT,
    LOGS_DIR,
    MECH_MARKETPLACE_GNOSIS_URL,
    MECH_MARKETPLACE_POLYGON_URL,
    _compute_config_hash,
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
# exactly what fetch_production.parse_tool_response records when the tool
# response is null/empty. "malformed" and "error" rows carried a real
# response and are not recoverable by requerying.
# WARNING: this bare literal must stay in lockstep with the status string
# that fetch_production.parse_tool_response emits for empty/null input; a
# rename there would silently zero candidates forever. The regression test
# ``test_candidate_parse_status_matches_fetch_production`` pins that
# contract so a future rename breaks a test here rather than in silence.
# (A shared Literal/StrEnum in fetch_production is the cleaner long-term
# fix but belongs to a base-branch/main change, not this PR.)
CANDIDATE_PARSE_STATUS = "missing_fields"

# Default lookback for the shard scan: shards whose filename date is older
# than this are skipped so legacy/unhealable rows are not re-globbed and
# re-queried forever as logs/ grows. Generous enough to cover the incident
# window that motivated this module while still bounding the daily scan.
DEFAULT_MAX_SHARD_AGE_DAYS = 120

# Matches the daily shard filename so the scan can derive each shard's date
# for the age bound: production_log_YYYY_MM_DD.jsonl.
SHARD_DATE_RE = re.compile(r"production_log_(\d{4})_(\d{2})_(\d{2})\.jsonl$")

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
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Requery the marketplace subgraph for tool responses by deliver id.

    The endpoint's delivers shape (nested ParsedDelivery vs legacy flat
    fields) is probed once via :func:`detect_delivers_schema`, and every
    returned deliver is mapped through :func:`extract_delivery_fields`, so
    a future schema change only needs handling in ``fetch_production``.
    Deliveries whose parsed payload has not been indexed upstream yet are
    omitted from the result.

    Queries in batches. A failed probe or batch is logged and skipped (its
    rows simply stay unrepaired until the next run) so a subgraph failure
    can never fail the backfill; the boolean second return value flags that
    such a failure happened so the caller can force a rebuild.

    :param marketplace_url: subgraph endpoint URL.
    :param deliver_ids: deliver ids to look up.
    :param batch_size: max ids per GraphQL request.
    :return: ``(responses, had_error)`` where responses maps deliver_id to
        the :func:`extract_delivery_fields` dict (model, tool_response,
        tool_hash) and had_error is True if any probe/batch failed.
    """
    responses: dict[str, dict[str, Any]] = {}
    try:
        schema = detect_delivers_schema(marketplace_url)
    except Exception as e:  # pylint: disable=broad-except
        log.warning(
            "Delivers schema probe failed against %s: %s -- skipping",
            marketplace_url,
            e,
        )
        return responses, True
    query_template = (
        DELIVERS_PARSED_BY_IDS_QUERY
        if schema == DELIVERS_SCHEMA_PARSED
        else DELIVERS_LEGACY_BY_IDS_QUERY
    )
    had_error = False
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
            had_error = True
            continue
        for deliver in data.get("delivers", []):
            fields = extract_delivery_fields(deliver, schema)
            if fields["parsed_missing"]:
                # Parsed payload not indexed upstream yet -- leave the row
                # a candidate for a future run.
                continue
            responses[deliver["id"]] = fields
    return responses, had_error


# ---------------------------------------------------------------------------
# Shard scanning & repair
# ---------------------------------------------------------------------------


def _load_shard(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Load all rows from one JSONL shard, tolerating malformed lines.

    A single corrupt/truncated JSONL line -- exactly the failure class this
    module exists to heal -- must never abort the scan, so a bad line is
    skipped with a warning (mirroring
    ``fetch_production._load_ids_from_file``'s scoped except) and counted
    rather than raised.

    :param path: path to the shard file.
    :return: ``(rows, skipped)`` -- row dicts in file order and the count
        of malformed lines skipped.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError) as e:
                skipped += 1
                log.warning(
                    "Skipping malformed line %d in shard %s: %s",
                    lineno,
                    path.name,
                    e,
                )
    return rows, skipped


def _shard_date(path: Path) -> Optional[date]:
    """Parse the calendar date encoded in a daily shard filename.

    :param path: shard path (``production_log_YYYY_MM_DD.jsonl``).
    :return: the parsed date, or None when the name does not match (e.g.
        ``production_log_legacy.jsonl``) so such shards are never age-gated.
    """
    match = SHARD_DATE_RE.search(path.name)
    if match is None:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _is_candidate(row: dict[str, Any]) -> bool:
    """Return whether a row is a repair candidate.

    :param row: production log row.
    :return: True when the row lost its response and carries a deliver_id.
    """
    return bool(
        row.get("prediction_parse_status") == CANDIDATE_PARSE_STATUS
        and row.get("deliver_id")
    )


def repair_row(
    row: dict[str, Any],
    tool_response: Optional[str],
    model: Optional[str] = None,
    tool_hash: Optional[str] = None,
) -> bool:
    """Re-parse a requeried toolResponse and repair the row in place.

    Only a response that parses to a *valid* prediction mutates the row;
    still-null or still-unparseable responses leave it untouched, so the
    row stays a candidate for future runs.

    The requeried delivery's ``model`` and ``tool_hash`` (returned by
    :func:`extract_delivery_fields` alongside the response) also backfill
    the row's ``model`` / ``tool_version`` / ``config_hash`` when those are
    currently null, so a repaired row buckets under its real tool version
    in by_tool_version/by_config rather than under "unknown". Existing
    (non-null) metadata is never overwritten.

    :param row: production log row (mutated in place on success).
    :param tool_response: requeried toolResponse, possibly still None.
    :param model: requeried delivery model, if available.
    :param tool_hash: requeried delivery tool hash, if available.
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
    # Backfill bucketing metadata (fill-if-null; never overwrite). Mirror
    # build_row: config_hash = _compute_config_hash(tool_version, model).
    if model is not None and row.get("model") is None:
        row["model"] = model
    if tool_hash is not None and row.get("tool_version") is None:
        row["tool_version"] = tool_hash
    if tool_hash is not None and row.get("config_hash") is None:
        row["config_hash"] = _compute_config_hash(tool_hash, row.get("model"))
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
    max_shard_age_days: Optional[int] = None,
) -> dict[str, Any]:
    """Scan all shards, requery lost responses, and repair rows in place.

    Never raises: a failed probe/batch, malformed line, unreadable shard,
    or shard-rewrite failure is logged and skipped, and the partial summary
    (with the repairs already persisted) is returned. ``backfill_error`` in
    the summary is True when any subgraph probe/batch or shard rewrite
    failed, so the caller can force a rebuild of the already-repaired rows.

    :param logs_dir: directory containing the daily JSONL shards.
    :param batch_size: max deliver ids per GraphQL request.
    :param max_shard_age_days: skip shards whose filename date is older
        than this many days (None disables the bound); give-up behavior for
        legacy/unhealable rows once they age out of the window. Shards with
        an unparseable filename date are never age-gated.
    :return: summary dict with ``shards_scanned``, ``candidates``,
        ``repaired``, ``skipped_lines``, ``backfill_error``, and a
        per-platform breakdown.
    """
    summary: dict[str, Any] = {
        "shards_scanned": 0,
        "candidates": 0,
        "repaired": 0,
        "skipped_lines": 0,
        "backfill_error": False,
        "platforms": {},
    }
    if not logs_dir.is_dir():
        log.info("Logs dir %s does not exist -- nothing to backfill", logs_dir)
        return summary

    cutoff: Optional[date] = None
    if max_shard_age_days is not None:
        cutoff = date.today() - timedelta(days=max_shard_age_days)

    # Pass 1: scan shards, collect candidate deliver ids per platform.
    shards: list[tuple[Path, list[dict[str, Any]]]] = []
    ids_by_platform: dict[str, list[str]] = {}
    for path in sorted(logs_dir.glob("*.jsonl")):
        if cutoff is not None:
            shard_dt = _shard_date(path)
            if shard_dt is not None and shard_dt < cutoff:
                log.info(
                    "Skipping shard %s older than %d days -- giving up",
                    path.name,
                    max_shard_age_days,
                )
                continue
        try:
            rows, skipped = _load_shard(path)
        except OSError as e:
            # Per-file isolation: an unreadable shard cannot sink the run.
            log.warning("Failed to read shard %s: %s -- skipping", path.name, e)
            summary["backfill_error"] = True
            continue
        summary["skipped_lines"] += skipped
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
    responses: dict[str, dict[str, dict[str, Any]]] = {}
    for platform, deliver_ids in ids_by_platform.items():
        unique_ids = sorted(set(deliver_ids))
        log.info(
            "%s: requerying %d deliver ids in batches of %d",
            platform,
            len(unique_ids),
            batch_size,
        )
        plat_responses, had_error = fetch_tool_responses(
            PLATFORM_MARKETPLACE_URLS[platform], unique_ids, batch_size
        )
        responses[platform] = plat_responses
        if had_error:
            summary["backfill_error"] = True

    # Pass 3: repair rows in place; rewrite only shards that changed. Each
    # shard's repair+rewrite is isolated so a failure on one cannot discard
    # the repairs already persisted to earlier shards. Per-shard/per-platform
    # counts are committed to the summary only AFTER the atomic rewrite
    # succeeds, so the returned counts reflect exactly what is on disk.
    for path, rows in shards:
        try:
            shard_repaired = 0
            shard_by_platform: dict[str, int] = {}
            for row in rows:
                if not _is_candidate(row):
                    continue
                platform = row.get("platform") or "unknown"
                fields = responses.get(platform, {}).get(row["deliver_id"])
                if fields is None:
                    continue
                if repair_row(
                    row,
                    fields.get("tool_response"),
                    fields.get("model"),
                    fields.get("tool_hash"),
                ):
                    shard_repaired += 1
                    shard_by_platform[platform] = shard_by_platform.get(platform, 0) + 1
            if shard_repaired:
                _rewrite_shard_atomic(path, rows)
                summary["repaired"] += shard_repaired
                for platform, count in shard_by_platform.items():
                    summary["platforms"][platform]["repaired"] += count
                log.info("Repaired %d rows in %s", shard_repaired, path.name)
        except Exception as e:  # pylint: disable=broad-except
            # Isolate this shard; already-persisted repairs are kept and a
            # rebuild is forced (backfill_error) since we cannot tell which
            # rows on later shards would have been repaired.
            log.warning(
                "Repair/rewrite failed for shard %s: %s -- continuing",
                path.name,
                e,
            )
            summary["backfill_error"] = True
            continue

    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _log_summary(summary: dict[str, Any]) -> None:
    """Log the backfill summary.

    :param summary: summary dict from :func:`backfill`.
    """
    log.info(
        "Backfill summary: %d shards scanned, %d candidate rows, %d repaired, "
        "%d malformed lines skipped, backfill_error=%s",
        summary["shards_scanned"],
        summary["candidates"],
        summary["repaired"],
        summary.get("skipped_lines", 0),
        summary.get("backfill_error", False),
    )
    for platform, stats in sorted(summary["platforms"].items()):
        log.info(
            "  %s: %d candidates, %d repaired",
            platform,
            stats["candidates"],
            stats["repaired"],
        )


def _emit_repaired_output(repaired: int, backfill_error: bool = False) -> None:
    """Emit the repaired count and error flag for workflow consumption.

    Prints ``repaired=<n>`` and ``backfill_error=<true|false>`` to stdout
    and appends the same lines to ``$GITHUB_OUTPUT`` when that env var is
    set, so the flywheel force-rebuild path can gate on either: repaired>0
    (rows changed) OR backfill_error (a partial rewrite/batch failure means
    some rows may have been repaired but we cannot tell which, so rebuild).

    :param repaired: number of rows repaired this run.
    :param backfill_error: True if any probe/batch/shard rewrite failed.
    """
    lines = [
        f"repaired={repaired}",
        f"backfill_error={'true' if backfill_error else 'false'}",
    ]
    for line in lines:
        print(line)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            for line in lines:
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
    parser.add_argument(
        "--max-shard-age-days",
        type=int,
        default=DEFAULT_MAX_SHARD_AGE_DAYS,
        help=(
            "Skip shards whose filename date is older than this many days "
            "so legacy/unhealable rows are not re-globbed and re-queried "
            "forever as logs/ grows (0 or negative disables the bound)"
        ),
    )
    return parser


def main() -> None:
    """CLI entry point.

    Backfill failures (subgraph/parsing/IO errors) never raise -- they are
    caught and reported as ``backfill_error=true`` with the repairs already
    persisted preserved in the count. Argument-parse errors and a fatal
    ``$GITHUB_OUTPUT`` write are the only non-zero exits.
    """
    args = _build_arg_parser().parse_args()
    max_age: Optional[int] = (
        args.max_shard_age_days if args.max_shard_age_days > 0 else None
    )
    repaired = 0
    backfill_error = False
    try:
        summary = backfill(args.logs_dir, args.batch_size, max_age)
        repaired = summary["repaired"]
        backfill_error = summary.get("backfill_error", False)
        _log_summary(summary)
    except Exception:  # pylint: disable=broad-except
        # Defense in depth: backfill is designed not to raise, but if it
        # ever does, force a rebuild rather than silently drop the signal.
        log.exception("Backfill failed (non-fatal: the flywheel must not block)")
        backfill_error = True
    _emit_repaired_output(repaired, backfill_error)


if __name__ == "__main__":
    main()
