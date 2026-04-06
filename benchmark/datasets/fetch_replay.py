"""
Fetch replay-ready dataset directly from on-chain subgraphs.

Combines delivery fetch, IPFS prompt extraction, and outcome matching
in a single pass. Outputs enriched JSONL ready for prompt_replay.py replay.

Usage:
    python benchmark/datasets/fetch_replay.py \
      --tool prediction-online \
      --lookback-days 7 \
      --output benchmark/results/replay_dataset.jsonl

    python benchmark/datasets/fetch_replay.py \
      --tool prediction-online \
      --lookback-days 30 \
      --sample-per-platform 50 --seed 42 \
      --output benchmark/results/replay_dataset_50x50.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmark.datasets.fetch_production import (
    IPFS_FETCH_DELAY,
    MECH_MARKETPLACE_GNOSIS_URL,
    MECH_MARKETPLACE_POLYGON_URL,
    QUESTION_DATA_SEPARATOR,
    classify_category,
    fetch_omen_resolved,
    fetch_polymarket_resolved,
    parse_tool_response,
)
from benchmark.prompt_replay import (
    ADDITIONAL_INFO_RE,
    USER_PROMPT_RE,
    _ipfs_hash_to_cid,
    fetch_ipfs_prompt,
    stratified_sample,
)

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

HTTP_TIMEOUT = 60
DEFAULT_BATCH_SIZE = 1000

# Delivery query that includes ipfsHashBytes in the same call
DELIVERS_WITH_IPFS_QUERY = """
{
  delivers(
    first: %(first)s
    skip: %(skip)s
    orderBy: blockTimestamp
    orderDirection: desc
    where: { blockTimestamp_gt: %(timestamp_gt)s }
  ) {
    id
    blockTimestamp
    model
    toolResponse
    marketplaceDelivery {
      ipfsHashBytes
    }
    request {
      id
      blockTimestamp
      parsedRequest {
        questionTitle
        tool
        content
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Subgraph helpers (reuse from fetch_production)
# ---------------------------------------------------------------------------

import requests

MAX_RETRIES = 3


def _post_graphql(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Post a GraphQL query and return the JSON response data."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
            if "errors" in body:
                error_msg = str(body["errors"])
                if "reorganized" in error_msg and attempt < MAX_RETRIES:
                    log.warning("Chain reorg, retrying (attempt %d/%d)", attempt, MAX_RETRIES)
                    time.sleep(attempt * 5)
                    continue
                raise RuntimeError(f"GraphQL errors: {body['errors']}")
            return body.get("data", {})
        except requests.exceptions.ReadTimeout:
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 10)
            else:
                raise
    return {}


def _paginated_fetch(
    url: str,
    query_template: str,
    template_vars: dict[str, Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Fetch all records using pagination."""
    all_records: list[dict[str, Any]] = []
    skip = 0
    while True:
        query = query_template % {**template_vars, "first": batch_size, "skip": skip}
        data = _post_graphql(url, {"query": query})
        batch = data.get("delivers", [])
        if not batch:
            break
        all_records.extend(batch)
        log.info("  fetched %d delivers (total %d)", len(batch), len(all_records))
        if len(batch) < batch_size:
            break
        skip += batch_size
    return all_records


# ---------------------------------------------------------------------------
# Core: fetch deliveries with IPFS hashes
# ---------------------------------------------------------------------------


def _extract_question_title(question: str) -> str:
    """Extract question title using the separator."""
    if not question:
        return ""
    return question.split(QUESTION_DATA_SEPARATOR)[0].strip()


def fetch_deliveries(
    marketplace_url: str,
    timestamp_gt: int,
    tool_filter: str,
) -> list[dict[str, Any]]:
    """Fetch deliveries for a specific tool, including IPFS hashes."""
    raw = _paginated_fetch(
        marketplace_url,
        DELIVERS_WITH_IPFS_QUERY,
        {"timestamp_gt": timestamp_gt},
    )

    deliveries = []
    skipped = 0
    for d in raw:
        request = d.get("request") or {}
        parsed = request.get("parsedRequest")
        if not parsed:
            skipped += 1
            continue

        tool = parsed.get("tool") or "unknown"
        if tool != tool_filter:
            continue

        question_title = _extract_question_title(parsed.get("questionTitle", ""))
        if not question_title:
            skipped += 1
            continue

        mp = d.get("marketplaceDelivery") or {}
        ipfs_hash = mp.get("ipfsHashBytes")

        # Parse request_context for market_id
        market_id = None
        try:
            content = json.loads(parsed.get("content", "") or "{}")
            ctx = content.get("request_context") or {}
            market_id = ctx.get("market_id")
        except (json.JSONDecodeError, TypeError):
            pass

        deliveries.append({
            "deliver_id": d["id"],
            "timestamp": int(d["blockTimestamp"]),
            "request_timestamp": int(request.get("blockTimestamp") or 0) or None,
            "model": d.get("model"),
            "tool_response": d.get("toolResponse"),
            "tool": tool,
            "question_title": question_title,
            "market_id": market_id,
            "ipfs_hash": ipfs_hash,
        })

    if skipped:
        log.info("  skipped %d deliveries (null parsedRequest or title)", skipped)
    log.info("  %d %s deliveries found", len(deliveries), tool_filter)

    return deliveries


# ---------------------------------------------------------------------------
# Match deliveries to resolved markets
# ---------------------------------------------------------------------------


def match_outcomes(
    deliveries: list[dict[str, Any]],
    resolved_after: int,
    platform: str,
) -> list[dict[str, Any]]:
    """Match deliveries to resolved markets and return matched rows."""
    log.info("Fetching resolved %s markets...", platform)
    if platform == "omen":
        markets = fetch_omen_resolved(resolved_after)
    else:
        markets = fetch_polymarket_resolved(resolved_after)
    log.info("  %d resolved markets found", len(markets))

    matched = []
    for d in deliveries:
        # Try market_id first
        market_data = None
        match_confidence = 0.0

        mid = d.get("market_id")
        if mid and mid in markets.by_id:
            market_data = markets.by_id[mid]
            match_confidence = 1.0

        # Fallback: title match
        if not market_data:
            key = d["question_title"].lower()
            if key in markets.by_title:
                market_data = markets.by_title[key]
                match_confidence = 1.0
            elif len(key) >= 20:
                for mt, md_data in markets.by_title.items():
                    if len(mt) >= 20 and (key.startswith(mt) or mt.startswith(key)):
                        market_data = md_data
                        match_confidence = 0.8
                        break

        if market_data is not None:
            parsed = parse_tool_response(d["tool_response"])
            if parsed["prediction_parse_status"] != "valid":
                continue

            matched.append({
                **d,
                "platform": platform,
                "final_outcome": market_data["outcome"],
                "resolved_at_ts": market_data["resolved_at_ts"],
                "match_confidence": match_confidence,
                "p_yes": parsed["p_yes"],
                "p_no": parsed["p_no"],
                "confidence": parsed.get("confidence"),
                "prediction_parse_status": "valid",
            })

    log.info("  %d/%d deliveries matched to outcomes", len(matched), len(deliveries))
    return matched


# ---------------------------------------------------------------------------
# Enrich with IPFS prompt extraction
# ---------------------------------------------------------------------------


def enrich_with_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fetch IPFS prompts and extract user_prompt + additional_information."""
    enriched = []
    for i, row in enumerate(rows):
        ipfs_hash = row.get("ipfs_hash")
        if not ipfs_hash:
            continue

        prompt_text = fetch_ipfs_prompt(ipfs_hash)
        if not prompt_text:
            continue

        up_match = USER_PROMPT_RE.search(prompt_text)
        ai_match = ADDITIONAL_INFO_RE.search(prompt_text)
        if not up_match:
            continue

        row["extracted_user_prompt"] = up_match.group(1).strip()
        row["extracted_additional_information"] = (
            ai_match.group(1).strip() if ai_match else ""
        )
        enriched.append(row)

        if (i + 1) % 10 == 0:
            log.info("  IPFS progress: %d/%d (%d enriched)", i + 1, len(rows), len(enriched))

        time.sleep(IPFS_FETCH_DELAY)

    log.info("  Enriched %d/%d rows with prompt data", len(enriched), len(rows))
    return enriched


# ---------------------------------------------------------------------------
# Build output rows
# ---------------------------------------------------------------------------


def build_output_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert matched+enriched rows to replay-ready format."""
    output = []
    for row in rows:
        delivery_ts = row["timestamp"]
        request_ts = row.get("request_timestamp")
        resolved_at_ts = row.get("resolved_at_ts")

        latency_s = None
        if request_ts and delivery_ts > request_ts:
            latency_s = delivery_ts - request_ts

        prediction_lead_time_days = None
        if resolved_at_ts and resolved_at_ts > delivery_ts:
            prediction_lead_time_days = round((resolved_at_ts - delivery_ts) / 86400, 1)

        def _ts_to_iso(ts: Optional[int]) -> Optional[str]:
            if ts is None:
                return None
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        question = row["question_title"]
        output.append({
            "row_id": row["deliver_id"],
            "deliver_id": row["deliver_id"],
            "schema_version": "1.0",
            "mode": "production_replay",
            "platform": row["platform"],
            "question_text": question,
            "tool_name": row["tool"],
            "model": row.get("model"),
            "p_yes": row["p_yes"],
            "p_no": row["p_no"],
            "prediction_parse_status": "valid",
            "confidence": row.get("confidence"),
            "final_outcome": row["final_outcome"],
            "requested_at": _ts_to_iso(request_ts),
            "predicted_at": _ts_to_iso(delivery_ts),
            "resolved_at": _ts_to_iso(resolved_at_ts),
            "latency_s": latency_s,
            "prediction_lead_time_days": prediction_lead_time_days,
            "category": classify_category(question),
            "match_confidence": row["match_confidence"],
            "extracted_user_prompt": row["extracted_user_prompt"],
            "extracted_additional_information": row["extracted_additional_information"],
        })
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def fetch_replay(
    tool_filter: str,
    lookback_days: int,
    output: Path,
    sample_per_platform: Optional[int] = None,
    seed: int = 42,
    skip_recent_days: int = 0,
) -> None:
    """Fetch replay-ready dataset: deliveries + outcomes + IPFS prompts."""
    now = int(time.time())
    cutoff = now - (lookback_days * 86400)
    upper_cutoff = now - (skip_recent_days * 86400) if skip_recent_days > 0 else None

    # 1. Fetch deliveries from both chains
    log.info("=== Fetching %s deliveries (last %d days) ===", tool_filter, lookback_days)

    log.info("Gnosis marketplace...")
    gnosis_deliveries = fetch_deliveries(
        MECH_MARKETPLACE_GNOSIS_URL, cutoff, tool_filter
    )
    log.info("Polygon marketplace...")
    polygon_deliveries = fetch_deliveries(
        MECH_MARKETPLACE_POLYGON_URL, cutoff, tool_filter
    )

    # Filter out recent deliveries if skip_recent_days is set
    if upper_cutoff is not None:
        before = len(gnosis_deliveries) + len(polygon_deliveries)
        gnosis_deliveries = [d for d in gnosis_deliveries if d["timestamp"] <= upper_cutoff]
        polygon_deliveries = [d for d in polygon_deliveries if d["timestamp"] <= upper_cutoff]
        after = len(gnosis_deliveries) + len(polygon_deliveries)
        log.info("Filtered to days %d-%d: %d -> %d deliveries", skip_recent_days, lookback_days, before, after)

    # 2. Match to resolved markets
    log.info("=== Matching to resolved markets ===")
    omen_matched = match_outcomes(gnosis_deliveries, cutoff, "omen")
    poly_matched = match_outcomes(polygon_deliveries, cutoff, "polymarket")

    all_matched = omen_matched + poly_matched
    log.info("Total matched: %d (omen=%d, polymarket=%d)",
             len(all_matched), len(omen_matched), len(poly_matched))

    if not all_matched:
        log.warning("No matched deliveries found")
        return

    # 3. Stratified sample before IPFS fetch
    if sample_per_platform is not None:
        all_matched = stratified_sample(
            # stratified_sample expects "final_outcome" key
            all_matched,
            sample_per_platform,
            seed,
        )
        log.info("Sampled %d rows for IPFS fetch", len(all_matched))

    # 4. Enrich with IPFS prompts
    log.info("=== Fetching IPFS prompts ===")
    enriched = enrich_with_prompts(all_matched)

    if not enriched:
        log.warning("No rows enriched with prompts")
        return

    # 5. Build output
    output_rows = build_output_rows(enriched)

    # Report
    by_platform: dict[str, int] = defaultdict(int)
    by_outcome: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in output_rows:
        plat = row["platform"]
        outcome = "yes" if row["final_outcome"] else "no"
        by_platform[plat] += 1
        by_outcome[plat][outcome] += 1
    for plat in sorted(by_platform):
        log.info("  %s: %d rows (yes=%d, no=%d)",
                 plat, by_platform[plat],
                 by_outcome[plat]["yes"], by_outcome[plat]["no"])

    # Write
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")

    log.info("Written %d rows to %s", len(output_rows), output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch replay-ready dataset from on-chain subgraphs.",
    )
    parser.add_argument(
        "--tool",
        type=str,
        default="prediction-online",
        help="Tool name to filter (default: prediction-online)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Fetch deliveries from the last N days (default: 7)",
    )
    parser.add_argument(
        "--sample-per-platform",
        type=int,
        default=None,
        help="Stratified sample N markets per platform (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42)",
    )
    parser.add_argument(
        "--skip-recent-days",
        type=int,
        default=0,
        help="Exclude the most recent N days (default: 0). Use to create non-overlapping windows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/replay_dataset.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    fetch_replay(
        tool_filter=args.tool,
        lookback_days=args.lookback_days,
        output=args.output,
        sample_per_platform=args.sample_per_platform,
        seed=args.seed,
        skip_recent_days=args.skip_recent_days,
    )


if __name__ == "__main__":
    main()
