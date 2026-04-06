"""
Score tournament predictions by matching against market resolutions.

Loads stored predictions from tournament_predictions.jsonl, checks which
markets have resolved, fills in final_outcome, and writes scorer-compatible
output to tournament_scored.jsonl.

Usage:
    python benchmark/score_tournament.py
    python benchmark/score_tournament.py --predictions path/to/predictions.jsonl
    python benchmark/score_tournament.py --output path/to/scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchmark.io import load_existing_ids, load_jsonl

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

DEFAULT_PREDICTIONS = Path(__file__).parent / "results" / "tournament_predictions.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "tournament_scored.jsonl"

OMEN_SUBGRAPH_URL = os.environ.get(
    "OMEN_SUBGRAPH_URL",
    "https://omen.subgraph.autonolas.tech",
)
POLYMARKET_GAMMA_URL = os.environ.get(
    "POLYMARKET_GAMMA_URL",
    "https://gamma-api.polymarket.com",
)

HTTP_TIMEOUT = 60
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# GraphQL helpers
# ---------------------------------------------------------------------------

# Query a single Omen market by address to check resolution
OMEN_MARKET_QUERY = """
{
  fixedProductMarketMaker(id: "%(market_address)s") {
    id
    currentAnswer
    currentAnswerTimestamp
    outcomes
  }
}
"""

# Batch query multiple Omen markets at once
OMEN_MARKETS_BATCH_QUERY = """
{
  fixedProductMarketMakers(
    first: %(first)s
    where: { id_in: %(ids)s }
  ) {
    id
    currentAnswer
    currentAnswerTimestamp
    outcomes
  }
}
"""


def _post_graphql(url: str, query: str) -> dict[str, Any]:
    """POST a GraphQL query with retry. Returns response data dict."""
    headers = {"Content-Type": "application/json"}
    data: dict[str, Any] = {}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url,
                json={"query": query},
                headers=headers,
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()
            if "errors" in body:
                raise RuntimeError(f"GraphQL errors: {body['errors']}")
            data = body.get("data", {})
            return data
        except requests.exceptions.ReadTimeout:
            if attempt < MAX_RETRIES:
                wait = attempt * 10
                log.warning(
                    "Timeout on %s (attempt %d/%d), retrying in %ds",
                    url,
                    attempt,
                    MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
            else:
                raise
    return data


# ---------------------------------------------------------------------------
# Resolution checkers
# ---------------------------------------------------------------------------


def check_omen_resolutions(
    market_addresses: list[str],
) -> dict[str, dict[str, Any]]:
    """Check resolution status for Omen markets.

    :param market_addresses: Omen market contract addresses (without omen_ prefix).
    :return: dict mapping market_address to resolution data. Only resolved markets.
    """
    if not market_addresses:
        return {}

    resolved: dict[str, dict[str, Any]] = {}

    # Batch query (subgraph supports up to 1000 per query)
    for i in range(0, len(market_addresses), 100):
        batch = market_addresses[i : i + 100]
        ids_json = json.dumps(batch)
        query = OMEN_MARKETS_BATCH_QUERY % {"first": len(batch), "ids": ids_json}
        data = _post_graphql(OMEN_SUBGRAPH_URL, query)

        for fpmm in data.get("fixedProductMarketMakers", []):
            current_answer = fpmm.get("currentAnswer")
            if current_answer is None:
                continue

            market_addr = fpmm.get("id", "")
            try:
                outcome_index = int(current_answer, 16)
            except (ValueError, TypeError):
                log.warning(
                    "Omen market %s: unparseable currentAnswer=%s",
                    market_addr,
                    current_answer,
                )
                continue

            # outcomes=["Yes","No"]: index 0 = Yes → True
            outcome = outcome_index == 0

            resolved_ts = fpmm.get("currentAnswerTimestamp")
            resolved_at = None
            if resolved_ts:
                resolved_at = datetime.fromtimestamp(
                    int(resolved_ts), tz=timezone.utc
                ).isoformat()

            resolved[market_addr] = {
                "outcome": outcome,
                "resolved_at": resolved_at,
            }

    return resolved


def check_polymarket_resolutions(
    condition_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Check resolution status for Polymarket markets via Gamma API.

    :param condition_ids: Polymarket condition IDs (without poly_ prefix).
    :return: dict mapping condition_id to resolution data. Only resolved markets.
    """
    if not condition_ids:
        return {}

    resolved: dict[str, dict[str, Any]] = {}

    for cid in condition_ids:
        try:
            resp = requests.get(
                f"{POLYMARKET_GAMMA_URL}/markets",
                params={"conditionId": cid},
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            markets = resp.json()
            if not markets:
                continue
            m = markets[0] if isinstance(markets, list) else markets
        except Exception as exc:
            log.warning("Polymarket fetch failed for %s: %s", cid, exc)
            continue

        # Check explicit resolved flag first, fall back to price proxy
        is_resolved = m.get("resolved", False)

        prices_raw = m.get("outcomePrices", "[]")
        try:
            prices = (
                json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            )
            prices = [float(p) for p in prices]
        except (json.JSONDecodeError, TypeError, ValueError):
            prices = []

        # Resolved if API says so OR price hit 1.0 (proxy)
        if not is_resolved and not any(p >= 0.99 for p in prices):
            continue

        outcomes_raw = m.get("outcomes", "[]")
        try:
            outcomes = (
                json.loads(outcomes_raw)
                if isinstance(outcomes_raw, str)
                else outcomes_raw
            )
        except (json.JSONDecodeError, TypeError):
            outcomes = []

        # Find the winning outcome
        winning_idx = None
        for idx, p in enumerate(prices):
            if p >= 0.99:
                winning_idx = idx
                break

        # Fallback: if API says resolved but prices haven't settled to 1.0,
        # pick the highest-priced outcome
        if winning_idx is None and is_resolved and prices:
            winning_idx = prices.index(max(prices))

        if winning_idx is None:
            continue

        if outcomes and winning_idx < len(outcomes):
            outcome = outcomes[winning_idx].lower() == "yes"
        else:
            outcome = winning_idx == 0  # Default: index 0 = Yes

        # Polymarket Gamma API doesn't give resolution timestamp directly
        # Use current time as an approximation
        resolved_at = datetime.now(timezone.utc).isoformat()

        resolved[cid] = {
            "outcome": outcome,
            "resolved_at": resolved_at,
        }

    return resolved


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load tournament predictions from JSONL."""
    return load_jsonl(path)


def load_existing_row_ids(path: Path) -> set[str]:
    """Load row IDs already in the scored output."""
    return load_existing_ids(path)


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------


def _fetch_resolution_map(
    pending: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collect unique markets from pending rows and query resolutions."""
    omen_addrs: dict[str, str] = {}
    poly_cids: dict[str, str] = {}

    for row in pending:
        market_addr = row.get("market_address", "")
        plat = row.get("platform", "")
        if plat == "omen" and market_addr:
            omen_addrs[market_addr] = row.get("market_id", "")
        elif plat == "polymarket" and market_addr:
            poly_cids[market_addr] = row.get("market_id", "")

    log.info(
        "Checking resolutions: %d Omen markets, %d Polymarket markets",
        len(omen_addrs),
        len(poly_cids),
    )

    omen_resolved = check_omen_resolutions(list(omen_addrs.keys()))
    poly_resolved = check_polymarket_resolutions(list(poly_cids.keys()))

    log.info(
        "  Resolved: %d Omen, %d Polymarket",
        len(omen_resolved),
        len(poly_resolved),
    )

    resolution_map: dict[str, dict[str, Any]] = {}
    resolution_map.update(omen_resolved)
    resolution_map.update(poly_resolved)
    return resolution_map


def _apply_resolution(
    row: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    """Apply a resolution to a prediction row. Returns the scored row."""
    scored_row = {**row}
    scored_row["final_outcome"] = resolution["outcome"]
    scored_row["resolved_at"] = resolution["resolved_at"]

    predicted_at = row.get("predicted_at")
    resolved_at = resolution["resolved_at"]
    if predicted_at and resolved_at:
        try:
            pred_dt = datetime.fromisoformat(predicted_at)
            res_dt = datetime.fromisoformat(resolved_at)
            lead_days = (res_dt - pred_dt).total_seconds() / 86400
            scored_row["prediction_lead_time_days"] = round(lead_days, 1)
        except (ValueError, TypeError):
            pass

    scored_row.pop("source_content", None)
    return scored_row


def score_tournament(
    predictions_path: Path,
    output_path: Path,
) -> None:
    """Match predictions against resolutions and write scored rows."""
    predictions = load_predictions(predictions_path)
    log.info("Loaded %d tournament predictions", len(predictions))

    pending = [r for r in predictions if r.get("final_outcome") is None]
    log.info("  %d pending (no final_outcome)", len(pending))

    if not pending:
        log.info("No pending predictions to score.")
        return

    resolution_map = _fetch_resolution_map(pending)
    if not resolution_map:
        log.info("No markets have resolved yet.")
        return

    existing_scored = load_existing_row_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_count = 0
    skipped = 0

    with open(output_path, "a", encoding="utf-8") as out:
        for row in pending:
            market_addr = row.get("market_address", "")
            if market_addr not in resolution_map:
                continue

            row_id = row.get("row_id", "")
            if row_id in existing_scored:
                skipped += 1
                continue

            scored_row = _apply_resolution(row, resolution_map[market_addr])
            out.write(json.dumps(scored_row, ensure_ascii=False) + "\n")
            out.flush()
            scored_count += 1

    log.info(
        "Done: %d predictions scored, %d skipped (already scored). Output: %s",
        scored_count,
        skipped,
        output_path,
    )

    if scored_count > 0:
        _update_predictions_file(predictions_path, resolution_map)


def _update_predictions_file(
    path: Path,
    resolution_map: dict[str, dict[str, Any]],
) -> None:
    """Update predictions JSONL to mark resolved markets (avoid re-querying)."""
    updated_lines: list[str] = []
    updates = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            market_addr = row.get("market_address", "")
            if row.get("final_outcome") is None and market_addr in resolution_map:
                resolution = resolution_map[market_addr]
                row["final_outcome"] = resolution["outcome"]
                row["resolved_at"] = resolution["resolved_at"]
                updates += 1
            updated_lines.append(json.dumps(row, ensure_ascii=False))

    # Atomic write: temp file + os.replace to avoid corruption on crash
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp:
            tmp.write("\n".join(updated_lines) + "\n")
        os.replace(tmp_path, str(path))
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    log.info("Updated %d rows in %s", updates, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Score tournament predictions against market resolutions."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(DEFAULT_PREDICTIONS),
        help=f"Path to tournament predictions JSONL (default: {DEFAULT_PREDICTIONS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output scored JSONL path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    score_tournament(
        predictions_path=Path(args.predictions),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
