"""
Fetch currently open (unresolved) prediction markets from Omen and Polymarket.

Queries the Omen subgraph for binary markets without a resolution, and
the Polymarket Gamma API for binary, non-resolved, non-neg-risk markets.
Output feeds into tournament.py for forward-looking predictions.

Usage:
    python benchmark/datasets/fetch_open.py --platform omen --dry-run
    python benchmark/datasets/fetch_open.py --platform all --min-liquidity 1000
    python benchmark/datasets/fetch_open.py --platform polymarket --max-markets 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from benchmark.datasets.fetch_production import classify_category

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

OMEN_SUBGRAPH_URL = os.environ.get(
    "OMEN_SUBGRAPH_URL",
    "https://omen.subgraph.autonolas.tech",
)
POLYMARKET_GAMMA_URL = os.environ.get(
    "POLYMARKET_GAMMA_URL",
    "https://gamma-api.polymarket.com",
)

DEFAULT_OUTPUT = Path(__file__).parent / "open_markets.jsonl"
DEFAULT_BATCH_SIZE = 1000
HTTP_TIMEOUT = 60
MAX_RETRIES = 3

# Omen market creators the trader bets on
OMEN_CREATORS = [
    "0xffc8029154ecd55abed15bd428ba596e7d23f557",  # Pearl
    "0x89c5cc945dd550bcffb72fe42bff002429f46fec",  # Quickstart (QS)
]

# Polymarket category slugs to iterate
POLYMARKET_CATEGORIES = [
    "business",
    "politics",
    "science",
    "technology",
    "health",
    "travel",
    "entertainment",
    "weather",
    "finance",
    "international",
]
POLYMARKET_WINDOW_DAYS = 30

# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

# Omen: fetch open binary markets (currentAnswer is null)
OMEN_OPEN_MARKETS_QUERY = """
{
  fixedProductMarketMakers(
    first: %(first)s
    skip: %(skip)s
    orderBy: creationTimestamp
    orderDirection: desc
    where: {
      currentAnswer: null
      outcomeSlotCount: 2
      creator_in: %(creators)s
    }
  ) {
    id
    title
    outcomes
    outcomeTokenMarginalPrices
    usdVolume
    usdLiquidityMeasure
    creationTimestamp
    openingTimestamp
    category
  }
}
"""

# ---------------------------------------------------------------------------
# HTTP / GraphQL helpers
# ---------------------------------------------------------------------------


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
# Omen: fetch open markets
# ---------------------------------------------------------------------------


def fetch_omen_open(max_markets: int = 500) -> list[dict[str, Any]]:
    """Fetch open binary markets from the Omen subgraph."""
    markets: list[dict[str, Any]] = []
    skip = 0
    creators_json = json.dumps(OMEN_CREATORS)

    while len(markets) < max_markets:
        query = OMEN_OPEN_MARKETS_QUERY % {
            "first": DEFAULT_BATCH_SIZE,
            "skip": skip,
            "creators": creators_json,
        }
        data = _post_graphql(OMEN_SUBGRAPH_URL, query)
        batch = data.get("fixedProductMarketMakers", [])
        if not batch:
            break

        for fpmm in batch:
            market_addr = fpmm.get("id", "")
            if not market_addr:
                continue

            outcomes = fpmm.get("outcomes") or []
            if len(outcomes) != 2:
                continue

            title = (fpmm.get("title") or "").strip()
            if not title:
                continue

            # Parse marginal prices for current probability
            prices = fpmm.get("outcomeTokenMarginalPrices") or []
            current_prob = None
            if len(prices) == 2:
                try:
                    current_prob = round(float(prices[0]), 4)
                except (ValueError, TypeError):
                    pass

            try:
                usd_volume = round(float(fpmm.get("usdVolume", 0)), 2)
            except (ValueError, TypeError):
                usd_volume = 0.0

            try:
                usd_liquidity = round(float(fpmm.get("usdLiquidityMeasure", 0)), 2)
            except (ValueError, TypeError):
                usd_liquidity = 0.0

            # Omen category field is often empty; fall back to keyword classifier
            category = (fpmm.get("category") or "").strip().lower()
            if not category or category == "unknown":
                category = classify_category(title)

            markets.append(
                {
                    "id": f"omen_{market_addr}",
                    "market_address": market_addr,
                    "platform": "omen",
                    "question_text": title,
                    "current_prob": current_prob,
                    "close_date": (
                        datetime.fromtimestamp(
                            int(fpmm.get("openingTimestamp", 0)), tz=timezone.utc
                        ).isoformat()
                        if fpmm.get("openingTimestamp")
                        else None
                    ),
                    "category": category,
                    "usd_volume": usd_volume,
                    "usd_liquidity": usd_liquidity,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            if len(markets) >= max_markets:
                break

        if len(batch) < DEFAULT_BATCH_SIZE:
            break
        skip += DEFAULT_BATCH_SIZE

    return markets


# ---------------------------------------------------------------------------
# Polymarket: fetch open markets via Gamma API
# ---------------------------------------------------------------------------


def _fetch_polymarket_tag_id(category: str) -> Optional[int]:
    """Fetch the numeric tag ID for a Polymarket category slug."""
    try:
        resp = requests.get(
            f"{POLYMARKET_GAMMA_URL}/tags/slug/{category}",
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        return resp.json().get("id")
    except Exception:
        return None


def _is_valid_polymarket_binary(m: dict[str, Any]) -> bool:
    """Check if a Polymarket entry is a valid open binary market."""
    outcomes_raw = m.get("outcomes", "[]")
    try:
        outcomes = (
            json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        )
    except (json.JSONDecodeError, TypeError):
        return False
    if len(outcomes) != 2 or not all(o.lower() in ("yes", "no") for o in outcomes):
        return False

    prices_raw = m.get("outcomePrices", "[]")
    try:
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        prices = [float(p) for p in prices]
    except (json.JSONDecodeError, TypeError, ValueError):
        prices = []
    if any(p >= 0.99 for p in prices):
        return False

    if m.get("negRisk", False):
        return False

    return bool((m.get("question") or "").strip())


def _parse_polymarket_entry(
    m: dict[str, Any],
    category: str,
    min_liquidity: float,
) -> dict[str, Any] | None:
    """Parse a single Polymarket API entry into a market dict, or None if invalid."""
    if not _is_valid_polymarket_binary(m):
        return None

    # Liquidity filter
    try:
        liquidity = float(m.get("liquidity", 0))
    except (ValueError, TypeError):
        liquidity = 0.0
    if liquidity <= 0 or liquidity < min_liquidity:
        return None

    question = (m.get("question") or "").strip()

    prices_raw = m.get("outcomePrices", "[]")
    try:
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        prices = [float(p) for p in prices]
    except (json.JSONDecodeError, TypeError, ValueError):
        prices = []

    condition_id = m.get("conditionId") or m.get("id", "")
    current_prob = prices[0] if len(prices) >= 2 else None
    try:
        volume = round(float(m.get("volume", 0)), 2)
    except (ValueError, TypeError):
        volume = 0.0

    end_date = m.get("endDate", "")
    close_date = None
    if end_date:
        try:
            close_date = datetime.fromisoformat(
                end_date.replace("Z", "+00:00")
            ).isoformat()
        except (ValueError, TypeError):
            pass

    return {
        "id": f"poly_{condition_id}",
        "market_address": condition_id,
        "platform": "polymarket",
        "question_text": question,
        "current_prob": (round(current_prob, 4) if current_prob is not None else None),
        "close_date": close_date,
        "category": category,
        "usd_volume": volume,
        "usd_liquidity": round(liquidity, 2),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_polymarket_open(
    max_markets: int = 500,
    window_days: int = POLYMARKET_WINDOW_DAYS,
    min_liquidity: float = 0.0,
) -> list[dict[str, Any]]:
    """Fetch open binary markets from Polymarket via the Gamma API."""
    now = datetime.now(timezone.utc)
    end_date_min = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date_max = (now + timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    seen_ids: set[str] = set()
    markets: list[dict[str, Any]] = []

    for category in POLYMARKET_CATEGORIES:
        if len(markets) >= max_markets:
            break

        tag_id = _fetch_polymarket_tag_id(category)
        if tag_id is None:
            log.debug("Skipping Polymarket category '%s' (no tag ID)", category)
            continue

        offset = 0
        while len(markets) < max_markets:
            try:
                resp = requests.get(
                    f"{POLYMARKET_GAMMA_URL}/markets",
                    params={
                        "tag_id": str(tag_id),
                        "end_date_min": end_date_min,
                        "end_date_max": end_date_max,
                        "limit": "300",
                        "offset": str(offset),
                        "closed": "false",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                batch = resp.json()
            except Exception as exc:
                log.warning("Polymarket fetch failed for '%s': %s", category, exc)
                break

            if not batch:
                break

            for m in batch:
                condition_id = m.get("conditionId") or m.get("id", "")
                if not condition_id or condition_id in seen_ids:
                    continue
                seen_ids.add(condition_id)

                parsed = _parse_polymarket_entry(m, category, min_liquidity)
                if parsed is not None:
                    markets.append(parsed)

                if len(markets) >= max_markets:
                    break

            if len(batch) < 300:
                break
            offset += 300

    return markets


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def load_existing_ids(path: Path) -> set[str]:
    """Load market IDs from an existing JSONL file for dedup."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            ids.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch open prediction markets for tournament mode."
    )
    parser.add_argument(
        "--platform",
        choices=["omen", "polymarket", "all"],
        default="all",
        help="Platform to fetch from (default: all)",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=500,
        help="Max markets per platform (default: 500)",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=0.0,
        help="Minimum USD liquidity (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=POLYMARKET_WINDOW_DAYS,
        help=f"Polymarket: markets closing within N days (default: {POLYMARKET_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and print stats, don't write files",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    existing_ids = set() if args.dry_run else load_existing_ids(output_path)

    # ---- Fetch markets ----
    all_markets: list[dict[str, Any]] = []

    if args.platform in ("omen", "all"):
        log.info("Fetching open markets from Omen subgraph...")
        omen = fetch_omen_open(max_markets=args.max_markets)
        log.info("  Omen: %d open binary markets", len(omen))
        all_markets.extend(omen)

    if args.platform in ("polymarket", "all"):
        log.info("Fetching open markets from Polymarket Gamma API...")
        poly = fetch_polymarket_open(
            max_markets=args.max_markets,
            window_days=args.window_days,
            min_liquidity=args.min_liquidity,
        )
        log.info("  Polymarket: %d open binary markets", len(poly))
        all_markets.extend(poly)

    # Apply liquidity filter for Omen (Polymarket filters inline)
    if args.min_liquidity > 0:
        before = len(all_markets)
        all_markets = [
            m for m in all_markets if m["usd_liquidity"] >= args.min_liquidity
        ]
        log.info(
            "Liquidity filter: %d → %d markets (dropped %d below $%.2f)",
            before,
            len(all_markets),
            before - len(all_markets),
            args.min_liquidity,
        )

    # Dedup against existing
    new_markets = [m for m in all_markets if m["id"] not in existing_ids]
    log.info(
        "Total: %d markets (%d new, %d already in output)",
        len(all_markets),
        len(new_markets),
        len(all_markets) - len(new_markets),
    )

    if not all_markets:
        log.info("No markets found.")
        return

    # Summary
    platforms: dict[str, int] = {}
    categories: dict[str, int] = {}
    for m in all_markets:
        platforms[m["platform"]] = platforms.get(m["platform"], 0) + 1
        cat = m.get("category") or "other"
        categories[cat] = categories.get(cat, 0) + 1

    log.info("  Platforms: %s", platforms)
    log.info("  Categories: %s", dict(sorted(categories.items())))
    log.info("  Sample questions:")
    for m in all_markets[:5]:
        prob = f" (p={m['current_prob']:.2f})" if m["current_prob"] else ""
        log.info("    - %s%s", m["question_text"][:90], prob)

    if args.dry_run:
        log.info("--dry-run: not writing files.")
        return

    # ---- Write new markets ----
    if new_markets:
        append_jsonl(output_path, new_markets)
        log.info("Wrote %d markets to %s", len(new_markets), output_path)
    else:
        log.info("No new markets to write.")


if __name__ == "__main__":
    main()
