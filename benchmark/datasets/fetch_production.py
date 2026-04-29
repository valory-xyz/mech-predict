"""
Fetch production prediction data from on-chain subgraphs.

Data flow:
  1. Bulk fetch all recent deliveries from marketplace subgraphs
     → question title, tool, model, p_yes, p_no
  2. Bulk fetch all resolved bets from prediction subgraphs
     → question title, outcome
  3. Match deliveries ↔ resolved markets by question title in memory
  4. Output → production_log.jsonl (append-only)

Usage:
    python benchmark/datasets/fetch_production.py
    python benchmark/datasets/fetch_production.py --lookback-days 30
    python benchmark/datasets/fetch_production.py --output path/to/log.jsonl
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from benchmark.categories import PLATFORM_ALLOWED_CATEGORIES
from benchmark.io import append_jsonl

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

QUESTION_DATA_SEPARATOR = "\u241f"

DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_BATCH_SIZE = 1000
HTTP_TIMEOUT = 60
PROBABILITY_SUM_TOLERANCE = 0.05

# Max age for pending deliveries (days). Deliveries older than this
# are dropped from the pending store to keep the state file small.
PENDING_MAX_AGE_DAYS = 90

# Subgraph endpoints — read from env with defaults
PREDICT_OMEN_SUBGRAPH_URL = os.environ.get(
    "PREDICT_OMEN_SUBGRAPH_URL",
    "https://predict-agents.subgraph.autonolas.tech",
)
PREDICT_POLYMARKET_SUBGRAPH_URL = os.environ.get(
    "PREDICT_POLYMARKET_SUBGRAPH_URL",
    "https://predict-polymarket-agents.subgraph.autonolas.tech",
)
MECH_MARKETPLACE_GNOSIS_URL = os.environ.get(
    "MECH_MARKETPLACE_GNOSIS_URL",
    "https://api.subgraph.autonolas.tech/api/proxy/marketplace-gnosis",
)
MECH_MARKETPLACE_POLYGON_URL = os.environ.get(
    "MECH_MARKETPLACE_POLYGON_URL",
    "https://api.subgraph.autonolas.tech/api/proxy/marketplace-polygon",
)
IPFS_GATEWAY_URL = os.environ.get(
    "IPFS_GATEWAY_URL",
    "https://gateway.autonolas.tech/ipfs",
)
IPFS_FETCH_DELAY = 0.2  # seconds between IPFS gateway requests

# Daily log file rotation
LOGS_DIR = Path(__file__).parent / "logs"
LEGACY_LOG_PATH = Path(__file__).parent / "production_log.jsonl"
DEDUP_LOOKBACK_DAYS = 7  # how many daily files to scan on state-loss recovery

# Category keywords for classifying prediction market questions.
# Matched using word boundaries (\b) to avoid substring false positives.
# Falls back to "other" if no keywords match.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "business": [
        "business",
        "corp",
        "corporate",
        "merger",
        "acquisition",
        "startup",
        "ceo",
        "cfo",
        "layoff",
        "hiring",
        "strike",
        "labor union",
        "trade union",
        "bankruptcy",
        "ipo",
        "company",
        "brand",
        "retail",
        "supply chain",
        "logistics",
        "management",
        "industry",
        "commercial",
        "monopoly",
        "antitrust",
        "executive",
        "stellantis",
        "byd",
        "tesla",
        "revenue",
        "profit",
        # merged from economics
        "economy",
        "economic",
        "inflation",
        "recession",
        "gdp",
        "cpi",
        "interest rate",
        "fed",
        "federal reserve",
        "central bank",
        "unemployment",
        "jobs report",
        "macro",
        "debt",
        "deficit",
        "yield curve",
        "treasury",
        "fiscal",
        "mortgage",
        "freddie mac",
        # merged from food
        "food",
        "drink",
        "restaurant",
        "dining",
        "mcdonalds",
        "starbucks",
        "burger",
        "meat",
        "plant-based",
        "agriculture",
        "farming",
        "crop",
        "harvest",
        "beer",
        "wine",
        "spirit",
        "coffee",
        "sugar",
        "grocery",
        "supermarket",
        "chef",
        "cooking",
    ],
    "curiosities": [
        "curiosities",
        "mystery",
        "ufo",
        "alien",
        "flat earth",
        "paranormal",
        "ghost",
        "psychic",
        "anomaly",
        "weird",
        "strange",
        "guinness",
        "record breaker",
        "bizarre",
        "hoax",
        "conspiracy",
        "qanon",
    ],
    "entertainment": [
        "entertainment",
        "movie",
        "film",
        "cinema",
        "hollywood",
        "actor",
        "actress",
        "netflix",
        "disney",
        "hbo",
        "box office",
        "oscar",
        "tv",
        "series",
        "streaming",
        "show",
        "theater",
        "gambling",
        "betting",
        "poker",
        "casino",
        "lottery",
        # merged from music
        "music",
        "song",
        "album",
        "artist",
        "concert",
        "spotify",
        "grammy",
        "billboard",
        "singer",
        "band",
        "rapper",
        "genre",
        "hip hop",
        "chart",
        "musical",
        "vocalist",
        # merged from arts
        "art",
        "arts",
        "museum",
        "painting",
        "auction",
        "sothebys",
        "christies",
        "gallery",
        "masterpiece",
        "sculpture",
        "exhibition",
        "cultural",
        "literature",
        "novel",
        "biography",
        "author",
        "poet",
        # merged from fashion
        "fashion",
        "clothing",
        "apparel",
        "luxury",
        "gucci",
        "prada",
        "nike",
        "adidas",
        "sneaker",
        "shoe",
        "runway",
        "designer",
        "style",
        "vogue",
        "wear",
        "textile",
        "fashion collection",
        "couture",
        "handbag",
        # merged from trending
        "trending",
        "viral",
        "trend",
        "tiktok",
        "meme",
        "challenge",
        "hashtag",
        "breaking",
        "hype",
        "buzz",
        "influencer",
        "youtuber",
        "streamer",
        "mrbeast",
        "drama",
        "cancel culture",
    ],
    "finance": [
        "finance",
        "financial",
        "stock",
        "share",
        "market",
        "wall street",
        "sp500",
        "nasdaq",
        "dow jones",
        "trade",
        "investor",
        "dividend",
        "portfolio",
        "hedge fund",
        "equity",
        "bond",
        "earnings",
        "bloomberg",
        "etf",
        "short",
        "long",
        "robinhood",
        "close",
        # merged from crypto
        "crypto",
        "cryptocurrency",
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "blockchain",
        "web3",
        "defi",
        "nft",
        "token",
        "wallet",
        "coinbase",
        "binance",
        "solana",
        "doge",
        "stablecoin",
        "altcoin",
        "mining",
        "ledger",
        "satoshi",
        "airdrop",
        "smart contract",
        "bull run",
    ],
    "health": [
        "health",
        "medicine",
        "medical",
        "doctor",
        "hospital",
        "virus",
        "disease",
        "cancer",
        "vaccine",
        "drug",
        "pharmaceutical",
        "fda",
        "covid",
        "pandemic",
        "therapy",
        "surgery",
        "mental health",
        "diet",
        "nutrition",
        "obesity",
        "who",
        "treatment",
    ],
    "international": [
        "international",
        "global",
        "war",
        "conflict",
        "ukraine",
        "russia",
        "israel",
        "gaza",
        "china",
        "un",
        "united nations",
        "nato",
        "treaty",
        "diplomacy",
        "foreign",
        "border",
        "geopolitics",
        "summit",
        "sanction",
        "ambassador",
        "territory",
        # merged from social
        "social",
        "society",
        "demographic",
        "population",
        "census",
        "birth rate",
        "inequality",
        "human rights",
        "protest",
        "civil rights",
        "gender",
        "race",
        "immigration",
        "poverty",
        "class",
        "community",
        "homelessness",
        "socio-economic",
        "student",
    ],
    "pets": [
        "pet",
        "pets",
        "dog",
        "cat",
        "puppy",
        "kitten",
        "veterinarian",
        "vet",
        "breed",
        "animal shelter",
        "adoption",
        "kibble",
        "leash",
        "domestic animal",
        # merged from animals
        "animal",
        "wildlife",
        "zoo",
        "species",
        "extinction",
        "wildlife conservation",
        "nature conservation",
        "lion",
        "tiger",
        "whale",
        "bear",
        "biodiversity",
        "safari",
        "jungle",
        "forest",
        "fauna",
        "marine",
    ],
    "politics": [
        "politics",
        "political",
        "election",
        "vote",
        "poll",
        "ballot",
        "democrat",
        "republican",
        "congress",
        "senate",
        "parliament",
        "president",
        "prime minister",
        "biden",
        "trump",
        "harris",
        "campaign",
        "legislation",
        "bill",
        "law",
        "supreme court",
        "governor",
        "mayor",
        "tory",
        "labour",
        "party",
        "impeachment",
        "regulatory",
        "uscis",
        "federal court",
    ],
    "science": [
        "science",
        "physics",
        "chemistry",
        "biology",
        "astronomy",
        "nasa",
        "space",
        "rocket",
        "spacex",
        "laboratory",
        "experiment",
        "discovery",
        "research",
        "scientist",
        "nobel prize",
        "atom",
        "molecule",
        "dna",
        "genetics",
        "telescope",
        "quantum",
        "fusion",
        "superconductor",
        "study",
        "peer-reviewed",
        "comet",
        "asteroid",
    ],
    "sports": [
        "sports",
        "sport",
        "football",
        "basketball",
        "soccer",
        "baseball",
        "nfl",
        "nba",
        "mlb",
        "fifa",
        "olympics",
        "world cup",
        "medal",
        "champion",
        "league",
        "sports team",
        "athlete",
        "score",
        "match",
        "tournament",
        "ufc",
        "boxing",
        "f1",
        "liverpool",
        "transfer",
        "player",
        "tennis",
        "grand slam",
    ],
    "sustainability": [
        "sustainability",
        "sustainable",
        "climate",
        "carbon",
        "green",
        "renewable",
        "solar",
        "wind",
        "energy",
        "electric vehicle",
        "ev",
        "emission",
        "pollution",
        "environment",
        "recycle",
        "plastic",
        "global warming",
        "net zero",
        "clean energy",
    ],
    "technology": [
        "technology",
        "tech",
        "ai",
        "artificial intelligence",
        "gpt",
        "llm",
        "software",
        "hardware",
        "app",
        "google",
        "apple",
        "microsoft",
        "meta",
        "server",
        "cloud",
        "algorithm",
        "robot",
        "cyber",
        "silicon",
        "chip",
        "semiconductor",
        "nvidia",
        "virtual reality",
        "metaverse",
        "device",
        "smartphone",
        "adobe",
        "semrush",
        # merged from internet
        "internet",
        "website",
        "domain",
        "url",
        "broadband",
        "fiber",
        "wifi",
        "5g",
        "browser",
        "search engine",
        "online",
        "digital",
        "connectivity",
        "network",
        "router",
        "isp",
        "cybersecurity",
        "hack",
        "ddos",
    ],
    "travel": [
        "travel",
        "tourism",
        "airline",
        "flight",
        "airport",
        "plane",
        "boeing",
        "airbus",
        "hotel",
        "resort",
        "visa",
        "passport",
        "destination",
        "cruise",
        "vacation",
        "booking",
        "airbnb",
        "expedia",
        "trip",
        "passenger",
        "transportation",
        "tour",
        "bus",
        "ntsb",
    ],
    "weather": [
        "weather",
        "forecast",
        "hurricane",
        "storm",
        "tornado",
        "temperature",
        "rain",
        "snow",
        "heatwave",
        "drought",
        "flood",
        "meteorology",
        "monsoon",
        "el nino",
        "tropical",
        "dissipate",
        "noaa",
    ],
}

# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

# Marketplace: bulk fetch all recent deliveries with prediction data
DELIVERS_QUERY = """
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

DELIVERS_BY_IDS_QUERY = """
{
  delivers(
    first: %(first)s
    where: { id_in: [%(ids)s] }
  ) {
    id
    marketplaceDelivery {
      ipfsHashBytes
      mechServiceMultisig
    }
  }
}
"""

# Omen: bulk fetch bets on markets that resolved after the cutoff.
# Filters by resolution time (currentAnswerTimestamp), not bet placement time.
OMEN_BETS_QUERY = """
{
  bets(
    first: %(first)s
    skip: %(skip)s
    orderBy: timestamp
    orderDirection: desc
    where: {
      fixedProductMarketMaker_: {
        currentAnswer_not: null
        currentAnswerTimestamp_gt: %(resolved_after)s
      }
    }
  ) {
    id
    timestamp
    outcomeIndex
    fixedProductMarketMaker {
      id
      currentAnswer
      currentAnswerTimestamp
      question
    }
  }
}
"""

# Polymarket: bulk fetch bets from a wide window, post-filter for resolved.
# The subgraph doesn't support filtering by resolution timestamp directly,
# so we fetch a broad candidate window and filter in Python.
POLYMARKET_BETS_QUERY = """
{
  bets(
    first: %(first)s
    skip: %(skip)s
    orderBy: blockTimestamp
    orderDirection: desc
    where: { blockTimestamp_gt: %(timestamp_gt)s }
  ) {
    id
    blockTimestamp
    outcomeIndex
    question {
      id
      metadata {
        title
        outcomes
      }
      resolution {
        winningIndex
        blockTimestamp
      }
    }
  }
}
"""

# How far back to fetch Polymarket bets to find recently resolved markets.
# Bets may have been placed months before the market resolves.
POLYMARKET_CANDIDATE_WINDOW_DAYS = 30

# ---------------------------------------------------------------------------
# Subgraph helpers
# ---------------------------------------------------------------------------


MAX_RETRIES = 3


def _post_graphql(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Post a GraphQL query and return the JSON response data. Retries on timeout."""
    headers = {"Content-Type": "application/json"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=HTTP_TIMEOUT
            )
            resp.raise_for_status()
            body = resp.json()
            if "errors" in body:
                raise RuntimeError(f"GraphQL errors from {url}: {body['errors']}")
            return body.get("data", {})
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
    return {}  # unreachable, but satisfies mypy


def _paginated_fetch(
    url: str,
    query_template: str,
    entity_key: str,
    template_vars: dict[str, Any],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Fetch all records from a subgraph using pagination."""
    all_records: list[dict[str, Any]] = []
    skip = 0

    while True:
        query = query_template % {**template_vars, "first": batch_size, "skip": skip}
        data = _post_graphql(url, {"query": query})
        batch = data.get(entity_key, [])
        if not batch:
            break
        all_records.extend(batch)
        log.info("  fetched %d %s (total %d)", len(batch), entity_key, len(all_records))
        if len(batch) < batch_size:
            break
        skip += batch_size

    return all_records


# ---------------------------------------------------------------------------
# Fetch deliveries (marketplace subgraphs)
# ---------------------------------------------------------------------------


def _parse_request_context(content_str: str) -> dict[str, Any]:
    """Parse request_context from parsedRequest.content JSON.

    Returns dict with market_id, market_type, market_prob, market_liquidity_usd,
    market_close_at if present (schema_version 2.0+). Empty dict otherwise.

    :param content_str: raw JSON string from parsedRequest.content.
    :return: dict with parsed market context fields, or empty dict.
    """
    if not content_str:
        return {}
    try:
        content = json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        return {}
    ctx = content.get("request_context")
    if not isinstance(ctx, dict):
        return {}
    return {
        "market_id": ctx.get("market_id"),
        "market_type": ctx.get("type"),
        "market_prob": ctx.get("market_prob"),
        "market_liquidity_usd": ctx.get("market_liquidity_usd"),
        "market_close_at": ctx.get("market_close_at"),
        "market_spread": ctx.get("market_spread"),
    }


def fetch_deliveries(
    marketplace_url: str,
    timestamp_gt: int,
) -> list[dict[str, Any]]:
    """Bulk fetch all recent deliveries with prediction data.

    Skips deliveries with null parsedRequest (IPFS failures on subgraph side).
    Extracts market_id from request_context when available (schema v2.0+).

    :param marketplace_url: GraphQL endpoint for the marketplace subgraph.
    :param timestamp_gt: only fetch deliveries after this UNIX timestamp.
    :return: list of delivery dicts.
    """
    raw = _paginated_fetch(
        marketplace_url,
        DELIVERS_QUERY,
        "delivers",
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

        question_title = _extract_question_title(parsed.get("questionTitle", ""))
        if not question_title:
            skipped += 1
            continue

        request_ts = int(request.get("blockTimestamp") or 0) or None
        delivery_ts = int(d["blockTimestamp"])
        ctx = _parse_request_context(parsed.get("content", ""))

        deliveries.append(
            {
                "deliver_id": d["id"],
                "timestamp": delivery_ts,
                "request_timestamp": request_ts,
                "model": d.get("model"),
                "tool_response": d.get("toolResponse"),
                "tool": parsed.get("tool") or "unknown",
                "question_title": question_title,
                "market_id": ctx.get("market_id"),
                "market_prob": ctx.get("market_prob"),
                "market_liquidity_usd": ctx.get("market_liquidity_usd"),
                "market_close_at": ctx.get("market_close_at"),
                "market_spread": ctx.get("market_spread"),
            }
        )

    if skipped:
        log.info("  skipped %d deliveries with null parsedRequest", skipped)

    has_market_id = sum(1 for d in deliveries if d["market_id"])
    if deliveries:
        log.info("  %d/%d deliveries have market_id", has_market_id, len(deliveries))

    return deliveries


# ---------------------------------------------------------------------------
# IPFS payload fetch (for source_content extraction)
# ---------------------------------------------------------------------------


def _ipfs_hash_to_cid(ipfs_hash: str) -> str:
    """Convert an IPFS hash from the subgraph to a base32 CIDv1.

    Handles two formats:
    - ``0x`` raw sha256 hash (from ``marketplaceDelivery.ipfsHashBytes``):
      wraps as CIDv1 (version=1, codec=dag-pb, sha256 multihash).
    - ``f``-prefixed hex CIDv1 (from ``mechDelivery.ipfsHash``):
      strips multibase prefix and re-encodes.

    The IPFS gateway expects base32 CIDv1 like ``bafybei...``.

    :param ipfs_hash: hex-encoded IPFS hash from the subgraph.
    :return: base32-encoded CIDv1 string.
    """
    if ipfs_hash.startswith("0x"):
        hash_bytes = bytes.fromhex(ipfs_hash[2:])
        # CIDv1: version=0x01, codec=0x70 (dag-pb), multihash=0x12 (sha256) + 0x20 (32 bytes) + hash
        cid_bytes = bytes([0x01, 0x70, 0x12, 0x20]) + hash_bytes
        return "b" + base64.b32encode(cid_bytes).decode().lower().rstrip("=")
    if ipfs_hash.startswith("f"):
        raw = bytes.fromhex(ipfs_hash[1:])
        return "b" + base64.b32encode(raw).decode().lower().rstrip("=")
    return ipfs_hash


def fetch_ipfs_metadata(ipfs_hash: str) -> Optional[dict[str, Any]]:
    """Fetch the full IPFS payload and return the metadata dict.

    The IPFS hash from the subgraph is a hex-encoded CIDv1 pointing to a
    directory. The directory contains one file named by request ID.

    :param ipfs_hash: hex-encoded IPFS hash from the subgraph.
    :return: the full metadata dict, or None if not available.
    """
    try:
        cid = _ipfs_hash_to_cid(ipfs_hash)
        dir_url = f"{IPFS_GATEWAY_URL}/{cid}/"

        # Fetch directory listing to find the file name
        resp = requests.get(dir_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()

        # Try parsing as JSON directly (in case gateway returns JSON directory listing)
        try:
            dir_data = resp.json()
            # If the response is already the payload (single-file directory auto-resolved)
            if isinstance(dir_data, dict) and "metadata" in dir_data:
                return dir_data.get("metadata") or {}
        except (json.JSONDecodeError, ValueError):
            pass

        # Parse HTML directory listing to find the file link
        links = re.findall(rf"/ipfs/{re.escape(cid)}/(\d+)", resp.text)
        if not links:
            log.debug("No files found in IPFS directory %s", cid)
            return None

        # Fetch the actual file (first match — there's typically one file per delivery)
        file_url = f"{IPFS_GATEWAY_URL}/{cid}/{links[0]}"
        file_resp = requests.get(file_url, timeout=HTTP_TIMEOUT)
        file_resp.raise_for_status()
        payload = file_resp.json()

        return payload.get("metadata") or {}
    except Exception as e:
        log.debug("Failed to fetch IPFS %s: %s", ipfs_hash, e)
        return None


def fetch_ipfs_source_content(ipfs_hash: str) -> Optional[dict[str, Any]]:
    """Fetch the IPFS payload and extract source_content from metadata.params.

    :param ipfs_hash: hex-encoded IPFS hash from the subgraph.
    :return: the source_content dict, or None if not available.
    """
    metadata = fetch_ipfs_metadata(ipfs_hash)
    if metadata is None:
        return None
    params = metadata.get("params") or {}
    return params.get("source_content")


# ---------------------------------------------------------------------------
# Fetch resolved markets (prediction subgraphs)
# ---------------------------------------------------------------------------


class ResolvedMarkets:
    """Resolved markets indexed by both market_id and question title."""

    def __init__(self) -> None:
        """Initialize empty market indexes."""
        self.by_id: dict[str, dict[str, Any]] = {}
        self.by_title: dict[str, dict[str, Any]] = {}
        self._seen: set[int] = set()

    def add(self, market_id: Optional[str], title: str, data: dict[str, Any]) -> None:
        """Add a resolved market to the indexes."""
        if market_id:
            self.by_id[market_id] = data
        if title:
            self.by_title[title.lower()] = data
        self._seen.add(id(data))

    def __len__(self) -> int:
        """Return the number of unique resolved markets."""
        return len(self._seen)

    def __bool__(self) -> bool:
        """Return True if any resolved markets are stored."""
        return len(self._seen) > 0


def fetch_omen_resolved(resolved_after: int) -> ResolvedMarkets:
    """Bulk fetch Omen markets that resolved after the given timestamp.

    Filters by resolution time (currentAnswerTimestamp), not bet placement time.
    Indexes by both market ID (fpmm address) and question title.

    :param resolved_after: UNIX timestamp; only include markets resolved after this.
    :return: ResolvedMarkets indexed by ID and title.
    """
    raw = _paginated_fetch(
        PREDICT_OMEN_SUBGRAPH_URL,
        OMEN_BETS_QUERY,
        "bets",
        {"resolved_after": resolved_after},
    )

    markets = ResolvedMarkets()
    for bet in raw:
        fpmm = bet.get("fixedProductMarketMaker") or {}
        current_answer = fpmm.get("currentAnswer")
        if current_answer is None:
            continue

        try:
            outcome = int(current_answer, 16)
        except (ValueError, TypeError):
            continue

        question_raw = fpmm.get("question", "")
        title = _extract_question_title(question_raw)
        if not title:
            continue

        resolved_at_ts = fpmm.get("currentAnswerTimestamp")
        data = {
            "outcome": outcome == 0,  # outcomes=["Yes","No"], index 0 = Yes
            "resolved_at_ts": int(resolved_at_ts) if resolved_at_ts else None,
        }

        # Omen market ID is the fpmm contract address (the bet entity's id prefix)
        # but the fpmm id from the subgraph is the FixedProductMarketMakerCreation id
        # which matches request_context.market_id
        market_id = fpmm.get("id")
        markets.add(market_id, title, data)

    return markets


def fetch_polymarket_resolved(resolved_after: int) -> ResolvedMarkets:
    """Bulk fetch Polymarket markets that resolved after the given timestamp.

    The subgraph doesn't support filtering bets by resolution time, so we:
    1. Fetch a wide candidate window (POLYMARKET_CANDIDATE_WINDOW_DAYS) of bets
    2. Post-filter to resolved questions only
    3. Only include markets where resolution.blockTimestamp > resolved_after
    4. Deduplicate by question ID

    :param resolved_after: UNIX timestamp; only include markets resolved after this.
    :return: ResolvedMarkets indexed by ID and title.
    """
    candidate_window = int(time.time()) - (POLYMARKET_CANDIDATE_WINDOW_DAYS * 86400)
    raw = _paginated_fetch(
        PREDICT_POLYMARKET_SUBGRAPH_URL,
        POLYMARKET_BETS_QUERY,
        "bets",
        {"timestamp_gt": candidate_window},
    )

    markets = ResolvedMarkets()
    for bet in raw:
        question = bet.get("question") or {}
        resolution = question.get("resolution")
        if resolution is None:
            continue

        resolved_at_ts = resolution.get("blockTimestamp")
        if not resolved_at_ts:
            continue

        # Only include markets that resolved after our cutoff
        resolved_at_int = int(resolved_at_ts)
        if resolved_at_int <= resolved_after:
            continue

        metadata = question.get("metadata") or {}
        title = (metadata.get("title") or "").strip()
        if not title:
            continue

        winning_index = int(resolution["winningIndex"])
        # winningIndex follows CLOB token order: 0 = Yes, 1 = No.
        # The subgraph outcomes array is unreliable (often ["No", "Yes"]),
        # so we ignore it and use the index directly.
        outcome = winning_index == 0

        data = {
            "outcome": outcome,
            "resolved_at_ts": resolved_at_int,
        }

        # Polymarket question ID matches request_context.market_id
        market_id = question.get("id")
        markets.add(market_id, title, data)

    return markets


# ---------------------------------------------------------------------------
# Question matching
# ---------------------------------------------------------------------------


def _extract_question_title(question: str) -> str:
    """Extract question title using the separator from production code."""
    if not question:
        return ""
    return question.split(QUESTION_DATA_SEPARATOR)[0].strip()


def _match_delivery(
    delivery: dict[str, Any],
    markets: ResolvedMarkets,
) -> tuple[Optional[dict[str, Any]], float]:
    """Match a delivery to a resolved market.

    Tries market_id first (deterministic), falls back to title matching (heuristic).
    Returns (market_data, match_confidence).

    :param delivery: delivery dict with question_title and optional market_id.
    :param markets: resolved markets to match against.
    :return: tuple of (market_data or None, match_confidence).
    """
    # 1. Deterministic match via market_id (from request_context, schema v2.0+)
    market_id = delivery.get("market_id")
    if market_id and market_id in markets.by_id:
        return markets.by_id[market_id], 1.0

    # 2. Fallback: title matching (for older requests without market_id)
    key = delivery["question_title"].lower()

    # Exact title match
    if key in markets.by_title:
        return markets.by_title[key], 1.0

    # Prefix match (min 20 chars to avoid false positives)
    if len(key) >= 20:
        for market_title, market_data in markets.by_title.items():
            if len(market_title) >= 20 and (
                key.startswith(market_title) or market_title.startswith(key)
            ):
                return market_data, 0.8

    return None, 0.0


# ---------------------------------------------------------------------------
# Tool response parsing
# ---------------------------------------------------------------------------


def parse_tool_response(tool_response: Optional[str]) -> dict[str, Any]:
    """Parse a toolResponse JSON string into p_yes, p_no, and parse status."""
    if not tool_response:
        return {
            "p_yes": None,
            "p_no": None,
            "confidence": None,
            "prediction_parse_status": "missing_fields",
        }

    # Check for known IPFS retrieval error messages (only short non-JSON responses)
    if len(tool_response) < 300 and tool_response.lstrip()[:1] != "{":
        lower = tool_response.lower()
        if "could not be retrieved" in lower or "failed to download" in lower:
            return {
                "p_yes": None,
                "p_no": None,
                "confidence": None,
                "prediction_parse_status": "error",
            }

    # Strategy 1: Direct JSON parse
    try:
        data = json.loads(tool_response)
        if isinstance(data, dict):
            p_yes = data.get("p_yes")
            p_no = data.get("p_no")

            if p_yes is not None and p_no is not None:
                p_yes = float(p_yes)
                p_no = float(p_no)

                if (
                    0.0 <= p_yes <= 1.0
                    and 0.0 <= p_no <= 1.0
                    and abs(p_yes + p_no - 1.0) <= PROBABILITY_SUM_TOLERANCE
                ):
                    return {
                        "p_yes": p_yes,
                        "p_no": p_no,
                        "confidence": (
                            float(data["confidence"])
                            if data.get("confidence") is not None
                            else None
                        ),
                        "prediction_parse_status": "valid",
                    }

            return {
                "p_yes": None,
                "p_no": None,
                "confidence": None,
                "prediction_parse_status": "malformed",
            }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Strategy 2: Regex extraction
    p_yes_match = re.search(r'"p_yes"\s*:\s*([\d.]+)', tool_response)
    p_no_match = re.search(r'"p_no"\s*:\s*([\d.]+)', tool_response)

    if p_yes_match and p_no_match:
        try:
            p_yes = float(p_yes_match.group(1))
            p_no = float(p_no_match.group(1))
            if (
                0.0 <= p_yes <= 1.0
                and 0.0 <= p_no <= 1.0
                and abs(p_yes + p_no - 1.0) <= PROBABILITY_SUM_TOLERANCE
            ):
                return {
                    "p_yes": p_yes,
                    "p_no": p_no,
                    "confidence": None,
                    "prediction_parse_status": "valid",
                }
        except ValueError:
            pass

    return {
        "p_yes": None,
        "p_no": None,
        "confidence": None,
        "prediction_parse_status": "malformed",
    }


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------


def classify_category(question_text: str, platform: Optional[str] = None) -> str:
    """Classify a question into a category using word-boundary keyword matching.

    When ``platform`` is provided, the classified category is filtered
    against that platform's upstream taxonomy
    (``PLATFORM_ALLOWED_CATEGORIES``); a keyword match for a category the
    platform never emits (e.g. ``travel`` for omen, or ``curiosities``
    for polymarket) drops to ``"other"`` so per-platform reports don't
    show categories the platform doesn't actually trade.

    :param question_text: market question, used for keyword matching.
    :param platform: scorer platform key (``"omen"`` or ``"polymarket"``).
        ``None`` is accepted for callers that don't yet know the
        platform — the classifier behaves as before.
    :return: category name, or ``"other"`` when no keyword matches or
        the matched category is outside ``platform``'s allowed set.
    """
    text_lower = question_text.lower()
    matched = "other"
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                matched = category
                break
        if matched != "other":
            break

    if platform is None:
        return matched
    allowed = PLATFORM_ALLOWED_CATEGORIES.get(platform)
    if allowed is None or matched in allowed:
        return matched
    return "other"


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _ts_to_iso(ts: Optional[int]) -> Optional[str]:
    """Convert a unix timestamp to ISO 8601 UTC string."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compute_config_hash(
    tool_hash: Optional[str],
    model: Optional[str],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """Compute a deterministic config hash from tool + model parameters.

    :param tool_hash: IPFS tool hash (e.g. ``bafyabc123``).
    :param model: model name (e.g. ``gpt-4.1``).
    :param temperature: optional temperature setting.
    :param max_tokens: optional max tokens setting.
    :return: first 12 chars of SHA256 hex digest, or None if no inputs.
    """
    if tool_hash is None and model is None:
        return None
    parts = [
        "" if tool_hash is None else str(tool_hash),
        "" if model is None else str(model),
        "" if temperature is None else str(temperature),
        "" if max_tokens is None else str(max_tokens),
    ]
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
    return h


# TODO: unify _make_row_id across runner, tournament, prompt_replay
# & fetch_production into benchmark/tools.py
def _make_row_id(platform: str, deliver_id: str) -> str:
    """Generate a deterministic row ID from platform + deliver_id."""
    h = hashlib.sha256(f"{platform}:{deliver_id}".encode()).hexdigest()[:12]
    return f"prod_{platform}_{h}"


def build_row(
    delivery: dict[str, Any],
    market: dict[str, Any],
    match_confidence: float,
    platform: str,
    ipfs_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a production_log row from a delivery matched to a resolved market.

    :param delivery: delivery dict from the subgraph.
    :param market: resolved market dict.
    :param match_confidence: confidence of the delivery-to-market match.
    :param platform: platform name (e.g. "omen", "polymarket").
    :param ipfs_metadata: optional IPFS metadata dict with tool_hash and params.
    :return: production log row dict.
    """
    question_text = delivery["question_title"]
    parsed = parse_tool_response(delivery["tool_response"])
    delivery_ts = delivery["timestamp"]
    request_ts = delivery.get("request_timestamp")
    resolved_at_ts = market["resolved_at_ts"]

    prediction_lead_time_days: Optional[float] = None
    if delivery_ts and resolved_at_ts and resolved_at_ts > delivery_ts:
        prediction_lead_time_days = round((resolved_at_ts - delivery_ts) / 86400, 1)

    # Block-level granularity (~5s Gnosis, ~12s Ethereum), not sub-second
    latency_s: Optional[int] = None
    if request_ts and delivery_ts and delivery_ts > request_ts:
        latency_s = delivery_ts - request_ts

    # Extract tool_version and config_hash from IPFS metadata if available
    tool_version: Optional[str] = None
    config_hash: Optional[str] = None
    if ipfs_metadata:
        tool_version = ipfs_metadata.get("tool_hash")
        params = ipfs_metadata.get("params") or {}
        config_hash = _compute_config_hash(
            tool_version,
            delivery["model"],
            params.get("temperature"),
            params.get("max_tokens"),
        )

    row = {
        "row_id": _make_row_id(platform, delivery["deliver_id"]),
        "deliver_id": delivery["deliver_id"],
        "schema_version": "1.0",
        "mode": "production_replay",
        "market_id": delivery.get("market_id"),
        "platform": platform,
        "question_text": question_text,
        "tool_name": delivery["tool"],
        "tool_version": tool_version,
        "model": delivery["model"],
        "config_hash": config_hash,
        "p_yes": parsed["p_yes"],
        "p_no": parsed["p_no"],
        "prediction_parse_status": parsed["prediction_parse_status"],
        "market_prob_at_prediction": delivery.get("market_prob"),
        "market_liquidity_at_prediction": delivery.get("market_liquidity_usd"),
        "market_spread_at_prediction": delivery.get("market_spread"),
        "market_close_at": delivery.get("market_close_at"),
        "final_outcome": market["outcome"],
        "requested_at": _ts_to_iso(request_ts),
        "predicted_at": _ts_to_iso(delivery_ts),
        "resolved_at": _ts_to_iso(resolved_at_ts),
        "latency_s": latency_s,
        "prediction_lead_time_days": prediction_lead_time_days,
        "category": classify_category(question_text, platform),
        "match_confidence": match_confidence,
    }
    return row


# ---------------------------------------------------------------------------
# Incremental state & deduplication
# ---------------------------------------------------------------------------


def load_fetch_state(state_path: Path) -> dict[str, Any]:
    """Load incremental fetch state from disk."""
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning(
                "Could not read fetch state from %s, starting fresh", state_path
            )
    return {}


def save_fetch_state(state_path: Path, state: dict[str, Any]) -> None:
    """Save incremental fetch state to disk."""
    state_path.write_text(json.dumps(state, indent=2))


def daily_log_path(logs_dir: Path, date: datetime | None = None) -> Path:
    """Return the daily log file path for the given date (default: today UTC).

    :param logs_dir: directory containing daily log files.
    :param date: date to use; defaults to today in UTC.
    :return: path like ``logs/production_log_2026_04_06.jsonl``.
    """
    if date is None:
        date = datetime.now(timezone.utc)
    return logs_dir / f"production_log_{date.strftime('%Y_%m_%d')}.jsonl"


def _daily_log_files(logs_dir: Path, n_days: int) -> list[Path]:
    """Return the last *n_days* daily log file paths that exist on disk.

    Checks today and the previous *n_days - 1* days (UTC).

    :param logs_dir: directory containing daily log files.
    :param n_days: number of days to look back (inclusive of today).
    :return: list of existing daily log file paths.
    """
    now = datetime.now(timezone.utc)
    paths: list[Path] = []
    for i in range(n_days):
        p = daily_log_path(logs_dir, now - timedelta(days=i))
        if p.exists():
            paths.append(p)
    return paths


def _load_ids_from_file(path: Path) -> set[str]:
    """Extract row IDs from a single JSONL file."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                ids.add(row["row_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def load_existing_row_ids(
    logs_dir: Path,
    *,
    state_loss: bool = False,
) -> set[str]:
    """Load existing row IDs from daily log files for deduplication.

    Normal case: reads only today's file (handles cursor overlap and
    same-day reruns).  State-loss recovery: reads the last
    ``DEDUP_LOOKBACK_DAYS`` daily files to catch duplicates across
    the lookback window.

    :param logs_dir: directory containing daily log files.
    :param state_loss: if True, widen dedup scope to last 7 days.
    :return: set of row IDs already written.
    """
    if state_loss:
        files = _daily_log_files(logs_dir, DEDUP_LOOKBACK_DAYS)
    else:
        files = [daily_log_path(logs_dir)]
        if not files[0].exists():
            files = []
    ids: set[str] = set()
    for path in files:
        ids |= _load_ids_from_file(path)
    return ids


def append_rows(output_path: Path, rows: list[dict[str, Any]]) -> int:
    """Append rows to the output JSONL file. Returns count of rows written."""
    return append_jsonl(output_path, rows)


# ---------------------------------------------------------------------------
# Pipeline: process one platform
# ---------------------------------------------------------------------------


def _match_and_build(
    deliveries: list[dict[str, Any]],
    resolved_markets: ResolvedMarkets,
    existing_ids: set[str],
    platform: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, int, int]:
    """Match deliveries to resolved markets and build rows.

    Returns (rows, still_pending, matched_by_id, matched_by_title,
             max_delivery_ts, max_resolved_ts).

    :param deliveries: list of delivery dicts.
    :param resolved_markets: resolved markets to match against.
    :param existing_ids: row IDs already written, for deduplication.
    :param platform: platform name (e.g. "omen", "polymarket").
    :return: tuple of (rows, still_pending, matched_by_id, matched_by_title,
        max_delivery_ts, max_resolved_ts).
    """
    rows: list[dict[str, Any]] = []
    still_pending: list[dict[str, Any]] = []
    matched_by_id = 0
    matched_by_title = 0
    max_delivery_ts = 0
    max_resolved_ts = 0

    for delivery in deliveries:
        row_id = _make_row_id(platform, delivery["deliver_id"])
        if row_id in existing_ids:
            continue

        market, confidence = _match_delivery(delivery, resolved_markets)
        if market is None:
            still_pending.append(delivery)
            continue

        if delivery.get("market_id") and confidence == 1.0:
            matched_by_id += 1
        else:
            matched_by_title += 1

        row = build_row(delivery, market, confidence, platform)
        rows.append(row)
        existing_ids.add(row_id)
        max_delivery_ts = max(max_delivery_ts, delivery["timestamp"])
        if market.get("resolved_at_ts"):
            max_resolved_ts = max(max_resolved_ts, market["resolved_at_ts"])

    return (
        rows,
        still_pending,
        matched_by_id,
        matched_by_title,
        max_delivery_ts,
        max_resolved_ts,
    )


def _fetch_delivery_info(
    deliver_ids: list[str],
    marketplace_url: str,
) -> dict[str, dict[str, Optional[str]]]:
    """Query the subgraph for IPFS hashes and mech addresses by deliver IDs.

    :param deliver_ids: list of deliver IDs to look up.
    :param marketplace_url: subgraph endpoint URL.
    :return: dict mapping deliver_id to
        ``{"ipfs_hash": ..., "mech": ...}``.
    """
    result: dict[str, dict[str, Optional[str]]] = {}
    for i in range(0, len(deliver_ids), DEFAULT_BATCH_SIZE):
        batch = deliver_ids[i : i + DEFAULT_BATCH_SIZE]
        ids_str = ", ".join(f'"{did}"' for did in batch)
        query = DELIVERS_BY_IDS_QUERY % {"first": len(batch), "ids": ids_str}

        try:
            resp = requests.post(
                marketplace_url,
                json={"query": query},
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {}).get("delivers", [])
            for d in data:
                mp = d.get("marketplaceDelivery") or {}
                result[d["id"]] = {
                    "ipfs_hash": mp.get("ipfsHashBytes"),
                    "mech": mp.get("mechServiceMultisig"),
                }
        except Exception as e:
            log.warning("Failed to fetch delivery info from subgraph: %s", e)
            for did in batch:
                result[did] = {"ipfs_hash": None, "mech": None}

    return result


def _fetch_tool_hash(ipfs_hash: str) -> Optional[str]:
    """Fetch tool_hash from IPFS metadata for a single delivery.

    :param ipfs_hash: hex-encoded IPFS hash from the subgraph.
    :return: tool_hash string, or None on failure.
    """
    metadata = fetch_ipfs_metadata(ipfs_hash)
    if not metadata:
        return None
    return metadata.get("tool_hash")


def _resolve_group_tool_hash(
    group_rows: list[dict[str, Any]],
    delivery_info: dict[str, dict[str, Optional[str]]],
) -> dict[int, Optional[str]]:
    """Resolve tool_hash for a (mech, tool) group using sample + binary search.

    Samples the earliest and latest delivery in the group. If both return
    the same tool_hash, applies it to all rows. If they differ (tool updated
    mid-batch), uses binary search to find the boundary.

    :param group_rows: rows in this group, sorted by timestamp.
    :param delivery_info: mapping of deliver_id to ipfs_hash/mech info.
    :return: dict mapping row index (in group_rows) to tool_hash.
    """
    result: dict[int, Optional[str]] = {}
    if not group_rows:
        return result

    def _get_hash(idx: int) -> Optional[str]:
        """Fetch tool_hash for the row at index idx."""
        row = group_rows[idx]
        info = delivery_info.get(row.get("deliver_id", ""), {})
        ipfs_hash = info.get("ipfs_hash")
        if not ipfs_hash:
            return None
        return _fetch_tool_hash(ipfs_hash)

    # Sample earliest and latest
    first_hash = _get_hash(0)
    last_hash = _get_hash(len(group_rows) - 1) if len(group_rows) > 1 else first_hash

    # If either sample failed (gateway error), use whichever succeeded.
    # If both match (or one is None), apply uniformly — no binary search.
    uniform_hash = None
    if first_hash is None and last_hash is None:
        return result  # both failed — leave all unknown
    if first_hash is None or last_hash is None or first_hash == last_hash:
        uniform_hash = first_hash or last_hash
        for i in range(len(group_rows)):
            result[i] = uniform_hash
        return result

    # Both non-None but different — tool genuinely changed mid-batch
    log.info(
        "IPFS enrichment: tool version changed mid-batch (%s → %s), "
        "binary searching %d rows",
        first_hash,
        last_hash,
        len(group_rows),
    )
    lo, hi = 0, len(group_rows) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        mid_hash = _get_hash(mid)
        if mid_hash == first_hash:
            lo = mid
        else:
            hi = mid

    # lo is last row with first_hash, hi is first row with last_hash
    for i in range(len(group_rows)):
        result[i] = first_hash if i <= lo else last_hash

    return result


def _enrich_rows_with_ipfs_metadata(
    rows: list[dict[str, Any]],
    marketplace_url: str,
) -> None:
    """Enrich matched rows with tool_version and config_hash from IPFS.

    Groups deliveries by (mech_address, tool_name) and samples the
    earliest + latest delivery per group to determine the tool_hash.
    If a version change is detected mid-batch, binary searches for
    the boundary. This reduces ~23K IPFS fetches to ~60-100.

    Mutates rows in place. Failures are silently skipped.

    :param rows: list of production log row dicts (mutated in place).
    :param marketplace_url: subgraph endpoint URL.
    """
    if not rows:
        return

    # Step 1: Batch fetch IPFS hashes + mech addresses from subgraph
    deliver_ids = [r["deliver_id"] for r in rows if r.get("deliver_id")]
    if not deliver_ids:
        return

    delivery_info = _fetch_delivery_info(deliver_ids, marketplace_url)
    has_hash = sum(1 for v in delivery_info.values() if v.get("ipfs_hash"))
    log.info(
        "IPFS enrichment: %d/%d deliveries have hashes",
        has_hash,
        len(deliver_ids),
    )

    # Step 2: Group rows by (mech, tool_name), skipping rows without IPFS data
    # Non-marketplace deliveries have no marketplaceDelivery record and
    # therefore no ipfs_hash or mech address — they stay tool_version=None.
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    skipped = 0
    for i, row in enumerate(rows):
        did = row.get("deliver_id", "")
        info = delivery_info.get(did, {})
        ipfs_hash = info.get("ipfs_hash")
        mech = info.get("mech")
        if not ipfs_hash or not mech:
            skipped += 1
            continue
        tool = row.get("tool_name", "unknown")
        key = (mech, tool)
        if key not in groups:
            groups[key] = []
        groups[key].append((i, row))

    if skipped:
        log.info(
            "IPFS enrichment: skipped %d rows without marketplace delivery",
            skipped,
        )

    log.info(
        "IPFS enrichment: %d (mech, tool) groups from %d rows",
        len(groups),
        len(rows),
    )

    # Step 3: For each group, resolve tool_hash via sample + binary search
    enriched = 0
    for (_mech, _tool), indexed_rows in groups.items():
        # Sort by delivery timestamp for correct binary search
        indexed_rows.sort(
            key=lambda x: x[1].get("predicted_at") or "",
        )
        group_rows = [r for _, r in indexed_rows]
        group_indices = [i for i, _ in indexed_rows]

        hash_map = _resolve_group_tool_hash(group_rows, delivery_info)

        for local_idx, tool_hash in hash_map.items():
            if tool_hash is None:
                continue
            row = rows[group_indices[local_idx]]
            config_hash = _compute_config_hash(
                tool_hash,
                row.get("model"),
            )
            row["tool_version"] = tool_hash
            row["config_hash"] = config_hash
            enriched += 1

    log.info("IPFS enrichment: %d/%d rows enriched", enriched, len(rows))


def process_platform(
    platform: str,
    marketplace_url: str,
    resolved_markets: ResolvedMarkets,
    delivery_ts_gt: int,
    existing_ids: set[str],
    pending_deliveries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Process one platform: fetch deliveries, match to resolved markets, build rows.

    Also retries previously pending (unmatched) deliveries against newly
    resolved markets. Unmatched deliveries are returned as still-pending
    for the next run.

    :param platform: platform name (e.g. "omen", "polymarket").
    :param marketplace_url: GraphQL endpoint for the marketplace subgraph.
    :param resolved_markets: resolved markets to match against.
    :param delivery_ts_gt: only fetch deliveries after this UNIX timestamp.
    :param existing_ids: row IDs already written, for deduplication.
    :param pending_deliveries: unmatched deliveries from previous runs.
    :return: tuple of (rows, still_pending, max_delivery_timestamp,
        max_resolved_timestamp).
    """
    # 1. Retry pending deliveries from previous runs
    rows_from_pending: list[dict[str, Any]] = []
    remaining_pending: list[dict[str, Any]] = []
    if pending_deliveries and resolved_markets:
        rows_from_pending, remaining_pending, _, _, _, _ = _match_and_build(
            pending_deliveries,
            resolved_markets,
            existing_ids,
            platform,
        )
        if rows_from_pending:
            log.info(
                "%s: matched %d previously pending deliveries",
                platform,
                len(rows_from_pending),
            )

    # 2. Fetch and process new deliveries
    log.info("%s: fetching deliveries...", platform)
    new_deliveries = fetch_deliveries(marketplace_url, delivery_ts_gt)
    log.info(
        "%s: %d new deliveries, %d resolved markets, %d pending from before",
        platform,
        len(new_deliveries),
        len(resolved_markets),
        len(pending_deliveries),
    )

    rows_from_new: list[dict[str, Any]] = []
    new_pending: list[dict[str, Any]] = []
    max_delivery_ts = 0
    max_resolved_ts = 0
    matched_by_id = 0
    matched_by_title = 0

    if new_deliveries:
        (
            rows_from_new,
            new_pending,
            matched_by_id,
            matched_by_title,
            max_delivery_ts,
            max_resolved_ts,
        ) = _match_and_build(new_deliveries, resolved_markets, existing_ids, platform)

    all_rows = rows_from_pending + rows_from_new
    all_pending = remaining_pending + new_pending

    # 3. Enrich matched rows with IPFS metadata (tool_version, config_hash)
    _enrich_rows_with_ipfs_metadata(all_rows, marketplace_url)

    # Prune old pending deliveries to keep state file small
    cutoff = int(time.time()) - (PENDING_MAX_AGE_DAYS * 86400)
    before_prune = len(all_pending)
    all_pending = [d for d in all_pending if d["timestamp"] > cutoff]
    pruned = before_prune - len(all_pending)
    if pruned:
        log.info(
            "%s: pruned %d pending deliveries older than %d days",
            platform,
            pruned,
            PENDING_MAX_AGE_DAYS,
        )

    total_matched = len(all_rows)
    log.info(
        "%s: %d matched (%d by market_id, %d by title, %d from pending), "
        "%d still pending, %d rows built",
        platform,
        total_matched,
        matched_by_id,
        matched_by_title,
        len(rows_from_pending),
        len(all_pending),
        len(all_rows),
    )
    return all_rows, all_pending, max_delivery_ts, max_resolved_ts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _migrate_legacy_log(legacy_path: Path, logs_dir: Path) -> None:
    """Move the old single-file production log into the daily logs directory.

    This is a one-time migration. After the move the legacy path no longer
    exists and all future reads go through the daily log directory.

    :param legacy_path: path to the old ``production_log.jsonl``.
    :param logs_dir: target ``logs/`` directory.
    """
    if not legacy_path.exists():
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    dest = logs_dir / "production_log_legacy.jsonl"
    shutil.move(str(legacy_path), str(dest))
    log.info("Migrated legacy log %s -> %s", legacy_path, dest)


def _update_platform_state(
    state: dict[str, Any],
    platform: str,
    prev_state: dict[str, Any],
    max_del_ts: int,
    max_res_ts: int,
    still_pending: list[dict[str, Any]],
    now: int,
) -> None:
    """Update incremental state for one platform.

    :param state: full state dict (mutated in place).
    :param platform: platform name (e.g. "omen", "polymarket").
    :param prev_state: previous state for this platform.
    :param max_del_ts: max delivery timestamp seen this run.
    :param max_res_ts: max resolved timestamp seen this run.
    :param still_pending: deliveries still unmatched.
    :param now: current UNIX timestamp.
    """
    if max_del_ts or max_res_ts or still_pending:
        state[platform] = {
            "last_delivery_timestamp": (
                (max_del_ts - 1)
                if max_del_ts
                else prev_state.get("last_delivery_timestamp", 0)
            ),
            "last_resolved_timestamp": (
                (max_res_ts - 1)
                if max_res_ts
                else prev_state.get("last_resolved_timestamp", 0)
            ),
            "pending_deliveries": still_pending,
            "last_run": _ts_to_iso(now),
        }


def _run_scorer_update(
    rows: list[dict[str, Any]],
    scores_path: Path,
    history_path: Path,
) -> None:
    """Run scorer.update() with fault isolation.

    Adds the project root to sys.path if needed (supports running as
    a script via ``python benchmark/datasets/fetch_production.py``).
    Catches all exceptions so fetch never crashes due to scoring bugs.

    :param rows: new production log rows.
    :param scores_path: path to scores.json.
    :param history_path: path to scores_history.jsonl.
    """
    try:
        _root = str(Path(__file__).resolve().parent.parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        # pylint: disable-next=import-outside-toplevel
        from benchmark.scorer import update as scorer_update

        scorer_update(rows, scores_path, history_path)
        log.info("Scores updated at %s", scores_path)
    except Exception:
        log.exception("Scorer update failed (non-fatal, fetch data is safe)")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for fetch_production.

    :return: configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Fetch production prediction data for benchmark scoring.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=int(os.environ.get("BENCHMARK_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)),
        help="How many days back to fetch (default: 7)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help="Directory for daily log files",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(__file__).parent / ".fetch_state.json",
        help="Incremental state file path",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Only process the last N rows (most recent first)",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "scores.json",
        help="Path to scores.json for incremental scoring",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "scores_history.jsonl",
        help="Path to scores_history.jsonl",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help=(
            "Skip the inline scorer.update() call. Use when the workflow "
            "will run `scorer --rebuild` explicitly afterwards."
        ),
    )
    return parser


def main() -> None:
    """CLI entry point for fetching production data."""
    args = _build_arg_parser().parse_args()

    # One-time migration: move old single-file log into logs/ directory
    legacy_path = args.logs_dir.parent / "production_log.jsonl"
    _migrate_legacy_log(legacy_path, args.logs_dir)

    now = int(time.time())
    lookback_ts = now - (args.lookback_days * 86400)

    state = load_fetch_state(args.state_file)
    state_loss = not state
    existing_ids = load_existing_row_ids(args.logs_dir, state_loss=state_loss)
    log.info(
        "Loaded %d existing row IDs for deduplication (state_loss=%s)",
        len(existing_ids),
        state_loss,
    )

    all_rows: list[dict[str, Any]] = []

    # Two separate cursors per platform:
    # - last_delivery_timestamp: for the marketplace delivery query
    # - last_resolved_timestamp: for the resolved markets query
    # These advance independently because deliveries and resolutions
    # happen at different times.

    # --- Omen (Gnosis chain) ---
    omen_state = state.get("omen", {})
    omen_delivery_ts = max(lookback_ts, omen_state.get("last_delivery_timestamp", 0))
    omen_resolved_ts = max(lookback_ts, omen_state.get("last_resolved_timestamp", 0))
    omen_pending = omen_state.get("pending_deliveries", [])
    log.info(
        "Omen: deliveries since %s, resolved since %s, %d pending",
        _ts_to_iso(omen_delivery_ts),
        _ts_to_iso(omen_resolved_ts),
        len(omen_pending),
    )
    omen_markets = fetch_omen_resolved(resolved_after=omen_resolved_ts)

    omen_rows, omen_still_pending, omen_max_del_ts, omen_max_res_ts = process_platform(
        "omen",
        MECH_MARKETPLACE_GNOSIS_URL,
        omen_markets,
        omen_delivery_ts,
        existing_ids,
        omen_pending,
    )
    all_rows.extend(omen_rows)

    # --- Polymarket (Polygon chain) ---
    poly_state = state.get("polymarket", {})
    poly_delivery_ts = max(lookback_ts, poly_state.get("last_delivery_timestamp", 0))
    poly_resolved_ts = max(lookback_ts, poly_state.get("last_resolved_timestamp", 0))
    poly_pending = poly_state.get("pending_deliveries", [])
    log.info(
        "Polymarket: deliveries since %s, resolved since %s, %d pending",
        _ts_to_iso(poly_delivery_ts),
        _ts_to_iso(poly_resolved_ts),
        len(poly_pending),
    )
    poly_markets = fetch_polymarket_resolved(resolved_after=poly_resolved_ts)

    poly_rows, poly_still_pending, poly_max_del_ts, poly_max_res_ts = process_platform(
        "polymarket",
        MECH_MARKETPLACE_POLYGON_URL,
        poly_markets,
        poly_delivery_ts,
        existing_ids,
        poly_pending,
    )
    all_rows.extend(poly_rows)

    # Apply --last-n truncation (rows are already newest-first from subgraph)
    if args.last_n is not None and len(all_rows) > args.last_n:
        log.info("Truncating to last %d rows (from %d)", args.last_n, len(all_rows))
        all_rows = all_rows[: args.last_n]

    # Write results to today's daily log file
    output_path = daily_log_path(args.logs_dir)
    if all_rows:
        written = append_rows(output_path, all_rows)
        log.info("Appended %d new rows to %s", written, output_path)

        # Incremental scoring — update accumulators in scores.json
        if args.no_score:
            log.info("Inline scoring skipped (--no-score)")
        else:
            _run_scorer_update(all_rows, args.scores, args.history)
    else:
        log.info("No new rows to write")

    # Update incremental state — separate cursors, subtract 1 for same-block safety.
    _update_platform_state(
        state,
        "omen",
        omen_state,
        omen_max_del_ts,
        omen_max_res_ts,
        omen_still_pending,
        now,
    )
    _update_platform_state(
        state,
        "polymarket",
        poly_state,
        poly_max_del_ts,
        poly_max_res_ts,
        poly_still_pending,
        now,
    )

    save_fetch_state(args.state_file, state)
    log.info("State saved to %s", args.state_file)

    valid = sum(1 for r in all_rows if r["prediction_parse_status"] == "valid")
    log.info(
        "Summary: %d total rows, %d valid predictions, %d missing/malformed",
        len(all_rows),
        valid,
        len(all_rows) - valid,
    )


if __name__ == "__main__":
    main()
