"""
Shared per-platform category taxonomies.

Mirrors the upstream platforms' own category lists so the benchmark
emits, scores, and reports categories the platforms actually trade.

Keep these in sync with:
    Omen:       valory-xyz/market-creator — DEFAULT_TOPICS in
                packages/valory/skills/market_creation_manager_abci/propose_questions.py
    Polymarket: valory-xyz/trader — POLYMARKET_CATEGORY_TAGS in
                packages/valory/connections/polymarket_client/connection.py

Imported by:
    - benchmark.analyze (weak-spot filter, per-platform report sections)
    - benchmark.datasets.fetch_production (platform-aware category classifier)
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

OMEN_CATEGORIES: frozenset[str] = frozenset(
    {
        "business",
        "cryptocurrency",
        "politics",
        "science",
        "technology",
        "trending",
        "social",
        "health",
        "sustainability",
        "internet",
        "food",
        "pets",
        "animals",
        "curiosities",
        "economy",
        "arts",
        "entertainment",
        "weather",
        "sports",
        "finance",
        "international",
    }
)

POLYMARKET_ACTIVE_CATEGORIES: frozenset[str] = frozenset(
    {
        "business",
        "politics",
        "science",
        "technology",
        "health",
        "entertainment",
        "weather",
        "finance",
        "international",
    }
)

ACTIVE_CATEGORIES: frozenset[str] = OMEN_CATEGORIES | POLYMARKET_ACTIVE_CATEGORIES


# Maps the scorer's platform key to the platform's allowed-category set.
# A row whose keyword-classified category is not in the platform's set
# should bucket as "other" — the upstream platform never emits that
# category, so reporting it as a category in that platform's report
# would mislead the reader.
PLATFORM_ALLOWED_CATEGORIES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "omen": OMEN_CATEGORIES,
        "polymarket": POLYMARKET_ACTIVE_CATEGORIES,
    }
)
