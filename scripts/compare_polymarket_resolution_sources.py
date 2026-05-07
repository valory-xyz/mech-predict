#!/usr/bin/env python3
"""Compare Polymarket resolution discovery through bets vs questions.

This diagnostic script starts from the benchmark fetch state file, extracts
pending Polymarket delivery ``market_id`` values, and compares two discovery
paths against the Polymarket prediction subgraph:

1. Current production path: ``bets -> question -> resolution`` via
   ``fetch_polymarket_resolved``.
2. Direct question path: ``question(id: delivery.market_id) -> resolution`` and
   ``questions(where: { resolution_: { blockTimestamp_gt: ... } })``.

The output is intended to show whether fetched Polymarket deliveries have valid
question IDs that are resolved but missed by the current bet-based discovery.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.datasets import fetch_production as fp  # noqa: E402


DEFAULT_STATE_FILE = REPO_ROOT / "benchmark" / "datasets" / ".fetch_state.json"
DEFAULT_BATCH_SIZE = 1000
QUESTION_ID_BATCH_SIZE = 25

POLYMARKET_QUESTIONS_QUERY = """
{
  questions(
    first: %(first)s
    skip: %(skip)s
    orderBy: id
    orderDirection: asc
    where: { resolution_: { blockTimestamp_gt: %(resolved_after)s } }
  ) {
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
"""


def _post_graphql(url: str, query: str, timeout: int) -> dict[str, Any]:
    """Post a GraphQL query and return the response data."""
    response = requests.post(
        url,
        json={"query": query},
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if "errors" in body:
        raise RuntimeError(f"GraphQL errors from {url}: {body['errors']}")
    return body.get("data", {})


def _fetch_questions_by_id(
    url: str,
    market_ids: list[str],
    timeout: int,
) -> dict[str, dict[str, Any] | None]:
    """Fetch direct question records for delivery market IDs."""
    questions: dict[str, dict[str, Any] | None] = {}
    for start in range(0, len(market_ids), QUESTION_ID_BATCH_SIZE):
        batch = market_ids[start : start + QUESTION_ID_BATCH_SIZE]
        fields = [
            f"q{i}: question(id: {json.dumps(market_id)}) "
            "{ id metadata { title outcomes } "
            "resolution { winningIndex blockTimestamp } }"
            for i, market_id in enumerate(batch)
        ]
        data = _post_graphql(url, "{ " + "\n".join(fields) + " }", timeout)
        for i, market_id in enumerate(batch):
            questions[market_id] = data.get(f"q{i}")
    return questions


def _fetch_resolved_questions(
    url: str,
    resolved_after: int,
    batch_size: int,
    timeout: int,
    delay_seconds: float,
) -> dict[str, dict[str, Any]]:
    """Fetch all resolved questions after a cutoff using direct question query."""
    questions: dict[str, dict[str, Any]] = {}
    skip = 0
    while True:
        query = POLYMARKET_QUESTIONS_QUERY % {
            "first": batch_size,
            "skip": skip,
            "resolved_after": resolved_after,
        }
        rows = _post_graphql(url, query, timeout).get("questions", [])
        for row in rows:
            questions[row["id"]] = row
        if len(rows) < batch_size:
            break
        skip += batch_size
        if delay_seconds:
            time.sleep(delay_seconds)
    return questions


def _load_pending_polymarket_deliveries(state_file: Path) -> list[dict[str, Any]]:
    """Load pending Polymarket deliveries from the benchmark fetch state."""
    state = json.loads(state_file.read_text())
    pending = state.get("polymarket", {}).get("pending_deliveries", [])
    if not isinstance(pending, list):
        raise ValueError(f"{state_file} does not contain polymarket.pending_deliveries")
    return pending


def _default_cutoff(state_file: Path, lookback_days: int) -> int:
    """Compute the same lower-bound style cutoff used by fetch_production."""
    state = json.loads(state_file.read_text())
    state_cutoff = int(state.get("polymarket", {}).get("last_resolved_timestamp", 0))
    lookback_cutoff = int(time.time()) - (lookback_days * 86400)
    return max(state_cutoff, lookback_cutoff)


def _summarize(
    pending: list[dict[str, Any]],
    questions_by_id: dict[str, dict[str, Any] | None],
    via_bets_ids: set[str],
    via_questions: dict[str, dict[str, Any]],
    resolved_after: int,
) -> dict[str, Any]:
    """Build a JSON-serializable comparison summary."""
    pending_id_counts = Counter(
        row.get("market_id") for row in pending if row.get("market_id")
    )
    pending_ids = set(pending_id_counts)
    via_questions_ids = set(via_questions)

    pending_with_question = {
        market_id
        for market_id, question in questions_by_id.items()
        if question is not None
    }
    pending_with_resolution_by_id = {
        market_id
        for market_id, question in questions_by_id.items()
        if question and question.get("resolution") is not None
    }
    pending_resolution_null = {
        market_id
        for market_id, question in questions_by_id.items()
        if question is not None and question.get("resolution") is None
    }

    pending_via_bets = pending_ids & via_bets_ids
    pending_via_questions = pending_ids & via_questions_ids
    missed_by_bets = sorted(pending_via_questions - via_bets_ids)

    return {
        "resolved_after": resolved_after,
        "pending_delivery_rows": len(pending),
        "pending_unique_market_ids": len(pending_ids),
        "pending_question_entities_found_by_id": len(pending_with_question),
        "pending_question_entities_missing_by_id": len(pending_ids - pending_with_question),
        "pending_ids_with_non_null_question_resolution": len(
            pending_with_resolution_by_id
        ),
        "pending_ids_with_null_question_resolution": len(pending_resolution_null),
        "resolved_via_current_bet_discovery": len(via_bets_ids),
        "resolved_via_direct_question_discovery": len(via_questions_ids),
        "pending_ids_resolved_via_current_bet_discovery": len(pending_via_bets),
        "pending_ids_resolved_via_direct_question_discovery": len(
            pending_via_questions
        ),
        "pending_resolved_questions_missed_by_current_discovery": len(
            missed_by_bets
        ),
        "sample_pending_resolved_questions_missed_by_current_discovery": [
            {
                "market_id": market_id,
                "delivery_count": pending_id_counts[market_id],
                "title": (via_questions[market_id].get("metadata") or {}).get(
                    "title"
                ),
                "resolution": via_questions[market_id].get("resolution"),
            }
            for market_id in missed_by_bets[:20]
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare Polymarket resolution discovery through current bet-based "
            "production logic and direct question-based lookup."
        )
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help="Path to benchmark fetch state containing polymarket.pending_deliveries.",
    )
    parser.add_argument(
        "--resolved-after",
        type=int,
        default=None,
        help=(
            "UNIX timestamp cutoff for resolved questions. Defaults to the max "
            "of state polymarket.last_resolved_timestamp and --lookback-days."
        ),
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Lookback used to compute the default --resolved-after cutoff.",
    )
    parser.add_argument(
        "--endpoint",
        default=fp.PREDICT_POLYMARKET_SUBGRAPH_URL,
        help="Polymarket prediction subgraph endpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Page size for direct resolved questions query.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.2,
        help="Delay between paginated direct question requests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON summary.",
    )
    return parser


def main() -> None:
    """Run the comparison."""
    args = _build_parser().parse_args()
    pending = _load_pending_polymarket_deliveries(args.state_file)
    pending_ids = sorted({row["market_id"] for row in pending if row.get("market_id")})
    resolved_after = (
        args.resolved_after
        if args.resolved_after is not None
        else _default_cutoff(args.state_file, args.lookback_days)
    )

    questions_by_id = _fetch_questions_by_id(args.endpoint, pending_ids, args.timeout)
    via_bets = fp.fetch_polymarket_resolved(resolved_after)
    via_questions = _fetch_resolved_questions(
        args.endpoint,
        resolved_after,
        args.batch_size,
        args.timeout,
        args.delay_seconds,
    )

    summary = _summarize(
        pending,
        questions_by_id,
        set(via_bets.by_id),
        via_questions,
        resolved_after,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
