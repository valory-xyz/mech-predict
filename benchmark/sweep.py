"""
Benchmark sweep: filter → enrich → score baseline → replay → score candidate → compare.

Orchestrates the full benchmark pipeline for local developer experimentation.
Modify tool code or switch models, run this script, see the comparison table.

The production log already contains baseline predictions. This script:
1. Filters rows matching your tool from the production log
2. Enriches them with source_content from IPFS (cached web evidence)
3. Scores those enriched rows as the baseline (production performance)
4. Replays only the candidate (your modified code/model) with cached source_content
5. Scores the candidate and prints a comparison table

Usage:
    # Test a code change on superforcaster (last 500 rows)
    python benchmark/sweep.py --last-n 500 --tools superforcaster --candidate-model gpt-4.1-2025-04-14

    # Reuse existing replay dataset (skip IPFS enrichment)
    python benchmark/sweep.py --dataset replay_dataset.jsonl --tools superforcaster --candidate-model gpt-4.1-2025-04-14

    # Skip baseline scoring (use pre-computed scores)
    python benchmark/sweep.py --dataset replay_dataset.jsonl --baseline-scores baseline.json --tools superforcaster --candidate-model gpt-4.1-2025-04-14
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import requests

from benchmark.compare import compare, format_markdown
from benchmark.datasets.fetch_production import (
    DELIVERS_BY_IDS_QUERY,
    IPFS_FETCH_DELAY,
    MECH_MARKETPLACE_GNOSIS_URL,
    MECH_MARKETPLACE_POLYGON_URL,
    fetch_ipfs_source_content,
)
from benchmark.runner import DEFAULT_MODEL, TASK_DEADLINE, TOOL_REGISTRY, replay
from benchmark.scorer import load_rows, score

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Scores = dict[str, Any]

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
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_PRODUCTION_LOG = Path(__file__).parent / "datasets" / "production_log.jsonl"
DEFAULT_LAST_N = 100
HTTP_TIMEOUT = 60
DEFAULT_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _fetch_ipfs_hashes_for_deliver_ids(
    deliver_ids: list[str],
    marketplace_url: str,
) -> dict[str, Optional[str]]:
    """Query the subgraph for IPFS hashes by deliver IDs.

    :param deliver_ids: list of deliver IDs to look up.
    :param marketplace_url: subgraph endpoint URL.
    :return: dict mapping deliver_id to ipfs_hash (or None).
    """
    result: dict[str, Optional[str]] = {}
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
                result[d["id"]] = mp.get("ipfsHashBytes")
        except Exception as e:
            log.warning("Failed to fetch IPFS hashes from subgraph: %s", e)
            for did in batch:
                result[did] = None

    return result


def step_filter_rows(
    production_log: Path,
    last_n: int,
    tools: list[str],
    output: Path,
) -> Path:
    """Filter last N rows from production log to matching tool + valid + resolved.

    :param production_log: path to production_log.jsonl.
    :param last_n: number of most recent rows to consider.
    :param tools: tool names to include.
    :param output: path to write filtered rows.
    :return: the output path.
    """
    log.info("=== FILTER ROWS (last %d, tools=%s) ===", last_n, tools)

    all_rows: list[dict[str, Any]] = []
    with open(production_log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))

    tail = all_rows if last_n == 0 else all_rows[-last_n:]

    filtered: list[dict[str, Any]] = [
        r
        for r in tail
        if r.get("tool_name") in tools
        and r.get("prediction_parse_status") == "valid"
        and r.get("final_outcome") is not None
    ]

    log.info(
        "  %d total rows → last %d → %d matched (tool + valid + resolved)",
        len(all_rows),
        len(tail),
        len(filtered),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in filtered:
            f.write(json.dumps(row) + "\n")

    return output


def step_enrich_rows(
    rows_path: Path,
    output: Path,
) -> Path:
    """Enrich existing rows with source_content from IPFS.

    Reads rows from *rows_path*, fetches IPFS hashes via subgraph, then
    fetches source_content from IPFS gateway. Only rows that get
    source_content are written to *output*.

    :param rows_path: JSONL file with rows that have deliver_id.
    :param output: path to write enriched replay dataset.
    :return: the output path.
    """
    log.info("=== ENRICH ROWS with source_content ===")

    rows: list[dict[str, Any]] = []
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        log.warning("  No rows to enrich")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("")
        return output

    # Collect deliver_ids by chain
    gnosis_ids: list[str] = []
    polygon_ids: list[str] = []
    for row in rows:
        did = row.get("deliver_id")
        if not did:
            continue
        platform = row.get("platform", "")
        if platform == "polymarket":
            polygon_ids.append(did)
        else:
            gnosis_ids.append(did)

    # Fetch IPFS hashes
    ipfs_hashes: dict[str, Optional[str]] = {}
    if gnosis_ids:
        log.info("  Fetching IPFS hashes for %d Gnosis deliveries...", len(gnosis_ids))
        ipfs_hashes.update(
            _fetch_ipfs_hashes_for_deliver_ids(gnosis_ids, MECH_MARKETPLACE_GNOSIS_URL)
        )
    if polygon_ids:
        log.info(
            "  Fetching IPFS hashes for %d Polygon deliveries...", len(polygon_ids)
        )
        ipfs_hashes.update(
            _fetch_ipfs_hashes_for_deliver_ids(
                polygon_ids, MECH_MARKETPLACE_POLYGON_URL
            )
        )

    has_hash = sum(1 for v in ipfs_hashes.values() if v)
    log.info("  %d/%d deliveries have IPFS hashes", has_hash, len(ipfs_hashes))

    # Fetch source_content from IPFS
    enriched_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        did = row.get("deliver_id")
        ipfs_hash = ipfs_hashes.get(did) if did else None

        if not ipfs_hash:
            continue

        sc = fetch_ipfs_source_content(ipfs_hash)
        if sc:
            enriched_rows.append({**row, "source_content": sc})

        if (i + 1) % 50 == 0:
            log.info(
                "  IPFS progress: %d/%d (%d enriched)",
                i + 1,
                len(rows),
                len(enriched_rows),
            )

        time.sleep(IPFS_FETCH_DELAY)

    log.info("  Enriched %d/%d rows with source_content", len(enriched_rows), len(rows))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in enriched_rows:
            f.write(json.dumps(row) + "\n")

    log.info("  Replay dataset written to %s (%d rows)", output, len(enriched_rows))
    return output


def step_score_baseline(
    dataset_path: Path,
    output: Path,
    tools: list[str],
) -> Scores:
    """Score the enriched dataset rows as the production baseline.

    Uses the p_yes/p_no/final_outcome already in the rows — no model re-run.
    Filters to rows matching *tools* so the baseline covers exactly the same
    tool subset as the candidate replay.

    :param dataset_path: enriched replay dataset JSONL.
    :param output: path to write baseline scores JSON.
    :param tools: tool names to include (must match --tools).
    :return: scores dict.
    """
    log.info("=== SCORE BASELINE (production predictions, tools=%s) ===", tools)
    all_rows = load_rows(dataset_path)
    rows = [r for r in all_rows if r.get("tool_name") in tools]
    if len(rows) < len(all_rows):
        log.info(
            "  Filtered %d → %d rows matching --tools", len(all_rows), len(rows)
        )
    if not rows:
        log.warning("  No rows to score")
        return {"overall": {}, "by_tool": {}, "by_platform": {}, "by_category": {}}

    scores = score(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2))
    log.info("  Baseline: %d rows scored → %s", len(rows), output)
    return scores


def step_replay(
    dataset: Path,
    output: Path,
    tools: list[str],
    model: str,
    timeout: int,
) -> Path:
    """Run the replay runner."""
    log.info("=== REPLAY: %s on %d tools ===", model, len(tools))
    replay(
        dataset_path=dataset,
        output_path=output,
        tools=tools,
        model=model,
        timeout=timeout,
    )
    return output


def step_score(results: Path, output: Path) -> Scores:
    """Score replay results."""
    log.info("=== SCORE: %s ===", results)
    rows = load_rows(results)
    if not rows:
        log.warning("  No rows to score in %s", results)
        return {"overall": {}, "by_tool": {}, "by_platform": {}, "by_category": {}}

    scores = score(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2))
    log.info("  Scored %d rows → %s", len(rows), output)
    return scores


def step_compare(
    baseline_scores: Scores,
    candidate_scores: Scores,
    output: Optional[Path],
) -> str:
    """Compare baseline and candidate scores."""
    log.info("=== COMPARE ===")
    comparison = compare(baseline_scores, candidate_scores)
    markdown = format_markdown(comparison)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown)
        log.info("  Comparison written to %s", output)

    return markdown


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sweep: filter → enrich → score baseline → "
            "replay candidate → score → compare."
        ),
    )

    # Data source
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Existing replay dataset (skips filter + enrichment steps)",
    )
    parser.add_argument(
        "--production-log",
        type=Path,
        default=DEFAULT_PRODUCTION_LOG,
        help=f"Production log to read rows from (default: {DEFAULT_PRODUCTION_LOG})",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=DEFAULT_LAST_N,
        help=(
            f"Filter last N rows from production log (default: {DEFAULT_LAST_N}). "
            "Use 0 for all rows."
        ),
    )

    # Baseline
    parser.add_argument(
        "--baseline-scores",
        type=Path,
        default=None,
        help="Existing baseline scores.json (skips baseline scoring)",
    )

    # Candidate
    parser.add_argument(
        "--candidate-model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model for candidate run (default: {DEFAULT_MODEL})",
    )

    # Shared
    parser.add_argument(
        "--tools",
        type=str,
        required=True,
        help="Comma-separated tool names to test",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TASK_DEADLINE,
        help=f"Per-tool timeout in seconds (default: {TASK_DEADLINE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for intermediate files (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
        help="Write comparison to file (default: stdout only)",
    )

    args = parser.parse_args()

    # Parse tools
    tools = [t.strip() for t in args.tools.split(",")]
    for t in tools:
        if t not in TOOL_REGISTRY:
            parser.error(f"Unknown tool: {t}. Available: {sorted(TOOL_REGISTRY)}")

    results_dir = args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Get the replay dataset (filter + enrich, or reuse)
    # ---------------------------------------------------------------
    if args.dataset:
        dataset_path = args.dataset
        log.info("Using existing dataset: %s", dataset_path)
    else:
        filtered_path = results_dir / "sweep_filtered.jsonl"
        step_filter_rows(
            production_log=args.production_log,
            last_n=args.last_n,
            tools=tools,
            output=filtered_path,
        )

        dataset_path = results_dir / "sweep_replay_dataset.jsonl"
        step_enrich_rows(
            rows_path=filtered_path,
            output=dataset_path,
        )

    # ---------------------------------------------------------------
    # Step 2: Score baseline (production predictions on enriched rows only)
    # ---------------------------------------------------------------
    if args.baseline_scores:
        log.info("Using existing baseline scores: %s", args.baseline_scores)
        with open(args.baseline_scores, encoding="utf-8") as f:
            baseline_scores = json.load(f)
    else:
        baseline_scores_path = results_dir / "sweep_baseline_scores.json"
        baseline_scores = step_score_baseline(dataset_path, baseline_scores_path, tools)

    # ---------------------------------------------------------------
    # Step 3: Replay candidate + score
    # ---------------------------------------------------------------
    candidate_model = args.candidate_model
    candidate_results = results_dir / f"sweep_candidate_{candidate_model}.jsonl"
    candidate_scores_path = (
        results_dir / f"sweep_candidate_{candidate_model}_scores.json"
    )

    step_replay(dataset_path, candidate_results, tools, candidate_model, args.timeout)
    candidate_scores = step_score(candidate_results, candidate_scores_path)

    # ---------------------------------------------------------------
    # Step 4: Compare
    # ---------------------------------------------------------------
    markdown = step_compare(baseline_scores, candidate_scores, args.comparison_output)
    print("\n" + markdown)


if __name__ == "__main__":
    main()
