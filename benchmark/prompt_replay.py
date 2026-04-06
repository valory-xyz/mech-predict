"""
Prompt-only replay for benchmarking prediction prompt changes.

Two subcommands:
  enrich  — stratified-sample, fetch original formatted prompts from IPFS,
            extract user_prompt and additional_information (slow, do once per sample size)
  replay  — re-format ALL enriched rows with current prompt template, send to
            LLM, score against production baseline on same markets (fast, iterate)

Usage:
    # Step 1: enrich with stratified sample (5 per platform, ~1 min)
    python -m benchmark.prompt_replay enrich \
      --production-log "benchmark-results (7)/production_log.jsonl" \
      --tool prediction-online --last-days 7 \
      --sample-per-platform 5 --seed 42 \
      --output benchmark/results/prediction_online_enriched_5x5.jsonl

    # Step 2: replay all enriched rows (iterate on prompt changes)
    python -m benchmark.prompt_replay replay \
      --dataset benchmark/results/prediction_online_enriched_5x5.jsonl \
      --model gpt-4.1-2025-04-14 \
      --output-dir benchmark/results/replay_5x5/

    # Step 3: score + compare
    python benchmark/scorer.py --input benchmark/results/replay_5x5/baseline.jsonl ...
    python benchmark/scorer.py --input benchmark/results/replay_5x5/candidate.jsonl ...
    python benchmark/compare.py --baseline ...baseline_scores.json --candidate ...candidate_scores.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import openai
import requests

from benchmark.datasets.fetch_production import (
    DELIVERS_BY_IDS_QUERY,
    IPFS_FETCH_DELAY,
    IPFS_GATEWAY_URL,
    MECH_MARKETPLACE_GNOSIS_URL,
    MECH_MARKETPLACE_POLYGON_URL,
    classify_category,
    parse_tool_response,
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
DEFAULT_BATCH_SIZE = 100
DEFAULT_MODEL = "gpt-4.1-2025-04-14"

# Regex to extract user_prompt and additional_information from the old
# formatted PREDICTION_PROMPT.  The old format uses triple-backtick fences.
USER_PROMPT_RE = re.compile(
    r"USER_PROMPT:\s*```\s*\n(.*?)\n```", re.DOTALL
)
ADDITIONAL_INFO_RE = re.compile(
    r"ADDITIONAL_INFORMATION:\s*```\s*\n(.*?)\n```", re.DOTALL
)


# ---------------------------------------------------------------------------
# IPFS helpers (adapted from sweep.py / fetch_production.py)
# ---------------------------------------------------------------------------


def _fetch_ipfs_hashes_for_deliver_ids(
    deliver_ids: list[str],
    marketplace_url: str,
) -> dict[str, Optional[str]]:
    """Query the subgraph for IPFS hashes by deliver IDs."""
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
            log.warning("Failed to fetch IPFS hashes: %s", e)
            for did in batch:
                result[did] = None

    return result


def _ipfs_hash_to_cid(ipfs_hash: str) -> str:
    """Convert a hex IPFS hash to base32 CIDv1."""
    import base64

    if ipfs_hash.startswith("0x"):
        hash_bytes = bytes.fromhex(ipfs_hash[2:])
        cid_bytes = bytes([0x01, 0x70, 0x12, 0x20]) + hash_bytes
        return "b" + base64.b32encode(cid_bytes).decode().lower().rstrip("=")
    if ipfs_hash.startswith("f"):
        raw = bytes.fromhex(ipfs_hash[1:])
        return "b" + base64.b32encode(raw).decode().lower().rstrip("=")
    return ipfs_hash


def fetch_ipfs_prompt(ipfs_hash: str) -> Optional[str]:
    """Fetch the full IPFS delivery payload and return the 'prompt' field.

    This is the formatted prompt that was sent to the LLM, containing
    both the user_prompt and additional_information baked in.
    """
    try:
        cid = _ipfs_hash_to_cid(ipfs_hash)
        dir_url = f"{IPFS_GATEWAY_URL}/{cid}/"

        resp = requests.get(dir_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()

        # Try direct JSON (auto-resolved single-file directory)
        try:
            payload = resp.json()
            if isinstance(payload, dict) and "prompt" in payload:
                return payload["prompt"]
        except (json.JSONDecodeError, ValueError):
            pass

        # Parse HTML directory listing
        links = re.findall(rf"/ipfs/{re.escape(cid)}/(\d+)", resp.text)
        if not links:
            return None

        file_url = f"{IPFS_GATEWAY_URL}/{cid}/{links[0]}"
        file_resp = requests.get(file_url, timeout=HTTP_TIMEOUT)
        file_resp.raise_for_status()
        payload = file_resp.json()
        return payload.get("prompt")
    except Exception as e:
        log.debug("Failed to fetch IPFS prompt %s: %s", ipfs_hash, e)
        return None


def extract_prompt_components(
    formatted_prompt: str,
) -> Optional[dict[str, str]]:
    """Extract user_prompt and additional_information from a formatted prompt."""
    up_match = USER_PROMPT_RE.search(formatted_prompt)
    ai_match = ADDITIONAL_INFO_RE.search(formatted_prompt)

    if not up_match:
        return None

    return {
        "user_prompt": up_match.group(1).strip(),
        "additional_information": ai_match.group(1).strip() if ai_match else "",
    }


# ---------------------------------------------------------------------------
# Enrich subcommand
# ---------------------------------------------------------------------------


def enrich(
    production_log: Path,
    tool_filter: str,
    output: Path,
    last_days: Optional[int] = None,
    sample_per_platform: Optional[int] = None,
    seed: int = 42,
) -> None:
    """Fetch IPFS prompts and extract components for replay.

    When --sample-per-platform is given, stratified sampling is done BEFORE
    the slow IPFS fetch so we only download what we need.
    """
    cutoff = None
    if last_days is not None:
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=last_days)

    # Load and filter
    rows: list[dict[str, Any]] = []
    with open(production_log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("tool_name") != tool_filter:
                continue
            if not row.get("deliver_id"):
                continue
            if row.get("prediction_parse_status") != "valid":
                continue
            if row.get("final_outcome") is None:
                continue
            if cutoff is not None:
                predicted = row.get("predicted_at")
                if predicted:
                    dt = datetime.fromisoformat(predicted.replace("Z", "+00:00"))
                    if dt < cutoff:
                        continue
            rows.append(row)

    log.info(
        "Loaded %d %s rows with deliver_id + valid predictions + known outcome",
        len(rows),
        tool_filter,
    )

    if not rows:
        log.warning("No rows to enrich")
        return

    # Stratified sample BEFORE IPFS fetch to avoid unnecessary downloads.
    if sample_per_platform is not None:
        rows = stratified_sample(rows, sample_per_platform, seed)
        log.info("Pre-sampled %d rows for IPFS fetch", len(rows))

    # Group by platform for subgraph queries
    gnosis_ids: list[str] = []
    polygon_ids: list[str] = []
    for row in rows:
        did = row["deliver_id"]
        if row.get("platform") == "polymarket":
            polygon_ids.append(did)
        else:
            gnosis_ids.append(did)

    # Fetch IPFS hashes
    ipfs_hashes: dict[str, Optional[str]] = {}
    if gnosis_ids:
        log.info("Fetching IPFS hashes for %d Gnosis deliveries...", len(gnosis_ids))
        ipfs_hashes.update(
            _fetch_ipfs_hashes_for_deliver_ids(gnosis_ids, MECH_MARKETPLACE_GNOSIS_URL)
        )
    if polygon_ids:
        log.info("Fetching IPFS hashes for %d Polygon deliveries...", len(polygon_ids))
        ipfs_hashes.update(
            _fetch_ipfs_hashes_for_deliver_ids(
                polygon_ids, MECH_MARKETPLACE_POLYGON_URL
            )
        )

    has_hash = sum(1 for v in ipfs_hashes.values() if v)
    log.info("%d/%d deliveries have IPFS hashes", has_hash, len(ipfs_hashes))

    # Fetch prompts and extract components
    enriched: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        did = row["deliver_id"]
        ipfs_hash = ipfs_hashes.get(did)
        if not ipfs_hash:
            continue

        prompt_text = fetch_ipfs_prompt(ipfs_hash)
        if not prompt_text:
            continue

        components = extract_prompt_components(prompt_text)
        if not components:
            log.debug(
                "Could not parse prompt components for %s", row["question_text"][:60]
            )
            continue

        enriched_row = {
            **row,
            "extracted_user_prompt": components["user_prompt"],
            "extracted_additional_information": components["additional_information"],
        }
        enriched.append(enriched_row)

        if (i + 1) % 10 == 0:
            log.info(
                "IPFS progress: %d/%d (%d enriched)", i + 1, len(rows), len(enriched)
            )

        time.sleep(IPFS_FETCH_DELAY)

    log.info("Enriched %d/%d rows", len(enriched), len(rows))

    # Report platform breakdown
    by_platform: dict[str, int] = defaultdict(int)
    by_outcome: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in enriched:
        plat = row.get("platform", "unknown")
        outcome = "yes" if row["final_outcome"] else "no"
        by_platform[plat] += 1
        by_outcome[plat][outcome] += 1
    for plat in sorted(by_platform):
        log.info(
            "  %s: %d rows (yes=%d, no=%d)",
            plat,
            by_platform[plat],
            by_outcome[plat]["yes"],
            by_outcome[plat]["no"],
        )

    # Write
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in enriched:
            f.write(json.dumps(row) + "\n")

    log.info("Written to %s", output)


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    rows: list[dict[str, Any]],
    sample_per_platform: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample rows with stratification by platform and outcome.

    Within each platform, splits rows by final_outcome (True/False),
    then samples proportionally from each stratum.  This prevents
    accidentally getting all-yes or all-no markets.
    """
    rng = random.Random(seed)

    # Group by platform
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_platform[row.get("platform", "unknown")].append(row)

    sampled: list[dict[str, Any]] = []
    for platform in sorted(by_platform):
        platform_rows = by_platform[platform]
        n = min(sample_per_platform, len(platform_rows))
        if n == 0:
            continue

        # Split by outcome
        yes_rows = [r for r in platform_rows if r["final_outcome"]]
        no_rows = [r for r in platform_rows if not r["final_outcome"]]

        # Proportional allocation
        yes_frac = len(yes_rows) / len(platform_rows) if platform_rows else 0.5
        n_yes = max(1, min(len(yes_rows), round(n * yes_frac)))
        n_no = max(1, min(len(no_rows), n - n_yes))
        # Adjust if one stratum is too small
        if n_yes > len(yes_rows):
            n_yes = len(yes_rows)
            n_no = min(len(no_rows), n - n_yes)
        if n_no > len(no_rows):
            n_no = len(no_rows)
            n_yes = min(len(yes_rows), n - n_no)

        sampled.extend(rng.sample(yes_rows, n_yes))
        sampled.extend(rng.sample(no_rows, n_no))

        log.info(
            "Sampled %s: %d rows (%d yes, %d no) from %d available",
            platform,
            n_yes + n_no,
            n_yes,
            n_no,
            len(platform_rows),
        )

    return sampled


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> Optional[str]:
    """Send a prompt to the LLM and return the response content."""
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=150,
        )
        return response.choices[0].message.content
    except Exception as e:
        log.warning("LLM call failed: %s", e)
        return None
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Row ID
# ---------------------------------------------------------------------------


def _make_row_id(prefix: str, tool_name: str, question_text: str, model: str) -> str:
    """Deterministic row ID."""
    payload = f"{tool_name}:{model}:{question_text}"
    h = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{prefix}_{tool_name}_{h}"


# ---------------------------------------------------------------------------
# Replay subcommand
# ---------------------------------------------------------------------------


def replay(
    dataset: Path,
    output_dir: Path,
    model: str,
) -> None:
    """Replay enriched rows through current prompt template vs production baseline.

    Uses ALL rows from the enriched dataset (sampling already happened in enrich).
    """
    # Import prompt template from tool module (picks up any code changes)
    from packages.valory.customs.prediction_request.prediction_request import (
        PREDICTION_PROMPT,
        SYSTEM_PROMPT_FORECASTER,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        return

    # Load enriched dataset
    sampled: list[dict[str, Any]] = []
    with open(dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sampled.append(json.loads(line))

    if not sampled:
        log.warning("No rows in dataset")
        return

    log.info("Loaded %d enriched rows from %s", len(sampled), dataset)

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline.jsonl"
    candidate_path = output_dir / "candidate.jsonl"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Track running scores
    baseline_brier_sum = 0.0
    candidate_brier_sum = 0.0
    n_scored = 0

    with (
        open(baseline_path, "w", encoding="utf-8") as bf,
        open(candidate_path, "w", encoding="utf-8") as cf,
    ):
        for i, row in enumerate(sampled):
            question = row["question_text"]
            user_prompt = row["extracted_user_prompt"]
            additional_info = row["extracted_additional_information"]
            outcome = row["final_outcome"]

            # --- Baseline row (original production prediction) ---
            baseline_row = {
                "row_id": _make_row_id("baseline", row["tool_name"], question, model),
                "schema_version": "1.0",
                "mode": "baseline",
                "platform": row.get("platform", "unknown"),
                "question_text": question,
                "tool_name": row["tool_name"],
                "model": row.get("model", model),
                "p_yes": row["p_yes"],
                "p_no": row["p_no"],
                "prediction_parse_status": "valid",
                "confidence": row.get("confidence"),
                "final_outcome": outcome,
                "predicted_at": row.get("predicted_at", now),
                "resolved_at": row.get("resolved_at"),
                "category": row.get("category") or classify_category(question),
            }
            bf.write(json.dumps(baseline_row) + "\n")

            # --- Candidate: re-format with current prompt template ---
            formatted_prompt = PREDICTION_PROMPT.format(
                user_prompt=user_prompt,
                additional_information=additional_info,
            )

            log.info(
                "[%d/%d] %s | %s | %s",
                i + 1,
                len(sampled),
                row.get("platform", "?"),
                "YES" if outcome else "NO",
                question[:70],
            )

            response_text = call_llm(
                model=model,
                system_prompt=SYSTEM_PROMPT_FORECASTER,
                user_prompt=formatted_prompt,
                api_key=api_key,
            )

            parsed = parse_tool_response(response_text)

            candidate_row = {
                "row_id": _make_row_id(
                    "candidate", row["tool_name"], question, model
                ),
                "schema_version": "1.0",
                "mode": "candidate",
                "platform": row.get("platform", "unknown"),
                "question_text": question,
                "tool_name": row["tool_name"],
                "model": model,
                "p_yes": parsed["p_yes"],
                "p_no": parsed["p_no"],
                "prediction_parse_status": parsed["prediction_parse_status"],
                "confidence": parsed.get("confidence"),
                "final_outcome": outcome,
                "predicted_at": now,
                "resolved_at": row.get("resolved_at"),
                "category": row.get("category") or classify_category(question),
            }
            cf.write(json.dumps(candidate_row) + "\n")
            cf.flush()

            # Running score
            outcome_val = 1.0 if outcome else 0.0
            b_brier = (row["p_yes"] - outcome_val) ** 2
            baseline_brier_sum += b_brier

            if parsed["p_yes"] is not None:
                c_brier = (parsed["p_yes"] - outcome_val) ** 2
                candidate_brier_sum += c_brier
                n_scored += 1
                log.info(
                    "  baseline p_yes=%.2f (brier=%.3f) | candidate p_yes=%.2f (brier=%.3f) | %s",
                    row["p_yes"],
                    b_brier,
                    parsed["p_yes"],
                    c_brier,
                    "BETTER" if c_brier < b_brier else "WORSE" if c_brier > b_brier else "SAME",
                )
            else:
                log.warning(
                    "  candidate parse failed: %s", parsed["prediction_parse_status"]
                )

    # Summary
    total = len(sampled)
    avg_baseline = baseline_brier_sum / total if total else 0
    avg_candidate = candidate_brier_sum / n_scored if n_scored else 0

    log.info("=" * 60)
    log.info("RESULTS: %d markets (%d candidate scored)", total, n_scored)
    log.info("  Baseline avg Brier:  %.4f", avg_baseline)
    log.info("  Candidate avg Brier: %.4f", avg_candidate)
    if avg_baseline > 0:
        delta_pct = (avg_candidate - avg_baseline) / avg_baseline * 100
        log.info(
            "  Delta: %+.4f (%+.1f%%) — %s",
            avg_candidate - avg_baseline,
            delta_pct,
            "IMPROVED" if avg_candidate < avg_baseline else "REGRESSED",
        )
    log.info("  Baseline:  %s", baseline_path)
    log.info("  Candidate: %s", candidate_path)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prompt-only replay for benchmarking prediction prompt changes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- enrich ---
    enrich_parser = subparsers.add_parser(
        "enrich",
        help="Fetch IPFS prompts and extract components for replay",
    )
    enrich_parser.add_argument(
        "--production-log",
        type=Path,
        required=True,
        help="Path to production_log.jsonl",
    )
    enrich_parser.add_argument(
        "--tool",
        type=str,
        default="prediction-online",
        help="Tool name to filter (default: prediction-online)",
    )
    enrich_parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/prediction_online_enriched.jsonl"),
        help="Output enriched JSONL",
    )
    enrich_parser.add_argument(
        "--last-days",
        type=int,
        default=None,
        help="Only include rows from the last N days (default: all)",
    )
    enrich_parser.add_argument(
        "--sample-per-platform",
        type=int,
        default=None,
        help="Stratified sample N markets per platform before IPFS fetch (default: all)",
    )
    enrich_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42)",
    )

    # --- replay ---
    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay enriched rows through current prompt template",
    )
    replay_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Enriched JSONL from enrich step",
    )
    replay_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    replay_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results/replay/"),
        help="Output directory for baseline.jsonl + candidate.jsonl",
    )

    args = parser.parse_args()

    if args.command == "enrich":
        enrich(
            production_log=args.production_log,
            tool_filter=args.tool,
            output=args.output,
            last_days=args.last_days,
            sample_per_platform=args.sample_per_platform,
            seed=args.seed,
        )
    elif args.command == "replay":
        replay(
            dataset=args.dataset,
            output_dir=args.output_dir,
            model=args.model,
        )


if __name__ == "__main__":
    main()
