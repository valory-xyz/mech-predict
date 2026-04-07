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
import base64
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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
from benchmark.io import load_jsonl

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
USER_PROMPT_RE = re.compile(r"USER_PROMPT:\s*```\s*\n(.*?)\n```", re.DOTALL)
ADDITIONAL_INFO_RE = re.compile(
    r"ADDITIONAL_INFORMATION:\s*```\s*\n(.*?)\n```", re.DOTALL
)

# Two-stage tool separator (reasoning_prompt + "////" + prediction_prompt)
TWO_STAGE_SEPARATOR = "////"

# Extraction from reasoning_prompt half (stage 1)
REASONING_USER_PROMPT_RE = re.compile(
    r"Here is the user's question:\s*(.*?)\nHere is some additional information",
    re.DOTALL,
)
REASONING_ADDITIONAL_INFO_RE = re.compile(
    r"<additional_information>\s*(.*?)\s*</additional_information>",
    re.DOTALL,
)

# Extraction from prediction_prompt half (stage 2)
PREDICTION_USER_INPUT_RE = re.compile(
    r"<user_input>\s*(.*?)\s*</user_input>",
    re.DOTALL,
)
PREDICTION_REASONING_RE = re.compile(
    r"The reasoning from the other AI is:\s*(.*?)\n\nCarefully consider",
    re.DOTALL,
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

    :param ipfs_hash: hex-encoded IPFS hash from the subgraph.
    :return: the formatted prompt string, or None if not available.
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


def _extract_reasoning_prompt_components(
    formatted_prompt: str,
) -> Optional[dict[str, str]]:
    """Extract components from a two-stage reasoning tool IPFS prompt.

    The IPFS prompt format is: reasoning_prompt + "////" + prediction_prompt.

    :param formatted_prompt: the full IPFS prompt string.
    :return: dict with user_prompt, additional_information, reasoning, user_input
        or None if extraction fails.
    """
    if TWO_STAGE_SEPARATOR not in formatted_prompt:
        return None

    reasoning_half, prediction_half = formatted_prompt.split(TWO_STAGE_SEPARATOR, 1)

    up_match = REASONING_USER_PROMPT_RE.search(reasoning_half)
    ai_match = REASONING_ADDITIONAL_INFO_RE.search(reasoning_half)
    ui_match = PREDICTION_USER_INPUT_RE.search(prediction_half)
    rr_match = PREDICTION_REASONING_RE.search(prediction_half)

    if not up_match or not ui_match:
        return None

    return {
        "user_prompt": up_match.group(1).strip(),
        "additional_information": ai_match.group(1).strip() if ai_match else "",
        "reasoning": rr_match.group(1).strip() if rr_match else "",
        "user_input": ui_match.group(1).strip(),
    }


def extract_prompt_components(
    formatted_prompt: str,
    tool_name: str = "prediction-online",
) -> Optional[dict[str, str]]:
    """Extract components from a formatted IPFS prompt, dispatching by tool.

    :param formatted_prompt: the full IPFS prompt string.
    :param tool_name: tool name to determine extraction strategy.
    :return: dict of extracted components, or None if extraction fails.
    """
    if tool_name.startswith("prediction-request-reasoning"):
        return _extract_reasoning_prompt_components(formatted_prompt)

    # Default: prediction-online / superforcaster format
    up_match = USER_PROMPT_RE.search(formatted_prompt)
    ai_match = ADDITIONAL_INFO_RE.search(formatted_prompt)

    if not up_match:
        return None

    return {
        "user_prompt": up_match.group(1).strip(),
        "additional_information": ai_match.group(1).strip() if ai_match else "",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_and_extract_prompts(
    rows: list[dict[str, Any]],
    ipfs_hashes: dict[str, Optional[str]],
    tool_name: str = "prediction-online",
) -> list[dict[str, Any]]:
    """Fetch IPFS prompts and extract components for replay.

    :param rows: list of row dicts with 'deliver_id' keys.
    :param ipfs_hashes: mapping of deliver_id to IPFS hash.
    :param tool_name: tool name to determine extraction strategy.
    :return: list of enriched row dicts.
    """
    enriched: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        did = row["deliver_id"]
        ipfs_hash = ipfs_hashes.get(did)
        if not ipfs_hash:
            continue

        prompt_text = fetch_ipfs_prompt(ipfs_hash)
        if not prompt_text:
            continue

        components = extract_prompt_components(prompt_text, tool_name=tool_name)
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
        # Two-stage tools also store reasoning and user_input
        if "reasoning" in components:
            enriched_row["extracted_reasoning"] = components["reasoning"]
        if "user_input" in components:
            enriched_row["extracted_user_input"] = components["user_input"]
        enriched.append(enriched_row)

        if (i + 1) % 10 == 0:
            log.info(
                "IPFS progress: %d/%d (%d enriched)", i + 1, len(rows), len(enriched)
            )

        time.sleep(IPFS_FETCH_DELAY)

    log.info("Enriched %d/%d rows", len(enriched), len(rows))
    return enriched


def _log_platform_breakdown(rows: list[dict[str, Any]]) -> None:
    """Log platform/outcome breakdown.

    :param rows: list of row dicts with 'platform' and 'final_outcome' keys.
    """
    by_platform: dict[str, int] = defaultdict(int)
    by_outcome: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
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


def _load_and_filter_rows(
    production_log: Path,
    tool_filter: str,
    last_days: Optional[int],
) -> list[dict[str, Any]]:
    """Load production log and filter for valid rows.

    :param production_log: path to production_log.jsonl.
    :param tool_filter: tool name to filter for.
    :param last_days: only include rows from the last N days.
    :return: filtered list of row dicts.
    """
    cutoff = None
    if last_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=last_days)

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
    return rows


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

    :param production_log: path to production_log.jsonl.
    :param tool_filter: tool name to filter for.
    :param output: path to write enriched JSONL.
    :param last_days: only include rows from the last N days.
    :param sample_per_platform: stratified sample N per platform before IPFS fetch.
    :param seed: random seed for sampling.
    """
    rows = _load_and_filter_rows(production_log, tool_filter, last_days)
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

    enriched = _fetch_and_extract_prompts(rows, ipfs_hashes, tool_name=tool_filter)
    _log_platform_breakdown(enriched)

    # Write
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in enriched:
            f.write(json.dumps(row) + "\n")

    log.info("Written to %s", output)


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def _brier_bucket(p_yes: float, outcome: bool) -> str:
    """Classify a prediction into a Brier-score bucket.

    :param p_yes: predicted probability of Yes outcome.
    :param outcome: actual outcome (True = Yes).
    :return: "good", "moderate", or "poor".
    """
    brier = (p_yes - int(outcome)) ** 2
    if brier <= 0.10:
        return "good"
    if brier <= 0.50:
        return "moderate"
    return "poor"


def stratified_sample(
    rows: list[dict[str, Any]],
    sample_per_platform: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample rows with stratification by platform, outcome, and Brier bucket.

    Each platform gets ``sample_per_platform`` rows (50:50 platform split).
    Within each platform, rows are grouped by (outcome, brier_bucket) and
    sampled proportionally, ensuring at least 1 row per non-empty stratum.

    :param rows: list of row dicts with 'platform', 'final_outcome', 'p_yes'.
    :param sample_per_platform: max rows to sample per platform.
    :param seed: random seed for reproducibility.
    :return: list of sampled row dicts.
    """
    rng = random.Random(seed)

    # Group by platform
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_platform[row.get("platform", "unknown")].append(row)

    sampled: list[dict[str, Any]] = []
    for platform in sorted(by_platform):
        platform_rows = by_platform[platform]
        budget = min(sample_per_platform, len(platform_rows))
        if budget == 0:
            continue

        # Group by (outcome, brier_bucket)
        strata: dict[tuple[bool, str], list[dict[str, Any]]] = defaultdict(list)
        for row in platform_rows:
            outcome = row["final_outcome"]
            bucket = _brier_bucket(row["p_yes"], outcome)
            strata[(outcome, bucket)].append(row)

        # Proportional allocation with floor of 1 per non-empty stratum
        non_empty = {k: v for k, v in strata.items() if v}
        total_available = sum(len(v) for v in non_empty.values())

        allocations: dict[tuple[bool, str], int] = {}
        remaining_budget = budget

        # First pass: give each stratum at least 1
        for key, stratum_rows in non_empty.items():
            allocations[key] = min(1, len(stratum_rows))
            remaining_budget -= allocations[key]

        # Second pass: distribute remaining proportionally
        if remaining_budget > 0 and total_available > 0:
            for key, stratum_rows in non_empty.items():
                extra = round(remaining_budget * len(stratum_rows) / total_available)
                allocations[key] += extra

            # Clamp to available and adjust
            for key in non_empty:
                allocations[key] = min(allocations[key], len(non_empty[key]))

            # Ensure total matches budget (distribute remainder largest-first)
            allocated = sum(allocations.values())
            deficit = budget - allocated
            if deficit > 0:
                sizes = {k: len(v) for k, v in non_empty.items()}
                for key in sorted(sizes, key=sizes.__getitem__, reverse=True):
                    can_add = len(non_empty[key]) - allocations[key]
                    add = min(deficit, can_add)
                    allocations[key] += add
                    deficit -= add
                    if deficit == 0:
                        break

        # Sample from each stratum
        platform_sampled = 0
        for key in sorted(non_empty):
            n = allocations[key]
            if n > 0:
                sampled.extend(rng.sample(non_empty[key], n))
                platform_sampled += n

        # Log breakdown
        outcome_labels = {True: "yes", False: "no"}
        detail_parts = []
        for key in sorted(non_empty):
            outcome, bucket = key
            n = allocations[key]
            detail_parts.append(f"{outcome_labels[outcome]}/{bucket}={n}")
        log.info(
            "Sampled %s: %d/%d rows [%s]",
            platform,
            platform_sampled,
            len(platform_rows),
            ", ".join(detail_parts),
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
    """Send a prompt to the LLM and return the response content.

    :param model: model identifier (e.g. gpt-4.1-2025-04-14, claude-4-sonnet-20250514).
    :param system_prompt: system message content.
    :param user_prompt: user message content.
    :param api_key: API key for the provider.
    :param temperature: sampling temperature.
    :param max_tokens: maximum tokens to generate.
    :return: response content string, or None on failure.
    """
    if "claude" in model:
        return _call_anthropic(
            model, system_prompt, user_prompt, api_key, temperature, max_tokens
        )
    return _call_openai(
        model, system_prompt, user_prompt, api_key, temperature, max_tokens
    )


def _call_openai(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    """Call OpenAI-compatible API.

    :param model: model identifier.
    :param system_prompt: system message content.
    :param user_prompt: user message content.
    :param api_key: OpenAI API key.
    :param temperature: sampling temperature.
    :param max_tokens: maximum tokens to generate.
    :return: response content string, or None on failure.
    """
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


def _call_anthropic(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    """Call Anthropic API.

    :param model: model identifier.
    :param system_prompt: system message content.
    :param user_prompt: user message content.
    :param api_key: Anthropic API key.
    :param temperature: sampling temperature.
    :param max_tokens: maximum tokens to generate.
    :return: response content string, or None on failure.
    """
    import anthropic  # pylint: disable=import-outside-toplevel

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text
    except Exception as e:
        log.warning("LLM call failed: %s", e)
        return None
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Row ID
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

PROBABILITY_SUM_TOLERANCE = 0.05


def parse_xml_prediction_response(
    response_text: Optional[str],
) -> dict[str, Any]:
    """Parse XML-tagged prediction response from the reasoning tool.

    Expects tags: <p_yes>, <p_no>, <confidence>, <info_utility>.
    Returns same schema as fetch_production.parse_tool_response.

    :param response_text: raw LLM response string.
    :return: dict with p_yes, p_no, confidence, prediction_parse_status.
    """
    if not response_text:
        return {
            "p_yes": None,
            "p_no": None,
            "confidence": None,
            "prediction_parse_status": "missing_fields",
        }

    tags: dict[str, Optional[float]] = {
        "p_yes": None,
        "p_no": None,
        "confidence": None,
        "info_utility": None,
    }
    for key in tags:
        try:
            value_str = response_text.split(f"<{key}>")[1].split(f"</{key}>")[0].strip()
            tags[key] = float(value_str)
        except (IndexError, ValueError):
            pass

    if (
        tags["p_yes"] is not None
        and tags["p_no"] is not None
        and 0.0 <= tags["p_yes"] <= 1.0
        and 0.0 <= tags["p_no"] <= 1.0
        and abs(tags["p_yes"] + tags["p_no"] - 1.0) <= PROBABILITY_SUM_TOLERANCE
    ):
        return {
            "p_yes": tags["p_yes"],
            "p_no": tags["p_no"],
            "confidence": tags["confidence"],
            "prediction_parse_status": "valid",
        }

    return {
        "p_yes": None,
        "p_no": None,
        "confidence": None,
        "prediction_parse_status": "malformed",
    }


def parse_response(response_text: Optional[str], tool_name: str) -> dict[str, Any]:
    """Route to the correct response parser based on tool name.

    :param response_text: raw LLM response string.
    :param tool_name: tool name to determine parsing strategy.
    :return: dict with p_yes, p_no, confidence, prediction_parse_status.
    """
    if tool_name.startswith("prediction-request-reasoning"):
        return parse_xml_prediction_response(response_text)
    return parse_tool_response(response_text)


# ---------------------------------------------------------------------------
# Row ID
# ---------------------------------------------------------------------------


# TODO: unify _make_row_id across runner, tournament, prompt_replay
# & fetch_production into benchmark/tools.py
def _make_row_id(prefix: str, tool_name: str, question_text: str, model: str) -> str:
    """Deterministic row ID."""
    payload = f"{tool_name}:{model}:{question_text}"
    h = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{prefix}_{tool_name}_{h}"


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------


def _log_replay_summary(
    sampled: list[dict[str, Any]],
    candidate_path: Path,
    baseline_brier_sum: float,
    candidate_brier_sum: float,
    total: int,
    n_scored: int,
    baseline_path: Path,
) -> None:
    """Compute accuracy and log the replay summary.

    :param sampled: list of sampled row dicts with production p_yes.
    :param candidate_path: path to the candidate JSONL file.
    :param baseline_brier_sum: sum of baseline Brier scores.
    :param candidate_brier_sum: sum of candidate Brier scores.
    :param total: total number of markets.
    :param n_scored: number of candidate predictions scored.
    :param baseline_path: path to the baseline JSONL file.
    """
    avg_baseline = baseline_brier_sum / total if total else 0
    avg_candidate = candidate_brier_sum / n_scored if n_scored else 0

    baseline_correct = sum(
        1
        for c in sampled
        if (c["p_yes"] > 0.5 and c["final_outcome"])
        or (c["p_yes"] < 0.5 and not c["final_outcome"])
    )

    candidate_correct = 0
    for cr in load_jsonl(candidate_path):
        if cr["p_yes"] is not None:
            if (cr["p_yes"] > 0.5 and cr["final_outcome"]) or (
                cr["p_yes"] < 0.5 and not cr["final_outcome"]
            ):
                candidate_correct += 1

    baseline_acc = baseline_correct / total if total else 0
    candidate_acc = candidate_correct / n_scored if n_scored else 0

    log.info("=" * 60)
    log.info("RESULTS: %d markets (%d candidate scored)", total, n_scored)
    log.info(
        "  Baseline avg Brier:  %.4f  Accuracy: %.1f%%",
        avg_baseline,
        baseline_acc * 100,
    )
    log.info(
        "  Candidate avg Brier: %.4f  Accuracy: %.1f%%",
        avg_candidate,
        candidate_acc * 100,
    )
    if avg_baseline > 0:
        delta_pct = (avg_candidate - avg_baseline) / avg_baseline * 100
        log.info(
            "  Brier delta: %+.4f (%+.1f%%) — %s",
            avg_candidate - avg_baseline,
            delta_pct,
            "IMPROVED" if avg_candidate < avg_baseline else "REGRESSED",
        )
    log.info("  Baseline:  %s", baseline_path)
    log.info("  Candidate: %s", candidate_path)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Two-stage replay helper
# ---------------------------------------------------------------------------


def _replay_reasoning_tool(
    *,
    row: dict[str, Any],
    phase: str,
    model: str,
    system_prompt: str,
    api_key: str,
    prediction_prompt_tpl: str,
    reasoning_prompt_tpl: str,
    parser_reasoning_response: Any,
) -> tuple[Optional[str], Optional[str]]:
    """Run the appropriate LLM call(s) for a two-stage reasoning tool.

    :param row: enriched row with extracted_user_prompt, extracted_reasoning, etc.
    :param phase: "prediction-only", "reasoning-only", or "both".
    :param model: LLM model identifier.
    :param system_prompt: system prompt for the LLM.
    :param api_key: API key for the LLM provider.
    :param prediction_prompt_tpl: PREDICTION_PROMPT template string.
    :param reasoning_prompt_tpl: REASONING_PROMPT template string.
    :param parser_reasoning_response: function to extract reasoning from XML tags.
    :return: (prediction_response, fresh_reasoning) — fresh_reasoning is None
        for prediction-only phase.
    """
    if phase == "prediction-only":
        # Hold reasoning fixed, only re-run stage 2
        formatted = prediction_prompt_tpl.format(
            USER_INPUT=row["extracted_user_prompt"],
            REASONING=row["extracted_reasoning"],
        )
        return (
            call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=formatted,
                api_key=api_key,
            ),
            None,
        )

    # Phase "reasoning-only" or "both" — re-run stage 1
    reasoning_formatted = reasoning_prompt_tpl.format(
        USER_PROMPT=row["extracted_user_prompt"],
        ADDITIONAL_INFOMATION=row["extracted_additional_information"],
    )
    reasoning_response = call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=reasoning_formatted,
        api_key=api_key,
    )

    if not reasoning_response:
        log.warning("  Stage 1 (reasoning) returned empty response")
        return None, None

    reasoning = parser_reasoning_response(reasoning_response)
    if not reasoning:
        log.warning("  Stage 1 (reasoning) parsing failed")
        return None, None

    # Stage 2 — prediction using fresh reasoning
    formatted = prediction_prompt_tpl.format(
        USER_INPUT=row["extracted_user_prompt"],
        REASONING=reasoning,
    )
    return (
        call_llm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=formatted,
            api_key=api_key,
        ),
        reasoning,
    )


# ---------------------------------------------------------------------------
# Replay subcommand
# ---------------------------------------------------------------------------


def replay(  # pylint: disable=too-many-statements
    dataset: Path,
    output_dir: Path,
    model: str,
    phase: str = "prediction-only",
) -> None:
    """Replay enriched rows through current prompt template vs production baseline.

    Uses ALL rows from the enriched dataset (sampling already happened in enrich).

    :param dataset: path to enriched JSONL from enrich step.
    :param output_dir: directory for baseline.jsonl + candidate.jsonl output.
    :param model: LLM model identifier.
    :param phase: replay phase — "prediction-only", "reasoning-only", or "both".
    """
    # Load enriched dataset first to detect tool
    sampled: list[dict[str, Any]] = load_jsonl(dataset)

    if not sampled:
        log.warning("No rows in dataset")
        return

    tool_name = sampled[0]["tool_name"]
    is_reasoning_tool = tool_name.startswith("prediction-request-reasoning")

    # Import prompt templates from the appropriate tool module
    if is_reasoning_tool:
        from packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning import (  # pylint: disable=import-outside-toplevel
            PREDICTION_PROMPT,
            REASONING_PROMPT,
            SYSTEM_PROMPT,
            parser_reasoning_response,
        )

        system_prompt = SYSTEM_PROMPT
    else:
        from packages.valory.customs.prediction_request.prediction_request import (  # pylint: disable=import-outside-toplevel
            PREDICTION_PROMPT,
            SYSTEM_PROMPT_FORECASTER,
        )

        system_prompt = SYSTEM_PROMPT_FORECASTER

    if "claude" in model:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            log.error("ANTHROPIC_API_KEY not set")
            return
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            log.error("OPENAI_API_KEY not set")
            return

    log.info(
        "Loaded %d enriched rows from %s (tool=%s, phase=%s)",
        len(sampled),
        dataset,
        tool_name,
        phase,
    )

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
            question, outcome = row["question_text"], row["final_outcome"]

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

            log.info(
                "[%d/%d] %s | %s | %s",
                i + 1,
                len(sampled),
                row.get("platform", "?"),
                "YES" if outcome else "NO",
                question[:70],
            )

            # --- Candidate: phase-aware prompt formatting + LLM call ---
            fresh_reasoning = None
            if is_reasoning_tool:
                response_text, fresh_reasoning = _replay_reasoning_tool(
                    row=row,
                    phase=phase,
                    model=model,
                    system_prompt=system_prompt,
                    api_key=api_key,
                    prediction_prompt_tpl=PREDICTION_PROMPT,
                    reasoning_prompt_tpl=REASONING_PROMPT,
                    parser_reasoning_response=parser_reasoning_response,
                )
            else:
                # Single-stage tool (prediction-online, etc.)
                formatted_prompt = PREDICTION_PROMPT.format(
                    user_prompt=row["extracted_user_prompt"],
                    additional_information=row["extracted_additional_information"],
                )
                response_text = call_llm(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=formatted_prompt,
                    api_key=api_key,
                )

            # Cache fresh reasoning for later prediction-only iteration
            if fresh_reasoning is not None:
                row["_fresh_reasoning"] = fresh_reasoning

            parsed = parse_response(response_text, tool_name)

            candidate_row = {
                "row_id": _make_row_id("candidate", row["tool_name"], question, model),
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
                    (
                        "BETTER"
                        if c_brier < b_brier
                        else "WORSE" if c_brier > b_brier else "SAME"
                    ),
                )
            else:
                log.warning(
                    "  candidate parse failed: %s", parsed["prediction_parse_status"]
                )

    # Summary
    total = len(sampled)

    _log_replay_summary(
        sampled,
        candidate_path,
        baseline_brier_sum,
        candidate_brier_sum,
        total,
        n_scored,
        baseline_path,
    )

    # Write updated enriched JSONL with fresh reasoning for next iteration
    has_fresh = any(r.get("_fresh_reasoning") for r in sampled)
    if has_fresh:
        enriched_path = output_dir / "enriched_with_new_reasoning.jsonl"
        with open(enriched_path, "w", encoding="utf-8") as ef:
            for row in sampled:
                updated = {**row}
                fresh = updated.pop("_fresh_reasoning", None)
                if fresh:
                    updated["extracted_reasoning"] = fresh
                ef.write(json.dumps(updated) + "\n")
        log.info("Wrote enriched dataset with new reasoning: %s", enriched_path)


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
        help="Stratified sample N rows per platform (distributed across outcome x brier strata) before IPFS fetch (default: all)",
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
    replay_parser.add_argument(
        "--phase",
        choices=["prediction-only", "reasoning-only", "both"],
        default="prediction-only",
        help=(
            "For two-stage tools: which stage(s) to replay. "
            "prediction-only = hold reasoning fixed, iterate stage 2; "
            "reasoning-only = re-run stage 1, hold stage 2 fixed; "
            "both = re-run both stages. Ignored for single-stage tools. "
            "(default: prediction-only)"
        ),
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
            phase=args.phase,
        )


if __name__ == "__main__":
    main()
