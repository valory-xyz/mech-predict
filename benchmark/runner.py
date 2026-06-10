"""
Cached replay runner for benchmark evaluation.

Replays resolved prediction market questions through prediction tools
with cached web content injected (source_content), producing
production_log.jsonl-compatible output for scoring.

Usage:
    python benchmark/runner.py --dataset path/to/replay.jsonl
    python benchmark/runner.py --dataset path/to/replay.jsonl --tools prediction-online,superforcaster
    python benchmark/runner.py --dataset path/to/replay.jsonl --model gpt-4.1-2025-04-14
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from benchmark.datasets.fetch_production import classify_category, parse_tool_response
from benchmark.io import load_existing_ids as load_existing_row_ids
from benchmark.io import load_jsonl as load_dataset
from benchmark.tools import (
    TOOL_REGISTRY,
    ToolTimeout,
    _can_use_sigalrm,
    alarm_handler,
    build_keychain,
    load_tool_run,
)

from packages.valory.skills.task_execution.utils.apis import KeyChain

# ---------------------------------------------------------------------------
# Logging — matches fetch_production.py style
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

DEFAULT_OUTPUT = Path(__file__).parent / "results" / "replay_results.jsonl"
DEFAULT_MODEL = "gpt-4.1-2025-04-14"
TASK_DEADLINE = 240  # seconds, matches production


# ---------------------------------------------------------------------------
# Row ID generation
# ---------------------------------------------------------------------------


# TODO: unify _make_row_id across runner, tournament, prompt_replay
# & fetch_production into benchmark/tools.py
def _make_row_id(tool_name: str, question_text: str, model: str) -> str:
    """Deterministic row ID from tool + question + model."""
    payload = f"{tool_name}:{model}:{question_text}"
    h = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"replay_{tool_name}_{h}"


# ---------------------------------------------------------------------------
# Core: run a single tool on a single question
# ---------------------------------------------------------------------------


def run_single(
    tool_name: str,
    question_text: str,
    source_content: dict[str, Any],
    model: str,
    api_keys: KeyChain,
    timeout: int = TASK_DEADLINE,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one tool on one question and return parsed result.

    :param tool_name: registered tool name.
    :param question_text: the prediction question.
    :param source_content: cached web content for replay.
    :param model: LLM model identifier.
    :param api_keys: KeyChain with API credentials.
    :param timeout: per-tool timeout in seconds.
    :param request_context: optional mech request_context dict (market_id,
        type, …) mirroring what the trader sends in production. Forwarded to
        tools that read it (e.g. factual_research-v2); omitted when None.
    :return: dict with p_yes, p_no, confidence, prediction_parse_status,
        latency_s, error.
    """
    run_fn = load_tool_run(tool_name)

    kwargs: dict[str, Any] = {
        "tool": tool_name,
        "prompt": question_text,
        "model": model,
        "api_keys": api_keys,
        "source_content": source_content,
        "counter_callback": None,
        "delivery_rate": 100,
    }
    if request_context is not None:
        kwargs["request_context"] = request_context

    start = time.monotonic()
    use_alarm = _can_use_sigalrm()
    old_handler = None

    try:
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(timeout)

        result_tuple = run_fn(**kwargs)
        elapsed = time.monotonic() - start

        # with_key_rotation appends api_keys → index 0 is always the result
        result_str = result_tuple[0]
        parsed = parse_tool_response(result_str)

        return {
            "latency_s": round(elapsed, 1),
            "error": None,
            **parsed,
        }
    except ToolTimeout:
        return {
            "p_yes": None,
            "p_no": None,
            "confidence": None,
            "prediction_parse_status": "timeout",
            "latency_s": timeout,
            "error": "timeout",
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        log.exception("Tool %s raised an exception", tool_name)
        return {
            "p_yes": None,
            "p_no": None,
            "confidence": None,
            "prediction_parse_status": "error",
            "latency_s": round(elapsed, 1),
            "error": str(e),
        }
    finally:
        if use_alarm:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Build output row
# ---------------------------------------------------------------------------


POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
POLYMARKET_PLATFORM = "polymarket"
GAMMA_TIMEOUT = 15


def _fetch_polymarket_description(condition_id: str) -> str | None:
    """Fetch a Polymarket market's resolution rules (Gamma ``description``).

    The benchmark stands in for the trader here: in production the trader
    fetches the resolution rules from Gamma and forwards them on the mech
    request (trader PR #989), and ``factual_research-v2`` only reads them. To
    reproduce that input under replay we fetch the same text by ``condition_id``
    and return ONLY the description — never prices, volume, or liquidity.

    Returns None on any failure (network error, non-200, no match, condition-id
    mismatch, or an empty/missing description), exactly mirroring the tool's
    production fallback to v1 behaviour.

    :param condition_id: CTF condition id hex (``0x…``); a leading ``poly_``
        prefix is stripped.
    :return: the resolution-rules text, or None.
    """
    cid = condition_id.strip()
    if cid.startswith("poly_"):
        cid = cid[len("poly_") :]
    if not cid:
        return None

    # Gamma's default query hides closed markets, so a resolved (benchmark)
    # market needs closed=true; an open (production) market is returned by the
    # default query. Try open first, then closed — the rules are identical.
    for params in ({"condition_ids": cid}, {"condition_ids": cid, "closed": "true"}):
        try:
            resp = requests.get(
                f"{POLYMARKET_GAMMA_URL}/markets",
                params=params,
                timeout=GAMMA_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            markets = resp.json()
        except Exception as exc:  # noqa: BLE001 — any failure → degrade to v1
            log.warning("Gamma description fetch failed for %s: %s", cid, exc)
            continue

        if not markets:
            continue
        market = markets[0] if isinstance(markets, list) else markets

        # Only trust a response that echoes back the exact condition id asked
        # for; a mismatch means the filter was ignored and the rules would
        # belong to some other market.
        returned_cid = str(market.get("conditionId", "")).lower()
        if not returned_cid or returned_cid != cid.lower():
            continue

        description = (market.get("description") or "").strip()
        if description:
            return description

    return None


def build_request_context(dataset_row: dict[str, Any]) -> dict[str, Any] | None:
    """Build a mech-style request_context from a dataset row.

    Mirrors the trader's ``Bet.to_request_context()`` so tools that read it in
    production (e.g. factual_research-v2) receive the same input under replay:
    ``market_id`` + ``type``, plus the Polymarket resolution rules under
    ``description`` (Omen rows carry none). The description is taken from the
    row when present, else fetched from Gamma the way the trader would. NO
    market price/liquidity is ever included — those tools must never see odds.
    Returns None when the row lacks a market id or platform.

    :param dataset_row: one dataset row with ``market_id`` and ``platform``
        (optionally a pre-baked ``description``).
    :return: request_context dict, or None.
    """
    market_id = dataset_row.get("market_id")
    platform = dataset_row.get("platform")
    if not market_id or not platform:
        return None

    context: dict[str, Any] = {"market_id": market_id, "type": platform}

    if platform == POLYMARKET_PLATFORM:
        # Prefer a pre-baked description (deterministic, offline); otherwise
        # simulate the trader and fetch it from Gamma.
        description = dataset_row.get("description") or _fetch_polymarket_description(
            str(market_id)
        )
        if description:
            context["description"] = description

    return context


def build_output_row(
    dataset_row: dict[str, Any],
    tool_name: str,
    model: str,
    run_result: dict[str, Any],
) -> dict[str, Any]:
    """Build a production_log.jsonl-compatible row from a replay result."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    question_text = dataset_row["question_text"]

    return {
        "row_id": _make_row_id(tool_name, question_text, model),
        "schema_version": "1.0",
        "mode": "cached_replay",
        "market_id": dataset_row.get("market_id"),
        "platform": dataset_row.get("platform", "unknown"),
        "question_text": question_text,
        "tool_name": tool_name,
        "tool_version": None,
        "model": model,
        "config_hash": None,
        "p_yes": run_result["p_yes"],
        "p_no": run_result["p_no"],
        "prediction_parse_status": run_result["prediction_parse_status"],
        "confidence": run_result.get("confidence"),
        "market_prob_at_prediction": None,
        "market_liquidity_at_prediction": None,
        "market_close_at": None,
        "final_outcome": dataset_row["final_outcome"],
        "requested_at": now,
        "predicted_at": now,
        "resolved_at": dataset_row.get("resolved_at"),
        "latency_s": run_result["latency_s"],
        "prediction_lead_time_days": None,
        "category": dataset_row.get("category")
        or classify_category(question_text, dataset_row.get("platform")),
        "match_confidence": 1.0,
    }


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------


def replay(
    dataset_path: Path,
    output_path: Path,
    tools: list[str],
    model: str,
    timeout: int = TASK_DEADLINE,
) -> None:
    """For each dataset row x each tool, run and append output."""
    dataset = load_dataset(dataset_path)
    log.info("Loaded %d replay cases from %s", len(dataset), dataset_path)

    api_keys = build_keychain()
    existing_ids = load_existing_row_ids(output_path)
    if existing_ids:
        log.info("Found %d existing rows (will skip duplicates)", len(existing_ids))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(dataset) * len(tools)
    done = 0
    skipped = 0
    errors = 0

    with open(output_path, "a", encoding="utf-8") as out:
        for row_idx, row in enumerate(dataset):
            if "question_text" not in row or "final_outcome" not in row:
                log.warning(
                    "Skipping row %d: missing required field (question_text or final_outcome)",
                    row_idx,
                )
                continue

            question = row["question_text"]
            source_content = row.get("source_content")

            if source_content is None:
                log.warning(
                    "Skipping row %d: no source_content (would trigger live web fetch): %s",
                    row_idx,
                    question[:60],
                )
                continue

            sc_mode = (
                source_content.get("mode", "unknown")
                if isinstance(source_content, dict)
                else "none"
            )

            for tool_name in tools:
                row_id = _make_row_id(tool_name, question, model)
                if row_id in existing_ids:
                    skipped += 1
                    done += 1
                    continue

                log.info(
                    "[%d/%d] %s (source:%s) | %s",
                    done + 1,
                    total,
                    tool_name,
                    sc_mode,
                    question[:80],
                )

                result = run_single(
                    tool_name=tool_name,
                    question_text=question,
                    source_content=source_content,
                    model=model,
                    api_keys=api_keys,
                    timeout=timeout,
                    request_context=build_request_context(row),
                )

                output_row = build_output_row(row, tool_name, model, result)
                out.write(json.dumps(output_row) + "\n")
                out.flush()

                status = result["prediction_parse_status"]
                if status != "valid":
                    errors += 1
                    log.warning(
                        "  -> %s (error=%s)",
                        status,
                        result.get("error"),
                    )
                else:
                    log.info(
                        "  -> p_yes=%.2f, latency=%ds",
                        result["p_yes"],
                        result["latency_s"],
                    )

                done += 1

    log.info(
        "Done: %d processed, %d skipped, %d errors",
        done - skipped,
        skipped,
        errors,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Replay resolved questions through prediction tools with cached content.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Input JSONL replay dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Comma-separated tool names (default: all registered tools)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use for predictions (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TASK_DEADLINE,
        help=f"Per-tool timeout in seconds (default: {TASK_DEADLINE})",
    )
    args = parser.parse_args()

    if args.tools:
        tools = [t.strip() for t in args.tools.split(",")]
        for t in tools:
            if t not in TOOL_REGISTRY:
                parser.error(f"Unknown tool: {t}. Available: {sorted(TOOL_REGISTRY)}")
    else:
        tools = sorted(TOOL_REGISTRY)

    replay(
        dataset_path=args.dataset,
        output_path=args.output,
        tools=tools,
        model=args.model,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
