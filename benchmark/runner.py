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
) -> dict[str, Any]:
    """Run one tool on one question and return parsed result.

    :param tool_name: registered tool name.
    :param question_text: the prediction question.
    :param source_content: cached web content for replay.
    :param model: LLM model identifier.
    :param api_keys: KeyChain with API credentials.
    :param timeout: per-tool timeout in seconds.
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
        "category": dataset_row.get("category") or classify_category(question_text),
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
