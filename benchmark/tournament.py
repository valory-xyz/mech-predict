"""
Tournament runner: forward-looking predictions on open markets.

Runs prediction tools on currently open markets with live web search
(source_content=None). Stores predictions with timestamps and captured
source_content for future cached replay. Markets are scored later via
score_tournament.py when they resolve.

Usage:
    python benchmark/tournament.py --markets benchmark/datasets/open_markets.jsonl
    python benchmark/tournament.py --markets open_markets.jsonl --tools prediction-online,superforcaster
    python benchmark/tournament.py --markets open_markets.jsonl --max-markets 10
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import platform
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv  # type: ignore[import-not-found]

from benchmark.datasets.fetch_production import classify_category, parse_tool_response

from packages.valory.skills.task_execution.utils.apis import KeyChain

load_dotenv()

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

DEFAULT_OUTPUT = Path(__file__).parent / "results" / "tournament_predictions.jsonl"
DEFAULT_MODEL = "gpt-4.1-2025-04-14"
TASK_DEADLINE = 240  # seconds, matches production


# ---------------------------------------------------------------------------
# Tool registry (duplicated from runner.py — will be extracted later)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Specification for a prediction tool."""

    module: str


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "prediction-online": ToolSpec(
        module="packages.valory.customs.prediction_request.prediction_request",
    ),
    "prediction-offline": ToolSpec(
        module="packages.valory.customs.prediction_request.prediction_request",
    ),
    "claude-prediction-online": ToolSpec(
        module="packages.valory.customs.prediction_request.prediction_request",
    ),
    "claude-prediction-offline": ToolSpec(
        module="packages.valory.customs.prediction_request.prediction_request",
    ),
    "superforcaster": ToolSpec(
        module="packages.valory.customs.superforcaster.superforcaster",
    ),
    "prediction-request-reasoning": ToolSpec(
        module="packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning",
    ),
    "prediction-request-reasoning-claude": ToolSpec(
        module="packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning",
    ),
    "prediction-request-rag": ToolSpec(
        module="packages.napthaai.customs.prediction_request_rag.prediction_request_rag",
    ),
    "prediction-request-rag-claude": ToolSpec(
        module="packages.napthaai.customs.prediction_request_rag.prediction_request_rag",
    ),
    "prediction-url-cot": ToolSpec(
        module="packages.napthaai.customs.prediction_url_cot.prediction_url_cot",
    ),
    "prediction-url-cot-claude": ToolSpec(
        module="packages.napthaai.customs.prediction_url_cot.prediction_url_cot",
    ),
    "prediction-offline-sme": ToolSpec(
        module="packages.nickcom007.customs.prediction_request_sme.prediction_request_sme",
    ),
    "prediction-online-sme": ToolSpec(
        module="packages.nickcom007.customs.prediction_request_sme.prediction_request_sme",
    ),
}


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------


def build_keychain() -> KeyChain:
    """Build a KeyChain for tournament mode.

    Sets return_source_content=true so tools capture their web content
    into used_params for future cached replay.

    :return: a KeyChain populated from environment variables.
    """
    services: dict[str, list[str]] = {
        "openai": [os.environ.get("OPENAI_API_KEY", "")],
        "anthropic": [os.environ.get("ANTHROPIC_API_KEY", "")],
        "google_api_key": [os.environ.get("GOOGLE_API_KEY", "")],
        "google_engine_id": [os.environ.get("GOOGLE_ENGINE_ID", "")],
        "serperapi": [os.environ.get("SERPER_API_KEY", "")],
        "openrouter": [os.environ.get("OPENROUTER_API_KEY", "")],
        "search_provider": [os.environ.get("SEARCH_PROVIDER", "google")],
        # Tournament: capture source_content for future cached replay
        "return_source_content": ["true"],
    }
    return KeyChain(services)


# ---------------------------------------------------------------------------
# Tool loading
# ---------------------------------------------------------------------------

_tool_cache: dict[str, Callable[..., Any]] = {}


def load_tool_run(tool_name: str) -> Callable[..., Any]:
    """Import and cache a tool's run() function."""
    if tool_name in _tool_cache:
        return _tool_cache[tool_name]

    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        raise ValueError(
            f"Unknown tool: {tool_name}. Available: {sorted(TOOL_REGISTRY)}"
        )

    module = importlib.import_module(spec.module)
    run_fn: Callable[..., Any] = module.run
    _tool_cache[tool_name] = run_fn
    return run_fn


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

_HAS_SIGALRM = platform.system() != "Windows"


def _can_use_sigalrm() -> bool:
    """SIGALRM only works on UNIX from the main thread."""
    return _HAS_SIGALRM and threading.current_thread() is threading.main_thread()


class _ToolTimeout(Exception):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:
    raise _ToolTimeout("Tool execution timed out")


# ---------------------------------------------------------------------------
# Row ID generation
# ---------------------------------------------------------------------------


def _make_row_id(tool_name: str, question_text: str, model: str) -> str:
    """Deterministic row ID from tool + question + model."""
    payload = f"{tool_name}:{model}:{question_text}"
    h = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"tourn_{tool_name}_{h}"


# ---------------------------------------------------------------------------
# Core: run a single tool on a single question (live search)
# ---------------------------------------------------------------------------


def run_single(
    tool_name: str,
    question_text: str,
    model: str,
    api_keys: KeyChain,
    timeout: int = TASK_DEADLINE,
) -> dict[str, Any]:
    """Run one tool on one question with live web search.

    :param tool_name: registered tool name.
    :param question_text: the prediction question.
    :param model: LLM model identifier.
    :param api_keys: KeyChain with API credentials.
    :param timeout: per-tool timeout in seconds.
    :return: dict with p_yes, p_no, confidence, prediction_parse_status,
        latency_s, error, source_content.
    """
    run_fn = load_tool_run(tool_name)

    kwargs: dict[str, Any] = {
        "tool": tool_name,
        "prompt": question_text,
        "model": model,
        "api_keys": api_keys,
        "source_content": None,  # None = live web search
        "counter_callback": None,
        "delivery_rate": 100,
    }

    start = time.monotonic()
    use_alarm = _can_use_sigalrm()
    old_handler = None

    try:
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)

        result_tuple = run_fn(**kwargs)
        elapsed = time.monotonic() - start

        # with_key_rotation appends api_keys → index 0 is always the result
        result_str = result_tuple[0]
        parsed = parse_tool_response(result_str)

        # Extract source_content from used_params if captured
        source_content = None
        if len(result_tuple) > 3 and isinstance(result_tuple[3], dict):
            source_content = result_tuple[3].get("source_content")

        return {
            "latency_s": round(elapsed, 1),
            "error": None,
            "source_content": source_content,
            **parsed,
        }
    except _ToolTimeout:
        return {
            "p_yes": None,
            "p_no": None,
            "confidence": None,
            "prediction_parse_status": "timeout",
            "latency_s": timeout,
            "error": "timeout",
            "source_content": None,
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
            "source_content": None,
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
    market: dict[str, Any],
    tool_name: str,
    model: str,
    run_result: dict[str, Any],
) -> dict[str, Any]:
    """Build a tournament prediction row."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    question_text = market["question_text"]

    return {
        "row_id": _make_row_id(tool_name, question_text, model),
        "schema_version": "1.0",
        "mode": "tournament",
        "market_id": market.get("id"),
        "market_address": market.get("market_address"),
        "platform": market.get("platform", "unknown"),
        "question_text": question_text,
        "tool_name": tool_name,
        "tool_version": None,
        "model": model,
        "p_yes": run_result["p_yes"],
        "p_no": run_result["p_no"],
        "prediction_parse_status": run_result["prediction_parse_status"],
        "confidence": run_result.get("confidence"),
        "market_prob_at_prediction": market.get("current_prob"),
        "market_close_at": market.get("close_date"),
        "final_outcome": None,  # Unknown until market resolves
        "predicted_at": now,
        "resolved_at": None,
        "latency_s": run_result["latency_s"],
        "prediction_lead_time_days": None,
        "category": market.get("category") or classify_category(question_text),
        "source_content": run_result.get("source_content"),
    }


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------


def load_markets(path: Path) -> list[dict[str, Any]]:
    """Load open markets from JSONL."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_row_ids(output_path: Path) -> set[str]:
    """Load row IDs of valid predictions for deduplication.

    Only rows with prediction_parse_status == "valid" are considered done.
    Failed/malformed rows are excluded so they can be retried.

    :param output_path: path to the predictions JSONL file.
    :return: set of row IDs that should be skipped.
    """
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    if row.get("prediction_parse_status") == "valid":
                        ids.add(row["row_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


# ---------------------------------------------------------------------------
# Main tournament loop
# ---------------------------------------------------------------------------


def run_tournament(
    markets_path: Path,
    output_path: Path,
    tools: list[str],
    model: str,
    max_markets: Optional[int] = None,
    timeout: int = TASK_DEADLINE,
) -> None:
    """Run all tool x market combos and append predictions."""
    markets = load_markets(markets_path)
    if max_markets is not None:
        markets = markets[:max_markets]

    log.info(
        "Tournament: %d markets x %d tools = %d combos",
        len(markets),
        len(tools),
        len(markets) * len(tools),
    )

    api_keys = build_keychain()
    existing_ids = load_existing_row_ids(output_path)
    if existing_ids:
        log.info(
            "Found %d existing predictions (will skip duplicates)", len(existing_ids)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    errors = 0
    total = len(markets) * len(tools)

    with open(output_path, "a", encoding="utf-8") as out:
        for market in markets:
            question = market.get("question_text", "")
            if not question:
                log.warning("Skipping market %s: no question_text", market.get("id"))
                continue

            for tool_name in tools:
                row_id = _make_row_id(tool_name, question, model)
                if row_id in existing_ids:
                    skipped += 1
                    done += 1
                    continue

                log.info(
                    "[%d/%d] %s | %s",
                    done + 1,
                    total,
                    tool_name,
                    question[:80],
                )

                result = run_single(
                    tool_name=tool_name,
                    question_text=question,
                    model=model,
                    api_keys=api_keys,
                    timeout=timeout,
                )

                output_row = build_output_row(market, tool_name, model, result)
                out.write(json.dumps(output_row, ensure_ascii=False) + "\n")
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
        "Done: %d processed, %d skipped, %d errors. Output: %s",
        done - skipped,
        skipped,
        errors,
        output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tournament: run predictions on open markets."
    )
    parser.add_argument(
        "--markets",
        type=str,
        required=True,
        help="Path to open_markets.jsonl",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=",".join(sorted(TOOL_REGISTRY)),
        help="Comma-separated tool names (default: all registered tools)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Max markets to process (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TASK_DEADLINE,
        help=f"Per-tool timeout in seconds (default: {TASK_DEADLINE})",
    )
    args = parser.parse_args()

    tool_list = [t.strip() for t in args.tools.split(",") if t.strip()]

    # Validate tools
    for t in tool_list:
        if t not in TOOL_REGISTRY:
            parser.error(f"Unknown tool: {t}. Available: {sorted(TOOL_REGISTRY)}")

    run_tournament(
        markets_path=Path(args.markets),
        output_path=Path(args.output),
        tools=tool_list,
        model=args.model,
        max_markets=args.max_markets,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
