"""
Tournament runner: forward-looking predictions on open markets.

Runs prediction tools on currently open markets with live web search
(source_content=None). Stores predictions with timestamps and captured
source_content for future cached replay. Markets are scored later via
score_tournament.py when they resolve.

Usage:
    python benchmark/tournament.py --markets benchmark/datasets/open_markets.jsonl
    python benchmark/tournament.py --markets open_markets.jsonl --tools prediction-online,superforcaster
    python benchmark/tournament.py --markets open_markets.jsonl --max-markets 10  # 10 per platform
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmark.datasets.fetch_production import classify_category, parse_tool_response
from benchmark.io import load_jsonl as load_markets
from benchmark.tools import (
    TOOL_REGISTRY,
    ToolTimeout,
    _can_use_sigalrm,
    alarm_handler,
    build_keychain,
    load_tool_run,
)

# isort treats `benchmark` as third-party (not in known_first_party=autonomy);
# pylint treats it as first-party. The two views disagree on import order.
# isort wins — silence pylint's complaint on the import that follows.
from dotenv import (  # type: ignore[import-not-found]  # pylint: disable=wrong-import-order
    load_dotenv,
)

from packages.valory.skills.task_execution.utils.apis import KeyChain

# ---------------------------------------------------------------------------
# Package hash lookup (for tool version audit trail)
# ---------------------------------------------------------------------------

PACKAGES_JSON = Path(__file__).resolve().parent.parent / "packages" / "packages.json"

# Patterns that look like API keys / tokens (long hex, base64, sk-... etc.)
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|[A-Za-z0-9+/]{32,}={0,2}|[0-9a-f]{32,})",
)


def _sanitize_error(error: str) -> str:
    """Redact potential secrets from error strings."""
    return _SECRET_RE.sub("REDACTED", error)


def _load_package_hashes() -> dict[str, str]:
    """Build a map from tool module path to IPFS hash.

    Reads packages/packages.json and maps the full module path used in
    TOOL_REGISTRY to its IPFS hash. For example, a packages.json entry
    ``"custom/valory/prediction_request/0.1.0": "bafybei..."`` becomes
    ``"packages.valory.customs.prediction_request.prediction_request": "bafybei..."``.

    :return: dict mapping full module path to IPFS hash, empty on error.
    """
    if not PACKAGES_JSON.exists():
        return {}
    try:
        data = json.loads(PACKAGES_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    result: dict[str, str] = {}
    for key, ipfs_hash in data.get("dev", {}).items():
        if not key.startswith("custom/"):
            continue
        # key format: custom/{author}/{package_name}/{version}
        parts = key.split("/")
        if len(parts) < 3:
            continue
        author, pkg = parts[1], parts[2]
        # tools use module path: packages.{author}.customs.{pkg}.{pkg}
        module_path = f"packages.{author}.customs.{pkg}.{pkg}"
        result[module_path] = ipfs_hash
    return result


_PACKAGE_HASHES: dict[str, str] = _load_package_hashes()


def get_tool_ipfs_hash(tool_name: str) -> Optional[str]:
    """Look up the IPFS package hash for a registered tool.

    :param tool_name: tool name from TOOL_REGISTRY.
    :return: IPFS hash string, or None if not found.
    """
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return None
    return _PACKAGE_HASHES.get(spec.module)


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
# Row ID generation
# ---------------------------------------------------------------------------


# TODO: unify _make_row_id across runner, tournament, prompt_replay
# & fetch_production into benchmark/tools.py
def _make_row_id(
    tool_name: str, market_id: str, market_platform: str, model: str
) -> str:
    """Deterministic row ID from tool + market + platform + model."""
    payload = f"{tool_name}:{model}:{market_platform}:{market_id}"
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
            old_handler = signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(timeout)

        result_tuple = run_fn(**kwargs)
        elapsed = time.monotonic() - start

        # with_key_rotation appends api_keys → index 0 is always the result
        result_str = result_tuple[0]
        parsed = parse_tool_response(result_str)

        # Extract source_content from used_params if captured
        source_content = None
        if len(result_tuple) > 4 and isinstance(result_tuple[4], dict):
            source_content = result_tuple[4].get("source_content")

        return {
            "latency_s": round(elapsed, 1),
            "error": None,
            "source_content": source_content,
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
            "error": _sanitize_error(str(e)),
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
        "row_id": _make_row_id(
            tool_name, market.get("id", ""), market.get("platform", ""), model
        ),
        "schema_version": "1.0",
        "mode": "tournament",
        "market_id": market.get("id"),
        "market_address": market.get("market_address"),
        "platform": market.get("platform", "unknown"),
        "question_text": question_text,
        "tool_name": tool_name,
        "tool_version": None,
        "tool_ipfs_hash": get_tool_ipfs_hash(tool_name),
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
        by_platform: dict[str, list[dict[str, Any]]] = {}
        for m in markets:
            by_platform.setdefault(m.get("platform", "unknown"), []).append(m)
        markets = []
        for plat_markets in by_platform.values():
            missing = [m.get("id") for m in plat_markets if not m.get("fetched_at")]
            if missing:
                log.warning(
                    "%d markets missing fetched_at (will sort last): %s",
                    len(missing),
                    missing[:5],
                )
            plat_markets.sort(key=lambda m: m.get("fetched_at", ""), reverse=True)
            markets.extend(plat_markets[:max_markets])

    log.info(
        "Tournament: %d markets x %d tools = %d combos",
        len(markets),
        len(tools),
        len(markets) * len(tools),
    )

    api_keys = build_keychain(return_source_content=True)
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
                market_id = market.get("id", "")
                market_platform = market.get("platform", "")
                row_id = _make_row_id(tool_name, market_id, market_platform, model)
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
        help="Max markets per platform to process (default: all)",
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
