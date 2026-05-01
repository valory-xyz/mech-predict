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

from dotenv import load_dotenv  # type: ignore[import-not-found]

from benchmark.datasets.fetch_production import classify_category, parse_tool_response
from benchmark.io import load_jsonl as load_markets
from benchmark.ipfs_loader import IpfsFetchError
from benchmark.tools import (
    ToolTimeout,
    _can_use_sigalrm,
    alarm_handler,
    build_keychain,
    load_tool_run,
)

from packages.valory.skills.task_execution.utils.apis import KeyChain

# ---------------------------------------------------------------------------
# Tournament tools — single source of truth (TOURNAMENT_IPFS_LOADER_SPEC §3.1)
# ---------------------------------------------------------------------------

TOURNAMENT_TOOLS_JSON = Path(__file__).resolve().parent / "tournament_tools.json"

# Patterns that look like API keys / tokens (long hex, base64, sk-... etc.)
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|[A-Za-z0-9+/]{32,}={0,2}|[0-9a-f]{32,})",
)


def _sanitize_error(error: str) -> str:
    """Redact potential secrets from error strings."""
    return _SECRET_RE.sub("REDACTED", error)


def load_tournament_tools(path: Path = TOURNAMENT_TOOLS_JSON) -> dict[str, str]:
    """Load the tournament tool→CID map from JSON.

    :param path: path to tournament_tools.json. Defaults to the file shipped
        alongside this module.
    :return: dict mapping tool_name → IPFS CID.
    :raises FileNotFoundError: if the file is missing.
    :raises ValueError: if the file is malformed (not a dict[str, str]).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Tournament tools config not found: {path}. "
            "Tournament cannot run without it."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tournament tools config is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Tournament tools config must be a JSON object, got {type(data).__name__}: {path}"
        )
    for tool_name, cid in data.items():
        if not isinstance(tool_name, str) or not isinstance(cid, str):
            raise ValueError(
                f"Tournament tools config must map str→str; "
                f"got {tool_name!r}→{cid!r} in {path}"
            )
        if not tool_name or not cid:
            raise ValueError(
                f"Tournament tools config has empty tool_name or CID: "
                f"{tool_name!r}→{cid!r} in {path}"
            )
    return data


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
    cid: str,
    timeout: int = TASK_DEADLINE,
    cache_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run one tool on one question with live web search.

    :param tool_name: tournament tool name (key in tournament_tools.json).
    :param question_text: the prediction question.
    :param model: LLM model identifier.
    :param api_keys: KeyChain with API credentials.
    :param cid: IPFS CID of the tool package to fetch and exec.
    :param timeout: per-tool timeout in seconds.
    :param cache_dir: optional override for the IPFS source cache directory.
    :return: dict with p_yes, p_no, confidence, prediction_parse_status,
        latency_s, error, source_content. ``prediction_parse_status`` is
        ``"ipfs_fetch_error"`` when the tool source can't be fetched/exec'd
        from IPFS; the row is recorded and the tournament continues so one
        bad CID doesn't poison the rest of the run.
    """
    try:
        run_fn = load_tool_run(tool_name, cid=cid, cache_dir=cache_dir)
    except IpfsFetchError as exc:
        log.warning("IPFS fetch failed: tool=%s cid=%s err=%s", tool_name, cid, exc)
        return {
            "p_yes": None,
            "p_no": None,
            "confidence": None,
            "prediction_parse_status": "ipfs_fetch_error",
            "latency_s": 0,
            "error": _sanitize_error(str(exc)),
            "source_content": None,
        }

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
    cid: str,
) -> dict[str, Any]:
    """Build a tournament prediction row.

    :param market: market dict (id, platform, question_text, etc.).
    :param tool_name: tournament tool name.
    :param model: LLM model identifier used by the tool.
    :param run_result: dict returned by ``run_single``.
    :param cid: IPFS CID of the tool package that produced ``run_result``;
        recorded as ``tool_ipfs_hash`` for the audit trail.
    :return: dict ready to serialize as a JSONL row.
    """
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
        "tool_ipfs_hash": cid,
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


def _skipped_global_timeout_row(
    market: dict[str, Any], tool_name: str, model: str, cid: str
) -> dict[str, Any]:
    """Build a row recording that we ran out of wall-clock budget."""
    return build_output_row(
        market,
        tool_name,
        model,
        {
            "p_yes": None,
            "p_no": None,
            "confidence": None,
            "prediction_parse_status": "skipped_global_timeout",
            "latency_s": 0,
            "error": "skipped_global_timeout",
            "source_content": None,
        },
        cid,
    )


def _select_markets(
    markets: list[dict[str, Any]], max_markets: Optional[int]
) -> list[dict[str, Any]]:
    """Bucket markets by platform, sort newest-first, take the top ``max_markets`` per platform.

    :param markets: full list loaded from the JSONL file.
    :param max_markets: max markets to keep per platform (None = keep all).
    :return: the filtered, ordered list.
    """
    if max_markets is None:
        return markets
    by_platform: dict[str, list[dict[str, Any]]] = {}
    for market in markets:
        by_platform.setdefault(market.get("platform", "unknown"), []).append(market)
    out: list[dict[str, Any]] = []
    for plat_markets in by_platform.values():
        missing = [m.get("id") for m in plat_markets if not m.get("fetched_at")]
        if missing:
            log.warning(
                "%d markets missing fetched_at (will sort last): %s",
                len(missing),
                missing[:5],
            )
        plat_markets.sort(key=lambda m: m.get("fetched_at", ""), reverse=True)
        out.extend(plat_markets[:max_markets])
    return out


def _log_result(result: dict[str, Any]) -> None:
    """Log the outcome of a single run_single call."""
    status = result["prediction_parse_status"]
    if status == "valid":
        log.info(
            "  -> p_yes=%.2f, latency=%ds",
            result["p_yes"],
            result["latency_s"],
        )
    else:
        log.warning("  -> %s (error=%s)", status, result.get("error"))


def run_tournament(
    markets_path: Path,
    output_path: Path,
    tools_to_cid: dict[str, str],
    model: str,
    max_markets: Optional[int] = None,
    timeout: int = TASK_DEADLINE,
    cache_dir: Optional[Path] = None,
    global_timeout: Optional[int] = None,
) -> None:
    """Run all tool x market combos and append predictions.

    :param markets_path: path to ``open_markets.jsonl``.
    :param output_path: JSONL path to append predictions to.
    :param tools_to_cid: ordered map ``{tool_name: ipfs_cid}`` defining which
        tools run and at which IPFS CID. Iteration order is preserved for the
        per-market inner loop, so ``--tools`` ordering controls execution
        order.
    :param model: LLM model identifier passed to each tool.
    :param max_markets: keep this many markets per platform (None = all).
    :param timeout: per-tool wall-clock timeout in seconds.
    :param cache_dir: optional override for the IPFS source cache directory.
    :param global_timeout: optional wall-clock budget (seconds) for the whole
        run. When set, unprocessed combos after the budget is exceeded are
        recorded as ``skipped_global_timeout`` rows.
    """
    tools = list(tools_to_cid.keys())
    markets = _select_markets(load_markets(markets_path), max_markets)

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
    started_at = time.monotonic()
    out_of_budget = False

    with open(output_path, "a", encoding="utf-8") as out:
        for market in markets:
            question = market.get("question_text", "")
            if not question:
                log.warning("Skipping market %s: no question_text", market.get("id"))
                continue

            for tool_name in tools:
                cid = tools_to_cid[tool_name]
                row_id = _make_row_id(
                    tool_name,
                    market.get("id", ""),
                    market.get("platform", ""),
                    model,
                )
                if row_id in existing_ids:
                    skipped += 1
                    done += 1
                    continue

                if (
                    global_timeout is not None
                    and time.monotonic() - started_at > global_timeout
                ):
                    out_of_budget = True
                    log.warning(
                        "Global timeout %ds exceeded; recording remaining "
                        "combos as skipped_global_timeout",
                        global_timeout,
                    )
                    row = _skipped_global_timeout_row(market, tool_name, model, cid)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    errors += 1
                    done += 1
                    continue

                log.info("[%d/%d] %s | %s", done + 1, total, tool_name, question[:80])
                result = run_single(
                    tool_name=tool_name,
                    question_text=question,
                    model=model,
                    api_keys=api_keys,
                    cid=cid,
                    timeout=timeout,
                    cache_dir=cache_dir,
                )
                output_row = build_output_row(market, tool_name, model, result, cid)
                out.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                out.flush()
                _log_result(result)
                if result["prediction_parse_status"] != "valid":
                    errors += 1
                done += 1

    log.info(
        "Done: %d processed, %d skipped, %d errors%s. Output: %s",
        done - skipped,
        skipped,
        errors,
        " (global-timeout fired)" if out_of_budget else "",
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
        default=None,
        help=(
            "Comma-separated subset of tools to run "
            "(default: every key in tournament_tools.json)"
        ),
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
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Override IPFS source cache directory "
            "(default: ~/.cache/mech-predict/tournament-tools)"
        ),
    )
    parser.add_argument(
        "--global-timeout",
        type=int,
        default=None,
        help=(
            "Wall-clock budget (seconds) for the whole run; remaining combos "
            "are recorded as skipped_global_timeout (default: unset)"
        ),
    )
    args = parser.parse_args()

    try:
        tournament_tools = load_tournament_tools()
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if args.tools is None:
        tool_list = list(tournament_tools.keys())
    else:
        tool_list = [t.strip() for t in args.tools.split(",") if t.strip()]
        for t in tool_list:
            if t not in tournament_tools:
                parser.error(
                    f"Unknown tool: {t}. "
                    f"Available in tournament_tools.json: {sorted(tournament_tools)}"
                )

    tools_to_cid: dict[str, str] = {t: tournament_tools[t] for t in tool_list}

    run_tournament(
        markets_path=Path(args.markets),
        output_path=Path(args.output),
        tools_to_cid=tools_to_cid,
        model=args.model,
        max_markets=args.max_markets,
        timeout=args.timeout,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        global_timeout=args.global_timeout,
    )


if __name__ == "__main__":
    main()
