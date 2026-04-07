"""
Prepare context for the Claude Code Action tool improvement workflow.

Reads scores.json and report.md, identifies the worst-performing tool,
and writes a single suggestion_context.md file with everything Claude
needs to diagnose and fix the tool — saving ~5 turns of file discovery.

Usage:
    python -m benchmark.prepare_suggestion
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
SCORES_PATH = RESULTS_DIR / "scores.json"
REPORT_PATH = RESULTS_DIR / "report.md"
OUTPUT_PATH = RESULTS_DIR / "suggestion_context.md"

BRIER_THRESHOLD = 0.27
MIN_VALID_N = 50
MIN_RELIABILITY = 0.80

# Standalone registry — no import from runner.py to avoid package sync dependency
TOOL_REGISTRY = {
    "prediction-online": "packages/valory/customs/prediction_request/prediction_request.py",
    "prediction-offline": "packages/valory/customs/prediction_request/prediction_request.py",
    "claude-prediction-online": "packages/valory/customs/prediction_request/prediction_request.py",
    "claude-prediction-offline": "packages/valory/customs/prediction_request/prediction_request.py",
    "superforcaster": "packages/valory/customs/superforcaster/superforcaster.py",
    "prediction-request-reasoning": "packages/napthaai/customs/prediction_request_reasoning/prediction_request_reasoning.py",
    "prediction-request-reasoning-claude": "packages/napthaai/customs/prediction_request_reasoning/prediction_request_reasoning.py",
    "prediction-request-rag": "packages/napthaai/customs/prediction_request_rag/prediction_request_rag.py",
    "prediction-request-rag-claude": "packages/napthaai/customs/prediction_request_rag/prediction_request_rag.py",
    "prediction-url-cot": "packages/napthaai/customs/prediction_url_cot/prediction_url_cot.py",
    "prediction-url-cot-claude": "packages/napthaai/customs/prediction_url_cot/prediction_url_cot.py",
    "prediction-offline-sme": "packages/nickcom007/customs/prediction_request_sme/prediction_request_sme.py",
    "prediction-online-sme": "packages/nickcom007/customs/prediction_request_sme/prediction_request_sme.py",
}


def _find_worst_tool(by_tool: dict[str, Any]) -> tuple[str, dict] | None:
    """Find the worst tool by Brier score that meets quality thresholds."""
    candidates = []
    for tool, stats in by_tool.items():
        brier = stats.get("brier")
        valid_n = stats.get("valid_n", 0)
        reliability = stats.get("reliability", 0)
        if (
            brier is not None
            and brier > BRIER_THRESHOLD
            and valid_n >= MIN_VALID_N
            and reliability is not None
            and reliability >= MIN_RELIABILITY
        ):
            candidates.append((tool, stats, brier))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0], candidates[0][1]


def _check_existing_work(tool_name: str) -> str | None:
    """Check for existing open PRs/issues for this tool."""
    for cmd in [
        ["gh", "pr", "list", "--label", "auto-improvement", "--state", "open", "--json", "title,number"],
        ["gh", "issue", "list", "--label", "auto-improvement", "--state", "open", "--json", "title,number"],
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                items = json.loads(result.stdout)
                for item in items:
                    if tool_name in item.get("title", ""):
                        return f"#{item['number']}: {item['title']}"
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            continue
    return None


def _extract_report_lines(report: str, tool_name: str) -> str:
    """Extract lines from report.md that mention this tool."""
    lines = []
    for line in report.splitlines():
        if tool_name in line:
            lines.append(line)
    return "\n".join(lines) if lines else "No mentions found in report."


def _build_context(
    tool_name: str,
    stats: dict,
    scores: dict,
    report: str,
) -> str:
    """Build the suggestion_context.md content."""
    source_file = TOOL_REGISTRY.get(tool_name, "UNKNOWN")

    sections = [
        f"# Suggestion Context: `{tool_name}`",
        "",
        "## Target Tool",
        f"- **Tool:** `{tool_name}`",
        f"- **Source file:** `{source_file}`",
        f"- **Brier:** {stats.get('brier')}",
        f"- **Accuracy:** {stats.get('accuracy')}",
        f"- **Sharpness:** {stats.get('sharpness')}",
        f"- **Reliability:** {stats.get('reliability')}",
        f"- **Valid predictions:** {stats.get('valid_n')}",
        f"- **Total rows:** {stats.get('n')}",
    ]

    # Parse breakdown
    parse = scores.get("parse_breakdown_by_tool", {}).get(tool_name)
    if parse:
        sections.extend(["", "## Parse Breakdown"])
        for status, count in sorted(parse.items()):
            sections.append(f"- {status}: {count}")

    # Overconfidence
    overconf = scores.get("overconfidence_by_tool", {}).get(tool_name)
    if overconf:
        sections.extend(["", "## Overconfidence"])
        sections.append(f"- High-confidence predictions (p>0.9 or p<0.1): {overconf.get('high_confidence_n')}")
        sections.append(f"- Wrong: {overconf.get('high_confidence_wrong')}")
        sections.append(f"- Wrong rate: {overconf.get('high_confidence_wrong_rate')}")

    # Calibration
    cal = scores.get("calibration_by_tool", {}).get(tool_name)
    if cal:
        sections.extend(["", "## Calibration"])
        sections.append("| Predicted Range | Avg Predicted | Realized | Gap | n |")
        sections.append("|-----------------|---------------|----------|-----|---|")
        for bucket in cal:
            sections.append(
                f"| {bucket['bin']} | {bucket['avg_predicted']} "
                f"| {bucket['realized_rate']} | {bucket['gap']} | {bucket['n']} |"
            )

    # Platform breakdown
    by_tp = scores.get("by_tool_platform", {})
    # Keys use " | " separator, e.g. "prediction-request-rag | omen"
    platform_rows = {k: v for k, v in by_tp.items() if k.startswith(f"{tool_name} | ")}
    if platform_rows:
        sections.extend(["", "## Platform Breakdown"])
        for key, pstats in sorted(platform_rows.items()):
            platform = key.split(" | ", 1)[1] if " | " in key else key
            sections.append(
                f"- **{platform}:** Brier {pstats.get('brier')}, "
                f"Acc {pstats.get('accuracy')}, n={pstats.get('n')}"
            )

    # Report mentions
    mentions = _extract_report_lines(report, tool_name)
    sections.extend(["", "## Report Mentions", mentions])

    # Top 3 tools for comparison
    by_tool = scores.get("by_tool", {})
    ranked = sorted(
        [(t, s) for t, s in by_tool.items() if s.get("brier") is not None and s.get("valid_n", 0) >= MIN_VALID_N],
        key=lambda x: x[1]["brier"],
    )
    if ranked:
        sections.extend(["", "## Best Tools (for comparison)"])
        for t, s in ranked[:3]:
            src = TOOL_REGISTRY.get(t, "UNKNOWN")
            sections.append(f"- `{t}` — Brier {s['brier']}, source: `{src}`")

    return "\n".join(sections)


def main() -> None:
    """Identify worst tool and write suggestion context."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not SCORES_PATH.exists():
        log.warning("scores.json not found at %s", SCORES_PATH)
        OUTPUT_PATH.write_text("NO_TARGET — scores.json not found\n")
        return

    scores = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
    report = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""

    target = _find_worst_tool(scores.get("by_tool", {}))
    if not target:
        log.info("No tool has Brier > %.2f — nothing to improve.", BRIER_THRESHOLD)
        OUTPUT_PATH.write_text("NO_TARGET — all tools performing above baseline\n")
        return

    tool_name, stats = target
    log.info("Worst tool: %s (Brier %.4f)", tool_name, stats["brier"])

    existing = _check_existing_work(tool_name)
    if existing:
        msg = f"DUPLICATE — existing open work for {tool_name}: {existing}\n"
        log.info(msg)
        OUTPUT_PATH.write_text(msg)
        return

    context = _build_context(tool_name, stats, scores, report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(context, encoding="utf-8")
    log.info("Wrote suggestion context to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
