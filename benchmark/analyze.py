"""
Generate a human-readable benchmark report.

Reads scores.json (current month accumulators) and scores_history.jsonl
(monthly snapshots) to produce a markdown report with rankings, weak
spots, and highlights.  Never reads raw log files.

Usage:
    python benchmark/analyze.py
    python benchmark/analyze.py --scores path/to/scores.json --output path/to/report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SCORES = Path(__file__).parent / "results" / "scores.json"
DEFAULT_HISTORY = Path(__file__).parent / "results" / "scores_history.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "report.md"

BRIER_RANDOM = 0.25
BRIER_WEAK_THRESHOLD = 0.40
RELIABILITY_ISSUE_THRESHOLD = 0.90
SAMPLE_SIZE_WARNING = 20
TREND_WORSENING_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_scores(path: Path) -> dict[str, Any]:
    """Load scores from a JSON file."""
    return json.loads(path.read_text())


def load_history(path: Path) -> list[dict[str, Any]]:
    """Load monthly snapshots from a JSONL file.

    :param path: path to ``scores_history.jsonl``.
    :return: list of monthly summary dicts.
    """
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def section_overall(scores: dict[str, Any]) -> str:
    """Generate the overall summary section."""
    o = scores["overall"]
    if scores["total_rows"] == 0:
        return "## Overall\n\nNo predictions to score."

    rel_str = f"{o['reliability']:.0%}" if o["reliability"] is not None else "N/A"
    brier_str = str(o["brier"]) if o["brier"] is not None else "N/A"
    acc_str = f"{o['accuracy']:.0%}" if o.get("accuracy") is not None else "N/A"
    sharp_str = f"{o['sharpness']:.4f}" if o.get("sharpness") is not None else "N/A"
    lines = [
        "## Overall",
        "",
        f"- Predictions scored: {scores['valid_rows']} / {scores['total_rows']}"
        f" ({rel_str} reliability)",
        f"- Overall Brier: {brier_str}",
        "  - 0.0 = perfect, 0.25 = random guessing, 1.0 = maximally wrong",
        f"- Accuracy: {acc_str}",
        f"- Sharpness: {sharp_str}",
        "  - 0.0 = all predictions at 50/50, 0.5 = maximally decisive",
    ]
    return "\n".join(lines)


def _sample_label(stats: dict[str, Any]) -> str:
    """Return a sample-size label for ranking context."""
    if stats.get("decision_worthy") is False:
        return " ⚠ low sample"
    return ""


def section_tool_ranking(scores: dict[str, Any]) -> str:
    """Generate the tool ranking section."""
    tools = scores.get("by_tool", {})
    ranked = sorted(
        tools.items(),
        key=lambda x: x[1].get("brier") if x[1].get("brier") is not None else 999,
    )

    lines = ["## Tool Ranking", ""]
    for i, (tool, stats) in enumerate(ranked, 1):
        flags = ""
        if (
            stats.get("reliability") is not None
            and stats["reliability"] < RELIABILITY_ISSUE_THRESHOLD
        ):
            flags = f" — {stats['reliability']:.0%} reliability"
        flags += _sample_label(stats)
        brier = stats["brier"] if stats["brier"] is not None else "N/A"
        acc = f"{stats['accuracy']:.0%}" if stats.get("accuracy") is not None else "N/A"
        sharp = (
            f"{stats['sharpness']:.4f}" if stats.get("sharpness") is not None else "N/A"
        )
        lines.append(
            f"{i}. **{tool}** — Brier: {brier}, Acc: {acc},"
            f" Sharp: {sharp} (n={stats['n']}){flags}"
        )

    return "\n".join(lines)


def section_platform(scores: dict[str, Any]) -> str:
    """Generate the platform comparison section."""
    platforms = scores.get("by_platform", {})
    lines = ["## Platform Comparison", ""]
    for platform, stats in sorted(
        platforms.items(),
        key=lambda x: x[1].get("brier") if x[1].get("brier") is not None else 999,
    ):
        lines.append(f"- **{platform}**: Brier: {stats['brier']} (n={stats['n']})")
    return "\n".join(lines)


def section_weak_spots(scores: dict[str, Any]) -> str:
    """Generate the weak spots section."""
    lines = ["## Weak Spots", ""]
    found = False

    for section_name, section_key in [
        ("category", "by_category"),
        ("platform", "by_platform"),
        ("tool", "by_tool"),
    ]:
        for name, stats in (scores.get(section_key) or {}).items():
            brier = stats.get("brier")
            if brier is not None and brier > BRIER_WEAK_THRESHOLD:
                found = True
                label = (
                    "anti-predictive (worse than coin flip)"
                    if brier > 0.5
                    else "weak performance"
                )
                lines.append(
                    f"- **{name}** ({section_name}): Brier {brier:.4f} (n={stats['n']})"
                    f" — {label}"
                )

    if not found:
        lines.append(
            f"No weak spots detected (all Brier scores below {BRIER_WEAK_THRESHOLD})."
        )

    return "\n".join(lines)


def section_reliability_issues(scores: dict[str, Any]) -> str:
    """Generate the reliability issues section."""
    lines = ["## Reliability Issues", ""]
    found = False

    for tool, stats in (scores.get("by_tool") or {}).items():
        rel = stats.get("reliability")
        if rel is not None and rel < RELIABILITY_ISSUE_THRESHOLD:
            found = True
            error_pct = (1 - rel) * 100
            lines.append(f"- **{tool}**: {error_pct:.1f}% error/malformed rate")

    if not found:
        lines.append("All tools above 90% reliability.")

    return "\n".join(lines)


def section_worst_predictions(scores: dict[str, Any], n: int = 10) -> str:
    """Generate the worst predictions section from scores.worst_10.

    :param scores: scores dict with ``worst_10`` list.
    :param n: max entries to show.
    :return: markdown section string.
    """
    entries = scores.get("worst_10", [])
    lines = ["## Worst Predictions", ""]
    if not entries:
        lines.append("No prediction data available.")
        return "\n".join(lines)

    for i, entry in enumerate(entries[:n], 1):
        outcome_str = "Yes" if entry["final_outcome"] else "No"
        q = entry.get("question_text", "?")
        if len(q) > 80:
            q = q[:77] + "..."
        lines.append(
            f'{i}. "{q}"'
            f"\n   {entry['tool_name']} predicted p_yes={entry['p_yes']:.2f},"
            f" outcome: {outcome_str} (Brier: {entry['brier']:.4f})"
            f"\n   Category: {entry.get('category', '?')},"
            f" Platform: {entry.get('platform', '?')}"
        )

    return "\n".join(lines)


def section_best_predictions(scores: dict[str, Any], n: int = 10) -> str:
    """Generate the best predictions section from scores.best_10.

    :param scores: scores dict with ``best_10`` list.
    :param n: max entries to show.
    :return: markdown section string.
    """
    entries = scores.get("best_10", [])
    lines = ["## Best Predictions", ""]
    if not entries:
        lines.append("No prediction data available.")
        return "\n".join(lines)

    for i, entry in enumerate(entries[:n], 1):
        outcome_str = "Yes" if entry["final_outcome"] else "No"
        q = entry.get("question_text", "?")
        if len(q) > 80:
            q = q[:77] + "..."
        lines.append(
            f'{i}. "{q}"'
            f"\n   {entry['tool_name']} predicted p_yes={entry['p_yes']:.2f},"
            f" outcome: {outcome_str} (Brier: {entry['brier']:.4f})"
            f"\n   Category: {entry.get('category', '?')},"
            f" Platform: {entry.get('platform', '?')}"
        )

    return "\n".join(lines)


def section_trend(
    history: list[dict[str, Any]],
    scores: dict[str, Any] | None = None,
) -> str:
    """Generate the trend section from monthly history + current month.

    :param history: list of monthly snapshot dicts from scores_history.jsonl.
    :param scores: current scores.json dict (appended as in-progress month).
    :return: markdown section string.
    """
    # Build full trend: completed months + current month
    trend: list[dict[str, Any]] = list(history)
    if scores and scores.get("current_month") and scores.get("overall"):
        trend.append(
            {
                "month": scores["current_month"],
                "overall": scores["overall"],
            }
        )

    lines = ["## Trend", ""]

    if not trend:
        lines.append("No trend data available.")
        return "\n".join(lines)

    for entry in trend:
        overall = entry.get("overall", {})
        brier = overall.get("brier")
        n = overall.get("n", 0)
        suffix = " *(in progress)*" if entry is trend[-1] and scores else ""
        lines.append(f"- {entry['month']}: Brier {brier} (n={n}){suffix}")

    # Check for worsening
    if len(trend) >= 2:
        prev = trend[-2].get("overall", {}).get("brier")
        curr = trend[-1].get("overall", {}).get("brier")
        if prev is not None and curr is not None:
            delta = curr - prev
            if delta > TREND_WORSENING_THRESHOLD:
                lines.append(
                    f"\n**Warning:** Brier worsened by {delta:.4f}"
                    f" ({prev:.4f} → {curr:.4f}) in the last month."
                )

    return "\n".join(lines)


def section_sample_size_warnings(scores: dict[str, Any]) -> str:
    """Generate the sample size warnings section."""
    lines = ["## Sample Size Warnings", ""]
    found = False

    for cat, stats in (scores.get("by_category") or {}).items():
        if stats["n"] < SAMPLE_SIZE_WARNING:
            found = True
            lines.append(
                f"- **{cat}**: only {stats['n']} questions — treat with caution"
            )

    if not found:
        lines.append("All categories have sufficient sample size.")

    return "\n".join(lines)


def section_tool_platform(scores: dict[str, Any]) -> str:
    """Tool x platform cross breakdown table."""
    data = scores.get("by_tool_platform", {})
    if not data:
        return "## Tool × Platform\n\nNo cross-breakdown data available."

    lines = [
        "## Tool × Platform",
        "",
        "| Tool | Platform | Brier | Accuracy | Sharpness | n |",
        "|------|----------|-------|----------|-----------|---|",
    ]
    for key, stats in sorted(
        data.items(),
        key=lambda x: x[1].get("brier") if x[1].get("brier") is not None else 999,
    ):
        parts = key.split(" | ")
        tool = parts[0] if parts else key
        platform = parts[1] if len(parts) > 1 else "?"
        brier = f"{stats['brier']:.4f}" if stats.get("brier") is not None else "N/A"
        acc = f"{stats['accuracy']:.0%}" if stats.get("accuracy") is not None else "N/A"
        sharp = (
            f"{stats['sharpness']:.4f}" if stats.get("sharpness") is not None else "N/A"
        )
        label = _sample_label(stats)
        lines.append(
            f"| {tool} | {platform} | {brier} | {acc} | {sharp} | {stats['n']}{label} |"
        )

    return "\n".join(lines)


def section_calibration(scores: dict[str, Any]) -> str:
    """Calibration analysis — are predictions overconfident or underconfident?"""
    cal = scores.get("calibration", [])
    if not cal:
        return "## Calibration\n\nNo calibration data available."

    lines = [
        "## Calibration",
        "",
        "| Predicted Range | Avg Predicted | Realized Yes-Rate | Gap | n |",
        "|-----------------|---------------|-------------------|-----|---|",
    ]
    for bucket in cal:
        if bucket.get("n", 0) == 0:
            continue
        avg_p = f"{bucket['avg_predicted']:.2f}"
        realized = f"{bucket['realized_rate']:.2f}"
        gap = bucket["gap"]
        gap_str = f"{gap:+.2f}"
        lines.append(
            f"| {bucket['bin']} | {avg_p} | {realized} | {gap_str} | {bucket['n']} |"
        )

    # Summary interpretation
    lines.append("")
    high_conf = [
        b for b in cal if b.get("avg_predicted", 0) > 0.7 and b.get("n", 0) > 0
    ]
    low_conf = [b for b in cal if b.get("avg_predicted", 0) < 0.3 and b.get("n", 0) > 0]
    if high_conf:
        avg_gap = sum(b["gap"] for b in high_conf) / len(high_conf)
        if avg_gap > 0.1:
            lines.append(
                "**High-confidence predictions are overconfident** — predicted high yes-probability"
                " but realized rate is much lower."
            )
        elif avg_gap < -0.1:
            lines.append(
                "**High-confidence predictions are underconfident** — realized rate exceeds predictions."
            )
    if low_conf:
        avg_gap = sum(b["gap"] for b in low_conf) / len(low_conf)
        if avg_gap < -0.1:
            lines.append(
                "**Low-confidence predictions are underconfident** — predicted low yes-probability"
                " but events happen more often than predicted."
            )

    return "\n".join(lines)


def section_parse_breakdown(scores: dict[str, Any]) -> str:
    """Per-tool parse status breakdown from scores.parse_breakdown.

    :param scores: scores dict with ``parse_breakdown`` mapping.
    :return: markdown section string.
    """
    by_tool = scores.get("parse_breakdown", {})
    if not by_tool:
        return "## Parse/Error Breakdown by Tool\n\nNo parse data available."

    lines = [
        "## Parse/Error Breakdown by Tool",
        "",
        "| Tool | Valid | Malformed | Missing | Error | Total |",
        "|------|-------|-----------|---------|-------|-------|",
    ]
    for tool in sorted(by_tool):
        c = by_tool[tool]
        total = sum(c.values())
        lines.append(
            f"| {tool} | {c.get('valid', 0)} | {c.get('malformed', 0)}"
            f" | {c.get('missing_fields', 0)} | {c.get('error', 0)} | {total} |"
        )

    return "\n".join(lines)


def section_latency(scores: dict[str, Any]) -> str:
    """Latency breakdown from scores.latency_reservoir.

    :param scores: scores dict with ``latency_reservoir`` mapping.
    :return: markdown section string.
    """
    by_tool = scores.get("latency_reservoir", {})
    if not by_tool:
        return "## Latency\n\nNo latency data available."

    # Filter out empty reservoirs
    by_tool = {t: vals for t, vals in by_tool.items() if vals}
    if not by_tool:
        return "## Latency\n\nNo latency data available."

    lines = [
        "## Latency (seconds)",
        "",
        "| Tool | Median | Mean | p95 | n |",
        "|------|--------|------|-----|---|",
    ]
    for tool in sorted(by_tool, key=lambda t: statistics.median(by_tool[t])):
        vals = sorted(by_tool[tool])
        med = statistics.median(vals)
        mean = statistics.mean(vals)
        p95_idx = min(int(len(vals) * 0.95), len(vals) - 1)
        p95 = vals[p95_idx]
        lines.append(
            f"| {tool} | {med:.0f}s | {mean:.0f}s | {p95:.0f}s | {len(vals)} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def generate_report(
    scores: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a full benchmark report from scores and history.

    :param scores: parsed ``scores.json`` dict.
    :param history: list of monthly snapshots from ``scores_history.jsonl``.
    :return: full markdown report string.
    """
    if history is None:
        history = []

    date = scores.get("generated_at", "")[:10] or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d"
    )

    sections = [
        f"# Benchmark Report — {date}",
        section_overall(scores),
        section_tool_ranking(scores),
        section_platform(scores),
        section_tool_platform(scores),
        section_calibration(scores),
        section_weak_spots(scores),
        section_reliability_issues(scores),
        section_parse_breakdown(scores),
        section_latency(scores),
        section_worst_predictions(scores),
        section_best_predictions(scores),
        section_trend(history, scores),
        section_sample_size_warnings(scores),
    ]

    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from scores.",
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    scores = load_scores(args.scores)
    history = load_history(args.history)
    print(
        f"Loaded scores ({scores.get('total_rows', 0)} rows), {len(history)} months of history"
    )

    report = generate_report(scores, history)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to {args.output}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
