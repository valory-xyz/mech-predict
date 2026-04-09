"""
Compare two benchmark scorer outputs.

Takes a baseline and candidate scores.json, computes deltas for every
shared dimension (overall, by_tool, by_platform, by_category), and
prints a markdown comparison table.

Usage:
    python benchmark/compare.py --baseline scores_gpt4o.json --candidate scores_gpt41.json
    python benchmark/compare.py --baseline scores_gpt4o.json --candidate scores_gpt41.json --output comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def _delta(baseline: Optional[float], candidate: Optional[float]) -> Optional[float]:
    """Compute delta between two values. Returns None if either is None."""
    if baseline is None or candidate is None:
        return None
    return round(candidate - baseline, 4)


def _direction(delta: Optional[float], lower_is_better: bool = True) -> str:
    """Classify a delta as improved, regressed, or unchanged."""
    if delta is None:
        return "—"
    if abs(delta) < 0.001:
        return "unchanged"
    if lower_is_better:
        return "improved" if delta < 0 else "regressed"
    return "improved" if delta > 0 else "regressed"


def _fmt(val: Optional[float]) -> str:
    """Format a float for display."""
    if val is None:
        return "—"
    return f"{val:.4f}"


def _fmt_delta(val: Optional[float]) -> str:
    """Format a delta with sign."""
    if val is None:
        return "—"
    return f"{val:+.4f}"


# ---------------------------------------------------------------------------
# Compare a single group stats pair
# ---------------------------------------------------------------------------


def compare_stats(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare two compute_group_stats dicts."""
    metrics = {
        "brier": {"lower_is_better": True},
        "log_loss": {"lower_is_better": True},
        "directional_accuracy": {"lower_is_better": False},
        "sharpness": {"lower_is_better": False},
        "reliability": {"lower_is_better": False},
    }

    result: dict[str, Any] = {
        "baseline_n": baseline.get("n", 0),
        "candidate_n": candidate.get("n", 0),
    }

    for metric, opts in metrics.items():
        b_val = baseline.get(metric)
        c_val = candidate.get(metric)
        d = _delta(b_val, c_val)
        result[metric] = {
            "baseline": b_val,
            "candidate": c_val,
            "delta": d,
            "direction": _direction(d, opts["lower_is_better"]),
        }

    return result


# ---------------------------------------------------------------------------
# Compare a dimension (by_tool, by_platform, by_category)
# ---------------------------------------------------------------------------


def compare_dimension(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compare two grouped breakdowns, returning deltas for shared keys."""
    all_keys = sorted(set(baseline) | set(candidate))
    result: dict[str, dict[str, Any]] = {}

    empty_stats: dict[str, Any] = {
        "brier": None,
        "log_loss": None,
        "directional_accuracy": None,
        "sharpness": None,
        "reliability": None,
        "n": 0,
        "valid_n": 0,
    }

    for key in all_keys:
        b = baseline.get(key, empty_stats)
        c = candidate.get(key, empty_stats)
        result[key] = compare_stats(b, c)

    return result


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare two full scorer outputs."""
    return {
        "overall": compare_stats(
            baseline.get("overall", {}),
            candidate.get("overall", {}),
        ),
        "by_tool": compare_dimension(
            baseline.get("by_tool", {}),
            candidate.get("by_tool", {}),
        ),
        "by_platform": compare_dimension(
            baseline.get("by_platform", {}),
            candidate.get("by_platform", {}),
        ),
        "by_category": compare_dimension(
            baseline.get("by_category", {}),
            candidate.get("by_category", {}),
        ),
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def _table_row(name: str, stats: dict[str, Any]) -> str:
    """Format one row of a comparison table."""
    b = stats["brier"]
    a = stats["directional_accuracy"]
    ll = stats.get("log_loss", {})
    # Show Brier direction + accuracy direction when they disagree
    if b["direction"] == a["direction"] or a["direction"] == "unchanged":
        direction = b["direction"]
    elif b["direction"] == "unchanged":
        direction = a["direction"]
    else:
        direction = f"{b['direction'][:3]}B/{a['direction'][:3]}A"
    ll_b = _fmt(ll.get("baseline")) if ll else "—"
    ll_c = _fmt(ll.get("candidate")) if ll else "—"
    ll_d = _fmt_delta(ll.get("delta")) if ll else "—"
    return (
        f"| {name:<35} "
        f"| {_fmt(b['baseline']):>8} "
        f"| {_fmt(b['candidate']):>8} "
        f"| {_fmt_delta(b['delta']):>8} "
        f"| {ll_b:>8} "
        f"| {ll_c:>8} "
        f"| {ll_d:>8} "
        f"| {_fmt(a['baseline']):>8} "
        f"| {_fmt(a['candidate']):>8} "
        f"| {_fmt_delta(a['delta']):>8} "
        f"| {stats['baseline_n']:>5} "
        f"| {stats['candidate_n']:>5} "
        f"| {direction:<10} |"
    )


def format_markdown(comparison: dict[str, Any]) -> str:
    """Format a full comparison as markdown."""
    lines: list[str] = []

    header = (
        f"| {'':35} "
        f"| {'B.Brier':>8} "
        f"| {'C.Brier':>8} "
        f"| {'Delta':>8} "
        f"| {'B.LL':>8} "
        f"| {'C.LL':>8} "
        f"| {'Delta':>8} "
        f"| {'B.DAcc':>8} "
        f"| {'C.DAcc':>8} "
        f"| {'Delta':>8} "
        f"| {'B.N':>5} "
        f"| {'C.N':>5} "
        f"| {'Direction':<10} |"
    )
    separator = (
        "|" + "|".join(["-" * 36] + ["-" * 9] * 9 + ["-" * 6] * 2 + ["-" * 11]) + "|"
    )

    # Overall
    lines.append("## Overall")
    lines.append("")
    lines.append(header)
    lines.append(separator)
    lines.append(_table_row("Overall", comparison["overall"]))
    lines.append("")

    # Per-tool
    by_tool = comparison.get("by_tool", {})
    if by_tool:
        lines.append("## By Tool")
        lines.append("")
        lines.append(header)
        lines.append(separator)
        for tool in sorted(by_tool):
            lines.append(_table_row(tool, by_tool[tool]))
        lines.append("")

    # Per-platform
    by_platform = comparison.get("by_platform", {})
    if by_platform:
        lines.append("## By Platform")
        lines.append("")
        lines.append(header)
        lines.append(separator)
        for plat in sorted(by_platform):
            lines.append(_table_row(plat, by_platform[plat]))
        lines.append("")

    # Per-category
    by_category = comparison.get("by_category", {})
    if by_category:
        lines.append("## By Category")
        lines.append("")
        lines.append(header)
        lines.append(separator)
        for cat in sorted(by_category):
            lines.append(_table_row(cat, by_category[cat]))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two benchmark scorer outputs.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline scores.json",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate scores.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write comparison to file (default: stdout)",
    )
    args = parser.parse_args()

    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        candidate = json.load(f)

    comparison = compare(baseline, candidate)
    markdown = format_markdown(comparison)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        print(f"Comparison written to {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
