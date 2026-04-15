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

from benchmark.compare import compare_stats
from benchmark.io import load_jsonl
from benchmark.scorer import DISAGREE_THRESHOLD, LARGE_TRADE_THRESHOLD, MIN_SAMPLE_SIZE

DEFAULT_SCORES = Path(__file__).parent / "results" / "scores.json"
DEFAULT_HISTORY = Path(__file__).parent / "results" / "scores_history.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "report.md"

BRIER_RANDOM = 0.25
BRIER_WEAK_THRESHOLD = 0.40
BSS_HARMFUL_THRESHOLD = 0.0
RELIABILITY_ISSUE_THRESHOLD = 0.90
SAMPLE_SIZE_WARNING = 20
TREND_WORSENING_THRESHOLD = 0.02

# Calibration interpretation thresholds (logit-scale Platt scaling).
# On the logit scale, deviations from 1.0/0.0 are larger than on the
# probability scale. These are initial values; validate against real data.
CAL_SLOPE_OVERCONFIDENT = 0.7
CAL_SLOPE_UNDERCONFIDENT = 1.3
CAL_INTERCEPT_NOTABLE = 0.3


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
    if not path.exists():
        return []
    return load_jsonl(path)


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
    acc_str = (
        f"{o['directional_accuracy']:.0%}"
        if o.get("directional_accuracy") is not None
        else "N/A"
    )
    no_sig = o.get("no_signal_rate")
    no_sig_str = f"{no_sig:.0%}" if no_sig is not None else "N/A"
    ll = o.get("log_loss")
    ll_str = f"{ll:.4f}" if ll is not None else "N/A"
    sharp_str = f"{o['sharpness']:.4f}" if o.get("sharpness") is not None else "N/A"
    bss = o.get("brier_skill_score")
    bss_str = f"{bss:+.4f}" if bss is not None else "N/A"
    baseline_str = str(o.get("baseline_brier", "N/A"))
    # Edge over market
    edge = o.get("edge")
    edge_n = o.get("edge_n", 0)
    edge_str = f"{edge:+.4f}" if edge is not None else "N/A"
    epr = o.get("edge_positive_rate")
    epr_str = f"{epr:.0%}" if epr is not None else "N/A"

    lines = [
        "## Overall",
        "",
        f"- Predictions scored: {scores['valid_rows']} / {scores['total_rows']}"
        f" ({rel_str} reliability)",
        f"- Overall Brier: {brier_str}",
        f"- Baseline Brier: {baseline_str}"
        " (naive predictor using observed base rate)",
        f"- Brier Skill Score: {bss_str}",
        "  - BSS > 0 = better than base rate, BSS = 0 = no skill,"
        " BSS < 0 = worse than base rate",
        f"- Edge over market: {edge_str} (n={edge_n})",
        "  - Positive = tool beats market consensus, negative = market wins",
        f"  - Edge positive rate: {epr_str}"
        " (fraction of questions where tool beat market)",
        f"- Directional Accuracy: {acc_str}"
        f" (n_directional={o.get('n_directional', 0)})",
        f"- No-signal rate: {no_sig_str}"
        f" ({o.get('no_signal_count', 0)} predictions at exactly 0.5)",
        f"- Log Loss: {ll_str}",
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
        acc = (
            f"{stats['directional_accuracy']:.0%}"
            if stats.get("directional_accuracy") is not None
            else "N/A"
        )
        sharp = (
            f"{stats['sharpness']:.4f}" if stats.get("sharpness") is not None else "N/A"
        )
        bss = stats.get("brier_skill_score")
        bss_str = f", BSS: {bss:+.4f}" if bss is not None else ""
        ll = stats.get("log_loss")
        ll_str = f", LogLoss: {ll:.4f}" if ll is not None else ""
        edge = stats.get("edge")
        edge_n = stats.get("edge_n", 0)
        edge_str = f", Edge: {edge:+.4f} (n={edge_n})" if edge is not None else ""
        lines.append(
            f"{i}. **{tool}** — Brier: {brier}{bss_str}{ll_str}{edge_str},"
            f" DirAcc: {acc}, Sharp: {sharp} (n={stats['n']}){flags}"
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
        baseline = stats.get("baseline_brier")
        bss = stats.get("brier_skill_score")
        yes_rate = stats.get("outcome_yes_rate")
        edge = stats.get("edge")
        edge_n = stats.get("edge_n", 0)
        parts = [f"Brier: {stats['brier']}"]
        if baseline is not None:
            parts.append(f"baseline: {baseline}")
        if bss is not None:
            parts.append(f"BSS: {bss:+.4f}")
        if edge is not None:
            parts.append(f"edge: {edge:+.4f} (n={edge_n})")
        if yes_rate is not None:
            parts.append(f"yes rate: {yes_rate:.0%}")
        parts.append(f"n={stats['n']}")
        lines.append(f"- **{platform}**: {', '.join(parts)}")
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
            bss = stats.get("brier_skill_score")
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
            elif bss is not None and bss < BSS_HARMFUL_THRESHOLD:
                baseline = stats.get("baseline_brier", "?")
                if stats.get("decision_worthy", True):
                    found = True
                    lines.append(
                        f"- **{name}** ({section_name}): BSS {bss:+.4f}"
                        f" — worse than base-rate predictor"
                        f" (Brier {brier} vs baseline {baseline},"
                        f" n={stats['n']})"
                    )

    if not found:
        lines.append("No weak spots detected.")

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
        "| Tool | Platform | Brier | BSS | LogLoss | Edge | Edge n | DirAcc | Sharpness | n |",
        "|------|----------|-------|-----|---------|------|--------|--------|-----------|---|",
    ]
    for key, stats in sorted(
        data.items(),
        key=lambda x: x[1].get("brier") if x[1].get("brier") is not None else 999,
    ):
        parts = key.split(" | ")
        tool = parts[0] if parts else key
        platform = parts[1] if len(parts) > 1 else "?"
        brier = f"{stats['brier']:.4f}" if stats.get("brier") is not None else "N/A"
        bss = stats.get("brier_skill_score")
        bss_str = f"{bss:+.4f}" if bss is not None else "N/A"
        ll = stats.get("log_loss")
        ll_str2 = f"{ll:.4f}" if ll is not None else "N/A"
        edge = stats.get("edge")
        edge_str = f"{edge:+.4f}" if edge is not None else "N/A"
        edge_n = stats.get("edge_n", 0)
        acc = (
            f"{stats['directional_accuracy']:.0%}"
            if stats.get("directional_accuracy") is not None
            else "N/A"
        )
        sharp = (
            f"{stats['sharpness']:.4f}" if stats.get("sharpness") is not None else "N/A"
        )
        label = _sample_label(stats)
        lines.append(
            f"| {tool} | {platform} | {brier} | {bss_str} | {ll_str2} | {edge_str}"
            f" | {edge_n} | {acc} | {sharp} | {stats['n']}{label} |"
        )

    return "\n".join(lines)


_EDGE_SECTION_HEADER = "## Edge Over Market (System Diagnostic)"


def section_edge_analysis(scores: dict[str, Any]) -> str:
    """Edge-over-market analysis — per platform, difficulty, and liquidity."""
    elig = scores.get("edge_eligibility", {})
    n_eligible = elig.get("n_eligible", 0)
    n_total = elig.get("n_total", 0)
    if n_eligible == 0:
        return (
            f"{_EDGE_SECTION_HEADER}\n\n"
            "No edge-eligible rows (need market_prob_at_prediction)."
        )

    pct = f" ({n_eligible / n_total:.1%} of total)" if n_total > 0 else ""
    lines = [
        _EDGE_SECTION_HEADER,
        "",
        "Edge measures whether prediction accuracy translates to trading"
        " value — it is not a tool ranking metric.",
        "",
        f"Edge-eligible rows: {n_eligible} / {n_total}{pct}",
        "",
    ]

    # Per-platform edge
    by_plat = scores.get("by_platform", {})
    if by_plat:
        lines.append("### By Platform")
        lines.append("")
        for plat, stats in sorted(by_plat.items()):
            edge = stats.get("edge")
            edge_n = stats.get("edge_n", 0)
            epr = stats.get("edge_positive_rate")
            if edge is not None:
                lines.append(
                    f"- **{plat}**: edge {edge:+.4f},"
                    f" positive rate {epr:.0%}, edge_n={edge_n}"
                )
        lines.append("")

    # Platform × difficulty
    pd = scores.get("by_platform_difficulty", {})
    pd_filtered = {
        k: v for k, v in pd.items() if v.get("edge") is not None and "unknown" not in k
    }
    if pd_filtered:
        lines.append("### By Platform × Difficulty")
        lines.append("")
        lines.append(
            "| Platform | Difficulty | Edge | Edge +rate | Edge n | Brier | n |"
        )
        lines.append(
            "|----------|-----------|------|------------|--------|-------|---|"
        )
        for key, stats in sorted(pd_filtered.items()):
            parts = key.split(" | ")
            plat, diff = parts[0], parts[1] if len(parts) > 1 else "?"
            edge = stats["edge"]
            epr = stats.get("edge_positive_rate", 0)
            brier = stats.get("brier")
            brier_str = f"{brier:.4f}" if brier is not None else "N/A"
            lines.append(
                f"| {plat} | {diff} | {edge:+.4f} | {epr:.0%}"
                f" | {stats.get('edge_n', 0)} | {brier_str} | {stats['n']} |"
            )
        lines.append("")

    # Platform × liquidity
    pl = scores.get("by_platform_liquidity", {})
    pl_filtered = {
        k: v for k, v in pl.items() if v.get("edge") is not None and "unknown" not in k
    }
    if pl_filtered:
        lines.append("### By Platform × Liquidity")
        lines.append("")
        lines.append(
            "| Platform | Liquidity | Edge | Edge +rate | Edge n | Brier | n |"
        )
        lines.append(
            "|----------|-----------|------|------------|--------|-------|---|"
        )
        for key, stats in sorted(pl_filtered.items()):
            parts = key.split(" | ")
            plat, liq = parts[0], parts[1] if len(parts) > 1 else "?"
            edge = stats["edge"]
            epr = stats.get("edge_positive_rate", 0)
            brier = stats.get("brier")
            brier_str = f"{brier:.4f}" if brier is not None else "N/A"
            lines.append(
                f"| {plat} | {liq} | {edge:+.4f} | {epr:.0%}"
                f" | {stats.get('edge_n', 0)} | {brier_str} | {stats['n']} |"
            )

    return "\n".join(lines)


_DIAG_SECTION_HEADER = "## Diagnostic Edge Metrics"


def _bias_label(bias: float) -> str:
    """Return a human-readable label for a directional bias value.

    :param bias: directional bias value.
    :return: label string.
    """
    if bias > 0:
        return "overestimates"
    if bias < 0:
        return "underestimates"
    return "no bias"


def _render_conditional_accuracy(
    overall: dict[str, Any], scores: dict[str, Any], lines: list[str]
) -> None:
    """Render conditional accuracy subsection.

    :param overall: overall stats dict.
    :param scores: full scores dict (for by_platform).
    :param lines: output list to append to.
    """
    disagree_n = overall.get("disagree_n", 0)
    lines.append("### Conditional Accuracy When Disagreeing")
    lines.append("")
    lines.append(
        "When the tool disagrees with the market enough to trigger a trade"
        f" (|p_yes - market_prob| > {DISAGREE_THRESHOLD}),"
        " is the tool or market closer to truth?"
    )
    lines.append("")

    ca = overall.get("conditional_accuracy_rate")
    if ca is not None:
        lines.append(
            f"- **Overall**: {ca:.0%} tool-wins (n={disagree_n} disagreements)"
        )
    else:
        lines.append(
            f"- **Overall**: insufficient data (n={disagree_n} disagreements,"
            f" need {MIN_SAMPLE_SIZE})"
        )

    for plat, stats in sorted(scores.get("by_platform", {}).items()):
        p_ca = stats.get("conditional_accuracy_rate")
        p_dn = stats.get("disagree_n", 0)
        if p_ca is not None:
            lines.append(f"- **{plat}**: {p_ca:.0%} tool-wins (n={p_dn})")
        elif p_dn > 0:
            lines.append(
                f"- **{plat}**: insufficient data (n={p_dn},"
                f" need {MIN_SAMPLE_SIZE})"
            )
    lines.append("")


def _render_disagreement_brier(overall: dict[str, Any], lines: list[str]) -> None:
    """Render disagreement-stratified Brier subsection.

    :param overall: overall stats dict.
    :param lines: output list to append to.
    """
    lines.append("### Disagreement-Stratified Brier")
    lines.append("")
    lines.append(
        "Brier score bucketed by how much the tool disagrees with the market."
        " Worse accuracy on large_trade = losing money where it matters."
    )
    lines.append("")

    bucket_labels = [
        ("no_trade", f"No trade (|d| \u2264 {DISAGREE_THRESHOLD})"),
        (
            "small_trade",
            f"Small trade ({DISAGREE_THRESHOLD} < |d| \u2264 {LARGE_TRADE_THRESHOLD})",
        ),
        ("large_trade", f"Large trade (|d| > {LARGE_TRADE_THRESHOLD})"),
    ]
    for bucket_key, label in bucket_labels:
        b = overall.get(f"brier_{bucket_key}")
        n = overall.get(f"n_{bucket_key}", 0)
        if b is not None:
            lines.append(f"- **{label}**: Brier {b:.4f} (n={n})")
        else:
            lines.append(f"- **{label}**: insufficient data (n={n})")
    lines.append("")


def _render_directional_bias(
    overall: dict[str, Any], scores: dict[str, Any], lines: list[str]
) -> None:
    """Render directional bias subsection.

    :param overall: overall stats dict.
    :param scores: full scores dict (for by_category).
    :param lines: output list to append to.
    """
    lines.append("### Directional Bias (When Tool Loses)")
    lines.append("")
    lines.append(
        "When the tool disagrees and the market was closer to truth,"
        " does the tool tend to overestimate (positive) or underestimate"
        " (negative)?"
    )
    lines.append("")

    bias = overall.get("directional_bias")
    n_losses = overall.get("n_bias_losses", 0)
    if bias is not None:
        lines.append(
            f"- **Overall**: {bias:+.4f} ({_bias_label(bias)}," f" n={n_losses} losses)"
        )
    else:
        lines.append(
            f"- **Overall**: insufficient data (n={n_losses} losses,"
            f" need {MIN_SAMPLE_SIZE})"
        )

    for cat, stats in sorted(scores.get("by_category", {}).items()):
        c_bias = stats.get("directional_bias")
        c_n = stats.get("n_bias_losses", 0)
        if c_bias is not None:
            lines.append(f"- **{cat}**: {c_bias:+.4f} ({_bias_label(c_bias)}, n={c_n})")
        elif c_n > 0:
            lines.append(
                f"- **{cat}**: insufficient data" f" (n={c_n}, need {MIN_SAMPLE_SIZE})"
            )
        else:
            lines.append(f"- **{cat}**: no losses to measure")


def section_diagnostic_metrics(scores: dict[str, Any]) -> str:
    """Conditional accuracy, disagreement-stratified Brier, and directional bias."""
    overall = scores.get("overall", {})
    disagree_n = overall.get("disagree_n", 0)
    if disagree_n == 0 and overall.get("edge_n", 0) == 0:
        return (
            f"{_DIAG_SECTION_HEADER}\n\n"
            "No edge-eligible rows — diagnostic metrics require "
            "market_prob_at_prediction."
        )

    lines = [
        _DIAG_SECTION_HEADER,
        "",
        "These metrics diagnose whether accuracy translates to profit and"
        " where the system loses. They are not used for tool ranking.",
        "",
    ]

    _render_conditional_accuracy(overall, scores, lines)
    _render_disagreement_brier(overall, lines)
    _render_directional_bias(overall, scores, lines)

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

    # ECE and calibration regression
    ece = scores.get("ece")
    if ece is not None:
        lines.append("")
        lines.append(f"**ECE (Expected Calibration Error):** {ece:.4f}")
        lines.append("  - 0 = perfectly calibrated, higher = worse")
    cal_int = scores.get("calibration_intercept")
    cal_slope = scores.get("calibration_slope")
    if cal_int is not None and cal_slope is not None:
        lines.append(
            f"**Calibration regression:** intercept={cal_int:+.4f},"
            f" slope={cal_slope:.4f}"
        )
        if cal_slope < CAL_SLOPE_OVERCONFIDENT:
            lines.append(
                "  - Slope < 1.0 → predictions too extreme"
                " (overpredicts high-confidence, underpredicts low-confidence)"
            )
        elif cal_slope > CAL_SLOPE_UNDERCONFIDENT:
            lines.append("  - Slope > 1.0 → predictions too compressed toward 0.5")
        # Intercept is evaluated at logit(p_yes)=0, i.e. p_yes=0.5.
        # Only interpret when slope is not too far from 1.0; with extreme
        # slopes the intercept alone is ambiguous.
        if abs(cal_slope - 1.0) < 0.4 and abs(cal_int) > CAL_INTERCEPT_NOTABLE:
            if cal_int > 0:
                lines.append(
                    "  - Positive intercept at p=0.5 midpoint"
                    " (tool underpredicts at the 50% probability point)"
                )
            else:
                lines.append(
                    "  - Negative intercept at p=0.5 midpoint"
                    " (tool overpredicts at the 50% probability point)"
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
                "**High-confidence bins overpredict** — predicted high yes-probability"
                " but realized rate is much lower."
            )
        elif avg_gap < -0.1:
            lines.append(
                "**High-confidence bins underpredict** — realized rate exceeds predictions."
            )
    if low_conf:
        avg_gap = sum(b["gap"] for b in low_conf) / len(low_conf)
        if avg_gap < -0.1:
            lines.append(
                "**Low-confidence bins underpredict** — predicted low yes-probability"
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


def section_base_rates(scores: dict[str, Any]) -> str:
    """Generate the base rate summary section per platform."""
    platforms = scores.get("by_platform", {})
    if not platforms:
        return "## Base Rates\n\nNo platform data available."

    header = "| Platform | Yes Rate | No Rate | Baseline Brier | Always-No Brier |"
    separator = "|----------|----------|---------|----------------|-----------------|"
    lines = [
        "## Base Rates",
        "",
        header,
        separator,
    ]
    for platform, stats in sorted(platforms.items()):
        yes_rate = stats.get("outcome_yes_rate")
        if yes_rate is None:
            continue
        no_rate = 1 - yes_rate
        baseline = stats.get("baseline_brier", 0)
        always_no_brier = round(yes_rate, 4)
        lines.append(
            f"| {platform} | {yes_rate:.0%} | {no_rate:.0%}"
            f" | {baseline} | {always_no_brier} |"
        )

    lines.extend(
        [
            "",
            "- **Baseline Brier**: Brier score of a predictor that always"
            " outputs the observed base rate (best no-skill strategy)",
            "- **Always-No Brier**: Brier score of a predictor that always"
            " outputs p_yes=0.0",
            "- Tools should score below the baseline to demonstrate"
            " predictive skill (BSS > 0)",
        ]
    )

    return "\n".join(lines)


def _delta_str(period_val: float | None, alltime_val: float | None) -> str:
    """Format a delta vs all-time with arrow."""
    if period_val is None or alltime_val is None:
        return ""
    delta = period_val - alltime_val
    arrow = "better" if delta < 0 else "worse" if delta > 0 else "same"
    return f" (delta vs all-time: {delta:+.4f} {arrow})"


def section_period(
    period_scores: dict[str, Any] | None,
    alltime_scores: dict[str, Any],
    label: str = "Since last report",
) -> str:
    """Generate a period comparison section (since-last-report or rolling 7d).

    :param period_scores: scores from the recent period.
    :param alltime_scores: all-time scores for delta comparison.
    :param label: section title.
    :return: markdown section string.
    """
    if period_scores is None:
        return f"## {label}\n\nNo period data available."

    po = period_scores.get("overall", {})
    ao = alltime_scores.get("overall", {})
    n = po.get("n", 0)
    valid_n = po.get("valid_n", 0)

    if n == 0:
        return f"## {label}\n\nNo new predictions since last report."

    brier = po.get("brier")
    brier_str = f"{brier:.4f}" if brier is not None else "N/A"
    ll = po.get("log_loss")
    ll_str = f"{ll:.4f}" if ll is not None else "N/A"

    lines = [
        f"## {label} (n={valid_n})",
        "",
        f"- Brier: {brier_str}{_delta_str(brier, ao.get('brier'))}",
        f"- Log Loss: {ll_str}{_delta_str(ll, ao.get('log_loss'))}",
    ]

    # Per-tool breakdown
    by_tool = period_scores.get("by_tool", {})
    if by_tool:
        at_tools = alltime_scores.get("by_tool", {})
        for tool, stats in sorted(
            by_tool.items(),
            key=lambda x: x[1].get("brier") if x[1].get("brier") is not None else 999,
        ):
            tb = stats.get("brier")
            if tb is None:
                continue
            at_b = at_tools.get(tool, {}).get("brier")
            lines.append(
                f"  - **{tool}**: {tb:.4f}"
                f"{_delta_str(tb, at_b)}"
                f" (n={stats['n']})"
            )

    return "\n".join(lines)


VERSION_DELTA_LOW_SAMPLE = 30


def _parse_tvm_key(key: str) -> tuple[str, str, str]:
    """Split a 'tool | version | mode' key. Pads if mode is missing (legacy)."""
    parts = [p.strip() for p in key.split("|")]
    while len(parts) < 3:
        parts.append("unknown")
    return parts[0], parts[1], parts[2]


def section_tool_version_breakdown(
    scores: dict[str, Any],
    title: str = "Tool × Version × Mode",
) -> str:
    """Per (tool, version, mode) metrics table — combines prod and tournament."""
    tvm = scores.get("by_tool_version_mode", {})
    if not tvm:
        return ""

    rows = sorted(
        (_parse_tvm_key(k) + (v,) for k, v in tvm.items()),
        key=lambda r: (r[0], r[1], r[2]),
    )

    lines = [
        f"## {title}",
        "",
        "| Tool | Version | Mode | n | valid | Brier | DirAcc | BSS |",
        "|------|---------|------|---:|---:|---:|---:|---:|",
    ]
    has_low_sample = False
    for tool, version, mode, stats in rows:
        n = stats.get("n", 0)
        valid_n = stats.get("valid_n", 0)
        low = n < VERSION_DELTA_LOW_SAMPLE
        if low:
            has_low_sample = True
        n_cell = f"{n} ⚠" if low else str(n)
        brier = stats.get("brier")
        brier_s = f"{brier:.4f}" if brier is not None else "—"
        acc = stats.get("directional_accuracy")
        acc_s = f"{acc:.0%}" if acc is not None else "—"
        bss = stats.get("brier_skill_score")
        bss_s = f"{bss:+.4f}" if bss is not None else "—"
        lines.append(
            f"| {tool} | `{version}` | {mode} | {n_cell} | {valid_n} | {brier_s} | {acc_s} | {bss_s} |"
        )

    if has_low_sample:
        lines.extend(
            [
                "",
                f"⚠ Rows marked with ⚠ have n < {VERSION_DELTA_LOW_SAMPLE}; metrics from these cells are statistically unreliable and superlatives should not be drawn from them.",
            ]
        )
    return "\n".join(lines)


def section_version_deltas(scores: dict[str, Any]) -> str:
    """Pairwise Brier deltas across versions/modes for each tool with multiple cells."""
    tvm = scores.get("by_tool_version_mode", {})
    if not tvm:
        return ""

    by_tool: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for key, stats in tvm.items():
        tool, version, mode = _parse_tvm_key(key)
        by_tool.setdefault(tool, []).append((version, mode, stats))

    multi = {t: cells for t, cells in by_tool.items() if len(cells) >= 2}
    if not multi:
        return ""

    lines = ["## Version Deltas", ""]
    for tool in sorted(multi):
        cells = sorted(multi[tool], key=lambda c: (c[0], c[1]))
        lines.append(f"### {tool}")
        lines.append("")
        lines.append(
            "| Baseline (version, mode) | Candidate (version, mode) | Brier Δ | Direction | n_b | n_c |"
        )
        lines.append("|---|---|---:|---|---:|---:|")
        for i, (v_b, m_b, s_b) in enumerate(cells):
            for v_c, m_c, s_c in cells[i + 1 :]:
                cmp = compare_stats(s_b, s_c)
                brier_cmp = cmp.get("brier", {})
                delta = brier_cmp.get("delta")
                direction = brier_cmp.get("direction") or "—"
                delta_s = f"{delta:+.4f}" if isinstance(delta, (int, float)) else "—"
                low = (
                    " ⚠"
                    if min(s_b.get("n", 0), s_c.get("n", 0)) < VERSION_DELTA_LOW_SAMPLE
                    else ""
                )
                lines.append(
                    f"| `{v_b}` / {m_b} | `{v_c}` / {m_c} | {delta_s}{low} | {direction} | {s_b.get('n', 0)} | {s_c.get('n', 0)} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


CALLOUT_DELTA = 0.03
CALLOUT_MIN_N = 30


def _relabel_heading(section_md: str, suffix: str) -> str:
    """Append *suffix* to the first ``## Heading`` line in *section_md*.

    Used to dup-render production sections with a "— Tournament" label
    without duplicating every section function.

    :param section_md: rendered markdown for a single section.
    :param suffix: string appended to the first ``## Heading`` line.
    :return: the section markdown with the heading relabelled.
    """
    lines = section_md.split("\n")
    if lines and lines[0].startswith("## "):
        lines[0] = lines[0] + suffix
    return "\n".join(lines)


def _has_tournament_data(scores_tournament: dict[str, Any] | None) -> bool:
    """Return True when a tournament scores dict has any rows to report."""
    return bool(scores_tournament and scores_tournament.get("total_rows", 0) > 0)


def _merged_tvm_scores(
    scores_prod: dict[str, Any],
    scores_tournament: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a scores-shaped dict whose ``by_tool_version_mode`` holds both modes.

    Keys are already of the form ``tool | version | mode`` so there is no
    risk of collision between the two files.

    :param scores_prod: production scores dict.
    :param scores_tournament: tournament scores dict, or None.
    :return: minimal scores dict with merged ``by_tool_version_mode``.
    """
    merged: dict[str, Any] = {"by_tool_version_mode": {}}
    merged["by_tool_version_mode"].update(scores_prod.get("by_tool_version_mode", {}))
    if scores_tournament:
        merged["by_tool_version_mode"].update(
            scores_tournament.get("by_tool_version_mode", {})
        )
    return merged


def section_tournament_callouts(
    scores_prod: dict[str, Any],
    scores_tournament: dict[str, Any] | None,
) -> str:
    """Flag tool versions whose tournament Brier diverges from prod.

    A row qualifies when tournament sample size is at least
    ``CALLOUT_MIN_N`` and the absolute Brier delta exceeds
    ``CALLOUT_DELTA``. Negative deltas become promotion candidates
    (tournament better); positive deltas become tournament regressions
    (tournament worse — a warning before the version reaches production).

    :param scores_prod: production scores dict (provides tool-level baselines).
    :param scores_tournament: tournament scores dict (candidate cells), or None.
    :return: markdown section, or empty string when no callouts qualify.
    """
    if not _has_tournament_data(scores_tournament):
        return ""

    prod_by_tool = scores_prod.get("by_tool", {}) if scores_prod else {}
    assert scores_tournament is not None  # narrowed by _has_tournament_data
    tournament_tvm = scores_tournament.get("by_tool_version_mode", {})

    promotions: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    regressions: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []

    for key, t_stats in tournament_tvm.items():
        tool, version, mode = _parse_tvm_key(key)
        if mode != "tournament":
            continue
        if t_stats.get("n", 0) < CALLOUT_MIN_N:
            continue
        t_brier = t_stats.get("brier")
        if t_brier is None:
            continue

        p_stats = prod_by_tool.get(tool)
        if not p_stats:
            continue
        p_brier = p_stats.get("brier")
        if p_brier is None:
            continue

        delta = t_brier - p_brier
        if delta <= -CALLOUT_DELTA:
            promotions.append((tool, version, t_stats, p_stats))
        elif delta >= CALLOUT_DELTA:
            regressions.append((tool, version, t_stats, p_stats))

    if not promotions and not regressions:
        return ""

    lines = ["## Tournament Callouts", ""]
    if promotions:
        lines.append("**Promotion candidates:**")
        lines.append("")
        for tool, version, t_stats, p_stats in promotions:
            delta = t_stats["brier"] - p_stats["brier"]
            lines.append(
                f"- `{tool}` version `{version}` — tournament Brier"
                f" {t_stats['brier']:.4f} (n={t_stats['n']}) vs production Brier"
                f" {p_stats['brier']:.4f} (n={p_stats['n']}). Δ {delta:+.4f}."
            )
        lines.append("")
    if regressions:
        lines.append("**Tournament regressions:**")
        lines.append("")
        for tool, version, t_stats, p_stats in regressions:
            delta = t_stats["brier"] - p_stats["brier"]
            lines.append(
                f"- `{tool}` version `{version}` — tournament Brier"
                f" {t_stats['brier']:.4f} (n={t_stats['n']}) vs production Brier"
                f" {p_stats['brier']:.4f} (n={p_stats['n']}). Δ {delta:+.4f}."
            )
    return "\n".join(lines).rstrip()


def generate_report(
    scores: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    period_scores: dict[str, Any] | None = None,
    rolling_scores: dict[str, Any] | None = None,
    include_tournament: bool = False,
    scores_tournament: dict[str, Any] | None = None,
    period_scores_tournament: dict[str, Any] | None = None,
    rolling_scores_tournament: dict[str, Any] | None = None,
) -> str:
    """Generate a full benchmark report from scores and history.

    Production-mode sections are rendered from ``scores`` /
    ``period_scores`` / ``rolling_scores``. When tournament scores are
    supplied and contain rows, a duplicate set of the mode-sensitive
    sections (Since Last Report, Last 7 Days Rolling, Overall, Tool
    Ranking) is rendered with a ``— Tournament`` suffix. The Tool ×
    Version × Mode breakdown merges both modes. A final "Tournament
    Callouts" section flags tool versions whose tournament Brier
    diverges materially from the same tool's production baseline.

    :param scores: parsed production ``scores.json`` dict (all-time).
    :param history: list of monthly snapshots from ``scores_history.jsonl``.
    :param period_scores: production scores since last report.
    :param rolling_scores: production scores from the last 7 days.
    :param include_tournament: master switch for rendering the Tool ×
        Version × Mode breakdown. When False, tournament inputs are
        ignored entirely.
    :param scores_tournament: parsed ``scores_tournament.json`` dict.
    :param period_scores_tournament: tournament since last report.
    :param rolling_scores_tournament: tournament last 7 days.
    :return: full markdown report string.
    """
    if history is None:
        history = []

    date = scores.get("generated_at", "")[:10] or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d"
    )

    render_tournament = include_tournament and _has_tournament_data(scores_tournament)
    # Local non-optional alias for mypy once _has_tournament_data has narrowed.
    tournament_scores: dict[str, Any] = scores_tournament or {}

    sections: list[str] = [f"# Benchmark Report — {date}"]

    # Since Last Report
    sections.append(section_period(period_scores, scores, "Since Last Report"))
    if render_tournament and period_scores_tournament is not None:
        sections.append(
            _relabel_heading(
                section_period(
                    period_scores_tournament,
                    tournament_scores,
                    "Since Last Report",
                ),
                " — Tournament",
            )
        )

    # Last 7 Days Rolling
    sections.append(section_period(rolling_scores, scores, "Last 7 Days Rolling"))
    if render_tournament and rolling_scores_tournament is not None:
        sections.append(
            _relabel_heading(
                section_period(
                    rolling_scores_tournament,
                    tournament_scores,
                    "Last 7 Days Rolling",
                ),
                " — Tournament",
            )
        )

    # Overall
    sections.append(section_overall(scores))
    if render_tournament:
        sections.append(
            _relabel_heading(section_overall(tournament_scores), " — Tournament")
        )

    # Base Rates — production only
    sections.append(section_base_rates(scores))

    # Tool Ranking
    sections.append(section_tool_ranking(scores))
    if render_tournament:
        sections.append(
            _relabel_heading(section_tool_ranking(tournament_scores), " — Tournament")
        )

    # Tool × Version × Mode — merged
    if include_tournament:
        merged = _merged_tvm_scores(scores, scores_tournament)
        tvm_section = section_tool_version_breakdown(
            merged, "Tool × Version × Mode (All-Time)"
        )
        if tvm_section:
            sections.append(tvm_section)
        if rolling_scores is not None:
            merged_rolling = _merged_tvm_scores(
                rolling_scores, rolling_scores_tournament
            )
            tvm_rolling = section_tool_version_breakdown(
                merged_rolling, "Tool × Version × Mode (Last 7 Days)"
            )
            if tvm_rolling:
                sections.append(tvm_rolling)

    # Remaining production-only sections
    sections.extend(
        [
            section_platform(scores),
            section_tool_platform(scores),
            section_edge_analysis(scores),
            section_diagnostic_metrics(scores),
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
    )

    # Tournament Callouts — cross-mode
    if render_tournament:
        callouts = section_tournament_callouts(scores, scores_tournament)
        if callouts:
            sections.append(callouts)

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
    parser.add_argument(
        "--period",
        type=Path,
        default=None,
        help="Period scores JSON (since last report, production)",
    )
    parser.add_argument(
        "--rolling",
        type=Path,
        default=None,
        help="Rolling 7-day scores JSON (production)",
    )
    parser.add_argument(
        "--scores-tournament",
        type=Path,
        default=None,
        help="Tournament scores JSON (all-time)",
    )
    parser.add_argument(
        "--period-tournament",
        type=Path,
        default=None,
        help="Tournament period scores JSON (since last report)",
    )
    parser.add_argument(
        "--rolling-tournament",
        type=Path,
        default=None,
        help="Tournament rolling 7-day scores JSON",
    )
    parser.add_argument(
        "--include-tournament",
        action="store_true",
        help=(
            "Render tournament-mode sections (duplicated per-mode blocks, "
            "merged Tool × Version × Mode, and Tournament Callouts)"
        ),
    )
    args = parser.parse_args()

    def _maybe_load(path: Path | None) -> dict[str, Any] | None:
        return load_scores(path) if path and path.exists() else None

    scores = load_scores(args.scores)
    history = load_history(args.history)
    period = _maybe_load(args.period)
    rolling = _maybe_load(args.rolling)
    scores_tournament = _maybe_load(args.scores_tournament)
    period_tournament = _maybe_load(args.period_tournament)
    rolling_tournament = _maybe_load(args.rolling_tournament)

    print(
        f"Loaded scores ({scores.get('total_rows', 0)} rows), "
        f"{len(history)} months of history"
    )

    report = generate_report(
        scores,
        history,
        period_scores=period,
        rolling_scores=rolling,
        include_tournament=args.include_tournament,
        scores_tournament=scores_tournament,
        period_scores_tournament=period_tournament,
        rolling_scores_tournament=rolling_tournament,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to {args.output}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
