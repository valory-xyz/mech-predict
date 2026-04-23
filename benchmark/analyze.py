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
from types import MappingProxyType
from typing import Any, Mapping

from benchmark import release_map
from benchmark.io import load_jsonl
from benchmark.scorer import (
    DISAGREE_THRESHOLD,
    LARGE_TRADE_THRESHOLD,
    MIN_SAMPLE_SIZE,
    brier_sort_key,
)
from benchmark.tool_usage import (
    failed_deployments,
    fetch_disabled_tools,
    iter_tools_with_disabled,
)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_HISTORY = DEFAULT_RESULTS_DIR / "scores_history.jsonl"

# Maps the scorer's platform keys to the deployment names used in report
# headers and Slack summaries. Exposed read-only so downstream modules
# (notify_slack) can reuse the mapping without drifting.
PLATFORM_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "omen": "Omenstrat",
        "polymarket": "Polystrat",
    }
)

ROLLING_WINDOW_DAYS = 3

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

# Categories currently emitted by the two upstream platforms.
# Keep these in sync with:
#   Omen:       valory-xyz/market-creator — DEFAULT_TOPICS in
#               packages/valory/skills/market_creation_manager_abci/propose_questions.py
#   Polymarket: valory-xyz/trader — POLYMARKET_CATEGORY_TAGS in
#               packages/valory/connections/polymarket_client/connection.py
# Historical labels not in either set (e.g. "travel", "crypto", "tech") are
# treated as legacy and skipped by weak-spot reporting.
OMEN_CATEGORIES: frozenset[str] = frozenset(
    {
        "business",
        "cryptocurrency",
        "politics",
        "science",
        "technology",
        "trending",
        "social",
        "health",
        "sustainability",
        "internet",
        "food",
        "pets",
        "animals",
        "curiosities",
        "economy",
        "arts",
        "entertainment",
        "weather",
        "sports",
        "finance",
        "international",
    }
)
POLYMARKET_ACTIVE_CATEGORIES: frozenset[str] = frozenset(
    {
        "business",
        "politics",
        "science",
        "technology",
        "health",
        "entertainment",
        "weather",
        "finance",
        "international",
    }
)
ACTIVE_CATEGORIES: frozenset[str] = OMEN_CATEGORIES | POLYMARKET_ACTIVE_CATEGORIES


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


def section_metric_reference() -> str:
    """Render the metric-reference legend shown at the top of every report.

    :return: markdown section string.
    """
    lines = [
        "## Metric References",
        "",
        (
            "All point-in-time metrics below are scoped to the rolling window "
            f"(last {ROLLING_WINDOW_DAYS} days) unless a section explicitly says "
            "otherwise."
        ),
        "",
        f"- **Brier** — ideal 0.00, coin-flip {BRIER_RANDOM}; lower is better.",
        (
            "- **Log Loss** — ideal 0.00; lower is better, punishes confident "
            "errors harder."
        ),
        (
            "- **BSS (Brier Skill Score)** — ideal > 0; negative means worse than "
            "the base-rate predictor."
        ),
        (
            "- **Edge over market** — ideal > 0; positive = tool beats market "
            "consensus. System diagnostic, not a tool ranking signal."
        ),
    ]
    return "\n".join(lines)


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
    no_sig_str = f"{no_sig:.2%}" if no_sig is not None else "N/A"
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
    """Return a sample-size label for ranking context.

    Splits the previously single "⚠ low sample" label into two cases so
    tools with many rows but zero valid parses are not mistaken for a
    small-sample problem:

    - ``n >= MIN_SAMPLE_SIZE`` and ``valid_n == 0``: all rows malformed,
      a pipeline issue, not a sample-size issue.
    - ``decision_worthy is False``: too few valid rows to be decision
      worthy (the scorer's own sample-size gate).

    ``decision_worthy`` is the primary signal because the scorer already
    computes and ships it on every group. Using ``stats.get(...) is
    False`` preserves the "missing key = empty label" safety property:
    a caller that passes a partial dict gets no warning rather than a
    spurious one.

    :param stats: group stats dict; ``n``, ``valid_n`` and
        ``decision_worthy`` are all consulted when present.
    :return: a leading-space string label, or empty when stats pass
        the sample-size gate or do not expose the required keys.
    """
    if stats.get("n", 0) >= MIN_SAMPLE_SIZE and stats.get("valid_n", None) == 0:
        return " ⚠ all malformed"
    if stats.get("decision_worthy") is False:
        return " ⚠ low sample"
    return ""


def section_tool_deployment_status(
    scores: dict[str, Any],
    disabled: dict[str, list[str] | None] | None = None,
) -> str:
    """Render which benchmarked tools are disabled on each deployment.

    Tools are listed only when they appear in ``scores.by_tool`` (i.e. were
    actually invoked) AND are on at least one deployment's
    ``IRRELEVANT_TOOLS`` list.  Deployments whose config fetch failed are
    called out explicitly so a reader never confuses "unavailable" with
    "nothing disabled".

    :param scores: parsed ``scores.json`` dict.
    :param disabled: pre-fetched ``{deployment: [tool_names] | None}`` map.
        Pass ``None`` to fetch on the fly (the daily-report default).  Pass
        an empty dict to skip the section entirely (tests use this to avoid
        the live GitHub fetch).
    :return: markdown section string, or ``""`` when the caller opted out.
    """
    if disabled is None:
        disabled = fetch_disabled_tools()

    # Explicit skip contract: an empty dict means "caller opted out" (used by
    # unit tests).  Returning "" means the section heading is omitted so the
    # report doesn't advertise a section that was never computed.
    if not disabled:
        return ""

    tools = scores.get("by_tool", {})
    # Preserve report-wide ordering (Brier ascending) so readers scan this
    # section in the same order as Tool Ranking below.
    ordered = [name for name, _ in sorted(tools.items(), key=brier_sort_key)]
    entries = iter_tools_with_disabled(ordered, disabled)

    lines = ["## Tool Deployment Status", ""]
    failed = failed_deployments(disabled)
    all_failed = len(failed) == len(disabled) and len(disabled) > 0
    if failed:
        lines.append(
            "> ⚠️ Could not fetch deployment config for: "
            f"{', '.join(failed)}. Status below may be incomplete."
        )
        lines.append("")

    if not entries:
        # If every fetch failed we already told the reader — don't also claim
        # "nothing is disabled", which would be a false negative.
        if all_failed:
            return "\n".join(lines).rstrip()
        lines.append(
            "_No benchmarked tools are disabled on any deployment._"
            if not failed
            else "_No disabled tools reported from the deployments that were fetched._"
        )
        return "\n".join(lines)

    for tool, deployments in entries:
        lines.append(f"- `{tool}` — disabled on {', '.join(deployments)}")
    return "\n".join(lines)


def section_tool_ranking(scores: dict[str, Any]) -> str:
    """Generate the tool ranking section."""
    tools = scores.get("by_tool", {})
    ranked = sorted(
        tools.items(),
        key=brier_sort_key,
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
        key=brier_sort_key,
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


def section_category(scores: dict[str, Any]) -> str:
    """Fleet-level category performance — Brier/LogLoss/BSS/Edge per category."""
    categories = scores.get("by_category") or {}
    lines = ["## Category Performance", ""]
    if not categories:
        lines.append("No per-category data available.")
        return "\n".join(lines)

    for category, stats in sorted(categories.items(), key=brier_sort_key):
        if stats["n"] < MIN_SAMPLE_SIZE:
            # Include the (noisy) Brier so the ascending-Brier sort is
            # traceable to the reader — otherwise a sparse category ranking
            # above a sufficient one looks arbitrary.
            noisy_brier = stats.get("brier")
            brier_hint = (
                f" noisy Brier: {noisy_brier}" if noisy_brier is not None else ""
            )
            lines.append(
                f"- **{category}**: insufficient data"
                f" (n={stats['n']}, need {MIN_SAMPLE_SIZE}){brier_hint}"
            )
            continue
        baseline = stats.get("baseline_brier")
        bss = stats.get("brier_skill_score")
        ll = stats.get("log_loss")
        yes_rate = stats.get("outcome_yes_rate")
        edge = stats.get("edge")
        edge_n = stats.get("edge_n", 0)
        brier = stats.get("brier")
        parts = [f"Brier: {brier}" if brier is not None else "Brier: N/A"]
        if ll is not None:
            parts.append(f"LogLoss: {ll:.4f}")
        if baseline is not None:
            parts.append(f"baseline: {baseline}")
        if bss is not None:
            parts.append(f"BSS: {bss:+.4f}")
        if edge is not None:
            parts.append(f"edge: {edge:+.4f} (n={edge_n})")
        if yes_rate is not None:
            parts.append(f"yes rate: {yes_rate:.0%}")
        parts.append(f"n={stats['n']}")
        # Flag homogeneous-outcome categories: a low Brier on a one-sided
        # category reflects the base rate, not predictive skill. Mirrors
        # the base-rate guard in notify_slack.py so human readers of the
        # markdown get the same warning as the Slack LLM.
        is_homogeneous = yes_rate is not None and yes_rate in (0.0, 1.0)
        prefix = "⚠ " if is_homogeneous else ""
        tail = (
            " — one-sided outcomes; Brier not meaningful here" if is_homogeneous else ""
        )
        lines.append(f"- {prefix}**{category}**: {', '.join(parts)}{tail}")
    return "\n".join(lines)


def section_tool_category(scores: dict[str, Any]) -> str:
    """Tool × category cross breakdown — gated by MIN_SAMPLE_SIZE."""
    data = scores.get("by_tool_category") or {}
    if not data:
        return "## Tool × Category\n\nNo cross-breakdown data available."

    ranked = sorted(data.items(), key=brier_sort_key)
    sufficient = [(k, s) for k, s in ranked if s["n"] >= MIN_SAMPLE_SIZE]
    sparse = [(k, s) for k, s in ranked if s["n"] < MIN_SAMPLE_SIZE]

    lines = [
        "## Tool × Category",
        "",
        f"> Cells with n < {MIN_SAMPLE_SIZE} are moved to a separate list below"
        " the ranking. This differs from Tool × Platform, which renders every"
        " cell inline and marks small samples with ⚠.",
        "",
        "| Tool | Category | Brier | BSS | LogLoss | Edge | Edge n | DirAcc | Sharpness | n |",
        "|------|----------|-------|-----|---------|------|--------|--------|-----------|---|",
    ]
    if not sufficient:
        lines.append(f"| _(no cells with n ≥ {MIN_SAMPLE_SIZE})_ | | | | | | | | | |")
    for key, stats in sufficient:
        parts = key.split(" | ")
        tool = parts[0] if parts else key
        category = parts[1] if len(parts) > 1 else "?"
        brier = f"{stats['brier']:.4f}" if stats.get("brier") is not None else "N/A"
        bss = stats.get("brier_skill_score")
        bss_str = f"{bss:+.4f}" if bss is not None else "N/A"
        ll = stats.get("log_loss")
        ll_str = f"{ll:.4f}" if ll is not None else "N/A"
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
            f"| {tool} | {category} | {brier} | {bss_str} | {ll_str} | {edge_str}"
            f" | {edge_n} | {acc} | {sharp} | {stats['n']}{label} |"
        )

    if sparse:
        lines.append("")
        lines.append(
            f"_{len(sparse)} cell(s) below n={MIN_SAMPLE_SIZE} threshold omitted"
            " from ranking. Examples:_"
        )
        for key, stats in sparse[:5]:
            parts = key.split(" | ")
            tool = parts[0] if parts else key
            category = parts[1] if len(parts) > 1 else "?"
            lines.append(
                f"- **{tool} | {category}**: insufficient data (n={stats['n']})"
            )

    return "\n".join(lines)


def section_weak_spots(scores: dict[str, Any]) -> str:
    """Generate the weak spots section."""
    lines = ["## Weak Spots", ""]
    found = False
    skipped_legacy: list[str] = []

    for section_name, section_key in [
        ("category", "by_category"),
        ("platform", "by_platform"),
        ("tool", "by_tool"),
    ]:
        for name, stats in (scores.get(section_key) or {}).items():
            if section_key == "by_category" and name not in ACTIVE_CATEGORIES:
                skipped_legacy.append(name)
                continue
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

    if skipped_legacy:
        lines.append("")
        lines.append(
            f"_Skipped {len(skipped_legacy)} legacy category label(s) not in the"
            f" current Omen or Polymarket taxonomy: "
            f"{', '.join(sorted(set(skipped_legacy)))}._"
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
    platform: str | None = None,
) -> str:
    """Generate the trend section from monthly history + current month.

    :param history: list of monthly snapshot dicts from scores_history.jsonl.
    :param scores: current scores.json dict (appended as in-progress month).
    :param platform: when set, an annotation warns the reader that the
        monthly history is fleet-wide and not scoped to that platform.
        ``scores_history.jsonl`` is only written for the combined prod
        accumulator, so the same numbers render in every per-platform report.
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
    if platform is not None:
        lines.append("_Fleet-wide monthly trend — not scoped to this platform._")
        lines.append("")

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
    """Generate the sample size warnings section.

    Gate: total rows per category < SAMPLE_SIZE_WARNING. Subsections
    like directional bias apply stricter gates on narrower denominators
    (e.g. n_losses within a category); the wording below makes that
    explicit so the "all categories sufficient" line does not read as
    contradicting a per-subsection "insufficient data" note.

    :param scores: parsed scores dict with a ``by_category`` mapping.
    :return: markdown section string.
    """
    lines = ["## Sample Size Warnings", ""]
    found = False

    for cat, stats in (scores.get("by_category") or {}).items():
        if stats["n"] < SAMPLE_SIZE_WARNING:
            found = True
            lines.append(
                f"- **{cat}**: only {stats['n']} questions"
                f" (< {SAMPLE_SIZE_WARNING}) — treat with caution"
            )

    if not found:
        lines.append(
            f"All categories have at least {SAMPLE_SIZE_WARNING} total"
            " questions (the category reporting gate)."
            " Subsections that use stricter gates on narrower"
            " denominators (e.g. n_losses in directional bias) may"
            " still flag specific categories as insufficient."
        )

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
        key=brier_sort_key,
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


def section_edge_analysis(scores: dict[str, Any], platform: str | None = None) -> str:
    """Edge-over-market analysis — per platform, difficulty, and liquidity.

    When ``platform`` is set, the scores input is already partitioned to
    one platform, so the per-platform / platform × difficulty / platform
    × liquidity sub-blocks are degenerate (one row each) and skipped.

    :param scores: parsed scores dict.
    :param platform: when set, suppresses the platform-comparison sub-blocks.
    :return: markdown section string.
    """
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

    # Sub-blocks below compare platforms; skip them when the input is
    # already platform-scoped.
    if platform is not None:
        return "\n".join(lines).rstrip()

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
    """Generate a period comparison section.

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
            key=brier_sort_key,
        ):
            tb = stats.get("brier")
            if tb is None:
                # Skip empty groups but still surface pipeline failures
                # (n > 0 with zero valid parses). Otherwise the scorer's
                # brier=None-on-valid_n==0 rule silently hides the tools
                # that _sample_label now labels 'all malformed'.
                if stats.get("n", 0) == 0:
                    continue
                lines.append(
                    f"  - **{tool}**: N/A" f" (n={stats['n']}){_sample_label(stats)}"
                )
                continue
            at_b = at_tools.get(tool, {}).get("brier")
            lines.append(
                f"  - **{tool}**: {tb:.4f}"
                f"{_delta_str(tb, at_b)}"
                f" (n={stats['n']}){_sample_label(stats)}"
            )

    return "\n".join(lines)


VERSION_DELTA_LOW_SAMPLE = 30
VERSION_DELTA_LOW_SAMPLE_STRICT = 300
VERSION_DELTA_UNCHANGED_EPSILON = 0.001


def _parse_tvm_key(key: str) -> tuple[str, str, str]:
    """Split a 'tool | version | mode' key. Pads if mode is missing (legacy)."""
    parts = [p.strip() for p in key.split("|")]
    while len(parts) < 3:
        parts.append("unknown")
    return parts[0], parts[1], parts[2]


def _version_label(cid: str, rm: dict[str, Any] | None = None) -> str:
    """Return the release-tag label for a CID, or the untagged fallback.

    Wraps :func:`benchmark.release_map.resolve` so callers inside
    ``analyze.py`` don't need to import the module directly.

    :param cid: IPFS CID string.
    :param rm: optional pre-loaded release map; defaults to the cached one.
    :return: release tag (e.g. ``"v0.17.2"``) or ``"untagged@..."``.
    """
    return release_map.resolve(cid, rm)


def _most_recent_prod_cid(
    tool: str,
    scores_prod: dict[str, Any],
    rm: dict[str, Any],
) -> str | None:
    """Return the production-mode CID with the latest release tag for *tool*.

    Iterates ``scores_prod["by_tool_version_mode"]``, keeps the cells
    whose tool matches *tool* and whose mode is ``production_replay``,
    and returns the CID of the latest (by release tag). Untagged CIDs
    fall through to the end; ties among them break arbitrarily but
    deterministically (by sort order of the fallback label).

    :param tool: runtime tool name.
    :param scores_prod: production scores dict.
    :param rm: release map.
    :return: production CID or None when no prod cell exists for *tool*.
    """
    tvm = scores_prod.get("by_tool_version_mode", {}) if scores_prod else {}
    tags_scanned = rm.get("tags_scanned", []) if rm else []
    candidates: list[tuple[str, str]] = []  # (cid, label)
    for key in tvm:
        t, cid, mode = _parse_tvm_key(key)
        if t != tool or mode != "production_replay":
            continue
        candidates.append((cid, release_map.resolve(cid, rm)))
    if not candidates:
        return None
    candidates.sort(key=lambda c: release_map.sort_key(c[1], tags_scanned))
    return candidates[-1][0]


def _pool_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool a list of stats dicts via n-weighted mean of row-mean metrics.

    Exact for Brier, LogLoss, directional accuracy, baseline Brier —
    all of which are row-level means whose weighted mean by n equals
    the pooled row mean.

    :param cells: list of stats dicts with ``n`` and per-metric fields.
    :return: single stats dict representing the pool.
    """
    total_n = sum(c.get("n", 0) or 0 for c in cells)
    if total_n == 0:
        return {"n": 0, "brier": None, "directional_accuracy": None}

    def _wmean(key: str) -> float | None:
        num = 0.0
        denom = 0
        for cell in cells:
            val = cell.get(key)
            if val is None:
                continue
            cell_n = cell.get("n", 0) or 0
            num += cell_n * val
            denom += cell_n
        return num / denom if denom else None

    return {
        "n": total_n,
        "brier": _wmean("brier"),
        "directional_accuracy": _wmean("directional_accuracy"),
        "log_loss": _wmean("log_loss"),
        "baseline_brier": _wmean("baseline_brier"),
    }


def section_tool_version_breakdown(
    scores: dict[str, Any],
    title: str = "Tool × Version × Mode",
    release_map_data: dict[str, Any] | None = None,
) -> str:
    """Per (tool, version, mode) metrics table — combines prod and tournament.

    The Version column shows release-tag labels (via
    :func:`benchmark.release_map.resolve`) when a CID resolves, or
    ``untagged@<short>`` otherwise. Rows are sorted by
    ``(tool, release_chronology, mode)`` so readers scan a tool's
    versions in deploy order.

    :param scores: scores dict containing ``by_tool_version_mode``.
    :param title: markdown heading text (without the leading ``##``).
    :param release_map_data: optional pre-loaded release map.
    :return: rendered markdown section, or empty string when empty.
    """
    tvm = scores.get("by_tool_version_mode", {})
    if not tvm:
        return ""

    if release_map_data is None:
        release_map_data = release_map.get_release_map()
    tags_scanned = release_map_data.get("tags_scanned", [])

    enriched: list[tuple[str, str, str, str, dict[str, Any]]] = []
    for key, stats in tvm.items():
        tool, cid, mode = _parse_tvm_key(key)
        label = release_map.resolve(cid, release_map_data)
        enriched.append((tool, cid, label, mode, stats))

    enriched.sort(
        key=lambda r: (r[0], release_map.sort_key(r[2], tags_scanned), r[3]),
    )

    lines = [
        f"## {title}",
        "",
        "| Tool | Version | Mode | n | valid | Brier | DirAcc | BSS |",
        "|------|---------|------|---:|---:|---:|---:|---:|",
    ]
    has_low_sample = False
    for tool, _cid, label, mode, stats in enriched:
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
            f"| {tool} | `{label}` | {mode} | {n_cell} | {valid_n}"
            f" | {brier_s} | {acc_s} | {bss_s} |"
        )

    if has_low_sample:
        lines.extend(
            [
                "",
                f"⚠ Rows marked with ⚠ have n < {VERSION_DELTA_LOW_SAMPLE};"
                " metrics from these cells are statistically unreliable and"
                " superlatives should not be drawn from them.",
            ]
        )
    return "\n".join(lines)


def _delta_direction(delta: float | None) -> str:
    """Return a one-word direction label for a Brier delta."""
    if delta is None:
        return "—"
    if delta < -VERSION_DELTA_UNCHANGED_EPSILON:
        return "improved"
    if delta > VERSION_DELTA_UNCHANGED_EPSILON:
        return "regressed"
    return "unchanged"


def _format_delta_row(
    baseline_label: str,
    candidate_label: str,
    baseline_stats: dict[str, Any],
    candidate_stats: dict[str, Any],
) -> str:
    """Render a single markdown row for a (baseline, candidate) pair.

    :param baseline_label: release-tag label (or pooled range).
    :param candidate_label: release-tag label for the candidate version.
    :param baseline_stats: baseline cell stats (or pooled stats dict).
    :param candidate_stats: candidate cell stats.
    :return: pipe-delimited markdown table row.
    """
    b_brier = baseline_stats.get("brier")
    c_brier = candidate_stats.get("brier")
    delta: float | None
    if b_brier is None or c_brier is None:
        delta = None
        delta_s = "—"
    else:
        delta = c_brier - b_brier
        delta_s = f"{delta:+.4f}"
    n_b = baseline_stats.get("n", 0) or 0
    n_c = candidate_stats.get("n", 0) or 0
    low_flag = " ⚠" if min(n_b, n_c) < VERSION_DELTA_LOW_SAMPLE_STRICT else ""
    return (
        f"| `{baseline_label}` | `{candidate_label}` | {delta_s}{low_flag}"
        f" | {_delta_direction(delta)} | {n_b} | {n_c} |"
    )


def section_version_deltas(
    scores: dict[str, Any],
    release_map_data: dict[str, Any] | None = None,
) -> str:
    """Per-tool, per-mode version timeline with prior + previous-pooled deltas.

    Replaces the alphabetical-by-CID pairwise table that was disabled
    in PR #215. Versions are ordered chronologically by release tag
    (via :mod:`benchmark.release_map`), with ``first_seen`` currently
    unused as a tiebreaker. For each (tool, mode) with ≥ 2 versions,
    two sub-tables render:

    - **vs prior version:** each V_i compared to V_{i-1}.
    - **vs previous pooled:** each V_i compared to the n-weighted pool
      of V_0..V_{i-1}.

    Within-mode only. Low-sample rows (``min(n) <
    VERSION_DELTA_LOW_SAMPLE_STRICT``) are flagged ⚠, not dropped.

    :param scores: merged scores dict whose ``by_tool_version_mode``
        holds cells for both production and tournament.
    :param release_map_data: optional pre-loaded release map.
    :return: markdown section, or empty string when no tool has ≥ 2
        versions in any single mode.
    """
    tvm = scores.get("by_tool_version_mode", {})
    if not tvm:
        return ""

    if release_map_data is None:
        release_map_data = release_map.get_release_map()
    tags_scanned = release_map_data.get("tags_scanned", [])

    # (tool, mode) -> list of (cid, label, stats) sorted by release chronology.
    by_tool_mode: dict[tuple[str, str], list[tuple[str, str, dict[str, Any]]]] = {}
    for key, stats in tvm.items():
        tool, cid, mode = _parse_tvm_key(key)
        label = release_map.resolve(cid, release_map_data)
        by_tool_mode.setdefault((tool, mode), []).append((cid, label, stats))

    multi = {k: cells for k, cells in by_tool_mode.items() if len(cells) >= 2}
    if not multi:
        return ""

    for cells in multi.values():
        cells.sort(key=lambda c: release_map.sort_key(c[1], tags_scanned))

    lines = ["## Version Deltas", ""]
    has_low_sample = False
    for (tool, mode), cells in sorted(multi.items()):
        lines.append(f"### {tool} ({mode})")
        lines.append("")

        lines.append("**vs prior version:**")
        lines.append("")
        lines.append("| Baseline | Candidate | Brier Δ | Direction | n_b | n_c |")
        lines.append("|---|---|---:|---|---:|---:|")
        for i in range(1, len(cells)):
            _, prior_label, prior_stats = cells[i - 1]
            _, cand_label, cand_stats = cells[i]
            if (
                min(prior_stats.get("n", 0) or 0, cand_stats.get("n", 0) or 0)
                < VERSION_DELTA_LOW_SAMPLE_STRICT
            ):
                has_low_sample = True
            lines.append(
                _format_delta_row(prior_label, cand_label, prior_stats, cand_stats)
            )
        lines.append("")

        lines.append("**vs previous pooled:**")
        lines.append("")
        lines.append(
            "| Baseline (pool) | Candidate | Brier Δ | Direction | n_b | n_c |"
        )
        lines.append("|---|---|---:|---|---:|---:|")
        for i in range(1, len(cells)):
            prior_cells = [c[2] for c in cells[:i]]
            pool_stats = _pool_cells(prior_cells)
            start_label = cells[0][1]
            end_label = cells[i - 1][1]
            pool_label = (
                start_label
                if start_label == end_label
                else f"{start_label}..{end_label}"
            )
            _, cand_label, cand_stats = cells[i]
            if (
                min(pool_stats.get("n", 0) or 0, cand_stats.get("n", 0) or 0)
                < VERSION_DELTA_LOW_SAMPLE_STRICT
            ):
                has_low_sample = True
            lines.append(
                _format_delta_row(pool_label, cand_label, pool_stats, cand_stats)
            )
        lines.append("")

    if has_low_sample:
        lines.append(
            f"⚠ Rows marked with ⚠ have min(n) <"
            f" {VERSION_DELTA_LOW_SAMPLE_STRICT}; the delta is within noise and"
            " the flagged version wasn't in production long enough to produce a"
            " load-bearing baseline."
        )
        lines.append("")
    lines.append(
        "The **vs previous pooled** table shows each candidate against the"
        " n-weighted pool of all earlier versions — the cumulative baseline."
        " For tools with exactly 2 versions, pool(V_0) equals V_0, so the"
        " pooled row matches the prior-version row; the two diverge once a"
        " tool has 3+ versions in that mode."
    )
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


def _annotate_with_window(section_md: str, window_label: str) -> str:
    """Insert an italicized window qualifier under the first ``## Heading``.

    :param section_md: rendered markdown for a single section.
    :param window_label: phrase describing the window (e.g. ``"last 3 days"``).
    :return: the section markdown with the qualifier line inserted.
    """
    lines = section_md.split("\n")
    if not lines or not lines[0].startswith("## "):
        return section_md
    note = f"_n= values below are over the {window_label}._"
    if len(lines) >= 2 and lines[1] == "":
        return "\n".join([lines[0], "", note, ""] + lines[2:])
    return "\n".join([lines[0], "", note, ""] + lines[1:])


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
    release_map_data: dict[str, Any] | None = None,
) -> str:
    """Flag tool versions whose tournament Brier diverges from prod.

    A row qualifies when tournament sample size is at least
    ``CALLOUT_MIN_N`` and the absolute Brier delta exceeds
    ``CALLOUT_DELTA``. Negative deltas become promotion candidates
    (tournament better); positive deltas become tournament regressions
    (tournament worse — a warning before the version reaches production).

    The production baseline is the **specific production CID with the
    latest release tag** for the same tool — not the tool-level
    aggregate. That way a rollout scenario (v2 partially in prod,
    tournament evaluating v2) compares against the right thing and
    doesn't get washed out by older versions' numbers.

    :param scores_prod: production scores dict.
    :param scores_tournament: tournament scores dict, or None.
    :param release_map_data: optional pre-loaded release map.
    :return: markdown section, or empty string when no callouts qualify.
    """
    if not _has_tournament_data(scores_tournament):
        return ""

    if release_map_data is None:
        release_map_data = release_map.get_release_map()

    assert scores_tournament is not None  # narrowed by _has_tournament_data
    tournament_tvm = scores_tournament.get("by_tool_version_mode", {})
    prod_tvm = (scores_prod or {}).get("by_tool_version_mode", {})

    # Callout tuple layout:
    #   tool, cand_cid, cand_label, t_stats, prod_cid, prod_label, p_stats
    Callout = tuple[str, str, str, dict[str, Any], str, str, dict[str, Any]]
    promotions: list[Callout] = []
    regressions: list[Callout] = []

    for key, t_stats in tournament_tvm.items():
        tool, cand_cid, mode = _parse_tvm_key(key)
        if mode != "tournament":
            continue
        if t_stats.get("n", 0) < CALLOUT_MIN_N:
            continue
        t_brier = t_stats.get("brier")
        if t_brier is None:
            continue

        prod_cid = _most_recent_prod_cid(tool, scores_prod or {}, release_map_data)
        if prod_cid is None:
            # Tournament-only tool — no prod baseline to compare against.
            continue
        if cand_cid == prod_cid:
            # Candidate has rolled out; comparing two samples of the same
            # version is eval-pipeline noise, not a promotion signal.
            continue
        p_stats = prod_tvm.get(f"{tool} | {prod_cid} | production_replay") or {}
        p_brier = p_stats.get("brier")
        if p_brier is None:
            continue

        cand_label = release_map.resolve(cand_cid, release_map_data)
        prod_label = release_map.resolve(prod_cid, release_map_data)

        delta = t_brier - p_brier
        entry: Callout = (
            tool,
            cand_cid,
            cand_label,
            t_stats,
            prod_cid,
            prod_label,
            p_stats,
        )
        if delta <= -CALLOUT_DELTA:
            promotions.append(entry)
        elif delta >= CALLOUT_DELTA:
            regressions.append(entry)

    if not promotions and not regressions:
        return ""

    def _bullet(entry: Callout) -> str:
        tool, _cand_cid, cand_label, t_stats, _prod_cid, prod_label, p_stats = entry
        delta = t_stats["brier"] - p_stats["brier"]
        return (
            f"- `{tool}` `{cand_label}` (tournament, n={t_stats['n']}) Brier"
            f" {t_stats['brier']:.4f} vs `{prod_label}` (production,"
            f" n={p_stats['n']}) Brier {p_stats['brier']:.4f}. Δ {delta:+.4f}."
        )

    lines = ["## Tournament Callouts", ""]
    if promotions:
        lines.append("**Promotion candidates:**")
        lines.append("")
        lines.extend(_bullet(e) for e in promotions)
        lines.append("")
    if regressions:
        lines.append("**Tournament regressions:**")
        lines.append("")
        lines.extend(_bullet(e) for e in regressions)
    return "\n".join(lines).rstrip()


def generate_report(
    scores: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    *,
    platform: str,
    period_scores: dict[str, Any] | None = None,
    rolling_scores: dict[str, Any] | None = None,
    include_tournament: bool = False,
    scores_tournament: dict[str, Any] | None = None,
    period_scores_tournament: dict[str, Any] | None = None,
    rolling_scores_tournament: dict[str, Any] | None = None,
    disabled_tools: dict[str, list[str] | None] | None = None,
) -> str:
    """Generate a platform-scoped benchmark report from scores and history.

    Every section is driven by scores already partitioned to ``platform``
    by the scorer, so the fleet-wide comparison sections (``section_platform``,
    ``section_tool_platform``, Platform × Difficulty, Platform × Liquidity)
    are dropped — they'd render a single-row view that adds noise without
    signal. The report header names the deployment this report covers.

    Production-mode sections are rendered from ``scores`` /
    ``period_scores`` / ``rolling_scores``. When tournament scores are
    supplied and contain rows, a duplicate set of the mode-sensitive
    sections is rendered with a ``— Tournament`` suffix.

    :param scores: parsed platform-scoped ``scores_<platform>.json`` dict.
    :param history: list of monthly snapshots from ``scores_history.jsonl``.
    :param platform: one of ``PLATFORM_LABELS`` keys (``"omen"`` or
        ``"polymarket"``). Drives the report header and gates platform-
        comparison sections.
    :param period_scores: production scores since last report.
    :param rolling_scores: production scores from the rolling window.
    :param include_tournament: master switch for rendering the Tool ×
        Version × Mode breakdown. When False, tournament inputs are
        ignored entirely.
    :param scores_tournament: parsed ``scores_tournament_<platform>.json`` dict.
    :param period_scores_tournament: tournament since last report.
    :param rolling_scores_tournament: tournament rolling window scores.
    :param disabled_tools: pre-fetched ``{deployment: [tool_names] | None}``
        map used by the Tool Deployment Status section.
    :return: full markdown report string.
    """
    if platform not in PLATFORM_LABELS:
        raise ValueError(
            f"platform must be one of {sorted(PLATFORM_LABELS)}, got {platform!r}"
        )
    platform_label = PLATFORM_LABELS[platform]

    if history is None:
        history = []

    date = scores.get("generated_at", "")[:10] or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d"
    )

    render_tournament = include_tournament and _has_tournament_data(scores_tournament)
    # Local non-optional alias for mypy once _has_tournament_data has narrowed.
    tournament_scores: dict[str, Any] = scores_tournament or {}

    sections: list[str] = [f"# Benchmark Report ({platform_label}) — {date}"]

    sections.append(section_metric_reference())

    # Since Last Report
    sections.append(section_period(period_scores, scores, "Since Last Report"))
    if render_tournament and _has_tournament_data(period_scores_tournament):
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

    rolling_heading = f"Last {ROLLING_WINDOW_DAYS} Days Rolling"
    sections.append(section_period(rolling_scores, scores, rolling_heading))
    if render_tournament and _has_tournament_data(rolling_scores_tournament):
        sections.append(
            _relabel_heading(
                section_period(
                    rolling_scores_tournament,
                    tournament_scores,
                    rolling_heading,
                ),
                " — Tournament",
            )
        )

    sections.append(section_base_rates(scores))
    sections.append(section_tool_deployment_status(scores, disabled=disabled_tools))

    rolling_for_sections = rolling_scores if rolling_scores is not None else scores
    rolling_tournament_for_sections = (
        rolling_scores_tournament
        if rolling_scores_tournament is not None
        else tournament_scores
    )
    rolling_suffix = f" (Last {ROLLING_WINDOW_DAYS} Days)"
    rolling_window_note = f"last {ROLLING_WINDOW_DAYS} days"

    def _rolling(section_md: str, heading_suffix: str = rolling_suffix) -> str:
        """Add the rolling-window heading suffix and n= qualifier note."""
        return _annotate_with_window(
            _relabel_heading(section_md, heading_suffix), rolling_window_note
        )

    sections.append(_rolling(section_tool_ranking(rolling_for_sections)))
    if render_tournament:
        sections.append(
            _rolling(
                section_tool_ranking(rolling_tournament_for_sections),
                f"{rolling_suffix} — Tournament",
            )
        )

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
                merged_rolling,
                f"Tool × Version × Mode (Last {ROLLING_WINDOW_DAYS} Days)",
            )
            if tvm_rolling:
                sections.append(tvm_rolling)
        deltas = section_version_deltas(merged)
        if deltas:
            sections.append(deltas)

    sections.extend(
        [
            _rolling(section_category(rolling_for_sections)),
            _rolling(section_tool_category(rolling_for_sections)),
            _rolling(section_diagnostic_metrics(rolling_for_sections)),
            _rolling(section_weak_spots(rolling_for_sections)),
            section_reliability_issues(scores),
            section_parse_breakdown(scores),
            section_trend(history, scores, platform=platform),
            section_sample_size_warnings(scores),
        ]
    )

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
    parser.add_argument(
        "--platform",
        required=True,
        choices=sorted(PLATFORM_LABELS),
        help="Platform to scope the report to (drives default paths + header).",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Override for scores file. Default: results/scores_<platform>.json.",
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report output path. Default: results/report_<platform>.md.",
    )
    parser.add_argument(
        "--period",
        type=Path,
        default=None,
        help=(
            "Period scores JSON (since last report). "
            "Default: results/period_scores_<platform>.json."
        ),
    )
    parser.add_argument(
        "--rolling",
        type=Path,
        default=None,
        help=(
            "Rolling scores JSON. " "Default: results/rolling_scores_<platform>.json."
        ),
    )
    parser.add_argument(
        "--scores-tournament",
        type=Path,
        default=None,
        help=(
            "Tournament scores JSON. "
            "Default: results/scores_tournament_<platform>.json."
        ),
    )
    parser.add_argument(
        "--period-tournament",
        type=Path,
        default=None,
        help=(
            "Tournament period scores JSON. "
            "Default: results/period_scores_tournament_<platform>.json."
        ),
    )
    parser.add_argument(
        "--rolling-tournament",
        type=Path,
        default=None,
        help=(
            "Tournament rolling scores JSON. "
            "Default: results/rolling_scores_tournament_<platform>.json."
        ),
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

    platform = args.platform
    results_dir = DEFAULT_RESULTS_DIR
    scores_path = args.scores or results_dir / f"scores_{platform}.json"
    output_path = args.output or results_dir / f"report_{platform}.md"
    period_path = args.period or results_dir / f"period_scores_{platform}.json"
    rolling_path = args.rolling or results_dir / f"rolling_scores_{platform}.json"
    scores_tournament_path = (
        args.scores_tournament or results_dir / f"scores_tournament_{platform}.json"
    )
    period_tournament_path = (
        args.period_tournament
        or results_dir / f"period_scores_tournament_{platform}.json"
    )
    rolling_tournament_path = (
        args.rolling_tournament
        or results_dir / f"rolling_scores_tournament_{platform}.json"
    )

    def _maybe_load(path: Path | None) -> dict[str, Any] | None:
        return load_scores(path) if path and path.exists() else None

    scores = load_scores(scores_path)
    history = load_history(args.history)
    period = _maybe_load(period_path)
    rolling = _maybe_load(rolling_path)
    scores_tournament = _maybe_load(scores_tournament_path)
    period_tournament = _maybe_load(period_tournament_path)
    rolling_tournament = _maybe_load(rolling_tournament_path)

    print(
        f"Loaded scores ({scores.get('total_rows', 0)} rows) for "
        f"{PLATFORM_LABELS[platform]}, {len(history)} months of history"
    )

    report = generate_report(
        scores,
        history,
        platform=platform,
        period_scores=period,
        rolling_scores=rolling,
        include_tournament=args.include_tournament,
        scores_tournament=scores_tournament,
        period_scores_tournament=period_tournament,
        rolling_scores_tournament=rolling_tournament,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written to {output_path}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
