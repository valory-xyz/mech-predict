"""
Generate a human-readable benchmark report for one platform deployment.

Reads scores_<platform>.json (current month accumulators) and
scores_history.jsonl (monthly snapshots) to produce a markdown report
with rankings, weak spots, and highlights. Never reads raw log files.

Usage:
    python -m benchmark.analyze --platform omen --include-tournament
    python -m benchmark.analyze --platform polymarket --include-tournament

Overrides (for ad-hoc renders against a specific scores file):
    python -m benchmark.analyze --platform omen --scores scores_omen.json --output report_omen.md
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from benchmark import release_map

# OMEN_CATEGORIES / POLYMARKET_ACTIVE_CATEGORIES are re-exported so
# callers that previously imported these constants from benchmark.analyze
# keep working after the constants moved to benchmark.categories.
from benchmark.categories import (  # noqa: F401  # pylint: disable=unused-import
    ACTIVE_CATEGORIES,
    OMEN_CATEGORIES,
    POLYMARKET_ACTIVE_CATEGORIES,
)
from benchmark.io import load_jsonl
from benchmark.scorer import (
    DISAGREE_THRESHOLD,
    LARGE_TRADE_THRESHOLD,
    MIN_SAMPLE_SIZE,
    brier_sort_key,
)
from benchmark.tool_usage import deployments_for_platform, fetch_disabled_tools

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

# Trailing-window length (in days) for the Current-window snapshot and
# the previous non-overlapping window used by the comparison sections.
# Read from BENCHMARK_ROLLING_WINDOW_DAYS so the CI workflow drives the
# Python constant and the ``--period-days`` flag passed to the scorer
# from one source of truth (see .github/workflows/benchmark_flywheel.yaml).
ROLLING_WINDOW_DAYS = int(os.environ.get("BENCHMARK_ROLLING_WINDOW_DAYS", "7"))

BRIER_RANDOM = 0.25
BRIER_WEAK_THRESHOLD = 0.40
BSS_HARMFUL_THRESHOLD = 0.0
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
    if not path.exists():
        return []
    return load_jsonl(path)


# ---------------------------------------------------------------------------
# Three-window comparison helpers
# ---------------------------------------------------------------------------
#
# The per-platform report renders each metric against three windows:
#
#   - Current 3d (the rolling window the report is anchored to)
#   - All-time (cumulative since the first scored row)
#   - Prev non-overlapping 3d (the 3-day window immediately preceding
#     the current one, shifted back by the window width so the two
#     windows share no rows)
#
# Delta formatting obeys two hard rules:
#   - No delta when either side has n < MIN_SAMPLE_SIZE. "insufficient
#     data" is rendered instead so readers never see a signed number
#     with no sample-size anchor.
#   - n is always cited next to the absolute value it measures.


_INSUFFICIENT = "insufficient data"


def _delta_cell(
    current: float | None,
    reference: float | None,
    current_n: int,
    reference_n: int,
    lower_is_better: bool = True,
) -> str:
    """Format a delta cell with sample-size guardrails.

    :param current: value from the current window.
    :param reference: value from the reference window (all-time or prev 3d).
    :param current_n: sample size for the current window.
    :param reference_n: sample size for the reference window.
    :param lower_is_better: when True (default, matches Brier/LogLoss),
        a negative delta renders as "better"; when False (for BSS,
        Edge, directional accuracy), a positive delta renders as
        "better" instead.
    :return: delta cell string.
    """
    if current is None or reference is None:
        return "N/A"
    if current_n < MIN_SAMPLE_SIZE or reference_n < MIN_SAMPLE_SIZE:
        return _INSUFFICIENT
    delta = current - reference
    if lower_is_better:
        direction = "better" if delta < 0 else "worse" if delta > 0 else "same"
    else:
        direction = "better" if delta > 0 else "worse" if delta < 0 else "same"
    return f"{delta:+.4f} {direction}"


def _value_cell(
    value: float | None,
    sample_n: int,
    decimals: int = 4,
) -> str:
    """Format an absolute-value cell with an n= anchor.

    :param value: metric value (already rounded by the scorer).
    :param sample_n: sample size that produced this value.
    :param decimals: number of decimal places to render.
    :return: cell string such as ``"0.2100 (n=42)"`` or ``"N/A (n=5)"``.
    """
    if value is None:
        return f"N/A (n={sample_n})"
    return f"{value:.{decimals}f} (n={sample_n})"


def _pct_cell(value: float | None, sample_n: int) -> str:
    """Format a percentage-value cell with an n= anchor.

    :param value: metric value in [0, 1] or ``None``.
    :param sample_n: sample size that produced this value.
    :return: cell string such as ``"70% (n=42)"``.
    """
    if value is None:
        return f"N/A (n={sample_n})"
    return f"{value:.0%} (n={sample_n})"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def section_metric_reference(include_scope_note: bool = True) -> str:
    """Render the metric-reference legend shown at the top of every report.

    :param include_scope_note: when True, prepends the rolling-vs-all-time
        scope explanation. The fleet report omits the note because it
        contains no rolling-scoped sections.
    :return: markdown section string.
    """
    lines = ["## Metric References", ""]
    if include_scope_note:
        lines.extend(
            [
                (
                    f"Every comparison below uses three windows: `Current "
                    f"{ROLLING_WINDOW_DAYS}d` (trailing {ROLLING_WINDOW_DAYS}-day "
                    "aggregate), `All-Time` (cumulative since first scored row), "
                    f"and `Prev {ROLLING_WINDOW_DAYS}d` (the immediately preceding "
                    f"non-overlapping {ROLLING_WINDOW_DAYS}-day window). Deltas "
                    "compare `Current` against each reference window."
                ),
                "",
                (
                    "Guardrails:"
                    " a delta is suppressed when either side has n < "
                    f"{MIN_SAMPLE_SIZE} (rendered as `insufficient data`);"
                    " n is cited next to every absolute value;"
                    " the Trend section is fleet-wide monthly, independent "
                    "of this report's platform scope."
                ),
                "",
            ]
        )
    lines.extend(
        [
            f"- **Brier** — ideal 0.00, coin-flip {BRIER_RANDOM}; lower is better.",
            (
                "- **Log Loss** — ideal 0.00; lower is better, punishes confident "
                "errors harder."
            ),
            (
                "- **BSS (Brier Skill Score)** — ideal > 0; negative means worse "
                "than the base-rate predictor."
            ),
            (
                "- **Edge over market** — ideal > 0; positive = tool beats "
                "market consensus. System diagnostic, not a tool ranking signal."
            ),
        ]
    )
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


def _active_tools_for_platform(
    disabled: dict[str, list[str] | None] | None,
    platform: str,
    scores: dict[str, Any],
    rolling_scores: dict[str, Any] | None = None,
) -> frozenset[str] | None:
    """Return tools currently active on at least one deployment of ``platform``.

    Computed as the union of (benchmarked tools - disabled tools) across
    every deployment of ``platform`` whose config fetch succeeded. A tool
    is "active" if any deployment has it enabled.

    Tool-name normalization mirrors ``section_tool_deployment_status``:
    underscores and hyphens are treated as interchangeable when comparing
    against the disabled list. The returned set uses the names as they
    appear in ``by_tool`` keys.

    :param disabled: ``{deployment: [tool_names] | None}`` map, where
        ``None`` indicates a fetch/parse failure for that deployment.
        ``None`` for the whole map (or an empty dict) is treated as
        "no deployment data available".
    :param platform: scorer platform key (``"omen"`` or ``"polymarket"``).
    :param scores: parsed platform-scoped all-time scores dict.
    :param rolling_scores: parsed platform-scoped current-window scores
        dict. The benchmarked-tool universe is the union of ``scores``
        and ``rolling_scores`` ``by_tool`` keys so a freshly-deployed
        tool with rolling data but no all-time history yet is still
        included in the active set.
    :return: frozenset of active tool names, or ``None`` when **every**
        deployment of this platform has ``disabled=None`` (full fetch
        failure for the platform). Callers fall back to "show all
        tools" plus a ``⚠ deployment config unavailable`` notice.
    """
    if not disabled:
        return None

    deployments = deployments_for_platform(platform)
    relevant = {name: disabled.get(name) for name in deployments}
    if all(disabled_tools is None for disabled_tools in relevant.values()):
        # Every deployment for this platform failed — caller renders the
        # warning and shows all tools rather than blanking the report.
        return None

    benchmarked = set((scores.get("by_tool") or {}).keys())
    if rolling_scores is not None:
        benchmarked |= set((rolling_scores.get("by_tool") or {}).keys())

    active: set[str] = set()
    for disabled_tools in relevant.values():
        if disabled_tools is None:
            continue
        disabled_set = {t.replace("_", "-") for t in disabled_tools}
        for tool in benchmarked:
            if tool.replace("_", "-") not in disabled_set:
                active.add(tool)

    return frozenset(active)


def _filter_by_active(
    items: list[tuple[str, Any]],
    active_tools: frozenset[str] | None,
    *,
    composite_key_separator: str | None = None,
) -> list[tuple[str, Any]]:
    """Drop entries whose tool name is not in ``active_tools``.

    :param items: ``[(key, stats), ...]`` ranked-iteration list.
    :param active_tools: set of tools currently deployed on the platform,
        or ``None`` to disable filtering (caller's fallback path when
        deployment config could not be fetched).
    :param composite_key_separator: when set, treat ``key`` as a
        composite ``"tool{sep}other"`` and filter on the first segment.
        Pass ``" | "`` for ``by_tool_category`` keys.
    :return: ``items`` with non-active entries removed; identity when
        ``active_tools`` is ``None``.
    """
    if active_tools is None:
        return items
    if composite_key_separator is None:
        return [(k, s) for k, s in items if k in active_tools]
    return [
        (k, s)
        for k, s in items
        if k.split(composite_key_separator, 1)[0] in active_tools
    ]


def section_tool_deployment_status(
    scores: dict[str, Any],
    disabled: dict[str, list[str] | None] | None = None,
    platform: str | None = None,
) -> str:
    """Render which benchmarked tools are active on each deployment.

    Deployments are filtered to ``platform`` when set; active tools are
    the benchmarked tools minus the disabled tools for that deployment.
    Failed-fetch deployments are called out so ``⚠️ unavailable`` is
    never confused with "all tools active".

    :param scores: parsed ``scores.json`` dict.
    :param disabled: pre-fetched ``{deployment: [tool_names] | None}`` map.
        Pass ``None`` to fetch on the fly (the daily-report default). Pass
        an empty dict to skip the section entirely (tests use this to avoid
        the live GitHub fetch).
    :param platform: when set, restrict the section to deployments matching
        this platform key (``"omen"`` or ``"polymarket"``).
    :return: markdown section string, or ``""`` when the caller opted out.
    """
    if disabled is None:
        disabled = fetch_disabled_tools()

    # Explicit skip contract: an empty dict means "caller opted out" (used by
    # unit tests).  Returning "" means the section heading is omitted so the
    # report doesn't advertise a section that was never computed.
    if not disabled:
        return ""

    if platform is not None:
        allowed = set(deployments_for_platform(platform))
        disabled = {name: v for name, v in disabled.items() if name in allowed}
        if not disabled:
            return ""

    tools = scores.get("by_tool", {})
    benchmarked = [name for name, _ in sorted(tools.items(), key=brier_sort_key)]

    heading = "## Tool Deployment Status"
    if platform is not None:
        heading = f"## Tool Deployment Status ({PLATFORM_LABELS[platform]})"
    lines = [heading, ""]

    failed = [name for name in disabled if disabled.get(name) is None]
    if failed:
        lines.append(
            "> ⚠️ Could not fetch deployment config for: "
            f"{', '.join(failed)}. Status below may be incomplete."
        )
        lines.append("")

    for deployment, disabled_tools in disabled.items():
        if disabled_tools is None:
            lines.append(f"- **{deployment}** — ⚠️ unavailable")
            continue
        disabled_set = {t.replace("_", "-") for t in disabled_tools}
        active = [t for t in benchmarked if t.replace("_", "-") not in disabled_set]
        if not active:
            lines.append(f"- **{deployment}** — no benchmarked tools active")
            continue
        tools_str = ", ".join(f"`{t}`" for t in active)
        lines.append(f"- **{deployment}** ({len(active)} active): {tools_str}")

    return "\n".join(lines)


def section_tool_ranking(
    scores: dict[str, Any],
    active_tools: frozenset[str] | None = None,
) -> str:
    """Generate the tool ranking section.

    :param scores: parsed platform-scoped scores dict.
    :param active_tools: when set, only tools in this set are ranked.
        ``None`` (the default) preserves the legacy "show every tool
        with rows" behaviour and is used as a fallback when deployment
        config could not be fetched.
    :return: markdown section string.
    """
    tools = scores.get("by_tool", {})
    ranked = _filter_by_active(
        sorted(tools.items(), key=brier_sort_key),
        active_tools,
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


def _always_majority(yes_rate: float | None) -> float | None:
    """Return ``max(yes_rate, 1-yes_rate)`` — the always-majority baseline.

    :param yes_rate: fraction of valid rows whose final outcome is YES.
    :return: always-majority accuracy in ``[0.5, 1.0]`` or ``None`` when
        ``yes_rate`` is missing.
    """
    if yes_rate is None:
        return None
    return max(yes_rate, 1.0 - yes_rate)


def _da_lift(
    directional_accuracy: float | None, yes_rate: float | None
) -> float | None:
    """Return directional-accuracy lift over the always-majority baseline.

    A positive value means the tool beat "always predict the majority
    outcome" on valid rows; a value at or below zero means the tool
    gave no edge over a constant baseline.

    :param directional_accuracy: tool's directional accuracy on valid rows.
    :param yes_rate: outcome yes rate on valid rows.
    :return: DA lift, or ``None`` when either input is missing.
    """
    majority = _always_majority(yes_rate)
    if directional_accuracy is None or majority is None:
        return None
    return directional_accuracy - majority


def section_tool_category(
    scores: dict[str, Any],
    active_tools: frozenset[str] | None = None,
) -> str:
    """Tool × category cross breakdown — primary metrics, gated by MIN_SAMPLE_SIZE.

    Renders the reviewer-specified column set: n, reliability, Brier,
    baseline Brier, BSS, directional accuracy, yes/no rate,
    always-majority baseline, and DA lift. Diagnostic metrics (edge,
    edge_n, log loss) render in a separate ``section_tool_category_diagnostics``
    section so this table stays scannable.

    :param scores: parsed per-platform rolling scores dict.
    :param active_tools: when set, only cells whose tool half is in this
        set render. ``None`` disables filtering (used as the fallback
        when deployment config could not be fetched).
    :return: markdown section string.
    """
    data = scores.get("by_tool_category") or {}
    if not data:
        return "## Tool × Category\n\nNo cross-breakdown data available."

    ranked = _filter_by_active(
        sorted(data.items(), key=brier_sort_key),
        active_tools,
        composite_key_separator=" | ",
    )
    sufficient = [(k, s) for k, s in ranked if s["n"] >= MIN_SAMPLE_SIZE]
    sparse = [(k, s) for k, s in ranked if s["n"] < MIN_SAMPLE_SIZE]

    lines = [
        "## Tool × Category",
        "",
        f"> Cells with n < {MIN_SAMPLE_SIZE} are moved to a separate list below"
        " the ranking. Diagnostic metrics (edge, log loss) live in the"
        " next section.",
        "",
        "| Tool | Category | n | Reliability | Brier | Baseline Brier | BSS"
        " | DirAcc | Yes% | No% | Always-majority | DA lift |",
        "|------|----------|---|-------------|-------|----------------|-----"
        "|--------|------|-----|-----------------|---------|",
    ]
    if not sufficient:
        lines.append(
            f"| _(no cells with n ≥ {MIN_SAMPLE_SIZE})_" " | | | | | | | | | | | |"
        )
    for key, stats in sufficient:
        parts = key.split(" | ")
        tool = parts[0] if parts else key
        category = parts[1] if len(parts) > 1 else "?"
        brier = f"{stats['brier']:.4f}" if stats.get("brier") is not None else "N/A"
        baseline = stats.get("baseline_brier")
        baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
        bss = stats.get("brier_skill_score")
        bss_str = f"{bss:+.4f}" if bss is not None else "N/A"
        acc_val = stats.get("directional_accuracy")
        acc = f"{acc_val:.0%}" if acc_val is not None else "N/A"
        yes = stats.get("outcome_yes_rate")
        yes_str = f"{yes:.0%}" if yes is not None else "N/A"
        no_str = f"{1 - yes:.0%}" if yes is not None else "N/A"
        majority = _always_majority(yes)
        majority_str = f"{majority:.0%}" if majority is not None else "N/A"
        lift = _da_lift(acc_val, yes)
        lift_str = f"{lift:+.4f}" if lift is not None else "N/A"
        reliability = stats.get("reliability")
        rel_str = f"{reliability:.0%}" if reliability is not None else "N/A"
        label = _sample_label(stats)
        lines.append(
            f"| {tool} | {category} | {stats['n']}{label} | {rel_str}"
            f" | {brier} | {baseline_str} | {bss_str} | {acc}"
            f" | {yes_str} | {no_str} | {majority_str} | {lift_str} |"
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


def section_tool_category_diagnostics(
    scores: dict[str, Any],
    active_tools: frozenset[str] | None = None,
) -> str:
    """Tool × category diagnostic metrics — edge, edge_n, log loss.

    Rendered as a follow-on to ``section_tool_category`` so the primary
    table stays compact while the diagnostic view still reads at the
    per-(tool, category) grain.

    :param scores: parsed per-platform rolling scores dict.
    :param active_tools: when set, only cells whose tool half is in this
        set render. ``None`` disables filtering (deployment-config
        fallback).
    :return: markdown section string.
    """
    data = scores.get("by_tool_category") or {}
    if not data:
        return "## Tool × Category Diagnostics\n\n" "No cross-breakdown data available."

    ranked = _filter_by_active(
        sorted(data.items(), key=brier_sort_key),
        active_tools,
        composite_key_separator=" | ",
    )
    sufficient = [(k, s) for k, s in ranked if s["n"] >= MIN_SAMPLE_SIZE]

    lines = [
        "## Tool × Category Diagnostics",
        "",
        f"> Edge, Edge n, and Log Loss for each cell above n = {MIN_SAMPLE_SIZE}.",
        "",
        "| Tool | Category | Edge | Edge n | Log Loss | n |",
        "|------|----------|------|--------|----------|---|",
    ]
    if not sufficient:
        lines.append(f"| _(no cells with n ≥ {MIN_SAMPLE_SIZE})_ | | | | | |")
        return "\n".join(lines)
    for key, stats in sufficient:
        parts = key.split(" | ")
        tool = parts[0] if parts else key
        category = parts[1] if len(parts) > 1 else "?"
        edge = stats.get("edge")
        edge_str = f"{edge:+.4f}" if edge is not None else "N/A"
        edge_n = stats.get("edge_n", 0)
        ll = stats.get("log_loss")
        ll_str = f"{ll:.4f}" if ll is not None else "N/A"
        lines.append(
            f"| {tool} | {category} | {edge_str} | {edge_n} | {ll_str}"
            f" | {stats['n']} |"
        )

    return "\n".join(lines)


def section_category_platform(scores: dict[str, Any]) -> str:
    """Category × platform cross breakdown — gated by MIN_SAMPLE_SIZE.

    :param scores: parsed combined ``scores.json`` dict with a
        ``by_category_platform`` mapping.
    :return: markdown section string.
    """
    data = scores.get("by_category_platform") or {}
    if not data:
        return "## Category × Platform\n\n" "No cross-breakdown data available."

    ranked = sorted(data.items(), key=brier_sort_key)
    sufficient = [(k, s) for k, s in ranked if s["n"] >= MIN_SAMPLE_SIZE]
    sparse = [(k, s) for k, s in ranked if s["n"] < MIN_SAMPLE_SIZE]

    lines = [
        "## Category × Platform",
        "",
        f"> Cells with n < {MIN_SAMPLE_SIZE} are moved to a separate list below.",
        "",
        "| Category | Platform | Brier | BSS | LogLoss | DirAcc | n |",
        "|----------|----------|-------|-----|---------|--------|---|",
    ]
    if not sufficient:
        lines.append(f"| _(no cells with n ≥ {MIN_SAMPLE_SIZE})_ | | | | | | |")
    for key, stats in sufficient:
        parts = key.split(" | ")
        category = parts[0] if parts else key
        platform = parts[1] if len(parts) > 1 else "?"
        brier = f"{stats['brier']:.4f}" if stats.get("brier") is not None else "N/A"
        bss = stats.get("brier_skill_score")
        bss_str = f"{bss:+.4f}" if bss is not None else "N/A"
        ll = stats.get("log_loss")
        ll_str = f"{ll:.4f}" if ll is not None else "N/A"
        acc = (
            f"{stats['directional_accuracy']:.0%}"
            if stats.get("directional_accuracy") is not None
            else "N/A"
        )
        lines.append(
            f"| {category} | {platform} | {brier} | {bss_str} | {ll_str} | {acc}"
            f" | {stats['n']} |"
        )

    if sparse:
        lines.append("")
        lines.append(
            f"_{len(sparse)} cell(s) below n={MIN_SAMPLE_SIZE} threshold omitted"
            " from ranking._"
        )
        for key, stats in sparse[:5]:
            parts = key.split(" | ")
            category = parts[0] if parts else key
            platform = parts[1] if len(parts) > 1 else "?"
            lines.append(
                f"- **{category} | {platform}**: insufficient data (n={stats['n']})"
            )

    return "\n".join(lines)


def section_tool_category_platform(scores: dict[str, Any]) -> str:
    """Tool × category × platform tri-dimensional breakdown.

    Gated by MIN_SAMPLE_SIZE. Rendered as a table so readers can rank
    the combined (tool, category, platform) slice directly from one
    artifact.

    :param scores: parsed combined ``scores.json`` dict with a
        ``by_tool_category_platform`` mapping.
    :return: markdown section string.
    """
    data = scores.get("by_tool_category_platform") or {}
    if not data:
        return "## Tool × Category × Platform\n\n" "No cross-breakdown data available."

    ranked = sorted(data.items(), key=brier_sort_key)
    sufficient = [(k, s) for k, s in ranked if s["n"] >= MIN_SAMPLE_SIZE]
    sparse_count = len(ranked) - len(sufficient)

    lines = [
        "## Tool × Category × Platform",
        "",
        f"> Cells with n < {MIN_SAMPLE_SIZE} are omitted.",
        "",
        "| Tool | Category | Platform | Brier | BSS | LogLoss | Edge | DirAcc | n |",
        "|------|----------|----------|-------|-----|---------|------|--------|---|",
    ]
    if not sufficient:
        lines.append(f"| _(no cells with n ≥ {MIN_SAMPLE_SIZE})_ | | | | | | | | |")
    for key, stats in sufficient:
        parts = key.split(" | ")
        tool = parts[0] if parts else key
        category = parts[1] if len(parts) > 1 else "?"
        platform = parts[2] if len(parts) > 2 else "?"
        brier = f"{stats['brier']:.4f}" if stats.get("brier") is not None else "N/A"
        bss = stats.get("brier_skill_score")
        bss_str = f"{bss:+.4f}" if bss is not None else "N/A"
        ll = stats.get("log_loss")
        ll_str = f"{ll:.4f}" if ll is not None else "N/A"
        edge = stats.get("edge")
        edge_str = f"{edge:+.4f}" if edge is not None else "N/A"
        acc = (
            f"{stats['directional_accuracy']:.0%}"
            if stats.get("directional_accuracy") is not None
            else "N/A"
        )
        lines.append(
            f"| {tool} | {category} | {platform} | {brier} | {bss_str} | {ll_str}"
            f" | {edge_str} | {acc} | {stats['n']} |"
        )

    if sparse_count:
        lines.append("")
        lines.append(
            f"_{sparse_count} cell(s) below n={MIN_SAMPLE_SIZE} threshold"
            " omitted from ranking._"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Three-window comparison sections
# ---------------------------------------------------------------------------


def section_platform_snapshot(rolling_scores: dict[str, Any]) -> str:
    """Render the current rolling-window platform snapshot.

    Metrics: n, reliability, Brier, baseline Brier, BSS, directional
    accuracy, outcome yes rate, and the always-majority baseline
    (``max(yes_rate, 1-yes_rate)``) so readers can eyeball whether the
    dataset is homogeneous enough for a low Brier to reflect base rate
    rather than prediction skill.

    :param rolling_scores: parsed ``rolling_scores_<platform>.json``.
    :return: markdown section string.
    """
    heading = f"## Platform Snapshot (Current {ROLLING_WINDOW_DAYS}d)"
    overall = rolling_scores.get("overall") or {}
    n = overall.get("n", 0)
    valid_n = overall.get("valid_n", 0)
    reliability = overall.get("reliability")
    brier = overall.get("brier")
    baseline = overall.get("baseline_brier")
    bss = overall.get("brier_skill_score")
    dir_acc = overall.get("directional_accuracy")
    yes_rate = overall.get("outcome_yes_rate")

    if n == 0:
        return f"{heading}\n\nNo rows scored in the current window."

    majority = _always_majority(yes_rate)
    majority_str = f"{majority:.0%}" if majority is not None else "N/A"
    lift = _da_lift(dir_acc, yes_rate)
    lift_str = f"{lift:+.4f}" if lift is not None else "N/A"
    no_rate = 1.0 - yes_rate if yes_rate is not None else None

    lines = [
        heading,
        "",
        f"- **n**: {n} (valid_n: {valid_n})",
        f"- **Reliability**: {_pct_cell(reliability, n)}",
        f"- **Brier**: {_value_cell(brier, valid_n)}",
        f"- **Baseline Brier**: {_value_cell(baseline, valid_n)}",
        f"- **BSS**: {_value_cell(bss, valid_n)}",
        f"- **Directional Accuracy**: {_pct_cell(dir_acc, valid_n)}",
        f"- **Outcome Yes Rate**: {_pct_cell(yes_rate, valid_n)}",
        f"- **Outcome No Rate**: {_pct_cell(no_rate, valid_n)}",
        f"- **Always-majority baseline**: {majority_str}",
        f"- **DA lift (DirAcc - always-majority)**: {lift_str}",
    ]
    return "\n".join(lines)


def _platform_metric_row(
    label: str,
    current: dict[str, Any],
    alltime: dict[str, Any],
    prev: dict[str, Any] | None,
    key: str,
    *,
    lower_is_better: bool = True,
    decimals: int = 4,
    is_pct: bool = False,
) -> str:
    """Build one comparison-table row for a platform-level metric.

    :param label: row label (e.g. ``"Brier"``).
    :param current: current-window overall stats dict.
    :param alltime: all-time overall stats dict.
    :param prev: previous-window overall stats dict, or ``None`` when
        prev-rolling scores are unavailable for this run.
    :param key: stats-dict key for the metric (e.g. ``"brier"``).
    :param lower_is_better: passed to ``_delta_cell`` direction labeling.
    :param decimals: decimal places for absolute values (ignored when
        ``is_pct`` is True).
    :param is_pct: when True, render absolute values as percentages.
    :return: markdown table row (leading and trailing pipe included).
    """
    c_val = current.get(key)
    a_val = alltime.get(key)
    p_val = prev.get(key) if prev is not None else None
    # valid_n drives sample-size gating for rate metrics; "reliability"
    # uses n (row count) not valid_n, so the caller passes the right
    # denominator key by using "n" as the metric key for that row.
    c_n = current.get("valid_n", current.get("n", 0))
    a_n = alltime.get("valid_n", alltime.get("n", 0))
    p_n = prev.get("valid_n", prev.get("n", 0)) if prev is not None else 0
    if key == "reliability":
        c_n = current.get("n", 0)
        a_n = alltime.get("n", 0)
        p_n = prev.get("n", 0) if prev is not None else 0

    value_fmt = _pct_cell if is_pct else lambda v, n: _value_cell(v, n, decimals)

    delta_alltime = _delta_cell(c_val, a_val, c_n, a_n, lower_is_better)
    if prev is None:
        prev_val_cell = "no prev window"
        delta_prev = "no prev window"
    else:
        prev_val_cell = value_fmt(p_val, p_n)
        delta_prev = _delta_cell(c_val, p_val, c_n, p_n, lower_is_better)
    return (
        f"| {label} | {value_fmt(c_val, c_n)} | {value_fmt(a_val, a_n)}"
        f" | {delta_alltime} | {prev_val_cell} | {delta_prev} |"
    )


def section_platform_comparison(
    rolling_scores: dict[str, Any],
    alltime_scores: dict[str, Any],
    prev_rolling_scores: dict[str, Any] | None,
) -> str:
    """Render the platform historical comparison table.

    Each metric row compares the current window against all-time and
    (when available) the previous non-overlapping window, with deltas
    and sample sizes gated by ``MIN_SAMPLE_SIZE``.

    :param rolling_scores: current rolling-window scores.
    :param alltime_scores: all-time / cumulative scores.
    :param prev_rolling_scores: previous non-overlapping rolling scores,
        or ``None`` when the upstream scoring step has not yet landed
        one on disk.
    :return: markdown section string.
    """
    heading = "## Platform Historical Comparison"
    current = rolling_scores.get("overall") or {}
    alltime = alltime_scores.get("overall") or {}
    prev = (prev_rolling_scores or {}).get("overall")

    lines = [
        heading,
        "",
        (
            "| Metric"
            f" | Current {ROLLING_WINDOW_DAYS}d"
            " | All-Time"
            " | Δ vs All-Time"
            f" | Prev {ROLLING_WINDOW_DAYS}d"
            f" | Δ vs Prev {ROLLING_WINDOW_DAYS}d |"
        ),
        "|--------|---------|----------|---------------|---------|-------------|",
        _platform_metric_row(
            "Reliability", current, alltime, prev, "reliability", is_pct=True
        ),
        _platform_metric_row("Brier", current, alltime, prev, "brier"),
        _platform_metric_row(
            "Baseline Brier", current, alltime, prev, "baseline_brier"
        ),
        _platform_metric_row(
            "BSS", current, alltime, prev, "brier_skill_score", lower_is_better=False
        ),
        _platform_metric_row(
            "Directional Accuracy",
            current,
            alltime,
            prev,
            "directional_accuracy",
            lower_is_better=False,
            is_pct=True,
        ),
        _platform_metric_row("Log Loss", current, alltime, prev, "log_loss"),
    ]
    return "\n".join(lines)


def _tool_universe(
    *score_dicts: dict[str, Any] | None,
) -> list[str]:
    """Return the ordered union of tool names across the given scores.

    Ordering: tools that appear in the first non-None scores dict keep
    its ranking order; tools unique to later dicts are appended in the
    order they're first seen. This keeps comparison rows aligned with
    the current-window ranking when it exists.

    :param score_dicts: any number of scores dicts (``None`` entries are
        ignored).
    :return: ordered list of unique tool names.
    """
    seen: dict[str, None] = {}
    for sd in score_dicts:
        if sd is None:
            continue
        by_tool = sd.get("by_tool") or {}
        ranked = sorted(by_tool.items(), key=brier_sort_key)
        for tool, _ in ranked:
            if tool not in seen:
                seen[tool] = None
    return list(seen)


def section_tool_comparison(
    rolling_scores: dict[str, Any],
    alltime_scores: dict[str, Any],
    prev_rolling_scores: dict[str, Any] | None,
    active_tools: frozenset[str] | None = None,
) -> str:
    """Render the tool historical comparison table.

    One row per tool, ranked by current-window Brier. Brier is shown for
    each window with n= anchors, plus deltas vs all-time and vs prev
    non-overlapping window. Low-sample cells are gated — deltas
    disappear rather than mislead.

    :param rolling_scores: current rolling-window scores.
    :param alltime_scores: all-time scores (used for Δ vs all-time).
    :param prev_rolling_scores: previous non-overlapping rolling scores,
        or ``None``.
    :param active_tools: when set, restrict the table to currently-deployed
        tools so historical/tournament-only entries don't clutter the
        per-platform report. ``None`` disables filtering (used as the
        fallback when deployment config could not be fetched).
    :return: markdown section string.
    """
    heading = "## Tool Historical Comparison"
    tools = _tool_universe(rolling_scores, alltime_scores)
    if active_tools is not None:
        tools = [t for t in tools if t in active_tools]
    if not tools:
        return f"{heading}\n\nNo tool data available."

    lines = [
        heading,
        "",
        (
            "| Tool"
            f" | Current {ROLLING_WINDOW_DAYS}d Brier"
            " | All-Time Brier"
            " | Δ vs All-Time"
            f" | Prev {ROLLING_WINDOW_DAYS}d Brier"
            f" | Δ vs Prev {ROLLING_WINDOW_DAYS}d |"
        ),
        "|------|----------|----------|---------------|---------|-------------|",
    ]
    cur_by_tool = rolling_scores.get("by_tool") or {}
    at_by_tool = alltime_scores.get("by_tool") or {}
    prev_by_tool = (prev_rolling_scores or {}).get("by_tool") or {}
    for tool in tools:
        c = cur_by_tool.get(tool, {})
        a = at_by_tool.get(tool, {})
        p = prev_by_tool.get(tool) if prev_rolling_scores is not None else None
        c_n = c.get("valid_n", 0) if c else 0
        a_n = a.get("valid_n", 0) if a else 0
        p_n = p.get("valid_n", 0) if p else 0
        c_brier = c.get("brier") if c else None
        a_brier = a.get("brier") if a else None
        p_brier = p.get("brier") if p else None

        flag = _sample_label(c) if c else ""
        delta_at = _delta_cell(c_brier, a_brier, c_n, a_n)
        if prev_rolling_scores is None:
            prev_val_cell = "no prev window"
            delta_prev = "no prev window"
        else:
            prev_val_cell = _value_cell(p_brier, p_n)
            delta_prev = _delta_cell(c_brier, p_brier, c_n, p_n)
        lines.append(
            f"| **{tool}**{flag}"
            f" | {_value_cell(c_brier, c_n)}"
            f" | {_value_cell(a_brier, a_n)}"
            f" | {delta_at}"
            f" | {prev_val_cell}"
            f" | {delta_prev} |"
        )
    return "\n".join(lines)


def section_tool_category_comparison(
    rolling_scores: dict[str, Any],
    alltime_scores: dict[str, Any],
    prev_rolling_scores: dict[str, Any] | None,
    active_tools: frozenset[str] | None = None,
) -> str:
    """Render the tool × category historical comparison table.

    One row per (tool, category) cell that clears ``MIN_SAMPLE_SIZE`` in
    the current window. Cells that lack current-window coverage are
    dropped from the ranking (they'd otherwise clutter the table with
    rows where the comparison is meaningless).

    :param rolling_scores: current rolling-window scores.
    :param alltime_scores: all-time scores.
    :param prev_rolling_scores: previous non-overlapping rolling scores.
    :param active_tools: when set, only cells whose tool half is in this
        set render. ``None`` disables filtering (deployment-config
        fallback).
    :return: markdown section string.
    """
    heading = "## Tool × Category Historical Comparison"
    cur = rolling_scores.get("by_tool_category") or {}
    at = alltime_scores.get("by_tool_category") or {}
    prev = (prev_rolling_scores or {}).get("by_tool_category") or {}

    ranked = _filter_by_active(
        sorted(cur.items(), key=brier_sort_key),
        active_tools,
        composite_key_separator=" | ",
    )
    sufficient = [(k, s) for k, s in ranked if s["n"] >= MIN_SAMPLE_SIZE]
    if not sufficient:
        return (
            f"{heading}\n\n"
            f"No (tool × category) cells clear n ≥ {MIN_SAMPLE_SIZE} in the "
            "current window."
        )

    lines = [
        heading,
        "",
        (
            "| Tool | Category"
            f" | Current {ROLLING_WINDOW_DAYS}d Brier"
            " | All-Time Brier"
            " | Δ vs All-Time"
            f" | Prev {ROLLING_WINDOW_DAYS}d Brier"
            f" | Δ vs Prev {ROLLING_WINDOW_DAYS}d |"
        ),
        "|------|----------|----------|----------|---------------|---------|-------------|",
    ]
    for key, c_stats in sufficient:
        parts = key.split(" | ")
        tool = parts[0] if parts else key
        cat = parts[1] if len(parts) > 1 else "?"
        a_stats = at.get(key, {})
        p_stats = prev.get(key, {}) if prev_rolling_scores is not None else None
        c_n = c_stats.get("valid_n", 0)
        a_n = a_stats.get("valid_n", 0)
        p_n = p_stats.get("valid_n", 0) if p_stats else 0
        c_brier = c_stats.get("brier")
        a_brier = a_stats.get("brier")
        p_brier = p_stats.get("brier") if p_stats else None

        delta_at = _delta_cell(c_brier, a_brier, c_n, a_n)
        if prev_rolling_scores is None:
            prev_val_cell = "no prev window"
            delta_prev = "no prev window"
        else:
            prev_val_cell = _value_cell(p_brier, p_n)
            delta_prev = _delta_cell(c_brier, p_brier, c_n, p_n)
        lines.append(
            f"| {tool} | {cat}"
            f" | {_value_cell(c_brier, c_n)}"
            f" | {_value_cell(a_brier, a_n)}"
            f" | {delta_at}"
            f" | {prev_val_cell}"
            f" | {delta_prev} |"
        )
    return "\n".join(lines)


def _diag_metric_label(metric_key: str) -> str:
    """Return the display label for a diagnostic metric key.

    :param metric_key: one of the diagnostic metric keys on a stats dict.
    :return: human-readable label.
    """
    labels = {
        "edge": "Edge",
        "log_loss": "Log Loss",
        "conditional_accuracy_rate": "Conditional Accuracy",
        "brier_large_trade": "Disagreement Brier (large trade)",
        "directional_bias": "Directional Bias",
    }
    return labels.get(metric_key, metric_key)


def section_diagnostics_comparison(
    rolling_scores: dict[str, Any],
    alltime_scores: dict[str, Any],
    prev_rolling_scores: dict[str, Any] | None,
    active_tools: frozenset[str] | None = None,
) -> str:
    """Render the diagnostics historical comparison table.

    Per tool, renders the diagnostic metrics (edge, log loss, conditional
    accuracy, disagreement Brier at large trade, directional bias) for
    the three windows with deltas. Uses the tool's per-metric
    denominator (``edge_n``, ``disagree_n``, ``n_large_trade``,
    ``n_bias_losses``) for sample-size gating so deltas aren't gated by
    the wrong ``valid_n``.

    :param rolling_scores: current rolling-window scores.
    :param alltime_scores: all-time scores.
    :param prev_rolling_scores: previous non-overlapping rolling scores.
    :param active_tools: when set, restrict the table to currently-deployed
        tools. ``None`` disables filtering (deployment-config fallback).
    :return: markdown section string.
    """
    heading = "## Diagnostics Historical Comparison"
    tools = _tool_universe(rolling_scores, alltime_scores)
    if active_tools is not None:
        tools = [t for t in tools if t in active_tools]
    if not tools:
        return f"{heading}\n\nNo tool data available."

    # For each metric, the stats dict key + the denominator key that
    # gates its sample size. Edge n lives in `edge_n`, not `valid_n`,
    # so a tool with low edge-eligible rows doesn't drag its Brier
    # delta into "insufficient data".
    metrics: list[tuple[str, str, bool]] = [
        ("edge", "edge_n", False),
        ("log_loss", "valid_n", True),
        ("conditional_accuracy_rate", "disagree_n", False),
        ("brier_large_trade", "n_large_trade", True),
        ("directional_bias", "n_bias_losses", True),
    ]

    cur_by_tool = rolling_scores.get("by_tool") or {}
    at_by_tool = alltime_scores.get("by_tool") or {}
    prev_by_tool = (prev_rolling_scores or {}).get("by_tool") or {}

    lines = [
        heading,
        "",
        (
            "| Tool | Metric"
            f" | Current {ROLLING_WINDOW_DAYS}d"
            " | All-Time"
            " | Δ vs All-Time"
            f" | Prev {ROLLING_WINDOW_DAYS}d"
            f" | Δ vs Prev {ROLLING_WINDOW_DAYS}d |"
        ),
        "|------|--------|----------|----------|---------------|---------|-------------|",
    ]
    header_len = len(lines)
    for tool in tools:
        c = cur_by_tool.get(tool, {})
        a = at_by_tool.get(tool, {})
        p = prev_by_tool.get(tool) if prev_rolling_scores is not None else None
        for metric_key, n_key, lower_is_better in metrics:
            c_val = c.get(metric_key) if c else None
            a_val = a.get(metric_key) if a else None
            p_val = p.get(metric_key) if p else None
            c_n = c.get(n_key, 0) if c else 0
            a_n = a.get(n_key, 0) if a else 0
            p_n = p.get(n_key, 0) if p else 0
            if c_val is None and a_val is None and p_val is None:
                continue

            delta_at = _delta_cell(c_val, a_val, c_n, a_n, lower_is_better)
            if prev_rolling_scores is None:
                prev_val_cell = "no prev window"
                delta_prev = "no prev window"
            else:
                prev_val_cell = _value_cell(p_val, p_n)
                delta_prev = _delta_cell(c_val, p_val, c_n, p_n, lower_is_better)
            lines.append(
                f"| **{tool}** | {_diag_metric_label(metric_key)}"
                f" | {_value_cell(c_val, c_n)}"
                f" | {_value_cell(a_val, a_n)}"
                f" | {delta_at}"
                f" | {prev_val_cell}"
                f" | {delta_prev} |"
            )
    # Pin against the header length captured above the data loop. Matching
    # a hardcoded count here broke once already (off-by-one vs the init
    # list) and would break again on any future header edit, so anchor
    # to the snapshot instead of a literal.
    if len(lines) == header_len:
        lines.append("| _(no diagnostic data)_ | | | | | | |")
    return "\n".join(lines)


def section_reliability_comparison(
    rolling_scores: dict[str, Any],
    alltime_scores: dict[str, Any],
    active_tools: frozenset[str] | None = None,
) -> str:
    """Render the reliability & parse quality comparison table.

    Per-tool reliability and parse-rate stats for the current window
    versus all-time. Prev-rolling intentionally omitted: reliability is
    a pipeline-health signal, not a tool-performance one, and comparing
    two short trailing windows against each other adds noise without
    signal.

    :param rolling_scores: current rolling-window scores.
    :param alltime_scores: all-time scores.
    :param active_tools: when set, restrict the table to currently-deployed
        tools. ``None`` disables filtering (deployment-config fallback).
    :return: markdown section string.
    """
    heading = "## Reliability & Parse Quality (Current vs All-Time)"
    tools = _tool_universe(rolling_scores, alltime_scores)
    if active_tools is not None:
        tools = [t for t in tools if t in active_tools]
    if not tools:
        return f"{heading}\n\nNo tool data available."

    cur_by_tool = rolling_scores.get("by_tool") or {}
    at_by_tool = alltime_scores.get("by_tool") or {}
    cur_parse = rolling_scores.get("parse_breakdown") or {}
    at_parse = alltime_scores.get("parse_breakdown") or {}

    def _rate(breakdown: dict[str, int], status: str) -> tuple[float | None, int]:
        total = sum(breakdown.values())
        if total == 0:
            return None, 0
        return breakdown.get(status, 0) / total, total

    lines = [
        heading,
        "",
        (
            "| Tool"
            f" | Current {ROLLING_WINDOW_DAYS}d Reliability"
            " | All-Time Reliability"
            " | Δ"
            f" | Current {ROLLING_WINDOW_DAYS}d Valid %"
            " | All-Time Valid %"
            " | Δ |"
        ),
        "|------|----------|----------|-----|----------|----------|-----|",
    ]
    for tool in tools:
        c = cur_by_tool.get(tool, {})
        a = at_by_tool.get(tool, {})
        c_n = c.get("n", 0) if c else 0
        a_n = a.get("n", 0) if a else 0
        c_rel = c.get("reliability") if c else None
        a_rel = a.get("reliability") if a else None

        c_valid_rate, c_parse_n = _rate(cur_parse.get(tool) or {}, "valid")
        a_valid_rate, a_parse_n = _rate(at_parse.get(tool) or {}, "valid")

        delta_rel = _delta_cell(c_rel, a_rel, c_n, a_n, lower_is_better=False)
        delta_valid = _delta_cell(
            c_valid_rate,
            a_valid_rate,
            c_parse_n,
            a_parse_n,
            lower_is_better=False,
        )
        lines.append(
            f"| **{tool}**"
            f" | {_pct_cell(c_rel, c_n)}"
            f" | {_pct_cell(a_rel, a_n)}"
            f" | {delta_rel}"
            f" | {_pct_cell(c_valid_rate, c_parse_n)}"
            f" | {_pct_cell(a_valid_rate, a_parse_n)}"
            f" | {delta_valid} |"
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

    lines = ["## Trend (Fleet-wide, Monthly)", ""]
    if platform is not None:
        lines.append("_Aggregated across all platforms — not scoped to this report._")
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

    Renders a single aggregate computed over all rows in the period, not a
    moving-average series. Deltas compare the period aggregate to the
    all-time aggregate.

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


def _has_scored_rows(scores: dict[str, Any] | None) -> bool:
    """Return True when a scores dict has at least one scored row.

    Used to collapse zero-row scoring output to the same "no data" signal
    a missing-file case emits, so downstream renderers don't have to
    distinguish the two for user-facing copy.

    :param scores: a parsed scores dict or ``None``.
    :return: True when the dict carries at least one row in ``total_rows``
        or its ``overall`` accumulator.
    """
    if not scores:
        return False
    if scores.get("total_rows", 0) > 0:
        return True
    overall = scores.get("overall") or {}
    return overall.get("n", 0) > 0


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


def generate_report(  # pylint: disable=too-many-statements
    scores: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    *,
    platform: str,
    rolling_scores: dict[str, Any] | None = None,
    prev_rolling_scores: dict[str, Any] | None = None,
    include_tournament: bool = False,
    scores_tournament: dict[str, Any] | None = None,
    rolling_scores_tournament: dict[str, Any] | None = None,
    disabled_tools: dict[str, list[str] | None] | None = None,
) -> str:
    """Generate a platform-scoped benchmark report from scores and history.

    Every section is driven by scores already partitioned to ``platform``
    by the scorer. Each metric is compared across three windows: current
    rolling (``rolling_scores``), cumulative (``scores``), and previous
    non-overlapping rolling (``prev_rolling_scores``). Deltas are
    suppressed when either side has n < ``MIN_SAMPLE_SIZE`` so a reader
    never sees a signed number without an adequate sample-size anchor.

    Overlapping-window comparisons (such as "since last report") are
    intentionally omitted — prev-rolling is the only change-over-time
    reference, and it is non-overlapping by construction.

    :param scores: parsed platform-scoped ``scores_<platform>.json`` dict.
    :param history: list of monthly snapshots from ``scores_history.jsonl``.
    :param platform: one of ``PLATFORM_LABELS`` keys (``"omen"`` or
        ``"polymarket"``). Drives the report header.
    :param rolling_scores: production scores from the current rolling window.
    :param prev_rolling_scores: production scores from the preceding
        non-overlapping rolling window, or ``None`` when the upstream
        scoring step has not landed one on disk.
    :param include_tournament: master switch for rendering the Tool ×
        Version × Mode breakdown. When False, tournament inputs are
        ignored entirely.
    :param scores_tournament: parsed ``scores_tournament_<platform>.json`` dict.
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

    # A zero-row prev scoring run (CI step succeeded but the window was
    # empty) lands as ``{"total_rows": 0, "overall": {}}`` on disk. Treat
    # that as "no prev window" so readers see the same explicit
    # placeholder they see when the file is missing, rather than a
    # stream of ``N/A (n=0)`` cells that silently merges the two cases.
    if prev_rolling_scores is not None and not _has_scored_rows(prev_rolling_scores):
        prev_rolling_scores = None

    # Hoist the deployment-config fetch above the section calls so the
    # comparison sections, the deployment-status section, and the
    # warning notice all see the same map. ``None`` (the default)
    # triggers a live fetch; an empty dict is the test-only opt-out.
    if disabled_tools is None:
        disabled_tools = fetch_disabled_tools()

    # Restrict the per-platform comparison sections to tools currently
    # deployed somewhere on this platform. Returns ``None`` when every
    # deployment fetch failed for this platform; the caller's fallback
    # is to skip the filter and prepend a one-line warning so the
    # reader knows the tool list is unfiltered for this run.
    active_tools = _active_tools_for_platform(
        disabled_tools, platform, scores, rolling_scores
    )

    sections: list[str] = [f"# Benchmark Report ({platform_label}) — {date}"]
    # Warning fires only when the caller actually attempted a fetch
    # (non-empty input) but every deployment for this platform failed.
    # Empty-dict callers are the unit-test opt-out and shouldn't see
    # the notice.
    if active_tools is None and disabled_tools:
        sections.append(
            "> ⚠️ Deployment config unavailable for this platform — comparison "
            "sections show every benchmarked tool, including tools that may "
            "no longer be deployed."
        )

    sections.append(section_metric_reference())

    if rolling_scores is None:
        sections.append(
            f"## Platform Snapshot (Current {ROLLING_WINDOW_DAYS}d)\n\n"
            f"Scores for the last {ROLLING_WINDOW_DAYS} days are "
            "unavailable — the scoring step did not produce "
            f"`rolling_scores_{platform}.json` for this run. All rolling "
            "sections (snapshot, platform/tool/tool×category/diagnostics "
            "comparisons, reliability) are omitted."
        )
    else:
        sections.append(section_platform_snapshot(rolling_scores))
        sections.append(
            section_platform_comparison(rolling_scores, scores, prev_rolling_scores)
        )
        sections.append(
            section_tool_comparison(
                rolling_scores, scores, prev_rolling_scores, active_tools=active_tools
            )
        )
        # Tool × Category — the reviewer asked for both the current-window
        # ranking table AND the historical comparison, so render the two
        # in sequence.
        sections.append(
            _relabel_heading(
                section_tool_category(rolling_scores, active_tools=active_tools),
                f" (Current {ROLLING_WINDOW_DAYS}d)",
            )
        )
        sections.append(
            _relabel_heading(
                section_tool_category_diagnostics(
                    rolling_scores, active_tools=active_tools
                ),
                f" (Current {ROLLING_WINDOW_DAYS}d)",
            )
        )
        sections.append(
            section_tool_category_comparison(
                rolling_scores,
                scores,
                prev_rolling_scores,
                active_tools=active_tools,
            )
        )
        sections.append(
            section_diagnostics_comparison(
                rolling_scores,
                scores,
                prev_rolling_scores,
                active_tools=active_tools,
            )
        )
        sections.append(
            section_reliability_comparison(
                rolling_scores, scores, active_tools=active_tools
            )
        )

    sections.append(
        section_tool_deployment_status(
            scores, disabled=disabled_tools, platform=platform
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
            rolling_window_note = f"last {ROLLING_WINDOW_DAYS} days"
            tvm_rolling = section_tool_version_breakdown(
                merged_rolling,
                f"Tool × Version × Mode (Last {ROLLING_WINDOW_DAYS} Days)",
            )
            if tvm_rolling:
                sections.append(_annotate_with_window(tvm_rolling, rolling_window_note))
        deltas = section_version_deltas(merged)
        if deltas:
            sections.append(deltas)

    sections.extend(
        [
            section_trend(history, None, platform=platform),
            _relabel_heading(section_sample_size_warnings(scores), " (All-Time)"),
        ]
    )

    if render_tournament:
        callouts = section_tournament_callouts(scores, scores_tournament)
        if callouts:
            sections.append(callouts)

    return "\n\n".join(sections) + "\n"


def generate_fleet_report(
    scores: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a cross-platform fleet report from the combined scores file.

    Consumes the fleet-wide ``by_platform``, ``by_category_platform`` and
    ``by_tool_category_platform`` aggregates so readers can rank tools
    and categories across platforms directly from one artifact. Does not
    duplicate the per-platform deep dives — those live in
    ``report_<platform>.md``.

    :param scores: parsed combined ``scores.json`` dict.
    :param history: list of monthly snapshots from ``scores_history.jsonl``.
    :return: full markdown report string.
    """
    if history is None:
        history = []

    date = scores.get("generated_at", "")[:10] or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d"
    )

    sections: list[str] = [
        f"# Benchmark Report (Fleet, Cross-Platform) — {date}",
        (
            "_Cross-platform view for direct category and tool × category ranking "
            "across platforms. All metrics here are all-time / cumulative — for "
            f"change-over-time and {ROLLING_WINDOW_DAYS}-day comparisons, see "
            "`report_omen.md` and `report_polymarket.md`._"
        ),
        section_metric_reference(include_scope_note=False),
        section_platform(scores),
        section_category_platform(scores),
        section_tool_category_platform(scores),
        section_trend(history, None),
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
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--platform",
        choices=sorted(PLATFORM_LABELS),
        help="Platform to scope the report to (drives default paths + header).",
    )
    mode_group.add_argument(
        "--fleet",
        action="store_true",
        help=(
            "Render a cross-platform fleet report from the combined "
            "scores.json. Uses by_category_platform and "
            "by_tool_category_platform for direct tri-dimensional ranking."
        ),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help=(
            "Override for scores file. Default: results/scores_<platform>.json "
            "(per-platform) or results/scores.json (fleet)."
        ),
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report output path. Default: results/report_<platform>.md.",
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
        "--prev-rolling",
        type=Path,
        default=None,
        help=(
            "Previous non-overlapping rolling scores JSON. Default: "
            "results/prev_rolling_scores_<platform>.json."
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
    results_dir = DEFAULT_RESULTS_DIR
    history = load_history(args.history)

    if args.fleet:
        scores_path = args.scores or results_dir / "scores.json"
        output_path = args.output or results_dir / "report_fleet.md"
        scores = load_scores(scores_path)
        print(
            f"Loaded fleet scores ({scores.get('total_rows', 0)} rows), "
            f"{len(history)} months of history"
        )
        report = generate_fleet_report(scores, history)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Report written to {output_path}")
        print(f"\n{report}")
        return

    platform = args.platform
    scores_path = args.scores or results_dir / f"scores_{platform}.json"
    output_path = args.output or results_dir / f"report_{platform}.md"
    rolling_path = args.rolling or results_dir / f"rolling_scores_{platform}.json"
    prev_rolling_path = (
        args.prev_rolling or results_dir / f"prev_rolling_scores_{platform}.json"
    )
    scores_tournament_path = (
        args.scores_tournament or results_dir / f"scores_tournament_{platform}.json"
    )
    rolling_tournament_path = (
        args.rolling_tournament
        or results_dir / f"rolling_scores_tournament_{platform}.json"
    )

    def _maybe_load(path: Path | None) -> dict[str, Any] | None:
        return load_scores(path) if path and path.exists() else None

    scores = load_scores(scores_path)
    rolling = _maybe_load(rolling_path)
    prev_rolling = _maybe_load(prev_rolling_path)
    scores_tournament = _maybe_load(scores_tournament_path)
    rolling_tournament = _maybe_load(rolling_tournament_path)

    print(
        f"Loaded scores ({scores.get('total_rows', 0)} rows) for "
        f"{PLATFORM_LABELS[platform]}, {len(history)} months of history"
    )

    report = generate_report(
        scores,
        history,
        platform=platform,
        rolling_scores=rolling,
        prev_rolling_scores=prev_rolling,
        include_tournament=args.include_tournament,
        scores_tournament=scores_tournament,
        rolling_scores_tournament=rolling_tournament,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written to {output_path}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
