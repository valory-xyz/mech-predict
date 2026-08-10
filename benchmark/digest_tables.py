# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""Build the benchmark Slack digest as computed tables.

Every cell here is read from a scorer artifact or is a pure function of
fields in one. No value passes through a language model. That is the point:
the previous digest was an LLM rendering of the markdown report, which meant a
mislabelled column ("BSS vs mkt" carrying a base-rate skill score) propagated
untouched for weeks and rows were dropped often enough that the prompt grew six
defensive sentences telling the model not to drop them. A prompt cannot be
unit-tested; this module can.

Inputs, all written by ``benchmark.scorer`` into the same results directory:

===========================================  ==============================
file                                          window
===========================================  ==============================
``scores_<platform>.json``                    all-time (since the cutoff)
``rolling_scores_<platform>.json``            W-1, the last 7 days
``prev_rolling_scores_<platform>.json``       W-2, the 7 days before that
``scores_tournament_<platform>.json``         tournament, all-time only
``roi_results.json``                          90-day ROI simulation
===========================================  ==============================

Tournament rows have no W-1/W-2 because rolling scorer invocations pass
``--skip-tournament-output``; those cells render the literal ``n/a`` rather
than a fabricated or back-filled number.

Two baselines, both shown, neither as a ratio
---------------------------------------------
``base`` is the no-skill floor (``yes_rate * (1 - yes_rate)``) and ``mkt`` is
the market's own Brier on the same markets. Both are printed in Brier units
beside the tool's Brier so the reader subtracts by eye. A skill-score ratio is
deliberately NOT rendered: its denominator sits off-screen, which is precisely
how a wrong denominator went unnoticed.

One table per Slack message
---------------------------
Slack renders roughly 3000 characters per message and splits longer text
mid-block, which breaks the code fence and destroys the alignment the whole
table depends on. :func:`build_digest_messages` therefore returns a LIST of
messages, each carrying at most one fenced table, for the caller to post in
order. This is the same reason the ROI companion is already posted as its own
message rather than appended to the digest.

Pure builder, stdlib only: no network, no LLM, no clock. A broken digest must
never break the daily post, so the entry point returns an empty list instead of
raising when its inputs are missing or unparseable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmark.roi_sim import RELIABILITY_GATE
from benchmark.scoring_primitives import MIN_SAMPLE_SIZE
from benchmark.slack_tables import Column, code_block, render_table

log = logging.getLogger(__name__)

NA = "n/a"

# Marks a window whose sample is too small to support a keep/demote call. The
# marker is ASCII on purpose: table cells stay ASCII so column padding cannot
# skew, and glyphs are reserved for prose outside the code block.
LOW_SAMPLE_MARK = "*"

# A delta between two windows is only rendered when BOTH sides clear the
# sample floor. Rendering a delta off a 19-row window invites exactly the
# false alarms this redesign exists to remove.
INSUFFICIENT = "insufficient"

WINDOW_FILES = {
    "at": "scores_{platform}.json",
    "w1": "rolling_scores_{platform}.json",
    "w2": "prev_rolling_scores_{platform}.json",
    "tournament": "scores_tournament_{platform}.json",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_by_tool(path: Path) -> dict[str, dict[str, Any]]:
    """Read one scores file and return its ``by_tool`` mapping.

    Missing or unparseable files yield an empty mapping: a window that did not
    run renders ``n/a`` cells rather than failing the whole post.

    :param path: path to a scores json file.
    :return: mapping of tool name to its stats dict; empty when unavailable.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.info("digest: no usable scores at %s (%s)", path, exc)
        return {}
    by_tool = payload.get("by_tool")
    if not isinstance(by_tool, dict):
        log.warning("digest: %s has no by_tool mapping", path)
        return {}
    return {str(k): v for k, v in by_tool.items() if isinstance(v, dict)}


def _load_roi(path: Path, platform: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Index the ROI simulation results by (tool name, mode) for this platform.

    :param path: path to roi_results.json.
    :param platform: platform key, e.g. "polymarket".
    :return: mapping of (tool_name, mode) to the ROI group dict.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.info("digest: no usable ROI results at %s (%s)", path, exc)
        return {}
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return {}
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict) or group.get("platform") != platform:
            continue
        key = (str(group.get("tool_name") or ""), str(group.get("mode") or ""))
        indexed[key] = group
    return indexed


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    """Check whether *value* is a real number (bools excluded).

    :param value: candidate value.
    :return: True when value is an int or float and not a bool.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _score(value: object) -> str:
    """Format a Brier-unit score to 4 decimals.

    :param value: candidate value.
    :return: formatted score, or ``n/a``.
    """
    return f"{value:.4f}" if _is_number(value) else NA


def _signed(value: object) -> str:
    """Format a signed Brier-unit score to 4 decimals.

    :param value: candidate value.
    :return: formatted score with explicit sign, or ``n/a``.
    """
    return f"{value:+.4f}" if _is_number(value) else NA


def _count(stats: dict[str, Any] | None) -> str:
    """Format a window's scored-row count, marking under-floor samples.

    :param stats: per-tool stats dict for that window, or None.
    :return: count cell, e.g. ``301`` or ``19 *``, or ``n/a``.
    """
    if not stats or not _is_number(stats.get("valid_n")):
        return NA
    valid_n = int(stats["valid_n"])
    mark = f" {LOW_SAMPLE_MARK}" if valid_n < MIN_SAMPLE_SIZE else ""
    return f"{valid_n}{mark}"


def _reliability(stats: dict[str, Any] | None) -> str:
    """Format a window's reliability as a percentage.

    Reliability is ``valid_n / n`` -- the share of attempts that produced a
    usable prediction. It qualifies the Brier printed beside it, because Brier
    is averaged over the valid rows only: a tool at 94% has its score computed
    on 94% of its attempts, and the missing rows are not a random sample.

    :param stats: per-tool stats dict for that window, or None.
    :return: percentage cell, e.g. ``97%``, or ``n/a``.
    """
    if not stats or not _is_number(stats.get("reliability")):
        return NA
    return f"{stats['reliability'] * 100:.0f}%"


def _has_floor(stats: dict[str, Any] | None) -> bool:
    """Check whether a window has enough scored rows to support a delta.

    :param stats: per-tool stats dict for that window, or None.
    :return: True when the window cleared MIN_SAMPLE_SIZE.
    """
    return bool(
        stats
        and _is_number(stats.get("valid_n"))
        and int(stats["valid_n"]) >= MIN_SAMPLE_SIZE
    )


def _delta(
    current: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    field: str,
    lower_is_better: bool,
) -> str:
    """Format the current-vs-reference movement of one metric.

    The direction word is spelled out because the sign alone is ambiguous
    across columns: a positive Brier delta is a regression while a positive
    Edge delta is an improvement, and readers were mixing them up.

    :param current: stats for the current window (W-1).
    :param reference: stats for the comparison window (W-2 or all-time).
    :param field: metric key present in both stats dicts.
    :param lower_is_better: True for Brier, False for Edge.
    :return: e.g. ``+0.0417 worse``, ``insufficient``, or ``n/a``.
    """
    if not current or not reference:
        return NA
    # isinstance rather than _is_number: the narrowing has to be visible to
    # the type checker for the subtraction below.
    a = current.get(field)
    b = reference.get(field)
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        return NA
    if isinstance(b, bool) or not isinstance(b, (int, float)):
        return NA
    if not (_has_floor(current) and _has_floor(reference)):
        return INSUFFICIENT
    diff = a - b
    improved = diff < 0 if lower_is_better else diff > 0
    return f"{diff:+.4f} {'better' if improved else 'worse'}"


def _roi(group: dict[str, Any] | None) -> str:
    """Format the simulated 90-day ROI midpoint.

    ``roi_sim`` already emits ``roi_mid`` in percent (5.94 means +5.9%), so
    it is formatted, never rescaled.

    :param group: ROI group dict for this (tool, mode), or None.
    :return: percentage cell, e.g. ``+15.3%``, or ``n/a``.
    """
    if not group or not _is_number(group.get("roi_mid")):
        return NA
    return f"{float(group['roi_mid']):+.1f}%"


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

_W2_COLUMNS = (
    Column("tool"),
    Column("n W-1"),
    Column("n W-2"),
    Column("rel W-1"),
    Column("Brier W-1"),
    Column("Brier W-2"),
    Column("Δ Brier"),
    Column("mkt W-1"),
    Column("mkt W-2"),
    Column("Edge W-1"),
    Column("Edge W-2"),
    Column("Δ Edge"),
    Column("ROI 90d"),
)

_AT_COLUMNS = (
    Column("tool"),
    Column("n W-1"),
    Column("n AT"),
    Column("rel W-1"),
    Column("Brier W-1"),
    Column("Brier AT"),
    Column("Δ Brier"),
    Column("base AT"),
    Column("mkt W-1"),
    Column("mkt AT"),
    Column("Edge W-1"),
    Column("Edge AT"),
    Column("Δ Edge"),
    Column("ROI 90d"),
)


def _sort_key(entry: tuple[str, dict[str, Any] | None, dict[str, Any] | None]) -> tuple:
    """Order rows by the decision metric, best first, unscored last.

    :param entry: (tool name, current stats, reference stats).
    :return: sort key placing higher Edge first and None-Edge rows last.
    """
    _, current, reference = entry
    for stats in (current, reference):
        if stats and _is_number(stats.get("edge")):
            return (0, -float(stats["edge"]))
    return (1, 0.0)


def _rows_w2(
    tools: Sequence[str],
    w1: dict[str, dict[str, Any]],
    w2: dict[str, dict[str, Any]],
    roi: dict[tuple[str, str], dict[str, Any]],
    mode: str,
) -> list[tuple[str, ...]]:
    """Build the W-1 vs W-2 table rows.

    :param tools: tool names to render, in the order given.
    :param w1: by_tool stats for the last 7 days.
    :param w2: by_tool stats for the preceding 7 days.
    :param roi: ROI groups indexed by (tool, mode).
    :param mode: "production" or "tournament", for the ROI lookup.
    :return: one cell tuple per tool.
    """
    rows: list[tuple[str, ...]] = []
    for tool in tools:
        a, b = w1.get(tool), w2.get(tool)
        rows.append(
            (
                tool,
                _count(a),
                _count(b),
                _reliability(a),
                _score((a or {}).get("brier")),
                _score((b or {}).get("brier")),
                _delta(a, b, "brier", lower_is_better=True),
                _score((a or {}).get("market_brier")),
                _score((b or {}).get("market_brier")),
                _signed((a or {}).get("edge")),
                _signed((b or {}).get("edge")),
                _delta(a, b, "edge", lower_is_better=False),
                _roi(roi.get((tool, mode))),
            )
        )
    return rows


def _rows_at(
    tools: Sequence[str],
    w1: dict[str, dict[str, Any]],
    at: dict[str, dict[str, Any]],
    roi: dict[tuple[str, str], dict[str, Any]],
    mode: str,
) -> list[tuple[str, ...]]:
    """Build the W-1 vs all-time table rows.

    :param tools: tool names to render, in the order given.
    :param w1: by_tool stats for the last 7 days (empty for tournament).
    :param at: by_tool stats for the all-time window.
    :param roi: ROI groups indexed by (tool, mode).
    :param mode: "production" or "tournament", for the ROI lookup.
    :return: one cell tuple per tool.
    """
    rows: list[tuple[str, ...]] = []
    for tool in tools:
        a, b = w1.get(tool), at.get(tool)
        rows.append(
            (
                tool,
                _count(a),
                _count(b),
                _reliability(a),
                _score((a or {}).get("brier")),
                _score((b or {}).get("brier")),
                _delta(a, b, "brier", lower_is_better=True),
                _score((b or {}).get("baseline_brier")),
                _score((a or {}).get("market_brier")),
                _score((b or {}).get("market_brier")),
                _signed((a or {}).get("edge")),
                _signed((b or {}).get("edge")),
                _delta(a, b, "edge", lower_is_better=False),
                _roi(roi.get((tool, mode))),
            )
        )
    return rows


def _scored(by_tool: dict[str, dict[str, Any]]) -> set[str]:
    """Return tools that produced a Brier in this window.

    :param by_tool: by_tool mapping from a scores file.
    :return: set of tool names with a numeric Brier.
    """
    return {t for t, s in by_tool.items() if _is_number(s.get("brier"))}


def _ordered(
    names: Iterable[str],
    current: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
) -> list[str]:
    """Order tool names by the decision metric.

    :param names: tool names to order.
    :param current: stats for the current window.
    :param reference: stats for the reference window.
    :return: tool names, best Edge first.
    """
    entries = [(n, current.get(n), reference.get(n)) for n in names]
    return [name for name, _, _ in sorted(entries, key=_sort_key)]


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

_ALERT_COLUMNS = (
    Column("what"),
    Column("tool"),
    Column("evidence"),
    Column("action"),
)


def _alert_rows(
    w1: dict[str, dict[str, Any]],
    at: dict[str, dict[str, Any]],
    tools: Sequence[str],
) -> list[tuple[str, ...]]:
    """Build the alerts table: conditions that block a call, not calls.

    Three conditions, all computed:

    * sample starved -- W-1 has fewer than MIN_SAMPLE_SIZE scored rows, so no
      keep/demote conclusion is supportable for that tool this week;
    * reliability breach -- W-1 reliability fell below RELIABILITY_GATE, which
      also excludes the tool from the ROI simulation;
    * platform below no-skill -- every tool's all-time Brier exceeds its own
      base-rate floor, which is a platform problem no tool swap addresses.

    Reliability is deliberately NOT alerted on a bare week-over-week delta: at
    the sample sizes involved a two-point drop is one or two extra failed rows,
    and the previous digest raised three such "worse" bullets in one message.

    :param w1: by_tool stats for the last 7 days.
    :param at: by_tool stats for the all-time window.
    :param tools: tool names under consideration.
    :return: alert rows, empty when nothing fires.
    """
    rows: list[tuple[str, ...]] = []

    starved = [t for t in tools if w1.get(t) and not _has_floor(w1[t])]
    for tool in starved:
        rows.append(
            (
                "sample starved",
                tool,
                f"{int(w1[tool]['valid_n'])} scored rows in W-1, "
                f"floor is {MIN_SAMPLE_SIZE}",
                "widen the window before any keep/demote call",
            )
        )

    for tool in tools:
        stats = w1.get(tool)
        if not stats or not _is_number(stats.get("reliability")):
            continue
        if stats["reliability"] < RELIABILITY_GATE:
            rows.append(
                (
                    "reliability breach",
                    tool,
                    f"{_reliability(stats)} in W-1, gate is " f"{RELIABILITY_GATE:.0%}",
                    "tool is excluded from ROI; investigate parse failures",
                )
            )

    scored = [at[t] for t in tools if at.get(t) and _is_number(at[t].get("edge"))]
    if scored and all(s["edge"] < 0 for s in scored):
        worst = max(s["edge"] for s in scored)
        rows.append(
            (
                "platform below no-skill",
                f"ALL ({len(scored)} tools)",
                f"every all-time Edge < 0; best is {worst:+.4f}",
                "upstream calibration, not a tool swap",
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_digest_messages(
    results_dir: Path,
    platform: str,
    roi_results: Path | None = None,
) -> list[str]:
    """Build the benchmark digest as one Slack message per table.

    Each returned message carries at most one fenced code block so Slack's
    per-message length limit can never split a table and break its fence.

    :param results_dir: directory holding the scorer output files.
    :param platform: platform key, e.g. "polymarket" or "omen".
    :param roi_results: path to roi_results.json; defaults to
        ``results_dir/roi_results.json``.
    :return: ordered Slack mrkdwn messages; empty when there is nothing to post.
    """
    windows = {
        key: _load_by_tool(results_dir / name.format(platform=platform))
        for key, name in WINDOW_FILES.items()
    }
    roi = _load_roi(roi_results or results_dir / "roi_results.json", platform)

    prod_tools = _scored(windows["at"]) | _scored(windows["w1"])
    tourn_tools = _scored(windows["tournament"])
    if not prod_tools and not tourn_tools:
        log.warning("digest: no scored tools for %s; skipping tables", platform)
        return []

    messages: list[str] = []

    if prod_tools:
        order_w2 = _ordered(prod_tools, windows["w1"], windows["w2"])
        order_at = _ordered(prod_tools, windows["w1"], windows["at"])
        messages.append(
            "*1a. PRODUCTION - W-1 vs W-2* (what changed this week)\n"
            + code_block(
                render_table(
                    _W2_COLUMNS,
                    _rows_w2(order_w2, windows["w1"], windows["w2"], roi, "production"),
                )
            )
        )
        messages.append(
            "*1b. PRODUCTION - W-1 vs AT* (this week against the longer baseline)\n"
            + code_block(
                render_table(
                    _AT_COLUMNS,
                    _rows_at(order_at, windows["w1"], windows["at"], roi, "production"),
                )
            )
        )

    if tourn_tools:
        order = _ordered(tourn_tools, {}, windows["tournament"])
        messages.append(
            "*2. TOURNAMENT - all-time pool* (no weekly comparison available)\n"
            + code_block(
                render_table(
                    _AT_COLUMNS,
                    _rows_at(order, {}, windows["tournament"], roi, "tournament"),
                )
            )
            + "\n>Tournament rows carry no W-1/W-2: rolling scorer runs pass "
            "`--skip-tournament-output`, so no weekly aggregate is written for "
            "a candidate. Its `AT` is also a different span than production's."
        )

    alerts = _alert_rows(windows["w1"], windows["at"], sorted(prod_tools))
    if alerts:
        messages.append(
            "*3. Alerts*\n" + code_block(render_table(_ALERT_COLUMNS, alerts))
        )

    return messages
