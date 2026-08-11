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
``scores_<platform>.json``                    cumulative; span is NOT fixed
``rolling_scores_<platform>.json``            W-1, the last 7 days
``prev_rolling_scores_<platform>.json``       W-2, the 7 days before that
``scores_tournament_<platform>.json``         tournament, all-time only
``roi_results.json``                          90-day ROI simulation
===========================================  ==============================

``scores_<platform>.json`` is a cumulative accumulator whose span is neither
all-time nor a calendar month, and its own ``current_month`` key does not
describe it: on the live artifact that key read ``2026-08`` while the file held
every row back to ``2026-07-01``, because the rollover snapshots history without
fully resetting the live file. A ``--rebuild`` writes only the newest month, so
the same filename can mean either span. The columns are therefore labelled
``cum`` and the section line prints the observed ``window_start..window_end``
rather than asserting a window nobody recorded.

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
Each message carries one native Block Kit ``table`` block, so Slack lays the
columns out itself -- nothing here pads or measures a cell. Messages stay
separate anyway: it keeps each table under Slack's per-message cell-character
cap and lets a reader thread a reply to one table rather than to all of them.

Pure builder, stdlib only: no network, no LLM, no clock. A broken digest must
never break the daily post, so the entry point returns an empty list instead of
raising when its inputs are missing or unparseable.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Collection, Iterable, Sequence

from benchmark.roi_sim import RELIABILITY_GATE
from benchmark.scoring_primitives import MIN_SAMPLE_SIZE
from benchmark.slack_blocks import Col, context, message, section, table_block

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


def _load_window(path: Path) -> str | None:
    """Describe the span the accumulator actually covers.

    Do NOT trust ``current_month`` for this. On the live artifact that field
    read ``2026-08`` while the file held every row back to ``2026-07-01`` --
    the month rollover snapshots history without fully resetting the live
    accumulator, so the stamp names the month it was last written in, not the
    window it covers. A ``--rebuild`` writes only the newest month, so the same
    filename means different spans depending on which path last touched it.
    ``window_start``/``window_end`` are the observed bounds and are the only
    honest source; without them the label says so rather than inventing one.

    :param path: path to a scores json file.
    :return: e.g. "2026-07-01..2026-08-11", or None when unavailable.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    start, end = payload.get("window_start"), payload.get("window_end")
    if not start or not end:
        return None
    return f"{str(start)[:10]}..{str(end)[:10]}"


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
    # roi_sim groups by (platform, tool, mode, MODEL), so several groups share
    # a (tool, mode) key -- 14 of them in a live file. Last-wins would drop a
    # real number: omen factual_research has a gpt-4.1 group at +15.8% plus two
    # model-less groups whose roi_mid is None, and the None one sorts last.
    # Keep the best-evidenced group instead: a real roi_mid beats None, then
    # the larger bet count.
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict) or group.get("platform") != platform:
            continue
        key = (str(group.get("tool_name") or ""), str(group.get("mode") or ""))
        incumbent = indexed.get(key)
        if incumbent is None or _roi_rank(group) > _roi_rank(incumbent):
            indexed[key] = group
    return indexed


def _roi_rank(group: dict[str, Any]) -> tuple[int, int]:
    """Rank competing ROI groups that share a (tool, mode) key.

    :param group: ROI group dict.
    :return: sort key; higher wins. A usable roi_mid dominates bet count.
    """
    has_roi = 1 if _is_number(group.get("roi_mid")) else 0
    n_bets = group.get("n_bets")
    # isinstance rather than _is_number: the narrowing has to be visible to
    # the type checker for the int() call.
    bets = int(n_bets) if isinstance(n_bets, (int, float)) else 0
    return (has_roi, bets)


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    """Check whether *value* is a real number (bools excluded).

    :param value: candidate value.
    :return: True when value is an int or float and not a bool.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num(value: object) -> float | None:
    """Return *value* as a float when it is a real number, else None.

    A single narrowing helper: ``_is_number`` answers the question but the type
    checker cannot see through it, so every caller was repeating an isinstance
    guard to satisfy mypy.

    :param value: candidate value.
    :return: the value as a float, or None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


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

    Renders ``valid_n/edge_n`` when the two pools differ. Brier averages over
    every valid row; Edge and mkt average over the subset that carried a market
    price. Printing only ``valid_n`` let an Edge computed on 11 rows sit
    unstarred beside ``n = 40``, so the marker keys off whichever pool is
    smaller.

    :param stats: per-tool stats dict for that window, or None.
    :return: count cell, e.g. ``301``, ``40/11`` or ``19 *``, or ``n/a``.
    """
    valid = _num((stats or {}).get("valid_n"))
    if valid is None:
        return NA
    valid_n = int(valid)
    edge = _num((stats or {}).get("edge_n"))
    edge_n = None if edge is None else int(edge)
    both = (
        f"{valid_n}/{edge_n}"
        if edge_n is not None and edge_n != valid_n
        else str(valid_n)
    )
    floor = min(valid_n, edge_n) if edge_n is not None else valid_n
    mark = f" {LOW_SAMPLE_MARK}" if floor < MIN_SAMPLE_SIZE else ""
    return f"{both}{mark}"


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


def _has_floor(stats: dict[str, Any] | None, field: str = "valid_n") -> bool:
    """Check whether a window has enough rows to support a delta.

    Takes the count field so an Edge delta is floored on ``edge_n`` while a
    Brier delta stays on ``valid_n`` -- the two average over different pools,
    and gating both on the larger one suppresses nothing.

    :param stats: per-tool stats dict for that window, or None.
    :param field: the count to gate on.
    :return: True when the window cleared MIN_SAMPLE_SIZE.
    """
    count = _num((stats or {}).get(field))
    return count is not None and int(count) >= MIN_SAMPLE_SIZE


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
    # Edge and mkt live on the edge-eligible pool; everything else on valid_n.
    count_field = "edge_n" if field in ("edge", "market_brier") else "valid_n"
    if not (_has_floor(current, count_field) and _has_floor(reference, count_field)):
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


def _roi_haircut(group: dict[str, Any] | None) -> str:
    """Format the simulated 90-day ROI net of execution costs.

    Point estimate only -- ``roi_sim`` publishes no CI for the haircut figure.
    Already in percent, like ``roi_mid``, so it is formatted, never rescaled.

    :param group: ROI group dict for this (tool, mode), or None.
    :return: percentage cell, e.g. ``+14.5%``, or ``n/a``.
    """
    if not group or not _is_number(group.get("roi_haircut")):
        return NA
    return f"{float(group['roi_haircut']):+.1f}%"


# Promote margin and floor from PROMOTE_DEMOTE_POLICY.md: promote only when the
# one-sided 95% lower bound on the paired edge clears delta, with n >= 30.
PROMOTE_DELTA = 0.04
# A Brier this far above its own base-rate floor is a signal; less is noise.
NO_SKILL_MARGIN = 0.01
# One-sided 95%: the normal-approximation z. The policy specifies a bootstrap;
# at n >= 30 on a mean this is the same call to well inside the margin, and it
# is computable from the stored sum/sum-of-squares without per-row data.
Z_ONE_SIDED_95 = 1.645

# A tool that wins fewer than half the markets where it DISAGREED with the price
# has no tradable edge, whatever its headline accuracy looks like.
COIN_FLIP = 0.50


def _edge_lower_bound(stats: dict[str, Any] | None) -> float | None:
    """One-sided 95% lower bound on the tool's mean edge over the market.

    :param stats: per-tool stats for a window.
    :return: the lower bound, or None when it cannot be computed.
    """
    edge = _num((stats or {}).get("edge"))
    sd = _num((stats or {}).get("edge_sd"))
    count = _num((stats or {}).get("edge_n"))
    if edge is None or sd is None or count is None or count < 2:
        return None
    return edge - Z_ONE_SIDED_95 * sd / math.sqrt(count)


def _below_no_skill(stats: dict[str, Any] | None) -> bool:
    """Is this window's Brier materially worse than its own base-rate floor?

    "Materially" matters: a live tool sat 0.0003 above its floor on 2,055
    markets while winning 65% of the 1,968 markets where it disagreed with the
    price. A hairline gap is noise, and the policy is explicit that a demote
    needs a sustained signal rather than a one-off reading.

    :param stats: per-tool stats for one window.
    :return: True when the tool is below no-skill by more than the margin.
    """
    brier = _num((stats or {}).get("brier"))
    base = _num((stats or {}).get("baseline_brier"))
    return brier is not None and base is not None and brier - base >= NO_SKILL_MARGIN


def _verdict(
    stats: dict[str, Any] | None,
    deployed: bool,
    recent: dict[str, Any] | None = None,
) -> str:
    """Apply the promote/demote gate to one tool.

    The rule is the policy's, stated once and used for both rosters:
    promote only when the lower bound on the edge clears the margin; demote on
    a sustained below-no-skill signal; otherwise keep. Anything the sample
    cannot support says so rather than guessing -- "not enough data yet" is a
    different answer from "no improvement", and the policy is explicit that
    conflating them is the failure to avoid.

    :param stats: per-tool stats for the decision window.
    :param deployed: True for a production tool, False for a candidate.
    :param recent: stats for the most recent week, used to confirm that a
        below-no-skill reading is sustained rather than a single window.
    :return: a short verdict string for the ``rec`` cell.
    """
    brier = _num((stats or {}).get("brier"))
    if brier is None:
        return "no data"
    edge_n = _num((stats or {}).get("edge_n"))
    if edge_n is None or edge_n < MIN_SAMPLE_SIZE:
        shown = NA if edge_n is None else int(edge_n)
        return f"insufficient (n={shown} < {MIN_SAMPLE_SIZE})"

    lower = _edge_lower_bound(stats)
    if lower is None:
        return "insufficient (no spread)"

    conditional = _num((stats or {}).get("conditional_accuracy_rate"))
    if conditional is not None and conditional < COIN_FLIP:
        loses = f"loses its disagreements ({conditional:.0%})"
        return f"demote: {loses}" if deployed else f"no: {loses}"

    if lower > PROMOTE_DELTA:
        return "keep (clears margin)" if deployed else "PROMOTE"

    # Sustained, not a one-off: both the month and the latest week must agree.
    if _below_no_skill(stats) and (recent is None or _below_no_skill(recent)):
        return "demote: below no-skill" if deployed else "no: below no-skill"

    return "keep" if deployed else f"no: {PROMOTE_DELTA - lower:+.3f} short"


def _guard_last_tools(verdicts: dict[str, str]) -> dict[str, str]:
    """Never let the report recommend emptying a platform.

    A demote means stop serving that tool. If EVERY deployed tool on a platform
    demotes, acting on all of them leaves the platform with no forecaster at
    all, which is worse than keeping the least-bad one. The policy anticipates
    this case and marks it rather than hiding it, so the verdicts survive and
    the consequence is stated beside them.

    :param verdicts: mapping of tool name to verdict, for ONE platform cohort.
    :return: the same mapping, annotated when every tool would be demoted.
    """
    demotes = [t for t, v in verdicts.items() if v.startswith("demote")]
    if not verdicts or len(demotes) < len(verdicts):
        return verdicts
    return {
        tool: f"{verdict} - NO REPLACEMENT, do not act alone"
        for tool, verdict in verdicts.items()
    }


def _headline(
    prod: dict[str, str], tourn: dict[str, str], platform: str
) -> dict[str, Any]:
    """Build the one-message answer: what to promote, what to demote.

    The tables carry the evidence; this carries the decision. A reader who
    stops after the first message must still leave with the right action, and
    with an honest count of how many tools the data could not judge.

    :param prod: verdicts for the deployed tools.
    :param tourn: verdicts for the candidates.
    :param platform: platform key, for the heading.
    :return: a webhook payload.
    """
    promote = sorted(t for t, v in tourn.items() if v == "PROMOTE")
    demote = sorted(t for t, v in prod.items() if v.startswith("demote"))
    unjudged = sorted(
        t for t, v in {**prod, **tourn}.items() if v.startswith("insufficient")
    )
    blocked = any("NO REPLACEMENT" in v for v in prod.values())

    if promote:
        verdict = f"PROMOTE {len(promote)}: " + ", ".join(f"`{t}`" for t in promote)
    elif demote and not blocked:
        verdict = f"DEMOTE {len(demote)}: " + ", ".join(f"`{t}`" for t in demote)
    elif demote:
        verdict = (
            f"NO ACTION - all {len(demote)} deployed tools fail the gate, so "
            "demoting them would leave this platform with no forecaster. "
            "Treat as a platform-level problem, not a tool swap."
        )
    else:
        verdict = "NO CHANGE - no candidate clears the promote margin."

    lines = [f"*{verdict}*"]
    if unjudged:
        lines.append(
            f"_{len(unjudged)} tool(s) had too little data to judge: "
            + ", ".join(f"`{t}`" for t in unjudged)
            + "._"
        )
    return message(
        f"Verdict ({platform})",
        [
            section(f"*TOOL VERDICT - {platform.upper()}*\n" + "\n".join(lines)),
            context(
                "Gate: promote needs the one-sided 95% lower bound on edge vs "
                f"the market to clear +{PROMOTE_DELTA} with n >= "
                f"{MIN_SAMPLE_SIZE}. A tool winning under half the markets where "
                "it disagreed with the price cannot be promoted whatever its "
                "headline accuracy says. Evidence in the tables below."
            ),
        ],
    )


def _conditional(stats: dict[str, Any] | None) -> str:
    """Format the win rate on markets where the tool disagreed with the price.

    :param stats: per-tool stats for that window.
    :return: e.g. ``64%``, or ``n/a``.
    """
    rate = _num((stats or {}).get("conditional_accuracy_rate"))
    return NA if rate is None else f"{rate * 100:.0f}%"


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

_W2_COLUMNS = (
    Col("tool"),
    Col("n W-1", align="right"),
    Col("n W-2", align="right"),
    Col("rel W-1", align="right"),
    Col("Brier W-1", align="right"),
    Col("Brier W-2", align="right"),
    Col("Δ Brier", align="right"),
    Col("mkt W-1", align="right"),
    Col("mkt W-2", align="right"),
    Col("Edge W-1", align="right"),
    Col("Edge W-2", align="right"),
    Col("Δ Edge", align="right"),
    Col("ROI 90d", align="right"),
    Col("w/costs", align="right"),
)

_AT_COLUMNS = (
    Col("tool"),
    Col("n W-1", align="right"),
    Col("n cum", align="right"),
    Col("rel W-1", align="right"),
    Col("Brier W-1", align="right"),
    Col("Brier cum", align="right"),
    Col("Δ Brier", align="right"),
    Col("base cum", align="right"),
    Col("mkt W-1", align="right"),
    Col("mkt cum", align="right"),
    Col("Edge W-1", align="right"),
    Col("Edge cum", align="right"),
    Col("Δ Edge", align="right"),
    Col("condAcc", align="right"),
    Col("ROI 90d", align="right"),
    Col("w/costs", align="right"),
    Col("rec", wrap=True),
)


def _sort_key(
    entry: tuple[str, dict[str, Any] | None, dict[str, Any] | None],
) -> tuple[int, float, str]:
    """Order rows by the decision metric, best first, unscored last.

    :param entry: (tool name, current stats, reference stats).
    :return: sort key placing higher Edge first and None-Edge rows last.
    """
    name, current, reference = entry
    for stats in (current, reference):
        if stats and _is_number(stats.get("edge")):
            return (0, -float(stats["edge"]), name)
    # Tool name breaks the tie: without it, Edge-less rows come out in set
    # iteration order, which differs between processes and makes the posted
    # table undiffable day over day. _ordered() also sorts its input, so this
    # is belt-and-braces -- deliberately. Removing EITHER is safe; removing
    # BOTH reintroduces the bug, which is what TestOrderStability asserts.
    return (1, 0.0, name)


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
                _roi_haircut(roi.get((tool, mode))),
            )
        )
    return rows


def _verdicts_for(
    tools: Sequence[str],
    at: dict[str, dict[str, Any]],
    deployed: bool,
    w1: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Apply the gate to a cohort, with the no-replacement guard.

    :param tools: tool names in the cohort.
    :param at: by_tool stats for the decision window.
    :param deployed: True for production tools.
    :param w1: last week's stats, used to confirm a sustained demote signal.
    :return: mapping of tool name to verdict.
    """
    verdicts = {
        t: _verdict(at.get(t), deployed=deployed, recent=(w1 or {}).get(t))
        for t in tools
    }
    return _guard_last_tools(verdicts) if deployed else verdicts


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
    deployed = mode == "production"
    verdicts = _verdicts_for(tools, at, deployed, w1)

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
                _conditional(b),
                _roi(roi.get((tool, mode))),
                _roi_haircut(roi.get((tool, mode))),
                verdicts[tool],
            )
        )
    return rows


def _scored(by_tool: dict[str, dict[str, Any]]) -> set[str]:
    """Return tools that produced a Brier in this window.

    :param by_tool: by_tool mapping from a scores file.
    :return: set of tool names with a numeric Brier.
    """
    return {t for t, s in by_tool.items() if _is_number(s.get("brier"))}


def _ran_but_unscored(
    by_tool: dict[str, dict[str, Any]], permitted: Collection[str]
) -> set[str]:
    """Return PREDICTION tools that ran but produced no usable prediction.

    A prediction tool at 100% parse failure has rows but no Brier, and
    selecting on Brier alone would hide exactly the tool most in need of
    attention. The allowlist is what makes this safe: the benchmark also sees
    tools that are not forecasters at all -- a question-proposing tool, for
    instance, never emits a ``p_yes``, so ``valid_n=0`` is its correct and
    unremarkable reading, not a failure. Admitting those would manufacture an
    alarm about a tool this report has no opinion on.

    :param by_tool: by_tool mapping from a scores file.
    :param permitted: prediction-tool allowlist.
    :return: names of permitted tools that ran without scoring.
    """
    return {
        tool
        for tool, stats in by_tool.items()
        if tool in permitted
        and not _is_number(stats.get("brier"))
        and _is_number(stats.get("n"))
        and int(stats["n"]) > 0
    }


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
    entries = [(n, current.get(n), reference.get(n)) for n in sorted(names)]
    return [name for name, _, _ in sorted(entries, key=_sort_key)]


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

_ALERT_COLUMNS = (
    Col("what"),
    Col("tool"),
    # Free-form prose, so Slack is allowed to wrap these two.
    Col("evidence", wrap=True),
    Col("action", wrap=True),
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

    # Two DIFFERENT conditions, against two different baselines. Collapsing
    # them is the same mistake that produced "BSS vs mkt": Edge is measured
    # against the MARKET, and a tool can lose to the market while still
    # beating its own no-skill floor (live: factual_research on Omen, Brier
    # 0.2234 vs base 0.2330 -- above no-skill -- with Edge -0.0266).
    # Floored, like every other claim in the message. Without this the digest
    # marked a tool "insufficient" in the table and then counted that same
    # window toward "the fleet is worse than the market" in the alerts.
    below_market = [
        at[t]
        for t in tools
        if at.get(t) and _is_number(at[t].get("edge")) and _has_floor(at[t], "edge_n")
    ]
    if len(below_market) >= 2 and all(s["edge"] < 0 for s in below_market):
        best = max(s["edge"] for s in below_market)
        rows.append(
            (
                "platform below market",
                f"ALL ({len(below_market)} tools over the floor)",
                f"every cumulative Edge < 0; best is {best:+.4f}",
                "upstream calibration, not a tool swap",
            )
        )

    scored = [
        t
        for t in tools
        if at.get(t) and _is_number(at[t].get("brier")) and _has_floor(at[t])
    ]
    below_floor = [
        t
        for t in scored
        if _is_number(at[t].get("baseline_brier"))
        and at[t]["brier"] > at[t]["baseline_brier"]
    ]
    if len(scored) >= 2 and below_floor and len(below_floor) == len(scored):
        rows.append(
            (
                "platform below no-skill",
                f"ALL ({len(below_floor)} tools over the floor)",
                "every cumulative Brier exceeds its own base-rate floor",
                "the fleet is worse than predicting the base rate",
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
    allowed_tools: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the benchmark digest as one webhook payload per table.

    :param results_dir: directory holding the scorer output files.
    :param platform: platform key, e.g. "polymarket" or "omen".
    :param roi_results: path to roi_results.json; defaults to
        ``results_dir/roi_results.json``.
    :param allowed_tools: when given, only these tools are rendered. Callers
        pass the prediction-tool registry, so tools that are not forecasters --
        a question-proposing tool never emits a ``p_yes`` -- cannot appear.
        Injected rather than imported to keep this module stdlib-only.
    :return: ordered webhook payloads; empty when there is nothing to post.
    """
    windows = {
        key: _load_by_tool(results_dir / name.format(platform=platform))
        for key, name in WINDOW_FILES.items()
    }
    roi = _load_roi(roi_results or results_dir / "roi_results.json", platform)
    # The accumulator's span is not fixed and not stated by its own
    # `current_month` key, so the section line carries the observed bounds.
    span = _load_window(results_dir / WINDOW_FILES["at"].format(platform=platform))
    month_label = span if span else "an UNSTATED span - see the docstring"

    prod_tools = _scored(windows["at"]) | _scored(windows["w1"])
    tourn_tools = _scored(windows["tournament"])
    if allowed_tools is not None:
        permitted = set(allowed_tools)
        prod_tools &= permitted
        tourn_tools &= permitted
        # A permitted tool that ran and scored nothing still belongs here.
        prod_tools |= _ran_but_unscored(windows["at"], permitted)
        prod_tools |= _ran_but_unscored(windows["w1"], permitted)
    if not prod_tools and not tourn_tools:
        log.warning("digest: no scored tools for %s; skipping tables", platform)
        return []

    messages: list[dict[str, Any]] = [
        _headline(
            _verdicts_for(sorted(prod_tools), windows["at"], True, windows["w1"]),
            _verdicts_for(sorted(tourn_tools), windows["tournament"], False),
            platform,
        )
    ]

    # Degrade LOUDLY. Both rolling files come from separate --period-days
    # scorer invocations that can fail independently of the all-time rebuild.
    # Without this the W-1/W-2 columns silently fill with n/a and the digest
    # reads as "no change this week" rather than "this week was not scored".
    missing = [
        label for key, label in (("w1", "W-1"), ("w2", "W-2")) if not windows[key]
    ]
    if missing:
        warning = (
            f":warning: *{' and '.join(missing)} unavailable* - the rolling "
            "scorer step produced no scores for this platform, so those "
            "columns read `n/a` below. They are NOT unchanged; they are "
            "unmeasured."
        )
        messages.append(
            message(f"{' and '.join(missing)} unavailable", [section(warning)])
        )

    if prod_tools:
        order_w2 = _ordered(prod_tools, windows["w1"], windows["w2"])
        order_at = _ordered(prod_tools, windows["w1"], windows["at"])
        messages.append(
            message(
                "1a. Production - W-1 vs W-2",
                [
                    section("*1a. PRODUCTION - W-1 vs W-2*  _what changed this week_"),
                    table_block(
                        _W2_COLUMNS,
                        _rows_w2(
                            order_w2, windows["w1"], windows["w2"], roi, "production"
                        ),
                    ),
                ],
            )
        )
        messages.append(
            message(
                "1b. Production - W-1 vs cumulative",
                [
                    section(
                        f"*1b. PRODUCTION - W-1 vs CUMULATIVE*  _this week against "
                        f"{month_label}_"
                    ),
                    table_block(
                        _AT_COLUMNS,
                        _rows_at(
                            order_at, windows["w1"], windows["at"], roi, "production"
                        ),
                    ),
                ],
            )
        )

    if tourn_tools:
        order = _ordered(tourn_tools, {}, windows["tournament"])
        messages.append(
            message(
                "2. Tournament - all-time pool",
                [
                    section(
                        "*2. TOURNAMENT - all-time pool*  _candidates; no weekly "
                        "comparison available_"
                    ),
                    table_block(
                        _AT_COLUMNS,
                        _rows_at(order, {}, windows["tournament"], roi, "tournament"),
                    ),
                    context(
                        "Two caveats. Tournament rows carry no W-1/W-2: rolling "
                        "scorer runs pass `--skip-tournament-output`, so no "
                        "weekly aggregate is written for a candidate. And the "
                        "columns headed `cum` hold the candidate's ALL-TIME "
                        "pool here, not a month -- a different span, over much "
                        "easier markets (`mkt` ~0.07 against ~0.18 in "
                        "production). Candidate and incumbent numbers are NOT "
                        "comparable row to row."
                    ),
                ],
            )
        )

    alerts = _alert_rows(windows["w1"], windows["at"], sorted(prod_tools))
    if alerts:
        messages.append(
            message(
                f"3. Alerts ({len(alerts)})",
                [
                    section("*3. ALERTS*  _conditions that block a call_"),
                    table_block(_ALERT_COLUMNS, alerts),
                ],
            )
        )

    return messages
