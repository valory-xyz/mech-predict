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

Every cell is computed from a scorer artifact -- no LLM anywhere -- so this
module is unit-testable.

Inputs, all written by ``benchmark.scorer`` into the same results directory:

===========================================  ==============================
file                                          window
===========================================  ==============================
``trailing_scores_<platform>.json``           trailing 90d (nightly --period-days 90)
``rolling_scores_<platform>.json``            W-1, the last 7 days
``prev_rolling_scores_<platform>.json``       W-2, the 7 days before that
``scores_tournament_<platform>.json``         tournament, all-time only
===========================================  ==============================

``base`` (no-skill floor) and ``mkt`` (market Brier) are shown in Brier units
beside the tool's Brier; a skill-score ratio is deliberately NOT rendered --
its denominator sits off-screen.

One table per Slack message: stays under Slack's per-message cell-character
cap and lets a reader thread a reply to one table.

Pure builder, stdlib only: no network, no LLM, no clock. A broken digest must
never break the daily post, so the entry point returns an empty list instead
of raising when its inputs are missing or unparseable.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Collection, Iterable, Sequence

from benchmark.roi_sim import RELIABILITY_GATE
from benchmark.scoring_primitives import MIN_SAMPLE_SIZE
from benchmark.slack_blocks import (
    Col,
    context,
    header,
    message,
    section,
    table_block,
)
from benchmark.slack_tables import display_width
from benchmark.tool_usage import normalize_tool_name

log = logging.getLogger(__name__)

NA = "n/a"

# ASCII on purpose: table cells stay ASCII so column padding cannot skew.
LOW_SAMPLE_MARK = "*"


# Deployment names, so the title matches what the team calls each platform.
PLATFORM_TITLES = {"polymarket": "Polystrat", "omen": "Omenstrat"}

WINDOW_FILES = {
    "at": "trailing_scores_{platform}.json",
    "w1": "rolling_scores_{platform}.json",
    "w2": "prev_rolling_scores_{platform}.json",
    "tournament": "scores_tournament_{platform}.json",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_by_tool(path: Path) -> dict[str, dict[str, Any]]:
    """Read one scores file and return its ``by_tool`` mapping.

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


def _as_of_date(path: Path) -> str | None:
    """Last date the scores file covers: window_end, else generated_at.

    The batch scorer stamps both bounds; ``window_end`` is the last
    observed prediction, so it can trail the run date when predictions pause.

    :param path: scores json path.
    :return: as-of date (YYYY-MM-DD) or None.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stamp = payload.get("window_end") or payload.get("generated_at")
    return str(stamp)[:10] if stamp else None


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------


def _num(value: object) -> float | None:
    """Return *value* as a float when it is a real number (bools excluded).

    :param value: candidate value.
    :return: the value as a float, or None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _score(value: object, signed: bool = False) -> str:
    """Format a Brier-unit score to 4 decimals.

    :param value: candidate value.
    :param signed: render an explicit sign (Edge columns).
    :return: formatted score, or ``n/a``.
    """
    if _num(value) is None:
        return NA
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _count(stats: dict[str, Any] | None) -> str:
    """Format a window's scored-row count, marking under-floor samples.

    Renders ``valid_n/edge_n`` when the pools differ: Brier averages over
    valid rows, Edge/mkt over the priced subset, and the low-sample marker
    keys off the smaller pool.

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
    """Format a window's reliability (``valid_n / n``) as a percentage.

    Qualifies the Brier beside it: the score is averaged over valid rows only.

    :param stats: per-tool stats dict for that window, or None.
    :return: percentage cell, e.g. ``97%``, or ``n/a``.
    """
    if not stats or _num(stats.get("reliability")) is None:
        return NA
    return f"{stats['reliability'] * 100:.0f}%"


def _has_floor(stats: dict[str, Any] | None, field: str = "valid_n") -> bool:
    """Check whether a window has enough rows to support a delta.

    Edge deltas floor on ``edge_n``, Brier deltas on ``valid_n`` -- the two
    average over different pools.

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

    The direction word is spelled out: the sign alone is ambiguous across
    columns (a positive Brier delta is a regression, a positive Edge delta
    an improvement).

    :param current: stats for the current window (W-1).
    :param reference: stats for the comparison window (W-2 or all-time).
    :param field: metric key present in both stats dicts.
    :param lower_is_better: True for Brier, False for Edge.
    :return: e.g. ``+0.0417 worse``, ``+0.0417 worse *`` (low sample), or ``n/a``.
    """
    if not current or not reference:
        return NA
    a = _num(current.get(field))
    b = _num(reference.get(field))
    if a is None or b is None:
        return NA
    # Edge and mkt live on the edge-eligible pool; everything else on valid_n.
    count_field = "edge_n" if field in ("edge", "market_brier") else "valid_n"
    diff = a - b
    improved = diff < 0 if lower_is_better else diff > 0
    cell = f"{diff:+.4f} {'better' if improved else 'worse'}"
    if not (_has_floor(current, count_field) and _has_floor(reference, count_field)):
        # Same cell, starred: under the sample floor the arithmetic is real
        # but not statistically significant -- the star the counts carry.
        return f"{cell} {LOW_SAMPLE_MARK}"
    return cell


COMPLETENESS_RATIO = 0.70


PROMOTE_DELTA = 0.04
# A Brier this far above its own base-rate floor is a signal; less is noise.
NO_SKILL_MARGIN = 0.01
# One-sided 95%: the normal-approximation z. The policy specifies a bootstrap;
# at n >= 30 on a mean this is the same call to well inside the margin, and it
# is computable from the stored sum/sum-of-squares without per-row data.
Z_ONE_SIDED_95 = 1.645

NO_SKILL_MARGIN = 0.01
# One-sided 95%: the normal-approximation z. The policy specifies a bootstrap;
# at n >= 30 on a mean this is the same call to well inside the margin, and it
# is computable from the stored sum/sum-of-squares without per-row data.
Z_ONE_SIDED_95 = 1.645

Z_ONE_SIDED_95 = 1.645

COIN_FLIP = 0.50


# The rule lives inside the header's own text so Slack cannot pad between
# rule and title; its length tracks the title rather than being fixed.
VERDICT_MARKER = {
    "promote": "🟢",
    "demote": "🟠",
    "blocked": "🔴",
    "none": "⚪",
}

TITLE_RULE_CHAR = "━"
TITLE_RULE_MIN = 24


def _headline(
    prod: dict[str, str],
    tourn: dict[str, str],
    platform: str,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Build the one-message answer: what to promote, what to demote.

    The tables carry the evidence; this carries the decision. A reader who
    stops after the first message must still leave with the right action.

    :param prod: verdicts for the deployed tools.
    :param tourn: verdicts for the candidates.
    :param platform: platform key, for the heading.
    :param as_of: last date the data covers, from the scores file -- this
        module has no clock.
    :return: a webhook payload.
    """
    promote = sorted(t for t, v in tourn.items() if v.startswith("PROMOTE"))
    demote = sorted(t for t, v in prod.items() if v.startswith("demote"))
    unjudged = sorted(
        t
        for t, v in {**prod, **tourn}.items()
        if v.startswith(("n=", "no spread", "needs --rebuild"))
    )
    # Asked of the shared survivor rule, NOT of the verdict text.
    blocked = bool(prod) and not _survivors(prod)

    if promote:
        state, token = "promote", f"PROMOTE {len(promote)}"
        detail = "Promote " + ", ".join(f"`{t}`" for t in promote) + "."
    elif demote and not blocked:
        state, token = "demote", f"DEMOTE {len(demote)}"
        detail = "Demote " + ", ".join(f"`{t}`" for t in demote) + "."
    elif demote:
        state, token = "blocked", "NO ACTION"
        # NOT len(demote): "blocked" means no tool SURVIVED, which includes
        # the ones the gate could not judge at all.
        detail = (
            "No deployed tool clears the gate, so demoting the "
            f"{len(demote)} that fail outright would leave this platform "
            "with no forecaster. Treat as a platform-level problem, not a "
            "tool swap."
        )
    else:
        state, token = "none", "NO CHANGE"
        detail = "No candidate clears the promote margin."

    label = PLATFORM_TITLES.get(platform, platform.title())
    marker = VERDICT_MARKER[state]
    # Date inside the rules, on the title line: platform, badge, outcome and
    # day are one block.
    # Single-space separators: the three framed lines must fit Slack's
    # 150-char header cap, and the verdict token pushed double-spacing over.
    stamp = f" \u00b7 {as_of}" if as_of else ""
    line = f"{marker} {label.upper()} \u00b7 REPORT V2 \u00b7 {token}{stamp}"
    rule = TITLE_RULE_CHAR * max(TITLE_RULE_MIN, display_width(line))
    title = f"{rule}\n{line}\n{rule}"

    lines = [detail]
    if unjudged:
        lines.append(
            f"_{len(unjudged)} tool(s) had too little data to judge: "
            + ", ".join(f"`{t}`" for t in unjudged)
            + "._"
        )
    return message(
        f"{label} {as_of or ''}: {token}".strip(),
        [
            header(title),
            section("\n".join(lines)),
            context(
                "*Legend:*\n"
                f"\u2022 {LOW_SAMPLE_MARK}: marks a window under "
                f"n = {MIN_SAMPLE_SIZE} -- the number is reported but not "
                "statistically significant. Keys off the smaller of the two "
                "pools below.\n"
                "\u2022 *n*: number of scored predictions, not distinct "
                "markets -- a market asked repeatedly counts each time. "
                "Shown as valid/edge when the Brier pool and the Edge pool "
                "differ.\n"
                "\u2022 *rel*: share of calls that returned a usable "
                "prediction.\n"
                "\u2022 *base*: score of always predicting the pool's "
                "average YES rate -- the zero-skill reference.\n"
                "\u2022 *mkt*: score of always predicting the current "
                "market price -- the crowd reference.\n"
                f"\u2022 *floor*: worst Edge the evidence still supports "
                f"(one-sided 95%); the promote test is floor > "
                f"+{PROMOTE_DELTA}.\n"
                "\u2022 *condAcc*: win rate on the markets where the tool "
                "DISAGREED with the price -- under 50% blocks a promote.\n"
                "\u2022 *rec*: the verdict -- see "
                "DAILY_REPORT_OPERATOR_GUIDE.md."
            ),
        ],
    )


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


def _below_reliability(stats: dict[str, Any] | None) -> bool:
    """Did this window fail to return a usable prediction often enough to matter?

    Reliability is ``valid_n / n``. A tool that answers four calls in five is
    not one the trader can lean on, whatever the four scored -- and the failures
    are not a random sample, so the Brier computed on them is not a fair read of
    the tool either.

    :param stats: per-tool stats for one window.
    :return: True when reliability is known and below the gate.
    """
    rate = _num((stats or {}).get("reliability"))
    return rate is not None and rate < RELIABILITY_GATE


def _verdict(
    stats: dict[str, Any] | None,
    deployed: bool,
    recent: dict[str, Any] | None = None,
) -> str:
    """Apply the gate, then flag low reliability as a WARNING rather than a veto.

    Reliability qualifies a good verdict instead of overturning it. A tool that
    misses calls may still be the best forecaster available, and refusing to
    promote it can leave a worse tool deployed -- so the number is surfaced on
    the row and the human decides. Only the positive verdicts are annotated: a
    demote needs no extra reason, and a row the gate could not judge is not
    made clearer by a second caveat.

    :param stats: per-tool stats for the decision window.
    :param deployed: True for a production tool, False for a candidate.
    :param recent: stats for the most recent week.
    :return: a short verdict string for the ``rec`` cell.
    """
    verdict = _verdict_core(stats, deployed, recent)
    rate = _num((stats or {}).get("reliability"))
    if (
        rate is not None
        and rate < RELIABILITY_GATE
        and verdict.startswith(("PROMOTE", "keep", "review"))
    ):
        return f"{verdict} (rel {rate:.0%})"
    return verdict


def _verdict_core(
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
        return f"n={shown} < {MIN_SAMPLE_SIZE}"

    lower = _edge_lower_bound(stats)
    if lower is None:
        # An accumulator written before `edge_sd` existed restores the field as
        # None and never re-arms on the incremental path, which is the daily
        # path in production. That is a MIGRATION state, not a thin sample, and
        # saying so is the difference between "run a rebuild" and "this tool
        # has no data".
        if (stats or {}).get("edge_sd", "missing") is None:
            return "needs --rebuild"
        return "no spread"

    # A cleared bound is the policy's promote condition and nothing below may
    # silently override it. Conditional accuracy is a strong diagnostic but it
    # is NOT in the policy, its 0.50 line carries no interval (today's live
    # values all straddle it on a Wilson 95%), so it qualifies a promote rather
    # than vetoing one.
    # Each reason NAMES THE COLUMN that triggered it rather than describing it
    # in prose. The reader can then check the verdict against a cell on the
    # same row, and the cell stays the width of the other columns instead of
    # wrapping to three lines.
    # Reliability is a HARD gate, and it runs BEFORE the floor test so an
    # unreliable tool can never read PROMOTE however good its bound looks. The
    # score is computed only on the calls that returned a usable prediction, so
    # a tool answering four in five is not 80% of a good tool -- the missing
    # fifth is not a random sample, and the ROI sim already refuses to simulate
    # it. Promoting something the simulator will not score is incoherent.
    #
    # Asymmetric on purpose, exactly like the no-skill rule below: a promote is
    # blocked on a single window, a demote needs both to agree. Unreliability
    # is usually an outage or an upstream API change, and one bad week should
    # retire nothing.
    conditional = _num((stats or {}).get("conditional_accuracy_rate"))
    if lower > PROMOTE_DELTA:
        if conditional is not None and conditional < COIN_FLIP:
            note = f"floor ok, condAcc {conditional:.0%}"
            return f"keep ({note})" if deployed else f"review: {note}"
        return "keep (floor ok)" if deployed else "PROMOTE"

    if conditional is not None and conditional < COIN_FLIP:
        reason = f"condAcc {conditional:.0%}"
        return f"demote: {reason}" if deployed else f"no: {reason}"

    # Sustained, not a one-off: both the 90d window and the latest week must agree.
    if _below_no_skill(stats) and (recent is None or _below_no_skill(recent)):
        return "demote: no-skill" if deployed else "no: no-skill"

    return "keep" if deployed else f"no: {PROMOTE_DELTA - lower:+.3f} short"


def _survivors(verdicts: dict[str, str]) -> list[str]:
    """Tools that could actually replace a demoted one.

    A verdict that is neither a demote nor a keep -- "n=12 < 30", "no spread",
    "needs --rebuild" -- is NOT a survivor. Counting it as one lets the report
    recommend retiring every tool it could assess while leaving the platform
    holding only the one it just said it could not judge.

    :param verdicts: mapping of tool name to verdict.
    :return: names of tools that survive.
    """
    return [
        tool
        for tool, verdict in verdicts.items()
        if verdict.startswith(("keep", "PROMOTE", "review"))
    ]


def _no_replacement_blocks(
    verdicts: dict[str, str], candidates: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """The no-replacement warning as blocks, or nothing when it does not apply.

    :param verdicts: mapping of tool name to verdict for one platform.
    :param candidates: tournament verdicts, which may hold a replacement.
    :return: a single context block, or an empty list.
    """
    note = _no_replacement_note(verdicts, candidates)
    return [context(note)] if note else []


def _no_replacement_note(
    verdicts: dict[str, str], candidates: dict[str, str] | None = None
) -> str | None:
    """The warning that belongs under a table where every tool demotes.

    Rendered once, adjacent to the rows it qualifies, rather than repeated in
    each row. Acting on the demotes one at a time would retire every forecaster
    on the platform, which is why it has to sit beside the table and not only
    in the headline message.

    Checks the tournament before claiming there is no replacement. Reading only
    the deployed cohort let the page assert "no forecaster at all" directly
    under a headline announcing a promotable candidate -- the one case where a
    replacement demonstrably exists and the operator should be told to reach
    for it.

    :param verdicts: mapping of tool name to verdict for one platform.
    :param candidates: tournament verdicts, which may hold a replacement.
    :return: the warning, or None when at least one tool survives.
    """
    if not verdicts or _survivors(verdicts):
        return None
    ready = sorted(t for t, v in (candidates or {}).items() if v.startswith("PROMOTE"))
    if ready:
        names = ", ".join(f"`{t}`" for t in ready)
        return (
            f":warning: *NO DEPLOYED REPLACEMENT* - all {len(verdicts)} "
            f"deployed tools demote, but {names} clears the promote gate in "
            "the tournament. Promote before retiring, not after."
        )
    return (
        f":warning: *NO REPLACEMENT* - all {len(verdicts)} deployed tools "
        "demote. Acting on these row by row would leave the platform with no "
        "forecaster at all; treat it as a platform-level problem."
    )


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
    # No cohort-level rewrite: the every-tool-demotes case is stated once
    # under the table by _no_replacement_note, not stamped into each row.
    return verdicts


def _floor(stats: dict[str, Any] | None) -> str:
    """Format the lower bound the ranking and the promote rule both use.

    Shown because it decides things. The report ranks on it and the gate tests
    it, so leaving it off-screen would repeat exactly the failure this module
    was written to stop: a number that drives a verdict while the reader can
    only see a different number beside it. It also explains an order that
    ``Edge`` alone appears to contradict.

    :param stats: per-tool stats for the decision window.
    :return: signed bound to 4 decimals, or ``n/a``.
    """
    lower = _edge_lower_bound(stats)
    return NA if lower is None else f"{lower:+.4f}"


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


def _cols(names: Iterable[str]) -> tuple[Col, ...]:
    """Column specs: ``tool`` left-aligned, every other column right-aligned.

    :param names: column header names.
    :return: Col spec tuple for ``table_block``.
    """
    return tuple(
        (
            Col(n)
            if n == "tool"
            else Col(n, wrap=True) if n == "rec" else Col(n, align="right")
        )
        for n in names
    )


_W2_COLUMNS = _cols(
    "#;tool;n W-2;n W-1;rel W-1;Brier W-2;Brier W-1;Δ Brier;base W-2;"
    "base W-1;mkt W-2;mkt W-1;Edge W-2;Edge W-1;Δ Edge".split(";")
)

_AT_COLUMNS = _cols(
    "#;tool;n 90d;n W-1;rel W-1;Brier 90d;Brier W-1;Δ Brier;base 90d;"
    "mkt 90d;mkt W-1;Edge 90d;floor;Edge W-1;Δ Edge;condAcc;rec".split(";")
)

# Tournament rows come from the all-time tournament accumulator, not the
# trailing window, so its headers keep the cum label.
_TOURN_COLUMNS = _cols(
    "#;tool;n cum;n W-1;rel W-1;Brier cum;Brier W-1;Δ Brier;base cum;"
    "mkt cum;mkt W-1;Edge cum;floor;Edge W-1;Δ Edge;condAcc;rec".split(";")
)


def _sort_key(
    entry: tuple[str, dict[str, Any] | None, dict[str, Any] | None],
) -> tuple[int, float, str]:
    """Order rows best-first on Edge; scored before unscored; name breaks ties.

    Edge, not an uncertainty-penalized bound: a bound would rank how well
    MEASURED a tool is, not how good.

    :param entry: (tool name, current stats, reference stats).
    :return: sort key.
    """
    name, current, reference = entry
    for stats in (current, reference):
        edge = _num((stats or {}).get("edge"))
        if edge is not None:
            return (0, -edge, name)
    # Belt-and-braces with _ordered()'s own sort against nondeterministic set
    # iteration order. Removing EITHER is safe; removing BOTH reintroduces the
    # bug, which is what TestOrderStability asserts.
    return (1, 0.0, name)


def _rows_w2(
    tools: Sequence[str],
    w1: dict[str, dict[str, Any]],
    w2: dict[str, dict[str, Any]],
) -> list[tuple[str, ...]]:
    """Build the W-2 vs W-1 table rows.

    :param tools: tool names to render, in the order given.
    :param w1: by_tool stats for the last 7 days.
    :param w2: by_tool stats for the preceding 7 days.
    :return: one cell tuple per tool.
    """
    ranked = [t for t in tools if _num((w1.get(t) or {}).get("edge")) is not None]

    rows: list[tuple[str, ...]] = []
    for tool in tools:
        a, b = w1.get(tool), w2.get(tool)
        rows.append(
            (
                str(ranked.index(tool) + 1) if tool in ranked else "-",
                tool,
                _count(b),
                _count(a),
                _reliability(a),
                _score((b or {}).get("brier")),
                _score((a or {}).get("brier")),
                _delta(a, b, "brier", lower_is_better=True),
                _score((b or {}).get("baseline_brier")),
                _score((a or {}).get("baseline_brier")),
                _score((b or {}).get("market_brier")),
                _score((a or {}).get("market_brier")),
                _score((b or {}).get("edge"), signed=True),
                _score((a or {}).get("edge"), signed=True),
                _delta(a, b, "edge", lower_is_better=False),
            )
        )
    return rows


def _rows_at(
    tools: Sequence[str],
    w1: dict[str, dict[str, Any]],
    at: dict[str, dict[str, Any]],
    deployed: bool,
) -> list[tuple[str, ...]]:
    """Build the reference-window vs W-1 table rows.

    :param tools: tool names to render, in the order given.
    :param w1: by_tool stats for the last 7 days (empty for tournament).
    :param at: by_tool stats for the reference window -- trailing 90d
        for production, the all-time pool for the tournament.
    :param deployed: True for production rows, False for candidates -- the
        verdict wording differs (keep/demote vs PROMOTE/no).
    :return: one cell tuple per tool.
    """
    ranked = [t for t in tools if _num((at.get(t) or {}).get("edge")) is not None]
    verdicts = _verdicts_for(tools, at, deployed, w1)

    rows: list[tuple[str, ...]] = []
    for tool in tools:
        a, b = w1.get(tool), at.get(tool)
        rows.append(
            (
                str(ranked.index(tool) + 1) if tool in ranked else "-",
                tool,
                _count(b),
                _count(a),
                _reliability(a),
                _score((b or {}).get("brier")),
                _score((a or {}).get("brier")),
                _delta(a, b, "brier", lower_is_better=True),
                _score((b or {}).get("baseline_brier")),
                _score((b or {}).get("market_brier")),
                _score((a or {}).get("market_brier")),
                _score((b or {}).get("edge"), signed=True),
                _floor(b),
                _score((a or {}).get("edge"), signed=True),
                _delta(a, b, "edge", lower_is_better=False),
                _conditional(b),
                verdicts[tool],
            )
        )
    return rows


def _scored(by_tool: dict[str, dict[str, Any]]) -> set[str]:
    """Return tools that produced a Brier in this window.

    :param by_tool: by_tool mapping from a scores file.
    :return: set of tool names with a numeric Brier.
    """
    return {t for t, s in by_tool.items() if _num(s.get("brier")) is not None}


def _ran_but_unscored(
    by_tool: dict[str, dict[str, Any]], permitted: Collection[str]
) -> set[str]:
    """Return PREDICTION tools that ran but produced no usable prediction.

    The allowlist keeps this from alarming on non-forecaster tools, whose
    ``valid_n=0`` is a correct reading, not a failure.

    :param by_tool: by_tool mapping from a scores file.
    :param permitted: prediction-tool allowlist.
    :return: names of permitted tools that ran without scoring.
    """
    return {
        tool
        for tool, stats in by_tool.items()
        if tool in permitted
        and _num(stats.get("brier")) is None
        and _num(stats.get("n")) is not None
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
    at_w2: dict[str, dict[str, Any]] | None = None,
) -> list[tuple[str, ...]]:
    """Build the alerts table: conditions that BLOCK a call, not calls.

    Reliability is deliberately NOT alerted on a bare week-over-week delta:
    at these sample sizes a two-point drop is one or two extra failed rows.

    :param w1: by_tool stats for the last 7 days.
    :param at: by_tool stats for the trailing 90d window.
    :param tools: tool names under consideration.
    :param at_w2: the prior week, used to detect a still-filling W-1.
    :return: alert rows, empty when nothing fires.
    """
    at_w2 = at_w2 or {}
    rows: list[tuple[str, ...]] = []

    starved = [t for t in tools if w1.get(t) and not _has_floor(w1[t])]
    for tool in starved:
        # _has_floor is False when valid_n is MISSING as well as small, so
        # this cannot KeyError on the field it just failed to find.
        starved_n = _num(w1[tool].get("valid_n"))
        shown = NA if starved_n is None else int(starved_n)
        rows.append(
            (
                "sample starved",
                tool,
                f"{shown} scored rows in W-1, floor is {MIN_SAMPLE_SIZE}",
                "widen the window before any keep/demote call",
            )
        )

    for tool in tools:
        stats = w1.get(tool)
        if not stats or _num(stats.get("reliability")) is None:
            continue
        if stats["reliability"] < RELIABILITY_GATE:
            rows.append(
                (
                    "reliability breach",
                    tool,
                    f"{_reliability(stats)} in W-1, gate is " f"{RELIABILITY_GATE:.0%}",
                    "row stays in ROI flagged \u26a0; investigate parse failures",
                )
            )

    # Two DIFFERENT conditions against two different baselines: Edge is
    # measured against the MARKET, and a tool can lose to the market while
    # beating its own no-skill floor. Floored, like every other claim here.
    below_market = [
        at[t]
        for t in tools
        if at.get(t)
        and _num(at[t].get("edge")) is not None
        and _has_floor(at[t], "edge_n")
    ]
    if len(below_market) >= 2 and all(s["edge"] < 0 for s in below_market):
        best = max(s["edge"] for s in below_market)
        rows.append(
            (
                "platform below market",
                f"ALL ({len(below_market)} tools over the floor)",
                f"every 90d Edge < 0; best is {best:+.4f}",
                "upstream calibration, not a tool swap. Check the "
                "recorded price band too: a trader-side bet filter "
                "can confine which markets this platform ever "
                "records, which changes what Edge is measured over",
            )
        )

    # Resolution censoring: the one alert that fires on an ABSENCE, so it is
    # computed here rather than inferred from any single row.
    w1_rows = sum(int(_num(w1[t].get("valid_n")) or 0) for t in tools if w1.get(t))
    w2_rows = sum(
        int(_num(at_w2[t].get("valid_n")) or 0) for t in tools if at_w2.get(t)
    )
    if w2_rows and w1_rows / w2_rows < COMPLETENESS_RATIO:
        rows.append(
            (
                "week still filling",
                f"ALL ({len(tools)} tools)",
                f"W-1 holds {w1_rows} scored rows against W-2's {w2_rows} "
                f"({w1_rows / w2_rows:.0%}); a row appears only once its market "
                "resolves, so the newest week is not yet comparable",
                "read the 90d column, not the weekly deltas",
            )
        )

    scored = [
        t
        for t in tools
        if at.get(t) and _num(at[t].get("brier")) is not None and _has_floor(at[t])
    ]
    below_floor = [
        t
        for t in scored
        if _num(at[t].get("baseline_brier")) is not None
        and at[t]["brier"] > at[t]["baseline_brier"]
    ]
    if len(scored) >= 2 and below_floor and len(below_floor) == len(scored):
        rows.append(
            (
                "platform below no-skill",
                f"ALL ({len(below_floor)} tools over the floor)",
                "every 90d Brier exceeds its own base-rate floor",
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
    allowed_tools: Collection[str] | None = None,
    deployed_tools: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the benchmark digest as one webhook payload per table.

    :param results_dir: directory holding the scorer output files.
    :param platform: platform key, e.g. "polymarket" or "omen".
    :param allowed_tools: when given, only these tools are rendered -- callers
        pass the prediction-tool registry. Injected to stay stdlib-only.
    :param deployed_tools: tools live on this platform's mechs right now
        (from the on-chain manifests). When given, tables 1a/1b list ONLY
        these; None means the lookup was unavailable and the scored roster
        renders unfiltered. Matched via normalize_tool_name, since manifests
        and scorer artifacts disagree on dash vs underscore.
    :return: ordered webhook payloads; empty when there is nothing to post.
    """
    windows = {
        key: _load_by_tool(results_dir / name.format(platform=platform))
        for key, name in WINDOW_FILES.items()
    }
    as_of = _as_of_date(results_dir / WINDOW_FILES["at"].format(platform=platform))

    prod_tools = _scored(windows["at"]) | _scored(windows["w1"])
    tourn_tools = _scored(windows["tournament"])
    if allowed_tools is not None:
        permitted = set(allowed_tools)
        prod_tools &= permitted
        tourn_tools &= permitted
        # A permitted tool that ran and scored nothing still belongs here.
        prod_tools |= _ran_but_unscored(windows["at"], permitted)
        prod_tools |= _ran_but_unscored(windows["w1"], permitted)
    if deployed_tools is not None:
        live = {normalize_tool_name(t) for t in deployed_tools}
        prod_tools = {t for t in prod_tools if normalize_tool_name(t) in live}

    if not prod_tools and not tourn_tools:
        log.warning("digest: no scored tools for %s; skipping tables", platform)
        return []

    messages: list[dict[str, Any]] = [
        _headline(
            _verdicts_for(sorted(prod_tools), windows["at"], True, windows["w1"]),
            _verdicts_for(sorted(tourn_tools), windows["tournament"], False),
            platform,
            as_of,
        )
    ]

    # Degrade LOUDLY: without this, a failed window-scorer run reads as
    # "no change this week" rather than "this week was not scored".
    missing = [
        label
        for key, label in (("w1", "W-1"), ("w2", "W-2"), ("at", "90d"))
        if not windows[key]
    ]
    if missing:
        warning = (
            f":warning: *{' and '.join(missing)} unavailable* - that "
            "window's scorer step produced no scores for this platform, so those "
            "columns read `n/a` below. They are NOT unchanged; they are "
            "unmeasured."
        )
        messages.append(
            message(f"{' and '.join(missing)} unavailable", [section(warning)])
        )

    # 1a compares the two weekly windows, so a tool absent from BOTH has
    # nothing to say there -- with a 90d reference window the roster includes
    # tools retired months ago, and rendering them in 1a is a row of pure n/a.
    # They still appear in 1b, where their 90d numbers are real.
    weekly_tools = prod_tools & (_scored(windows["w1"]) | _scored(windows["w2"]))
    if prod_tools:
        # The 90d reference FIRST here, weekly first in 1a: _sort_key ranks on
        # the first window argument that has an edge, so the argument order IS
        # the ranking metric the caption names.
        order_at = _ordered(prod_tools, windows["at"], windows["w1"])
        if weekly_tools:
            order_w2 = _ordered(weekly_tools, windows["w1"], windows["w2"])
            messages.append(
                message(
                    "1a. Production - W-2 vs W-1",
                    [
                        section(
                            "*1a. PRODUCTION W-2 vs W-1 - ranked by `Edge W-1`*"
                            + (
                                "\n_Currently deployed tools only._"
                                if deployed_tools is not None
                                else ""
                            )
                        ),
                        table_block(
                            _W2_COLUMNS,
                            _rows_w2(order_w2, windows["w1"], windows["w2"]),
                        ),
                    ],
                )
            )
        messages.append(
            message(
                "1b. Production - 90D vs W-1",
                [
                    section(
                        "*1b. PRODUCTION 90D vs W-1 - ranked by `Edge 90d`*"
                        + (
                            "\n_Currently deployed tools only._"
                            if deployed_tools is not None
                            else ""
                        )
                    ),
                    table_block(
                        _AT_COLUMNS,
                        _rows_at(order_at, windows["w1"], windows["at"], True),
                    ),
                ]
                + _no_replacement_blocks(
                    _verdicts_for(
                        sorted(prod_tools), windows["at"], True, windows["w1"]
                    ),
                    _verdicts_for(sorted(tourn_tools), windows["tournament"], False),
                ),
            )
        )

    if tourn_tools:
        order = _ordered(tourn_tools, {}, windows["tournament"])
        messages.append(
            message(
                "2. Tournament - all-time pool",
                [
                    section(
                        "*2. TOURNAMENT - ranked by `Edge cum`*\n"
                        "_cum: the candidate's whole scored pool._"
                    ),
                    table_block(
                        _TOURN_COLUMNS,
                        _rows_at(order, {}, windows["tournament"], False),
                    ),
                    context(
                        "*Caveats:*\n"
                        "\u2022 Candidates have no weekly columns: the "
                        "rolling scorer skips tournament output, so only "
                        "their all-time pool is scored.\n"
                        "\u2022 Candidate rows are NOT comparable to "
                        "production rows: candidates answer a different, "
                        "much easier market pool (compare the *mkt* "
                        "columns), over a different span."
                    ),
                ],
            )
        )

    alerts = _alert_rows(
        windows["w1"], windows["at"], sorted(prod_tools), windows["w2"]
    )
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
