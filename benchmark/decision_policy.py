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
"""Pure deployment policy shared by report renderers and offline audit replay.

Extracted without changing thresholds or verdict precedence. Roster coverage and
input availability are handled by the report layer, separately from this policy.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from benchmark.roi_sim import RELIABILITY_GATE
from benchmark.scoring_primitives import MIN_SAMPLE_SIZE

NA = "n/a"

PROMOTE_DELTA = 0.04
# A Brier this far above its own base-rate floor is a signal; less is noise.
NO_SKILL_MARGIN = 0.01
# One-sided 95%: the normal-approximation z. The policy specifies a bootstrap;
# at n >= 30 on a mean this is the same call to well inside the margin, and it
# is computable from the stored sum/sum-of-squares without per-row data.
Z_ONE_SIDED_95 = 1.645

COIN_FLIP = 0.50


def _num(value: object) -> float | None:
    """Return *value* as a float when it is a real number (bools excluded).

    :param value: candidate value.
    :return: the value as a float, or None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


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


def _edge_lower_bound(stats: dict[str, Any] | None) -> float | None:
    """One-sided 95% lower bound on the mean edge, or None without a spread.

    :param stats: per-tool stats for the decision window.
    :return: the bound, or None.
    """
    edge = _num((stats or {}).get("edge"))
    sd = _num((stats or {}).get("edge_sd"))
    count = _num((stats or {}).get("edge_n"))
    if edge is None or sd is None or count is None or count < 2:
        return None
    return edge - Z_ONE_SIDED_95 * sd / math.sqrt(count)


def _below_no_skill(stats: dict[str, Any] | None) -> bool:
    """Is this window's Brier materially worse than its own base rate?

    Material means >= NO_SKILL_MARGIN: a hairline gap is noise, not a demote.

    :param stats: per-tool stats for one window.
    :return: True when below no-skill by more than the margin.
    """
    brier = _num((stats or {}).get("brier"))
    base = _num((stats or {}).get("baseline_brier"))
    return brier is not None and base is not None and brier - base >= NO_SKILL_MARGIN


def _below_reliability(stats: dict[str, Any] | None) -> bool:
    """Did too few calls return a usable prediction this window?

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
    """Gate verdict; low reliability annotates positive outcomes.

    ``PROMOTE (rel 60%)`` -- the warning rides along, never vetoes: a tool
    that misses calls may still be the best forecaster available.

    :param stats: per-tool stats for the decision window.
    :param deployed: True for a production tool, False for a candidate.
    :param recent: stats for the most recent week.
    :return: a short verdict string for the ``rec`` cell.
    """
    verdict = _verdict_core(stats, deployed, recent)
    rate = _num((stats or {}).get("reliability"))
    if _below_reliability(stats) and verdict.startswith(("PROMOTE", "keep", "review")):
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

    # Conditional accuracy under 50% VETOES a candidate promote (review:,
    # never PROMOTE) but only annotates a deployed keep -- destructive
    # actions need the stronger signal, eligibility gating is cheap.
    # Each reason NAMES THE COLUMN that triggered it rather than describing it
    # in prose. The reader can then check the verdict against a cell on the
    # same row, and the cell stays the width of the other columns instead of
    # wrapping to three lines.
    # Reliability never vetoes: the _verdict wrapper annotates a positive
    # verdict with (rel NN%) instead -- a tool that misses calls may still
    # be the best forecaster available.
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
    # An absent or thin week is "we know nothing new" and must never take
    # the destructive action; a 3-row lucky week must not erase a 500-row
    # demote signal either -- the week counts only when it clears the same
    # sample floor as every other cross-window comparison. Candidates carry
    # no weekly window; their "no:" is non-destructive, so one reading is
    # enough.
    sustained = _below_no_skill(stats) and (
        (_has_floor(recent) and _below_no_skill(recent))
        if deployed
        else (recent is None or _below_no_skill(recent))
    )
    if sustained:
        return "demote: no-skill" if deployed else "no: no-skill"

    return "keep" if deployed else f"no: {PROMOTE_DELTA - lower:+.3f} short"


def _survivors(verdicts: dict[str, str]) -> list[str]:
    """Tools that could actually replace a demoted one.

    "n=12 < 30" or "no spread" is NOT a survivor: retiring every judged tool
    must not leave the platform holding only what could not be judged.

    :param verdicts: mapping of tool name to verdict.
    :return: names of tools that survive.
    """
    return [
        tool
        for tool, verdict in verdicts.items()
        if verdict.startswith(("keep", "PROMOTE", "review"))
    ]


def _verdicts_for(
    tools: Sequence[str],
    at: dict[str, dict[str, Any]],
    deployed: bool,
    w1: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Apply the gate to one cohort.

    :param tools: tool names.
    :param at: decision-window stats (trailing 90d, or tournament pool).
    :param deployed: True for production tools.
    :param w1: weekly stats, for the sustained-demote confirmation.
    :return: mapping of tool name to verdict.
    """
    verdicts = {
        t: _verdict(at.get(t), deployed=deployed, recent=(w1 or {}).get(t))
        for t in tools
    }
    # No cohort-level rewrite: the every-tool-demotes case is stated once
    # under the table by _no_replacement_note, not stamped into each row.
    return verdicts


def _decision_state(
    prod: dict[str, str], tourn: dict[str, str]
) -> tuple[str, str, list[str], list[str]]:
    """Return marker state, token, promotions, and actionable demotions."""
    promote = sorted(
        tool for tool, verdict in tourn.items() if verdict.startswith("PROMOTE")
    )
    demote = sorted(
        tool for tool, verdict in prod.items() if verdict.startswith("demote")
    )
    blocked = bool(prod) and not _survivors(prod)
    if blocked:
        if promote:
            # A qualified candidate is the only safe next action. Demotions
            # become actionable only after that replacement is deployed.
            return "promote", f"PROMOTE {len(promote)} FIRST", promote, []
        return "blocked", "NO ACTION", [], demote
    if promote and demote:
        return (
            "demote",
            f"PROMOTE {len(promote)} · DEMOTE {len(demote)}",
            promote,
            demote,
        )
    if promote:
        return "promote", f"PROMOTE {len(promote)}", promote, []
    if demote:
        return "demote", f"DEMOTE {len(demote)}", [], demote
    return "none", "NO CHANGE", [], []
