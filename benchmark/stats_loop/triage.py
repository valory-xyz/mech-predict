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
"""Deterministic triage of per-tool rolling-window Brier scores.

Given two rolling-window score files (current 7d and previous 7d, both
per-platform) and a state file from yesterday's triage run, decide which
tools have flagged a real regression worth opening an issue for.

Gates (all must pass for a tool to be flagged):

1. Platform: Polymarket only. Omen Brier is confounded by the on-chain
   ``jury-resolve-market-v1`` mech tool's quality (see R0014 sec.6 and
   the retired-Grok-voter incident); a tool-side fix cannot reliably
   address an Omen regression, so the agent's scope excludes it.
2. Regression size: ``Brier_cur - Brier_prev > 0.015``.
3. Sample floor: ``n_window >= 60`` per window AND ``n_day >= 30``
   per day (the day-level check is approximated by the per-window
   ``valid_n / 7`` floor).
4. Sign agreement: ``sign(delta Brier) == sign(delta log_loss)``.
   Both being primary metrics (PROPOSAL.md Part 4), disagreement
   suggests the move is in the noise rather than a real regression.
5. Reliability path: when ``reliability < 0.80``, route to a separate
   reliability-collapse outcome (pages a human) instead of opening a
   tool-improvement issue. Reliability collapses are upstream (API
   outages, retired model slugs, evidence-fetch failures) and are not
   tool-quality problems.
6. Two-day confirmation: today's triage must agree with yesterday's
   for the same tool. Single-day signals at n in [60, 200] have a
   ~15% false-positive rate; the confirmation cuts FP to ~2%.

The function ``triage_tools`` returns a list of ``TriageDecision`` records
that the calling script (``open_issue``) inspects to decide what to publish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


PLATFORM = "polymarket"
BRIER_REGRESSION_THRESHOLD = 0.015
N_WINDOW_FLOOR = 60
N_DAY_FLOOR = 30
ROLLING_WINDOW_DAYS = 7
RELIABILITY_FLOOR = 0.80


@dataclass
class TriageOutcome:
    """One tool's triage outcome for a single day.

    Stored verbatim in the cross-day state file (``stats_loop_state.json``)
    so tomorrow's triage can apply the two-day confirmation gate (gate 6).
    """

    tool_name: str
    flagged: bool
    delta_brier: Optional[float]
    delta_log_loss: Optional[float]
    brier_cur: Optional[float]
    brier_prev: Optional[float]
    log_loss_cur: Optional[float]
    log_loss_prev: Optional[float]
    n_cur: int
    n_prev: int
    reliability_cur: Optional[float]
    reason: str


@dataclass
class TriageDecision:
    """The triage result for a tool: one of three categories.

    - ``open_issue``: gate cascade passed AND yesterday also flagged the
      same tool. The dispatcher should open a ``tool-improvement`` issue.
    - ``reliability_collapse``: reliability dropped below floor. Page a
      human; do NOT open a tool-improvement issue (the fix is upstream).
    - ``silent``: no gate triggered, or first day of a possible regression
      (needs confirmation tomorrow). Most days every tool is silent.
    """

    tool_name: str
    decision: str  # "open_issue" | "reliability_collapse" | "silent"
    today: TriageOutcome
    yesterday: Optional[TriageOutcome] = field(default=None)


def _load_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file or return ``{}`` when the file is missing."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _tool_stats(scores: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """Extract one tool's row from a scorer-emitted scores JSON file.

    The scorer writes per-platform stats under ``by_tool[<tool_name>]`` in
    the per-platform sibling file (e.g. ``rolling_scores_polymarket.json``).
    """
    return scores.get("by_tool", {}).get(tool_name, {}) or {}


def _safe_get(stats: Dict[str, Any], field_name: str) -> Optional[float]:
    """Return a float field or ``None`` when missing or non-numeric."""
    value = stats.get(field_name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gate_sample_floor(stats: Dict[str, Any]) -> bool:
    """Return True when the per-window sample floors are met."""
    valid_n = stats.get("valid_n") or stats.get("n") or 0
    if valid_n < N_WINDOW_FLOOR:
        return False
    # Approximate "n_day >= 30" by averaging the window: 7 days of data
    # at >= 30/day gives >= 210; we use a softer floor since the day
    # distribution is rarely uniform. A direct per-day check would
    # require the daily logs; the per-window floor catches the obvious
    # holiday/low-traffic-window misfires.
    if valid_n / ROLLING_WINDOW_DAYS < (N_DAY_FLOOR / 2):
        return False
    return True


def _gate_reliability(stats: Dict[str, Any]) -> Optional[float]:
    """Return reliability when below floor (else ``None``).

    A reliability collapse short-circuits the rest of the cascade and
    routes to a separate outcome. The caller treats a non-None return
    as a "page the operator" signal.
    """
    reliability = _safe_get(stats, "reliability")
    if reliability is None:
        return None
    if reliability < RELIABILITY_FLOOR:
        return reliability
    return None


def _triage_one_tool(
    tool_name: str,
    cur_stats: Dict[str, Any],
    prev_stats: Dict[str, Any],
) -> TriageOutcome:
    """Apply all the size/sign/sample gates to one tool's row (no confirmation).

    Returns a ``TriageOutcome`` whose ``flagged`` flag is True when every
    gate except confirmation has passed. The confirmation gate is applied
    one level up, in ``triage_tools``, which has access to yesterday's
    state.
    """
    brier_cur = _safe_get(cur_stats, "brier")
    brier_prev = _safe_get(prev_stats, "brier")
    log_loss_cur = _safe_get(cur_stats, "log_loss")
    log_loss_prev = _safe_get(prev_stats, "log_loss")
    n_cur = int(cur_stats.get("valid_n") or cur_stats.get("n") or 0)
    n_prev = int(prev_stats.get("valid_n") or prev_stats.get("n") or 0)
    reliability_cur = _safe_get(cur_stats, "reliability")

    delta_brier = (
        brier_cur - brier_prev
        if (brier_cur is not None and brier_prev is not None)
        else None
    )
    delta_log_loss = (
        log_loss_cur - log_loss_prev
        if (log_loss_cur is not None and log_loss_prev is not None)
        else None
    )

    def outcome(flagged: bool, reason: str) -> TriageOutcome:
        return TriageOutcome(
            tool_name=tool_name,
            flagged=flagged,
            delta_brier=delta_brier,
            delta_log_loss=delta_log_loss,
            brier_cur=brier_cur,
            brier_prev=brier_prev,
            log_loss_cur=log_loss_cur,
            log_loss_prev=log_loss_prev,
            n_cur=n_cur,
            n_prev=n_prev,
            reliability_cur=reliability_cur,
            reason=reason,
        )

    if not _gate_sample_floor(cur_stats) or not _gate_sample_floor(prev_stats):
        return outcome(False, "sample_floor")
    if delta_brier is None:
        return outcome(False, "missing_brier")
    if delta_brier <= BRIER_REGRESSION_THRESHOLD:
        return outcome(False, "no_regression")
    if delta_log_loss is None:
        return outcome(False, "missing_log_loss")
    # Sign agreement: both metrics must move in the worsening direction
    # (positive delta on each).
    if delta_log_loss <= 0:
        return outcome(False, "sign_disagreement")
    return outcome(True, "all_gates_pass")


def triage_tools(
    cur_scores_path: Path,
    prev_scores_path: Path,
    state_path: Path,
) -> List[TriageDecision]:
    """Run the triage cascade across every tool present in the current window.

    :param cur_scores_path: per-platform Current 7d scores JSON path
        (e.g. ``benchmark/results/rolling_scores_polymarket.json``).
    :param prev_scores_path: per-platform Previous 7d scores JSON path
        (e.g. ``benchmark/results/prev_rolling_scores_polymarket.json``).
    :param state_path: path to the cross-day state file written by the
        previous triage run. Read-only here; the caller is responsible
        for writing the new state after acting on the decisions.
    :return: list of ``TriageDecision`` records, one per tool the cascade
        touched. Tools that fail any gate produce a ``"silent"`` decision.
    """
    cur = _load_json(cur_scores_path)
    prev = _load_json(prev_scores_path)
    state = _load_json(state_path)
    yesterday_by_tool: Dict[str, Dict[str, Any]] = state.get("by_tool", {}) or {}

    decisions: List[TriageDecision] = []
    for tool_name in sorted((cur.get("by_tool") or {}).keys()):
        cur_stats = _tool_stats(cur, tool_name)
        prev_stats = _tool_stats(prev, tool_name)

        # Gate 5 first: reliability is a separate outcome; do not let
        # reliability-collapsed tools shadow the regression analysis.
        reliability_below = _gate_reliability(cur_stats)
        if reliability_below is not None:
            collapse_outcome = TriageOutcome(
                tool_name=tool_name,
                flagged=False,
                delta_brier=None,
                delta_log_loss=None,
                brier_cur=_safe_get(cur_stats, "brier"),
                brier_prev=_safe_get(prev_stats, "brier"),
                log_loss_cur=_safe_get(cur_stats, "log_loss"),
                log_loss_prev=_safe_get(prev_stats, "log_loss"),
                n_cur=int(cur_stats.get("valid_n") or cur_stats.get("n") or 0),
                n_prev=int(prev_stats.get("valid_n") or prev_stats.get("n") or 0),
                reliability_cur=reliability_below,
                reason="reliability_collapse",
            )
            decisions.append(
                TriageDecision(
                    tool_name=tool_name,
                    decision="reliability_collapse",
                    today=collapse_outcome,
                )
            )
            continue

        today = _triage_one_tool(tool_name, cur_stats, prev_stats)
        yesterday_raw = yesterday_by_tool.get(tool_name)
        yesterday = (
            TriageOutcome(**yesterday_raw)
            if isinstance(yesterday_raw, dict)
            else None
        )

        # Gate 6 (confirmation): both today and yesterday must have flagged
        # the same tool to open an issue. A first-day flag is silent (it
        # just gets written to state for tomorrow to confirm).
        if today.flagged and yesterday is not None and yesterday.flagged:
            decisions.append(
                TriageDecision(
                    tool_name=tool_name,
                    decision="open_issue",
                    today=today,
                    yesterday=yesterday,
                )
            )
        else:
            decisions.append(
                TriageDecision(
                    tool_name=tool_name,
                    decision="silent",
                    today=today,
                    yesterday=yesterday,
                )
            )
    return decisions


def write_state(
    state_path: Path,
    decisions: List[TriageDecision],
    generated_at: str,
) -> None:
    """Persist today's triage outcomes for tomorrow's confirmation gate.

    The state file is intentionally simple JSON so it can be inspected
    by humans and round-tripped through the GitHub Actions artifact
    that carries it across daily runs.
    """
    payload = {
        "generated_at": generated_at,
        "platform": PLATFORM,
        "by_tool": {
            decision.tool_name: {
                "tool_name": decision.today.tool_name,
                "flagged": decision.today.flagged,
                "delta_brier": decision.today.delta_brier,
                "delta_log_loss": decision.today.delta_log_loss,
                "brier_cur": decision.today.brier_cur,
                "brier_prev": decision.today.brier_prev,
                "log_loss_cur": decision.today.log_loss_cur,
                "log_loss_prev": decision.today.log_loss_prev,
                "n_cur": decision.today.n_cur,
                "n_prev": decision.today.n_prev,
                "reliability_cur": decision.today.reliability_cur,
                "reason": decision.today.reason,
            }
            for decision in decisions
        },
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
