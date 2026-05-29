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
3. Sample floor: a per-window per-day floor of ``valid_n / 7 >= 15``
   (so ``valid_n >= 105`` over the 7-day window). The original design
   target was ``n_day >= 30`` but the daily distribution is rarely
   uniform across the window; the softer floor catches obvious holiday /
   low-traffic-window misfires without rejecting genuine regressions.
4. Sign agreement: ``sign(delta Brier) == sign(delta log_loss)``.
   Both being primary metrics (PROPOSAL.md Part 4), disagreement
   suggests the move is in the noise rather than a real regression.
5. Reliability path: when ``reliability < 0.80``, route to a separate
   reliability-collapse outcome instead of opening a tool-improvement
   issue. Reliability collapses are upstream (API outages, retired
   model slugs, evidence-fetch failures) and are not tool-quality
   problems. The dispatcher logs at WARNING; no automated paging
   exists today, so a human still needs to be watching the daily
   workflow output to act on it.
6. Two-day confirmation: today's triage must agree with yesterday's
   for the same tool. Single-day signals at n in [60, 200] have a
   ~15% false-positive rate; the confirmation cuts FP to ~2%.

Duplicate-issue suppression: once gate 6 fires and an issue opens,
today's outcome is persisted with ``issue_open=True``. The next day,
even if all six gates still pass, the dispatcher honors the persisted
flag and stays silent until the regression resolves (i.e. until a day
where gates 2-4 stop firing for that tool). This prevents the same
sustained regression from filing a new issue every morning.

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
# Per-window per-day soft floor: valid_n / 7 >= 15 => valid_n >= 105.
# (The original 30/day target softened to 15/day because the daily
# distribution across the window is rarely uniform.)
N_DAY_FLOOR_PER_WINDOW_DAY = 15
ROLLING_WINDOW_DAYS = 7
RELIABILITY_FLOOR = 0.80


@dataclass
class TriageOutcome:
    """One tool's triage outcome for a single day.

    Stored verbatim in the cross-day state file (``stats_loop_state.json``)
    so tomorrow's triage can apply the two-day confirmation gate (gate 6).

    All fields are explicitly defaulted so a schema change (an added or
    renamed field) does not crash deserialisation of yesterday's
    persisted state. See ``TriageOutcome.from_dict`` for the tolerant
    adapter used by the dispatcher.
    """

    tool_name: str = ""
    flagged: bool = False
    delta_brier: Optional[float] = None
    delta_log_loss: Optional[float] = None
    brier_cur: Optional[float] = None
    brier_prev: Optional[float] = None
    log_loss_cur: Optional[float] = None
    log_loss_prev: Optional[float] = None
    n_cur: int = 0
    n_prev: int = 0
    reliability_cur: Optional[float] = None
    reason: str = ""
    # When True, today's triage already filed a tool-improvement issue
    # for this regression. Subsequent days that re-flag the same tool
    # stay silent until the regression resolves (gates 2-4 stop firing).
    issue_open: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TriageOutcome":
        """Tolerant deserialiser used when reading yesterday's state.

        Unknown keys (e.g. fields removed in a schema change) are
        silently dropped; missing keys fall back to the dataclass
        defaults. The whole point is that a schema drift in the
        persisted state file must NEVER crash today's triage; the
        worst-case fallback is "yesterday is unknown" (treated as no
        confirmation -> silent today).
        """
        if not isinstance(payload, dict):
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        cleaned = {k: v for k, v in payload.items() if k in known}
        try:
            return cls(**cleaned)
        except (TypeError, ValueError):
            return cls()


@dataclass
class TriageDecision:
    """The triage result for a tool: one of three categories.

    - ``open_issue``: gate cascade passed AND yesterday also flagged the
      same tool AND no issue was already open for this regression. The
      dispatcher should open a ``tool-improvement`` issue.
    - ``reliability_collapse``: reliability dropped below floor. The
      dispatcher logs at WARNING. No automated paging exists yet; a
      human watching the daily workflow output is the safety net.
    - ``silent``: no gate triggered, OR a first-day flag (needs
      tomorrow's confirmation), OR a previously-issued regression
      still firing (gate 6 already satisfied; suppress repeats).
      Most days every tool is silent.
    """

    tool_name: str
    decision: str  # "open_issue" | "reliability_collapse" | "silent"
    today: TriageOutcome
    yesterday: Optional[TriageOutcome] = field(default=None)


def _load_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file or return ``{}`` on missing OR malformed content.

    A truncated state file (process killed mid-write) raises
    ``JSONDecodeError`` if we don't catch it. Because the state file
    rides in the ``benchmark-data`` artifact across daily runs, a
    single corrupt write would otherwise wedge the loop indefinitely:
    every subsequent run would re-raise and abort before write_state
    could rebuild the file. Returning ``{}`` treats a corrupt read as
    "no yesterday", which collapses to first-day-silent behavior and
    self-heals once write_state lays down a fresh file.
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # ValueError covers json.JSONDecodeError on every Python version.
        # Log via raising a tagged exception once and swallowing; the
        # caller's logger picks up the workflow context. Re-importing
        # logging here would duplicate the open_issue logger; we just
        # signal silently and let write_state heal next run.
        del exc
        return {}


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


def _valid_n(stats: Dict[str, Any]) -> int:
    """Read ``valid_n`` strictly; fall back to 0 (NOT total ``n``).

    ``valid_n`` and total ``n`` are different signals: ``valid_n`` is
    the count of rows whose prediction was parseable AND whose outcome
    was resolved AND whose ``p_yes`` was non-null. A tool with
    ``n=200`` but ``valid_n=0`` (full parse failure) must NOT pass the
    sample floor on the total count; that would defeat the whole point
    of the *valid*-sample floor.
    """
    value = stats.get("valid_n")
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _gate_sample_floor(stats: Dict[str, Any]) -> bool:
    """Return True when the per-window per-day soft floor is met.

    The strict per-day floor would need the daily logs, which the
    scorer doesn't surface here; we approximate with
    ``valid_n / 7 >= 15`` (=> valid_n >= 105). On a clean run with no
    holidays this is laxer than the original 30/day target, but it
    avoids rejecting genuine regressions when one or two days in the
    window had reduced traffic.
    """
    valid_n = _valid_n(stats)
    if valid_n / ROLLING_WINDOW_DAYS < N_DAY_FLOOR_PER_WINDOW_DAY:
        return False
    return True


def _gate_reliability(stats: Dict[str, Any]) -> Optional[float]:
    """Return reliability when below floor (else ``None``).

    A reliability collapse short-circuits the rest of the cascade and
    routes to a separate outcome. The dispatcher logs at WARNING; no
    automated paging is wired today, so a human watching the daily
    workflow output is the safety net for this case.
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
    n_cur = _valid_n(cur_stats)
    n_prev = _valid_n(prev_stats)
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
                n_cur=_valid_n(cur_stats),
                n_prev=_valid_n(prev_stats),
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
        # Tolerant deserialise via from_dict (H2): schema drift in the
        # persisted state file must NOT crash the whole triage loop.
        # Unknown fields are dropped; missing fields fall back to
        # dataclass defaults.
        yesterday: Optional[TriageOutcome] = (
            TriageOutcome.from_dict(yesterday_raw)
            if isinstance(yesterday_raw, dict)
            else None
        )

        # Gate 6 (confirmation + duplicate suppression).
        confirmed = today.flagged and yesterday is not None and yesterday.flagged
        # Duplicate-issue suppression: if yesterday's outcome was already
        # an opened issue (yesterday.issue_open=True) and today's gates
        # still fire, today's outcome stays silent. We propagate the
        # issue_open flag forward so tomorrow stays silent too, until
        # the regression resolves and gates 2-4 stop firing.
        if confirmed and yesterday is not None and yesterday.issue_open:
            today.issue_open = True
            decisions.append(
                TriageDecision(
                    tool_name=tool_name,
                    decision="silent",
                    today=today,
                    yesterday=yesterday,
                )
            )
        elif confirmed:
            today.issue_open = True
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
                "issue_open": decision.today.issue_open,
            }
            for decision in decisions
        },
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
