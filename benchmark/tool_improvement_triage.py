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
"""Daily tool-improvement triage on rolling-window Brier scores.

Runs as a step inside ``benchmark_flywheel.yaml`` after the per-platform
``Analyze`` job. Reads two per-platform rolling-window score files
(produced by ``scorer.py``) plus a cross-day state file carried in the
``benchmark-data`` artifact. For each tool, applies a deterministic gate
cascade:

1. Platform = whichever platforms are listed in ``ENABLED_PLATFORMS``.
   Today that is ``["polymarket"]``; Omen is excluded because its
   Brier is confounded by the on-chain ``jury-resolve-market-v1`` mech
   tool's quality (a tool-side fix cannot reliably address an
   Omen-only regression while the jury is in the loop). The triage
   iterates each enabled platform separately and files one issue per
   (tool, platform) regression.
2. Trigger (either is sufficient), evaluated on the first window
   source in the assessment ladder that meets the sample floor:
   - Regression: ``Brier_cur - Brier_prev > 0.040`` AND
     ``sign(delta Brier) == sign(delta log_loss)``. Calibrated
     2026-05-29 against 26 days of CI data: 0.040 alone yields ~1
     issue/week per active tool.
   - Level: the tool's **Brier Skill Score** is below
     ``BSS_LEVEL_FLOOR`` -- it is materially worse than its base-rate
     reference (``BSS = 1 - brier / base_rate_brier``, computed per
     group by ``scorer.py``). This is *market-relative*: unlike an
     absolute Brier floor, it does not keep flagging a tool that is at
     the ceiling of an efficient, hard-to-beat market. Falls back to
     the legacy absolute floor (``Brier_cur > 0.25``) only when the
     skill score is missing or non-finite.
3. Assessment ladder (anti-stall; a thin week no longer means silence):
   a. calendar 7d windows (``W-1``) when ``valid_n >= 105`` on both;
   b. else production count windows (last ``COUNT_WINDOW_N`` valid rows
      vs the preceding ``COUNT_WINDOW_N``, ``C-1``, 45d age cap);
   c. else tournament count windows (``T-1``, rows from
      ``results/tournament_scored.jsonl`` -- shadow data, labeled as
      such in the issue and never blended with production);
   d. else ``insufficient_data`` -> a ``deployment-review`` widen-sample
      note (add to tournament / route traffic / retire, with roster
      status), guarded by a ``COUNT_RAMP_DAYS`` ramp for newly-deployed
      tools and a ``DR_RECENT_CLOSE_DAYS`` cooldown after a manual
      close.
4. ``reliability >= 0.80``, checked only on a floor-sized calendar
   sample (on a thin slice the number is noise and must not preempt
   the count/tournament fallback); collapses route to a WARNING log.

Duplicate-issue suppression: once a tool has an open ``tool-improvement``
issue on GitHub, the triage stays silent for that tool. ``prior_open``
is seeded from the live ``gh issue list`` query at the start of
``main()`` rather than from yesterday's state file alone; a PR that
fixes the issue (closing it) re-arms the trigger after the
``RECENT_CLOSE_DAYS`` cooldown elapses, and a PR still under review
keeps the tool quiet for as long as the issue remains open.

Recently-closed cooldown: any ``(tool, platform)`` with a close in the
last ``RECENT_CLOSE_DAYS`` days is silenced regardless of trigger. The
log line per tool with a close in the recent past explicitly states
whether the cooldown is ACTIVE or has ELAPSED, so an operator can see
why a known-bad tool isn't firing today (or why it is firing again
after a recent close).

Lineage-descendant routing: a tool that fires but already has a merged
fix variant in ``tool_lineage.json`` (i.e. it is some other tool's
``parent``) is NOT a new fix request -- repeated re-fixing of the same
ancestor without promoting the result is the churn this stops. Such a
fire is routed to ``descendant_exists`` and emitted as a *visible*
promotion-review note under the ``tool-promotion-review`` label (its own
dedup, posted once per ``(tool, platform)``). That label is deliberately
NOT ``tool-improvement``, so the label-routed coding agent is not
invoked; the note asks a human to promote an existing variant (judged on
BSS-vs-market, not raw Brier) or accept the lineage is at its ceiling and
retire it. By design there is no ``RECENT_CLOSE_DAYS`` cooldown for these
notes: the signal is "the deployed ancestor is still underperforming and
its fix is unpromoted", so a note closed without changing that condition
re-posts on the next firing run (until the variant is promoted or the
tool retired), rather than going quiet on a stale close.

For each ``open_issue`` decision, the script calls ``gh issue create``
with the ``tool-improvement`` label. The label routes the issue to
``tool-improvement-agent`` in the agent-skills monorepo. A failed
``gh`` call surfaces as a CI red (non-zero exit) so a silently-lost
dispatch does not also wedge tomorrow's duplicate-suppression.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ENABLED_PLATFORMS = ["polymarket"]
BRIER_REGRESSION_THRESHOLD = 0.040
BRIER_LEVEL_THRESHOLD = 0.25
# Time-based silence applied to every closed ``tool-improvement`` issue:
# any new fire for the same ``(tool, platform)`` is suppressed for this many
# days after the most recent close (merge OR manual), regardless of trigger.
# Replaces the previous Brier-band re-arm hysteresis: a partial fix that
# leaves the Brier above ``BRIER_LEVEL_THRESHOLD`` is allowed to re-fire
# the day after the window elapses, rather than being silenced forever by
# a stuck cooldown marker (the dead-zone failure mode of the re-arm band).
RECENT_CLOSE_DAYS = 1
VALID_N_PER_WINDOW_FLOOR = 105
RELIABILITY_FLOOR = 0.80
ROLLING_WINDOW_DAYS = 7
# Market-relative level floor. The tool is flagged when its Brier Skill
# Score (BSS = 1 - brier / base_rate_brier, already computed per group by
# the scorer) drops below this floor -- i.e. it is materially worse than
# simply predicting the base rate. Unlike an absolute Brier floor, BSS
# adapts to market difficulty: on an efficient market whose own Brier is
# high, an absolute floor sits BELOW the achievable frontier and fires
# forever, so no prompt fix can ever clear it. -0.10 keeps roughly the same
# sensitivity as the legacy 0.25 Brier floor on the observed base-rate
# *Brier* of ~0.22 (yes-rate ~0.36 -> yes*(1-yes) ~= 0.22): the floor then
# fires at Brier ~= 0.22 * 1.10 ~= 0.24. NB the referent is the base-rate
# Brier, NOT the yes-rate itself.
BSS_LEVEL_FLOOR = -0.10
# --- Count-window fallback (anti-stall) ---------------------------------
# A tool whose calendar windows are below VALID_N_PER_WINDOW_FLOOR is not
# assessable on the 7d cadence, and under calendar-only gates it stays
# silent FOREVER once its traffic thins -- the loop stalls exactly on the
# newest (least-requested) tools. The fallback re-evaluates such a tool on
# its last COUNT_WINDOW_N valid rows vs the preceding COUNT_WINDOW_N (same
# regression / level triggers, same downstream gates), so a starved tool is
# judged whenever it has accumulated enough rows, however long that took.
COUNT_WINDOW_N = VALID_N_PER_WINDOW_FLOOR
# Age cap for count-window rows: a window stitched from months-old rows
# describes a tool that no longer exists (prompt/model drift), so rows
# older than this never enter a count window. Tools whose recent rows
# cannot fill even one window surface as ``insufficient_data`` and are
# routed to a human deployment-review note instead of silence.
COUNT_WINDOW_MAX_AGE_DAYS = 45
# A tool younger than this (first valid row within the ramp period) is
# accruing data normally, not starved: filing a widen-sample note on a
# variant deployed three days ago would page a human about a non-problem.
# Below-count-floor tools inside the ramp stay silent (reason="ramping").
COUNT_RAMP_DAYS = 7
# Cooldown for deployment-review notes after a manual close: without it the
# insufficient_data condition re-fires deterministically every daily run
# and re-files the note the day after a human closed it.
DR_RECENT_CLOSE_DAYS = 7
# Row-level tournament results (written by the flywheel's tournament arm).
# Tournament rows shadow-score the SAME daily questions without serving
# on-chain traffic, so they are the riskless data source for tools whose
# production traffic is too thin to assess. They are a SEPARATE slice from
# production (different execution contract) -- never blended into one
# metric, always labeled T-1/T-2 when they drive a fire.
TOURNAMENT_SCORED_NAME = "tournament_scored.jsonl"
TOURNAMENT_TOOLS_PATH = Path(__file__).resolve().parent / "tournament_tools.json"

DEFAULT_REPO = "valory-xyz/mech-predict"
DEFAULT_LABEL = "tool-improvement"
# A tool that fires a Brier signal but ALREADY has a merged fix variant in
# tool_lineage.json is not a new fix request -- repeated re-fixing of the
# same ancestor without promoting the result is the churn this routing
# stops. Such a fire is routed to a visible promotion-review note under this
# separate label so the coding agent (label-routed on ``tool-improvement``)
# is NOT invoked. The same label also carries the widen-sample notes for
# tools below the count floor: it is the single human-decision funnel for
# anything that needs a deployment action (promote / route traffic /
# retire) rather than a code fix.
PROMOTION_LABEL = "deployment-review"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
LINEAGE_PATH = Path(__file__).resolve().parent.parent / "tool_lineage.json"

log = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict[str, Any]:
    """Return JSON contents of ``path`` or ``{}`` if missing/corrupt."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# Parses ``[tool-improvement] `<tool>`: <metric> on <platform> W-1`` where
# <metric> is "Brier regression", "Brier above level", or "BSS below floor".
# Backticks around the tool are required; the platform is captured from the
# trailing ``on <platform> W-1`` (calendar windows) or ``on <platform> C-1``
# (count-window fallback) segment so suppression can key on the
# (tool, platform) pair regardless of the metric wording or window kind. A
# manually-filed issue that omits either segment logs a warning and is skipped.
_TITLE_RE = re.compile(r"\[tool-improvement\]\s+`([^`]+)`.*\bon\s+(\S+)\s+[WCT]-1\b")
# Deployment-review notes (promotion + widen-sample) use their own label +
# title so they dedup independently of the fix issues and never match the
# coding agent's ``tool-improvement`` title parser. Anchored to end-of-title
# (the platform is the last token); the generated-title <-> regex contract is
# pinned by TestPromoTitleRoundTrip.
_PROMO_TITLE_RE = re.compile(r"\[deployment-review\]\s+`([^`]+)`.*\bon\s+(\S+)\s*$")
# Migration shim (see main()): open notes filed before the label rename still
# carry the legacy label + title prefix and must keep suppressing duplicates.
_LEGACY_PROMOTION_LABEL = "tool-promotion-review"
_LEGACY_PROMO_TITLE_RE = re.compile(
    r"\[tool-promotion-review\]\s+`([^`]+)`.*\bon\s+(\S+)\s*$"
)


def _load_lineage_children(path: Path = LINEAGE_PATH) -> Dict[str, List[str]]:
    """Map each tool to its variant descendants (children) from ``tool_lineage.json``."""
    # A tool present as some other tool's ``parent`` has at least one merged
    # fix variant (the ledger is updated on merge), so its presence as a key
    # here means "a fix for this tool already exists". Warn (rather than
    # silently returning {}) when the file EXISTS but does not parse: an empty
    # children map disables descendant routing, so a corrupt ledger would
    # silently resume the very re-fixing churn this stops.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as exc:
        log.warning(
            "tool_lineage.json exists but failed to parse (%s); descendant "
            "routing DISABLED this run -- promotion notes will not fire.",
            exc,
        )
        data = {}
    children: Dict[str, List[str]] = {}
    for name, meta in (data.get("tools") or {}).items():
        parent = (meta or {}).get("parent")
        if parent:
            children.setdefault(parent, []).append(name)
    return children


def _descendants(tool: str, children: Dict[str, List[str]]) -> List[str]:
    """All transitive variant descendants of ``tool`` (sorted, de-duped)."""
    # Walks the parent->children map so a chained lineage
    # (factual_research -> v1 -> v2 -> v3) surfaces the leaf tips, not just the
    # direct children; ``seen`` (seeded with the root) guards against a cycle.
    out: List[str] = []
    seen: set = {tool}  # seed with the root so a back-edge can't re-add it
    stack = list(children.get(tool, []))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        out.append(node)
        stack.extend(children.get(node, []))
    return sorted(out)


def _below_level(*, brier_cur: Optional[float], bss_cur: Optional[float]) -> bool:
    """Level trigger: is the tool materially worse than its base-rate reference?"""
    # Keyword-only: both params are Optional[float], so a positional swap would
    # type-check silently.
    # Prefer the market-relative Brier Skill Score (adapts to market
    # difficulty); fall back to the legacy absolute Brier floor when the skill
    # score is missing (pre-BSS score files / unit fixtures) OR not finite.
    # A NaN BSS is plausible (scorer's base_rate_brier = yes*(1-yes) is 0 on an
    # all-one-outcome window, and json round-trips NaN); `NaN < floor` is
    # False, which would silently disable the level trigger -- so treat a
    # non-finite BSS the same as missing and use the absolute Brier fallback.
    if bss_cur is not None and math.isfinite(bss_cur):
        return bss_cur < BSS_LEVEL_FLOOR
    if bss_cur is not None:
        log.warning(
            "brier_skill_score is non-finite (%r); falling back to the "
            "absolute Brier floor for the level trigger.",
            bss_cur,
        )
    if brier_cur is None:
        return False
    return brier_cur > BRIER_LEVEL_THRESHOLD


def _open_issue_tools(
    repo: str, label: str, title_re: "re.Pattern[str]" = _TITLE_RE
) -> Optional[List[Tuple[str, str]]]:
    """Return (tool, platform) pairs for open issues; ``None`` on gh error, ``[]`` on zero."""
    # Distinguishing None (transient gh-CLI failure) from [] (gh
    # succeeded with zero matching open issues) lets the caller fall
    # back to the state file on transient errors, rather than treating
    # the empty result as "no issues open" and refiling every regression.
    cmd = [
        "gh",
        "issue",
        "list",
        "--repo",
        repo,
        "--label",
        label,
        "--state",
        "open",
        "--json",
        "title",
        # gh has no "unlimited" sentinel: --limit 0 is rejected with
        # rc=1 ("invalid limit: 0"). Use a finite cap large enough that
        # we never hit it in practice (orders of magnitude above the
        # realistic max of simultaneously-open tool-improvement issues).
        "--limit",
        "1000",
    ]
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("gh issue list failed (%s); falling back to state file only", exc)
        return None
    if r.returncode != 0:
        log.warning(
            "gh issue list rc=%d stderr=%r; falling back to state file only",
            r.returncode,
            r.stderr[:200],
        )
        return None
    try:
        rows = json.loads(r.stdout or "[]")
    except ValueError as exc:
        log.warning(
            "gh issue list returned unparseable JSON (%s); "
            "falling back to state file only",
            exc,
        )
        return None
    pairs: List[Tuple[str, str]] = []
    for row in rows:
        title = row.get("title") or ""
        m = title_re.search(title)
        if m:
            pairs.append((m.group(1), m.group(2)))
        else:
            log.warning(
                "%s issue title did not match the expected format; "
                "skipping (no suppression): %r",
                label,
                title[:120],
            )
    return pairs


def _closed_issue_pairs(
    repo: str,
    label: str,
    title_re: "re.Pattern[str]" = _TITLE_RE,
) -> Optional[List[Tuple[str, str, datetime]]]:
    """Return `(tool, platform, closed_at)` triples for closed tool-improvement issues.

    Caller buckets the result by ``closed_at`` against ``RECENT_CLOSE_DAYS``
    to drive the recently-closed silence (issues within the window suppress
    re-fire of the same ``(tool, platform)``).

    Returns ``None`` on transient gh-CLI failure so callers can skip the
    cooldown rather than open a noisy issue against a half-resolved state.

    :param repo: GitHub ``owner/repo`` slug.
    :param label: issue label to filter on (e.g. ``"tool-improvement"``).
    :param title_re: title regex whose two capture groups yield
        ``(tool, platform)``; defaults to the tool-improvement format.
    :return: list of ``(tool, platform, closed_at)`` triples, or ``None`` on
        gh error. ``closed_at`` is a timezone-aware UTC ``datetime``.
    """
    cmd = [
        "gh",
        "issue",
        "list",
        "--repo",
        repo,
        "--label",
        label,
        "--state",
        "closed",
        "--json",
        "title,closedAt",
        "--limit",
        "1000",
    ]
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning(
            "gh issue list (closed) failed (%s); skipping recently-closed cooldown",
            exc,
        )
        return None
    if r.returncode != 0:
        log.warning(
            "gh issue list (closed) rc=%d stderr=%r; "
            "skipping recently-closed cooldown",
            r.returncode,
            r.stderr[:200],
        )
        return None
    try:
        rows = json.loads(r.stdout or "[]")
    except ValueError as exc:
        log.warning(
            "gh issue list (closed) returned unparseable JSON (%s); "
            "skipping recently-closed cooldown",
            exc,
        )
        return None
    triples: List[Tuple[str, str, datetime]] = []
    for row in rows:
        title = row.get("title") or ""
        m = title_re.search(title)
        if not m:
            log.warning(
                "closed tool-improvement issue title did not match the expected "
                "format; skipping (no suppression): %r",
                title[:120],
            )
            continue
        closed_at_raw = row.get("closedAt") or ""
        try:
            closed_at = datetime.fromisoformat(closed_at_raw.replace("Z", "+00:00"))
        except ValueError:
            log.warning(
                "closed tool-improvement issue has unparseable closedAt=%r; "
                "skipping",
                closed_at_raw,
            )
            continue
        triples.append((m.group(1), m.group(2), closed_at))
    return triples


def _most_recent_close_per_tool(
    closed_issues: Optional[List[Tuple[str, str, datetime]]],
    platform: str,
) -> Dict[str, datetime]:
    """Return ``{tool: most_recent_close_at}`` for ``platform``.

    Per ``(tool, platform)`` we keep only the most recent close timestamp so a
    long-lived tool with several historical closes is summarised by its
    latest close (the one that governs the cooldown window).

    :param closed_issues: list of ``(tool, platform, closed_at)`` triples from
        the live gh query, or ``None`` if the query was skipped / failed.
    :param platform: the platform under triage; entries on other platforms are
        ignored.
    :return: mapping from tool name to its most recent close ``datetime``
        (timezone-aware UTC). Empty dict if ``closed_issues`` is ``None`` or
        has no entries for ``platform``.
    """
    if closed_issues is None:
        return {}
    most_recent: Dict[str, datetime] = {}
    for tool_name, tool_platform, closed_at in closed_issues:
        if tool_platform != platform:
            continue
        existing = most_recent.get(tool_name)
        if existing is None or closed_at > existing:
            most_recent[tool_name] = closed_at
    return most_recent


def _log_cooldown_status(
    most_recent_close: Dict[str, datetime],
    cur_tools: List[str],
    platform: str,
    now_dt: datetime,
    n_days: int,
) -> None:
    """Log per-tool cooldown status for any current tool with a recent close.

    For each tool present in the current rolling scores AND in
    ``most_recent_close``, emit one INFO log line stating whether the
    ``RECENT_CLOSE_DAYS`` cooldown is ACTIVE or has ELAPSED. The status is
    bounded to closes within ``2 * n_days + 7`` days so the operator sees the
    transition without permanent log noise from ancient closes.

    :param most_recent_close: ``{tool: closed_at}`` from
        :func:`_most_recent_close_per_tool` for the current platform.
    :param cur_tools: tool names present in today's rolling scores; the log is
        scoped to these so retired tools don't add noise.
    :param platform: platform under triage (rendered in the log message).
    :param now_dt: reference "now" used to compute days-since-close.
    :param n_days: the ``RECENT_CLOSE_DAYS`` value in effect.
    """
    horizon = timedelta(days=2 * n_days + 7)
    cooldown = timedelta(days=n_days)
    cutoff = now_dt - cooldown
    cur_tool_set = set(cur_tools)
    for tool_name, last_close in sorted(most_recent_close.items()):
        if tool_name not in cur_tool_set:
            continue
        age = now_dt - last_close
        if age > horizon:
            continue
        # Mirror the gate's predicate exactly (``last_close >= cutoff``) so
        # the log never says ACTIVE on a close the gate is about to let
        # fire. Floored ``age.days`` is fine for the human-readable count
        # but must not drive the status decision.
        status = "ACTIVE" if last_close >= cutoff else "ELAPSED"
        log.info(
            "cooldown %s: tool=%r platform=%r closed %d day(s) ago (N=%d)",
            status,
            tool_name,
            platform,
            age.days,
            n_days,
        )


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp into a UTC-aware datetime (``None`` if invalid)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _load_count_rows(
    logs_dir: Path, platform: str, now: datetime
) -> Dict[str, List[Dict[str, Any]]]:
    """Per-tool valid rows for ``platform`` from the daily JSONLs, oldest first.

    Rows are deduped by ``row_id`` (the daily files are cumulative
    snapshots), filtered by the scorer's validity predicate
    (``prediction_parse_status == "valid"``, resolved outcome, non-null
    ``p_yes``) plus the ``COUNT_WINDOW_MAX_AGE_DAYS`` age cap, and sorted
    by ``predicted_at`` -- the exact predicate the issue body documents so
    the coding agent's Step-1 reproduction sees the same slice.

    :param logs_dir: directory containing the daily production-log JSONLs.
    :param platform: platform to filter rows to (e.g. ``"polymarket"``).
    :param now: reference "now" for the age cap.
    :return: per-tool lists of valid rows, sorted oldest-first.
    """
    dedup: Dict[str, Dict[str, Any]] = {}
    for path in sorted(logs_dir.glob("production_log_*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            row_id = row.get("row_id")
            if row_id:
                dedup[row_id] = row
    cutoff = now - timedelta(days=COUNT_WINDOW_MAX_AGE_DAYS)
    by_tool: Dict[str, List[Dict[str, Any]]] = {}
    for row in dedup.values():
        if row.get("platform") != platform:
            continue
        # The calendar by_tool the fallback substitutes for is the scorer's
        # PRODUCTION-mode partition; tournament rows must not leak into
        # count windows or the two paths diverge on mixed logs.
        if "tournament" in (row.get("mode") or ""):
            continue
        if row.get("prediction_parse_status") != "valid":
            continue
        if row.get("final_outcome") is None or row.get("p_yes") is None:
            continue
        predicted_at = _parse_ts(row.get("predicted_at"))
        if predicted_at is None or predicted_at < cutoff:
            continue
        tool_name = row.get("tool_name")
        if tool_name:
            by_tool.setdefault(tool_name, []).append(row)
    for rows in by_tool.values():
        rows.sort(key=lambda r: r.get("predicted_at") or "")
    return by_tool


def _load_tournament_count_rows(
    scored_path: Path, platform: str, now: datetime
) -> Dict[str, List[Dict[str, Any]]]:
    """Per-tool valid TOURNAMENT rows for ``platform``, oldest first.

    Same validity predicate and age cap as :func:`_load_count_rows`, but
    sourced from the flywheel's row-level tournament output and keeping
    ONLY tournament-mode rows. Missing file -> ``{}`` (no tournament arm).

    :param scored_path: path to ``tournament_scored.jsonl``.
    :param platform: platform to filter rows to.
    :param now: reference "now" for the age cap.
    :return: per-tool lists of valid tournament rows, sorted oldest-first.
    """
    if not scored_path.is_file():
        return {}
    dedup: Dict[str, Dict[str, Any]] = {}
    try:
        text = scored_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        row_id = row.get("row_id") or row.get("deliver_id")
        if row_id:
            dedup[row_id] = row
    cutoff = now - timedelta(days=COUNT_WINDOW_MAX_AGE_DAYS)
    by_tool: Dict[str, List[Dict[str, Any]]] = {}
    for row in dedup.values():
        if row.get("platform") != platform:
            continue
        if "tournament" not in (row.get("mode") or "tournament"):
            continue
        if row.get("prediction_parse_status") != "valid":
            continue
        if row.get("final_outcome") is None or row.get("p_yes") is None:
            continue
        predicted_at = _parse_ts(row.get("predicted_at"))
        if predicted_at is None or predicted_at < cutoff:
            continue
        tool_name = row.get("tool_name")
        if tool_name:
            by_tool.setdefault(tool_name, []).append(row)
    for rows in by_tool.values():
        rows.sort(key=lambda r: r.get("predicted_at") or "")
    return by_tool


def _tournament_roster(path: Path = TOURNAMENT_TOOLS_PATH) -> set:
    """Tool names currently in the tournament (dash/underscore-insensitive).

    :param path: path to ``tournament_tools.json``.
    :return: set of normalized (dashes as underscores) tool names.
    """
    data = _load_json(path)
    return {str(k).replace("-", "_") for k in data} if isinstance(data, dict) else set()


def _in_roster(tool: str, roster: set) -> bool:
    """True when ``tool`` is in the tournament roster (naming-insensitive).

    :param tool: tool name (dash or underscore spelling).
    :param roster: output of :func:`_tournament_roster`.
    :return: membership flag.
    """
    return tool.replace("-", "_") in roster


def _window_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scorer-compatible metrics for a count window of validity-gated rows.

    Emits the same keys the scorer's ``by_tool`` entries carry (``brier``,
    ``log_loss``, ``brier_skill_score``, ``valid_n``) so the trigger cascade
    in :func:`triage` consumes calendar and count windows interchangeably.
    ``brier_skill_score`` uses the window's own base-rate Brier
    (``yes_rate * (1 - yes_rate)``) as reference, matching the scorer's
    group-level BSS; an all-one-outcome window yields NaN, which
    :func:`_below_level` already treats as "fall back to the absolute floor".

    :param rows: validity-gated rows of one count window (non-empty).
    :return: metrics dict keyed like a scorer ``by_tool`` entry, plus the
        window's ``span_start`` / ``span_end`` dates.
    """
    n = len(rows)
    eps = 1e-15
    brier_sum = 0.0
    log_loss_sum = 0.0
    yes = 0
    for row in rows:
        outcome = 1.0 if row["final_outcome"] else 0.0
        p_yes = float(row["p_yes"])
        brier_sum += (p_yes - outcome) ** 2
        clipped = min(max(p_yes, eps), 1 - eps)
        log_loss_sum += -(
            outcome * math.log(clipped) + (1 - outcome) * math.log(1 - clipped)
        )
        yes += int(outcome)
    brier = brier_sum / n
    yes_rate = yes / n
    base_rate_brier = yes_rate * (1 - yes_rate)
    bss = 1 - brier / base_rate_brier if base_rate_brier > 0 else float("nan")
    return {
        "brier": round(brier, 4),
        "log_loss": round(log_loss_sum / n, 4),
        "brier_skill_score": round(bss, 4) if math.isfinite(bss) else bss,
        "valid_n": n,
        "span_start": (rows[0].get("predicted_at") or "")[:10],
        "span_end": (rows[-1].get("predicted_at") or "")[:10],
    }


def _count_window_stats(
    tool_rows: List[Dict[str, Any]], n: int = COUNT_WINDOW_N
) -> Dict[str, Any]:
    """Count-window stats: C-1 = last ``n`` rows, C-2 = the preceding ``n``.

    ``{"insufficient": True, "total": k}`` when the tool cannot fill even
    one window (k < n) -- routed by main() to a deployment-review
    widen-sample note. When only C-1 fills (n <= total < 2n), ``prev`` is
    ``None`` and the trigger cascade runs level-only (no regression check
    without a disjoint baseline).

    :param tool_rows: one tool's valid rows, sorted oldest-first.
    :param n: rows per count window.
    :return: stats dict (``insufficient``/``total``/``cur``/``prev``/
        ``spans``/``first_predicted_at``).
    """
    total = len(tool_rows)
    first_predicted_at = tool_rows[0].get("predicted_at") if tool_rows else None
    if total < n:
        return {
            "insufficient": True,
            "total": total,
            "first_predicted_at": first_predicted_at,
        }
    cur = _window_metrics(tool_rows[-n:])
    prev = _window_metrics(tool_rows[-2 * n : -n]) if total >= 2 * n else None
    spans = {"cur": f"{cur['span_start']} - {cur['span_end']}"}
    if prev:
        spans["prev"] = f"{prev['span_start']} - {prev['span_end']}"
    return {
        "insufficient": False,
        "total": total,
        "cur": cur,
        "prev": prev,
        "spans": spans,
    }


def triage(
    cur: Dict[str, Any],
    prev: Dict[str, Any],
    prior_state: Dict[str, Any],
    platform: str = "polymarket",
    open_now: Optional[List[Any]] = None,
    closed_issues: Optional[List[Tuple[str, str, datetime]]] = None,
    now: Optional[datetime] = None,
    lineage_children: Optional[Dict[str, List[str]]] = None,
    count_windows: Optional[Dict[str, Dict[str, Any]]] = None,
    tournament_windows: Optional[Dict[str, Dict[str, Any]]] = None,
    tournament_roster: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Apply the gate cascade to ``cur`` vs ``prev`` and return one decision dict per tool."""
    # ``count_windows`` (per-tool output of ``_count_window_stats``) feeds
    # the anti-stall fallback: a tool below the calendar sample floor is
    # re-evaluated on its count windows instead of going silent. ``None`` /
    # missing tool preserves the legacy calendar-only behavior.
    # pylint: disable=too-many-locals,too-many-statements
    # ``lineage_children`` maps a tool to its merged fix variants (from
    # tool_lineage.json); a tool that fires but already has a variant is
    # routed to ``descendant_exists`` (a visible promotion note) instead of
    # a redundant fix issue -- see the module docstring.
    children = lineage_children or {}
    # ``open_now`` may be a list of bare tool names (legacy single-
    # platform callers) OR a list of ``(tool, platform)`` tuples (multi-
    # platform live-gh callers). Tuple entries are filtered to the
    # current ``platform`` so an open issue on tool ``X`` on platform
    # ``A`` does not suppress the same tool on platform ``B``.
    # Suppression source-of-truth: ``open_now`` (the live gh result)
    # when provided; falls back to ``prior_state`` only when the caller
    # did not run the live query. Closing the GitHub issue re-arms the
    # trigger after the ``RECENT_CLOSE_DAYS`` cooldown elapses (or
    # immediately if the cooldown is already past).
    if open_now is not None:
        prior_open = {}
        for entry in open_now:
            if isinstance(entry, tuple):
                tool_name, tool_platform = entry
                if tool_platform == platform:
                    prior_open[tool_name] = True
            else:
                prior_open[entry] = True
    else:
        prior_open = {
            t: v.get("issue_open", False)
            for t, v in (prior_state.get("by_tool") or {}).items()
        }
    # Recently-closed silence: any issue closed within the last
    # RECENT_CLOSE_DAYS suppresses re-fire of the same (tool, platform)
    # regardless of trigger. Replaces the previous Brier-band re-arm
    # hysteresis: a partial fix that leaves the Brier above
    # ``BRIER_LEVEL_THRESHOLD`` is allowed to re-fire the day after the
    # window elapses, rather than being silenced forever by a stuck
    # cooldown marker (the dead-zone failure mode of the re-arm band).
    # ``closed_issues=None`` (caller skipped the query) means we apply no
    # silence -- same fail-open principle as ``open_now=None``.
    now_dt = now or datetime.now(timezone.utc)
    cutoff = now_dt - timedelta(days=RECENT_CLOSE_DAYS)
    most_recent_close = _most_recent_close_per_tool(closed_issues, platform)
    recently_closed_set = {
        tool_name
        for tool_name, last_close in most_recent_close.items()
        if last_close >= cutoff
    }
    cur_tools = list((cur.get("by_tool") or {}).keys())
    _log_cooldown_status(
        most_recent_close, cur_tools, platform, now_dt, RECENT_CLOSE_DAYS
    )
    decisions: List[Dict[str, Any]] = []
    # Union with count-window tools: a fully-starved tool (ZERO rows in the
    # current rolling window) never appears in cur.by_tool at all, so
    # calendar-only iteration silenced exactly the worst starvation case.
    # A tool known only from the raw-log count windows enters with empty
    # calendar stats (n_cur=0) and flows into the count fallback /
    # widen-sample routing below.
    cur_by_tool = cur.get("by_tool") or {}
    for tool in sorted(
        set(cur_by_tool) | set(count_windows or {}) | set(tournament_windows or {})
    ):
        c = cur_by_tool.get(tool) or {}
        p = (prev.get("by_tool") or {}).get(tool, {})
        d: Dict[str, Any] = {
            "tool": tool,
            "platform": platform,
            "brier_cur": c.get("brier"),
            "brier_prev": p.get("brier"),
            "n_cur": c.get("valid_n") or 0,
            "n_prev": p.get("valid_n") or 0,
            "reliability_cur": c.get("reliability"),
            "issue_open": False,
        }
        rel = c.get("reliability")
        if (
            rel is not None
            and rel < RELIABILITY_FLOOR
            and d["n_cur"] >= VALID_N_PER_WINDOW_FLOOR
        ):
            # Reliability is only meaningful on a floor-sized sample: on a
            # thin calendar slice (n=3, one parse error -> 0.67) the old
            # ordering declared reliability_collapse and made the count
            # fallback unreachable for exactly the starved tools it exists
            # for. Sub-floor tools fall through to the count path, whose
            # windows are validity-gated anyway.
            d.update(decision="reliability_collapse", reason="reliability_collapse")
            decisions.append(d)
            continue
        if (
            d["n_cur"] < VALID_N_PER_WINDOW_FLOOR
            or d["n_prev"] < VALID_N_PER_WINDOW_FLOOR
        ):
            # Count-window fallback (anti-stall). Rebinding ``c``/``p`` to
            # the count windows lets the whole trigger cascade below run
            # unchanged -- _window_metrics emits the same keys the scorer
            # does. Three sub-cases:
            #   * no count data for this tool -> legacy silent sample_floor;
            #   * below the count floor even cumulatively -> the tool cannot
            #     be assessed at all: surface as ``insufficient_data`` so
            #     main() posts a deployment-review widen-sample note
            #     (silence here is exactly how starved tools stall the loop);
            #   * count windows available -> evaluate the same triggers on
            #     them, tagged ``window_kind: count`` for title/body.
            cw = (count_windows or {}).get(tool)
            tw = (tournament_windows or {}).get(tool)
            d["in_tournament"] = _in_roster(tool, tournament_roster or set())
            d["tournament_total"] = (tw or {}).get("total", 0)
            if not cw and not tw:
                d.update(decision="silent", reason="sample_floor")
                decisions.append(d)
                continue
            # Source preference: production count windows first (the real
            # contract); tournament count windows as the riskless fallback
            # for tools production barely requests. Never blended.
            picked: Optional[Dict[str, Any]] = None
            picked_kind = ""
            if cw and not cw.get("insufficient"):
                picked, picked_kind = cw, "count"
            elif tw and not tw.get("insufficient"):
                picked, picked_kind = tw, "tournament"
            if picked is None:
                first_seen = _parse_ts((cw or tw or {}).get("first_predicted_at"))
                if (
                    first_seen is not None
                    and (now_dt - first_seen).days < COUNT_RAMP_DAYS
                ):
                    # A newly-deployed variant accruing its first rows is
                    # ramping, not starved -- no human page.
                    d.update(
                        decision="silent",
                        reason="ramping",
                        cw_total=(cw or {}).get("total", 0),
                    )
                    decisions.append(d)
                    continue
                d.update(
                    decision="insufficient_data",
                    reason="widen_sample",
                    window_kind="count",
                    cw_total=(cw or {}).get("total", 0),
                )
                decisions.append(d)
                continue
            d["calendar_n_cur"] = d["n_cur"]
            c, p = picked["cur"], picked.get("prev") or {}
            d.update(
                window_kind=picked_kind,
                brier_cur=c.get("brier"),
                brier_prev=p.get("brier"),
                n_cur=c.get("valid_n") or 0,
                n_prev=p.get("valid_n") or 0,
                cw_spans=picked.get("spans"),
                cw_total=picked.get("total"),
            )
        bc, bp = c.get("brier"), p.get("brier")
        if bc is None or (
            bp is None and d.get("window_kind") not in ("count", "tournament")
        ):
            d.update(decision="silent", reason="missing_brier")
            decisions.append(d)
            continue
        if bp is None:
            # Count-window with only C-1 filled: no disjoint baseline, so
            # the regression trigger cannot run -- the level trigger below
            # still can. delta 0.0 keeps the regression branch inert.
            delta_brier = 0.0
        else:
            delta_brier = bc - bp
            d["delta_brier"] = delta_brier
        # Two triggers express the self-improvement loop:
        # - regression: today's Brier worsened by more than the threshold
        #   AND log_loss agrees on the worsening direction. Sign
        #   agreement filters noisy borderline regressions.
        # - level: the tool's Brier Skill Score is below the market-relative
        #   floor -- it is materially worse than its base-rate reference.
        #   Using BSS (not an absolute Brier) means an efficient, hard-to-
        #   beat market does not keep an at-ceiling tool flagged forever
        #   (an absolute floor sits below the achievable frontier there).
        #   Level is checked independently of sign-agreement.
        bss_cur = c.get("brier_skill_score")
        d["bss_cur"] = bss_cur
        d["bss_prev"] = p.get("brier_skill_score")
        level_hit = _below_level(brier_cur=bc, bss_cur=bss_cur)
        regressed = False
        if delta_brier > BRIER_REGRESSION_THRESHOLD:
            lc, lp = c.get("log_loss"), p.get("log_loss")
            if lc is None or lp is None:
                if not level_hit:
                    d.update(decision="silent", reason="missing_log_loss")
                    decisions.append(d)
                    continue
                # else fall through with regressed=False; level fires
            elif lc - lp > 0:
                regressed = True
            elif not level_hit:
                d.update(decision="silent", reason="sign_disagreement")
                decisions.append(d)
                continue
        # Trigger label is shared by the recently-closed silence below, the
        # duplicate_suppressed path, and the open_issue path; compute once.
        trigger = "regression" if regressed else "level_floor"
        # Recently-closed silence: a tool whose issue closed within the last
        # RECENT_CLOSE_DAYS is silenced regardless of trigger (regression OR
        # level_floor) and regardless of how it closed (merge OR manual).
        # Gives engineers room before the next page after a close. Gated on
        # prior_open being false because while the issue is still open the
        # duplicate_suppressed path below owns the suppression.
        if (
            (regressed or level_hit)
            and not prior_open.get(tool)
            and tool in recently_closed_set
        ):
            d.update(
                decision="silent",
                reason="recently_closed",
                trigger=trigger,
                issue_open=False,
            )
            decisions.append(d)
            continue
        if not regressed and not level_hit:
            d.update(decision="silent", reason="no_regression")
            decisions.append(d)
            continue
        # Lineage-descendant awareness: a tool that fires but already has a
        # merged fix variant in tool_lineage.json is NOT a new fix request.
        # Repeated re-fixing of the same ancestor without promoting the result
        # is the churn this stops. Route to a visible
        # promotion-review note (emitted in main under PROMOTION_LABEL) so
        # the label-routed coding agent is not invoked. Placed after the
        # cooldown/no-trigger gates (a recently-closed or non-firing tool
        # stays quiet) but before the duplicate/open paths (a stale open fix
        # issue must not mask the promotion signal).
        # ALL transitive descendants, not just direct children: for a chained
        # lineage (e.g. factual_research -> v1 -> v2 -> v3) the promotable tip
        # is a grandchild, so a direct-children-only list would point a
        # reviewer at a stale variant.
        descendants = _descendants(tool, children)
        if descendants:
            d.update(
                decision="descendant_exists",
                reason=trigger,
                trigger=trigger,
                descendants=sorted(descendants),
            )
            decisions.append(d)
            continue
        # Duplicate-issue suppression: if an issue is already open for
        # this tool (regression or level signal), stay silent. The
        # ``trigger`` field records the original signal for state-file
        # bookkeeping and analytics.
        if prior_open.get(tool):
            d.update(
                decision="silent",
                reason="duplicate_suppressed",
                trigger=trigger,
                issue_open=True,
            )
        else:
            d.update(
                decision="open_issue",
                reason=trigger,
                trigger=trigger,
                issue_open=True,
            )
        decisions.append(d)
    return decisions


def write_state(
    state_path: Path, decisions: List[Dict[str, Any]], generated_at: str
) -> None:
    """Persist today's per-tool outcomes to ``state_path`` for tomorrow's run."""
    payload = {
        "generated_at": generated_at,
        "platforms": sorted(
            {d.get("platform", "") for d in decisions if d.get("platform")}
        ),
        "by_tool": {
            d["tool"]: {
                "tool_name": d["tool"],
                "platform": d.get("platform"),
                "decision": d["decision"],
                "reason": d["reason"],
                "trigger": d.get("trigger"),
                "issue_open": d["issue_open"],
                "delta_brier": d.get("delta_brier"),
                "brier_cur": d.get("brier_cur"),
                "brier_prev": d.get("brier_prev"),
                "bss_cur": d.get("bss_cur"),
                "bss_prev": d.get("bss_prev"),
                "descendants": d.get("descendants"),
                "dispatch_failed": d.get("dispatch_failed", False),
                "n_cur": d.get("n_cur"),
                "n_prev": d.get("n_prev"),
                "reliability_cur": d.get("reliability_cur"),
            }
            for d in decisions
        },
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _window_iso(now: datetime, days: int = ROLLING_WINDOW_DAYS) -> Dict[str, str]:
    """Compute W-1 and W-2 ISO ranges relative to ``now`` (window length ``days``)."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    w = timedelta(days=days)
    return {
        "w1_start": (now - w).strftime(fmt),
        "w1_end": now.strftime(fmt),
        "w2_start": (now - 2 * w).strftime(fmt),
        "w2_end": (now - w).strftime(fmt),
    }


def build_issue_title(
    tool: str,
    platform: str = "polymarket",
    reason: str = "regression",
    bss_cur: Optional[float] = None,
    window_kind: str = "calendar",
) -> str:
    """Issue title format for ``tool`` that the agent's Step 1 parser expects."""
    # The agent reads the trigger reason from the body's ``- Trigger:`` line and
    # only the platform from the title's ``on <platform> W-1``/``C-1`` anchor,
    # so the metric wording here is free to match the body: name BSS when the
    # level trigger fired on the skill score, keep the legacy "Brier above
    # level" for the absolute-Brier fallback path. ``_TITLE_RE`` keys on
    # ``on <platform> [WC]-1``, so (tool, platform) dedup is unaffected by the
    # window kind: a calendar fire suppresses a later count fire and vice
    # versa.
    tag = {"count": "C-1", "tournament": "T-1"}.get(window_kind, "W-1")
    if reason == "level_floor":
        if bss_cur is not None and math.isfinite(bss_cur):
            return f"[tool-improvement] `{tool}`: BSS below floor on {platform} {tag}"
        return f"[tool-improvement] `{tool}`: Brier above level on {platform} {tag}"
    return f"[tool-improvement] `{tool}`: Brier regression on {platform} {tag}"


def build_issue_body(
    decision: Dict[str, Any],
    polymarket_stats: Dict[str, Any],
    artifact_url: str,
    window_iso: Dict[str, str],
    platforms_monitored: Optional[List[str]] = None,
) -> str:
    """Render the data-only Markdown issue body for one flagged tool."""
    tool = decision["tool"]
    platform = decision.get("platform", "polymarket")
    monitored = platforms_monitored or [platform]
    other = sorted(p for p in monitored if p != platform)
    cross_ref = (
        f" Other monitored platforms ({', '.join(other)}) are scored in "
        "the same artifact and may be inspected for cross-reference, but "
        "a proposed fix should address this issue's platform."
        if other
        else ""
    )
    pm = json.dumps(polymarket_stats, indent=2, sort_keys=True)
    reason = decision.get("reason", "regression")
    if reason == "level_floor":
        bss_cur = decision.get("bss_cur")
        skill_clause = (
            f"a Brier Skill Score of {bss_cur:+.4f} (below the {BSS_LEVEL_FLOOR:+.2f} "
            "floor -- materially worse than its base-rate reference)"
            if bss_cur is not None and math.isfinite(bss_cur)
            else f"a Brier persistently above {BRIER_LEVEL_THRESHOLD:.2f}"
        )
        headline = (
            f"`{tool}` on {platform} has {skill_clause} in the most recent "
            "7-day window. This is a self-improvement opportunity: the tool "
            "is not actively getting worse, but it is materially below its "
            "reference, so a structural change is worth attempting."
        )
        signal = "level signal"
    else:
        headline = (
            f"`{tool}` on {platform} shows a Brier regression in the most "
            "recent 7-day window."
        )
        signal = "regression signal"
    if decision.get("window_kind") in ("count", "tournament"):
        is_tournament = decision.get("window_kind") == "tournament"
        if reason == "level_floor":
            headline = headline.replace(
                "in the most recent 7-day window",
                "on the count-window check (calendar windows are below the "
                "sample floor for this tool, so the fallback count-window "
                "rule applies)",
            )
        else:
            headline = (
                f"`{tool}` on {platform} shows a Brier regression on the "
                "count-window check (calendar windows are below the sample "
                "floor for this tool, so the fallback count-window rule "
                "applies)."
            )
        spans = decision.get("cw_spans") or {}
        n_prev = decision.get("n_prev") or 0
        brier_prev = decision.get("brier_prev")
        body = _COUNT_BODY_TEMPLATE.format(
            headline=headline,
            tool=tool,
            brier_cur=decision["brier_cur"],
            bss_clause=(
                f", BSS vs base rate: **{decision['bss_cur']:+.4f}**"
                if decision.get("bss_cur") is not None
                and math.isfinite(decision["bss_cur"])
                else ""
            ),
            prev_clause=(
                f"**{brier_prev:.4f}** (n={n_prev})"
                if brier_prev is not None
                else "not available (fewer than 2x the window size accrued)"
            ),
            delta_clause=(
                f"\n- Delta (C-1 - C-2): **{decision['delta_brier']:+.4f}**"
                if decision.get("delta_brier") is not None
                else ""
            ),
            n_cur=decision["n_cur"],
            n_prev=n_prev,
            span_cur=spans.get("cur", "n/a"),
            span_prev=spans.get("prev", "n/a"),
            brier_prev_s=f"{brier_prev:.4f}" if brier_prev is not None else "n/a",
            reason=reason,
            signal=signal,
            platforms=sorted(monitored),
            platform=platform,
            cross_ref=cross_ref,
            calendar_n=decision.get("calendar_n_cur", 0),
            floor=VALID_N_PER_WINDOW_FLOOR,
            max_age=COUNT_WINDOW_MAX_AGE_DAYS,
            window_days=ROLLING_WINDOW_DAYS,
            pm=pm,
            artifact_url=artifact_url,
        )
        if is_tournament:
            # Same structure, tournament-sourced: retag the windows and
            # prepend the source caveat so no reader mistakes the slice
            # for production data.
            body = body.replace("C-1", "T-1").replace("C-2", "T-2")
            body = body.replace(
                "## Windows (count-based",
                "**Source: tournament arm** -- shadow-scores the same daily "
                "questions with no on-chain traffic; separate execution "
                "contract from production, so validate any fix against the "
                "production payload before shipping. Production rows in the "
                "same period were below the sample floor.\n\n"
                "## Windows (tournament count-based",
            )
            body = body.replace(
                "deduped by `row_id`",
                "sourced from `results/tournament_scored.jsonl`, "
                "deduped by `row_id`",
            )
        return body
    return _BODY_TEMPLATE.format(
        headline=headline,
        brier_cur=decision["brier_cur"],
        brier_prev=decision["brier_prev"],
        delta=decision.get("delta_brier") or 0.0,
        n_cur=decision["n_cur"],
        n_prev=decision["n_prev"],
        reason=reason,
        signal=signal,
        platforms=sorted(monitored),
        platform=platform,
        cross_ref=cross_ref,
        w1_start=window_iso["w1_start"],
        w1_end=window_iso["w1_end"],
        w2_start=window_iso["w2_start"],
        w2_end=window_iso["w2_end"],
        pm=pm,
        artifact_url=artifact_url,
    )


_BODY_TEMPLATE = """## Summary

{headline}

- Current 7d (**W-1**) Brier: **{brier_cur:.4f}** (n={n_cur})
- Previous 7d (**W-2**, non-overlapping) Brier: **{brier_prev:.4f}** (n={n_prev})
- Delta: **{delta:+.4f}**
- Trigger: **{reason}**
- Platforms monitored: {platforms}; this issue is scoped to **{platform}**.{cross_ref}

This issue records the {signal}, not a diagnosis. The cause has not been identified.

@valory-coding-agent

## Windows

| Window | Role | Range (predicted_at, UTC) |
|---|---|---|
| **W-1** (current) | the regression to explain | `{w1_start}` - `{w1_end}` |
| **W-2** (previous, disjoint) | comparison baseline for any proposed fix | `{w2_start}` - `{w2_end}` |

## Baseline stats (machine-readable)

```baseline-stats-{platform}
{pm}
```

## Investigation

Artifact: {artifact_url}

Download with `gh run download <run-id> --name benchmark-data`. The artifact contains the daily JSONL logs and `results/scores_{platform}.json`. Reproduce the headline number from the raw rows before forming any hypothesis.

If your investigation concludes that a code change is warranted, **before editing anything** read the [Tool-improvement housekeeping rules](../blob/main/CLAUDE.md#tool-improvement-housekeeping-rules) section of `CLAUDE.md` at the repo root. It is the canonical reference for: in-place edit vs new-version spawn decision, naming convention, the `tool_lineage.json` ledger, and the side-effect file list.
"""


_COUNT_BODY_TEMPLATE = """## Summary

{headline}

- Current count-window (**C-1**, last {n_cur} valid rows) Brier: **{brier_cur:.4f}**{bss_clause}
- Previous count-window (**C-2**, preceding {n_prev} valid rows) Brier: {prev_clause}{delta_clause}
- Trigger: **{reason}** (count-window fallback)
- Platforms monitored: {platforms}; this issue is scoped to **{platform}**.{cross_ref}

This issue records the {signal}, not a diagnosis. The cause has not been identified.

@valory-coding-agent

## Windows (count-based -- reproduce by row count, not calendar)

Predicate: rows with `tool_name == {tool} AND platform == {platform} AND prediction_parse_status == "valid" AND final_outcome != null AND p_yes != null AND predicted_at >= now - {max_age}d`, deduped by `row_id`, sorted by `predicted_at` ascending.

| Window | Definition | predicted_at span | n | Brier |
|---|---|---|---|---|
| **C-1** (current) | last {n_cur} rows of the sorted slice | {span_cur} | {n_cur} | {brier_cur:.4f} |
| **C-2** (previous) | preceding {n_prev} rows | {span_prev} | {n_prev} | {brier_prev_s} |

Calendar context: the {window_days}d calendar windows do not both meet the {floor}-row sample floor for this tool (current-window valid n: {calendar_n}), which is why the calendar triage is silent and the count-window fallback fired.

## Baseline stats (machine-readable)

```baseline-stats-{platform}
{pm}
```

## Investigation

Artifact: {artifact_url}

Download with `gh run download <run-id> --name benchmark-data`. The artifact contains the daily JSONL logs and `results/scores_{platform}.json`. Reproduce the headline number from the raw rows (predicate above) before forming any hypothesis.

If your investigation concludes that a code change is warranted, **before editing anything** read the [Tool-improvement housekeeping rules](../blob/main/CLAUDE.md#tool-improvement-housekeeping-rules) section of `CLAUDE.md` at the repo root. It is the canonical reference for: in-place edit vs new-version spawn decision, naming convention, the `tool_lineage.json` ledger, and the side-effect file list.
"""


def build_widen_sample_title(tool: str, platform: str = "polymarket") -> str:
    """Title for the deployment-review widen-sample note (own dedup key + label)."""
    return (
        f"[deployment-review] `{tool}`: below count floor - widen sample on {platform}"
    )


_WIDEN_SAMPLE_BODY_TEMPLATE = """## Not enough data to judge this tool -- pick one option

`{tool}` on {platform}: **{total} scored predictions in {max_age} days** (need {n}).
Tournament arm: {tournament_line}

1. **Add it to the tournament** -- shadow-scores daily questions, no production risk, assessable in ~1 week.{tournament_done}
2. **Route production traffic** to it.
3. **Retire it.**

No code change requested; the coding agent is not invoked. This note will not repost while open, and stays quiet {dr_days}d after a close.
"""


def build_widen_sample_body(
    decision: Dict[str, Any], platform: str = "polymarket"
) -> str:
    """Render the deployment-review widen-sample note for a below-count-floor tool."""
    in_tournament = decision.get("in_tournament")
    t_total = decision.get("tournament_total") or 0
    if in_tournament and t_total:
        tournament_line = f"IN tournament, {t_total} rows accrued so far."
    elif in_tournament:
        tournament_line = "IN tournament (no scored rows yet)."
    else:
        tournament_line = "NOT in tournament."
    return _WIDEN_SAMPLE_BODY_TEMPLATE.format(
        tool=decision["tool"],
        platform=platform,
        total=decision.get("cw_total", 0),
        max_age=COUNT_WINDOW_MAX_AGE_DAYS,
        n=COUNT_WINDOW_N,
        tournament_line=tournament_line,
        tournament_done=(
            " (already done -- data is accruing)" if in_tournament else ""
        ),
        dr_days=DR_RECENT_CLOSE_DAYS,
    )


def _open_issue(repo: str, label: str, title: str, body: str, dry_run: bool) -> int:
    """Call ``gh issue create`` and return the subprocess return code (0 on success)."""
    if dry_run:
        log.info("DRY-RUN: would open issue title=%s", title)
        return 0
    cmd = [
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--label",
        label,
        "--title",
        title,
        "--body",
        body,
    ]
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        log.error("gh issue create timed out after 30s")
        return 124
    if r.returncode != 0:
        log.error("gh issue create failed (rc=%d): %s", r.returncode, r.stderr)
        return r.returncode
    log.info(
        "Issue opened: %s",
        (r.stdout or "").strip().splitlines()[-1] if r.stdout else "(no url)",
    )
    return 0


def _f(v: Optional[float], p: str) -> str:
    """Format ``v`` with format spec ``p`` or return ``n/a``."""
    return f"{v:{p}}" if v is not None else "n/a"


def _log_decision(d: Dict[str, Any]) -> None:
    """Emit one greppable INFO line summarising a triage decision."""
    log.info(
        "triage %s %s decision=%s reason=%s delta_brier=%s n_cur=%d "
        "brier_cur=%s brier_prev=%s issue_open=%s",
        d.get("platform", "?"),
        d.get("tool", "?"),
        d.get("decision", "?"),
        d.get("reason", "?"),
        _f(d.get("delta_brier"), "+.4f"),
        d.get("n_cur", 0) or 0,
        _f(d.get("brier_cur"), ".4f"),
        _f(d.get("brier_prev"), ".4f"),
        d.get("issue_open", False),
    )


def build_promotion_title(tool: str, platform: str = "polymarket") -> str:
    """Title for the visible promotion-review note (its own dedup key + label)."""
    return f"[deployment-review] `{tool}`: merged fix variant exists on {platform}"


_PROMOTION_BODY_TEMPLATE = """## A fix already exists -- decide on deployment, not more code

`{tool}` on {platform} fired **{trigger}** again (current-window Brier {brier_cur}, BSS {bss_cur}), but merged fix variant(s) already exist:

{descendant_list}

Do NOT request another fix (this note deliberately does not tag the coding agent). Decide:

1. **Deploy the variant to production** if it beats the market (judge on BSS-vs-market / edge, not raw Brier) -- and once production data reaches ~{n} predictions, **remove it from the tournament** to free the slot.
2. **Retire / mute the lineage** if no variant beats the market (it is at its ceiling).

Baseline stats: see the matching `[tool-improvement]` issue (if open) and the `benchmark-data` artifact.
"""


def build_promotion_body(
    decision: Dict[str, Any],
    platform: str = "polymarket",
    tournament_roster: Optional[set] = None,
    tournament_by_tool: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the visible promotion-review note for a tool that already has a fix variant.

    :param decision: the triage decision dict (``descendants`` et al.).
    :param platform: platform the note is scoped to.
    :param tournament_roster: normalized tournament roster names.
    :param tournament_by_tool: ``by_tool`` dict of the tournament scores file.
    :return: markdown body.
    """
    descendants = decision.get("descendants", [])
    roster = tournament_roster or set()
    t_by = tournament_by_tool or {}

    def _dline(v: str) -> str:
        stats = t_by.get(v) or t_by.get(v.replace("-", "_")) or {}
        if _in_roster(v, roster):
            n = stats.get("valid_n")
            brier = stats.get("brier")
            if n:
                return (
                    f"- `{v}` -- in tournament: n={n}, Brier {brier} "
                    "(shadow data, accruing daily)"
                )
            return f"- `{v}` -- in tournament (no scored rows yet)"
        return f"- `{v}` -- NOT in tournament (add it to accrue riskless data)"

    dlist = "\n".join(_dline(v) for v in descendants) or "- (see tool_lineage.json)"
    return _PROMOTION_BODY_TEMPLATE.format(
        n=COUNT_WINDOW_N,
        tool=decision["tool"],
        platform=platform,
        trigger=decision.get("trigger", "level_floor"),
        brier_cur=_f(decision.get("brier_cur"), ".4f"),
        bss_cur=_f(decision.get("bss_cur"), "+.4f"),
        descendant_list=dlist,
    )


def _ensure_label(repo: str, label: str, dry_run: bool) -> None:
    """Create ``label`` if missing (idempotent; a pre-existing label is a no-op)."""
    if dry_run:
        return
    # Build the argv as a variable (matching the other gh calls in this
    # module) so bandit does not raise B607 on the literal ``gh`` path.
    cmd = [
        "gh",
        "label",
        "create",
        label,
        "--repo",
        repo,
        "--color",
        "0e8a16",
        "--description",
        "Tool has a merged fix variant; promote/retire decision (no auto-fix)",
    ]
    # Degrade gracefully like every other gh helper in this module: a missing
    # gh binary / timeout must not abort the whole run before write_state.
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("gh label create failed to run (%s); continuing", exc)
        return
    # ``already exists`` is the expected idempotent path; surface any other
    # non-zero (e.g. token lacks label scope) so the root cause is visible
    # rather than showing up only as a downstream ``gh issue create`` error.
    if r.returncode != 0 and "already exists" not in (r.stderr or "").lower():
        log.warning(
            "gh label create %r returned rc=%d: %s",
            label,
            r.returncode,
            (r.stderr or "").strip()[:200],
        )


def main() -> int:
    """CLI entry point invoked by benchmark_flywheel.yaml (non-zero on dispatch failure)."""
    # pylint: disable=too-many-locals,too-many-statements
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument(
        "--state",
        type=Path,
        default=RESULTS_DIR / "tool_improvement_triage_state.json",
    )
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--label", default=DEFAULT_LABEL)
    p.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--platforms",
        default=",".join(ENABLED_PLATFORMS),
        help="Comma-separated platforms to triage (default: ENABLED_PLATFORMS).",
    )
    args = p.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
    )

    platforms = [p_.strip() for p_ in args.platforms.split(",") if p_.strip()]
    log.info(
        "triage thresholds: regression=%.3f level_bss=%.2f "
        "level_brier_fallback=%.2f valid_n=%d reliability=%.2f "
        "window_days=%d platforms=%s dry_run=%s",
        BRIER_REGRESSION_THRESHOLD,
        BSS_LEVEL_FLOOR,
        BRIER_LEVEL_THRESHOLD,
        VALID_N_PER_WINDOW_FLOOR,
        RELIABILITY_FLOOR,
        ROLLING_WINDOW_DAYS,
        platforms,
        args.dry_run,
    )

    if args.run_id:
        artifact_url = (
            f"https://github.com/{args.repo}/actions/runs/{args.run_id}#artifacts"
        )
    else:
        log.warning("GITHUB_RUN_ID not set; issue body will carry a placeholder URL.")
        artifact_url = "<no run id>"

    now = datetime.now(timezone.utc)
    window_iso = _window_iso(now)
    state = _load_json(args.state)
    # Live sources-of-truth for the two close-related gates:
    #  * ``open_now`` (open issues on GitHub) drives the duplicate-issue
    #    suppression. The state file is consulted only as a fallback when
    #    the gh query fails (returns ``None``).
    #  * ``closed_issues`` (recently closed issues on GitHub) drives the
    #    ``RECENT_CLOSE_DAYS`` cooldown silence. Same fail-open principle:
    #    a transient gh failure skips the cooldown rather than refile.
    # The state file is no longer consulted for cooldown bookkeeping; the
    # gate runs entirely off the live gh closed-list result.
    open_now = _open_issue_tools(args.repo, args.label)
    if open_now:
        log.info("triage open issues on GitHub: %s", sorted(open_now))
    closed_issues = _closed_issue_pairs(args.repo, args.label)
    if closed_issues:
        log.info(
            "triage closed-issue history (last %d shown): %s",
            min(10, len(closed_issues)),
            sorted((t, p, c.strftime("%Y-%m-%d")) for t, p, c in closed_issues[:10]),
        )

    # Lineage: tools with a merged fix variant are routed to a promotion
    # note instead of a redundant fix issue. ``promo_open`` dedups the note
    # under its own label so it is posted once per (tool, platform), not
    # daily.
    lineage_children = _load_lineage_children()
    log.info("triage lineage: %d tools have >=1 fix variant", len(lineage_children))
    promo_open = _open_issue_tools(args.repo, PROMOTION_LABEL, _PROMO_TITLE_RE)
    # Migration shim for the tool-promotion-review -> deployment-review label
    # rename: notes opened under the OLD label would otherwise be invisible
    # to the dedup and get re-posted daily under the new one. Query both and
    # union; a gh error on EITHER makes the combined result None (fail-closed,
    # same rule as below). Remove once no open legacy-labelled notes remain.
    legacy_open = _open_issue_tools(
        args.repo, _LEGACY_PROMOTION_LABEL, _LEGACY_PROMO_TITLE_RE
    )
    if promo_open is None or legacy_open is None:
        promo_open = None
    else:
        promo_open = list(promo_open) + list(legacy_open)
    # Recently-closed deployment-review notes: a manual close must silence
    # the same (tool, platform) note for DR_RECENT_CLOSE_DAYS -- without
    # this the insufficient_data condition re-fires deterministically and
    # re-files the note the day after a human closed it. Fail-closed on gh
    # error (treated like an open note: skip filing).
    dr_closed = _closed_issue_pairs(args.repo, PROMOTION_LABEL, _PROMO_TITLE_RE)
    dr_closed_legacy = _closed_issue_pairs(
        args.repo, _LEGACY_PROMOTION_LABEL, _LEGACY_PROMO_TITLE_RE
    )
    if dr_closed is None or dr_closed_legacy is None:
        dr_recent_close: Optional[set] = None
    else:
        dr_cutoff = now - timedelta(days=DR_RECENT_CLOSE_DAYS)
        dr_recent_close = {
            (t, p_)
            for t, p_, closed_at in list(dr_closed) + list(dr_closed_legacy)
            if closed_at >= dr_cutoff
        }
    # ``promo_open`` is a None/[]/[...] tristate. ``promo_open_set`` collapses
    # None -> empty, so it is NOT sufficient on its own: the dispatch branch
    # below MUST keep its explicit ``if promo_open is None`` guard (skip on gh
    # error) to avoid duplicate notes. Do not "simplify" to promo_open_set only.
    promo_open_set = {tuple(x) for x in (promo_open or [])}

    all_decisions: List[Dict[str, Any]] = []
    n_opened = n_failed = n_collapse = n_silent = n_promo = n_widen = 0
    for platform in platforms:
        cur = _load_json(RESULTS_DIR / f"rolling_scores_{platform}.json")
        prev = _load_json(RESULTS_DIR / f"prev_rolling_scores_{platform}.json")
        platform_scores = _load_json(RESULTS_DIR / f"scores_{platform}.json")
        # Count-window fallback input (anti-stall): per-tool last-N /
        # previous-N stats from the raw daily logs. The logs dir is derived
        # from RESULTS_DIR (its sibling) so test monkeypatching of
        # RESULTS_DIR isolates this path too; missing logs (unit tests,
        # partial checkouts) degrade to the legacy calendar-only behavior
        # inside triage().
        logs_dir = RESULTS_DIR.parent / "datasets" / "logs"
        count_windows: Dict[str, Dict[str, Any]] = {}
        if logs_dir.is_dir():
            count_rows = _load_count_rows(logs_dir, platform, now)
            count_windows = {
                tool: _count_window_stats(rows) for tool, rows in count_rows.items()
            }
            log.info(
                "count-window fallback: %d tools with row data on %s",
                len(count_windows),
                platform,
            )
        # Tournament fallback inputs: row-level shadow results + roster.
        tournament_rows = _load_tournament_count_rows(
            RESULTS_DIR / TOURNAMENT_SCORED_NAME, platform, now
        )
        tournament_windows = {
            tool: _count_window_stats(rows) for tool, rows in tournament_rows.items()
        }
        if tournament_windows:
            log.info(
                "tournament fallback: %d tools with tournament rows on %s",
                len(tournament_windows),
                platform,
            )
        roster = _tournament_roster()
        tournament_scores = _load_json(
            RESULTS_DIR / f"scores_tournament_{platform}.json"
        )
        tournament_by_tool = tournament_scores.get("by_tool") or {}
        decisions = triage(
            cur,
            prev,
            state,
            platform=platform,
            open_now=open_now,
            closed_issues=closed_issues,
            now=now,
            lineage_children=lineage_children,
            count_windows=count_windows,
            tournament_windows=tournament_windows,
            tournament_roster=roster,
        )
        all_decisions.extend(decisions)

        for d in decisions:
            _log_decision(d)
            if d["decision"] == "open_issue":
                stats_pm = (platform_scores.get("by_tool") or {}).get(d["tool"], {})
                body = build_issue_body(
                    d,
                    stats_pm,
                    artifact_url,
                    window_iso,
                    platforms_monitored=platforms,
                )
                title = build_issue_title(
                    d["tool"],
                    platform,
                    d.get("reason", "regression"),
                    d.get("bss_cur"),
                    window_kind=d.get("window_kind", "calendar"),
                )
                rc = _open_issue(args.repo, args.label, title, body, args.dry_run)
                if rc == 0:
                    n_opened += 1
                else:
                    n_failed += 1
                    # A failed gh issue create must not be persisted as
                    # "issue is open"; otherwise a later gh-list-failure
                    # day would fall back to the state file and silently
                    # suppress tomorrow's run for a nonexistent issue.
                    d["issue_open"] = False
                    d["reason"] = "open_issue_failed"
                    # decision-independent flag so a state-file consumer can
                    # detect "we tried to dispatch and it failed" without
                    # knowing every per-decision reason string.
                    d["dispatch_failed"] = True
            elif d["decision"] == "reliability_collapse":
                log.warning(
                    "RELIABILITY COLLAPSE on %s (%s): reliability=%.3f (< %.2f). "
                    "Tool-improvement issue NOT opened.",
                    d["tool"],
                    platform,
                    d.get("reliability_cur") or float("nan"),
                    RELIABILITY_FLOOR,
                )
                n_collapse += 1
            elif d["decision"] == "descendant_exists":
                # Fired, but a merged fix variant already exists: post a
                # visible promotion-review note (deduped by PROMOTION_LABEL)
                # rather than a redundant fix issue. Does NOT invoke the
                # coding agent.
                tool = d["tool"]
                descendants = d.get("descendants")
                key = (tool, platform)
                if promo_open is None:
                    # The open-promo-notes query itself failed (transient gh
                    # error), so ``promo_open_set`` is unreliable. This branch is
                    # fail-CLOSED: unlike the fix-issue path (which on a gh error
                    # falls back to the state file and keeps operating), a promo
                    # note has no fallback source, so we skip creation entirely
                    # rather than risk a duplicate.
                    log.warning(
                        "descendant_exists on %s/%s: promo-note list unavailable "
                        "(gh error); skipping to avoid a duplicate note.",
                        tool,
                        platform,
                    )
                    n_silent += 1
                elif key in promo_open_set:
                    log.info(
                        "descendant_exists on %s/%s (variants=%s): promotion "
                        "note already open; skipping.",
                        tool,
                        platform,
                        descendants,
                    )
                    n_silent += 1
                else:
                    log.info(
                        "descendant_exists on %s/%s: fix variant(s) %s already "
                        "merged -> opening promotion-review note (no auto-fix).",
                        tool,
                        platform,
                        descendants,
                    )
                    _ensure_label(args.repo, PROMOTION_LABEL, args.dry_run)
                    rc = _open_issue(
                        args.repo,
                        PROMOTION_LABEL,
                        build_promotion_title(tool, platform),
                        build_promotion_body(
                            d,
                            platform,
                            tournament_roster=roster,
                            tournament_by_tool=tournament_by_tool,
                        ),
                        args.dry_run,
                    )
                    if rc == 0:
                        n_promo += 1
                        promo_open_set.add(key)
                    else:
                        n_failed += 1
                        # Mirror the fix-issue path: record the failure so
                        # write_state does not persist a look-alike success.
                        d["reason"] = "promotion_note_failed"
                        d["dispatch_failed"] = True
            elif d["decision"] == "insufficient_data":
                # Below the count floor even cumulatively: the tool cannot
                # be assessed at all. Route to a deployment-review
                # widen-sample note (human decides: route traffic or
                # retire). Shares the promo-note label + open-set so at most
                # one deployment-review note per (tool, platform) is open at
                # a time; same fail-closed rule on a gh listing error.
                key = (d["tool"], platform)
                if dr_recent_close is None:
                    log.warning(
                        "insufficient_data on %s/%s: closed deployment-review "
                        "list unavailable (gh error); skipping to avoid "
                        "re-filing over a fresh manual close.",
                        d["tool"],
                        platform,
                    )
                    n_silent += 1
                elif key in dr_recent_close:
                    log.info(
                        "insufficient_data on %s/%s: deployment-review note "
                        "closed within %dd; cooldown active, skipping.",
                        d["tool"],
                        platform,
                        DR_RECENT_CLOSE_DAYS,
                    )
                    n_silent += 1
                elif promo_open is None:
                    log.warning(
                        "insufficient_data on %s/%s: deployment-review note "
                        "list unavailable (gh error); skipping to avoid a "
                        "duplicate note.",
                        d["tool"],
                        platform,
                    )
                    n_silent += 1
                elif key in promo_open_set:
                    log.info(
                        "insufficient_data on %s/%s: deployment-review note "
                        "already open; skipping.",
                        d["tool"],
                        platform,
                    )
                    n_silent += 1
                else:
                    log.info(
                        "insufficient_data on %s/%s (total=%s < %d): opening "
                        "deployment-review widen-sample note.",
                        d["tool"],
                        platform,
                        d.get("cw_total"),
                        COUNT_WINDOW_N,
                    )
                    _ensure_label(args.repo, PROMOTION_LABEL, args.dry_run)
                    rc = _open_issue(
                        args.repo,
                        PROMOTION_LABEL,
                        build_widen_sample_title(d["tool"], platform),
                        build_widen_sample_body(d, platform),
                        args.dry_run,
                    )
                    if rc == 0:
                        n_widen += 1
                        promo_open_set.add(key)
                    else:
                        n_failed += 1
                        d["reason"] = "widen_sample_note_failed"
                        d["dispatch_failed"] = True
            else:
                n_silent += 1

    log.info(
        "triage summary: %d opened, %d promotion-note, %d widen-sample, "
        "%d failed, %d reliability_collapse, %d silent",
        n_opened,
        n_promo,
        n_widen,
        n_failed,
        n_collapse,
        n_silent,
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not all_decisions:
        # No tool produced a decision: a valid-but-empty rolling file
        # would otherwise overwrite the prior state with {by_tool: {}}
        # and silently drop every cooldown/issue_open marker on the
        # fallback path. Surface the empty-input case and keep the
        # existing state intact.
        log.error(
            "No decisions produced (empty cur for every platform?); "
            "state file NOT overwritten."
        )
        return 1
    write_state(args.state, all_decisions, generated_at)
    log.info("State saved to %s", args.state)
    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
