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
2. Trigger (either is sufficient):
   - Regression: ``Brier_cur - Brier_prev > 0.040`` AND
     ``sign(delta Brier) == sign(delta log_loss)``. Calibrated
     2026-05-29 against 26 days of CI data: 0.040 alone yields ~1
     issue/week per active tool.
   - Level: ``Brier_cur > 0.25`` regardless of delta. The loop is
     self-improvement, not just self-stabilization; a tool whose
     Brier is persistently 0.30 is a candidate for improvement
     even when it is not actively getting worse.
3. ``valid_n / 7 >= 15`` on both windows (so ``valid_n >= 105``).
4. ``reliability >= 0.80`` (collapses route to a WARNING log instead
   of opening a tool-improvement issue; an operator watching the
   workflow output is the safety net).

Duplicate-issue suppression: once a tool has an open ``tool-improvement``
issue on GitHub, the triage stays silent for that tool. ``prior_open``
is seeded from the live ``gh issue list`` query at the start of
``main()`` rather than from yesterday's state file alone, so a PR that
fixes the issue (closing it) re-arms the trigger immediately; a PR
still under review keeps the tool quiet for as long as the issue
remains open.

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
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ENABLED_PLATFORMS = ["polymarket"]
BRIER_REGRESSION_THRESHOLD = 0.040
BRIER_LEVEL_THRESHOLD = 0.25
VALID_N_PER_WINDOW_FLOOR = 105
RELIABILITY_FLOOR = 0.80
ROLLING_WINDOW_DAYS = 7

DEFAULT_REPO = "valory-xyz/mech-predict"
DEFAULT_LABEL = "tool-improvement"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

log = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict[str, Any]:
    """Return JSON contents of ``path`` or ``{}`` if missing/corrupt."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _open_issue_tools(repo: str, label: str) -> Optional[List[str]]:
    """Return tool names with an open ``label``-labelled issue on ``repo``.

    Returns ``None`` on any gh error (caller falls back to the state file
    for suppression) and ``[]`` only when the gh call succeeded with zero
    matching open issues. Distinguishing the two avoids treating a
    transient gh-CLI failure as "no issues open", which would otherwise
    drop suppression and refile every regression.
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
        "open",
        "--json",
        "title",
        "--limit",
        "200",
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
    tools: List[str] = []
    for row in rows:
        title = row.get("title") or ""
        start = title.find("`")
        end = title.find("`", start + 1)
        if 0 <= start < end:
            tools.append(title[start + 1 : end])
    return tools


def triage(
    cur: Dict[str, Any],
    prev: Dict[str, Any],
    prior_state: Dict[str, Any],
    platform: str = "polymarket",
    open_now: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Apply the gate cascade to ``cur`` vs ``prev`` and return one decision dict per tool.

    Suppression source-of-truth: ``open_now`` (the live ``gh issue list``
    result) when provided; falls back to ``prior_state`` only when the
    caller did not run the live query. Closing the GitHub issue therefore
    re-arms the trigger on the next run regardless of state-file content.
    """
    if open_now is not None:
        prior_open = {tool: True for tool in open_now}
    else:
        prior_open = {
            t: v.get("issue_open", False)
            for t, v in (prior_state.get("by_tool") or {}).items()
        }
    decisions: List[Dict[str, Any]] = []
    for tool in sorted((cur.get("by_tool") or {}).keys()):
        c = cur["by_tool"][tool]
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
        if rel is not None and rel < RELIABILITY_FLOOR:
            d.update(decision="reliability_collapse", reason="reliability_collapse")
            decisions.append(d)
            continue
        if (
            d["n_cur"] < VALID_N_PER_WINDOW_FLOOR
            or d["n_prev"] < VALID_N_PER_WINDOW_FLOOR
        ):
            d.update(decision="silent", reason="sample_floor")
            decisions.append(d)
            continue
        bc, bp = c.get("brier"), p.get("brier")
        if bc is None or bp is None:
            d.update(decision="silent", reason="missing_brier")
            decisions.append(d)
            continue
        delta_brier = bc - bp
        d["delta_brier"] = delta_brier
        # Two triggers express the self-improvement loop:
        # - regression: today's Brier worsened by more than the threshold
        #   AND log_loss agrees on the worsening direction. Sign
        #   agreement filters noisy borderline regressions.
        # - level: today's Brier exceeds an absolute floor. The tool is
        #   not getting worse but it is persistently inaccurate, which
        #   is a self-improvement opportunity. Level is checked
        #   independently of sign-agreement: a high-level tool may still
        #   warrant a fix even when log_loss disagrees on the delta.
        above_level = bc > BRIER_LEVEL_THRESHOLD
        regressed = False
        if delta_brier > BRIER_REGRESSION_THRESHOLD:
            lc, lp = c.get("log_loss"), p.get("log_loss")
            if lc is None or lp is None:
                if not above_level:
                    d.update(decision="silent", reason="missing_log_loss")
                    decisions.append(d)
                    continue
                # else fall through with regressed=False; level fires
            elif lc - lp > 0:
                regressed = True
            elif not above_level:
                d.update(decision="silent", reason="sign_disagreement")
                decisions.append(d)
                continue
        if not regressed and not above_level:
            d.update(decision="silent", reason="no_regression")
            decisions.append(d)
            continue
        # Duplicate-issue suppression: if an issue is already open for
        # this tool (regression or level signal), stay silent and keep
        # issue_open=True so tomorrow also stays silent until the
        # human/agent closes the open GitHub issue.
        reason = "regression" if regressed else "level_floor"
        if prior_open.get(tool):
            d.update(decision="silent", reason="duplicate_suppressed", issue_open=True)
        else:
            d.update(decision="open_issue", reason=reason, issue_open=True)
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
                "issue_open": d["issue_open"],
                "delta_brier": d.get("delta_brier"),
                "brier_cur": d.get("brier_cur"),
                "brier_prev": d.get("brier_prev"),
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
) -> str:
    """Issue title format for ``tool`` that the agent's Step 1 parser expects."""
    if reason == "level_floor":
        return f"[tool-improvement] `{tool}`: Brier above level on {platform} W-1"
    return f"[tool-improvement] `{tool}`: Brier regression on {platform} W-1"


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
        headline = (
            f"`{tool}` on {platform} has a Brier persistently above "
            f"{BRIER_LEVEL_THRESHOLD:.2f} in the most recent 7-day window. "
            "This is a self-improvement opportunity: the tool is not "
            "actively getting worse, but its level is high enough that a "
            "structural change is worth attempting."
        )
        signal = "level signal"
    else:
        headline = (
            f"`{tool}` on {platform} shows a Brier regression in the most "
            "recent 7-day window."
        )
        signal = "regression signal"
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
"""


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
        r = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=30
        )
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


def main() -> int:
    """CLI entry point invoked by benchmark_flywheel.yaml (non-zero on dispatch failure)."""
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
        "triage thresholds: regression=%.3f level=%.2f valid_n=%d reliability=%.2f "
        "window_days=%d platforms=%s dry_run=%s",
        BRIER_REGRESSION_THRESHOLD,
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
    # Live source-of-truth for duplicate-issue suppression: a tool whose
    # issue is currently open on GitHub stays silent; one whose issue was
    # closed (PR merged or human dismissed) re-arms the trigger
    # immediately on the next run. The state file is no longer consulted
    # for suppression (it remains informational, persisted for debugging).
    open_now = _open_issue_tools(args.repo, args.label)
    if open_now:
        log.info("triage open issues on GitHub: %s", sorted(open_now))

    all_decisions: List[Dict[str, Any]] = []
    n_opened = n_failed = n_collapse = n_silent = 0
    for platform in platforms:
        cur = _load_json(RESULTS_DIR / f"rolling_scores_{platform}.json")
        prev = _load_json(RESULTS_DIR / f"prev_rolling_scores_{platform}.json")
        platform_scores = _load_json(RESULTS_DIR / f"scores_{platform}.json")
        decisions = triage(cur, prev, state, platform=platform, open_now=open_now)
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
                    d["tool"], platform, d.get("reason", "regression")
                )
                rc = _open_issue(args.repo, args.label, title, body, args.dry_run)
                if rc == 0:
                    n_opened += 1
                else:
                    n_failed += 1
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
            else:
                n_silent += 1

    log.info(
        "triage summary: %d opened, %d failed, %d reliability_collapse, %d silent",
        n_opened,
        n_failed,
        n_collapse,
        n_silent,
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_state(args.state, all_decisions, generated_at)
    log.info("State saved to %s", args.state)
    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
