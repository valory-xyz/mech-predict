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

1. Platform = Polymarket only (Omen Brier is confounded by the on-chain
   ``jury-resolve-market-v1`` mech tool's quality).
2. ``Brier_cur - Brier_prev > 0.040``. Calibrated 2026-05-29 against
   26 days of CI data: 0.015 + two-day confirmation produced ~0.5
   issues/week (too quiet); 0.040 alone yields ~1 issue/week per
   active tool.
3. ``valid_n / 7 >= 15`` on both windows (so ``valid_n >= 105``).
4. ``sign(delta Brier) == sign(delta log_loss)`` (both primary metrics
   from PROPOSAL.md Part 4 must agree on the worsening direction).
5. ``reliability >= 0.80`` (collapses route to a WARNING log instead
   of opening a tool-improvement issue; an operator watching the
   workflow output is the safety net).

Duplicate-issue suppression: once a tool opens an issue, the persisted
``issue_open=True`` flag stays true for as long as the gates keep
firing. Tomorrow stays silent. Once the regression resolves (gates
stop firing), the flag clears and a fresh issue can open later.

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
from typing import Any, Dict, List

ENABLED_PLATFORMS = ["polymarket"]
BRIER_REGRESSION_THRESHOLD = 0.040
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


def triage(
    cur: Dict[str, Any],
    prev: Dict[str, Any],
    prior_state: Dict[str, Any],
    platform: str = "polymarket",
) -> List[Dict[str, Any]]:
    """Apply the gate cascade to ``cur`` vs ``prev`` and return one decision dict per tool."""
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
        if delta_brier <= BRIER_REGRESSION_THRESHOLD:
            d.update(decision="silent", reason="no_regression")
            decisions.append(d)
            continue
        lc, lp = c.get("log_loss"), p.get("log_loss")
        if lc is None or lp is None or lc - lp <= 0:
            d.update(decision="silent", reason="sign_disagreement")
            decisions.append(d)
            continue
        # All gates pass. Duplicate-issue suppression: if an issue is
        # already open for this tool, stay silent but keep issue_open=True
        # so tomorrow also stays silent until the regression resolves.
        if prior_open.get(tool):
            d.update(decision="silent", reason="duplicate_suppressed", issue_open=True)
        else:
            d.update(decision="open_issue", reason="all_gates_pass", issue_open=True)
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


def build_issue_title(tool: str, platform: str = "polymarket") -> str:
    """Issue title format for ``tool`` that the agent's Step 1 parser expects."""
    return f"[tool-improvement] `{tool}`: Brier regression on {platform} W-1"


def build_issue_body(
    decision: Dict[str, Any],
    polymarket_stats: Dict[str, Any],
    combined_stats: Dict[str, Any],
    artifact_url: str,
    window_iso: Dict[str, str],
) -> str:
    """Render the data-only Markdown issue body for one flagged tool."""
    tool = decision["tool"]
    platform = decision.get("platform", "polymarket")
    pm = json.dumps(polymarket_stats, indent=2, sort_keys=True)
    cb = json.dumps({"tool": tool, "stats": combined_stats}, indent=2, sort_keys=True)
    return (
        f"**`{tool}` on {platform}** Brier "
        f"**{decision['brier_cur']:.4f}** (n={decision['n_cur']}, current 7d window W-1) "
        f"vs **{decision['brier_prev']:.4f}** (n={decision['n_prev']}, prev 7d window "
        "W-2 non-overlapping), delta "
        f"**{decision['delta_brier']:+.4f}**. This is the data; the cause is not "
        "yet diagnosed.\n\n"
        "This issue is a regression signal, not a diagnosis. The "
        "investigating agent must pull the linked benchmark artifact, slice "
        "the raw rows, identify the failure mode, and decide whether a "
        "structural fix is warranted.\n\n"
        "@valory-coding-agent (tool-improvement-agent) - investigate per "
        "the `tool-improvement` pipeline.\n\n"
        "## Window definitions\n\n"
        "| Window | Role | Range (predicted_at, UTC) |\n"
        "|---|---|---|\n"
        f"| **W-1** (current) | the regression you must explain | "
        f"`{window_iso['w1_start']}` - `{window_iso['w1_end']}` |\n"
        f"| **W-2** (prev, disjoint) | what PR-CI will re-benchmark against "
        f"| `{window_iso['w2_start']}` - `{window_iso['w2_end']}` |\n\n"
        "The disjointness is the honesty constraint: you investigate W-1, "
        "PR-CI validates on W-2.\n\n"
        f"## {platform.capitalize()} baseline stats (machine-readable)\n\n"
        f"```baseline-stats-{platform}\n{pm}\n```\n\n"
        "(The combined cross-platform `baseline-stats` block follows for "
        f"cross-reference, but the investigation is **{platform}-only** "
        "per the agent's scope rules.)\n\n"
        f"```baseline-stats\n{cb}\n```\n\n"
        "## Data - where to investigate\n\n"
        "The full `benchmark-data` artifact for the flywheel run that "
        "produced these numbers is at:\n\n"
        f"> **{artifact_url}**\n\n"
        "Download it with `gh run download <run-id> --name benchmark-data` "
        "and read the files below in the **listed order** - do not skip "
        "ahead to raw rows before checking the breakdowns:\n\n"
        f"1. **`results/scores_{platform}.json`** - all the per-slice "
        "aggregates the daily scorer produced. The trigger fired against "
        f'`by_tool["{tool}"]`; the agent must decompose the regression '
        "by inspecting these sub-keys for the same tool:\n"
        f'   - `by_tool_category["{tool} | <category>"]` - is the '
        "regression localized to one market category (politics, business, "
        "sports, etc.)?\n"
        f'   - `by_tool_version_mode["{tool} | <version> | <mode>"]` - '
        "did a recent tool version bump introduce the regression?\n"
        f'   - `by_difficulty["<bucket>"]` cross-referenced with the '
        f"tool - is `{tool}` failing specifically on hard markets?\n"
        f'   - `by_liquidity["<bucket>"]` cross-referenced - does '
        "illiquidity correlate?\n"
        f'   - `by_tool_platform["{tool} | {platform}"]` - sanity '
        "check vs the headline number above.\n"
        "   A regression localizes when a sub-cell satisfies "
        "`cell_brier > tool_brier + 0.05` AND `cell_n >= 30`. Cells under "
        "that threshold are statistically noisy at this n.\n"
        f"2. **`results/report_{platform}.md`** - the same numbers in "
        "human-readable form, including `Tool x Category` and "
        "`Tool x Category Historical Comparison` tables that mirror "
        "the JSON above. Use this to sanity-check the decomposition.\n"
        f"3. **`datasets/logs/production_log_<YYYY_MM_DD>.jsonl`** - raw "
        f'rows. Filter on `tool_name == "{tool}"` AND `platform == '
        f'"{platform}"` AND `prediction_parse_status == "valid"` AND '
        "`final_outcome` not null. Sample 10-20 misses from the worst "
        "localized cell and read the prompt + tool response.\n\n"
        "The agent must reproduce the headline number above from step 3 "
        "raw rows BEFORE forming any hypothesis. If the reproduction "
        "does not match within +/- 0.001, the scoring artifact may have "
        "been rewritten between trigger and dispatch; close the issue "
        "with that note rather than acting on stale data.\n"
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
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("gh issue create failed (rc=%d): %s", r.returncode, r.stderr)
        return r.returncode
    log.info(
        "Issue opened: %s",
        (r.stdout or "").strip().splitlines()[-1] if r.stdout else "(no url)",
    )
    return 0


def _log_decision(d: Dict[str, Any]) -> None:
    """Emit one greppable INFO line summarising a triage decision."""
    db = d.get("delta_brier")
    bc = d.get("brier_cur")
    bp = d.get("brier_prev")
    log.info(
        "triage %s %s decision=%s reason=%s delta_brier=%s n_cur=%d "
        "brier_cur=%s brier_prev=%s issue_open=%s",
        d.get("platform", "?"),
        d.get("tool", "?"),
        d.get("decision", "?"),
        d.get("reason", "?"),
        f"{db:+.4f}" if db is not None else "n/a",
        d.get("n_cur", 0) or 0,
        f"{bc:.4f}" if bc is not None else "n/a",
        f"{bp:.4f}" if bp is not None else "n/a",
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
    p.add_argument("--scores", type=Path, default=RESULTS_DIR / "scores.json")
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
        "triage thresholds: brier_threshold=%.3f valid_n_floor=%d "
        "reliability_floor=%.2f rolling_window_days=%d platforms=%s dry_run=%s",
        BRIER_REGRESSION_THRESHOLD,
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

    cross_scores = _load_json(args.scores)
    now = datetime.now(timezone.utc)
    window_iso = _window_iso(now)
    state = _load_json(args.state)

    all_decisions: List[Dict[str, Any]] = []
    n_opened = n_failed = n_collapse = n_silent = 0
    for platform in platforms:
        cur = _load_json(RESULTS_DIR / f"rolling_scores_{platform}.json")
        prev = _load_json(RESULTS_DIR / f"prev_rolling_scores_{platform}.json")
        platform_scores = _load_json(RESULTS_DIR / f"scores_{platform}.json")
        decisions = triage(cur, prev, state, platform=platform)
        all_decisions.extend(decisions)

        for d in decisions:
            _log_decision(d)
            if d["decision"] == "open_issue":
                stats_pm = (platform_scores.get("by_tool") or {}).get(d["tool"], {})
                stats_cb = (cross_scores.get("by_tool") or {}).get(d["tool"], {})
                body = build_issue_body(d, stats_pm, stats_cb, artifact_url, window_iso)
                title = build_issue_title(d["tool"], platform)
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
