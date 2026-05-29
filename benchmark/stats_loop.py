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

PLATFORM = "polymarket"
BRIER_REGRESSION_THRESHOLD = 0.040
VALID_N_PER_WINDOW_FLOOR = 105
RELIABILITY_FLOOR = 0.80
ROLLING_WINDOW_DAYS = 7

DEFAULT_REPO = "valory-xyz/mech-predict"
DEFAULT_LABEL = "tool-improvement"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

log = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict[str, Any]:
    """Return JSON contents or ``{}`` on missing / corrupt file.

    The state file rides in the ``benchmark-data`` artifact across
    daily runs; a truncated write would wedge the loop without this
    heal. Missing rolling-score files happen on first deploys.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def triage(
    cur: Dict[str, Any],
    prev: Dict[str, Any],
    prior_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply the gate cascade. Returns one decision dict per tool.

    Each decision has ``tool``, ``decision`` (``open_issue`` |
    ``reliability_collapse`` | ``silent``), ``reason``, ``issue_open``
    (for state propagation), and the underlying numbers used in the
    issue body (``brier_cur``, ``brier_prev``, ``delta_brier``,
    ``n_cur``, ``n_prev``, ``reliability_cur``).
    """
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
    """Persist today's per-tool outcomes for tomorrow's run."""
    payload = {
        "generated_at": generated_at,
        "platform": PLATFORM,
        "by_tool": {
            d["tool"]: {
                "tool_name": d["tool"],
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
    """Compute W-1 and W-2 ISO ranges relative to ``now``."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    w = timedelta(days=days)
    return {
        "w1_start": (now - w).strftime(fmt),
        "w1_end": now.strftime(fmt),
        "w2_start": (now - 2 * w).strftime(fmt),
        "w2_end": (now - w).strftime(fmt),
    }


def build_issue_title(tool: str) -> str:
    """Title format that the agent's Step 1 parser expects."""
    return f"[tool-improvement] `{tool}`: Brier regression on polymarket W-1"


def build_issue_body(
    decision: Dict[str, Any],
    polymarket_stats: Dict[str, Any],
    combined_stats: Dict[str, Any],
    artifact_url: str,
    window_iso: Dict[str, str],
) -> str:
    """Render the data-only issue body for one flagged tool.

    Contains only objective signals: the headline regression numbers,
    two JSON stats blocks, the artifact URL, and a description of the
    window definitions. No diagnosis, no hints. The agent pulls the
    artifact, slices the rows, and forms its own hypothesis.
    """
    tool = decision["tool"]
    pm = json.dumps(polymarket_stats, indent=2, sort_keys=True)
    cb = json.dumps({"tool": tool, "stats": combined_stats}, indent=2, sort_keys=True)
    return (
        f"**Polystrat `{tool}`** Brier "
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
        "## Polymarket baseline stats (machine-readable)\n\n"
        f"```baseline-stats-polymarket\n{pm}\n```\n\n"
        "(The combined cross-platform `baseline-stats` block follows for "
        "cross-reference, but the investigation is **polymarket-only** "
        "per the agent's scope rules.)\n\n"
        f"```baseline-stats\n{cb}\n```\n\n"
        "## Data - where to investigate\n\n"
        "The full `benchmark-data` artifact for the flywheel run that "
        "produced these numbers is at:\n\n"
        f"> **{artifact_url}**\n\n"
        "Download it with `gh run download <run-id> --name benchmark-data` "
        "and read:\n\n"
        "- `datasets/logs/production_log_<YYYY_MM_DD>.jsonl` - raw rows. "
        f'Filter on `tool_name == "{tool}"` AND `platform == '
        '"polymarket"` AND `prediction_parse_status == "valid"` AND '
        "`final_outcome` not null.\n"
        "- `results/scores_polymarket.json` - what the daily scorer wrote "
        "(matches the baseline block above).\n"
        "- `results/report_polymarket.md` - the human-readable report.\n\n"
        "The agent must reproduce the headline number above from the raw "
        "rows before forming any hypothesis.\n"
    )


def _open_issue(repo: str, label: str, title: str, body: str, dry_run: bool) -> int:
    """Call ``gh issue create``. Returns subprocess return code."""
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


def main() -> int:
    """CLI entry point invoked by benchmark_flywheel.yaml."""
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument(
        "--cur-scores",
        type=Path,
        default=RESULTS_DIR / f"rolling_scores_{PLATFORM}.json",
    )
    p.add_argument(
        "--prev-scores",
        type=Path,
        default=RESULTS_DIR / f"prev_rolling_scores_{PLATFORM}.json",
    )
    p.add_argument("--state", type=Path, default=RESULTS_DIR / "stats_loop_state.json")
    p.add_argument(
        "--platform-scores", type=Path, default=RESULTS_DIR / f"scores_{PLATFORM}.json"
    )
    p.add_argument("--scores", type=Path, default=RESULTS_DIR / "scores.json")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--label", default=DEFAULT_LABEL)
    p.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
    )

    cur = _load_json(args.cur_scores)
    prev = _load_json(args.prev_scores)
    state = _load_json(args.state)
    decisions = triage(cur, prev, state)

    if args.run_id:
        artifact_url = (
            f"https://github.com/{args.repo}/actions/runs/{args.run_id}#artifacts"
        )
    else:
        log.warning("GITHUB_RUN_ID not set; issue body will carry a placeholder URL.")
        artifact_url = "<no run id>"

    pm_scores = _load_json(args.platform_scores)
    cross_scores = _load_json(args.scores)
    now = datetime.now(timezone.utc)
    window_iso = _window_iso(now)

    n_opened = n_failed = n_collapse = n_silent = 0
    for d in decisions:
        if d["decision"] == "open_issue":
            stats_pm = (pm_scores.get("by_tool") or {}).get(d["tool"], {})
            stats_cb = (cross_scores.get("by_tool") or {}).get(d["tool"], {})
            body = build_issue_body(d, stats_pm, stats_cb, artifact_url, window_iso)
            title = build_issue_title(d["tool"])
            rc = _open_issue(args.repo, args.label, title, body, args.dry_run)
            if rc == 0:
                n_opened += 1
            else:
                n_failed += 1
        elif d["decision"] == "reliability_collapse":
            log.warning(
                "RELIABILITY COLLAPSE on %s: reliability=%.3f (< %.2f). "
                "Tool-improvement issue NOT opened.",
                d["tool"],
                d.get("reliability_cur") or float("nan"),
                RELIABILITY_FLOOR,
            )
            n_collapse += 1
        else:
            n_silent += 1

    log.info(
        "stats-loop summary: %d opened, %d failed, %d reliability_collapse, %d silent",
        n_opened,
        n_failed,
        n_collapse,
        n_silent,
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_state(args.state, decisions, generated_at)
    log.info("State saved to %s", args.state)
    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
