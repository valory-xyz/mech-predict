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
"""Dispatcher entrypoint: triage today's scores, open issues, persist state.

Runs as a step inside ``.github/workflows/benchmark_flywheel.yaml`` after the
per-platform ``Analyze`` jobs have produced the rolling-window score files.

For each ``TriageDecision`` returned by ``triage.triage_tools``:

- ``open_issue``: builds a data-only issue body and calls ``gh issue
  create --label tool-improvement``. The label routes the issue to
  ``tool-improvement-agent`` in the agent-skills monorepo.
- ``reliability_collapse``: prints a warning to stderr (operators tail
  the workflow logs) and DOES NOT open a tool-improvement issue.
  Reliability collapses are upstream and need a human.
- ``silent``: no action.

The script always writes the new state file at the end so tomorrow's
triage can apply the confirmation gate. State persists across daily
runs via the existing ``benchmark-data`` GitHub Actions artifact.

The issue body itself is described by the dispatch contract (see the
in-file comment around ``build_issue_body``) and matches the
``pipelines/tool-improvement.md`` Step 1 parser in the
``tool-improvement-agent`` directory of the agent-skills monorepo.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from benchmark.stats_loop.triage import (
    PLATFORM,
    TriageDecision,
    TriageOutcome,
    triage_tools,
    write_state,
)


log = logging.getLogger(__name__)

DEFAULT_REPO = "valory-xyz/mech-predict"
DEFAULT_LABEL = "tool-improvement"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_CUR_SCORES = RESULTS_DIR / f"rolling_scores_{PLATFORM}.json"
DEFAULT_PREV_SCORES = RESULTS_DIR / f"prev_rolling_scores_{PLATFORM}.json"
DEFAULT_STATE = RESULTS_DIR / "stats_loop_state.json"
DEFAULT_SCORES = RESULTS_DIR / "scores.json"
DEFAULT_PLATFORM_SCORES = RESULTS_DIR / f"scores_{PLATFORM}.json"


def _format_baseline_block(stats: Dict[str, Any]) -> str:
    """Serialize a tool's stats block as canonical, machine-readable JSON.

    The agent's pipeline (``tool-improvement-agent``) parses this block
    from the issue body verbatim and uses it as the contract that PR-CI's
    re-bench compares against. The keys MUST match what
    ``benchmark/scorer.py`` writes into ``by_tool[<name>]``.
    """
    return json.dumps(stats, indent=2, sort_keys=True)


def _artifact_url(run_id: str) -> str:
    """Build the GitHub Actions UI URL for this run's artifacts page."""
    return (
        f"https://github.com/{DEFAULT_REPO}/actions/runs/{run_id}#artifacts"
    )


def build_issue_body(
    *,
    decision: TriageDecision,
    polymarket_stats: Dict[str, Any],
    combined_stats: Dict[str, Any],
    artifact_url: str,
    window_iso: Dict[str, str],
) -> str:
    """Render the data-only issue body for one flagged tool.

    The body deliberately contains only objective signals: the headline
    regression numbers, two JSON stats blocks, the artifact URL, and a
    description of the window definitions. No diagnosis, no hints. The
    agent's pipeline pulls the artifact, slices the rows, and forms its
    own hypothesis.

    :param decision: the ``open_issue`` triage decision for this tool.
    :param polymarket_stats: the tool's cumulative Polymarket stats from
        ``scores_polymarket.json::by_tool[<name>]``.
    :param combined_stats: the tool's cross-platform stats from
        ``scores.json::by_tool[<name>]`` (for cross-reference).
    :param artifact_url: GitHub Actions UI URL for the ``benchmark-data``
        artifact carrying the raw rows and reports.
    :param window_iso: dict with ``w1_start``, ``w1_end``, ``w2_start``,
        ``w2_end`` ISO-8601 timestamps for the two rolling windows.
    :return: Markdown issue body, ASCII-only.
    """
    today = decision.today
    polymarket_block = _format_baseline_block(polymarket_stats)
    combined_block = _format_baseline_block({
        "tool": today.tool_name,
        "stats": combined_stats,
    })
    return (
        f"**Polystrat `{today.tool_name}`** Brier "
        f"**{today.brier_cur:.4f}** (n={today.n_cur}, current 7d window W-1) "
        f"vs **{today.brier_prev:.4f}** (n={today.n_prev}, prev 7d window "
        "W-2 non-overlapping), delta "
        f"**{today.delta_brier:+.4f}**. This is the data; the cause is not "
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
        f"```baseline-stats-polymarket\n{polymarket_block}\n```\n\n"
        "(The combined cross-platform `baseline-stats` block follows for "
        "cross-reference, but the investigation is **polymarket-only** "
        "per the agent's scope rules.)\n\n"
        f"```baseline-stats\n{combined_block}\n```\n\n"
        "## Data - where to investigate\n\n"
        "The full `benchmark-data` artifact for the flywheel run that "
        "produced these numbers is at:\n\n"
        f"> **{artifact_url}**\n\n"
        "Download it with `gh run download <run-id> --name benchmark-data` "
        "and read:\n\n"
        "- `datasets/logs/production_log_<YYYY_MM_DD>.jsonl` - raw rows. "
        f"Filter on `tool_name == \"{today.tool_name}\"` AND `platform == "
        "\"polymarket\"` AND `prediction_parse_status == \"valid\"` AND "
        "`final_outcome` not null.\n"
        "- `results/scores_polymarket.json` - what the daily scorer wrote "
        "(matches the baseline block above).\n"
        "- `results/report_polymarket.md` - the human-readable report.\n\n"
        "The agent must reproduce the headline number above from the raw "
        "rows before forming any hypothesis.\n"
    )


def build_issue_title(decision: TriageDecision) -> str:
    """Issue title that the agent's Step 1 parser expects.

    Format: ``[tool-improvement] <tool>: Brier regression on polymarket W-1``.
    """
    return (
        f"[tool-improvement] `{decision.today.tool_name}`: "
        "Brier regression on polymarket W-1"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _window_iso_from_args(now: datetime, days: int = 7) -> Dict[str, str]:
    """Compute the W-1 and W-2 ISO ranges based on the run wall-clock.

    The scorer's per-window slicing uses calendar-day boundaries. This
    approximation is good enough for issue-body context; the agent
    re-derives the exact ranges from the daily log filenames when it
    pulls the artifact.
    """
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    w1_end = now
    w1_start = w1_end.replace() - _delta_days(days)
    w2_end = w1_start
    w2_start = w2_end - _delta_days(days)
    return {
        "w1_start": w1_start.strftime(fmt),
        "w1_end": w1_end.strftime(fmt),
        "w2_start": w2_start.strftime(fmt),
        "w2_end": w2_end.strftime(fmt),
    }


def _delta_days(days: int):
    from datetime import timedelta
    return timedelta(days=days)


def open_github_issue(
    *,
    repo: str,
    label: str,
    title: str,
    body: str,
    dry_run: bool,
) -> int:
    """Call ``gh issue create`` to publish the issue.

    Returns 0 on success, the gh subprocess return code otherwise.
    In dry-run mode prints what would be executed and returns 0.
    """
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
    if dry_run:
        log.info("DRY-RUN: would execute gh issue create with title=%s", title)
        return 0
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(
            "gh issue create failed (rc=%d): %s",
            result.returncode,
            result.stderr,
        )
    else:
        log.info("Issue opened: %s", result.stdout.strip())
    return result.returncode


def main() -> int:
    """CLI entry point invoked by ``benchmark_flywheel.yaml``.

    Reads two per-platform rolling-window score files + the cross-day
    state file, runs the triage cascade, opens issues for tools that
    pass all gates, and persists the new state.
    """
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--cur-scores", type=Path, default=DEFAULT_CUR_SCORES)
    parser.add_argument("--prev-scores", type=Path, default=DEFAULT_PREV_SCORES)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--platform-scores", type=Path, default=DEFAULT_PLATFORM_SCORES
    )
    parser.add_argument(
        "--scores", type=Path, default=DEFAULT_SCORES,
        help="Cumulative cross-platform scores for the cross-reference block.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--run-id", default=os.environ.get("GITHUB_RUN_ID", ""),
        help="GitHub Actions run id; used to build the artifact URL.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the actions; do not call gh.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )

    decisions = triage_tools(
        cur_scores_path=args.cur_scores,
        prev_scores_path=args.prev_scores,
        state_path=args.state,
    )

    now = datetime.now(timezone.utc)
    window_iso = _window_iso_from_args(now)
    artifact_url = _artifact_url(args.run_id) if args.run_id else "<no run id>"

    polymarket_scores = (
        json.loads(args.platform_scores.read_text(encoding="utf-8"))
        if args.platform_scores.exists()
        else {}
    )
    cross_scores = (
        json.loads(args.scores.read_text(encoding="utf-8"))
        if args.scores.exists()
        else {}
    )

    n_opened = 0
    n_reliability = 0
    n_silent = 0
    for decision in decisions:
        if decision.decision == "open_issue":
            stats_pm = polymarket_scores.get("by_tool", {}).get(
                decision.today.tool_name, {}
            )
            stats_combined = cross_scores.get("by_tool", {}).get(
                decision.today.tool_name, {}
            )
            body = build_issue_body(
                decision=decision,
                polymarket_stats=stats_pm,
                combined_stats=stats_combined,
                artifact_url=artifact_url,
                window_iso=window_iso,
            )
            title = build_issue_title(decision)
            rc = open_github_issue(
                repo=args.repo,
                label=args.label,
                title=title,
                body=body,
                dry_run=args.dry_run,
            )
            if rc == 0:
                n_opened += 1
        elif decision.decision == "reliability_collapse":
            log.warning(
                "RELIABILITY COLLAPSE on %s: reliability=%.3f (< %.2f). "
                "Page an operator; tool-improvement issue NOT opened.",
                decision.today.tool_name,
                decision.today.reliability_cur or float("nan"),
                0.80,
            )
            n_reliability += 1
        else:
            n_silent += 1

    log.info(
        "stats-loop summary: %d issue(s) opened, %d reliability collapse(s), "
        "%d tool(s) silent",
        n_opened,
        n_reliability,
        n_silent,
    )

    write_state(args.state, decisions, _now_iso())
    log.info("State saved to %s", args.state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
