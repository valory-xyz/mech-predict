"""
Compute benchmark metrics from replay results and post a PR comment.

Reads baseline.jsonl and candidate.jsonl produced by prompt_replay.py replay,
computes Brier score / accuracy / overconfident-wrong metrics, and optionally
posts a comparison table as a GitHub PR comment.

Usage:
    # Local (print report to stdout):
    python -m benchmark.ci_replay \
      --baseline benchmark/results/ci_replay/baseline.jsonl \
      --candidate benchmark/results/ci_replay/candidate.jsonl

    # CI (post to PR):
    python -m benchmark.ci_replay \
      --baseline benchmark/results/ci_replay/baseline.jsonl \
      --candidate benchmark/results/ci_replay/candidate.jsonl \
      --pr 192 --repo valory-xyz/mech-predict --triggered-by LOCKhart07
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.io import load_jsonl


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Brier score, accuracy, and overconfident-wrong count.

    :param rows: list of prediction rows with p_yes, final_outcome.
    :return: dict with brier, accuracy, overconf_wrong, n, and by_platform.
    """
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_platform[row.get("platform", "unknown")].append(row)

    def _metrics(subset: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [r for r in subset if r.get("p_yes") is not None]
        n = len(valid)
        if n == 0:
            return {"brier": None, "accuracy": None, "overconf_wrong": 0, "n": 0}

        brier_sum = 0.0
        correct = 0
        overconf_wrong = 0
        for r in valid:
            p_yes = r["p_yes"]
            outcome = r["final_outcome"]
            outcome_val = 1.0 if outcome else 0.0
            brier_sum += (p_yes - outcome_val) ** 2
            predicted_yes = p_yes > 0.5
            if predicted_yes == outcome:
                correct += 1
            if max(p_yes, 1 - p_yes) > 0.90 and predicted_yes != outcome:
                overconf_wrong += 1

        return {
            "brier": brier_sum / n,
            "accuracy": correct / n,
            "overconf_wrong": overconf_wrong,
            "n": n,
        }

    overall = _metrics(rows)
    overall["by_platform"] = {
        p: _metrics(prows) for p, prows in sorted(by_platform.items())
    }
    return overall


def _fmt_delta(
    baseline_val: float | None,
    candidate_val: float | None,
    lower_is_better: bool = True,
) -> str:
    """Format a delta cell with arrow indicator."""
    if baseline_val is None or candidate_val is None or baseline_val == 0:
        return "N/A"
    delta_pct = (candidate_val - baseline_val) / abs(baseline_val) * 100
    return f"{delta_pct:+.1f}%"


def _fmt_metric_row(
    name: str, b_val: Any, c_val: Any, fmt: str, lower_is_better: bool = True
) -> str:
    """Format a single metric row for the markdown table."""
    if b_val is None:
        b_str = "N/A"
    elif fmt == "pct":
        b_str = f"{b_val * 100:.1f}%"
    elif fmt == "int":
        b_str = str(b_val)
    else:
        b_str = f"{b_val:.4f}"

    if c_val is None:
        c_str = "N/A"
    elif fmt == "pct":
        c_str = f"{c_val * 100:.1f}%"
    elif fmt == "int":
        c_str = str(c_val)
    else:
        c_str = f"{c_val:.4f}"

    delta = _fmt_delta(b_val, c_val, lower_is_better)
    return f"| {name} | {b_str} | {c_str} | {delta} |"


def _metrics_table(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Build a markdown comparison table from two metric dicts."""
    lines = [
        "| Metric | Baseline (prod) | Candidate (PR) | Delta |",
        "|--------|-----------------|----------------|-------|",
        _fmt_metric_row(
            "Brier score",
            baseline["brier"],
            candidate["brier"],
            "float",
            lower_is_better=True,
        ),
        _fmt_metric_row(
            "Accuracy",
            baseline["accuracy"],
            candidate["accuracy"],
            "pct",
            lower_is_better=False,
        ),
        _fmt_metric_row(
            "Overconf-wrong",
            baseline["overconf_wrong"],
            candidate["overconf_wrong"],
            "int",
            lower_is_better=True,
        ),
    ]
    return "\n".join(lines)


def format_report(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    meta: dict[str, str],
) -> str:
    """Format the full benchmark report as markdown.

    :param baseline: metrics dict from compute_metrics.
    :param candidate: metrics dict from compute_metrics.
    :param meta: dict with tool, phase, sample, seed, triggered_by.
    :return: markdown string.
    """
    parts = [
        "<!-- benchmark-result -->",
        f"## Benchmark: {meta.get('tool', 'unknown')}",
        "",
        _metrics_table(baseline, candidate),
        "",
    ]

    # Per-platform breakdown
    b_platforms = baseline.get("by_platform", {})
    c_platforms = candidate.get("by_platform", {})
    all_platforms = sorted(set(b_platforms) | set(c_platforms))

    if len(all_platforms) > 1:
        detail_lines = ["<details><summary>Per-platform breakdown</summary>", ""]
        for plat in all_platforms:
            b_plat = b_platforms.get(
                plat, {"brier": None, "accuracy": None, "overconf_wrong": 0, "n": 0}
            )
            c_plat = c_platforms.get(
                plat, {"brier": None, "accuracy": None, "overconf_wrong": 0, "n": 0}
            )
            detail_lines.append(f"### {plat.title()} (n={b_plat['n']})")
            detail_lines.append("")
            detail_lines.append(_metrics_table(b_plat, c_plat))
            detail_lines.append("")
        detail_lines.append("</details>")
        parts.extend(detail_lines)
        parts.append("")

    # Footer
    footer_parts = [f"{baseline['n']} markets"]
    if meta.get("seed"):
        footer_parts.append(f"seed {meta['seed']}")
    if meta.get("phase"):
        footer_parts.append(f"phase: {meta['phase']}")
    if meta.get("triggered_by"):
        footer_parts.append(f"triggered by @{meta['triggered_by']}")
    parts.append(f"*{' | '.join(footer_parts)}*")

    return "\n".join(parts)


def post_or_update_comment(report: str, pr_number: int, repo: str) -> None:
    """Post or update a benchmark comment on a GitHub PR.

    Finds existing comment by <!-- benchmark-result --> marker and updates it,
    or creates a new one.

    :param report: markdown report string.
    :param pr_number: PR number.
    :param repo: owner/repo string.
    """
    # Find existing benchmark comment
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{pr_number}/comments",
            "--paginate",
            "--jq",
            '[.[] | select(.body | contains("<!-- benchmark-result -->")) | .id] | first',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    existing_id = result.stdout.strip()

    if existing_id and existing_id != "null":
        # Update existing comment
        subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/issues/comments/{existing_id}",
                "-X",
                "PATCH",
                "-f",
                f"body={report}",
                "--silent",
            ],
            check=True,
        )
        print(f"Updated existing comment {existing_id}")
    else:
        # Create new comment
        subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/issues/{pr_number}/comments",
                "-f",
                f"body={report}",
                "--silent",
            ],
            check=True,
        )
        print(f"Posted new comment on PR #{pr_number}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute benchmark metrics and optionally post PR comment.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline.jsonl from replay",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to candidate.jsonl from replay",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="PR number to post comment on",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="GitHub repo (owner/repo)",
    )
    parser.add_argument(
        "--triggered-by",
        type=str,
        default=None,
        help="GitHub username who triggered the benchmark",
    )

    args = parser.parse_args()

    baseline_rows = load_jsonl(args.baseline)
    candidate_rows = load_jsonl(args.candidate)

    if not baseline_rows:
        print("ERROR: No baseline rows found", file=sys.stderr)
        sys.exit(1)

    baseline_metrics = compute_metrics(baseline_rows)
    candidate_metrics = compute_metrics(candidate_rows)

    # Infer tool from data
    tool = baseline_rows[0].get("tool_name", "unknown")

    meta = {
        "tool": tool,
        "triggered_by": args.triggered_by,
    }

    report = format_report(baseline_metrics, candidate_metrics, meta)

    # Always print to stdout
    print(report)

    # Post to PR if requested
    if args.pr and args.repo:
        post_or_update_comment(report, args.pr, args.repo)


if __name__ == "__main__":
    main()
