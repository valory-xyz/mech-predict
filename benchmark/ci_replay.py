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
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from benchmark.io import load_jsonl

# Status buckets emitted by parse_response. Listed in a fixed order so the PR
# comment breakdown is diffable across runs.
PARSE_STATUS_BUCKETS = ("valid", "missing_fields", "malformed", "error")

# Upper bound on failure bodies to inline in the PR comment. Keeps the comment
# within GitHub's 65k-character limit even when every market fails, and keeps
# CI log output readable.
MAX_FAILURE_BODIES_IN_COMMENT = 5
MAX_FAILURE_BODY_CHARS = 600


def _compute_parse_reliability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise parse reliability from ``prediction_parse_status`` values.

    The on-chain mech protocol only delivers the four-field JSON; a row that
    fails to parse is still an on-chain deliver + payment (see issue #221).
    Reliability is therefore a first-class metric, not a subset of Brier.

    :param rows: list of prediction rows with 'prediction_parse_status'.
    :return: dict with total, valid, parse_rate, breakdown.
    """
    total = len(rows)
    breakdown = {b: 0 for b in PARSE_STATUS_BUCKETS}
    for r in rows:
        status = r.get("prediction_parse_status") or "error"
        if status not in breakdown:
            status = "error"
        breakdown[status] += 1
    valid = breakdown["valid"]
    return {
        "total": total,
        "valid": valid,
        "parse_rate": (valid / total) if total else None,
        "breakdown": breakdown,
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Brier score, accuracy, and overconfident-wrong count.

    :param rows: list of prediction rows with p_yes, final_outcome.
    :return: dict with brier, accuracy, overconf_wrong, n, parse_reliability,
        and by_platform.
    """
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_platform[row.get("platform") or "unknown"].append(row)

    def _metrics(subset: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [r for r in subset if r.get("p_yes") is not None]
        n = len(valid)
        if n == 0:
            return {
                "brier": None,
                "directional_accuracy": None,
                "n_directional": 0,
                "overconf_wrong": 0,
                "overconf_wrong_rate": None,
                "n": 0,
            }

        brier_sum = 0.0
        correct = 0
        n_directional = 0
        overconf_wrong = 0
        for r in valid:
            p_yes = r["p_yes"]
            outcome = r["final_outcome"]
            outcome_val = 1.0 if outcome else 0.0
            brier_sum += (p_yes - outcome_val) ** 2
            predicted_yes = p_yes > 0.5
            if p_yes != 0.5:
                n_directional += 1
                if predicted_yes == outcome:
                    correct += 1
            if max(p_yes, 1 - p_yes) > 0.80 and predicted_yes != outcome:
                overconf_wrong += 1

        return {
            "brier": brier_sum / n,
            "directional_accuracy": (
                correct / n_directional if n_directional > 0 else None
            ),
            "n_directional": n_directional,
            "overconf_wrong": overconf_wrong,
            # Denominator is n (not n_directional) to normalize against
            # total valid sample size; p_yes==0.5 rows can never be
            # overconfident-wrong (max(0.5,0.5)=0.5 < 0.80 threshold)
            # so the numerator is unaffected.
            "overconf_wrong_rate": round(overconf_wrong / n, 4) if n > 0 else None,
            "n": n,
        }

    overall = _metrics(rows)
    overall["parse_reliability"] = _compute_parse_reliability(rows)
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
    if baseline_val is None or candidate_val is None:
        return "N/A"
    if baseline_val == 0:
        if candidate_val == 0:
            return "0.0%"
        return "+∞%" if candidate_val > 0 else "-∞%"
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
            "Directional Accuracy",
            baseline["directional_accuracy"],
            candidate["directional_accuracy"],
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
        _fmt_metric_row(
            "Overconf-wrong rate",
            baseline["overconf_wrong_rate"],
            candidate["overconf_wrong_rate"],
            "float",
            lower_is_better=True,
        ),
    ]
    return "\n".join(lines)


def _load_filter_stats(candidate_path: Path) -> Optional[dict[str, Any]]:
    """Load ``filter_stats.json`` written alongside candidate.jsonl, if present.

    :param candidate_path: path to candidate.jsonl.
    :return: parsed stats dict, or None if absent / unreadable.
    """
    stats_path = candidate_path.parent / "filter_stats.json"
    if not stats_path.exists():
        return None
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _format_reliability_block(
    candidate: dict[str, Any],
    failure_rows: list[dict[str, Any]],
    filter_stats: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Render the Reliability section: candidate parse rate + pre-filter stats.

    Baseline parse rate is 100% by construction (the enrich step drops
    non-valid rows), so it is not reported as part of a baseline-vs-candidate
    comparison. The two observations here are one-sided:

    - candidate parse rate: did the PR's code produce parseable responses?
    - pre-filter: did the upstream enrich filter's "drop non-valid" invariant
      hold? (non-zero ``not_valid_parse`` rejects = silent regression of the
      invariant that made baseline 100% trustworthy in the first place)

    :param candidate: metrics dict including ``parse_reliability``.
    :param failure_rows: rows from ``candidate_failures.jsonl`` (may be empty).
    :param filter_stats: optional dict from the filter_stats.json sidecar.
    :return: markdown lines.
    """
    c_rel = candidate["parse_reliability"]
    c_total = c_rel["total"]
    c_valid = c_rel["valid"]
    c_rate = c_rel["parse_rate"] or 0.0
    c_marker = "✅" if c_valid == c_total else "⚠️"

    lines: list[str] = [
        "**Reliability**",
        "",
        f"- Candidate parse rate: {c_valid}/{c_total} "
        f"({c_rate * 100:.1f}%) {c_marker}",
    ]

    # Breakdown only surfaces when candidate drifted — keeps the happy-path
    # line count short.
    if c_valid < c_total:
        bd = c_rel["breakdown"]
        breakdown_str = ", ".join(f"{k}={bd[k]}" for k in PARSE_STATUS_BUCKETS)
        lines.append(f"  - Breakdown: {breakdown_str}")

    if filter_stats is not None:
        r = filter_stats.get("rejected", {}) or {}
        accepted = filter_stats.get("accepted", 0)
        total_rej = sum(r.values())
        not_valid = r.get("not_valid_parse", 0)
        pf_marker = "✅" if not_valid == 0 else "⚠️"
        lines.append(
            f"- Pre-filter (enrich): {accepted} accepted, {total_rej} rejected, "
            f"not_valid_parse={not_valid} {pf_marker}"
        )
        # Scoping breakdown only when any rows were rejected — otherwise four
        # zeroes just add noise to the happy path.
        if total_rej > 0:
            scoping = ", ".join(
                [
                    f"wrong_tool={r.get('wrong_tool', 0)}",
                    f"no_deliver_id={r.get('no_deliver_id', 0)}",
                    f"no_outcome={r.get('no_outcome', 0)}",
                    f"older_than_cutoff={r.get('older_than_cutoff', 0)}",
                ]
            )
            lines.append(f"  - Scoping: {scoping}")

    lines.append("")

    if failure_rows:
        n_shown = min(len(failure_rows), MAX_FAILURE_BODIES_IN_COMMENT)
        lines.append(
            f"<details><summary>Candidate parse failures "
            f"({len(failure_rows)} total; first {n_shown} shown)</summary>"
        )
        lines.append("")
        for fr in failure_rows[:n_shown]:
            body = (fr.get("raw_response") or "").replace("```", "ʼʼʼ")
            if len(body) > MAX_FAILURE_BODY_CHARS:
                body = body[:MAX_FAILURE_BODY_CHARS] + "…"
            question = (fr.get("question_text", "")[:120]).replace("`", "ʼ")
            lines.append(
                f"**{fr.get('prediction_parse_status', 'error')}** — `{question}`"
            )
            lines.append("")
            lines.append("```")
            lines.append(body)
            lines.append("```")
            lines.append("")
        lines.append("</details>")
        lines.append("")

    return lines


def format_report(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    meta: dict[str, str],
    failure_rows: list[dict[str, Any]] | None = None,
    filter_stats: Optional[dict[str, Any]] = None,
) -> str:
    """Format the full benchmark report as markdown.

    :param baseline: metrics dict from compute_metrics.
    :param candidate: metrics dict from compute_metrics.
    :param meta: dict with tool, phase, sample, seed, triggered_by.
    :param failure_rows: optional parse-failure rows loaded from
        candidate_failures.jsonl. When non-empty, bodies are inlined in a
        collapsed <details> block.
    :param filter_stats: optional ``{accepted, rejected}`` dict from
        filter_stats.json sidecar. When present, a Pre-filter block is
        rendered so an upstream filter regression would be visible.
    :return: markdown string.
    """
    tool = meta.get("tool", "unknown")
    parts: list[str] = [
        f"<!-- benchmark-result:{tool} -->",
        f"## Benchmark: {tool}",
        "",
        _metrics_table(baseline, candidate),
        "",
    ]
    parts.extend(_format_reliability_block(candidate, failure_rows or [], filter_stats))

    # Per-platform breakdown
    b_platforms = baseline.get("by_platform", {})
    c_platforms = candidate.get("by_platform", {})
    all_platforms = sorted(set(b_platforms) | set(c_platforms))

    if len(all_platforms) > 1:
        detail_lines = ["<details><summary>Per-platform breakdown</summary>", ""]
        for plat in all_platforms:
            b_plat = b_platforms.get(
                plat,
                {
                    "brier": None,
                    "directional_accuracy": None,
                    "n_directional": 0,
                    "overconf_wrong": 0,
                    "overconf_wrong_rate": None,
                    "n": 0,
                },
            )
            c_plat = c_platforms.get(
                plat,
                {
                    "brier": None,
                    "directional_accuracy": None,
                    "n_directional": 0,
                    "overconf_wrong": 0,
                    "overconf_wrong_rate": None,
                    "n": 0,
                },
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


def post_comment(report: str, pr_number: int, repo: str) -> None:
    """Post a benchmark comment on a GitHub PR.

    Always creates a new comment so results flow chronologically with
    the conversation and history is preserved.

    :param report: markdown report string.
    :param pr_number: PR number.
    :param repo: owner/repo string.
    """
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
    print(f"Posted benchmark comment on PR #{pr_number}")


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

    # prompt_replay writes candidate_failures.jsonl next to candidate.jsonl
    # whenever any candidate failed to parse. Missing file = zero failures.
    failures_path = args.candidate.parent / "candidate_failures.jsonl"
    failure_rows = load_jsonl(failures_path) if failures_path.exists() else []

    # prompt_replay copies filter_stats.json into output_dir when the sidecar
    # was present at enrich time. Missing file = no stats (older pipelines).
    filter_stats = _load_filter_stats(args.candidate)

    baseline_metrics = compute_metrics(baseline_rows)
    candidate_metrics = compute_metrics(candidate_rows)

    # Infer tool from data
    tool = baseline_rows[0].get("tool_name", "unknown")

    meta = {
        "tool": tool,
        "triggered_by": args.triggered_by,
    }

    report = format_report(
        baseline_metrics,
        candidate_metrics,
        meta,
        failure_rows=failure_rows,
        filter_stats=filter_stats,
    )

    # Always print to stdout
    print(report)

    # Post to PR if requested
    if args.pr and args.repo:
        post_comment(report, args.pr, args.repo)


if __name__ == "__main__":
    main()
