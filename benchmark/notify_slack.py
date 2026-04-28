"""
Post an AI-summarized benchmark report to Slack for one platform deployment.

Reads a per-platform markdown report (``report_<platform>.md``), sends it
to OpenAI for a concise Slack-formatted summary scoped to the named
deployment, and posts the result via an incoming webhook.

Usage:
    python -m benchmark.notify_slack --report benchmark/results/report_omen.md --platform-label Omenstrat
    python -m benchmark.notify_slack --report benchmark/results/report_polymarket.md --platform-label Polystrat

When ``--platform-label`` is omitted, the deployment name is inferred
from the report filename (``report_omen.md`` -> Omenstrat,
``report_polymarket.md`` -> Polystrat).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.request import Request, urlopen

from benchmark.analyze import (
    PLATFORM_LABELS,
    ROLLING_WINDOW_DAYS,
    VERSION_DELTA_LOW_SAMPLE_STRICT,
)
from benchmark.scorer import MIN_SAMPLE_SIZE
from benchmark.tools import TOOL_REGISTRY

log = logging.getLogger(__name__)

_PROMPT_HEADER = f"""\
Summarize this Olas Predict benchmark report for the *{{platform_label}}* deployment using EXACTLY this structure (output will be posted to Slack). The report carries three windows per metric: `Current {ROLLING_WINDOW_DAYS}d` (trailing {ROLLING_WINDOW_DAYS}-day aggregate), `All-Time` (cumulative), and `Prev {ROLLING_WINDOW_DAYS}d` (the immediately preceding non-overlapping {ROLLING_WINDOW_DAYS}-day window). Never mix numbers across windows; if you cite a value, state which window it came from. Do NOT compare platforms, reference tools or deployments belonging to other platforms, or cite metrics from another platform's rows.

*Summary:* 2-3 sentence high-level takeaway for {{platform_label}}. Lead with the Current-{ROLLING_WINDOW_DAYS}d platform Brier (from the "Platform Snapshot" section). Then name the direction of change: "Δ vs All-Time" from the "Platform Historical Comparison" row for Brier, and "Δ vs Prev {ROLLING_WINDOW_DAYS}d" from the same row. If either delta shows `insufficient data`, say so plainly instead of guessing.

*Eligibility for the tool ranking section below:* a row is eligible only if its Current {ROLLING_WINDOW_DAYS}d n is at least {MIN_SAMPLE_SIZE} AND the row carries no ⚠ low sample / ⚠ all malformed flag."""


_PROMPT_RANKING_NONE = f"""\
*Tool performance:*
• _no eligible tools — every row in the Tool Historical Comparison table is below n={MIN_SAMPLE_SIZE} or flagged_"""


_PROMPT_RANKING_ALL = f"""\
*Tool performance:*
• `tool-name` — Current {ROLLING_WINDOW_DAYS}d Brier `X.XXXX` (n=X), Δ vs All-Time `±X.XXXX direction`, Δ vs Prev {ROLLING_WINDOW_DAYS}d `±X.XXXX direction`
(list ALL eligible rows from the "Tool Historical Comparison" table, sorted by Current {ROLLING_WINDOW_DAYS}d Brier ascending. Use the exact delta strings from that table; if a delta is `insufficient data` or `no prev window`, write those words verbatim — never invent a number.)"""


def _prompt_ranking_split(top_k: int) -> str:
    """Top-K + Worst-K ranking block for ``top_k`` ≥ 1.

    :param top_k: number of rows in each of the Top tools and Worst tools
        bullets. Caller must guarantee the eligible set has strictly more
        than ``2 * top_k`` rows so the two slices are non-overlapping.
    :return: ranking block as a string suitable for inserting between
        the prompt header and footer.
    """
    return f"""\
*Top tools:*
• `tool-name` — Current {ROLLING_WINDOW_DAYS}d Brier `X.XXXX` (n=X), Δ vs All-Time `±X.XXXX direction`, Δ vs Prev {ROLLING_WINDOW_DAYS}d `±X.XXXX direction`
(list top {top_k} eligible rows from the "Tool Historical Comparison" table, sorted by Current {ROLLING_WINDOW_DAYS}d Brier ascending. Use the exact delta strings from that table; if a delta is `insufficient data` or `no prev window`, write those words verbatim — never invent a number.)

*Worst tools:*
• `tool-name` — Current {ROLLING_WINDOW_DAYS}d Brier `X.XXXX` (n=X), Δ vs All-Time `±X.XXXX direction`, Δ vs Prev {ROLLING_WINDOW_DAYS}d `±X.XXXX direction`
(list bottom {top_k} eligible rows from the same table, sorted by Current {ROLLING_WINDOW_DAYS}d Brier descending.)"""


_PROMPT_FOOTER = f"""\
*Deployment status:* if the report has a "Tool Deployment Status ({{platform_label}})" section, list one line per deployment with its count of active tools only (do NOT enumerate the tool names — the full report has them and the Slack message stays readable). Skip deployments marked `⚠️ unavailable` after noting briefly that their config fetch failed.

*Tool × Category:* from the "Tool × Category (Current {ROLLING_WINDOW_DAYS}d)" section, list every cell that clears the sample-size threshold. Use format: • `tool` × `category` — Brier `X.XXXX` (n=X, Current {ROLLING_WINDOW_DAYS}d), DirAcc X%, Always-majority X%, DA lift `±X.XXXX`. If DA lift is ≤ 0 say " — no lift over always-majority" inline so the reader isn't misled by a low Brier on a homogeneous-outcome cell. Never cite rows from the "below n=X threshold omitted" list. FALLBACK: if fewer than 2 rows clear the threshold, write exactly "insufficient tool × category data" as the only bullet in this section.

If the "Tool × Category Historical Comparison" table has any row where `Δ vs Prev {ROLLING_WINDOW_DAYS}d` is a signed number (not `insufficient data`, not `no prev window`), add a single follow-up bullet naming the largest absolute-value movement and its direction.

*Tool versions:* If the report has a "Version Deltas" section, summarize up to 5 of the most significant flagged changes, one bullet per row.

REQUIRED bullet format — reproduce exactly, with both versions wrapped in backticks:
• `tool-name` (mode): `baseline-label` → `candidate-label` — Brier Δ X.XXXX direction (n_b=X, n_c=X)

Example (copy this style exactly):
• `prediction-request-reasoning` (production_replay): `v0.16.5` → `v0.17.0` — Brier Δ -0.0725 improved (n_b=433, n_c=4485)

Rules:
- The baseline and candidate labels come verbatim from the Baseline and Candidate columns of the report's "**vs prior version:**" sub-table (they look like `v0.17.0` or `untagged@bafybei1`). Never invent labels, never truncate, never summarize them as generic "v1/v2".
- Only include rows where min(n_b, n_c) ≥ {VERSION_DELTA_LOW_SAMPLE_STRICT} and direction is "regressed" or "improved". Never include rows marked ⚠ — the flagged samples are too small to be reliable.
- Skip this section entirely if the Version Deltas section is absent or has no rows without ⚠.

*Tournament callouts:* If the report has a "Tournament Callouts" section, list each callout as a single bullet: tool name, release-tag labels for both tournament and production versions (in backticks, e.g. `v0.17.2` and `v0.17.0`), tournament Brier + n, production Brier + n, Brier Δ. Lead promotion candidates with "promotion candidate:" and tournament regressions with "watch:". Skip this section entirely if no Tournament Callouts section is present in the report.

*Diagnostics:*
If the report has a "Diagnostics Historical Comparison" section, for each tool that carries at least one row with a signed delta (not `insufficient data`, not `no prev window`), summarize up to two metrics with the largest movement. Use format: • `tool` — `metric` Current {ROLLING_WINDOW_DAYS}d `X.XXXX` (n=X), Δ vs All-Time `±X.XXXX direction`, Δ vs Prev {ROLLING_WINDOW_DAYS}d `±X.XXXX direction`. Skip the section if no tool has a signed delta.

*Reliability:* from the "Reliability & Parse Quality" comparison table, list every tool whose Current {ROLLING_WINDOW_DAYS}d Reliability or Valid % has a non-`insufficient data` delta vs All-Time and the delta is negative (regression). Use format: • `tool` — Reliability X% (n=X) vs All-Time X% (Δ -X.XXXX worse). If no tool regressed, skip this section.

*Recommended actions:* 2-3 concrete next steps for {{platform_label}} based on the Current {ROLLING_WINDOW_DAYS}d data. Anchor each action to a specific row in the comparison tables. If the Current {ROLLING_WINDOW_DAYS}d → Prev {ROLLING_WINDOW_DAYS}d delta shows a regression, call it out explicitly.

Rules:
- Never mix windows in a single claim. Every cited number must be paired with its window label (Current {ROLLING_WINDOW_DAYS}d, All-Time, or Prev {ROLLING_WINDOW_DAYS}d).
- Deltas never stand alone — always cite both sides' n (or state the delta was `insufficient data` / `no prev window` verbatim from the table).
- Do not make claims from cells flagged ⚠ low sample or all malformed.
- Tool names with hyphens vs underscores are DIFFERENT tools — use exact names.
- Wrap tool names, Brier scores, and Edge scores in backticks.
- Slack mrkdwn only: *bold* (single asterisk), `code`. No **double asterisks**.
- No greetings or preamble.
- Edge over market: positive = tool beats market, negative = market beats tool. Read it as a system-level diagnostic — tools are still ranked by Brier.
- "Accuracy" in the report means "Directional Accuracy" — it excludes predictions at exactly 0.5 (no signal).
- Log Loss: like Brier but punishes confidently-wrong predictions harder.
- Some tools listed below are third-party (not ours). Completely exclude them — never mention, rank, compare, or recommend actions for third-party tools anywhere in the summary."""


_VALID_PLATFORM_LABELS: frozenset[str] = frozenset(PLATFORM_LABELS.values())


def _count_eligible_tools(report_text: str) -> int:
    """Count tools in the Tool Historical Comparison table that pass the eligibility floor.

    A row is eligible when its Current-window n is at least
    ``MIN_SAMPLE_SIZE`` and the tool name carries no ⚠ flag from
    ``_sample_label`` (``⚠ low sample`` or ``⚠ all malformed``).

    :param report_text: full markdown report for one platform.
    :return: count of eligible rows; ``0`` when the section is absent.
    """
    # The block terminates on the next ``^## `` heading OR at end-of-report
    # so this helper is robust to Tool Historical Comparison being the
    # final section.
    block_match = re.search(
        r"^## Tool Historical Comparison\n(.*?)(?=^## |\Z)",
        report_text,
        re.S | re.M,
    )
    if block_match is None:
        return 0

    eligible = 0
    for line in block_match.group(1).splitlines():
        if not line.startswith("| **"):
            continue
        if "⚠" in line:
            continue
        n_match = re.search(r"\(n=(\d+)\)", line)
        if n_match and int(n_match.group(1)) >= MIN_SAMPLE_SIZE:
            eligible += 1
    return eligible


def _compute_top_k(eligible_count: int) -> int:
    """Return the per-side bullet count for the Top/Worst split.

    Constraint: Top K and Worst K must be disjoint, so ``K + K < N``,
    i.e. ``K <= floor((N - 1) / 2)``. Capped at 3 so the message stays
    scannable on large deployments.

    A return of ``0`` is the "render a single ranked list" signal —
    used when ``N`` is too small to support a non-overlapping split
    (``N`` is 0, 1, or 2).

    :param eligible_count: number of tools that clear the eligibility
        floor in the Tool Historical Comparison table.
    :return: ``K`` for the Top/Worst split, or ``0`` to switch to a
        combined "Tool performance" listing.
    """
    if eligible_count <= 2:
        return 0
    return min(3, (eligible_count - 1) // 2)


def _build_system_prompt(platform_label: str, eligible_count: int) -> str:
    """Assemble the system prompt with the right tool-ranking block.

    The ranking block dispatches on ``eligible_count`` so the LLM
    only ever sees one section convention per request:

    - ``0`` eligible tools → placeholder bullet under ``*Tool performance:*``
    - ``1-2`` eligible → all eligible rows under ``*Tool performance:*``
    - ``3+`` eligible → ``*Top tools:*`` / ``*Worst tools:*`` with K bullets
      each, where K = ``min(3, floor((N-1)/2))``

    :param platform_label: deployment name to thread into the prompt.
        Must be one of ``benchmark.analyze.PLATFORM_LABELS`` values.
    :param eligible_count: number of tools that clear the eligibility
        floor in the markdown report's Tool Historical Comparison table.
        Drives the ranking-block dispatch.
    :return: fully formatted system prompt string.
    :raises ValueError: when ``platform_label`` is empty or unknown.
    """
    if not platform_label:
        raise ValueError("platform_label must be non-empty")
    if platform_label not in _VALID_PLATFORM_LABELS:
        raise ValueError(
            f"platform_label must be one of {sorted(_VALID_PLATFORM_LABELS)},"
            f" got {platform_label!r}"
        )

    top_k = _compute_top_k(eligible_count)
    if eligible_count == 0:
        ranking_block = _PROMPT_RANKING_NONE
    elif top_k == 0:
        ranking_block = _PROMPT_RANKING_ALL
    else:
        ranking_block = _prompt_ranking_split(top_k)

    template = "\n\n".join([_PROMPT_HEADER, ranking_block, _PROMPT_FOOTER])
    return template.format(platform_label=platform_label)


MODEL = "gpt-4.1-mini"


OUR_TOOLS: set[str] = set(TOOL_REGISTRY)


def _tool_ownership_context(report_text: str) -> str:
    """List third-party tools found in the report so the LLM can ignore them."""
    report_tools: list[str] = []
    for line in report_text.splitlines():
        # "1. **tool-name** — ..."
        if line.lstrip()[:3].rstrip(".").isdigit() and "**" in line:
            parts = line.split("**")
            if len(parts) >= 2:
                report_tools.append(parts[1])

    seen = dict.fromkeys(report_tools)
    theirs = [t for t in seen if t not in OUR_TOOLS]
    if not theirs:
        return ""
    return f"Third-party tools (ignore these): {', '.join(theirs)}"


def summarize_report(report_text: str, api_key: str, platform_label: str) -> str:
    """Call OpenAI to produce a short Slack-formatted summary.

    :param report_text: full markdown report for a single platform.
    :param api_key: OpenAI API key.
    :param platform_label: deployment name (e.g. ``Omenstrat``) threaded into
        the system prompt so the LLM frames the summary correctly and never
        mixes platforms.
    :return: Slack-formatted summary string.
    """
    ownership = _tool_ownership_context(report_text)
    user_content = report_text
    if ownership:
        user_content = f"{ownership}\n\n{report_text}"
    eligible_count = _count_eligible_tools(report_text)
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": _build_system_prompt(platform_label, eligible_count),
                },
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 1200,
            "temperature": 0.2,
        }
    ).encode()

    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())

    return body["choices"][0]["message"]["content"].strip()


def _build_report_url() -> str | None:
    """Build a link to the benchmark report artifact.

    Prefers ``REPORT_ARTIFACT_URL`` (set by a prior workflow step that
    queries the API after uploading the artifact).  Falls back to the
    generic run URL when the exact artifact link isn't available.

    :return: URL string, or None if not running in CI.
    """
    artifact_url = os.environ.get("REPORT_ARTIFACT_URL")
    if artifact_url:
        return artifact_url
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


def post_to_slack(webhook_url: str, summary: str) -> None:
    """POST a message to a Slack incoming webhook."""
    payload = json.dumps({"text": summary}).encode()
    req = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=15) as resp:
        resp.read()


_PLATFORM_LABEL_BY_STEM: Mapping[str, str] = MappingProxyType(
    {f"report_{key}": label for key, label in PLATFORM_LABELS.items()}
)


def _infer_platform_label(report_path: Path) -> str | None:
    """Derive the deployment label from the report filename.

    ``analyze.py`` writes ``report_<platform>.md`` per key in
    ``PLATFORM_LABELS``; an explicit ``--platform-label`` CLI arg
    overrides this inference.

    :param report_path: path to the report markdown file.
    :return: deployment label, or None if the filename doesn't match.
    """
    return _PLATFORM_LABEL_BY_STEM.get(report_path.stem)


def main() -> None:
    """Read report, summarize, post. Skip gracefully if keys missing."""
    parser = argparse.ArgumentParser(description="Post benchmark summary to Slack")
    parser.add_argument("--report", type=Path, required=True, help="Path to report.md")
    parser.add_argument(
        "--platform-label",
        default=None,
        help=(
            "Deployment name (e.g. 'Omenstrat', 'Polystrat') threaded into the "
            "LLM summary prompt. Inferred from the report filename when omitted."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print summary without posting to Slack"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")

    if not api_key:
        log.info("OPENAI_API_KEY not set, skipping Slack notification.")
        return

    if not webhook_url and not args.dry_run:
        log.info("SLACK_WEBHOOK_URL not set, skipping Slack notification.")
        return

    if not args.report.exists():
        log.warning("Report not found: %s", args.report)
        sys.exit(1)

    report_text = args.report.read_text(encoding="utf-8")
    if not report_text.strip():
        log.warning("Report is empty: %s", args.report)
        return

    platform_label = args.platform_label or _infer_platform_label(args.report)
    if platform_label is None:
        log.error(
            "Cannot determine platform label for %s. Pass --platform-label "
            "or name the report file report_<platform>.md.",
            args.report,
        )
        sys.exit(1)

    # Extract heading from report (e.g. "# Benchmark Report (Omenstrat) — 2026-04-03")
    heading = "*Benchmark Report*"
    first_line = report_text.split("\n", 1)[0]
    if first_line.startswith("# "):
        heading = f"*{first_line.lstrip('# ').strip()}*"

    log.info("Summarizing %s report with %s...", platform_label, MODEL)
    summary = f"{heading}\n\n{summarize_report(report_text, api_key, platform_label)}"

    # Append link to full report if running in GitHub Actions
    report_url = _build_report_url()
    if report_url:
        summary += f"\n<{report_url}|Full report>"

    if args.dry_run:
        print(summary)
        return

    log.info("Posting to Slack...")
    post_to_slack(webhook_url, summary)
    log.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
