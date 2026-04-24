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
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.request import Request, urlopen

from benchmark.analyze import PLATFORM_LABELS, VERSION_DELTA_LOW_SAMPLE_STRICT
from benchmark.tools import TOOL_REGISTRY

log = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT_TEMPLATE = f"""\
Summarize this Olas Predict benchmark report for the *{{platform_label}}* deployment using EXACTLY this structure (output will be posted to Slack). Every number in this report is already scoped to {{platform_label}} — do NOT compare platforms or reference the other deployment.

*Summary:* 2-3 sentence high-level takeaway for {{platform_label}} — lead with what changed since last report and in the last 7 days. Only mention all-time numbers for context. Include deltas vs all-time where available.

*Top tools:*
• `tool-name` — Brier `X.XX`, LogLoss `X.XX`, Edge `±X.XX` (n=X), directional accuracy X%, one word on why
(list top 3, rank by Brier. Log Loss and Edge are shown alongside for context)

*Worst tools:*
• `tool-name` — Brier `X.XX`, LogLoss `X.XX`, Edge `±X.XX` (n=X), directional accuracy X%, one word on why
(list bottom 3, ignore tools with 0% reliability or < 50 predictions)

*Deployment status:* if the report has a "Tool Deployment Status" section, include only deployments whose name starts with the lowercase of "{{platform_label}}". One line per qualifying deployment listing its disabled tools. Skip deployments with no disabled tools, and skip the entire block if no qualifying deployments have content. The "Tool Deployment Status" section is fleet-wide in the report source — do NOT mention deployments belonging to other platforms. Note any fetch-failure banner briefly, but only when it concerns a {{platform_label}}-linked deployment.

*Category performance:* from the "Category Performance" section, list every category with sufficient data (skip rows flagged "insufficient data"). Use format: • `category` — Brier `X.XX`, Edge `±X.XX` (n=X). Call out the single strongest and weakest category inline (e.g. " — strongest" / " — weakest"). This is the answer to "where do our agents do well vs poorly" for {{platform_label}} — always include when data is present. IMPORTANT: a category with "yes rate: 0%" or "yes rate: 100%" has homogeneous outcomes — a low Brier there reflects the base rate, not prediction skill. If you cite such a category as "strongest", append " (homogeneous outcomes — reflects base rate)" so the reader isn't misled.

*Tool × Category highlights:* from the "Tool × Category" section, pick 2–4 standout tool-category combinations (best performers, worst performers, or weaknesses). Only use rows above the sample-size threshold — never cite rows from the "below n=X threshold omitted" list. If all tools underperform on a category, say that explicitly ("tools struggle on X across the board"). FALLBACK: if fewer than 2 rows clear the sample-size threshold in the Tool × Category ranking table, do NOT cite any sparse examples or fabricate — write exactly "insufficient tool × category data" as the only bullet in this section.

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
If the report includes "Diagnostic Edge Metrics", summarize:
• Conditional accuracy: X% tool-wins when disagreeing (n=X) — when the tool would trigger a trade, how often is it closer to truth than the market?
• Disagreement Brier (large trade): X.XX — prediction accuracy on high-disagreement questions where PnL impact is highest
• Directional bias: ±X.XX — positive = tool overestimates, negative = underestimates, near 0 = no systematic bias
Only include this section if the report has diagnostic metric data. Skip if insufficient data.

*Recommended actions:* 2-3 concrete next steps for {{platform_label}} based on the data. If edge is negative for all tools, this is important — recommend specific improvements.

Rules:
- Tool names with hyphens vs underscores are DIFFERENT tools — use exact names.
- Wrap tool names, Brier scores, and Edge scores in backticks.
- Slack mrkdwn only: *bold* (single asterisk), `code`. No **double asterisks**.
- No greetings or preamble.
- Edge over market: positive = tool beats market, negative = market beats tool. This is a system-level diagnostic — it shows whether prediction accuracy translates to trading value, but tools are ranked by Brier (prediction quality).
- "Accuracy" in the report means "Directional Accuracy" — it excludes predictions at exactly 0.5 (no signal). Include the no-signal rate if it's notable.
- Log Loss: like Brier but punishes confidently-wrong predictions harder. Include alongside Brier.
- ECE (Expected Calibration Error): how well calibrated predictions are. Include if present.
- Some tools listed below are third-party (not ours). Completely exclude them — never mention, rank, compare, or recommend actions for third-party tools anywhere in the summary."""


_VALID_PLATFORM_LABELS: frozenset[str] = frozenset(PLATFORM_LABELS.values())


def _build_system_prompt(platform_label: str) -> str:
    """Fill in the platform label on the system prompt template.

    :param platform_label: deployment name to reference throughout the
        summary. Must match one of the labels defined in
        ``benchmark.analyze.PLATFORM_LABELS`` so a typo at the workflow
        level (e.g. ``--platform-label Omenstrap``) surfaces loudly instead
        of reaching the LLM.
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
    return SUMMARY_SYSTEM_PROMPT_TEMPLATE.format(platform_label=platform_label)


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
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": _build_system_prompt(platform_label),
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
