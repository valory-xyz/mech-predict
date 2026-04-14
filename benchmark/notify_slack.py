"""
Post an AI-summarized benchmark report to Slack.

Reads a markdown report, sends it to OpenAI for a concise Slack-formatted
summary, and posts the result via an incoming webhook.

Usage:
    python benchmark/notify_slack.py --report benchmark/results/report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from benchmark.tools import TOOL_REGISTRY

log = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """\
Summarize this Olas Predict benchmark report using EXACTLY this structure (output will be posted to Slack).

*Summary:* 2-3 sentence high-level takeaway — lead with what changed since last report and in the last 7 days. Only mention all-time numbers for context. Include deltas vs all-time where available.

*Top tools:*
• `tool-name` — Brier `X.XX`, LogLoss `X.XX`, Edge `±X.XX` (n=X), directional accuracy X%, one word on why
(list top 3, rank by Brier. Log Loss and Edge are shown alongside for context)

*Worst tools:*
• `tool-name` — Brier `X.XX`, LogLoss `X.XX`, Edge `±X.XX` (n=X), directional accuracy X%, one word on why
(list bottom 3, ignore tools with 0% reliability or < 50 predictions)

*Platform performance:*
• `platform` — Brier `X.XX`, LogLoss `X.XX`, Edge `±X.XX` (n=X), BSS `±X.XX`, n=X
(list all platforms. Edge = how much tool beats market consensus; positive = profitable signal)

*Edge by difficulty:* if the report has "Platform × Difficulty" data, summarize which difficulty level has the best/worst edge per platform (1 line per platform)

*Category performance:* from the "Category Performance" section, list every category with sufficient data (skip rows flagged "insufficient data"). Use format: • `category` — Brier `X.XX`, Edge `±X.XX` (n=X). Call out the single strongest and weakest category inline (e.g. " — strongest" / " — weakest"). This is the fleet-level answer to "where do our agents do well vs poorly" — always include when data is present. IMPORTANT: a category with "yes rate: 0%" or "yes rate: 100%" has homogeneous outcomes — a low Brier there reflects the base rate, not prediction skill. If you cite such a category as "strongest", append " (homogeneous outcomes — reflects base rate)" so the reader isn't misled.

*Fleet × Category highlights:* from the "Tool × Category" section, pick 2–4 standout tool-category combinations (best performers, worst performers, or fleet-wide weaknesses). Only use rows above the sample-size threshold — never cite rows from the "below n=X threshold omitted" list. If all tools underperform on a category, say that explicitly ("fleet struggles on X across tools").

*Regressions:* any tools or metrics that worsened vs prior period. Say "None" if trend data shows no worsening. "Regression" means worse over TIME, not just a bad score.

*Tool versions:* if the report has a "Tool × Version × Mode" (cumulative or 7d), or "Version Deltas" section, summarize per-version observations: which tool versions changed, mode (production_replay vs tournament), Brier delta where shown. Show full hashes (no truncation) wrapped in backticks. CRITICAL: any row whose n cell ends in ⚠ or whose underlying n is below 30 is a small sample — when mentioning such a row you MUST lead with "small sample (n=X)" before stating any metric, and you MUST NOT use the words "best", "highest", "lowest", "worst", or any superlative for that row. Compare across versions only when both have n >= 30. Skip the section entirely if no Tool × Version × Mode or Version Deltas content is present.

*Diagnostics:*
If the report includes "Diagnostic Edge Metrics", summarize:
• Conditional accuracy: X% tool-wins when disagreeing (n=X) — when the tool would trigger a trade, how often is it closer to truth than the market?
• Disagreement Brier (large trade): X.XX — prediction accuracy on high-disagreement questions where PnL impact is highest
• Directional bias: ±X.XX — positive = tool overestimates, negative = underestimates, near 0 = no systematic bias
Only include this section if the report has diagnostic metric data. Skip if insufficient data.

*Recommended actions:* 2-3 concrete next steps based on the data. If edge is negative for all tools, this is important — recommend specific improvements.

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


def summarize_report(report_text: str, api_key: str) -> str:
    """Call OpenAI to produce a short Slack-formatted summary."""
    ownership = _tool_ownership_context(report_text)
    user_content = report_text
    if ownership:
        user_content = f"{ownership}\n\n{report_text}"
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
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


def main() -> None:
    """Read report, summarize, post. Skip gracefully if keys missing."""
    parser = argparse.ArgumentParser(description="Post benchmark summary to Slack")
    parser.add_argument("--report", type=Path, required=True, help="Path to report.md")
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

    # Extract heading from report (e.g. "# Benchmark Report — 2026-04-03")
    heading = "*Benchmark Report*"
    first_line = report_text.split("\n", 1)[0]
    if first_line.startswith("# "):
        heading = f"*{first_line.lstrip('# ').strip()}*"

    log.info("Summarizing report with %s...", MODEL)
    summary = f"{heading}\n\n{summarize_report(report_text, api_key)}"

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
