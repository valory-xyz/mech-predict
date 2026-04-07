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

*Summary:* 2-3 sentence high-level takeaway — what's working, what's not, any trends.

*Top tools:*
• `tool-name` — Brier `X.XX`, accuracy X%, one word on why (e.g. "strong calibration")
(list top 3)

*Worst tools:*
• `tool-name` — Brier `X.XX`, accuracy X%, one word on why (e.g. "overconfident", "anti-predictive")
(list bottom 3, ignore tools with 0% reliability or < 50 predictions)

*Weak categories:* list categories with Brier > 0.40 and brief note

*Regressions:* any tools or metrics that worsened vs prior period. Say "None" if trend data shows no worsening. "Regression" means worse over TIME, not just a bad score.

*Recommended actions:* 2-3 concrete next steps based on the data

Rules:
- Tool names with hyphens vs underscores are DIFFERENT tools — use exact names.
- Wrap tool names and Brier scores in backticks.
- Slack mrkdwn only: *bold* (single asterisk), `code`. No **double asterisks**.
- No greetings or preamble.
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
            "max_tokens": 400,
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


def _build_run_url() -> str | None:
    """Build a GitHub Actions run URL from environment variables."""
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
    run_url = _build_run_url()
    if run_url:
        summary += f"\n<{run_url}|Full report>"

    if args.dry_run:
        print(summary)
        return

    log.info("Posting to Slack...")
    post_to_slack(webhook_url, summary)
    log.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
