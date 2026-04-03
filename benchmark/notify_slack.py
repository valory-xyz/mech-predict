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

log = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """\
Summarize this benchmark report for Slack. Max 5 bullet points.
Include: overall Brier/accuracy, worst tool or category, regressions, reliability issues.
Only flag problems.

Tool names with hyphens vs underscores are DIFFERENT tools — do not conflate them.

Format rules (Slack mrkdwn, NOT markdown):
- Bold: *text* (single asterisk, NOT double)
- Code: `text`
- Each bullet on its own line starting with •
- No **double asterisks**, no _underscores for italic_
- No greetings or preamble"""

MODEL = "gpt-4.1-nano"


def summarize_report(report_text: str, api_key: str) -> str:
    """Call OpenAI to produce a short Slack-formatted summary."""
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": report_text},
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

    log.info("Summarizing report with %s...", MODEL)
    summary = summarize_report(report_text, api_key)

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
