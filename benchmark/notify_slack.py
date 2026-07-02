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
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from benchmark.analyze import (
    PLATFORM_LABELS,
    ROLLING_WINDOW_DAYS,
)
from benchmark.scorer import MIN_SAMPLE_SIZE
from benchmark.tools import TOOL_REGISTRY

log = logging.getLogger(__name__)

_PROMPT_HEADER = f"""\
Summarize this Olas Predict benchmark report for the *{{platform_label}}* deployment using EXACTLY the structure below. The output posts to a Slack channel as internal team telemetry — keep it technical, decision-first, and scannable on roughly one screen. The report carries three windows per metric: `Current {ROLLING_WINDOW_DAYS}d` (trailing {ROLLING_WINDOW_DAYS}-day aggregate), `All-Time` (cumulative), and `Prev {ROLLING_WINDOW_DAYS}d` (the immediately preceding non-overlapping {ROLLING_WINDOW_DAYS}-day window). Never mix windows in one claim; pair every cited number with its window label. Do NOT compare platforms or cite tools, deployments, or rows belonging to another platform.

*Headline:* one or two lines answering "does {{platform_label}} beat the market?".
- BEGIN the line with the status emoji read from the "Platform Snapshot" `Edge over market` value (Current {ROLLING_WINDOW_DAYS}d): 🔴 when edge < 0 (the platform's tools lose to the market on average), 🟡 when 0 ≤ edge < 0.02 (roughly at the market), 🟢 when edge ≥ 0.02 (beats the market). Then state the verdict in words, the edge value, and the "% of calls beat market" figure from that same line. If the `Edge over market` line reads `N/A`, say platform edge is unavailable this run instead of guessing.
- Then, kept clearly distinct from edge so the two are never conflated: `BSS` (skill vs the base-rate predictor — a dumb always-majority forecaster, NOT the market) and `Brier`, both Current {ROLLING_WINDOW_DAYS}d, each with its week-over-week direction from the "Platform Historical Comparison" `Δ vs Prev {ROLLING_WINDOW_DAYS}d` column (state `insufficient data` verbatim when the delta is suppressed).

*Eligibility for the per-tool edge table below:* a row is eligible only if its Current {ROLLING_WINDOW_DAYS}d n in the "Tool Historical Comparison" table is at least {MIN_SAMPLE_SIZE} AND it carries no ⚠ low sample / ⚠ all malformed flag."""


_PROMPT_BEATS_MARKET_NONE = f"""\
*Beats market? — edge over market (Current {ROLLING_WINDOW_DAYS}d):*
• _no eligible tools — every row in the Tool Historical Comparison table is below n={MIN_SAMPLE_SIZE} or flagged_"""


_PROMPT_BEATS_MARKET = f"""\
First output the literal bold line `*Beats market? — edge over market (Current {ROLLING_WINDOW_DAYS}d):*` on its own, then a GitHub-flavored markdown table (pipe `|` syntax with a `---` separator row). For EVERY eligible tool, read its `Edge` metric row from the "Diagnostics Historical Comparison" section and emit ONE table row, sorted by Current {ROLLING_WINDOW_DAYS}d edge descending (best first). Columns: `tool | edge | WoW | verdict`. `edge` = the Edge row's Current {ROLLING_WINDOW_DAYS}d value; `WoW` = its `Δ vs Prev {ROLLING_WINDOW_DAYS}d` cell copied verbatim (or `insufficient data` / `no prev window`); `verdict` = 🟢 beats when edge >= 0, 🔴 loses when edge < 0. Never invent an edge or a delta. Skip any eligible tool with no `Edge` row. Do NOT wrap cell values in backticks inside the table. Example:
| tool | edge | WoW | verdict |
|---|---|---|---|
| tool-a | +0.0123 | +0.0045 better | 🟢 beats |
| tool-b | -0.0210 | -0.0100 worse | 🔴 loses |"""


_PROMPT_FOOTER = f"""\
Output the literal bold line `*Promote watch — tournament candidates:*` then a markdown table (pipe `|` syntax + `---` separator). REQUIRED whenever the report contains a "## Tournament Callouts" heading — one table row per Callouts data row, never drop, merge, or sample-gate a row (skip the section ONLY when there is no "## Tournament Callouts" heading). Columns: `tool | ver | n | Brier | vs prod`. Count the Callouts data rows (N) and emit EXACTLY N table rows. Order: 🟢 (promote) rows first, then 🔴 (watch) rows, then the rest by Brier ascending. Fill `vs prod` from each row's "vs Production" cell: a 🟢 tag → "🟢 promote vs <prodver> <prodBrier> Δ<delta>"; a 🔴 tag → "🔴 watch vs <prodver> ..."; any other cell ("— no prod baseline" etc.) → copy that note verbatim. `ver` = the version label with any leading "untagged@" stripped (e.g. `untagged@bafybeia` -> bafybeia). When a row's n carries `⚠`, append " ⚠" inside its `n` cell. Do NOT wrap cell values in backticks. The report's `BSS` column is skill vs the base-rate predictor, NOT the market — never present a candidate as "beating the market". After the table you MUST, whenever no row carries a 🟢 tag, output this exact italic line on its own (it is the honest "nothing to promote" signal — do not omit it): _No candidate clears the promote gate vs a deployed sibling yet._ Example:
| tool | ver | n | Brier | vs prod |
|---|---|---|---|---|
| tool_full_search | bafybeih | 665 | 0.1857 | — no prod baseline |
| tool_v2 | bafybeix | 9 ⚠ | 0.3672 | — no prod baseline |

Output the literal bold line `*Regressions:*` then a markdown table — columns `what | Brier (Cur {ROLLING_WINDOW_DAYS}d) | note` — with up to 3 rows, most material first. Draw from: the "Platform Historical Comparison" Brier `Δ vs Prev {ROLLING_WINDOW_DAYS}d` (when worse); any tool whose "Tool Historical Comparison" Brier `Δ vs Prev {ROLLING_WINDOW_DAYS}d` is a signed worse value; and the single worst cell in "Tool × Category (Current {ROLLING_WINDOW_DAYS}d)" (highest Brier among rows above n={MIN_SAMPLE_SIZE}, `note` = "worst category"). Skip this section entirely when nothing regressed. Example:
| what | Brier (Cur {ROLLING_WINDOW_DAYS}d) | note |
|---|---|---|
| tool-a | 0.2487 (n=10266) | WoW +0.0317 worse |
| tool-a x weather | 0.3530 (n=1440) | worst category |

*Action:* 1-2 concrete next steps for {{platform_label}} as plain `*bold*`-free bullets (not a table), each anchored to a specific row above (e.g. investigate a negative-edge tool, hold the roster when no candidate clears the gate, chase the worst category). A tournament candidate with "— no prod baseline" is NOT promotable on Brier alone — it has no deployed sibling to beat — so never recommend promoting one; at most flag it as one to keep accumulating. No filler, no restating the headline.

Rules:
- Output layout: *Headline:* and *Action:* are Slack mrkdwn prose (use `code` backticks, single-asterisk *bold*, no **double asterisks**). The *Beats market?*, *Promote watch*, and *Regressions* sections are each a bold header line followed by a GitHub-flavored markdown table (pipe syntax with a `---` separator) — the post-processor reformats those tables into aligned monospace for Slack, so just emit clean pipe tables and do not pre-align or wrap cells in backticks. Use 🟢 / 🟡 / 🔴 for status — never ✅.
- Edge over market = paired tool-vs-market Brier on the same rows: positive = the tool beats the price you bet against. This is the money-relevant "beats market" signal. BSS = skill vs the base-rate (always-majority) predictor, NOT the market — keep the two strictly distinct and never call BSS a market comparison.
- Never mix windows in one claim; pair every number with its window label. A delta never stands alone — cite its n, or write `insufficient data` / `no prev window` verbatim from the table.
- Do not draw claims from cells flagged ⚠ low sample or all malformed. Tool names with hyphens vs underscores are DIFFERENT tools — use exact names.
- No greetings, no preamble, no trailing summary. Never invent a metric, tool, or version label.
- Some tools in the report are third-party (not ours). Exclude them entirely — never mention, rank, compare, or recommend actions for them."""


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


def _build_system_prompt(platform_label: str, eligible_count: int) -> str:
    """Assemble the edge-led digest system prompt.

    The per-tool "Beats market?" block dispatches on ``eligible_count``
    so the LLM only ever sees one convention per request:

    - ``0`` eligible tools → a single placeholder bullet.
    - ``1+`` eligible → list every eligible tool's edge, best first.

    Each platform deploys only a handful of production tools, so the
    eligible set is small and a single edge-sorted list reads cleanly —
    no Top/Worst split is needed.

    :param platform_label: deployment name to thread into the prompt.
        Must be one of ``benchmark.analyze.PLATFORM_LABELS`` values.
    :param eligible_count: number of tools that clear the eligibility
        floor in the markdown report's Tool Historical Comparison table.
        Drives the beats-market-block dispatch.
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

    beats_block = (
        _PROMPT_BEATS_MARKET_NONE if eligible_count == 0 else _PROMPT_BEATS_MARKET
    )
    template = "\n\n".join([_PROMPT_HEADER, beats_block, _PROMPT_FOOTER])
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


def _is_table_row(line: str) -> bool:
    """True when ``line`` looks like a markdown table row (``| a | b |``)."""
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 1


def _is_separator_row(line: str) -> bool:
    """True when ``line`` is a markdown header/body separator (``|---|---|``)."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(c and set(c) <= set("-: ") and "-" in c for c in cells)


def _split_row(line: str) -> list[str]:
    """Split a ``| a | b |`` row into stripped cell strings."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _tables_to_monospace(text: str) -> str:
    """Convert markdown pipe tables in ``text`` to aligned monospace blocks.

    Slack does not render markdown tables — it shows the raw ``|`` pipes.
    The LLM is reliable at emitting clean markdown tables but not at hand-
    aligning monospace columns (long tool names skew the layout), so the
    prompt asks for pipe tables and this pass turns each one into a fenced
    code block with space-padded columns that line up in Slack's monospace
    font. Non-table lines (the bold headers, the headline, the action
    bullets) pass through untouched.

    Alignment uses character counts; status emojis (🟢/🔴) sit in the last
    column of every table, so their double-width rendering only affects
    trailing space and never pushes a later column out of line.

    :param text: the LLM summary, possibly containing markdown tables.
    :return: the same text with every markdown table replaced by an
        aligned, triple-backtick-fenced monospace table.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not _is_table_row(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        # Gather the contiguous run of table rows.
        block: list[str] = []
        while i < len(lines) and _is_table_row(lines[i]):
            block.append(lines[i])
            i += 1
        rows = [_split_row(b) for b in block if not _is_separator_row(b)]
        if not rows:
            continue
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        widths = [max(len(r[c]) for r in rows) for c in range(ncol)]
        out.append("```")
        for r in rows:
            out.append("  ".join(r[c].ljust(widths[c]) for c in range(ncol)).rstrip())
        out.append("```")
    return "\n".join(out)


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
    with urlopen(
        req, timeout=120
    ) as resp:  # nosec B310 — fixed https URL, not user-controlled
        body = json.loads(resp.read())

    return _tables_to_monospace(body["choices"][0]["message"]["content"].strip())


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
    """POST a message to a Slack incoming webhook.

    On rejection Slack returns the reason as a short plaintext body
    (e.g. ``invalid_payload``, ``no_text``, ``too_many_attachments``).
    ``urlopen`` raises :class:`HTTPError` before that body is read, so we
    catch it, surface the reason, and re-raise — otherwise the failure is
    an opaque ``HTTP Error 400: Bad Request`` with no actionable detail.

    :param webhook_url: Slack incoming-webhook URL (from a secret).
    :param summary: message text to post.
    :raises RuntimeError: if Slack rejects the payload, with its reason.
    """
    payload = json.dumps({"text": summary}).encode()
    req = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(
            req, timeout=15
        ) as resp:  # nosec B310 — webhook URL from secret, not user input
            resp.read()
    except HTTPError as exc:
        reason = exc.read().decode("utf-8", errors="replace").strip()
        log.error(
            "Slack rejected the message (HTTP %s): %s",
            exc.code,
            reason or "<empty body>",
        )
        raise RuntimeError(
            f"Slack webhook rejected the payload (HTTP {exc.code}): "
            f"{reason or 'no reason given'}"
        ) from exc


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
