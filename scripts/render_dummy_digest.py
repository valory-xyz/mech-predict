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
"""Render the digest exactly as ``notify_slack`` would post it, to a file.

Same call path as the daily job: :func:`benchmark.digest_tables.
build_digest_messages` for the tables and :func:`benchmark.roi_slack.
build_roi_section` for the ROI companion, one Slack message per block. The only
thing not reproduced is the LLM prose block, which costs money and is the part
being replaced; its contents are inventoried instead.

Also emits a coverage table: every ``## `` section of the live markdown report,
and whether the computed digest carries it.

Usage::

    python -m scripts.render_dummy_digest --results DIR --reports DIR --out FILE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.digest_tables import build_digest_messages
from benchmark.roi_slack import build_roi_section
from benchmark.tools import TOOL_REGISTRY

PLATFORMS = (("polymarket", "Polystrat"), ("omen", "Omenstrat"))

# Section of the live report -> how the computed digest covers it.
# "covered" / "partial" / "missing"; the note is the reason.
COVERAGE: dict[str, tuple[str, str]] = {
    "Metric References": (
        "missing",
        "no legend in the posted messages: `*`, `insufficient`, `base`, `mkt`, "
        "`Edge` are undefined for the reader",
    ),
    "Platform Snapshot": (
        "missing",
        "the headline. Platform Brier + n + reliability. Computable from the "
        "`overall` block of each window file",
    ),
    "Platform Historical Comparison": (
        "missing",
        "platform Brier delta vs AT and vs W-2 -- the direction-of-travel line. "
        "Computable from `overall` across the three window files",
    ),
    "Tool Historical Comparison": (
        "covered",
        "tables 1a/1b, and strictly richer: adds mkt, Edge, base, rel, ROI",
    ),
    "Tool x Category": (
        "missing",
        "DA lift answers a different question (beat always-majority?) than "
        "Brier/Edge. Computable from `by_tool_category`",
    ),
    "Tool x Category Diagnostics": ("missing", "computable from `by_tool_category`"),
    "Tool x Category Historical Comparison": (
        "missing",
        "single largest-movement bullet; low value",
    ),
    "Diagnostics Historical Comparison": (
        "partial",
        "Edge IS covered and improved (windowed, with mkt beside it). Log Loss, "
        "Conditional Accuracy, Disagreement Brier and Directional Bias are not. "
        "All present in `by_tool`",
    ),
    "Reliability & Parse Quality": (
        "partial",
        "`rel W-1` column + gate-breach alert. No rel AT/W-2, so no regression "
        "is detectable; no Valid % counterpart",
    ),
    "Tool Deployment Status": (
        "missing",
        "the only gap NOT computable from the scorer artifacts -- "
        "`tool_usage.fetch_valid_tools()` resolves it over the network",
    ),
    "Tool x Version x Mode": (
        "missing",
        "per-version rows. Computable from `by_tool_version_mode`",
    ),
    "Version Deltas": (
        "missing",
        "how a bad rollout gets caught. Computable from `by_tool_version_mode`",
    ),
    "Trend (Fleet-wide, Monthly)": (
        "missing",
        "fleet-wide, not platform-scoped; arguably belongs only in the report",
    ),
    "Sample Size Warnings": (
        "partial",
        "per-tool low samples carry `*` and raise an alert; the category-level "
        "gate statement is absent",
    ),
    "Tournament Callouts": (
        "partial",
        "BIGGEST GAP. Table 2 has the tool, n, Brier, base, mkt, Edge and ROI, "
        "but NOT the Version label, the `vs Production` comparison, the "
        "promote/watch verdict, or active-CID scoping -- so it renders dead "
        "candidates the report drops",
    ),
}

MARK = {"covered": "OK      ", "partial": "PARTIAL ", "missing": "MISSING "}


def _report_sections(path: Path) -> list[str]:
    """List the ``## `` headings of a rendered markdown report.

    :param path: report_<platform>.md path.
    :return: heading texts without the leading hashes.
    """
    if not path.exists():
        return []
    return [
        line[3:].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    ]


def _coverage_for(heading: str) -> tuple[str, str]:
    """Look up a heading's coverage, matching on prefix.

    Report headings carry parenthetical suffixes ("(Current 7d)") that the
    coverage keys omit.

    :param heading: report heading text.
    :return: (status, note); ("missing", "") when unmapped.
    """
    normalized = heading.replace("×", "x")
    for key, value in COVERAGE.items():
        if normalized.startswith(key):
            return value
    return ("missing", "unmapped -- review")


def _render_platform(
    results: Path, reports: Path, platform: str, label: str
) -> list[str]:
    """Render one platform's full message sequence plus its coverage table.

    :param results: scorer results directory.
    :param reports: directory holding report_<platform>.md.
    :param platform: platform key.
    :param label: deployment label used in the digest heading.
    :return: output lines.
    """
    out: list[str] = [f"\n\n{'=' * 78}\n{label.upper()}  ({platform})\n{'=' * 78}"]

    report_path = reports / f"report_{platform}.md"
    out.append(
        "\n--- MESSAGE 1 -- heading + LLM prose (UNCHANGED by the redesign) ---\n"
    )
    heading = "*Benchmark Report*"
    if report_path.exists():
        first = report_path.read_text(encoding="utf-8").split("\n", 1)[0]
        if first.startswith("# "):
            heading = f"*{first.lstrip('# ').strip()}*"
    out.append(heading)
    out.append(
        "\n[LLM-generated block not reproduced -- it costs an API call and is\n"
        " the part being replaced. Today it emits: Summary, Tool performance\n"
        " (or Top/Worst), Deployment status, Tool x Category, Tool versions,\n"
        " Tournament callouts, Diagnostics, Reliability, Recommended actions.]\n"
        "<...|Full report>"
    )

    messages = build_digest_messages(results, platform, allowed_tools=TOOL_REGISTRY)
    if not messages:
        out.append("\n--- NO COMPUTED TABLES (no scored tools for this platform) ---")
    for index, message in enumerate(messages, start=2):
        out.append(
            f"\n--- MESSAGE {index} -- computed, {len(message)} chars "
            f"(Slack splits ~3000) ---\n"
        )
        out.append(message)

    roi = build_roi_section(results / "roi_results.json", platform)
    out.append(f"\n--- MESSAGE {len(messages) + 2} -- ROI companion (UNCHANGED) ---\n")
    out.append(roi or "(no ROI section for this platform)")

    out.append(f"\n\n--- COVERAGE vs report_{platform}.md ---\n")
    sections = _report_sections(report_path)
    if not sections:
        out.append("(report not found; coverage table skipped)")
        return out
    tally: dict[str, int] = {"covered": 0, "partial": 0, "missing": 0}
    for heading in sections:
        status, note = _coverage_for(heading)
        tally[status] += 1
        out.append(f"{MARK[status]} {heading}")
        if status != "covered":
            out.append(f"         -> {note}")
    out.append(
        f"\n{tally['covered']} covered / {tally['partial']} partial / "
        f"{tally['missing']} missing  (of {len(sections)} report sections)"
    )
    return out


def main() -> None:
    """Render the dummy digest for every platform to one file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    as_of: Any = "unknown"
    scores = args.results / "scores.json"
    if scores.exists():
        as_of = json.loads(scores.read_text(encoding="utf-8")).get(
            "generated_at", "unknown"
        )

    lines = [
        "DUMMY BENCHMARK DIGEST -- rendered through the same call path as the",
        "daily job (digest_tables.build_digest_messages + roi_slack.",
        "build_roi_section), with BENCHMARK_COMPUTED_TABLES on.",
        "",
        f"scores generated_at : {as_of}",
        f"results dir         : {args.results}",
        "",
        "Each '--- MESSAGE n ---' block is a SEPARATE Slack post: Slack splits",
        "at ~3000 characters and a split breaks the ``` fence, which is why the",
        "tables are not concatenated.",
        "",
        "=" * 78,
        "WHAT IS MISSING -- vs the 9 content blocks the Slack digest posts TODAY",
        "=" * 78,
        "",
        "COVERED (1)",
        "  Tool performance / Top+Worst -> tables 1a/1b, strictly richer",
        "",
        "PARTIAL (4)",
        "  Tournament callouts -> BIGGEST GAP. No Version label, no `vs",
        "      Production`, no promote/watch verdict, no active-CID scoping.",
        "      The verdict IS the point of that section.",
        "  Diagnostics         -> Edge covered and improved; Log Loss,",
        "      Conditional Accuracy, Disagreement Brier, Directional Bias absent.",
        "  Reliability         -> `rel W-1` column + gate alert; no rel AT/W-2",
        "      so no regression is detectable, and no Valid % counterpart.",
        "  Recommended actions -> replaced by the computed Alerts table; the",
        "      per-tool keep/watch/demote `rec` column is deferred (needs the",
        "      market-clustered bootstrap).",
        "",
        "MISSING (4)",
        "  Summary           -> the headline: platform Brier + direction of",
        "      travel. Computable from the `overall` block of each window file.",
        "  Version Deltas    -> how a bad rollout gets caught. Computable from",
        "      `by_tool_version_mode`.",
        "  Tool x Category   -> DA lift asks 'beat always-majority?', a",
        "      different question. Computable from `by_tool_category`.",
        "  Deployment status -> the ONLY gap not computable from the scorer",
        "      artifacts; resolved over the network by fetch_valid_tools().",
        "",
        "ALSO MISSING (labelling, not content)",
        "  * no legend: `*`, `insufficient`, `base`, `mkt`, `Edge` undefined",
        "  * `AT` is hard-coded but the label becomes `MTD` under the",
        "    mech-analytics flag",
        "  * tables carry no platform label and no date (message 1 only)",
        "  * no active-deployment filter: a demoted tool would persist forever,",
        "    indistinguishable from a live one",
        "",
        "PRE-EXISTING DEFECT surfaced by this exercise",
        "  `OUR_TOOLS = set(TOOL_REGISTRY)` in notify_slack, but TOOL_REGISTRY",
        "  is the BENCHMARK-RUNNABLE registry, not an ownership list.",
        "  `propose-question` is ours (packages/valory/customs/propose_question)",
        "  and is deployed on Omen with n=124, valid_n=0, reliability 0% -- the",
        "  single most alarming row in the data -- yet it is classified",
        "  third-party and suppressed from Slack TODAY. The computed tables",
        "  inherit this (parity, not a regression). Ownership should come from",
        "  the package author, not the registry.",
    ]
    for platform, label in PLATFORMS:
        lines.extend(_render_platform(args.results, args.reports, platform, label))

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
