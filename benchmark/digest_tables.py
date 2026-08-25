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
"""Build the benchmark Slack digest as computed tables.

Every cell is computed from a scorer artifact -- no LLM anywhere -- so this
module is unit-testable.

Inputs, all written by ``benchmark.scorer`` into the same results directory:

===========================================  ==============================
file                                          window
===========================================  ==============================
``trailing_scores_<platform>.json``           trailing 90d (nightly --period-days 90)
``rolling_scores_<platform>.json``            W-1, the last 7 days
``prev_rolling_scores_<platform>.json``       W-2, the 7 days before that
``scores_tournament_<platform>.json``         tournament, all-time only
===========================================  ==============================

``base`` (no-skill floor) and ``mkt`` (market Brier) are shown in Brier units
beside the tool's Brier; a skill-score ratio is deliberately NOT rendered --
its denominator sits off-screen.

One table per Slack message: stays under Slack's per-message cell-character
cap and lets a reader thread a reply to one table.

Pure builder, stdlib only: no network, no LLM, no clock. A broken digest must
never break the daily post, so the entry point returns an empty list instead
of raising when its inputs are missing or unparseable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Collection, Iterable, Sequence

from benchmark.roi_sim import RELIABILITY_GATE
from benchmark.scoring_primitives import MIN_SAMPLE_SIZE
from benchmark.slack_blocks import (
    Col,
    context,
    header,
    message,
    section,
    table_block,
)
from benchmark.slack_tables import display_width

log = logging.getLogger(__name__)

NA = "n/a"

# ASCII on purpose: table cells stay ASCII so column padding cannot skew.
LOW_SAMPLE_MARK = "*"

# A delta renders only when BOTH windows clear the sample floor.

# Deployment names, so the title matches what the team calls each platform.
PLATFORM_TITLES = {"polymarket": "Polystrat", "omen": "Omenstrat"}

WINDOW_FILES = {
    "at": "trailing_scores_{platform}.json",
    "w1": "rolling_scores_{platform}.json",
    "w2": "prev_rolling_scores_{platform}.json",
    "tournament": "scores_tournament_{platform}.json",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_by_tool(path: Path) -> dict[str, dict[str, Any]]:
    """Read one scores file and return its ``by_tool`` mapping.

    :param path: path to a scores json file.
    :return: mapping of tool name to its stats dict; empty when unavailable.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.info("digest: no usable scores at %s (%s)", path, exc)
        return {}
    by_tool = payload.get("by_tool")
    if not isinstance(by_tool, dict):
        log.warning("digest: %s has no by_tool mapping", path)
        return {}
    return {str(k): v for k, v in by_tool.items() if isinstance(v, dict)}


def _as_of_date(path: Path) -> str | None:
    """Last date the scores file covers: window_end, else generated_at.

    The batch scorer stamps both bounds; ``window_end`` is the last
    observed prediction, so it can trail the run date when predictions pause.

    :param path: scores json path.
    :return: as-of date (YYYY-MM-DD) or None.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stamp = payload.get("window_end") or payload.get("generated_at")
    return str(stamp)[:10] if stamp else None


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------


def _num(value: object) -> float | None:
    """Return *value* as a float when it is a real number (bools excluded).

    :param value: candidate value.
    :return: the value as a float, or None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _score(value: object, signed: bool = False) -> str:
    """Format a Brier-unit score to 4 decimals.

    :param value: candidate value.
    :param signed: render an explicit sign (Edge columns).
    :return: formatted score, or ``n/a``.
    """
    if _num(value) is None:
        return NA
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _count(stats: dict[str, Any] | None) -> str:
    """Format a window's scored-row count, marking under-floor samples.

    Renders ``valid_n/edge_n`` when the pools differ: Brier averages over
    valid rows, Edge/mkt over the priced subset, and the low-sample marker
    keys off the smaller pool.

    :param stats: per-tool stats dict for that window, or None.
    :return: count cell, e.g. ``301``, ``40/11`` or ``19 *``, or ``n/a``.
    """
    valid = _num((stats or {}).get("valid_n"))
    if valid is None:
        return NA
    valid_n = int(valid)
    edge = _num((stats or {}).get("edge_n"))
    edge_n = None if edge is None else int(edge)
    both = (
        f"{valid_n}/{edge_n}"
        if edge_n is not None and edge_n != valid_n
        else str(valid_n)
    )
    floor = min(valid_n, edge_n) if edge_n is not None else valid_n
    mark = f" {LOW_SAMPLE_MARK}" if floor < MIN_SAMPLE_SIZE else ""
    return f"{both}{mark}"


def _reliability(stats: dict[str, Any] | None) -> str:
    """Format a window's reliability (``valid_n / n``) as a percentage.

    Qualifies the Brier beside it: the score is averaged over valid rows only.

    :param stats: per-tool stats dict for that window, or None.
    :return: percentage cell, e.g. ``97%``, or ``n/a``.
    """
    if not stats or _num(stats.get("reliability")) is None:
        return NA
    return f"{stats['reliability'] * 100:.0f}%"


def _has_floor(stats: dict[str, Any] | None, field: str = "valid_n") -> bool:
    """Check whether a window has enough rows to support a delta.

    Edge deltas floor on ``edge_n``, Brier deltas on ``valid_n`` -- the two
    average over different pools.

    :param stats: per-tool stats dict for that window, or None.
    :param field: the count to gate on.
    :return: True when the window cleared MIN_SAMPLE_SIZE.
    """
    count = _num((stats or {}).get(field))
    return count is not None and int(count) >= MIN_SAMPLE_SIZE


def _delta(
    current: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    field: str,
    lower_is_better: bool,
) -> str:
    """Format the current-vs-reference movement of one metric.

    The direction word is spelled out: the sign alone is ambiguous across
    columns (a positive Brier delta is a regression, a positive Edge delta
    an improvement).

    :param current: stats for the current window (W-1).
    :param reference: stats for the comparison window (W-2 or all-time).
    :param field: metric key present in both stats dicts.
    :param lower_is_better: True for Brier, False for Edge.
    :return: e.g. ``+0.0417 worse``, ``+0.0417 worse *`` (low sample), or ``n/a``.
    """
    if not current or not reference:
        return NA
    a = _num(current.get(field))
    b = _num(reference.get(field))
    if a is None or b is None:
        return NA
    # Edge and mkt live on the edge-eligible pool; everything else on valid_n.
    count_field = "edge_n" if field in ("edge", "market_brier") else "valid_n"
    diff = a - b
    improved = diff < 0 if lower_is_better else diff > 0
    cell = f"{diff:+.4f} {'better' if improved else 'worse'}"
    if not (_has_floor(current, count_field) and _has_floor(reference, count_field)):
        # Same cell, starred: under the sample floor the arithmetic is real
        # but not statistically significant -- the star the counts carry.
        return f"{cell} {LOW_SAMPLE_MARK}"
    return cell


COMPLETENESS_RATIO = 0.70


# The rule lives inside the header's own text so Slack cannot pad between
# rule and title; its length tracks the title rather than being fixed.
TITLE_RULE_CHAR = "━"
TITLE_RULE_MIN = 24


def _headline(platform: str, as_of: str | None = None) -> dict[str, Any]:
    """Build the title message that opens a platform's report.

    :param platform: platform key, for the heading.
    :param as_of: the last date the data covers -- taken from the scores file,
        not a clock; this module has no clock.
    :return: a webhook payload.
    """
    label = PLATFORM_TITLES.get(platform, platform.title())
    # Date inside the rules, on the title line: platform and day are one block.
    stamp = f"  \u00b7  {as_of}" if as_of else ""
    line = f"{label.upper()}  \u00b7  REPORT V2{stamp}"
    rule = TITLE_RULE_CHAR * max(TITLE_RULE_MIN, display_width(line))
    return message(
        f"{label} {as_of or ''}".strip(),
        [
            header(f"{rule}\n{line}\n{rule}"),
            context(
                "*Legend:*\n"
                f"\u2022 {LOW_SAMPLE_MARK}: marks a window under "
                f"n = {MIN_SAMPLE_SIZE} -- the number is reported but not "
                "statistically significant.\n"
                "\u2022 *n*: number of scored predictions, not distinct "
                "markets -- a market asked repeatedly counts each time.\n"
                "\u2022 *base*: score of always predicting the pool's "
                "average YES rate -- the zero-skill reference.\n"
                "\u2022 *mkt*: score of always predicting the current "
                "market price -- the crowd reference."
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------


def _cols(names: Iterable[str]) -> tuple[Col, ...]:
    """Column specs: ``tool`` left-aligned, every other column right-aligned.

    :param names: column header names.
    :return: Col spec tuple for ``table_block``.
    """
    return tuple(Col(n) if n == "tool" else Col(n, align="right") for n in names)


_W2_COLUMNS = _cols(
    "#;tool;n W-2;n W-1;rel W-1;Brier W-2;Brier W-1;Δ Brier;base W-2;"
    "base W-1;mkt W-2;mkt W-1;Edge W-2;Edge W-1;Δ Edge".split(";")
)

_AT_COLUMNS = _cols(
    "#;tool;n 90d;n W-1;rel W-1;Brier 90d;Brier W-1;Δ Brier;base 90d;"
    "mkt 90d;mkt W-1;Edge 90d;Edge W-1;Δ Edge".split(";")
)

# Tournament rows come from the all-time tournament accumulator, not the
# trailing window, so its headers keep the cum label.
_TOURN_COLUMNS = _cols(
    "#;tool;n cum;n W-1;rel W-1;Brier cum;Brier W-1;Δ Brier;base cum;"
    "mkt cum;mkt W-1;Edge cum;Edge W-1;Δ Edge".split(";")
)


def _sort_key(
    entry: tuple[str, dict[str, Any] | None, dict[str, Any] | None],
) -> tuple[int, float, str]:
    """Order rows best-first on Edge; scored before unscored; name breaks ties.

    Edge, not an uncertainty-penalized bound: a bound would rank how well
    MEASURED a tool is, not how good.

    :param entry: (tool name, current stats, reference stats).
    :return: sort key.
    """
    name, current, reference = entry
    for stats in (current, reference):
        edge = _num((stats or {}).get("edge"))
        if edge is not None:
            return (0, -edge, name)
    # Belt-and-braces with _ordered()'s own sort against nondeterministic set
    # iteration order. Removing EITHER is safe; removing BOTH reintroduces the
    # bug, which is what TestOrderStability asserts.
    return (1, 0.0, name)


def _rows_w2(
    tools: Sequence[str],
    w1: dict[str, dict[str, Any]],
    w2: dict[str, dict[str, Any]],
) -> list[tuple[str, ...]]:
    """Build the W-2 vs W-1 table rows.

    :param tools: tool names to render, in the order given.
    :param w1: by_tool stats for the last 7 days.
    :param w2: by_tool stats for the preceding 7 days.
    :return: one cell tuple per tool.
    """
    ranked = [t for t in tools if _num((w1.get(t) or {}).get("edge")) is not None]

    rows: list[tuple[str, ...]] = []
    for tool in tools:
        a, b = w1.get(tool), w2.get(tool)
        rows.append(
            (
                str(ranked.index(tool) + 1) if tool in ranked else "-",
                tool,
                _count(b),
                _count(a),
                _reliability(a),
                _score((b or {}).get("brier")),
                _score((a or {}).get("brier")),
                _delta(a, b, "brier", lower_is_better=True),
                _score((b or {}).get("baseline_brier")),
                _score((a or {}).get("baseline_brier")),
                _score((b or {}).get("market_brier")),
                _score((a or {}).get("market_brier")),
                _score((b or {}).get("edge"), signed=True),
                _score((a or {}).get("edge"), signed=True),
                _delta(a, b, "edge", lower_is_better=False),
            )
        )
    return rows


def _rows_at(
    tools: Sequence[str],
    w1: dict[str, dict[str, Any]],
    at: dict[str, dict[str, Any]],
) -> list[tuple[str, ...]]:
    """Build the reference-window vs W-1 table rows.

    :param tools: tool names to render, in the order given.
    :param w1: by_tool stats for the last 7 days (empty for tournament).
    :param at: by_tool stats for the reference window -- trailing 90d
        for production, the all-time pool for the tournament.
    :return: one cell tuple per tool.
    """
    ranked = [t for t in tools if _num((at.get(t) or {}).get("edge")) is not None]

    rows: list[tuple[str, ...]] = []
    for tool in tools:
        a, b = w1.get(tool), at.get(tool)
        rows.append(
            (
                str(ranked.index(tool) + 1) if tool in ranked else "-",
                tool,
                _count(b),
                _count(a),
                _reliability(a),
                _score((b or {}).get("brier")),
                _score((a or {}).get("brier")),
                _delta(a, b, "brier", lower_is_better=True),
                _score((b or {}).get("baseline_brier")),
                _score((b or {}).get("market_brier")),
                _score((a or {}).get("market_brier")),
                _score((b or {}).get("edge"), signed=True),
                _score((a or {}).get("edge"), signed=True),
                _delta(a, b, "edge", lower_is_better=False),
            )
        )
    return rows


def _scored(by_tool: dict[str, dict[str, Any]]) -> set[str]:
    """Return tools that produced a Brier in this window.

    :param by_tool: by_tool mapping from a scores file.
    :return: set of tool names with a numeric Brier.
    """
    return {t for t, s in by_tool.items() if _num(s.get("brier")) is not None}


def _ran_but_unscored(
    by_tool: dict[str, dict[str, Any]], permitted: Collection[str]
) -> set[str]:
    """Return PREDICTION tools that ran but produced no usable prediction.

    The allowlist keeps this from alarming on non-forecaster tools, whose
    ``valid_n=0`` is a correct reading, not a failure.

    :param by_tool: by_tool mapping from a scores file.
    :param permitted: prediction-tool allowlist.
    :return: names of permitted tools that ran without scoring.
    """
    return {
        tool
        for tool, stats in by_tool.items()
        if tool in permitted
        and _num(stats.get("brier")) is None
        and _num(stats.get("n")) is not None
        and int(stats["n"]) > 0
    }


def _ordered(
    names: Iterable[str],
    current: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
) -> list[str]:
    """Order tool names by the decision metric.

    :param names: tool names to order.
    :param current: stats for the current window.
    :param reference: stats for the reference window.
    :return: tool names, best Edge first.
    """
    entries = [(n, current.get(n), reference.get(n)) for n in sorted(names)]
    return [name for name, _, _ in sorted(entries, key=_sort_key)]


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

_ALERT_COLUMNS = (
    Col("what"),
    Col("tool"),
    # Free-form prose, so Slack is allowed to wrap these two.
    Col("evidence", wrap=True),
    Col("action", wrap=True),
)


def _alert_rows(
    w1: dict[str, dict[str, Any]],
    at: dict[str, dict[str, Any]],
    tools: Sequence[str],
    at_w2: dict[str, dict[str, Any]] | None = None,
) -> list[tuple[str, ...]]:
    """Build the alerts table: conditions that BLOCK a call, not calls.

    Reliability is deliberately NOT alerted on a bare week-over-week delta:
    at these sample sizes a two-point drop is one or two extra failed rows.

    :param w1: by_tool stats for the last 7 days.
    :param at: by_tool stats for the trailing 90d window.
    :param tools: tool names under consideration.
    :param at_w2: the prior week, used to detect a still-filling W-1.
    :return: alert rows, empty when nothing fires.
    """
    at_w2 = at_w2 or {}
    rows: list[tuple[str, ...]] = []

    starved = [t for t in tools if w1.get(t) and not _has_floor(w1[t])]
    for tool in starved:
        # _has_floor is False when valid_n is MISSING as well as small, so
        # this cannot KeyError on the field it just failed to find.
        starved_n = _num(w1[tool].get("valid_n"))
        shown = NA if starved_n is None else int(starved_n)
        rows.append(
            (
                "sample starved",
                tool,
                f"{shown} scored rows in W-1, floor is {MIN_SAMPLE_SIZE}",
                "widen the window before any keep/demote call",
            )
        )

    for tool in tools:
        stats = w1.get(tool)
        if not stats or _num(stats.get("reliability")) is None:
            continue
        if stats["reliability"] < RELIABILITY_GATE:
            rows.append(
                (
                    "reliability breach",
                    tool,
                    f"{_reliability(stats)} in W-1, gate is " f"{RELIABILITY_GATE:.0%}",
                    "tool is excluded from ROI; investigate parse failures",
                )
            )

    # Two DIFFERENT conditions against two different baselines: Edge is
    # measured against the MARKET, and a tool can lose to the market while
    # beating its own no-skill floor. Floored, like every other claim here.
    below_market = [
        at[t]
        for t in tools
        if at.get(t)
        and _num(at[t].get("edge")) is not None
        and _has_floor(at[t], "edge_n")
    ]
    if len(below_market) >= 2 and all(s["edge"] < 0 for s in below_market):
        best = max(s["edge"] for s in below_market)
        rows.append(
            (
                "platform below market",
                f"ALL ({len(below_market)} tools over the floor)",
                f"every 90d Edge < 0; best is {best:+.4f}",
                "upstream calibration, not a tool swap. Check the "
                "recorded price band too: a trader-side bet filter "
                "can confine which markets this platform ever "
                "records, which changes what Edge is measured over",
            )
        )

    # Resolution censoring: the one alert that fires on an ABSENCE, so it is
    # computed here rather than inferred from any single row.
    w1_rows = sum(int(_num(w1[t].get("valid_n")) or 0) for t in tools if w1.get(t))
    w2_rows = sum(
        int(_num(at_w2[t].get("valid_n")) or 0) for t in tools if at_w2.get(t)
    )
    if w2_rows and w1_rows / w2_rows < COMPLETENESS_RATIO:
        rows.append(
            (
                "week still filling",
                f"ALL ({len(tools)} tools)",
                f"W-1 holds {w1_rows} scored rows against W-2's {w2_rows} "
                f"({w1_rows / w2_rows:.0%}); a row appears only once its market "
                "resolves, so the newest week is not yet comparable",
                "read the 90d column, not the weekly deltas",
            )
        )

    scored = [
        t
        for t in tools
        if at.get(t) and _num(at[t].get("brier")) is not None and _has_floor(at[t])
    ]
    below_floor = [
        t
        for t in scored
        if _num(at[t].get("baseline_brier")) is not None
        and at[t]["brier"] > at[t]["baseline_brier"]
    ]
    if len(scored) >= 2 and below_floor and len(below_floor) == len(scored):
        rows.append(
            (
                "platform below no-skill",
                f"ALL ({len(below_floor)} tools over the floor)",
                "every 90d Brier exceeds its own base-rate floor",
                "the fleet is worse than predicting the base rate",
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_digest_messages(
    results_dir: Path,
    platform: str,
    allowed_tools: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the benchmark digest as one webhook payload per table.

    :param results_dir: directory holding the scorer output files.
    :param platform: platform key, e.g. "polymarket" or "omen".
    :param allowed_tools: when given, only these tools are rendered -- callers
        pass the prediction-tool registry. Injected to stay stdlib-only.
    :return: ordered webhook payloads; empty when there is nothing to post.
    """
    windows = {
        key: _load_by_tool(results_dir / name.format(platform=platform))
        for key, name in WINDOW_FILES.items()
    }
    as_of = _as_of_date(results_dir / WINDOW_FILES["at"].format(platform=platform))

    prod_tools = _scored(windows["at"]) | _scored(windows["w1"])
    tourn_tools = _scored(windows["tournament"])
    if allowed_tools is not None:
        permitted = set(allowed_tools)
        prod_tools &= permitted
        tourn_tools &= permitted
        # A permitted tool that ran and scored nothing still belongs here.
        prod_tools |= _ran_but_unscored(windows["at"], permitted)
        prod_tools |= _ran_but_unscored(windows["w1"], permitted)
    if not prod_tools and not tourn_tools:
        log.warning("digest: no scored tools for %s; skipping tables", platform)
        return []

    messages: list[dict[str, Any]] = [_headline(platform, as_of)]

    # Degrade LOUDLY: without this, a failed window-scorer run reads as
    # "no change this week" rather than "this week was not scored".
    missing = [
        label
        for key, label in (("w1", "W-1"), ("w2", "W-2"), ("at", "90d"))
        if not windows[key]
    ]
    if missing:
        warning = (
            f":warning: *{' and '.join(missing)} unavailable* - that "
            "window's scorer step produced no scores for this platform, so those "
            "columns read `n/a` below. They are NOT unchanged; they are "
            "unmeasured."
        )
        messages.append(
            message(f"{' and '.join(missing)} unavailable", [section(warning)])
        )

    # 1a compares the two weekly windows, so a tool absent from BOTH has
    # nothing to say there -- with a 90d reference window the roster includes
    # tools retired months ago, and rendering them in 1a is a row of pure n/a.
    # They still appear in 1b, where their 90d numbers are real.
    weekly_tools = prod_tools & (_scored(windows["w1"]) | _scored(windows["w2"]))
    if prod_tools:
        # The 90d reference FIRST here, weekly first in 1a: _sort_key ranks on
        # the first window argument that has an edge, so the argument order IS
        # the ranking metric the caption names.
        order_at = _ordered(prod_tools, windows["at"], windows["w1"])
        if weekly_tools:
            order_w2 = _ordered(weekly_tools, windows["w1"], windows["w2"])
            messages.append(
                message(
                    "1a. Production - W-2 vs W-1",
                    [
                        section("*1a. PRODUCTION W-2 vs W-1 - ranked by `Edge W-1`*"),
                        table_block(
                            _W2_COLUMNS,
                            _rows_w2(order_w2, windows["w1"], windows["w2"]),
                        ),
                    ],
                )
            )
        messages.append(
            message(
                "1b. Production - 90D vs W-1",
                [
                    section(
                        "*1b. PRODUCTION 90D vs W-1 - ranked by `Edge 90d`*\n"
                        "_90d: trailing 90-day window, recomputed nightly._"
                    ),
                    table_block(
                        _AT_COLUMNS,
                        _rows_at(order_at, windows["w1"], windows["at"]),
                    ),
                ],
            )
        )

    if tourn_tools:
        order = _ordered(tourn_tools, {}, windows["tournament"])
        messages.append(
            message(
                "2. Tournament - all-time pool",
                [
                    section(
                        "*2. TOURNAMENT - ranked by `Edge cum`*\n"
                        "_cum: the candidate's whole scored pool._"
                    ),
                    table_block(
                        _TOURN_COLUMNS,
                        _rows_at(order, {}, windows["tournament"]),
                    ),
                    context(
                        "*Caveats:*\n"
                        "\u2022 Candidates have no weekly columns: the "
                        "rolling scorer skips tournament output, so only "
                        "their all-time pool is scored.\n"
                        "\u2022 Candidate rows are NOT comparable to "
                        "production rows: candidates answer a different, "
                        "much easier market pool (compare the *mkt* "
                        "columns), over a different span."
                    ),
                ],
            )
        )

    alerts = _alert_rows(
        windows["w1"], windows["at"], sorted(prod_tools), windows["w2"]
    )
    if alerts:
        messages.append(
            message(
                f"3. Alerts ({len(alerts)})",
                [
                    section("*3. ALERTS*  _conditions that block a call_"),
                    table_block(_ALERT_COLUMNS, alerts),
                ],
            )
        )

    return messages
