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
"""Render the ROI companion section posted as its own message after the daily benchmark Slack digest.

Pure builder over ``roi_results.json`` (written by ``benchmark.roi_sim``):
no network, no LLM, stdlib only (the model display-name map is imported from
``benchmark.roi_sim``, itself stdlib-only). :func:`build_roi_section` returns a Slack
mrkdwn snippet -- one intro line plus a compact fixed-width table inside a
fenced code block -- or None when there is nothing to post (missing or
unparseable results file, or no groups for the platform). Callers treat
None as "skip the section"; a broken ROI section must never break the daily
post, so this module never raises for bad input data.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from benchmark.roi_sim import MODEL_DISPLAY, _is_number, roi_display_sort_key

log = logging.getLogger(__name__)

# Slack desktop code blocks SCROLL horizontally -- they do NOT wrap -- so a
# wide table renders fine; MAX_LINE_WIDTH is only a truncation backstop for
# pathological tool/flag text. Numeric columns are never capped (see _CAPS),
# so a full mirror of the report table fits comfortably under this bound.
MAX_LINE_WIDTH = 160

# Truncation guard for very wide deployments: keep the top rows by staked
# capital and point at the full report for the rest.
MAX_TABLE_ROWS = 14

# Column order mirrors the full report (render_report in benchmark.roi_sim):
# tool | mode | model | preds | bets | Brier all | Brier bets | staked | ROI |
# w/costs | flags.
_HEADERS = (
    "tool",
    "mode",
    "model",
    "preds",
    "bets",
    "Brier all",
    "Brier bets",
    "staked",
    "ROI (95% CI)",
    "w/costs",
    "flags",
)
# Per-column width caps. Only the TOOL name and the FLAGS text may ever
# ellipsize; every other column -- mode, model, and all numeric cells
# (preds, bets, Brier all, Brier bets, staked, ROI, w/costs) -- is
# content-driven with NO cap (None), so a 6-digit bet count like 109722
# always renders in full.
_TOOL_CAP = 30
_FLAGS_CAP = 24
_CAPS: tuple[int | None, ...] = (
    _TOOL_CAP,  # tool
    None,  # mode
    None,  # model
    None,  # preds
    None,  # bets
    None,  # Brier all
    None,  # Brier bets
    None,  # staked
    None,  # ROI (95% CI)
    None,  # w/costs
    _FLAGS_CAP,  # flags
)

# Abbreviated mode labels keep the width budget for the tool column.
_MODE_SHORT = {"production": "prod", "tournament": "tourn"}

# Compact spellings for the roi_sim flags; the full text stays in the report.
_FLAG_SHORT = {
    "few bets - anecdotal": "few bets",
    "low sample": "low n",
    "no eligible rows in window": "no eligible",
}
_PARSE_RELIABILITY_RE = re.compile(r"(\d+)% parse reliability")
_SEP = " | "
# Backstop only: if even the capped tool/flags widths push a line past
# MAX_LINE_WIDTH (pathological text), shrink the tool column first (down to
# this floor), then flags. Numeric columns are never touched.
_TOOL_MIN = 12
_FLAGS_MIN = 6

_ELLIPSIS = "…"

# _HEADERS, _CAPS, and _row_cells() are parallel per-column tuples that
# _render_table zips together; a length mismatch would silently drop a
# column. Pin the header/cap coupling at import time; _render_table pins the
# row arity at render time.
assert len(_HEADERS) == len(_CAPS), "_HEADERS and _CAPS must stay the same length"


def _as_int(value: object) -> int:
    """Coerce a JSON count field to int, defaulting to 0.

    :param value: candidate value.
    :return: integer value, or 0 when not a number.
    """
    return int(value) if _is_number(value) else 0


def _as_float(value: object) -> float:
    """Coerce a JSON numeric field to float, defaulting to 0.0.

    :param value: candidate value.
    :return: float value, or 0.0 when not a number.
    """
    return float(value) if _is_number(value) else 0.0


def _fmt_signed(value: float) -> str:
    """Format a signed percentage figure compactly.

    One decimal below 100 in magnitude, none above -- keeps worst-case
    cell width bounded for the fixed-width table.

    :param value: percentage value.
    :return: signed display string (no % suffix).
    """
    return f"{value:+.0f}" if abs(value) >= 100 else f"{value:+.1f}"


def _fmt_roi_ci(roi: object, ci: object) -> str:
    """Format the "ROI (95% CI)" table cell.

    :param roi: pooled ROI in percent, or None.
    :param ci: [low, high] CI bounds, or None.
    :return: display string, e.g. ``+12.3% (+4.1,+20.9)``.
    """
    if not _is_number(roi):
        return "n/a"
    roi_text = f"{_fmt_signed(float(roi))}%"
    if (
        isinstance(ci, list)
        and len(ci) == 2
        and _is_number(ci[0])
        and _is_number(ci[1])
    ):
        return f"{roi_text} ({_fmt_signed(float(ci[0]))},{_fmt_signed(float(ci[1]))})"
    return roi_text


def _fmt_brier(group: dict[str, Any]) -> tuple[str, str]:
    """Format the two Brier cells ("Brier all", "Brier bets").

    Mirrors ``render_report``'s two columns: each cell is the 3-decimal
    score, or ``n/a`` when its field is missing/non-numeric.

    :param group: group dict from roi_results.json.
    :return: ``(Brier all, Brier bets)`` display cells, e.g.
        ``("0.218", "0.221")`` or ``("n/a", "n/a")``.
    """
    brier_all = group.get("brier_all")
    brier_bets = group.get("brier_bets")
    left = f"{float(brier_all):.3f}" if _is_number(brier_all) else "n/a"
    right = f"{float(brier_bets):.3f}" if _is_number(brier_bets) else "n/a"
    return left, right


def _fmt_staked(group: dict[str, Any]) -> str:
    """Format the "staked" cell as a plain 2-decimal figure.

    Matches the report's 2-decimal USDC number but drops the " USDC" suffix
    to save table width (the intro already frames the numbers as simulated
    USDC stake). Never truncated -- the column is content-driven.

    :param group: group dict from roi_results.json.
    :return: 2-decimal staked amount, e.g. ``274305.00``.
    """
    return f"{_as_float(group.get('staked')):.2f}"


def _compact_flags(flags: object) -> str:
    """Compress a group's flags list into a short cell.

    Known roi_sim flags map to short spellings (``few bets - anecdotal``
    -> ``few bets``, the parse-reliability warning -> ``⚠ parse NN%``);
    unknown flags keep only their lead phrase (text before an em- or
    hyphen-dash separator).

    :param flags: raw flags value from the group dict.
    :return: comma-joined compact flag text (empty when no flags).
    """
    if not isinstance(flags, list):
        return ""
    parts = []
    for flag in flags:
        text = str(flag)
        reliability = _PARSE_RELIABILITY_RE.search(text)
        if reliability:
            text = f"⚠ parse {reliability.group(1)}%"
        else:
            short = text.split(" — ", maxsplit=1)[0].split(" - ", maxsplit=1)[0]
            text = _FLAG_SHORT.get(text, short).strip()
        if text:
            parts.append(text)
    return ", ".join(parts)


def _fit(text: str, width: int) -> str:
    """Truncate *text* to *width* characters, marking cuts with an ellipsis.

    :param text: cell text.
    :param width: maximum width in characters.
    :return: text of length <= width.
    """
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + _ELLIPSIS


def _format_line(cells: tuple[str, ...], widths: list[int]) -> str:
    """Render one table line: fitted, padded cells joined by the separator.

    :param cells: one string per column.
    :param widths: column widths (same length as cells).
    :return: single table line (trailing whitespace stripped).
    """
    padded = [_fit(cell, width).ljust(width) for cell, width in zip(cells, widths)]
    return _SEP.join(padded).rstrip()


def _render_table(rows: list[tuple[str, ...]]) -> list[str]:
    """Render header + divider + data lines.

    Each column width is content-driven: max(header, widest cell). Numeric
    columns (and mode/model) carry no cap, so their cells never ellipsize.
    The tool and flags columns are capped; only they may be truncated. As a
    backstop, if the resulting line still exceeds MAX_LINE_WIDTH (pathological
    tool/flag text), the tool column shrinks first, then flags -- numeric
    columns are never touched.

    :param rows: pre-formatted cell tuples, one per table row. Each tuple
        MUST have exactly ``len(_HEADERS)`` cells. An empty list returns no
        lines (callers only render the block when there are bet rows).
    :return: table lines (no code-block fences).
    """
    if not rows:
        return []
    for row in rows:
        assert len(row) == len(_HEADERS), (
            f"row has {len(row)} cells, expected {len(_HEADERS)} "
            "(_HEADERS / _CAPS / _row_cells out of sync)"
        )
    widths = []
    for i, (header, cap) in enumerate(zip(_HEADERS, _CAPS)):
        content = max(len(header), *(len(row[i]) for row in rows))
        widths.append(content if cap is None else min(cap, content))
    overflow = sum(widths) + len(_SEP) * (len(_HEADERS) - 1) - MAX_LINE_WIDTH
    if overflow > 0:
        take = min(overflow, widths[0] - _TOOL_MIN)
        widths[0] -= take
        overflow -= take
    if overflow > 0:
        widths[-1] = max(_FLAGS_MIN, widths[-1] - overflow)
    lines = [
        _format_line(_HEADERS, widths),
        _format_line(tuple("-" * width for width in widths), widths),
    ]
    lines.extend(_format_line(row, widths) for row in rows)
    return lines


def _model_cell(group: dict[str, Any]) -> str:
    """Short display name for the group's underlying LLM.

    Full model names stay in the json/report; the table shows the
    ``MODEL_DISPLAY`` short form when one exists. Older results files
    without a "model" field render an empty cell.

    :param group: group dict from roi_results.json.
    :return: display text for the model column.
    """
    model = group.get("model")
    if not isinstance(model, str):
        return ""
    return MODEL_DISPLAY.get(model, model)


def _row_cells(group: dict[str, Any]) -> tuple[str, ...]:
    """Build the table cells for one bet-carrying group.

    :param group: group dict from roi_results.json.
    :return: one string per table column.
    """
    mode = str(group.get("mode") or "")
    brier_all_cell, brier_bets_cell = _fmt_brier(group)
    return (
        str(group.get("tool_name") or "unknown"),
        _MODE_SHORT.get(mode, mode),
        _model_cell(group),
        str(_as_int(group.get("n_eligible"))),
        str(_as_int(group.get("n_bets"))),
        brier_all_cell,
        brier_bets_cell,
        _fmt_staked(group),
        _fmt_roi_ci(group.get("roi_mid"), group.get("roi_ci")),
        # w/costs is point-only: passing ci=None makes _fmt_roi_ci emit the
        # bare "+X.X%" (or "n/a") with no CI suffix.
        _fmt_roi_ci(group.get("roi_haircut"), None),
        _compact_flags(group.get("flags")),
    )


def _display_sort_key(
    group: dict[str, Any],
) -> tuple[int, float, int, str, str, str]:
    """Table display order for one group: delegates to the shared key.

    Extracts the group's scalars and hands them to
    :func:`benchmark.roi_sim.roi_display_sort_key` so this Slack table and
    the markdown report can never disagree on row order.

    :param group: group dict from roi_results.json.
    :return: sort key tuple.
    """
    return roi_display_sort_key(
        group.get("roi_mid"),
        _as_int(group.get("n_bets")),
        str(group.get("tool_name") or ""),
        str(group.get("mode") or ""),
        str(group.get("model") or ""),
    )


def _load_results(results_path: Path) -> dict[str, Any] | None:
    """Read and parse roi_results.json, tolerating any failure.

    A missing file is expected (the roi_sim step may have produced no results
    yet) and stays quiet. A present-but-unreadable or malformed file signals
    upstream pipeline breakage (a broken roi_sim step, a truncated artifact),
    so it is logged at WARNING -- otherwise the vanished ROI section is
    indistinguishable from "no data yet".

    :param results_path: path to the results file.
    :return: parsed payload dict, or None when missing/unparseable.
    """
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning(
            "ROI results at %s are present but unreadable (%s); "
            "skipping the ROI section.",
            results_path,
            exc,
        )
        return None
    if not isinstance(payload, dict):
        log.warning(
            "ROI results at %s parsed to %s, not an object; "
            "skipping the ROI section.",
            results_path,
            type(payload).__name__,
        )
        return None
    return payload


def build_roi_section(results_path: Path, platform: str) -> str | None:
    """Build the Slack mrkdwn ROI companion section for one platform.

    :param results_path: path to roi_results.json (from benchmark.roi_sim).
    :param platform: platform key ("omen" / "polymarket").
    :return: mrkdwn section text, or None when the results file is
        missing/unparseable or carries no groups for the platform --
        callers then append nothing.
    """
    payload = _load_results(results_path)
    if payload is None:
        return None
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return None
    platform_groups = [
        group
        for group in groups
        if isinstance(group, dict) and group.get("platform") == platform
    ]
    if not platform_groups:
        return None

    # Forward-compatible deployment filter: groups explicitly marked
    # "active": false (tools not currently deployed) stay out of the table
    # and the zero-bet listing, surfacing only as a compact trailing count.
    # Older results files without the key render every group as before.
    inactive_groups = [g for g in platform_groups if g.get("active") is False]
    platform_groups = [g for g in platform_groups if g.get("active") is not False]

    window_days = payload.get("window_days")
    window_text = (
        f"trailing {int(window_days)}d"
        if _is_number(window_days)
        else "trailing window"
    )
    lines = [
        f"*Simulated trader ROI* — {window_text}, same decision rules for "
        "every tool (companion sim; full report in the benchmark-data artifact):"
    ]

    prediction_groups = [g for g in platform_groups if g.get("is_prediction_tool")]
    bet_groups = [g for g in prediction_groups if _as_int(g.get("n_bets")) > 0]
    if bet_groups:
        extra = 0
        if len(bet_groups) > MAX_TABLE_ROWS:
            extra = len(bet_groups) - MAX_TABLE_ROWS
            bet_groups = sorted(bet_groups, key=lambda g: -_as_float(g.get("staked")))[
                :MAX_TABLE_ROWS
            ]
        bet_groups.sort(key=_display_sort_key)
        lines.append("```")
        lines.extend(_render_table([_row_cells(g) for g in bet_groups]))
        if extra:
            lines.append(f"{_ELLIPSIS} +{extra} more rows in the full report")
        lines.append("```")

    # A tool counts as zero-bet only when NONE of its groups (any mode)
    # placed a bet -- a tool betting in one mode must not be listed as idle.
    betting_tools = {
        str(g.get("tool_name") or "unknown")
        for g in prediction_groups
        if _as_int(g.get("n_bets")) > 0
    }
    zero_bet_tools = sorted(
        {
            str(g.get("tool_name") or "unknown")
            for g in prediction_groups
            if _as_int(g.get("n_bets")) == 0
        }
        - betting_tools
    )
    if zero_bet_tools:
        lines.append("no bets in window: " + ", ".join(zero_bet_tools))

    excluded_tools = {
        str(g.get("tool_name") or "unknown")
        for g in platform_groups
        if not g.get("is_prediction_tool")
    }
    if excluded_tools:
        lines.append(
            f"excluded non-prediction tools: {len(excluded_tools)} " "— see report"
        )

    # A tool counts as not-deployed only when ALL of its groups carry the
    # explicit "active": false marker.
    active_tool_names = {str(g.get("tool_name") or "unknown") for g in platform_groups}
    inactive_tools = {
        str(g.get("tool_name") or "unknown") for g in inactive_groups
    } - active_tool_names
    if inactive_tools:
        lines.append(f"not deployed/active: {len(inactive_tools)} tools")
    return "\n".join(lines)
