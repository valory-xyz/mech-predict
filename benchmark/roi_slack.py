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
"""Render the ROI companion section appended to the daily benchmark Slack post.

Pure builder over ``roi_results.json`` (written by ``benchmark.roi_sim``):
no network, no LLM, stdlib only. :func:`build_roi_section` returns a Slack
mrkdwn snippet -- one intro line plus a compact fixed-width table inside a
fenced code block -- or None when there is nothing to append (missing or
unparseable results file, or no groups for the platform). Callers treat
None as "append nothing"; a broken ROI section must never break the daily
post, so this module never raises for bad input data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeGuard

# Slack renders ~monospace code blocks without wrapping up to roughly this
# width on a desktop client; every table line is kept at or under it.
MAX_LINE_WIDTH = 100

# Truncation guard for very wide deployments: keep the top rows by staked
# capital and point at the full report for the rest.
MAX_TABLE_ROWS = 14

_HEADERS = ("tool", "mode", "preds", "bets", "ROI (95% CI)", "w/costs", "flags")
# Per-column width caps; actual widths are content-driven up to these.
_CAPS = (28, 5, 6, 5, 20, 8, 22)

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
# When the capped widths still exceed MAX_LINE_WIDTH, shrink the tool
# column first (down to this floor), then the flags column.
_TOOL_MIN = 12
_FLAGS_MIN = 6

_ELLIPSIS = "…"


def _is_num(value: object) -> TypeGuard[float]:
    """Return True for a real (non-bool) int/float.

    The TypeGuard narrows the value to float for the caller's arithmetic
    (mirrors ``benchmark.roi_sim._is_number``).

    :param value: candidate value.
    :return: True when usable as a number.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_int(value: object) -> int:
    """Coerce a JSON count field to int, defaulting to 0.

    :param value: candidate value.
    :return: integer value, or 0 when not a number.
    """
    return int(value) if _is_num(value) else 0


def _as_float(value: object) -> float:
    """Coerce a JSON numeric field to float, defaulting to 0.0.

    :param value: candidate value.
    :return: float value, or 0.0 when not a number.
    """
    return float(value) if _is_num(value) else 0.0


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
    if not _is_num(roi):
        return "n/a"
    roi_text = f"{_fmt_signed(float(roi))}%"
    if isinstance(ci, list) and len(ci) == 2 and _is_num(ci[0]) and _is_num(ci[1]):
        return f"{roi_text} ({_fmt_signed(float(ci[0]))},{_fmt_signed(float(ci[1]))})"
    return roi_text


def _fmt_roi_point(roi: object) -> str:
    """Format the "w/costs" table cell (point estimate only).

    The haircut CI stays in the full report; the compact table keeps
    the with-costs column to the point value so lines fit the width cap.

    :param roi: haircut ROI in percent, or None.
    :return: display string.
    """
    if not _is_num(roi):
        return "n/a"
    return f"{_fmt_signed(float(roi))}%"


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
            text = _FLAG_SHORT.get(text, text.split(" — ")[0].split(" - ")[0]).strip()
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
    """Render header + divider + data lines, all within MAX_LINE_WIDTH.

    Widths are content-driven up to the per-column caps; if the capped
    total still exceeds the width budget the tool column shrinks first,
    then flags -- both truncations are marked with an ellipsis.

    :param rows: pre-formatted cell tuples, one per table row.
    :return: table lines (no code-block fences).
    """
    widths = [
        min(
            cap,
            max(len(header), *(len(row[i]) for row in rows)),
        )
        for i, (header, cap) in enumerate(zip(_HEADERS, _CAPS))
    ]
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


def _row_cells(group: dict[str, Any]) -> tuple[str, ...]:
    """Build the table cells for one bet-carrying group.

    :param group: group dict from roi_results.json.
    :return: one string per table column.
    """
    mode = str(group.get("mode") or "")
    return (
        str(group.get("tool_name") or "unknown"),
        _MODE_SHORT.get(mode, mode),
        str(_as_int(group.get("n_eligible"))),
        str(_as_int(group.get("n_bets"))),
        _fmt_roi_ci(group.get("roi_mid"), group.get("roi_ci")),
        _fmt_roi_point(group.get("roi_haircut")),
        _compact_flags(group.get("flags")),
    )


def _display_sort_key(group: dict[str, Any]) -> tuple[int, int, str]:
    """Table display order: production first, then bet count descending.

    :param group: group dict from roi_results.json.
    :return: sort key tuple.
    """
    return (
        0 if group.get("mode") == "production" else 1,
        -_as_int(group.get("n_bets")),
        str(group.get("tool_name") or ""),
    )


def _load_results(results_path: Path) -> dict[str, Any] | None:
    """Read and parse roi_results.json, tolerating any failure.

    :param results_path: path to the results file.
    :return: parsed payload dict, or None when missing/unparseable.
    """
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
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

    window_days = payload.get("window_days")
    window_text = (
        f"trailing {int(window_days)}d" if _is_num(window_days) else "trailing window"
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
    return "\n".join(lines)
