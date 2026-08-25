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
"""Fixed-width Slack table renderer, extracted from ``benchmark.roi_slack``.

Pure and stdlib-only: same ``(columns, rows)`` in, same bytes out. Padding is
display-width-aware (:func:`display_width`) so emoji-presented glyphs cannot
skew the columns to their right; only columns with an explicit cap are ever
ellipsized.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

SEPARATOR = " | "
ELLIPSIS = "…"


@dataclass(frozen=True)
class Column:
    """One table column: its header text and optional width cap.

    :param header: header cell text; also participates in width computation.
    :param cap: maximum display width, or None for uncapped (content-driven).
    """

    header: str
    cap: int | None = None


# Codepoints Slack renders with EMOJI presentation even though Unicode marks
# them Neutral or Ambiguous, so ``east_asian_width`` alone under-counts them.
# Only symbols that actually appear in a table cell need listing; U+26A0 is
# the parse-reliability warning emitted by ``roi_slack._compact_flags``.
_EMOJI_PRESENTATION_EXTRAS = frozenset("\u26a0\u2705\u274c\u2757\u2b50")

# U+FE0F (variation selector-16) upgrades the PRECEDING character to emoji
# presentation. It has no width of its own, and it makes its base two columns.
_VS16 = "\ufe0f"


def char_width(char: str) -> int:
    """Return the number of columns a single character occupies.

    :param char: one character.
    :return: 0, 1 or 2 columns.
    """
    if char == _VS16:
        # Width contributed by the sequence is accounted on the base char.
        return 0
    if char in _EMOJI_PRESENTATION_EXTRAS:
        return 2
    return 2 if unicodedata.east_asian_width(char) in "WF" else 1


def display_width(text: str) -> int:
    """Return the number of columns *text* occupies when Slack renders it.

    :param text: cell text.
    :return: display width in columns.
    """
    total = 0
    for index, char in enumerate(text):
        width = char_width(char)
        # A base character explicitly followed by VS16 is emoji-presented.
        if width == 1 and text[index + 1 : index + 2] == _VS16:
            width = 2
        total += width
    return total


def fit(text: str, width: int) -> str:
    """Truncate *text* to *width* display columns, marking cuts with an ellipsis.

    :param text: cell text.
    :param width: maximum width in display columns.
    :return: text whose display width is <= width.
    """
    if display_width(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    kept: list[str] = []
    used = 0
    # Reserve one column for the ellipsis itself.
    for index, char in enumerate(text):
        this_width = char_width(char)
        if this_width == 1 and text[index + 1 : index + 2] == _VS16:
            this_width = 2
        if used + this_width > width - 1:
            break
        kept.append(char)
        used += this_width
    return "".join(kept) + ELLIPSIS


def pad(text: str, width: int) -> str:
    """Left-align *text* in a field of *width* display columns.

    :param text: cell text (already fitted).
    :param width: field width in display columns.
    :return: text padded with spaces to the requested display width.
    """
    return text + " " * max(0, width - display_width(text))


def format_line(cells: Sequence[str], widths: Sequence[int]) -> str:
    """Render one table line: fitted, padded cells joined by the separator.

    :param cells: one string per column.
    :param widths: column widths (same length as cells).
    :return: single table line, trailing whitespace stripped.
    """
    return SEPARATOR.join(
        pad(fit(cell, width), width) for cell, width in zip(cells, widths)
    ).rstrip()


def column_widths(
    columns: Sequence[Column], rows: Iterable[Sequence[str]]
) -> list[int]:
    """Compute each column's rendered width.

    Width is the widest of the header and every cell, then clamped to the
    column's cap when it has one.

    :param columns: column specs.
    :param rows: table rows, each a sequence of cell strings.
    :return: one width per column.
    """
    materialized = [tuple(row) for row in rows]
    widths = []
    for index, column in enumerate(columns):
        content = max(
            [display_width(column.header)]
            + [display_width(row[index]) for row in materialized]
        )
        widths.append(content if column.cap is None else min(column.cap, content))
    return widths


def render_table(columns: Sequence[Column], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render header + divider + data lines, without code-block fences.

    :param columns: column specs; length must match every row's cell count.
    :param rows: pre-formatted cell sequences, one per table row. An empty
        list renders nothing.
    :return: table lines, without code-block fences.
    :raises AssertionError: when a row's cell count differs from the column
        count.
    """
    if not rows:
        return []
    for row in rows:
        assert len(row) == len(
            columns
        ), f"row has {len(row)} cells, expected {len(columns)}: {row!r}"
    widths = column_widths(columns, rows)
    header = format_line([column.header for column in columns], widths)
    divider = format_line(["-" * width for width in widths], widths)
    return [header, divider] + [format_line(row, widths) for row in rows]
