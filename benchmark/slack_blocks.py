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
"""Build Slack Block Kit ``table`` blocks for the benchmark digest.

Slack lays the columns out itself, so nothing here pads, measures or truncates
a cell. That removes a whole class of defect the fixed-width renderer has to
defend against: display-width miscounts on non-ASCII glyphs, a 239-character
line that only scrolls on desktop, and column skew whenever a cell contains
something wider than ``len()`` reports.

Verified against Slack, not assumed
-----------------------------------
Every constraint below was checked against ``https://slack.com/api/blocks.validate``,
which needs no token and no scopes:

* a cell is ``{"type": "cell_text", "text": "..."}``. ``raw_text`` and an
  ``elements`` array are both rejected, and ``style`` is not an allowed
  property on a cell -- so a header row cannot be bolded;
* an EMPTY cell string is rejected, so every blank must be a placeholder;
* ``column_settings`` accepts ``align`` (``left`` / ``right`` / ``center``
  only -- ``start``/``end``/``justify`` are rejected) and ``is_wrapped``;
* 20 columns validate, 21 do not;
* a ``section`` and a ``table`` coexist in one message, and two tables do too.

An incoming webhook accepts these blocks (HTTP 200 against a live webhook), so
this needs no bot token, no OAuth scope and no transport change.

Fail loud, not silent
---------------------
Over-long TEXT is truncated by Slack; over-long BLOCKS are REJECTED, which on a
webhook surfaces as an opaque HTTP 400. There is no graceful degradation, so
:func:`table_block` asserts the documented budgets locally rather than letting
an unattended CI job discover them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

# Verified against blocks.validate: 20 columns pass, 21 fail.
MAX_COLUMNS = 20
MAX_ROWS = 100

# Slack caps the aggregate of every table cell in a message. Kept as a module
# constant so callers can assert it before POSTing rather than after.
MAX_TABLE_CHARS = 10_000

# An empty cell string fails validation, so blanks need a visible stand-in.
EMPTY_CELL = "-"

_ALIGNMENTS = frozenset({"left", "right", "center"})


@dataclass(frozen=True)
class Col:
    """One table column.

    :param header: header cell text.
    :param align: ``left``, ``right`` or ``center``. Numeric columns read far
        better right-aligned, which the fixed-width renderer could not do.
    :param wrap: whether Slack may wrap the cell. False keeps an identifier
        such as a 37-character tool name on one line.
    """

    header: str
    align: str = "left"
    wrap: bool = False


def cell(text: str) -> dict[str, Any]:
    """Build one table cell.

    :param text: cell text; an empty string is replaced, since Slack rejects it.
    :return: a ``cell_text`` element.
    """
    return {"type": "cell_text", "text": text if text else EMPTY_CELL}


def table_block(
    columns: Sequence[Col], rows: Sequence[Sequence[str]]
) -> dict[str, Any]:
    """Build a ``table`` block from a column spec and pre-formatted rows.

    The first rendered row is the header. Slack has no header-row flag and no
    per-cell styling, so it is an ordinary row -- Slack styles it itself.

    :param columns: column specs; length must match every row's cell count.
    :param rows: pre-formatted cell strings, one sequence per row.
    :return: a Block Kit ``table`` block.
    :raises AssertionError: on a row-arity mismatch, an unknown alignment, or a
        documented Slack budget being exceeded -- all of which Slack would
        otherwise reject as an opaque HTTP 400 on an unattended run.
    """
    assert (
        len(columns) <= MAX_COLUMNS
    ), f"{len(columns)} columns exceeds Slack's limit of {MAX_COLUMNS}"
    assert (
        len(rows) + 1 <= MAX_ROWS
    ), f"{len(rows) + 1} rows exceeds Slack's limit of {MAX_ROWS}"
    for column in columns:
        assert (
            column.align in _ALIGNMENTS
        ), f"alignment {column.align!r} is not one of {sorted(_ALIGNMENTS)}"
    for row in rows:
        assert len(row) == len(
            columns
        ), f"row has {len(row)} cells, expected {len(columns)}: {row!r}"

    header = [cell(column.header) for column in columns]
    body = [[cell(value) for value in row] for row in rows]

    total = sum(len(column.header) for column in columns) + sum(
        len(value) for row in rows for value in row
    )
    assert (
        total <= MAX_TABLE_CHARS
    ), f"table cells total {total} chars, over Slack's {MAX_TABLE_CHARS} cap"

    return {
        "type": "table",
        "rows": [header] + body,
        "column_settings": [
            {"align": column.align, "is_wrapped": column.wrap} for column in columns
        ],
    }


def section(text: str) -> dict[str, Any]:
    """Build a mrkdwn section block, used as a table's caption.

    :param text: Slack mrkdwn.
    :return: a ``section`` block.
    """
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def context(text: str) -> dict[str, Any]:
    """Build a context block, used for legends and footnotes.

    :param text: Slack mrkdwn.
    :return: a ``context`` block.
    """
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}


def message(fallback: str, blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Wrap blocks in a webhook payload.

    ``text`` is required as the notification fallback: it is what a reader sees
    in a push notification and in clients that cannot render the blocks.

    :param fallback: plain-text summary of the message.
    :param blocks: Block Kit blocks.
    :return: the JSON payload to POST to an incoming webhook.
    """
    return {"text": fallback, "blocks": list(blocks)}
