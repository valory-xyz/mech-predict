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

Constraints, verified against ``https://slack.com/api/blocks.validate``
(over-budget blocks are REJECTED, surfacing as an opaque HTTP 400 on a
webhook, so ``table_block`` asserts the documented budgets locally):

* a cell is ``{"type": "cell_text", "text": "..."}``. ``raw_text`` and an
  ``elements`` array are both rejected, and ``style`` is not an allowed
  property on a cell -- so a header row cannot be bolded;
* an EMPTY cell string is rejected, so every blank must be a placeholder;
* ``column_settings`` accepts ``align`` (``left`` / ``right`` / ``center``
  only -- ``start``/``end``/``justify`` are rejected) and ``is_wrapped``;
* 20 columns validate, 21 do not;
* a ``section`` and a ``table`` coexist in one message, and two tables do too.
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
    :param align: ``left``, ``right`` or ``center``.
    :param wrap: whether Slack may wrap the cell.
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

    The first rendered row is the header (Slack has no header-row flag).

    :param columns: column specs; length must match every row's cell count.
    :param rows: pre-formatted cell strings, one sequence per row.
    :return: a Block Kit ``table`` block.
    :raises AssertionError: on a row-arity mismatch, an unknown alignment, or
        a documented Slack budget being exceeded -- otherwise an opaque HTTP
        400 on an unattended run.
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

    header_row = [cell(column.header) for column in columns]
    body = [[cell(value) for value in row] for row in rows]

    total = sum(len(column.header) for column in columns) + sum(
        len(value) for row in rows for value in row
    )
    assert (
        total <= MAX_TABLE_CHARS
    ), f"table cells total {total} chars, over Slack's {MAX_TABLE_CHARS} cap"

    return {
        "type": "table",
        "rows": [header_row] + body,
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


def header(text: str) -> dict[str, Any]:
    """Build a header block -- Slack's large-title style.

    ``plain_text`` only (mrkdwn is rejected); Slack caps it at 150 characters.

    :param text: the title, plain text.
    :return: a ``header`` block.
    """
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": text[:150], "emoji": True},
    }


def context(text: str) -> dict[str, Any]:
    """Build a context block, used for legends and footnotes.

    :param text: Slack mrkdwn.
    :return: a ``context`` block.
    """
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}


def message(fallback: str, blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Wrap blocks in a webhook payload.

    :param fallback: plain-text summary, the notification fallback text.
    :param blocks: Block Kit blocks.
    :return: the JSON payload to POST to an incoming webhook.
    """
    return {"text": fallback, "blocks": list(blocks)}
