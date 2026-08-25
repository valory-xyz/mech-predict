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
"""Tests for the Block Kit table builder.

Each assertion mirrors a constraint verified against Slack's own
``blocks.validate`` endpoint. They are asserted locally so CI needs no network
and so an unattended run fails here rather than on an opaque HTTP 400.
"""

import pytest
from benchmark.slack_blocks import (
    Col,
    EMPTY_CELL,
    MAX_COLUMNS,
    MAX_TABLE_CHARS,
    cell,
    message,
    table_block,
)


class TestCellSchema:
    """Slack rejects anything but a cell_text with a non-empty string."""

    def test_cell_is_cell_text(self) -> None:
        """The only shape Slack accepts is cell_text with a plain string."""
        assert cell("0.3485") == {"type": "cell_text", "text": "0.3485"}

    def test_empty_cell_gets_a_placeholder(self) -> None:
        """An empty string fails validation, so blanks render a stand-in."""
        assert cell("")["text"] == EMPTY_CELL

    def test_header_row_is_an_ordinary_row(self) -> None:
        """Slack styles the first row itself; cells carry no style property.

        A `style` key on a cell is rejected by blocks.validate, so a bold
        header cannot be requested and must not be emitted.
        """
        block = table_block((Col("tool"),), [("alpha",)])
        assert block["rows"][0] == [{"type": "cell_text", "text": "tool"}]
        assert all("style" not in c for row in block["rows"] for c in row)


class TestColumnSettings:
    """Alignment and wrapping are the two settings Slack accepts."""

    def test_settings_track_the_columns(self) -> None:
        """Every column contributes one settings entry, in order."""
        block = table_block((Col("tool"), Col("n", align="right")), [("alpha", "44")])
        assert block["column_settings"] == [
            {"align": "left", "is_wrapped": False},
            {"align": "right", "is_wrapped": False},
        ]

    def test_unknown_alignment_is_rejected_locally(self) -> None:
        """`start`/`end`/`justify` are rejected by Slack, so reject them here."""
        with pytest.raises(AssertionError):
            table_block((Col("tool", align="justify"),), [("alpha",)])


class TestBudgets:
    """Over-long blocks are REJECTED by Slack, never truncated."""

    def test_column_ceiling(self) -> None:
        """21 columns fail validation, so fail before POSTing."""
        columns = tuple(Col(f"c{i}") for i in range(MAX_COLUMNS + 1))
        with pytest.raises(AssertionError):
            table_block(columns, [tuple("x" for _ in columns)])

    def test_cell_character_cap(self) -> None:
        """The aggregate cell budget is asserted, not discovered at runtime."""
        with pytest.raises(AssertionError):
            table_block((Col("tool"),), [("x" * (MAX_TABLE_CHARS + 1),)])

    def test_row_arity_mismatch_is_loud(self) -> None:
        """A short row raises rather than silently dropping a column."""
        with pytest.raises(AssertionError):
            table_block((Col("a"), Col("b")), [("only-one",)])


class TestMessage:
    """A webhook payload always carries a notification fallback."""

    def test_fallback_text_is_present(self) -> None:
        """`text` is what a push notification shows; blocks alone are silent."""
        payload = message("1a. Production", [table_block((Col("a"),), [("x",)])])
        assert payload["text"] == "1a. Production"
        assert payload["blocks"][0]["type"] == "table"
