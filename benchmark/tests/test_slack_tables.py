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
"""Tests for the shared fixed-width Slack table renderer."""

import pytest
from benchmark.slack_tables import (
    Column,
    code_block,
    display_width,
    fit,
    render_table,
)


class TestDisplayWidth:
    """Display width must count what Slack actually renders, not characters."""

    def test_ascii_width_is_length(self) -> None:
        """ASCII text occupies one column per character."""
        assert display_width("Brier W-1") == len("Brier W-1")

    @pytest.mark.parametrize("glyph", ["⚠️", "✅", "\U0001f7e2"])
    def test_wide_glyphs_count_double(self, glyph: str) -> None:
        """Emoji-presentation glyphs occupy two columns, not one."""
        assert display_width(glyph) > len(glyph.rstrip("️"))


class TestAlignment:
    """Every rendered line must agree on column boundaries."""

    def test_separator_positions_match_across_rows(self) -> None:
        """Header, divider and data rows share separator offsets."""
        columns = (Column("tool"), Column("n"), Column("Brier"))
        rows = [("superforcaster-polymarket-v1", "301", "0.3491"), ("x", "1", "0.1")]
        lines = render_table(columns, rows)
        offsets = [[i for i, ch in enumerate(line) if ch == "|"] for line in lines[:-1]]
        assert len({tuple(o) for o in offsets}) == 1, lines

    def test_wide_glyph_row_stays_aligned(self) -> None:
        """A row carrying a double-width glyph does not skew the table.

        Mutation check: reverting slack_tables.pad/display_width to len()
        makes this fail, because the glyph row comes out one column short.
        """
        columns = (Column("tool"), Column("flags"))
        rows = [("alpha", "⚠ parse 94%"), ("beta", "clean")]
        lines = render_table(columns, rows)
        widths = {len(line.split(" | ")[0]) for line in lines}
        assert len(widths) == 1, lines

    def test_row_arity_mismatch_is_loud(self) -> None:
        """A short row raises rather than silently dropping a column."""
        columns = (Column("a"), Column("b"))
        with pytest.raises(AssertionError):
            render_table(columns, [("only-one",)])


class TestCaps:
    """Only capped columns may lose content."""

    def test_uncapped_column_never_truncates(self) -> None:
        """A long identifier renders in full when its column has no cap."""
        name = "superforcaster-polymarket-v4-with-a-very-long-suffix"
        lines = render_table((Column("tool"), Column("n")), [(name, "1")])
        assert name in lines[-1]

    def test_capped_column_ellipsizes(self) -> None:
        """A capped column is cut to its cap and marked."""
        lines = render_table((Column("flags", 8),), [("way too much text",)])
        cell = lines[-1]
        assert display_width(cell) == 8
        assert cell.endswith("…")

    def test_fit_respects_display_width_not_length(self) -> None:
        """Fitting a wide-glyph string budgets two columns per glyph."""
        assert display_width(fit("\U0001f7e2\U0001f7e2\U0001f7e2", 5)) <= 5


class TestEmpty:
    """Empty input renders nothing at all."""

    def test_no_rows_renders_no_lines(self) -> None:
        """Callers skip the whole block when there is nothing to show."""
        assert render_table((Column("a"),), []) == []

    def test_code_block_of_nothing_is_empty(self) -> None:
        """An empty table produces no fence, not an empty fenced block."""
        assert code_block([]) == ""
