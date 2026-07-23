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
"""Tests for benchmark/roi_slack.py — the ROI companion Slack section."""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pytest
from benchmark import notify_slack, roi_slack
from benchmark.roi_slack import (
    MAX_LINE_WIDTH,
    MAX_TABLE_ROWS,
    STALE_AFTER_DAYS,
    _HEADERS,
    _load_results,
    _render_table,
    build_roi_section,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group(
    tool: str = "test-tool",
    platform: str = "omen",
    mode: str = "production",
    model: Any = "gpt-4.1-2025-04-14",
    n_eligible: int = 40,
    n_bets: int = 20,
    staked: float = 25.0,
    brier_all: Any = 0.218,
    brier_bets: Any = 0.221,
    roi_mid: Any = 12.3,
    roi_ci: Any = None,
    roi_haircut: Any = 8.4,
    roi_haircut_ci: Any = None,
    flags: Any = None,
    is_prediction_tool: bool = True,
    parse_reliability: Any = 1.0,
    active: Any = None,
) -> dict[str, Any]:
    """Build a minimal roi_results.json group entry.

    ``active`` is only written when not None (the key is absent in older
    results files and only emitted by the deployment filter).

    :param tool: tool name for the group.
    :param platform: platform key (omen / polymarket).
    :param mode: deployment mode (production / tournament).
    :param model: underlying LLM identifier.
    :param n_eligible: count of eligible predictions.
    :param n_bets: count of simulated bets.
    :param staked: total simulated stake.
    :param brier_all: mean Brier over all eligible predictions.
    :param brier_bets: mean Brier over the gated bet subset.
    :param roi_mid: mid-point ROI percentage.
    :param roi_ci: ROI 95% CI pair, or None for the default.
    :param roi_haircut: cost-adjusted ROI percentage.
    :param roi_haircut_ci: cost-adjusted ROI 95% CI pair, or None for the default.
    :param flags: list of warning flags, or None for none.
    :param is_prediction_tool: whether the tool is a prediction tool.
    :param parse_reliability: parse-reliability ratio.
    :param active: deployment-filter flag; omitted from the entry when None.
    :return: the group entry dict.
    """
    entry: dict[str, Any] = {
        "platform": platform,
        "tool_name": tool,
        "mode": mode,
        "model": model,
        "n_eligible": n_eligible,
        "n_bets": n_bets,
        "staked": staked,
        "brier_all": brier_all,
        "brier_bets": brier_bets,
        "roi_mid": roi_mid,
        "roi_ci": roi_ci if roi_ci is not None else [4.1, 20.9],
        "roi_haircut": roi_haircut,
        "roi_haircut_ci": (
            roi_haircut_ci if roi_haircut_ci is not None else [-1.2, 18.0]
        ),
        "flags": flags or [],
        "is_prediction_tool": is_prediction_tool,
        "parse_reliability": parse_reliability,
    }
    if active is not None:
        entry["active"] = active
    return entry


def _write_results(
    tmp_path: Path,
    groups: list[dict[str, Any]],
    window_days: int = 90,
    as_of: Optional[str] = None,
) -> Path:
    """Write a roi_results.json fixture and return its path.

    :param tmp_path: pytest tmp_path fixture.
    :param groups: group entries to embed in the payload.
    :param window_days: trailing window length in days.
    :param as_of: results timestamp; defaults to today so the freshness guard
        treats the fixture as current. Pass an old date to exercise staleness.
    :return: path of the written roi_results.json.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc).date().isoformat()
    path = tmp_path / "roi_results.json"
    payload = {
        "as_of": as_of,
        "window_days": window_days,
        "window_start": "2026-04-08T00:00:00+00:00",
        "window_end": f"{as_of}T00:00:00+00:00",
        "groups": groups,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _code_block_lines(section: str) -> list[str]:
    """Extract the lines inside the fenced code block of a section.

    :param section: rendered Slack section text.
    :return: the lines between the two code fences.
    """
    lines = section.splitlines()
    fences = [i for i, line in enumerate(lines) if line == "```"]
    assert len(fences) == 2, f"expected one fenced code block, got {section!r}"
    return lines[fences[0] + 1 : fences[1]]


# ---------------------------------------------------------------------------
# build_roi_section — happy path
# ---------------------------------------------------------------------------


class TestBuildRoiSectionHappyPath:
    """The section renders intro + code-block table for bet-carrying groups."""

    def test_intro_line_and_window_days(self, tmp_path: Path) -> None:
        """First line is the bold intro citing the window from the json.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(tmp_path, [_group()], window_days=21)
        section = build_roi_section(path, "omen")
        assert section is not None
        first = section.splitlines()[0]
        assert first.startswith("*Simulated trader ROI*")
        assert "trailing 21d" in first

    def test_code_block_carries_all_columns(self, tmp_path: Path) -> None:
        """Header row inside the code block names every column.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(tmp_path, [_group()])
        section = build_roi_section(path, "omen")
        assert section is not None
        header = _code_block_lines(section)[0]
        for column in (
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
        ):
            assert column in header, f"missing column: {column}"
        assert "Brier all->bets" not in header

    def test_row_renders_counts_roi_and_ci(self, tmp_path: Path) -> None:
        """A data row shows n_eligible, n_bets, and the ROI with its CI.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(
                    tool="alpha",
                    n_eligible=40,
                    n_bets=20,
                    roi_mid=12.3,
                    roi_ci=[4.1, 20.9],
                )
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert len(rows) == 1
        assert "alpha" in rows[0]
        assert "40" in rows[0]
        assert "20" in rows[0]
        assert "+12.3% (+4.1,+20.9)" in rows[0]

    def test_row_renders_brier_and_staked(self, tmp_path: Path) -> None:
        """A data row carries the two Brier cells separately, plus staked.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [_group(tool="alpha", brier_all=0.218, brier_bets=0.221, staked=274305.0)],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        # The two scores land in two distinct cells, not one arrow cell.
        assert "0.218" in rows[0]
        assert "0.221" in rows[0]
        assert "0.218 -> 0.221" not in rows[0]
        assert "274305.00" in rows[0]

    def test_absent_brier_fields_render_na(self, tmp_path: Path) -> None:
        """Missing Brier fields render two 'n/a' cells (one per column).

        :param tmp_path: pytest tmp_path fixture.
        """
        group = _group(tool="alpha")
        del group["brier_all"]
        del group["brier_bets"]
        path = _write_results(tmp_path, [group])
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        # ROI / w-costs are populated, so the only n/a cells are the two
        # Brier columns.
        assert rows[0].count("n/a") == 2
        assert "n/a -> n/a" not in rows[0]

    def test_one_brier_field_missing_renders_per_column_na(
        self, tmp_path: Path
    ) -> None:
        """A single missing Brier field only blanks its own column.

        :param tmp_path: pytest tmp_path fixture.
        """
        group = _group(tool="alpha", brier_all=0.207)
        del group["brier_bets"]
        path = _write_results(tmp_path, [group])
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "0.207" in rows[0]
        assert rows[0].count("n/a") == 1

    def test_sorted_by_roi_desc_across_modes(self, tmp_path: Path) -> None:
        """Higher ROI ranks first even when it is a tournament tool.

        A tournament tool with better ROI now legitimately precedes a
        lower-ROI production tool -- mode is no longer a sort key.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="p-low", mode="production", roi_mid=3.0, n_bets=50),
                _group(tool="t-high", mode="tournament", roi_mid=25.0, n_bets=5),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "t-high" in rows[0]
        assert "p-low" in rows[1]

    def test_roi_none_sorts_last(self, tmp_path: Path) -> None:
        """A group with a None ROI sorts below every ROI-bearing row.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="no-roi", roi_mid=None, n_bets=10),
                _group(tool="has-roi", roi_mid=5.0, n_bets=10),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "has-roi" in rows[0]
        assert "no-roi" in rows[1]

    def test_equal_roi_sorted_by_bets_desc(self, tmp_path: Path) -> None:
        """Equal-ROI rows fall back to bet count descending.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="few", roi_mid=10.0, n_bets=3),
                _group(tool="many", roi_mid=10.0, n_bets=30),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "many" in rows[0]
        assert "few" in rows[1]

    def test_zero_bet_groups_stay_out_of_table(self, tmp_path: Path) -> None:
        """Groups with n_bets == 0 never occupy a table row.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [_group(tool="bettor", n_bets=10), _group(tool="idle", n_bets=0)],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)
        assert not any("idle" in row for row in rows)

    def test_other_platform_groups_ignored(self, tmp_path: Path) -> None:
        """Only the requested platform's groups render.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="omen-tool", platform="omen"),
                _group(tool="poly-tool", platform="polymarket"),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "omen-tool" in section
        assert "poly-tool" not in section

    def test_flags_compacted(self, tmp_path: Path) -> None:
        """Verbose flags render as their short spellings in the flags cell.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [_group(flags=["few bets - anecdotal", "low sample"])],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "few bets, low n" in rows[0]
        assert "anecdotal" not in rows[0]

    def test_parse_reliability_flag_compacted(self, tmp_path: Path) -> None:
        """The parse-reliability warning compacts to '⚠ parse NN%'.

        :param tmp_path: pytest tmp_path fixture.
        """
        long_flag = "⚠ 45% parse reliability — possible response-format gap"
        path = _write_results(tmp_path, [_group(flags=[long_flag])])
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "⚠ parse 45%" in rows[0]
        assert "response-format" not in rows[0]

    def test_mode_labels_abbreviated(self, tmp_path: Path) -> None:
        """Modes render as 'prod' / 'tourn' to keep the table narrow.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="p-tool", mode="production"),
                _group(tool="t-tool", mode="tournament"),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "prod" in rows[0] and "production" not in rows[0]
        assert "tourn" in rows[1] and "tournament" not in rows[1]


class TestModelColumn:
    """Groups are split per underlying LLM; the table shows a model column."""

    def test_model_rendered_short(self, tmp_path: Path) -> None:
        """Known full model names render as their short display form.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(tmp_path, [_group(model="gpt-4.1-2025-04-14")])
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "gpt-4.1" in rows[0]
        assert "gpt-4.1-2025-04-14" not in section

    def test_unmapped_model_kept_verbatim(self, tmp_path: Path) -> None:
        """Models without a display mapping render their full name.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(tmp_path, [_group(model="claude-sonnet-4-6")])
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "claude-sonnet-4-6" in rows[0]

    def test_split_groups_render_separate_rows(self, tmp_path: Path) -> None:
        """Same tool+mode under two models -> two table rows.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="split-tool", model="gpt-4.1-2025-04-14", n_bets=9),
                _group(tool="split-tool", model="gpt-4o-2024-08-06", n_bets=9),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert len(rows) == 2
        assert all("split-tool" in row for row in rows)
        assert "gpt-4.1" in rows[0]
        assert "gpt-4o" in rows[1]

    def test_absent_model_field_renders_empty_cell(self, tmp_path: Path) -> None:
        """Older results files without a model field still render rows.

        :param tmp_path: pytest tmp_path fixture.
        """
        group = _group()
        del group["model"]
        path = _write_results(tmp_path, [group])
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert len(rows) == 1
        assert "test-tool" in rows[0]


# ---------------------------------------------------------------------------
# Zero-bet and excluded lines
# ---------------------------------------------------------------------------


class TestZeroBetLine:
    """Prediction tools with zero bets are listed on one compact line."""

    def test_zero_bet_tools_listed(self, tmp_path: Path) -> None:
        """Zero-bet prediction tools appear comma-joined after the block.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="bettor", n_bets=10),
                _group(tool="idle-b", n_bets=0),
                _group(tool="idle-a", n_bets=0),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "no bets in window: idle-a, idle-b" in section

    def test_line_omitted_when_all_tools_bet(self, tmp_path: Path) -> None:
        """No zero-bet line when every prediction tool placed bets.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(tmp_path, [_group()])
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "no bets in window" not in section

    def test_tool_betting_in_any_mode_not_listed(self, tmp_path: Path) -> None:
        """Keep a tool off the idle line when any of its modes has bets.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="multi", mode="production", n_bets=0),
                _group(tool="multi", mode="tournament", n_bets=8),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "no bets in window" not in section

    def test_tool_betting_under_any_model_not_listed(self, tmp_path: Path) -> None:
        """Keep a tool off the idle line when any of its models has bets.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="multi", model="gpt-4.1-2025-04-14", n_bets=0),
                _group(tool="multi", model="gpt-4o-2024-08-06", n_bets=8),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "no bets in window" not in section


class TestActiveFilter:
    """Groups marked "active": false are excluded and counted, not rendered."""

    def test_inactive_excluded_from_table_and_counted(self, tmp_path: Path) -> None:
        """active:false tools leave the table; a trailing count appears.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="live-tool", n_bets=10),
                _group(tool="retired-a", n_bets=5, active=False),
                _group(tool="retired-b", n_bets=0, active=False),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "retired-a" not in section
        assert "retired-b" not in section
        assert "not deployed/active: 2 tools" in section

    def test_inactive_excluded_from_zero_bet_line(self, tmp_path: Path) -> None:
        """A zero-bet inactive tool is counted, never listed as idle.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(tool="live-tool", n_bets=10),
                _group(tool="retired", n_bets=0, active=False),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "no bets in window" not in section
        assert "not deployed/active: 1 tools" in section

    def test_active_true_renders_normally(self, tmp_path: Path) -> None:
        """An explicit active:true group renders like any other.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(tmp_path, [_group(tool="live-tool", active=True)])
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "live-tool" in section
        assert "not deployed/active" not in section

    def test_absent_active_key_back_compat(self, tmp_path: Path) -> None:
        """Older results files without the key behave exactly as before.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [_group(tool="bettor", n_bets=10), _group(tool="idle", n_bets=0)],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "bettor" in section
        assert "no bets in window: idle" in section
        assert "not deployed/active" not in section


class TestExcludedLine:
    """Non-prediction tools are summarized as a count only."""

    def test_excluded_count_rendered(self, tmp_path: Path) -> None:
        """Count of distinct non-prediction tools, names omitted.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path,
            [
                _group(),
                _group(tool="question-gen", is_prediction_tool=False, n_bets=0),
                _group(tool="service-mech", is_prediction_tool=False, n_bets=0),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "excluded non-prediction tools: 2 — see report" in section
        assert "question-gen" not in section

    def test_line_omitted_when_none_excluded(self, tmp_path: Path) -> None:
        """No excluded line when every group is a prediction tool.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(tmp_path, [_group()])
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "excluded non-prediction tools" not in section


# ---------------------------------------------------------------------------
# None returns
# ---------------------------------------------------------------------------


class TestReturnsNone:
    """Callers append nothing when there is nothing to render."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """Missing results file -> None.

        :param tmp_path: pytest tmp_path fixture.
        """
        assert build_roi_section(tmp_path / "absent.json", "omen") is None

    def test_unparseable_file(self, tmp_path: Path) -> None:
        """Corrupt JSON -> None.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = tmp_path / "roi_results.json"
        path.write_text("{not json", encoding="utf-8")
        assert build_roi_section(path, "omen") is None

    def test_non_dict_payload(self, tmp_path: Path) -> None:
        """A JSON list payload -> None.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = tmp_path / "roi_results.json"
        path.write_text("[1, 2]", encoding="utf-8")
        assert build_roi_section(path, "omen") is None

    def test_no_groups_for_platform(self, tmp_path: Path) -> None:
        """Groups exist but none on the requested platform -> None.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(tmp_path, [_group(platform="polymarket")])
        assert build_roi_section(path, "omen") is None

    def test_empty_groups(self, tmp_path: Path) -> None:
        """Empty groups list -> None.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(tmp_path, [])
        assert build_roi_section(path, "omen") is None


# ---------------------------------------------------------------------------
# Truncation + width bounds
# ---------------------------------------------------------------------------


class TestTruncation:
    """Wide deployments keep the top rows by staked and point at the report."""

    def test_more_than_max_rows_truncates(self, tmp_path: Path) -> None:
        """17 bet groups -> MAX_TABLE_ROWS rows + '+N more' line.

        :param tmp_path: pytest tmp_path fixture.
        """
        groups = [
            _group(tool=f"tool-{i:02d}", n_bets=10 + i, staked=float(i))
            for i in range(MAX_TABLE_ROWS + 3)
        ]
        path = _write_results(tmp_path, groups)
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)
        # header + divider + MAX_TABLE_ROWS data rows + "+N more" line
        assert len(rows) == 2 + MAX_TABLE_ROWS + 1
        assert "+3 more rows in the full report" in rows[-1]

    def test_kept_rows_are_top_by_staked(self, tmp_path: Path) -> None:
        """The lowest-staked groups are the ones dropped.

        :param tmp_path: pytest tmp_path fixture.
        """
        groups = [
            _group(tool=f"tool-{i:02d}", staked=float(i))
            for i in range(MAX_TABLE_ROWS + 2)
        ]
        path = _write_results(tmp_path, groups)
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "tool-00" not in section
        assert "tool-01" not in section
        assert f"tool-{MAX_TABLE_ROWS + 1:02d}" in section

    def test_at_max_rows_no_truncation(self, tmp_path: Path) -> None:
        """Exactly MAX_TABLE_ROWS groups render fully with no more-line.

        :param tmp_path: pytest tmp_path fixture.
        """
        groups = [_group(tool=f"tool-{i:02d}") for i in range(MAX_TABLE_ROWS)]
        path = _write_results(tmp_path, groups)
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "more rows in the full report" not in section


class TestLineWidth:
    """The tool name + numeric cells are never truncated; flags may shrink."""

    def test_long_tool_name_renders_in_full(self, tmp_path: Path) -> None:
        """A long tool name is never truncated (Slack scrolls horizontally).

        Numeric columns (preds, bets, Brier, staked, ROI, w/costs) never carry
        the ellipsis marker, however extreme their values.

        :param tmp_path: pytest tmp_path fixture.
        """
        long_tool = "a-very-long-tool-name-that-exceeds-the-old-column-cap-x" * 2
        long_flag = "⚠ 45% parse reliability — possible response-format gap"
        groups = [
            _group(
                tool=long_tool,
                model="claude-3-5-sonnet-20241022",
                n_eligible=999999,
                n_bets=888888,
                brier_all=0.123456,
                brier_bets=0.654321,
                staked=1234567.89,
                roi_mid=-100.0,
                roi_ci=[-1234.5, 6789.0],
                roi_haircut=-100.0,
                flags=[long_flag, "few bets - anecdotal", "low sample"],
            )
        ]
        path = _write_results(tmp_path, groups)
        section = build_roi_section(path, "omen")
        assert section is not None
        data_row = _code_block_lines(section)[2]
        # The tool cell is the first column, left-justified to its own width,
        # so the row starts with the full name -- no ellipsis.
        assert data_row.startswith(long_tool)
        for value in ("999999", "888888", "0.123", "0.654", "1234567.89"):
            assert value in data_row, f"numeric cell truncated: {value}"
        # Flags is NOT sacrificed: the long tool name is the overflow cause, so
        # clipping flags cannot restore MAX_LINE_WIDTH. The backstop leaves flags
        # at its own cap instead of shrinking it to the floor ("⚠ par…").
        assert "parse 45%" in data_row

    def test_backstop_shrinks_flags_only_when_it_restores_the_bound(self) -> None:
        """Flags shrinks iff that alone brings the line under the target.

        Tool-caused overflow: flags keeps its capped width (clipping it cannot
        help). Flags-caused (small) overflow: flags shrinks and the line fits.
        """
        n = len(_HEADERS)
        # (1) Tool is the cause -> line far over the target; flags untouched.
        tool_row = ("x" * 200,) + ("y",) * (n - 2) + ("flagflag",)
        line = _render_table([tool_row])[-1]
        assert len(line) > MAX_LINE_WIDTH
        assert line.split(" | ")[-1] == "flagflag"  # not shrunk to the floor

        # (2) A moderate tool leaves a small overflow that flags (capped) can
        # absorb -> flags shrinks and the line lands on the target.
        flags_row = ("t" * 50,) + ("y",) * (n - 2) + ("f" * 30,)
        line2 = _render_table([flags_row])[-1]
        assert len(line2) <= MAX_LINE_WIDTH
        assert line2.split(" | ")[-1].endswith("…")  # flags was shrunk

    def test_six_digit_bets_not_ellipsized(self, tmp_path: Path) -> None:
        """A 6-digit bet count renders in full, never '1097…'.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = _write_results(
            tmp_path, [_group(tool="alpha", n_eligible=200000, n_bets=109722)]
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        data_row = _code_block_lines(section)[2]
        assert "109722" in data_row
        assert "…" not in data_row

    def test_typical_rows_fit(self, tmp_path: Path) -> None:
        """A realistic multi-row table also honors the width cap.

        :param tmp_path: pytest tmp_path fixture.
        """
        groups = [
            _group(tool="prediction-request-reasoning", n_bets=120),
            _group(tool="prediction-online", mode="tournament", n_bets=40),
        ]
        path = _write_results(tmp_path, groups)
        section = build_roi_section(path, "omen")
        assert section is not None
        for line in _code_block_lines(section):
            assert len(line) <= MAX_LINE_WIDTH, f"too wide: {line!r}"


# ---------------------------------------------------------------------------
# notify_slack separate-message hook
# ---------------------------------------------------------------------------


class TestNotifySlackHook:
    """notify_slack posts the ROI section as its own best-effort message."""

    @staticmethod
    def _run_main(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        extra_argv: list[str] | None = None,
        roi_env: str | None = None,
        post_stub: Optional[Callable[[str, str], None]] = None,
    ) -> list[str]:
        """Drive notify_slack.main with network + LLM stubbed; return posts.

        :param monkeypatch: pytest monkeypatch fixture.
        :param tmp_path: pytest tmp_path fixture.
        :param extra_argv: extra CLI arguments appended after --report.
        :param roi_env: value for the ROI_SECTION env var, or None to leave it unset.
        :param post_stub: optional post_to_slack replacement (url, text) -> None;
            defaults to one that records the posted text.
        :return: texts posted to the Slack webhook stub.
        """
        report = tmp_path / "report_omen.md"
        report.write_text(
            "# Benchmark Report (Omenstrat) — 2026-07-06\n\nbody\n",
            encoding="utf-8",
        )
        for var in (
            "REPORT_ARTIFACT_URL",
            "GITHUB_SERVER_URL",
            "GITHUB_REPOSITORY",
            "GITHUB_RUN_ID",
            "ROI_SECTION",
        ):
            monkeypatch.delenv(var, raising=False)
        if roi_env is not None:
            monkeypatch.setenv("ROI_SECTION", roi_env)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/T000")
        posted: list[str] = []

        def _record(url: str, text: str) -> None:
            """Record every posted text, then delegate to ``post_stub`` if given."""
            posted.append(text)
            if post_stub is not None:
                post_stub(url, text)

        monkeypatch.setattr(
            notify_slack, "summarize_report", lambda *a, **k: "LLM SUMMARY"
        )
        monkeypatch.setattr(notify_slack, "post_to_slack", _record)
        argv = ["notify_slack", "--report", str(report)] + (extra_argv or [])
        monkeypatch.setattr(sys, "argv", argv)
        notify_slack.main()
        return posted

    def test_section_posted_as_separate_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The ROI section is a second, standalone message after the digest.

        Posting it separately keeps its fenced code block from being split by
        Slack's long-text rendering when the digest is large.

        :param monkeypatch: pytest monkeypatch fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        calls: list[tuple[Path, str]] = []

        def _fake_builder(results_path: Path, platform: str) -> str:
            calls.append((results_path, platform))
            return "ROI_MARKER_SECTION"

        monkeypatch.setattr(notify_slack, "build_roi_section", _fake_builder)
        posted = self._run_main(monkeypatch, tmp_path)
        assert len(posted) == 2
        assert "LLM SUMMARY" in posted[0]
        assert "ROI_MARKER_SECTION" not in posted[0]
        assert posted[1] == "ROI_MARKER_SECTION"
        # Platform key derived from the report's platform label.
        assert calls == [(Path("benchmark/results/roi_results.json"), "omen")]

    def test_dry_run_prints_digest_and_roi_section(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--dry-run prints both the digest and the ROI section and posts nothing.

        :param monkeypatch: pytest monkeypatch fixture.
        :param tmp_path: pytest tmp_path fixture.
        :param capsys: pytest capsys fixture.
        """
        monkeypatch.setattr(
            notify_slack, "build_roi_section", lambda *a, **k: "ROI_MARKER_SECTION"
        )
        posted = self._run_main(monkeypatch, tmp_path, extra_argv=["--dry-run"])
        out = capsys.readouterr().out
        assert "LLM SUMMARY" in out
        assert "ROI_MARKER_SECTION" in out
        assert not posted  # dry-run must never call post_to_slack

    def test_roi_post_failure_never_breaks_digest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failure posting the ROI message leaves the digest post intact.

        :param monkeypatch: pytest monkeypatch fixture.
        :param tmp_path: pytest tmp_path fixture.
        :param caplog: pytest caplog fixture.
        """
        monkeypatch.setattr(
            notify_slack, "build_roi_section", lambda *a, **k: "ROI_MARKER_SECTION"
        )
        posted: list[str] = []

        def _post(url: str, text: str) -> None:
            posted.append(text)
            if text == "ROI_MARKER_SECTION":
                raise RuntimeError("slack down")

        with caplog.at_level(logging.WARNING, logger="benchmark.notify_slack"):
            self._run_main(monkeypatch, tmp_path, post_stub=_post)
        assert "LLM SUMMARY" in posted[0]
        assert posted[1] == "ROI_MARKER_SECTION"
        assert "Posting the ROI section failed" in caplog.text

    def test_roi_results_flag_passed_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--roi-results overrides the default results path.

        :param monkeypatch: pytest monkeypatch fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        calls: list[Path] = []
        monkeypatch.setattr(
            notify_slack,
            "build_roi_section",
            lambda results_path, platform: calls.append(results_path),
        )
        custom = tmp_path / "custom_roi.json"
        self._run_main(monkeypatch, tmp_path, ["--roi-results", str(custom)])
        assert calls == [custom]

    def test_builder_none_appends_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A None from the builder leaves the summary untouched.

        :param monkeypatch: pytest monkeypatch fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        monkeypatch.setattr(notify_slack, "build_roi_section", lambda *a, **k: None)
        posted = self._run_main(monkeypatch, tmp_path)
        assert len(posted) == 1
        assert posted[0].rstrip().endswith("LLM SUMMARY")

    def test_builder_exception_never_breaks_post(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An exploding builder logs a warning; the post still goes out.

        :param monkeypatch: pytest monkeypatch fixture.
        :param tmp_path: pytest tmp_path fixture.
        :param caplog: pytest caplog fixture.
        """

        def _boom(results_path: Path, platform: str) -> str:
            raise RuntimeError("boom")

        monkeypatch.setattr(notify_slack, "build_roi_section", _boom)
        with caplog.at_level(logging.WARNING, logger="benchmark.notify_slack"):
            posted = self._run_main(monkeypatch, tmp_path)
        assert len(posted) == 1
        assert "LLM SUMMARY" in posted[0]
        assert "ROI section build failed" in caplog.text

    def test_env_var_off_disables_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ROI_SECTION=off skips the builder entirely.

        :param monkeypatch: pytest monkeypatch fixture.
        :param tmp_path: pytest tmp_path fixture.
        """

        def _must_not_run(results_path: Path, platform: str) -> str:
            raise AssertionError("builder must not be called when ROI_SECTION=off")

        monkeypatch.setattr(notify_slack, "build_roi_section", _must_not_run)
        posted = self._run_main(monkeypatch, tmp_path, roi_env="off")
        assert len(posted) == 1
        assert "LLM SUMMARY" in posted[0]


# ---------------------------------------------------------------------------
# _render_table structural guards
# ---------------------------------------------------------------------------


class TestRenderTableGuards:
    """_render_table protects against empty input and column drift."""

    def test_empty_rows_returns_no_lines(self) -> None:
        """No rows -> no lines (never a max() on an empty header width)."""
        lines = _render_table([])
        assert isinstance(lines, list) and not lines

    def test_full_width_row_renders(self) -> None:
        """A correctly-sized row produces header + divider + one data line."""
        row = tuple(f"c{i}" for i in range(len(_HEADERS)))
        lines = _render_table([row])
        assert len(lines) == 3

    def test_short_row_raises_not_silently_truncates(self) -> None:
        """A row with fewer cells than _HEADERS raises rather than dropping a column."""
        short = tuple(f"c{i}" for i in range(len(_HEADERS) - 1))
        with pytest.raises(AssertionError):
            _render_table([short])

    def test_long_row_raises(self) -> None:
        """A row with more cells than _HEADERS also raises."""
        long_row = tuple(f"c{i}" for i in range(len(_HEADERS) + 1))
        with pytest.raises(AssertionError):
            _render_table([long_row])


# ---------------------------------------------------------------------------
# Freshness + malformed-payload robustness
# ---------------------------------------------------------------------------


class TestRoiSectionFreshness:
    """Stale results are flagged; malformed payloads are logged, not silent."""

    def test_fresh_results_have_no_stale_marker(self, tmp_path: Path) -> None:
        """A results file dated today renders without the stale marker."""
        section = build_roi_section(_write_results(tmp_path, [_group()]), "omen")
        assert section is not None
        assert "stale" not in section

    def test_old_results_are_marked_stale(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Results older than the cadence get a visible marker + a WARNING."""
        path = _write_results(tmp_path, [_group()], as_of="2020-01-01")
        with caplog.at_level(logging.WARNING):
            section = build_roi_section(path, "omen")
        assert section is not None
        assert "*(stale — as of 2020-01-01)*" in section
        assert "days old" in caplog.text

    def test_malformed_groups_warns_and_skips(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A payload whose 'groups' is not a list is logged, not silently dropped."""
        path = tmp_path / "roi_results.json"
        path.write_text(json.dumps({"groups": {"oops": 1}}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            assert build_roi_section(path, "omen") is None
        assert "no 'groups' list" in caplog.text

    def test_load_results_quiet_on_missing_warns_on_broken(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing file stays silent (no data yet); a corrupt file warns."""
        with caplog.at_level(logging.WARNING):
            assert _load_results(tmp_path / "absent.json") is None
        assert not caplog.records

        corrupt = tmp_path / "roi_results.json"
        corrupt.write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            assert _load_results(corrupt) is None
        assert "unreadable" in caplog.text

    def test_garbled_timestamp_warns_and_skips_marker(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A present-but-unparseable as_of logs a WARNING and skips the marker."""
        for bad in ("not-a-date", 12345):
            path = tmp_path / "roi_results.json"
            path.write_text(
                json.dumps({"window_days": 90, "as_of": bad, "groups": [_group()]}),
                encoding="utf-8",
            )
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                section = build_roi_section(path, "omen")
            assert section is not None and "stale" not in section
            assert "unparseable 'as_of'" in caplog.text

    def test_absent_timestamp_is_quiet(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A legacy payload with no as_of skips the check silently (age unknown)."""
        path = tmp_path / "roi_results.json"
        path.write_text(
            json.dumps({"window_days": 90, "groups": [_group()]}), encoding="utf-8"
        )
        with caplog.at_level(logging.WARNING):
            section = build_roi_section(path, "omen")
        assert section is not None and "stale" not in section
        assert not caplog.records

    def test_future_dated_results_are_not_stale(self, tmp_path: Path) -> None:
        """A future as_of (negative age) is never marked stale."""
        future = (datetime.now(timezone.utc) + timedelta(days=5)).date().isoformat()
        section = build_roi_section(
            _write_results(tmp_path, [_group()], as_of=future), "omen"
        )
        assert section is not None and "stale" not in section

    def test_stale_boundary_is_strict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Age == STALE_AFTER_DAYS is not stale (`>` is strict); just over is."""
        frozen = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        fake_dt = MagicMock(wraps=datetime)
        fake_dt.now.return_value = frozen
        monkeypatch.setattr(roi_slack, "datetime", fake_dt)
        at = (frozen - timedelta(days=STALE_AFTER_DAYS)).isoformat()
        over = (frozen - timedelta(days=STALE_AFTER_DAYS, seconds=1)).isoformat()
        at_section = build_roi_section(
            _write_results(tmp_path, [_group()], as_of=at), "omen"
        )
        over_section = build_roi_section(
            _write_results(tmp_path, [_group()], as_of=over), "omen"
        )
        assert at_section is not None and "stale" not in at_section
        assert over_section is not None and "stale" in over_section
