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
from pathlib import Path
from typing import Any

import pytest
from benchmark import notify_slack
from benchmark.roi_slack import (
    MAX_LINE_WIDTH,
    MAX_TABLE_ROWS,
    build_roi_section,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group(
    tool: str = "test-tool",
    platform: str = "omen",
    mode: str = "production",
    n_eligible: int = 40,
    n_bets: int = 20,
    staked: float = 25.0,
    roi_mid: Any = 12.3,
    roi_ci: Any = None,
    roi_haircut: Any = 8.4,
    roi_haircut_ci: Any = None,
    flags: Any = None,
    is_prediction_tool: bool = True,
    parse_reliability: Any = 1.0,
) -> dict[str, Any]:
    """Build a minimal roi_results.json group entry."""
    return {
        "platform": platform,
        "tool_name": tool,
        "mode": mode,
        "n_eligible": n_eligible,
        "n_bets": n_bets,
        "staked": staked,
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


def _write_results(
    tmp_path: Path,
    groups: list[dict[str, Any]],
    window_days: int = 90,
) -> Path:
    """Write a roi_results.json fixture and return its path."""
    path = tmp_path / "roi_results.json"
    payload = {
        "as_of": "2026-07-06",
        "window_days": window_days,
        "window_start": "2026-04-08T00:00:00+00:00",
        "window_end": "2026-07-07T00:00:00+00:00",
        "groups": groups,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _code_block_lines(section: str) -> list[str]:
    """Extract the lines inside the fenced code block of a section."""
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
        """First line is the bold intro citing the window from the json."""
        path = _write_results(tmp_path, [_group()], window_days=21)
        section = build_roi_section(path, "omen")
        assert section is not None
        first = section.splitlines()[0]
        assert first.startswith("*Simulated trader ROI*")
        assert "trailing 21d" in first

    def test_code_block_carries_all_columns(self, tmp_path: Path) -> None:
        """Header row inside the code block names every column."""
        path = _write_results(tmp_path, [_group()])
        section = build_roi_section(path, "omen")
        assert section is not None
        header = _code_block_lines(section)[0]
        for column in (
            "tool",
            "mode",
            "preds",
            "bets",
            "ROI (95% CI)",
            "w/costs",
            "flags",
        ):
            assert column in header, f"missing column: {column}"

    def test_row_renders_counts_roi_and_ci(self, tmp_path: Path) -> None:
        """A data row shows n_eligible, n_bets, and the ROI with its CI."""
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

    def test_production_sorts_before_tournament(self, tmp_path: Path) -> None:
        """Production rows precede tournament rows regardless of bet count."""
        path = _write_results(
            tmp_path,
            [
                _group(tool="t-big", mode="tournament", n_bets=99),
                _group(tool="p-small", mode="production", n_bets=5),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "p-small" in rows[0]
        assert "t-big" in rows[1]

    def test_within_mode_sorted_by_bets_desc(self, tmp_path: Path) -> None:
        """Within one mode, more bets rank higher."""
        path = _write_results(
            tmp_path,
            [
                _group(tool="few", n_bets=3),
                _group(tool="many", n_bets=30),
            ],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "many" in rows[0]
        assert "few" in rows[1]

    def test_zero_bet_groups_stay_out_of_table(self, tmp_path: Path) -> None:
        """Groups with n_bets == 0 never occupy a table row."""
        path = _write_results(
            tmp_path,
            [_group(tool="bettor", n_bets=10), _group(tool="idle", n_bets=0)],
        )
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)
        assert not any("idle" in row for row in rows)

    def test_other_platform_groups_ignored(self, tmp_path: Path) -> None:
        """Only the requested platform's groups render."""
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
        """Verbose flags render as their short spellings in the flags cell."""
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
        """The parse-reliability warning compacts to '⚠ parse NN%'."""
        long_flag = "⚠ 45% parse reliability — possible response-format gap"
        path = _write_results(tmp_path, [_group(flags=[long_flag])])
        section = build_roi_section(path, "omen")
        assert section is not None
        rows = _code_block_lines(section)[2:]
        assert "⚠ parse 45%" in rows[0]
        assert "response-format" not in rows[0]

    def test_mode_labels_abbreviated(self, tmp_path: Path) -> None:
        """Modes render as 'prod' / 'tourn' to keep the table narrow."""
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


# ---------------------------------------------------------------------------
# Zero-bet and excluded lines
# ---------------------------------------------------------------------------


class TestZeroBetLine:
    """Prediction tools with zero bets are listed on one compact line."""

    def test_zero_bet_tools_listed(self, tmp_path: Path) -> None:
        """Zero-bet prediction tools appear comma-joined after the block."""
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
        """No zero-bet line when every prediction tool placed bets."""
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


class TestExcludedLine:
    """Non-prediction tools are summarized as a count only."""

    def test_excluded_count_rendered(self, tmp_path: Path) -> None:
        """Count of distinct non-prediction tools, names omitted."""
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
        """No excluded line when every group is a prediction tool."""
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
        """Missing results file -> None."""
        assert build_roi_section(tmp_path / "absent.json", "omen") is None

    def test_unparseable_file(self, tmp_path: Path) -> None:
        """Corrupt JSON -> None."""
        path = tmp_path / "roi_results.json"
        path.write_text("{not json", encoding="utf-8")
        assert build_roi_section(path, "omen") is None

    def test_non_dict_payload(self, tmp_path: Path) -> None:
        """A JSON list payload -> None."""
        path = tmp_path / "roi_results.json"
        path.write_text("[1, 2]", encoding="utf-8")
        assert build_roi_section(path, "omen") is None

    def test_no_groups_for_platform(self, tmp_path: Path) -> None:
        """Groups exist but none on the requested platform -> None."""
        path = _write_results(tmp_path, [_group(platform="polymarket")])
        assert build_roi_section(path, "omen") is None

    def test_empty_groups(self, tmp_path: Path) -> None:
        """Empty groups list -> None."""
        path = _write_results(tmp_path, [])
        assert build_roi_section(path, "omen") is None


# ---------------------------------------------------------------------------
# Truncation + width bounds
# ---------------------------------------------------------------------------


class TestTruncation:
    """Wide deployments keep the top rows by staked and point at the report."""

    def test_more_than_max_rows_truncates(self, tmp_path: Path) -> None:
        """17 bet groups -> MAX_TABLE_ROWS rows + '+N more' line."""
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
        """The lowest-staked groups are the ones dropped."""
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
        """Exactly MAX_TABLE_ROWS groups render fully with no more-line."""
        groups = [_group(tool=f"tool-{i:02d}") for i in range(MAX_TABLE_ROWS)]
        path = _write_results(tmp_path, groups)
        section = build_roi_section(path, "omen")
        assert section is not None
        assert "more rows in the full report" not in section


class TestLineWidth:
    """Every code-block line stays within the Slack-friendly width cap."""

    def test_long_names_and_flags_fit(self, tmp_path: Path) -> None:
        """Very long tool names / flags are truncated, not overflowed."""
        long_flag = "⚠ 45% parse reliability — possible response-format gap"
        groups = [
            _group(
                tool="a-very-long-tool-name-that-exceeds-the-column-cap-x" * 2,
                n_eligible=99999,
                n_bets=88888,
                roi_mid=-100.0,
                roi_ci=[-1234.5, 6789.0],
                roi_haircut=-100.0,
                flags=[long_flag, "few bets - anecdotal", "low sample"],
            )
        ]
        path = _write_results(tmp_path, groups)
        section = build_roi_section(path, "omen")
        assert section is not None
        for line in _code_block_lines(section):
            assert len(line) <= MAX_LINE_WIDTH, f"too wide: {line!r}"

    def test_typical_rows_fit(self, tmp_path: Path) -> None:
        """A realistic multi-row table also honors the width cap."""
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
# notify_slack append hook
# ---------------------------------------------------------------------------


class TestNotifySlackHook:
    """notify_slack appends the ROI section without ever breaking the post."""

    @staticmethod
    def _run_main(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        extra_argv: list[str] | None = None,
        roi_env: str | None = None,
    ) -> list[str]:
        """Drive notify_slack.main with network + LLM stubbed; return posts."""
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
        monkeypatch.setattr(
            notify_slack, "summarize_report", lambda *a, **k: "LLM SUMMARY"
        )
        monkeypatch.setattr(
            notify_slack, "post_to_slack", lambda url, text: posted.append(text)
        )
        argv = ["notify_slack", "--report", str(report)] + (extra_argv or [])
        monkeypatch.setattr(sys, "argv", argv)
        notify_slack.main()
        return posted

    def test_section_appended_after_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The builder's text lands at the end of the posted message."""
        calls: list[tuple[Path, str]] = []

        def _fake_builder(results_path: Path, platform: str) -> str:
            calls.append((results_path, platform))
            return "ROI_MARKER_SECTION"

        monkeypatch.setattr(notify_slack, "build_roi_section", _fake_builder)
        posted = self._run_main(monkeypatch, tmp_path)
        assert len(posted) == 1
        assert posted[0].index("LLM SUMMARY") < posted[0].index("ROI_MARKER_SECTION")
        assert posted[0].endswith("ROI_MARKER_SECTION")
        # Platform key derived from the report's platform label.
        assert calls == [(Path("benchmark/results/roi_results.json"), "omen")]

    def test_roi_results_flag_passed_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--roi-results overrides the default results path."""
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
        """A None from the builder leaves the summary untouched."""
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
        """An exploding builder logs a warning; the post still goes out."""

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
        """ROI_SECTION=off skips the builder entirely."""

        def _must_not_run(results_path: Path, platform: str) -> str:
            raise AssertionError("builder must not be called when ROI_SECTION=off")

        monkeypatch.setattr(notify_slack, "build_roi_section", _must_not_run)
        posted = self._run_main(monkeypatch, tmp_path, roi_env="off")
        assert len(posted) == 1
        assert "LLM SUMMARY" in posted[0]
