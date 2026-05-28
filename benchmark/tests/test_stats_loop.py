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
"""Tests for benchmark/stats_loop/triage.py.

Covers each gate in the cascade:

- Polymarket-only scope (Omen rows are not even loaded by this module;
  the caller selects the per-platform sibling file).
- Regression size threshold (0.015).
- Sample-floor gate.
- Sign-agreement gate (Brier and log_loss must move in the same
  direction).
- Reliability-collapse short-circuit.
- Two-day confirmation gate.
- write_state round-trips the data triage_tools needs from a prior run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from benchmark.stats_loop.triage import (
    BRIER_REGRESSION_THRESHOLD,
    N_WINDOW_FLOOR,
    RELIABILITY_FLOOR,
    TriageOutcome,
    triage_tools,
    write_state,
)


def _scores_doc(by_tool: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build a minimal scores.json doc with the fields triage reads."""
    return {
        "generated_at": "2026-05-28T03:45:00Z",
        "total_rows": sum(t["n"] for t in by_tool.values()),
        "valid_rows": sum(t["valid_n"] for t in by_tool.values()),
        "by_tool": by_tool,
    }


def _tool_stats(
    *,
    n: int = 200,
    valid_n: int = 200,
    brier: float = 0.20,
    log_loss: float = 0.60,
    reliability: float = 0.98,
) -> Dict[str, Any]:
    """Build one row in by_tool with the fields triage actually reads."""
    return {
        "n": n,
        "valid_n": valid_n,
        "brier": brier,
        "log_loss": log_loss,
        "reliability": reliability,
    }


def _write(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _make_yesterday_state_path(tmp_path: Path, flagged_tools: list) -> Path:
    """Write a state file as if yesterday's triage had flagged these tools."""
    state = {
        "generated_at": "2026-05-27T03:45:00Z",
        "platform": "polymarket",
        "by_tool": {
            tool_name: {
                "tool_name": tool_name,
                "flagged": True,
                "delta_brier": 0.020,
                "delta_log_loss": 0.050,
                "brier_cur": 0.230,
                "brier_prev": 0.210,
                "log_loss_cur": 0.700,
                "log_loss_prev": 0.650,
                "n_cur": 200,
                "n_prev": 220,
                "reliability_cur": 0.99,
                "reason": "all_gates_pass",
            }
            for tool_name in flagged_tools
        },
    }
    p = tmp_path / "stats_loop_state.json"
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return p


class TestRegressionSizeGate:
    """Δ Brier > 0.015 is required."""

    def test_below_threshold_does_not_flag(self, tmp_path: Path) -> None:
        cur = _scores_doc({"tool_a": _tool_stats(brier=0.215, log_loss=0.620)})
        prev = _scores_doc({"tool_a": _tool_stats(brier=0.210, log_loss=0.610)})
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        yesterday = _make_yesterday_state_path(tmp_path, ["tool_a"])
        decisions = triage_tools(cur_p, prev_p, yesterday)
        assert decisions[0].decision == "silent"
        assert decisions[0].today.reason == "no_regression"

    def test_above_threshold_passes_size_gate(self, tmp_path: Path) -> None:
        cur = _scores_doc({"tool_a": _tool_stats(brier=0.240, log_loss=0.700)})
        prev = _scores_doc({"tool_a": _tool_stats(brier=0.210, log_loss=0.610)})
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        yesterday = _make_yesterday_state_path(tmp_path, ["tool_a"])
        decisions = triage_tools(cur_p, prev_p, yesterday)
        # With yesterday's confirmation present, all gates -> open_issue.
        assert decisions[0].decision == "open_issue"

    def test_threshold_boundary_does_not_flag(self, tmp_path: Path) -> None:
        # Delta exactly == threshold (0.015) must NOT pass the strict gate.
        cur = _scores_doc({
            "tool_a": _tool_stats(brier=0.215, log_loss=0.620),
        })
        prev = _scores_doc({
            "tool_a": _tool_stats(brier=0.200, log_loss=0.610),
        })
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        decisions = triage_tools(cur_p, prev_p, tmp_path / "no_state.json")
        # Delta is exactly the threshold; the gate uses > not >=, so silent.
        assert decisions[0].today.flagged is False
        assert decisions[0].today.reason == "no_regression"


class TestSampleFloorGate:
    """Both windows must have valid_n >= N_WINDOW_FLOOR."""

    def test_below_floor_silences(self, tmp_path: Path) -> None:
        cur = _scores_doc({
            "tool_a": _tool_stats(n=40, valid_n=40, brier=0.30, log_loss=0.80),
        })
        prev = _scores_doc({
            "tool_a": _tool_stats(brier=0.20, log_loss=0.60),
        })
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        decisions = triage_tools(cur_p, prev_p, tmp_path / "no_state.json")
        assert decisions[0].today.flagged is False
        assert decisions[0].today.reason == "sample_floor"

    def test_at_floor_evaluates(self, tmp_path: Path) -> None:
        # N_WINDOW_FLOOR is 60; with 7 days of data the per-day approximation
        # needs >= 30/2 = 15/day = 105 in the window. Use that floor.
        cur = _scores_doc({
            "tool_a": _tool_stats(
                n=105, valid_n=105, brier=0.30, log_loss=0.80
            ),
        })
        prev = _scores_doc({
            "tool_a": _tool_stats(
                n=105, valid_n=105, brier=0.20, log_loss=0.60
            ),
        })
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        yesterday = _make_yesterday_state_path(tmp_path, ["tool_a"])
        decisions = triage_tools(cur_p, prev_p, yesterday)
        assert decisions[0].decision == "open_issue"


class TestSignAgreementGate:
    """Brier and log_loss must agree on the direction of the regression."""

    def test_log_loss_moves_opposite_direction_silences(
        self, tmp_path: Path
    ) -> None:
        # Brier worsens, log_loss improves -> disagreement -> silent.
        cur = _scores_doc({
            "tool_a": _tool_stats(brier=0.240, log_loss=0.580),
        })
        prev = _scores_doc({
            "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
        })
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        decisions = triage_tools(cur_p, prev_p, tmp_path / "no_state.json")
        assert decisions[0].today.flagged is False
        assert decisions[0].today.reason == "sign_disagreement"


class TestReliabilityCollapse:
    """reliability < RELIABILITY_FLOOR short-circuits to a separate outcome."""

    def test_collapse_returns_reliability_collapse_decision(
        self, tmp_path: Path
    ) -> None:
        cur = _scores_doc({
            "tool_a": _tool_stats(
                brier=0.240, log_loss=0.700, reliability=0.65
            ),
        })
        prev = _scores_doc({"tool_a": _tool_stats()})
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        decisions = triage_tools(cur_p, prev_p, tmp_path / "no_state.json")
        assert decisions[0].decision == "reliability_collapse"
        assert decisions[0].today.reason == "reliability_collapse"
        assert decisions[0].today.reliability_cur == pytest.approx(0.65)

    def test_reliability_collapse_does_not_open_issue_even_with_yesterday(
        self, tmp_path: Path
    ) -> None:
        # Even if yesterday flagged the tool, today's reliability collapse
        # must short-circuit; we want a human-paging not an LLM patch.
        cur = _scores_doc({
            "tool_a": _tool_stats(
                brier=0.240, log_loss=0.700, reliability=0.50
            ),
        })
        prev = _scores_doc({"tool_a": _tool_stats()})
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        yesterday = _make_yesterday_state_path(tmp_path, ["tool_a"])
        decisions = triage_tools(cur_p, prev_p, yesterday)
        assert decisions[0].decision == "reliability_collapse"


class TestTwoDayConfirmationGate:
    """An issue opens only when today AND yesterday both flagged the tool."""

    def test_first_day_flag_is_silent(self, tmp_path: Path) -> None:
        cur = _scores_doc({
            "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
        })
        prev = _scores_doc({
            "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
        })
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        # Yesterday's state file does not exist (or has no entry for tool_a).
        decisions = triage_tools(cur_p, prev_p, tmp_path / "no_state.json")
        assert decisions[0].decision == "silent"
        # The outcome is still flagged=True today (so tomorrow can confirm).
        assert decisions[0].today.flagged is True

    def test_yesterday_silent_today_flag_is_silent(self, tmp_path: Path) -> None:
        cur = _scores_doc({
            "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
        })
        prev = _scores_doc({
            "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
        })
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        # Yesterday's state explicitly records flagged=False for tool_a.
        state_path = tmp_path / "stats_loop_state.json"
        state_path.write_text(
            json.dumps({
                "generated_at": "2026-05-27T03:45:00Z",
                "platform": "polymarket",
                "by_tool": {
                    "tool_a": {
                        "tool_name": "tool_a",
                        "flagged": False,
                        "delta_brier": 0.005,
                        "delta_log_loss": 0.010,
                        "brier_cur": 0.215,
                        "brier_prev": 0.210,
                        "log_loss_cur": 0.620,
                        "log_loss_prev": 0.610,
                        "n_cur": 200,
                        "n_prev": 200,
                        "reliability_cur": 0.99,
                        "reason": "no_regression",
                    }
                },
            }, indent=2),
            encoding="utf-8",
        )
        decisions = triage_tools(cur_p, prev_p, state_path)
        assert decisions[0].decision == "silent"

    def test_two_day_confirmation_opens_issue(self, tmp_path: Path) -> None:
        cur = _scores_doc({
            "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
        })
        prev = _scores_doc({
            "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
        })
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        yesterday = _make_yesterday_state_path(tmp_path, ["tool_a"])
        decisions = triage_tools(cur_p, prev_p, yesterday)
        assert decisions[0].decision == "open_issue"
        assert decisions[0].yesterday is not None
        assert decisions[0].yesterday.flagged is True


class TestStateRoundTrip:
    """write_state produces a file triage_tools can read for confirmation."""

    def test_write_then_read(self, tmp_path: Path) -> None:
        cur = _scores_doc({
            "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
        })
        prev = _scores_doc({
            "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
        })
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        # Day 1: no yesterday yet; first run -> silent.
        state_path = tmp_path / "stats_loop_state.json"
        day1 = triage_tools(cur_p, prev_p, state_path)
        assert day1[0].decision == "silent"
        write_state(state_path, day1, "2026-05-27T03:45:00Z")
        assert state_path.exists()
        # Day 2: yesterday now has flagged=True for tool_a (from write_state).
        # The same numbers today -> open_issue.
        day2 = triage_tools(cur_p, prev_p, state_path)
        assert day2[0].decision == "open_issue"


class TestMultipleTools:
    """When many tools are present, each is evaluated independently."""

    def test_one_flags_others_silent(self, tmp_path: Path) -> None:
        cur = _scores_doc({
            "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
            "tool_b": _tool_stats(brier=0.200, log_loss=0.600),
            "tool_c": _tool_stats(brier=0.250, log_loss=0.580),  # sign disagree
        })
        prev = _scores_doc({
            "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            "tool_b": _tool_stats(brier=0.210, log_loss=0.610),
            "tool_c": _tool_stats(brier=0.210, log_loss=0.610),
        })
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        yesterday = _make_yesterday_state_path(tmp_path, ["tool_a", "tool_b", "tool_c"])
        decisions = {d.tool_name: d for d in triage_tools(cur_p, prev_p, yesterday)}
        assert decisions["tool_a"].decision == "open_issue"
        assert decisions["tool_b"].decision == "silent"  # no regression
        assert decisions["tool_c"].decision == "silent"  # sign disagreement
