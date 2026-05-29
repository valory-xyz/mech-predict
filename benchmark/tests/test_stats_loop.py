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

from benchmark.stats_loop.triage import (TriageOutcome, triage_tools,
                                         write_state)


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
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.215, log_loss=0.620),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.200, log_loss=0.610),
            }
        )
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        decisions = triage_tools(cur_p, prev_p, tmp_path / "no_state.json")
        # Delta is exactly the threshold; the gate uses > not >=, so silent.
        assert decisions[0].today.flagged is False
        assert decisions[0].today.reason == "no_regression"


class TestSampleFloorGate:
    """Both windows must meet the per-day approximation N_DAY_FLOOR_PER_WINDOW_DAY."""

    def test_below_floor_silences(self, tmp_path: Path) -> None:
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(n=40, valid_n=40, brier=0.30, log_loss=0.80),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.20, log_loss=0.60),
            }
        )
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        decisions = triage_tools(cur_p, prev_p, tmp_path / "no_state.json")
        assert decisions[0].today.flagged is False
        assert decisions[0].today.reason == "sample_floor"

    def test_at_floor_evaluates(self, tmp_path: Path) -> None:
        # Per-day approximation: valid_n / 7 >= 15 -> valid_n >= 105.
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(n=105, valid_n=105, brier=0.30, log_loss=0.80),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(n=105, valid_n=105, brier=0.20, log_loss=0.60),
            }
        )
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        yesterday = _make_yesterday_state_path(tmp_path, ["tool_a"])
        decisions = triage_tools(cur_p, prev_p, yesterday)
        assert decisions[0].decision == "open_issue"


class TestSignAgreementGate:
    """Brier and log_loss must agree on the direction of the regression."""

    def test_log_loss_moves_opposite_direction_silences(self, tmp_path: Path) -> None:
        # Brier worsens, log_loss improves -> disagreement -> silent.
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.580),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
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
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700, reliability=0.65),
            }
        )
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
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700, reliability=0.50),
            }
        )
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
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        # Yesterday's state file does not exist (or has no entry for tool_a).
        decisions = triage_tools(cur_p, prev_p, tmp_path / "no_state.json")
        assert decisions[0].decision == "silent"
        # The outcome is still flagged=True today (so tomorrow can confirm).
        assert decisions[0].today.flagged is True

    def test_yesterday_silent_today_flag_is_silent(self, tmp_path: Path) -> None:
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        # Yesterday's state explicitly records flagged=False for tool_a.
        state_path = tmp_path / "stats_loop_state.json"
        state_path.write_text(
            json.dumps(
                {
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
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        decisions = triage_tools(cur_p, prev_p, state_path)
        assert decisions[0].decision == "silent"

    def test_two_day_confirmation_opens_issue(self, tmp_path: Path) -> None:
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
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
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
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
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
                "tool_b": _tool_stats(brier=0.200, log_loss=0.600),
                "tool_c": _tool_stats(brier=0.250, log_loss=0.580),  # sign disagree
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
                "tool_b": _tool_stats(brier=0.210, log_loss=0.610),
                "tool_c": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
        cur_p, prev_p = tmp_path / "cur.json", tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        yesterday = _make_yesterday_state_path(tmp_path, ["tool_a", "tool_b", "tool_c"])
        decisions = {d.tool_name: d for d in triage_tools(cur_p, prev_p, yesterday)}
        assert decisions["tool_a"].decision == "open_issue"
        assert decisions["tool_b"].decision == "silent"  # no regression
        assert decisions["tool_c"].decision == "silent"  # sign disagreement


class TestDuplicateIssueSuppression:
    """H1: a sustained regression must not file a new issue every day.

    Once gate 6 confirms and an issue opens, ``today.issue_open`` is
    persisted to state. Tomorrow, even though the gates still fire,
    the dispatcher honors the persisted flag and stays silent.
    """

    def _persistent_regression(self, tmp_path: Path) -> tuple[Path, Path]:
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
        cur_p = tmp_path / "cur.json"
        prev_p = tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        return cur_p, prev_p

    def test_day_2_opens_issue_first_time(self, tmp_path: Path) -> None:
        cur_p, prev_p = self._persistent_regression(tmp_path)
        # Yesterday flagged but did not file an issue (no issue_open).
        yesterday = _make_yesterday_state_path(tmp_path, ["tool_a"])
        decisions = triage_tools(cur_p, prev_p, yesterday)
        assert decisions[0].decision == "open_issue"
        assert decisions[0].today.issue_open is True

    def test_day_3_stays_silent_after_day_2_opened(self, tmp_path: Path) -> None:
        cur_p, prev_p = self._persistent_regression(tmp_path)
        # Yesterday flagged AND already filed an issue.
        state_path = tmp_path / "stats_loop_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "generated_at": "2026-05-28T03:45:00Z",
                    "platform": "polymarket",
                    "by_tool": {
                        "tool_a": {
                            "tool_name": "tool_a",
                            "flagged": True,
                            "delta_brier": 0.030,
                            "delta_log_loss": 0.090,
                            "brier_cur": 0.240,
                            "brier_prev": 0.210,
                            "log_loss_cur": 0.700,
                            "log_loss_prev": 0.610,
                            "n_cur": 200,
                            "n_prev": 200,
                            "reliability_cur": 0.99,
                            "reason": "all_gates_pass",
                            "issue_open": True,
                        }
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        decisions = triage_tools(cur_p, prev_p, state_path)
        # Same gates still fire, but a new issue is NOT opened.
        assert decisions[0].decision == "silent"
        # The issue_open flag propagates forward so day 4 also stays silent.
        assert decisions[0].today.issue_open is True

    def test_regression_resolves_clears_issue_open(self, tmp_path: Path) -> None:
        # Brier improved today; gate 2 no longer fires -> today.flagged
        # is False -> today.issue_open should be False on the persisted
        # outcome (the regression has resolved, future day's confirmation
        # will start from scratch).
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.205, log_loss=0.600),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
        cur_p = tmp_path / "cur.json"
        prev_p = tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        state_path = tmp_path / "stats_loop_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "by_tool": {
                        "tool_a": {
                            "tool_name": "tool_a",
                            "flagged": True,
                            "issue_open": True,
                        }
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        decisions = triage_tools(cur_p, prev_p, state_path)
        assert decisions[0].decision == "silent"
        assert decisions[0].today.flagged is False
        # New default: issue_open starts False on a fresh outcome.
        assert decisions[0].today.issue_open is False


class TestSchemaTolerantDeserialization:
    """H2: a schema-drifted state file must not crash triage."""

    def test_unknown_field_in_yesterday_is_ignored(self, tmp_path: Path) -> None:
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
        cur_p = tmp_path / "cur.json"
        prev_p = tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        state_path = tmp_path / "stats_loop_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "by_tool": {
                        "tool_a": {
                            "tool_name": "tool_a",
                            "flagged": True,
                            # Field that does not exist on TriageOutcome:
                            "some_future_field": "should_be_dropped",
                        }
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Must not raise; should treat yesterday as flagged-but-no-issue
        # and produce a normal open_issue decision today.
        decisions = triage_tools(cur_p, prev_p, state_path)
        assert decisions[0].decision == "open_issue"

    def test_completely_missing_fields_falls_back_to_defaults(
        self, tmp_path: Path
    ) -> None:
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
        cur_p = tmp_path / "cur.json"
        prev_p = tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        state_path = tmp_path / "stats_loop_state.json"
        # Yesterday's payload is minimal -- only tool_name and flagged.
        # Every other field falls back to TriageOutcome defaults.
        state_path.write_text(
            json.dumps(
                {
                    "by_tool": {
                        "tool_a": {
                            "tool_name": "tool_a",
                            "flagged": True,
                        }
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        decisions = triage_tools(cur_p, prev_p, state_path)
        assert decisions[0].decision == "open_issue"
        assert decisions[0].yesterday is not None
        # Defaults from the schema-tolerant adapter:
        assert decisions[0].yesterday.brier_cur is None
        assert decisions[0].yesterday.n_cur == 0

    def test_TriageOutcome_from_dict_with_non_dict_returns_default(self) -> None:
        out = TriageOutcome.from_dict("not a dict")  # type: ignore[arg-type]
        assert out == TriageOutcome()

    def test_TriageOutcome_from_dict_with_bad_type_does_not_raise(self) -> None:
        # Dataclasses don't type-check at construction; passing a list
        # where int is expected succeeds. The H2 guarantee is "do not
        # crash"; pedantic type checking is intentionally out of scope.
        # The next consumer downstream (e.g. _safe_get) is responsible
        # for narrowing -- here we just assert no exception.
        out = TriageOutcome.from_dict({"tool_name": "t", "n_cur": []})
        assert out.tool_name == "t"


class TestCorruptStateFile:
    """M3: corrupt/truncated state file must not wedge the loop."""

    def test_truncated_state_treated_as_no_yesterday(self, tmp_path: Path) -> None:
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.240, log_loss=0.700),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
        cur_p = tmp_path / "cur.json"
        prev_p = tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        state_path = tmp_path / "stats_loop_state.json"
        # Truncated JSON (process killed mid-write).
        state_path.write_text('{"by_tool": {"too', encoding="utf-8")
        # Must not raise; falls back to first-day-silent behavior.
        decisions = triage_tools(cur_p, prev_p, state_path)
        assert decisions[0].decision == "silent"
        assert decisions[0].yesterday is None


class TestValidNStrictness:
    """M5: valid_n must NOT fall back to total n."""

    def test_zero_valid_n_with_nonzero_n_fails_sample_floor(
        self, tmp_path: Path
    ) -> None:
        # 200 rows but 0 valid (full parse failure) must NOT pass the
        # sample floor. The original `valid_n or n or 0` would have
        # incorrectly fallen back to n=200 and let the tool through.
        cur = _scores_doc(
            {
                "tool_a": _tool_stats(
                    n=200,
                    valid_n=0,
                    brier=0.240,
                    log_loss=0.700,
                ),
            }
        )
        prev = _scores_doc(
            {
                "tool_a": _tool_stats(brier=0.210, log_loss=0.610),
            }
        )
        cur_p = tmp_path / "cur.json"
        prev_p = tmp_path / "prev.json"
        _write(cur_p, cur)
        _write(prev_p, prev)
        decisions = triage_tools(cur_p, prev_p, tmp_path / "no_state.json")
        assert decisions[0].today.flagged is False
        assert decisions[0].today.reason == "sample_floor"


class TestOpenIssueModule:
    """M1: coverage for the dispatcher functions in open_issue.py."""

    def _decision(self) -> "TriageDecision":  # noqa: F821 - imported below
        from benchmark.stats_loop.triage import TriageDecision, TriageOutcome

        outcome = TriageOutcome(
            tool_name="factual_research",
            flagged=True,
            delta_brier=0.024,
            delta_log_loss=0.060,
            brier_cur=0.243,
            brier_prev=0.219,
            log_loss_cur=0.711,
            log_loss_prev=0.651,
            n_cur=187,
            n_prev=212,
            reliability_cur=0.99,
            reason="all_gates_pass",
            issue_open=True,
        )
        return TriageDecision(
            tool_name="factual_research",
            decision="open_issue",
            today=outcome,
            yesterday=outcome,
        )

    def test_build_issue_title_format(self) -> None:
        from benchmark.stats_loop.open_issue import build_issue_title

        title = build_issue_title(self._decision())
        # The agent's pipeline parses this format; it is a contract.
        assert title == (
            "[tool-improvement] `factual_research`: "
            "Brier regression on polymarket W-1"
        )

    def test_build_issue_body_includes_headline_and_artifact_url(
        self,
    ) -> None:
        from benchmark.stats_loop.open_issue import (_window_iso_from_args,
                                                     build_issue_body)

        body = build_issue_body(
            decision=self._decision(),
            polymarket_stats={"brier": 0.243, "n": 187},
            combined_stats={"brier": 0.230, "n": 412},
            artifact_url="https://example.test/artifacts/42",
            window_iso=_window_iso_from_args(
                __import__("datetime").datetime(
                    2026,
                    5,
                    28,
                    3,
                    45,
                    0,
                    tzinfo=__import__("datetime").timezone.utc,
                )
            ),
        )
        assert "factual_research" in body
        assert "0.2430" in body  # brier_cur formatted
        assert "0.2190" in body  # brier_prev formatted
        assert "+0.0240" in body  # delta formatted
        assert "https://example.test/artifacts/42" in body
        assert "baseline-stats-polymarket" in body
        # The combined block is included for cross-reference but the
        # body must say polymarket-only.
        assert "polymarket-only" in body

    def test_build_issue_body_asserts_on_missing_brier(self) -> None:
        # L2: a None brier in the open_issue path must fail loudly.
        from benchmark.stats_loop.open_issue import build_issue_body
        from benchmark.stats_loop.triage import TriageDecision, TriageOutcome

        bad_outcome = TriageOutcome(
            tool_name="tool_a",
            flagged=True,
            delta_brier=None,  # would have crashed at format time
            brier_cur=None,
            brier_prev=None,
        )
        decision = TriageDecision(
            tool_name="tool_a",
            decision="open_issue",
            today=bad_outcome,
        )
        with pytest.raises(AssertionError):
            build_issue_body(
                decision=decision,
                polymarket_stats={},
                combined_stats={},
                artifact_url="x",
                window_iso={
                    "w1_start": "a",
                    "w1_end": "b",
                    "w2_start": "c",
                    "w2_end": "d",
                },
            )

    def test_artifact_url_uses_provided_repo(self) -> None:
        # M2: the URL must point at args.repo, not the hardcoded default.
        from benchmark.stats_loop.open_issue import _artifact_url

        url = _artifact_url("acme/fork", "12345")
        assert "acme/fork" in url
        assert "12345" in url
        assert "valory-xyz/mech-predict" not in url

    def test_safe_load_json_returns_empty_for_missing(self, tmp_path: Path) -> None:
        from benchmark.stats_loop.open_issue import _safe_load_json

        assert _safe_load_json(tmp_path / "missing.json") == {}

    def test_safe_load_json_returns_empty_for_corrupt(self, tmp_path: Path) -> None:
        from benchmark.stats_loop.open_issue import _safe_load_json

        p = tmp_path / "broken.json"
        p.write_text("{not valid json", encoding="utf-8")
        # Must not raise; returns empty dict.
        assert _safe_load_json(p) == {}

    def test_window_iso_w2_precedes_w1(self) -> None:
        # L1 regression catcher: w2_end must equal w1_start and w2 must
        # precede w1 (non-overlapping windows).
        from benchmark.stats_loop.open_issue import _window_iso_from_args

        now = __import__("datetime").datetime(
            2026,
            5,
            28,
            3,
            45,
            0,
            tzinfo=__import__("datetime").timezone.utc,
        )
        iso = _window_iso_from_args(now, days=7)
        assert iso["w1_end"] == "2026-05-28T03:45:00Z"
        assert iso["w1_start"] == "2026-05-21T03:45:00Z"
        assert iso["w2_end"] == iso["w1_start"]
        assert iso["w2_start"] == "2026-05-14T03:45:00Z"

    def test_open_github_issue_dry_run_returns_zero(self) -> None:
        from benchmark.stats_loop.open_issue import open_github_issue

        rc, url = open_github_issue(
            repo="acme/test",
            label="tool-improvement",
            title="title",
            body="body",
            dry_run=True,
        )
        assert rc == 0
        assert url is None

    def test_open_github_issue_returns_failure_on_subprocess_error(
        self, monkeypatch
    ) -> None:
        # H3: when gh fails, the function returns non-zero AND no URL.
        # The caller (main) must propagate that to a non-zero exit.
        from benchmark.stats_loop import open_issue

        class _FakeResult:
            def __init__(self) -> None:
                self.returncode = 1
                self.stdout = ""
                self.stderr = "gh: token expired"

        monkeypatch.setattr(
            open_issue.subprocess, "run", lambda *_a, **_kw: _FakeResult()
        )
        rc, url = open_issue.open_github_issue(
            repo="acme/test",
            label="tool-improvement",
            title="title",
            body="body",
            dry_run=False,
        )
        assert rc == 1
        assert url is None
