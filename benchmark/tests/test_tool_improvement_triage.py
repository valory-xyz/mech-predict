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
"""Tests for benchmark/tool_improvement_triage.py.

Covers the gate cascade, duplicate-issue suppression, the state
round-trip, and the issue body/title format the agent's parser
expects.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from benchmark.tool_improvement_triage import (
    BRIER_LEVEL_RE_ARM,
    BRIER_LEVEL_THRESHOLD,
    BRIER_REGRESSION_THRESHOLD,
    RELIABILITY_FLOOR,
    VALID_N_PER_WINDOW_FLOOR,
    _load_json,
    _open_issue,
    _open_issue_tools,
    _window_iso,
    build_issue_body,
    build_issue_title,
    main,
    triage,
    write_state,
)


def _stats(
    *,
    n: int = 200,
    valid_n: int = 200,
    brier: float = 0.20,
    log_loss: float = 0.60,
    reliability: float = 0.98,
) -> Dict[str, Any]:
    return {
        "n": n,
        "valid_n": valid_n,
        "brier": brier,
        "log_loss": log_loss,
        "reliability": reliability,
    }


def _scores(**by_tool: Dict[str, Any]) -> Dict[str, Any]:
    return {"by_tool": by_tool}


def _prior_state_with_issue(*tools: str) -> Dict[str, Any]:
    return {"by_tool": {t: {"issue_open": True} for t in tools}}


class TestRegressionSizeGate:
    """Delta Brier > 0.040 is required (calibrated 2026-05-29)."""

    def test_below_threshold_silent(self) -> None:
        """Delta Brier below the threshold leaves the tool silent."""
        d = triage(_scores(a=_stats(brier=0.185)), _scores(a=_stats(brier=0.160)), {})
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "no_regression"

    def test_above_threshold_opens(self) -> None:
        """Delta Brier above the threshold opens an issue."""
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            {},
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["issue_open"] is True

    def test_at_threshold_silent(self) -> None:
        """Delta exactly equal to the threshold is silent (gate uses > not >=)."""
        # Delta == threshold (0.040) must NOT pass (gate uses > not >=).
        # cur kept below the level floor (0.20) so level does not fire.
        d = triage(
            _scores(a=_stats(brier=0.180, log_loss=0.620)),
            _scores(a=_stats(brier=0.140, log_loss=0.610)),
            {},
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "no_regression"


class TestSampleFloor:
    """valid_n >= 105 required on both windows."""

    def test_below_floor_silent(self) -> None:
        """valid_n below 105 silences the tool."""
        d = triage(
            _scores(a=_stats(n=40, valid_n=40, brier=0.30)),
            _scores(a=_stats(brier=0.20)),
            {},
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "sample_floor"

    def test_at_floor_evaluates(self) -> None:
        """valid_n exactly 105 passes the sample-floor gate."""
        d = triage(
            _scores(a=_stats(n=105, valid_n=105, brier=0.30, log_loss=0.80)),
            _scores(a=_stats(n=105, valid_n=105, brier=0.20, log_loss=0.60)),
            {},
        )
        assert d[0]["decision"] == "open_issue"

    def test_valid_n_strict_no_n_fallback(self) -> None:
        """High total n with zero valid_n must fail the sample floor."""
        # 200 total rows but 0 valid (full parse failure) -> must NOT
        # fall back to total n; must fail sample_floor.
        d = triage(
            _scores(a=_stats(n=200, valid_n=0, brier=0.30)),
            _scores(a=_stats(brier=0.20)),
            {},
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "sample_floor"


class TestSignAgreement:
    """Brier and log_loss must agree on the worsening direction."""

    def test_log_loss_improves_silent(self) -> None:
        """Sign disagreement on a regression silences when level stays low."""
        # Brier worsens, log_loss improves -> regression-path sign
        # disagreement; level stays below 0.20 -> overall silent.
        d = triage(
            _scores(a=_stats(brier=0.190, log_loss=0.580)),
            _scores(a=_stats(brier=0.140, log_loss=0.610)),
            {},
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "sign_disagreement"


class TestReliabilityCollapse:
    """reliability < 0.80 routes to separate outcome."""

    def test_collapse(self) -> None:
        """Reliability below 0.80 routes to a separate outcome."""
        d = triage(
            _scores(a=_stats(brier=0.260, reliability=0.65)), _scores(a=_stats()), {}
        )
        assert d[0]["decision"] == "reliability_collapse"

    def test_collapse_short_circuits_even_with_prior_issue(self) -> None:
        """Reliability collapse takes precedence over a prior open issue."""
        # An open issue must NOT keep a tool in the regression path
        # once reliability collapses.
        prior = _prior_state_with_issue("a")
        d = triage(
            _scores(a=_stats(brier=0.260, reliability=0.50)), _scores(a=_stats()), prior
        )
        assert d[0]["decision"] == "reliability_collapse"


class TestDuplicateIssueSuppression:
    """Sustained regressions file at most one issue (silent until resolved)."""

    def test_day_1_opens_without_prior_state(self) -> None:
        """First-day regression with no prior state opens an issue."""
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            {},
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["issue_open"] is True

    def test_day_2_silent_when_issue_already_open(self) -> None:
        """A persisted open issue suppresses today's would-be dispatch."""
        prior = _prior_state_with_issue("a")
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            prior,
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "duplicate_suppressed"
        # issue_open propagates so tomorrow also stays silent.
        assert d[0]["issue_open"] is True

    def test_resolution_clears_issue_open(self) -> None:
        """Brier recovery clears issue_open so a future regression can fire."""
        # Today Brier improved AND sits below the level floor -> no
        # trigger fires -> issue_open=False on today's outcome so
        # tomorrow can open fresh if needed.
        prior = _prior_state_with_issue("a")
        d = triage(
            _scores(a=_stats(brier=0.155, log_loss=0.600)),
            _scores(a=_stats(brier=0.160, log_loss=0.610)),
            prior,
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "no_regression"
        assert d[0]["issue_open"] is False


class TestStateRoundTrip:
    """write_state writes a file triage can read on the next run."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """write_state output is readable by the next day's triage call."""
        cur = _scores(a=_stats(brier=0.260, log_loss=0.700))
        prev = _scores(a=_stats(brier=0.210, log_loss=0.610))
        state_path = tmp_path / "state.json"
        # Day 1: no state -> opens.
        day1 = triage(cur, prev, {})
        assert day1[0]["decision"] == "open_issue"
        write_state(state_path, day1, "2026-05-27T03:45:00Z")
        # Day 2: read state, same numbers -> silent (duplicate-suppressed).
        state = _load_json(state_path)
        day2 = triage(cur, prev, state)
        assert day2[0]["decision"] == "silent"
        assert day2[0]["issue_open"] is True


class TestMultipleTools:
    """Each tool evaluated independently."""

    def test_one_flags_others_silent(self) -> None:
        """Three tools with different fates resolve independently."""
        cur = _scores(
            a=_stats(brier=0.260, log_loss=0.700),  # regression + above level
            b=_stats(brier=0.150, log_loss=0.600),  # no regression, below level
            c=_stats(brier=0.180, log_loss=0.580),  # sign disagree, below level
        )
        prev = _scores(
            a=_stats(brier=0.210, log_loss=0.610),
            b=_stats(brier=0.160, log_loss=0.610),
            c=_stats(brier=0.130, log_loss=0.610),
        )
        d = {x["tool"]: x for x in triage(cur, prev, {})}
        assert d["a"]["decision"] == "open_issue"
        assert d["b"]["decision"] == "silent"
        assert d["c"]["decision"] == "silent"
        assert d["c"]["reason"] == "sign_disagreement"


class TestCorruptOrMissingState:
    """A bad state file must not wedge the loop."""

    def test_missing_state_file(self, tmp_path: Path) -> None:
        """A non-existent state path collapses to no-prior-state."""
        # Non-existent path -> _load_json returns {} -> triage proceeds.
        state = _load_json(tmp_path / "no_such.json")
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            state,
        )
        assert d[0]["decision"] == "open_issue"

    def test_truncated_state_file(self, tmp_path: Path) -> None:
        """A truncated state file is treated as missing rather than crashing."""
        # Process killed mid-write -> JSONDecodeError -> _load_json
        # returns {} -> triage proceeds.
        state_path = tmp_path / "state.json"
        state_path.write_text('{"by_tool": {"too', encoding="utf-8")
        state = _load_json(state_path)
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            state,
        )
        assert d[0]["decision"] == "open_issue"

    def test_state_missing_by_tool_key(self) -> None:
        """A state dict without by_tool is treated as no-prior-state."""
        # Old/foreign state file -> {} treated as "no prior issues".
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            {"unrelated": "stuff"},
        )
        assert d[0]["decision"] == "open_issue"


class TestIssueBodyContract:
    """The agent's Step 1 parser depends on title + body formatting."""

    def _decision(self) -> Dict[str, Any]:
        return {
            "tool": "factual_research",
            "brier_cur": 0.243,
            "brier_prev": 0.219,
            "delta_brier": 0.024,
            "n_cur": 187,
            "n_prev": 212,
            "decision": "open_issue",
            "reason": "regression",
            "issue_open": True,
        }

    def test_title_format(self) -> None:
        """Title matches the format the agent's Step 1 parser expects."""
        title = build_issue_title("factual_research")
        assert title == (
            "[tool-improvement] `factual_research`: "
            "Brier regression on polymarket W-1"
        )

    def test_body_contains_headline_and_artifact(self) -> None:
        """Issue body carries the headline numbers and the artifact URL."""
        body = build_issue_body(
            self._decision(),
            polymarket_stats={"brier": 0.243, "n": 187},
            artifact_url="https://example.test/artifacts/42",
            window_iso=_window_iso(datetime(2026, 5, 28, 3, 45, tzinfo=timezone.utc)),
        )
        # Headline numbers visible.
        assert "0.2430" in body
        assert "0.2190" in body
        assert "+0.0240" in body
        assert "factual_research" in body
        # Artifact URL passes through verbatim.
        assert "https://example.test/artifacts/42" in body
        # Window markers (any downstream parser reads these).
        assert "W-1" in body
        assert "W-2" in body
        # Baseline blocks present (machine-readable contract).
        assert "```baseline-stats-polymarket" in body
        assert "```baseline-stats" in body
        assert "@valory-coding-agent" in body
        assert "tool-improvement-agent" not in body

    def test_body_ascii_only(self) -> None:
        """Issue body is pure ASCII to avoid encoding issues in gh CLI."""
        body = build_issue_body(
            self._decision(),
            polymarket_stats={},
            artifact_url="https://example.test/x",
            window_iso=_window_iso(datetime(2026, 5, 28, 3, 45, tzinfo=timezone.utc)),
        )
        body.encode("ascii")  # raises if any non-ASCII char snuck in.


class TestLevelTrigger:
    """The level-floor trigger fires when Brier persistently exceeds the floor."""

    def test_level_above_floor_no_regression_opens(self) -> None:
        """A tool above the level floor opens even without a regression."""
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.610)),
            _scores(a=_stats(brier=0.260, log_loss=0.610)),
            {},
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "level_floor"
        assert d[0]["issue_open"] is True

    def test_level_below_floor_no_regression_silent(self) -> None:
        """A tool below the level floor and not regressing stays silent."""
        d = triage(
            _scores(a=_stats(brier=0.180, log_loss=0.610)),
            _scores(a=_stats(brier=0.180, log_loss=0.610)),
            {},
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "no_regression"

    def test_level_floor_does_not_require_sign_agreement(self) -> None:
        """Level trigger fires even when log_loss disagrees with Brier delta."""
        # Brier worsens AND above floor; log_loss improves. Sign-disagreement
        # silences the regression path, but level fires independently.
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.580)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            {},
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "level_floor"

    def test_regression_reason_wins_over_level(self) -> None:
        """When both trigger, the issue is labelled as regression."""
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            {},
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "regression"

    def test_open_issue_suppresses_level_too(self) -> None:
        """An open issue suppresses level-floor firings just like regressions."""
        prior = _prior_state_with_issue("a")
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.610)),
            _scores(a=_stats(brier=0.260, log_loss=0.610)),
            prior,
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "duplicate_suppressed"
        assert d[0]["issue_open"] is True


class TestConstants:
    """The calibrated constants are part of the public contract."""

    def test_threshold(self) -> None:
        """Brier regression threshold is the calibrated 0.040 value."""
        assert BRIER_REGRESSION_THRESHOLD == 0.040

    def test_level_threshold(self) -> None:
        """Brier level-floor threshold is the no-skill baseline (0.25)."""
        assert BRIER_LEVEL_THRESHOLD == 0.25

    def test_sample_floor(self) -> None:
        """Sample floor is 105 (the 15/day * 7d aggregate)."""
        assert VALID_N_PER_WINDOW_FLOOR == 105  # 15/day * 7d

    def test_reliability_floor(self) -> None:
        """Reliability floor is the documented 0.80 value."""
        assert RELIABILITY_FLOOR == 0.80


class TestWindowISO:
    """W-1 and W-2 are non-overlapping 7-day windows ending at `now`."""

    def test_disjoint_windows(self) -> None:
        """W-2 ends exactly where W-1 starts (non-overlapping windows)."""
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        w = _window_iso(now)
        assert w["w1_end"] == "2026-05-28T12:00:00Z"
        assert w["w1_start"] == "2026-05-21T12:00:00Z"
        assert w["w2_end"] == "2026-05-21T12:00:00Z"  # = w1_start
        assert w["w2_start"] == "2026-05-14T12:00:00Z"


class TestPlatformPropagation:
    """Platform identity flows through triage and title rendering."""

    def test_triage_records_platform_on_each_decision(self) -> None:
        """Every decision dict carries the platform passed to triage()."""
        cur = _scores(a=_stats(brier=0.260, log_loss=0.700))
        prev = _scores(a=_stats(brier=0.210, log_loss=0.610))
        d = triage(cur, prev, {}, platform="omen")
        assert d[0]["platform"] == "omen"

    def test_triage_default_platform_is_polymarket(self) -> None:
        """Omitting the platform kwarg keeps the legacy default."""
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            {},
        )
        assert d[0]["platform"] == "polymarket"

    def test_title_carries_platform(self) -> None:
        """build_issue_title interpolates the platform into the title."""
        assert build_issue_title("foo", "omen") == (
            "[tool-improvement] `foo`: Brier regression on omen W-1"
        )

    def test_title_changes_per_trigger_reason(self) -> None:
        """Level-floor issues get a distinct title style."""
        assert build_issue_title("foo", "polymarket", "level_floor") == (
            "[tool-improvement] `foo`: Brier above level on polymarket W-1"
        )
        assert build_issue_title("foo", "polymarket", "regression") == (
            "[tool-improvement] `foo`: Brier regression on polymarket W-1"
        )

    def test_body_single_platform_no_cross_ref(self) -> None:
        """When only one platform is monitored, no cross-ref text is added."""
        body = build_issue_body(
            self._decision(),
            polymarket_stats={},
            artifact_url="https://example.test/x",
            window_iso=_window_iso(datetime(2026, 5, 28, 3, 45, tzinfo=timezone.utc)),
            platforms_monitored=["polymarket"],
        )
        assert "Other monitored platforms" not in body
        assert "['polymarket']" in body

    def test_body_multi_platform_carries_cross_ref(self) -> None:
        """When multiple platforms are monitored, cross-ref text names the others."""
        body = build_issue_body(
            self._decision(),
            polymarket_stats={},
            artifact_url="https://example.test/x",
            window_iso=_window_iso(datetime(2026, 5, 28, 3, 45, tzinfo=timezone.utc)),
            platforms_monitored=["polymarket", "omen"],
        )
        assert "Other monitored platforms (omen)" in body
        assert "this issue's platform" in body

    def _decision(self) -> Dict[str, Any]:
        """Reusable decision fixture for issue-body tests."""
        return {
            "tool": "factual_research",
            "platform": "polymarket",
            "brier_cur": 0.243,
            "brier_prev": 0.219,
            "delta_brier": 0.024,
            "n_cur": 187,
            "n_prev": 212,
            "decision": "open_issue",
            "reason": "regression",
            "issue_open": True,
        }


@pytest.mark.parametrize(
    "case",
    [
        # Gate 1 (sample floor) wins over later gates.
        {"cur_n": 50, "prev_n": 200, "expected": "sample_floor"},
        {"cur_n": 200, "prev_n": 50, "expected": "sample_floor"},
        # Gate 5 (reliability) wins over regression.
        {"cur_rel": 0.5, "expected": "reliability_collapse"},
    ],
)
def test_gate_precedence(case: Dict[str, Any]) -> None:
    """Higher-priority gates short-circuit lower ones (parametrized)."""
    cur = _stats(
        n=case.get("cur_n", 200),
        valid_n=case.get("cur_n", 200),
        brier=0.30,
        log_loss=0.80,
        reliability=case.get("cur_rel", 0.95),
    )
    prev = _stats(
        n=case.get("prev_n", 200),
        valid_n=case.get("prev_n", 200),
        brier=0.20,
        log_loss=0.60,
    )
    d = triage(_scores(a=cur), _scores(a=prev), {})
    if case["expected"] == "reliability_collapse":
        assert d[0]["decision"] == "reliability_collapse"
    else:
        assert d[0]["reason"] == case["expected"]


class TestOpenNowSuppression:
    """Live gh issue list overlay is the suppression source-of-truth."""

    def test_open_now_suppresses_regardless_of_state(self) -> None:
        """A tool in open_now is suppressed even when state says issue_open=False."""
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            {},  # empty state file
            open_now=["a"],
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "duplicate_suppressed"

    def test_open_now_empty_re_arms_trigger(self) -> None:
        """Closing the issue (absent from open_now) re-arms even if state says issue_open=True."""
        prior = _prior_state_with_issue("a")  # state pins issue_open=True
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            prior,
            open_now=[],  # GitHub says no open issue
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "regression"


class TestOpenIssueSubprocess:
    """_open_issue subprocess paths: dry-run, success, failure, timeout."""

    def test_dry_run_returns_zero_without_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dry-run path returns 0 and does NOT shell out."""
        called = {"n": 0}

        def boom(*_a: Any, **_k: Any) -> None:
            called["n"] += 1
            raise RuntimeError("subprocess.run must not be called in dry-run")

        monkeypatch.setattr("benchmark.tool_improvement_triage.subprocess.run", boom)
        rc = _open_issue("r/r", "tool-improvement", "t", "b", dry_run=True)
        assert rc == 0
        assert called["n"] == 0

    def test_nonzero_rc_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed gh issue create surfaces its rc so main() exits non-zero."""
        result = SimpleNamespace(returncode=1, stdout="", stderr="gh: forbidden")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        rc = _open_issue("r/r", "tool-improvement", "t", "b", dry_run=False)
        assert rc == 1

    def test_success_rc_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A successful gh issue create returns 0."""
        result = SimpleNamespace(
            returncode=0, stdout="https://example/issues/1", stderr=""
        )
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        rc = _open_issue("r/r", "tool-improvement", "t", "b", dry_run=False)
        assert rc == 0

    def test_timeout_returns_124(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """subprocess.TimeoutExpired surfaces as rc=124 so main() counts a failure."""

        def boom(*_a: Any, **_k: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="gh", timeout=30)

        monkeypatch.setattr("benchmark.tool_improvement_triage.subprocess.run", boom)
        rc = _open_issue("r/r", "tool-improvement", "t", "b", dry_run=False)
        assert rc == 124


class TestOpenIssueToolsParser:
    """_open_issue_tools backtick title parsing + gh-error fallback."""

    def test_parses_backtick_tool_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tool name between first two backticks is extracted."""
        stdout = (
            '[{"title": "[tool-improvement] `superforcaster`: Brier '
            'regression on polymarket W-1"},'
            ' {"title": "[tool-improvement] `factual_research`: Brier '
            'above level on polymarket W-1"}]'
        )
        result = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        tools = _open_issue_tools("r/r", "tool-improvement")
        assert tools == ["superforcaster", "factual_research"]

    def test_gh_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-zero gh exit yields None so caller falls back to state file."""
        # Distinct from "success with zero open issues" which returns [].
        result = SimpleNamespace(returncode=4, stdout="", stderr="auth required")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        assert _open_issue_tools("r/r", "tool-improvement") is None

    def test_gh_success_zero_issues_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successful gh with no open issues yields [] (re-arms suppression)."""
        result = SimpleNamespace(returncode=0, stdout="[]", stderr="")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        assert _open_issue_tools("r/r", "tool-improvement") == []


class TestLevelFloorIssueBody:
    """build_issue_body renders the level_floor signal with its own headline."""

    def test_level_floor_body_headline(self) -> None:
        """level_floor reason yields the persistent-level headline + signal label."""
        decision = {
            "tool": "factual_research",
            "platform": "polymarket",
            "brier_cur": 0.262,
            "brier_prev": 0.255,
            "delta_brier": 0.007,
            "n_cur": 210,
            "n_prev": 212,
            "reason": "level_floor",
        }
        body = build_issue_body(
            decision,
            polymarket_stats={"brier": 0.262, "valid_n": 210},
            artifact_url="https://example/artifact",
            window_iso={
                "w1_start": "2026-05-22T00:00:00Z",
                "w1_end": "2026-05-29T00:00:00Z",
                "w2_start": "2026-05-15T00:00:00Z",
                "w2_end": "2026-05-22T00:00:00Z",
            },
        )
        assert "persistently above" in body
        assert "level signal" in body
        assert "factual_research" in body


def _prior_level_floor_state(*tools: str) -> Dict[str, Any]:
    """Prior state where each tool's last open_issue carried trigger=level_floor.

    The cooldown reads the dedicated ``trigger`` field (not ``reason``)
    because ``reason`` gets clobbered with ``duplicate_suppressed`` every
    day the GitHub issue is open. Simulating an open-then-closed level
    issue means injecting the ``trigger`` marker the live pipeline would
    have set when it first opened.
    """
    return {
        "by_tool": {
            t: {
                "issue_open": False,
                "reason": "duplicate_suppressed",
                "trigger": "level_floor",
            }
            for t in tools
        }
    }


def _state_from_decisions(decisions: list) -> Dict[str, Any]:
    """Build a prior_state dict from triage() output (mimics write_state)."""
    return {
        "by_tool": {
            d["tool"]: {
                "decision": d["decision"],
                "reason": d["reason"],
                "trigger": d.get("trigger"),
                "issue_open": d["issue_open"],
            }
            for d in decisions
        }
    }


class TestLevelFloorCooldown:
    """A closed level_floor issue stays suppressed until Brier drops below the re-arm band."""

    def test_closed_level_floor_suppressed_while_brier_above_rearm(self) -> None:
        """Live open_now=[] + brier still above 0.22 -> cooldown, no re-open."""
        # Prior state: level_floor issue was open and is now closed (so
        # open_now=[]). Brier is still above the re-arm band -> must NOT
        # reopen.
        prior = _prior_level_floor_state("a")
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.610)),
            _scores(a=_stats(brier=0.260, log_loss=0.610)),
            prior,
            open_now=[],
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "level_floor_cooldown"
        assert d[0]["trigger"] == "level_floor"
        # Cooldown is a local suppression after the GitHub issue closed,
        # so issue_open=False (no live issue exists). The trigger marker
        # carries the cooldown state across runs.
        assert d[0]["issue_open"] is False

    def test_closed_level_floor_re_arms_below_band(self) -> None:
        """Brier dropped below BRIER_LEVEL_RE_ARM -> cooldown lifts and tool can re-fire later."""
        # Brier below 0.22 today + below the level floor + no
        # regression -> silent/no_regression, cooldown effectively
        # over.
        prior = _prior_level_floor_state("a")
        d = triage(
            _scores(a=_stats(brier=0.150, log_loss=0.580)),
            _scores(a=_stats(brier=0.160, log_loss=0.610)),
            prior,
            open_now=[],
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "no_regression"

    def test_cooldown_does_not_block_regression_reason(self) -> None:
        """Hysteresis applies to level_floor only; regression-reason re-fires immediately."""
        # Same tool had a closed level_floor issue, but today it
        # genuinely regresses (delta > 0.040 + sign agreement) -> the
        # cooldown must not stop a regression dispatch.
        prior = _prior_level_floor_state("a")
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            prior,
            open_now=[],
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "regression"

    def test_cooldown_constant_below_threshold(self) -> None:
        """The re-arm constant must sit strictly below the level threshold."""
        assert BRIER_LEVEL_RE_ARM < BRIER_LEVEL_THRESHOLD

    def test_cooldown_survives_duplicate_suppressed_lifecycle(self) -> None:
        """End-to-end lifecycle: open -> N days suppressed -> close -> still cooldown.

        @OjusWiZard test_tool_improvement_triage.py:727 thread: the previous
        cooldown test injected reason=level_floor directly, skipping the
        duplicate_suppressed overwrite that happens on every real day the
        issue is open. This test runs the actual day-by-day state machine
        to prove the trigger marker survives the entire suppressed period.
        """
        # Day 1: Brier 0.30, no prior state -> open level_floor issue
        cur = _scores(a=_stats(brier=0.300, log_loss=0.620))
        prev = _scores(a=_stats(brier=0.295, log_loss=0.610))
        d1 = triage(cur, prev, {}, open_now=[])
        assert d1[0]["decision"] == "open_issue"
        assert d1[0]["trigger"] == "level_floor"
        state = _state_from_decisions(d1)

        # Days 2..8: GitHub issue is open, Brier still 0.30. Every day
        # triage writes reason=duplicate_suppressed. The trigger marker
        # must persist through ALL of these writes.
        for _ in range(7):
            dN = triage(cur, prev, state, open_now=["a"])
            assert dN[0]["decision"] == "silent"
            assert dN[0]["reason"] == "duplicate_suppressed"
            assert dN[0]["trigger"] == "level_floor"  # the load-bearing claim
            state = _state_from_decisions(dN)

        # Day 9: operator closes the GitHub issue as wontfix.
        # Brier is still 0.30 (above 0.22 re-arm band). The cooldown must
        # engage and stop a new issue from firing.
        d9 = triage(cur, prev, state, open_now=[])
        assert d9[0]["decision"] == "silent"
        assert d9[0]["reason"] == "level_floor_cooldown"
        assert d9[0]["trigger"] == "level_floor"

    def test_cooldown_holds_in_hysteresis_band(self) -> None:
        """Brier in (BRIER_LEVEL_RE_ARM, BRIER_LEVEL_THRESHOLD] keeps cooldown alive.

        Without the cooldown check running before the no_regression guard,
        a Brier in this band exits via no_regression (above_level is False)
        and silently clears the marker. This test forces the band traversal.
        """
        prior = _prior_level_floor_state("a")
        # Brier 0.230 sits inside (0.22, 0.25] -> above_level=False, no
        # regression, but cooldown must still fire.
        d = triage(
            _scores(a=_stats(brier=0.230, log_loss=0.600)),
            _scores(a=_stats(brier=0.225, log_loss=0.595)),
            prior,
            open_now=[],
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "level_floor_cooldown"
        assert d[0]["trigger"] == "level_floor"


class TestLevelFloorReArmAndRegressionCoverage:
    """Closes the test gap @OjusWiZard flagged on the re-arm + missing_log_loss branches."""

    def test_open_now_empty_re_arms_level_floor(self) -> None:
        """Re-arm path for reason=level_floor (no prior level cooldown, no regression)."""
        # No prior state -> no cooldown. Brier above 0.25, no
        # regression -> level_floor fires.
        d = triage(
            _scores(a=_stats(brier=0.270, log_loss=0.610)),
            _scores(a=_stats(brier=0.265, log_loss=0.610)),
            {},
            open_now=[],
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "level_floor"

    def test_missing_log_loss_above_level_fires_level_floor(self) -> None:
        """Missing log_loss + delta > threshold + above_level=True -> level fires."""
        # Regression-path sign-agreement can't be computed because
        # log_loss is None on cur. But level_floor still fires because
        # Brier is above 0.25 -> open_issue with reason="level_floor".
        d = triage(
            _scores(a=_stats(brier=0.280, log_loss=None)),  # type: ignore[arg-type]
            _scores(a=_stats(brier=0.230, log_loss=0.610)),
            {},
            open_now=[],
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "level_floor"

    def test_missing_log_loss_below_level_silent(self) -> None:
        """Missing log_loss + delta > threshold + below level -> silent/missing_log_loss."""
        d = triage(
            _scores(a=_stats(brier=0.210, log_loss=None)),  # type: ignore[arg-type]
            _scores(a=_stats(brier=0.160, log_loss=0.610)),
            {},
            open_now=[],
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "missing_log_loss"


class TestMainExitCode:
    """main()'s exit code is the CI contract for detecting lost dispatches."""

    @staticmethod
    def _write_scores(results_dir: Path, brier_cur: float, brier_prev: float) -> None:
        """Drop matching cur/prev rolling score files for a single tool 'a'."""
        results_dir.mkdir(parents=True, exist_ok=True)
        cur = {"by_tool": {"a": _stats(brier=brier_cur, log_loss=0.700)}}
        prev = {"by_tool": {"a": _stats(brier=brier_prev, log_loss=0.610)}}
        (results_dir / "rolling_scores_polymarket.json").write_text(json.dumps(cur))
        (results_dir / "prev_rolling_scores_polymarket.json").write_text(
            json.dumps(prev)
        )
        (results_dir / "scores_polymarket.json").write_text(json.dumps(cur))

    def test_main_returns_nonzero_when_open_issue_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A failed gh issue create surfaces as main() returning 1."""
        results_dir = tmp_path / "results"
        self._write_scores(results_dir, brier_cur=0.260, brier_prev=0.210)
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.RESULTS_DIR", results_dir
        )

        # gh issue list succeeds with no open issues; gh issue create fails.
        call_count = {"n": 0}

        def fake_run(cmd: list, **_k: Any) -> SimpleNamespace:
            call_count["n"] += 1
            if "list" in cmd:
                return SimpleNamespace(returncode=0, stdout="[]", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="forbidden")

        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run", fake_run
        )
        monkeypatch.setattr(
            "sys.argv",
            ["triage", "--state", str(tmp_path / "state.json"), "--repo", "r/r"],
        )
        rc = main()
        assert rc == 1
        # Verify state.json was written and the failed open issue is
        # NOT persisted as issue_open=True (state-poisoning guard).
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["by_tool"]["a"]["issue_open"] is False
        assert state["by_tool"]["a"]["reason"] == "open_issue_failed"

    def test_main_returns_one_on_empty_cur_without_overwriting_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Empty cur -> log.error + return 1 + state file untouched."""
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True)
        # Valid JSON but empty by_tool -> zero decisions.
        empty: Dict[str, Any] = {"by_tool": {}}
        (results_dir / "rolling_scores_polymarket.json").write_text(json.dumps(empty))
        (results_dir / "prev_rolling_scores_polymarket.json").write_text(
            json.dumps(empty)
        )
        (results_dir / "scores_polymarket.json").write_text(json.dumps(empty))
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.RESULTS_DIR", results_dir
        )
        # Pre-existing state should NOT be overwritten.
        state_path = tmp_path / "state.json"
        prior = {"by_tool": {"a": {"issue_open": True, "reason": "level_floor"}}}
        state_path.write_text(json.dumps(prior))

        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: SimpleNamespace(returncode=0, stdout="[]", stderr=""),
        )
        monkeypatch.setattr(
            "sys.argv",
            ["triage", "--state", str(state_path), "--repo", "r/r"],
        )
        rc = main()
        assert rc == 1
        # Prior state preserved verbatim.
        assert json.loads(state_path.read_text()) == prior
