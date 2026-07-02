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
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from benchmark.tool_improvement_triage import (
    BRIER_LEVEL_THRESHOLD,
    BRIER_REGRESSION_THRESHOLD,
    BSS_LEVEL_FLOOR,
    PROMOTION_LABEL,
    RECENT_CLOSE_DAYS,
    RELIABILITY_FLOOR,
    VALID_N_PER_WINDOW_FLOOR,
    _below_level,
    _closed_issue_pairs,
    _load_json,
    _load_lineage_children,
    _open_issue,
    _open_issue_tools,
    _window_iso,
    build_issue_body,
    build_issue_title,
    build_promotion_body,
    build_promotion_title,
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
    brier_skill_score: float | None = None,
) -> Dict[str, Any]:
    s: Dict[str, Any] = {
        "n": n,
        "valid_n": valid_n,
        "brier": brier,
        "log_loss": log_loss,
        "reliability": reliability,
    }
    # Only set the skill score when a test explicitly exercises the
    # market-relative floor; omitting it drives the backward-compatible
    # absolute-Brier fallback (matches pre-BSS score files).
    if brier_skill_score is not None:
        s["brier_skill_score"] = brier_skill_score
    return s


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

    def test_open_now_tuple_keyed_by_platform(self) -> None:
        """(tool, platform) tuples scope suppression to the matching platform only."""
        # Issue open for tool 'a' on omen; we are triaging polymarket -->
        # the omen issue MUST NOT suppress the polymarket dispatch.
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            {},
            platform="polymarket",
            open_now=[("a", "omen")],
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "regression"

    def test_open_now_tuple_same_platform_suppresses(self) -> None:
        """(tool, platform) tuple on the matching platform DOES suppress."""
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            {},
            platform="polymarket",
            open_now=[("a", "polymarket")],
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "duplicate_suppressed"


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

    def test_parses_tool_and_platform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """(tool, platform) pairs are extracted from the standard title format."""
        stdout = (
            '[{"title": "[tool-improvement] `superforcaster`: Brier '
            'regression on polymarket W-1"},'
            ' {"title": "[tool-improvement] `factual_research`: Brier '
            'above level on omen W-1"}]'
        )
        result = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        pairs = _open_issue_tools("r/r", "tool-improvement")
        assert pairs == [("superforcaster", "polymarket"), ("factual_research", "omen")]

    def test_skips_title_without_backticks(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A label-matching issue filed manually (no backticks) is skipped + warned."""
        stdout = '[{"title": "[tool-improvement] superforcaster bad on polymarket"}]'
        result = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        with caplog.at_level(logging.WARNING):
            assert _open_issue_tools("r/r", "tool-improvement") == []
        assert any(
            "did not match the expected format" in r.message for r in caplog.records
        )

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

    def test_gh_invocation_uses_valid_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the gh CLI contract: --limit must be a positive integer."""
        # Regression guard: a previous version passed --limit 0 thinking
        # it meant "unlimited"; gh rejects that with rc=1 and the live-gh
        # suppression silently degraded to the state-file fallback.
        captured = {}

        def fake_run(cmd, **_k):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")

        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run", fake_run
        )
        _open_issue_tools("r/r", "tool-improvement")
        cmd = captured["cmd"]
        assert "--limit" in cmd
        limit_val = cmd[cmd.index("--limit") + 1]
        assert int(limit_val) >= 1, f"gh requires --limit >= 1, got {limit_val!r}"


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


class TestLevelFloorAndRegressionCoverage:
    """Closes the test gap @OjusWiZard flagged on the missing_log_loss branches."""

    def test_open_now_empty_fires_level_floor(self) -> None:
        """Re-fire path for reason=level_floor (no prior cooldown, no regression)."""
        # No prior state, no closed_issues -> no cooldown. Brier above
        # 0.25, no regression -> level_floor fires.
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

    def test_missing_brier_cur_silent(self) -> None:
        """Missing Brier on the current window -> silent/missing_brier (covers #311 g)."""
        d = triage(
            _scores(a=_stats(brier=None)),  # type: ignore[arg-type]
            _scores(a=_stats(brier=0.200)),
            {},
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "missing_brier"

    def test_missing_brier_prev_silent(self) -> None:
        """Missing Brier on the previous window -> silent/missing_brier."""
        d = triage(
            _scores(a=_stats(brier=0.260)),
            _scores(a=_stats(brier=None)),  # type: ignore[arg-type]
            {},
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "missing_brier"

    def test_write_state_shape_for_silent_decision(self, tmp_path: Path) -> None:
        """write_state persists delta_brier=None on silent/sample_floor (covers #311 h)."""
        # Sample floor short-circuits BEFORE delta_brier is computed, so
        # the persisted payload must carry delta_brier=None without
        # crashing the JSON serializer.
        d = triage(
            _scores(a=_stats(n=40, valid_n=40, brier=0.300)),
            _scores(a=_stats(brier=0.200)),
            {},
        )
        state_path = tmp_path / "state.json"
        write_state(state_path, d, "2026-06-01T03:45:00Z")
        payload = json.loads(state_path.read_text())
        row = payload["by_tool"]["a"]
        assert row["decision"] == "silent"
        assert row["reason"] == "sample_floor"
        assert row["delta_brier"] is None
        assert row["brier_cur"] == 0.300

    def test_write_state_shape_for_reliability_collapse(self, tmp_path: Path) -> None:
        """write_state persists delta_brier=None on reliability_collapse."""
        d = triage(
            _scores(a=_stats(brier=0.260, reliability=0.65)),
            _scores(a=_stats()),
            {},
        )
        state_path = tmp_path / "state.json"
        write_state(state_path, d, "2026-06-01T03:45:00Z")
        payload = json.loads(state_path.read_text())
        row = payload["by_tool"]["a"]
        assert row["decision"] == "reliability_collapse"
        assert row["delta_brier"] is None


def _closed_triples(*pairs: Any, days_ago: int = 0, hours_ago: int = 0) -> Any:
    """Helper: build a `(tool, platform, closed_at)` list for ``triage(closed_issues=...)``.

    :param pairs: (tool, platform) tuples to time-stamp identically.
    :param days_ago: how many days ago each close happened (used to land
        inside or outside the RECENT_CLOSE_DAYS window).
    :param hours_ago: additional hours offset for fractional-day testing
        (e.g. ``hours_ago=43`` lands the close in the ``(1d, 2d)`` band that
        exercises the gate-vs-log status agreement on N=1).
    :return: list of (tool, platform, closed_at) triples.
    """
    closed_at = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc) - timedelta(
        days=days_ago, hours=hours_ago
    )
    return [(t, p, closed_at) for t, p in pairs]


_NOW = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)


class TestRecentlyClosedSilence:
    """Issues closed within RECENT_CLOSE_DAYS silence any re-fire of the same pair.

    Replaces the previous Brier-band re-arm hysteresis: a partial fix that
    leaves the Brier above ``BRIER_LEVEL_THRESHOLD`` is allowed to re-fire
    after the window elapses. Tests exercise both triggers + the "no
    trigger" pass-through.
    """

    def test_recently_closed_above_level_silenced(self) -> None:
        """Tool would fire level_floor but closed within window -> silent."""
        closed = _closed_triples(("a", "polymarket"), days_ago=0)
        d = triage(
            _scores(a=_stats(brier=0.300, log_loss=0.620)),
            _scores(a=_stats(brier=0.295, log_loss=0.610)),
            {},
            open_now=[],
            closed_issues=closed,
            now=_NOW,
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "recently_closed"
        assert d[0]["trigger"] == "level_floor"
        assert d[0]["issue_open"] is False

    def test_recently_closed_regression_silenced(self) -> None:
        """Regression delta crossed but closed recently -> silent."""
        closed = _closed_triples(("a", "polymarket"), days_ago=0)
        d = triage(
            _scores(a=_stats(brier=0.270, log_loss=0.700)),
            _scores(a=_stats(brier=0.210, log_loss=0.610)),
            {},
            open_now=[],
            closed_issues=closed,
            now=_NOW,
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "recently_closed"
        assert d[0]["trigger"] == "regression"

    def test_recently_closed_but_no_trigger_passes_through(self) -> None:
        """No trigger fires -> cooldown gate is irrelevant; standard no_regression."""
        closed = _closed_triples(("a", "polymarket"), days_ago=0)
        d = triage(
            _scores(a=_stats(brier=0.180, log_loss=0.580)),
            _scores(a=_stats(brier=0.180, log_loss=0.580)),
            {},
            open_now=[],
            closed_issues=closed,
            now=_NOW,
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "no_regression"

    def test_recently_closed_filtered_by_platform(self) -> None:
        """Close on `(tool, platform_A)` must not silence `(tool, platform_B)`."""
        closed = _closed_triples(("a", "omen"), days_ago=1)
        d = triage(
            _scores(a=_stats(brier=0.300, log_loss=0.620)),
            _scores(a=_stats(brier=0.295, log_loss=0.610)),
            {},
            platform="polymarket",
            open_now=[],
            closed_issues=closed,
            now=_NOW,
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "level_floor"

    def test_closed_issues_none_applies_no_cooldown(self) -> None:
        """Gh failure path: closed_issues=None -> no cooldown, fires normally."""
        d = triage(
            _scores(a=_stats(brier=0.300, log_loss=0.620)),
            _scores(a=_stats(brier=0.295, log_loss=0.610)),
            {},
            open_now=[],
            closed_issues=None,
            now=_NOW,
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "level_floor"

    def test_close_exactly_at_window_boundary_silenced(self) -> None:
        """Close on the exact RECENT_CLOSE_DAYS boundary is still recent (>= cutoff)."""
        closed = _closed_triples(("a", "polymarket"), days_ago=RECENT_CLOSE_DAYS)
        d = triage(
            _scores(a=_stats(brier=0.300, log_loss=0.620)),
            _scores(a=_stats(brier=0.295, log_loss=0.610)),
            {},
            open_now=[],
            closed_issues=closed,
            now=_NOW,
        )
        # Boundary uses >= so the close at exactly N days ago still silences.
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "recently_closed"

    def test_recent_close_days_constant_is_positive(self) -> None:
        """RECENT_CLOSE_DAYS must be a positive integer or the gate becomes a no-op."""
        assert isinstance(RECENT_CLOSE_DAYS, int)
        assert RECENT_CLOSE_DAYS > 0


class TestCloseOlderThanWindow:
    """Closes older than ``RECENT_CLOSE_DAYS`` are out of cooldown -> tool can re-fire.

    This is the dead-zone fix: under the previous Brier-band re-arm a partial
    fix that left the Brier above ``BRIER_LEVEL_THRESHOLD`` could silence a
    tool forever. The flat N-day cooldown elapses on its own.
    """

    def test_old_close_does_not_silence_above_level_tool(self) -> None:
        """Closed > N days ago + Brier still above threshold -> level_floor fires fresh."""
        # The same scenario that under the old re-arm marker would have
        # been silenced indefinitely: Brier 0.260 (above threshold), no
        # regression, but the close is outside the cooldown window.
        closed = _closed_triples(("a", "polymarket"), days_ago=RECENT_CLOSE_DAYS + 2)
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.610)),
            _scores(a=_stats(brier=0.260, log_loss=0.610)),
            {},
            open_now=[],
            closed_issues=closed,
            now=_NOW,
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "level_floor"

    def test_most_recent_close_wins_when_both_old_and_new_exist(self) -> None:
        """A pair with BOTH an old and a recent close -> recent close drives the cooldown."""
        old = _closed_triples(("a", "polymarket"), days_ago=RECENT_CLOSE_DAYS + 9)
        new = _closed_triples(("a", "polymarket"), days_ago=0)
        d = triage(
            _scores(a=_stats(brier=0.300, log_loss=0.620)),
            _scores(a=_stats(brier=0.295, log_loss=0.610)),
            {},
            open_now=[],
            closed_issues=old + new,
            now=_NOW,
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "recently_closed"

    def test_old_close_with_benign_brier_silent_no_regression(self) -> None:
        """Old close + benign Brier -> standard no_regression silence, no spurious gate fires."""
        closed = _closed_triples(("a", "polymarket"), days_ago=RECENT_CLOSE_DAYS + 5)
        d = triage(
            _scores(a=_stats(brier=0.180, log_loss=0.580)),
            _scores(a=_stats(brier=0.180, log_loss=0.580)),
            {},
            open_now=[],
            closed_issues=closed,
            now=_NOW,
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "no_regression"


class TestCooldownStatusLogging:
    """Per-tool cooldown status is logged as INFO so the operator can see why a tool is (or isn't) firing."""

    def test_active_cooldown_logs_active(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A tool with a close within the window logs cooldown ACTIVE."""
        closed = _closed_triples(("a", "polymarket"), days_ago=0)
        with caplog.at_level(logging.INFO, logger="benchmark.tool_improvement_triage"):
            triage(
                _scores(a=_stats(brier=0.300, log_loss=0.620)),
                _scores(a=_stats(brier=0.295, log_loss=0.610)),
                {},
                open_now=[],
                closed_issues=closed,
                now=_NOW,
            )
        matches = [
            r.getMessage()
            for r in caplog.records
            if "cooldown ACTIVE" in r.getMessage()
        ]
        assert len(matches) == 1
        assert "'a'" in matches[0]
        assert "'polymarket'" in matches[0]
        assert f"N={RECENT_CLOSE_DAYS}" in matches[0]

    def test_elapsed_cooldown_logs_elapsed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A tool whose close is past the window logs cooldown ELAPSED."""
        closed = _closed_triples(("a", "polymarket"), days_ago=RECENT_CLOSE_DAYS + 2)
        with caplog.at_level(logging.INFO, logger="benchmark.tool_improvement_triage"):
            triage(
                _scores(a=_stats(brier=0.180, log_loss=0.580)),
                _scores(a=_stats(brier=0.180, log_loss=0.580)),
                {},
                open_now=[],
                closed_issues=closed,
                now=_NOW,
            )
        matches = [
            r.getMessage()
            for r in caplog.records
            if "cooldown ELAPSED" in r.getMessage()
        ]
        assert len(matches) == 1
        assert "'a'" in matches[0]

    def test_no_close_history_no_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """A tool with no recent close history produces no cooldown log line."""
        with caplog.at_level(logging.INFO, logger="benchmark.tool_improvement_triage"):
            triage(
                _scores(a=_stats(brier=0.180, log_loss=0.580)),
                _scores(a=_stats(brier=0.180, log_loss=0.580)),
                {},
                open_now=[],
                closed_issues=[],
                now=_NOW,
            )
        assert not any(
            "cooldown ACTIVE" in r.getMessage() or "cooldown ELAPSED" in r.getMessage()
            for r in caplog.records
        )

    def test_ancient_close_outside_horizon_not_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A close far past the log horizon (2*N+7 days) produces no log noise."""
        closed = _closed_triples(
            ("a", "polymarket"), days_ago=2 * RECENT_CLOSE_DAYS + 30
        )
        with caplog.at_level(logging.INFO, logger="benchmark.tool_improvement_triage"):
            triage(
                _scores(a=_stats(brier=0.180, log_loss=0.580)),
                _scores(a=_stats(brier=0.180, log_loss=0.580)),
                {},
                open_now=[],
                closed_issues=closed,
                now=_NOW,
            )
        assert not any(
            "cooldown ACTIVE" in r.getMessage() or "cooldown ELAPSED" in r.getMessage()
            for r in caplog.records
        )

    def test_retired_tool_with_close_not_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A tool not in current rolling scores (retired) doesn't log a cooldown line."""
        # closed_issues references "b" but cur only has "a" -> the cur_tools
        # gate keeps the retired tool's log line out.
        closed = _closed_triples(("b", "polymarket"), days_ago=0)
        with caplog.at_level(logging.INFO, logger="benchmark.tool_improvement_triage"):
            triage(
                _scores(a=_stats(brier=0.180, log_loss=0.580)),
                _scores(a=_stats(brier=0.180, log_loss=0.580)),
                {},
                open_now=[],
                closed_issues=closed,
                now=_NOW,
            )
        assert not any(
            "'b'" in r.getMessage() and "cooldown" in r.getMessage()
            for r in caplog.records
        )

    def test_fractional_band_log_status_matches_gate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Log status must agree with the gate in the (N, N+1)-day band.

        Regression guard for the gate-vs-log mismatch flagged in PR #329
        review: with N=1, a close ~43h ago lands in the (1d, 2d) band where
        ``age.days == 1`` would falsely say ACTIVE while the gate actually
        lets the issue fire. The fix mirrors the gate's predicate exactly
        (``last_close >= cutoff``) so the log and the decision can never
        disagree on a tool the operator is consulting the log about.

        :param caplog: pytest log capture fixture.
        """
        # 43 hours ago with N=1 -> outside the gate window (gate fires),
        # but age.days == 1 -> floored-day status would have said ACTIVE.
        closed = _closed_triples(("a", "polymarket"), hours_ago=43)
        with caplog.at_level(logging.INFO, logger="benchmark.tool_improvement_triage"):
            decisions = triage(
                _scores(a=_stats(brier=0.300, log_loss=0.620)),
                _scores(a=_stats(brier=0.295, log_loss=0.610)),
                {},
                open_now=[],
                closed_issues=closed,
                now=_NOW,
            )
        # Gate fires the issue (close is past the N=1 window).
        assert decisions[0]["decision"] == "open_issue"
        assert decisions[0]["reason"] == "level_floor"
        # Log must agree: ELAPSED, not ACTIVE.
        log_lines = [
            r.getMessage() for r in caplog.records if "cooldown" in r.getMessage()
        ]
        assert len(log_lines) == 1
        assert "cooldown ELAPSED" in log_lines[0]
        assert "cooldown ACTIVE" not in log_lines[0]


class TestClosedIssuePairs:
    """``_closed_issue_pairs`` backtick title + ``closedAt`` parsing + gh-error fallback.

    Mirrors :class:`TestOpenIssueToolsParser` for the closed-issue sibling
    which adds ``closedAt`` ISO parsing on top of the title-regex flow. These
    are the fail-open paths the cooldown design leans on.
    """

    def test_parses_tool_platform_and_closed_at(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(tool, platform, closed_at) triples are extracted with timezone-aware UTC."""
        stdout = (
            '[{"title": "[tool-improvement] `superforcaster`: Brier '
            'regression on polymarket W-1", "closedAt": "2026-06-05T10:30:00Z"},'
            ' {"title": "[tool-improvement] `factual_research`: Brier '
            'above level on omen W-1", "closedAt": "2026-06-04T08:15:00Z"}]'
        )
        result = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        triples = _closed_issue_pairs("r/r", "tool-improvement")
        assert triples is not None
        assert len(triples) == 2
        assert triples[0][0] == "superforcaster"
        assert triples[0][1] == "polymarket"
        assert triples[0][2] == datetime(2026, 6, 5, 10, 30, 0, tzinfo=timezone.utc)
        assert triples[1][0] == "factual_research"
        assert triples[1][1] == "omen"

    def test_skips_title_without_backticks(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A label-matching issue filed manually (no backticks) is skipped + warned."""
        stdout = (
            '[{"title": "[tool-improvement] superforcaster bad on polymarket",'
            ' "closedAt": "2026-06-05T10:30:00Z"}]'
        )
        result = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        with caplog.at_level(logging.WARNING):
            assert _closed_issue_pairs("r/r", "tool-improvement") == []
        assert any(
            "did not match the expected format" in r.message for r in caplog.records
        )

    def test_skips_unparseable_closed_at(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed ``closedAt`` field is skipped + warned (cooldown stays fail-open)."""
        stdout = (
            '[{"title": "[tool-improvement] `superforcaster`: Brier '
            'above level on polymarket W-1", "closedAt": "not-a-date"}]'
        )
        result = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        with caplog.at_level(logging.WARNING):
            assert _closed_issue_pairs("r/r", "tool-improvement") == []
        assert any("unparseable closedAt" in r.message for r in caplog.records)

    def test_gh_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-zero gh exit yields ``None`` so the caller skips the cooldown gate."""
        # Distinct from "success with zero closed issues" which returns [].
        result = SimpleNamespace(returncode=4, stdout="", stderr="auth required")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        assert _closed_issue_pairs("r/r", "tool-improvement") is None

    def test_gh_timeout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``subprocess.TimeoutExpired`` is caught and returns ``None`` (fail-open)."""

        def fake_run(*_a: Any, **_k: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd="gh", timeout=30)

        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run", fake_run
        )
        assert _closed_issue_pairs("r/r", "tool-improvement") is None

    def test_gh_oserror_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``OSError`` (gh binary missing) is caught and returns ``None`` (fail-open)."""

        def fake_run(*_a: Any, **_k: Any) -> Any:
            raise OSError("gh: command not found")

        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run", fake_run
        )
        assert _closed_issue_pairs("r/r", "tool-improvement") is None

    def test_gh_invalid_json_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unparseable JSON on stdout yields ``None`` so cooldown stays fail-open."""
        result = SimpleNamespace(returncode=0, stdout="not json", stderr="")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        assert _closed_issue_pairs("r/r", "tool-improvement") is None

    def test_gh_success_zero_issues_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successful gh with no closed issues yields ``[]`` (no cooldown anywhere)."""
        result = SimpleNamespace(returncode=0, stdout="[]", stderr="")
        monkeypatch.setattr(
            "benchmark.tool_improvement_triage.subprocess.run",
            lambda *a, **k: result,
        )
        assert _closed_issue_pairs("r/r", "tool-improvement") == []


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


class TestBssLevelFloor:
    """Level trigger is market-relative (BSS) when the skill score is present."""

    def test_bss_below_floor_opens(self) -> None:
        """BSS below the floor fires the level trigger even with a modest Brier."""
        d = triage(
            _scores(a=_stats(brier=0.22, brier_skill_score=BSS_LEVEL_FLOOR - 0.10)),
            _scores(a=_stats(brier=0.22, brier_skill_score=BSS_LEVEL_FLOOR - 0.10)),
            {},
        )
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "level_floor"

    def test_high_brier_but_positive_skill_is_silent(self) -> None:
        """The market-ceiling case: high absolute Brier but the tool still beats
        its base rate (BSS > floor) must NOT fire -- this is the whole point of
        the market-relative floor (the legacy absolute 0.25 floor would fire)."""
        assert 0.30 > BRIER_LEVEL_THRESHOLD  # would trip the legacy absolute floor
        d = triage(
            _scores(a=_stats(brier=0.30, brier_skill_score=0.05)),
            _scores(a=_stats(brier=0.30, brier_skill_score=0.05)),
            {},
        )
        assert d[0]["decision"] == "silent"
        assert d[0]["reason"] == "no_regression"

    def test_missing_bss_falls_back_to_absolute_floor(self) -> None:
        """No skill score -> legacy absolute Brier floor still applies."""
        assert _below_level(0.30, None) is True
        assert _below_level(0.20, None) is False
        d = triage(_scores(a=_stats(brier=0.30)), _scores(a=_stats(brier=0.30)), {})
        assert d[0]["decision"] == "open_issue"
        assert d[0]["reason"] == "level_floor"


class TestLineageDescendant:
    """A tool with a merged fix variant is routed to a promotion note, not a fix."""

    def _firing(self) -> Dict[str, Any]:
        # Level-floor fire (BSS below the floor), stable across windows.
        return _scores(a=_stats(brier=0.22, brier_skill_score=BSS_LEVEL_FLOOR - 0.10))

    def test_descendant_routes_to_promotion(self) -> None:
        d = triage(
            self._firing(),
            self._firing(),
            {},
            lineage_children={"a": ["a-v4"]},
        )
        assert d[0]["decision"] == "descendant_exists"
        assert d[0]["descendants"] == ["a-v4"]
        assert d[0]["issue_open"] is False  # not a fix issue

    def test_no_descendant_opens_normally(self) -> None:
        d = triage(self._firing(), self._firing(), {}, lineage_children={})
        assert d[0]["decision"] == "open_issue"

    def test_descendant_check_is_after_no_trigger(self) -> None:
        """A tool that does not fire is silent even if it has descendants."""
        d = triage(
            _scores(a=_stats(brier=0.20, brier_skill_score=0.10)),
            _scores(a=_stats(brier=0.20, brier_skill_score=0.10)),
            {},
            lineage_children={"a": ["a-v4"]},
        )
        assert d[0]["decision"] == "silent"

    def test_regression_with_descendant_also_routed(self) -> None:
        """A regression-triggered fire with a descendant is also promotion-routed."""
        d = triage(
            _scores(a=_stats(brier=0.260, log_loss=0.700, brier_skill_score=0.20)),
            _scores(a=_stats(brier=0.210, log_loss=0.610, brier_skill_score=0.20)),
            {},
            lineage_children={"a": ["a-v4"]},
        )
        assert d[0]["decision"] == "descendant_exists"
        assert d[0]["trigger"] == "regression"


class TestLineageLoader:
    """tool_lineage.json parsing into parent -> children."""

    def test_children_map(self, tmp_path: Path) -> None:
        lineage = {
            "tools": {
                "v1": {"parent": None},
                "v2": {"parent": "v1"},
                "v4": {"parent": "v1"},
                "orphan": {"parent": "other"},
            }
        }
        p = tmp_path / "tool_lineage.json"
        p.write_text(json.dumps(lineage), encoding="utf-8")
        children = _load_lineage_children(p)
        assert sorted(children["v1"]) == ["v2", "v4"]
        assert children["other"] == ["orphan"]
        assert "v2" not in children  # leaf, no children

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert _load_lineage_children(tmp_path / "nope.json") == {}


class TestPromotionNote:
    """The visible promotion-review note format."""

    def test_title_and_body(self) -> None:
        title = build_promotion_title("superforcaster-polymarket-v1", "polymarket")
        assert title.startswith("[tool-promotion-review]")
        assert "`superforcaster-polymarket-v1`" in title
        assert "on polymarket" in title
        body = build_promotion_body(
            {
                "tool": "superforcaster-polymarket-v1",
                "trigger": "level_floor",
                "brier_cur": 0.3679,
                "bss_cur": -0.26,
                "descendants": ["superforcaster-polymarket-v4"],
            },
            "polymarket",
        )
        assert "not a new fix request" in body.lower()
        assert "`superforcaster-polymarket-v4`" in body
        assert "BSS-vs-market" in body
        # Must NOT invoke the coding agent.
        assert "@valory-coding-agent" not in body
        assert PROMOTION_LABEL == "tool-promotion-review"
