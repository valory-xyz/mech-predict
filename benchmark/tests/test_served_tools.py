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
"""Tests for benchmark/served_tools.py (no network: fetches are injected)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from benchmark import served_tools
from benchmark.tool_improvement_triage import triage


class TestRepoTools:
    """The packages.json parse (the shipping source of truth)."""

    def test_finds_shipped_packages(self) -> None:
        """Parsing packages.json yields custom/ package names."""
        tools = served_tools.repo_tools()
        assert "superforcaster_polymarket_v1" in tools
        assert "factual_research" in tools
        assert all(tool == tool.strip() for tool in tools)

    def test_ignores_non_custom_entries(self) -> None:
        """Only custom/ packages count (not agents, skills, protocols)."""
        tools = served_tools.repo_tools()
        assert not any("/" in tool for tool in tools)

    def test_unreadable_packages_json_is_empty(
        self, monkeypatch: Any, tmp_path: Any
    ) -> None:
        """A missing ledger yields an empty set rather than an exception."""
        monkeypatch.setattr(served_tools, "PACKAGES_JSON_PATH", tmp_path / "nope.json")
        assert served_tools.repo_tools() == set()


class TestServedAndActionable:
    """Intersection semantics with an injected selectable-tools map."""

    VALID: Dict[str, Optional[List[str]]] = {
        "omenstrat Pearl": ["factual_research", "superforcaster"],
        "polystrat Pearl": [
            "superforcaster-polymarket-v1",
            "resolve-market-reasoning-gpt-4.1",
        ],
    }

    def test_served_unions_all_deployments(self) -> None:
        """Without a platform filter every deployment contributes.

        Names come back normalized (underscores -> dashes), which is why
        ``factual_research`` is asserted in its dashed form here.
        """
        served = served_tools.served_tools(valid=self.VALID)
        assert "factual-research" in served
        assert "superforcaster-polymarket-v1" in served

    def test_platform_filter(self) -> None:
        """A platform filter keeps only that platform's deployments."""
        served = served_tools.served_tools("polymarket", self.VALID)
        assert "superforcaster-polymarket-v1" in served
        assert "factual-research" not in served

    def test_name_convention_is_normalized(self) -> None:
        """Underscore/dash spellings of one tool match."""
        served = served_tools.served_tools(valid=self.VALID)
        assert "factual-research" in served

    def test_actionable_is_the_intersection(self) -> None:
        """Actionable = selectable AND shipped here; both exclusions hold."""
        actionable = served_tools.actionable_tools(valid=self.VALID)
        # selectable and shipped here (package spelling is returned)
        assert "superforcaster_polymarket_v1" in actionable
        # selectable but shipped by another repo
        assert "resolve-market-reasoning-gpt-4.1" not in actionable
        # shipped here but not selectable anywhere
        assert "superforcaster_polymarket_v3" not in actionable

    def test_versions_are_distinct_not_collapsed(self) -> None:
        """A served -v1 must NOT make sibling -v3 look actionable."""
        actionable = served_tools.actionable_tools(valid=self.VALID)
        assert "factual_research_v1" not in actionable
        assert "factual_research_v3" not in actionable

    def test_failed_deployment_contributes_nothing(self) -> None:
        """A None (fetch failed) deployment is skipped, not treated as empty."""
        valid: Dict[str, Optional[List[str]]] = {
            "omenstrat Pearl": None,
            "polystrat Pearl": ["superforcaster-polymarket-v1"],
        }
        served = served_tools.served_tools(valid=valid)
        assert served == {"superforcaster-polymarket-v1"}

    def test_all_failed_yields_empty(self) -> None:
        """Everything unavailable -> empty actionable set (never a false 'none')."""
        valid: Dict[str, Optional[List[str]]] = {
            "omenstrat Pearl": None,
            "polystrat Pearl": None,
        }
        assert served_tools.actionable_tools(valid=valid) == set()


class TestTournamentRoster:
    """Roster read is failure-tolerant."""

    def test_missing_roster_is_empty(self, monkeypatch: Any, tmp_path: Any) -> None:
        """An absent roster file yields an empty set, not an exception."""
        monkeypatch.setattr(
            served_tools, "TOURNAMENT_TOOLS_PATH", tmp_path / "nope.json"
        )
        assert served_tools.tournament_tools() == set()


class TestTriageActionableFilter:
    """triage() honours the actionable set."""

    def test_non_actionable_tool_is_silent(self) -> None:
        """A tool outside the actionable set never reaches the triggers."""
        stats = {
            "n": 200,
            "valid_n": 200,
            "brier": 0.40,
            "log_loss": 1.2,
            "reliability": 0.98,
        }
        decisions = triage(
            {"by_tool": {"tournament-only-tool": stats}},
            {"by_tool": {"tournament-only-tool": {**stats, "brier": 0.20}}},
            {},
            actionable={"some-other-tool"},
        )
        assert decisions[0]["decision"] == "silent"
        assert decisions[0]["reason"] == "not_actionable"

    def test_actionable_tool_is_assessed(self) -> None:
        """A tool inside the set is evaluated normally (fires here)."""
        stats = {
            "n": 200,
            "valid_n": 200,
            "brier": 0.40,
            "log_loss": 1.2,
            "reliability": 0.98,
        }
        decisions = triage(
            {"by_tool": {"served-tool": stats}},
            {"by_tool": {"served-tool": {**stats, "brier": 0.20, "log_loss": 0.6}}},
            {},
            actionable={"served-tool"},
        )
        assert decisions[0]["decision"] == "open_issue"

    def test_none_means_assess_everything(self) -> None:
        """actionable=None (discovery failed) preserves the legacy behaviour."""
        stats = {
            "n": 200,
            "valid_n": 200,
            "brier": 0.40,
            "log_loss": 1.2,
            "reliability": 0.98,
        }
        decisions = triage(
            {"by_tool": {"anything": stats}},
            {"by_tool": {"anything": {**stats, "brier": 0.20, "log_loss": 0.6}}},
            {},
            actionable=None,
        )
        assert decisions[0]["decision"] == "open_issue"
