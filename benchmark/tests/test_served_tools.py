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


class TestRepoTools:
    """The registry parse."""

    def test_finds_known_tools(self) -> None:
        """Parsing tools.py yields the registered tool names."""
        tools = served_tools.repo_tools()
        assert "superforcaster-polymarket-v1" in tools
        assert "factual_research" in tools
        assert all(tool == tool.strip() for tool in tools)


class TestServedAndActionable:
    """Intersection semantics with an injected selectable-tools map."""

    VALID: Dict[str, Optional[List[str]]] = {
        "omenstrat Pearl": ["factual_research", "superforcaster"],
        "polystrat Pearl": [
            "superforcaster-polymarket-v1",
            "superforcaster_full_search",
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
        assert "superforcaster-full-search" in served

    def test_actionable_is_the_intersection(self) -> None:
        """Actionable = selectable AND buildable here; both exclusions hold."""
        actionable = served_tools.actionable_tools(valid=self.VALID)
        # selectable and in this repo
        assert "superforcaster-polymarket-v1" in actionable
        # selectable but built elsewhere
        assert "superforcaster_full_search" not in actionable
        # in this repo but not selectable anywhere
        assert "superforcaster-polymarket-v3" not in actionable

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
