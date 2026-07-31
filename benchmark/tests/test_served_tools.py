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

import json
import sys
from typing import Any, Dict, List, Optional

from benchmark import served_tools


class TestRepoTools:
    """The packages.json parse (the shipping source of truth)."""

    @staticmethod
    def _ledger(tmp_path: Any, monkeypatch: Any, payload: dict) -> None:
        """Point repo_tools() at a synthetic packages.json.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        :param payload: ledger contents to write.
        """
        path = tmp_path / "packages.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(served_tools, "PACKAGES_JSON_PATH", path)

    def test_parses_custom_keys_across_authors(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Any author segment is accepted; the package name is the 3rd field.

        Pinned on a synthetic ledger rather than the live one: asserting that
        a specific tool ships today couples the parse test to the roster, so
        retiring a tool would break it for an unrelated reason.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        """
        self._ledger(
            tmp_path,
            monkeypatch,
            {
                "dev": {
                    "custom/valory/superforcaster_polymarket_v1/0.1.0": "bafy1",
                    "custom/napthaai/prediction_request_rag_v1/0.1.0": "bafy2",
                    "custom/some-other.author/odd_name_v9/1.2.3": "bafy3",
                },
                "third_party": {"custom/thirdparty/vendor_tool/0.1.0": "bafy4"},
            },
        )
        assert served_tools.repo_tools() == {
            "superforcaster_polymarket_v1",
            "prediction_request_rag_v1",
            "odd_name_v9",
            "vendor_tool",
        }

    def test_ignores_non_custom_keys_even_when_they_contain_custom(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Only keys STARTING with ``custom/`` count.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        """
        self._ledger(
            tmp_path,
            monkeypatch,
            {
                "dev": {
                    "agent/valory/custom_agent/0.1.0": "bafy1",
                    "skill/valory/some_custom/0.1.0": "bafy2",
                    "protocol/valory/acn/1.1.0": "bafy3",
                    "custom/valory/real_tool/0.1.0": "bafy4",
                }
            },
        )
        assert served_tools.repo_tools() == {"real_tool"}

    def test_null_and_absent_sections_are_tolerated(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """``dev: null`` and a missing ``third_party`` must not raise.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        """
        self._ledger(tmp_path, monkeypatch, {"dev": None})
        assert served_tools.repo_tools() == set()
        self._ledger(
            tmp_path, monkeypatch, {"third_party": {"custom/v/t/0.1.0": "bafy"}}
        )
        assert served_tools.repo_tools() == {"t"}

    def test_unreadable_packages_json_is_unknown(
        self, monkeypatch: Any, tmp_path: Any
    ) -> None:
        """An unreadable ledger yields None (unknown), not an empty set.

        :param monkeypatch: pytest monkeypatch fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        monkeypatch.setattr(served_tools, "PACKAGES_JSON_PATH", tmp_path / "nope.json")
        assert served_tools.repo_tools() is None


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
        assert served is not None
        assert "factual-research" in served
        assert "superforcaster-polymarket-v1" in served

    def test_platform_filter(self) -> None:
        """A platform filter keeps only that platform's deployments."""
        served = served_tools.served_tools("polymarket", self.VALID)
        assert served is not None
        assert "superforcaster-polymarket-v1" in served
        assert "factual-research" not in served

    def test_name_convention_is_normalized(self) -> None:
        """Underscore/dash spellings of one tool match."""
        served = served_tools.served_tools(valid=self.VALID)
        assert served is not None
        assert "factual-research" in served

    def test_actionable_is_the_intersection(self) -> None:
        """Actionable = selectable AND shipped here; both exclusions hold."""
        actionable = served_tools.actionable_tools(valid=self.VALID)
        assert actionable is not None
        # selectable and shipped here (package spelling is returned)
        assert "superforcaster_polymarket_v1" in actionable
        # selectable but shipped by another repo
        assert "resolve-market-reasoning-gpt-4.1" not in actionable
        # shipped here but not selectable anywhere
        assert "superforcaster_polymarket_v3" not in actionable

    def test_versions_are_distinct_not_collapsed(self) -> None:
        """A served -v1 must NOT make sibling -v3 look actionable."""
        actionable = served_tools.actionable_tools(valid=self.VALID)
        assert actionable is not None
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

    def test_all_failed_yields_unknown_not_empty(self) -> None:
        """Everything unavailable -> None (UNKNOWN), never a false 'nothing'.

        An empty set is a real answer meaning "this platform serves nothing
        we ship". A total discovery failure must not be spelled the same way:
        a consumer that treats them alike silences every tool on a
        rate-limited morning and reports it as a quiet day.
        """
        valid: Dict[str, Optional[List[str]]] = {
            "omenstrat Pearl": None,
            "polystrat Pearl": None,
        }
        assert served_tools.served_tools(valid=valid) is None
        assert served_tools.actionable_tools(valid=valid) is None

    def test_resolved_but_nothing_shipped_is_a_real_empty(self) -> None:
        """Discovery worked and nothing matched -> empty set, NOT None."""
        valid: Dict[str, Optional[List[str]]] = {
            "polystrat Pearl": ["some-tool-we-do-not-ship"],
        }
        assert served_tools.actionable_tools(valid=valid) == set()

    def test_unknown_propagates_from_an_unreadable_ledger(
        self, monkeypatch: Any, tmp_path: Any
    ) -> None:
        """An unreadable packages.json is UNKNOWN too, not 'we ship nothing'.

        :param monkeypatch: pytest monkeypatch fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        monkeypatch.setattr(
            served_tools, "PACKAGES_JSON_PATH", tmp_path / "missing.json"
        )
        valid: Dict[str, Optional[List[str]]] = {
            "polystrat Pearl": ["superforcaster-polymarket-v1"],
        }
        assert served_tools.actionable_tools(valid=valid) is None

    def test_platform_scope_ignores_other_platforms_failures(self) -> None:
        """A resolved omen deployment must not mask a failed polymarket one."""
        valid: Dict[str, Optional[List[str]]] = {
            "omenstrat Pearl": ["superforcaster"],
            "polystrat Pearl": None,
        }
        assert served_tools.served_tools("omen", valid) == {"superforcaster"}
        assert served_tools.served_tools("polymarket", valid) is None


class TestTournamentRoster:
    """Roster read is failure-tolerant."""

    def test_missing_roster_is_empty(self, monkeypatch: Any, tmp_path: Any) -> None:
        """An absent roster file yields an empty set, not an exception."""
        monkeypatch.setattr(
            served_tools, "TOURNAMENT_TOOLS_PATH", tmp_path / "nope.json"
        )
        assert served_tools.tournament_tools() == set()


class TestMainCli:
    """The CLI: 65 lines that had no test, and where two bugs lived."""

    @staticmethod
    def _run(
        monkeypatch: Any,
        capsys: Any,
        tmp_path: Any,
        argv: list,
        valid: Dict[str, Optional[List[str]]],
        ledger: Optional[dict] = None,
        roster: Optional[dict] = None,
    ) -> tuple:
        """Run main() against injected discovery, ledger and roster.

        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        :param tmp_path: pytest tmp_path fixture.
        :param argv: argv tail (without the program name).
        :param valid: injected fetch_valid_tools() result.
        :param ledger: synthetic packages.json payload.
        :param roster: synthetic tournament_tools.json payload.
        :return: (exit_code, stdout).
        """
        monkeypatch.setattr(served_tools, "fetch_valid_tools", lambda: valid)
        pkg = tmp_path / "packages.json"
        pkg.write_text(
            json.dumps(ledger if ledger is not None else {"dev": {}}), encoding="utf-8"
        )
        monkeypatch.setattr(served_tools, "PACKAGES_JSON_PATH", pkg)
        ros = tmp_path / "tournament_tools.json"
        ros.write_text(json.dumps(roster or {}), encoding="utf-8")
        monkeypatch.setattr(served_tools, "TOURNAMENT_TOOLS_PATH", ros)
        monkeypatch.setattr(sys, "argv", ["served_tools"] + argv)
        code = served_tools.main()
        return code, capsys.readouterr().out

    def test_roster_tag_survives_the_naming_split(
        self, monkeypatch: Any, capsys: Any, tmp_path: Any
    ) -> None:
        """A tool spelled with '_' here and '-' on the roster is still tagged.

        The real spelling pair: packages.json says ``factual_research_v2``,
        tournament_tools.json says ``factual_research-v2``. Comparing them
        raw can never match, so the tag was unreachable for every versioned
        tool on the roster.

        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        code, out = self._run(
            monkeypatch,
            capsys,
            tmp_path,
            [],
            {"polystrat Pearl": ["factual-research-v2"]},
            ledger={"dev": {"custom/valory/factual_research_v2/0.1.0": "bafy"}},
            roster={"factual_research-v2": "bafyroster"},
        )
        assert code == 0
        assert "[also in tournament]" in out

    def test_tournament_only_is_not_reported_as_shipped_only(
        self, monkeypatch: Any, capsys: Any, tmp_path: Any
    ) -> None:
        """A shipped, rostered, unserved tool reads 'tournament only'.

        'shipped only' vs 'tournament only' is what a deployment-review
        reader uses to decide whether a tool already has a tournament slot;
        printing it backwards actively misleads them into re-adding it.

        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        _, out = self._run(
            monkeypatch,
            capsys,
            tmp_path,
            [],
            {"polystrat Pearl": ["something-else"]},
            ledger={"dev": {"custom/valory/factual_research_v2/0.1.0": "bafy"}},
            roster={"factual_research-v2": "bafyroster"},
        )
        assert "factual_research_v2   (tournament only" in out
        assert "factual_research_v2   (shipped only" not in out

    def test_roster_only_tools_appear_in_the_text_report(
        self, monkeypatch: Any, capsys: Any, tmp_path: Any
    ) -> None:
        """Rostered tools that ship no package must not be silently omitted.

        These are the motivating '#416 class' -- they are neither in
        ``repo - actionable`` nor in ``served - repo``, so they fell through
        both loops and showed up only under --json.

        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        _, out = self._run(
            monkeypatch,
            capsys,
            tmp_path,
            [],
            {"polystrat Pearl": ["superforcaster-polymarket-v1"]},
            ledger={"dev": {"custom/valory/superforcaster_polymarket_v1/0.1.0": "b"}},
            roster={"predict-fine-tuned-calibrated": "bafy"},
        )
        assert "predict-fine-tuned-calibrated" in out
        assert "ships no package here" in out

    def test_exit_code_is_scoped_to_the_requested_platform(
        self, monkeypatch: Any, capsys: Any, tmp_path: Any
    ) -> None:
        """`--platform omen` must exit 1 when omen failed, even if polymarket worked.

        fetch_valid_tools() always resolves both, so judging success over the
        unfiltered map let a total failure of the REQUESTED platform exit 0 --
        and anything gating on the exit code then consumed an empty actionable
        list as ground truth.

        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        valid: Dict[str, Optional[List[str]]] = {
            "omenstrat Pearl": None,
            "polystrat Pearl": ["superforcaster-polymarket-v1"],
        }
        code_omen, _ = self._run(
            monkeypatch, capsys, tmp_path, ["--platform", "omen"], valid
        )
        assert code_omen == 1
        code_pm, _ = self._run(
            monkeypatch, capsys, tmp_path, ["--platform", "polymarket"], valid
        )
        assert code_pm == 0

    def test_exit_code_when_everything_fails(
        self, monkeypatch: Any, capsys: Any, tmp_path: Any
    ) -> None:
        """All deployments unavailable -> exit 1 and an explicit UNKNOWN.

        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        code, out = self._run(
            monkeypatch,
            capsys,
            tmp_path,
            [],
            {"omenstrat Pearl": None, "polystrat Pearl": None},
        )
        assert code == 1
        assert "UNKNOWN" in out

    def test_json_schema_and_null_for_unresolved(
        self, monkeypatch: Any, capsys: Any, tmp_path: Any
    ) -> None:
        """--json keys are stable and an unresolved deployment emits null.

        Inverting the ``sorted(tools) if tools is not None else None``
        conditional would emit null for HEALTHY deployments and TypeError on
        the failed one; nothing pinned it.

        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        code, out = self._run(
            monkeypatch,
            capsys,
            tmp_path,
            ["--json"],
            {"omenstrat Pearl": None, "polystrat Pearl": ["superforcaster"]},
            ledger={"dev": {"custom/valory/superforcaster/0.1.0": "bafy"}},
        )
        payload = json.loads(out)
        assert set(payload) == {
            "selectable_by_deployment",
            "repo_tools",
            "tournament_roster",
            "actionable",
        }
        assert payload["selectable_by_deployment"]["omenstrat Pearl"] is None
        assert payload["selectable_by_deployment"]["polystrat Pearl"] == [
            "superforcaster"
        ]
        assert payload["actionable"] == ["superforcaster"]
        assert code == 0

    def test_json_reports_unknown_as_null_not_empty(
        self, monkeypatch: Any, capsys: Any, tmp_path: Any
    ) -> None:
        """Total failure renders actionable as null, never [].

        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        :param tmp_path: pytest tmp_path fixture.
        """
        code, out = self._run(
            monkeypatch,
            capsys,
            tmp_path,
            ["--json"],
            {"omenstrat Pearl": None, "polystrat Pearl": None},
        )
        assert json.loads(out)["actionable"] is None
        assert code == 1
