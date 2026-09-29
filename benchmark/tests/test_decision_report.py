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
"""Complete roster and rule-evidence regression tests."""

import copy
from typing import Any

import pytest
from benchmark.decision_report import (
    build_decision_record,
    decision_markdown,
    decision_text,
)
from benchmark.digest_tables import WINDOW_FILES

NEW_TOOL = "superforcaster-market-aware-olas-predict-r1-14b"


def _stats(**changes: Any) -> dict[str, Any]:
    result = {
        "n": 100,
        "valid_n": 100,
        "brier": 0.20,
        "baseline_brier": 0.24,
        "edge": -0.01,
        "edge_n": 80,
        "edge_sd": 0.05,
        "conditional_accuracy_rate": 0.60,
        "disagree_n": 40,
        "reliability": 1.0,
    }
    result.update(changes)
    return result


def _payloads(rows: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"by_tool": copy.deepcopy(rows) if key != "tournament" else {}}
        for key in WINDOW_FILES
    }


def test_seventh_deployed_tool_is_not_filtered_by_execution_registry() -> None:
    """Reproduce seven deployed tools versus six registered tools."""
    rows = {f"tool-{i}": _stats(brier=0.32 if i < 4 else 0.20) for i in range(6)}
    rows[NEW_TOOL] = _stats()
    record = build_decision_record(
        "polymarket",
        _payloads(rows),
        {"status": "complete", "tools": list(rows)},
        list(rows)[:-1],
    )
    assert len(record["production"]) == 7
    assert record["production"][NEW_TOOL]["verdict"] == "keep"
    assert record["decision"]["token"] == "DEMOTE 4"
    assert record["counts"]["remaining_deployed"] == 3
    assert record["counts"]["remaining_assessed"] == 3
    assert "3 remain deployed" in decision_text(record)


def test_unscored_unknown_and_non_prediction_tools_are_accounted_for() -> None:
    """Lack of scores never silently removes a manifest member."""
    record = build_decision_record(
        "omen",
        _payloads({"known": _stats()}),
        {
            "status": "complete",
            "tools": ["known", "empty", "unknown", "utility"],
            "non_prediction_tools": ["utility"],
        },
        ["known", "empty"],
    )
    rows = record["production"]
    assert rows["empty"]["verdict"] == "no data"
    assert rows["unknown"]["classification"] == "unknown"
    assert rows["utility"]["classification"] == "non_prediction"
    assert record["counts"]["manifest_tools"] == 4
    assert record["counts"]["remaining_deployed"] == 3
    assert record["counts"]["remaining_unassessed"] == 2
    assert "unknown tool classification" in decision_text(record)
    assert "utility" in decision_markdown(record)


def test_aliases_match_across_windows_without_merging_duplicate_score_keys() -> None:
    """Only collisions in inputs consumed by the verdict prevent assessment."""
    payloads = _payloads({"tool_name": _stats()})
    payloads["w1"]["by_tool"] = {"tool-name": _stats(brier=0.30)}
    roster = {"status": "complete", "tools": ["tool-name", "tool_name"]}
    record = build_decision_record("omen", payloads, roster, [])
    assert len(record["production"]) == 1
    row = next(iter(record["production"].values()))
    assert row["evidence"]["w1"]["brier"] == 0.30
    assert row["evidence"]["at"]["brier"] == 0.20
    payloads["w1"]["by_tool"]["tool_name"] = _stats()
    record = build_decision_record("omen", payloads, roster, [])
    assert record["production"]["tool_name"]["verdict"] == "keep"
    assert record["production"]["tool_name"]["evidence"]["w1"]["brier"] is None
    payloads["at"]["by_tool"]["tool_name"]["brier"] = 0.31
    record = build_decision_record("omen", payloads, roster, [])
    assert record["production"]["tool_name"]["verdict"].startswith("ambiguous")
    assert record["counts"]["remaining_assessed"] == 0
    assert record["decision"]["token"] == "NO ACTION"


@pytest.mark.parametrize("window", ["at", "w1", "w2", "tournament"])
def test_alias_collision_is_local_to_tool_and_consumed_window(window: str) -> None:
    """Neither optional-window aliases nor an unrelated tool suppress verdicts."""
    rows = {
        "bad": _stats(brier=0.31),
        "good": _stats(),
        "alias-tool": _stats(brier=0.31),
    }
    payloads = _payloads(rows)
    payloads[window]["by_tool"].update(
        {"alias-tool": _stats(brier=0.31), "alias_tool": _stats(brier=0.33)}
    )
    record = build_decision_record(
        "omen", payloads, {"status": "complete", "tools": list(rows)}, list(rows)
    )
    assert "bad" in record["decision"]["demote"]
    row = record["production"]["alias-tool"]
    if window in ("at", "w1"):
        assert row["verdict"].startswith("ambiguous")
        assert row["evidence"][window]["brier"] is None
        assert record["counts"]["remaining_unassessed"] == 1
    else:
        assert "alias-tool" in record["decision"]["demote"]
    assert record["errors"] == []


def test_ambiguous_candidate_does_not_block_other_promotions() -> None:
    """Overlapping candidate aggregates are not merged or selected arbitrarily."""
    payloads = _payloads({"good": _stats()})
    payloads["tournament"]["by_tool"] = {
        name: _stats(edge=0.20) for name in ("alias-tool", "alias_tool", "candidate")
    }
    record = build_decision_record(
        "omen",
        payloads,
        {"status": "complete", "tools": ["good"]},
        ["alias-tool", "candidate"],
    )
    assert record["decision"]["promote"] == ["candidate"]
    assert record["tournament"]["alias-tool"]["evidence"]["tournament"]["brier"] is None


def test_failed_week_cannot_supply_a_survivor_but_valid_empty_week_keeps_policy() -> (
    None
):
    """An unavailable baseline confirmation cannot justify another removal."""
    rows = {
        "conditional": _stats(conditional_accuracy_rate=0.40),
        "baseline": _stats(brier=0.31),
    }
    payloads = _payloads(rows)
    payloads["w1"] = {}
    roster = {"status": "complete", "tools": list(rows)}
    record = build_decision_record("omen", payloads, roster, list(rows))
    assert record["production"]["conditional"]["verdict"].startswith("demote")
    assert record["production"]["baseline"]["verdict"].startswith("unavailable")
    assert record["decision"]["demote"] == []
    assert record["counts"]["remaining_assessed"] == 0
    payloads["w1"] = {"by_tool": {}}
    record = build_decision_record("omen", payloads, roster, list(rows))
    assert record["production"]["baseline"]["verdict"] == "keep"
    assert record["decision"]["demote"] == ["conditional"]


def test_conditional_demotion_and_floor_keep_do_not_require_week() -> None:
    """Existing earlier gates can establish a safe action without weekly data."""
    rows = {
        "conditional": _stats(conditional_accuracy_rate=0.40),
        "floor": _stats(edge=0.20, brier=0.31),
    }
    payloads = _payloads(rows)
    payloads["w1"] = {}
    record = build_decision_record(
        "omen", payloads, {"status": "complete", "tools": list(rows)}, list(rows)
    )
    assert record["decision"]["demote"] == ["conditional"]
    assert record["production"]["floor"]["verdict"] == "keep (floor ok)"


def test_roster_failure_suppresses_proposals_and_remaining_count() -> None:
    """An incomplete roster cannot establish a safe surviving cohort."""
    record = build_decision_record(
        "omen",
        _payloads({"bad": _stats(brier=0.30), "good": _stats()}),
        {"status": "unavailable", "tools": ["bad", "good"]},
        [],
    )
    assert record["decision"]["token"] == "DECISION UNAVAILABLE"
    assert record["decision"]["demote"] == []
    assert record["counts"]["remaining_deployed"] is None
    assert "remain deployed" not in decision_text(record)
    text = decision_text(record)
    assert "finding only; action blocked" in text
    assert "Brier 0.3000" in text and "baseline 0.2400" in text
    assert "Roster incomplete: 2 resolved manifest tools" in text
    assert "*Next step:*" not in text


def test_unassessed_tool_does_not_allow_removing_every_assessed_tool() -> None:
    """Preserve the existing no-last-assessed-forecaster guard."""
    record = build_decision_record(
        "omen",
        _payloads({"bad": _stats(brier=0.30)}),
        {"status": "complete", "tools": ["bad", "unknown"]},
        [],
    )
    assert record["decision"]["token"] == "NO ACTION"
    assert record["decision"]["demote"] == []
    assert record["production"]["bad"]["verdict"].startswith("demote")
    assert record["counts"]["remaining_deployed"] == 2
    assert record["counts"]["remaining_assessed"] == 0
    assert record["counts"]["remaining_flagged"] == 1
    assert record["counts"]["remaining_unassessed"] == 1
    assert "1 flagged but retained" in decision_text(record)
    assert "finding only; action blocked" in decision_text(record)
    assert "Brier 0.3000" in decision_text(record)


def test_baseline_reason_shows_both_gaps_and_correct_denominators() -> None:
    """Positive market Edge does not prevent a below-baseline finding."""
    payloads = _payloads({"bad": _stats(brier=0.31, edge=0.01), "good": _stats()})
    payloads["w1"]["by_tool"]["bad"].update(brier=0.34, valid_n=35, edge_n=31)
    record = build_decision_record(
        "omen", payloads, {"status": "complete", "tools": ["bad", "good"]}, []
    )
    row = record["production"]["bad"]
    assert row["rule"] == "sustained_baseline"
    text = decision_text(record)
    for evidence in (
        "Brier 0.3100",
        "baseline 0.2400",
        "gap +0.0700",
        "gap +0.1000",
        "scored n=35",
        "priced n=80",
    ):
        assert evidence in text
        assert evidence in decision_markdown(record)
    assert "Sustained no-skill: Edge" not in text


def test_conditional_rule_keeps_threshold_and_exposes_disagreement_count() -> None:
    """This PR exposes the 14/30 evidence without changing the existing rule."""
    record = build_decision_record(
        "omen",
        _payloads(
            {
                "bad": _stats(conditional_accuracy_rate=14 / 30, disagree_n=30),
                "good": _stats(),
            }
        ),
        {"status": "complete", "tools": ["bad", "good"]},
        [],
    )
    assert record["decision"]["token"] == "DEMOTE 1"
    text = decision_text(record)
    assert "46.7%" in text and "disagreements n=30" in text
    assert "Edge lower bound" in text and "priced n=80" in text
    assert "below 50%" in text
