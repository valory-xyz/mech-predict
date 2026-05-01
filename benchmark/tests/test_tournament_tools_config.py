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
"""Tests for benchmark/tournament.py:load_tournament_tools."""

import json
from pathlib import Path

import pytest

from benchmark.tournament import TOURNAMENT_TOOLS_JSON, load_tournament_tools


def test_loads_valid_file(tmp_path: Path) -> None:
    """A well-formed dict[str, str] loads as-is."""
    f = tmp_path / "tournament_tools.json"
    f.write_text(
        json.dumps({"superforcaster": "bafycid1", "factual_research": "bafycid2"})
    )
    result = load_tournament_tools(f)
    assert result == {"superforcaster": "bafycid1", "factual_research": "bafycid2"}


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    """Missing config is loud, not silent."""
    f = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_tournament_tools(f)
    assert "Tournament cannot run without it" in str(exc_info.value)


def test_malformed_json_raises_valueerror(tmp_path: Path) -> None:
    """Invalid JSON syntax → ValueError, not silent fallback."""
    f = tmp_path / "tournament_tools.json"
    f.write_text("not json {{{")
    with pytest.raises(ValueError) as exc_info:
        load_tournament_tools(f)
    assert "valid JSON" in str(exc_info.value)


def test_non_dict_raises_valueerror(tmp_path: Path) -> None:
    """A JSON list / string / number is not a valid config."""
    f = tmp_path / "tournament_tools.json"
    f.write_text(json.dumps(["just", "a", "list"]))
    with pytest.raises(ValueError) as exc_info:
        load_tournament_tools(f)
    assert "JSON object" in str(exc_info.value)


def test_non_string_value_raises_valueerror(tmp_path: Path) -> None:
    """Values must be strings (CIDs), not e.g. dicts."""
    f = tmp_path / "tournament_tools.json"
    f.write_text(json.dumps({"superforcaster": {"nested": "bafy"}}))
    with pytest.raises(ValueError) as exc_info:
        load_tournament_tools(f)
    assert "str→str" in str(exc_info.value)


def test_empty_value_raises_valueerror(tmp_path: Path) -> None:
    """Empty CID is not allowed."""
    f = tmp_path / "tournament_tools.json"
    f.write_text(json.dumps({"superforcaster": ""}))
    with pytest.raises(ValueError) as exc_info:
        load_tournament_tools(f)
    assert "empty" in str(exc_info.value)


def test_empty_dict_is_valid(tmp_path: Path) -> None:
    """An empty dict is structurally valid (no tools to run)."""
    f = tmp_path / "tournament_tools.json"
    f.write_text("{}")
    assert load_tournament_tools(f) == {}


def test_shipped_file_loads_and_is_nonempty() -> None:
    """The committed benchmark/tournament_tools.json is well-formed."""
    result = load_tournament_tools(TOURNAMENT_TOOLS_JSON)
    assert isinstance(result, dict)
    assert len(result) > 0
    # Sanity-check shape: keys are tool names, values are CID-shaped strings
    for tool_name, cid in result.items():
        assert isinstance(tool_name, str) and tool_name
        assert isinstance(cid, str) and cid.startswith("bafy")
