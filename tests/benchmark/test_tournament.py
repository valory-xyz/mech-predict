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
"""Tests for benchmark/tournament.py"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from benchmark.tournament import (
    TOOL_REGISTRY,
    _make_row_id,
    build_output_row,
    load_existing_row_ids,
    load_markets,
    run_single,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market(
    market_id: str = "omen_0xabc",
    question: str = "Will X happen?",
    platform: str = "omen",
    prob: float = 0.65,
    close_date: str | None = None,
    category: str = "politics",
) -> dict[str, Any]:
    return {
        "id": market_id,
        "market_address": "0xabc",
        "platform": platform,
        "question_text": question,
        "current_prob": prob,
        "close_date": close_date,
        "category": category,
    }


def _run_result(
    p_yes: float = 0.72,
    p_no: float = 0.28,
    status: str = "valid",
    latency: float = 12.5,
    source_content: dict | None = None,
) -> dict[str, Any]:
    return {
        "p_yes": p_yes,
        "p_no": p_no,
        "confidence": 0.8,
        "prediction_parse_status": status,
        "latency_s": latency,
        "error": None,
        "source_content": source_content,
    }


# ---------------------------------------------------------------------------
# _make_row_id
# ---------------------------------------------------------------------------


class TestMakeRowId:
    def test_deterministic(self) -> None:
        id1 = _make_row_id("tool-a", "market_1", "omen", "model-1")
        id2 = _make_row_id("tool-a", "market_1", "omen", "model-1")
        assert id1 == id2

    def test_different_tools(self) -> None:
        id1 = _make_row_id("tool-a", "market_1", "omen", "model-1")
        id2 = _make_row_id("tool-b", "market_1", "omen", "model-1")
        assert id1 != id2

    def test_different_markets_same_question(self) -> None:
        """Two markets with same question but different IDs get different row IDs."""
        id1 = _make_row_id("tool-a", "omen_0x1", "omen", "model-1")
        id2 = _make_row_id("tool-a", "poly_abc", "polymarket", "model-1")
        assert id1 != id2

    def test_different_platforms_same_market_id(self) -> None:
        id1 = _make_row_id("tool-a", "0xabc", "omen", "model-1")
        id2 = _make_row_id("tool-a", "0xabc", "polymarket", "model-1")
        assert id1 != id2

    def test_prefix(self) -> None:
        row_id = _make_row_id("prediction-online", "m1", "omen", "m")
        assert row_id.startswith("tourn_prediction-online_")


# ---------------------------------------------------------------------------
# build_output_row
# ---------------------------------------------------------------------------


class TestBuildOutputRow:
    def test_basic_row(self) -> None:
        market = _market()
        result = _run_result()
        row = build_output_row(market, "prediction-online", "gpt-4.1", result)

        assert row["mode"] == "tournament"
        assert row["final_outcome"] is None
        assert row["p_yes"] == 0.72
        assert row["market_prob_at_prediction"] == 0.65
        assert row["platform"] == "omen"
        assert row["tool_name"] == "prediction-online"
        assert row["schema_version"] == "1.0"

    def test_stores_source_content(self) -> None:
        market = _market()
        sc = {"pages": {"http://example.com": "<html>...</html>"}}
        result = _run_result(source_content=sc)
        row = build_output_row(market, "tool", "model", result)
        assert row["source_content"] == sc

    def test_none_source_content(self) -> None:
        market = _market()
        result = _run_result(source_content=None)
        row = build_output_row(market, "tool", "model", result)
        assert row["source_content"] is None

    def test_error_result(self) -> None:
        market = _market()
        result = _run_result(p_yes=None, p_no=None, status="error")
        row = build_output_row(market, "tool", "model", result)
        assert row["prediction_parse_status"] == "error"
        assert row["p_yes"] is None
        assert row["final_outcome"] is None


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


class TestJsonlIO:
    def test_load_markets(self, tmp_path: Path) -> None:
        f = tmp_path / "markets.jsonl"
        f.write_text(
            json.dumps(_market("m1")) + "\n" + json.dumps(_market("m2")) + "\n"
        )
        markets = load_markets(f)
        assert len(markets) == 2
        assert markets[0]["id"] == "m1"

    def test_load_existing_row_ids_valid_only(self, tmp_path: Path) -> None:
        f = tmp_path / "predictions.jsonl"
        f.write_text(
            '{"row_id": "tourn_a_123", "prediction_parse_status": "valid"}\n'
            '{"row_id": "tourn_b_456", "prediction_parse_status": "malformed"}\n'
            '{"row_id": "tourn_c_789", "prediction_parse_status": "valid"}\n'
        )
        ids = load_existing_row_ids(f)
        assert ids == {"tourn_a_123", "tourn_c_789"}

    def test_load_existing_skips_errors(self, tmp_path: Path) -> None:
        f = tmp_path / "predictions.jsonl"
        f.write_text(
            '{"row_id": "tourn_a_1", "prediction_parse_status": "error"}\n'
            '{"row_id": "tourn_b_2", "prediction_parse_status": "timeout"}\n'
        )
        ids = load_existing_row_ids(f)
        assert ids == set()

    def test_load_existing_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "predictions.jsonl"
        assert load_existing_row_ids(f) == set()


# ---------------------------------------------------------------------------
# run_single (mocked — no API keys)
# ---------------------------------------------------------------------------


class TestRunSingle:
    """Tests for run_single with mocked tool execution."""

    @patch("benchmark.tournament.load_tool_run")
    def test_valid_result(self, mock_load: MagicMock) -> None:
        mock_fn = MagicMock(
            return_value=(
                '{"p_yes": 0.7, "p_no": 0.3, "confidence": 0.8}',
                None,
                None,
                None,  # counter_callback
                {"source_content": {"pages": {"http://x.com": "<html>"}}},
            )
        )
        mock_load.return_value = mock_fn

        from packages.valory.skills.task_execution.utils.apis import KeyChain

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("prediction-online", "Will X?", "gpt-4.1", keys)

        assert result["prediction_parse_status"] == "valid"
        assert result["p_yes"] == 0.7
        assert result["source_content"] == {"pages": {"http://x.com": "<html>"}}
        assert result["error"] is None

    @patch("benchmark.tournament.load_tool_run")
    def test_tool_exception(self, mock_load: MagicMock) -> None:
        mock_fn = MagicMock(side_effect=RuntimeError("API down"))
        mock_load.return_value = mock_fn

        from packages.valory.skills.task_execution.utils.apis import KeyChain

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("prediction-online", "Will X?", "gpt-4.1", keys)

        assert result["prediction_parse_status"] == "error"
        assert result["p_yes"] is None
        assert "API down" in result["error"]
        assert result["source_content"] is None

    @patch("benchmark.tournament.load_tool_run")
    def test_malformed_response(self, mock_load: MagicMock) -> None:
        mock_fn = MagicMock(return_value=("not json", None, None, {}))
        mock_load.return_value = mock_fn

        from packages.valory.skills.task_execution.utils.apis import KeyChain

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("prediction-online", "Will X?", "gpt-4.1", keys)

        assert result["prediction_parse_status"] != "valid"


# ---------------------------------------------------------------------------
# run_tournament integration (mocked tools, no API keys)
# ---------------------------------------------------------------------------


class TestRunTournament:
    """Integration test for the main tournament loop."""

    @patch("benchmark.tournament.build_keychain")
    @patch("benchmark.tournament.run_single")
    def test_writes_predictions(
        self,
        mock_run: MagicMock,
        mock_keys: MagicMock,
        tmp_path: Path,
    ) -> None:
        from benchmark.tournament import run_tournament

        mock_keys.return_value = MagicMock()
        mock_run.return_value = _run_result()

        markets_path = tmp_path / "markets.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        markets_path.write_text(
            json.dumps(_market("omen_0x1", "Will A?"))
            + "\n"
            + json.dumps(_market("omen_0x2", "Will B?"))
            + "\n"
        )

        run_tournament(markets_path, output_path, ["prediction-online"], "gpt-4.1")

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert row["mode"] == "tournament"
        assert row["final_outcome"] is None
        assert row["p_yes"] == 0.72

    @patch("benchmark.tournament.build_keychain")
    @patch("benchmark.tournament.run_single")
    def test_skips_valid_dedup(
        self,
        mock_run: MagicMock,
        mock_keys: MagicMock,
        tmp_path: Path,
    ) -> None:
        from benchmark.tournament import run_tournament

        mock_keys.return_value = MagicMock()
        mock_run.return_value = _run_result()

        markets_path = tmp_path / "markets.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        markets_path.write_text(json.dumps(_market("omen_0x1")) + "\n")

        # First run
        run_tournament(markets_path, output_path, ["prediction-online"], "gpt-4.1")
        assert mock_run.call_count == 1

        # Second run — should skip (valid prediction exists)
        mock_run.reset_mock()
        run_tournament(markets_path, output_path, ["prediction-online"], "gpt-4.1")
        assert mock_run.call_count == 0

    @patch("benchmark.tournament.build_keychain")
    @patch("benchmark.tournament.run_single")
    def test_retries_malformed(
        self,
        mock_run: MagicMock,
        mock_keys: MagicMock,
        tmp_path: Path,
    ) -> None:
        from benchmark.tournament import run_tournament

        mock_keys.return_value = MagicMock()

        markets_path = tmp_path / "markets.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        markets_path.write_text(json.dumps(_market("omen_0x1")) + "\n")

        # First run — malformed result
        mock_run.return_value = _run_result(p_yes=None, p_no=None, status="malformed")
        run_tournament(markets_path, output_path, ["prediction-online"], "gpt-4.1")
        assert mock_run.call_count == 1

        # Second run — should retry (malformed not skipped)
        mock_run.reset_mock()
        mock_run.return_value = _run_result()
        run_tournament(markets_path, output_path, ["prediction-online"], "gpt-4.1")
        assert mock_run.call_count == 1
