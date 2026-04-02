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
"""Tests for benchmark/score_tournament.py"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from benchmark.score_tournament import (
    check_omen_resolutions,
    check_polymarket_resolutions,
    load_predictions,
    score_tournament,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prediction(
    row_id: str = "tourn_tool_abc",
    market_id: str = "omen_0xabc",
    market_address: str = "0xabc",
    platform: str = "omen",
    p_yes: float = 0.72,
    final_outcome: bool | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "schema_version": "1.0",
        "mode": "tournament",
        "market_id": market_id,
        "market_address": market_address,
        "platform": platform,
        "question_text": "Will X happen?",
        "tool_name": "prediction-online",
        "model": "gpt-4.1",
        "p_yes": p_yes,
        "p_no": 1 - p_yes,
        "prediction_parse_status": "valid",
        "confidence": 0.8,
        "market_prob_at_prediction": 0.65,
        "final_outcome": final_outcome,
        "predicted_at": "2026-03-15T10:00:00+00:00",
        "resolved_at": None,
        "latency_s": 12.5,
        "category": "politics",
        "source_content": None,
    }


# ---------------------------------------------------------------------------
# check_omen_resolutions
# ---------------------------------------------------------------------------


class TestCheckOmenResolutions:
    @patch("benchmark.score_tournament._post_graphql")
    def test_resolved_market(self, mock_gql: MagicMock) -> None:
        mock_gql.return_value = {
            "fixedProductMarketMakers": [
                {
                    "id": "0xabc",
                    "currentAnswer": "0x0000000000000000000000000000000000000000000000000000000000000000",
                    "currentAnswerTimestamp": "1712000000",
                    "outcomes": ["Yes", "No"],
                }
            ]
        }

        result = check_omen_resolutions(["0xabc"])
        assert "0xabc" in result
        assert result["0xabc"]["outcome"] is True  # index 0 = Yes
        assert result["0xabc"]["resolved_at"] is not None

    @patch("benchmark.score_tournament._post_graphql")
    def test_resolved_no(self, mock_gql: MagicMock) -> None:
        """currentAnswer=0x01 → outcome index 1 = No → False."""
        mock_gql.return_value = {
            "fixedProductMarketMakers": [
                {
                    "id": "0xdef",
                    "currentAnswer": "0x0000000000000000000000000000000000000000000000000000000000000001",
                    "currentAnswerTimestamp": "1712000000",
                    "outcomes": ["Yes", "No"],
                }
            ]
        }

        result = check_omen_resolutions(["0xdef"])
        assert result["0xdef"]["outcome"] is False

    @patch("benchmark.score_tournament._post_graphql")
    def test_unresolved_skipped(self, mock_gql: MagicMock) -> None:
        mock_gql.return_value = {
            "fixedProductMarketMakers": [
                {
                    "id": "0xabc",
                    "currentAnswer": None,
                    "currentAnswerTimestamp": None,
                    "outcomes": ["Yes", "No"],
                }
            ]
        }

        result = check_omen_resolutions(["0xabc"])
        assert len(result) == 0

    def test_empty_input(self) -> None:
        result = check_omen_resolutions([])
        assert result == {}


# ---------------------------------------------------------------------------
# check_polymarket_resolutions
# ---------------------------------------------------------------------------


class TestCheckPolymarketResolutions:
    @patch("benchmark.score_tournament.requests.get")
    def test_resolved_yes(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "conditionId": "cid_1",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["1.0", "0.0"]',
            }
        ]
        mock_get.return_value = mock_resp

        result = check_polymarket_resolutions(["cid_1"])
        assert "cid_1" in result
        assert result["cid_1"]["outcome"] is True

    @patch("benchmark.score_tournament.requests.get")
    def test_resolved_no(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "conditionId": "cid_2",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.0", "1.0"]',
            }
        ]
        mock_get.return_value = mock_resp

        result = check_polymarket_resolutions(["cid_2"])
        assert result["cid_2"]["outcome"] is False

    @patch("benchmark.score_tournament.requests.get")
    def test_unresolved_skipped(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "conditionId": "cid_3",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.55", "0.45"]',
            }
        ]
        mock_get.return_value = mock_resp

        result = check_polymarket_resolutions(["cid_3"])
        assert len(result) == 0

    def test_empty_input(self) -> None:
        result = check_polymarket_resolutions([])
        assert result == {}


# ---------------------------------------------------------------------------
# score_tournament (integration-style with mocked network)
# ---------------------------------------------------------------------------


class TestScoreTournament:
    """Tests for the full scoring pipeline."""

    @patch("benchmark.score_tournament.check_polymarket_resolutions")
    @patch("benchmark.score_tournament.check_omen_resolutions")
    def test_scores_resolved_markets(
        self,
        mock_omen: MagicMock,
        mock_poly: MagicMock,
        tmp_path: Path,
    ) -> None:
        pred_path = tmp_path / "predictions.jsonl"
        out_path = tmp_path / "scored.jsonl"

        # Two predictions for same market (different tools)
        p1 = _prediction(row_id="tourn_a_1", market_address="0xabc")
        p2 = _prediction(row_id="tourn_b_2", market_address="0xabc")
        pred_path.write_text(
            json.dumps(p1) + "\n" + json.dumps(p2) + "\n"
        )

        mock_omen.return_value = {
            "0xabc": {
                "outcome": True,
                "resolved_at": "2026-04-01T12:00:00+00:00",
            }
        }
        mock_poly.return_value = {}

        score_tournament(pred_path, out_path)

        # Both predictions should be scored
        scored = [json.loads(l) for l in out_path.read_text().strip().split("\n")]
        assert len(scored) == 2
        assert all(s["final_outcome"] is True for s in scored)
        assert all(s["resolved_at"] is not None for s in scored)

        # source_content should be stripped from scored output
        assert all("source_content" not in s for s in scored)

    @patch("benchmark.score_tournament.check_polymarket_resolutions")
    @patch("benchmark.score_tournament.check_omen_resolutions")
    def test_skips_already_resolved(
        self,
        mock_omen: MagicMock,
        mock_poly: MagicMock,
        tmp_path: Path,
    ) -> None:
        pred_path = tmp_path / "predictions.jsonl"
        out_path = tmp_path / "scored.jsonl"

        p = _prediction(final_outcome=True)  # Already resolved
        pred_path.write_text(json.dumps(p) + "\n")

        score_tournament(pred_path, out_path)

        # Nothing to check — all predictions already have outcomes
        mock_omen.assert_not_called()
        mock_poly.assert_not_called()

    @patch("benchmark.score_tournament.check_polymarket_resolutions")
    @patch("benchmark.score_tournament.check_omen_resolutions")
    def test_dedup_market_queries(
        self,
        mock_omen: MagicMock,
        mock_poly: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Multiple predictions for same market → only one resolution query."""
        pred_path = tmp_path / "predictions.jsonl"
        out_path = tmp_path / "scored.jsonl"

        preds = [
            _prediction(row_id=f"tourn_t{i}_{i}", market_address="0xsame")
            for i in range(5)
        ]
        pred_path.write_text("\n".join(json.dumps(p) for p in preds) + "\n")

        mock_omen.return_value = {}
        mock_poly.return_value = {}

        score_tournament(pred_path, out_path)

        # check_omen_resolutions receives deduplicated list
        call_args = mock_omen.call_args[0][0]
        assert call_args == ["0xsame"]  # Only one unique address

    @patch("benchmark.score_tournament.check_polymarket_resolutions")
    @patch("benchmark.score_tournament.check_omen_resolutions")
    def test_updates_predictions_file(
        self,
        mock_omen: MagicMock,
        mock_poly: MagicMock,
        tmp_path: Path,
    ) -> None:
        """After scoring, predictions file should be updated with outcomes."""
        pred_path = tmp_path / "predictions.jsonl"
        out_path = tmp_path / "scored.jsonl"

        p = _prediction(row_id="tourn_a_1", market_address="0xabc")
        pred_path.write_text(json.dumps(p) + "\n")

        mock_omen.return_value = {
            "0xabc": {
                "outcome": False,
                "resolved_at": "2026-04-01T12:00:00+00:00",
            }
        }
        mock_poly.return_value = {}

        score_tournament(pred_path, out_path)

        # Check predictions file was updated
        updated = json.loads(pred_path.read_text().strip())
        assert updated["final_outcome"] is False
        assert updated["resolved_at"] == "2026-04-01T12:00:00+00:00"

    @patch("benchmark.score_tournament.check_polymarket_resolutions")
    @patch("benchmark.score_tournament.check_omen_resolutions")
    def test_computes_lead_time(
        self,
        mock_omen: MagicMock,
        mock_poly: MagicMock,
        tmp_path: Path,
    ) -> None:
        pred_path = tmp_path / "predictions.jsonl"
        out_path = tmp_path / "scored.jsonl"

        p = _prediction(
            row_id="tourn_a_1",
            market_address="0xabc",
        )
        pred_path.write_text(json.dumps(p) + "\n")

        mock_omen.return_value = {
            "0xabc": {
                "outcome": True,
                "resolved_at": "2026-03-25T10:00:00+00:00",
            }
        }
        mock_poly.return_value = {}

        score_tournament(pred_path, out_path)

        scored = json.loads(out_path.read_text().strip())
        assert scored["prediction_lead_time_days"] == 10.0  # 10 days later


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------


class TestLoadPredictions:
    def test_load(self, tmp_path: Path) -> None:
        f = tmp_path / "preds.jsonl"
        f.write_text(
            json.dumps(_prediction("r1")) + "\n"
            + json.dumps(_prediction("r2")) + "\n"
        )
        rows = load_predictions(f)
        assert len(rows) == 2
        assert rows[0]["row_id"] == "r1"
