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
"""Tests for benchmark/datasets/fetch_open.py"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from benchmark.datasets.fetch_open import (
    append_jsonl,
    fetch_omen_open,
    fetch_polymarket_open,
    load_existing_ids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _omen_fpmm(
    market_id: str = "0xabc",
    title: str = "Will X happen?",
    prices: list[str] | None = None,
    outcomes: list[str] | None = None,
    volume: str = "100.0",
    liquidity: str = "50.0",
    category: str = "politics",
) -> dict[str, Any]:
    """Build a mock Omen fixedProductMarketMaker response."""
    return {
        "id": market_id,
        "title": title,
        "outcomes": outcomes or ["Yes", "No"],
        "outcomeTokenMarginalPrices": prices or ["0.65", "0.35"],
        "usdVolume": volume,
        "usdLiquidityMeasure": liquidity,
        "creationTimestamp": "1711900000",
        "openingTimestamp": "1711900000",
        "category": category,
    }


def _poly_market(
    condition_id: str = "cid_123",
    question: str = "Will Y happen?",
    outcomes: str = '["Yes", "No"]',
    prices: str = '["0.55", "0.45"]',
    liquidity: float = 5000.0,
    volume: float = 10000.0,
    neg_risk: bool = False,
    end_date: str = "2026-06-01T00:00:00Z",
) -> dict[str, Any]:
    return {
        "conditionId": condition_id,
        "question": question,
        "outcomes": outcomes,
        "outcomePrices": prices,
        "liquidity": liquidity,
        "volume": volume,
        "negRisk": neg_risk,
        "endDate": end_date,
        "createdAt": "2026-03-01T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# fetch_omen_open
# ---------------------------------------------------------------------------


class TestFetchOmenOpen:
    """Tests for Omen open market fetching."""

    @patch("benchmark.datasets.fetch_open._post_graphql")
    def test_basic_fetch(self, mock_gql: MagicMock) -> None:
        mock_gql.return_value = {
            "fixedProductMarketMakers": [
                _omen_fpmm("0x001", "Will BTC hit 100k?"),
                _omen_fpmm("0x002", "Will ETH merge?"),
            ]
        }

        markets = fetch_omen_open(max_markets=10)
        assert len(markets) == 2
        assert markets[0]["id"] == "omen_0x001"
        assert markets[0]["platform"] == "omen"
        assert markets[0]["question_text"] == "Will BTC hit 100k?"
        assert markets[0]["current_prob"] == 0.65

    @patch("benchmark.datasets.fetch_open._post_graphql")
    def test_skips_non_binary(self, mock_gql: MagicMock) -> None:
        fpmm = _omen_fpmm()
        fpmm["outcomes"] = ["A", "B", "C"]  # Not binary
        mock_gql.return_value = {"fixedProductMarketMakers": [fpmm]}

        markets = fetch_omen_open(max_markets=10)
        assert len(markets) == 0

    @patch("benchmark.datasets.fetch_open._post_graphql")
    def test_skips_empty_title(self, mock_gql: MagicMock) -> None:
        fpmm = _omen_fpmm()
        fpmm["title"] = ""
        mock_gql.return_value = {"fixedProductMarketMakers": [fpmm]}

        markets = fetch_omen_open(max_markets=10)
        assert len(markets) == 0

    @patch("benchmark.datasets.fetch_open._post_graphql")
    def test_max_markets_limit(self, mock_gql: MagicMock) -> None:
        fpmms = [_omen_fpmm(f"0x{i:03x}", f"Q{i}") for i in range(20)]
        mock_gql.return_value = {"fixedProductMarketMakers": fpmms}

        markets = fetch_omen_open(max_markets=5)
        assert len(markets) == 5

    @patch("benchmark.datasets.fetch_open._post_graphql")
    def test_missing_prices_sets_prob_none(self, mock_gql: MagicMock) -> None:
        fpmm = _omen_fpmm()
        fpmm["outcomeTokenMarginalPrices"] = None
        mock_gql.return_value = {"fixedProductMarketMakers": [fpmm]}

        markets = fetch_omen_open(max_markets=10)
        assert len(markets) == 1
        assert markets[0]["current_prob"] is None


# ---------------------------------------------------------------------------
# fetch_polymarket_open
# ---------------------------------------------------------------------------


class TestFetchPolymarketOpen:
    """Tests for Polymarket open market fetching."""

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_basic_fetch(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_tag.return_value = 42

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [_poly_market()]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        markets = fetch_polymarket_open(max_markets=10)
        assert len(markets) >= 1
        m = markets[0]
        assert m["id"] == "poly_cid_123"
        assert m["platform"] == "polymarket"
        assert m["question_text"] == "Will Y happen?"
        assert m["current_prob"] == 0.55

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_skips_neg_risk(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_tag.return_value = 42

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [_poly_market(neg_risk=True)]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        markets = fetch_polymarket_open(max_markets=10)
        assert len(markets) == 0

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_skips_resolved(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        """Market with price >= 0.99 is effectively resolved."""
        mock_tag.return_value = 42

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            _poly_market(prices='["1.0", "0.0"]')
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        markets = fetch_polymarket_open(max_markets=10)
        assert len(markets) == 0

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_liquidity_filter(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_tag.return_value = 42

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            _poly_market(condition_id="c1", liquidity=500),
            _poly_market(condition_id="c2", liquidity=2000),
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        markets = fetch_polymarket_open(max_markets=10, min_liquidity=1000)
        assert len(markets) == 1
        assert markets[0]["id"] == "poly_c2"


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


class TestJsonlIO:
    """Tests for JSONL loading and writing."""

    def test_load_existing_ids(self, tmp_path: Path) -> None:
        f = tmp_path / "markets.jsonl"
        f.write_text(
            '{"id": "omen_0x1"}\n'
            '{"id": "poly_abc"}\n'
        )
        ids = load_existing_ids(f)
        assert ids == {"omen_0x1", "poly_abc"}

    def test_load_existing_ids_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "markets.jsonl"
        assert load_existing_ids(f) == set()

    def test_append_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        append_jsonl(f, [{"id": "a"}, {"id": "b"}])
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "a"

    def test_append_preserves_existing(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text('{"id": "existing"}\n')
        append_jsonl(f, [{"id": "new"}])
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 2
