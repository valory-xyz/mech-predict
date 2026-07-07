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
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from benchmark.datasets.fetch_open import (
    POLYMARKET_MAX_PAGES_PER_CATEGORY,
    POLYMARKET_PAGE_LIMIT,
    fetch_omen_open,
    fetch_polymarket_open,
)
from benchmark.io import append_jsonl, load_existing_ids

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
    created_at: str = "2026-03-01T00:00:00Z",
) -> dict[str, Any]:
    """Build a mock Polymarket market response."""
    return {
        "conditionId": condition_id,
        "question": question,
        "outcomes": outcomes,
        "outcomePrices": prices,
        "liquidity": liquidity,
        "volume": volume,
        "negRisk": neg_risk,
        "endDate": end_date,
        "createdAt": created_at,
    }


def _poly_response(batch: list[dict[str, Any]]) -> MagicMock:
    """Wrap a market batch in a mock HTTP response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = batch
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _hours_ago(hours: float) -> str:
    """ISO timestamp ``hours`` before now (UTC, Z-suffixed)."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


# ---------------------------------------------------------------------------
# fetch_omen_open
# ---------------------------------------------------------------------------


class TestFetchOmenOpen:
    """Tests for Omen open market fetching."""

    @patch("benchmark.datasets.fetch_open._post_graphql")
    def test_basic_fetch(self, mock_gql: MagicMock) -> None:
        """Test basic Omen market fetch returns normalized records."""
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
        """Test that non-binary markets are skipped."""
        fpmm = _omen_fpmm()
        fpmm["outcomes"] = ["A", "B", "C"]  # Not binary
        mock_gql.return_value = {"fixedProductMarketMakers": [fpmm]}

        markets = fetch_omen_open(max_markets=10)
        assert len(markets) == 0

    @patch("benchmark.datasets.fetch_open._post_graphql")
    def test_skips_empty_title(self, mock_gql: MagicMock) -> None:
        """Test that markets with empty titles are skipped."""
        fpmm = _omen_fpmm()
        fpmm["title"] = ""
        mock_gql.return_value = {"fixedProductMarketMakers": [fpmm]}

        markets = fetch_omen_open(max_markets=10)
        assert len(markets) == 0

    @patch("benchmark.datasets.fetch_open._post_graphql")
    def test_max_markets_limit(self, mock_gql: MagicMock) -> None:
        """Test that max_markets parameter limits results."""
        fpmms = [_omen_fpmm(f"0x{i:03x}", f"Q{i}") for i in range(20)]
        mock_gql.return_value = {"fixedProductMarketMakers": fpmms}

        markets = fetch_omen_open(max_markets=5)
        assert len(markets) == 5

    @patch("benchmark.datasets.fetch_open._post_graphql")
    def test_missing_prices_sets_prob_none(self, mock_gql: MagicMock) -> None:
        """Test that missing prices result in None probability."""
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
    def test_basic_fetch(self, mock_tag: MagicMock, mock_get: MagicMock) -> None:
        """Test basic Polymarket fetch returns normalized records."""
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
    def test_skips_neg_risk(self, mock_tag: MagicMock, mock_get: MagicMock) -> None:
        """Test that negRisk markets are skipped."""
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
    def test_skips_resolved(self, mock_tag: MagicMock, mock_get: MagicMock) -> None:
        """Market with price >= 0.99 is effectively resolved."""
        mock_tag.return_value = 42

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [_poly_market(prices='["1.0", "0.0"]')]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        markets = fetch_polymarket_open(max_markets=10)
        assert len(markets) == 0

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_liquidity_filter(self, mock_tag: MagicMock, mock_get: MagicMock) -> None:
        """Test that markets below min_liquidity are filtered out."""
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

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_existing_ids_do_not_count_toward_cap(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        """Already-known markets are skipped without consuming max_markets slots."""
        mock_tag.return_value = 42
        mock_get.return_value = _poly_response(
            [
                _poly_market(condition_id="c1"),
                _poly_market(condition_id="c2"),
                _poly_market(condition_id="c3"),
            ]
        )

        markets = fetch_polymarket_open(max_markets=2, existing_ids={"poly_c1"})
        assert [m["id"] for m in markets] == ["poly_c2", "poly_c3"]

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_created_cutoff_skips_old_markets(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        """Markets older than the cutoff are skipped without hiding later new ones."""
        mock_tag.return_value = 42
        # One backdated entry must not mask new markets listed after it —
        # Gamma's newest-first ordering is not trusted strictly.
        mock_get.return_value = _poly_response(
            [
                _poly_market(condition_id="new", created_at=_hours_ago(1)),
                _poly_market(condition_id="old", created_at=_hours_ago(48)),
                _poly_market(condition_id="after_old", created_at=_hours_ago(2)),
            ]
        )

        markets = fetch_polymarket_open(max_markets=10, created_within_hours=24)
        assert [m["id"] for m in markets] == ["poly_new", "poly_after_old"]

    @patch("benchmark.datasets.fetch_open.POLYMARKET_CATEGORIES", ["business"])
    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_fully_stale_page_stops_category(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        """A full page entirely older than the cutoff ends the category scan."""
        mock_tag.return_value = 42
        stale_page = _poly_response(
            [
                _poly_market(condition_id=f"old_{i}", created_at=_hours_ago(48))
                for i in range(POLYMARKET_PAGE_LIMIT)
            ]
        )
        mock_get.side_effect = [stale_page]  # a second request would raise

        markets = fetch_polymarket_open(max_markets=10, created_within_hours=24)
        assert not markets
        assert mock_get.call_count == 1

    @patch("benchmark.datasets.fetch_open.POLYMARKET_CATEGORIES", ["business"])
    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_partially_stale_page_continues_pagination(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        """A full page with at least one recent market keeps paginating."""
        mock_tag.return_value = 42
        mixed_page = _poly_response(
            [_poly_market(condition_id="recent_0", created_at=_hours_ago(1))]
            + [
                _poly_market(condition_id=f"old_{i}", created_at=_hours_ago(48))
                for i in range(POLYMARKET_PAGE_LIMIT - 1)
            ]
        )
        next_page = _poly_response(
            [_poly_market(condition_id="recent_1", created_at=_hours_ago(2))]
        )
        mock_get.side_effect = [mixed_page, next_page]

        markets = fetch_polymarket_open(max_markets=10, created_within_hours=24)
        assert [m["id"] for m in markets] == ["poly_recent_0", "poly_recent_1"]
        assert mock_get.call_count == 2

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_created_cutoff_keeps_unparseable_timestamps(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        """Markets with missing/invalid createdAt are kept and don't stop the scan."""
        mock_tag.return_value = 42
        no_created = _poly_market(condition_id="undated", created_at="")
        mock_get.return_value = _poly_response(
            [
                no_created,
                _poly_market(condition_id="new", created_at=_hours_ago(1)),
            ]
        )

        markets = fetch_polymarket_open(max_markets=10, created_within_hours=24)
        assert [m["id"] for m in markets] == ["poly_undated", "poly_new"]

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_created_cutoff_handles_tz_naive_timestamps(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        """A createdAt without Z/offset is treated as UTC instead of raising TypeError."""
        mock_tag.return_value = 42
        naive_new = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%S"  # no Z suffix, no offset
        )
        naive_old = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        mock_get.return_value = _poly_response(
            [
                _poly_market(condition_id="naive_new", created_at=naive_new),
                _poly_market(condition_id="naive_old", created_at=naive_old),
            ]
        )

        markets = fetch_polymarket_open(max_markets=10, created_within_hours=24)
        assert [m["id"] for m in markets] == ["poly_naive_new"]

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_requests_newest_first_ordering(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        """The Gamma API is asked for newest-first ordering by creation time."""
        mock_tag.return_value = 42
        mock_get.return_value = _poly_response([_poly_market()])

        fetch_polymarket_open(max_markets=10)
        params = mock_get.call_args.kwargs["params"]
        assert params["order"] == "createdAt"
        assert params["ascending"] == "false"

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_logs_already_known_skip_count(
        self, mock_tag: MagicMock, mock_get: MagicMock, caplog: Any
    ) -> None:
        """The fetch logs how many markets were skipped as already known."""
        mock_tag.return_value = 42
        mock_get.return_value = _poly_response(
            [
                _poly_market(condition_id="known_1"),
                _poly_market(condition_id="known_2"),
                _poly_market(condition_id="fresh"),
            ]
        )

        with caplog.at_level(logging.INFO, logger="benchmark.datasets.fetch_open"):
            markets = fetch_polymarket_open(
                max_markets=10, existing_ids={"poly_known_1", "poly_known_2"}
            )

        assert [m["id"] for m in markets] == ["poly_fresh"]
        assert "1 new" in caplog.text
        assert "2 already-known skipped" in caplog.text

    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_known_and_old_market_counts_as_known_skipped(
        self, mock_tag: MagicMock, mock_get: MagicMock, caplog: Any
    ) -> None:
        """A market both already-known and pre-cutoff is counted as known-skipped."""
        # On a daily run virtually every known market is also older than the
        # cutoff; if the cutoff skip shadowed the known check, the skip counter
        # would always read 0 and lose its diagnostic value.
        mock_tag.return_value = 42
        mock_get.return_value = _poly_response(
            [
                _poly_market(condition_id="fresh", created_at=_hours_ago(1)),
                _poly_market(condition_id="known_old", created_at=_hours_ago(48)),
            ]
        )

        with caplog.at_level(logging.INFO, logger="benchmark.datasets.fetch_open"):
            markets = fetch_polymarket_open(
                max_markets=10,
                existing_ids={"poly_known_old"},
                created_within_hours=24,
            )

        assert [m["id"] for m in markets] == ["poly_fresh"]
        assert "1 new" in caplog.text
        assert "1 already-known skipped" in caplog.text

    @patch("benchmark.datasets.fetch_open.POLYMARKET_CATEGORIES", ["business"])
    @patch("benchmark.datasets.fetch_open.requests.get")
    @patch("benchmark.datasets.fetch_open._fetch_polymarket_tag_id")
    def test_pagination_safety_cap(
        self, mock_tag: MagicMock, mock_get: MagicMock
    ) -> None:
        """Pagination stops after the per-category page cap even if nothing new is found."""
        mock_tag.return_value = 42
        pages = [
            _poly_response(
                [
                    _poly_market(condition_id=f"p{page}_{i}")
                    for i in range(POLYMARKET_PAGE_LIMIT)
                ]
            )
            for page in range(POLYMARKET_MAX_PAGES_PER_CATEGORY + 3)
        ]
        mock_get.side_effect = pages
        all_known = {
            f"poly_p{page}_{i}"
            for page in range(POLYMARKET_MAX_PAGES_PER_CATEGORY + 3)
            for i in range(POLYMARKET_PAGE_LIMIT)
        }

        markets = fetch_polymarket_open(max_markets=10, existing_ids=all_known)
        assert not markets
        assert mock_get.call_count == POLYMARKET_MAX_PAGES_PER_CATEGORY


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


class TestJsonlIO:
    """Tests for JSONL loading and writing."""

    def test_load_existing_ids(self, tmp_path: Path) -> None:
        """Test loading existing IDs from a JSONL file."""
        f = tmp_path / "markets.jsonl"
        f.write_text('{"id": "omen_0x1"}\n' + '{"id": "poly_abc"}\n')
        ids = load_existing_ids(f, key="id")
        assert ids == {"omen_0x1", "poly_abc"}

    def test_load_existing_ids_empty(self, tmp_path: Path) -> None:
        """Test loading IDs from a non-existent file returns empty set."""
        f = tmp_path / "markets.jsonl"
        assert load_existing_ids(f, key="id") == set()

    def test_append_jsonl(self, tmp_path: Path) -> None:
        """Test appending records to a JSONL file."""
        f = tmp_path / "out.jsonl"
        append_jsonl(f, [{"id": "a"}, {"id": "b"}])
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "a"

    def test_append_preserves_existing(self, tmp_path: Path) -> None:
        """Test that appending preserves existing file content."""
        f = tmp_path / "out.jsonl"
        f.write_text('{"id": "existing"}\n')
        append_jsonl(f, [{"id": "new"}])
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 2
