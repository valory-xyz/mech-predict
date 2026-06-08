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

"""Unit tests for benchmark/runner.py request_context construction.

These cover the trader-simulation logic: building a mech-style request_context
from a dataset row, including the Polymarket resolution-rules fetch that the
benchmark performs in the trader's place (factual_research-v2 only reads the
forwarded ``description``; it never contacts Polymarket).
"""

from typing import Any
from unittest.mock import MagicMock, patch

from benchmark.runner import _fetch_polymarket_description, build_request_context

RUNNER = "benchmark.runner"

# A Gamma `/markets` object. Deliberately carries price/volume/liquidity fields
# so the tests can prove only the description is extracted — these are exactly
# the "odds" the prediction tools must never see.
FAKE_GAMMA_MARKET = {
    "conditionId": "0xabc123",
    "question": "Will X ship?",
    "description": "RULES_SENTINEL: resolves YES if X ships before 2026-01-01.",
    "endDate": "2026-01-01T00:00:00Z",
    "outcomePrices": '["0.82", "0.18"]',
    "lastTradePrice": 0.82,
    "volume": "1234567",
    "liquidity": "98765",
}

_PRICE_TOKENS = (
    "0.82",
    "0.18",
    "outcomePrices",
    "lastTradePrice",
    "volume",
    "liquidity",
    "1234567",
    "98765",
)


def _mock_gamma_response(status: int = 200, payload: Any = None) -> MagicMock:
    """Build a mock requests.Response for a Gamma query."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = [] if payload is None else payload
    return resp


class TestFetchPolymarketDescription:
    """Tests for _fetch_polymarket_description — the trader-simulation fetch."""

    @patch(f"{RUNNER}.requests.get")
    def test_extracts_only_description(self, mock_get: MagicMock) -> None:
        """A successful fetch returns ONLY the description text, no prices."""
        mock_get.return_value = _mock_gamma_response(payload=[FAKE_GAMMA_MARKET])

        out = _fetch_polymarket_description("0xABC123")

        assert out == FAKE_GAMMA_MARKET["description"]
        for token in _PRICE_TOKENS:
            assert token not in out

    @patch(f"{RUNNER}.requests.get")
    def test_strips_poly_prefix_before_query(self, mock_get: MagicMock) -> None:
        """A `poly_` prefix is stripped so Gamma gets the bare condition id."""
        mock_get.return_value = _mock_gamma_response(payload=[FAKE_GAMMA_MARKET])

        _fetch_polymarket_description("poly_0xabc123")

        assert mock_get.call_args.kwargs["params"]["condition_ids"] == "0xabc123"

    @patch(f"{RUNNER}.requests.get")
    def test_condition_id_mismatch_returns_none(self, mock_get: MagicMock) -> None:
        """A response echoing a different conditionId is rejected."""
        wrong = dict(FAKE_GAMMA_MARKET, conditionId="0xdeadbeef")
        mock_get.return_value = _mock_gamma_response(payload=[wrong])

        assert _fetch_polymarket_description("0xabc123") is None

    @patch(f"{RUNNER}.requests.get")
    def test_empty_description_returns_none(self, mock_get: MagicMock) -> None:
        """A market with no resolution text yields None."""
        nodesc = dict(FAKE_GAMMA_MARKET, description="")
        mock_get.return_value = _mock_gamma_response(payload=[nodesc])

        assert _fetch_polymarket_description("0xabc123") is None

    @patch(f"{RUNNER}.requests.get")
    def test_non_200_returns_none(self, mock_get: MagicMock) -> None:
        """A non-200 status degrades to None."""
        mock_get.return_value = _mock_gamma_response(
            status=503, payload=[FAKE_GAMMA_MARKET]
        )

        assert _fetch_polymarket_description("0xabc123") is None

    @patch(f"{RUNNER}.requests.get")
    def test_network_error_returns_none(self, mock_get: MagicMock) -> None:
        """A raised request exception degrades to None, never propagates."""
        mock_get.side_effect = Exception("boom")

        assert _fetch_polymarket_description("0xabc123") is None

    @patch(f"{RUNNER}.requests.get")
    def test_falls_back_to_closed_query(self, mock_get: MagicMock) -> None:
        """An empty open-market result retries with closed=true (resolved)."""
        mock_get.side_effect = [
            _mock_gamma_response(payload=[]),  # open query → nothing
            _mock_gamma_response(payload=[FAKE_GAMMA_MARKET]),  # closed → hit
        ]

        out = _fetch_polymarket_description("0xabc123")

        assert out == FAKE_GAMMA_MARKET["description"]
        assert mock_get.call_count == 2
        assert mock_get.call_args_list[0].kwargs["params"].get("closed") is None
        assert mock_get.call_args_list[1].kwargs["params"].get("closed") == "true"


class TestBuildRequestContext:
    """Tests for build_request_context — dataset row → mech request_context."""

    def test_returns_none_without_market_id(self) -> None:
        """A row missing the market id yields no context."""
        assert build_request_context({"platform": "polymarket"}) is None

    def test_returns_none_without_platform(self) -> None:
        """A row missing the platform yields no context."""
        assert build_request_context({"market_id": "0xabc"}) is None

    @patch(f"{RUNNER}._fetch_polymarket_description")
    def test_omen_carries_id_and_type_no_fetch(self, mock_fetch: MagicMock) -> None:
        """An Omen row gets id + type only and never hits Gamma."""
        ctx = build_request_context({"market_id": "0xfpmm", "platform": "omen"})

        assert ctx == {"market_id": "0xfpmm", "type": "omen"}
        mock_fetch.assert_not_called()

    @patch(f"{RUNNER}._fetch_polymarket_description")
    def test_polymarket_prefers_prebaked_description(
        self, mock_fetch: MagicMock
    ) -> None:
        """A row carrying a description uses it and skips the network."""
        ctx = build_request_context(
            {
                "market_id": "0xabc",
                "platform": "polymarket",
                "description": "pre-baked rules",
            }
        )

        assert ctx == {
            "market_id": "0xabc",
            "type": "polymarket",
            "description": "pre-baked rules",
        }
        mock_fetch.assert_not_called()

    @patch(f"{RUNNER}._fetch_polymarket_description")
    def test_polymarket_fetches_when_absent(self, mock_fetch: MagicMock) -> None:
        """A polymarket row without a description fetches it by market_id."""
        mock_fetch.return_value = "fetched rules"

        ctx = build_request_context({"market_id": "0xabc", "platform": "polymarket"})

        assert ctx == {
            "market_id": "0xabc",
            "type": "polymarket",
            "description": "fetched rules",
        }
        mock_fetch.assert_called_once_with("0xabc")

    @patch(f"{RUNNER}._fetch_polymarket_description")
    def test_polymarket_omits_description_when_fetch_fails(
        self, mock_fetch: MagicMock
    ) -> None:
        """A failed fetch leaves the context without a description (v1 fallback)."""
        mock_fetch.return_value = None

        ctx = build_request_context({"market_id": "0xabc", "platform": "polymarket"})

        assert ctx == {"market_id": "0xabc", "type": "polymarket"}
