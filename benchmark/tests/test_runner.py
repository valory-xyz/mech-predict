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
forwarded ``description``; it never contacts Polymarket) and the opt-in market
context (price, close time, liquidity, spread) that a market-aware tool needs.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from benchmark.runner import (
    _fetch_polymarket_description,
    build_output_row,
    build_request_context,
    extract_extras,
    replay,
    run_single,
)
from benchmark.scorer import _is_edge_eligible

RUNNER = "benchmark.runner"

# A Gamma `/markets` object. Deliberately carries price/volume/liquidity fields
# so the tests can prove only the description is extracted. Market odds reach a
# tool from the dataset row under --market-context, never from this fetch.
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


# A dataset row carrying every production market field, all in range.
MARKET_ROW: dict[str, Any] = {
    "market_id": "0xabc",
    "platform": "polymarket",
    "description": "pre-baked rules",
    "market_prob_at_prediction": 0.42,
    "market_close_at": "2026-08-31T12:00:00Z",
    "market_liquidity_at_prediction": 12345.6,
    "market_spread_at_prediction": 0.02,
}


class TestBuildRequestContextMarketContext:
    """Tests for the opt-in market-context fields (price, close, liquidity, spread)."""

    def test_off_by_default(self) -> None:
        """Without the flag a row full of odds still yields a blind context."""
        ctx = build_request_context(MARKET_ROW)

        assert ctx == {
            "market_id": "0xabc",
            "type": "polymarket",
            "description": "pre-baked rules",
        }

    def test_adds_all_trader_fields(self) -> None:
        """With the flag on, every in-range field is forwarded under its trader name."""
        ctx = build_request_context(MARKET_ROW, with_market_context=True)

        assert ctx == {
            "market_id": "0xabc",
            "type": "polymarket",
            "description": "pre-baked rules",
            "market_prob": 0.42,
            "market_close_at": "2026-08-31T12:00:00Z",
            "market_liquidity_usd": 12345.6,
            "market_spread": 0.02,
        }

    def test_omits_none_fields(self) -> None:
        """Fields the row does not carry are omitted, never sent as None."""
        ctx = build_request_context(
            {
                "market_id": "0xfpmm",
                "platform": "omen",
                "market_prob_at_prediction": 0.7,
            },
            with_market_context=True,
        )

        assert ctx == {"market_id": "0xfpmm", "type": "omen", "market_prob": 0.7}

    @pytest.mark.parametrize("bad_prob", [-0.1, 1.5, float("nan"), "0.4", True, None])
    def test_out_of_range_price_omitted(self, bad_prob: Any) -> None:
        """A price outside [0, 1], the wrong type, or a bool is dropped."""
        row = {**MARKET_ROW, "market_prob_at_prediction": bad_prob}

        ctx = build_request_context(row, with_market_context=True)

        assert ctx is not None
        assert "market_prob" not in ctx

    @pytest.mark.parametrize("bad_spread", [-0.01, 1.01, "wide", None])
    def test_out_of_range_spread_omitted(self, bad_spread: Any) -> None:
        """A spread outside [0, 1] is dropped, matching the trader's clamp."""
        row = {**MARKET_ROW, "market_spread_at_prediction": bad_spread}

        ctx = build_request_context(row, with_market_context=True)

        assert ctx is not None
        assert "market_spread" not in ctx

    def test_negative_liquidity_omitted(self) -> None:
        """Negative liquidity is not a real depth reading, so it is dropped."""
        row = {**MARKET_ROW, "market_liquidity_at_prediction": -1.0}

        ctx = build_request_context(row, with_market_context=True)

        assert ctx is not None
        assert "market_liquidity_usd" not in ctx

    @pytest.mark.parametrize("bad_close", ["", "   ", 1756640000, None])
    def test_blank_close_at_omitted(self, bad_close: Any) -> None:
        """A blank or non-string close time is dropped rather than forwarded."""
        row = {**MARKET_ROW, "market_close_at": bad_close}

        ctx = build_request_context(row, with_market_context=True)

        assert ctx is not None
        assert "market_close_at" not in ctx

    def test_close_at_is_stripped(self) -> None:
        """Surrounding whitespace is trimmed off the close time."""
        row = {**MARKET_ROW, "market_close_at": "  2026-08-31T12:00:00Z  "}

        ctx = build_request_context(row, with_market_context=True)

        assert ctx is not None
        assert ctx["market_close_at"] == "2026-08-31T12:00:00Z"

    def test_amm_fee_never_set(self) -> None:
        """The dataset has no amm_fee counterpart, so it is never invented."""
        ctx = build_request_context(MARKET_ROW, with_market_context=True)

        assert ctx is not None
        assert "amm_fee" not in ctx


class TestReplayThreadsMarketContext:
    """The --market-context flag must reach build_request_context from replay()."""

    @staticmethod
    def _run_replay(tmp_path: Path, with_market_context: bool) -> MagicMock:
        """Run replay() over one synthetic row with every tool call mocked.

        :param tmp_path: pytest temporary directory.
        :param with_market_context: value to thread through replay().
        :return: the patched build_request_context mock.
        """
        dataset = tmp_path / "dataset.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    **MARKET_ROW,
                    "question_text": "Will X ship?",
                    "final_outcome": 1,
                    "source_content": {"mode": "cached", "sources": []},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with (
            patch(f"{RUNNER}.build_keychain"),
            patch(f"{RUNNER}.build_request_context") as mock_ctx,
            patch(f"{RUNNER}.run_single") as mock_run,
        ):
            mock_ctx.return_value = {"market_id": "0xabc", "type": "polymarket"}
            mock_run.return_value = {
                "p_yes": 0.5,
                "p_no": 0.5,
                "confidence": 0.5,
                "prediction_parse_status": "valid",
                "latency_s": 1.0,
                "error": None,
            }
            replay(
                dataset_path=dataset,
                output_path=tmp_path / "out.jsonl",
                tools=["superforcaster-market-aware"],
                model="test-model",
                with_market_context=with_market_context,
            )
        return mock_ctx

    def test_flag_on_is_forwarded(self, tmp_path: Path) -> None:
        """replay(with_market_context=True) asks for the market fields."""
        mock_ctx = self._run_replay(tmp_path, with_market_context=True)

        mock_ctx.assert_called_once()
        assert mock_ctx.call_args.kwargs["with_market_context"] is True

    def test_default_is_blind(self, tmp_path: Path) -> None:
        """replay() without the flag keeps the historical blind behaviour."""
        mock_ctx = self._run_replay(tmp_path, with_market_context=False)

        mock_ctx.assert_called_once()
        assert mock_ctx.call_args.kwargs["with_market_context"] is False


# A replay dataset row carrying every market column the production log has.
SCORED_ROW: dict[str, Any] = {
    **MARKET_ROW,
    "question_text": "Will X ship?",
    "final_outcome": 1,
    "resolved_at": "2026-09-01T00:00:00Z",
    "prediction_lead_time_days": 3.5,
    "category": "tech",
}

# What run_single returns for a clean, parseable delivery.
VALID_RESULT: dict[str, Any] = {
    "p_yes": 0.7,
    "p_no": 0.3,
    "confidence": 0.8,
    "prediction_parse_status": "valid",
    "latency_s": 12.0,
    "error": None,
    "extras": {"p_independent": 0.6, "research_class": "rich"},
}


class TestExtractExtras:
    """Tests for extract_extras - the non-core payload keys a tool emits."""

    def test_keeps_non_core_keys(self) -> None:
        """Market-aware reasoning fields survive; the scored core fields do not."""
        payload = json.dumps(
            {
                "p_yes": 0.7,
                "p_no": 0.3,
                "confidence": 0.8,
                "info_utility": 0.5,
                "p_independent": 0.62,
                "researchability": "high",
                "research_class": "rich",
                "evidence_quality": 4,
            }
        )

        assert extract_extras(payload) == {
            "p_independent": 0.62,
            "researchability": "high",
            "research_class": "rich",
            "evidence_quality": 4,
        }

    def test_core_only_payload_yields_empty(self) -> None:
        """A plain predictor contributes nothing beyond the scored columns."""
        payload = json.dumps({"p_yes": 0.7, "p_no": 0.3, "confidence": 0.8})

        assert extract_extras(payload) == {}

    @pytest.mark.parametrize(
        "raw",
        [
            "not json at all",
            "",
            "[1, 2, 3]",
            '"a bare string"',
            "null",
            None,
            123,
        ],
    )
    def test_unusable_payloads_yield_empty(self, raw: Any) -> None:
        """Anything that is not a JSON object degrades to {} instead of raising.

        :param raw: a tool response that cannot yield extras.
        """
        assert extract_extras(raw) == {}


class TestRunSingleExtras:
    """run_single must surface extras on success and on every failure path."""

    @staticmethod
    def _run(result_str: str) -> dict[str, Any]:
        """Call run_single with a stub tool returning ``result_str``.

        :param result_str: the raw tool response to feed back.
        :return: the run_single result dict.
        """
        run_fn = MagicMock(return_value=(result_str, None, None, None))
        with patch(f"{RUNNER}.load_tool_run", return_value=run_fn):
            return run_single(
                tool_name="superforcaster-market-aware",
                question_text="Will X ship?",
                source_content={"mode": "cached", "sources": []},
                model="test-model",
                api_keys=MagicMock(),
            )

    def test_valid_payload_carries_extras(self) -> None:
        """A market-aware payload keeps its reasoning fields under extras."""
        result = self._run(
            json.dumps({"p_yes": 0.7, "p_no": 0.3, "p_independent": 0.62})
        )

        assert result["prediction_parse_status"] == "valid"
        assert result["extras"] == {"p_independent": 0.62}

    def test_extras_never_shadow_core_fields(self) -> None:
        """parse_tool_response wins on p_yes/p_no; extras cannot overwrite them."""
        result = self._run(json.dumps({"p_yes": 0.7, "p_no": 0.3, "note": "hi"}))

        assert result["p_yes"] == 0.7
        assert result["p_no"] == 0.3
        assert result["extras"] == {"note": "hi"}

    def test_malformed_payload_still_returns_extras(self) -> None:
        """An unparseable response yields empty extras rather than a KeyError."""
        result = self._run("total garbage")

        assert result["prediction_parse_status"] != "valid"
        assert result["extras"] == {}

    def test_tool_exception_returns_empty_extras(self) -> None:
        """A raising tool still produces the extras key downstream code reads."""
        run_fn = MagicMock(side_effect=RuntimeError("boom"))
        with patch(f"{RUNNER}.load_tool_run", return_value=run_fn):
            result = run_single(
                tool_name="superforcaster-market-aware",
                question_text="Will X ship?",
                source_content={"mode": "cached", "sources": []},
                model="test-model",
                api_keys=MagicMock(),
            )

        assert result["prediction_parse_status"] == "error"
        assert result["extras"] == {}


class TestBuildOutputRowMarketColumns:
    """build_output_row must carry the market baseline the scorer needs."""

    def test_carries_market_columns(self) -> None:
        """Price, liquidity, close time and lead time come from the dataset row."""
        row = build_output_row(
            SCORED_ROW, "superforcaster-market-aware", "m", VALID_RESULT
        )

        assert row["market_prob_at_prediction"] == 0.42
        assert row["market_liquidity_at_prediction"] == 12345.6
        assert row["market_close_at"] == "2026-08-31T12:00:00Z"
        assert row["prediction_lead_time_days"] == 3.5

    def test_absent_columns_stay_none(self) -> None:
        """A dataset row without market columns keeps the historical Nones."""
        bare = {"question_text": "Will X ship?", "final_outcome": 0}

        row = build_output_row(bare, "superforcaster", "m", VALID_RESULT)

        assert row["market_prob_at_prediction"] is None
        assert row["market_liquidity_at_prediction"] is None
        assert row["market_close_at"] is None
        assert row["prediction_lead_time_days"] is None

    def test_carries_tool_extras(self) -> None:
        """The non-core payload keys land under tool_extras."""
        row = build_output_row(
            SCORED_ROW, "superforcaster-market-aware", "m", VALID_RESULT
        )

        assert row["tool_extras"] == {
            "p_independent": 0.6,
            "research_class": "rich",
        }

    def test_missing_extras_key_yields_empty_dict(self) -> None:
        """A run_result from older code paths must not blow up the row build."""
        legacy = {k: v for k, v in VALID_RESULT.items() if k != "extras"}

        row = build_output_row(SCORED_ROW, "superforcaster", "m", legacy)

        assert row["tool_extras"] == {}

    def test_scorer_sees_an_edge_row(self) -> None:
        """The built row satisfies the scorer's edge-row predicate end to end."""
        row = build_output_row(
            SCORED_ROW, "superforcaster-market-aware", "m", VALID_RESULT
        )

        assert _is_edge_eligible(row)
