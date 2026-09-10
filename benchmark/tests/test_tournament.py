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
from benchmark.ipfs_loader import IpfsFetchError
from benchmark.tournament import (
    _make_row_id,
    build_output_row,
    build_request_context,
    load_existing_row_ids,
    load_markets,
    run_single,
    run_tournament,
)

from packages.valory.skills.task_execution.utils.apis import KeyChain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market(
    market_id: str = "omen_0xabc",
    question: str = "Will X happen?",
    platform: str = "omen",
    prob: float | None = 0.65,
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
    """Tests for _make_row_id."""

    def test_deterministic(self) -> None:
        """Test that identical inputs produce identical row IDs."""
        id1 = _make_row_id("tool-a", "market_1", "omen", "model-1")
        id2 = _make_row_id("tool-a", "market_1", "omen", "model-1")
        assert id1 == id2

    def test_different_tools(self) -> None:
        """Test that different tools produce different row IDs."""
        id1 = _make_row_id("tool-a", "market_1", "omen", "model-1")
        id2 = _make_row_id("tool-b", "market_1", "omen", "model-1")
        assert id1 != id2

    def test_different_markets_same_question(self) -> None:
        """Two markets with same question but different IDs get different row IDs."""
        id1 = _make_row_id("tool-a", "omen_0x1", "omen", "model-1")
        id2 = _make_row_id("tool-a", "poly_abc", "polymarket", "model-1")
        assert id1 != id2

    def test_different_platforms_same_market_id(self) -> None:
        """Test that different platforms produce different row IDs."""
        id1 = _make_row_id("tool-a", "0xabc", "omen", "model-1")
        id2 = _make_row_id("tool-a", "0xabc", "polymarket", "model-1")
        assert id1 != id2

    def test_prefix(self) -> None:
        """Test that row ID starts with expected prefix."""
        row_id = _make_row_id("prediction-online", "m1", "omen", "m")
        assert row_id.startswith("tourn_prediction-online_")

    def test_market_context_flag_changes_id(self) -> None:
        """A blind and a market-context run of one market are two rows."""
        blind = _make_row_id("tool-a", "m1", "omen", "model-1")
        priced = _make_row_id(
            "tool-a", "m1", "omen", "model-1", with_market_context=True
        )
        assert blind != priced
        assert blind == _make_row_id(
            "tool-a", "m1", "omen", "model-1", with_market_context=False
        )


# ---------------------------------------------------------------------------
# build_output_row
# ---------------------------------------------------------------------------


class TestBuildOutputRow:
    """Tests for build_output_row."""

    def test_basic_row(self) -> None:
        """Test basic row construction with valid inputs."""
        market = _market()
        result = _run_result()
        row = build_output_row(
            market, "prediction-online", "gpt-4.1", result, "bafycid1"
        )

        assert row["mode"] == "tournament"
        assert row["final_outcome"] is None
        assert row["p_yes"] == 0.72
        assert row["market_prob_at_prediction"] == 0.65
        assert row["platform"] == "omen"
        assert row["tool_name"] == "prediction-online"
        assert row["tool_ipfs_hash"] == "bafycid1"
        assert row["schema_version"] == "1.0"

    def test_stores_source_content(self) -> None:
        """Test that source content is preserved in the row."""
        market = _market()
        sc = {"pages": {"http://example.com": "<html>...</html>"}}
        result = _run_result(source_content=sc)
        row = build_output_row(market, "tool", "model", result, "bafycid1")
        assert row["source_content"] == sc

    def test_none_source_content(self) -> None:
        """Test that None source content is stored as None."""
        market = _market()
        result = _run_result(source_content=None)
        row = build_output_row(market, "tool", "model", result, "bafycid1")
        assert row["source_content"] is None

    def test_error_result(self) -> None:
        """Test row construction with error result."""
        market = _market()
        result = _run_result(p_yes=None, p_no=None, status="error")  # type: ignore[arg-type]
        row = build_output_row(market, "tool", "model", result, "bafycid1")
        assert row["prediction_parse_status"] == "error"
        assert row["p_yes"] is None
        assert row["final_outcome"] is None


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


class TestJsonlIO:
    """Tests for JSONL I/O functions."""

    def test_load_markets(self, tmp_path: Path) -> None:
        """Test loading markets from JSONL file."""
        f = tmp_path / "markets.jsonl"
        f.write_text(
            json.dumps(_market("m1")) + "\n" + json.dumps(_market("m2")) + "\n"
        )
        markets = load_markets(f)
        assert len(markets) == 2
        assert markets[0]["id"] == "m1"

    def test_load_existing_row_ids_valid_only(self, tmp_path: Path) -> None:
        """Test that only valid prediction row IDs are loaded."""
        f = tmp_path / "predictions.jsonl"
        f.write_text(
            '{"row_id": "tourn_a_123", "prediction_parse_status": "valid"}\n'
            '{"row_id": "tourn_b_456", "prediction_parse_status": "malformed"}\n'
            '{"row_id": "tourn_c_789", "prediction_parse_status": "valid"}\n'
        )
        ids = load_existing_row_ids(f)
        assert ids == {"tourn_a_123", "tourn_c_789"}

    def test_load_existing_skips_errors(self, tmp_path: Path) -> None:
        """Test that error and timeout rows are skipped."""
        f = tmp_path / "predictions.jsonl"
        f.write_text(
            '{"row_id": "tourn_a_1", "prediction_parse_status": "error"}\n'
            '{"row_id": "tourn_b_2", "prediction_parse_status": "timeout"}\n'
        )
        ids = load_existing_row_ids(f)
        assert ids == set()

    def test_load_existing_empty(self, tmp_path: Path) -> None:
        """Test loading from non-existent file returns empty set."""
        f = tmp_path / "predictions.jsonl"
        assert load_existing_row_ids(f) == set()


# ---------------------------------------------------------------------------
# run_single (mocked — no API keys)
# ---------------------------------------------------------------------------


class TestRunSingle:
    """Tests for run_single with mocked tool execution."""

    @patch("benchmark.tournament.load_tool_run")
    def test_valid_result(self, mock_load: MagicMock) -> None:
        """Test run_single with a valid prediction result."""
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

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("prediction-online", "Will X?", "gpt-4.1", keys, "bafycid1")

        assert result["prediction_parse_status"] == "valid"
        assert result["p_yes"] == 0.7
        assert result["source_content"] == {"pages": {"http://x.com": "<html>"}}
        assert result["error"] is None

    @patch("benchmark.tournament.load_tool_run")
    def test_tool_exception(self, mock_load: MagicMock) -> None:
        """Test run_single when tool raises an exception."""
        mock_fn = MagicMock(side_effect=RuntimeError("API down"))
        mock_load.return_value = mock_fn

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("prediction-online", "Will X?", "gpt-4.1", keys, "bafycid1")

        assert result["prediction_parse_status"] == "error"
        assert result["p_yes"] is None
        assert "API down" in result["error"]
        assert result["source_content"] is None

    @patch("benchmark.tournament.load_tool_run")
    def test_malformed_response(self, mock_load: MagicMock) -> None:
        """Test run_single with a malformed tool response."""
        mock_fn = MagicMock(return_value=("not json", None, None, {}))
        mock_load.return_value = mock_fn

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("prediction-online", "Will X?", "gpt-4.1", keys, "bafycid1")

        assert result["prediction_parse_status"] != "valid"

    # When a tool's `with_key_rotation` bare-except returns an error-JSON
    # (p_yes=null, error="<msg>"), the embedded error must be surfaced on
    # the result so the prediction row carries a diagnostic instead of
    # just `malformed` with no context.
    @patch("benchmark.tournament.load_tool_run")
    def test_error_json_from_tool_is_captured(self, mock_load: MagicMock) -> None:
        """Tool error-JSON has its `error` field captured on the result."""
        mock_fn = MagicMock(
            return_value=(
                '{"p_yes": null, "p_no": null, "confidence": 0.0, '
                '"info_utility": 0.0, '
                '"error": "SubQuestions is not fully defined"}',
                None,
                None,
                None,
                {},
            )
        )
        mock_load.return_value = mock_fn

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("prediction-online", "Will X?", "gpt-4.1", keys, "bafycid1")

        assert result["prediction_parse_status"] == "malformed"
        assert result["p_yes"] is None
        assert result["error"] == "SubQuestions is not fully defined"

    @patch("benchmark.tournament.load_tool_run")
    def test_ipfs_fetch_error_yields_row(self, mock_load: MagicMock) -> None:
        """A bad CID is recorded as ipfs_fetch_error, not raised."""
        mock_load.side_effect = IpfsFetchError(
            "IPFS fetch failed: cid=bafybadcid status=404"
        )

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single(
            "prediction-online", "Will X?", "gpt-4.1", keys, "bafybadcid"
        )

        assert result["prediction_parse_status"] == "ipfs_fetch_error"
        assert result["p_yes"] is None
        assert result["source_content"] is None
        assert "bafybadcid" in result["error"]


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
        """Test that predictions are written to output file."""

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

        run_tournament(
            markets_path,
            output_path,
            {"prediction-online": "bafycid1"},
            "gpt-4.1",
        )

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
        """Test that valid predictions are skipped on re-run."""

        mock_keys.return_value = MagicMock()
        mock_run.return_value = _run_result()

        markets_path = tmp_path / "markets.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        markets_path.write_text(json.dumps(_market("omen_0x1")) + "\n")

        # First run
        run_tournament(
            markets_path,
            output_path,
            {"prediction-online": "bafycid1"},
            "gpt-4.1",
        )
        assert mock_run.call_count == 1

        # Second run — should skip (valid prediction exists)
        mock_run.reset_mock()
        run_tournament(
            markets_path,
            output_path,
            {"prediction-online": "bafycid1"},
            "gpt-4.1",
        )
        assert mock_run.call_count == 0

    @patch("benchmark.tournament.build_keychain")
    @patch("benchmark.tournament.run_single")
    def test_retries_malformed(
        self,
        mock_run: MagicMock,
        mock_keys: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that malformed predictions are retried on re-run."""

        mock_keys.return_value = MagicMock()

        markets_path = tmp_path / "markets.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        markets_path.write_text(json.dumps(_market("omen_0x1")) + "\n")

        # First run — malformed result
        mock_run.return_value = _run_result(p_yes=None, p_no=None, status="malformed")  # type: ignore[arg-type]
        run_tournament(
            markets_path,
            output_path,
            {"prediction-online": "bafycid1"},
            "gpt-4.1",
        )
        assert mock_run.call_count == 1

        # Second run — should retry (malformed not skipped)
        mock_run.reset_mock()
        mock_run.return_value = _run_result()
        run_tournament(
            markets_path,
            output_path,
            {"prediction-online": "bafycid1"},
            "gpt-4.1",
        )
        assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# build_request_context
# ---------------------------------------------------------------------------


class TestBuildRequestContext:
    """Tests for the trader-shaped request_context built per open market."""

    def test_carries_every_trader_field(self) -> None:
        """A complete market row yields the full trader-shaped context."""
        market = _market(
            "poly_0xdead",
            platform="polymarket",
            prob=0.42,
            close_date="2026-10-01T00:00:00Z",
        )
        market["usd_liquidity"] = 1500.5

        context = build_request_context(market)

        assert context == {
            "market_id": "0xabc",
            "type": "polymarket",
            "market_prob": 0.42,
            "market_close_at": "2026-10-01T00:00:00Z",
            "market_liquidity_usd": 1500.5,
        }

    @pytest.mark.parametrize(
        "liquidity",
        [None, -1.0, "100", True],
        ids=["none", "negative", "string", "bool"],
    )
    def test_unusable_liquidity_is_omitted(self, liquidity: Any) -> None:
        """An absent or invalid liquidity is dropped, not forwarded as junk."""
        market = _market()
        market["usd_liquidity"] = liquidity
        context = build_request_context(market)
        assert context is not None
        assert "market_liquidity_usd" not in context

    def test_market_id_is_the_raw_address_not_the_prefixed_row_id(self) -> None:
        """The trader sends the bare condition id / FPMM address, not poly_/omen_."""
        market = _market("poly_0xdead", platform="polymarket")
        market["market_address"] = "0xdead"
        context = build_request_context(market)
        assert context is not None
        assert context["market_id"] == "0xdead"

    def test_market_id_falls_back_to_row_id_without_address(self) -> None:
        """A row with no market_address still yields a usable context."""
        market = _market("poly_0xdead", platform="polymarket")
        market["market_address"] = None
        context = build_request_context(market)
        assert context is not None
        assert context["market_id"] == "poly_0xdead"

    def test_omen_market(self) -> None:
        """An Omen row carries its own platform as the context type."""
        context = build_request_context(_market(platform="omen", prob=0.65))
        assert context is not None
        assert context["type"] == "omen"
        assert context["market_prob"] == 0.65

    @pytest.mark.parametrize(
        "field", ["id", "platform"], ids=["no-market-id", "no-platform"]
    )
    def test_missing_identity_field_yields_none(self, field: str) -> None:
        """A row without a market id or a platform has no usable context."""
        market = _market()
        market[field] = None
        assert build_request_context(market) is None

    @pytest.mark.parametrize(
        "prob",
        [None, 1.5, -0.1, "0.5", True, float("nan")],
        ids=["none", "above-one", "negative", "string", "bool", "nan"],
    )
    def test_unusable_price_is_omitted(self, prob: Any) -> None:
        """An absent or out-of-range price is dropped, not forwarded as junk."""
        market = _market(prob=prob)
        context = build_request_context(market)
        assert context is not None
        assert "market_prob" not in context

    @pytest.mark.parametrize("prob", [0.0, 1.0], ids=["zero", "one"])
    def test_price_bounds_are_inclusive(self, prob: float) -> None:
        """A price of exactly 0 or 1 is a real price and is forwarded."""
        context = build_request_context(_market(prob=prob))
        assert context is not None
        assert context["market_prob"] == prob

    @pytest.mark.parametrize(
        "close_date", [None, "", "   ", 1234], ids=["none", "empty", "blank", "number"]
    )
    def test_unusable_close_date_is_omitted(self, close_date: Any) -> None:
        """A missing or non-string close date is dropped."""
        market = _market(close_date=close_date)
        context = build_request_context(market)
        assert context is not None
        assert "market_close_at" not in context

    def test_close_date_is_stripped(self) -> None:
        """Surrounding whitespace never reaches the tool."""
        context = build_request_context(_market(close_date="  2026-10-01T00:00:00Z "))
        assert context is not None
        assert context["market_close_at"] == "2026-10-01T00:00:00Z"

    def test_description_absent_without_network_fetch(self) -> None:
        """A Polymarket row with no description gets none.

        A live run must not add a per-market HTTP round trip to find one.
        """
        with patch("benchmark.runner.requests.get") as mock_get:
            context = build_request_context(
                _market("poly_0xdead", platform="polymarket")
            )
        assert context is not None
        assert "description" not in context
        mock_get.assert_not_called()

    def test_description_is_never_forwarded(self) -> None:
        """open_markets.jsonl carries no rules; a stray key is not forwarded either."""
        market = _market("poly_0x1", platform="polymarket")
        market["description"] = "Resolves YES if X occurs."
        context = build_request_context(market)
        assert context is not None
        assert "description" not in context


# ---------------------------------------------------------------------------
# run_single: request_context forwarding and payload extras
# ---------------------------------------------------------------------------


class TestRunSingleRequestContext:
    """Tests that run_single forwards the context and keeps payload extras."""

    @staticmethod
    def _mock_tool(payload: str) -> MagicMock:
        """Build a mock tool run function returning ``payload``."""
        return MagicMock(return_value=(payload, None, None, None, {}))

    @patch("benchmark.tournament.load_tool_run")
    def test_context_forwarded_to_tool(self, mock_load: MagicMock) -> None:
        """A supplied request_context reaches the tool's kwargs verbatim."""
        mock_fn = self._mock_tool('{"p_yes": 0.7, "p_no": 0.3, "confidence": 0.8}')
        mock_load.return_value = mock_fn
        context = {"market_id": "poly_0x1", "type": "polymarket", "market_prob": 0.4}

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        run_single(
            "superforcaster-market-aware",
            "Will X?",
            "gpt-4.1",
            keys,
            "bafycid1",
            request_context=context,
        )

        assert mock_fn.call_args.kwargs["request_context"] == context

    @patch("benchmark.tournament.load_tool_run")
    def test_no_context_key_when_none(self, mock_load: MagicMock) -> None:
        """Without a context the kwarg is absent, not None."""
        mock_fn = self._mock_tool('{"p_yes": 0.7, "p_no": 0.3, "confidence": 0.8}')
        mock_load.return_value = mock_fn

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        run_single("prediction-online", "Will X?", "gpt-4.1", keys, "bafycid1")

        assert "request_context" not in mock_fn.call_args.kwargs

    @patch("benchmark.tournament.load_tool_run")
    def test_extras_capture_market_aware_fields(self, mock_load: MagicMock) -> None:
        """The reasoning fields a market-aware tool emits survive the run."""
        mock_load.return_value = self._mock_tool(
            '{"p_yes": 0.6, "p_no": 0.4, "confidence": 0.8, "info_utility": 0.5, '
            '"p_independent": 0.7, "researchability": "high", '
            '"research_class": "news", "evidence_quality": 0.9}'
        )

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single(
            "superforcaster-market-aware", "Will X?", "gpt-4.1", keys, "bafycid1"
        )

        assert result["extras"] == {
            "p_independent": 0.7,
            "researchability": "high",
            "research_class": "news",
            "evidence_quality": 0.9,
        }

    @patch("benchmark.tournament.load_tool_run")
    def test_extras_never_shadow_core_fields(self, mock_load: MagicMock) -> None:
        """The parsed p_yes wins over anything the raw payload carries."""
        mock_load.return_value = self._mock_tool(
            '{"p_yes": 0.6, "p_no": 0.4, "confidence": 0.8, "p_independent": 0.7}'
        )

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("tool", "Will X?", "gpt-4.1", keys, "bafycid1")

        assert result["p_yes"] == 0.6
        assert "p_yes" not in result["extras"]

    @patch("benchmark.tournament.load_tool_run")
    def test_malformed_payload_yields_empty_extras(self, mock_load: MagicMock) -> None:
        """A non-JSON response leaves extras empty instead of raising."""
        mock_load.return_value = self._mock_tool("not json")

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("tool", "Will X?", "gpt-4.1", keys, "bafycid1")

        assert result["prediction_parse_status"] != "valid"
        assert result["extras"] == {}

    @patch("benchmark.tournament.load_tool_run")
    def test_raising_tool_yields_empty_extras(self, mock_load: MagicMock) -> None:
        """A crashing tool still returns an extras key."""
        mock_load.return_value = MagicMock(side_effect=RuntimeError("API down"))

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("tool", "Will X?", "gpt-4.1", keys, "bafycid1")

        assert result["prediction_parse_status"] == "error"
        assert result["extras"] == {}

    @patch("benchmark.tournament.load_tool_run")
    def test_ipfs_fetch_error_yields_empty_extras(self, mock_load: MagicMock) -> None:
        """An unfetchable CID still returns an extras key."""
        mock_load.side_effect = IpfsFetchError("IPFS fetch failed: cid=bad status=404")

        keys = KeyChain({"openai": ["fake"], "search_provider": ["google"]})
        result = run_single("tool", "Will X?", "gpt-4.1", keys, "bad")

        assert result["prediction_parse_status"] == "ipfs_fetch_error"
        assert result["extras"] == {}


# ---------------------------------------------------------------------------
# build_output_row: tool_extras column
# ---------------------------------------------------------------------------


class TestBuildOutputRowExtras:
    """Tests for the tool_extras column on tournament rows."""

    def test_extras_stored(self) -> None:
        """Payload extras are stored under tool_extras."""
        result = _run_result()
        result["extras"] = {"p_independent": 0.7, "research_class": "news"}
        row = build_output_row(_market(), "tool", "model", result, "bafycid1")
        assert row["tool_extras"] == {"p_independent": 0.7, "research_class": "news"}

    def test_run_result_without_extras(self) -> None:
        """A run_result from the older signature yields an empty dict."""
        row = build_output_row(_market(), "tool", "model", _run_result(), "bafycid1")
        assert row["tool_extras"] == {}


# ---------------------------------------------------------------------------
# run_tournament: the loop hands each market's context to run_single
# ---------------------------------------------------------------------------


class TestRunTournamentRequestContext:
    """Integration test proving the tournament no longer runs tools blind."""

    @patch("benchmark.tournament.build_keychain")
    @patch("benchmark.tournament.run_single")
    def test_context_and_extras_reach_the_row(
        self,
        mock_run: MagicMock,
        mock_keys: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Each market's price is forwarded and the tool's extras are written."""
        mock_keys.return_value = MagicMock()
        result = _run_result()
        result["extras"] = {"p_independent": 0.55}
        mock_run.return_value = result

        markets_path = tmp_path / "markets.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        market = _market(
            "poly_0x1", platform="polymarket", prob=0.31, close_date="2026-10-01T00:00Z"
        )
        markets_path.write_text(json.dumps(market) + "\n")

        run_tournament(
            markets_path,
            output_path,
            {"superforcaster-market-aware": "bafycid1"},
            "gpt-4.1",
            with_market_context=True,
        )

        forwarded = mock_run.call_args.kwargs["request_context"]
        assert forwarded == {
            "market_id": "0xabc",
            "type": "polymarket",
            "market_prob": 0.31,
            "market_close_at": "2026-10-01T00:00Z",
        }
        row = json.loads(output_path.read_text().strip())
        assert row["tool_extras"] == {"p_independent": 0.55}
        assert row["market_context"] is True

    @patch("benchmark.tournament.build_keychain")
    @patch("benchmark.tournament.run_single")
    def test_blind_by_default(
        self,
        mock_run: MagicMock,
        mock_keys: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Without --market-context no tool sees a price, and the row says so."""
        mock_keys.return_value = MagicMock()
        mock_run.return_value = _run_result()
        markets_path = tmp_path / "markets.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        markets_path.write_text(json.dumps(_market(prob=0.31)) + "\n")

        run_tournament(
            markets_path, output_path, {"prediction-online": "bafycid1"}, "gpt-4.1"
        )

        assert mock_run.call_args.kwargs["request_context"] is None
        row = json.loads(output_path.read_text().strip())
        assert row["market_context"] is False

    @patch("benchmark.tournament.build_keychain")
    @patch("benchmark.tournament.run_single")
    def test_blind_and_priced_arms_share_one_file(
        self,
        mock_run: MagicMock,
        mock_keys: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A market predicted blind is re-run when the price is switched on."""
        mock_keys.return_value = MagicMock()
        mock_run.return_value = _run_result()
        markets_path = tmp_path / "markets.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        markets_path.write_text(json.dumps(_market(prob=0.31)) + "\n")
        tools = {"superforcaster-market-aware": "bafycid1"}

        run_tournament(markets_path, output_path, tools, "gpt-4.1")
        run_tournament(
            markets_path, output_path, tools, "gpt-4.1", with_market_context=True
        )

        rows = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert [r["market_context"] for r in rows] == [False, True]
        assert len({r["row_id"] for r in rows}) == 2

    @patch("benchmark.tournament.build_keychain")
    @patch("benchmark.tournament.run_single")
    def test_warns_when_no_market_carried_a_price(
        self,
        mock_run: MagicMock,
        mock_keys: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """--market-context with unpriced markets is reported, not silent."""
        mock_keys.return_value = MagicMock()
        mock_run.return_value = _run_result()
        markets_path = tmp_path / "markets.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        markets_path.write_text(json.dumps(_market(prob=None)) + "\n")

        with caplog.at_level("WARNING", logger="benchmark.tournament"):
            run_tournament(
                markets_path,
                output_path,
                {"superforcaster-market-aware": "bafycid1"},
                "gpt-4.1",
                with_market_context=True,
            )

        assert "market context: 0/1 rows carried a usable price" in caplog.text

    @patch("benchmark.tournament.build_keychain")
    @patch("benchmark.tournament.run_single")
    def test_market_without_identity_still_runs(
        self,
        mock_run: MagicMock,
        mock_keys: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A market with no id yields a None context rather than skipping it."""
        mock_keys.return_value = MagicMock()
        mock_run.return_value = _run_result()

        market = _market()
        market["id"] = None
        markets_path = tmp_path / "markets.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        markets_path.write_text(json.dumps(market) + "\n")

        run_tournament(
            markets_path,
            output_path,
            {"prediction-online": "bafycid1"},
            "gpt-4.1",
        )

        assert mock_run.call_args.kwargs["request_context"] is None
        assert output_path.read_text().strip()
