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
"""Tests for benchmark/mech_analytics_client.py.

Focus: the mapping from mech-analytics's endpoint response to the row shape
``_accumulate_row`` reads, and the paging + cursor behaviour of
``iter_scored_rows``. No live HTTP — ``requests.Session.get`` is patched with
a queue of fake responses so we exercise the real code paths without a
network dependency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from benchmark import mech_analytics_client as mac


@pytest.fixture
def sample_api_row() -> dict[str, Any]:
    """One response row shaped like the live endpoint returns."""
    return {
        "request_id": "req-1",
        "tool": "superforcaster",
        "tool_version": "abc123",
        "platform": "omen",
        "question_title": "Will X happen by 2026?",
        "p_yes": 0.7,
        "p_no": 0.3,
        "confidence": 0.8,
        "prediction_parse_status": "valid",
        "market_prob_at_prediction": 0.55,
        "market_liquidity_usd": 12000.0,
        "resolved_outcome": 1.0,
        "requested_at": "2026-07-01T00:00:00Z",
        "delivered_at": "2026-07-01T00:00:30Z",
    }


class TestMapRow:
    def test_maps_scored_row_to_accumulator_shape(
        self, sample_api_row: dict[str, Any]
    ) -> None:
        # Fields the accumulator reads are what we care about — assert on
        # the exact keys _accumulate_row keys off, not on incidental fields.
        row = mac._map_row(sample_api_row)

        assert row["tool_name"] == "superforcaster"
        assert row["tool_version"] == "abc123"
        assert row["platform"] == "omen"
        assert row["question_text"] == "Will X happen by 2026?"
        assert row["p_yes"] == 0.7
        assert row["prediction_parse_status"] == "valid"
        assert row["market_prob_at_prediction"] == 0.55
        assert row["market_liquidity_at_prediction"] == 12000.0

    def test_resolved_outcome_1_maps_to_true(self, sample_api_row: dict) -> None:
        assert mac._map_row(sample_api_row)["final_outcome"] is True

    def test_resolved_outcome_0_maps_to_false(self, sample_api_row: dict) -> None:
        sample_api_row["resolved_outcome"] = 0.0
        assert mac._map_row(sample_api_row)["final_outcome"] is False

    def test_resolved_outcome_none_stays_none(self, sample_api_row: dict) -> None:
        # Unresolved rows must reach _accumulate_row with final_outcome=None
        # so the calibration + worst/best paths skip them.
        sample_api_row["resolved_outcome"] = None
        assert mac._map_row(sample_api_row)["final_outcome"] is None

    def test_latency_derived_from_timestamps(self, sample_api_row: dict) -> None:
        # delivered_at - requested_at = 30s in the fixture.
        assert mac._map_row(sample_api_row)["latency_s"] == 30.0

    def test_negative_latency_clamped_to_none(self, sample_api_row: dict) -> None:
        # A clock-skew or bookkeeping bug that produces delivered_at <
        # requested_at must not feed a nonsense negative into the latency
        # reservoir (which the accumulator reservoir-samples for reports).
        sample_api_row["delivered_at"] = "2026-06-30T23:59:00Z"
        assert mac._map_row(sample_api_row)["latency_s"] is None

    def test_missing_timestamps_gives_none_latency(
        self, sample_api_row: dict
    ) -> None:
        sample_api_row["delivered_at"] = None
        assert mac._map_row(sample_api_row)["latency_s"] is None

    def test_grouping_fields_absent_on_endpoint_are_none(
        self, sample_api_row: dict
    ) -> None:
        # by_mode / by_config_hash / by_horizon depend on fields the
        # endpoint doesn't carry today. Absent → None so the accumulator
        # uses its own defaults instead of KeyError.
        row = mac._map_row(sample_api_row)
        assert row["mode"] is None
        assert row["config_hash"] is None
        assert row["prediction_lead_time_days"] is None


class TestIterScoredRowsPaging:
    """Cursor-based paging through the endpoint, no live HTTP."""

    def _fake_response(self, rows: list[dict], next_cursor: str | None) -> Any:
        return SimpleNamespace(
            json=lambda: {"rows": rows, "next_cursor": next_cursor},
            raise_for_status=lambda: None,
        )

    def test_single_page_yields_all_rows_once(
        self, monkeypatch: pytest.MonkeyPatch, sample_api_row: dict
    ) -> None:
        monkeypatch.setenv("MECH_ANALYTICS_URL", "http://mech-analytics.test")
        responses = [self._fake_response([sample_api_row], next_cursor=None)]
        with patch.object(
            mac.requests.Session, "get", side_effect=lambda *a, **kw: responses.pop(0)
        ):
            rows = list(
                mac.iter_scored_rows(since=datetime(2026, 7, 1, tzinfo=timezone.utc))
            )
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "superforcaster"

    def test_multi_page_walks_cursor_until_exhausted(
        self, monkeypatch: pytest.MonkeyPatch, sample_api_row: dict
    ) -> None:
        monkeypatch.setenv("MECH_ANALYTICS_URL", "http://mech-analytics.test")
        second_row = dict(sample_api_row, request_id="req-2")
        third_row = dict(sample_api_row, request_id="req-3")
        responses = [
            self._fake_response([sample_api_row], next_cursor="cur-1"),
            self._fake_response([second_row], next_cursor="cur-2"),
            self._fake_response([third_row], next_cursor=None),
        ]
        with patch.object(
            mac.requests.Session, "get", side_effect=lambda *a, **kw: responses.pop(0)
        ):
            request_ids = [
                row["request_id"]
                for row in mac.iter_scored_rows(
                    since=datetime(2026, 7, 1, tzinfo=timezone.utc)
                )
            ]
        assert request_ids == ["req-1", "req-2", "req-3"]

    def test_missing_url_raises_before_any_http(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MECH_ANALYTICS_URL", raising=False)
        with pytest.raises(mac.MechAnalyticsError, match="MECH_ANALYTICS_URL"):
            # Consume the generator so the pre-flight config check fires.
            list(mac.iter_scored_rows(since=datetime(2026, 7, 1, tzinfo=timezone.utc)))
