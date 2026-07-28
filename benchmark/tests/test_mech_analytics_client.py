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
``accumulate_row`` reads, and the paging + cursor behaviour of
``iter_scored_rows``. No live HTTP — ``requests.Session.get`` is patched with
a queue of fake responses so we exercise the real code paths without a
network dependency.
"""

# pylint: disable=redefined-outer-name,protected-access

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
        "market_spread_at_prediction": 0.02,
        "market_id": "0xabcd",
        "resolved_outcome": 1.0,
        "resolved_at": "2026-07-02T12:00:00Z",
        "brier": 0.09,
        "log_loss": 0.3567,
        "edge": 0.15,
        "directional_correct": True,
        "requested_at": "2026-07-01T00:00:00Z",
        "delivered_at": "2026-07-01T00:00:30Z",
    }


class TestMapRow:
    """Field mapping from the endpoint's response shape to the accumulator shape."""

    def test_maps_scored_row_to_accumulator_shape(
        self, sample_api_row: dict[str, Any]
    ) -> None:
        """Endpoint field names route onto the exact keys ``accumulate_row`` reads."""
        # Fields the accumulator reads are what we care about — assert on
        # the exact keys accumulate_row keys off, not on incidental fields.
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
        """Numeric 1.0 resolves to ``final_outcome=True``."""
        assert mac._map_row(sample_api_row)["final_outcome"] is True

    def test_resolved_outcome_0_maps_to_false(self, sample_api_row: dict) -> None:
        """Numeric 0.0 resolves to ``final_outcome=False``."""
        sample_api_row["resolved_outcome"] = 0.0
        assert mac._map_row(sample_api_row)["final_outcome"] is False

    def test_resolved_outcome_none_stays_none(self, sample_api_row: dict) -> None:
        """Unresolved rows pass through with ``final_outcome=None`` so the accumulator skips them."""
        # so the calibration + worst/best paths skip them.
        sample_api_row["resolved_outcome"] = None
        assert mac._map_row(sample_api_row)["final_outcome"] is None

    def test_latency_derived_from_timestamps(self, sample_api_row: dict) -> None:
        """``latency_s`` is derived from ``delivered_at - requested_at``."""
        # delivered_at - requested_at = 30s in the fixture.
        assert mac._map_row(sample_api_row)["latency_s"] == 30.0

    def test_negative_latency_clamped_to_none(self, sample_api_row: dict) -> None:
        """A clock-skew delivered<requested must not feed a negative into the reservoir."""
        # reservoir (which the accumulator reservoir-samples for reports).
        sample_api_row["delivered_at"] = "2026-06-30T23:59:00Z"
        assert mac._map_row(sample_api_row)["latency_s"] is None

    def test_negative_latency_logs_warning(
        self, sample_api_row: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Negative latency logs a warning naming the offending row."""
        # Silent None on a negative delta would thin the latency stats
        # without any breadcrumb; the warning gives operators something
        # to grep for on an ingest anomaly.
        sample_api_row["delivered_at"] = "2026-06-30T23:59:00Z"
        with caplog.at_level("WARNING", logger=mac.log.name):
            result = mac._map_row(sample_api_row)
        assert result["latency_s"] is None
        assert any(
            "negative latency" in rec.message
            and sample_api_row["request_id"] in rec.message
            for rec in caplog.records
        ), f"expected a negative-latency warning; got: {[r.message for r in caplog.records]}"

    def test_missing_timestamps_gives_none_latency(self, sample_api_row: dict) -> None:
        """Missing timestamps yield ``latency_s=None`` rather than raising."""
        sample_api_row["delivered_at"] = None
        assert mac._map_row(sample_api_row)["latency_s"] is None

    def test_passthrough_fields_carry_endpoint_values(
        self, sample_api_row: dict[str, Any]
    ) -> None:
        """market_id + resolved_at + brier/log_loss/edge/directional pass through."""
        row = mac._map_row(sample_api_row)
        assert row["market_id"] == "0xabcd"
        assert row["market_spread_at_prediction"] == 0.02
        assert row["resolved_at"] == "2026-07-02T12:00:00Z"
        assert row["brier"] == 0.09
        assert row["log_loss"] == 0.3567
        assert row["edge"] == 0.15
        assert row["directional_correct"] is True

    def test_grouping_fields_absent_on_endpoint_are_none(
        self, sample_api_row: dict
    ) -> None:
        """Grouping keys absent on the endpoint stay None so the accumulator uses defaults."""
        # by_mode / by_config_hash / by_horizon depend on fields the
        # endpoint doesn't carry today. Absent → None so the accumulator
        # uses its own defaults instead of KeyError.
        row = mac._map_row(sample_api_row)
        assert row["mode"] is None
        assert row["config_hash"] is None
        assert row["prediction_lead_time_days"] is None


class TestValidatedProbability:
    """Range + type gate for p_yes / p_no / market_prob."""

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 0.99, "0.7"])
    def test_valid_values_pass(self, value: Any) -> None:
        """In-range numbers (and numeric strings) coerce cleanly."""
        result, ok = mac._validated_probability(value)
        assert ok is True
        assert result == float(value)

    def test_none_passes_as_none(self) -> None:
        """None (absent field) is not a validation failure."""
        result, ok = mac._validated_probability(None)
        assert result is None
        assert ok is True

    @pytest.mark.parametrize("value", [-0.01, 1.01, 1.5, -1.0, float("inf"), float("nan")])
    def test_out_of_range_fails(self, value: float) -> None:
        """Values outside [0, 1] (including NaN / inf) are rejected."""
        result, ok = mac._validated_probability(value)
        assert result is None
        assert ok is False

    @pytest.mark.parametrize("value", ["nope", {}, [], object()])
    def test_non_numeric_fails(self, value: Any) -> None:
        """Non-numeric junk is rejected without raising."""
        result, ok = mac._validated_probability(value)
        assert result is None
        assert ok is False


class TestMapRowPredictionValidation:
    """Prediction-parse-status demotion when the endpoint ships invalid ranges."""

    def test_valid_row_stays_valid(self, sample_api_row: dict) -> None:
        """A row with in-range fields keeps its parse status."""
        row = mac._map_row(sample_api_row)
        assert row["prediction_parse_status"] == "valid"
        assert row["p_yes"] == 0.7

    def test_out_of_range_p_yes_demotes_valid_to_malformed(
        self, sample_api_row: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An out-of-range p_yes demotes the parse status and nulls the field."""
        sample_api_row["p_yes"] = 1.5
        with caplog.at_level("WARNING", logger=mac.log.name):
            row = mac._map_row(sample_api_row)
        assert row["prediction_parse_status"] == "malformed"
        assert row["p_yes"] is None
        assert any("out-of-range" in r.message for r in caplog.records)

    def test_out_of_range_market_prob_demotes(self, sample_api_row: dict) -> None:
        """An out-of-range market_prob also demotes to malformed."""
        sample_api_row["market_prob_at_prediction"] = -0.2
        row = mac._map_row(sample_api_row)
        assert row["prediction_parse_status"] == "malformed"
        assert row["market_prob_at_prediction"] is None

    def test_already_malformed_stays_malformed(self, sample_api_row: dict) -> None:
        """Non-valid rows aren't re-graded (invariant only enforced on 'valid')."""
        sample_api_row["prediction_parse_status"] = "missing_fields"
        sample_api_row["p_yes"] = 5.0
        row = mac._map_row(sample_api_row)
        assert row["prediction_parse_status"] == "missing_fields"


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
        """A single-page response yields each row exactly once, then stops."""
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
        """The paginator follows ``next_cursor`` across pages until it goes null."""
        monkeypatch.setenv("MECH_ANALYTICS_URL", "http://mech-analytics.test")
        second_row = dict(sample_api_row, request_id="req-2")
        third_row = dict(sample_api_row, request_id="req-3")
        responses = [
            self._fake_response([sample_api_row], next_cursor="cur-1"),
            self._fake_response([second_row], next_cursor="cur-2"),
            self._fake_response([third_row], next_cursor=None),
        ]
        captured_params: list[dict] = []

        def _capturing_get(*_args: Any, **kwargs: Any) -> Any:
            # Shallow-copy params: the client mutates the dict in place
            # across pages (pop("since"), set("cursor")), so keeping a
            # reference would collapse all captures to the final state.
            captured_params.append(dict(kwargs.get("params") or {}))
            return responses.pop(0)

        with patch.object(
            mac.requests.Session, "get", side_effect=_capturing_get
        ):
            request_ids = [
                row["request_id"]
                for row in mac.iter_scored_rows(
                    since=datetime(2026, 7, 1, tzinfo=timezone.utc)
                )
            ]
        assert request_ids == ["req-1", "req-2", "req-3"]

        # Argument-aware assertions on the cursor handoff. A broken client
        # that dropped ``params["cursor"] = cursor`` or kept ``since`` on
        # page 2+ would still pass the rows-collected check above; these
        # assertions are what actually validates the plumbing.
        assert len(captured_params) == 3
        assert captured_params[0].get("since") is not None
        assert "cursor" not in captured_params[0]
        assert captured_params[1].get("cursor") == "cur-1"
        assert "since" not in captured_params[1]
        assert captured_params[2].get("cursor") == "cur-2"
        assert "since" not in captured_params[2]

    def test_missing_url_raises_before_any_http(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing ``MECH_ANALYTICS_URL`` raises immediately, before any HTTP call."""
        monkeypatch.delenv("MECH_ANALYTICS_URL", raising=False)
        with pytest.raises(mac.MechAnalyticsError, match="MECH_ANALYTICS_URL"):
            # Consume the generator so the pre-flight config check fires.
            list(mac.iter_scored_rows(since=datetime(2026, 7, 1, tzinfo=timezone.utc)))

    def test_non_list_rows_raises_mech_analytics_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Endpoint schema drift on ``rows`` surfaces as MechAnalyticsError."""
        # Without the isinstance guard a payload like {"rows": {}} would
        # raise an opaque AttributeError mid-iteration. The guard raises
        # a typed error at the boundary instead.
        monkeypatch.setenv("MECH_ANALYTICS_URL", "http://mech-analytics.test")
        bad_response = SimpleNamespace(
            json=lambda: {"rows": {"unexpected": "dict"}, "next_cursor": None},
            raise_for_status=lambda: None,
        )
        with patch.object(
            mac.requests.Session, "get", side_effect=lambda *a, **kw: bad_response
        ):
            with pytest.raises(mac.MechAnalyticsError, match="unexpected type dict"):
                list(
                    mac.iter_scored_rows(
                        since=datetime(2026, 7, 1, tzinfo=timezone.utc)
                    )
                )

    def test_non_dict_payload_raises_mech_analytics_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Top-level array or non-dict payload raises instead of AttributeError."""
        monkeypatch.setenv("MECH_ANALYTICS_URL", "http://mech-analytics.test")
        bad_response = SimpleNamespace(
            json=lambda: ["unexpected", "array"],
            raise_for_status=lambda: None,
        )
        with patch.object(
            mac.requests.Session, "get", side_effect=lambda *a, **kw: bad_response
        ):
            with pytest.raises(mac.MechAnalyticsError, match="expected dict"):
                list(
                    mac.iter_scored_rows(
                        since=datetime(2026, 7, 1, tzinfo=timezone.utc)
                    )
                )

    def test_missing_rows_key_raises_mech_analytics_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing 'rows' key raises (distinct from a legitimate rows=[])."""
        monkeypatch.setenv("MECH_ANALYTICS_URL", "http://mech-analytics.test")
        bad_response = SimpleNamespace(
            json=lambda: {"error": "something went wrong"},
            raise_for_status=lambda: None,
        )
        with patch.object(
            mac.requests.Session, "get", side_effect=lambda *a, **kw: bad_response
        ):
            with pytest.raises(mac.MechAnalyticsError, match="missing 'rows' key"):
                list(
                    mac.iter_scored_rows(
                        since=datetime(2026, 7, 1, tzinfo=timezone.utc)
                    )
                )

    def test_empty_rows_list_ends_pagination_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit rows=[] with null cursor is a valid empty last page."""
        monkeypatch.setenv("MECH_ANALYTICS_URL", "http://mech-analytics.test")
        empty_response = SimpleNamespace(
            json=lambda: {"rows": [], "next_cursor": None},
            raise_for_status=lambda: None,
        )
        with patch.object(
            mac.requests.Session, "get", side_effect=lambda *a, **kw: empty_response
        ):
            rows = list(
                mac.iter_scored_rows(since=datetime(2026, 7, 1, tzinfo=timezone.utc))
            )
        assert rows == []

    def test_max_pages_hit_with_cursor_pending_raises(
        self, monkeypatch: pytest.MonkeyPatch, sample_api_row: dict
    ) -> None:
        """Paginator raises when max_pages is hit with cursor still pending."""
        monkeypatch.setenv("MECH_ANALYTICS_URL", "http://mech-analytics.test")
        always_more = SimpleNamespace(
            json=lambda: {"rows": [sample_api_row], "next_cursor": "cur-x"},
            raise_for_status=lambda: None,
        )
        with patch.object(
            mac.requests.Session, "get", side_effect=lambda *a, **kw: always_more
        ):
            with pytest.raises(mac.MechAnalyticsError, match="max_pages=3"):
                list(
                    mac.iter_scored_rows(
                        since=datetime(2026, 7, 1, tzinfo=timezone.utc),
                        max_pages=3,
                    )
                )
