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
"""Tests for benchmark/roi_sim.py."""

import json
import logging
import random
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, get_args

import pytest
from benchmark.roi_sim import (
    Bet,
    CLAUDE_HARDCODED_MODEL,
    FLAG_NO_ELIGIBLE,
    MAX_BET,
    MODEL_DISPLAY,
    MODEL_UNKNOWN,
    NO_BET_GATES,
    NoBetReason,
    PENDING,
    PLATFORM_GATES,
    REJECT_REASONS,
    RELIABILITY_GATE,
    RejectReason,
    TOURNAMENT_MODEL_OVERRIDES,
    _mode_label,
    _resolve_model,
    cluster_bootstrap_ci,
    compute_group_stats,
    eligibility_reason,
    load_input_rows,
    main,
    render_report,
    simulate,
    simulate_row,
    window_bounds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OMEN = PLATFORM_GATES["omen"]
POLYMARKET = PLATFORM_GATES["polymarket"]

WINDOW_START = datetime(2026, 4, 10, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 7, 9, tzinfo=timezone.utc)


def _row(
    p_yes: Any = 0.7,
    market_prob: Any = 0.5,
    outcome: Any = True,
    status: str = "valid",
    tool: str = "test-tool",
    platform: str = "omen",
    mode: str = "production_replay",
    spread: Any = None,
    predicted_at: Any = "2026-07-01T12:00:00Z",
    market_id: str | None = None,
    row_id: str | None = None,
    model: Any = None,
) -> dict[str, Any]:
    """Build a minimal benchmark row for ROI simulation testing."""
    return {
        "row_id": row_id or f"test_{uuid.uuid4().hex[:12]}",
        "mode": mode,
        "platform": platform,
        "tool_name": tool,
        "model": model,
        "prediction_parse_status": status,
        "p_yes": p_yes,
        "market_prob_at_prediction": market_prob,
        "market_spread_at_prediction": spread,
        "final_outcome": outcome,
        "predicted_at": predicted_at,
        "market_id": market_id or f"m_{uuid.uuid4().hex[:8]}",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to *path* as JSON lines.

    :param path: destination file.
    :param rows: rows to serialize.
    """
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Gate boundaries
# ---------------------------------------------------------------------------


class TestGateBoundaries:
    """Gate comparisons are strict and decimal-rounded."""

    def test_edge_equal_min_edge_no_bet(self) -> None:
        """An edge exactly equal to min_edge does not bet (strict >)."""
        # 0.53 - 0.50 rounds to exactly min_edge 0.03 on Omen.
        result = simulate_row(_row(p_yes=0.53, market_prob=0.50), OMEN)
        assert result == "skip_edge_below_floor"

    def test_edge_just_above_min_edge_bets(self) -> None:
        """An edge strictly above min_edge bets."""
        result = simulate_row(_row(p_yes=0.54, market_prob=0.50), OMEN)
        assert isinstance(result, Bet)

    def test_edge_equal_max_edge_no_bet(self) -> None:
        """An edge exactly equal to max_edge does not bet (strict <)."""
        # 0.85 - 0.55 rounds to exactly max_edge 0.30 on Polymarket.
        row = _row(p_yes=0.85, market_prob=0.55, platform="polymarket")
        assert simulate_row(row, POLYMARKET) == "skip_edge_above_cap"

    def test_edge_just_below_max_edge_bets(self) -> None:
        """An edge strictly below max_edge bets."""
        row = _row(p_yes=0.84, market_prob=0.55, platform="polymarket")
        assert isinstance(simulate_row(row, POLYMARKET), Bet)

    def test_9dp_rounding_kills_ieee_boundary_leak(self) -> None:
        """Raw-float boundary leaks are closed by 9dp rounding.

        0.55 - 0.54 in IEEE floats is 0.010000000000000009 which strictly
        exceeds min_edge 0.01; rounded to 9 decimals it is exactly 0.01 and
        must NOT bet.
        """
        assert (0.55 - 0.54) > 0.01  # documents the raw-float leak
        row = _row(p_yes=0.55, market_prob=0.54, platform="polymarket")
        assert simulate_row(row, POLYMARKET) == "skip_edge_below_floor"

    def test_no_side_complement_edge_boundary_after_rounding(self) -> None:
        """A NO-side edge on the floor only after 9dp rounding must not bet.

        NO side: p_side = 1 - 0.45 = 0.55, price = 1 - 0.48 = 0.52. The raw
        IEEE edge 0.030000000000000027 strictly exceeds Omen's min_edge 0.03
        (would bet without rounding); rounded to 9 decimals it is exactly
        0.03 and must NOT bet. Kills a complement-symmetry mutant that drops
        the round() on NO-side edges.
        """
        assert ((1.0 - 0.45) - (1.0 - 0.48)) > 0.03  # raw-float leak
        assert round((1.0 - 0.45) - (1.0 - 0.48), 9) == 0.03
        result = simulate_row(_row(p_yes=0.45, market_prob=0.48), OMEN)
        assert result == "skip_edge_below_floor"


# ---------------------------------------------------------------------------
# Favored-side floor
# ---------------------------------------------------------------------------


class TestFavoredSideFloor:
    """The oracle-prob floor makes both platforms favored-side-only."""

    def test_p_side_below_half_never_bets(self) -> None:
        """A favored-side probability below 0.5 never bets despite edge."""
        # side = YES (0.45 >= 0.40), edge 0.05 clears Omen's 0.03 floor,
        # but p_side 0.45 < 0.50 so the oracle gate rejects first.
        result = simulate_row(_row(p_yes=0.45, market_prob=0.40), OMEN)
        assert result == "skip_oracle_prob"

    def test_p_side_exactly_half_passes_floor(self) -> None:
        """p_side exactly 0.5 passes the floor (>= comparison)."""
        bet = simulate_row(_row(p_yes=0.5, market_prob=0.40), OMEN)
        assert isinstance(bet, Bet)
        assert bet.side_yes is True

    def test_oracle_floor_9dp_rounding_admits_boundary_row(self) -> None:
        """A raw p_side just below 0.5 that rounds to 0.5 at 9dp must BET.

        p_yes = 0.4999999996 < 0.5 as a raw float, but round(p_side, 9) is
        exactly 0.5 and the floor is >= on the ROUNDED value. Removing the
        round() flips this row to skip_oracle_prob, so this test kills that
        mutant.
        """
        raw_p_yes = 0.4999999996
        assert raw_p_yes < 0.5  # raw float sits below the floor
        assert round(raw_p_yes, 9) == 0.5  # 9dp rounding lands ON it
        bet = simulate_row(_row(p_yes=raw_p_yes, market_prob=0.40), OMEN)
        assert isinstance(bet, Bet)
        assert bet.side_yes is True

    def test_no_side_selection(self) -> None:
        """The NO side is selected when p_yes < market price."""
        bet = simulate_row(_row(p_yes=0.3, market_prob=0.5), OMEN)
        assert isinstance(bet, Bet)
        assert bet.side_yes is False
        # NO-side price = 1 - 0.5 = 0.5.
        assert bet.price == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Spread gate
# ---------------------------------------------------------------------------


class TestSpreadGate:
    """Spread gate: enforced when usable, skipped when missing/NaN/bool."""

    def _pm_row(self, spread: Any) -> dict[str, Any]:
        """Build a Polymarket row that passes all non-spread gates.

        :param spread: market_spread_at_prediction value under test.
        :return: row dict.
        """
        return _row(p_yes=0.65, market_prob=0.55, platform="polymarket", spread=spread)

    def test_missing_spread_skips_gate(self) -> None:
        """A missing (None) spread skips the gate and bets."""
        assert isinstance(simulate_row(self._pm_row(None), POLYMARKET), Bet)

    def test_nan_spread_skips_gate(self) -> None:
        """A NaN spread counts as missing and bets."""
        result = simulate_row(self._pm_row(float("nan")), POLYMARKET)
        assert isinstance(result, Bet)

    def test_bool_spread_skips_gate(self) -> None:
        """A bool spread is not a number and skips the gate."""
        assert isinstance(simulate_row(self._pm_row(True), POLYMARKET), Bet)

    def test_wide_spread_rejects(self) -> None:
        """A spread above spread_max rejects the row."""
        assert simulate_row(self._pm_row(0.15), POLYMARKET) == "skip_spread"

    def test_spread_at_cap_bets(self) -> None:
        """A spread exactly at spread_max passes (<= comparison)."""
        assert isinstance(simulate_row(self._pm_row(0.10), POLYMARKET), Bet)


# ---------------------------------------------------------------------------
# Stake sizing
# ---------------------------------------------------------------------------


class TestStakeSizing:
    """Kelly-proxy stake: f = clamp(edge/(1-a), 0, 1); min(2.5, f*100)."""

    def test_stake_capped_at_max_bet(self) -> None:
        """A large edge caps the stake at MAX_BET."""
        bet = simulate_row(_row(p_yes=0.95, market_prob=0.50), OMEN)
        assert isinstance(bet, Bet)
        assert bet.stake == MAX_BET

    def test_stake_below_cap_uses_kelly_fraction(self) -> None:
        """A small edge stakes f * 100 below the cap."""
        # edge = round(0.561 - 0.55, 9) = 0.011; f = 0.011 / 0.45.
        row = _row(p_yes=0.561, market_prob=0.55, platform="polymarket")
        bet = simulate_row(row, POLYMARKET)
        assert isinstance(bet, Bet)
        assert bet.stake == pytest.approx(0.011 / 0.45 * 100.0)
        assert bet.stake < MAX_BET


# ---------------------------------------------------------------------------
# Haircut variant
# ---------------------------------------------------------------------------


class TestHaircutVariant:
    """Haircut variant shares the mid bet set; collapse keeps the row."""

    def test_haircut_collapse_keeps_row_with_zero_pnl(self) -> None:
        """A haircut edge at/below the floor zeroes stake_h/pnl_h only."""
        # mid edge 0.04 bets on Omen; haircut edge 0.55 - 0.53 = 0.02 <= 0.03.
        bet = simulate_row(_row(p_yes=0.55, market_prob=0.51), OMEN)
        assert isinstance(bet, Bet)
        assert bet.stake > 0.0
        assert bet.stake_haircut == 0.0
        assert bet.pnl_haircut == 0.0

    def test_haircut_prices_and_pnl(self) -> None:
        """The haircut variant re-sizes and settles at price + haircut."""
        bet = simulate_row(_row(p_yes=0.70, market_prob=0.50, outcome=True), OMEN)
        assert isinstance(bet, Bet)
        # a_h = 0.52; edge_h = 0.18; f_h = 0.18/0.48 -> stake capped at 2.5.
        assert bet.stake_haircut == MAX_BET
        assert bet.pnl_haircut == pytest.approx(MAX_BET * (1.0 / 0.52 - 1.0))


# ---------------------------------------------------------------------------
# Settlement arithmetic
# ---------------------------------------------------------------------------


class TestSettlement:
    """Winner-take-all settlement at the clamped buy price."""

    def test_yes_side_win(self) -> None:
        """A winning YES bet pays stake * (1/a - 1)."""
        bet = simulate_row(_row(p_yes=0.7, market_prob=0.5, outcome=True), OMEN)
        assert isinstance(bet, Bet)
        assert bet.win is True
        assert bet.pnl == pytest.approx(bet.stake * (1.0 / 0.5 - 1.0))
        assert (bet.stake + bet.pnl) > 0

    def test_yes_side_loss(self) -> None:
        """A losing YES bet loses exactly its stake."""
        bet = simulate_row(_row(p_yes=0.7, market_prob=0.5, outcome=False), OMEN)
        assert isinstance(bet, Bet)
        assert bet.win is False
        assert bet.pnl == pytest.approx(-bet.stake)

    def test_no_side_win_on_false_outcome(self) -> None:
        """A NO bet wins when the outcome is False."""
        bet = simulate_row(_row(p_yes=0.3, market_prob=0.5, outcome=False), OMEN)
        assert isinstance(bet, Bet)
        assert bet.win is True
        assert bet.pnl == pytest.approx(bet.stake * (1.0 / 0.5 - 1.0))

    def test_no_side_loss_on_true_outcome(self) -> None:
        """A NO bet loses when the outcome is True."""
        bet = simulate_row(_row(p_yes=0.3, market_prob=0.5, outcome=True), OMEN)
        assert isinstance(bet, Bet)
        assert bet.win is False
        assert bet.pnl == pytest.approx(-bet.stake)


# ---------------------------------------------------------------------------
# Pooled ROI
# ---------------------------------------------------------------------------


class TestPooledRoi:
    """ROI is pooled capital-weighted, never a mean of per-bet ROIs."""

    def test_pooled_roi_not_mean_of_per_bet_rois(self) -> None:
        """Group ROI equals 100*sum(pnl)/sum(stake), not the per-bet mean."""
        rows = [
            _row(
                p_yes=0.561,
                market_prob=0.55,
                outcome=True,
                platform="polymarket",
                market_id="m1",
            ),
            _row(
                p_yes=0.85,
                market_prob=0.56,
                outcome=False,
                platform="polymarket",
                market_id="m2",
            ),
        ]
        sims = [simulate_row(row, POLYMARKET) for row in rows]
        bets = [bet for bet in sims if isinstance(bet, Bet)]
        assert len(bets) == len(rows)
        stakes = [bet.stake for bet in bets]
        pnls = [bet.pnl for bet in bets]
        pooled = 100.0 * sum(pnls) / sum(stakes)
        per_bet_mean = sum(
            100.0 * pnl / stake for pnl, stake in zip(pnls, stakes)
        ) / len(bets)

        stats = compute_group_stats(rows, POLYMARKET)
        assert stats["n_bets"] == 2
        assert stats["staked"] == pytest.approx(sum(stakes))
        assert stats["roi_mid"] == pytest.approx(pooled)
        # The two definitions genuinely differ on this bet set.
        assert abs(pooled - per_bet_mean) > 1.0
        assert stats["roi_mid"] != pytest.approx(per_bet_mean)
        # Exact companion stats: one win out of two bets, and with only two
        # markets the top-3 concentration share is exactly 1.0.
        assert stats["win_rate"] == 0.5
        assert stats["top3_pnl_share"] == 1.0


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCi:
    """Market-clustered bootstrap with a fresh fixed-seed RNG per call."""

    _BETS = [
        ("m1", 2.5, 2.5),
        ("m2", 2.5, -2.5),
        ("m3", 2.0, 1.0),
        ("m1", 1.0, -1.0),
    ]

    def test_same_rows_identical_ci_twice(self) -> None:
        """Two calls on the same bets return bit-identical bounds."""
        assert cluster_bootstrap_ci(self._BETS) == cluster_bootstrap_ci(self._BETS)

    def test_ci_isolated_from_global_random_state(self) -> None:
        """Global random state never leaks into the CI computation."""
        random.seed(0)
        first = cluster_bootstrap_ci(self._BETS)
        random.seed(999)
        random.random()
        assert cluster_bootstrap_ci(self._BETS) == first

    def test_ci_none_below_two_clusters(self) -> None:
        """Fewer than 2 market clusters yields no CI."""
        assert cluster_bootstrap_ci([("m1", 2.0, 1.0), ("m1", 1.0, -1.0)]) is None

    def test_zero_stake_rows_do_not_form_clusters(self) -> None:
        """Zero-stake rows are excluded before clustering."""
        assert cluster_bootstrap_ci([("m1", 0.0, 0.0), ("m2", 2.0, 1.0)]) is None

    def test_ci_bounds_ordered(self) -> None:
        """CI low bound never exceeds the high bound."""
        ci = cluster_bootstrap_ci(self._BETS)
        assert ci is not None
        assert ci[0] <= ci[1]

    def test_haircut_ci_deterministic_and_mid_ci_unperturbed(self) -> None:
        """Haircut CI is reproducible and never perturbs the mid CI.

        Each CI call uses a FRESH Random(seed): computing the haircut CI in
        between two mid-CI computations must leave the mid CI bit-identical,
        and two group-stat computations must agree on both CIs. Kills a
        mutant that shares one RNG across the two CI calls.
        """
        rows = [
            _row(p_yes=0.70, market_prob=0.50, outcome=True, market_id="m1"),
            _row(p_yes=0.72, market_prob=0.52, outcome=False, market_id="m2"),
            _row(p_yes=0.68, market_prob=0.50, outcome=True, market_id="m3"),
            _row(p_yes=0.80, market_prob=0.55, outcome=False, market_id="m4"),
        ]
        bets = [b for b in (simulate_row(r, OMEN) for r in rows) if isinstance(b, Bet)]
        assert len(bets) == 4
        mid_rows = [(b.market_id, b.stake, b.pnl) for b in bets]
        haircut_rows = [(b.market_id, b.stake_haircut, b.pnl_haircut) for b in bets]

        mid_before = cluster_bootstrap_ci(mid_rows)
        haircut_first = cluster_bootstrap_ci(haircut_rows)
        mid_after = cluster_bootstrap_ci(mid_rows)
        haircut_second = cluster_bootstrap_ci(haircut_rows)
        assert mid_before is not None
        assert haircut_first is not None
        assert haircut_first == haircut_second
        assert mid_before == mid_after

        stats_1 = compute_group_stats(rows, OMEN)
        stats_2 = compute_group_stats(rows, OMEN)
        assert stats_1["roi_haircut_ci"] == list(haircut_first)
        assert stats_1["roi_haircut_ci"] == stats_2["roi_haircut_ci"]
        assert stats_1["roi_ci"] == list(mid_before)
        assert stats_1["roi_ci"] == stats_2["roi_ci"]
        # Exact non-degenerate companion stats on this 4-market bet set:
        # 2 wins / 4 bets, and all four |per-market PnL| are 2.5 (every
        # stake caps at MAX_BET on a 0.5-ish price), so the top-3 share is
        # exactly 7.5 / 10.0.
        assert stats_1["win_rate"] == 0.5
        assert stats_1["top3_pnl_share"] == 0.75


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestDedup:
    """row_id dedup: first-seen wins, files scanned in sorted name order.

    :param tmp_path: pytest tmp_path fixture (per test).
    """

    def test_first_seen_wins_across_shards(self, tmp_path: Path) -> None:
        """A duplicated row_id keeps the copy from the earlier shard.

        :param tmp_path: pytest tmp_path fixture.
        """
        logs = tmp_path / "logs"
        logs.mkdir()
        dup = _row(p_yes=0.6, row_id="prod_dup_1")
        later = dict(dup, p_yes=0.9)
        _write_jsonl(logs / "production_log_2026_07_02.jsonl", [later])
        _write_jsonl(logs / "production_log_2026_07_01.jsonl", [dup])
        rows = load_input_rows(logs, tmp_path / "missing_tournament.jsonl")
        assert len(rows) == 1
        assert rows[0]["p_yes"] == 0.6  # sorted filename order: 07_01 first

    def test_tournament_rows_appended_after_shards(self, tmp_path: Path) -> None:
        """Tournament rows load after all shards, with their own row_ids.

        :param tmp_path: pytest tmp_path fixture.
        """
        logs = tmp_path / "logs"
        logs.mkdir()
        _write_jsonl(logs / "production_log_2026_07_01.jsonl", [_row(row_id="prod_a")])
        tournament = tmp_path / "tournament_scored.jsonl"
        _write_jsonl(tournament, [_row(row_id="tourn_a", mode="tournament")])
        rows = load_input_rows(logs, tournament)
        assert [row["row_id"] for row in rows] == ["prod_a", "tourn_a"]


# ---------------------------------------------------------------------------
# Eligibility ladder
# ---------------------------------------------------------------------------


class TestEligibility:
    """Eligibility ladder rejects with the first failing reason."""

    def _reason(self, row: dict[str, Any]) -> str | None:
        """Run the ladder against the fixed test window.

        :param row: row under test.
        :return: reject reason or None.
        """
        return eligibility_reason(row, WINDOW_START, WINDOW_END)

    def test_valid_row_is_eligible(self) -> None:
        """A well-formed in-window row passes the ladder."""
        assert self._reason(_row()) is None

    def test_invalid_parse_status(self) -> None:
        """A non-valid parse status rejects."""
        assert self._reason(_row(status="malformed")) == "invalid_parse"

    def test_bool_p_yes_rejects(self) -> None:
        """A bool p_yes is not a probability."""
        assert self._reason(_row(p_yes=True)) == "bad_p_yes"

    def test_nan_p_yes_rejects(self) -> None:
        """A NaN p_yes rejects."""
        assert self._reason(_row(p_yes=float("nan"))) == "bad_p_yes"

    def test_out_of_range_p_yes_rejects(self) -> None:
        """A p_yes outside [0, 1] rejects."""
        assert self._reason(_row(p_yes=1.5)) == "bad_p_yes"

    def test_p_yes_bounds_are_closed(self) -> None:
        """p_yes of exactly 0.0 and 1.0 are eligible (closed interval)."""
        assert self._reason(_row(p_yes=0.0)) is None
        assert self._reason(_row(p_yes=1.0)) is None

    def test_market_prob_bounds_are_open(self) -> None:
        """market_prob of exactly 0.0 / 1.0 rejects (open interval)."""
        assert self._reason(_row(market_prob=0.0)) == "bad_market_prob"
        assert self._reason(_row(market_prob=1.0)) == "bad_market_prob"

    def test_string_outcome_rejects(self) -> None:
        """A string final_outcome is not a strict JSON bool."""
        assert self._reason(_row(outcome="true")) == "bad_final_outcome"

    def test_null_outcome_is_pending(self) -> None:
        """A null final_outcome (unresolved) classifies as pending."""
        assert self._reason(_row(outcome=None)) == "pending"

    def test_pending_vs_bad_final_outcome_ladder(self) -> None:
        """Null is PENDING; any other non-bool is bad_final_outcome.

        The certified ladder distinguishes an unresolved row (null) from a
        malformed one ("true" strings, 0/1 ints); collapsing the two loses
        the pending count and inflates the reject counters.
        """
        assert self._reason(_row(outcome=None)) == "pending"
        assert self._reason(_row(outcome="true")) == "bad_final_outcome"
        assert self._reason(_row(outcome=0)) == "bad_final_outcome"
        assert self._reason(_row(outcome=1)) == "bad_final_outcome"
        assert self._reason(_row(outcome=True)) is None
        assert self._reason(_row(outcome=False)) is None

    def test_unparseable_predicted_at_is_out_of_window(self) -> None:
        """An unparseable predicted_at counts as out_of_window."""
        assert self._reason(_row(predicted_at="garbage")) == "out_of_window"

    def test_out_of_window_rejects(self) -> None:
        """A predicted_at before the window start rejects."""
        row = _row(predicted_at="2026-01-01T00:00:00Z")
        assert self._reason(row) == "out_of_window"

    def test_window_checked_before_parse_status(self) -> None:
        """The ladder order pins out_of_window ahead of invalid_parse."""
        row = _row(predicted_at="2026-01-01T00:00:00Z", status="malformed")
        assert self._reason(row) == "out_of_window"

    def test_naive_timestamp_treated_as_utc(self) -> None:
        """A naive predicted_at is interpreted as UTC."""
        assert self._reason(_row(predicted_at="2026-07-01T12:00:00")) is None


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


class TestWindowing:
    """Trailing window semantics around --as-of."""

    def test_window_bounds_include_whole_as_of_day(self) -> None:
        """The window ends at midnight UTC of as-of + 1, exclusive."""
        start, end = window_bounds(date(2026, 7, 8), 1)
        assert start == datetime(2026, 7, 8, tzinfo=timezone.utc)
        assert end == datetime(2026, 7, 9, tzinfo=timezone.utc)

    def test_window_length_in_days(self) -> None:
        """The window spans exactly window_days days."""
        start, end = window_bounds(date(2026, 7, 8), 90)
        assert (end - start).days == 90
        assert start == datetime(2026, 4, 10, tzinfo=timezone.utc)

    def test_as_of_day_inclusive_next_day_exclusive(self) -> None:
        """Rows late on the as-of day are in; as-of + 1 midnight is out."""
        start, end = window_bounds(date(2026, 7, 8), 1)
        late = _row(predicted_at="2026-07-08T23:59:59Z")
        assert eligibility_reason(late, start, end) is None
        next_day = _row(predicted_at="2026-07-09T00:00:00Z")
        assert eligibility_reason(next_day, start, end) == "out_of_window"

    def test_window_start_inclusive(self) -> None:
        """A row exactly at window_start is in the window."""
        start, end = window_bounds(date(2026, 7, 8), 1)
        boundary = _row(predicted_at="2026-07-08T00:00:00Z")
        assert eligibility_reason(boundary, start, end) is None


# ---------------------------------------------------------------------------
# Group counters (window-scoped n_rows_seen, n_pending, no-bet attribution)
# ---------------------------------------------------------------------------


class TestGroupCounters:
    """simulate() group counters follow the certified contract."""

    def test_n_rows_seen_window_scoped_and_pending_counted(self) -> None:
        """n_rows_seen counts only in-window rows; null outcome is pending.

        Out-of-window and unparseable-timestamp rows stay OUT of
        n_rows_seen (they are still counted as out_of_window rejects);
        pending rows count in n_rows_seen and n_pending, never in rejects.
        """
        rows = [
            _row(),  # eligible
            _row(outcome=None),  # in-window, pending
            _row(predicted_at="2026-01-01T00:00:00Z"),  # out of window
            _row(predicted_at="garbage"),  # unparseable -> out_of_window
            _row(status="malformed"),  # in-window, invalid_parse
        ]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        assert len(groups) == 1
        group = groups[0]
        assert group["n_rows_seen"] == 3  # eligible + pending + invalid_parse
        assert group["n_pending"] == 1
        assert group["n_eligible"] == 1
        assert group["rejects"]["out_of_window"] == 2
        assert group["rejects"]["invalid_parse"] == 1
        assert group["rejects"]["bad_final_outcome"] == 0

    def test_no_bet_gate_attribution_and_coverage(self) -> None:
        """No-bets are attributed to their first-failing gate per group."""
        rows = [
            _row(p_yes=0.70, market_prob=0.50),  # bet
            _row(p_yes=0.45, market_prob=0.40),  # skip_oracle_prob
            _row(p_yes=0.53, market_prob=0.50),  # skip_edge_below_floor
            _row(p_yes=0.45, market_prob=0.40),  # skip_oracle_prob
        ]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        assert len(groups) == 1
        group = groups[0]
        assert set(group["no_bet"]) == set(NO_BET_GATES)
        assert group["no_bet"]["skip_oracle_prob"] == 2
        assert group["no_bet"]["skip_edge_below_floor"] == 1
        assert group["no_bet"]["skip_edge_above_cap"] == 0
        assert group["no_bet"]["skip_spread"] == 0
        assert group["no_bet"]["skip_zero_stake"] == 0
        assert group["n_bets"] == 1
        assert group["coverage_pct"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """main() over synthetic shards: outputs parse and are deterministic."""

    def _build_inputs(self, tmp_path: Path) -> tuple[Path, Path]:
        """Write 2 platforms x 2 tools of synthetic rows.

        :param tmp_path: pytest tmp_path fixture.
        :return: (logs_dir, tournament_input) paths.
        """
        logs = tmp_path / "logs"
        logs.mkdir()
        production = [
            _row(tool="tool-a", p_yes=0.7, market_prob=0.5, outcome=True),
            _row(tool="tool-a", p_yes=0.8, market_prob=0.5, outcome=False),
            _row(tool="tool-b", p_yes=0.65, market_prob=0.5, outcome=True),
            _row(tool="tool-b", p_yes=0.75, market_prob=0.5, outcome=True),
            _row(
                tool="tool-a",
                platform="polymarket",
                p_yes=0.65,
                market_prob=0.55,
                spread=0.05,
                outcome=True,
            ),
            _row(
                tool="tool-a",
                platform="polymarket",
                p_yes=0.70,
                market_prob=0.55,
                spread=0.05,
                outcome=False,
            ),
        ]
        _write_jsonl(logs / "production_log_2026_07_01.jsonl", production)
        tournament = [
            _row(tool="tool-a", mode="tournament", p_yes=0.7, market_prob=0.5),
            _row(tool="tool-a", mode="tournament", p_yes=0.6, market_prob=0.5),
            _row(
                tool="tool-b",
                mode="tournament",
                platform="polymarket",
                p_yes=0.65,
                market_prob=0.55,
            ),
            _row(
                tool="tool-b",
                mode="tournament",
                platform="polymarket",
                p_yes=0.75,
                market_prob=0.55,
                outcome=False,
            ),
        ]
        tournament_path = tmp_path / "tournament_scored.jsonl"
        _write_jsonl(tournament_path, tournament)
        return logs, tournament_path

    def _run_main(
        self,
        monkeypatch: pytest.MonkeyPatch,
        logs: Path,
        tournament: Path,
        results: Path,
    ) -> None:
        """Invoke main() with a synthetic argv.

        :param monkeypatch: pytest monkeypatch fixture.
        :param logs: logs directory.
        :param tournament: tournament_scored.jsonl path.
        :param results: results output directory.
        """
        argv = [
            "roi_sim",
            "--logs-dir",
            str(logs),
            "--tournament-input",
            str(tournament),
            "--results-dir",
            str(results),
            "--as-of",
            "2026-07-08",
            "--window-days",
            "90",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        main()

    def test_smoke_outputs_parse_and_are_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both md reports + json are produced, sane, and reproducible.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        """
        logs, tournament = self._build_inputs(tmp_path)
        results_1 = tmp_path / "results_1"
        self._run_main(monkeypatch, logs, tournament, results_1)

        payload = json.loads(
            (results_1 / "roi_results.json").read_text(encoding="utf-8")
        )
        assert payload["as_of"] == "2026-07-08"
        assert payload["window_days"] == 90
        groups = payload["groups"]
        keys = {(g["platform"], g["tool_name"], g["mode"], g["model"]) for g in groups}
        assert ("omen", "tool-a", "production", MODEL_UNKNOWN) in keys
        assert ("omen", "tool-a", "tournament", MODEL_UNKNOWN) in keys
        assert ("polymarket", "tool-a", "production", MODEL_UNKNOWN) in keys
        assert ("polymarket", "tool-b", "tournament", MODEL_UNKNOWN) in keys
        for group in groups:
            assert group["n_eligible"] <= group["n_rows_seen"]
            assert group["n_pending"] == 0  # all synthetic rows are resolved
            assert set(group["rejects"]) == {
                "out_of_window",
                "invalid_parse",
                "bad_p_yes",
                "bad_market_prob",
                "bad_final_outcome",
            }
            assert set(group["no_bet"]) == set(NO_BET_GATES)
            assert (
                group["n_bets"] + sum(group["no_bet"].values()) == group["n_eligible"]
            )
            assert group["coverage_pct"] is not None
            assert "roi_haircut_ci" in group
            assert "few bets - anecdotal" in group["flags"]  # all n_bets < 10
            assert "low sample" in group["flags"]  # all n_eligible < 30
            assert group["is_prediction_tool"] is True  # all rows parse
            assert group["parse_reliability"] == pytest.approx(1.0)

        report_omen = (results_1 / "report_roi_omen.md").read_text(encoding="utf-8")
        report_poly = (results_1 / "report_roi_polymarket.md").read_text(
            encoding="utf-8"
        )
        for report in (report_omen, report_poly):
            assert "| tool | mode | model | n preds | n bets |" in report
            assert "tool-a" in report
        # Production rows sort before tournament rows in the table.
        assert report_omen.index("| tool-a | production |") < report_omen.index(
            "| tool-a | tournament |"
        )

        # Byte-identical on a second run with the same artifacts + as-of.
        results_2 = tmp_path / "results_2"
        self._run_main(monkeypatch, logs, tournament, results_2)
        for name in (
            "roi_results.json",
            "report_roi_omen.md",
            "report_roi_polymarket.md",
        ):
            assert (results_1 / name).read_bytes() == (results_2 / name).read_bytes()


# ---------------------------------------------------------------------------
# Tool policy (is_prediction_tool, parse_reliability, rendering)
# ---------------------------------------------------------------------------


class TestToolPolicy:
    """Data-driven tool policy: no allowlist, nothing silently dropped."""

    def _render(self, groups: list[dict[str, Any]], platform: str = "omen") -> str:
        """Render a platform report against the fixed test window.

        :param groups: simulate() output.
        :param platform: platform key.
        :return: markdown report text.
        """
        return render_report(
            platform, groups, date(2026, 7, 8), 90, WINDOW_START, WINDOW_END
        )

    def test_non_prediction_tool_json_and_excluded_line(self) -> None:
        """A never-parsing tool leaves the md table but stays in JSON.

        A tool with no valid-parse row anywhere is excluded from the md
        table but kept in JSON with is_prediction_tool False, and listed
        on the Excluded line.
        """
        rows = [
            _row(tool="pred-tool"),
            _row(tool="propose-question", status="malformed"),
            _row(tool="propose-question", status="malformed"),
        ]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        by_tool = {g["tool_name"]: g for g in groups}
        assert by_tool["propose-question"]["is_prediction_tool"] is False
        assert by_tool["propose-question"]["parse_reliability"] == 0.0
        assert by_tool["pred-tool"]["is_prediction_tool"] is True

        report = self._render(groups)
        assert "| pred-tool |" in report
        assert "| propose-question |" not in report
        assert (
            "Excluded (no parseable prediction in any row): "
            "propose-question (2 rows)" in report
        )
        assert "indicates a parser/format gap" in report

    def test_excluded_line_sorted_by_row_count_desc(self) -> None:
        """The Excluded line sorts non-prediction tools by row count desc."""
        rows = (
            [_row(tool="pred-tool")]
            + [_row(tool="gen-small", status="malformed")]
            + [_row(tool="gen-big", status="malformed") for _ in range(3)]
        )
        report = self._render(simulate(rows, WINDOW_START, WINDOW_END))
        assert "gen-big (3 rows), gen-small (1 row)" in report

    def test_zero_eligible_prediction_tool_stays_in_table(self) -> None:
        """A prediction tool with zero eligible rows stays in the table.

        Valid-parse rows OUTSIDE the window classify the tool as a
        prediction tool; with zero eligible in-window rows the row stays
        in the table flagged 'no eligible rows in window'.
        """
        rows = [
            _row(tool="stale-tool", predicted_at="2026-01-01T00:00:00Z"),
            _row(tool="live-tool"),
        ]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        stale = next(g for g in groups if g["tool_name"] == "stale-tool")
        assert stale["is_prediction_tool"] is True
        assert stale["n_eligible"] == 0
        assert stale["parse_reliability"] is None  # denominator 0 in window
        assert stale["flags"] == [FLAG_NO_ELIGIBLE]

        report = self._render(groups)
        assert "| stale-tool | production | unknown | 0 | 0 |" in report
        assert FLAG_NO_ELIGIBLE in report
        assert "Excluded" not in report  # both tools are prediction tools

    def test_parse_reliability_at_gate_exactly_no_flag(self) -> None:
        """parse_reliability exactly at the 0.80 gate is NOT flagged."""
        rows = [_row() for _ in range(4)] + [_row(status="malformed")]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        assert len(groups) == 1
        group = groups[0]
        assert group["parse_reliability"] == pytest.approx(RELIABILITY_GATE)
        assert not any("parse reliability" in flag for flag in group["flags"])

    def test_parse_reliability_below_gate_flag_text(self) -> None:
        """Below-gate parse reliability yields the percentage flag.

        The flag text appears in both the JSON flags and the rendered
        table.
        """
        rows = [_row(), _row()] + [_row(status="malformed") for _ in range(3)]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        assert len(groups) == 1
        group = groups[0]
        expected = "⚠ 40% parse reliability — possible response-format gap"
        assert group["parse_reliability"] == pytest.approx(0.4)
        assert expected in group["flags"]
        assert expected in self._render(groups)

    def test_parse_reliability_arithmetic_counts_at_parse_rung(self) -> None:
        """parse_reliability counts at the parse rung, in-window only.

        Later-rung failures and pending rows count as valid-parse;
        out-of-window rows are excluded entirely.
        """
        rows = [
            _row(),  # eligible
            _row(outcome=None),  # pending -- passes parse rung
            _row(p_yes=1.5),  # bad_p_yes -- passes parse rung
            _row(status="malformed"),  # invalid_parse
            _row(status="malformed", predicted_at="2026-01-01T00:00:00Z"),
        ]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        assert len(groups) == 1
        group = groups[0]
        assert group["n_rows_seen"] == 4
        assert group["rejects"]["invalid_parse"] == 1
        assert group["parse_reliability"] == pytest.approx(3 / 4)

    def test_no_platform_data_still_lists_excluded(self) -> None:
        """Only non-prediction groups: no-data line plus Excluded line."""
        rows = [_row(tool="gen-only", status="malformed")]
        report = self._render(simulate(rows, WINDOW_START, WINDOW_END))
        assert "_No data for this platform in the window._" in report
        assert "gen-only (1 row)" in report


# ---------------------------------------------------------------------------
# Input integrity (malformed lines, corrupt shards, empty input)
# ---------------------------------------------------------------------------


class TestInputIntegrity:
    """Broken input aborts loudly instead of publishing an empty report."""

    def test_malformed_lines_skipped_and_counted(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Garbage / JSON-array / empty lines are skipped; valid row loads.

        The shard holds 4 lines (garbage, JSON array, blank, valid row):
        2 bad out of 4 total is exactly the 50% threshold, NOT above it,
        so the run proceeds with the valid row and a WARNING carrying the
        per-file bad-line count.

        :param tmp_path: pytest tmp_path fixture.
        :param caplog: pytest log-capture fixture.
        """
        logs = tmp_path / "logs"
        logs.mkdir()
        shard = logs / "production_log_2026_07_01.jsonl"
        valid = _row(row_id="good_row")
        shard.write_text(
            "{not json at all\n"
            + json.dumps([1, 2, 3])
            + "\n"
            + "\n"
            + json.dumps(valid)
            + "\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="benchmark.roi_sim"):
            rows = load_input_rows(logs, tmp_path / "missing_tournament.jsonl")
        assert len(rows) == 1
        assert rows[0]["row_id"] == "good_row"
        assert "skipped 2 unparseable line(s) out of 4" in caplog.text
        assert shard.name in caplog.text

    def test_mostly_garbage_shard_aborts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A shard with >50% unparseable lines exits 1 with an annotation.

        :param tmp_path: pytest tmp_path fixture.
        :param capsys: pytest capture fixture.
        """
        logs = tmp_path / "logs"
        logs.mkdir()
        shard = logs / "production_log_2026_07_01.jsonl"
        shard.write_text(
            "garbage-1\ngarbage-2\ngarbage-3\n" + json.dumps(_row()) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as excinfo:
            load_input_rows(logs, tmp_path / "missing_tournament.jsonl")
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "::error::roi_sim:" in out
        assert "failed to parse" in out

    def test_zero_rows_loaded_aborts_main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() with no input rows exits 1 instead of a green empty run.

        Defense-in-depth companion to the workflow's
        ``if_no_artifact_found: fail`` download step: an artifact-layout
        drift that leaves the logs dir empty must fail the run.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capture fixture.
        """
        logs = tmp_path / "logs"
        logs.mkdir()  # exists but holds no shards
        argv = [
            "roi_sim",
            "--logs-dir",
            str(logs),
            "--tournament-input",
            str(tmp_path / "missing_tournament.jsonl"),
            "--results-dir",
            str(tmp_path / "results"),
            "--as-of",
            "2026-07-08",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "::error::roi_sim: no input rows loaded — check artifact layout" in out
        assert not (tmp_path / "results").exists()  # nothing was published


# ---------------------------------------------------------------------------
# Golden values (bootstrap CI quantiles, Brier formula)
# ---------------------------------------------------------------------------


class TestGoldenValues:
    """Exact expected numbers pin the statistical formulas against mutation."""

    def test_bootstrap_ci_golden_value(self) -> None:
        """cluster_bootstrap_ci reproduces the pinned reference draw exactly.

        Derivation (self-contained reference re-implementation below): the
        six bet rows aggregate into five market clusters in first-occurrence
        order -- m1 (3.5, 1.5), m2 (2.5, -2.5), m3 (2.0, 1.0),
        m4 (1.5, -1.5), m5 (3.0 pnl on 1.0 stake). A fresh
        ``random.Random(12345)`` draws B = 2000 replicates of n = 5 clusters
        via ``randrange(n)``; each replicate's statistic is
        100 * sum(pnl) / sum(stake); bounds are the pinned index convention
        ``sorted_samples[int(0.025 * N)]`` / ``[int(0.975 * N)]``. The
        reference is INDEPENDENT of the module's constants, so silently
        widening/narrowing the quantiles (e.g. 0.025/0.975 -> 0.05/0.95,
        which yields (-60.0, 100.0) here), changing the seed, B, or the
        index convention all fail this test. The hard-coded literals pin
        the same values a second time, independent of the reference code.
        """
        bet_rows = [
            ("m1", 2.5, 2.5),
            ("m2", 2.5, -2.5),
            ("m3", 2.0, 1.0),
            ("m4", 1.5, -1.5),
            ("m5", 1.0, 3.0),
            ("m1", 1.0, -1.0),
        ]
        # Reference algorithm, re-implemented independently of the module.
        sums = [(3.5, 1.5), (2.5, -2.5), (2.0, 1.0), (1.5, -1.5), (1.0, 3.0)]
        n_clusters = len(sums)
        rng = random.Random(12345)
        samples: list[float] = []
        for _ in range(2000):
            stake_total = 0.0
            pnl_total = 0.0
            for _ in range(n_clusters):
                stake_part, pnl_part = sums[rng.randrange(n_clusters)]
                stake_total += stake_part
                pnl_total += pnl_part
            if stake_total <= 0.0:
                continue
            samples.append(100.0 * pnl_total / stake_total)
        samples.sort()
        n_kept = len(samples)
        expected = (samples[int(0.025 * n_kept)], samples[int(0.975 * n_kept)])

        ci = cluster_bootstrap_ci(bet_rows)
        assert ci is not None
        assert ci == expected  # bit-identical to the reference draw
        assert (round(ci[0], 6), round(ci[1], 6)) == (-66.666667, 120.0)

    def test_brier_golden_values(self) -> None:
        """Brier columns match hand-computed exact values.

        Two eligible Omen rows: p_yes 0.8 / outcome True (Brier 0.04, bets:
        edge 0.30 > 0.03) and p_yes 0.6 / outcome False (Brier 0.36, no bet:
        edge 0.02 <= 0.03). brier_all = (0.04 + 0.36) / 2 = 0.20 while
        brier_bets = 0.04 alone -- dropping the square yields -0.2 there
        and fails. A one-row group pins 0.36 = (0.6 - 0)^2 on brier_all
        too (the un-squared value would be 0.6).
        """
        rows = [
            _row(p_yes=0.8, market_prob=0.5, outcome=True, market_id="m1"),
            _row(p_yes=0.6, market_prob=0.58, outcome=False, market_id="m2"),
        ]
        stats = compute_group_stats(rows, OMEN)
        assert stats["n_bets"] == 1  # only the 0.8 row clears the edge floor
        assert stats["brier_all"] == pytest.approx(0.20)
        assert stats["brier_bets"] == pytest.approx(0.04)

        lone = [_row(p_yes=0.6, market_prob=0.5, outcome=False)]
        assert compute_group_stats(lone, OMEN)["brier_all"] == pytest.approx(0.36)


# ---------------------------------------------------------------------------
# Sentinel types stay in sync with the runtime tuples
# ---------------------------------------------------------------------------


class TestSentinelTypes:
    """Literal aliases mirror the runtime reason tuples exactly."""

    def test_literal_aliases_match_runtime_tuples(self) -> None:
        """NoBetReason/RejectReason and the tuples must never drift apart."""
        assert set(get_args(NoBetReason)) == set(NO_BET_GATES)
        assert set(get_args(RejectReason)) == set(REJECT_REASONS) | {PENDING}


# ---------------------------------------------------------------------------
# Deterministic ordering (tie-breaks) + row-classification edges
# ---------------------------------------------------------------------------


class TestDeterministicOrdering:
    """Ordering under ties is part of the byte-reproducibility contract."""

    def test_report_sort_tool_name_tie_break_on_equal_n_bets(self) -> None:
        """Two tools with equal n_bets sort by tool_name in the report."""
        rows = [  # inserted in REVERSE of the expected report order
            _row(tool="b-tool", p_yes=0.7, market_prob=0.5),
            _row(tool="a-tool", p_yes=0.7, market_prob=0.5),
        ]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        by_tool = {g["tool_name"]: g for g in groups}
        # Precondition: the -n_bets sort key genuinely ties.
        assert by_tool["a-tool"]["n_bets"] == by_tool["b-tool"]["n_bets"] == 1
        report = render_report(
            "omen", groups, date(2026, 7, 8), 90, WINDOW_START, WINDOW_END
        )
        assert report.index("| a-tool |") < report.index("| b-tool |")

    def test_simulate_group_ordering_tie_breaks(self) -> None:
        """simulate() orders groups by platform, prod-first, mode, tool."""
        rows = [  # inserted in REVERSE of the expected sorted order
            _row(tool="tool-a", platform="polymarket"),
            _row(tool="tool-a", mode="tournament"),
            _row(tool="tool-a", mode="shadow"),
            _row(tool="tool-b"),
            _row(tool="tool-a"),
        ]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        assert [(g["platform"], g["mode"], g["tool_name"]) for g in groups] == [
            ("omen", "production", "tool-a"),
            ("omen", "production", "tool-b"),
            ("omen", "shadow", "tool-a"),
            ("omen", "tournament", "tool-a"),
            ("polymarket", "production", "tool-a"),
        ]

    def test_unrecognized_platform_rows_warned_and_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Rows on a platform without a gate config are skipped with a warn.

        :param caplog: pytest log-capture fixture.
        """
        with caplog.at_level(logging.WARNING, logger="benchmark.roi_sim"):
            groups = simulate([_row(platform="kalshi")], WINDOW_START, WINDOW_END)
        assert not groups
        assert "platforms without a gate config" in caplog.text

    def test_mode_label_unknown_mode_passthrough(self) -> None:
        """An unknown mode string passes through verbatim (stays visible)."""
        assert _mode_label({"mode": "shadow"}) == "shadow"


# ---------------------------------------------------------------------------
# Model column (provenance, per-model split, display, ordering)
# ---------------------------------------------------------------------------


class TestModelColumn:
    """Underlying-LLM column: provenance rules + per-model group split."""

    def test_same_tool_two_models_split_and_sums_match_single_group(self) -> None:
        """Two models within one (platform, tool, mode) yield two rows.

        The model dimension only PARTITIONS a group -- gates and stakes
        never read it -- so summing n_eligible / n_bets / staked / PnL over
        the model-split rows must reproduce the old single-group row
        computed on the same data under one common model.
        """
        base: list[dict[str, Any]] = [
            {"p_yes": 0.70, "market_prob": 0.50, "outcome": True, "market_id": "m1"},
            {"p_yes": 0.80, "market_prob": 0.50, "outcome": False, "market_id": "m2"},
            {"p_yes": 0.53, "market_prob": 0.50, "outcome": True, "market_id": "m3"},
            {"p_yes": 0.65, "market_prob": 0.50, "outcome": True, "market_id": "m4"},
        ]
        models = ["gpt-4.1-2025-04-14", "gpt-4o-2024-08-06"]
        split_rows = [
            _row(model=models[i % 2], **kwargs) for i, kwargs in enumerate(base)
        ]
        merged_rows = [_row(model="gpt-4.1-2025-04-14", **kwargs) for kwargs in base]

        split = simulate(split_rows, WINDOW_START, WINDOW_END)
        merged = simulate(merged_rows, WINDOW_START, WINDOW_END)
        assert len(merged) == 1
        assert len(split) == 2
        assert {g["model"] for g in split} == set(models)
        assert {(g["platform"], g["tool_name"], g["mode"]) for g in split} == {
            ("omen", "test-tool", "production")
        }
        for field in ("n_rows_seen", "n_eligible", "n_bets"):
            assert sum(g[field] for g in split) == merged[0][field]
        assert sum(g["staked"] for g in split) == pytest.approx(merged[0]["staked"])

        def pnl(group: dict[str, Any]) -> float:
            """Recover a group's total PnL from its pooled ROI.

            :param group: simulate() group entry with staked > 0.
            :return: total PnL in USDC.
            """
            return group["staked"] * group["roi_mid"] / 100.0

        split_pnl = sum(pnl(g) for g in split if g["staked"] > 0)
        assert split_pnl == pytest.approx(pnl(merged[0]))

    def test_tournament_override_beats_runner_stamp(self) -> None:
        """Hardcoding tools get their model corrected on tournament rows.

        predict-fine-tuned serves a fixed vLLM checkpoint (MODEL_BY_TOOL)
        and claude-* tools hardcode claude-sonnet-4-6; both ignore the CI
        runner's --model stamp, so the stamp must never reach the report.
        A kwarg-honoring tournament tool keeps its stamp.
        """
        rows = [
            _row(
                tool="predict-fine-tuned",
                mode="tournament",
                model="gpt-4.1-2025-04-14",
            ),
            _row(
                tool="claude-prediction-online-v1",
                mode="tournament",
                model="gpt-4.1-2025-04-14",
            ),
            _row(
                tool="superforcaster",
                mode="tournament",
                model="gpt-4o-2024-08-06",
            ),
        ]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        by_tool = {g["tool_name"]: g["model"] for g in groups}
        assert by_tool["predict-fine-tuned"] == "qwen-14b-fine-tuned"
        assert by_tool["claude-prediction-online-v1"] == CLAUDE_HARDCODED_MODEL
        assert by_tool["superforcaster"] == "gpt-4o-2024-08-06"
        # Every override entry resolves regardless of the stamp.
        for tool, served in TOURNAMENT_MODEL_OVERRIDES.items():
            row = _row(tool=tool, mode="tournament", model="stamped-anything")
            assert _resolve_model(row, "tournament") == served

    def test_production_model_passthrough_and_unknown_fallback(self) -> None:
        """Production rows trust the payload-derived model; else unknown.

        The tournament overrides never apply to production rows (their
        model is payload-derived, not a runner stamp); missing / None /
        empty / non-string model values all resolve to MODEL_UNKNOWN.
        """
        passthrough = _row(model="gpt-4.1-2025-04-14")
        assert _resolve_model(passthrough, "production") == "gpt-4.1-2025-04-14"
        # A claude-named tool's PRODUCTION row is trusted, not overridden.
        claude_prod = _row(tool="claude-prediction-online-v1", model="other-model")
        assert _resolve_model(claude_prod, "production") == "other-model"
        for bad_model in (None, "", 42):
            assert _resolve_model(_row(model=bad_model), "production") == MODEL_UNKNOWN
        missing = _row()
        del missing["model"]
        assert _resolve_model(missing, "production") == MODEL_UNKNOWN
        groups = simulate([missing], WINDOW_START, WINDOW_END)
        assert groups[0]["model"] == MODEL_UNKNOWN

    def test_report_shortens_display_names_json_keeps_full(self) -> None:
        """The md table shows short display names; JSON keeps full names."""
        assert MODEL_DISPLAY["gpt-4.1-2025-04-14"] == "gpt-4.1"
        assert MODEL_DISPLAY["gpt-4o-2024-08-06"] == "gpt-4o"
        rows = [
            _row(tool="tool-a", model="gpt-4.1-2025-04-14"),
            _row(tool="tool-b", model="qwen-14b-fine-tuned"),
        ]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        by_tool = {g["tool_name"]: g["model"] for g in groups}
        assert by_tool["tool-a"] == "gpt-4.1-2025-04-14"  # JSON: full name
        report = render_report(
            "omen", groups, date(2026, 7, 8), 90, WINDOW_START, WINDOW_END
        )
        assert "| tool-a | production | gpt-4.1 |" in report
        assert "gpt-4.1-2025-04-14" not in report
        # Names outside the display map render verbatim.
        assert "| tool-b | production | qwen-14b-fine-tuned |" in report

    def test_ordering_model_tie_break_deterministic(self) -> None:
        """Groups tied on every other key sort by model, in JSON and md."""
        rows = [  # inserted in REVERSE of the expected order
            _row(tool="tool-a", model="b-model", p_yes=0.7, market_prob=0.5),
            _row(tool="tool-a", model="a-model", p_yes=0.7, market_prob=0.5),
        ]
        groups = simulate(rows, WINDOW_START, WINDOW_END)
        assert [g["model"] for g in groups] == ["a-model", "b-model"]
        # Precondition: the report's -n_bets sort key genuinely ties.
        assert groups[0]["n_bets"] == groups[1]["n_bets"] == 1
        report = render_report(
            "omen", groups, date(2026, 7, 8), 90, WINDOW_START, WINDOW_END
        )
        assert report.index("| tool-a | production | a-model |") < report.index(
            "| tool-a | production | b-model |"
        )
