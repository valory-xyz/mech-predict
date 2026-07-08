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
"""Simulate trader ROI from stored benchmark predictions (accuracy-benchmark companion).

Replays every tool's stored predictions through the production trader's own
decision rules -- per-platform gates plus a Kelly-proxy stake -- at the market
price captured when the prediction was made, and settles each bet with the
known market resolution. This is a port of a certified simulation contract
(METHOD_SPEC v1.4, cross-validated against live trader ledgers via an
independent re-implementation triangulated to exact agreement): favored-side
betting only, decimal-rounded gate comparisons, pooled capital-weighted ROI,
and a market-clustered bootstrap CI with fixed seeds.

Inputs are the benchmark's own accumulated artifacts (daily production log
shards plus the scored tournament predictions) -- no new data capture, no
LLM calls, no network access. Stdlib only by design so light CI jobs can run
it without the full dependency stack.

Tool policy (data-driven, mirroring the accuracy benchmark's no-allowlist +
reliability-flag approach): every (platform, tool, mode) group present in
the data is simulated and serialized to roi_results.json -- no tool
allowlist, nothing silently dropped. A tool is classified a prediction tool
(``is_prediction_tool``) when at least one of its loaded rows ANYWHERE (all
rows, not window-limited) carries a valid-parse prediction. The markdown
tables show ALL prediction-tool groups: zero-eligible ones stay visible with
a "no eligible rows in window" flag, and groups whose in-window parse
reliability (``parse_reliability`` = valid-parse rows / (valid-parse rows +
invalid_parse rejects), counted at the parse rung) falls below
RELIABILITY_GATE (0.80, the accuracy benchmark's reliability-gate threshold)
are flagged as a possible response-format gap. Non-prediction groups
(question generators, service mechs -- no parseable prediction in any row)
are omitted from the table and summarized on one compact line below it; a
known prediction tool appearing on that line indicates a parser/format gap.

Determinism: the outputs are a pure function of the input artifacts and the
--as-of date. Fixed bootstrap seeds and no wall-clock dependence beyond the
window cutoff mean that re-running with the same artifacts and the same
--as-of produces byte-identical roi_results.json and report files.

Usage:
    python -m benchmark.roi_sim
    python -m benchmark.roi_sim --window-days 90 --as-of 2026-07-08
    python -m benchmark.roi_sim --logs-dir benchmark/datasets/logs \
        --tournament-input benchmark/results/tournament_scored.jsonl \
        --results-dir benchmark/results
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeGuard

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOGS_DIR = Path(__file__).parent / "datasets" / "logs"
DEFAULT_TOURNAMENT_INPUT = Path(__file__).parent / "results" / "tournament_scored.jsonl"
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"

# Mode strings as they appear in the input rows (mirrors benchmark/scorer.py).
PRODUCTION_MODE = "production_replay"
TOURNAMENT_MODE = "tournament"

# Trailing window on predicted_at. 90 days is the default remedy for thin
# segments (bets, not predictions, are the binding count once gates cut).
DEFAULT_WINDOW_DAYS = 90

# All gate comparisons happen on values rounded to this many decimals.
# Raw IEEE floats admit/reject boundary rows (e.g. 0.55 - 0.54 > 0.01 in
# binary floats); the certified contract pins 9-decimal rounding BEFORE
# every gate comparison, including the oracle-prob floor.
EDGE_DECIMALS = 9

# Buy prices are clamped into this range so 1/price payouts stay finite.
PRICE_MIN = 0.01
PRICE_MAX = 0.99

# Kelly-proxy stake: stake = min(MAX_BET, f * BANKROLL_NOMINAL) with
# f = clamp(edge / (1 - price), 0, 1) * KELLY_FRACTION. "Enough"-wealth mode:
# capital is never binding, matching the certified contract.
MAX_BET = 2.5
BANKROLL_NOMINAL = 100.0
KELLY_FRACTION = 1.0

# Market-clustered bootstrap: B replicates, fresh random.Random(BOOT_SEED)
# per CI computation so results never depend on global random state.
BOOT_B = 2000
BOOT_SEED = 12345

# Flag thresholds. MIN_SAMPLE_SIZE reuses the accuracy benchmark's precedent
# (scorer.MIN_SAMPLE_SIZE = 30); FEW_BETS_THRESHOLD flags rows whose bet
# count is too small for anything beyond anecdote.
MIN_SAMPLE_SIZE = 30
FEW_BETS_THRESHOLD = 10
FLAG_FEW_BETS = "few bets - anecdotal"
FLAG_LOW_SAMPLE = "low sample"

# Prediction-tool rows with zero eligible in-window rows stay in the table
# (all prediction tools visible, never dropped) under this flag.
FLAG_NO_ELIGIBLE = "no eligible rows in window"

# Parse-reliability flag threshold; mirrors the accuracy benchmark's
# reliability gate (scorer.RELIABILITY_GATE = 0.80). Strict <: a group at
# exactly 0.80 is not flagged.
RELIABILITY_GATE = 0.80

# Eligibility reject reasons, in ladder order (first failure wins). A null
# final_outcome is NOT a reject: it classifies as PENDING (unresolved) and is
# counted separately per group.
REJECT_REASONS = (
    "out_of_window",
    "invalid_parse",
    "bad_p_yes",
    "bad_market_prob",
    "bad_final_outcome",
)

PENDING = "pending"

# First-failing-gate no-bet reasons, in gate order (certified naming).
NO_BET_GATES = (
    "skip_oracle_prob",
    "skip_edge_below_floor",
    "skip_edge_above_cap",
    "skip_spread",
    "skip_zero_stake",
)

PLATFORM_LABELS = {"omen": "Omen", "polymarket": "Polymarket"}


@dataclass(frozen=True)
class GateConfig:
    """Frozen per-platform trader gate + cost configuration (v1.4 canonical)."""

    # Favored-side floor on round(p_side, 9); 0.50 on both platforms is the
    # kelly strategy DEFAULT (the 0.10 Polymarket floor was a fleet-only
    # service override, not representative of a default-config trader).
    min_oracle_prob: float
    # Edge floor, STRICT >: an edge exactly equal to min_edge does NOT bet.
    min_edge: float
    # Edge cap, STRICT <: caps overconfident bets. Live Omen never had a cap
    # (1.0 = no cap); the 0.30 Polymarket cap is the current production value.
    max_edge: float
    # Spread cap; the gate is SKIPPED entirely when the row carries no usable
    # spread (tournament rows never do).
    spread_max: float
    # Flat additive cost on the buy price for the "with costs" variant:
    # Omen +0.02 (AMM-fee proxy), Polymarket +0.08 (half-spread proxy).
    haircut: float


# One frozen trader config per platform -- the strategy defaults the live
# traders run, identical for every tool.
PLATFORM_GATES: dict[str, GateConfig] = {
    "polymarket": GateConfig(
        min_oracle_prob=0.50,
        min_edge=0.01,
        max_edge=0.30,
        spread_max=0.10,
        haircut=0.08,
    ),
    "omen": GateConfig(
        min_oracle_prob=0.50,
        min_edge=0.03,
        max_edge=1.0,
        spread_max=0.10,
        haircut=0.02,
    ),
}


@dataclass(frozen=True)
class Bet:
    """One simulated bet: the mid-price variant plus its haircut companion.

    Both variants share ONE bet set selected by the mid-price gates only;
    the haircut affects price/PnL, never selection. A haircut stake that
    collapses below the edge floor is recorded as stake_haircut = 0 /
    pnl_haircut = 0 but the row stays in the bet set.
    """

    market_id: Any
    side_yes: bool
    price: float
    stake: float
    pnl: float
    win: bool
    stake_haircut: float
    pnl_haircut: float


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* into [low, high].

    :param value: input value.
    :param low: lower bound.
    :param high: upper bound.
    :return: clamped value.
    """
    return max(low, min(high, value))


def _is_number(value: object) -> TypeGuard[float]:
    """Return True when *value* is a real (non-bool, non-NaN) number.

    JSON true/false decodes to Python bool, which is an int subclass; the
    certified ladder rejects bools wherever a probability is expected. The
    TypeGuard narrows the value to float for the caller's arithmetic.

    :param value: candidate value.
    :return: True for usable numeric values.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not math.isnan(value)


def _parse_predicted_at(value: Any) -> datetime | None:
    """Parse a predicted_at timestamp (ISO 8601; Z -> +00:00; naive -> UTC).

    :param value: raw timestamp value from a row.
    :return: timezone-aware datetime, or None when unparseable.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def window_bounds(as_of: date, window_days: int) -> tuple[datetime, datetime]:
    """Compute the trailing simulation window for an --as-of date.

    The window covers exactly *window_days* days of predicted_at values and
    INCLUDES the whole as-of day: window_end is midnight UTC of as-of + 1
    (exclusive), window_start = window_end - window_days.

    :param as_of: last (fully included) calendar day of the window.
    :param window_days: window length in days.
    :return: (window_start, window_end) as aware UTC datetimes; rows are in
        the window when window_start <= predicted_at < window_end.
    """
    window_end = datetime(
        as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc
    ) + timedelta(days=1)
    return window_end - timedelta(days=window_days), window_end


# ---------------------------------------------------------------------------
# Input loading (dedup on row_id, first-seen wins)
# ---------------------------------------------------------------------------


def load_input_rows(logs_dir: Path, tournament_input: Path) -> list[dict[str, Any]]:
    """Load and dedup all input rows from the benchmark artifacts.

    Production shards (production_log_*.jsonl) are scanned in sorted
    filename order, then the scored tournament file. Dedup is on row_id,
    first-seen wins -- the collector re-emits ~4% of production rows into two
    consecutive daily shards; the copies are usually verbatim but can differ
    (collector re-emission), so first-seen-wins in sorted filename order is
    the deterministic rule. Rows without a string row_id are kept as-is
    (never deduped).

    :param logs_dir: directory holding production_log_*.jsonl shards.
    :param tournament_input: path to tournament_scored.jsonl (resolved rows
        only; tournament_predictions.jsonl is deliberately NOT read).
    :return: deduped rows in scan order.
    """
    paths: list[Path] = []
    if logs_dir.is_dir():
        paths.extend(sorted(logs_dir.glob("production_log_*.jsonl")))
    else:
        log.warning("Logs dir %s does not exist; no production rows", logs_dir)
    if tournament_input.is_file():
        paths.append(tournament_input)
    else:
        log.warning(
            "Tournament input %s does not exist; no tournament rows",
            tournament_input,
        )

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    n_duplicates = 0
    n_bad_lines = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    n_bad_lines += 1
                    continue
                if not isinstance(row, dict):
                    n_bad_lines += 1
                    continue
                row_id = row.get("row_id")
                if isinstance(row_id, str):
                    if row_id in seen_ids:
                        n_duplicates += 1
                        continue
                    seen_ids.add(row_id)
                rows.append(row)
    log.info(
        "Loaded %d rows from %d files (%d duplicate row_ids dropped, "
        "%d unparseable lines skipped)",
        len(rows),
        len(paths),
        n_duplicates,
        n_bad_lines,
    )
    return rows


# ---------------------------------------------------------------------------
# Eligibility ladder
# ---------------------------------------------------------------------------


def eligibility_reason(
    row: dict[str, Any], window_start: datetime, window_end: datetime
) -> str | None:
    """Apply the eligibility ladder to one row; first failure wins.

    Ladder (in order): predicted_at parses and lies in the window;
    prediction_parse_status is "valid"; p_yes is a non-bool number in the
    CLOSED interval [0, 1]; market_prob_at_prediction is a non-bool number
    in the OPEN interval (0, 1); final_outcome is a strict JSON bool. A null
    final_outcome is PENDING (unresolved, counted separately); any other
    non-bool (strings, 0/1) is bad_final_outcome -- only resolved rows
    simulate either way.

    :param row: input row.
    :param window_start: inclusive window start (aware UTC).
    :param window_end: exclusive window end (aware UTC).
    :return: reject reason from REJECT_REASONS, PENDING for a null
        final_outcome, or None when eligible.
    """
    predicted_at = _parse_predicted_at(row.get("predicted_at"))
    if predicted_at is None or not window_start <= predicted_at < window_end:
        return "out_of_window"
    if row.get("prediction_parse_status") != "valid":
        return "invalid_parse"
    p_yes = row.get("p_yes")
    if not _is_number(p_yes) or not 0.0 <= p_yes <= 1.0:
        return "bad_p_yes"
    market_prob = row.get("market_prob_at_prediction")
    if not _is_number(market_prob) or not 0.0 < market_prob < 1.0:
        return "bad_market_prob"
    final_outcome = row.get("final_outcome")
    if final_outcome is None:
        return PENDING
    if not isinstance(final_outcome, bool):
        return "bad_final_outcome"
    return None


# ---------------------------------------------------------------------------
# Single-row simulation (gates + stake + settlement)
# ---------------------------------------------------------------------------


def simulate_row(row: dict[str, Any], gates: GateConfig) -> Bet | str:
    """Run one ELIGIBLE row through the trader's gates and settle it.

    Side selection: bet the side the tool favors relative to the captured
    price (side_yes = p_yes >= market_prob). Gates are evaluated in order on
    the mid price only; the FIRST failing gate names the no-bet (one of
    NO_BET_GATES). All gate comparisons use round(x, EDGE_DECIMALS) values.
    The haircut variant is settled over the SAME bet set at price + haircut;
    when the haircut edge no longer clears the edge floor the haircut
    stake/PnL collapse to zero but the row stays a bet.

    :param row: eligible row (see :func:`eligibility_reason`).
    :param gates: platform gate configuration.
    :return: a :class:`Bet`, or the first-failing gate name from
        NO_BET_GATES when a gate rejects the row.
    """
    p_yes = float(row["p_yes"])
    market_prob = float(row["market_prob_at_prediction"])
    outcome = row["final_outcome"]

    side_yes = p_yes >= market_prob
    p_side = p_yes if side_yes else 1.0 - p_yes
    m_side = market_prob if side_yes else 1.0 - market_prob
    price = _clamp(m_side, PRICE_MIN, PRICE_MAX)
    edge = round(p_side - price, EDGE_DECIMALS)

    # Gate 1: oracle-prob floor (favored-side rule; >= on the rounded value).
    if round(p_side, EDGE_DECIMALS) < gates.min_oracle_prob:
        return "skip_oracle_prob"
    # Gate 2: edge floor, STRICT >.
    if not edge > gates.min_edge:
        return "skip_edge_below_floor"
    # Gate 3: edge cap, STRICT <.
    if not edge < gates.max_edge:
        return "skip_edge_above_cap"
    # Gate 4: spread cap -- only when the row carries a usable spread value;
    # missing/None/NaN/bool spread skips the gate entirely.
    spread = row.get("market_spread_at_prediction")
    if _is_number(spread) and spread > gates.spread_max:
        return "skip_spread"
    # Gate 5: zero-stake (Kelly-proxy sizing).
    fraction = _clamp(edge / (1.0 - price), 0.0, 1.0) * KELLY_FRACTION
    stake = min(MAX_BET, fraction * BANKROLL_NOMINAL)
    if stake <= 0.0:
        return "skip_zero_stake"

    win = (outcome is True) if side_yes else (outcome is False)
    pnl = stake * (1.0 / price - 1.0) if win else -stake
    # Runtime invariant from the certified contract: payout > 0 iff win.
    assert ((stake + pnl) > 0) == win

    price_haircut = _clamp(m_side + gates.haircut, PRICE_MIN, PRICE_MAX)
    edge_haircut = round(p_side - price_haircut, EDGE_DECIMALS)
    if edge_haircut <= gates.min_edge:
        # Haircut price no longer clears the floor: zero stake/PnL, row stays.
        stake_haircut = 0.0
        pnl_haircut = 0.0
    else:
        fraction_haircut = (
            _clamp(edge_haircut / (1.0 - price_haircut), 0.0, 1.0) * KELLY_FRACTION
        )
        stake_haircut = min(MAX_BET, fraction_haircut * BANKROLL_NOMINAL)
        pnl_haircut = (
            stake_haircut * (1.0 / price_haircut - 1.0) if win else -stake_haircut
        )
        if stake_haircut > 0.0:
            assert ((stake_haircut + pnl_haircut) > 0) == win

    return Bet(
        market_id=row.get("market_id"),
        side_yes=side_yes,
        price=price,
        stake=stake,
        pnl=pnl,
        win=win,
        stake_haircut=stake_haircut,
        pnl_haircut=pnl_haircut,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI (market-clustered, fixed seed)
# ---------------------------------------------------------------------------


def cluster_bootstrap_ci(
    bet_rows: list[tuple[Any, float, float]],
) -> tuple[float, float] | None:
    """95% CI for pooled ROI via a market-clustered bootstrap.

    Resamples MARKET clusters, not bets: (stake, pnl) are pre-aggregated per
    market_id over bet rows with stake > 0, cluster order pinned to first
    occurrence in the bet list. B = BOOT_B replicates drawn with a fresh
    ``random.Random(BOOT_SEED)`` (results never depend on global random
    state); each replicate draws n clusters via ``randrange(n)`` and its
    statistic is 100 * sum(pnl) / sum(stake); zero-stake replicates are
    skipped. Bounds use the pinned index convention
    ``sorted_samples[int(q * N)]`` (not interpolated percentiles).

    :param bet_rows: (market_id, stake, pnl) triples for one variant.
    :return: (low, high), or None when clusters < 2 or fewer than B/2
        replicates survive.
    """
    clusters: dict[Any, list[float]] = {}
    for market_id, stake, pnl in bet_rows:
        if stake <= 0.0:
            continue
        aggregate = clusters.setdefault(market_id, [0.0, 0.0])
        aggregate[0] += stake
        aggregate[1] += pnl
    if len(clusters) < 2:
        return None

    sums = list(clusters.values())  # insertion order = first occurrence
    n_clusters = len(sums)
    rng = random.Random(
        BOOT_SEED
    )  # nosec B311 — deterministic statistical bootstrap, not cryptographic
    samples: list[float] = []
    for _ in range(BOOT_B):
        stake_total = 0.0
        pnl_total = 0.0
        for _ in range(n_clusters):
            stake_part, pnl_part = sums[rng.randrange(n_clusters)]
            stake_total += stake_part
            pnl_total += pnl_part
        if stake_total <= 0.0:
            continue
        samples.append(100.0 * pnl_total / stake_total)
    if len(samples) < BOOT_B / 2:
        return None
    samples.sort()
    n_kept = len(samples)
    return samples[int(0.025 * n_kept)], samples[int(0.975 * n_kept)]


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------


def _mean_brier(rows: list[dict[str, Any]]) -> float | None:
    """Mean Brier score of eligible rows: (p_yes - outcome)^2.

    :param rows: eligible rows (p_yes and bool final_outcome guaranteed).
    :return: mean Brier score, or None for an empty list.
    """
    if not rows:
        return None
    total = 0.0
    for row in rows:
        outcome = 1.0 if row["final_outcome"] else 0.0
        total += (float(row["p_yes"]) - outcome) ** 2
    return total / len(rows)


def _top3_pnl_share(bets: list[Bet]) -> float | None:
    """Share of total |per-market PnL| carried by the 3 largest markets.

    Absolute values on the mid variant -- a big LOSS market counts. High
    values mean the headline ROI is concentration-driven.

    :param bets: mid-variant bet set.
    :return: share in [0, 1], or None when total |PnL| is zero.
    """
    per_market: dict[Any, float] = {}
    for bet in bets:
        per_market[bet.market_id] = per_market.get(bet.market_id, 0.0) + bet.pnl
    magnitudes = sorted((abs(v) for v in per_market.values()), reverse=True)
    total = sum(magnitudes)
    if total <= 0.0:
        return None
    return sum(magnitudes[:3]) / total


def compute_group_stats(
    eligible_rows: list[dict[str, Any]], gates: GateConfig
) -> dict[str, Any]:
    """Simulate one (platform, tool, mode) group and compute its stats.

    ROI is POOLED and capital-weighted: 100 * sum(pnl) / sum(stake) -- never
    a mean of per-bet ROIs. Each price variant gets its own market-clustered
    CI (fresh fixed-seed RNG per CI call, so the order of the two calls can
    never change either result). No-bets are attributed to their
    first-failing gate.

    :param eligible_rows: rows that passed the eligibility ladder.
    :param gates: platform gate configuration.
    :return: stats dict (full float precision; rounding happens at render /
        serialization time).
    """
    bets: list[Bet] = []
    bet_rows: list[dict[str, Any]] = []
    no_bet = {gate: 0 for gate in NO_BET_GATES}
    for row in eligible_rows:
        result = simulate_row(row, gates)
        if isinstance(result, Bet):
            bets.append(result)
            bet_rows.append(row)
        else:
            no_bet[result] += 1

    n_bets = len(bets)
    staked = sum(b.stake for b in bets)
    pnl_total = sum(b.pnl for b in bets)
    staked_haircut = sum(b.stake_haircut for b in bets)
    pnl_haircut_total = sum(b.pnl_haircut for b in bets)
    ci = cluster_bootstrap_ci([(b.market_id, b.stake, b.pnl) for b in bets])
    ci_haircut = cluster_bootstrap_ci(
        [(b.market_id, b.stake_haircut, b.pnl_haircut) for b in bets]
    )
    return {
        "n_eligible": len(eligible_rows),
        "n_bets": n_bets,
        "no_bet": no_bet,
        "coverage_pct": (
            100.0 * n_bets / len(eligible_rows) if eligible_rows else None
        ),
        "staked": staked,
        "roi_mid": 100.0 * pnl_total / staked if staked > 0.0 else None,
        "roi_ci": list(ci) if ci is not None else None,
        "roi_haircut": (
            100.0 * pnl_haircut_total / staked_haircut if staked_haircut > 0.0 else None
        ),
        "roi_haircut_ci": list(ci_haircut) if ci_haircut is not None else None,
        "brier_all": _mean_brier(eligible_rows),
        "brier_bets": _mean_brier(bet_rows),
        "win_rate": (sum(1 for b in bets if b.win) / n_bets) if n_bets else None,
        "top3_pnl_share": _top3_pnl_share(bets),
    }


def _mode_label(row: dict[str, Any]) -> str:
    """Map a row's mode field to the report label.

    Missing mode defaults to production (matches scorer's historical
    default); unknown modes pass through verbatim so they stay visible.

    :param row: input row.
    :return: "production", "tournament", or the raw mode string.
    """
    mode = row.get("mode") or PRODUCTION_MODE
    if mode == PRODUCTION_MODE:
        return "production"
    if mode == TOURNAMENT_MODE:
        return "tournament"
    return str(mode)


def _parse_reliability_flag(parse_reliability: float) -> str:
    """Build the low-parse-reliability flag text.

    :param parse_reliability: in-window parse reliability in [0, 1].
    :return: flag string with the percentage shown.
    """
    return (
        f"⚠ {parse_reliability:.0%} parse reliability — " "possible response-format gap"
    )


def simulate(
    rows: list[dict[str, Any]], window_start: datetime, window_end: datetime
) -> list[dict[str, Any]]:
    """Group rows by (platform, tool, mode) and simulate every group.

    Every group present in the data is simulated -- no tool allowlist. Rows
    on platforms without a gate config are ignored (and counted in the log).
    n_rows_seen is WINDOW-SCOPED (certified contract): it counts the group's
    deduped rows whose predicted_at parses and falls in the window, before
    the parse/validity rungs; pending (unresolved) rows are counted in
    n_pending, not in rejects.

    Two tool-policy fields per group (see module docstring): parse_reliability
    = n_valid_parse / (n_valid_parse + invalid_parse rejects) over the
    group's IN-WINDOW rows, counted at the parse rung (None when the
    denominator is 0), and is_prediction_tool = whether the TOOL has at
    least one valid-parse row anywhere in the loaded data (all rows, not
    window-limited).

    :param rows: deduped input rows.
    :param window_start: inclusive window start (aware UTC).
    :param window_end: exclusive window end (aware UTC).
    :return: per-group stats dicts, deterministically sorted.
    """
    # A tool that ever produced a parseable prediction is a prediction tool.
    prediction_tools = {
        str(row.get("tool_name") or "unknown")
        for row in rows
        if row.get("prediction_parse_status") == "valid"
    }
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    n_skipped_platform = 0
    for row in rows:
        platform = row.get("platform")
        if platform not in PLATFORM_GATES:
            n_skipped_platform += 1
            continue
        tool_name = str(row.get("tool_name") or "unknown")
        mode = _mode_label(row)
        key = (platform, tool_name, mode)
        group = grouped.setdefault(
            key,
            {
                "n_rows_seen": 0,
                "n_pending": 0,
                "rejects": {reason: 0 for reason in REJECT_REASONS},
                "eligible": [],
            },
        )
        reason = eligibility_reason(row, window_start, window_end)
        if reason != "out_of_window":
            group["n_rows_seen"] += 1
        if reason is None:
            group["eligible"].append(row)
        elif reason == PENDING:
            group["n_pending"] += 1
        else:
            group["rejects"][reason] += 1
    if n_skipped_platform:
        log.warning(
            "Ignored %d rows on platforms without a gate config",
            n_skipped_platform,
        )

    results: list[dict[str, Any]] = []
    for (platform, tool_name, mode), group in grouped.items():
        stats = compute_group_stats(group["eligible"], PLATFORM_GATES[platform])
        # Parse reliability over IN-WINDOW rows, counted at the parse rung:
        # every in-window row either rejects as invalid_parse or passes the
        # parse rung (it may still fail later rungs), so the denominator
        # n_valid_parse + invalid_parse equals n_rows_seen.
        n_invalid_parse = group["rejects"]["invalid_parse"]
        n_valid_parse = group["n_rows_seen"] - n_invalid_parse
        parse_denominator = n_valid_parse + n_invalid_parse
        parse_reliability = (
            n_valid_parse / parse_denominator if parse_denominator else None
        )
        flags: list[str] = []
        if stats["n_eligible"] == 0:
            flags.append(FLAG_NO_ELIGIBLE)
        else:
            if stats["n_bets"] < FEW_BETS_THRESHOLD:
                flags.append(FLAG_FEW_BETS)
            if stats["n_eligible"] < MIN_SAMPLE_SIZE:
                flags.append(FLAG_LOW_SAMPLE)
        if parse_reliability is not None and parse_reliability < RELIABILITY_GATE:
            flags.append(_parse_reliability_flag(parse_reliability))
        entry: dict[str, Any] = {
            "platform": platform,
            "tool_name": tool_name,
            "mode": mode,
            "n_rows_seen": group["n_rows_seen"],
            "n_pending": group["n_pending"],
            "rejects": group["rejects"],
            "parse_reliability": parse_reliability,
            "is_prediction_tool": tool_name in prediction_tools,
            "flags": flags,
        }
        entry.update(stats)
        results.append(entry)
    results.sort(
        key=lambda r: (
            r["platform"],
            0 if r["mode"] == "production" else 1,
            r["mode"],
            r["tool_name"],
        )
    )
    return results


# ---------------------------------------------------------------------------
# Rendering + serialization
# ---------------------------------------------------------------------------


def _round_floats(value: Any) -> Any:
    """Recursively round floats to 6 decimals for stable JSON output.

    :param value: arbitrary JSON-serializable structure.
    :return: same structure with all floats rounded.
    """
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item) for item in value]
    return value


def _fmt_pct(value: float | None) -> str:
    """Format a percentage for the report.

    :param value: percentage value or None.
    :return: display string.
    """
    return "n/a" if value is None else f"{value:+.1f}%"


def _fmt_roi_ci(roi: float | None, ci: list[float] | None) -> str:
    """Format an "ROI (CI)" report cell (mid and haircut variants alike).

    :param roi: pooled ROI in percent, or None.
    :param ci: [low, high] CI bounds, or None.
    :return: display string.
    """
    if roi is None:
        return "n/a"
    if ci is None:
        return f"{_fmt_pct(roi)} (CI n/a)"
    return f"{_fmt_pct(roi)} ({ci[0]:+.1f}, {ci[1]:+.1f})"


def _fmt_brier(brier_all: float | None, brier_bets: float | None) -> str:
    """Format the "Brier all->bets" report cell.

    :param brier_all: Brier over all eligible predictions, or None.
    :param brier_bets: Brier over the gated bet subset, or None.
    :return: display string.
    """
    left = "n/a" if brier_all is None else f"{brier_all:.3f}"
    right = "n/a" if brier_bets is None else f"{brier_bets:.3f}"
    return f"{left} -> {right}"


def _excluded_line(excluded: list[dict[str, Any]]) -> str:
    """Summarize non-prediction groups on one compact line.

    Row counts are per TOOL over ALL loaded rows for that tool's groups on
    the platform (in-window rows plus out_of_window rejects), matching the
    any-row classification; sorted by row count descending, then tool name.

    :param excluded: groups with is_prediction_tool False (one platform).
    :return: single-line summary string.
    """
    counts: dict[str, int] = {}
    for group in excluded:
        n_rows = group["n_rows_seen"] + group["rejects"]["out_of_window"]
        counts[group["tool_name"]] = counts.get(group["tool_name"], 0) + n_rows
    parts = [
        f"{tool} ({n} row{'s' if n != 1 else ''})"
        for tool, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return (
        "Excluded (no parseable prediction in any row): "
        + ", ".join(parts)
        + " — a known prediction tool appearing here indicates a "
        "parser/format gap."
    )


def render_report(
    platform: str,
    groups: list[dict[str, Any]],
    as_of: date,
    window_days: int,
    window_start: datetime,
    window_end: datetime,
) -> str:
    """Render one platform's markdown ROI report.

    Rows are sorted production-first, then by bet count descending; rows
    below sample thresholds are flagged, never dropped. The table shows
    prediction-tool groups only (ALL of them, including zero-eligible ones,
    which carry the "no eligible rows in window" flag); non-prediction
    groups are summarized on one compact line below the table, sorted by
    row count descending.

    :param platform: platform key ("omen" / "polymarket").
    :param groups: all group stats (any platform; filtered here).
    :param as_of: window as-of date.
    :param window_days: window length in days.
    :param window_start: inclusive window start.
    :param window_end: exclusive window end.
    :return: markdown document text.
    """
    label = PLATFORM_LABELS.get(platform, platform)
    lines = [
        f"# Simulated trader ROI - {label} - trailing {window_days} days",
        "",
        (
            f"Window: {window_start.isoformat()} <= `predicted_at` < "
            f"{window_end.isoformat()} (as-of {as_of.isoformat()})."
        ),
        (
            "ROI = total PnL / total staked over bets placed in this window "
            "(capital-weighted, pooled; not annualized)."
        ),
        (
            "Brier all->bets: ALL eligible predictions vs the gated bet "
            "subset. Low-sample rows are flagged, never dropped."
        ),
        "",
    ]
    platform_groups = [g for g in groups if g["platform"] == platform]
    rows = [g for g in platform_groups if g["is_prediction_tool"]]
    excluded = [g for g in platform_groups if not g["is_prediction_tool"]]
    if not rows:
        lines.append("_No data for this platform in the window._")
        if excluded:
            lines.append("")
            lines.append(_excluded_line(excluded))
        return "\n".join(lines) + "\n"

    rows.sort(
        key=lambda g: (
            0 if g["mode"] == "production" else 1,
            -g["n_bets"],
            g["tool_name"],
            g["mode"],
        )
    )
    lines.append(
        "| tool | mode | n preds | n bets | Brier all->bets | staked "
        "| ROI (95% CI) | ROI w/ costs | flags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for g in rows:
        lines.append(
            "| {tool} | {mode} | {n_preds} | {n_bets} | {brier} | {staked} "
            "| {roi} | {roi_h} | {flags} |".format(
                tool=g["tool_name"],
                mode=g["mode"],
                n_preds=g["n_eligible"],
                n_bets=g["n_bets"],
                brier=_fmt_brier(g["brier_all"], g["brier_bets"]),
                staked=f"{g['staked']:.2f} USDC",
                roi=_fmt_roi_ci(g["roi_mid"], g["roi_ci"]),
                roi_h=_fmt_roi_ci(g["roi_haircut"], g["roi_haircut_ci"]),
                flags="; ".join(g["flags"]),
            )
        )
    if excluded:
        lines.append("")
        lines.append(_excluded_line(excluded))
    return "\n".join(lines) + "\n"


def write_outputs(
    groups: list[dict[str, Any]],
    results_dir: Path,
    as_of: date,
    window_days: int,
    window_start: datetime,
    window_end: datetime,
) -> None:
    """Write roi_results.json plus the two per-platform markdown reports.

    Outputs carry no timestamps other than the as_of / window fields, and
    JSON keys are sorted -- same artifacts + same as-of reproduce all three
    files byte-for-byte.

    :param groups: per-group stats from :func:`simulate`.
    :param results_dir: output directory (created if missing).
    :param as_of: window as-of date.
    :param window_days: window length in days.
    :param window_start: inclusive window start.
    :param window_end: exclusive window end.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": as_of.isoformat(),
        "window_days": window_days,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "groups": groups,
    }
    json_path = results_dir / "roi_results.json"
    json_path.write_text(
        json.dumps(_round_floats(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %s", json_path)
    for platform in sorted(PLATFORM_GATES):
        report_path = results_dir / f"report_roi_{platform}.md"
        report_path.write_text(
            render_report(
                platform, groups, as_of, window_days, window_start, window_end
            ),
            encoding="utf-8",
        )
        log.info("Wrote %s", report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: load artifacts, simulate, write reports."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Simulate trader ROI per (platform, tool, mode) from stored "
            "benchmark predictions over a trailing window."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory with production_log_*.jsonl shards. "
        "Default: benchmark/datasets/logs",
    )
    parser.add_argument(
        "--tournament-input",
        type=Path,
        default=DEFAULT_TOURNAMENT_INPUT,
        help="Scored tournament rows (tournament_scored.jsonl). "
        "Default: benchmark/results/tournament_scored.jsonl",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for roi_results.json + reports. "
        "Default: benchmark/results",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"Trailing window length in days. Default: {DEFAULT_WINDOW_DAYS}",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Last (fully included) day of the window, YYYY-MM-DD. "
        "Default: today UTC.",
    )
    args = parser.parse_args()

    as_of = (
        date.fromisoformat(args.as_of)
        if args.as_of
        else datetime.now(timezone.utc).date()
    )
    window_start, window_end = window_bounds(as_of, args.window_days)
    log.info(
        "Simulating trailing %d days: %s <= predicted_at < %s",
        args.window_days,
        window_start.isoformat(),
        window_end.isoformat(),
    )
    rows = load_input_rows(args.logs_dir, args.tournament_input)
    groups = simulate(rows, window_start, window_end)
    log.info(
        "Simulated %d groups (%d bets total)",
        len(groups),
        sum(g["n_bets"] for g in groups),
    )
    write_outputs(
        groups, args.results_dir, as_of, args.window_days, window_start, window_end
    )


if __name__ == "__main__":
    main()
