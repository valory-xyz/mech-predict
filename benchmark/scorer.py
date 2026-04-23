"""
Score production prediction data.

Reads production_log.jsonl and computes:
  - Overall Brier score and reliability
  - Per-tool, per-platform, per-category, per-horizon breakdowns
  - Monthly trend

Usage:
    python benchmark/scorer.py
    python benchmark/scorer.py --input path/to/log.jsonl --output path/to/scores.json
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import logging
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from benchmark.io import load_jsonl as load_rows

DEFAULT_INPUT = Path(__file__).parent / "datasets" / "production_log.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "scores.json"
DEFAULT_OUTPUT_TOURNAMENT = Path(__file__).parent / "results" / "scores_tournament.json"
DEFAULT_HISTORY = Path(__file__).parent / "results" / "scores_history.jsonl"
DEFAULT_DEDUP = Path(__file__).parent / "results" / "scored_row_ids.json"
DEFAULT_LOGS_DIR = Path(__file__).parent / "datasets" / "logs"

PRODUCTION_MODE = "production_replay"
TOURNAMENT_MODE = "tournament"
_KNOWN_MODES = frozenset({PRODUCTION_MODE, TOURNAMENT_MODE})
# Modes we've already logged a warning for, so a bad-data jsonl with 10k
# identical unknown-mode rows doesn't spam the log 10k times.
_WARNED_UNKNOWN_MODES: set[str] = set()


def _derive_tournament_path(scores_path: Path) -> Path:
    """Return the tournament scores path paired with *scores_path*.

    Convention: ``<stem>.json`` -> ``<stem>_tournament.json`` in the same dir.
    Used so a single ``--output`` flag (or default) implies both files.

    :param scores_path: production scores path.
    :return: paired tournament scores path.
    """
    return scores_path.with_name(f"{scores_path.stem}_tournament{scores_path.suffix}")


def _partition_rows_by_mode(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into (production, tournament) lists.

    Rows are routed by ``row["mode"]``. Missing mode defaults to
    production — matches the historical default in ``_accumulate_row``.

    :param rows: input rows (any mode).
    :return: tuple of (production_rows, tournament_rows).
    """
    prod: list[dict[str, Any]] = []
    tourn: list[dict[str, Any]] = []
    for row in rows:
        mode = row.get("mode") or PRODUCTION_MODE
        if mode not in _KNOWN_MODES and mode not in _WARNED_UNKNOWN_MODES:
            _WARNED_UNKNOWN_MODES.add(mode)
            logging.getLogger(__name__).warning(
                "Unknown mode %r — routing to production. If this is a new"
                " mode, add it to _KNOWN_MODES and route it explicitly.",
                mode,
            )
        if mode == TOURNAMENT_MODE:
            tourn.append(row)
        else:
            prod.append(row)
    return prod, tourn


# Platforms the scorer emits a dedicated scores file for. A file is written
# for every entry even when the partition is empty, so consumers can assume
# the path exists.
PLATFORMS: tuple[str, ...] = ("omen", "polymarket")


def _derive_platform_path(base_path: Path, platform: str) -> Path:
    """Return the per-platform sibling of ``base_path``.

    Convention: ``<stem>.json`` -> ``<stem>_<platform>.json`` in the same
    directory. Composed with ``_derive_tournament_path`` this yields
    ``scores_tournament_omen.json`` etc.

    :param base_path: base scores path (combined output).
    :param platform: one of ``PLATFORMS``.
    :return: sibling path scoped to ``platform``.
    """
    return base_path.with_name(f"{base_path.stem}_{platform}{base_path.suffix}")


def _partition_rows_by_platform(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group rows by ``row['platform']`` for the platforms we report on.

    Rows with unknown/missing platforms stay in the combined output but are
    excluded from per-platform partitions — the daily report only needs
    omen and polymarket.

    :param rows: input rows (any platform).
    :return: ``{platform: [rows]}`` with one entry per ``PLATFORMS`` value.
        Empty lists are returned for platforms with no rows (so callers can
        still emit an empty-but-valid output file).
    """
    buckets: dict[str, list[dict[str, Any]]] = {plat: [] for plat in PLATFORMS}
    for row in rows:
        plat = row.get("platform")
        if plat in buckets:
            buckets[plat].append(row)
    return buckets


LATENCY_RESERVOIR_SIZE = 200
CALIBRATION_PAIRS_RESERVOIR_SIZE = 50_000
_RESERVOIR_RNG = random.Random(42)
WORST_BEST_SIZE = 10

RELIABILITY_GATE = 0.80
MIN_SAMPLE_SIZE = 30
MIN_CALIBRATION_BIN_SIZE = 20

# Diagnostic edge metric thresholds (PROPOSAL.md Stage 4).
# Fixed for now — will version if changed.
DISAGREE_THRESHOLD = 0.03
LARGE_TRADE_THRESHOLD = 0.10

# Keys that must be persisted in scores.json for incremental resume.
# Used in update() and rebuild() when merging accumulators into output.
_ACCUM_KEYS = (
    "brier_sum",
    "correct_count",
    "n_directional",
    "no_signal_count",
    "sharpness_sum",
    "outcome_yes_count",
    "log_loss_sum",
    "edge_sum",
    "edge_n",
    "edge_positive_count",
    # Diagnostic edge metrics (PROPOSAL.md Stage 4)
    "disagree_tool_win_count",
    "disagree_n",
    "brier_sum_no_trade",
    "n_no_trade",
    "brier_sum_small_trade",
    "n_small_trade",
    "brier_sum_large_trade",
    "n_large_trade",
    "bias_sum",
    "n_bias_losses",
)


# ---------------------------------------------------------------------------
# Brier score computation
# ---------------------------------------------------------------------------


def _load_dedup_ids(dedup_path: Path) -> set[str]:
    """Load scored row IDs from the dedup file.

    Persists across month rollovers so replayed historical rows
    are still skipped.

    :param dedup_path: path to ``scored_row_ids.json``.
    :return: set of row IDs.
    """
    if not dedup_path.exists():
        return set()
    try:
        return set(json.loads(dedup_path.read_text()))
    except (json.JSONDecodeError, OSError):
        return set()


def _save_dedup_ids(dedup_path: Path, ids: set[str]) -> None:
    """Save scored row IDs to the dedup file.

    :param dedup_path: path to ``scored_row_ids.json``.
    :param ids: set of row IDs to persist.
    """
    dedup_path.parent.mkdir(parents=True, exist_ok=True)
    dedup_path.write_text(json.dumps(sorted(ids)))


def brier_score(p_yes: float, outcome: bool) -> float:
    """Compute Brier score for a single prediction."""
    return (p_yes - (1.0 if outcome else 0.0)) ** 2


def edge_score(p_yes: float, market_prob: float, outcome: bool) -> float:
    """Compute edge over market for a single prediction.

    Edge = market_brier - tool_brier. Positive means the tool's prediction
    was closer to the outcome than the market's price.

    :param p_yes: tool's predicted probability.
    :param market_prob: market probability at prediction time.
    :param outcome: actual outcome (True = yes).
    :return: edge score (positive = tool beat market).
    """
    outcome_val = 1.0 if outcome else 0.0
    market_brier = (market_prob - outcome_val) ** 2
    tool_brier = (p_yes - outcome_val) ** 2
    return market_brier - tool_brier


_LOG_LOSS_EPSILON = 1e-15


def log_loss_score(p_yes: float, outcome: bool) -> float:
    """Compute log loss for a single prediction.

    :param p_yes: predicted probability of yes.
    :param outcome: actual outcome.
    :return: log loss value (lower is better).
    """
    p = max(_LOG_LOSS_EPSILON, min(1 - _LOG_LOSS_EPSILON, p_yes))
    if outcome:
        return -math.log(p)
    return -math.log(1 - p)


def _is_edge_eligible(row: dict[str, Any]) -> bool:
    """Check if a row has all fields needed for edge-over-market calculation."""
    return (
        row.get("prediction_parse_status") == "valid"
        and row.get("final_outcome") is not None
        and row.get("p_yes") is not None
        and row.get("market_prob_at_prediction") is not None
    )


# ---------------------------------------------------------------------------
# Diagnostic edge metric helpers (PROPOSAL.md Stage 4)
# ---------------------------------------------------------------------------


def classify_disagreement(p_yes: float, market_prob: float, outcome: bool) -> str:
    """Classify whether tool or market was closer to the outcome.

    :param p_yes: tool's predicted probability.
    :param market_prob: market probability at prediction time.
    :param outcome: actual outcome (True = yes).
    :return: ``"tool_win"``, ``"market_win"``, or ``"tie"``.
    """
    outcome_val = 1.0 if outcome else 0.0
    tool_dist = abs(p_yes - outcome_val)
    market_dist = abs(market_prob - outcome_val)
    if tool_dist < market_dist:
        return "tool_win"
    if tool_dist > market_dist:
        return "market_win"
    return "tie"


def disagree_bucket(p_yes: float, market_prob: float) -> str:
    """Bucket a prediction by disagreement magnitude with the market.

    :param p_yes: tool's predicted probability.
    :param market_prob: market probability at prediction time.
    :return: ``"no_trade"``, ``"small_trade"``, or ``"large_trade"``.
    """
    d = round(abs(p_yes - market_prob), 10)
    if d <= DISAGREE_THRESHOLD:
        return "no_trade"
    if d <= LARGE_TRADE_THRESHOLD:
        return "small_trade"
    return "large_trade"


# ---------------------------------------------------------------------------
# Difficulty and liquidity classification
# ---------------------------------------------------------------------------

# Thresholds are initial values from PROPOSAL.md. Adjust after inspecting
# the actual data distribution from the first scorer run.
DIFFICULTY_THRESHOLDS = (0.15, 0.3)
LIQUIDITY_THRESHOLDS = (500.0, 5000.0)


def classify_difficulty(market_prob: float | None) -> str:
    """Classify market difficulty based on distance from 0.5.

    Uses market_prob_at_prediction (not final market price).

    :param market_prob: market probability at prediction time.
    :return: difficulty bucket name.
    """
    if market_prob is None:
        return "unknown"
    distance = round(abs(market_prob - 0.5), 10)
    lo, hi = DIFFICULTY_THRESHOLDS
    if distance < lo:
        return "hard"
    if distance <= hi:
        return "medium"
    return "easy"


def classify_liquidity(liquidity_usd: float | None) -> str:
    """Classify market liquidity into buckets.

    :param liquidity_usd: market liquidity in USD at prediction time.
    :return: liquidity bucket name.
    """
    if liquidity_usd is None:
        return "unknown"
    lo, hi = LIQUIDITY_THRESHOLDS
    if liquidity_usd < lo:
        return "low"
    if liquidity_usd <= hi:
        return "medium"
    return "high"


_DIAGNOSTIC_NONE: dict[str, Any] = {
    "edge": None,
    "edge_n": 0,
    "edge_positive_rate": None,
    "conditional_accuracy_rate": None,
    "disagree_n": 0,
    "brier_no_trade": None,
    "n_no_trade": 0,
    "brier_small_trade": None,
    "n_small_trade": 0,
    "brier_large_trade": None,
    "n_large_trade": 0,
    "directional_bias": None,
    "n_bias_losses": 0,
}


def _compute_edge_diagnostics(
    edge_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute edge-over-market and diagnostic metrics for edge-eligible rows.

    :param edge_rows: valid rows that have market_prob_at_prediction.
    :return: dict with edge, conditional accuracy, disagreement Brier,
        and directional bias metrics.
    """
    if not edge_rows:
        return dict(_DIAGNOSTIC_NONE)

    edges = [
        edge_score(r["p_yes"], r["market_prob_at_prediction"], r["final_outcome"])
        for r in edge_rows
    ]
    edge_avg = round(sum(edges) / len(edges), 4)
    edge_positive = sum(1 for e in edges if e > 0)
    edge_pos_rate = round(edge_positive / len(edges), 4)

    # Diagnostic edge metrics
    tool_wins = 0
    disagree_n = 0
    bucket_brier: dict[str, float] = {
        "no_trade": 0.0,
        "small_trade": 0.0,
        "large_trade": 0.0,
    }
    bucket_n: dict[str, int] = {"no_trade": 0, "small_trade": 0, "large_trade": 0}
    bias_sum = 0.0
    n_bias_losses = 0

    for r in edge_rows:
        p_yes = r["p_yes"]
        market_prob = r["market_prob_at_prediction"]
        outcome = r["final_outcome"]
        brier = brier_score(p_yes, outcome)

        bucket = disagree_bucket(p_yes, market_prob)
        bucket_brier[bucket] += brier
        bucket_n[bucket] += 1

        if bucket != "no_trade":
            result = classify_disagreement(p_yes, market_prob, outcome)
            if result != "tie":
                disagree_n += 1
                if result == "tool_win":
                    tool_wins += 1
                else:
                    bias_sum += p_yes - (1.0 if outcome else 0.0)
                    n_bias_losses += 1

    diag: dict[str, Any] = {
        "edge": edge_avg,
        "edge_n": len(edge_rows),
        "edge_positive_rate": edge_pos_rate,
        "disagree_n": disagree_n,
        "n_bias_losses": n_bias_losses,
    }

    diag["conditional_accuracy_rate"] = (
        round(tool_wins / disagree_n, 4) if disagree_n >= MIN_SAMPLE_SIZE else None
    )

    for bucket in ("no_trade", "small_trade", "large_trade"):
        diag[f"n_{bucket}"] = bucket_n[bucket]
        diag[f"brier_{bucket}"] = (
            round(bucket_brier[bucket] / bucket_n[bucket], 4)
            if bucket_n[bucket] >= MIN_SAMPLE_SIZE
            else None
        )

    diag["directional_bias"] = (
        round(bias_sum / n_bias_losses, 4) if n_bias_losses >= MIN_SAMPLE_SIZE else None
    )

    return diag


def compute_group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Brier score and reliability for a group of rows.

    All rows count toward reliability. Only valid rows with
    final_outcome count toward Brier.

    :param rows: list of prediction row dicts.
    :return: dict with brier, accuracy, sharpness, reliability, n, valid_n,
        decision_worthy.
    """
    total = len(rows)
    _none_stats: dict[str, Any] = {
        "brier": None,
        "directional_accuracy": None,
        "n_directional": 0,
        "no_signal_rate": None,
        "no_signal_count": 0,
        "log_loss": None,
        "sharpness": None,
        "reliability": None,
        "n": total,
        "valid_n": 0,
        "decision_worthy": False,
        "outcome_yes_rate": None,
        "baseline_brier": None,
        "brier_skill_score": None,
        **_DIAGNOSTIC_NONE,
    }
    if total == 0:
        return dict(_none_stats)

    valid = [
        r
        for r in rows
        if r["prediction_parse_status"] == "valid"
        and r["final_outcome"] is not None
        and r["p_yes"] is not None
    ]
    reliability = len(valid) / total
    worthy = len(valid) >= MIN_SAMPLE_SIZE

    if not valid:
        result = dict(_none_stats)
        result["reliability"] = round(reliability, 4)
        return result

    brier_scores = [brier_score(r["p_yes"], r["final_outcome"]) for r in valid]
    avg_brier = sum(brier_scores) / len(brier_scores)

    # Directional accuracy — exclude p_yes == 0.5 (no signal)
    directional = [r for r in valid if r["p_yes"] != 0.5]
    no_signal_count = len(valid) - len(directional)
    no_signal_rate = round(no_signal_count / len(valid), 4)
    if directional:
        correct = sum(
            1 for r in directional if (r["p_yes"] > 0.5) == r["final_outcome"]
        )
        dir_accuracy = round(correct / len(directional), 4)
    else:
        correct = 0
        dir_accuracy = None

    sharpness = sum(abs(r["p_yes"] - 0.5) for r in valid) / len(valid)

    # Log loss
    ll_sum = sum(log_loss_score(r["p_yes"], r["final_outcome"]) for r in valid)
    avg_log_loss = round(ll_sum / len(valid), 4)

    # BSS — matches _derive_group() computation
    yes_rate = sum(1 for r in valid if r["final_outcome"]) / len(valid)
    baseline_brier = round(yes_rate * (1 - yes_rate), 4)
    brier_rounded = round(avg_brier, 4)
    if baseline_brier > 0:
        bss = round(1 - (brier_rounded / baseline_brier), 4)
    else:
        bss = None

    # Edge over market
    edge_rows = [r for r in valid if r.get("market_prob_at_prediction") is not None]
    edge_diag = _compute_edge_diagnostics(edge_rows)

    return {
        "brier": brier_rounded,
        "directional_accuracy": dir_accuracy,
        "n_directional": len(directional),
        "no_signal_count": no_signal_count,
        "no_signal_rate": no_signal_rate,
        "log_loss": avg_log_loss,
        "sharpness": round(sharpness, 4),
        "reliability": round(reliability, 4),
        "n": total,
        "valid_n": len(valid),
        "decision_worthy": worthy,
        "outcome_yes_rate": round(yes_rate, 4),
        "baseline_brier": baseline_brier,
        "brier_skill_score": bss,
        **edge_diag,
    }


# ---------------------------------------------------------------------------
# Horizon classification
# ---------------------------------------------------------------------------


def classify_horizon(lead_time_days: float | None) -> str:
    """Classify prediction lead time into a horizon bucket."""
    if lead_time_days is None:
        return "unknown"
    if lead_time_days < 7:
        return "short_lt_7d"
    if lead_time_days <= 30:
        return "medium_7_30d"
    return "long_gt_30d"


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    """Group rows by a field value."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get(key) or "unknown"].append(row)
    return dict(groups)


def group_by_month(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group rows by month and compute full stats per month."""
    months: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        predicted_at = row.get("predicted_at")
        if predicted_at:
            month = predicted_at[:7]  # "YYYY-MM"
            months[month].append(row)

    trend = []
    for month in sorted(months):
        stats = compute_group_stats(months[month])
        stats["month"] = month
        trend.append(stats)
    return trend


def group_by_horizon(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group rows by prediction lead time horizon."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        horizon = classify_horizon(row.get("prediction_lead_time_days"))
        groups[horizon].append(row)

    return {h: compute_group_stats(group) for h, group in groups.items()}


def _composite_key(row: dict[str, Any], fields: list[str]) -> str:
    """Build a composite grouping key from multiple fields."""
    return " | ".join(str(row.get(f) or "unknown") for f in fields)


def group_by_composite(
    rows: list[dict[str, Any]],
    fields: list[str],
    *,
    horizon: bool = False,
) -> dict[str, Any]:
    """Group rows by a composite key, optionally sub-grouped by horizon.

    When *horizon* is False, returns ``{key: stats}``.
    When *horizon* is True, returns ``{key: {horizon_bucket: stats}}``.

    :param rows: list of prediction row dicts.
    :param fields: field names to form the composite key.
    :param horizon: whether to sub-group by horizon bucket.
    :return: dict mapping composite keys to stats or horizon sub-dicts.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _composite_key(row, fields)
        groups[key].append(row)

    if not horizon:
        return {k: compute_group_stats(g) for k, g in groups.items()}

    result: dict[str, dict[str, Any]] = {}
    for key, group in groups.items():
        result[key] = group_by_horizon(group)
    return result


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

CALIBRATION_BINS = [
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.01),
]


def _bin_label(lo: float, hi: float) -> str:
    """Human-readable label for a calibration bin."""
    hi_display = 1.0 if hi > 1.0 else hi
    return f"{lo:.1f}-{hi_display:.1f}"


def compute_calibration(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bucket valid predictions by p_yes range, compare predicted vs realized.

    Returns a list of bin dicts sorted by probability range:
    ``{"bin": "0.7-0.8", "avg_predicted": 0.74, "realized_rate": 0.81, "n": 42, "gap": -0.07}``

    Positive gap = overconfident (predicted > realized).
    Negative gap = underconfident (predicted < realized).

    :param rows: list of prediction row dicts.
    :return: list of calibration bin dicts.
    """
    valid = [
        r
        for r in rows
        if r.get("prediction_parse_status") == "valid"
        and r.get("final_outcome") is not None
        and r.get("p_yes") is not None
    ]

    bins: dict[str, list[dict[str, Any]]] = {
        _bin_label(lo, hi): [] for lo, hi in CALIBRATION_BINS
    }

    for row in valid:
        p = row["p_yes"]
        for lo, hi in CALIBRATION_BINS:
            if lo <= p < hi:
                bins[_bin_label(lo, hi)].append(row)
                break

    result = []
    for lo, hi in CALIBRATION_BINS:
        label = _bin_label(lo, hi)
        group = bins[label]
        if not group:
            continue
        avg_pred = sum(r["p_yes"] for r in group) / len(group)
        realized = sum(1 for r in group if r["final_outcome"]) / len(group)
        gap = round(avg_pred - realized, 4)
        result.append(
            {
                "bin": label,
                "avg_predicted": round(avg_pred, 4),
                "realized_rate": round(realized, 4),
                "n": len(group),
                "gap": gap,
            }
        )

    return result


def compute_ece(
    bins: list[dict[str, Any]],
    min_bin_n: int = MIN_CALIBRATION_BIN_SIZE,
) -> float | None:
    """Compute Expected Calibration Error from calibration bins.

    Bins with fewer than *min_bin_n* samples are excluded to avoid
    noisy estimates (per PROPOSAL.md: min 20 samples per bin).

    :param bins: list of calibration bin dicts with n, gap.
    :param min_bin_n: minimum samples for a bin to be included.
    :return: ECE value, or None if no qualifying bins.
    """
    populated = [b for b in bins if b.get("n", 0) >= min_bin_n]
    if not populated:
        return None
    total_n = sum(b["n"] for b in populated)
    if total_n == 0:
        return None
    weighted_gap = sum(b["n"] * abs(b["gap"]) for b in populated)
    return round(weighted_gap / total_n, 4)


_CAL_REG_NONE = {"calibration_intercept": None, "calibration_slope": None}

# Minimum valid predictions required for calibration regression.
MIN_CAL_REG_ROWS = 30


def compute_calibration_regression(
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Compute calibration intercept and slope via Platt scaling on the logit scale.

    Fits ``logit(P(y=1|p)) = intercept + slope * logit(p_yes)`` on
    individual predictions using ``scipy.optimize.minimize`` (Nelder-Mead).

    Both parameters live on the logit scale:

    - **slope = 1.0, intercept = 0.0**: perfectly calibrated
    - **slope < 1.0**: overconfident (predictions too extreme)
    - **slope > 1.0**: underconfident (predictions too compressed)
    - **intercept != 0**: systematic bias on the logit scale

    :param rows: list of prediction row dicts (must have p_yes, final_outcome).
    :return: dict with ``calibration_intercept`` and ``calibration_slope``
        (None if fewer than ``MIN_CAL_REG_ROWS`` valid rows, uniform
        predictions, or optimization failure).
    """

    valid = [
        r
        for r in rows
        if r.get("prediction_parse_status") == "valid"
        and r.get("final_outcome") is not None
        and r.get("p_yes") is not None
    ]
    if len(valid) < MIN_CAL_REG_ROWS:
        return dict(_CAL_REG_NONE)

    ps_list = [r["p_yes"] for r in valid]
    ys_list = [1.0 if r["final_outcome"] else 0.0 for r in valid]

    # Uniform predictions → slope is unidentifiable
    if len(set(ps_list)) < 2:
        return dict(_CAL_REG_NONE)

    # Vectorized computation with numpy
    eps = 1e-15
    ps_arr = np.asarray(ps_list)
    ys_arr = np.asarray(ys_list)
    logit_p = np.log(np.clip(ps_arr, eps, 1 - eps) / np.clip(1 - ps_arr, eps, 1 - eps))

    def _neg_log_likelihood(params: list[float]) -> float:
        intercept, slope = params
        z = intercept + slope * logit_p
        # Numerically stable: -sum(y*z - log(1+exp(z)))
        return float(np.sum(np.logaddexp(0.0, z) - ys_arr * z))

    try:
        result = minimize(  # type: ignore[call-overload]
            _neg_log_likelihood,
            x0=[0.0, 1.0],  # start at perfect calibration
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-8},
        )
        if not result.success:
            return dict(_CAL_REG_NONE)
        intercept, slope = result.x
        return {
            "calibration_intercept": round(float(intercept), 4),
            "calibration_slope": round(float(slope), 4),
        }
    except (ValueError, RuntimeError):
        return dict(_CAL_REG_NONE)


def _compute_calibration_regression_from_bins(
    bins: list[dict[str, Any]],
    min_bin_n: int = MIN_CALIBRATION_BIN_SIZE,
) -> dict[str, float | None]:
    """Legacy migration fallback: calibration regression from bin averages.

    Only runs for scores.json files written before ``_calibration_pairs``
    was introduced. Once ``rebuild()`` is run, pairs are populated and
    the row-level logistic path in ``_finalize_scores`` takes over.

    :param bins: list of calibration bin dicts.
    :param min_bin_n: minimum samples for a bin to be included.
    :return: dict with intercept and slope (None if < 3 qualifying bins).
    """
    populated = [
        b
        for b in bins
        if b.get("n", 0) >= min_bin_n and b.get("avg_predicted") is not None
    ]
    if len(populated) < 3:
        return dict(_CAL_REG_NONE)

    weights = [b["n"] for b in populated]
    xs = [b["avg_predicted"] for b in populated]
    ys = [b["realized_rate"] for b in populated]
    total_w = sum(weights)

    mean_x = sum(w * x for w, x in zip(weights, xs)) / total_w
    mean_y = sum(w * y for w, y in zip(weights, ys)) / total_w

    num = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys))
    den = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))

    if abs(den) < 1e-12:
        return dict(_CAL_REG_NONE)

    slope = num / den
    intercept = mean_y - slope * mean_x
    return {
        "calibration_intercept": round(intercept, 4),
        "calibration_slope": round(slope, 4),
    }


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------


def brier_sort_key(item: tuple[str, dict[str, Any]]) -> float:
    """Sort key for ranking (name, stats) entries by Brier ascending.

    Used by both the scorer CLI preview and the analyze.py report
    sections so there is a single source of truth. Uses an explicit
    ``is not None`` check; ``.get("brier") or 999`` collapses
    ``brier=0.0`` (perfect score) to 999 via falsy-or and sorts best
    cells to the end.

    :param item: a (key, stats) pair from a ``by_*`` dim dict.
    :return: the Brier value, or 999.0 when Brier is None.
    """
    brier = item[1].get("brier")
    return brier if brier is not None else 999.0


def _score_latency_reservoir(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Return the last ``LATENCY_RESERVOIR_SIZE`` latencies per tool in insertion order.

    :param rows: input rows.
    :return: ``{tool_name: [latencies]}``. Tools with no ``latency_s``
        values are omitted.
    """
    reservoir: dict[str, list[float]] = {}
    for row in rows:
        latency = row.get("latency_s")
        if latency is None:
            continue
        tool = row.get("tool_name") or "unknown"
        reservoir.setdefault(tool, []).append(latency)
    for tool, samples in reservoir.items():
        if len(samples) > LATENCY_RESERVOIR_SIZE:
            reservoir[tool] = samples[-LATENCY_RESERVOIR_SIZE:]
    return reservoir


def _score_extreme_predictions(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the top-``WORST_BEST_SIZE`` worst and best predictions by Brier.

    Considers only valid rows (parse status ``"valid"`` with both
    ``p_yes`` and ``final_outcome`` present). Deduplicated by
    ``question_text``: each question contributes at most one entry per
    list.

    :param rows: input rows.
    :return: ``(worst, best)``. Worst sorted by Brier descending, best
        ascending. Each truncated to ``WORST_BEST_SIZE``.
    """
    worst_by_q: dict[str, dict[str, Any]] = {}
    best_by_q: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("prediction_parse_status") != "valid":
            continue
        p_yes = row.get("p_yes")
        outcome = row.get("final_outcome")
        if p_yes is None or outcome is None:
            continue
        question = row.get("question_text")
        if not question:
            continue
        entry = {
            "question_text": question,
            "tool_name": row.get("tool_name") or "unknown",
            "p_yes": p_yes,
            "final_outcome": outcome,
            "brier": round(brier_score(p_yes, outcome), 4),
            "platform": row.get("platform") or "unknown",
            "category": row.get("category"),
        }
        prev_worst = worst_by_q.get(question)
        if prev_worst is None or entry["brier"] > prev_worst["brier"]:
            worst_by_q[question] = entry
        prev_best = best_by_q.get(question)
        if prev_best is None or entry["brier"] < prev_best["brier"]:
            best_by_q[question] = entry

    worst = sorted(worst_by_q.values(), key=lambda e: e["brier"], reverse=True)
    best = sorted(best_by_q.values(), key=lambda e: e["brier"])
    return worst[:WORST_BEST_SIZE], best[:WORST_BEST_SIZE]


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all scores from production log rows."""
    total = len(rows)
    overall = compute_group_stats(rows)

    # Per-tool
    by_tool = {
        tool: compute_group_stats(group)
        for tool, group in group_by(rows, "tool_name").items()
    }

    # Per-platform
    by_platform = {
        platform: compute_group_stats(group)
        for platform, group in group_by(rows, "platform").items()
    }

    # Per-category
    by_category = {
        cat: compute_group_stats(group)
        for cat, group in group_by(rows, "category").items()
    }

    # Per-horizon
    by_horizon = group_by_horizon(rows)

    # Per-difficulty (requires market_prob_at_prediction)
    difficulty_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        d = classify_difficulty(row.get("market_prob_at_prediction"))
        difficulty_groups[d].append(row)
    by_difficulty = {k: compute_group_stats(g) for k, g in difficulty_groups.items()}

    # Per-liquidity (requires market_liquidity_at_prediction)
    liquidity_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        liq = classify_liquidity(row.get("market_liquidity_at_prediction"))
        liquidity_groups[liq].append(row)
    by_liquidity = {k: compute_group_stats(g) for k, g in liquidity_groups.items()}

    # Platform × difficulty cross breakdown
    pd_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        plat = row.get("platform") or "unknown"
        diff = classify_difficulty(row.get("market_prob_at_prediction"))
        pd_groups[f"{plat} | {diff}"].append(row)
    by_platform_difficulty = {k: compute_group_stats(g) for k, g in pd_groups.items()}

    # Platform × liquidity cross breakdown
    pl_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        plat = row.get("platform") or "unknown"
        liq = classify_liquidity(row.get("market_liquidity_at_prediction"))
        pl_groups[f"{plat} | {liq}"].append(row)
    by_platform_liquidity = {k: compute_group_stats(g) for k, g in pl_groups.items()}

    # Tool × platform cross breakdown
    by_tool_platform = group_by_composite(rows, ["tool_name", "platform"])

    # Tool × category cross breakdown
    by_tool_category = group_by_composite(rows, ["tool_name", "category"])

    # Tool × version (normalized: tool_version OR tool_ipfs_hash) cross breakdown
    tv_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tvm_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tool = row.get("tool_name") or "unknown"
        version = row.get("tool_version") or row.get("tool_ipfs_hash") or "unknown"
        mode = row.get("mode") or "production_replay"
        tv_groups[f"{tool} | {version}"].append(row)
        tvm_groups[f"{tool} | {version} | {mode}"].append(row)
    by_tool_version = {k: compute_group_stats(g) for k, g in tv_groups.items()}
    by_tool_version_mode = {k: compute_group_stats(g) for k, g in tvm_groups.items()}

    # Tool × platform × horizon breakdown
    by_tool_platform_horizon = group_by_composite(
        rows,
        ["tool_name", "platform"],
        horizon=True,
    )

    # Monthly trend
    trend = group_by_month(rows)

    # Calibration — overall and per-tool
    calibration = compute_calibration(rows)
    ece = compute_ece(calibration)
    cal_reg = compute_calibration_regression(rows)
    calibration_by_tool = {
        tool: compute_calibration(group)
        for tool, group in group_by(rows, "tool_name").items()
    }

    # Edge eligibility reporting — same schema as _finalize_scores()
    n_edge_eligible = sum(1 for r in rows if _is_edge_eligible(r))
    valid_n = overall["valid_n"]
    edge_eligibility = {
        "n_total": total,
        "n_eligible": n_edge_eligible,
        "n_excluded": total - n_edge_eligible,
        "exclusion_reasons": {
            "invalid_or_incomplete": total - valid_n,
            "missing_market_prob": valid_n - n_edge_eligible,
        },
    }

    # Schema parity with _accumulate_and_write: both score() and the
    # incremental path write the same finalized dict shape to
    # scores_<platform>.json. Kept even when generate_report does not
    # render Latency / Worst / Best so consumers reading the JSON
    # directly see the same keys regardless of which path produced it.
    latency_reservoir = _score_latency_reservoir(rows)
    worst_10, best_10 = _score_extreme_predictions(rows)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_rows": total,
        "valid_rows": overall["valid_n"],
        "overall": overall,
        "by_tool": by_tool,
        "by_platform": by_platform,
        "by_category": by_category,
        "by_horizon": by_horizon,
        "by_difficulty": by_difficulty,
        "by_liquidity": by_liquidity,
        "by_platform_difficulty": by_platform_difficulty,
        "by_platform_liquidity": by_platform_liquidity,
        "by_tool_platform": by_tool_platform,
        "by_tool_category": by_tool_category,
        "by_tool_platform_horizon": by_tool_platform_horizon,
        "by_tool_version": by_tool_version,
        "by_tool_version_mode": by_tool_version_mode,
        "trend": trend,
        "calibration": calibration,
        "ece": ece,
        **cal_reg,
        "calibration_by_tool": calibration_by_tool,
        "edge_eligibility": edge_eligibility,
        "latency_reservoir": latency_reservoir,
        "worst_10": worst_10,
        "best_10": best_10,
    }


# ---------------------------------------------------------------------------
# Incremental scoring — accumulator helpers
# ---------------------------------------------------------------------------


def _empty_group() -> dict[str, Any]:
    """Return a fresh group accumulator."""
    return {
        "n": 0,
        "valid_n": 0,
        "brier_sum": 0.0,
        "correct_count": 0,
        "n_directional": 0,
        "no_signal_count": 0,
        "sharpness_sum": 0.0,
        "outcome_yes_count": 0,
        "log_loss_sum": 0.0,
        "edge_sum": 0.0,
        "edge_n": 0,
        "edge_positive_count": 0,
        # Diagnostic edge metrics
        "disagree_tool_win_count": 0,
        "disagree_n": 0,
        "brier_sum_no_trade": 0.0,
        "n_no_trade": 0,
        "brier_sum_small_trade": 0.0,
        "n_small_trade": 0,
        "brier_sum_large_trade": 0.0,
        "n_large_trade": 0,
        "bias_sum": 0.0,
        "n_bias_losses": 0,
    }


def _empty_scores(current_month: str) -> dict[str, Any]:
    """Return a fresh scores.json structure with empty accumulators.

    :param current_month: YYYY-MM string for the current month.
    :return: dict matching the accumulator schema.
    """
    return {
        "current_month": current_month,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": _empty_group(),
        "by_tool": {},
        "by_platform": {},
        "by_category": {},
        "by_horizon": {},
        "by_tool_platform": {},
        "by_tool_category": {},
        "by_tool_version": {},
        "by_tool_version_mode": {},
        "by_config": {},
        "by_difficulty": {},
        "by_liquidity": {},
        "by_platform_difficulty": {},
        "by_platform_liquidity": {},
        "calibration": {
            _bin_label(lo, hi): {"count": 0, "outcome_sum": 0, "predicted_sum": 0.0}
            for lo, hi in CALIBRATION_BINS
        },
        "parse_breakdown": {},
        "latency_reservoir": {},
        "worst_10": [],
        "best_10": [],
        "calibration_pairs": [],
    }


def _accumulate_group(group: dict[str, Any], row: dict[str, Any]) -> None:
    """Merge one row into a group accumulator (mutates *group*).

    :param group: accumulator dict with n, valid_n, brier_sum, etc.
    :param row: a production log row dict.
    """
    group["n"] += 1
    is_valid = (
        row.get("prediction_parse_status") == "valid"
        and row.get("final_outcome") is not None
        and row.get("p_yes") is not None
    )
    if is_valid:
        group["valid_n"] += 1
        p_yes = row["p_yes"]
        outcome = row["final_outcome"]
        group["brier_sum"] += brier_score(p_yes, outcome)
        group["log_loss_sum"] += log_loss_score(p_yes, outcome)
        # Directional accuracy — exclude p_yes == 0.5
        if p_yes != 0.5:
            group["n_directional"] += 1
            if (p_yes > 0.5) == outcome:
                group["correct_count"] += 1
        else:
            group["no_signal_count"] += 1
        group["sharpness_sum"] += abs(p_yes - 0.5)
        if outcome:
            group["outcome_yes_count"] += 1
        # Edge over market
        market_prob = row.get("market_prob_at_prediction")
        if market_prob is not None:
            brier = brier_score(p_yes, outcome)
            edge = edge_score(p_yes, market_prob, outcome)
            group["edge_sum"] += edge
            group["edge_n"] += 1
            if edge > 0:
                group["edge_positive_count"] += 1
            # Diagnostic edge metrics
            bucket = disagree_bucket(p_yes, market_prob)
            group[f"brier_sum_{bucket}"] += brier
            group[f"n_{bucket}"] += 1
            if bucket != "no_trade":
                result = classify_disagreement(p_yes, market_prob, outcome)
                if result != "tie":
                    group["disagree_n"] += 1
                    if result == "tool_win":
                        group["disagree_tool_win_count"] += 1
                    else:
                        group["bias_sum"] += p_yes - (1.0 if outcome else 0.0)
                        group["n_bias_losses"] += 1


def _derive_group(group: dict[str, Any]) -> dict[str, Any]:
    """Compute derived fields from a group accumulator.

    Returns a new dict with all accumulator fields plus derived
    brier, accuracy, sharpness, reliability, and decision_worthy.

    :param group: accumulator dict.
    :return: dict with accumulators and derived stats.
    """
    result = dict(group)
    n = group["n"]
    valid_n = group["valid_n"]
    if n == 0:
        result.update(
            brier=None,
            directional_accuracy=None,
            no_signal_rate=None,
            log_loss=None,
            sharpness=None,
            reliability=None,
            decision_worthy=False,
            outcome_yes_rate=None,
            baseline_brier=None,
            brier_skill_score=None,
        )
    elif valid_n == 0:
        result.update(
            brier=None,
            directional_accuracy=None,
            no_signal_rate=None,
            log_loss=None,
            sharpness=None,
            reliability=round(0.0, 4),
            decision_worthy=False,
            outcome_yes_rate=None,
            baseline_brier=None,
            brier_skill_score=None,
        )
    else:
        brier = round(group["brier_sum"] / valid_n, 4)
        yes_rate = group["outcome_yes_count"] / valid_n
        baseline_brier = round(yes_rate * (1 - yes_rate), 4)
        n_dir = group.get("n_directional", 0)
        no_sig = group.get("no_signal_count", 0)
        result["brier"] = brier
        result["directional_accuracy"] = (
            round(group["correct_count"] / n_dir, 4) if n_dir > 0 else None
        )
        result["n_directional"] = n_dir
        result["no_signal_count"] = no_sig
        result["no_signal_rate"] = round(no_sig / valid_n, 4)
        result["log_loss"] = round(group["log_loss_sum"] / valid_n, 4)
        result["sharpness"] = round(group["sharpness_sum"] / valid_n, 4)
        result["reliability"] = round(valid_n / n, 4)
        result["decision_worthy"] = valid_n >= MIN_SAMPLE_SIZE
        result["outcome_yes_rate"] = round(yes_rate, 4)
        result["baseline_brier"] = baseline_brier
        if baseline_brier > 0:
            result["brier_skill_score"] = round(1 - (brier / baseline_brier), 4)
        else:
            result["brier_skill_score"] = None

    # Edge over market — derived from edge accumulators
    edge_n = group.get("edge_n", 0)
    result["edge_n"] = edge_n
    if edge_n > 0:
        result["edge"] = round(group["edge_sum"] / edge_n, 4)
        result["edge_positive_rate"] = round(group["edge_positive_count"] / edge_n, 4)
    else:
        result["edge"] = None
        result["edge_positive_rate"] = None

    # Diagnostic edge metrics — conditional accuracy, disagreement Brier, bias
    _derive_diagnostic_metrics(group, result)

    return result


def _derive_diagnostic_metrics(group: dict[str, Any], result: dict[str, Any]) -> None:
    """Derive diagnostic edge metrics from accumulators into *result*.

    :param group: accumulator dict with diagnostic keys.
    :param result: output dict to populate (mutated in place).
    """
    disagree_n = group.get("disagree_n", 0)
    result["disagree_n"] = disagree_n
    if disagree_n >= MIN_SAMPLE_SIZE:
        result["conditional_accuracy_rate"] = round(
            group["disagree_tool_win_count"] / disagree_n, 4
        )
    else:
        result["conditional_accuracy_rate"] = None

    for bucket in ("no_trade", "small_trade", "large_trade"):
        n_bucket = group.get(f"n_{bucket}", 0)
        result[f"n_{bucket}"] = n_bucket
        if n_bucket >= MIN_SAMPLE_SIZE:
            result[f"brier_{bucket}"] = round(
                group[f"brier_sum_{bucket}"] / n_bucket, 4
            )
        else:
            result[f"brier_{bucket}"] = None

    n_losses = group.get("n_bias_losses", 0)
    result["n_bias_losses"] = n_losses
    if n_losses >= MIN_SAMPLE_SIZE:
        result["directional_bias"] = round(group["bias_sum"] / n_losses, 4)
    else:
        result["directional_bias"] = None


def _ensure_and_accumulate(
    dimension: dict[str, dict[str, Any]],
    key: str,
    row: dict[str, Any],
) -> None:
    """Ensure a key exists in a dimension dict and accumulate the row into it.

    :param dimension: dict mapping group keys to accumulator dicts.
    :param key: the group key to accumulate into.
    :param row: a production log row dict.
    """
    if key not in dimension:
        dimension[key] = _empty_group()
    _accumulate_group(dimension[key], row)


def _update_extreme_list(
    entries: list[dict[str, Any]],
    new_entry: dict[str, Any],
    keep: str = "worst",
) -> list[dict[str, Any]]:
    """Update a deduplicated worst/best list with a new entry.

    Keeps one entry per unique ``question_text``. For ``keep="worst"``,
    replaces an existing entry only if the new Brier is higher. For
    ``keep="best"``, replaces only if lower.

    :param entries: current list of extreme entries.
    :param new_entry: candidate entry to insert.
    :param keep: ``"worst"`` to keep highest Brier, ``"best"`` for lowest.
    :return: updated list, sorted and truncated to ``WORST_BEST_SIZE``.
    """
    question = new_entry["question_text"]
    existing = {e["question_text"]: e for e in entries}
    is_better = (
        (lambda new, old: new > old)
        if keep == "worst"
        else (lambda new, old: new < old)
    )

    if question in existing:
        if is_better(new_entry["brier"], existing[question]["brier"]):
            entries = [e for e in entries if e["question_text"] != question]
            entries.append(new_entry)
    else:
        entries.append(new_entry)

    reverse = keep == "worst"
    entries.sort(key=lambda x: x["brier"], reverse=reverse)
    return entries[:WORST_BEST_SIZE]


def _accumulate_calibration(scores: dict[str, Any], row: dict[str, Any]) -> None:
    """Accumulate calibration bins and pairs for a valid row.

    :param scores: the full scores dict with calibration accumulators.
    :param row: a valid production log row dict.
    """
    p = row["p_yes"]
    for lo, hi in CALIBRATION_BINS:
        if lo <= p < hi:
            label = _bin_label(lo, hi)
            bucket = scores["calibration"][label]
            bucket["count"] += 1
            bucket["outcome_sum"] += 1 if row["final_outcome"] else 0
            bucket["predicted_sum"] += p
            break
    # Store (p_yes, outcome) for row-level calibration regression.
    # Reservoir sampling: once at capacity, randomly replace entries
    # so the sample remains representative without unbounded growth.
    pair = [p, 1 if row["final_outcome"] else 0]
    pairs = scores["calibration_pairs"]
    if len(pairs) < CALIBRATION_PAIRS_RESERVOIR_SIZE:
        pairs.append(pair)
    else:
        idx = _RESERVOIR_RNG.randint(0, scores["overall"]["valid_n"] - 1)
        if idx < CALIBRATION_PAIRS_RESERVOIR_SIZE:
            pairs[idx] = pair


def _accumulate_row(scores: dict[str, Any], row: dict[str, Any]) -> None:
    """Merge one row into all accumulator dimensions (mutates *scores*).

    :param scores: the full scores dict with accumulators.
    :param row: a production log row dict.
    """
    _accumulate_group(scores["overall"], row)

    tool = row.get("tool_name") or "unknown"
    platform = row.get("platform") or "unknown"
    category = row.get("category") or "unknown"
    horizon = classify_horizon(row.get("prediction_lead_time_days"))
    # Production rows store the IPFS hash in `tool_version`; tournament rows
    # store it in `tool_ipfs_hash`. Normalize so both populate the same key.
    tool_version = row.get("tool_version") or row.get("tool_ipfs_hash") or "unknown"
    mode = row.get("mode") or "production_replay"
    config_hash = row.get("config_hash") or "unknown"

    _ensure_and_accumulate(scores["by_tool"], tool, row)
    _ensure_and_accumulate(scores["by_platform"], platform, row)
    _ensure_and_accumulate(scores["by_category"], category, row)
    _ensure_and_accumulate(scores["by_horizon"], horizon, row)
    _ensure_and_accumulate(scores["by_tool_platform"], f"{tool} | {platform}", row)
    _ensure_and_accumulate(scores["by_tool_category"], f"{tool} | {category}", row)
    _ensure_and_accumulate(scores["by_tool_version"], f"{tool} | {tool_version}", row)
    _ensure_and_accumulate(
        scores["by_tool_version_mode"],
        f"{tool} | {tool_version} | {mode}",
        row,
    )
    _ensure_and_accumulate(scores["by_config"], f"{tool} | {config_hash}", row)

    difficulty = classify_difficulty(row.get("market_prob_at_prediction"))
    _ensure_and_accumulate(scores["by_difficulty"], difficulty, row)

    liquidity = classify_liquidity(row.get("market_liquidity_at_prediction"))
    _ensure_and_accumulate(scores["by_liquidity"], liquidity, row)

    _ensure_and_accumulate(
        scores["by_platform_difficulty"], f"{platform} | {difficulty}", row
    )
    _ensure_and_accumulate(
        scores["by_platform_liquidity"], f"{platform} | {liquidity}", row
    )

    # Calibration buckets + pairs
    is_valid = (
        row.get("prediction_parse_status") == "valid"
        and row.get("final_outcome") is not None
        and row.get("p_yes") is not None
    )
    if is_valid:
        _accumulate_calibration(scores, row)

    # Parse breakdown
    status = row.get("prediction_parse_status") or "unknown"
    if tool not in scores["parse_breakdown"]:
        scores["parse_breakdown"][tool] = {}
    tool_pb = scores["parse_breakdown"][tool]
    tool_pb[status] = tool_pb.get(status, 0) + 1

    # Latency reservoir (last N per tool)
    latency = row.get("latency_s")
    if latency is not None:
        if tool not in scores["latency_reservoir"]:
            scores["latency_reservoir"][tool] = []
        reservoir = scores["latency_reservoir"][tool]
        reservoir.append(latency)
        if len(reservoir) > LATENCY_RESERVOIR_SIZE:
            reservoir.pop(0)

    # Worst / best 10 (deduplicated by question_text)
    question_text = row.get("question_text")
    if is_valid and question_text:
        row_brier = brier_score(row["p_yes"], row["final_outcome"])
        entry = {
            "question_text": question_text,
            "tool_name": tool,
            "p_yes": row["p_yes"],
            "final_outcome": row["final_outcome"],
            "brier": round(row_brier, 4),
            "platform": platform,
            "category": category,
        }
        scores["worst_10"] = _update_extreme_list(
            scores["worst_10"],
            entry,
            keep="worst",
        )
        scores["best_10"] = _update_extreme_list(
            scores["best_10"],
            entry,
            keep="best",
        )


def _finalize_scores(scores: dict[str, Any]) -> dict[str, Any]:
    """Derive all computed fields from accumulators.

    Returns a new dict suitable for writing to ``scores.json``.

    :param scores: raw accumulator dict.
    :return: finalized dict with derived stats.
    """
    result: dict[str, Any] = {
        "current_month": scores["current_month"],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    result["overall"] = _derive_group(scores["overall"])
    result["total_rows"] = scores["overall"]["n"]
    result["valid_rows"] = scores["overall"]["valid_n"]

    for dim in (
        "by_tool",
        "by_platform",
        "by_category",
        "by_horizon",
        "by_tool_platform",
        "by_tool_category",
        "by_tool_version",
        "by_tool_version_mode",
        "by_config",
        "by_difficulty",
        "by_liquidity",
        "by_platform_difficulty",
        "by_platform_liquidity",
    ):
        result[dim] = {k: _derive_group(v) for k, v in scores[dim].items()}

    # Edge eligibility from accumulators
    overall_n = scores["overall"]["n"]
    overall_valid_n = scores["overall"]["valid_n"]
    overall_edge_n = scores["overall"].get("edge_n", 0)
    result["edge_eligibility"] = {
        "n_total": overall_n,
        "n_eligible": overall_edge_n,
        "n_excluded": overall_n - overall_edge_n,
        "exclusion_reasons": {
            "invalid_or_incomplete": overall_n - overall_valid_n,
            "missing_market_prob": overall_valid_n - overall_edge_n,
        },
    }

    # Calibration — derive avg_predicted, realized_rate, gap
    cal_result = []
    for lo, hi in CALIBRATION_BINS:
        label = _bin_label(lo, hi)
        bucket = scores["calibration"].get(label, {})
        count = bucket.get("count", 0)
        if count == 0:
            continue
        avg_pred = bucket["predicted_sum"] / count
        realized = bucket["outcome_sum"] / count
        gap = round(avg_pred - realized, 4)
        cal_result.append(
            {
                "bin": label,
                "avg_predicted": round(avg_pred, 4),
                "realized_rate": round(realized, 4),
                "n": count,
                "gap": gap,
            }
        )
    result["calibration"] = cal_result
    result["ece"] = compute_ece(cal_result)

    # Row-level logistic regression from stored pairs; bin-level fallback
    pairs = scores.get("calibration_pairs", [])
    if pairs:
        # Build minimal row dicts for compute_calibration_regression
        pair_rows = [
            {"prediction_parse_status": "valid", "p_yes": p, "final_outcome": bool(o)}
            for p, o in pairs
        ]
        result.update(compute_calibration_regression(pair_rows))
    else:
        result.update(_compute_calibration_regression_from_bins(cal_result))

    result["parse_breakdown"] = scores["parse_breakdown"]
    result["latency_reservoir"] = scores["latency_reservoir"]
    result["worst_10"] = scores["worst_10"]
    result["best_10"] = scores["best_10"]

    return result


# ---------------------------------------------------------------------------
# Incremental scoring — public API
# ---------------------------------------------------------------------------


def _snapshot_month(
    scores: dict[str, Any],
    history_path: Path,
) -> None:
    """Append the current month's final state to the history file.

    :param scores: current accumulator state (will be finalized for snapshot).
    :param history_path: path to ``scores_history.jsonl``.
    """
    finalized = _finalize_scores(scores)
    entry = {
        "month": scores["current_month"],
        "overall": finalized["overall"],
        "by_tool": finalized["by_tool"],
        "by_platform": finalized["by_platform"],
        "calibration": finalized["calibration"],
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_history(history_path: Path) -> list[dict[str, Any]]:
    """Read monthly snapshots from ``scores_history.jsonl``.

    :param history_path: path to the history file.
    :return: list of monthly summary dicts.
    """
    entries: list[dict[str, Any]] = []
    if not history_path.exists():
        return entries
    with open(history_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_scores_for_resume(scores_path: Path) -> dict[str, Any] | None:
    """Load scores.json and reconstruct the raw accumulator state.

    The saved format includes both derived fields (brier, directional_accuracy)
    and raw accumulators (brier_sum, correct_count). This function extracts
    the raw accumulators into the internal format used by ``_accumulate_row``.

    :param scores_path: path to ``scores.json``.
    :return: raw accumulator dict or None.
    """
    if not scores_path.exists():
        return None
    try:
        data = json.loads(scores_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if "current_month" not in data or "brier_sum" not in data.get("overall", {}):
        return None

    def _restore_group(g: dict[str, Any]) -> dict[str, Any]:
        restored = _empty_group()
        restored["n"] = g["n"]
        restored["valid_n"] = g["valid_n"]
        restored["brier_sum"] = g["brier_sum"]
        restored["correct_count"] = g["correct_count"]
        restored["n_directional"] = g.get("n_directional", 0)
        restored["no_signal_count"] = g.get("no_signal_count", 0)
        restored["sharpness_sum"] = g["sharpness_sum"]
        restored["outcome_yes_count"] = g.get("outcome_yes_count", 0)
        restored["log_loss_sum"] = g.get("log_loss_sum", 0.0)
        restored["edge_sum"] = g.get("edge_sum", 0.0)
        restored["edge_n"] = g.get("edge_n", 0)
        restored["edge_positive_count"] = g.get("edge_positive_count", 0)
        # Diagnostic edge metrics — default to 0 for pre-existing scores
        for key in (
            "disagree_tool_win_count",
            "disagree_n",
            "brier_sum_no_trade",
            "n_no_trade",
            "brier_sum_small_trade",
            "n_small_trade",
            "brier_sum_large_trade",
            "n_large_trade",
            "bias_sum",
            "n_bias_losses",
        ):
            restored[key] = g.get(key, restored[key])
        return restored

    scores: dict[str, Any] = {
        "current_month": data["current_month"],
        "generated_at": data.get("generated_at", ""),
        "overall": _restore_group(data["overall"]),
    }
    for dim in (
        "by_tool",
        "by_platform",
        "by_category",
        "by_horizon",
        "by_tool_platform",
        "by_tool_category",
        "by_tool_version",
        "by_tool_version_mode",
        "by_config",
        "by_difficulty",
        "by_liquidity",
        "by_platform_difficulty",
        "by_platform_liquidity",
    ):
        scores[dim] = {}
        for key, group in data.get(dim, {}).items():
            scores[dim][key] = _restore_group(group)

    # Restore calibration accumulators
    if "_calibration_accum" in data:
        scores["calibration"] = data["_calibration_accum"]
    else:
        scores["calibration"] = {
            _bin_label(lo, hi): {"count": 0, "outcome_sum": 0, "predicted_sum": 0.0}
            for lo, hi in CALIBRATION_BINS
        }

    scores["parse_breakdown"] = data.get("parse_breakdown", {})
    scores["latency_reservoir"] = data.get("latency_reservoir", {})
    scores["worst_10"] = data.get("worst_10", [])
    scores["best_10"] = data.get("best_10", [])
    scores["calibration_pairs"] = data.get("_calibration_pairs", [])

    # Preserve legacy scored_row_ids so update() can migrate them
    # to the separate dedup file on first run after upgrade.
    if "scored_row_ids" in data:
        scores["scored_row_ids"] = set(data["scored_row_ids"])

    return scores


def _accumulate_and_write(
    rows: list[dict[str, Any]],
    scores_path: Path,
    history_path: Path | None,
    emit_history: bool,
) -> dict[str, Any]:
    """Load existing scores (if any), accumulate *rows*, write output.

    Shared post-dedup implementation of update(). Callers pass an
    already-deduplicated, single-mode slice of rows. When
    ``emit_history`` is False (tournament path), the month-boundary
    snapshot step is skipped and ``history_path`` may be None.

    :param rows: pre-deduplicated rows for a single mode.
    :param scores_path: output path for the accumulator dict.
    :param history_path: optional history file for monthly snapshots.
    :param emit_history: when False, skip the snapshot step entirely.
    :return: finalized scores dict (also written to disk).
    """
    today_month = datetime.now(timezone.utc).strftime("%Y-%m")

    existing = _load_scores_for_resume(scores_path)
    if existing is not None:
        scores = existing
        if scores["current_month"] != today_month:
            if emit_history and history_path is not None:
                _snapshot_month(scores, history_path)
                scores = _empty_scores(today_month)
            else:
                # Tournament path: keep accumulating across month boundaries
                # so cross-mode comparisons (and callout thresholds) don't
                # reset to zero on the 1st of every month. Only advance the
                # stored month label.
                scores["current_month"] = today_month
    else:
        scores = _empty_scores(today_month)

    # Drop any migrated dedup set — dedup is owned by the caller now.
    scores.pop("scored_row_ids", None)

    for row in rows:
        _accumulate_row(scores, row)

    finalized = _finalize_scores(scores)
    output = dict(finalized)
    output["overall"] = {
        **finalized["overall"],
        **{k: scores["overall"][k] for k in _ACCUM_KEYS},
    }
    for dim in (
        "by_tool",
        "by_platform",
        "by_category",
        "by_horizon",
        "by_tool_platform",
        "by_tool_category",
        "by_tool_version",
        "by_tool_version_mode",
        "by_config",
        "by_difficulty",
        "by_liquidity",
        "by_platform_difficulty",
        "by_platform_liquidity",
    ):
        for key, group in scores[dim].items():
            output[dim][key] = {
                **finalized[dim][key],
                **{k: group[k] for k in _ACCUM_KEYS},
            }
    output["_calibration_accum"] = scores["calibration"]
    output["_calibration_pairs"] = scores["calibration_pairs"]

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(output, indent=2))

    return finalized


def update(
    new_rows: list[dict[str, Any]],
    scores_path: Path = DEFAULT_OUTPUT,
    history_path: Path = DEFAULT_HISTORY,
    dedup_path: Path | None = None,
    tournament_scores_path: Path | None = None,
) -> dict[str, Any]:
    """Incrementally merge new rows into the scores accumulators.

    Splits rows by ``row["mode"]``: production rows land in
    ``scores_path``; tournament rows land in ``tournament_scores_path``
    (derived from ``scores_path`` when omitted). Dedup is global across
    both modes. Monthly history snapshots are emitted for the production
    file only.

    :param new_rows: list of log row dicts (any mode).
    :param scores_path: path to production ``scores.json``.
    :param history_path: path to ``scores_history.jsonl`` (production only).
    :param dedup_path: path to ``scored_row_ids.json`` (shared).
    :param tournament_scores_path: path to ``scores_tournament.json``.
        Derived from ``scores_path`` when None.
    :return: finalized production scores dict. Tournament result is
        written to disk but not returned (backward compat).
    """
    if dedup_path is None:
        dedup_path = scores_path.parent / "scored_row_ids.json"
    if tournament_scores_path is None:
        tournament_scores_path = _derive_tournament_path(scores_path)

    scored_ids = _load_dedup_ids(dedup_path)
    # Legacy migration: if an older scores.json still carries scored_row_ids
    # inline (pre-dedup-file format), fold it into the shared dedup set now
    # so the downstream partitioned accumulators start from a clean slate.
    for legacy_path in (scores_path, tournament_scores_path):
        if legacy_path.exists():
            try:
                legacy = json.loads(legacy_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            legacy_ids = legacy.get("scored_row_ids")
            if legacy_ids:
                scored_ids.update(legacy_ids)

    deduped_rows: list[dict[str, Any]] = []
    skipped = 0
    no_id = 0
    for row in new_rows:
        row_id = row.get("row_id")
        if not row_id:
            no_id += 1
            deduped_rows.append(row)
            continue
        if row_id in scored_ids:
            skipped += 1
            continue
        deduped_rows.append(row)
        scored_ids.add(row_id)

    _save_dedup_ids(dedup_path, scored_ids)

    if skipped:
        logging.getLogger(__name__).warning(
            "Skipped %d duplicate rows (already scored)", skipped
        )
    if no_id:
        logging.getLogger(__name__).warning(
            "%d rows without row_id cannot be deduplicated", no_id
        )

    prod_rows, tourn_rows = _partition_rows_by_mode(deduped_rows)

    prod_result = _accumulate_and_write(
        prod_rows, scores_path, history_path, emit_history=True
    )
    _accumulate_and_write(tourn_rows, tournament_scores_path, None, emit_history=False)

    # Per-platform accumulators each track their own _ACCUM_KEYS state so
    # future update() calls merge correctly. Dedup is shared via the
    # caller-level scored_row_ids.json, so a row can never double-count.
    for platform, plat_prod in _partition_rows_by_platform(prod_rows).items():
        _accumulate_and_write(
            plat_prod,
            _derive_platform_path(scores_path, platform),
            None,
            emit_history=False,
        )
    for platform, plat_tourn in _partition_rows_by_platform(tourn_rows).items():
        _accumulate_and_write(
            plat_tourn,
            _derive_platform_path(tournament_scores_path, platform),
            None,
            emit_history=False,
        )

    return prod_result


def _collect_rebuild_rows(
    logs_dir: Path,
    tournament_input: Path | None,
) -> list[dict[str, Any]]:
    """Load production log rows + optional tournament rows for rebuild."""
    pattern = str(logs_dir / "production_log_*.jsonl")
    files = sorted(glob_mod.glob(pattern))

    rows: list[dict[str, Any]] = []
    for filepath in files:
        rows.extend(load_rows(Path(filepath)))

    if tournament_input is not None and tournament_input.exists():
        rows.extend(load_rows(tournament_input))

    return rows


def _rebuild_single_mode(
    mode_rows: list[dict[str, Any]],
    scores_path: Path,
    history_path: Path | None,
    emit_history: bool,
) -> tuple[dict[str, Any], set[str]]:
    """Rebuild one mode's scores from a pre-filtered row list.

    Returns the finalized scores plus the set of row_ids seen (for
    dedup-file regeneration at the caller level).

    :param mode_rows: rows already filtered to a single mode.
    :param scores_path: output path for the accumulator dict.
    :param history_path: optional history file for monthly snapshots.
    :param emit_history: when False, snapshots are skipped and
        ``history_path`` is ignored.
    :return: tuple of (finalized scores dict, set of row_ids consumed).
    """
    if not mode_rows:
        scores = _empty_scores(datetime.now(timezone.utc).strftime("%Y-%m"))
        finalized = _finalize_scores(scores)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_path.write_text(json.dumps(finalized, indent=2))
        return finalized, set()

    mode_rows.sort(key=lambda r: r.get("predicted_at") or "")

    months: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mode_rows:
        predicted_at = row.get("predicted_at")
        month = predicted_at[:7] if predicted_at else "unknown"
        months[month].append(row)

    if emit_history and history_path is not None and history_path.exists():
        history_path.unlink()

    sorted_months = sorted(months.keys())
    last_month = sorted_months[-1]
    all_row_ids: set[str] = set()

    for month in sorted_months[:-1]:
        scores = _empty_scores(month)
        for row in months[month]:
            row_id = row.get("row_id")
            if row_id and row_id in all_row_ids:
                continue
            _accumulate_row(scores, row)
            if row_id:
                all_row_ids.add(row_id)
        if emit_history and history_path is not None:
            _snapshot_month(scores, history_path)

    scores = _empty_scores(last_month)
    for row in months[last_month]:
        row_id = row.get("row_id")
        if row_id and row_id in all_row_ids:
            continue
        _accumulate_row(scores, row)
        if row_id:
            all_row_ids.add(row_id)

    finalized = _finalize_scores(scores)
    output = dict(finalized)
    output["overall"] = {
        **finalized["overall"],
        **{k: scores["overall"][k] for k in _ACCUM_KEYS},
    }
    for dim in (
        "by_tool",
        "by_platform",
        "by_category",
        "by_horizon",
        "by_tool_platform",
        "by_tool_category",
        "by_tool_version",
        "by_tool_version_mode",
        "by_config",
        "by_difficulty",
        "by_liquidity",
        "by_platform_difficulty",
        "by_platform_liquidity",
    ):
        for key, group in scores[dim].items():
            output[dim][key] = {
                **finalized[dim][key],
                **{k: group[k] for k in _ACCUM_KEYS},
            }
    output["_calibration_accum"] = scores["calibration"]
    output["_calibration_pairs"] = scores["calibration_pairs"]

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(output, indent=2))

    return finalized, all_row_ids


def rebuild(
    logs_dir: Path = DEFAULT_LOGS_DIR,
    scores_path: Path = DEFAULT_OUTPUT,
    history_path: Path = DEFAULT_HISTORY,
    dedup_path: Path | None = None,
    tournament_input: Path | None = None,
    tournament_scores_path: Path | None = None,
) -> dict[str, Any]:
    """Rebuild both production and tournament scores from all log files.

    Production rows land in ``scores_path`` (default ``scores.json``)
    and emit monthly history snapshots. Tournament rows (from
    ``tournament_input`` and/or ``mode=tournament`` entries in the logs)
    land in ``tournament_scores_path`` (default
    ``scores_tournament.json``) and do **not** emit history.

    Dedup is global across both modes: the combined set of row_ids is
    written to ``dedup_path``.

    :param logs_dir: directory containing daily log files.
    :param scores_path: output path for production ``scores.json``.
    :param history_path: output path for ``scores_history.jsonl``
        (production only).
    :param dedup_path: path to ``scored_row_ids.json``.
    :param tournament_input: optional path to ``tournament_scored.jsonl``
        to merge into the rebuild input.
    :param tournament_scores_path: path to ``scores_tournament.json``.
        Derived from ``scores_path`` when None.
    :return: finalized production scores dict.
    """
    if dedup_path is None:
        dedup_path = scores_path.parent / "scored_row_ids.json"
    if tournament_scores_path is None:
        tournament_scores_path = _derive_tournament_path(scores_path)

    all_rows = _collect_rebuild_rows(logs_dir, tournament_input)

    if not all_rows:
        scores = _empty_scores(datetime.now(timezone.utc).strftime("%Y-%m"))
        finalized = _finalize_scores(scores)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_path.write_text(json.dumps(finalized, indent=2))
        tournament_scores_path.parent.mkdir(parents=True, exist_ok=True)
        tournament_scores_path.write_text(json.dumps(finalized, indent=2))
        # Empty per-platform files so the paths always exist.
        for platform in PLATFORMS:
            _derive_platform_path(scores_path, platform).write_text(
                json.dumps(finalized, indent=2)
            )
            _derive_platform_path(tournament_scores_path, platform).write_text(
                json.dumps(finalized, indent=2)
            )
        _save_dedup_ids(dedup_path, set())
        return finalized

    prod_rows, tourn_rows = _partition_rows_by_mode(all_rows)

    prod_finalized, prod_ids = _rebuild_single_mode(
        prod_rows, scores_path, history_path, emit_history=True
    )
    _, tourn_ids = _rebuild_single_mode(
        tourn_rows, tournament_scores_path, None, emit_history=False
    )

    # History is emitted for the combined accumulator only; per-platform
    # rebuilds skip the monthly snapshot step.
    for platform, plat_prod in _partition_rows_by_platform(prod_rows).items():
        _rebuild_single_mode(
            plat_prod,
            _derive_platform_path(scores_path, platform),
            None,
            emit_history=False,
        )
    for platform, plat_tourn in _partition_rows_by_platform(tourn_rows).items():
        _rebuild_single_mode(
            plat_tourn,
            _derive_platform_path(tournament_scores_path, platform),
            None,
            emit_history=False,
        )

    _save_dedup_ids(dedup_path, prod_ids | tourn_ids)

    return prod_finalized


# ---------------------------------------------------------------------------
# Period scoring — score a time window of log files
# ---------------------------------------------------------------------------


def _extract_date_from_log_path(path: str) -> str:
    """Extract a sortable date string from a log file path.

    Handles both ``YYYY-MM-DD.jsonl`` and ``production_log_YYYY_MM_DD.jsonl``
    naming conventions.

    :param path: file path string.
    :return: date string in ``YYYY-MM-DD`` format, or ``""`` if unparseable.
    """

    name = Path(path).stem
    # production_log_YYYY_MM_DD → YYYY-MM-DD
    m = re.search(r"(\d{4})[_-](\d{2})[_-](\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def score_period_split(
    logs_dir: Path = DEFAULT_LOGS_DIR,
    days: int = 1,
    tournament_input: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score the last *days* days, returning (production, tournament).

    Same collection rules as ``score_period`` but returns a tuple so
    callers can write both files separately.

    :param logs_dir: directory containing daily log files.
    :param days: score rows from the last N calendar days.
    :param tournament_input: optional path to ``tournament_scored.jsonl``
        whose rows are filtered to the same window and merged.
    :return: tuple of (production_scores, tournament_scores).
    """
    prod_rows, tourn_rows = _load_period_rows(logs_dir, days, tournament_input)
    return score(prod_rows), score(tourn_rows)


def _score_rows_by_platform(
    prod_rows: list[dict[str, Any]],
    tourn_rows: list[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Score combined + per-platform partitions of pre-loaded rows in one pass.

    :param prod_rows: production-mode rows (any platform).
    :param tourn_rows: tournament-mode rows (any platform).
    :return: ``{"all": (prod, tourn), "omen": (prod, tourn),
        "polymarket": (prod, tourn)}``. Unknown-platform rows stay in
        ``"all"`` only.
    """
    result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        "all": (score(prod_rows), score(tourn_rows)),
    }
    prod_by_plat = _partition_rows_by_platform(prod_rows)
    tourn_by_plat = _partition_rows_by_platform(tourn_rows)
    for platform in PLATFORMS:
        result[platform] = (
            score(prod_by_plat[platform]),
            score(tourn_by_plat[platform]),
        )
    return result


def score_period_split_by_platform(
    logs_dir: Path = DEFAULT_LOGS_DIR,
    days: int = 1,
    tournament_input: Path | None = None,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Score the last *days* days, returning combined + per-platform results.

    :param logs_dir: directory containing daily log files.
    :param days: score rows from the last N calendar days.
    :param tournament_input: optional path to ``tournament_scored.jsonl``
        whose rows are filtered to the same window and merged.
    :return: ``{"all": (prod, tourn), "omen": (prod, tourn),
        "polymarket": (prod, tourn)}``. Each ``(prod, tourn)`` tuple
        matches the shape of ``score_period_split``. Unknown-platform
        rows stay in ``"all"`` only.
    """
    prod_rows, tourn_rows = _load_period_rows(logs_dir, days, tournament_input)
    return _score_rows_by_platform(prod_rows, tourn_rows)


def _parse_predicted_at(value: Any) -> datetime | None:
    """Parse a row's ``predicted_at`` field into a UTC ``datetime``.

    :param value: raw ``predicted_at`` value from the row dict.
    :return: parsed ``datetime``, or ``None`` when missing or unparseable.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_period_rows(
    logs_dir: Path,
    days: int,
    tournament_input: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load + filter + mode-partition period rows.

    :param logs_dir: directory containing daily log files.
    :param days: score rows from the last N calendar days.
    :param tournament_input: optional path to ``tournament_scored.jsonl``
        whose rows are filtered to the same window and merged.
    :return: ``(production_rows, tournament_rows)`` both filtered to the
        period window.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    daily_pattern = str(logs_dir / "????-??-??.jsonl")
    prod_pattern = str(logs_dir / "production_log_*.jsonl")
    all_files = sorted(
        set(glob_mod.glob(daily_pattern) + glob_mod.glob(prod_pattern)),
        key=_extract_date_from_log_path,
    )

    rows: list[dict[str, Any]] = []
    for filepath in all_files:
        for row in load_rows(Path(filepath)):
            predicted_at = _parse_predicted_at(row.get("predicted_at"))
            if predicted_at is not None and predicted_at >= cutoff:
                rows.append(row)

    if tournament_input is not None and tournament_input.exists():
        for row in load_rows(tournament_input):
            predicted_at = _parse_predicted_at(row.get("predicted_at"))
            if predicted_at is not None and predicted_at >= cutoff:
                rows.append(row)

    return _partition_rows_by_mode(rows)


def score_period(
    logs_dir: Path = DEFAULT_LOGS_DIR,
    days: int = 1,
    tournament_input: Path | None = None,
) -> dict[str, Any]:
    """Score production rows in the last *days* days (backward-compat).

    Backward-compat wrapper around ``score_period_split``; returns the
    production partition only. Callers that need the tournament scores
    should use ``score_period_split`` directly.

    :param logs_dir: directory containing daily log files.
    :param days: score rows from the last N calendar days.
    :param tournament_input: optional path to ``tournament_scored.jsonl``
        whose rows are filtered to the same window and merged.
    :return: production scores dict.
    """
    prod, _ = score_period_split(logs_dir, days, tournament_input)
    return prod


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the scorer CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Score production prediction data.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input JSONL file path (legacy full-recompute mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Production scores JSON output path",
    )
    parser.add_argument(
        "--output-tournament",
        type=Path,
        default=None,
        help=(
            "Tournament scores JSON output path. If omitted, derived from "
            "--output by appending '_tournament' to the stem."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild scores from all log files in --logs-dir",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory containing daily log files (for --rebuild)",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help="Path to scores_history.jsonl",
    )
    parser.add_argument(
        "--period-days",
        type=int,
        default=None,
        help="Score only the last N daily log files (for period reports)",
    )
    parser.add_argument(
        "--tournament-input",
        type=Path,
        default=None,
        help="Optional path to tournament_scored.jsonl to merge into scoring",
    )
    parser.add_argument(
        "--update",
        type=Path,
        default=None,
        help="Incrementally merge rows from PATH into scores.json via update()",
    )
    return parser


def _cli_update(args: argparse.Namespace, output_tournament: Path) -> None:
    """Handle the ``--update`` CLI mode."""
    rows = load_rows(args.update)
    result = update(
        rows,
        args.output,
        args.history,
        tournament_scores_path=output_tournament,
    )
    overall = result["overall"]
    print(
        f"Merged {len(rows)} rows from {args.update}: Brier={overall['brier']},"
        f" n={overall['n']}"
    )


def _cli_period(args: argparse.Namespace, output_tournament: Path) -> None:
    """Handle the ``--period-days`` CLI mode."""
    results = score_period_split_by_platform(
        logs_dir=args.logs_dir,
        days=args.period_days,
        tournament_input=args.tournament_input,
    )
    prod_result, tourn_result = results["all"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prod_result, indent=2))
    output_tournament.parent.mkdir(parents=True, exist_ok=True)
    output_tournament.write_text(json.dumps(tourn_result, indent=2))

    for platform in PLATFORMS:
        plat_prod, plat_tourn = results[platform]
        _derive_platform_path(args.output, platform).write_text(
            json.dumps(plat_prod, indent=2)
        )
        _derive_platform_path(output_tournament, platform).write_text(
            json.dumps(plat_tourn, indent=2)
        )

    overall = prod_result["overall"]
    t_overall = tourn_result["overall"]
    print(
        f"Period ({args.period_days}d) production: Brier={overall['brier']},"
        f" n={overall['n']} | tournament: Brier={t_overall['brier']},"
        f" n={t_overall['n']}"
    )


def _cli_rebuild(args: argparse.Namespace, output_tournament: Path) -> None:
    """Handle the ``--rebuild`` CLI mode."""
    print(f"Rebuilding scores from {args.logs_dir}")
    result = rebuild(
        logs_dir=args.logs_dir,
        scores_path=args.output,
        history_path=args.history,
        tournament_input=args.tournament_input,
        tournament_scores_path=output_tournament,
    )
    print(
        f"Scores written to {args.output} (production) and "
        f"{output_tournament} (tournament)"
    )
    overall = result["overall"]
    print(
        f"Production: Brier={overall['brier']},"
        f" DirAcc={overall.get('directional_accuracy')}, n={overall['n']}"
    )


def _cli_legacy_full_recompute(
    args: argparse.Namespace, output_tournament: Path
) -> dict[str, Any]:
    """Legacy full-recompute path; returns the production scores dict."""
    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    prod_rows, tourn_rows = _partition_rows_by_mode(rows)
    results = _score_rows_by_platform(prod_rows, tourn_rows)
    result, tourn_result = results["all"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    output_tournament.parent.mkdir(parents=True, exist_ok=True)
    output_tournament.write_text(json.dumps(tourn_result, indent=2))

    for platform in PLATFORMS:
        plat_prod, plat_tourn = results[platform]
        _derive_platform_path(args.output, platform).write_text(
            json.dumps(plat_prod, indent=2)
        )
        _derive_platform_path(output_tournament, platform).write_text(
            json.dumps(plat_tourn, indent=2)
        )

    print(
        f"Scores written to {args.output} (production, n={result['overall']['n']}) "
        f"and {output_tournament} (tournament, n={tourn_result['overall']['n']})"
    )
    return result


def main() -> None:
    """CLI entry point for scoring."""
    args = _build_arg_parser().parse_args()

    output_tournament: Path = args.output_tournament or _derive_tournament_path(
        args.output
    )

    if args.update is not None:
        _cli_update(args, output_tournament)
        return

    if args.period_days is not None:
        _cli_period(args, output_tournament)
        return

    if args.rebuild:
        _cli_rebuild(args, output_tournament)
        return

    result = _cli_legacy_full_recompute(args, output_tournament)

    # Print summary
    overall = result["overall"]
    print(
        f"\nOverall: Brier={overall['brier']}, DirAcc={overall.get('directional_accuracy')},"
        f" Sharpness={overall['sharpness']}, Reliability={overall['reliability']},"
        f" n={overall['n']}"
    )

    if overall["reliability"] is not None and overall["reliability"] < RELIABILITY_GATE:
        print(
            f"WARNING: Reliability {overall['reliability']} is below {RELIABILITY_GATE} gate"
        )

    print("\nBy tool (decision-worthy):")
    ranked = sorted(result["by_tool"].items(), key=brier_sort_key)
    for tool, stats in ranked:
        flags = []
        if stats["reliability"] is not None and stats["reliability"] < RELIABILITY_GATE:
            flags.append("UNRELIABLE")
        if not stats["decision_worthy"]:
            flags.append(f"LOW-SAMPLE<{MIN_SAMPLE_SIZE}")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"  {tool}: Brier={stats['brier']}, DirAcc={stats.get('directional_accuracy')}, Sharp={stats['sharpness']}, n={stats['n']}{suffix}"
        )

    print("\nBy platform:")
    for platform, stats in result["by_platform"].items():
        print(f"  {platform}: Brier={stats['brier']}, n={stats['n']}")

    print("\nBy tool × platform:")
    for key, stats in sorted(
        result["by_tool_platform"].items(),
        key=brier_sort_key,
    ):
        print(f"  {key}: Brier={stats['brier']}, n={stats['n']}")

    print("\nBy tool × category:")
    for key, stats in sorted(
        result["by_tool_category"].items(),
        key=brier_sort_key,
    ):
        print(f"  {key}: Brier={stats['brier']}, n={stats['n']}")

    print("\nCalibration (overall):")
    print(f"  {'Bin':<10} {'Predicted':>10} {'Realized':>10} {'Gap':>8} {'n':>6}")
    for b in result["calibration"]:
        direction = "over" if b["gap"] > 0 else "under" if b["gap"] < 0 else ""
        print(
            f"  {b['bin']:<10} {b['avg_predicted']:>10.4f} {b['realized_rate']:>10.4f}"
            f" {b['gap']:>+8.4f} {b['n']:>6}  {direction}"
        )


if __name__ == "__main__":
    main()
