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
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

# isort treats `benchmark` as third-party (not in known_first_party=autonomy);
# pylint treats it as first-party. The two views disagree on import order.
# isort wins; silence pylint's complaint about the resulting block.
from benchmark.grouping import (  # pylint: disable=wrong-import-order
    _derive_group,
    accumulate_row,
)
from benchmark.io import load_jsonl as load_rows
from benchmark.scoring_primitives import (
    CALIBRATION_BINS,
    LATENCY_RESERVOIR_SIZE,
    MIN_SAMPLE_SIZE,
    WORST_BEST_SIZE,
    _bin_label,
    _empty_group,
    brier_score,
    classify_difficulty,
    classify_disagreement,
    classify_horizon,
    classify_liquidity,
    disagree_bucket,
    edge_score,
    log_loss_score,
)
from scipy.optimize import (  # type: ignore[import-untyped]  # pylint: disable=wrong-import-order
    minimize,
)

DEFAULT_INPUT = Path(__file__).parent / "datasets" / "production_log.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "scores.json"
DEFAULT_OUTPUT_TOURNAMENT = Path(__file__).parent / "results" / "scores_tournament.json"
DEFAULT_HISTORY = Path(__file__).parent / "results" / "scores_history.jsonl"
DEFAULT_DEDUP = Path(__file__).parent / "results" / "scored_row_ids.json"
DEFAULT_LOGS_DIR = Path(__file__).parent / "datasets" / "logs"

# Data-source markers written into ``scores.json``. The legacy incremental
# path (``update``) and legacy log-based ``rebuild`` stamp
# ``SOURCE_LEGACY_LOGS``. The mech-analytics-fed rebuild stamps
# ``SOURCE_MECH_ANALYTICS``. ``update()`` reads this on entry so a
# ``USE_MECH_ANALYTICS_ROWS`` rollback can detect a source flip and rebuild
# from ``logs/`` before merging incremental rows — the two paths dedup on
# disjoint key namespaces (``request_id`` vs ``platform:deliver_id``), so
# without the rebuild every row inside the fetch lookback double-counts.
SOURCE_LEGACY_LOGS = "legacy_logs"
SOURCE_MECH_ANALYTICS = "mech_analytics"

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
    production — matches the historical default in ``accumulate_row``.

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


RELIABILITY_GATE = 0.80
MIN_CALIBRATION_BIN_SIZE = 20

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
    "market_brier_sum",
    "edge_sq_sum",
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


def _is_edge_eligible(row: dict[str, Any]) -> bool:
    """Check if a row has all fields needed for edge-over-market calculation."""
    return (
        row.get("prediction_parse_status") == "valid"
        and row.get("final_outcome") is not None
        and row.get("p_yes") is not None
        and row.get("market_prob_at_prediction") is not None
    )


_DIAGNOSTIC_NONE: dict[str, Any] = {
    "edge": None,
    "market_brier": None,
    "edge_sd": None,
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
    # See grouping.accumulate_row: the market's Brier is averaged over the
    # edge-eligible rows, NOT reconstructed from the tool's all-valid-row
    # ``brier`` plus ``edge`` -- those two average over different pools.
    market_briers = [
        (r["market_prob_at_prediction"] - (1.0 if r["final_outcome"] else 0.0)) ** 2
        for r in edge_rows
    ]
    market_brier_avg = round(sum(market_briers) / len(market_briers), 4)
    mean_edge = sum(edges) / len(edges)
    variance = sum(e * e for e in edges) / len(edges) - mean_edge * mean_edge
    edge_sd = round(math.sqrt(variance), 6) if variance > 0 else 0.0
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
        "market_brier": market_brier_avg,
        "edge_sd": edge_sd if len(edges) >= 2 else None,
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


def score(  # pylint: disable=too-many-statements,too-many-locals
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
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

    # Category × platform cross breakdown
    cp_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cat = row.get("category") or "unknown"
        plat = row.get("platform") or "unknown"
        cp_groups[f"{cat} | {plat}"].append(row)
    by_category_platform = {k: compute_group_stats(g) for k, g in cp_groups.items()}

    # Tool × category × platform cross breakdown
    by_tool_category_platform = group_by_composite(
        rows, ["tool_name", "category", "platform"]
    )

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

    # Observed bounds, so the rendered report can name the span it is showing
    # instead of inferring one from a filename or a stale month stamp.
    stamps = sorted(r["predicted_at"] for r in rows if r.get("predicted_at"))

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_start": stamps[0] if stamps else None,
        "window_end": stamps[-1] if stamps else None,
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
        "by_category_platform": by_category_platform,
        "by_tool_category_platform": by_tool_category_platform,
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
        "by_category_platform": {},
        "by_tool_category_platform": {},
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
    # Carried through so the rendered report can name the span it is showing.
    for bound in ("window_start", "window_end"):
        if scores.get(bound):
            result[bound] = scores[bound]
    result["valid_rows"] = scores["overall"]["valid_n"]

    for dim in (
        "by_tool",
        "by_platform",
        "by_category",
        "by_horizon",
        "by_tool_platform",
        "by_tool_category",
        "by_category_platform",
        "by_tool_category_platform",
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


def _upsert_month_snapshot(
    scores: dict[str, Any],
    month: str,
    history_path: Path,
) -> None:
    """Replace the snapshot row for ``month`` in-place, preserving other months.

    The mech-analytics-fed rebuild fetches a widened trailing window
    each night and re-derives every month in it from the current row
    set. ``_snapshot_month`` unconditionally appends, so a naive port
    would either double-write months on every rebuild or need to unlink
    ``scores_history.jsonl`` (blowing away months older than the fetch
    window). Upserting by ``month`` keeps older months untouched.

    :param scores: accumulator state for ``month`` (will be finalized).
    :param month: ``YYYY-MM`` key to replace.
    :param history_path: path to ``scores_history.jsonl``.
    """
    finalized = _finalize_scores(scores)
    new_entry = {
        "month": month,
        "overall": finalized["overall"],
        "by_tool": finalized["by_tool"],
        "by_platform": finalized["by_platform"],
        "calibration": finalized["calibration"],
    }
    kept: list[dict[str, Any]] = []
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                # Fail loud on malformed history — silently dropping the
                # line here would remove it from the rewrite permanently
                # (the only place the file's contents live is on disk).
                entry = json.loads(stripped)
                if entry.get("month") != month:
                    kept.append(entry)
    kept.append(new_entry)
    kept.sort(key=lambda e: e.get("month") or "")
    # Atomic rewrite: write to a sibling tmp file and rename. os.replace is
    # atomic on POSIX, so a crash or cancelled job mid-write leaves the
    # original file intact rather than truncating everything to zero.
    history_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = history_path.with_suffix(history_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in kept:
            f.write(json.dumps(entry) + "\n")
    os.replace(tmp_path, history_path)


def _load_scores_for_resume(scores_path: Path) -> dict[str, Any] | None:
    """Load scores.json and reconstruct the raw accumulator state.

    The saved format includes both derived fields (brier, directional_accuracy)
    and raw accumulators (brier_sum, correct_count). This function extracts
    the raw accumulators into the internal format used by ``accumulate_row``.

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
        # None, NOT 0.0, when resuming from a scores.json written before this
        # field existed: edge_n is restored in full, so a 0.0 sum would derive
        # market_brier ~= 0 and read as "the market is nearly perfect" forever
        # (the incremental path is the normal daily path; --rebuild is the
        # exception). None makes _derive_group emit no value until a rebuild.
        restored["market_brier_sum"] = g.get("market_brier_sum")
        restored["edge_sq_sum"] = g.get("edge_sq_sum")
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
        "by_category_platform",
        "by_tool_category_platform",
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
    source: str = SOURCE_LEGACY_LOGS,
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
    :param source: fallback data-source marker. Used only when the file
        does not yet exist or predates the marker; otherwise the
        existing marker is preserved so a downstream ``update()`` call
        (tournament merge, etc.) cannot disarm the flag-rollback
        detection by silently rewriting a ``mech_analytics`` file with
        the ``legacy_logs`` default.
    :return: finalized scores dict (also written to disk).
    """
    today_month = datetime.now(timezone.utc).strftime("%Y-%m")

    existing_source = _read_scores_source(scores_path)
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
        accumulate_row(scores, row)

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
        "by_category_platform",
        "by_tool_category_platform",
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
    output["source"] = existing_source if existing_source is not None else source

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(output, indent=2))

    return finalized


def _read_scores_source(scores_path: Path) -> str | None:
    """Read the ``source`` marker from an existing scores file.

    Returns ``None`` when the file is missing, malformed, or predates
    the marker (all treated the same: no rollback signal available).

    :param scores_path: path to ``scores.json``.
    :return: source marker string, or ``None``.
    """
    if not scores_path.exists():
        return None
    try:
        data = json.loads(scores_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    source = data.get("source")
    return source if isinstance(source, str) else None


def _maybe_rebuild_after_source_flip(
    scores_path: Path,
    history_path: Path,
    dedup_path: Path,
    tournament_scores_path: Path,
) -> None:
    """Detect a rollback from a non-legacy source and rebuild from logs.

    Reads the ``source`` marker on ``scores_path``. Rebuilds only when
    the marker is non-legacy AND the current run is flag-off — that is
    the actual rollback shape. On a flag-on night the marker is
    ``SOURCE_MECH_ANALYTICS`` by design (the ``Score`` step just
    stamped it) and the tournament-merge step's downstream
    ``scorer --update`` call would otherwise read the same marker as a
    rollback, wipe the just-built MTD accumulator, rebuild from a
    frozen ``logs/``, and stamp ``SOURCE_LEGACY_LOGS`` on disk —
    silently disarming the safety net for a real future rollback.

    When triggered, wipes ``scores.json``, its per-platform siblings,
    and ``scored_row_ids.json``, then invokes :func:`rebuild` to
    repopulate all-time state from ``logs/``. Downstream ``update()``
    then merges the incoming ``new_rows`` on top of a matching dedup
    namespace and cannot double-count.

    :param scores_path: path to production ``scores.json``.
    :param history_path: path to ``scores_history.jsonl``.
    :param dedup_path: path to ``scored_row_ids.json``.
    :param tournament_scores_path: path to ``scores_tournament.json``.
    """
    source = _read_scores_source(scores_path)
    if source is None or source == SOURCE_LEGACY_LOGS:
        return
    if _use_mech_analytics_rows():
        return

    log = logging.getLogger(__name__)
    log.warning(
        "scores.json carries source=%r (non-legacy); wiping and "
        "rebuilding from %s so the request_id / platform:deliver_id "
        "dedup namespaces stop overlapping.",
        source,
        DEFAULT_LOGS_DIR,
    )
    scores_path.unlink(missing_ok=True)
    dedup_path.unlink(missing_ok=True)
    for platform in PLATFORMS:
        _derive_platform_path(scores_path, platform).unlink(missing_ok=True)
    rebuild(
        logs_dir=DEFAULT_LOGS_DIR,
        scores_path=scores_path,
        history_path=history_path,
        dedup_path=dedup_path,
        tournament_scores_path=tournament_scores_path,
    )


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

    # Detect a rollback from the mech-analytics path. That path keys the
    # accumulator on ``request_id``; the legacy incremental path here
    # keys on ``platform:deliver_id``. The two dedup namespaces never
    # collide, so if we resumed the mech-analytics-stamped file, every
    # row inside the fetch lookback would enter ``update()`` as "new"
    # and be counted a second time on top of the already-baked-in
    # request_id-keyed totals. Wipe scores + dedup and rebuild from
    # ``logs/`` once so subsequent incremental merges are safe.
    _maybe_rebuild_after_source_flip(
        scores_path, history_path, dedup_path, tournament_scores_path
    )

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
        finalized["source"] = SOURCE_LEGACY_LOGS
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
            accumulate_row(scores, row)
            if row_id:
                all_row_ids.add(row_id)
        if emit_history and history_path is not None:
            _snapshot_month(scores, history_path)

    scores = _empty_scores(last_month)
    for row in months[last_month]:
        row_id = row.get("row_id")
        if row_id and row_id in all_row_ids:
            continue
        accumulate_row(scores, row)
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
        "by_category_platform",
        "by_tool_category_platform",
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
    output["source"] = SOURCE_LEGACY_LOGS

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
        finalized["source"] = SOURCE_LEGACY_LOGS
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


def _resolution_lag_days() -> int:
    """Days of extra trailing window to fetch beyond ``since``.

    The endpoint filters ``since`` / ``until`` on ``requested_at`` (the
    mech request timestamp, source: ``mech-analytics`` handler docstring
    ``etl/api/routes/data.py``). Predictions late in month M that only
    resolve early in month M+1 fail the ``resolved=True`` filter on M's
    last run and fall out of window on M+1's first run. Fetching a lag
    tail past ``since`` and re-deriving every month it covers plugs
    that hole.

    Overridable via ``MECH_ANALYTICS_RESOLUTION_LAG_DAYS`` for staging
    smoke tests that want a shorter window. Default 90 days covers Omen
    / Polymarket's tail resolution latency comfortably.

    :return: lag window in days (positive integer; clamped to >=0).
    """
    raw = os.getenv("MECH_ANALYTICS_RESOLUTION_LAG_DAYS", "")
    try:
        value = int(raw)
    except ValueError:
        return 90
    return max(value, 0)


def rebuild_from_mech_analytics(
    since: datetime,
    until: datetime | None = None,
    scores_path: Path = DEFAULT_OUTPUT,
    history_path: Path = DEFAULT_HISTORY,
    chain_id: int | None = None,
) -> dict[str, Any]:
    """Rebuild scores from mech-analytics's ``/v1/data/scored-rows`` endpoint.

    Post off-chain migration the marketplace-subgraph + IPFS row-fetch layer
    can't reach individual mech requests. mech-analytics is the new public
    read path — it ingests from the predict-api data lake, scores each row
    once, and serves the results. This function is the mech-analytics-fed
    counterpart to :func:`rebuild`: same output shape, same
    ``scores_<platform>.json`` files, same downstream consumers
    (``analyze.py`` needs no changes).

    Gated by the ``USE_MECH_ANALYTICS_ROWS`` env var at the CLI layer;
    default off so nothing changes until we flip.

    **Timestamp windowing.** The endpoint applies ``since`` / ``until``
    to ``requested_at``. Rows are bucketed into months here by the same
    ``requested_at`` field so per-month attribution matches what the
    endpoint served. To close the resolution-lag hole (predictions late
    in month M that resolve early in M+1), the effective fetch window
    is ``[since - _resolution_lag_days(), until]`` and every prior
    month covered by that window is re-derived from the current
    response and upserted into ``scores_history.jsonl`` on each run.
    Only ``since``'s calendar month lands in ``scores.json`` (the live
    accumulator). Months older than the lag window are preserved.

    :param since: timezone-aware datetime for the current-month anchor.
        The fetch reaches back beyond this by
        ``_resolution_lag_days()`` internally.
    :param until: timezone-aware datetime, exclusive upper bound. When
        ``None``, the endpoint returns up to now.
    :param scores_path: output path for the production ``scores.json``.
    :param history_path: output path for ``scores_history.jsonl``.
    :param chain_id: optional chain filter (100 = Gnosis, 137 = Polygon).
        ``None`` fetches every chain the endpoint serves.
    :return: finalized production scores dict.
    :raises MechAnalyticsError: when ``scores_history.jsonl`` is missing
        and the ``MECH_ANALYTICS_ALLOW_EMPTY_HISTORY`` opt-in is not set.
        The rebuild only fetches a trailing window, so a missing history
        file would silently start us from scratch and drop prior months'
        trend rows on the floor.
    """
    # Local import so a missing ``requests`` install can't break the module
    # at import time on the sweep / tournament paths that don't use it.
    # pylint: disable=import-outside-toplevel
    import importlib

    from benchmark.mech_analytics_client import iter_scored_rows

    # ``classify_category`` lives in ``fetch_production``, which lazily
    # imports ``scorer.update`` on its incremental path — a static ``from``
    # here would surface as a cyclic-import warning even though both edges
    # resolve inside functions. Route through ``importlib`` so pylint's
    # static tracer doesn't count this edge.
    classify_category = importlib.import_module(
        "benchmark.datasets.fetch_production"
    ).classify_category

    # pylint: disable=import-outside-toplevel
    from benchmark.mech_analytics_client import MechAnalyticsError

    # History guard. Refuse whenever ``scores_history.jsonl`` is missing,
    # regardless of whether the current-month accumulator exists — the
    # CI ``Clear cached scores`` step wipes both files together, and the
    # narrower ``and scores_path.exists()`` version of this check let
    # that wipe through undetected (prior months' trend rows silently
    # gone). Truly-fresh setups opt in via
    # ``MECH_ANALYTICS_ALLOW_EMPTY_HISTORY=true`` so the guard is a
    # deliberate override, not a silent skip.
    # pylint: disable=import-outside-toplevel
    from benchmark.scoring_primitives import parse_truthy_env

    allow_empty_history = parse_truthy_env("MECH_ANALYTICS_ALLOW_EMPTY_HISTORY")
    if not history_path.exists() and not allow_empty_history:
        raise MechAnalyticsError(
            f"scores_history.jsonl missing at {history_path}; the "
            "mech-analytics rebuild only fetches a trailing window and "
            "cannot repopulate historical snapshots on its own. Seed "
            "history via the legacy rebuild path once, or set "
            "MECH_ANALYTICS_ALLOW_EMPTY_HISTORY=true to accept starting "
            "from an empty history."
        )

    lag_days = _resolution_lag_days()
    # Snap the widened lower bound to the 1st of its calendar month so the
    # oldest month bucket is fully covered. Without this,
    # ``since - lag_days`` lands mid-month and the oldest bucket only
    # holds day-N-through-31 rows; the subsequent upsert would then
    # replace the previously-complete history snapshot for that month
    # with a truncated re-derivation, and every month permanently loses
    # its first ~0-30 days the moment it slides to the trailing edge of
    # the lag window. Snapping costs at most 30 extra days of fetch;
    # ``since`` is already a tz-aware month start so replace(day=1)
    # keeps the 00:00 UTC anchor.
    effective_since = (since - timedelta(days=lag_days)).replace(day=1)
    current_month = since.strftime("%Y-%m")

    # Bucket rows by ``requested_at`` month so late-resolving predictions
    # from prior months land back in the month they were requested in,
    # not silently in the current-month accumulator (which is what a
    # single flat pass into ``_accumulate_and_write`` would do).
    all_rows: list[dict[str, Any]] = []
    prior_month_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current_month_rows: list[dict[str, Any]] = []
    for row in iter_scored_rows(
        since=effective_since, until=until, chain_id=chain_id, resolved=True
    ):
        # ``category`` is derived locally from ``question_text`` — the
        # endpoint carries the title but not the classified category, and
        # ``accumulate_row`` groups on this key.
        question_text = row.get("question_text")
        platform = row.get("platform")
        if question_text:
            row["category"] = classify_category(question_text, platform)
        request_id = row.get("request_id")
        if not request_id:
            raise MechAnalyticsError(f"mech-analytics row missing request_id: {row!r}")
        row["row_id"] = row.get("row_id") or request_id
        all_rows.append(row)

        requested_at = row.get("requested_at") or ""
        row_month = requested_at[:7] if requested_at else current_month
        if row_month == current_month:
            current_month_rows.append(row)
        elif row_month < current_month:
            prior_month_rows[row_month].append(row)
        # Rows dated after the current-month anchor (clock skew or ingest
        # of pre-scheduled markets) fold into the current bucket. Losing
        # them would be worse than accepting a small dating drift.
        else:
            current_month_rows.append(row)

    # No dedup filter on this path. ``iter_scored_rows`` returns the full
    # window every rebuild, so filtering against a persisted
    # ``scored_row_ids.json`` would drop rows that legitimately appear in
    # later runs' windows too. Combined with the ``scores_path.unlink``
    # below (fresh accumulator per rebuild), the earlier "read the dedup
    # set, skip seen rows, save the union" logic collapsed the
    # accumulator to a one-day slice from the second night onward.
    # Full-window refetch is idempotent by construction; the dedup file
    # is owned by the legacy ``--update`` incremental path and is
    # neither read nor written here.

    # A "rebuild" must start from _empty_scores, not silently merge into
    # an existing accumulator. _accumulate_and_write resumes when the
    # current_month matches, so unlink the aggregate files up front to
    # force the fresh path. history_path is deliberately preserved —
    # only scores_path (this month's live accumulator) gets wiped.
    scores_path.unlink(missing_ok=True)
    for platform in ("omen", "polymarket"):
        _derive_platform_path(scores_path, platform).unlink(missing_ok=True)

    # mech-analytics has no tournament partition — leave existing
    # scores_tournament*.json files untouched. Overwriting them with
    # empty would clobber accumulated tournament state on every rebuild,
    # and the subsequent workflow ``--update`` step would merge only
    # that run's fresh tournament rows onto the emptied file.
    prod_result = _accumulate_and_write(
        current_month_rows,
        scores_path,
        history_path,
        emit_history=False,
        source=SOURCE_MECH_ANALYTICS,
    )

    for platform, plat_rows in _partition_rows_by_platform(current_month_rows).items():
        _accumulate_and_write(
            plat_rows,
            _derive_platform_path(scores_path, platform),
            None,
            emit_history=False,
            source=SOURCE_MECH_ANALYTICS,
        )

    # Re-derive every prior month covered by the widened fetch window
    # from that month's row bucket and upsert the snapshot into
    # scores_history.jsonl. Months older than the lag window are left
    # untouched — the upsert only replaces rows for months present in
    # the current response.
    for month in sorted(prior_month_rows):
        month_scores = _empty_scores(month)
        for row in prior_month_rows[month]:
            accumulate_row(month_scores, row)
        _upsert_month_snapshot(month_scores, month, history_path)

    return prod_result


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
    offset_days: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score a windowed slice of rows, returning (production, tournament).

    Same collection rules as ``score_period`` but returns a tuple so
    callers can write both files separately.

    :param logs_dir: directory containing daily log files.
    :param days: score rows from a window ``days`` days wide.
    :param tournament_input: optional path to ``tournament_scored.jsonl``
        whose rows are filtered to the same window and merged.
    :param offset_days: number of days to shift the window back from "now".
        ``offset_days=days`` selects the immediately-preceding non-overlapping
        window.
    :return: tuple of (production_scores, tournament_scores).
    """
    prod_rows, tourn_rows = _load_period_rows(
        logs_dir, days, tournament_input, offset_days
    )
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
    offset_days: int = 0,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Score a windowed slice of rows, returning combined + per-platform results.

    :param logs_dir: directory containing daily log files.
    :param days: score rows from a window ``days`` days wide.
    :param tournament_input: optional path to ``tournament_scored.jsonl``
        whose rows are filtered to the same window and merged.
    :param offset_days: number of days to shift the window back from "now".
        ``offset_days=days`` selects the immediately-preceding non-overlapping
        window (used to compute prev-rolling scores without overlap).
    :return: ``{"all": (prod, tourn), "omen": (prod, tourn),
        "polymarket": (prod, tourn)}``. Each ``(prod, tourn)`` tuple
        matches the shape of ``score_period_split``. Unknown-platform
        rows stay in ``"all"`` only.
    """
    prod_rows, tourn_rows = _load_period_rows(
        logs_dir, days, tournament_input, offset_days
    )
    return _score_rows_by_platform(prod_rows, tourn_rows)


def _parse_predicted_at(value: Any) -> datetime | None:
    """Parse a row's ``predicted_at`` field into a UTC-aware ``datetime``.

    Naive timestamps (no offset suffix) are assumed UTC so the returned
    value is always comparable against a tz-aware cutoff.

    :param value: raw ``predicted_at`` value from the row dict.
    :return: parsed UTC-aware ``datetime``, or ``None`` when missing or
        unparseable.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _load_period_rows(
    logs_dir: Path,
    days: int,
    tournament_input: Path | None,
    offset_days: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load + filter + mode-partition period rows.

    :param logs_dir: directory containing daily log files.
    :param days: score rows from a window ``days`` days wide.
    :param tournament_input: optional path to ``tournament_scored.jsonl``
        whose rows are filtered to the same window and merged.
    :param offset_days: number of days to shift the window back from "now".
        ``0`` means the trailing window ending at "now" (current window);
        ``days`` means the immediately-preceding non-overlapping window.
    :return: ``(production_rows, tournament_rows)`` both filtered to the
        period window.
    :raises ValueError: when ``days`` or ``offset_days`` is negative.
    """
    if days < 0 or offset_days < 0:
        raise ValueError(
            f"days and offset_days must be >= 0 (got days={days},"
            f" offset_days={offset_days})"
        )
    now = datetime.now(timezone.utc)
    window_end = now - timedelta(days=offset_days)
    window_start = window_end - timedelta(days=days)

    daily_pattern = str(logs_dir / "????-??-??.jsonl")
    prod_pattern = str(logs_dir / "production_log_*.jsonl")
    all_files = sorted(
        set(glob_mod.glob(daily_pattern) + glob_mod.glob(prod_pattern)),
        key=_extract_date_from_log_path,
    )

    def _in_window(predicted_at: datetime | None) -> bool:
        if predicted_at is None:
            return False
        return window_start <= predicted_at < window_end

    rows: list[dict[str, Any]] = []
    for filepath in all_files:
        for row in load_rows(Path(filepath)):
            if _in_window(_parse_predicted_at(row.get("predicted_at"))):
                rows.append(row)

    if tournament_input is not None and tournament_input.exists():
        for row in load_rows(tournament_input):
            if _in_window(_parse_predicted_at(row.get("predicted_at"))):
                rows.append(row)

    return _partition_rows_by_mode(rows)


def score_period(
    logs_dir: Path = DEFAULT_LOGS_DIR,
    days: int = 1,
    tournament_input: Path | None = None,
    offset_days: int = 0,
) -> dict[str, Any]:
    """Score production rows in a windowed slice (backward-compat).

    Backward-compat wrapper around ``score_period_split``; returns the
    production partition only. Callers that need the tournament scores
    should use ``score_period_split`` directly.

    :param logs_dir: directory containing daily log files.
    :param days: score rows from a window ``days`` days wide.
    :param tournament_input: optional path to ``tournament_scored.jsonl``
        whose rows are filtered to the same window and merged.
    :param offset_days: number of days to shift the window back from "now".
    :return: production scores dict.
    """
    prod, _ = score_period_split(logs_dir, days, tournament_input, offset_days)
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
        "--period-offset-days",
        type=int,
        default=0,
        help=(
            "Shift the scoring window back by N days. Combined with "
            "--period-days=N this selects the immediately-preceding "
            "non-overlapping window (used for prev-rolling scores)."
        ),
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
    parser.add_argument(
        "--skip-tournament-output",
        action="store_true",
        help=(
            "Do not write the paired tournament scores file (and its "
            "per-platform splits) in --period-days mode. Use for "
            "rolling-window invocations whose tournament output is not "
            "consumed by the report — keeps results/ free of dead files."
        ),
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
        offset_days=args.period_offset_days,
    )
    prod_result, tourn_result = results["all"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prod_result, indent=2))

    # Rolling-window invocations pass --skip-tournament-output: the
    # report scopes tournament to all-time only, so per-platform rolling
    # tournament files would just be dead output sitting in results/.
    write_tournament = not args.skip_tournament_output
    if write_tournament:
        output_tournament.parent.mkdir(parents=True, exist_ok=True)
        output_tournament.write_text(json.dumps(tourn_result, indent=2))

    for platform in PLATFORMS:
        plat_prod, plat_tourn = results[platform]
        _derive_platform_path(args.output, platform).write_text(
            json.dumps(plat_prod, indent=2)
        )
        if write_tournament:
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
    """Handle the ``--rebuild`` CLI mode.

    When ``USE_MECH_ANALYTICS_ROWS`` is set truthy in the environment,
    reroutes to :func:`rebuild_from_mech_analytics` instead of reading
    log files. Off-chain migration: mech-analytics is the new source of
    the scored-row stream. Defaults off; nothing changes until we flip.

    :param args: parsed CLI namespace with ``output``, ``history``,
        ``logs_dir``, and ``tournament_input``.
    :param output_tournament: derived path for the tournament scores json.
    """
    if _use_mech_analytics_rows():
        # pylint: disable=import-outside-toplevel
        from benchmark.scoring_primitives import start_of_current_month_utc

        since = start_of_current_month_utc()
        print(f"Rebuilding scores from mech-analytics (since {since.isoformat()})")
        result = rebuild_from_mech_analytics(
            since=since,
            scores_path=args.output,
            history_path=args.history,
        )
    else:
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


def _use_mech_analytics_rows() -> bool:
    """Return True when the feature flag routes rebuilds via mech-analytics."""
    # pylint: disable=import-outside-toplevel
    from benchmark.scoring_primitives import use_mech_analytics_rows

    return use_mech_analytics_rows()


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
