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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.io import load_jsonl as load_rows

DEFAULT_INPUT = Path(__file__).parent / "datasets" / "production_log.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "scores.json"
DEFAULT_HISTORY = Path(__file__).parent / "results" / "scores_history.jsonl"
DEFAULT_LOGS_DIR = Path(__file__).parent / "datasets" / "logs"

LATENCY_RESERVOIR_SIZE = 200
WORST_BEST_SIZE = 10

RELIABILITY_GATE = 0.80
MIN_SAMPLE_SIZE = 30


# ---------------------------------------------------------------------------
# Brier score computation
# ---------------------------------------------------------------------------


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


def _is_edge_eligible(row: dict[str, Any]) -> bool:
    """Check if a row has all fields needed for edge-over-market calculation."""
    return (
        row.get("prediction_parse_status") == "valid"
        and row.get("final_outcome") is not None
        and row.get("p_yes") is not None
        and row.get("market_prob_at_prediction") is not None
    )


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
    distance = abs(market_prob - 0.5)
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


def compute_group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Brier score and reliability for a group of rows.

    All rows count toward reliability. Only valid rows with
    final_outcome count toward Brier.

    :param rows: list of prediction row dicts.
    :return: dict with brier, accuracy, sharpness, reliability, n, valid_n,
        decision_worthy.
    """
    total = len(rows)
    if total == 0:
        return {
            "brier": None,
            "accuracy": None,
            "sharpness": None,
            "reliability": None,
            "n": 0,
            "valid_n": 0,
            "decision_worthy": False,
        }

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
        return {
            "brier": None,
            "accuracy": None,
            "sharpness": None,
            "reliability": round(reliability, 4),
            "n": total,
            "valid_n": 0,
            "decision_worthy": False,
        }

    scores = [brier_score(r["p_yes"], r["final_outcome"]) for r in valid]
    avg_brier = sum(scores) / len(scores)

    # p_yes == 0.5 counted as incorrect (no directional signal)
    correct = sum(1 for r in valid if (r["p_yes"] > 0.5) == r["final_outcome"])
    accuracy = correct / len(valid)

    sharpness = sum(abs(r["p_yes"] - 0.5) for r in valid) / len(valid)

    # Edge over market — only for rows with market_prob_at_prediction
    edge_rows = [r for r in valid if r.get("market_prob_at_prediction") is not None]
    if edge_rows:
        edges = [
            edge_score(r["p_yes"], r["market_prob_at_prediction"], r["final_outcome"])
            for r in edge_rows
        ]
        edge_avg = round(sum(edges) / len(edges), 4)
        edge_positive = sum(1 for e in edges if e > 0)
        edge_pos_rate = round(edge_positive / len(edges), 4)
    else:
        edge_avg = None
        edge_pos_rate = None

    return {
        "brier": round(avg_brier, 4),
        "accuracy": round(accuracy, 4),
        "sharpness": round(sharpness, 4),
        "reliability": round(reliability, 4),
        "n": total,
        "valid_n": len(valid),
        "decision_worthy": worthy,
        "edge": edge_avg,
        "edge_n": len(edge_rows),
        "edge_positive_rate": edge_pos_rate,
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
        groups[row.get(key, "unknown")].append(row)
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
    return " | ".join(str(row.get(f, "unknown")) for f in fields)


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


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------


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
        plat = row.get("platform", "unknown")
        diff = classify_difficulty(row.get("market_prob_at_prediction"))
        pd_groups[f"{plat} | {diff}"].append(row)
    by_platform_difficulty = {k: compute_group_stats(g) for k, g in pd_groups.items()}

    # Platform × liquidity cross breakdown
    pl_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        plat = row.get("platform", "unknown")
        liq = classify_liquidity(row.get("market_liquidity_at_prediction"))
        pl_groups[f"{plat} | {liq}"].append(row)
    by_platform_liquidity = {k: compute_group_stats(g) for k, g in pl_groups.items()}

    # Tool × platform cross breakdown
    by_tool_platform = group_by_composite(rows, ["tool_name", "platform"])

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
    calibration_by_tool = {
        tool: compute_calibration(group)
        for tool, group in group_by(rows, "tool_name").items()
    }

    # Edge eligibility reporting — mutually exclusive exclusion reasons
    n_edge_eligible = sum(1 for r in rows if _is_edge_eligible(r))
    n_invalid_parse = sum(
        1 for r in rows if r.get("prediction_parse_status") != "valid"
    )
    n_missing_outcome = sum(
        1
        for r in rows
        if r.get("prediction_parse_status") == "valid"
        and r.get("final_outcome") is None
    )
    n_missing_p_yes = sum(
        1
        for r in rows
        if r.get("prediction_parse_status") == "valid"
        and r.get("final_outcome") is not None
        and r.get("p_yes") is None
    )
    n_missing_market_prob = sum(
        1
        for r in rows
        if r.get("prediction_parse_status") == "valid"
        and r.get("final_outcome") is not None
        and r.get("p_yes") is not None
        and r.get("market_prob_at_prediction") is None
    )
    edge_eligibility = {
        "n_total": total,
        "n_eligible": n_edge_eligible,
        "n_excluded": total - n_edge_eligible,
        "exclusion_reasons": {
            "invalid_parse": n_invalid_parse,
            "missing_outcome": n_missing_outcome,
            "missing_p_yes": n_missing_p_yes,
            "missing_market_prob": n_missing_market_prob,
        },
    }

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
        "by_tool_platform_horizon": by_tool_platform_horizon,
        "trend": trend,
        "calibration": calibration,
        "calibration_by_tool": calibration_by_tool,
        "edge_eligibility": edge_eligibility,
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
        "sharpness_sum": 0.0,
        "outcome_yes_count": 0,
        "edge_sum": 0.0,
        "edge_n": 0,
        "edge_positive_count": 0,
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
        "by_tool_version": {},
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
        "scored_row_ids": set(),
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
        if (p_yes > 0.5) == outcome:
            group["correct_count"] += 1
        group["sharpness_sum"] += abs(p_yes - 0.5)
        if outcome:
            group["outcome_yes_count"] += 1
        # Edge over market
        market_prob = row.get("market_prob_at_prediction")
        if market_prob is not None:
            edge = edge_score(p_yes, market_prob, outcome)
            group["edge_sum"] += edge
            group["edge_n"] += 1
            if edge > 0:
                group["edge_positive_count"] += 1


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
            accuracy=None,
            sharpness=None,
            reliability=None,
            decision_worthy=False,
        )
    elif valid_n == 0:
        result.update(
            brier=None,
            accuracy=None,
            sharpness=None,
            reliability=round(0.0, 4),
            decision_worthy=False,
        )
    else:
        brier = round(group["brier_sum"] / valid_n, 4)
        yes_rate = group["outcome_yes_count"] / valid_n
        baseline_brier = round(yes_rate * (1 - yes_rate), 4)
        result["brier"] = brier
        result["accuracy"] = round(group["correct_count"] / valid_n, 4)
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
    return result


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


def _accumulate_row(scores: dict[str, Any], row: dict[str, Any]) -> None:
    """Merge one row into all accumulator dimensions (mutates *scores*).

    :param scores: the full scores dict with accumulators.
    :param row: a production log row dict.
    """
    _accumulate_group(scores["overall"], row)

    tool = row.get("tool_name", "unknown")
    platform = row.get("platform", "unknown")
    category = row.get("category", "unknown")
    horizon = classify_horizon(row.get("prediction_lead_time_days"))
    tool_version = row.get("tool_version") or "unknown"
    config_hash = row.get("config_hash") or "unknown"

    _ensure_and_accumulate(scores["by_tool"], tool, row)
    _ensure_and_accumulate(scores["by_platform"], platform, row)
    _ensure_and_accumulate(scores["by_category"], category, row)
    _ensure_and_accumulate(scores["by_horizon"], horizon, row)
    _ensure_and_accumulate(scores["by_tool_platform"], f"{tool} | {platform}", row)
    _ensure_and_accumulate(scores["by_tool_version"], f"{tool} | {tool_version}", row)
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

    # Calibration buckets
    is_valid = (
        row.get("prediction_parse_status") == "valid"
        and row.get("final_outcome") is not None
        and row.get("p_yes") is not None
    )
    if is_valid:
        p = row["p_yes"]
        for lo, hi in CALIBRATION_BINS:
            if lo <= p < hi:
                label = _bin_label(lo, hi)
                bucket = scores["calibration"][label]
                bucket["count"] += 1
                bucket["outcome_sum"] += 1 if row["final_outcome"] else 0
                bucket["predicted_sum"] += p
                break

    # Parse breakdown
    status = row.get("prediction_parse_status", "unknown")
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
    if is_valid:
        row_brier = brier_score(row["p_yes"], row["final_outcome"])
        entry = {
            "question_text": row.get("question_text", ""),
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
        "by_tool_version",
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

    The saved format includes both derived fields (brier, accuracy) and
    raw accumulators (brier_sum, correct_count). This function extracts
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
        return {
            "n": g["n"],
            "valid_n": g["valid_n"],
            "brier_sum": g["brier_sum"],
            "correct_count": g["correct_count"],
            "sharpness_sum": g["sharpness_sum"],
            "outcome_yes_count": g.get("outcome_yes_count", 0),
            "edge_sum": g.get("edge_sum", 0.0),
            "edge_n": g.get("edge_n", 0),
            "edge_positive_count": g.get("edge_positive_count", 0),
        }

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
        "by_tool_version",
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
    scores["scored_row_ids"] = set(data.get("scored_row_ids", []))

    return scores


def update(
    new_rows: list[dict[str, Any]],
    scores_path: Path = DEFAULT_OUTPUT,
    history_path: Path = DEFAULT_HISTORY,
) -> dict[str, Any]:
    """Incrementally merge new rows into the scores accumulators.

    If ``scores.json`` exists with valid accumulators, loads and extends.
    Otherwise initializes fresh accumulators.

    Handles month boundaries: if the stored ``current_month`` differs from
    today's month, snapshots the completed month to ``scores_history.jsonl``
    and resets accumulators.

    :param new_rows: list of production log row dicts.
    :param scores_path: path to ``scores.json``.
    :param history_path: path to ``scores_history.jsonl``.
    :return: finalized scores dict (also written to disk).
    """
    today_month = datetime.now(timezone.utc).strftime("%Y-%m")

    existing = _load_scores_for_resume(scores_path)
    if existing is not None:
        scores = existing
        # Month boundary check
        if scores["current_month"] != today_month:
            _snapshot_month(scores, history_path)
            scores = _empty_scores(today_month)
    else:
        scores = _empty_scores(today_month)

    scored_ids: set[str] = scores.get("scored_row_ids", set())
    skipped = 0
    for row in new_rows:
        row_id = row.get("row_id")
        if row_id and row_id in scored_ids:
            skipped += 1
            continue
        _accumulate_row(scores, row)
        if row_id:
            scored_ids.add(row_id)
    scores["scored_row_ids"] = scored_ids

    if skipped:
        logging.getLogger(__name__).warning(
            "Skipped %d duplicate rows (already scored)", skipped
        )

    # Write raw accumulators (for future incremental loads)
    finalized = _finalize_scores(scores)
    # Merge accumulators into the output so next load can resume
    output = dict(finalized)
    _accum_keys = (
        "brier_sum",
        "correct_count",
        "sharpness_sum",
        "outcome_yes_count",
        "edge_sum",
        "edge_n",
        "edge_positive_count",
    )
    output["overall"] = {
        **finalized["overall"],
        **{k: scores["overall"][k] for k in _accum_keys},
    }
    for dim in (
        "by_tool",
        "by_platform",
        "by_category",
        "by_horizon",
        "by_tool_platform",
        "by_tool_version",
        "by_config",
        "by_difficulty",
        "by_liquidity",
        "by_platform_difficulty",
        "by_platform_liquidity",
    ):
        for key, group in scores[dim].items():
            output[dim][key] = {
                **finalized[dim][key],
                **{k: group[k] for k in _accum_keys},
            }
    # Preserve raw calibration accumulators alongside derived
    output["_calibration_accum"] = scores["calibration"]
    output["scored_row_ids"] = sorted(scores.get("scored_row_ids", set()))

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(output, indent=2))

    return finalized


def rebuild(
    logs_dir: Path = DEFAULT_LOGS_DIR,
    scores_path: Path = DEFAULT_OUTPUT,
    history_path: Path = DEFAULT_HISTORY,
) -> dict[str, Any]:
    """Rebuild scores.json from all log files in the logs directory.

    Reads all ``production_log_*.jsonl`` files (including legacy), processes
    rows month by month, and writes snapshots + final scores.

    :param logs_dir: directory containing daily log files.
    :param scores_path: output path for ``scores.json``.
    :param history_path: output path for ``scores_history.jsonl``.
    :return: finalized scores dict.
    """

    # Collect all log files
    pattern = str(logs_dir / "production_log_*.jsonl")
    files = sorted(glob_mod.glob(pattern))

    all_rows: list[dict[str, Any]] = []
    for filepath in files:
        all_rows.extend(load_rows(Path(filepath)))

    if not all_rows:
        scores = _empty_scores(datetime.now(timezone.utc).strftime("%Y-%m"))
        finalized = _finalize_scores(scores)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_path.write_text(json.dumps(finalized, indent=2))
        return finalized

    # Sort rows by predicted_at to process chronologically
    all_rows.sort(key=lambda r: r.get("predicted_at") or "")

    # Group by month and process
    months: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        predicted_at = row.get("predicted_at")
        if predicted_at:
            month = predicted_at[:7]
        else:
            month = "unknown"
        months[month].append(row)

    # Clear history file for rebuild
    if history_path.exists():
        history_path.unlink()

    sorted_months = sorted(months.keys())
    last_month = sorted_months[-1]

    # Collect all row_ids across all months for dedup on subsequent update() calls
    all_row_ids: set[str] = set()

    # Process all months except the last one as snapshots
    for month in sorted_months[:-1]:
        scores = _empty_scores(month)
        for row in months[month]:
            _accumulate_row(scores, row)
            row_id = row.get("row_id")
            if row_id:
                all_row_ids.add(row_id)
        _snapshot_month(scores, history_path)

    # The last month becomes the current scores.json
    scores = _empty_scores(last_month)
    for row in months[last_month]:
        _accumulate_row(scores, row)
        row_id = row.get("row_id")
        if row_id:
            all_row_ids.add(row_id)
    scores["scored_row_ids"] = all_row_ids

    finalized = _finalize_scores(scores)
    # Write with accumulators for future incremental use
    output = dict(finalized)
    _accum_keys = (
        "brier_sum",
        "correct_count",
        "sharpness_sum",
        "outcome_yes_count",
        "edge_sum",
        "edge_n",
        "edge_positive_count",
    )
    output["overall"] = {
        **finalized["overall"],
        **{k: scores["overall"][k] for k in _accum_keys},
    }
    for dim in (
        "by_tool",
        "by_platform",
        "by_category",
        "by_horizon",
        "by_tool_platform",
        "by_tool_version",
        "by_config",
        "by_difficulty",
        "by_liquidity",
        "by_platform_difficulty",
        "by_platform_liquidity",
    ):
        for key, group in scores[dim].items():
            output[dim][key] = {
                **finalized[dim][key],
                **{k: group[k] for k in _accum_keys},
            }
    output["_calibration_accum"] = scores["calibration"]
    output["scored_row_ids"] = sorted(scores.get("scored_row_ids", set()))

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(output, indent=2))

    return finalized


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for scoring."""
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
        help="Output JSON file path",
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
    args = parser.parse_args()

    if args.rebuild:
        print(f"Rebuilding scores from {args.logs_dir}")
        result = rebuild(
            logs_dir=args.logs_dir,
            scores_path=args.output,
            history_path=args.history,
        )
        print(f"Scores written to {args.output}")
        overall = result["overall"]
        print(
            f"Overall: Brier={overall['brier']}, Accuracy={overall['accuracy']},"
            f" n={overall['n']}"
        )
        return

    # Legacy full-recompute mode
    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    result = score(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Scores written to {args.output}")

    # Print summary
    overall = result["overall"]
    print(
        f"\nOverall: Brier={overall['brier']}, Accuracy={overall['accuracy']},"
        f" Sharpness={overall['sharpness']}, Reliability={overall['reliability']},"
        f" n={overall['n']}"
    )

    if overall["reliability"] is not None and overall["reliability"] < RELIABILITY_GATE:
        print(
            f"WARNING: Reliability {overall['reliability']} is below {RELIABILITY_GATE} gate"
        )

    print("\nBy tool (decision-worthy):")
    ranked = sorted(result["by_tool"].items(), key=lambda x: x[1].get("brier") or 999)
    for tool, stats in ranked:
        flags = []
        if stats["reliability"] is not None and stats["reliability"] < RELIABILITY_GATE:
            flags.append("UNRELIABLE")
        if not stats["decision_worthy"]:
            flags.append(f"LOW-SAMPLE<{MIN_SAMPLE_SIZE}")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"  {tool}: Brier={stats['brier']}, Acc={stats['accuracy']}, Sharp={stats['sharpness']}, n={stats['n']}{suffix}"
        )

    print("\nBy platform:")
    for platform, stats in result["by_platform"].items():
        print(f"  {platform}: Brier={stats['brier']}, n={stats['n']}")

    print("\nBy tool × platform:")
    for key, stats in sorted(
        result["by_tool_platform"].items(),
        key=lambda x: x[1].get("brier") or 999,
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
