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

    return {
        "brier": round(avg_brier, 4),
        "accuracy": round(accuracy, 4),
        "sharpness": round(sharpness, 4),
        "reliability": round(reliability, 4),
        "n": total,
        "valid_n": len(valid),
        "decision_worthy": worthy,
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

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_rows": total,
        "valid_rows": overall["valid_n"],
        "overall": overall,
        "by_tool": by_tool,
        "by_platform": by_platform,
        "by_category": by_category,
        "by_horizon": by_horizon,
        "by_tool_platform": by_tool_platform,
        "by_tool_platform_horizon": by_tool_platform_horizon,
        "trend": trend,
        "calibration": calibration,
        "calibration_by_tool": calibration_by_tool,
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
        "calibration": {
            _bin_label(lo, hi): {"count": 0, "outcome_sum": 0, "predicted_sum": 0.0}
            for lo, hi in CALIBRATION_BINS
        },
        "parse_breakdown": {},
        "latency_reservoir": {},
        "worst_10": [],
        "best_10": [],
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
        result["brier"] = round(group["brier_sum"] / valid_n, 4)
        result["accuracy"] = round(group["correct_count"] / valid_n, 4)
        result["sharpness"] = round(group["sharpness_sum"] / valid_n, 4)
        result["reliability"] = round(valid_n / n, 4)
        result["decision_worthy"] = valid_n >= MIN_SAMPLE_SIZE
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

    # Worst / best 10
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
        worst = scores["worst_10"]
        worst.append(entry)
        worst.sort(key=lambda x: x["brier"], reverse=True)
        scores["worst_10"] = worst[:WORST_BEST_SIZE]

        best = scores["best_10"]
        best.append(entry)
        best.sort(key=lambda x: x["brier"])
        scores["best_10"] = best[:WORST_BEST_SIZE]


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
    ):
        result[dim] = {k: _derive_group(v) for k, v in scores[dim].items()}

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

    scores: dict[str, Any] = {
        "current_month": data["current_month"],
        "generated_at": data.get("generated_at", ""),
        "overall": {
            "n": data["overall"]["n"],
            "valid_n": data["overall"]["valid_n"],
            "brier_sum": data["overall"]["brier_sum"],
            "correct_count": data["overall"]["correct_count"],
            "sharpness_sum": data["overall"]["sharpness_sum"],
        },
    }
    for dim in (
        "by_tool",
        "by_platform",
        "by_category",
        "by_horizon",
        "by_tool_platform",
        "by_tool_version",
        "by_config",
    ):
        scores[dim] = {}
        for key, group in data.get(dim, {}).items():
            scores[dim][key] = {
                "n": group["n"],
                "valid_n": group["valid_n"],
                "brier_sum": group["brier_sum"],
                "correct_count": group["correct_count"],
                "sharpness_sum": group["sharpness_sum"],
            }

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

    for row in new_rows:
        _accumulate_row(scores, row)

    # Write raw accumulators (for future incremental loads)
    finalized = _finalize_scores(scores)
    # Merge accumulators into the output so next load can resume
    output = dict(finalized)
    output["overall"] = {
        **finalized["overall"],
        **{
            k: scores["overall"][k]
            for k in ("brier_sum", "correct_count", "sharpness_sum")
        },
    }
    for dim in (
        "by_tool",
        "by_platform",
        "by_category",
        "by_horizon",
        "by_tool_platform",
        "by_tool_version",
        "by_config",
    ):
        for key, group in scores[dim].items():
            output[dim][key] = {
                **finalized[dim][key],
                **{
                    k: group[k] for k in ("brier_sum", "correct_count", "sharpness_sum")
                },
            }
    # Preserve raw calibration accumulators alongside derived
    output["_calibration_accum"] = scores["calibration"]

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

    # Process all months except the last one as snapshots
    for month in sorted_months[:-1]:
        scores = _empty_scores(month)
        for row in months[month]:
            _accumulate_row(scores, row)
        _snapshot_month(scores, history_path)

    # The last month becomes the current scores.json
    scores = _empty_scores(last_month)
    for row in months[last_month]:
        _accumulate_row(scores, row)

    finalized = _finalize_scores(scores)
    # Write with accumulators for future incremental use
    output = dict(finalized)
    output["overall"] = {
        **finalized["overall"],
        **{
            k: scores["overall"][k]
            for k in ("brier_sum", "correct_count", "sharpness_sum")
        },
    }
    for dim in (
        "by_tool",
        "by_platform",
        "by_category",
        "by_horizon",
        "by_tool_platform",
        "by_tool_version",
        "by_config",
    ):
        for key, group in scores[dim].items():
            output[dim][key] = {
                **finalized[dim][key],
                **{
                    k: group[k] for k in ("brier_sum", "correct_count", "sharpness_sum")
                },
            }
    output["_calibration_accum"] = scores["calibration"]

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
