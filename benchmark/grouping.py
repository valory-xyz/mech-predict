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
"""Per-row accumulator merging for the incremental scoring pipeline.

Given the ``scores`` dict shape defined in ``scorer._empty_scores``,
``accumulate_row`` merges one production log row into every grouping
dimension (by_tool, by_platform, by_horizon, ...), the calibration
bucket + reservoir, the parse breakdown, the latency reservoir, and
the deduplicated worst/best lists.

Split out of ``scorer`` so downstream code that only needs the
row-merge entry point (e.g. batch backfills, tests) can import it
without pulling in the CLI, rebuild, or period-scoring surface.

The single public entry point is :func:`accumulate_row`. The rest are
underscore-prefixed helpers used by the scorer's finalize path.
"""

from __future__ import annotations

from typing import Any

from benchmark.scoring_primitives import (
    CALIBRATION_BINS,
    CALIBRATION_PAIRS_RESERVOIR_SIZE,
    LATENCY_RESERVOIR_SIZE,
    MIN_SAMPLE_SIZE,
    WORST_BEST_SIZE,
    _RESERVOIR_RNG,
    _bin_label,
    _derive_diagnostic_metrics,
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
            # Market Brier over the edge-eligible pool. NOT derivable as
            # brier+edge: those average over different row pools.
            if group.get("market_brier_sum") is not None:
                group["market_brier_sum"] += (
                    market_prob - (1.0 if outcome else 0.0)
                ) ** 2
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
        market_sum = group.get("market_brier_sum")
        result["market_brier"] = (
            None if market_sum is None else round(market_sum / edge_n, 4)
        )
        result["edge_positive_rate"] = round(group["edge_positive_count"] / edge_n, 4)
    else:
        result["edge"] = None
        result["market_brier"] = None
        result["edge_positive_rate"] = None

    # Diagnostic edge metrics — conditional accuracy, disagreement Brier, bias
    _derive_diagnostic_metrics(group, result)

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


def accumulate_row(scores: dict[str, Any], row: dict[str, Any]) -> None:
    """Merge one row into all accumulator dimensions (mutates *scores*).

    :param scores: the full scores dict with accumulators.
    :param row: a production log row dict.
    """
    # Observed row bounds; current_month misstates the span after rollover
    # (see benchmark.digest_tables._as_of_date).
    stamp = row.get("predicted_at")
    if stamp:
        first = scores.get("window_start")
        if first is None or stamp < first:
            scores["window_start"] = stamp
        last = scores.get("window_end")
        if last is None or stamp > last:
            scores["window_end"] = stamp
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
    _ensure_and_accumulate(
        scores["by_category_platform"], f"{category} | {platform}", row
    )
    _ensure_and_accumulate(
        scores["by_tool_category_platform"],
        f"{tool} | {category} | {platform}",
        row,
    )
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
