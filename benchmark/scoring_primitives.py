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
"""Pure scoring primitives shared by scorer.py and grouping.py.

Constants, classification helpers, and single-row score functions that
have no dependency on ``scorer`` or ``grouping``. Anything that can be
evaluated for one prediction row without touching accumulators lives
here.

Downstream modules should import from this module rather than reach
back into ``scorer`` for these building blocks — that keeps the
grouping layer free of a cyclic dependency on the CLI/rebuild code
that lives in ``scorer``.
"""

from __future__ import annotations

import math
import os
import random
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Feature-flag parsing
# ---------------------------------------------------------------------------

USE_MECH_ANALYTICS_ROWS_ENV = "USE_MECH_ANALYTICS_ROWS"


def use_mech_analytics_rows() -> bool:
    """Return True when USE_MECH_ANALYTICS_ROWS is set to ``true``."""
    return parse_truthy_env(USE_MECH_ANALYTICS_ROWS_ENV)


def parse_truthy_env(name: str) -> bool:
    """Read ``name`` from env and return True only for ``true`` (case-insensitive).

    The parser was previously broader (``1`` / ``yes`` / ``on``) but the
    CI workflow gates compare the raw env var to the literal ``'true'``
    in YAML. Any value the Python side accepted that YAML did not (e.g.
    ``1``) would put the two sides on opposite branches: Python would
    take the mech-analytics path while CI would still delete
    ``scores_history.jsonl`` because the YAML gate read the flag as off.
    Narrowing here keeps both sides in lockstep.

    :param name: env var name to read.
    :return: True when the value is ``true`` (case-insensitive, trimmed).
    """
    return os.getenv(name, "").strip().lower() == "true"


def start_of_current_month_utc() -> datetime:
    """First moment of the current UTC month.

    Shared "main scores" window anchor for the mech-analytics-fed
    scorer and analyzer paths. Mirrors the monthly-accumulator
    semantics of on-disk ``scores_<platform>.json``.

    :return: timezone-aware datetime at 00:00:00 UTC on day 1.
    """
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Reservoir / sample-size constants
# ---------------------------------------------------------------------------

LATENCY_RESERVOIR_SIZE = 200
CALIBRATION_PAIRS_RESERVOIR_SIZE = 50_000
_RESERVOIR_RNG = random.Random(
    42
)  # nosec B311 — reservoir sampling for reproducible scoring
WORST_BEST_SIZE = 10

MIN_SAMPLE_SIZE = 30

# Diagnostic edge metric thresholds (PROPOSAL.md Stage 4).
# Fixed for now — will version if changed.
DISAGREE_THRESHOLD = 0.03
LARGE_TRADE_THRESHOLD = 0.10

# Thresholds are initial values from PROPOSAL.md. Adjust after inspecting
# the actual data distribution from the first scorer run.
DIFFICULTY_THRESHOLDS = (0.15, 0.3)
LIQUIDITY_THRESHOLDS = (500.0, 5000.0)

_LOG_LOSS_EPSILON = 1e-15


# ---------------------------------------------------------------------------
# Single-row score functions
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
# Difficulty, liquidity, and horizon classification
# ---------------------------------------------------------------------------


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
# Calibration bin definitions
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


# ---------------------------------------------------------------------------
# Group accumulator helpers
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
