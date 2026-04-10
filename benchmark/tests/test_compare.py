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
"""Tests for benchmark/compare.py — diagnostic edge metric deltas."""

from typing import Any

from benchmark.compare import compare_stats, format_markdown


def _stats(**overrides: Any) -> dict[str, Any]:
    """Build a minimal stats dict with defaults."""
    base: dict[str, Any] = {
        "brier": 0.25,
        "log_loss": 0.5,
        "directional_accuracy": 0.7,
        "sharpness": 0.2,
        "reliability": 0.95,
        "n": 100,
        "valid_n": 95,
        "conditional_accuracy_rate": None,
        "brier_no_trade": None,
        "brier_small_trade": None,
        "brier_large_trade": None,
        "directional_bias": None,
    }
    base.update(overrides)
    return base


class TestCompareDiagnosticMetrics:
    """Tests for diagnostic metric deltas in compare_stats."""

    def test_conditional_accuracy_improved(self) -> None:
        """Higher conditional accuracy → improved."""
        baseline = _stats(conditional_accuracy_rate=0.65)
        candidate = _stats(conditional_accuracy_rate=0.75)
        result = compare_stats(baseline, candidate)

        ca = result["conditional_accuracy_rate"]
        assert ca["baseline"] == 0.65
        assert ca["candidate"] == 0.75
        assert ca["delta"] == 0.1
        assert ca["direction"] == "improved"

    def test_brier_large_trade_improved(self) -> None:
        """Lower brier on large trades → improved."""
        baseline = _stats(brier_large_trade=0.30)
        candidate = _stats(brier_large_trade=0.25)
        result = compare_stats(baseline, candidate)

        bl = result["brier_large_trade"]
        assert bl["delta"] == -0.05
        assert bl["direction"] == "improved"

    def test_directional_bias_toward_zero_improved(self) -> None:
        """Bias moving from +0.08 toward 0 → improved."""
        baseline = _stats(directional_bias=0.08)
        candidate = _stats(directional_bias=0.03)
        result = compare_stats(baseline, candidate)

        db = result["directional_bias"]
        assert db["baseline"] == 0.08
        assert db["candidate"] == 0.03
        # abs delta: 0.03 - 0.08 = -0.05 (lower abs = improved)
        assert db["delta"] == -0.05
        assert db["direction"] == "improved"

    def test_directional_bias_sign_flip_improved(self) -> None:
        """Bias from +0.08 to -0.02 → improved (closer to 0)."""
        baseline = _stats(directional_bias=0.08)
        candidate = _stats(directional_bias=-0.02)
        result = compare_stats(baseline, candidate)

        db = result["directional_bias"]
        # abs: 0.02 - 0.08 = -0.06
        assert db["delta"] == -0.06
        assert db["direction"] == "improved"

    def test_directional_bias_away_from_zero_regressed(self) -> None:
        """Bias from +0.02 to -0.08 → regressed (further from 0)."""
        baseline = _stats(directional_bias=0.02)
        candidate = _stats(directional_bias=-0.08)
        result = compare_stats(baseline, candidate)

        db = result["directional_bias"]
        # abs: 0.08 - 0.02 = +0.06
        assert db["delta"] == 0.06
        assert db["direction"] == "regressed"

    def test_none_values_produce_dashes(self) -> None:
        """None baseline or candidate → delta is None, direction is '—'."""
        baseline = _stats(conditional_accuracy_rate=None)
        candidate = _stats(conditional_accuracy_rate=0.75)
        result = compare_stats(baseline, candidate)

        ca = result["conditional_accuracy_rate"]
        assert ca["delta"] is None
        assert ca["direction"] == "—"

    def test_format_markdown_includes_diagnostics(self) -> None:
        """format_markdown includes diagnostic section when data present."""
        baseline = _stats(
            conditional_accuracy_rate=0.65,
            brier_large_trade=0.30,
            directional_bias=0.08,
        )
        candidate = _stats(
            conditional_accuracy_rate=0.75,
            brier_large_trade=0.25,
            directional_bias=0.03,
        )
        comparison = {
            "overall": compare_stats(baseline, candidate),
            "by_tool": {},
            "by_platform": {},
            "by_category": {},
        }
        md = format_markdown(comparison)
        assert "## Diagnostic Edge Metrics" in md
        assert "Conditional Accuracy" in md
        assert "Brier (large trade)" in md
        assert "Directional Bias" in md

    def test_format_markdown_hides_diagnostics_when_all_none(self) -> None:
        """No diagnostic table when all values are None."""
        baseline = _stats()  # all diagnostic keys are None
        candidate = _stats()
        comparison = {
            "overall": compare_stats(baseline, candidate),
            "by_tool": {},
            "by_platform": {},
            "by_category": {},
        }
        md = format_markdown(comparison)
        assert "## Diagnostic Edge Metrics" not in md
