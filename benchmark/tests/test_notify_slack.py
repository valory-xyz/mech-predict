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
"""Tests for benchmark/notify_slack.py — platform-scoped Slack summaries."""

from pathlib import Path

import pytest

from benchmark.analyze import PLATFORM_LABELS
from benchmark.notify_slack import (
    SUMMARY_SYSTEM_PROMPT_TEMPLATE,
    _build_system_prompt,
    _infer_platform_label,
)


class TestBuildSystemPrompt:
    """_build_system_prompt threads the deployment label through the template."""

    def test_omenstrat_label_appears_in_prompt(self) -> None:
        """Omenstrat label renders into the Summary / Category / Actions sections."""
        prompt = _build_system_prompt("Omenstrat")
        assert "Omenstrat" in prompt
        # Spot-check the sentences that should carry the label so a future
        # template refactor doesn't silently drop deployment scoping.
        assert "for the *Omenstrat* deployment" in prompt
        assert "scoped to Omenstrat" in prompt

    def test_polystrat_label_appears_in_prompt(self) -> None:
        """Polystrat label renders symmetrically to Omenstrat."""
        prompt = _build_system_prompt("Polystrat")
        assert "Polystrat" in prompt
        assert "for the *Polystrat* deployment" in prompt

    def test_no_cross_platform_leakage(self) -> None:
        """Omenstrat prompt must not reference Polystrat and vice versa."""
        omen = _build_system_prompt("Omenstrat")
        assert "Polystrat" not in omen

        poly = _build_system_prompt("Polystrat")
        assert "Omenstrat" not in poly

    def test_template_no_longer_instructs_platform_comparison(self) -> None:
        """Single-platform summaries must not ask the LLM to 'list all platforms'.

        The legacy fleet-wide prompt had ``*Platform performance:* list all
        platforms`` and ``*Edge by difficulty:* ... per platform`` blocks.
        Both are meaningless in per-platform mode and must not bleed into
        the new template.
        """
        # Check the raw template so the assertion is independent of any
        # particular platform substitution.
        assert "list all platforms" not in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "one line per platform" not in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "Platform × Difficulty" not in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        # "*Platform performance:*" was a dedicated bullet in the old prompt.
        assert "*Platform performance:*" not in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_template_still_carries_core_sections(self) -> None:
        """Core single-platform sections remain wired up after the refactor."""
        assert "*Summary:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Top tools:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Worst tools:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Category performance:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Tool × Category highlights:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Tournament callouts:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Diagnostics:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE
        assert "*Recommended actions:*" in SUMMARY_SYSTEM_PROMPT_TEMPLATE


class TestInferPlatformLabel:
    """_infer_platform_label recovers the deployment label from the filename."""

    def test_omen_report(self) -> None:
        """report_omen.md -> Omenstrat."""
        assert _infer_platform_label(Path("/tmp/report_omen.md")) == "Omenstrat"

    def test_polymarket_report(self) -> None:
        """report_polymarket.md -> Polystrat."""
        assert _infer_platform_label(Path("/tmp/report_polymarket.md")) == "Polystrat"

    def test_unknown_stem_returns_none(self) -> None:
        """Unrecognised filenames get None so the caller can error explicitly."""
        assert _infer_platform_label(Path("/tmp/report.md")) is None
        assert _infer_platform_label(Path("/tmp/report_gnosis.md")) is None


class TestPromptRejectsUnformattedPlaceholder:
    """Guard against a missed ``{platform_label}`` replacement."""

    def test_template_contains_placeholder(self) -> None:
        """Template must include a ``{platform_label}`` placeholder."""
        # Without the placeholder, _build_system_prompt becomes a no-op and
        # the LLM loses deployment scoping silently.
        assert "{platform_label}" in SUMMARY_SYSTEM_PROMPT_TEMPLATE

    def test_build_raises_on_empty_label(self) -> None:
        """Empty label is rejected — would render "for the ** deployment"."""
        with pytest.raises(ValueError, match="platform_label"):
            _build_system_prompt("")

    def test_build_raises_on_unknown_label(self) -> None:
        """A label outside PLATFORM_LABELS is rejected before reaching the LLM.

        Guards against a workflow-level typo like ``--platform-label Omenstrap``
        silently producing a deployment-mislabeled summary.
        """
        with pytest.raises(ValueError, match="must be one of"):
            _build_system_prompt("Omenstrap")

    def test_labels_tracked_from_analyze(self) -> None:
        """Every ``benchmark.analyze.PLATFORM_LABELS`` value is accepted.

        Reusing the same import surface means a rename in analyze.py
        (e.g. Omenstrat -> Omen Strat) doesn't drift the two modules out
        of sync.
        """
        for label in PLATFORM_LABELS.values():
            _build_system_prompt(label)
