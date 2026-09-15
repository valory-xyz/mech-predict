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
"""Tests for benchmark/sweep.py: the market-context arm reaches replay intact.

A sweep scores a whole candidate file, so the two market-context arms must
never share one. These tests pin the per-arm filename and the flag's path from
the CLI down to ``replay``; every tool call is mocked.
"""

import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from benchmark.sweep import main, step_replay

SWEEP = "benchmark.sweep"


class TestStepReplayThreadsTheFlag:
    """step_replay must forward the arm, and only by keyword."""

    @patch(f"{SWEEP}.replay")
    @pytest.mark.parametrize("flag", [True, False], ids=["on", "off"])
    def test_forwards_the_flag(self, mock_replay: MagicMock, flag: bool) -> None:
        """The value handed to step_replay reaches replay unchanged."""
        step_replay(
            Path("in.jsonl"),
            Path("out.jsonl"),
            ["superforcaster-market-aware"],
            "gpt-4.1",
            240,
            with_market_context=flag,
        )

        assert mock_replay.call_args.kwargs["with_market_context"] is flag

    def test_flag_is_keyword_only(self) -> None:
        """A positional insert must raise rather than silently flip the arm."""
        param = inspect.signature(step_replay).parameters["with_market_context"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


class TestCandidateFilePerArm:
    """Each arm writes its own candidate file, so scores never mix."""

    @staticmethod
    def _run(tmp_path: Path, argv: list[str]) -> MagicMock:
        """Run sweep.main over a stub dataset with every step mocked.

        :param tmp_path: pytest temporary directory.
        :param argv: extra CLI arguments.
        :return: the patched step_replay mock.
        """
        dataset = tmp_path / "dataset.jsonl"
        dataset.write_text("", encoding="utf-8")
        args = [
            "sweep.py",
            "--dataset",
            str(dataset),
            "--tools",
            "superforcaster-market-aware",
            "--output-dir",
            str(tmp_path),
            *argv,
        ]
        with (
            patch("sys.argv", args),
            patch(f"{SWEEP}.step_score_baseline", return_value={}),
            patch(f"{SWEEP}.step_score", return_value={}),
            patch(f"{SWEEP}.step_compare", return_value=""),
            patch(f"{SWEEP}.step_replay") as mock_replay,
        ):
            main()
        return mock_replay

    def test_blind_and_priced_use_different_files(self, tmp_path: Path) -> None:
        """The two arms never resume into, or score with, each other's rows."""
        blind: Any = self._run(tmp_path, []).call_args.args[1]
        priced: Any = self._run(tmp_path, ["--market-context"]).call_args.args[1]

        assert blind != priced
        assert "market_context" in priced.name
        assert "market_context" not in blind.name

    def test_flag_reaches_step_replay(self, tmp_path: Path) -> None:
        """--market-context on the CLI arrives as the keyword argument."""
        call = self._run(tmp_path, ["--market-context"]).call_args

        assert call.kwargs["with_market_context"] is True
