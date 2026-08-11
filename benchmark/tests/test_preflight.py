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
"""Tests for the workflow preflight.

The preflight used to live as a Python snippet inside benchmark_replay.yaml,
where no linter, type checker or test could see it — the guard against silent
misconfiguration was itself unguarded. These tests are the point of the
extraction: every branch of the module is pinned.
"""

import sys
import types
from pathlib import Path

import pytest
from benchmark import preflight
from benchmark.preflight import main
from benchmark.tools import TOOL_REGISTRY


class TestPreflight:
    """Every exit path of the preflight, driven as the workflow drives it."""

    def test_registered_pairing_passes(self, capsys: pytest.CaptureFixture) -> None:
        """A registered baseline with a registered structured candidate: 0."""
        assert main(["superforcaster", "superforcaster-polymarket-v4"]) == 0
        assert "preflight ok" in capsys.readouterr().out

    def test_empty_candidate_defaults_to_tool(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """In-place runs pass an empty candidate; it must not fail the check."""
        assert main(["superforcaster_full_search", ""]) == 0
        out = capsys.readouterr().out
        assert "candidate=superforcaster_full_search" in out

    def test_unregistered_tool_fails_with_annotation(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """An unregistered name exits 1 with a ::error:: annotation.

        The annotation is load-bearing: it is what surfaces on the Actions
        run summary instead of a bare traceback in the log.

        :param capsys: pytest capsys fixture.
        """
        assert main(["no-such-tool", ""]) == 1
        out = capsys.readouterr().out
        assert out.startswith("::error::")
        assert "no-such-tool" in out
        assert "TOOL_REGISTRY" in out

    def test_unregistered_candidate_behind_registered_baseline_fails(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """The candidate is validated too, not just the baseline."""
        assert main(["superforcaster", "no-such-candidate"]) == 1
        assert "no-such-candidate" in capsys.readouterr().out

    def test_missing_argument_fails(self, capsys: pytest.CaptureFixture) -> None:
        """No tool argument at all is a hard error, not a crash."""
        assert main([]) == 1
        assert "::error::" in capsys.readouterr().out

    def test_malformed_schema_fails_with_annotation(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """A present-but-unusable schema annotates instead of raising bare.

        This is the asymmetry fix: the registry-missing branch already
        annotated, while a schema ValueError previously escaped bare.

        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        """
        bad = types.ModuleType("preflight_bad_module")
        bad.PredictionResult = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "preflight_bad_module", bad)
        monkeypatch.setitem(
            TOOL_REGISTRY,
            "preflight-bad-tool",
            type(TOOL_REGISTRY["superforcaster"])(
                module="preflight_bad_module", family="superforcaster"
            ),
        )
        assert main(["superforcaster", "preflight-bad-tool"]) == 1
        out = capsys.readouterr().out
        assert out.startswith("::error::")
        assert "preflight-bad-tool" in out

    def test_import_crashing_module_annotates_not_tracebacks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """A tool whose module raises at import must annotate, not traceback.

        A module broken by a bad merge raises whatever its top-level code
        hits (RuntimeError, NameError, SyntaxError) -- none of which the old
        (ValueError, ImportError) tuple covered, so exactly the
        misconfiguration class this preflight exists for escaped as the bare
        traceback the docstring promises it prevents.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        """
        crasher = "preflight_import_crasher"
        (tmp_path / f"{crasher}.py").write_text(
            'raise RuntimeError("module-level explosion")\n', encoding="utf-8"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setitem(
            TOOL_REGISTRY,
            "preflight-crash-tool",
            type(TOOL_REGISTRY["superforcaster"])(
                module=crasher, family="superforcaster"
            ),
        )
        assert main(["superforcaster", "preflight-crash-tool"]) == 1
        out = capsys.readouterr().out
        assert out.startswith("::error::"), f"escaped bare: {out[:80]!r}"
        assert "RuntimeError" in out and "module-level explosion" in out

    def test_import_error_annotates(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """The plain-ImportError branch (missing dependency) is pinned too.

        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        """
        monkeypatch.setitem(
            TOOL_REGISTRY,
            "preflight-missing-dep-tool",
            type(TOOL_REGISTRY["superforcaster"])(
                module="no_such_module_anywhere_xyz", family="superforcaster"
            ),
        )
        assert main(["superforcaster", "preflight-missing-dep-tool"]) == 1
        out = capsys.readouterr().out
        assert out.startswith("::error::")
        assert "ModuleNotFoundError" in out or "ImportError" in out

    def test_module_entrypoint_matches_workflow_invocation(self) -> None:
        """Pin the ``python -m benchmark.preflight`` contract the workflow uses."""
        assert hasattr(preflight, "main")
        assert callable(preflight.main)
