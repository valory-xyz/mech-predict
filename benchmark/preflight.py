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
"""Benchmark preflight: fail a misconfigured run before anything is spent.

Invoked by ``benchmark_replay.yaml`` right after dependency install and
before the log download / tournament fetch / enrich stages. Lives here as a
real module -- not a snippet inside the workflow YAML -- so the same lint,
type and test gates that cover the code it protects cover the guard itself.

Failures print GitHub Actions ``::error::`` annotations, so both failure
modes (unregistered name, unusable schema) surface identically on the run
summary instead of one being a bare traceback in the log.
"""

from __future__ import annotations

import sys

from benchmark.prompt_replay import _get_structured_output_schema
from benchmark.tools import TOOL_REGISTRY


def main(argv: list[str]) -> int:
    """Validate the requested tool pairing against the registry.

    :param argv: ``[tool, candidate_tool]``; an empty candidate means an
        in-place run (candidate defaults to the tool itself).
    :return: process exit code -- 0 when the pairing is replayable, 1 when
        the run would be doomed and must not proceed to the download stages.
    """
    if len(argv) < 1 or not argv[0]:
        print("::error::preflight called without a tool name")
        return 1
    tool = argv[0]
    candidate = argv[1] if len(argv) > 1 and argv[1] else tool

    missing = sorted({name for name in (tool, candidate) if name not in TOOL_REGISTRY})
    if missing:
        print(
            f"::error::not registered in benchmark/tools.py TOOL_REGISTRY: "
            f"{missing}. Add a ToolSpec entry (same family as the parent). "
            f"Registered: {sorted(TOOL_REGISTRY)}"
        )
        return 1

    try:
        # Raises ValueError naming tool+module on a present-but-unusable
        # schema; returns None for plain-prompt tools, which is fine.
        _get_structured_output_schema(candidate)
    except (ValueError, ImportError) as exc:
        print(f"::error::{exc}")
        return 1

    print(f"preflight ok: tool={tool} candidate={candidate}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
