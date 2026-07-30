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
"""Which tools are selectable in production, and which of those we can build.

The triage assesses whatever appears in the scored data, which includes
tools that live only in the tournament and are deployed nowhere. A fix
issue on such a tool has no validation path (its PR cannot be replayed
from production rows) and no deployment consequence, so the triage needs
to know the deployed set.

Discovery is delegated to :mod:`benchmark.tool_usage`, which already
resolves it with no local manifest: latest ``valory-xyz/trader`` release
-> each deployment's ``service.yaml`` -> its ``valid_mechs`` allow-list
-> the mech-marketplace subgraph -> each mech's on-chain metadata CID ->
the IPFS manifest's ``tools``. That chain answers "what can the live
trader select?", which is the question that matters, and it needs no
service ids, registry addresses or mech lists stored here.

This module adds the second half: intersect that set with the tools this
repository can build (``benchmark/tools.py``). Only the intersection is
actionable for the improvement loop -- a tool we cannot build cannot be
fixed here, and a tool nobody serves has nowhere for a fix to land.

Read-only: GitHub API, subgraph and IPFS gateway GETs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from benchmark.tool_usage import (
    DEPLOYMENT_TO_PLATFORM,
    fetch_valid_tools,
    normalize_tool_name,
)

log = logging.getLogger(__name__)

TOOLS_REGISTRY_PATH = Path(__file__).resolve().parent / "tools.py"
TOURNAMENT_TOOLS_PATH = Path(__file__).resolve().parent / "tournament_tools.json"

_TOOL_SPEC_RE = re.compile(r'^\s*"([a-z0-9_.-]+)":\s*ToolSpec\(', re.M)


def repo_tools() -> Set[str]:
    """Tool names this repository can build, from ``benchmark/tools.py``.

    Parsed textually so the registry (and every tool module's imports) does
    not have to be importable just to answer this.

    :return: registered tool names.
    """
    text = TOOLS_REGISTRY_PATH.read_text(encoding="utf-8")
    return set(_TOOL_SPEC_RE.findall(text))


def tournament_tools() -> Set[str]:
    """Tool names currently on the tournament roster.

    :return: roster names; empty when the file is missing or unreadable.
    """
    try:
        data = json.loads(TOURNAMENT_TOOLS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("tournament roster unreadable at %s", TOURNAMENT_TOOLS_PATH)
        return set()
    return set(data) if isinstance(data, dict) else set()


def served_tools(
    platform: Optional[str] = None,
    valid: Optional[Dict[str, Optional[List[str]]]] = None,
) -> Set[str]:
    """Tools selectable by the live trader, normalized for naming drift.

    :param platform: restrict to deployments of this platform (``"omen"`` /
        ``"polymarket"``); all deployments when omitted.
    :param valid: pre-fetched :func:`benchmark.tool_usage.fetch_valid_tools`
        result; fetched when omitted.
    :return: served tool names (dash-normalized). A deployment whose
        resolution failed contributes nothing and is logged.
    """
    valid = fetch_valid_tools() if valid is None else valid
    out: Set[str] = set()
    for deployment, tools in valid.items():
        if platform and DEPLOYMENT_TO_PLATFORM.get(deployment) != platform:
            continue
        if tools is None:
            log.warning("%s: selectable tools unavailable (fetch failed)", deployment)
            continue
        out.update(normalize_tool_name(tool) for tool in tools)
    return out


def actionable_tools(
    platform: Optional[str] = None,
    valid: Optional[Dict[str, Optional[List[str]]]] = None,
) -> Set[str]:
    """Tools BOTH selectable in production AND buildable from this repo.

    This is the set the improvement loop may open fix issues for: anything
    outside it either cannot be edited here or would have no deployment
    consequence.

    :param platform: restrict to one platform.
    :param valid: pre-fetched selectable-tools map.
    :return: actionable tool names, spelled as this repo spells them.
    """
    served = served_tools(platform, valid)
    return {tool for tool in repo_tools() if normalize_tool_name(tool) in served}


def main() -> int:
    """CLI: print the selectable set, this repo's set, and their intersection.

    :return: exit code (1 when no deployment could be resolved).
    """
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--platform", default=None, help="restrict to omen / polymarket"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.WARNING)

    valid = fetch_valid_tools()
    repo = repo_tools()
    repo_normalized = {normalize_tool_name(tool) for tool in repo}
    roster = tournament_tools()
    served = served_tools(args.platform, valid)
    actionable = actionable_tools(args.platform, valid)
    resolved = any(tools is not None for tools in valid.values())

    if args.json:
        print(
            json.dumps(
                {
                    "selectable_by_deployment": {
                        deployment: (sorted(tools) if tools is not None else None)
                        for deployment, tools in valid.items()
                    },
                    "repo_tools": sorted(repo),
                    "tournament_roster": sorted(roster),
                    "actionable": sorted(actionable),
                },
                indent=2,
            )
        )
        return 0 if resolved else 1

    print("SELECTABLE IN PRODUCTION (trader valid_mechs -> marketplace -> IPFS)")
    for deployment, tools in valid.items():
        platform = DEPLOYMENT_TO_PLATFORM.get(deployment, "?")
        if args.platform and platform != args.platform:
            continue
        if tools is None:
            print(f"  {deployment} [{platform}] -- unavailable (fetch failed)")
            continue
        print(f"  {deployment} [{platform}]")
        for tool in sorted(tools):
            known = normalize_tool_name(tool) in repo_normalized
            print(f"      - {tool}{'' if known else '   (not in this repo)'}")

    print("\nACTIONABLE (selectable AND buildable here)")
    for tool in sorted(actionable):
        print(f"  * {tool}{'   [also in tournament]' if tool in roster else ''}")

    print("\nNOT ACTIONABLE")
    for tool in sorted(repo - actionable):
        where = "tournament only" if tool in roster else "repo only"
        print(f"  - {tool}   ({where}, not selectable in production)")
    for tool in sorted(served - repo_normalized):
        print(f"  - {tool}   (selectable but not buildable from this repo)")

    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(main())
