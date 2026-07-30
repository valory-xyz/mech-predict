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

This module adds the second half: intersect that set with the tool
packages this repository actually ships (``packages/packages.json``,
the ledger ``autonomy packages lock`` maintains and deployment consumes
-- not ``benchmark/tools.py``, which is a benchmark-harness convenience
list that drifts from what is shipped). Only the intersection is
actionable for the improvement loop -- a tool we do not ship cannot be
fixed here, and a tool nobody serves has nowhere for a fix to land.

Mech manifests advertise tool NAMES only (no CIDs), so matching is by
name with underscores and dashes treated as interchangeable
(``superforcaster_full_search`` the package == ``superforcaster-full-search``
the served tool). Version suffixes are NOT stripped: ``-v1`` and ``-v3``
are different tools and only the ones actually advertised count as
served.

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
    deployments_for_platform,
    fetch_valid_tools,
    normalize_tool_name,
)
from benchmark.tournament_tools import roster_names

log = logging.getLogger(__name__)

PACKAGES_JSON_PATH = (
    Path(__file__).resolve().parent.parent / "packages" / "packages.json"
)
TOURNAMENT_TOOLS_PATH = Path(__file__).resolve().parent / "tournament_tools.json"

# packages.json keys look like ``custom/<author>/<package_name>/<version>``.
_CUSTOM_KEY_RE = re.compile(r"^custom/[^/]+/([^/]+)/")


def repo_tools() -> Optional[Set[str]]:
    """Tool packages this repository ships, or ``None`` when unknowable.

    The package ledger is the shipping source of truth (maintained by
    ``autonomy packages lock``, consumed by deployment), unlike
    ``benchmark/tools.py`` which only lists what the harness can run.

    Returns ``None`` rather than an empty set when the ledger cannot be read
    (truncated mid-``lock``, permissions, CI disk hiccup): "we ship nothing"
    and "we could not find out" must not be the same answer, because a
    consumer that treats them alike silently concludes nothing is actionable.

    :return: package names of every ``custom/`` entry, or ``None`` on failure.
    """
    try:
        data = json.loads(PACKAGES_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("packages.json unreadable at %s", PACKAGES_JSON_PATH)
        return None
    names: Set[str] = set()
    for section in ("dev", "third_party"):
        for key in data.get(section) or {}:
            match = _CUSTOM_KEY_RE.match(key)
            if match:
                names.add(match.group(1))
    return names


def tournament_tools() -> Optional[Set[str]]:
    """Tool names on the tournament roster, normalized; ``None`` if unknown.

    Thin delegate to :func:`benchmark.tournament_tools.roster_names` so this
    module is not a fourth, subtly-different parse of the same file. In
    particular the names come back in the UNDERSCORE namespace, which is what
    makes membership tests against package names correct -- comparing raw
    roster keys (dash spelling) to package names (underscore spelling) can
    never match for a versioned tool.

    :return: roster names folded into this module's single comparison
        namespace (``normalize_tool_name``), or ``None`` when unknown.
    """
    raw = roster_names(TOURNAMENT_TOOLS_PATH)
    return None if raw is None else {normalize_tool_name(n) for n in raw}


def served_tools(
    platform: Optional[str] = None,
    valid: Optional[Dict[str, Optional[List[str]]]] = None,
) -> Optional[Set[str]]:
    """Tools selectable by the live trader, normalized for naming drift.

    Returns ``None`` when every deployment IN SCOPE failed to resolve, so a
    total discovery outage is distinguishable from "this platform genuinely
    serves nothing". :func:`fetch_valid_tools` never raises -- a single
    failure at the trader-release lookup nulls every deployment at once -- so
    without this distinction a rate-limited morning is indistinguishable from
    a quiet one, and the improvement loop would silently assess nothing.

    :param platform: restrict to deployments of this platform (``"omen"`` /
        ``"polymarket"``); all deployments when omitted.
    :param valid: pre-fetched :func:`benchmark.tool_usage.fetch_valid_tools`
        result; fetched when omitted.
    :return: served tool names (dash-normalized), or ``None`` when nothing in
        scope could be resolved.
    """
    valid = fetch_valid_tools() if valid is None else valid
    in_scope = set(deployments_for_platform(platform)) if platform else set(valid)
    out: Set[str] = set()
    resolved_any = False
    for deployment, tools in valid.items():
        if deployment not in in_scope:
            continue
        if tools is None:
            log.warning("%s: selectable tools unavailable (fetch failed)", deployment)
            continue
        resolved_any = True
        out.update(normalize_tool_name(tool) for tool in tools)
    if not resolved_any:
        log.warning(
            "no deployment resolved%s; selectable set is UNKNOWN, not empty.",
            f" for platform {platform}" if platform else "",
        )
        return None
    return out


def actionable_tools(
    platform: Optional[str] = None,
    valid: Optional[Dict[str, Optional[List[str]]]] = None,
) -> Optional[Set[str]]:
    """Tools BOTH selectable in production AND buildable from this repo.

    This is the set the improvement loop may open fix issues for: anything
    outside it either cannot be edited here or would have no deployment
    consequence.

    ``None`` propagates from either input (discovery outage, unreadable
    ledger) and means "unknown" -- the caller must then assess everything
    rather than silence every tool. An empty SET is a real answer.

    :param platform: restrict to one platform.
    :param valid: pre-fetched selectable-tools map.
    :return: actionable tool names as this repo spells them, or ``None``.
    """
    served = served_tools(platform, valid)
    repo = repo_tools()
    if served is None or repo is None:
        return None
    return {tool for tool in repo if normalize_tool_name(tool) in served}


def main() -> int:
    """CLI: print the selectable set, this repo's set, and their intersection.

    :return: exit code (1 when nothing IN SCOPE could be resolved).
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
    repo_normalized = {normalize_tool_name(t) for t in repo} if repo else set()
    roster = tournament_tools()
    served = served_tools(args.platform, valid)
    actionable = actionable_tools(args.platform, valid)
    # Scope the exit code to what was ASKED for: fetch_valid_tools() always
    # resolves both deployments, so judging success over the unfiltered map
    # let `--platform omen` exit 0 on a total omen failure just because
    # polymarket happened to work.
    resolved = served is not None

    def _roster_has(tool: str) -> Optional[bool]:
        """Roster membership across the dash/underscore namespaces.

        :param tool: tool name in either spelling.
        :return: membership, or ``None`` when the roster is unknown.
        """
        if roster is None:
            return None
        return normalize_tool_name(tool) in roster

    if args.json:
        print(
            json.dumps(
                {
                    "selectable_by_deployment": {
                        deployment: (sorted(tools) if tools is not None else None)
                        for deployment, tools in valid.items()
                    },
                    "repo_tools": sorted(repo) if repo is not None else None,
                    "tournament_roster": sorted(roster) if roster is not None else None,
                    "actionable": (
                        sorted(actionable) if actionable is not None else None
                    ),
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
            print(f"      - {tool}{'' if known else '   (not shipped by this repo)'}")

    print("\nACTIONABLE (selectable AND shipped by this repo)")
    if actionable is None:
        print("  UNKNOWN -- discovery did not resolve; assess all tools")
    else:
        for tool in sorted(actionable):
            in_roster = _roster_has(tool)
            tag = {
                True: "   [also in tournament]",
                None: "   [tournament UNKNOWN]",
            }.get(in_roster, "")
            print(f"  * {tool}{tag}")

    print("\nNOT ACTIONABLE")
    for tool in sorted((repo or set()) - (actionable or set())):
        in_roster = _roster_has(tool)
        where = {
            True: "tournament only",
            None: "tournament membership unknown",
        }.get(in_roster, "shipped only")
        print(f"  - {tool}   ({where}, not selectable in production)")
    for name in sorted(served or set()):
        if name not in repo_normalized:
            print(f"  - {name}   (selectable but not shipped by this repo)")
    # Roster-only tools ship no package and are served by nobody, so they fall
    # outside both sets above -- yet they are the #416 class this whole change
    # exists for (predict-fine-tuned-calibrated among them). Omitting them from
    # the text report while listing them in --json hides the motivating case
    # from the only output a human reads.
    for name in sorted((roster or set()) - repo_normalized - (served or set())):
        print(f"  - {name}   (tournament only, ships no package here)")

    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(main())
