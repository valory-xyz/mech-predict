"""
Tournament tool→CID map: the single source of truth, with zero heavy deps.

This module deliberately imports only the standard library so that lightweight
consumers (e.g. ``benchmark.analyze`` running in CI's minimal env) can read the
tournament tool→CID map without dragging in the open-autonomy / ``aea`` stack
that ``benchmark.tournament`` pulls in via ``ipfs_loader`` and ``KeyChain``.

``benchmark.tournament`` re-imports ``load_tournament_tools`` from here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Set

# ---------------------------------------------------------------------------
# Tournament tools — single source of truth (TOURNAMENT_IPFS_LOADER_SPEC §3.1)
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

TOURNAMENT_TOOLS_JSON = Path(__file__).resolve().parent / "tournament_tools.json"


def roster_names(path: Path = TOURNAMENT_TOOLS_JSON) -> Optional[Set[str]]:
    """Tournament roster names, normalized; ``None`` when the roster is unknown.

    The canonical reader for "is this tool in the tournament". It exists here,
    in the dependency-light source of truth, because three modules were each
    answering that question with their own parse and disagreeing about two
    things that matter:

    * **Naming.** Package directories use underscores while registered tool
      names use dashes, so any versioned tool is spelled BOTH ways depending
      on where you read it. Keys come back AS WRITTEN and each caller folds
      them with its own canonical normalizer -- deliberately not normalized
      here, because a second normalizer folding the opposite way from
      ``tool_usage.normalize_tool_name`` is precisely the trap that made the
      unnormalized comparisons unreachable in the first place.
    * **Missing vs corrupt.** A MISSING file is the legitimate "no tournament
      configured" state and yields an empty set. An unreadable or malformed
      file yields ``None`` -- reporting a tool as "NOT in the tournament" off
      a corrupt roster would mislead a deployment-review reader into
      re-adding a tool that already has a slot.

    :param path: path to ``tournament_tools.json``.
    :return: roster keys as written, ``set()`` when no roster is configured,
        or ``None`` when the file exists but cannot be read as an object.
    """
    if not path.is_file():
        return set()  # no tournament configured -- a real answer, not a failure
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning(
            "tournament roster %s is unreadable/corrupt; membership UNKNOWN.",
            path,
        )
        return None
    if not isinstance(data, dict):
        log.warning(
            "tournament roster %s is not a JSON object; membership UNKNOWN.",
            path,
        )
        return None
    return {str(k) for k in data}


def load_tournament_tools(path: Path = TOURNAMENT_TOOLS_JSON) -> dict[str, str]:
    """Load the tournament tool→CID map from JSON.

    :param path: path to tournament_tools.json. Defaults to the file shipped
        alongside this module.
    :return: dict mapping tool_name → IPFS CID.
    :raises FileNotFoundError: if the file is missing.
    :raises ValueError: if the file is malformed (not a dict[str, str]).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Tournament tools config not found: {path}. "
            "Tournament cannot run without it."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tournament tools config is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Tournament tools config must be a JSON object, got {type(data).__name__}: {path}"
        )
    for tool_name, cid in data.items():
        if not isinstance(tool_name, str) or not isinstance(cid, str):
            raise ValueError(
                f"Tournament tools config must map str→str; "
                f"got {tool_name!r}→{cid!r} in {path}"
            )
        if not tool_name or not cid:
            raise ValueError(
                f"Tournament tools config has empty tool_name or CID: "
                f"{tool_name!r}→{cid!r} in {path}"
            )
    return data
