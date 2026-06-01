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
from pathlib import Path

# ---------------------------------------------------------------------------
# Tournament tools — single source of truth (TOURNAMENT_IPFS_LOADER_SPEC §3.1)
# ---------------------------------------------------------------------------

TOURNAMENT_TOOLS_JSON = Path(__file__).resolve().parent / "tournament_tools.json"


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
