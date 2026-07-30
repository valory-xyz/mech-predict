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
"""Discover which tools our mechs actually SERVE, read on-chain.

The triage assesses whatever appears in the scored data, which includes
tools that live only in the tournament and are not deployed anywhere. A
fix issue on such a tool has no validation path (its PR cannot be
benchmarked from production rows) and no deployment consequence, so the
triage needs to know the deployed set.

A manifest committed here would rot on every deploy, so this module
recomputes it per run, from chain state only:

1. For each known mech service, read
   ``ComplementaryServiceMetadata.mapServiceHashes(serviceId)`` -> a
   bytes32 IPFS digest.
2. Rebuild the CIDv1 prefix (``f01701220`` + digest) and fetch the
   metadata manifest from the IPFS gateway.
3. The manifest's ``tools`` array is the served-tool list -- the same
   view the trader has when it picks a tool.

The only local constants are the service registry below: chain, service
id and the ComplementaryServiceMetadata address per mech. Those change
approximately never (unlike tool CIDs, which churn every deploy).

Read-only: one ``eth_call`` per service and one gateway GET per distinct
manifest. No writes, no transactions, no private repo access.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# The autonolas RPC/IPFS gateways reject requests without a browser-like
# User-Agent (HTTP 403), so set one explicitly on every call.
_USER_AGENT = "Mozilla/5.0 (mech-predict served-tools discovery)"

GATEWAY = os.environ.get("IPFS_GATEWAY_URL", "https://gateway.autonolas.tech/ipfs")
RPC_URLS: Dict[str, str] = {
    "gnosis": os.environ.get(
        "GNOSIS_RPC_URL", "https://rpc-gate.autonolas.tech/gnosis-rpc/"
    ),
    "polygon": os.environ.get(
        "POLYGON_RPC_URL", "https://rpc-gate.autonolas.tech/polygon-rpc/"
    ),
    "base": os.environ.get("BASE_RPC_URL", "https://rpc-gate.autonolas.tech/base-rpc/"),
    "optimism": os.environ.get(
        "OPTIMISM_RPC_URL", "https://rpc-gate.autonolas.tech/optimism-rpc/"
    ),
}

# Function selector: the first 4 bytes of the keccak-256 hash of the
# signature mapServiceHashes(uint256), verified against the deployed
# ComplementaryServiceMetadata contract.
MAP_SERVICE_HASHES_SELECTOR = "0x4a086201"

# Our mech services. Sourced from the deployment env files; kept here because
# they are stable identifiers (a service id and a registry address), NOT
# per-deploy state. Add a row when a new mech service is created.
MECH_SERVICES: Tuple[Dict[str, str], ...] = (
    {
        "name": "mech_mm_predict",
        "chain": "gnosis",
        "service_id": "2182",
        "csm": "0x0598081D48FB80B0A7E52FAD2905AE9beCd6fC69",
    },
    {
        "name": "mech_mm_predict_clone",
        "chain": "gnosis",
        "service_id": "2198",
        "csm": "0x0598081D48FB80B0A7E52FAD2905AE9beCd6fC69",
    },
    {
        "name": "single_mech_mm_predict",
        "chain": "gnosis",
        "service_id": "2235",
        "csm": "0x0598081D48FB80B0A7E52FAD2905AE9beCd6fC69",
    },
    {
        "name": "single_mech_mm_predict_2340",
        "chain": "gnosis",
        "service_id": "2340",
        "csm": "0x0598081D48FB80B0A7E52FAD2905AE9beCd6fC69",
    },
    {
        "name": "single_mech_mm_predict_2359",
        "chain": "gnosis",
        "service_id": "2359",
        "csm": "0x0598081D48FB80B0A7E52FAD2905AE9beCd6fC69",
    },
    {
        "name": "single_mech_mm_predict_2360",
        "chain": "gnosis",
        "service_id": "2360",
        "csm": "0x0598081D48FB80B0A7E52FAD2905AE9beCd6fC69",
    },
    {
        "name": "nvm_native_mech_mm_predict_2469",
        "chain": "gnosis",
        "service_id": "2469",
        "csm": "0x0598081D48FB80B0A7E52FAD2905AE9beCd6fC69",
    },
    {
        "name": "single_polygon_mech_mm_predict_21",
        "chain": "polygon",
        "service_id": "21",
        "csm": "0xDC175E77d11246c79B23D7088750eb59160DD6b7",
    },
    {
        "name": "polygon_mech_mm_predict_25",
        "chain": "polygon",
        "service_id": "25",
        "csm": "0xDC175E77d11246c79B23D7088750eb59160DD6b7",
    },
    {
        "name": "polygon_mech_mm_predict_44",
        "chain": "polygon",
        "service_id": "44",
        "csm": "0xDC175E77d11246c79B23D7088750eb59160DD6b7",
    },
)

# Platform each chain's mechs answer for, so the served set can be compared
# against a platform-scoped triage run.
CHAIN_PLATFORM: Dict[str, str] = {"gnosis": "omen", "polygon": "polymarket"}

TOOLS_REGISTRY_PATH = Path(__file__).resolve().parent / "tools.py"
TOURNAMENT_TOOLS_PATH = Path(__file__).resolve().parent / "tournament_tools.json"


def _rpc_call(rpc_url: str, to: str, data: str, timeout: int = 20) -> Optional[str]:
    """One ``eth_call``; returns the hex result or ``None`` on any failure.

    :param rpc_url: JSON-RPC endpoint.
    :param to: contract address.
    :param data: ABI-encoded calldata.
    :param timeout: seconds.
    :return: hex string result, or ``None``.
    """
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [{"to": to, "data": data}, "latest"],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        rpc_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # nosec B310
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.warning("eth_call failed on %s: %s", rpc_url, exc)
        return None
    if "error" in body:
        log.warning("eth_call returned an error: %s", body["error"])
        return None
    result = body.get("result")
    return result if isinstance(result, str) else None


def _fetch_manifest_tools(metadata_hash: str, timeout: int = 20) -> List[str]:
    """Fetch an IPFS metadata manifest and return its ``tools`` list.

    :param metadata_hash: CIDv1 hex form (``f01701220`` + digest).
    :param timeout: seconds.
    :return: served tool names (empty on any failure).
    """
    req = urllib.request.Request(
        f"{GATEWAY}/{metadata_hash}", headers={"User-Agent": _USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # nosec B310
            manifest = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.warning("IPFS fetch failed for %s: %s", metadata_hash, exc)
        return []
    tools = manifest.get("tools")
    return [str(t) for t in tools] if isinstance(tools, list) else []


def discover_served_tools(
    services: Tuple[Dict[str, str], ...] = MECH_SERVICES,
) -> Dict[str, Dict[str, Any]]:
    """Read the served-tool set of every known mech service, from chain.

    :param services: service registry rows (chain / service_id / csm).
    :return: ``{service_name: {chain, platform, service_id, metadata_hash,
        tools, error}}``; ``error`` is set when the service could not be
        read (its ``tools`` is then empty).
    """
    out: Dict[str, Dict[str, Any]] = {}
    manifest_cache: Dict[str, List[str]] = {}
    for service in services:
        name = service["name"]
        chain = service["chain"]
        entry: Dict[str, Any] = {
            "chain": chain,
            "platform": CHAIN_PLATFORM.get(chain, chain),
            "service_id": service["service_id"],
            "metadata_hash": None,
            "tools": [],
            "error": None,
        }
        rpc_url = RPC_URLS.get(chain)
        if not rpc_url:
            entry["error"] = f"no RPC configured for chain {chain!r}"
            out[name] = entry
            continue
        # mapServiceHashes(uint256) -> bytes32
        calldata = MAP_SERVICE_HASHES_SELECTOR + f"{int(service['service_id']):064x}"
        raw = _rpc_call(rpc_url, service["csm"], calldata)
        if not raw or raw == "0x" or set(raw[2:]) == {"0"}:
            entry["error"] = "no metadata hash on chain (0x0 or unreadable)"
            out[name] = entry
            continue
        metadata_hash = "f01701220" + raw[2:].rjust(64, "0")
        entry["metadata_hash"] = metadata_hash
        if metadata_hash not in manifest_cache:
            manifest_cache[metadata_hash] = _fetch_manifest_tools(metadata_hash)
        entry["tools"] = sorted(manifest_cache[metadata_hash])
        if not entry["tools"]:
            entry["error"] = "manifest unreadable or carries no tools"
        out[name] = entry
    return out


def repo_tools() -> Set[str]:
    """Tool names this repository can build, from ``benchmark/tools.py``.

    Parsed textually so importing the registry (and every tool module's
    dependencies) is not required.

    :return: registered tool names.
    """
    text = TOOLS_REGISTRY_PATH.read_text(encoding="utf-8")
    return set(re.findall(r'^\s*"([a-z0-9_.-]+)":\s*ToolSpec\(', text, re.M))


def tournament_tools() -> Set[str]:
    """Tool names currently on the tournament roster.

    :return: roster names (empty when the file is missing/corrupt).
    """
    try:
        data = json.loads(TOURNAMENT_TOOLS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return set(data) if isinstance(data, dict) else set()


def actionable_tools(
    served: Optional[Dict[str, Dict[str, Any]]] = None,
    platform: Optional[str] = None,
) -> Set[str]:
    """Tools that are BOTH served on-chain and buildable from this repo.

    That intersection is the actionable set: a tool the triage can open a
    fix issue for and whose fix has somewhere to land. Tools served but
    absent here cannot be edited; tools present here but unserved have no
    deployment consequence (they are tournament candidates at most).

    :param served: output of :func:`discover_served_tools`; discovered when
        omitted.
    :param platform: restrict to mechs answering for this platform.
    :return: tool names.
    """
    served = discover_served_tools() if served is None else served
    names: Set[str] = set()
    for entry in served.values():
        if platform and entry.get("platform") != platform:
            continue
        names.update(entry.get("tools") or [])
    return names & repo_tools()


def main() -> int:
    """CLI: print the served set, the repo set, and their intersection.

    :return: process exit code (1 if no service could be read).
    """
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--platform",
        default=None,
        help="restrict to one platform (omen / polymarket)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.WARNING)

    served = discover_served_tools()
    repo = repo_tools()
    roster = tournament_tools()
    actionable = actionable_tools(served, args.platform)

    if args.json:
        print(
            json.dumps(
                {
                    "served_by_service": served,
                    "repo_tools": sorted(repo),
                    "tournament_roster": sorted(roster),
                    "actionable": sorted(actionable),
                },
                indent=2,
            )
        )
        return 0 if any(e["tools"] for e in served.values()) else 1

    print("SERVED ON-CHAIN")
    served_all: Set[str] = set()
    for name, entry in served.items():
        if args.platform and entry["platform"] != args.platform:
            continue
        head = f"  {name}  [{entry['chain']}/{entry['platform']} service={entry['service_id']}]"
        if entry["error"]:
            print(f"{head}  -- {entry['error']}")
            continue
        print(head)
        for tool in entry["tools"]:
            mark = "" if tool in repo else "   (not in this repo)"
            print(f"      - {tool}{mark}")
        served_all.update(entry["tools"])

    print("\nACTIONABLE (served AND buildable here)")
    for tool in sorted(actionable):
        tags = []
        if tool in roster:
            tags.append("also in tournament")
        print(f"  * {tool}" + (f"   [{', '.join(tags)}]" if tags else ""))

    print("\nNOT ACTIONABLE")
    for tool in sorted(repo - served_all):
        where = "tournament only" if tool in roster else "repo only"
        print(f"  - {tool}   ({where})")
    for tool in sorted(served_all - repo):
        print(f"  - {tool}   (served but not buildable from this repo)")

    return 0 if any(e["tools"] for e in served.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
