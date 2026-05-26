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
"""Resolve the per-deployment set of selectable tools so the daily report can annotate them.

The trader no longer filters tools with a per-tool allow/deny list. Instead
each deployment's ``service.yaml`` ships a ``valid_mechs`` allow-list of mech
contract addresses, and the trader may select **any** tool offered by those
mechs in the marketplace. We therefore resolve the allow-list dynamically:

1. Find the latest ``valory-xyz/trader`` release tag.
2. Read each deployment's ``service.yaml`` at that tag and parse ``valid_mechs``.
3. Resolve each mech address to its tools via the marketplace subgraph
   (mech → on-chain metadata CID → IPFS metadata ``tools`` list).
4. A deployment's selectable tools are the union across its ``valid_mechs``.

We return a ``{deployment: [tool_names] | None}`` map; ``None`` signals a fetch
or parse failure for that deployment so consumers can render "unavailable"
rather than falsely claiming a tool is (or is not) selectable.
"""

from __future__ import annotations

import json
import logging
import re
from types import MappingProxyType
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

# Single source of truth for every deployment we report on. Each entry maps
# the deployment name to ``(trader_service_dir, chain, platform)``:
#   * ``trader_service_dir`` selects which trader ``service.yaml`` to read
#     for ``valid_mechs`` at the latest trader release.
#   * ``chain`` selects the marketplace subgraph the mech addresses live on.
#   * ``platform`` is the scorer key used to filter the report section.
# Adding a new deployment requires editing exactly this one structure; the
# public ``DEPLOYMENTS`` tuple and ``DEPLOYMENT_TO_PLATFORM`` mapping below
# are derived from it.
_DEPLOYMENTS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        "omenstrat Pearl": ("trader_pearl", "gnosis", "omen"),
        "polystrat Pearl": ("polymarket_trader", "polygon", "polymarket"),
    }
)

# Deployments we report on. Order is the column order in the rendered line.
DEPLOYMENTS: tuple[str, ...] = tuple(_DEPLOYMENTS)

# Maps each deployment name to the scorer's platform key. Drives the
# per-platform filter on the Tool Deployment Status section.
DEPLOYMENT_TO_PLATFORM: Mapping[str, str] = MappingProxyType(
    {name: platform for name, (_service, _chain, platform) in _DEPLOYMENTS.items()}
)


def deployments_for_platform(platform: str) -> tuple[str, ...]:
    """Return the deployment names belonging to ``platform``, in declared order.

    :param platform: scorer platform key (``"omen"`` or ``"polymarket"``).
    :return: deployments matching ``platform``, preserving ``DEPLOYMENTS`` order.
    """
    return tuple(
        name for name in DEPLOYMENTS if DEPLOYMENT_TO_PLATFORM.get(name) == platform
    )


# Source URLs.
# Latest published release (pre-releases intentionally included: the trader
# is shipped via pre-release tags like ``v0.38.7-rc1`` and that is what is
# actually deployed). ``per_page=1`` keeps the response to one element, so
# ``[0]`` is the newest tag. Any network / parse failure here degrades the
# whole Tool Deployment Status section to ``⚠️ unavailable`` (never raises).
TRADER_RELEASES_URL = (
    "https://api.github.com/repos/valory-xyz/trader/releases?per_page=1"
)
# Filled with the resolved release tag and the trader service dir.
TRADER_SERVICE_YAML_URL = (
    "https://raw.githubusercontent.com/valory-xyz/trader/"
    "{ref}/packages/valory/services/{service}/service.yaml"
)
# Olas Mech Marketplace subgraphs, per chain.
MARKETPLACE_SUBGRAPH_URL: Mapping[str, str] = MappingProxyType(
    {
        "gnosis": "https://api.subgraph.autonolas.tech/api/proxy/marketplace-gnosis",
        "polygon": "https://api.subgraph.autonolas.tech/api/proxy/marketplace-polygon",
    }
)
# IPFS gateway and the CIDv1 prefix mech-interact prepends to a metadata hash.
IPFS_GATEWAY_URL = "https://gateway.autonolas.tech/ipfs"
CID_PREFIX = "f01701220"

# Fetch timeout (seconds). Short so a stalled endpoint never blocks the
# daily report pipeline.
FETCH_TIMEOUT = 30

# Matches the ``valid_mechs`` env-override default in a trader ``service.yaml``:
# ``valid_mechs: ${VALID_MECHS:list:["0x..","0x.."]}`` on a single line. The
# inner ``[...]`` is valid JSON (array of double-quoted addresses); ``[^\]]*``
# is safe because addresses never contain ``]``.
_VALID_MECHS_RE = re.compile(
    r"valid_mechs\s*:\s*\$\{VALID_MECHS:list:(?P<list>\[[^\]]*\])\}"
)

# Subgraph query: resolve a set of mech addresses to their on-chain metadata
# CID. ``meches`` is keyed internally by id but carries the contract
# ``address``; ``{addrs}`` is rendered with a comma-separated list of quoted
# addresses at call time via ``.format(addrs=...)``.
_MECHES_QUERY = (
    "{{ meches(first: 1000, where: {{address_in: [{addrs}]}}) "
    "{{ address service {{ metadata {{ metadata }} }} }} }}"
)


def _normalize_tool_name(name: str) -> str:
    """Return a canonical form for cross-convention tool-name matching.

    Mech metadata and the benchmark logs sometimes spell the same tool
    ``prediction_request_X`` vs ``prediction-request-X``. Treat underscores
    and hyphens as interchangeable so we don't under-report.

    :param name: raw tool name.
    :return: canonical form with ``_`` replaced by ``-``.
    """
    return name.replace("_", "-")


def _http_get(url: str) -> str:
    """GET ``url`` and return the body as text.

    Propagates ``URLError`` (or subclasses) on network failure so callers
    can decide how to degrade.

    :param url: HTTP(S) URL to fetch.
    :return: response body decoded as UTF-8.
    """
    req = Request(url, headers={"User-Agent": "mech-predict-benchmark"})
    with urlopen(
        req, timeout=FETCH_TIMEOUT
    ) as resp:  # nosec B310 — fixed registry/subgraph/IPFS URLs
        return resp.read().decode("utf-8")


def _post_graphql(url: str, query: str) -> dict[str, Any]:
    """POST a GraphQL ``query`` to ``url`` and return the ``data`` object.

    Propagates ``URLError`` (or subclasses) on network failure.

    :param url: GraphQL endpoint.
    :param query: GraphQL query string.
    :return: the ``data`` object from the JSON response.
    :raises ValueError: when the response reports GraphQL ``errors`` or when
        ``data`` is missing / not a JSON object. Spec-compliant subgraphs
        always send a dict; defending here converts a hypothetical
        ``AttributeError`` from downstream ``data.get(...)`` calls into the
        normal ``None``-fallback path in ``fetch_valid_tools``.
    """
    body = json.dumps({"query": query}).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "mech-predict-benchmark",
        },
        method="POST",
    )
    with urlopen(
        req, timeout=FETCH_TIMEOUT
    ) as resp:  # nosec B310 — fixed marketplace subgraph URLs
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("errors"):
        raise ValueError(f"GraphQL errors from {url}: {payload['errors']}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"unexpected data type from {url}: {type(data).__name__}")
    return data


def _parse_json_string_list(raw: str) -> list[str]:
    """Parse a JSON-encoded array-of-strings.

    :param raw: JSON document expected to be an array of strings.
    :return: the parsed list.
    :raises ValueError: if ``raw`` is not a JSON array of strings.
    """
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError("expected a JSON array of strings")
    return parsed


def latest_trader_ref() -> str:
    """Return the tag of the most recent ``valory-xyz/trader`` release.

    The releases endpoint lists published releases (pre-releases included)
    newest-first, so ``[0].tag_name`` is the latest deployable trader version.
    Propagates ``URLError`` (or subclasses) on network failure.

    :return: the release tag (e.g. ``"v0.38.0-rc1"``).
    :raises ValueError: when no release or no ``tag_name`` is present.
    """
    releases = json.loads(_http_get(TRADER_RELEASES_URL))
    if not isinstance(releases, list) or not releases:
        raise ValueError("no trader releases found")
    tag = releases[0].get("tag_name")
    if not isinstance(tag, str) or not tag:
        raise ValueError("latest trader release has no tag_name")
    return tag


def parse_valid_mechs(yaml_source: str) -> list[str]:
    """Extract the ``valid_mechs`` allow-list from a trader ``service.yaml``.

    Addresses are returned lowercased so every downstream caller receives a
    canonical-cased value without needing to know about the subgraph's
    lowercase normalisation convention.

    :param yaml_source: full text of a trader ``service.yaml``.
    :return: list of mech contract addresses (empty list is a real state:
        "no mechs allow-listed").
    :raises ValueError: when the ``valid_mechs`` env-default line is absent
        or its payload is not a JSON array of strings.
    """
    match = _VALID_MECHS_RE.search(yaml_source)
    if match is None:
        raise ValueError("no valid_mechs env default found in service.yaml")
    return [addr.lower() for addr in _parse_json_string_list(match.group("list"))]


def fetch_tools_for_metadata(metadata_hash: str) -> list[str]:
    """Fetch the tools manifest for one mech metadata hash from IPFS.

    The subgraph stores the metadata as a hex hash; mech-interact resolves
    it to a CIDv1 by prepending ``CID_PREFIX`` (after dropping any ``0x``).
    Propagates ``URLError`` (or subclasses) on network failure.

    :param metadata_hash: hex metadata hash from the subgraph
        (e.g. ``0x204398...``).
    :return: the tool names advertised in the manifest's ``tools`` array.
    :raises ValueError: when the manifest has no JSON ``tools`` array of strings.
    """
    digest = metadata_hash[2:] if metadata_hash.startswith("0x") else metadata_hash
    url = f"{IPFS_GATEWAY_URL}/{CID_PREFIX}{digest}"
    manifest = json.loads(_http_get(url))
    tools = manifest.get("tools") if isinstance(manifest, dict) else None
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise ValueError(f"metadata {metadata_hash} has no tools array of strings")
    return tools


def resolve_mech_tools(addresses: list[str], subgraph_url: str) -> list[str]:
    """Resolve a deployment's ``valid_mechs`` to the union of tools they offer.

    Looks up each address' on-chain metadata CID via the marketplace
    subgraph, then fetches each distinct manifest from IPFS and unions the
    advertised tools. Distinct metadata hashes are de-duplicated so mechs
    sharing a manifest are fetched once. Propagates ``URLError`` on a
    subgraph/IPFS network failure and ``ValueError`` on a GraphQL error or
    a malformed IPFS manifest, so the caller can mark the deployment
    unavailable.

    Two cases are treated as hard failures so the deployment falls back to
    ``⚠️ unavailable`` rather than to a silently shrunk allow-list:

    * The subgraph response omits any requested address (typo'd
      ``valid_mechs`` entry, mech not yet indexed, address on the wrong
      chain, etc.).
    * Every requested address is returned but none carries a usable
      on-chain metadata hash (``service`` is ``null`` or ``metadata`` is
      empty) — without this guard the deployment would resolve to ``[]``
      and render "no benchmarked tools active", indistinguishable from a
      legitimately empty ``valid_mechs``.

    :param addresses: mech contract addresses (the parsed ``valid_mechs``).
    :param subgraph_url: marketplace subgraph endpoint for the chain.
    :return: sorted union of advertised tool names; ``[]`` when ``addresses``
        is empty (a real "no mechs allow-listed" state, not a failure).
    :raises ValueError: when the subgraph does not return every requested
        address (the message lists the missing ones), or when the resolved
        mechs collectively yield no on-chain metadata hashes.
    """
    if not addresses:
        return []
    quoted = ",".join(f'"{addr.lower()}"' for addr in addresses)
    data = _post_graphql(subgraph_url, _MECHES_QUERY.format(addrs=quoted))

    meches = data.get("meches") or []
    returned = {(mech.get("address") or "").lower() for mech in meches}
    requested = {addr.lower() for addr in addresses}
    missing = requested - returned
    if missing:
        raise ValueError(
            "subgraph did not return mech address(es): " + ", ".join(sorted(missing))
        )

    metadata_hashes: set[str] = set()
    for mech in meches:
        manifests = (mech.get("service") or {}).get("metadata") or []
        if manifests and manifests[0].get("metadata"):
            metadata_hashes.add(manifests[0]["metadata"])

    if not metadata_hashes:
        raise ValueError(
            "subgraph returned all addresses but none have on-chain metadata"
        )

    tools: set[str] = set()
    for metadata_hash in metadata_hashes:
        tools.update(fetch_tools_for_metadata(metadata_hash))
    return sorted(tools)


def fetch_valid_tools() -> dict[str, list[str] | None]:
    """Return ``{deployment: [selectable_tool_names] | None}`` for all deployments.

    For each deployment: read its ``service.yaml`` at the latest trader
    release, parse ``valid_mechs``, and resolve those mechs to the union of
    tools they advertise in the marketplace. ``None`` signals a fetch or
    parse failure for that deployment so the renderer can say "unavailable"
    instead of falsely claiming a tool is or is not selectable. Failures are
    logged but never raised — the daily report must never be blocked by a
    flaky GitHub/subgraph/IPFS fetch.

    :return: selectable-tools map keyed by deployment name.
    """
    valid: dict[str, list[str] | None] = {name: None for name in DEPLOYMENTS}

    try:
        ref = latest_trader_ref()
    except (URLError, ValueError, OSError) as exc:
        log.warning("trader release resolution failed: %s", exc)
        return valid  # every deployment stays None

    for deployment, (service, chain, _platform) in _DEPLOYMENTS.items():
        try:
            yaml_source = _http_get(
                TRADER_SERVICE_YAML_URL.format(ref=ref, service=service)
            )
            mechs = parse_valid_mechs(yaml_source)
            valid[deployment] = resolve_mech_tools(
                mechs, MARKETPLACE_SUBGRAPH_URL[chain]
            )
        except (URLError, ValueError, OSError) as exc:
            log.warning("%s selectable-tools resolution failed: %s", deployment, exc)

    return valid
