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
"""Standard-library-only Marketplace endpoint selection."""

import base64
import os


def is_squid(url: str) -> bool:
    """Recognise the explicitly configured Polygon endpoint."""
    configured = os.environ.get("MARKETPLACE_POLYGON_SQUID_URL")
    return bool(configured) and url == configured


def require_polygon_url(url: str) -> str:
    """Fail clearly before a Polygon fetch when configuration is missing."""
    if not url:
        raise ValueError(
            "MARKETPLACE_POLYGON_SQUID_URL is required for Polygon Marketplace reads"
        )
    return url


SUBGRAPH_UNHANDLED_TYPE = "[unhandled type]"


def ipfs_hash_to_cid(ipfs_hash: str) -> str:
    """Convert an IPFS hash from the subgraph to a base32 CIDv1.

    Handles two formats:
    - ``0x`` raw sha256 hash (from ``marketplaceDelivery.ipfsHashBytes``):
      wraps as CIDv1 (version=1, codec=dag-pb, sha256 multihash).
    - ``f``-prefixed hex CIDv1 (from ``mechDelivery.ipfsHash``):
      strips multibase prefix and re-encodes.

    The IPFS gateway expects base32 CIDv1 like ``bafybei...``.

    :param ipfs_hash: hex-encoded IPFS hash from the subgraph.
    :return: base32-encoded CIDv1 string.
    """
    if ipfs_hash.startswith("0x"):
        hash_bytes = bytes.fromhex(ipfs_hash[2:])
        # CIDv1: version=0x01, codec=0x70 (dag-pb), multihash=0x12 (sha256) + 0x20 (32 bytes) + hash
        cid_bytes = bytes([0x01, 0x70, 0x12, 0x20]) + hash_bytes
        return "b" + base64.b32encode(cid_bytes).decode().lower().rstrip("=")
    if ipfs_hash.startswith("f"):
        raw = bytes.fromhex(ipfs_hash[1:])
        return "b" + base64.b32encode(raw).decode().lower().rstrip("=")
    return ipfs_hash
