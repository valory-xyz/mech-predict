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
"""Read Marketplace IPFS payloads without relying on Graph file data sources."""

import json
import os
import time
from functools import lru_cache

import requests
from benchmark.datasets.marketplace_config import (
    SUBGRAPH_UNHANDLED_TYPE,
    ipfs_hash_to_cid,
)


def payload_path(digest: str, request_id: str | None = None) -> str:
    """Convert a sha256 digest to a CID; delivery files use decimal request ids."""
    raw = bytes.fromhex(digest.removeprefix("0x"))
    if len(raw) != 32:
        raise ValueError("Marketplace IPFS digest must be 32 bytes")
    cid = ipfs_hash_to_cid("0x" + raw.hex())
    return cid if request_id is None else cid + "/" + str(int(request_id, 16))


def parsed_payload(content: str, digest: str, delivery: bool = False) -> dict:
    """Match the existing Graph parser's fields while retaining the full payload."""
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    sentinel = SUBGRAPH_UNHANDLED_TYPE

    def text(value: object) -> str:
        return value if isinstance(value, str) else sentinel

    if delivery:
        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        return {
            "hash": payload_path(digest),
            "content": content,
            "response": text(payload.get("result")),
            "model": text(metadata.get("model")),
            "tool": text(metadata.get("tool")),
            "toolHash": text(metadata.get("tool_hash")),
        }
    prompt = text(payload.get("prompt"))
    # Same delimiter as the Graph request parser; inner quotes are retained.
    title = ""
    marker = "With the given question"
    if marker in prompt:
        tail = prompt.split(marker, 1)[1]
        first = tail.find('"')
        end = tail.find('" and the', first + 1)
        if 0 <= first < end:
            title = tail[first + 1 : end]
    return {
        "hash": payload_path(digest),
        "content": content,
        "prompt": prompt,
        "tool": text(payload.get("tool")),
        "questionTitle": title,
    }


@lru_cache(maxsize=4096)
def fetch_payload(digest: str, request_id: str | None = None) -> dict | None:
    """Return None for absent/malformed digests; retry transient gateway failures.

    A permanent miss is row-local, just like a null Graph parsed payload.
    Exhausted transient failures still raise before a page can be committed.
    Malformed content retains Graph sentinel fields and its original body.

    :param digest: sha256 digest from the squid, as hex.
    :param request_id: hex request id for a delivery, or None for a request.
    :return: parsed payload, or None for an absent payload or malformed digest.
    :raises RuntimeError: if transient gateway failures exhaust retries.
    :raises requests.RequestException: if the gateway rejects the request.
    """
    try:
        path = payload_path(digest, request_id)
    except (ValueError, TypeError, AttributeError):
        return None
    gateway = os.environ.get(
        "IPFS_GATEWAY_URL", "https://gateway.autonolas.tech/ipfs"
    ).rstrip("/")
    suffixes = ("",) if request_id is not None else ("/metadata.json", "")
    for suffix in suffixes:
        for attempt in range(4):
            try:
                response = requests.get(f"{gateway}/{path}{suffix}", timeout=60)
                if response.status_code in (400, 404, 410):
                    break
                response.raise_for_status()
                return parsed_payload(response.text, digest, request_id is not None)
            except requests.RequestException as error:
                status = (
                    error.response.status_code
                    if isinstance(error, requests.HTTPError)
                    and error.response is not None
                    else None
                )
                if status is not None and status != 429 and status < 500:
                    raise
                if attempt == 3:
                    raise RuntimeError(
                        f"IPFS fetch failed for digest {digest}, request {request_id}"
                    ) from error
                time.sleep(2**attempt)
    return None
