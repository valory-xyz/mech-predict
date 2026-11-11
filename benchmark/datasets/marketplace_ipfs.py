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

import base64
import json
import os
from functools import lru_cache

import requests


def payload_path(digest: str, request_id: str | None = None) -> str:
    """Convert a sha256 digest to a CID; delivery files use decimal request ids."""
    raw = bytes.fromhex(digest.removeprefix("0x"))
    if len(raw) != 32:
        raise ValueError("Marketplace IPFS digest must be 32 bytes")
    cid = "b" + base64.b32encode(b"\x01\x70\x12\x20" + raw).decode().lower().rstrip("=")
    return cid if request_id is None else cid + "/" + str(int(request_id, 16))


def parsed_payload(content: str, digest: str, delivery: bool = False) -> dict:
    """Match the existing Graph parser's fields while retaining the full payload."""
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    sentinel = "[unhandled type]"

    def text(value):
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
        if first >= 0 and end > first:
            title = tail[first + 1 : end]
    return {
        "hash": payload_path(digest),
        "content": content,
        "prompt": prompt,
        "tool": text(payload.get("tool")),
        "questionTitle": title,
    }


@lru_cache(maxsize=4096)
def fetch_payload(digest: str, request_id: str | None = None) -> dict:
    """Try both supported layouts. Unavailable payloads raise, permitting retries."""
    path = payload_path(digest, request_id)
    gateway = os.environ.get(
        "IPFS_GATEWAY_URL", "https://gateway.autonolas.tech/ipfs"
    ).rstrip("/")
    # Directory-style payloads contain metadata.json; legacy files are bare.
    for suffix in ("/metadata.json", ""):
        response = requests.get(f"{gateway}/{path}{suffix}", timeout=60)
        if response.status_code == 404 and suffix:
            continue
        response.raise_for_status()
        return parsed_payload(response.text, digest, request_id is not None)
    raise RuntimeError("Marketplace IPFS payload unavailable")
