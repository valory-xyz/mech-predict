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
"""Gateway failures must be row-local for permanent misses and retry transients."""

from unittest.mock import patch

import requests
from benchmark.datasets import marketplace_ipfs as ipfs

DIGEST = "0x" + "11" * 32


def response(status: int, text: str = '{"prompt":"p"}') -> requests.Response:
    """Build a real requests response so HTTP exception semantics are exercised."""
    result = requests.Response()
    result.status_code = status
    result._content = text.encode()  # pylint: disable=protected-access
    result.url = "https://ipfs.test/payload"
    return result


def test_delivery_fetches_bare_file_first_and_keeps_cid() -> None:
    """A delivery never probes beneath an IPFS file."""
    ipfs.fetch_payload.cache_clear()
    with patch.object(
        ipfs.requests, "get", return_value=response(200, '{"result":"ok"}')
    ) as get:
        payload = ipfs.fetch_payload(DIGEST, "0x01")
    assert payload is not None
    assert (
        payload["hash"] == "bafybeiarceirceirceirceirceirceirceirceirceirceirceirceirce"
    )
    assert get.call_count == 1
    assert get.call_args.args[0].endswith("/1")


def test_request_404_falls_back_then_returns_missing() -> None:
    """Two definitive misses yield null rather than aborting the page."""
    ipfs.fetch_payload.cache_clear()
    with patch.object(
        ipfs.requests, "get", side_effect=[response(404), response(404)]
    ) as get:
        assert ipfs.fetch_payload(DIGEST) is None
    assert get.call_count == 2
    assert get.call_args_list[0].args[0].endswith("/metadata.json")
    assert not get.call_args_list[1].args[0].endswith("/metadata.json")


def test_transient_gateway_failure_is_retried() -> None:
    """A temporary 503 cannot turn a valid payload into a permanent miss."""
    ipfs.fetch_payload.cache_clear()
    with (
        patch.object(
            ipfs.requests, "get", side_effect=[response(503), response(200)]
        ) as get,
        patch.object(ipfs.time, "sleep"),
    ):
        assert ipfs.fetch_payload(DIGEST) is not None
    assert get.call_count == 2


def test_bad_digest_does_not_make_gateway_request() -> None:
    """Malformed source digests behave like absent Graph payloads."""
    ipfs.fetch_payload.cache_clear()
    with patch.object(ipfs.requests, "get") as get:
        assert ipfs.fetch_payload("0x11") is None
    get.assert_not_called()
