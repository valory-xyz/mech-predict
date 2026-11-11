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
"""Marketplace squid query and payload compatibility regressions."""

import json
from unittest.mock import patch

import pytest
from graphql import parse

from benchmark.datasets.marketplace_ipfs import parsed_payload, payload_path
from benchmark.datasets.marketplace_squid import hydrate, is_squid, squid_query


def test_only_configured_endpoint_uses_squid(monkeypatch):
    monkeypatch.setenv("MARKETPLACE_POLYGON_SQUID_URL", "https://squid.test")
    assert is_squid("https://squid.test")
    assert not is_squid("https://graph.test")


def test_delivery_page_keeps_stable_order_and_response_keys():
    result = squid_query("""{ delivers(first: 2, skip: 4, orderBy: blockTimestamp,
      orderDirection: desc, where: {blockTimestamp_gt: 10}) {
        id request { id parsedRequest { questionTitle tool content } }
    } }""")
    parse(result)
    assert "limit: 2" in result
    assert "offset: 4" in result
    assert "blockTimestamp_DESC, id_DESC" in result
    assert "parsedRequest" not in result
    with patch(
        "benchmark.datasets.marketplace_squid.fetch_payload",
        return_value={"questionTitle": "Q"},
    ):
        data = hydrate(
            {
                "delivers": [
                    {
                        "id": "d",
                        "request": {
                            "id": "r",
                            "_squidRequest": {"ipfsHashBytes": "digest"},
                        },
                    }
                ]
            }
        )
    assert data["delivers"][0]["request"]["parsedRequest"]["questionTitle"] == "Q"


def test_payload_failure_propagates_before_page_can_be_consumed():
    with patch(
        "benchmark.datasets.marketplace_squid.fetch_payload",
        side_effect=RuntimeError("offline"),
    ):
        with pytest.raises(RuntimeError, match="offline"):
            hydrate(
                {
                    "delivers": [
                        {"request": {"_squidRequest": {"ipfsHashBytes": "digest"}}}
                    ]
                }
            )


def test_ipfs_paths_and_question_parser_match_graph():
    assert payload_path("0x" + "ab" * 32, "0x10").endswith("/16")
    row = parsed_payload(
        json.dumps(
            {
                "prompt": 'With the given question "Will X say "yes"?" and the rest',
                "tool": "prediction",
            }
        ),
        "0x" + "ab" * 32,
    )
    assert row["questionTitle"] == 'Will X say "yes"?'


def test_delivery_identifiers_preserve_existing_shard_dedup():
    from benchmark.datasets.marketplace_squid import (
        graph_delivery_id,
        squid_delivery_id,
    )
    tx = "0x" + "ab" * 32
    request_id = "0x" + "cd" * 32
    for squid_id, old_id in [(tx + "-258", tx + "02010000"),
                              (tx + "-" + request_id, tx + request_id[2:])]:
        assert graph_delivery_id(squid_id) == old_id
        assert squid_delivery_id(old_id) == squid_id
        query = squid_query('{ delivers(first: 1, where: {id_in: ["' + old_id + '"]}) { id } }')
        assert squid_id in query
        assert hydrate({"delivers": [{"id": squid_id}]})["delivers"][0]["id"] == old_id
