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
import subprocess
import sys
from unittest.mock import patch

import pytest
from graphql import parse

# isort: split

from benchmark.datasets.marketplace_ipfs import parsed_payload, payload_path
from benchmark.datasets.marketplace_squid import (
    graph_delivery_id,
    hydrate,
    is_squid,
    squid_delivery_id,
    squid_query,
)


def test_only_configured_endpoint_uses_squid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the explicitly configured endpoint selects OpenReader."""
    monkeypatch.setenv("MARKETPLACE_POLYGON_SQUID_URL", "https://squid.test")
    assert is_squid("https://squid.test")
    assert not is_squid("https://graph.test")


def test_delivery_page_keeps_stable_order_and_response_keys() -> None:
    """Collection translation preserves deterministic pages and parsed row keys."""
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


def test_payload_failure_propagates_before_page_can_be_consumed() -> None:
    """Payload failures prevent a successfully hydrated partial page."""
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


def test_ipfs_paths_and_question_parser_match_graph() -> None:
    """IPFS paths and quoted question parsing match existing Graph behavior."""
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


def test_delivery_identifiers_preserve_existing_shard_dedup() -> None:
    """Both delivery ID encodings round-trip without changing shard row IDs."""

    tx = "0x" + "ab" * 32
    request_id = "0x" + "cd" * 32
    for squid_id, old_id in [
        (tx + "-258", tx + "02010000"),
        (tx + "-" + request_id, tx + request_id[2:]),
    ]:
        assert graph_delivery_id(squid_id) == old_id
        assert squid_delivery_id(old_id) == squid_id
        query = squid_query(
            '{ delivers(first: 1, where: {id_in: ["' + old_id + '"]}) { id } }'
        )
        assert squid_id in query
        assert hydrate({"delivers": [{"id": squid_id}]})["delivers"][0]["id"] == old_id


def test_parsed_deliveries_translation_and_missing_payload_are_row_local() -> None:
    """The by-ID query joins the delivery layout, and missing siblings do not abort."""
    query = squid_query(
        '{ parsedDeliveries(first: 1000, where: {id_in: ["old"]}) { id content } }'
    )
    assert "marketplaceDelivery" in query and "requestIdBytes" in query
    with patch(
        "benchmark.datasets.marketplace_squid.fetch_payload",
        side_effect=lambda digest, request_id=None: (
            None if digest == "missing" else {"content": "valid"}
        ),
    ) as fetch:
        result = hydrate(
            {
                "parsedDeliveries": [
                    {
                        "id": "a",
                        "_squidDelivery": {
                            "ipfsHashBytes": "good",
                            "requestIdBytes": "0x01",
                        },
                    },
                    {
                        "id": "b",
                        "_squidDelivery": {
                            "ipfsHashBytes": "missing",
                            "requestIdBytes": "0x02",
                        },
                    },
                ]
            }
        )
    assert result == {"parsedDeliveries": [{"id": "a", "content": "valid"}]}
    assert {call.args for call in fetch.call_args_list} == {
        ("good", "0x01"),
        ("missing", "0x02"),
    }


def test_stdlib_tools_import_without_optional_dependencies() -> None:
    """ROI imports must work when site-packages are unavailable."""
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import benchmark.tool_usage; import benchmark.roi_sim",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_missing_polygon_url_fails_before_fetching_or_writing() -> None:
    """Misconfiguration must name the env before any platform work starts."""
    from benchmark.datasets import (
        fetch_production as fp,  # pylint: disable=import-outside-toplevel
    )

    with (
        patch.object(sys, "argv", ["fetch_production"]),
        patch.object(fp, "MECH_MARKETPLACE_POLYGON_URL", ""),
        patch.object(fp, "process_platform") as process,
        patch.object(fp, "append_rows") as append,
        pytest.raises(ValueError, match="MARKETPLACE_POLYGON_SQUID_URL"),
    ):
        fp.main()
    process.assert_not_called()
    append.assert_not_called()
