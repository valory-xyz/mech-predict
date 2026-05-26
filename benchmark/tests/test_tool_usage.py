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
"""Tests for benchmark/tool_usage.py and the deployment-status report section."""

import json
from typing import Callable
from urllib.error import URLError

import pytest
from benchmark import tool_usage
from benchmark.analyze import section_tool_deployment_status


class TestDeploymentConfigInvariants:
    """Lock invariants on the single ``_DEPLOYMENTS`` source of truth.

    With ``DEPLOYMENTS`` and ``DEPLOYMENT_TO_PLATFORM`` derived directly
    from ``_DEPLOYMENTS``, drift between those three views is no longer
    possible by construction. The remaining real invariant is that every
    chain referenced in ``_DEPLOYMENTS`` has a marketplace subgraph URL —
    adding a new chain without wiring up the URL would crash
    ``fetch_valid_tools`` at lookup time.
    """

    def test_every_chain_has_a_subgraph_url(self) -> None:
        """Each deployment's chain has a marketplace subgraph URL."""
        deployments = tool_usage._DEPLOYMENTS  # pylint: disable=protected-access
        for name, (_service, chain, _platform) in deployments.items():
            assert (
                chain in tool_usage.MARKETPLACE_SUBGRAPH_URL
            ), f"{name} resolves on chain {chain!r} with no subgraph URL"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _service_yaml(addresses: list[str]) -> str:
    """Build a trader service.yaml stub carrying a valid_mechs env default."""
    payload = json.dumps(addresses)
    return (
        "    models:\n"
        "      params:\n"
        "        args:\n"
        "          some_other_param: ${SOME_OTHER:int:5}\n"
        f"          valid_mechs: ${{VALID_MECHS:list:{payload}}}\n"
        "          trailing_param: ${TRAILING:bool:true}\n"
    )


def _scores_with_tools(tool_names: list[str]) -> dict:
    """Build a minimal scores dict with the given tools in by_tool."""
    return {
        "generated_at": "2026-04-14T06:00:00Z",
        "total_rows": 10,
        "valid_rows": 10,
        "overall": {"brier": 0.25, "reliability": 1.0, "n": 10},
        "by_tool": {
            name: {"brier": 0.2 + idx * 0.01, "n": 10}
            for idx, name in enumerate(tool_names)
        },
    }


# ---------------------------------------------------------------------------
# latest_trader_ref
# ---------------------------------------------------------------------------


class TestLatestTraderRef:
    """Tests for latest_trader_ref."""

    def test_returns_latest_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """First (newest) release's tag_name is returned; URL pins ``per_page=1``."""
        seen: dict[str, str] = {}

        def fake_get(url: str) -> str:
            seen["url"] = url
            return json.dumps([{"tag_name": "v1.2.3-rc1"}, {"tag_name": "v1.2.2"}])

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        assert tool_usage.latest_trader_ref() == "v1.2.3-rc1"
        assert "per_page=1" in seen["url"]

    def test_empty_release_list_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No releases -> ValueError (so the caller degrades, not crashes)."""
        monkeypatch.setattr(tool_usage, "_http_get", lambda url: "[]")
        with pytest.raises(ValueError, match="no trader releases"):
            tool_usage.latest_trader_ref()

    def test_missing_tag_name_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A release object without tag_name -> ValueError."""
        monkeypatch.setattr(
            tool_usage, "_http_get", lambda url: json.dumps([{"id": 1}])
        )
        with pytest.raises(ValueError, match="no tag_name"):
            tool_usage.latest_trader_ref()


# ---------------------------------------------------------------------------
# parse_valid_mechs
# ---------------------------------------------------------------------------


class TestParseValidMechs:
    """Tests for parse_valid_mechs."""

    def test_happy_path(self) -> None:
        """The address list is extracted in order and lowercased at the boundary."""
        yaml_source = _service_yaml(["0xAAA", "0xbBB"])
        assert tool_usage.parse_valid_mechs(yaml_source) == ["0xaaa", "0xbbb"]

    def test_empty_list_returns_empty(self) -> None:
        """An empty allow-list is a real state ([], not an error)."""
        assert tool_usage.parse_valid_mechs(_service_yaml([])) == []

    def test_missing_valid_mechs_raises(self) -> None:
        """Absent valid_mechs line -> ValueError (no silent empty result)."""
        with pytest.raises(ValueError, match="no valid_mechs"):
            tool_usage.parse_valid_mechs("models:\n  params: {}\n")

    def test_non_string_items_raise(self) -> None:
        """A non-string array element -> ValueError."""
        bad = "      valid_mechs: ${VALID_MECHS:list:[1,2,3]}\n"
        with pytest.raises(ValueError, match="array of strings"):
            tool_usage.parse_valid_mechs(bad)


# ---------------------------------------------------------------------------
# fetch_tools_for_metadata
# ---------------------------------------------------------------------------


class TestFetchToolsForMetadata:
    """Tests for fetch_tools_for_metadata."""

    def test_parses_tools_and_builds_cid_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """0x is stripped and CID_PREFIX prepended; tools array is returned."""
        seen: dict[str, str] = {}

        def fake_get(url: str) -> str:
            seen["url"] = url
            return json.dumps({"name": "Mech", "tools": ["a", "b"]})

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        tools = tool_usage.fetch_tools_for_metadata("0xdeadbeef")
        assert tools == ["a", "b"]
        assert seen["url"] == (
            f"{tool_usage.IPFS_GATEWAY_URL}/{tool_usage.CID_PREFIX}deadbeef"
        )

    def test_missing_tools_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A manifest with no tools array -> ValueError."""
        monkeypatch.setattr(
            tool_usage, "_http_get", lambda url: json.dumps({"name": "Mech"})
        )
        with pytest.raises(ValueError, match="no tools array"):
            tool_usage.fetch_tools_for_metadata("0xabc")

    def test_non_string_tools_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A tools array with non-string items -> ValueError."""
        monkeypatch.setattr(
            tool_usage, "_http_get", lambda url: json.dumps({"tools": [1, 2]})
        )
        with pytest.raises(ValueError, match="no tools array"):
            tool_usage.fetch_tools_for_metadata("0xabc")

    def test_html_response_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """IPFS gateway returning HTML (cache miss / 504) -> ValueError (caught upstream)."""
        monkeypatch.setattr(tool_usage, "_http_get", lambda url: "<html>504</html>")
        with pytest.raises(ValueError):
            tool_usage.fetch_tools_for_metadata("0xabc")


# ---------------------------------------------------------------------------
# resolve_mech_tools
# ---------------------------------------------------------------------------


class TestResolveMechTools:
    """Tests for resolve_mech_tools."""

    def test_empty_addresses_makes_no_network_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty allow-list short-circuits to [] without touching the subgraph."""

        def boom(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("network must not be touched for empty allow-list")

        monkeypatch.setattr(tool_usage, "_post_graphql", boom)
        monkeypatch.setattr(tool_usage, "fetch_tools_for_metadata", boom)
        assert tool_usage.resolve_mech_tools([], "http://subgraph") == []

    def test_unions_and_dedupes_shared_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tools across mechs are unioned; a shared metadata hash is fetched once."""
        monkeypatch.setattr(
            tool_usage,
            "_post_graphql",
            lambda url, query: {
                "meches": [
                    {"address": "0xa", "service": {"metadata": [{"metadata": "0x11"}]}},
                    {"address": "0xb", "service": {"metadata": [{"metadata": "0x11"}]}},
                    {"address": "0xc", "service": {"metadata": [{"metadata": "0x22"}]}},
                ]
            },
        )
        fetched: list[str] = []

        def fake_tools(metadata_hash: str) -> list[str]:
            fetched.append(metadata_hash)
            return {"0x11": ["superforcaster", "factual_research"], "0x22": ["jury"]}[
                metadata_hash
            ]

        monkeypatch.setattr(tool_usage, "fetch_tools_for_metadata", fake_tools)
        tools = tool_usage.resolve_mech_tools(["0xa", "0xb", "0xc"], "http://subgraph")
        # Sorted union across all distinct manifests.
        assert tools == ["factual_research", "jury", "superforcaster"]
        # The shared 0x11 manifest is fetched exactly once (dedup), not twice.
        assert sorted(fetched) == ["0x11", "0x22"]

    def test_query_lowercases_and_quotes_addresses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The subgraph query embeds the addresses, lowercased and quoted."""
        captured: dict[str, str] = {}

        def fake_post(url: str, query: str) -> dict:
            captured["query"] = query
            # Return both requested addresses *with* metadata so resolve_mech_tools
            # doesn't short-circuit on missing-address / empty-metadata
            # ValueErrors before we get to inspect the recorded query string.
            return {
                "meches": [
                    {
                        "address": "0xabc",
                        "service": {"metadata": [{"metadata": "0x11"}]},
                    },
                    {
                        "address": "0xdef",
                        "service": {"metadata": [{"metadata": "0x11"}]},
                    },
                ]
            }

        monkeypatch.setattr(tool_usage, "_post_graphql", fake_post)
        monkeypatch.setattr(tool_usage, "fetch_tools_for_metadata", lambda _h: ["t"])
        tool_usage.resolve_mech_tools(["0xAbC", "0xDEF"], "http://subgraph")
        assert '"0xabc"' in captured["query"]
        assert '"0xdef"' in captured["query"]
        assert "0xAbC" not in captured["query"]

    def test_subgraph_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A GraphQL/network error is raised so the caller can mark unavailable."""

        def fake_post(url: str, query: str) -> dict:
            raise URLError("subgraph down")

        monkeypatch.setattr(tool_usage, "_post_graphql", fake_post)
        with pytest.raises(URLError):
            tool_usage.resolve_mech_tools(["0xa"], "http://subgraph")

    def test_missing_subgraph_addresses_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Subgraph omitting any requested address raises ValueError listing them."""
        monkeypatch.setattr(
            tool_usage,
            "_post_graphql",
            lambda url, query: {
                "meches": [
                    {
                        "address": "0xa",
                        "service": {"metadata": [{"metadata": "0x11"}]},
                    }
                ]
            },
        )

        def boom(_metadata_hash: str) -> list[str]:
            raise AssertionError("IPFS must not be touched after missing-addr raise")

        monkeypatch.setattr(tool_usage, "fetch_tools_for_metadata", boom)
        with pytest.raises(ValueError) as excinfo:
            tool_usage.resolve_mech_tools(
                ["0xA", "0xMissing1", "0xMissing2"], "http://subgraph"
            )
        # Resolved address must not appear; missing ones must.
        message = str(excinfo.value)
        assert "0xmissing1" in message
        assert "0xmissing2" in message
        assert "0xa" not in message

    def test_all_addresses_missing_raises_with_every_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty meches list raises with every requested address (pins subtraction direction)."""
        monkeypatch.setattr(
            tool_usage, "_post_graphql", lambda url, query: {"meches": []}
        )

        with pytest.raises(ValueError) as excinfo:
            tool_usage.resolve_mech_tools(["0xa", "0xb"], "http://subgraph")
        message = str(excinfo.value)
        assert "0xa" in message and "0xb" in message

    def test_all_addresses_returned_but_no_metadata_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Addresses present but every mech lacks on-chain metadata -> ValueError."""
        monkeypatch.setattr(
            tool_usage,
            "_post_graphql",
            lambda url, query: {
                "meches": [
                    {"address": "0xa", "service": None},
                    {"address": "0xb", "service": {"metadata": []}},
                ]
            },
        )

        def boom(_metadata_hash: str) -> list[str]:
            raise AssertionError("IPFS must not be touched when no metadata is present")

        monkeypatch.setattr(tool_usage, "fetch_tools_for_metadata", boom)
        with pytest.raises(ValueError, match="none have on-chain metadata"):
            tool_usage.resolve_mech_tools(["0xa", "0xb"], "http://subgraph")

    def test_partial_metadata_unions_present_skips_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mechs with ``service=None`` / empty metadata are skipped when at least one resolves."""
        monkeypatch.setattr(
            tool_usage,
            "_post_graphql",
            lambda url, query: {
                "meches": [
                    {"address": "0xa", "service": None},
                    {
                        "address": "0xb",
                        "service": {"metadata": [{"metadata": "0x22"}]},
                    },
                ]
            },
        )

        def fake_tools(metadata_hash: str) -> list[str]:
            assert (
                metadata_hash == "0x22"
            ), f"unexpected metadata fetch: {metadata_hash}"
            return ["superforcaster"]

        monkeypatch.setattr(tool_usage, "fetch_tools_for_metadata", fake_tools)
        assert tool_usage.resolve_mech_tools(["0xa", "0xb"], "http://subgraph") == [
            "superforcaster"
        ]


# ---------------------------------------------------------------------------
# fetch_valid_tools (end-to-end resolution + failure handling)
# ---------------------------------------------------------------------------

RELEASE_TAG = "v9.9.9-test"
PEARL_MECH = "0xpearlmech"
POLY_MECH = "0xpolymech"
PEARL_META = "0x1111"
POLY_META = "0x2222"


def _routed_http_get(
    pearl_addrs: list[str], poly_addrs: list[str]
) -> Callable[[str], str]:
    """Build a fake _http_get routing by URL: releases, service.yamls, IPFS."""

    def fake_get(url: str) -> str:
        if "api.github.com" in url:
            return json.dumps([{"tag_name": RELEASE_TAG}])
        if "trader_pearl" in url:
            assert RELEASE_TAG in url  # yaml fetched at the resolved release tag
            return _service_yaml(pearl_addrs)
        if "polymarket_trader" in url:
            assert RELEASE_TAG in url
            return _service_yaml(poly_addrs)
        if url.endswith(PEARL_META[2:]):
            return json.dumps({"tools": ["prediction-online", "shared-tool"]})
        if url.endswith(POLY_META[2:]):
            return json.dumps({"tools": ["echo", "shared-tool"]})
        raise AssertionError(f"unexpected URL: {url}")

    return fake_get


def _routed_post_graphql(url: str, query: str) -> dict:
    """Fake subgraph: route by chain, return each chain's mech metadata."""
    if "marketplace-gnosis" in url:
        return {
            "meches": [
                {
                    "address": PEARL_MECH,
                    "service": {"metadata": [{"metadata": PEARL_META}]},
                }
            ]
        }
    if "marketplace-polygon" in url:
        return {
            "meches": [
                {
                    "address": POLY_MECH,
                    "service": {"metadata": [{"metadata": POLY_META}]},
                }
            ]
        }
    raise AssertionError(f"unexpected subgraph URL: {url}")


class TestFetchValidTools:
    """Tests for fetch_valid_tools and its per-deployment failure isolation."""

    def test_both_deployments_succeed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each deployment resolves to the sorted union of its mechs' tools."""
        monkeypatch.setattr(
            tool_usage, "_http_get", _routed_http_get([PEARL_MECH], [POLY_MECH])
        )
        monkeypatch.setattr(tool_usage, "_post_graphql", _routed_post_graphql)
        result = tool_usage.fetch_valid_tools()
        assert result["omenstrat Pearl"] == ["prediction-online", "shared-tool"]
        assert result["polystrat Pearl"] == ["echo", "shared-tool"]

    def test_release_fetch_fails_blanks_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed release lookup leaves every deployment None (never raised)."""

        def fake_get(url: str) -> str:
            if "api.github.com" in url:
                raise URLError("github down")
            raise AssertionError("should not reach per-deployment fetch")

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        result = tool_usage.fetch_valid_tools()
        assert all(v is None for v in result.values())

    def test_one_deployment_yaml_fails_isolated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing service.yaml nulls only that deployment; the other resolves."""
        base = _routed_http_get([PEARL_MECH], [POLY_MECH])

        def fake_get(url: str) -> str:
            if "trader_pearl" in url:
                raise URLError("yaml 404")
            return base(url)

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        monkeypatch.setattr(tool_usage, "_post_graphql", _routed_post_graphql)
        result = tool_usage.fetch_valid_tools()
        assert result["omenstrat Pearl"] is None
        assert result["polystrat Pearl"] == ["echo", "shared-tool"]

    def test_one_subgraph_fails_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A subgraph failure nulls only the affected deployment."""
        monkeypatch.setattr(
            tool_usage, "_http_get", _routed_http_get([PEARL_MECH], [POLY_MECH])
        )

        def fake_post(url: str, query: str) -> dict:
            if "marketplace-gnosis" in url:
                raise URLError("subgraph down")
            return _routed_post_graphql(url, query)

        monkeypatch.setattr(tool_usage, "_post_graphql", fake_post)
        result = tool_usage.fetch_valid_tools()
        assert result["omenstrat Pearl"] is None
        assert result["polystrat Pearl"] == ["echo", "shared-tool"]

    def test_malformed_yaml_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A service.yaml missing valid_mechs nulls that deployment, never raises."""
        base = _routed_http_get([PEARL_MECH], [POLY_MECH])

        def fake_get(url: str) -> str:
            if "trader_pearl" in url:
                return "garbage: yaml: without: valid_mechs\n"
            return base(url)

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        monkeypatch.setattr(tool_usage, "_post_graphql", _routed_post_graphql)
        result = tool_usage.fetch_valid_tools()
        assert result["omenstrat Pearl"] is None
        assert result["polystrat Pearl"] == ["echo", "shared-tool"]

    def test_empty_valid_mechs_is_empty_list_not_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty valid_mechs resolves to [] (a real state) without a subgraph call."""
        monkeypatch.setattr(tool_usage, "_http_get", _routed_http_get([], [POLY_MECH]))

        def fake_post(url: str, query: str) -> dict:
            assert "marketplace-gnosis" not in url  # no mechs -> no gnosis query
            return _routed_post_graphql(url, query)

        monkeypatch.setattr(tool_usage, "_post_graphql", fake_post)
        result = tool_usage.fetch_valid_tools()
        assert result["omenstrat Pearl"] == []
        assert result["polystrat Pearl"] == ["echo", "shared-tool"]


# ---------------------------------------------------------------------------
# Rendering of the deployment-status section (allow-list semantics)
# ---------------------------------------------------------------------------


class TestSectionToolDeploymentStatus:
    """Rendering review: every code path produces the output a reader expects."""

    def test_active_tools_rendered(self) -> None:
        """Each deployment shows the benchmarked tools that ARE selectable."""
        scores = _scores_with_tools(["echo", "prediction-online"])
        valid: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["prediction-online"],
            "polystrat Pearl": ["echo", "prediction-online"],
        }
        result = section_tool_deployment_status(scores, valid=valid)
        assert "## Tool Deployment Status" in result
        pearl_line = next(
            line for line in result.splitlines() if "omenstrat Pearl" in line
        )
        assert "`prediction-online`" in pearl_line
        assert "`echo`" not in pearl_line
        poly_line = next(
            line for line in result.splitlines() if "polystrat Pearl" in line
        )
        assert "`echo`" in poly_line
        assert "`prediction-online`" in poly_line

    def test_tool_in_config_but_not_benchmarked_is_hidden(self) -> None:
        """Allow-list entries naming non-benchmarked tools don't invent rows."""
        scores = _scores_with_tools(["prediction-online"])
        valid: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["prediction-online", "echo"],  # echo not benchmarked
            "polystrat Pearl": ["echo"],  # only non-benchmarked -> empty
        }
        result = section_tool_deployment_status(scores, valid=valid)
        assert "`echo`" not in result
        pearl_line = next(
            line for line in result.splitlines() if "omenstrat Pearl" in line
        )
        assert "`prediction-online`" in pearl_line
        poly_line = next(
            line for line in result.splitlines() if "polystrat Pearl" in line
        )
        assert "no benchmarked tools active" in poly_line

    def test_full_allow_list_means_all_tools_active(self) -> None:
        """With every tool selectable, each appears once per deployment."""
        scores = _scores_with_tools(["prediction-online"])
        valid: dict[str, list[str] | None] = {
            name: ["prediction-online"] for name in tool_usage.DEPLOYMENTS
        }
        result = section_tool_deployment_status(scores, valid=valid)
        assert result.count("`prediction-online`") == len(tool_usage.DEPLOYMENTS)

    def test_empty_allow_list_means_no_active(self) -> None:
        """An empty allow-list renders 'no benchmarked tools active'."""
        scores = _scores_with_tools(["prediction-online"])
        valid: dict[str, list[str] | None] = {
            name: [] for name in tool_usage.DEPLOYMENTS
        }
        result = section_tool_deployment_status(scores, valid=valid)
        assert "`prediction-online`" not in result
        assert result.count("no benchmarked tools active") == len(
            tool_usage.DEPLOYMENTS
        )

    def test_normalizes_underscore_hyphen(self) -> None:
        """Allow-list spelling with underscores still matches hyphen tool names."""
        scores = _scores_with_tools(["prediction-request-reasoning-claude"])
        valid: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["prediction_request_reasoning_claude"],
            "polystrat Pearl": [],
        }
        result = section_tool_deployment_status(scores, valid=valid)
        pearl_line = next(
            line for line in result.splitlines() if "omenstrat Pearl" in line
        )
        assert "`prediction-request-reasoning-claude`" in pearl_line

    def test_fetch_failure_renders_unavailable_banner(self) -> None:
        """Failed deployments render ⚠️ unavailable instead of a false 'active' list."""
        scores = _scores_with_tools(["echo"])
        valid: dict[str, list[str] | None] = {
            "omenstrat Pearl": None,
            "polystrat Pearl": ["echo"],
        }
        result = section_tool_deployment_status(scores, valid=valid)
        assert "Could not fetch deployment config" in result
        pearl_line = next(
            line
            for line in result.splitlines()
            if line.startswith("- **omenstrat Pearl**")
        )
        poly_line = next(
            line
            for line in result.splitlines()
            if line.startswith("- **polystrat Pearl**")
        )
        assert "⚠️ unavailable" in pearl_line
        assert "`echo`" in poly_line

    def test_empty_dict_opts_out_entirely(self) -> None:
        """Empty dict means "caller opted out" and returns an empty string."""
        scores = _scores_with_tools(["echo"])
        result = section_tool_deployment_status(scores, valid={})
        assert result == ""

    def test_all_fetches_fail_no_false_active_claim(self) -> None:
        """If everything fails, no deployment claims active tools."""
        scores = _scores_with_tools(["echo"])
        valid: dict[str, list[str] | None] = {
            name: None for name in tool_usage.DEPLOYMENTS
        }
        result = section_tool_deployment_status(scores, valid=valid)
        assert "Could not fetch deployment config" in result
        assert "`echo`" not in result
        assert result.count("⚠️ unavailable") == len(tool_usage.DEPLOYMENTS)

    def test_active_tools_ordered_by_brier_ranking(self) -> None:
        """Active tools render in the same Brier-ascending order as Tool Ranking."""
        scores = {
            "generated_at": "2026-04-14T06:00:00Z",
            "total_rows": 30,
            "valid_rows": 30,
            "overall": {"brier": 0.25, "reliability": 1.0, "n": 30},
            "by_tool": {
                "slow-tool": {"brier": 0.45, "n": 10},
                "fast-tool": {"brier": 0.15, "n": 10},
                "mid-tool": {"brier": 0.30, "n": 10},
            },
        }
        valid: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["slow-tool", "fast-tool", "mid-tool"],
            "polystrat Pearl": [],
        }
        result = section_tool_deployment_status(scores, valid=valid)
        pearl_line = next(
            line for line in result.splitlines() if "omenstrat Pearl" in line
        )
        fast_idx = pearl_line.index("`fast-tool`")
        mid_idx = pearl_line.index("`mid-tool`")
        slow_idx = pearl_line.index("`slow-tool`")
        assert fast_idx < mid_idx < slow_idx
