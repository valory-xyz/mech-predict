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

from urllib.error import URLError

import pytest
from benchmark import tool_usage
from benchmark.analyze import section_tool_deployment_status
from benchmark.tool_usage import DEPLOYMENTS, DEPLOYMENT_TO_PLATFORM


class TestDeploymentPlatformMappingInvariants:
    """Lock the DEPLOYMENTS ↔ DEPLOYMENT_TO_PLATFORM coverage invariant.

    ``deployments_for_platform`` filters ``DEPLOYMENTS`` through
    ``DEPLOYMENT_TO_PLATFORM``. If a deployment lands in the list but
    not in the mapping, it silently disappears from the platform-scoped
    Tool Deployment Status section. The tests below fail loudly at add
    time so nobody has to rediscover the bug in production.
    """

    def test_every_deployment_has_a_platform(self) -> None:
        """No name in DEPLOYMENTS is missing from DEPLOYMENT_TO_PLATFORM."""
        missing = [name for name in DEPLOYMENTS if name not in DEPLOYMENT_TO_PLATFORM]
        assert not missing, (
            f"deployment(s) {missing} in DEPLOYMENTS have no platform mapping — "
            "deployments_for_platform would silently drop them"
        )

    def test_no_orphan_mapping_entries(self) -> None:
        """Every mapping key is also present in DEPLOYMENTS (no drift the other way)."""
        orphans = [name for name in DEPLOYMENT_TO_PLATFORM if name not in DEPLOYMENTS]
        assert not orphans, (
            f"mapping key(s) {orphans} have no matching DEPLOYMENTS entry — "
            "the deployment was removed but the mapping wasn't cleaned up"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OMEN_TRADER_REF = "v9.9.9-omen"
POLY_TRADER_REF = "v8.8.8-poly"
PEARL_ONLY_MARKER = "pearl-only-marker"
POLY_ONLY_MARKER = "poly-only-marker"

# Distinct service_version per template so a swap or boundary-bleed is
# detectable. Note PREDICT_SERVICE_TEMPLATE is intentionally NOT a
# substring of PREDICT_POLYMARKET_SERVICE_TEMPLATE.
OPERATE_APP_TS_SAMPLE = f"""
export const PREDICT_SERVICE_TEMPLATE: ServiceTemplate = {{
  name: 'Predict Agent',
  hash: 'bafyomen',
  service_version: '{OMEN_TRADER_REF}',
  configurations: {{
    env_variables: {{
      GENAI_API_KEY: {{ name: 'Gemini' }},
    }},
  }},
}};

export const PREDICT_POLYMARKET_SERVICE_TEMPLATE: ServiceTemplate = {{
  name: 'Predict Polymarket Agent',
  hash: 'bafypoly',
  service_version: '{POLY_TRADER_REF}',
  configurations: {{
    env_variables: {{}},
  }},
}};
"""

TRADER_PEARL_YAML_SAMPLE = f"""
name: trader_pearl
models:
  params:
    args:
      tools_accuracy_hash: ${{TOOLS_ACCURACY_HASH:str:QmExample}}
      valid_tools: ${{VALID_TOOLS:list:["superforcaster","factual_research","{PEARL_ONLY_MARKER}"]}}
      something_else: 1
"""

TRADER_POLY_YAML_SAMPLE = f"""
name: polymarket_trader
models:
  params:
    args:
      valid_tools: ${{VALID_TOOLS:list:["superforcaster","{POLY_ONLY_MARKER}"]}}
"""


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
# parse_service_version
# ---------------------------------------------------------------------------


class TestParseServiceVersion:
    """Tests for parse_service_version."""

    def test_extracts_per_template(self) -> None:
        """Each template resolves to its own service_version, not the other's."""
        assert (
            tool_usage.parse_service_version(
                OPERATE_APP_TS_SAMPLE, "PREDICT_SERVICE_TEMPLATE"
            )
            == OMEN_TRADER_REF
        )
        assert (
            tool_usage.parse_service_version(
                OPERATE_APP_TS_SAMPLE, "PREDICT_POLYMARKET_SERVICE_TEMPLATE"
            )
            == POLY_TRADER_REF
        )

    def test_identity_survives_template_reorder(self) -> None:
        """Declaring polymarket first must not swap the resolved versions."""
        halves = OPERATE_APP_TS_SAMPLE.split(
            "export const PREDICT_POLYMARKET_SERVICE_TEMPLATE", maxsplit=1
        )
        reordered = (
            "export const PREDICT_POLYMARKET_SERVICE_TEMPLATE" + halves[1] + halves[0]
        )
        assert (
            tool_usage.parse_service_version(reordered, "PREDICT_SERVICE_TEMPLATE")
            == OMEN_TRADER_REF
        )
        assert (
            tool_usage.parse_service_version(
                reordered, "PREDICT_POLYMARKET_SERVICE_TEMPLATE"
            )
            == POLY_TRADER_REF
        )

    def test_missing_template_raises(self) -> None:
        """An absent template raises rather than returning a wrong version."""
        with pytest.raises(ValueError, match="no service_version found for template"):
            tool_usage.parse_service_version(
                OPERATE_APP_TS_SAMPLE, "NONEXISTENT_TEMPLATE"
            )

    def test_missing_service_version_does_not_borrow_next_template(self) -> None:
        """A template without its own service_version raises, not borrows.

        Strips the omen template's service_version line while leaving the
        polymarket one intact. A naive lazy-gap regex would bleed across
        the ``export const`` boundary and wrongly return the poly version.
        """
        broken = OPERATE_APP_TS_SAMPLE.replace(
            f"  service_version: '{OMEN_TRADER_REF}',\n", ""
        )
        with pytest.raises(ValueError, match="no service_version found for template"):
            tool_usage.parse_service_version(broken, "PREDICT_SERVICE_TEMPLATE")

    def test_export_const_in_comment_is_not_a_boundary(self) -> None:
        """Only a line-start ``export const`` ends a template.

        A comment that merely mentions ``export const`` between the
        template name and its ``service_version`` must not make the
        boundary-guard abort and raise. Regression lock for the
        line-start anchor.
        """
        commented = OPERATE_APP_TS_SAMPLE.replace(
            "  name: 'Predict Agent',\n",
            "  name: 'Predict Agent',\n"
            "  // historically aliased, see export const LEGACY_TEMPLATE\n",
        )
        assert (
            tool_usage.parse_service_version(commented, "PREDICT_SERVICE_TEMPLATE")
            == OMEN_TRADER_REF
        )

    def test_path_traversal_ref_is_rejected(self) -> None:
        """A ``service_version`` with path-traversal chars is a parse failure.

        The value flows into the trader yaml fetch URL, so a ``/`` (or any
        non git-ref char) must raise rather than build a URL escaping
        ``valory-xyz/trader/<ref>/``.
        """
        evil = OPERATE_APP_TS_SAMPLE.replace(
            f"service_version: '{OMEN_TRADER_REF}'",
            "service_version: 'v0.38.0/../../evil'",
        )
        with pytest.raises(ValueError, match="not a valid git ref"):
            tool_usage.parse_service_version(evil, "PREDICT_SERVICE_TEMPLATE")


# ---------------------------------------------------------------------------
# parse_valid_tools
# ---------------------------------------------------------------------------


class TestParseValidTools:
    """Tests for parse_valid_tools."""

    def test_happy_path(self) -> None:
        """The env-override default list is parsed in declared order."""
        assert tool_usage.parse_valid_tools(TRADER_PEARL_YAML_SAMPLE) == [
            "superforcaster",
            "factual_research",
            PEARL_ONLY_MARKER,
        ]

    def test_empty_allow_list_returns_empty_not_none(self) -> None:
        """An empty allow-list is a real state, distinct from a failure."""
        src = "      valid_tools: ${VALID_TOOLS:list:[]}\n"
        assert tool_usage.parse_valid_tools(src) == []

    def test_missing_valid_tools_raises(self) -> None:
        """No valid_tools line raises — there is no irrelevant_tools fallback."""
        src = '      irrelevant_tools: ${IRRELEVANT_TOOLS:list:["native-transfer"]}\n'
        with pytest.raises(
            ValueError, match="no valid_tools env default found in service.yaml"
        ):
            tool_usage.parse_valid_tools(src)

    def test_non_string_items_raise(self) -> None:
        """A non-string array element raises ValueError."""
        src = '      valid_tools: ${VALID_TOOLS:list:["superforcaster", 2]}\n'
        with pytest.raises(ValueError, match="must be a JSON array of strings"):
            tool_usage.parse_valid_tools(src)


# ---------------------------------------------------------------------------
# fetch_valid_tools (two-stage resolution + failure handling)
# ---------------------------------------------------------------------------


class TestFetchValidTools:
    """Tests for fetch_valid_tools and its failure semantics."""

    def test_both_deployments_succeed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both Pearl slots populated; trader yaml fetched at the pinned ref."""
        seen: list[str] = []

        def fake_get(url: str) -> str:
            seen.append(url)
            if "olas-operate-app" in url:
                return OPERATE_APP_TS_SAMPLE
            if "trader_pearl" in url:
                return TRADER_PEARL_YAML_SAMPLE
            if "polymarket_trader" in url:
                return TRADER_POLY_YAML_SAMPLE
            raise AssertionError(f"unexpected url {url}")

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        result = tool_usage.fetch_valid_tools()

        pearl = result["omenstrat Pearl"]
        poly = result["polystrat Pearl"]
        assert pearl is not None and PEARL_ONLY_MARKER in pearl
        assert poly is not None and POLY_ONLY_MARKER in poly
        # Each trader yaml was fetched at the version parsed from trader.ts.
        pearl_url = next(u for u in seen if "trader_pearl" in u)
        poly_url = next(u for u in seen if "polymarket_trader" in u)
        assert OMEN_TRADER_REF in pearl_url
        assert POLY_TRADER_REF in poly_url

    def test_operate_app_fetch_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """operate-app failure -> every deployment None (no trader fetch)."""

        def fake_get(url: str) -> str:
            if "olas-operate-app" in url:
                raise URLError("network down")
            raise AssertionError("trader yaml must not be fetched")

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        result = tool_usage.fetch_valid_tools()
        assert result == {name: None for name in DEPLOYMENTS}

    def test_one_trader_yaml_fails_isolates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single trader yaml failure -> only that deployment None."""

        def fake_get(url: str) -> str:
            if "olas-operate-app" in url:
                return OPERATE_APP_TS_SAMPLE
            if "trader_pearl" in url:
                raise URLError("trader pearl yaml 500")
            return TRADER_POLY_YAML_SAMPLE

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        result = tool_usage.fetch_valid_tools()
        assert result["omenstrat Pearl"] is None
        poly = result["polystrat Pearl"]
        assert poly is not None and POLY_ONLY_MARKER in poly

    def test_missing_service_version_for_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A template missing service_version -> that deployment None only."""
        broken_ts = OPERATE_APP_TS_SAMPLE.replace(
            f"  service_version: '{OMEN_TRADER_REF}',\n", ""
        )

        def fake_get(url: str) -> str:
            if "olas-operate-app" in url:
                return broken_ts
            if "polymarket_trader" in url:
                return TRADER_POLY_YAML_SAMPLE
            raise AssertionError("pearl trader yaml must not be fetched")

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        result = tool_usage.fetch_valid_tools()
        assert result["omenstrat Pearl"] is None
        assert result["polystrat Pearl"] is not None

    def test_malformed_yaml_is_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A trader yaml with no valid_tools -> None, never blocks the report."""

        def fake_get(url: str) -> str:
            if "olas-operate-app" in url:
                return OPERATE_APP_TS_SAMPLE
            return "name: trader\nmodels: {}\n"

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        result = tool_usage.fetch_valid_tools()
        assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# Rendering of the deployment-status section
# ---------------------------------------------------------------------------


class TestSectionToolDeploymentStatus:
    """Rendering review: every code path produces the output a reader expects."""

    def test_active_tools_rendered(self) -> None:
        """Each deployment shows the benchmarked tools that ARE allow-listed."""
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

    def test_allow_listed_but_not_benchmarked_is_hidden(self) -> None:
        """Allow-list entries naming non-benchmarked tools don't appear."""
        scores = _scores_with_tools(["prediction-online"])
        valid: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["echo", "prediction-online"],  # echo not benchmarked
            "polystrat Pearl": ["prediction-online"],
        }
        result = section_tool_deployment_status(scores, valid=valid)
        assert "`echo`" not in result
        pearl_line = next(
            line for line in result.splitlines() if "omenstrat Pearl" in line
        )
        assert "`prediction-online`" in pearl_line

    def test_underscore_hyphen_are_distinct_tools(self) -> None:
        """The ``_``/``-`` separator is part of tool identity, not noise.

        ``prediction_request_reasoning_claude`` and
        ``prediction-request-reasoning-claude`` are different tools; an
        allow-list entry for the underscore form must NOT activate the
        benchmarked hyphen form.
        """
        scores = _scores_with_tools(["prediction-request-reasoning-claude"])
        valid: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["prediction_request_reasoning_claude"],
            "polystrat Pearl": [],
        }
        result = section_tool_deployment_status(scores, valid=valid)
        pearl_line = next(
            line for line in result.splitlines() if "omenstrat Pearl" in line
        )
        assert "`prediction-request-reasoning-claude`" not in pearl_line
        assert "no benchmarked tools active" in pearl_line

    def test_full_allow_list_means_all_tools_active(self) -> None:
        """When the allow-list covers every benchmarked tool, all render."""
        scores = _scores_with_tools(["prediction-online"])
        valid: dict[str, list[str] | None] = {
            name: ["prediction-online"] for name in DEPLOYMENTS
        }
        result = section_tool_deployment_status(scores, valid=valid)
        assert result.count("`prediction-online`") == len(DEPLOYMENTS)

    def test_empty_allow_list_means_no_active(self) -> None:
        """An empty allow-list renders 'no benchmarked tools active'."""
        scores = _scores_with_tools(["echo"])
        valid: dict[str, list[str] | None] = {name: [] for name in DEPLOYMENTS}
        result = section_tool_deployment_status(scores, valid=valid)
        assert "`echo`" not in result
        assert result.count("no benchmarked tools active") == len(DEPLOYMENTS)

    def test_fetch_failure_renders_unavailable_banner(self) -> None:
        """Failed deployments render ⚠️ unavailable, not a false active list."""
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
        valid: dict[str, list[str] | None] = {name: None for name in DEPLOYMENTS}
        result = section_tool_deployment_status(scores, valid=valid)
        assert "Could not fetch deployment config" in result
        assert "`echo`" not in result
        assert result.count("⚠️ unavailable") == len(DEPLOYMENTS)

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
            name: ["slow-tool", "fast-tool", "mid-tool"] for name in DEPLOYMENTS
        }
        result = section_tool_deployment_status(scores, valid=valid)
        pearl_line = next(
            line for line in result.splitlines() if "omenstrat Pearl" in line
        )
        fast_idx = pearl_line.index("`fast-tool`")
        mid_idx = pearl_line.index("`mid-tool`")
        slow_idx = pearl_line.index("`slow-tool`")
        assert fast_idx < mid_idx < slow_idx
