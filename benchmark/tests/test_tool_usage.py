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
from urllib.error import URLError

import pytest

from benchmark import tool_usage
from benchmark.analyze import section_tool_deployment_status

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PEARL_ONLY_MARKER = "omenstrat-only-marker"
POLYSTRAT_ONLY_MARKER = "polystrat-only-marker"

OPERATE_APP_TS_SAMPLE = f"""
export const PREDICT_SERVICE_TEMPLATE = {{
  env_variables: {{
    IRRELEVANT_TOOLS: {{
      name: 'Irrelevant tools',
      description: '',
      value:
        '["native-transfer","echo","prediction-online-lite","{PEARL_ONLY_MARKER}"]',
      provision_type: EnvProvisionType.FIXED,
    }},
    GENAI_API_KEY: {{ name: 'Gemini' }},
  }},
}};

export const PREDICT_POLYMARKET_SERVICE_TEMPLATE = {{
  env_variables: {{
    IRRELEVANT_TOOLS: {{
      name: 'Irrelevant tools',
      description: '',
      value:
        '["native-transfer","echo","prediction-online-lite","{POLYSTRAT_ONLY_MARKER}"]',
      provision_type: EnvProvisionType.FIXED,
    }},
  }},
}};
"""

QUICKSTART_JSON_SAMPLE = json.dumps(
    {
        "name": "Trader Agent",
        "env_variables": {
            "IRRELEVANT_TOOLS": {
                "name": "Irrelevant tools",
                "value": json.dumps(
                    [
                        "native-transfer",
                        "echo",
                        "openai-gpt-4",
                    ]
                ),
            },
        },
    }
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
# parse_operate_app_ts
# ---------------------------------------------------------------------------


class TestParseOperateAppTs:
    """Tests for parse_operate_app_ts."""

    def test_extracts_both_templates(self) -> None:
        """Each template's block is identified by name, not by file order."""
        result = tool_usage.parse_operate_app_ts(OPERATE_APP_TS_SAMPLE)
        # Template-unique markers prove the mapping didn't silently swap.
        assert PEARL_ONLY_MARKER in result["omenstrat Pearl"]
        assert PEARL_ONLY_MARKER not in result["polystrat Pearl"]
        assert POLYSTRAT_ONLY_MARKER in result["polystrat Pearl"]
        assert POLYSTRAT_ONLY_MARKER not in result["omenstrat Pearl"]

    def test_identity_survives_template_reorder(self) -> None:
        """If the file declares polystrat before omenstrat, labels still match."""
        halves = OPERATE_APP_TS_SAMPLE.split(
            "export const PREDICT_POLYMARKET_SERVICE_TEMPLATE", maxsplit=1
        )
        reordered = (
            "export const PREDICT_POLYMARKET_SERVICE_TEMPLATE" + halves[1] + halves[0]
        )
        result = tool_usage.parse_operate_app_ts(reordered)
        assert PEARL_ONLY_MARKER in result["omenstrat Pearl"]
        assert POLYSTRAT_ONLY_MARKER in result["polystrat Pearl"]

    def test_rejects_missing_template(self) -> None:
        """Missing a template raises instead of silently returning a partial dict."""
        single = OPERATE_APP_TS_SAMPLE.split(
            "export const PREDICT_POLYMARKET_SERVICE_TEMPLATE", maxsplit=1
        )[0]
        with pytest.raises(
            ValueError, match="no IRRELEVANT_TOOLS block found for template"
        ):
            tool_usage.parse_operate_app_ts(single)

    def test_rejects_non_string_items(self) -> None:
        """Array with non-string items raises ValueError."""
        bad = OPERATE_APP_TS_SAMPLE.replace('"echo"', "123")
        with pytest.raises(ValueError, match="must be a JSON array of strings"):
            tool_usage.parse_operate_app_ts(bad)


# ---------------------------------------------------------------------------
# parse_quickstart_config
# ---------------------------------------------------------------------------


class TestParseQuickstartConfig:
    """Tests for parse_quickstart_config."""

    def test_happy_path(self) -> None:
        """Standard shape returns the parsed list."""
        assert tool_usage.parse_quickstart_config(QUICKSTART_JSON_SAMPLE) == [
            "native-transfer",
            "echo",
            "openai-gpt-4",
        ]

    def test_missing_path_raises(self) -> None:
        """Missing env_variables.IRRELEVANT_TOOLS raises KeyError."""
        bad = json.dumps({"env_variables": {}})
        with pytest.raises(KeyError):
            tool_usage.parse_quickstart_config(bad)


# ---------------------------------------------------------------------------
# fetch_disabled_tools (fetch-failure handling)
# ---------------------------------------------------------------------------


class TestFetchDisabledTools:
    """Tests for fetch_disabled_tools and its failure semantics."""

    def test_both_sources_succeed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All three deployment slots populated when both HTTP calls succeed."""

        def fake_get(url: str) -> str:
            if "olas-operate-app" in url:
                return OPERATE_APP_TS_SAMPLE
            return QUICKSTART_JSON_SAMPLE

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        result = tool_usage.fetch_disabled_tools()
        pearl = result["omenstrat Pearl"]
        assert pearl is not None
        assert set(pearl) >= {"native-transfer", "echo", PEARL_ONLY_MARKER}
        assert result["omenstrat QS"] == ["native-transfer", "echo", "openai-gpt-4"]
        polystrat = result["polystrat Pearl"]
        assert polystrat is not None and POLYSTRAT_ONLY_MARKER in polystrat

    def test_operate_app_fetch_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """operate-app failure -> None for both Pearl slots, QS still works."""

        def fake_get(url: str) -> str:
            if "olas-operate-app" in url:
                raise URLError("network down")
            return QUICKSTART_JSON_SAMPLE

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        result = tool_usage.fetch_disabled_tools()
        assert result["omenstrat Pearl"] is None
        assert result["polystrat Pearl"] is None
        assert result["omenstrat QS"] == ["native-transfer", "echo", "openai-gpt-4"]

    def test_quickstart_fetch_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Quickstart failure -> None for QS only, Pearl slots still populated."""

        def fake_get(url: str) -> str:
            if "olas-operate-app" in url:
                return OPERATE_APP_TS_SAMPLE
            raise URLError("network down")

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        result = tool_usage.fetch_disabled_tools()
        assert result["omenstrat QS"] is None
        assert result["omenstrat Pearl"] is not None

    def test_malformed_response_is_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parse errors are swallowed so the daily report is never blocked."""

        def fake_get(url: str) -> str:
            return "not valid content"

        monkeypatch.setattr(tool_usage, "_http_get", fake_get)
        result = tool_usage.fetch_disabled_tools()
        assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# Rendering of the deployment-status section
# ---------------------------------------------------------------------------


class TestSectionToolDeploymentStatus:
    """Rendering review: every code path produces the output a reader expects."""

    def test_active_tools_rendered(self) -> None:
        """Each deployment shows the benchmarked tools that are NOT disabled."""
        scores = _scores_with_tools(["echo", "prediction-online"])
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["echo"],
            "omenstrat QS": ["echo", "prediction-online"],
            "polystrat Pearl": [],
        }
        result = section_tool_deployment_status(scores, disabled=disabled)
        assert "## Tool Deployment Status" in result
        pearl_line = next(
            line for line in result.splitlines() if "omenstrat Pearl" in line
        )
        assert "`prediction-online`" in pearl_line
        assert "`echo`" not in pearl_line
        qs_line = next(line for line in result.splitlines() if "omenstrat QS" in line)
        assert "no benchmarked tools active" in qs_line
        poly_line = next(
            line for line in result.splitlines() if "polystrat Pearl" in line
        )
        assert "`echo`" in poly_line
        assert "`prediction-online`" in poly_line

    def test_tool_in_config_but_not_benchmarked_is_hidden(self) -> None:
        """Disabled entries naming non-benchmarked tools don't affect the active list."""
        scores = _scores_with_tools(["prediction-online"])
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": ["echo"],  # not in scores, ignored
            "omenstrat QS": [],
            "polystrat Pearl": [],
        }
        result = section_tool_deployment_status(scores, disabled=disabled)
        assert "`echo`" not in result
        pearl_line = next(
            line for line in result.splitlines() if "omenstrat Pearl" in line
        )
        assert "`prediction-online`" in pearl_line

    def test_no_disabled_tools_means_all_tools_active(self) -> None:
        """With nothing disabled, every benchmarked tool appears on every deployment."""
        scores = _scores_with_tools(["prediction-online"])
        disabled: dict[str, list[str] | None] = {
            name: [] for name in tool_usage.DEPLOYMENTS
        }
        result = section_tool_deployment_status(scores, disabled=disabled)
        assert result.count("`prediction-online`") == len(tool_usage.DEPLOYMENTS)

    def test_fetch_failure_renders_unavailable_banner(self) -> None:
        """Failed deployments render ⚠️ unavailable instead of a false 'active' list."""
        scores = _scores_with_tools(["echo"])
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": None,
            "omenstrat QS": ["echo"],
            "polystrat Pearl": None,
        }
        result = section_tool_deployment_status(scores, disabled=disabled)
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
        qs_line = next(
            line
            for line in result.splitlines()
            if line.startswith("- **omenstrat QS**")
        )
        assert "⚠️ unavailable" in pearl_line
        assert "⚠️ unavailable" in poly_line
        assert "no benchmarked tools active" in qs_line

    def test_empty_dict_opts_out_entirely(self) -> None:
        """Empty dict means "caller opted out" and returns an empty string."""
        scores = _scores_with_tools(["echo"])
        result = section_tool_deployment_status(scores, disabled={})
        assert result == ""

    def test_all_fetches_fail_no_false_active_claim(self) -> None:
        """If everything fails, no deployment claims active tools."""
        scores = _scores_with_tools(["echo"])
        disabled: dict[str, list[str] | None] = {
            name: None for name in tool_usage.DEPLOYMENTS
        }
        result = section_tool_deployment_status(scores, disabled=disabled)
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
        disabled: dict[str, list[str] | None] = {
            "omenstrat Pearl": [],
            "omenstrat QS": [],
            "polystrat Pearl": [],
        }
        result = section_tool_deployment_status(scores, disabled=disabled)
        pearl_line = next(
            line for line in result.splitlines() if "omenstrat Pearl" in line
        )
        fast_idx = pearl_line.index("`fast-tool`")
        mid_idx = pearl_line.index("`mid-tool`")
        slow_idx = pearl_line.index("`slow-tool`")
        assert fast_idx < mid_idx < slow_idx
