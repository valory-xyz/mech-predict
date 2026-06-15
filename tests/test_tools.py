# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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

"""Tool tests running in isolated venvs matching production component.yaml dependencies.

Each test class specifies a component.yaml and module path.  The test creates
(or reuses) a virtual environment with exactly the dependencies declared in
the component.yaml, then runs the tool inside that environment as a subprocess.
This ensures tests exercise the same dependency versions as production.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.conftest import run_tool_in_isolated_venv
from tests.shared_constants import (
    DEFAULT_CALLABLE,
    DELIVER_MSG_PREVIEW_LENGTH,
    RESULT_KEY_DELIVER_MSG,
    RESULT_KEY_ERRORS,
    RESULT_KEY_MODEL,
    RESULT_KEY_RESULTS,
    RESULT_KEY_SUCCESS,
    RESULT_KEY_TOOL,
)

PACKAGES_DIR = Path(__file__).parent.parent / "packages"
COMPONENT_YAML_FILENAME = "component.yaml"


def _component_config(relative_path: str) -> str:
    """Build the full path to a component.yaml from a relative package path."""
    return str(PACKAGES_DIR / relative_path / COMPONENT_YAML_FILENAME)


def _module_path_from_config(component_yaml: str) -> str:
    """Derive the Python module path from a component.yaml path.

    e.g. '.../packages/valory/customs/prediction_request/component.yaml'
      -> 'packages.valory.customs.prediction_request.prediction_request'
    """
    component_dir = Path(component_yaml).parent
    module_name = component_dir.name
    packages_idx = component_dir.parts.index("packages")
    package_parts = component_dir.parts[packages_idx:]
    return ".".join((*package_parts, module_name))


# Component configs (component.yaml paths)
# The prediction tools below were forked to _v1 in the anthropic SDK bump
# (originals deleted); these point at the live _v1 packages.
PREDICTION_REQUEST_V1_CONFIG = _component_config("valory/customs/prediction_request_v1")
PREDICTION_REQUEST_RAG_V1_CONFIG = _component_config(
    "napthaai/customs/prediction_request_rag_v1"
)
PREDICTION_REQUEST_REASONING_V1_CONFIG = _component_config(
    "napthaai/customs/prediction_request_reasoning_v1"
)
PREDICTION_URL_COT_V1_CONFIG = _component_config(
    "napthaai/customs/prediction_url_cot_v1"
)
DALLE_REQUEST_CONFIG = _component_config("victorpolisetty/customs/dalle_request")
SUPERFORCASTER_CONFIG = _component_config("valory/customs/superforcaster")
SUPERFORCASTER_POLYMARKET_V1_CONFIG = _component_config(
    "valory/customs/superforcaster_polymarket_v1"
)
FACTUAL_RESEARCH_CONFIG = _component_config("valory/customs/factual_research")

# Prompts
PREDICTION_PROMPT = (
    "Please take over the role of a Data Scientist to evaluate the given question. "
    'With the given question "Will Apple release iPhone 17 by March 2025?" '
    "and the `yes` option represented by `Yes` and the `no` option represented by `No`, "
    "what are the respective probabilities of `p_yes` and `p_no` occurring?"
)
PREDICTION_RAG_PROMPT = (
    'With the given question "Will NASA\'s Artemis II mission launch by December 31, 2026?" '
    "and the `yes` option represented by `Yes` and the `no` option represented by `No`, "
    "what are the respective probabilities of `p_yes` and `p_no` occurring?"
)
DALLE_PROMPT = "Generate an image of a futuristic cityscape."


def _format_failure(failure: Dict[str, Any]) -> str:
    """Format a single test failure into a readable string."""
    deliver_msg = failure.get(RESULT_KEY_DELIVER_MSG, "")[:DELIVER_MSG_PREVIEW_LENGTH]
    errors = "; ".join(failure[RESULT_KEY_ERRORS])
    return (
        f"  model={failure[RESULT_KEY_MODEL]}, tool={failure[RESULT_KEY_TOOL]}:\n"
        f"    errors: {errors}\n"
        f"    deliver_msg: {deliver_msg}"
    )


def _assert_all_passed(results: List[Dict[str, Any]]) -> None:
    """Assert all tool invocation results passed, with detailed failure messages."""
    assert results, "No test results returned from isolated runner."
    failures = [r for r in results if not r[RESULT_KEY_SUCCESS]]
    if not failures:
        return
    details = "\n".join(_format_failure(f) for f in failures)
    pytest.fail(f"{len(failures)}/{len(results)} tool invocations failed:\n{details}")


class BaseIsolatedToolTest:
    """Base class for tool tests that run in isolated component.yaml venvs."""

    component_yaml: str
    prompts: list
    callable_name: str = DEFAULT_CALLABLE
    validate_prediction: bool = True

    def test_run(self) -> None:
        """Run the tool in an isolated venv and validate results."""
        output = run_tool_in_isolated_venv(
            component_yaml=self.component_yaml,
            module_path=_module_path_from_config(self.component_yaml),
            prompts=self.prompts,
            callable_name=self.callable_name,
            validate_prediction=self.validate_prediction,
        )
        _assert_all_passed(output[RESULT_KEY_RESULTS])


class TestPredictionOnline(BaseIsolatedToolTest):
    """Test Prediction Online (v1)."""

    component_yaml = PREDICTION_REQUEST_V1_CONFIG
    prompts = [PREDICTION_PROMPT]


class TestPredictionRAG(BaseIsolatedToolTest):
    """Test Prediction RAG (v1)."""

    component_yaml = PREDICTION_REQUEST_RAG_V1_CONFIG
    prompts = [PREDICTION_RAG_PROMPT]


class TestPredictionReasoning(BaseIsolatedToolTest):
    """Test Prediction Reasoning (v1)."""

    component_yaml = PREDICTION_REQUEST_REASONING_V1_CONFIG
    prompts = [PREDICTION_PROMPT]


class TestPredictionCOT(BaseIsolatedToolTest):
    """Test Prediction COT (v1)."""

    component_yaml = PREDICTION_URL_COT_V1_CONFIG
    prompts = [PREDICTION_PROMPT]


class TestDALLEGeneration(BaseIsolatedToolTest):
    """Test DALL-E Generation."""

    component_yaml = DALLE_REQUEST_CONFIG
    prompts = [DALLE_PROMPT]
    validate_prediction = False


class TestSuperforcaster(BaseIsolatedToolTest):
    """Test Superforcaster."""

    component_yaml = SUPERFORCASTER_CONFIG
    prompts = [PREDICTION_PROMPT]


class TestSuperforcasterPolymarketV1(BaseIsolatedToolTest):
    """Test Superforcaster (Polymarket v1, uncalibrated)."""

    component_yaml = SUPERFORCASTER_POLYMARKET_V1_CONFIG
    prompts = [PREDICTION_PROMPT]


# Note: no ``TestSuperforcasterPolymarketV3`` here — matches the convention
# already in place for ``factual_research-v3`` (PR #339, also fable-5).
# The org's CLAUDE_API_KEY tier doesn't yet authorize ``claude-fable-5``
# (released 2026-06-09), so a live invocation returns "Model claude-fable-5
# not supported." and the test would always fail. Re-add this class once
# the org tier gains fable-5 access; the dispatch + dependency-import
# coverage that OjusWiZard's review asked for (PR #340 L259) is still
# fully exercised by the in-process unit tests in
# ``packages/valory/customs/superforcaster_polymarket_v3/tests/`` —
# notably ``TestV3LLMClientAnthropicCompletions`` (7 tests) and
# ``TestV3RunEndToEnd`` (2 tests).


class TestFactualResearch(BaseIsolatedToolTest):
    """Test Factual Research."""

    component_yaml = FACTUAL_RESEARCH_CONFIG
    prompts = [PREDICTION_PROMPT]
