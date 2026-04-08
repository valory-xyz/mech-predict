"""
Shared tool infrastructure for benchmark scripts.

Provides the tool registry, keychain builder, tool loader, and timeout
helpers used by runner.py, tournament.py, sweep.py, and notify_slack.py.
"""

from __future__ import annotations

import importlib
import os
import platform
import threading
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from packages.valory.skills.task_execution.utils.apis import KeyChain

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Specification for a prediction tool."""

    module: str


TOOL_REGISTRY: dict[str, ToolSpec] = {
    # valory/prediction_request
    "prediction-online": ToolSpec(
        module="packages.valory.customs.prediction_request.prediction_request",
    ),
    "prediction-offline": ToolSpec(
        module="packages.valory.customs.prediction_request.prediction_request",
    ),
    "claude-prediction-online": ToolSpec(
        module="packages.valory.customs.prediction_request.prediction_request",
    ),
    "claude-prediction-offline": ToolSpec(
        module="packages.valory.customs.prediction_request.prediction_request",
    ),
    # valory/superforcaster
    "superforcaster": ToolSpec(
        module="packages.valory.customs.superforcaster.superforcaster",
    ),
    # napthaai/prediction_request_reasoning
    "prediction-request-reasoning": ToolSpec(
        module="packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning",
    ),
    "prediction-request-reasoning-claude": ToolSpec(
        module="packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning",
    ),
    # napthaai/prediction_request_rag
    "prediction-request-rag": ToolSpec(
        module="packages.napthaai.customs.prediction_request_rag.prediction_request_rag",
    ),
    "prediction-request-rag-claude": ToolSpec(
        module="packages.napthaai.customs.prediction_request_rag.prediction_request_rag",
    ),
    # napthaai/prediction_url_cot
    "prediction-url-cot": ToolSpec(
        module="packages.napthaai.customs.prediction_url_cot.prediction_url_cot",
    ),
    "prediction-url-cot-claude": ToolSpec(
        module="packages.napthaai.customs.prediction_url_cot.prediction_url_cot",
    ),
    # nickcom007/prediction_request_sme
    "prediction-offline-sme": ToolSpec(
        module="packages.nickcom007.customs.prediction_request_sme.prediction_request_sme",
    ),
    "prediction-online-sme": ToolSpec(
        module="packages.nickcom007.customs.prediction_request_sme.prediction_request_sme",
    ),
}


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------


def build_keychain(*, return_source_content: bool = False) -> "KeyChain":
    """Build a KeyChain from environment variables.

    Each service gets a single-key list.  Missing keys become empty strings
    so that tools relying on other providers don't crash at import time.

    :param return_source_content: when True the keychain tells tools to
        capture web content into used_params (tournament mode).  When False
        (default) tools skip capture (replay mode).
    :return: a KeyChain populated from environment variables.
    """
    from packages.valory.skills.task_execution.utils.apis import (  # pylint: disable=import-outside-toplevel
        KeyChain,
    )

    services: dict[str, list[str]] = {
        "openai": [os.environ.get("OPENAI_API_KEY", "")],
        "anthropic": [os.environ.get("ANTHROPIC_API_KEY", "")],
        "google_api_key": [os.environ.get("GOOGLE_API_KEY", "")],
        "google_engine_id": [os.environ.get("GOOGLE_ENGINE_ID", "")],
        "serperapi": [os.environ.get("SERPER_API_KEY", "")],
        "openrouter": [os.environ.get("OPENROUTER_API_KEY", "")],
        "search_provider": [os.environ.get("SEARCH_PROVIDER", "google")],
        "return_source_content": ["true" if return_source_content else "false"],
    }
    return KeyChain(services)


# ---------------------------------------------------------------------------
# Tool loading
# ---------------------------------------------------------------------------

_tool_cache: dict[str, Callable[..., Any]] = {}


def load_tool_run(tool_name: str) -> Callable[..., Any]:
    """Import and cache a tool's run() function."""
    if tool_name in _tool_cache:
        return _tool_cache[tool_name]

    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        raise ValueError(
            f"Unknown tool: {tool_name}. Available: {sorted(TOOL_REGISTRY)}"
        )

    module = importlib.import_module(spec.module)
    run_fn: Callable[..., Any] = module.run
    _tool_cache[tool_name] = run_fn
    return run_fn


# ---------------------------------------------------------------------------
# Timeout helpers
# ---------------------------------------------------------------------------

_HAS_SIGALRM = platform.system() != "Windows"


def _can_use_sigalrm() -> bool:
    """SIGALRM only works on UNIX from the main thread."""
    return _HAS_SIGALRM and threading.current_thread() is threading.main_thread()


class ToolTimeout(Exception):
    """Raised when a tool exceeds its execution deadline."""


def alarm_handler(signum: int, frame: Any) -> None:
    """Signal handler that raises ToolTimeout on SIGALRM."""
    raise ToolTimeout("Tool execution timed out")
