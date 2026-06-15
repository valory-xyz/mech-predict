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
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from packages.valory.skills.task_execution.utils.apis import KeyChain

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

# Closed set of prompt-schema families. Typed as a Literal so a registry typo
# (e.g. ``family="reasonning"``) is a mypy error rather than a silent route to
# the default branch — the exact silent-misroute class this field exists to kill.
FamilyName = Literal[
    "reasoning", "rag", "superforcaster", "factual_research", "default"
]


@dataclass(frozen=True)
class ToolSpec:
    """Specification for a prediction tool.

    ``family`` is the single source of truth for prompt-schema dispatch. It
    decides both (a) which attribute symbols the replay path reads from the
    tool's module (``PREDICTION_PROMPT`` / ``SYSTEM_PROMPT`` / ``ESTIMATE_USER``
    / …) and the kwargs it formats them with, and (b) which regex extractor the
    enrich path uses to parse the IPFS prompt. Both ``prompt_replay`` dispatch
    sites read this field via ``_baseline_family`` so they cannot drift.

    It is required (no default) on purpose: a silent wrong default is exactly
    the failure mode that left enrichment parsing 0 rows for sibling baselines.
    A new ``-v<n+1>`` variant must declare the same family as its parent.
    """

    module: str
    family: FamilyName


TOOL_REGISTRY: dict[str, ToolSpec] = {
    # valory/prediction_request_v1
    "prediction-online-v1": ToolSpec(
        module="packages.valory.customs.prediction_request_v1.prediction_request_v1",
        family="default",
    ),
    "prediction-offline-v1": ToolSpec(
        module="packages.valory.customs.prediction_request_v1.prediction_request_v1",
        family="default",
    ),
    "claude-prediction-online-v1": ToolSpec(
        module="packages.valory.customs.prediction_request_v1.prediction_request_v1",
        family="default",
    ),
    "claude-prediction-offline-v1": ToolSpec(
        module="packages.valory.customs.prediction_request_v1.prediction_request_v1",
        family="default",
    ),
    # valory/superforcaster
    "superforcaster": ToolSpec(
        module="packages.valory.customs.superforcaster.superforcaster",
        family="superforcaster",
    ),
    # valory/superforcaster_polymarket_v1
    "superforcaster-polymarket-v1": ToolSpec(
        module=(
            "packages.valory.customs.superforcaster_polymarket_v1"
            ".superforcaster_polymarket_v1"
        ),
        family="superforcaster",
    ),
    # valory/superforcaster_polymarket_v3 — sibling of v2; both depart from v1.
    # v3 swaps the default LLM model from gpt-4.1 to claude-fable-5 via the
    # dual-SDK convention; family is unchanged (the parent + sibling all share
    # the same superforcaster PREDICTION_PROMPT schema).
    "superforcaster-polymarket-v3": ToolSpec(
        module=(
            "packages.valory.customs.superforcaster_polymarket_v3"
            ".superforcaster_polymarket_v3"
        ),
        family="superforcaster",
    ),
    # napthaai/prediction_request_reasoning_v1
    "prediction-request-reasoning-v1": ToolSpec(
        module="packages.napthaai.customs.prediction_request_reasoning_v1.prediction_request_reasoning_v1",
        family="reasoning",
    ),
    "prediction-request-reasoning-claude-v1": ToolSpec(
        module="packages.napthaai.customs.prediction_request_reasoning_v1.prediction_request_reasoning_v1",
        family="reasoning",
    ),
    # napthaai/prediction_request_rag_v1
    "prediction-request-rag-v1": ToolSpec(
        module="packages.napthaai.customs.prediction_request_rag_v1.prediction_request_rag_v1",
        family="rag",
    ),
    "prediction-request-rag-claude-v1": ToolSpec(
        module="packages.napthaai.customs.prediction_request_rag_v1.prediction_request_rag_v1",
        family="rag",
    ),
    # napthaai/prediction_url_cot_v1 — rag-shaped (PREDICTION_PROMPT + SYSTEM_PROMPT,
    # <user_prompt> XML layout), NOT the default backtick format.
    "prediction-url-cot-v1": ToolSpec(
        module="packages.napthaai.customs.prediction_url_cot_v1.prediction_url_cot_v1",
        family="rag",
    ),
    "prediction-url-cot-claude-v1": ToolSpec(
        module="packages.napthaai.customs.prediction_url_cot_v1.prediction_url_cot_v1",
        family="rag",
    ),
    # valory/factual_research
    "factual_research": ToolSpec(
        module="packages.valory.customs.factual_research.factual_research",
        family="factual_research",
    ),
    # valory/factual_research_v1 — declares family="factual_research" (same as
    # its parent) so prompt_replay dispatch reads the correct schema/regex pair.
    "factual_research-v1": ToolSpec(
        module="packages.valory.customs.factual_research_v1.factual_research_v1",
        family="factual_research",
    ),
    # valory/factual_research_v2 — declares family="factual_research" (same as
    # its parent) so prompt_replay dispatch reads the correct schema/regex pair.
    "factual_research-v2": ToolSpec(
        module="packages.valory.customs.factual_research_v2.factual_research_v2",
        family="factual_research",
    ),
    # valory/factual_research_v3 — same prompts + Polymarket-rules pipeline as
    # v2, swaps the LLM backend from OpenAI gpt-4.1 to Anthropic claude-fable-5
    # via JSON-schema prompt injection + Pydantic model_validate_json on the
    # response (the Anthropic SDK's forced-tool-use mechanism is NOT used).
    # Family unchanged (parent + siblings share the same ESTIMATE_USER /
    # REFRAME_USER / SYNTHESIS_USER templates and resolution_rules kwarg).
    "factual_research-v3": ToolSpec(
        module="packages.valory.customs.factual_research_v3.factual_research_v3",
        family="factual_research",
    ),
    # nickcom007/prediction_request_sme — default schema: PREDICTION_PROMPT uses
    # {user_prompt}/{additional_information} and the module exports no
    # SYSTEM_PROMPT_FORECASTER. (It is NOT superforcaster-shaped despite
    # exporting only PREDICTION_PROMPT.)
    "prediction-offline-sme": ToolSpec(
        module="packages.nickcom007.customs.prediction_request_sme.prediction_request_sme",
        family="default",
    ),
    "prediction-online-sme": ToolSpec(
        module="packages.nickcom007.customs.prediction_request_sme.prediction_request_sme",
        family="default",
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
_ipfs_cache: dict[str, Callable[..., Any]] = {}


def load_tool_run(
    tool_name: str,
    *,
    cid: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Callable[..., Any]:
    """Resolve a tool's run() callable.

    Two modes:

    - ``cid`` is set (tournament): fetch the tool source from IPFS by CID,
      ``exec`` it, return the run callable. Cached in-process by CID — two
      runs in the same process resolving the same name to different CIDs
      get distinct callables.
    - ``cid`` is None (replay scripts): import via ``TOOL_REGISTRY`` from
      on-disk packages. Cached by tool name.

    :param tool_name: tool identifier (only used for error messages and the
        importlib cache key).
    :param cid: when provided, fetch from IPFS instead of importing.
    :param cache_dir: optional override for the IPFS source cache location.
    :return: the tool's run callable.
    """
    if cid is not None:
        cached = _ipfs_cache.get(cid)
        if cached is not None:
            return cached
        from benchmark.ipfs_loader import (  # pylint: disable=import-outside-toplevel
            DEFAULT_CACHE_DIR,
            load_tool_from_ipfs,
        )

        run_fn = load_tool_from_ipfs(cid, cache_dir or DEFAULT_CACHE_DIR)
        _ipfs_cache[cid] = run_fn
        return run_fn

    if tool_name in _tool_cache:
        return _tool_cache[tool_name]

    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        raise ValueError(
            f"Unknown tool: {tool_name}. Available: {sorted(TOOL_REGISTRY)}"
        )

    module = importlib.import_module(spec.module)
    run_fn = module.run
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
