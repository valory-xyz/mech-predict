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
"""Resolve each Pearl deployment's VALID_TOOLS allow-list so the report can annotate it.

Each Pearl deployment ships a pinned ``valory-xyz/trader`` release. We
read the operate-app ``main`` ``trader.ts`` to learn that pin
(``service_version`` per template), then read ``VALID_TOOLS`` from the
trader release's ``service.yaml``. We return a
``{deployment: [tool_names] | None}`` map; ``None`` signals a fetch or
parse failure for that deployment so consumers can render "unavailable"
rather than falsely claiming an allow-list.

operate-app is read from ``main`` (not a release tag) because operate-app
release publishing is unreliable; trader is read from the exact tag
operate-app pins.
"""

from __future__ import annotations

import json
import logging
import re
from types import MappingProxyType
from typing import Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

# Deployments we report on. Order is the column order in the rendered line.
DEPLOYMENTS: tuple[str, ...] = (
    "omenstrat Pearl",
    "polystrat Pearl",
)

# Maps each deployment name to the scorer's platform key. Drives the
# per-platform filter on the Tool Deployment Status section.
DEPLOYMENT_TO_PLATFORM: Mapping[str, str] = MappingProxyType(
    {
        "omenstrat Pearl": "omen",
        "polystrat Pearl": "polymarket",
    }
)

# Per-deployment resolution config: the operate-app exported template
# identifier whose ``service_version`` pins the trader release, and the
# trader service directory whose ``service.yaml`` carries ``VALID_TOOLS``.
_DEPLOYMENT_CONFIG: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "omenstrat Pearl": ("PREDICT_SERVICE_TEMPLATE", "trader_pearl"),
        "polystrat Pearl": (
            "PREDICT_POLYMARKET_SERVICE_TEMPLATE",
            "polymarket_trader",
        ),
    }
)


def deployments_for_platform(platform: str) -> tuple[str, ...]:
    """Return the deployment names belonging to ``platform``, in declared order.

    :param platform: scorer platform key (``"omen"`` or ``"polymarket"``).
    :return: deployments matching ``platform``, preserving ``DEPLOYMENTS`` order.
    """
    return tuple(
        name for name in DEPLOYMENTS if DEPLOYMENT_TO_PLATFORM.get(name) == platform
    )


# Source URLs. operate-app is pinned to ``main`` (release publishing is
# unreliable upstream); trader is pinned to the tag operate-app declares.
OPERATE_APP_TRADER_TS_URL = (
    "https://raw.githubusercontent.com/valory-xyz/olas-operate-app/"
    "main/frontend/constants/serviceTemplates/service/trader.ts"
)
TRADER_SERVICE_YAML_URL = (
    "https://raw.githubusercontent.com/valory-xyz/trader/"
    "{ref}/packages/valory/services/{service}/service.yaml"
)

# Fetch timeout (seconds). Short so a stalled GitHub never blocks the
# daily report pipeline.
FETCH_TIMEOUT = 10

# Captures the ``VALID_TOOLS`` env-override default from a trader
# ``service.yaml``. The payload is a JSON array of double-quoted
# strings; tool names never contain ``]`` so ``[^\]]*`` is a safe,
# newline-tolerant body match.
_VALID_TOOLS_RE = re.compile(
    r"valid_tools\s*:\s*\$\{VALID_TOOLS:list:(?P<list>\[[^\]]*\])\}"
)


def _normalize_tool_name(name: str) -> str:
    """Return a canonical form for cross-convention tool-name matching.

    operate-app/trader lists sometimes use ``prediction_request_X`` while
    the benchmark logs use ``prediction-request-X`` for the same tool;
    config authors defensively list both variants.  Treat underscores and
    hyphens as interchangeable so we don't under-report.

    :param name: raw tool name.
    :return: canonical form with ``_`` replaced by ``-``.
    """
    return name.replace("_", "-")


def _http_get(url: str) -> str:
    """GET ``url`` and return the body as text.

    Propagates ``URLError`` (or subclasses) on network failure so callers
    can decide how to degrade.

    :param url: HTTP(S) URL to fetch.
    :return: response body decoded as UTF-8.
    """
    req = Request(url, headers={"User-Agent": "mech-predict-benchmark"})
    with urlopen(
        req, timeout=FETCH_TIMEOUT
    ) as resp:  # nosec B310 — fixed mech-tool registry URL
        return resp.read().decode("utf-8")


def _parse_json_string_list(raw: str) -> list[str]:
    """Parse a JSON-encoded array-of-strings.

    :param raw: JSON document expected to be an array of strings.
    :return: the parsed list.
    :raises ValueError: if ``raw`` is not a JSON array of strings.
    """
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError("VALID_TOOLS value must be a JSON array of strings")
    return parsed


def parse_service_version(ts_source: str, template_name: str) -> str:
    """Extract ``service_version`` for one named operate-app template.

    Anchored to the exported identifier with a word boundary so a reorder
    of the template declarations cannot silently swap versions, and so
    ``PREDICT_SERVICE_TEMPLATE`` does not match inside
    ``PREDICT_POLYMARKET_SERVICE_TEMPLATE``. The gap before
    ``service_version`` is forbidden from crossing the next
    ``export const`` so a template missing its own ``service_version``
    raises rather than silently borrowing the following template's.

    :param ts_source: full text of operate-app ``trader.ts``.
    :param template_name: exported template identifier
        (e.g. ``PREDICT_SERVICE_TEMPLATE``).
    :return: the pinned trader tag (e.g. ``v0.38.0-rc1``).
    :raises ValueError: when the template or its ``service_version`` is absent.
    """
    pattern = re.compile(
        rf"{re.escape(template_name)}\b(?:(?!export const)[\s\S])*?"
        r"service_version\s*:\s*'(?P<ver>[^']+)'"
    )
    match = pattern.search(ts_source)
    if match is None:
        raise ValueError(f"no service_version found for template {template_name}")
    return match.group("ver")


def parse_valid_tools(yaml_source: str) -> list[str]:
    """Extract the ``VALID_TOOLS`` allow-list from a trader ``service.yaml``.

    There is intentionally no ``irrelevant_tools`` fallback: a missing
    ``valid_tools`` env default is treated as a parse failure so the
    caller renders "unavailable" rather than silently inverting
    semantics.

    :param yaml_source: full text of the trader ``service.yaml``.
    :return: the parsed allow-list (possibly empty — an empty allow-list
        is a legitimate "no tools allowed" state, distinct from a failure).
    :raises ValueError: when the ``valid_tools`` env default is absent or
        its payload is not a JSON array of strings.
    """
    match = _VALID_TOOLS_RE.search(yaml_source)
    if match is None:
        raise ValueError("no valid_tools env default found in service.yaml")
    return _parse_json_string_list(match.group("list"))


def fetch_valid_tools() -> dict[str, list[str] | None]:
    """Return ``{deployment: [valid_tool_names] | None}`` for all deployments.

    For each deployment: read operate-app ``main`` ``trader.ts`` once,
    resolve that deployment's trader ``service_version``, fetch the trader
    ``service.yaml`` at that tag, and parse ``VALID_TOOLS``. ``None``
    signals a fetch or parse failure for that deployment so the renderer
    can say "unavailable" instead of falsely claiming an allow-list.
    Failures are logged but never raised — the daily report must never be
    blocked by a flaky GitHub fetch.

    :return: valid-tools map keyed by deployment name.
    """
    valid: dict[str, list[str] | None] = {name: None for name in DEPLOYMENTS}

    try:
        ts_source = _http_get(OPERATE_APP_TRADER_TS_URL)
    except (URLError, OSError) as exc:
        log.warning("operate-app trader.ts fetch failed: %s", exc)
        return valid

    for deployment, (template, service) in _DEPLOYMENT_CONFIG.items():
        try:
            ref = parse_service_version(ts_source, template)
            yaml_source = _http_get(
                TRADER_SERVICE_YAML_URL.format(ref=ref, service=service)
            )
            valid[deployment] = parse_valid_tools(yaml_source)
        except (URLError, ValueError, OSError) as exc:
            log.warning("%s valid_tools resolution failed: %s", deployment, exc)

    return valid
