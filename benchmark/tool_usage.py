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
"""Fetch per-deployment IRRELEVANT_TOOLS lists so the daily report can annotate them.

Each consumer (olas-operate-app Pearl, quickstart QS) stores the list as a
JSON-encoded string in a public config on GitHub ``main``.  We return a
``{deployment: [tool_names] | None}`` map; ``None`` signals a fetch or parse
failure for that deployment so consumers can render "unavailable" rather
than falsely claiming "no tools disabled".
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
    "omenstrat QS",
    "polystrat Pearl",
)

# Maps each deployment name to the scorer's platform key. Drives the
# per-platform filter on the Tool Deployment Status section.
DEPLOYMENT_TO_PLATFORM: Mapping[str, str] = MappingProxyType(
    {
        "omenstrat Pearl": "omen",
        "omenstrat QS": "omen",
        "polystrat Pearl": "polymarket",
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


# Source URLs (GitHub raw, ``main`` branch).
OPERATE_APP_TRADER_TS_URL = (
    "https://raw.githubusercontent.com/valory-xyz/olas-operate-app/"
    "main/frontend/constants/serviceTemplates/service/trader.ts"
)
QUICKSTART_CONFIG_URL = (
    "https://raw.githubusercontent.com/valory-xyz/quickstart/"
    "main/configs/config_predict_trader.json"
)

# Fetch timeout (seconds). Short so a stalled GitHub never blocks the
# daily report pipeline.
FETCH_TIMEOUT = 10

# Matches an ``IRRELEVANT_TOOLS`` block anchored to a specific exported
# template name (e.g. ``PREDICT_SERVICE_TEMPLATE``).  The capture group is
# the JSON-array-of-strings payload under ``value:``.  Anchoring to the
# template name protects us against silent relabel if the two template
# declarations are ever reordered in ``trader.ts``.
_TS_TEMPLATE_BLOCK_TEMPLATE = (
    r"{template_name}\b[\s\S]*?"
    r"IRRELEVANT_TOOLS\s*:\s*\{{[^}}]*?value\s*:\s*'(?P<value>\[[^']*\])'"
)
_OMENSTRAT_PEARL_TEMPLATE = "PREDICT_SERVICE_TEMPLATE"
_POLYSTRAT_PEARL_TEMPLATE = "PREDICT_POLYMARKET_SERVICE_TEMPLATE"


def _normalize_tool_name(name: str) -> str:
    """Return a canonical form for cross-convention tool-name matching.

    Operate-app and quickstart lists sometimes use ``prediction_request_X``
    while the benchmark logs use ``prediction-request-X`` for the same tool;
    the config authors defensively list both variants.  Treat underscores
    and hyphens as interchangeable so we don't under-report.

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
    with urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        return resp.read().decode("utf-8")


def _extract_template_irrelevant_tools(source: str, template_name: str) -> list[str]:
    """Extract the IRRELEVANT_TOOLS list for one named template.

    :param source: full text of ``trader.ts``.
    :param template_name: exported template identifier
        (e.g. ``PREDICT_SERVICE_TEMPLATE``).
    :return: parsed list of tool names.
    :raises ValueError: when no IRRELEVANT_TOOLS block is found for this template.
    """
    pattern = re.compile(
        _TS_TEMPLATE_BLOCK_TEMPLATE.format(template_name=template_name),
        re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        raise ValueError(
            f"no IRRELEVANT_TOOLS block found for template {template_name}"
        )
    return _parse_json_string_list(match.group("value"))


def _parse_json_string_list(raw: str) -> list[str]:
    """Parse a JSON-encoded array-of-strings.

    :param raw: JSON document expected to be an array of strings.
    :return: the parsed list.
    :raises ValueError: if ``raw`` is not a JSON array of strings.
    """
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError("IRRELEVANT_TOOLS value must be a JSON array of strings")
    return parsed


def parse_operate_app_ts(source: str) -> dict[str, list[str]]:
    """Parse IRRELEVANT_TOOLS for each exported template in ``trader.ts``.

    Template identity is anchored to the exported identifier
    (``PREDICT_SERVICE_TEMPLATE`` for omenstrat Pearl and
    ``PREDICT_POLYMARKET_SERVICE_TEMPLATE`` for polystrat Pearl), not file
    order, so silent relabel is impossible if the two declarations are
    ever reordered.  Propagates ``ValueError`` from
    ``_extract_template_irrelevant_tools`` when either template's block
    is missing.

    :param source: full text of ``trader.ts``.
    :return: ``{"omenstrat Pearl": [...], "polystrat Pearl": [...]}``.
    """
    return {
        "omenstrat Pearl": _extract_template_irrelevant_tools(
            source, _OMENSTRAT_PEARL_TEMPLATE
        ),
        "polystrat Pearl": _extract_template_irrelevant_tools(
            source, _POLYSTRAT_PEARL_TEMPLATE
        ),
    }


def parse_quickstart_config(source: str) -> list[str]:
    """Parse ``env_variables.IRRELEVANT_TOOLS.value`` from the quickstart JSON.

    Propagates ``KeyError`` when the expected path is missing, and
    ``ValueError`` (from ``_parse_json_string_list`` or ``json.loads``)
    when the value is not a JSON array of strings.

    :param source: full text of the quickstart ``config_predict_trader.json``.
    :return: the parsed list of irrelevant tool names.
    """
    data = json.loads(source)
    raw = data["env_variables"]["IRRELEVANT_TOOLS"]["value"]
    return _parse_json_string_list(raw)


def fetch_disabled_tools() -> dict[str, list[str] | None]:
    """Return ``{deployment: [tool_names] | None}`` for all deployments.

    ``None`` signals a fetch or parse failure for that deployment so the
    renderer can say "unavailable" instead of falsely claiming nothing is
    disabled.  Failures are logged but never raised — the daily report
    must never be blocked by a flaky GitHub fetch.

    :return: disabled-tools map keyed by deployment name.
    """
    disabled: dict[str, list[str] | None] = {name: None for name in DEPLOYMENTS}

    try:
        ts_source = _http_get(OPERATE_APP_TRADER_TS_URL)
        pearl = parse_operate_app_ts(ts_source)
        disabled["omenstrat Pearl"] = pearl["omenstrat Pearl"]
        disabled["polystrat Pearl"] = pearl["polystrat Pearl"]
    except (URLError, ValueError, OSError) as exc:
        log.warning("operate-app fetch/parse failed: %s", exc)

    try:
        qs_source = _http_get(QUICKSTART_CONFIG_URL)
        disabled["omenstrat QS"] = parse_quickstart_config(qs_source)
    except (URLError, ValueError, KeyError, OSError) as exc:
        log.warning("quickstart fetch/parse failed: %s", exc)

    return disabled
