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

"""Every model a tool can ask for must be priced by the vendored task_execution.

`TokenCounterCallback.TOKEN_PRICES` raises `ValueError` for an unpriced model
AFTER the LLM call has already succeeded, so a tool naming a model the table
does not carry burns the spend and then fails every delivery. The failure is
invisible until production.

This guard discovers tools from the filesystem rather than from a hand-written
list, so a tool added tomorrow is covered without anyone remembering to add it.
It makes no API calls and needs no keys.

It checks the VENDORED table (the `third_party` pin in `packages/packages.json`),
not mech's main branch: the pin is what the deployed agent actually runs, and
checking main would go green while production still rejected the model.
"""

import json
import re
import urllib.request
from pathlib import Path
from typing import Dict, Set

import pytest
import yaml

PACKAGES = Path(__file__).parent.parent / "packages"
TASK_EXECUTION_PIN = "skill/valory/task_execution/0.1.0"
GATEWAY = "https://gateway.autonolas.tech/ipfs"

# A model name is only a risk if it can reach TokenCounterCallback, which counts
# COMPLETION calls. Embedding calls never pass through it, so constants naming an
# embedding model are out of scope rather than exempted.
_EMBEDDING = re.compile(r"EMBEDDING")

# tool directory name -> why its unpriced models cannot break a delivery.
# Keep this EMPTY unless the reason is structural. "We'll price it later" is not
# a reason; add the price to mech instead.
EXEMPT: Dict[str, str] = {
    # Tournament-only: absent from agents/mech_predict/aea-config.yaml customs,
    # so it is never bundled into the agent and never served. Its qwen models
    # must be priced in mech BEFORE it is ever added to customs.
    "finetuned_prediction": "tournament-only, not in aea-config customs",
}

# Findings this guard raised that are not fixed here, tracked so CI is green
# while they are outstanding. The check still runs and, the day the finding is
# released and re-pinned, the entry no longer matches a real defect, this test
# fails, and whoever re-pinned is told to delete it. It cannot rot.
KNOWN_UNPRICED: Dict[str, str] = {}


def _vendored_benchmarks_source() -> str:
    """Return the vendored `benchmarks.py` source that the deployed agent runs.

    Prefers the synced copy on disk. That copy is gitignored and only populated
    by `autonomy packages sync` (the integration-tests env, which runs this
    file, syncs first), so a run without a prior sync falls back to fetching
    the pinned CID.

    :return: the source text of the pinned `task_execution/utils/benchmarks.py`.
    """
    on_disk = PACKAGES / "valory/skills/task_execution/utils/benchmarks.py"
    if on_disk.is_file():
        return on_disk.read_text(encoding="utf-8")

    pin = json.loads((PACKAGES / "packages.json").read_text(encoding="utf-8"))
    cid = pin["third_party"][TASK_EXECUTION_PIN]
    url = f"{GATEWAY}/{cid}/task_execution/utils/benchmarks.py"
    request = urllib.request.Request(url, headers={"User-Agent": "pricing-guard/1"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:  # nosec B310
            return response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        pytest.skip(f"vendored task_execution unavailable on disk and at {cid}: {exc}")
        raise  # pragma: no cover - pytest.skip raises


def _priced_models() -> Set[str]:
    """Parse the TOKEN_PRICES keys out of the vendored source.

    Parsed as text rather than imported: mech-predict vendors `task_execution`,
    so an import can silently resolve to a different copy than the pinned one.

    :return: the model names that have an entry in TOKEN_PRICES.
    """
    source = _vendored_benchmarks_source()
    block = re.search(r"TOKEN_PRICES\s*=\s*\{(.*?)\n    \}", source, re.DOTALL)
    assert block, "TOKEN_PRICES not found in the vendored benchmarks.py"
    return set(re.findall(r'"([^"]+)"\s*:\s*\{', block.group(1)))


def _declared_models(tool_dir: Path) -> Set[str]:
    """Every completion model this tool can end up asking for.

    Two surfaces, because tools use both: `default_model` in component.yaml
    (which task_execution passes when the requester omits `model`) and the
    `*MODEL*` / `*ENGINE*` constants in the sources.

    :param tool_dir: the tool package directory (holding component.yaml).
    :return: the non-empty model names the tool declares.
    """
    declared: Set[str] = set()

    config = yaml.safe_load((tool_dir / "component.yaml").read_text(encoding="utf-8"))
    for key, value in (config.get("params") or {}).items():
        if "model" in key.lower() and isinstance(value, str):
            declared.add(value)

    for source_file in tool_dir.glob("*.py"):
        source = source_file.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(
            r'^([A-Z_]*(?:MODEL|ENGINE)[A-Z_]*)\s*=\s*"([^"]+)"', source, re.MULTILINE
        ):
            name, value = match.groups()
            if not _EMBEDDING.search(name):
                declared.add(value)

    return {model for model in declared if model}


TOOL_DIRS = sorted(PACKAGES.glob("*/customs/*/component.yaml"))


def test_tools_were_discovered() -> None:
    """Guard the guard: an empty discovery would make every case vacuous."""
    assert len(TOOL_DIRS) > 20, f"only {len(TOOL_DIRS)} tools discovered"


@pytest.mark.parametrize("component", TOOL_DIRS, ids=lambda p: p.parent.name)
def test_declared_models_are_priced(component: Path) -> None:
    """Every model this tool can ask for is in the vendored TOKEN_PRICES."""
    tool_dir = component.parent
    if tool_dir.name in EXEMPT:
        pytest.skip(f"{tool_dir.name}: {EXEMPT[tool_dir.name]}")
    unpriced = sorted(_declared_models(tool_dir) - _priced_models())

    if tool_dir.name in KNOWN_UNPRICED:
        # Assert the defect is STILL there before excusing it. Without this the
        # entry would outlive its fix and silently suppress a real regression.
        assert unpriced, (
            f"{tool_dir.name} is listed in KNOWN_UNPRICED but every model it "
            f"declares is now priced. The fix landed: delete its entry."
        )
        pytest.xfail(f"{tool_dir.name}: {KNOWN_UNPRICED[tool_dir.name]}")

    assert not unpriced, (
        f"{tool_dir.name} declares {unpriced}, absent from the vendored "
        f"TOKEN_PRICES ({TASK_EXECUTION_PIN}). The price lookup raises AFTER "
        f"the LLM call succeeds, so every delivery would burn spend and fail. "
        f"Add the model to mech's benchmarks.py, release it, and re-pin here."
    )
