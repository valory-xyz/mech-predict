# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025-2026 Valory AG
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
"""The script allows the user to generate the metadata of the tools"""

import argparse
import copy
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT_DIR = "./packages"
CUSTOMS = "customs"
METADATA_FILE_PATH = "metadata.json"
COMPONENT_YAML = "component.yaml"
ENTRY_POINT = "entry_point"
SCHEMA_REGISTRY_PATH = Path(__file__).parent / "tool_schemas.yaml"
DEFAULT_KIND = "prediction"
ALLOWED_TOOLS = "ALLOWED_TOOLS"
AVAILABLE_TOOLS = "AVAILABLE_TOOLS"
# Ordered — ALLOWED_TOOLS wins when a module defines both.
TOOLS_IDENTIFIERS: Tuple[str, ...] = (ALLOWED_TOOLS, AVAILABLE_TOOLS)
# Every Valory operated mech must identify the Mech Terms in its metadata.
# Emitting the link from the template (rather than adding it by hand after a
# republish) is what keeps it from being lost on a regenerate-from-source.
TERMS_URL = "https://www.valory.xyz/terms/mechs"
DEFAULT_BENCHMARK_METRIC = "accuracy"
BENCHMARK_WINDOWS: Tuple[str, ...] = ("7d", "30d", "90d", "all")
# One DNS label: letters, digits and inner hyphens, at most 63 characters.
HOSTNAME_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.IGNORECASE)
# `name` is deliberately absent: every mech must be named explicitly.
METADATA_TEMPLATE: Dict[str, Any] = {
    "description": "The mech executes AI tasks requested on-chain and delivers the results to the requester.",
    "inputFormat": "ipfs-v0.1",
    "outputFormat": "ipfs-v0.1",
    "image": "tbd",
    "termsUrl": TERMS_URL,
    "tools": [],
    "toolMetadata": {},
}


def find_customs_folders(packages_root: Path) -> List[Path]:
    """Find all the customs folders inside the packages dir."""
    return [p for p in packages_root.rglob("*") if p.is_dir() and p.name == CUSTOMS]


def get_immediate_subfolders(folder_path: Path) -> List[Path]:
    """Find all the subfolders inside the dir."""
    return [item for item in folder_path.iterdir() if item.is_dir()]


def import_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    """Import the py file as a module."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{module_name}' from '{file_path}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_allowed_tools(module: ModuleType) -> List[str]:
    """Return the first defined tools list on the module, in identifier order."""
    for k in TOOLS_IDENTIFIERS:
        tools = getattr(module, k, None)
        if isinstance(tools, list):
            return list(tools)
    return []


def parse_tool_folder(sub: Path, allow_import_errors: bool) -> Optional[Dict[str, Any]]:
    """Parse a single tool directory into an entry dict, or None if unusable."""
    yaml_path = sub / COMPONENT_YAML
    if not yaml_path.is_file():
        print(f"Skipping {sub}: no {COMPONENT_YAML}")
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f.read())

    entry: Dict[str, Any] = {
        "author": data.get("author"),
        "tool_name": data.get("name"),
        "description": data.get("description"),
        "allowed_tools": [],
    }

    entry_point = data.get(ENTRY_POINT)
    if not entry_point:
        msg = f"{yaml_path} has no '{ENTRY_POINT}' field"
        if allow_import_errors:
            print(f"Warning: {msg}")
            return entry
        raise ValueError(msg)

    py_path = sub / entry_point
    if not py_path.is_file():
        msg = f"Entry point {py_path} declared in {yaml_path} does not exist"
        if allow_import_errors:
            print(f"Warning: {msg}")
            return entry
        raise FileNotFoundError(msg)

    module_name = f"{data.get('author', 'unknown')}_{sub.name}_{py_path.stem}"
    try:
        mod = import_module_from_path(module_name, py_path)
    except Exception as e:
        msg = f"Failed to import {py_path}: {e}"
        if allow_import_errors:
            print(f"Warning: {msg}")
            return entry
        raise

    entry["allowed_tools"] = _extract_allowed_tools(mod)
    return entry


def generate_tools_data(
    packages_root: Path, allow_import_errors: bool
) -> List[Dict[str, Any]]:
    """Generate the tools data needed for the metadata.json."""
    tools_data: List[Dict[str, Any]] = []
    for folder in find_customs_folders(packages_root):
        print(f"\n Matched folder: {folder}")
        for sub in get_immediate_subfolders(folder):
            print(f"  └── Subfolder: {sub}")
            entry = parse_tool_folder(sub, allow_import_errors)
            if entry:
                tools_data.append(entry)
    return tools_data


def load_schema_registry(path: Path) -> Dict[str, Any]:
    """Load schema registry, validate structure."""
    with open(path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f.read()) or {}
    defaults = reg.get("defaults") or {}
    tool_kinds = reg.get("tool_kinds") or {}
    if DEFAULT_KIND not in defaults:
        raise ValueError(
            f"Schema registry {path} must define a '{DEFAULT_KIND}' default"
        )
    for kind, schemas in defaults.items():
        if "input" not in schemas or "output" not in schemas:
            raise ValueError(
                f"Schema registry kind '{kind}' missing 'input' or 'output'"
            )
    for wire_name, kind in tool_kinds.items():
        if kind not in defaults:
            raise ValueError(
                f"Schema registry maps '{wire_name}' to unknown kind '{kind}'; "
                f"known kinds: {sorted(defaults)}"
            )
    return {"defaults": defaults, "tool_kinds": tool_kinds}


def build_tools_metadata(
    tools_data: List[Dict[str, Any]],
    registry: Dict[str, Any],
    template: Dict[str, Any],
    skip_tools: List[str],
) -> Dict[str, Any]:
    """Build the metadata.json from the tools data."""
    result: Dict[str, Any] = copy.deepcopy(template)
    defaults = registry["defaults"]
    tool_kinds = registry["tool_kinds"]
    skip_set = set(skip_tools)

    for entry in tools_data:
        author = entry.get("author", "")
        tool_name = entry.get("tool_name", "")
        allowed = entry.get("allowed_tools") or []
        if not allowed:
            print(
                f"Warning: '{tool_name}' by '{author}' has no allowed tools/invalid format!"
            )
            continue

        for tool in allowed:
            if tool in skip_set:
                print(f"Skipping tool (via --skip-tool): {tool}")
                continue
            if tool in result["toolMetadata"]:
                raise ValueError(
                    f"Duplicate wire name '{tool}' found in '{author}/{tool_name}'"
                )
            kind = tool_kinds.get(tool, DEFAULT_KIND)
            schemas = defaults[kind]
            result["tools"].append(tool)
            result["toolMetadata"][tool] = {
                "name": tool_name,
                "description": entry.get("description", ""),
                "input": schemas["input"],
                "output": schemas["output"],
            }

    return result


def validate_operator_domain(domain: str) -> str:
    """Return the domain if it is a bare hostname: no scheme, path, port or trailing dot."""
    labels = domain.split(".")
    if len(labels) < 2 or not all(HOSTNAME_LABEL.fullmatch(label) for label in labels):
        raise ValueError(
            f"--operator-domain must be a bare hostname such as 'valory.xyz', got {domain!r}"
        )
    return domain


def build_operator(
    name: Optional[str], domain: Optional[str], contact: Optional[str]
) -> Optional[Dict[str, str]]:
    """Build the operator block, or None when no operator flag was given."""
    if name is None and domain is None and contact is None:
        return None
    if not name or not domain:
        raise ValueError(
            "--operator-name and --operator-domain are both required "
            "to emit an operator block"
        )
    operator = {"name": name, "domain": validate_operator_domain(domain)}
    if contact:
        operator["contact"] = contact
    return operator


def parse_benchmark_value(raw: str) -> Tuple[str, float]:
    """Parse a TOOL=VALUE pair; VALUE must lie within 0..1."""
    tool, sep, value_text = raw.partition("=")
    if not sep or not tool:
        raise argparse.ArgumentTypeError(f"expected TOOL=VALUE, got {raw!r}")
    try:
        value = float(value_text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{raw!r}: value is not a number") from e
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{raw!r}: value must be within 0..1")
    return tool, value


def build_benchmarks(
    values: List[Tuple[str, float]],
    metric: str,
    window: Optional[str],
    url: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Map each tool to its benchmark object; metric, window and url are shared."""
    if not values:
        return {}
    if not window or not url:
        raise ValueError(
            "--benchmark-window and --benchmark-url are required "
            "when --benchmark-value is given"
        )
    benchmarks: Dict[str, Dict[str, Any]] = {}
    for tool, value in values:
        if tool in benchmarks:
            raise ValueError(f"--benchmark-value given twice for '{tool}'")
        benchmarks[tool] = {
            "metric": metric,
            "value": value,
            "window": window,
            "url": url,
        }
    return benchmarks


def attach_benchmarks(
    metadata: Dict[str, Any], benchmarks: Dict[str, Dict[str, Any]]
) -> None:
    """Add each benchmark to its tool's entry; every tool named must be in the output."""
    unknown = set(benchmarks) - set(metadata["toolMetadata"])
    if unknown:
        raise ValueError(
            f"--benchmark-value names tools missing from the output: {sorted(unknown)}"
        )
    for tool, benchmark in benchmarks.items():
        metadata["toolMetadata"][tool]["benchmark"] = benchmark


def build_template(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the top-level manifest fields from the CLI arguments."""
    fixed = copy.deepcopy(METADATA_TEMPLATE)
    template: Dict[str, Any] = {
        "name": args.name,
        "description": args.description,
        "inputFormat": fixed["inputFormat"],
        "outputFormat": fixed["outputFormat"],
        "image": args.image,
    }
    if args.url:
        template["url"] = args.url
    template["termsUrl"] = args.terms_url
    operator = build_operator(
        args.operator_name, args.operator_domain, args.operator_contact
    )
    if operator:
        template["operator"] = operator
    template["tools"] = fixed["tools"]
    template["toolMetadata"] = fixed["toolMetadata"]
    return template


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate metadata.json from custom tool packages."
    )
    parser.add_argument("--packages-root", type=Path, default=Path(ROOT_DIR))
    parser.add_argument("--output", type=Path, default=Path(METADATA_FILE_PATH))
    parser.add_argument(
        "--name", type=str, required=True, help="Human-readable name of the mech."
    )
    parser.add_argument(
        "--description", type=str, default=METADATA_TEMPLATE["description"]
    )
    parser.add_argument("--image", type=str, default=METADATA_TEMPLATE["image"])
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Off-chain request endpoint of the mech; omitted when not given.",
    )
    parser.add_argument("--terms-url", type=str, default=METADATA_TEMPLATE["termsUrl"])
    parser.add_argument(
        "--operator-name", type=str, default=None, help="Operator, free text."
    )
    parser.add_argument(
        "--operator-domain",
        type=str,
        default=None,
        help="Operator's bare hostname, e.g. valory.xyz; serves the domain proof.",
    )
    parser.add_argument(
        "--operator-contact", type=str, default=None, help="Operator contact."
    )
    parser.add_argument(
        "--benchmark-value",
        action="append",
        default=[],
        type=parse_benchmark_value,
        metavar="TOOL=VALUE",
        help="Benchmark value (0..1) for one tool (repeatable).",
    )
    parser.add_argument(
        "--benchmark-metric",
        type=str,
        default=DEFAULT_BENCHMARK_METRIC,
        help="Benchmark metric shared by every tool.",
    )
    parser.add_argument(
        "--benchmark-window",
        choices=BENCHMARK_WINDOWS,
        default=None,
        help="Benchmark window shared by every tool.",
    )
    parser.add_argument(
        "--benchmark-url",
        type=str,
        default=None,
        help="Analytics endpoint covering every tool of this mech.",
    )
    parser.add_argument(
        "--allow-import-errors",
        action="store_true",
        help="Log and continue on per-tool parse/import failures instead of raising.",
    )
    parser.add_argument(
        "--skip-tool",
        action="append",
        default=[],
        metavar="WIRE_NAME",
        help="Exclude this tool from the output (repeatable).",
    )
    parser.add_argument("--schema-registry", type=Path, default=SCHEMA_REGISTRY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Run the generate_metadata script."""
    args = parse_args(argv)

    template = build_template(args)
    benchmarks = build_benchmarks(
        args.benchmark_value,
        args.benchmark_metric,
        args.benchmark_window,
        args.benchmark_url,
    )
    registry = load_schema_registry(args.schema_registry)
    tools_data = generate_tools_data(args.packages_root, args.allow_import_errors)

    metadata = build_tools_metadata(tools_data, registry, template, args.skip_tool)
    attach_benchmarks(metadata, benchmarks)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata has been stored to {args.output}")


if __name__ == "__main__":
    main()
