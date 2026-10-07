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

"""Tests for the metadata generator's manifest fields."""

import json
import textwrap
from pathlib import Path
from typing import Any, Dict

import pytest

from scripts.generate_metadata import (
    DEFAULT_BENCHMARK_METRIC,
    METADATA_TEMPLATE,
    SCHEMA_REGISTRY_PATH,
    TERMS_URL,
    load_schema_registry,
    main,
    validate_operator_domain,
)

SAMPLE_NAME = "Test Mech"
SAMPLE_URL = "https://mech.example.test"
SAMPLE_OPERATOR = ("Valory", "valory.xyz", "mechs@valory.xyz")
SAMPLE_BENCHMARK_URL = "https://analytics.example.test/v1/metrics/mech/100/0xabc"
SAMPLE_TOOL = "prediction_alpha"
OTHER_TOOL = "prediction_beta"
TOOL_SCHEMA_FIELDS = ("name", "description", "input", "output")
TOOL_DESCRIPTION = "Predicts things."
BASE_ARGS = ("--name", SAMPLE_NAME)
BENCHMARK_ARGS = (
    "--benchmark-window",
    "30d",
    "--benchmark-url",
    SAMPLE_BENCHMARK_URL,
)


def _write_tool_package(packages_root: Path, tool: str) -> None:
    """Create a minimal custom tool package that registers one wire name."""
    tool_dir = packages_root / "author" / "customs" / tool
    tool_dir.mkdir(parents=True)
    (tool_dir / "component.yaml").write_text(
        textwrap.dedent(f"""\
            name: {tool}
            author: author
            description: {TOOL_DESCRIPTION}
            entry_point: {tool}.py
            """),
        encoding="utf-8",
    )
    (tool_dir / f"{tool}.py").write_text(
        f'ALLOWED_TOOLS = ["{tool}"]\n', encoding="utf-8"
    )


def _generate(tmp_path: Path, *extra_args: str, tools: tuple = ()) -> Dict[str, Any]:
    """Run the generator against a packages root holding `tools` and load its output."""
    packages_root = tmp_path / "packages"
    packages_root.mkdir()
    for tool in tools:
        _write_tool_package(packages_root, tool)
    output = tmp_path / "metadata.json"
    main(["--packages-root", str(packages_root), "--output", str(output), *extra_args])
    return json.loads(output.read_text(encoding="utf-8"))


def test_name_is_required(tmp_path: Path) -> None:
    """There is no default mech name; omitting --name is a usage error."""
    with pytest.raises(SystemExit):
        _generate(tmp_path)
    assert "name" not in METADATA_TEMPLATE


def test_name_flag_reaches_the_output(tmp_path: Path) -> None:
    """--name is the only source of the manifest name."""
    assert _generate(tmp_path, *BASE_ARGS)["name"] == SAMPLE_NAME


def test_generated_metadata_carries_the_terms_link_by_default(tmp_path: Path) -> None:
    """A regenerate-from-source keeps termsUrl; nothing has to add it by hand."""
    metadata = _generate(tmp_path, *BASE_ARGS)
    assert metadata["termsUrl"] == TERMS_URL == "https://www.valory.xyz/terms/mechs"


def test_terms_url_flag_overrides_the_default(tmp_path: Path) -> None:
    """--terms-url reaches the output, like --name and --image do."""
    metadata = _generate(tmp_path, *BASE_ARGS, "--terms-url", "https://example.test/t")
    assert metadata["termsUrl"] == "https://example.test/t"


@pytest.mark.parametrize("field", ["description", "image", "termsUrl"])
def test_template_fixed_fields_reach_the_output(tmp_path: Path, field: str) -> None:
    """Each fixed template field reaches the output."""
    assert _generate(tmp_path, *BASE_ARGS)[field] == METADATA_TEMPLATE[field]


def test_url_flag_reaches_the_output(tmp_path: Path) -> None:
    """--url carries the off-chain endpoint, so a regenerate keeps it."""
    assert _generate(tmp_path, *BASE_ARGS, "--url", SAMPLE_URL)["url"] == SAMPLE_URL


def test_url_is_omitted_when_not_given(tmp_path: Path) -> None:
    """An on-chain-only mech has no url key rather than an empty one."""
    assert "url" not in _generate(tmp_path, *BASE_ARGS)


def test_operator_block_reaches_the_output(tmp_path: Path) -> None:
    """The three operator flags land under one `operator` object."""
    name, domain, contact = SAMPLE_OPERATOR
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        "--operator-name",
        name,
        "--operator-domain",
        domain,
        "--operator-contact",
        contact,
    )
    assert metadata["operator"] == {"name": name, "domain": domain, "contact": contact}


def test_operator_contact_is_optional(tmp_path: Path) -> None:
    """Without --operator-contact the block has no contact key."""
    name, domain, _ = SAMPLE_OPERATOR
    metadata = _generate(
        tmp_path, *BASE_ARGS, "--operator-name", name, "--operator-domain", domain
    )
    assert metadata["operator"] == {"name": name, "domain": domain}


def test_operator_is_omitted_when_no_operator_flag_is_given(tmp_path: Path) -> None:
    """No operator flags means no operator block."""
    assert "operator" not in _generate(tmp_path, *BASE_ARGS)


@pytest.mark.parametrize(
    "flags",
    [
        ("--operator-name", "Valory"),
        ("--operator-domain", "valory.xyz"),
        ("--operator-contact", "mechs@valory.xyz"),
        ("--operator-name", "Valory", "--operator-contact", "mechs@valory.xyz"),
        ("--operator-name", "", "--operator-domain", "valory.xyz"),
    ],
)
def test_operator_requires_both_name_and_domain(tmp_path: Path, flags: tuple) -> None:
    """A partial operator block is an error, not a silently incomplete one."""
    with pytest.raises(ValueError, match="--operator-name and --operator-domain"):
        _generate(tmp_path, *BASE_ARGS, *flags)


@pytest.mark.parametrize(
    "domain",
    [
        "https://valory.xyz",
        "valory.xyz/",
        "valory.xyz/.well-known",
        "valory.xyz.",
        ".valory.xyz",
        "valory.xyz:443",
        "valory",
        "val ory.xyz",
    ],
)
def test_operator_domain_rejects_anything_but_a_bare_hostname(
    tmp_path: Path, domain: str
) -> None:
    """Scheme, path, port, trailing dot and single labels are all rejected."""
    with pytest.raises(ValueError, match="bare hostname"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            "--operator-name",
            "Valory",
            "--operator-domain",
            domain,
        )


@pytest.mark.parametrize("domain", ["-valory.xyz", "valory-.xyz", "valory..xyz", ""])
def test_validate_operator_domain_rejects_malformed_labels(domain: str) -> None:
    """Labels with edge hyphens or empty labels are rejected by the validator itself."""
    with pytest.raises(ValueError, match="bare hostname"):
        validate_operator_domain(domain)


@pytest.mark.parametrize("domain", ["valory.xyz", "mechs.valory.xyz", "a-b.co.uk"])
def test_operator_domain_accepts_bare_hostnames(tmp_path: Path, domain: str) -> None:
    """A plain hostname passes through unchanged."""
    metadata = _generate(
        tmp_path, *BASE_ARGS, "--operator-name", "Valory", "--operator-domain", domain
    )
    assert metadata["operator"]["domain"] == domain


def test_tool_entry_keeps_schema_fields_alongside_benchmark(tmp_path: Path) -> None:
    """A benchmark is added to the tool entry without displacing its schema fields."""
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        *BENCHMARK_ARGS,
        "--benchmark-value",
        f"{SAMPLE_TOOL}=0.83",
        tools=(SAMPLE_TOOL,),
    )
    entry = metadata["toolMetadata"][SAMPLE_TOOL]
    schemas = load_schema_registry(SCHEMA_REGISTRY_PATH)["defaults"]["prediction"]
    assert metadata["tools"] == [SAMPLE_TOOL]
    assert set(entry) == {*TOOL_SCHEMA_FIELDS, "benchmark"}
    assert entry["name"] == SAMPLE_TOOL
    assert entry["description"] == TOOL_DESCRIPTION
    assert entry["input"] == schemas["input"]
    assert entry["output"] == schemas["output"]
    assert entry["benchmark"] == {
        "metric": DEFAULT_BENCHMARK_METRIC,
        "value": 0.83,
        "window": "30d",
        "url": SAMPLE_BENCHMARK_URL,
    }


def test_benchmark_metric_flag_overrides_the_default(tmp_path: Path) -> None:
    """--benchmark-metric replaces the default metric name."""
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        *BENCHMARK_ARGS,
        "--benchmark-metric",
        "brier",
        "--benchmark-value",
        f"{SAMPLE_TOOL}=0.5",
        tools=(SAMPLE_TOOL,),
    )
    assert metadata["toolMetadata"][SAMPLE_TOOL]["benchmark"]["metric"] == "brier"


def test_tool_without_a_benchmark_value_has_no_benchmark_key(tmp_path: Path) -> None:
    """Only tools that were given a value carry a benchmark object."""
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        *BENCHMARK_ARGS,
        "--benchmark-value",
        f"{SAMPLE_TOOL}=0.83",
        tools=(SAMPLE_TOOL, OTHER_TOOL),
    )
    assert "benchmark" in metadata["toolMetadata"][SAMPLE_TOOL]
    assert set(metadata["toolMetadata"][OTHER_TOOL]) == set(TOOL_SCHEMA_FIELDS)


def test_benchmark_value_for_a_tool_missing_from_the_output_raises(
    tmp_path: Path,
) -> None:
    """A mistyped or skipped tool name fails loudly instead of dropping the benchmark."""
    with pytest.raises(ValueError, match="missing from the output"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            *BENCHMARK_ARGS,
            "--benchmark-value",
            f"{OTHER_TOOL}=0.83",
            tools=(SAMPLE_TOOL,),
        )


def test_benchmark_value_given_twice_for_one_tool_raises(tmp_path: Path) -> None:
    """Two values for one tool is ambiguous; neither silently wins."""
    with pytest.raises(ValueError, match="given twice"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            *BENCHMARK_ARGS,
            "--benchmark-value",
            f"{SAMPLE_TOOL}=0.8",
            "--benchmark-value",
            f"{SAMPLE_TOOL}=0.9",
            tools=(SAMPLE_TOOL,),
        )


@pytest.mark.parametrize(
    "raw",
    [
        f"{SAMPLE_TOOL}=-0.1",
        f"{SAMPLE_TOOL}=1.01",
        f"{SAMPLE_TOOL}=nan",
        f"{SAMPLE_TOOL}=high",
        f"{SAMPLE_TOOL}=",
        SAMPLE_TOOL,
        "=0.5",
    ],
)
def test_benchmark_value_must_be_tool_equals_unit_interval(
    tmp_path: Path, raw: str
) -> None:
    """Anything but TOOL=<number in 0..1> is a usage error."""
    with pytest.raises(SystemExit):
        _generate(
            tmp_path,
            *BASE_ARGS,
            *BENCHMARK_ARGS,
            "--benchmark-value",
            raw,
            tools=(SAMPLE_TOOL,),
        )


@pytest.mark.parametrize("value", ["0", "1", "0.5"])
def test_benchmark_value_accepts_the_unit_interval_bounds(
    tmp_path: Path, value: str
) -> None:
    """0 and 1 are valid values, not off-by-one rejections."""
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        *BENCHMARK_ARGS,
        "--benchmark-value",
        f"{SAMPLE_TOOL}={value}",
        tools=(SAMPLE_TOOL,),
    )
    assert metadata["toolMetadata"][SAMPLE_TOOL]["benchmark"]["value"] == float(value)


def test_benchmark_window_rejects_unknown_values(tmp_path: Path) -> None:
    """The window is one of the spec's four literals."""
    with pytest.raises(SystemExit):
        _generate(tmp_path, *BASE_ARGS, "--benchmark-window", "14d")


@pytest.mark.parametrize(
    "flags",
    [
        ("--benchmark-window", "30d"),
        ("--benchmark-url", SAMPLE_BENCHMARK_URL),
        (),
    ],
)
def test_benchmark_value_requires_window_and_url(tmp_path: Path, flags: tuple) -> None:
    """A value without the shared window and url cannot form a complete benchmark."""
    with pytest.raises(ValueError, match="--benchmark-window and --benchmark-url"):
        _generate(
            tmp_path,
            *BASE_ARGS,
            *flags,
            "--benchmark-value",
            f"{SAMPLE_TOOL}=0.5",
            tools=(SAMPLE_TOOL,),
        )


def test_regenerated_manifest_keeps_every_spec_field(tmp_path: Path) -> None:
    """One full invocation yields url, termsUrl, operator and a per-tool benchmark."""
    name, domain, contact = SAMPLE_OPERATOR
    metadata = _generate(
        tmp_path,
        *BASE_ARGS,
        "--url",
        SAMPLE_URL,
        "--operator-name",
        name,
        "--operator-domain",
        domain,
        "--operator-contact",
        contact,
        *BENCHMARK_ARGS,
        "--benchmark-value",
        f"{SAMPLE_TOOL}=0.83",
        tools=(SAMPLE_TOOL,),
    )
    assert list(metadata) == [
        "name",
        "description",
        "inputFormat",
        "outputFormat",
        "image",
        "url",
        "termsUrl",
        "operator",
        "tools",
        "toolMetadata",
    ]
    assert metadata["url"] == SAMPLE_URL
    assert metadata["termsUrl"] == TERMS_URL
    assert metadata["operator"]["domain"] == domain
    assert metadata["toolMetadata"][SAMPLE_TOOL]["benchmark"]["value"] == 0.83
