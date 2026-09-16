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

"""Tests for the metadata generator's fixed fields."""

import json
from pathlib import Path

import pytest

from scripts.generate_metadata import METADATA_TEMPLATE, TERMS_URL, main


def _generate(tmp_path: Path, *extra_args: str) -> dict:
    """Run the generator against an empty packages root and load its output."""
    packages_root = tmp_path / "packages"
    packages_root.mkdir()
    output = tmp_path / "metadata.json"
    main(["--packages-root", str(packages_root), "--output", str(output), *extra_args])
    return json.loads(output.read_text(encoding="utf-8"))


def test_generated_metadata_carries_the_terms_link_by_default(tmp_path: Path) -> None:
    """A regenerate-from-source keeps termsUrl; nothing has to add it by hand."""
    metadata = _generate(tmp_path)
    assert metadata["termsUrl"] == TERMS_URL == "https://www.valory.xyz/terms/mechs"


def test_terms_url_flag_overrides_the_default(tmp_path: Path) -> None:
    """--terms-url reaches the output, like --name and --image do."""
    metadata = _generate(tmp_path, "--terms-url", "https://example.test/terms")
    assert metadata["termsUrl"] == "https://example.test/terms"


@pytest.mark.parametrize("field", ["name", "description", "image", "termsUrl"])
def test_template_fixed_fields_reach_the_output(tmp_path: Path, field: str) -> None:
    """Each fixed template field reaches the output."""
    assert _generate(tmp_path)[field] == METADATA_TEMPLATE[field]
