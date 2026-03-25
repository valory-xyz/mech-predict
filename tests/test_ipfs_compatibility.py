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

"""Tests that all custom tool packages are compatible with the IPFS handler.

The IPFS connection handler (ipfs/connection.py:264) reads every file in a
downloaded package with ``open(path, encoding="utf-8", mode="r")``.  This
means:

1. No **directories** may appear in the fingerprint — they cause
   ``IsADirectoryError``.
2. No **binary files** may appear in the fingerprint — they cause
   ``UnicodeDecodeError``.

These tests catch both issues *before* deployment by reading the fingerprint
entries from each ``component.yaml`` and verifying they are readable text.
"""

import os
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

CUSTOM_PACKAGES = sorted(ROOT.glob("packages/*/customs/*/component.yaml"))


def _get_fingerprint_files(component_yaml: Path) -> List[str]:
    """Return the list of fingerprinted file paths from a component.yaml."""
    with open(component_yaml, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return list(data.get("fingerprint", {}).keys())


@pytest.mark.parametrize(
    "component_yaml",
    CUSTOM_PACKAGES,
    ids=[str(p.parent.name) for p in CUSTOM_PACKAGES],
)
class TestIpfsCompatibility:
    """Verify every fingerprinted file is IPFS-handler safe."""

    def test_no_directories_in_fingerprint(self, component_yaml: Path) -> None:
        """Fingerprint entries must be files, not directories."""
        pkg_dir = component_yaml.parent
        for rel_path in _get_fingerprint_files(component_yaml):
            full = pkg_dir / rel_path
            assert not full.is_dir(), (
                f"'{rel_path}' in {component_yaml.parent.name}/component.yaml "
                f"fingerprint is a directory. The IPFS handler will crash with "
                f"IsADirectoryError. Add '{rel_path.split('/')[0]}/*' to "
                f"fingerprint_ignore_patterns."
            )

    def test_all_fingerprinted_files_are_utf8(self, component_yaml: Path) -> None:
        """Every fingerprinted file must be readable as UTF-8 text."""
        pkg_dir = component_yaml.parent
        for rel_path in _get_fingerprint_files(component_yaml):
            full = pkg_dir / rel_path
            if not full.exists() or full.is_dir():
                continue
            try:
                full.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pytest.fail(
                    f"'{rel_path}' in {component_yaml.parent.name}/component.yaml "
                    f"fingerprint is not valid UTF-8. The IPFS handler will crash "
                    f"with UnicodeDecodeError. Store binary data as base64 in a "
                    f".py file instead."
                )
