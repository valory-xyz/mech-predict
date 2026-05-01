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
"""Tests for benchmark/ipfs_loader.py — network mocked."""

import io
import tarfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from benchmark.ipfs_loader import (
    IPFS_GATEWAY,
    IpfsFetchError,
    fetch_tool_package,
    load_tool_from_ipfs,
)

CID = "bafybeitestcid"

COMPONENT_YAML = """\
name: testtool
author: valory
version: 0.1.0
type: custom
entry_point: testtool.py
callable: run
"""

ENTRY_PY = """\
def run(**kwargs):
    return ("ok", None, None, None, None)
"""


def _build_tar(
    wrapper_dir: str = "testtool",
    files: dict[str, str] | None = None,
    extra_root_prefix: str = CID,
) -> bytes:
    """Build a tar archive matching the gateway's wrapping convention.

    Layout: ``{cid}/{wrapper_dir}/{filename}`` for each file.

    :param wrapper_dir: name of the inner package directory.
    :param files: filename → text content map (default: a valid
        component.yaml + testtool.py pair).
    :param extra_root_prefix: outer directory entry name (the gateway uses
        the CID; tests can override).
    :return: tar archive bytes.
    """
    if files is None:
        files = {"component.yaml": COMPONENT_YAML, "testtool.py": ENTRY_PY}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        # Add the directory entries so the archive looks realistic.
        for dir_path in (extra_root_prefix, f"{extra_root_prefix}/{wrapper_dir}"):
            info = tarfile.TarInfo(name=dir_path)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=f"{extra_root_prefix}/{wrapper_dir}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _mock_response(status: int, content: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    return resp


class TestFetchToolPackage:
    """Tests for fetch_tool_package — happy path, cache, retries, fatal codes."""

    @patch("benchmark.ipfs_loader.requests.get")
    def test_happy_path_one_get(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """A single TAR fetch yields component.yaml + entry-point file."""
        mock_get.return_value = _mock_response(200, _build_tar())
        result = fetch_tool_package(CID, cache_dir=tmp_path)

        assert mock_get.call_count == 1
        assert result["component.yaml"] == COMPONENT_YAML
        assert result["testtool.py"] == ENTRY_PY
        # Cached flat under the CID dir
        assert (tmp_path / CID / "component.yaml").read_text() == COMPONENT_YAML
        assert (tmp_path / CID / "testtool.py").read_text() == ENTRY_PY
        # URL + headers — TAR accept header is what unlocks the wrapper
        call_args = mock_get.call_args
        assert call_args[0][0] == f"{IPFS_GATEWAY}/{CID}/"
        assert call_args[1]["headers"]["Accept"] == "application/x-tar"

    @patch("benchmark.ipfs_loader.requests.get")
    def test_cache_hit_skips_network(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Pre-populated cache means no HTTP calls."""
        cache_root = tmp_path / CID
        cache_root.mkdir(parents=True)
        (cache_root / "component.yaml").write_text(COMPONENT_YAML)
        (cache_root / "testtool.py").write_text(ENTRY_PY)

        result = fetch_tool_package(CID, cache_dir=tmp_path)

        assert mock_get.call_count == 0
        assert result["component.yaml"] == COMPONENT_YAML
        assert result["testtool.py"] == ENTRY_PY

    @patch.object(time, "sleep")
    @patch("benchmark.ipfs_loader.requests.get")
    def test_5xx_retries_then_succeeds(
        self,
        mock_get: MagicMock,
        _mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """5xx triggers retry; eventual 200 returns the package."""
        mock_get.side_effect = [
            _mock_response(503),
            _mock_response(200, _build_tar()),
        ]
        result = fetch_tool_package(CID, cache_dir=tmp_path)

        assert mock_get.call_count == 2
        assert result["component.yaml"] == COMPONENT_YAML

    @patch.object(time, "sleep")
    @patch("benchmark.ipfs_loader.requests.get")
    def test_5xx_exhausts_retries(
        self,
        mock_get: MagicMock,
        _mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """3 consecutive 5xx → IpfsFetchError mentioning the CID."""
        mock_get.return_value = _mock_response(503)
        with pytest.raises(IpfsFetchError) as exc_info:
            fetch_tool_package(CID, cache_dir=tmp_path)
        assert CID in str(exc_info.value)
        assert mock_get.call_count == 3  # _MAX_ATTEMPTS

    @patch.object(time, "sleep")
    @patch("benchmark.ipfs_loader.requests.get")
    def test_404_no_retry(
        self,
        mock_get: MagicMock,
        _mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """4xx is fatal — exactly one HTTP call, then IpfsFetchError."""
        mock_get.return_value = _mock_response(404)
        with pytest.raises(IpfsFetchError) as exc_info:
            fetch_tool_package(CID, cache_dir=tmp_path)
        assert "404" in str(exc_info.value)
        assert mock_get.call_count == 1

    @patch.object(time, "sleep")
    @patch("benchmark.ipfs_loader.requests.get")
    def test_connection_error_retries(
        self,
        mock_get: MagicMock,
        _mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Connection errors trigger retry, not fatal failure."""
        mock_get.side_effect = [
            requests.ConnectionError("dns down"),
            _mock_response(200, _build_tar()),
        ]
        result = fetch_tool_package(CID, cache_dir=tmp_path)
        assert mock_get.call_count == 2
        assert result["component.yaml"] == COMPONENT_YAML

    @patch("benchmark.ipfs_loader.requests.get")
    def test_malformed_tar_raises(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """A response that isn't a valid tar archive raises IpfsFetchError."""
        mock_get.return_value = _mock_response(200, b"this is not a tarball")
        with pytest.raises(IpfsFetchError):
            fetch_tool_package(CID, cache_dir=tmp_path)

    @patch("benchmark.ipfs_loader.requests.get")
    def test_archive_without_component_yaml(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """A tar that doesn't contain component.yaml raises IpfsFetchError."""
        mock_get.return_value = _mock_response(
            200, _build_tar(files={"random.py": "x = 1\n"})
        )
        with pytest.raises(IpfsFetchError) as exc_info:
            fetch_tool_package(CID, cache_dir=tmp_path)
        assert "component.yaml" in str(exc_info.value)

    @patch("benchmark.ipfs_loader.requests.get")
    def test_component_yaml_without_entry_point(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """component.yaml without entry_point raises IpfsFetchError."""
        mock_get.return_value = _mock_response(
            200, _build_tar(files={"component.yaml": "name: x\n"})
        )
        with pytest.raises(IpfsFetchError) as exc_info:
            fetch_tool_package(CID, cache_dir=tmp_path)
        assert "entry_point" in str(exc_info.value)

    @patch("benchmark.ipfs_loader.requests.get")
    def test_entry_point_missing_from_archive(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """component.yaml referencing a missing file raises IpfsFetchError."""
        # component.yaml says testtool.py but the archive only contains component.yaml
        mock_get.return_value = _mock_response(
            200, _build_tar(files={"component.yaml": COMPONENT_YAML})
        )
        with pytest.raises(IpfsFetchError) as exc_info:
            fetch_tool_package(CID, cache_dir=tmp_path)
        assert "testtool.py" in str(exc_info.value)


class TestLoadToolFromIpfs:
    """End-to-end tests: fetch + exec returns a callable."""

    @patch("benchmark.ipfs_loader.requests.get")
    def test_returns_callable(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """A successful fetch + exec returns the named callable."""
        mock_get.return_value = _mock_response(200, _build_tar())
        run_fn = load_tool_from_ipfs(CID, cache_dir=tmp_path)
        assert callable(run_fn)
        # Call it to confirm it's the run we wrote, not something else
        result = run_fn()
        assert result == ("ok", None, None, None, None)

    @patch("benchmark.ipfs_loader.requests.get")
    def test_missing_callable_raises(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """component.yaml pointing at an absent callable raises IpfsFetchError."""
        bad_yaml = COMPONENT_YAML.replace("callable: run", "callable: not_defined")
        mock_get.return_value = _mock_response(
            200, _build_tar(files={"component.yaml": bad_yaml, "testtool.py": ENTRY_PY})
        )
        with pytest.raises(IpfsFetchError) as exc_info:
            load_tool_from_ipfs(CID, cache_dir=tmp_path)
        assert "not_defined" in str(exc_info.value)
