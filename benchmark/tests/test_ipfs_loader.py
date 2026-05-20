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
import sys
import tarfile
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
import requests
from benchmark.ipfs_loader import (
    IPFS_GATEWAY,
    IpfsFetchError,
    _validate_entry_point_name,
    fetch_tool_package,
    load_tool_from_ipfs,
)

CID = "bafybeitestcid"


# Tests share a fixed `CID` constant while swapping out the on-disk
# source per case, so without this teardown a later test would hit a
# previous test's module cached under that CID in `sys.modules`.
@pytest.fixture(autouse=True)
def _clear_tournament_tool_modules() -> Iterator[None]:
    """Evict cached tournament-tool modules between tests.

    :yield: control to the test, then evict on teardown.
    """
    yield
    stale = [k for k in sys.modules if k.startswith("mech_predict_tournament_tool_")]
    for k in stale:
        sys.modules.pop(k, None)


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


def _mock_response(
    status: int,
    content: bytes = b"",
    headers: dict[str, str] | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    # Real requests responses expose `headers` as a dict-like. Default to
    # empty so headers.get(...) returns None rather than a MagicMock.
    resp.headers = headers or {}
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

    @pytest.mark.parametrize(
        "bad_entry",
        [
            "../escape.py",
            "../../escape.py",
            "/tmp/absolute.py",
            "sub/dir.py",
            "back\\slash.py",
            "..",
            ".",
            "",
        ],
    )
    @patch("benchmark.ipfs_loader.requests.get")
    def test_path_traversal_entry_point_rejected(
        self, mock_get: MagicMock, tmp_path: Path, bad_entry: str
    ) -> None:
        """Malicious entry_point values are rejected before any disk write."""
        bad_yaml = COMPONENT_YAML.replace(
            "entry_point: testtool.py", f"entry_point: {bad_entry!r}"
        )
        mock_get.return_value = _mock_response(
            200,
            _build_tar(files={"component.yaml": bad_yaml, "testtool.py": ENTRY_PY}),
        )
        with pytest.raises(IpfsFetchError) as exc_info:
            fetch_tool_package(CID, cache_dir=tmp_path)
        assert "entry_point" in str(exc_info.value)
        # Nothing was written outside cache_dir/{cid}/
        # No .py files anywhere except (possibly) inside the cid dir itself.
        for p in tmp_path.rglob("*.py"):
            assert p.parent == tmp_path / CID, f"file escaped cache: {p}"

    @patch("benchmark.ipfs_loader.requests.get")
    def test_cache_hit_with_traversal_entry_point_rejected(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """Cache-hit path validates entry_point even when yaml is already on disk."""
        cache_root = tmp_path / CID
        cache_root.mkdir(parents=True)
        bad_yaml = COMPONENT_YAML.replace(
            "entry_point: testtool.py", "entry_point: '../escape.py'"
        )
        (cache_root / "component.yaml").write_text(bad_yaml)
        with pytest.raises(IpfsFetchError) as exc_info:
            fetch_tool_package(CID, cache_dir=tmp_path)
        assert "entry_point" in str(exc_info.value)
        assert mock_get.call_count == 0

    @patch("benchmark.ipfs_loader.requests.get")
    def test_content_length_over_cap_rejected(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """Content-Length exceeding the cap aborts before reading the body."""
        mock_get.return_value = _mock_response(
            200,
            content=_build_tar(),
            headers={"Content-Length": str(64 * 1024 * 1024)},
        )
        with pytest.raises(IpfsFetchError) as exc_info:
            fetch_tool_package(CID, cache_dir=tmp_path)
        assert "too large" in str(exc_info.value)

    @patch("benchmark.ipfs_loader.requests.get")
    def test_content_length_under_cap_accepted(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """A small, declared Content-Length passes the guard cleanly."""
        tar_bytes = _build_tar()
        mock_get.return_value = _mock_response(
            200,
            content=tar_bytes,
            headers={"Content-Length": str(len(tar_bytes))},
        )
        result = fetch_tool_package(CID, cache_dir=tmp_path)
        assert result["component.yaml"] == COMPONENT_YAML

    @patch("benchmark.ipfs_loader.requests.get")
    def test_content_length_unparseable_treated_as_unknown(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """A non-numeric Content-Length doesn't raise — body still served."""
        mock_get.return_value = _mock_response(
            200,
            content=_build_tar(),
            headers={"Content-Length": "not-a-number"},
        )
        result = fetch_tool_package(CID, cache_dir=tmp_path)
        assert result["component.yaml"] == COMPONENT_YAML

    @patch("benchmark.ipfs_loader.requests.get")
    def test_corrupt_cached_yaml_evicts_and_refetches(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """Unparseable cached component.yaml is evicted; we re-fetch from gateway."""
        cache_root = tmp_path / CID
        cache_root.mkdir(parents=True)
        # YAML that fails safe_load: unclosed flow mapping.
        (cache_root / "component.yaml").write_text("entry_point: [unclosed\n")
        mock_get.return_value = _mock_response(200, _build_tar())

        result = fetch_tool_package(CID, cache_dir=tmp_path)

        assert mock_get.call_count == 1
        assert result["component.yaml"] == COMPONENT_YAML
        # Cache repopulated with the fresh yaml.
        assert (cache_root / "component.yaml").read_text() == COMPONENT_YAML


class TestValidateEntryPointName:
    """Direct unit tests for the entry_point validator."""

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            ".",
            "..",
            "../foo.py",
            "foo/bar.py",
            "foo\\bar.py",
            "/abs/path.py",
            "trailing/",
        ],
    )
    def test_rejects(self, bad: str) -> None:
        """Each malicious form raises IpfsFetchError."""
        with pytest.raises(IpfsFetchError):
            _validate_entry_point_name(bad, CID)

    @pytest.mark.parametrize("good", ["tool.py", "my_tool.py", "x.py", "tool"])
    def test_accepts(self, good: str) -> None:
        """Plain filenames pass through."""
        _validate_entry_point_name(good, CID)


class TestLoadToolFromIpfs:
    """End-to-end tests: fetch + import returns a callable."""

    @patch("benchmark.ipfs_loader.requests.get")
    def test_returns_callable(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """A successful fetch + import returns the named callable."""
        mock_get.return_value = _mock_response(200, _build_tar())
        run_fn = load_tool_from_ipfs(CID, cache_dir=tmp_path)
        assert callable(run_fn)
        # Call it to confirm it's the run we wrote, not something else
        result = run_fn()
        assert result == ("ok", None, None, None, None)

    # Regression test for the loader's original `exec(src, {})` path:
    # classes got `__module__ = 'builtins'` and pydantic 2.13 couldn't
    # resolve forward refs like `List[str]` when the OpenAI SDK validated
    # a `response_format=SomeModel` argument. Latent on PR #251 because
    # the smoke test ran superforcaster, which doesn't use pydantic.
    @patch("benchmark.ipfs_loader.requests.get")
    def test_pydantic_forward_refs_resolve(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """Loaded tools whose classes use forward refs (`List[str]`) must work."""
        pydantic_py = (
            "from typing import List\n"
            "from pydantic import BaseModel, Field\n"
            "\n"
            "class SubQuestions(BaseModel):\n"
            "    sub_questions: List[str] = Field(...)\n"
            "\n"
            "def run(**kwargs):\n"
            "    parsed = SubQuestions.model_validate(\n"
            "        {'sub_questions': ['q1']}\n"
            "    )\n"
            "    return (parsed.model_dump_json(), None, None, None, None)\n"
        )
        mock_get.return_value = _mock_response(
            200,
            _build_tar(
                files={"component.yaml": COMPONENT_YAML, "testtool.py": pydantic_py}
            ),
        )
        run_fn = load_tool_from_ipfs(CID, cache_dir=tmp_path)
        result_str, *_ = run_fn()
        assert "q1" in result_str

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

    @patch("benchmark.ipfs_loader.requests.get")
    def test_syntax_error_in_entry_point_raises_ipfs_fetch_error(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """A SyntaxError in the entry-point source surfaces as IpfsFetchError."""
        bad_py = "def run(:\n    pass\n"
        mock_get.return_value = _mock_response(
            200,
            _build_tar(files={"component.yaml": COMPONENT_YAML, "testtool.py": bad_py}),
        )
        with pytest.raises(IpfsFetchError) as exc_info:
            load_tool_from_ipfs(CID, cache_dir=tmp_path)
        assert "module load failed" in str(exc_info.value)

    @patch("benchmark.ipfs_loader.requests.get")
    def test_import_error_at_module_level_raises_ipfs_fetch_error(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """A failing top-level import surfaces as IpfsFetchError, not ImportError."""
        bad_py = (
            "import nonexistent_module_xyz_qqq\n"
            + "def run(**kwargs):\n"
            + "    return ('ok',)\n"
        )
        mock_get.return_value = _mock_response(
            200,
            _build_tar(files={"component.yaml": COMPONENT_YAML, "testtool.py": bad_py}),
        )
        with pytest.raises(IpfsFetchError) as exc_info:
            load_tool_from_ipfs(CID, cache_dir=tmp_path)
        assert "module load failed" in str(exc_info.value)

    @patch("benchmark.ipfs_loader.requests.get")
    def test_module_level_runtime_error_raises_ipfs_fetch_error(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """An exception thrown by module-level code surfaces as IpfsFetchError."""
        bad_py = (
            "raise RuntimeError('boom')\n"
            + "def run(**kwargs):\n"
            + "    return ('ok',)\n"
        )
        mock_get.return_value = _mock_response(
            200,
            _build_tar(files={"component.yaml": COMPONENT_YAML, "testtool.py": bad_py}),
        )
        with pytest.raises(IpfsFetchError) as exc_info:
            load_tool_from_ipfs(CID, cache_dir=tmp_path)
        assert "module load failed" in str(exc_info.value)

    @patch("benchmark.ipfs_loader.requests.get")
    def test_component_loader_value_error_surfaces_as_ipfs_fetch_error(
        self, mock_get: MagicMock, tmp_path: Path
    ) -> None:
        """A ValueError from ComponentPackageLoader surfaces as IpfsFetchError."""
        # Yaml is missing 'callable' → ComponentPackageLoader.load raises ValueError.
        bad_yaml = "name: x\nentry_point: testtool.py\n"
        mock_get.return_value = _mock_response(
            200,
            _build_tar(files={"component.yaml": bad_yaml, "testtool.py": ENTRY_PY}),
        )
        with pytest.raises(IpfsFetchError) as exc_info:
            load_tool_from_ipfs(CID, cache_dir=tmp_path)
        # Either "load failed" (if it gets to ComponentPackageLoader) or earlier
        # parse-time rejection — either way the wrapper contained the error.
        assert "IPFS fetch failed" in str(exc_info.value)
