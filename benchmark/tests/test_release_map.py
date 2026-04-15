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
"""Tests for benchmark/release_map.py."""

import json
from typing import Any
from unittest.mock import patch

import pytest

from benchmark import release_map
from benchmark.release_map import (
    UNTAGGED_PREFIX,
    _build,
    get_release_map,
    resolve,
    sort_key,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Wipe module-level cache between tests."""
    release_map._CACHE = None  # pylint: disable=protected-access


def _fake_releases(
    *tags_in_order: tuple[str, str],
) -> list[dict[str, str]]:
    """Build a list of fake release descriptors in chronological order."""
    return [{"tagName": t, "createdAt": ts} for t, ts in tags_in_order]


def _packages_json(*entries: tuple[str, str]) -> dict[str, Any]:
    """Build a fake packages.json body with the given custom entries."""
    return {"dev": dict(entries)}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TestBuild:
    """Tests for the private _build() function."""

    def test_first_tag_wins(self) -> None:
        """A CID seen in v1.0.0 and v1.1.0 maps to v1.0.0."""
        releases = _fake_releases(
            ("v1.0.0", "2026-01-01T00:00:00Z"),
            ("v1.1.0", "2026-02-01T00:00:00Z"),
        )
        pkg_by_tag = {
            "v1.0.0": _packages_json(("custom/v/tool/0.1.0", "cidA")),
            "v1.1.0": _packages_json(("custom/v/tool/0.1.0", "cidA")),
        }

        with (
            patch.object(release_map, "_run_gh_release_list", return_value=releases),
            patch.object(
                release_map,
                "_run_git_show_packages_json",
                side_effect=pkg_by_tag.get,
            ),
        ):
            result = _build()

        assert result["cid_to_tag"]["cidA"] == "v1.0.0"

    def test_cid_reintroduction_preserves_origin(self) -> None:
        """CID present in v1.0.0, absent in v1.1.0, back in v1.2.0 -> v1.0.0."""
        releases = _fake_releases(
            ("v1.0.0", "2026-01-01T00:00:00Z"),
            ("v1.1.0", "2026-02-01T00:00:00Z"),
            ("v1.2.0", "2026-03-01T00:00:00Z"),
        )
        pkg_by_tag = {
            "v1.0.0": _packages_json(("custom/v/tool/0.1.0", "cidX")),
            "v1.1.0": _packages_json(("custom/v/tool/0.1.0", "cidY")),
            "v1.2.0": _packages_json(("custom/v/tool/0.1.0", "cidX")),
        }
        with (
            patch.object(release_map, "_run_gh_release_list", return_value=releases),
            patch.object(
                release_map,
                "_run_git_show_packages_json",
                side_effect=pkg_by_tag.get,
            ),
        ):
            result = _build()

        assert result["cid_to_tag"]["cidX"] == "v1.0.0"
        assert result["cid_to_tag"]["cidY"] == "v1.1.0"

    def test_shared_cid_across_runtime_tools_resolves_once(self) -> None:
        """One package CID covers many runtime tool names. Map holds one entry."""
        releases = _fake_releases(("v1.0.0", "2026-01-01T00:00:00Z"))
        # Only the package key matters; the map is CID-keyed.
        pkg = _packages_json(("custom/valory/prediction_request/0.1.0", "cidShared"))
        with (
            patch.object(release_map, "_run_gh_release_list", return_value=releases),
            patch.object(release_map, "_run_git_show_packages_json", return_value=pkg),
        ):
            result = _build()

        assert list(result["cid_to_tag"].keys()) == ["cidShared"]
        assert (
            result["cid_to_package"]["cidShared"]
            == "custom/valory/prediction_request/0.1.0"
        )

    def test_non_custom_keys_ignored(self) -> None:
        """agent/service/protocol entries in packages.json are not indexed."""
        releases = _fake_releases(("v1.0.0", "2026-01-01T00:00:00Z"))
        pkg = _packages_json(
            ("custom/v/tool/0.1.0", "cidCustom"),
            ("agent/valory/mech_predict/0.1.0", "cidAgent"),
            ("service/valory/mech_predict/0.1.0", "cidService"),
            ("protocol/open_aea/signing/1.0.0", "cidProto"),
        )
        with (
            patch.object(release_map, "_run_gh_release_list", return_value=releases),
            patch.object(release_map, "_run_git_show_packages_json", return_value=pkg),
        ):
            result = _build()

        assert set(result["cid_to_tag"].keys()) == {"cidCustom"}

    def test_chronological_tag_order(self) -> None:
        """Published-at timestamp (not semver) drives first-tag assignment.

        In this fixture v1.2.0 was published before v1.1.0 — a hotfix
        scenario. `_run_gh_release_list` already sorts by createdAt so
        the caller sees v1.0.0, v1.2.0, v1.1.0 in that order, and cidNew
        is first observed at v1.2.0.
        """
        releases = _fake_releases(
            ("v1.0.0", "2026-01-01T00:00:00Z"),
            ("v1.2.0", "2026-02-01T00:00:00Z"),
            ("v1.1.0", "2026-03-01T00:00:00Z"),
        )
        pkg_by_tag = {
            "v1.0.0": _packages_json(("custom/v/tool/0.1.0", "cidOld")),
            "v1.2.0": _packages_json(("custom/v/tool/0.1.0", "cidNew")),
            "v1.1.0": _packages_json(("custom/v/tool/0.1.0", "cidNew")),
        }
        with (
            patch.object(release_map, "_run_gh_release_list", return_value=releases),
            patch.object(
                release_map,
                "_run_git_show_packages_json",
                side_effect=pkg_by_tag.get,
            ),
        ):
            result = _build()

        assert result["cid_to_tag"]["cidNew"] == "v1.2.0"

    def test_missing_packages_json_at_old_tag_skipped(self) -> None:
        """A tag without packages/packages.json is silently skipped."""
        releases = _fake_releases(
            ("v0.0.1", "2026-01-01T00:00:00Z"),
            ("v1.0.0", "2026-02-01T00:00:00Z"),
        )
        pkg_by_tag: dict[str, Any] = {
            "v0.0.1": None,  # missing
            "v1.0.0": _packages_json(("custom/v/tool/0.1.0", "cid1")),
        }
        with (
            patch.object(release_map, "_run_gh_release_list", return_value=releases),
            patch.object(
                release_map,
                "_run_git_show_packages_json",
                side_effect=pkg_by_tag.get,
            ),
        ):
            result = _build()

        assert result["cid_to_tag"]["cid1"] == "v1.0.0"
        assert result["tags_scanned"] == ["v1.0.0"]

    def test_build_failure_yields_empty_map(self) -> None:
        """Failure of the release-list call returns an empty, shape-valid map."""
        with patch.object(release_map, "_run_gh_release_list", return_value=[]):
            result = _build()

        assert not result["cid_to_tag"]
        assert not result["cid_to_package"]
        assert not result["tags_scanned"]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCache:
    """Tests for the module-level cache in get_release_map()."""

    def test_cache_reuses_first_build(self) -> None:
        """Calling get_release_map twice runs _build only once."""
        with patch.object(
            release_map, "_build", return_value={"cid_to_tag": {"cidA": "v1.0.0"}}
        ) as mock_build:
            get_release_map()
            get_release_map()
            get_release_map()
        assert mock_build.call_count == 1

    def test_force_rebuild_bypasses_cache(self) -> None:
        """force_rebuild=True runs _build again."""
        with patch.object(
            release_map, "_build", return_value={"cid_to_tag": {}}
        ) as mock_build:
            get_release_map()
            get_release_map(force_rebuild=True)
        assert mock_build.call_count == 2


# ---------------------------------------------------------------------------
# Tests for resolve
# ---------------------------------------------------------------------------


class TestResolve:
    """Tests for resolve()."""

    def test_returns_tag_when_present(self) -> None:
        """A known CID returns its tag."""
        rm = {"cid_to_tag": {"cidA": "v0.17.2"}}
        assert resolve("cidA", rm) == "v0.17.2"

    def test_returns_untagged_label_when_missing(self) -> None:
        """An unknown CID returns 'untagged@<short>'."""
        rm: dict[str, Any] = {"cid_to_tag": {}}
        result = resolve("bafybei1234567890abc", rm)
        assert result.startswith(UNTAGGED_PREFIX)
        assert "bafybei1" in result

    def test_never_raises_on_empty_map(self) -> None:
        """An empty release_map never raises."""
        assert resolve("anything", {}).startswith(UNTAGGED_PREFIX)
        assert resolve("anything", {"cid_to_tag": None}).startswith(UNTAGGED_PREFIX)

    def test_empty_cid_returns_untagged(self) -> None:
        """Empty or None CID is safely handled."""
        rm = {"cid_to_tag": {"cidA": "v1.0.0"}}
        assert resolve("", rm).startswith(UNTAGGED_PREFIX)


# ---------------------------------------------------------------------------
# Tests for sort_key
# ---------------------------------------------------------------------------


class TestSortKey:
    """Tests for sort_key()."""

    def test_orders_by_tag_chronology(self) -> None:
        """Tagged versions sort by their index in tags_scanned."""
        tags = ["v1.0.0", "v1.1.0", "v1.2.0"]
        items = [("v1.2.0", "c"), ("v1.0.0", "a"), ("v1.1.0", "b")]
        items.sort(key=lambda x: sort_key(x[0], tags))
        assert [x[1] for x in items] == ["a", "b", "c"]

    def test_untagged_sorts_after_tagged(self) -> None:
        """Untagged entries sort after all tagged ones."""
        tags = ["v1.0.0", "v1.1.0"]
        items = [
            ("v1.1.0", "tagged_new"),
            (f"{UNTAGGED_PREFIX}bafybei1", "dev"),
            ("v1.0.0", "tagged_old"),
        ]
        items.sort(key=lambda x: sort_key(x[0], tags))
        assert [x[1] for x in items] == ["tagged_old", "tagged_new", "dev"]

    def test_untagged_fallback_to_first_seen(self) -> None:
        """Multiple untagged entries sort by first_seen."""
        tags: list[str] = []
        items = [
            (f"{UNTAGGED_PREFIX}b2", "later", "2026-03-01T00:00:00Z"),
            (f"{UNTAGGED_PREFIX}a1", "earlier", "2026-01-01T00:00:00Z"),
        ]
        items.sort(key=lambda x: sort_key(x[0], tags, first_seen=x[2]))
        assert [x[1] for x in items] == ["earlier", "later"]


# ---------------------------------------------------------------------------
# CLI JSON surface (smoke test)
# ---------------------------------------------------------------------------


class TestCliSmoke:
    """Smoke tests for the CLI surface."""

    def test_main_prints_map_as_json(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default CLI invocation prints the map as JSON."""
        fake_map = {
            "generated_at": "2026-04-15T00:00:00Z",
            "tags_scanned": ["v1.0.0"],
            "cid_to_tag": {"cidA": "v1.0.0"},
            "cid_to_package": {"cidA": "custom/v/tool/0.1.0"},
        }
        monkeypatch.setattr("sys.argv", ["release_map"])
        with patch.object(release_map, "_build", return_value=fake_map):
            release_map.main()
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["cid_to_tag"] == {"cidA": "v1.0.0"}

    def test_main_resolve_single_cid(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--cid flag resolves and prints a single tag."""
        monkeypatch.setattr("sys.argv", ["release_map", "--cid", "cidA"])
        with patch.object(
            release_map,
            "_build",
            return_value={"cid_to_tag": {"cidA": "v1.0.0"}},
        ):
            release_map.main()
        assert capsys.readouterr().out.strip() == "v1.0.0"
