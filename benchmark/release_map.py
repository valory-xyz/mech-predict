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
"""Build and query the CID -> release tag map on demand.

The map is computed by shelling out to ``gh release list`` and
``git show``, cached in memory for the process lifetime, and never
written to disk. Downstream callers import :func:`get_release_map` or
:func:`resolve` and the map is built lazily on first access.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_REPO = "valory-xyz/mech-predict"
DEFAULT_LIMIT = 200
UNTAGGED_PREFIX = "untagged@"
UNTAGGED_CID_CHARS = 8

_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def _run_gh_release_list(repo: str, limit: int) -> list[dict[str, str]]:
    """Return release descriptors sorted chronologically (oldest first).

    :param repo: ``owner/name`` slug passed to ``gh release list``.
    :param limit: ``--limit`` value (number of releases to fetch).
    :return: list of ``{"tagName": str, "createdAt": str}`` sorted by
        ``createdAt`` ascending. Empty list on failure.
    """
    try:
        out = subprocess.check_output(
            [
                "gh",
                "release",
                "list",
                "--repo",
                repo,
                "--limit",
                str(limit),
                "--json",
                "tagName,createdAt",
            ],
            text=True,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ) as err:
        stderr = getattr(err, "stderr", "") or ""
        log.warning("gh release list failed: %s | stderr=%s", err, stderr.strip())
        return []
    try:
        releases = json.loads(out)
    except json.JSONDecodeError as err:
        log.warning("gh release list returned non-JSON: %s", err)
        return []
    if len(releases) >= limit:
        log.warning(
            "gh release list hit --limit=%d; older tags may be truncated and "
            "CIDs from those tags will be misattributed to later tags. "
            "Raise DEFAULT_LIMIT or paginate.",
            limit,
        )
    return sorted(releases, key=lambda r: r.get("createdAt", ""))


def _run_git_show_packages_json(tag: str) -> dict[str, Any] | None:
    """Fetch ``packages.json`` content at *tag* via ``git show``.

    :param tag: a release tag (e.g. ``"v0.17.2"``).
    :return: parsed JSON dict, or None if the file doesn't exist at
        that tag or git/json parsing fails. Never raises.
    """
    try:
        out = subprocess.check_output(
            ["git", "show", f"refs/tags/{tag}:packages/packages.json"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=5,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ) as err:
        # Tag missing packages.json, shallow clone, or git unavailable.
        # All treated as "skip this tag". Debug-level so early tags that
        # legitimately predate packages.json don't spam WARN every run.
        stderr = getattr(err, "stderr", "") or ""
        log.debug(
            "git show %s:packages/packages.json failed: %s | stderr=%s",
            tag,
            err,
            stderr.strip(),
        )
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as err:
        log.debug("git show %s:packages/packages.json returned non-JSON: %s", tag, err)
        return None


def _build(repo: str = DEFAULT_REPO, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """Walk every release tag and record the first tag each CID appeared in.

    :param repo: ``owner/name`` slug passed to ``gh release list``.
    :param limit: ``--limit`` value (number of releases to fetch).
    :return: dict with keys ``generated_at``, ``tags_scanned``,
        ``cid_to_tag``, ``cid_to_package``. Empty dicts on complete
        failure (no raises).
    """
    releases = _run_gh_release_list(repo, limit)
    tags_scanned: list[str] = []
    cid_to_tag: dict[str, str] = {}
    cid_to_package: dict[str, str] = {}

    for rel in releases:
        tag = rel.get("tagName")
        if not tag:
            continue
        packages_json = _run_git_show_packages_json(tag)
        if packages_json is None:
            continue
        tags_scanned.append(tag)
        for key, cid in (packages_json.get("dev") or {}).items():
            if not isinstance(key, str) or not isinstance(cid, str):
                continue
            if not key.startswith("custom/"):
                continue
            # First tag wins — reintroduced CIDs keep their origin tag.
            if cid not in cid_to_tag:
                cid_to_tag[cid] = tag
                cid_to_package[cid] = key

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tags_scanned": tags_scanned,
        "cid_to_tag": cid_to_tag,
        "cid_to_package": cid_to_package,
    }


def get_release_map(
    repo: str = DEFAULT_REPO,
    limit: int = DEFAULT_LIMIT,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Return the cached release map. Builds on first call; cached after.

    Cache is keyed on ``(repo, limit)`` so non-default callers get their
    own entry instead of silently receiving a map built for a different
    repo or limit.

    :param repo: ``owner/name`` slug passed to ``gh release list``.
    :param limit: ``--limit`` value (number of releases to fetch).
    :param force_rebuild: when True, bypass the cache entry for this
        ``(repo, limit)`` and rebuild. Used by tests; production callers
        typically don't need this.
    :return: dict with keys ``generated_at``, ``tags_scanned``,
        ``cid_to_tag``, ``cid_to_package``. Always returns a shape-valid
        dict, even on failure (empty inner dicts).
    """
    key = (repo, limit)
    if force_rebuild or key not in _CACHE:
        _CACHE[key] = _build(repo=repo, limit=limit)
    return _CACHE[key]


def _untagged_label(cid: str) -> str:
    """Return the fallback label for a CID not present in the map."""
    short = cid[:UNTAGGED_CID_CHARS] if cid else "unknown"
    return f"{UNTAGGED_PREFIX}{short}"


def resolve(cid: str, release_map: dict[str, Any] | None = None) -> str:
    """Return the release tag for *cid*, or ``untagged@<short>`` when absent.

    Never raises — callers can display whatever comes back verbatim.

    :param cid: IPFS CID string (``"bafybei..."``).
    :param release_map: optional pre-loaded map. When None, a cached
        build is used via :func:`get_release_map`.
    :return: release tag (e.g. ``"v0.17.2"``) or ``"untagged@<first-8>"``.
    """
    if not cid:
        return _untagged_label("")
    if release_map is None:
        release_map = get_release_map()
    tag = (release_map.get("cid_to_tag") or {}).get(cid)
    if tag:
        return tag
    return _untagged_label(cid)


def sort_key(
    tag_label: str,
    tags_scanned: list[str],
    first_seen: str | None = None,
) -> tuple:
    """Return a sort key for ordering versions by release chronology.

    Tagged versions sort by their index in *tags_scanned* (earliest
    first). Untagged labels sort after all tagged ones, ordered by
    *first_seen* timestamp (None last).

    :param tag_label: label returned by :func:`resolve` — a release tag
        or an ``untagged@...`` string.
    :param tags_scanned: ordered tag list from the release map.
    :param first_seen: optional ISO timestamp for fallback ordering of
        untagged labels.
    :return: tuple usable as a sort key.
    """
    is_untagged = tag_label.startswith(UNTAGGED_PREFIX)
    if is_untagged:
        # (1, "", first_seen_or_empty, tag_label) sorts after all tagged
        # entries. tag_label is included as a deterministic tiebreak so
        # multiple untagged labels with no first_seen still sort stably
        # across runs (the label already encodes the CID's first 8 chars).
        return (1, "", first_seen or "~", tag_label)
    try:
        index = tags_scanned.index(tag_label)
    except ValueError:
        # Tag not in the scanned list (shouldn't happen in practice) —
        # sort at the end of the tagged group.
        index = len(tags_scanned)
    return (0, index, first_seen or "")


# ---------------------------------------------------------------------------
# CLI — inspection only. Never writes to a committed file.
# ---------------------------------------------------------------------------


def _cli_coverage(scores_path: str, release_map: dict[str, Any]) -> int:
    """Print CID resolution coverage against a ``scores.json`` file.

    :param scores_path: path to a scores JSON file with
        ``by_tool_version_mode`` keys.
    :param release_map: pre-built release map.
    :return: process exit code (0 always; diagnostic only).
    """
    with open(scores_path, encoding="utf-8") as f:
        scores = json.load(f)
    tvm = scores.get("by_tool_version_mode", {})
    total = 0
    resolved = 0
    unresolved: list[tuple[str, str]] = []
    for key in tvm:
        parts = [p.strip() for p in key.split("|")]
        if len(parts) != 3:
            continue
        tool, cid, _mode = parts
        if cid in ("unknown", ""):
            continue
        total += 1
        tag = (release_map.get("cid_to_tag") or {}).get(cid)
        if tag:
            resolved += 1
        else:
            unresolved.append((tool, cid[:12]))
    pct = (100.0 * resolved / total) if total else 0.0
    print(f"Coverage: {resolved}/{total} ({pct:.0f}%)")
    for tool, cid_short in unresolved:
        print(f"  unresolved: tool={tool}  cid={cid_short}...")
    return 0


def main() -> int:
    """CLI entry point for inspection.

    :return: process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Inspect the CID -> release tag map for mech-predict.",
    )
    parser.add_argument(
        "--cid",
        help="Resolve a single CID and print the tag (or untagged label).",
    )
    parser.add_argument(
        "--coverage",
        help=(
            "Path to a scores.json file. Report what fraction of "
            "by_tool_version_mode CIDs can be resolved."
        ),
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help="GitHub repo slug (owner/name)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of releases to fetch.",
    )
    args = parser.parse_args()

    # Always rebuild for the CLI — it's inspection of live state.
    release_map = _build(repo=args.repo, limit=args.limit)

    if args.cid:
        print(resolve(args.cid, release_map))
        return 0

    if args.coverage:
        return _cli_coverage(args.coverage, release_map)

    json.dump(release_map, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
