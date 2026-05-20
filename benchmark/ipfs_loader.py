"""
IPFS-backed tool loader for tournament mode.

Fetches tool source from a public IPFS gateway by CID, caches on disk,
imports the entry-point as a proper module, and returns the run callable.
Used by tournament mode so candidate tool versions can be benchmarked
without merging them to packages.json.

One TAR GET per tool: ``Accept: application/x-tar`` on the CID returns a
tarball whose top-level entry is a wrapper directory (named after the
package, e.g. ``superforcaster/``) containing ``component.yaml``, the
entry-point ``.py``, and noise we ignore (``__init__.py``, ``tests/``).
We extract just the two files we need into ``cache_dir/{cid}/`` and
hand them to ``ComponentPackageLoader.load``.

The entry-point is loaded via ``importlib.util.spec_from_file_location``
so classes defined in it get a real ``__module__`` registered in
``sys.modules``. ``exec(source, namespace)`` is unsafe for tool code
that uses pydantic ``BaseModel`` with forward references (e.g.
``List[str]``) — pydantic resolves forward refs by looking up
``cls.__module__`` in ``sys.modules`` and falls over with
``class-not-fully-defined`` when the class was exec'd into a bare dict
namespace whose ``__module__`` resolves to ``'builtins'``.
"""

from __future__ import annotations

import importlib.util
import io
import logging
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests
import yaml

from packages.valory.skills.task_execution.utils.ipfs import ComponentPackageLoader

IPFS_GATEWAY = "https://gateway.autonolas.tech/ipfs"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mech-predict" / "tournament-tools"

_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SEC = 1.0
_TAR_ACCEPT_HEADER = {"Accept": "application/x-tar"}
# Refuse tarballs over 32 MiB — real tool packages are <1 MiB. Caps heap
# blow-up from a bad CID before resp.content materializes the body.
_MAX_PACKAGE_BYTES = 32 * 1024 * 1024

log = logging.getLogger(__name__)


class IpfsFetchError(Exception):
    """Raised when fetching a tool package from IPFS fails irrecoverably."""


def _validate_entry_point_name(name: str, cid: str) -> None:
    """Reject entry_point values that aren't a single safe filename.

    ``component.yaml``'s ``entry_point`` is used as a filename under
    ``cache_dir/{cid}/``. ``pathlib`` joins literally:
    ``cache_dir/{cid}/../../foo`` escapes the cache root,
    ``/tmp/foo`` replaces it. Both are blocked here.

    :param name: the raw ``entry_point`` value from component.yaml.
    :param cid: CID being processed (for error messages).
    :raises IpfsFetchError: if ``name`` is not a single safe path component.
    """
    has_separator = "/" in name or "\\" in name
    is_dot_path = name in (".", "..")
    is_absolute = bool(name) and Path(name).is_absolute()
    if (
        not name
        or has_separator
        or is_dot_path
        or is_absolute
        or Path(name).name != name
    ):
        raise IpfsFetchError(
            f"IPFS fetch failed: cid={cid} entry_point {name!r} "
            f"is not a single safe filename"
        )


def _get_tar_with_retries(url: str, *, cid: str) -> bytes:
    """GET ``url`` as a tar archive with retries on 5xx / connection errors.

    4xx is fatal — bad CID, no point retrying.

    :param url: full gateway URL.
    :param cid: CID being fetched (for error messages).
    :return: raw response body bytes.
    :raises IpfsFetchError: on final failure.
    """
    last_status: Optional[int] = None
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(
                url,
                headers=_TAR_ACCEPT_HEADER,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_BACKOFF_BASE_SEC * (2 ** (attempt - 1)))
                continue
            raise IpfsFetchError(
                f"IPFS fetch failed: cid={cid} url={url} "
                f"after {_MAX_ATTEMPTS} attempts: {exc!r}"
            ) from exc

        last_status = resp.status_code
        if resp.status_code == 200:
            content_length = resp.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError:
                    declared = -1
                if declared > _MAX_PACKAGE_BYTES:
                    raise IpfsFetchError(
                        f"IPFS fetch failed: cid={cid} response too large "
                        f"({declared} bytes > {_MAX_PACKAGE_BYTES} byte cap)"
                    )
            return resp.content
        if 400 <= resp.status_code < 500:
            # Client errors are fatal: no point retrying a 404 / 410.
            raise IpfsFetchError(
                f"IPFS fetch failed: cid={cid} url={url} "
                f"status={resp.status_code} (no retry on 4xx)"
            )
        if attempt < _MAX_ATTEMPTS:
            time.sleep(_BACKOFF_BASE_SEC * (2 ** (attempt - 1)))

    raise IpfsFetchError(
        f"IPFS fetch failed: cid={cid} url={url} "
        f"last_status={last_status} last_exc={last_exc!r} "
        f"after {_MAX_ATTEMPTS} attempts"
    )


def _cache_path(cache_dir: Path, cid: str, filename: str) -> Path:
    return cache_dir / cid / filename


def _extract_files_from_tar(tar_bytes: bytes, cid: str) -> tuple[str, str, str]:
    """Pull ``component.yaml`` and the entry-point ``.py`` out of a tar archive.

    The autonolas gateway wraps the package directory inside a top-level
    entry named after the CID, then a subdirectory named after the
    package. We don't know the subdirectory name, so we walk the archive
    looking for any ``*/component.yaml``, then resolve ``entry_point``
    relative to that.

    :param tar_bytes: raw tar archive bytes from the gateway.
    :param cid: the CID (for error messages).
    :return: tuple ``(entry_point_name, component_yaml_text, entry_point_text)``.
    :raises IpfsFetchError: if the archive doesn't contain the expected files.
    """
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r|") as tar:
            # Streaming-mode (`r|`) tar requires sequential read. Drop the
            # leading cid/ so keys are {wrapper_dir}/file.
            for member in tar:
                if not member.isfile():
                    continue
                path_parts = Path(member.name).parts
                if len(path_parts) < 3:
                    # Expect cid/wrapper/file at minimum
                    continue
                rel = "/".join(path_parts[1:])
                f = tar.extractfile(member)
                if f is None:
                    continue
                members[rel] = f.read()
    except tarfile.TarError as exc:
        raise IpfsFetchError(
            f"IPFS fetch failed: cid={cid} archive is not a valid tar: {exc!r}"
        ) from exc

    component_key = next(
        (k for k in members if k.endswith("/component.yaml") and k.count("/") == 1),
        None,
    )
    if component_key is None:
        raise IpfsFetchError(
            f"IPFS fetch failed: cid={cid} archive contains no component.yaml"
        )
    component_yaml_text = members[component_key].decode("utf-8")
    component_yaml = yaml.safe_load(component_yaml_text)
    if not isinstance(component_yaml, dict) or "entry_point" not in component_yaml:
        raise IpfsFetchError(
            f"IPFS fetch failed: cid={cid} component.yaml missing 'entry_point'"
        )
    entry_point_name: str = component_yaml["entry_point"]
    _validate_entry_point_name(entry_point_name, cid)

    wrapper_dir = component_key.split("/", 1)[0]
    entry_key = f"{wrapper_dir}/{entry_point_name}"
    if entry_key not in members:
        raise IpfsFetchError(
            f"IPFS fetch failed: cid={cid} entry_point '{entry_point_name}' "
            f"not in archive"
        )
    entry_text = members[entry_key].decode("utf-8")
    return entry_point_name, component_yaml_text, entry_text


def fetch_tool_package(cid: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, str]:
    """Fetch a tool package by CID into a dict suitable for ComponentPackageLoader.

    One GET: ``{IPFS_GATEWAY}/{cid}/`` with ``Accept: application/x-tar``
    returns a tarball of the published package. We extract
    ``component.yaml`` and the entry-point ``.py`` from the wrapper
    subdirectory and write them flat to ``cache_dir/{cid}/`` so future
    runs hit the disk cache and skip the network.

    :param cid: IPFS CID of the tool package.
    :param cache_dir: where to store cached tool sources.
    :return: dict ``{"component.yaml": yaml_text, entry_point: py_text}``.
    :raises IpfsFetchError: on gateway / network / archive failures.  # noqa: DAR402
    """
    component_cache = _cache_path(cache_dir, cid, "component.yaml")

    if component_cache.exists():
        component_yaml_text = component_cache.read_text(encoding="utf-8")
        try:
            component_yaml = yaml.safe_load(component_yaml_text)
        except yaml.YAMLError:
            # Corrupt on-disk yaml (e.g. disk full, SIGKILL mid-write).
            # Evict and fall through to re-fetch rather than crashing the
            # tournament loop with a YAMLError that escapes IpfsFetchError.
            log.warning("[ipfs] cid=%s cache=corrupt evicting", cid)
            component_cache.unlink(missing_ok=True)
            component_yaml = None
        if isinstance(component_yaml, dict) and "entry_point" in component_yaml:
            entry_point_name = component_yaml["entry_point"]
            _validate_entry_point_name(entry_point_name, cid)
            entry_cache = _cache_path(cache_dir, cid, entry_point_name)
            if entry_cache.exists():
                entry_text = entry_cache.read_text(encoding="utf-8")
                log.info("[ipfs] cid=%s cache=hit", cid)
                return {
                    "component.yaml": component_yaml_text,
                    entry_point_name: entry_text,
                }

    url = f"{IPFS_GATEWAY}/{cid}/"
    tar_bytes = _get_tar_with_retries(url, cid=cid)
    entry_point_name, component_yaml_text, entry_text = _extract_files_from_tar(
        tar_bytes, cid
    )

    component_cache.parent.mkdir(parents=True, exist_ok=True)
    component_cache.write_text(component_yaml_text, encoding="utf-8")
    _cache_path(cache_dir, cid, entry_point_name).write_text(
        entry_text, encoding="utf-8"
    )
    log.info("[ipfs] cid=%s cache=miss entry_point=%s", cid, entry_point_name)

    return {"component.yaml": component_yaml_text, entry_point_name: entry_text}


def load_tool_from_ipfs(
    cid: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Callable[..., Any]:
    """Fetch a tool package from IPFS, import it, return its run callable.

    The entry-point file on disk (written by ``fetch_tool_package``) is
    loaded via ``importlib.util.spec_from_file_location`` under a
    CID-derived module name. The module is registered in ``sys.modules``
    so introspection-based libraries (pydantic forward refs, dataclasses,
    typing.get_type_hints) can resolve names from the module's namespace.

    :param cid: IPFS CID of the tool package.
    :param cache_dir: on-disk cache location.
    :return: the tool's run callable (``component.yaml:callable``).
    :raises IpfsFetchError: on gateway / network failures, on
        ``component.yaml`` problems, or if the entry-point module can't
        be imported or doesn't expose the named callable.
    """
    package = fetch_tool_package(cid, cache_dir=cache_dir)
    try:
        (
            _component_yaml,
            _entry_point_source,
            callable_name,
        ) = ComponentPackageLoader.load(package)
    except Exception as exc:  # pylint: disable=broad-except
        raise IpfsFetchError(
            f"IPFS fetch failed: cid={cid} component package load failed: {exc!r}"
        ) from exc

    # The entry-point name is the only non-yaml key in the package dict;
    # `fetch_tool_package` wrote that file to `cache_dir/{cid}/{name}`.
    entry_point_name = next(k for k in package if k != "component.yaml")
    entry_path = _cache_path(cache_dir, cid, entry_point_name)

    # Unique module name per CID — lets a candidate CID and the
    # production CID for the same tool coexist in one process.
    module_name = f"mech_predict_tournament_tool_{cid}"

    cached_mod = sys.modules.get(module_name)
    if cached_mod is not None and hasattr(cached_mod, callable_name):
        return getattr(cached_mod, callable_name)

    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise IpfsFetchError(
            f"IPFS fetch failed: cid={cid} could not build import spec "
            f"for {entry_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pylint: disable=broad-except
        sys.modules.pop(module_name, None)
        raise IpfsFetchError(
            f"IPFS fetch failed: cid={cid} entry_point module load failed: {exc!r}"
        ) from exc

    if not hasattr(module, callable_name):
        sys.modules.pop(module_name, None)
        raise IpfsFetchError(
            f"IPFS fetch failed: cid={cid} callable '{callable_name}' "
            f"not found in entry_point module"
        )
    return getattr(module, callable_name)
