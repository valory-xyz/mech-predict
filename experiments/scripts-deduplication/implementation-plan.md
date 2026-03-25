# Implementation Plan: Script Deduplication via `aea-helpers` Plugin

**Date:** 2026-03-23
**Companion doc:** [scripts-deduplication-report.md](./scripts-deduplication-report.md)

---

## Overview

This plan covers the end-to-end work to consolidate `bump.py`, `check_dependencies.py`, and `check_doc_ipfs_hashes.py` into a new `aea-helpers` plugin under `open-autonomy/plugins/` and migrate all 7 repos.

**Dependency chain:**

```
Phase 1 ─── Create aea-helpers plugin in open-autonomy ───┐
  Task 1.0: Scaffold plugin package                        │
  Task 1.1: aea-helpers check-doc-hashes                   │
  Task 1.2: aea-helpers check-dependencies                 │  (1.1–1.3 parallelizable)
  Task 1.3: aea-helpers bump-dependencies                  │
                                                            ▼
Phase 2 ─── Release plugin ────────────────────────────────┐
  Task 2.1: Tests + CI validation                          │
  Task 2.2: Publish aea-helpers to PyPI                    │
                                                            ▼
Phase 3 ─── Migrate downstream repos ──────────────────────
  Task 3.1: open-autonomy (self-migration)
  Task 3.2: mech-predict                                    │
  Task 3.3: mech-agents-fun                                 │
  Task 3.4: mech-interact                                   │  (all parallelizable)
  Task 3.5: mech                                            │
  Task 3.6: mech-client                                     │
  Task 3.7: mech-server                                     │
```

---

## Phase 1: Create `aea-helpers` Plugin

> **Repo:** `open-autonomy`
> **Branch:** `feat/aea-helpers-plugin`
> **Location:** `plugins/aea-helpers/`

---

### Task 1.0: Scaffold the plugin package

Create the plugin structure following the `aea-test-autonomy` pattern:

```
plugins/aea-helpers/
├── setup.py
├── pyproject.toml
├── LICENSE
├── README.md
├── aea_helpers/
│   ├── __init__.py
│   ├── bump_dependencies.py
│   ├── check_dependencies.py
│   └── check_doc_hashes.py
└── tests/
    ├── __init__.py
    ├── test_bump_dependencies.py
    ├── test_check_dependencies.py
    └── test_check_doc_hashes.py
```

#### `setup.py` key fields

```python
setup(
    name="aea-helpers",
    version="0.1.0",
    author="Valory AG",
    license="Apache-2.0",
    packages=find_packages(include=["aea_helpers*"]),
    entry_points={
        "console_scripts": [
            "aea-helpers=aea_helpers.cli:cli",
        ],
    },
    install_requires=[
        "open-autonomy>=0.21.0,<1.0.0",
        "click>=8.1.0,<9",
        "requests>=2.28.0,<3",
        "toml>=0.10,<1",
    ],
    python_requires=">=3.10",
)
```

Key points:
- **Depends on `open-autonomy`** — required because `bump.py` needs `autonomy.cli.helpers.ipfs_hash.load_configuration` and `check_doc_ipfs_hashes.py` needs autonomy's overridden `get_package_manager`, both for Service package type support
- No circular dependency: `aea-helpers → open-autonomy → open-aea` (one-way)
- Exposes `aea-helpers` as a console script entry point
- Follows the same license and author conventions as `aea-test-autonomy`

#### `aea_helpers/cli.py` — CLI entry point

```python
import click

from aea_helpers.bump_dependencies import bump_dependencies
from aea_helpers.check_dependencies import check_dependencies
from aea_helpers.check_doc_hashes import check_doc_hashes


@click.group()
@click.version_option()
def cli():
    """AEA helper utilities for CI and dependency management."""


cli.add_command(bump_dependencies)
cli.add_command(check_dependencies)
cli.add_command(check_doc_hashes)
```

#### Integration with open-autonomy repo

Add to `tox.ini` in open-autonomy (following `aea-test-autonomy` pattern):
```ini
python -m pip install --no-deps {toxinidir}{/}plugins{/}aea-helpers
```

Add to `scripts/bump.py` version tracking:
```python
"aea-helpers": {
    "description": "AEA helper utilities for CI",
    "file": "plugins/aea-helpers/setup.py",
}
```

#### Files created

| File | Action |
|---|---|
| `plugins/aea-helpers/setup.py` | **Create** |
| `plugins/aea-helpers/pyproject.toml` | **Create** (build-system config) |
| `plugins/aea-helpers/LICENSE` | **Create** (Apache-2.0) |
| `plugins/aea-helpers/aea_helpers/__init__.py` | **Create** |
| `plugins/aea-helpers/aea_helpers/cli.py` | **Create** — CLI entry point |
| `plugins/aea-helpers/tests/__init__.py` | **Create** |

---

### Task 1.1: `aea-helpers check-doc-hashes`

**Risk: Low** — open-autonomy's version is already the superset; downstream repos use fewer features.

**Starting point:** `open-autonomy/scripts/check_doc_ipfs_hashes.py` (496 lines)

#### Step 1.1.1 — Create command module

Create `plugins/aea-helpers/aea_helpers/check_doc_hashes.py`:

- Move the script logic into this module
- Wrap with Click decorators:
  ```python
  @click.command(name="check-doc-hashes")
  @click.option("--fix", is_flag=True, help="Fix mismatched hashes in-place")
  @click.option("--skip-hash", multiple=True, help="IPFS hashes to skip (repeatable)")
  @click.option("--doc-path", multiple=True, default=("docs",), help="Doc directories to scan")
  ```
- Replace the hardcoded `HASH_SKIPS` list with the `--skip-hash` option
- Replace hardcoded `DOCS_DIR = Path("docs")` with the `--doc-path` option
- Keep `get_packages_from_repository()` and `PACKAGE_MAPPING_REGEX` — they're no-ops for repos that don't reference cross-repo packages

#### Step 1.1.2 — Add tests

Create `plugins/aea-helpers/tests/test_check_doc_hashes.py`:

- Test `--fix` mode updates hashes
- Test check mode (no `--fix`) exits non-zero on mismatch
- Test `--skip-hash` skips specified hashes
- Test with empty docs directory (no-op, exit 0)

#### Files changed

| File | Action |
|---|---|
| `plugins/aea-helpers/aea_helpers/check_doc_hashes.py` | **Create** |
| `plugins/aea-helpers/tests/test_check_doc_hashes.py` | **Create** |

---

### Task 1.2: `aea-helpers check-dependencies`

**Risk: Medium** — four distinct variants have diverged. The mech-interact version is the best starting point.

**Starting point:** `mech-interact/scripts/check_dependencies.py` (652 lines)

#### Step 1.2.1 — Create command module

Create `plugins/aea-helpers/aea_helpers/check_dependencies.py`:

- Start from mech-interact's version (already has Click CLI structure with `Pipfile`, `ToxFile`, `PyProjectToml` classes)
- Merge open-autonomy's `update_tox_ini()` logic for handling:
  - `*` wildcard versions → empty string
  - `extras` dicts → `[extra1,extra2]version`
  - `git+` deps → `git+URL@ref#egg=name`
- Add `--exclude` option (repeatable) to replace all hardcoded hacks:
  ```python
  @click.command(name="check-dependencies")
  @click.option("--check", is_flag=True, help="Validate only, do not update files")
  @click.option("--exclude", multiple=True, help="Package names to exclude (repeatable)")
  @click.option("--packages", "packages_dir", type=click.Path(exists=True), help="Packages directory")
  @click.option("--tox", "tox_path", type=click.Path(exists=True), help="tox.ini path")
  @click.option("--pipfile", "pipfile_path", type=click.Path(exists=True), help="Pipfile path")
  @click.option("--pyproject", "pyproject_path", type=click.Path(exists=True), help="pyproject.toml path")
  ```
- Auto-detect config file: if `Pipfile` exists use it, otherwise use `pyproject.toml`
- `--check` exits non-zero on mismatch (for CI); update mode rewrites files

#### Step 1.2.2 — Handle variant differences

| Variant feature | How to handle |
|---|---|
| mech repos hardcode `requests` version | Use `--exclude requests` in their tox.ini instead |
| open-autonomy pops `open-aea-ledger-solana`, `solders` | Use `--exclude open-aea-ledger-solana --exclude solders` in their tox.ini |
| mech-server uses content comparison (no git) | The Click CLI version from mech-interact already doesn't use git — handled |
| open-autonomy handles `git+` deps in tox.ini | Merge into the `ToxFile` class |
| open-autonomy handles `extras` dicts from Pipfile | Merge into the `Pipfile` class |
| mech-server handles `^` and `v` version prefixes | Already handled by `PyProjectToml` class in mech-interact |

#### Step 1.2.3 — Add tests

Create `plugins/aea-helpers/tests/test_check_dependencies.py`:

- Test check mode with matching deps (exit 0)
- Test check mode with mismatched deps (exit 1)
- Test update mode rewrites pyproject.toml correctly
- Test update mode rewrites tox.ini correctly
- Test `--exclude` skips specified packages
- Test auto-detection of Pipfile vs pyproject.toml
- Test with Pipfile input (extras, git deps, `*` wildcards)
- Test with pyproject.toml input (caret versions, `v` prefix)

#### Step 1.2.4 — Validate against all 7 repos

Before merging, run the consolidated command against each repo's actual files:

```bash
# For each repo, in a temporary checkout:
aea-helpers check-dependencies --check --exclude <repo-specific-excludes>
```

#### Files changed

| File | Action |
|---|---|
| `plugins/aea-helpers/aea_helpers/check_dependencies.py` | **Create** (~500-600 lines after cleanup) |
| `plugins/aea-helpers/tests/test_check_dependencies.py` | **Create** |

---

### Task 1.3: `aea-helpers bump-dependencies`

**Risk: Low** — all copies are nearly identical; the timeout bug gets fixed automatically.

**Starting point:** `mech-predict/scripts/bump.py` (318 lines)

#### Step 1.3.1 — Create command module

Create `plugins/aea-helpers/aea_helpers/bump_dependencies.py`:

- Move script logic into this module
- **Keep the `autonomy` import** — `load_configuration` from `autonomy.cli.helpers.ipfs_hash` is required because `PackageManagerV1.update_package_hashes()` iterates all packages including Service types, and only autonomy's loader supports `PackageType.SERVICE`. Dropping it causes a `KeyError`.
- Wrap with Click decorators:
  ```python
  @click.command(name="bump-dependencies")
  @click.option("--sync/--no-sync", default=False, help="Run packages sync after bumping")
  @click.option("--no-cache", is_flag=True, help="Bypass version cache")
  @click.option("-d", "--dep", multiple=True, help="Extra dependencies to bump")
  @click.option("-s", "--source", type=click.Choice(["pypi", "github"]), default="github", help="Version source")
  ```
- Keep the `TIMEOUT = 30.0` constant (fixes the mech-client/mech-server bug)

#### Step 1.3.2 — Add tests

Create `plugins/aea-helpers/tests/test_bump_dependencies.py`:

- Test with mocked GitHub API responses
- Test `--no-cache` bypasses cached versions
- Test `--sync` triggers package sync
- Test timeout is applied to all HTTP requests

#### Files changed

| File | Action |
|---|---|
| `plugins/aea-helpers/aea_helpers/bump_dependencies.py` | **Create** |
| `plugins/aea-helpers/tests/test_bump_dependencies.py` | **Create** |

---

## Phase 2: Release Plugin

> **Prerequisite:** All Phase 1 tasks merged into open-autonomy.

### Task 2.1: CI validation

- Ensure open-autonomy CI passes with the new plugin
- Run the 3 commands against open-autonomy's own repo as a smoke test:
  ```bash
  aea-helpers check-doc-hashes
  aea-helpers check-dependencies --check --exclude open-aea-ledger-solana --exclude solders
  ```
- Verify `aea-helpers --help` shows all 3 commands

### Task 2.2: Publish to PyPI

- Publish `aea-helpers` package to PyPI
- Downstream repos can now `pip install aea-helpers`

---

## Phase 3: Migrate Downstream Repos

> **Prerequisite:** `aea-helpers` is published on PyPI.
> All repo migrations are independent and can run in parallel.

### Common migration steps (per repo)

#### Step A — Add `aea-helpers` dependency

Must be added in **two places** (the `check_dependencies` script validates consistency between them):

`pyproject.toml`:
```toml
[tool.poetry.dependencies]
aea-helpers = ">=0.1.0"
```

Or `Pipfile` (for open-autonomy):
```
[dev-packages]
aea-helpers = ">=0.1.0"
```

**AND** `tox.ini` `[deps-packages]` section:
```ini
[deps-packages]
deps =
    ...
    aea-helpers>=0.1.0
```

#### Step B — Update tox.ini: script invocations → CLI commands

Replace:
```ini
[testenv:check-dependencies]
allowlist_externals = {toxinidir}/scripts/check_dependencies.py
commands =
    autonomy packages sync
    {toxinidir}/scripts/check_dependencies.py
```

With:
```ini
[testenv:check-dependencies]
commands =
    autonomy packages sync
    aea-helpers check-dependencies --check <repo-specific-excludes>
```

Replace:
```ini
[testenv:check-doc-hashes]
allowlist_externals = {toxinidir}/scripts/check_doc_ipfs_hashes.py
commands =
    aea init --reset --author ci --remote --ipfs --ipfs-node "/dns/registry.autonolas.tech/tcp/443/https"
    aea packages sync
    {toxinidir}/scripts/check_doc_ipfs_hashes.py
```

With:
```ini
[testenv:check-doc-hashes]
commands =
    aea init --reset --author ci --remote --ipfs --ipfs-node "/dns/registry.autonolas.tech/tcp/443/https"
    aea packages sync
    aea-helpers check-doc-hashes
```

Replace:
```ini
[testenv:fix-doc-hashes]
allowlist_externals = {toxinidir}/scripts/check_doc_ipfs_hashes.py
commands = {toxinidir}/scripts/check_doc_ipfs_hashes.py --fix
```

With:
```ini
[testenv:fix-doc-hashes]
commands = aea-helpers check-doc-hashes --fix
```

#### Step C — Remove `scripts` from linting targets

For repos where `scripts/` will be **deleted entirely** (mech, mech-interact, mech-server), remove `scripts`/`utils` from all of these tox environments:

| tox environment | Change |
|---|---|
| `[testenv:bandit]` | Remove `bandit -s B101 -r scripts` line |
| `[testenv:black]` | Remove `scripts` from `black ... scripts` |
| `[testenv:black-check]` | Remove `scripts` from `black --check ... scripts` |
| `[testenv:isort]` | Remove `isort scripts/` line |
| `[testenv:isort-check]` | Remove `scripts` from `isort --check-only ... scripts` |
| `[testenv:flake8]` | Remove `flake8 scripts` line |
| `[testenv:mypy]` | Remove `mypy scripts ...` line |
| `[testenv:pylint]` | Remove `scripts` from `pylint ... scripts` |
| `[testenv:darglint]` | Remove `scripts` from `darglint scripts ...` |
| `[flake8]` | Remove `scripts` from `application-import-names` |

For repos where `scripts/` still has repo-specific scripts (mech-predict, mech-agents-fun, mech-client), keep `scripts` in the linting targets.

#### Step D — Delete duplicated scripts

Remove these files from `scripts/` (or `utils/` for mech-server):
- `bump.py`
- `check_dependencies.py`
- `check_doc_ipfs_hashes.py`
- `__init__.py` (only if no other scripts remain)

#### Step E — Verify CI passes

Run the full tox suite to confirm nothing breaks.

---

### Task 3.1: `open-autonomy` (self-migration)

**Repo-specific details:**

- **Excludes for check-dependencies:** `--exclude open-aea-ledger-solana --exclude solders`
- **Extra tox envs using scripts:** `[testenv:lock-packages]` calls `scripts/check_doc_ipfs_hashes.py --fix` and `scripts/generate_package_list.py` — update the hash line to `aea-helpers check-doc-hashes --fix`, keep the `generate_package_list.py` reference
- **Extra tox envs using scripts:** `[testenv:check-doc-links-hashes]` calls 3 scripts — update only the `check_doc_ipfs_hashes.py` line
- **scripts/ directory:** Keep — still has `check_copyright.py`, `check_doc_links.py`, `check_ipfs_hashes_pushed.py`, `freeze_dependencies.py`, `generate_api_documentation.py`, `generate_contract_list.py`, `generate_package_list.py`, `whitelist.py`
- **Delete:** `scripts/bump.py`, `scripts/check_dependencies.py`, `scripts/check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in linting targets (directory still exists with other scripts)

---

### Task 3.2: `mech-predict`

- **Excludes for check-dependencies:** `--exclude requests` (currently hardcodes `requests==2.28.2`)
- **scripts/ directory:** Keep — still has `generate_metadata.py`, `publish_metadata.py`, `test_tool.py`, `test_tools.py`
- **Delete:** `scripts/bump.py`, `scripts/check_dependencies.py`, `scripts/check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in linting targets

---

### Task 3.3: `mech-agents-fun`

- **Excludes for check-dependencies:** `--exclude requests` (currently hardcodes `requests>=2.28.1,<2.33.0`)
- **scripts/ directory:** Keep — still has 5 test scripts
- **Delete:** `scripts/bump.py`, `scripts/check_dependencies.py`, `scripts/check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in linting targets

---

### Task 3.4: `mech-interact`

- **Excludes for check-dependencies:** None expected (mech-interact's version had no hardcoded hacks)
- **Extra cleanup:** Delete `scripts/compare_hashes.py` (one-off dev script with hardcoded personal path)
- **scripts/ directory:** **Delete entirely**
- **Delete:** `scripts/bump.py`, `scripts/check_dependencies.py`, `scripts/check_doc_ipfs_hashes.py`, `scripts/compare_hashes.py`, `scripts/__init__.py`
- **Linting:** Remove `scripts` from all linting targets and `application-import-names`

---

### Task 3.5: `mech`

- **Excludes for check-dependencies:** `--exclude requests` (currently hardcodes `requests==2.32.5`)
- **scripts/ directory:** **Delete entirely**
- **Delete:** `scripts/bump.py`, `scripts/check_dependencies.py`, `scripts/check_doc_ipfs_hashes.py`, `scripts/__init__.py`
- **Linting:** Remove `scripts` from all linting targets and `application-import-names`

---

### Task 3.6: `mech-client`

- **No `check_dependencies.py` or `check_doc_ipfs_hashes.py`** — only `bump.py` needs migration
- **scripts/ directory:** Keep — still has `whitelist.py`, `benchmark.sh`
- **Delete:** `scripts/bump.py`
- **Linting:** Keep `scripts` in linting targets
- **Note:** This repo only has `[testenv:bandit]` and `[testenv:vulture]` referencing scripts — no check-dependencies or check-doc-hashes tox envs to update

---

### Task 3.7: `mech-server`

- **Uses `utils/` not `scripts/`** — all references in tox.ini say `utils` instead of `scripts`
- **Excludes for check-dependencies:** `--exclude requests` (currently hardcodes `requests==2.32.5`)
- **utils/ directory:** **Delete entirely**
- **Delete:** `utils/bump.py`, `utils/check_dependencies.py`, `utils/check_doc_ipfs_hashes.py`, `utils/__init__.py`
- **Linting:** Remove `utils` from all linting targets; change `application-import-names = packages,tests,utils` to `application-import-names = packages,tests`

---

## Summary: Files Changed Across All Repos

### `open-autonomy` (Phase 1 + Task 3.1)

| File | Action |
|---|---|
| `plugins/aea-helpers/setup.py` | Create |
| `plugins/aea-helpers/pyproject.toml` | Create |
| `plugins/aea-helpers/LICENSE` | Create |
| `plugins/aea-helpers/aea_helpers/__init__.py` | Create |
| `plugins/aea-helpers/aea_helpers/cli.py` | Create |
| `plugins/aea-helpers/aea_helpers/bump_dependencies.py` | Create |
| `plugins/aea-helpers/aea_helpers/check_dependencies.py` | Create |
| `plugins/aea-helpers/aea_helpers/check_doc_hashes.py` | Create |
| `plugins/aea-helpers/tests/test_bump_dependencies.py` | Create |
| `plugins/aea-helpers/tests/test_check_dependencies.py` | Create |
| `plugins/aea-helpers/tests/test_check_doc_hashes.py` | Create |
| `scripts/bump.py` | Delete |
| `scripts/check_dependencies.py` | Delete |
| `scripts/check_doc_ipfs_hashes.py` | Delete |
| `tox.ini` | Edit (update script references → plugin commands, add plugin install) |

### Each downstream repo (Tasks 3.2–3.7)

| File | Action |
|---|---|
| `scripts/bump.py` or `utils/bump.py` | Delete |
| `scripts/check_dependencies.py` or `utils/check_dependencies.py` | Delete (where exists) |
| `scripts/check_doc_ipfs_hashes.py` or `utils/check_doc_ipfs_hashes.py` | Delete (where exists) |
| `scripts/__init__.py` or `utils/__init__.py` | Delete (if directory is being removed) |
| `tox.ini` | Edit (script refs → `aea-helpers` commands, add plugin dep) |
| `pyproject.toml` or `Pipfile` | Edit (add `aea-helpers` dependency) |

---

## Execution Order Checklist

```
[ ] Phase 1 — Create aea-helpers plugin (PR: feat/aea-helpers-plugin)
    [ ] 1.0  Scaffold plugin package structure
    [ ] 1.1  Implement aea-helpers check-doc-hashes + tests
    [ ] 1.2  Implement aea-helpers check-dependencies + tests
    [ ] 1.3  Implement aea-helpers bump-dependencies + tests (keeps autonomy import for Service support)
    [ ] 1.x  Validate: run plugin commands against open-autonomy's own repo

[ ] Phase 2 — Release
    [ ] 2.1  CI green on open-autonomy
    [ ] 2.2  Publish aea-helpers to PyPI

[ ] Phase 3 — Downstream migration (all independent, parallelizable)
    [ ] 3.1  open-autonomy: delete old scripts, update tox.ini, add plugin dep
    [ ] 3.2  mech-predict: delete scripts, update tox.ini, add plugin dep
    [ ] 3.3  mech-agents-fun: delete scripts, update tox.ini, add plugin dep
    [ ] 3.4  mech-interact: delete scripts/, update tox.ini, add plugin dep
    [ ] 3.5  mech: delete scripts/, update tox.ini, add plugin dep
    [ ] 3.6  mech-client: delete bump.py, add plugin dep
    [ ] 3.7  mech-server: delete utils/, update tox.ini, add plugin dep
```
