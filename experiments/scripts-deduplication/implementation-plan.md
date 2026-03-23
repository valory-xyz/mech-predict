# Implementation Plan: Script Deduplication

**Date:** 2026-03-23
**Companion doc:** [scripts-deduplication-report.md](./scripts-deduplication-report.md)

---

## Overview

This plan covers the end-to-end work to consolidate `bump.py`, `check_dependencies.py`, and `check_doc_ipfs_hashes.py` into the `open-autonomy` CLI and migrate all 7 repos.

**Dependency chain:**

```
Phase 1 ─── Add 3 CLI commands to open-autonomy ───┐
  Task 1.1: autonomy check-doc-hashes               │
  Task 1.2: autonomy check-dependencies              │  (can be parallelized)
  Task 1.3: autonomy bump-dependencies               │
                                                      ▼
Phase 2 ─── Release open-autonomy ──────────────────┐
  Task 2.1: Tests + CI validation                    │
  Task 2.2: Cut release with new commands            │
                                                      ▼
Phase 3 ─── Migrate downstream repos ──────────────────
  Task 3.1: open-autonomy (self-migration)
  Task 3.2: mech-predict                              │
  Task 3.3: mech-agents-fun                           │
  Task 3.4: mech-interact                             │  (can be parallelized)
  Task 3.5: mech                                      │
  Task 3.6: mech-client                               │
  Task 3.7: mech-server                               │
```

---

## Phase 1: Add CLI Commands to `open-autonomy`

> **Repo:** `open-autonomy`
> **Branch:** `feat/consolidate-scripts`
> All three tasks below are independent and can be developed in parallel.

---

### Task 1.1: `autonomy check-doc-hashes`

**Risk: Low**

**Starting point:** `open-autonomy/scripts/check_doc_ipfs_hashes.py` (496 lines — already the superset)

#### Step 1.1.1 — Create CLI module

Create `autonomy/cli/check_doc_hashes.py`:

- Move the script logic into this module
- Wrap with Click decorators:
  ```
  @click.command(name="check-doc-hashes")
  @click.option("--fix", is_flag=True, help="Fix mismatched hashes in-place")
  @click.option("--skip-hash", multiple=True, help="IPFS hashes to skip (repeatable)")
  @click.option("--doc-path", multiple=True, default=("docs",), help="Documentation directories to scan")
  ```
- Replace the hardcoded `HASH_SKIPS` list with the `--skip-hash` option
- Replace hardcoded `DOCS_DIR = Path("docs")` with the `--doc-path` option
- Keep `get_packages_from_repository()` and `PACKAGE_MAPPING_REGEX` (open-autonomy features) — they're no-ops for repos that don't reference cross-repo packages

#### Step 1.1.2 — Register in CLI

Edit `autonomy/cli/core.py`:

```python
# Add import
from autonomy.cli.check_doc_hashes import check_doc_hashes

# Add registration
cli.add_command(check_doc_hashes)
```

#### Step 1.1.3 — Add tests

Create `tests/test_autonomy/test_cli/test_check_doc_hashes.py`:

- Test `--fix` mode updates hashes
- Test check mode (no `--fix`) exits non-zero on mismatch
- Test `--skip-hash` skips specified hashes
- Test with empty docs directory (no-op, exit 0)
- Follow existing pattern: extend `BaseCliTest` from `tests/test_autonomy/test_cli/base.py`

#### Files changed

| File | Action |
|---|---|
| `autonomy/cli/check_doc_hashes.py` | **Create** — main command implementation |
| `autonomy/cli/core.py` | **Edit** — add import + `cli.add_command()` |
| `tests/test_autonomy/test_cli/test_check_doc_hashes.py` | **Create** — tests |

---

### Task 1.2: `autonomy check-dependencies`

**Risk: Medium**

**Starting point:** `mech-interact/scripts/check_dependencies.py` (652 lines — most complete version)

This is the hardest task because four distinct variants have diverged. The mech-interact version already has `Pipfile`, `ToxFile`, and `PyProjectToml` classes with proper Click CLI and `--check` flag. It needs merging with open-autonomy's Pipfile extras/git handling.

#### Step 1.2.1 — Create CLI module

Create `autonomy/cli/check_dependencies.py`:

- Start from mech-interact's version (already has Click CLI structure)
- Merge open-autonomy's `update_tox_ini()` logic for handling:
  - `*` wildcard versions → empty string
  - `extras` dicts → `[extra1,extra2]version`
  - `git+` deps → `git+URL@ref#egg=name`
- Add `--exclude` option (repeatable) to replace all hardcoded hacks:
  ```
  @click.command(name="check-dependencies")
  @click.option("--check", is_flag=True, help="Validate only, do not update files")
  @click.option("--exclude", multiple=True, help="Package names to exclude (repeatable)")
  @click.option("--packages", "packages_dir", type=click.Path(exists=True), help="Packages directory path")
  @click.option("--tox", "tox_path", type=click.Path(exists=True), help="tox.ini path")
  @click.option("--pipfile", "pipfile_path", type=click.Path(exists=True), help="Pipfile path")
  @click.option("--pyproject", "pyproject_path", type=click.Path(exists=True), help="pyproject.toml path")
  ```
- Auto-detect config file: if `Pipfile` exists use it, otherwise use `pyproject.toml`
- The `--check` flag should exit non-zero on mismatch (for CI), update mode should rewrite files

#### Step 1.2.2 — Handle variant differences

Specific logic that must be unified:

| Variant feature | How to handle |
|---|---|
| mech repos hardcode `requests` version | Use `--exclude requests` in their tox.ini instead |
| open-autonomy pops `open-aea-ledger-solana`, `solders` | Use `--exclude open-aea-ledger-solana --exclude solders` in their tox.ini |
| mech-server uses content comparison (no git) | The Click CLI version from mech-interact already doesn't use git — this is handled |
| open-autonomy handles `git+` deps in tox.ini | Merge this into the `ToxFile` class |
| open-autonomy handles `extras` dicts from Pipfile | Merge this into the `Pipfile` class |
| mech-server handles `^` and `v` version prefixes | Already handled by `PyProjectToml` class in mech-interact |

#### Step 1.2.3 — Register in CLI

Edit `autonomy/cli/core.py`:

```python
from autonomy.cli.check_dependencies import check_dependencies
cli.add_command(check_dependencies)
```

#### Step 1.2.4 — Add tests

Create `tests/test_autonomy/test_cli/test_check_dependencies.py`:

- Test check mode with matching deps (exit 0)
- Test check mode with mismatched deps (exit 1)
- Test update mode rewrites pyproject.toml correctly
- Test update mode rewrites tox.ini correctly
- Test `--exclude` skips specified packages
- Test auto-detection of Pipfile vs pyproject.toml
- Test with Pipfile input (extras, git deps, `*` wildcards)
- Test with pyproject.toml input (caret versions, `v` prefix)

#### Step 1.2.5 — Validate against all 7 repos

Before merging, run the consolidated command against each repo's actual files:

```bash
# For each repo, in a temporary checkout:
autonomy check-dependencies --check --exclude <repo-specific-excludes>
```

This catches format differences that tests might miss.

#### Files changed

| File | Action |
|---|---|
| `autonomy/cli/check_dependencies.py` | **Create** — main command implementation (~500-600 lines after cleanup) |
| `autonomy/cli/core.py` | **Edit** — add import + `cli.add_command()` |
| `tests/test_autonomy/test_cli/test_check_dependencies.py` | **Create** — tests |

---

### Task 1.3: `autonomy bump-dependencies`

**Risk: Low**

**Starting point:** `mech-predict/scripts/bump.py` (318 lines — most up-to-date, has `TIMEOUT = 30.0`)

#### Step 1.3.1 — Create CLI module

Create `autonomy/cli/bump_dependencies.py`:

- Move script logic into this module
- Wrap with Click decorators:
  ```
  @click.command(name="bump-dependencies")
  @click.option("--sync/--no-sync", default=False, help="Run autonomy packages sync after bumping")
  @click.option("--no-cache", is_flag=True, help="Bypass version cache")
  @click.option("-d", "--dep", multiple=True, help="Extra dependencies to bump")
  @click.option("-s", "--source", type=click.Choice(["pypi", "github"]), default="github", help="Version source")
  ```
- Keep the `TIMEOUT = 30.0` constant (fixes the mech-client/mech-server bug)
- The existing `sys.argv` parsing maps cleanly to Click options

#### Step 1.3.2 — Register in CLI

Edit `autonomy/cli/core.py`:

```python
from autonomy.cli.bump_dependencies import bump_dependencies
cli.add_command(bump_dependencies)
```

#### Step 1.3.3 — Add tests

Create `tests/test_autonomy/test_cli/test_bump_dependencies.py`:

- Test with mocked GitHub API responses
- Test `--no-cache` bypasses cached versions
- Test `--sync` triggers package sync
- Test timeout is applied to all HTTP requests

#### Files changed

| File | Action |
|---|---|
| `autonomy/cli/bump_dependencies.py` | **Create** — main command implementation |
| `autonomy/cli/core.py` | **Edit** — add import + `cli.add_command()` |
| `tests/test_autonomy/test_cli/test_bump_dependencies.py` | **Create** — tests |

---

## Phase 2: Release `open-autonomy`

> **Prerequisite:** All Phase 1 tasks merged.

### Task 2.1: CI validation

- Ensure all existing open-autonomy CI passes with the new commands
- Run the 3 new commands against open-autonomy's own repo as a smoke test:
  ```bash
  autonomy check-doc-hashes
  autonomy check-dependencies --check --exclude open-aea-ledger-solana --exclude solders
  ```
- Verify the commands appear in `autonomy --help`

### Task 2.2: Cut release

- Bump version in open-autonomy
- Tag and release
- Downstream repos will pin to this version (or higher)

---

## Phase 3: Migrate Downstream Repos

> **Prerequisite:** Phase 2 release is published.
> All repo migrations are independent and can run in parallel.

Each migration follows the same pattern. Repo-specific details are noted below.

### Common migration steps (per repo)

#### Step A — Bump open-autonomy version

Update the open-autonomy dependency to the Phase 2 release version in `pyproject.toml` (or `Pipfile` for open-autonomy itself) and `tox.ini`.

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
    autonomy check-dependencies --check <repo-specific-excludes>
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
    autonomy check-doc-hashes
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
commands = autonomy check-doc-hashes --fix
```

#### Step C — Update tox.ini: remove `scripts` from linting targets

For repos where `scripts/` will be **deleted entirely** (mech, mech-interact, mech-server), remove `scripts` from all of these tox environments:

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
- **Extra tox envs using scripts:** `[testenv:lock-packages]` calls `scripts/check_doc_ipfs_hashes.py --fix` and `scripts/generate_package_list.py` — update the hash line to `autonomy check-doc-hashes --fix`, keep the `generate_package_list.py` reference
- **Extra tox envs using scripts:** `[testenv:check-doc-links-hashes]` calls 3 scripts — update only the `check_doc_ipfs_hashes.py` line
- **scripts/ directory:** Keep — still has `check_copyright.py`, `check_doc_links.py`, `check_ipfs_hashes_pushed.py`, `freeze_dependencies.py`, `generate_api_documentation.py`, `generate_contract_list.py`, `generate_package_list.py`, `whitelist.py`
- **Delete:** `scripts/bump.py`, `scripts/check_dependencies.py`, `scripts/check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in linting targets (directory still exists with other scripts)

---

### Task 3.2: `mech-predict`

**Repo-specific details:**

- **Excludes for check-dependencies:** `--exclude requests` (currently hardcodes `requests==2.28.2`)
- **scripts/ directory:** Keep — still has `generate_metadata.py`, `publish_metadata.py`, `test_tool.py`, `test_tools.py`
- **Delete:** `scripts/bump.py`, `scripts/check_dependencies.py`, `scripts/check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in linting targets (directory still exists with other scripts)

---

### Task 3.3: `mech-agents-fun`

**Repo-specific details:**

- **Excludes for check-dependencies:** `--exclude requests` (currently hardcodes `requests>=2.28.1,<2.33.0`)
- **scripts/ directory:** Keep — still has 5 test scripts (`test_image_gen.py`, etc.)
- **Delete:** `scripts/bump.py`, `scripts/check_dependencies.py`, `scripts/check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in linting targets (directory still exists with other scripts)

---

### Task 3.4: `mech-interact`

**Repo-specific details:**

- **Excludes for check-dependencies:** None expected (mech-interact's version had no hardcoded hacks — it was the most clean version)
- **Extra cleanup:** Delete `scripts/compare_hashes.py` (one-off dev script with hardcoded personal path)
- **scripts/ directory:** **Delete entirely** — no repo-specific scripts remain
- **Delete:** `scripts/bump.py`, `scripts/check_dependencies.py`, `scripts/check_doc_ipfs_hashes.py`, `scripts/compare_hashes.py`, `scripts/__init__.py`
- **Linting:** Remove `scripts` from all linting targets and `application-import-names`

---

### Task 3.5: `mech`

**Repo-specific details:**

- **Excludes for check-dependencies:** `--exclude requests` (currently hardcodes `requests==2.32.5`)
- **scripts/ directory:** **Delete entirely** — no repo-specific scripts remain
- **Delete:** `scripts/bump.py`, `scripts/check_dependencies.py`, `scripts/check_doc_ipfs_hashes.py`, `scripts/__init__.py`
- **Linting:** Remove `scripts` from all linting targets and `application-import-names`

---

### Task 3.6: `mech-client`

**Repo-specific details:**

- **No `check_dependencies.py` or `check_doc_ipfs_hashes.py`** — only `bump.py` needs migration
- **scripts/ directory:** Keep — still has `whitelist.py`, `benchmark.sh`
- **Delete:** `scripts/bump.py`
- **Linting:** Keep `scripts` in linting targets (directory still exists)
- **Note:** This repo only has `[testenv:bandit]` and `[testenv:vulture]` referencing scripts — no check-dependencies or check-doc-hashes tox envs to update

---

### Task 3.7: `mech-server`

**Repo-specific details:**

- **Uses `utils/` not `scripts/`** — all references in tox.ini say `utils` instead of `scripts`
- **Excludes for check-dependencies:** `--exclude requests` (currently hardcodes `requests==2.32.5`)
- **utils/ directory:** **Delete entirely** — no repo-specific scripts remain
- **Delete:** `utils/bump.py`, `utils/check_dependencies.py`, `utils/check_doc_ipfs_hashes.py`, `utils/__init__.py`
- **Linting:** Remove `utils` from all linting targets; change `application-import-names = packages,tests,utils` to `application-import-names = packages,tests`

---

## Summary: Files Changed Across All Repos

### `open-autonomy` (Phase 1 + Phase 2 + Task 3.1)

| File | Action |
|---|---|
| `autonomy/cli/check_doc_hashes.py` | Create |
| `autonomy/cli/check_dependencies.py` | Create |
| `autonomy/cli/bump_dependencies.py` | Create |
| `autonomy/cli/core.py` | Edit (3 imports + 3 `add_command`) |
| `tests/test_autonomy/test_cli/test_check_doc_hashes.py` | Create |
| `tests/test_autonomy/test_cli/test_check_dependencies.py` | Create |
| `tests/test_autonomy/test_cli/test_bump_dependencies.py` | Create |
| `scripts/bump.py` | Delete |
| `scripts/check_dependencies.py` | Delete |
| `scripts/check_doc_ipfs_hashes.py` | Delete |
| `tox.ini` | Edit (update script references → CLI commands) |

### Each downstream repo (Tasks 3.2–3.7)

| File | Action |
|---|---|
| `scripts/bump.py` or `utils/bump.py` | Delete |
| `scripts/check_dependencies.py` or `utils/check_dependencies.py` | Delete (where exists) |
| `scripts/check_doc_ipfs_hashes.py` or `utils/check_doc_ipfs_hashes.py` | Delete (where exists) |
| `scripts/__init__.py` or `utils/__init__.py` | Delete (if directory is being removed) |
| `tox.ini` | Edit |
| `pyproject.toml` or `Pipfile` | Edit (bump open-autonomy version) |

---

## Execution Order Checklist

```
[ ] Phase 1 — open-autonomy CLI commands (PR: feat/consolidate-scripts)
    [ ] 1.1  Create autonomy/cli/check_doc_hashes.py + tests
    [ ] 1.2  Create autonomy/cli/check_dependencies.py + tests
    [ ] 1.3  Create autonomy/cli/bump_dependencies.py + tests
    [ ] 1.x  Register all 3 in autonomy/cli/core.py
    [ ] 1.x  Validate: run new commands against open-autonomy's own repo

[ ] Phase 2 — Release
    [ ] 2.1  CI green on open-autonomy
    [ ] 2.2  Tag + release new open-autonomy version

[ ] Phase 3 — Downstream migration (all independent, parallelizable)
    [ ] 3.1  open-autonomy: delete old scripts, update tox.ini
    [ ] 3.2  mech-predict: delete scripts, update tox.ini, bump OA version
    [ ] 3.3  mech-agents-fun: delete scripts, update tox.ini, bump OA version
    [ ] 3.4  mech-interact: delete scripts/, update tox.ini, bump OA version
    [ ] 3.5  mech: delete scripts/, update tox.ini, bump OA version
    [ ] 3.6  mech-client: delete bump.py, bump OA version
    [ ] 3.7  mech-server: delete utils/, update tox.ini, bump OA version
```
