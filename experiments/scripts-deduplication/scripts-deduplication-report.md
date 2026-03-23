# Script Deduplication Across Mech Repositories

**Date:** 2026-03-23
**Status:** Proposal for Review
**Author:** Engineering
**Repos in scope:** mech-predict, mech-agents-fun, mech-interact, mech, mech-client, mech-server, open-autonomy

---

## Executive Summary

Three Python scripts — `bump.py`, `check_dependencies.py`, and `check_doc_ipfs_hashes.py` — are copy-pasted across **7 repositories** (6 mech repos + open-autonomy). These scripts are used exclusively in CI (via tox) for dependency bumping, dependency validation, and documentation hash checking.

The copies have silently diverged over time, introducing real bugs (missing HTTP timeouts in mech-client and mech-server) and inconsistent behavior (hardcoded version hacks that differ per repo). Consolidating them into the `open-autonomy` CLI eliminates ~4,500 lines of duplicated code, fixes existing bugs by default, and ensures future improvements propagate automatically.

---

## 1. Problem

| Issue | Impact |
|---|---|
| **Maintenance burden** | Bug fixes and improvements must be manually copied to 7 repos. This is routinely forgotten. |
| **Silent drift** | Copies have diverged — same script behaves differently across repos (details in Section 3). |
| **Real bugs from drift** | `mech-client` and `mech-server` `bump.py` are missing HTTP request timeouts — a fix that was applied to other repos but never propagated. |
| **CI noise** | All scripts are included in linting targets (bandit, black, isort, flake8, mypy, pylint, darglint) — linting infrastructure code that nobody actively develops. |
| **Onboarding friction** | Contributors encounter scripts that look identical across repos but behave differently. |

---

## 2. Current State: What Exists Where

### 2.1 Duplicated Scripts (the problem)

| Script | mech-predict | mech-agents-fun | mech-interact | mech | mech-client | mech-server | open-autonomy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `bump.py` | 318 lines | 317 lines | 315 lines | 318 lines | 266 lines | 318 lines | 317 lines |
| `check_dependencies.py` | 175 lines | 175 lines | **652 lines** | 175 lines | — | 185 lines | 202 lines |
| `check_doc_ipfs_hashes.py` | 372 lines | 372 lines | 366 lines | 372 lines | — | 372 lines | **496 lines** |

> **Note:** mech-server uses a `utils/` directory instead of `scripts/`. All other repos use `scripts/`.

### 2.2 Repo-Specific Scripts (not duplicated, not in scope)

| Script | Repo | Purpose | Action |
|---|---|---|---|
| `generate_metadata.py` | mech-predict | Generates tool metadata JSON from package YAML/Python files | Keep per-repo (or migrate to CLI tooling later) |
| `publish_metadata.py` | mech-predict | Pushes metadata.json to IPFS | Keep per-repo (can be replaced by CLI) |
| `test_tool.py`, `test_tools.py` | mech-predict | Tests prediction tools with sample data | Keep per-repo |
| `test_image_gen.py`, `test_recraft_image_gen.py`, `test_short_maker.py`, `test_stabilityai_request.py`, `test_video_gen.py` | mech-agents-fun | Tests AI image/video generation tools | Keep per-repo |
| `compare_hashes.py` | mech-interact | Compares package hashes between repos | Remove — contains hardcoded personal path, one-off dev script |
| `whitelist.py` | mech-client | Vulture dead-code analysis whitelist | Keep per-repo (inherently repo-specific) |
| `benchmark.sh` | mech-client | Benchmarks `mechx` CLI throughput | Keep or remove if unused |

### 2.3 How Scripts Are Currently Invoked

Scripts are invoked **only through tox environments** (not directly in GitHub Actions workflows). Typical pattern in `tox.ini`:

```ini
[testenv:check-dependencies]
allowlist_externals = {toxinidir}/scripts/check_dependencies.py
commands = {toxinidir}/scripts/check_dependencies.py

[testenv:check-doc-hashes]
allowlist_externals = {toxinidir}/scripts/check_doc_ipfs_hashes.py
commands = {toxinidir}/scripts/check_doc_ipfs_hashes.py
```

Additionally, `scripts/` is included as a target in every linting tox environment (bandit, black, isort, flake8, mypy, pylint, darglint) and in the flake8 `application-import-names` config.

---

## 3. Drift Analysis

### 3.1 `bump.py` — 4 variants across 7 repos

| Variant | Repos | What's different |
|---|---|---|
| **A — Reference** | mech-predict, mech, open-autonomy | Uses `TIMEOUT = 30.0` constant for HTTP requests. Copyright 2023–2026. |
| **B — Inline timeout** | mech-agents-fun, mech-interact | Uses `timeout=30` inline instead of constant. Functionally equivalent, cosmetic diff. |
| **C — No timeout (BUG)** | mech-client, mech-server | **`requests.get()` calls have no timeout parameter.** This means HTTP requests can hang indefinitely in CI. mech-client also has stale copyright header (2023 only). |

**Root cause:** The timeout fix was applied to some repos and not propagated to others.

### 3.2 `check_dependencies.py` — 4 distinct variants (worst drift)

| Variant | Repos | Input format | Mode | Hardcoded hacks |
|---|---|---|---|---|
| **A — Basic pyproject.toml** | mech-predict, mech-agents-fun, mech | Reads `pyproject.toml` (poetry deps) | Update + git-diff check | `requests` version hardcoded (`==2.28.2` or `>=2.28.1,<2.33.0`) |
| **B — Improved pyproject.toml** | mech-server | Reads `pyproject.toml` via `Path` objects | Update + content comparison (no git dependency) | `requests==2.32.5` hardcoded, handles `^` and `v` version prefixes |
| **C — Pipfile-based** | open-autonomy | Reads `Pipfile` (dev-packages + packages) | Update + git-diff check | Pops `open-aea-ledger-solana` and `solders`; handles `*` wildcards, extras dicts, `git+` deps |
| **D — Full CLI (most complete)** | mech-interact | Reads **both** Pipfile and pyproject.toml | Separate `--check` (validate-only) and update modes via Click CLI | Proper `Pipfile`, `ToxFile`, `PyProjectToml` classes with check/update/dump methods |

**Key observations:**
- The mech-interact version (652 lines) is already a near-production-ready consolidated solution with proper CLI, multiple config format support, and separate check vs. update modes.
- Every repo hardcodes different package exclusions — these should be CLI flags.
- open-autonomy reads `Pipfile` while all mech repos read `pyproject.toml` — a consolidated version must support both.

### 3.3 `check_doc_ipfs_hashes.py` — 2 variants

| Variant | Repos | Key differences |
|---|---|---|
| **A — Standard** | mech-predict, mech-agents-fun, mech, mech-server, mech-interact | 366–372 lines. Core IPFS hash validation against packages.json. Minor string formatting diffs between repos. |
| **B — Superset** | open-autonomy | **496 lines.** Adds `get_packages_from_repository()` (fetches hashes from GitHub releases), `PACKAGE_MAPPING_REGEX` (matches JSON hash mappings), and `HASH_SKIPS` (hardcoded placeholder hashes to ignore). |

**Key observation:** open-autonomy's version is a strict superset. All downstream copies are simplified versions that dropped features they didn't need.

---

## 4. Consolidation Recommendation

### 4.1 Where to consolidate: `open-autonomy` CLI

All three scripts should be consolidated into the **open-autonomy CLI** as top-level commands.

**Why not tomte (as previously suggested for `bump.py`)?**

| Consideration | Tomte | Open-Autonomy |
|---|---|---|
| Already depends on `aea` / `autonomy`? | No — would need new `[bump]` optional extra | Yes |
| Current scope | Linter wrappers (black, isort, flake8, etc.) | Framework CLI (packages, deployment, analysis) |
| All three scripts import from `aea`? | Yes — adding as dependency changes tomte's scope | Already available |
| Repos using scripts already depend on? | open-autonomy (always) | open-autonomy (always) |

Adding `open-aea` + `open-autonomy` as tomte dependencies (even as optionals) would change tomte from a lightweight linter wrapper to a framework-aware tool. All three scripts naturally belong in open-autonomy because they use core package management APIs (`PackageManagerV1`, `load_configuration`, `PackageId`).

**Why top-level commands (not an `autonomy dev` subgroup)?**

- The `autonomy` CLI already mixes user-facing and CI/developer commands at the top level (e.g. `autonomy analyse handlers`, `autonomy packages lock --check` are CI-only).
- A `dev` subgroup adds an extra word to every CI invocation for no functional benefit.
- There is no existing precedent for a `dev` subgroup in the codebase.
- The `check-*` naming pattern is already established with `autonomy check-packages`.

### 4.2 Proposed CLI Commands

| Current script | Proposed command | Key options |
|---|---|---|
| `bump.py` | `autonomy bump-dependencies` | `--sync`, `--no-cache`, `--source SOURCE` |
| `check_dependencies.py` | `autonomy check-dependencies` | `--check` (validate-only), `--update`, `--exclude PACKAGE` (repeatable), `--pipfile PATH`, `--pyproject PATH` |
| `check_doc_ipfs_hashes.py` | `autonomy check-doc-hashes` | `--fix`, `--skip-hash HASH` (repeatable), `--paths GLOB` |

### 4.3 Migration per repo (after commands are available)

**tox.ini changes:**

```ini
# BEFORE
[testenv:check-dependencies]
allowlist_externals = {toxinidir}/scripts/check_dependencies.py
commands = {toxinidir}/scripts/check_dependencies.py

# AFTER
[testenv:check-dependencies]
commands = autonomy check-dependencies --check
```

```ini
# BEFORE
[testenv:check-doc-hashes]
allowlist_externals = {toxinidir}/scripts/check_doc_ipfs_hashes.py
commands = {toxinidir}/scripts/check_doc_ipfs_hashes.py

# AFTER
[testenv:check-doc-hashes]
commands = autonomy check-doc-hashes
```

**Linting cleanup — remove `scripts` from all linting targets:**

```ini
# BEFORE (repeated across bandit, black, isort, flake8, mypy, pylint, darglint)
commands = bandit -s B101 -r scripts
commands = black {env:SERVICE_SPECIFIC_PACKAGES} scripts
commands = mypy scripts --disallow-untyped-defs --config-file tox.ini
# etc.

# AFTER — remove "scripts" from each command if no repo-specific scripts remain
commands = bandit -s B101 -r {env:SERVICE_SPECIFIC_PACKAGES}
commands = black {env:SERVICE_SPECIFIC_PACKAGES}
# etc.
```

**Delete files:**
- Remove `bump.py`, `check_dependencies.py`, `check_doc_ipfs_hashes.py`, `__init__.py` from `scripts/` (or `utils/` for mech-server)
- For repos where `scripts/` only contained these files (mech, mech-server): delete the directory entirely

---

## 5. Implementation Plan

### Phase 1: `check_doc_ipfs_hashes.py` → `autonomy check-doc-hashes`

**Risk: Low** — open-autonomy's version is already the superset; downstream repos use fewer features.

| Step | Detail |
|---|---|
| Starting point | open-autonomy's 496-line version |
| Changes needed | Replace hardcoded `HASH_SKIPS` with `--skip-hash` CLI option; add `--paths` for configurable doc directories |
| Wire into CLI | Add to `autonomy` CLI |
| Test | Existing CI in open-autonomy validates this already |

### Phase 2: `check_dependencies.py` → `autonomy check-dependencies`

**Risk: Medium** — the copies have meaningfully diverged. The mech-interact version is the best starting point.

| Step | Detail |
|---|---|
| Starting point | mech-interact's 652-line version (has `Pipfile`, `ToxFile`, `PyProjectToml` classes, Click CLI, `--check` flag) |
| Changes needed | Add `--exclude PACKAGE` option to replace hardcoded hacks; auto-detect Pipfile vs pyproject.toml; merge open-autonomy's Pipfile extras/git handling |
| Wire into CLI | Add to `autonomy` CLI |
| Risk mitigation | Run consolidated version against all 7 repos in a test branch before merging |

### Phase 3: `bump.py` → `autonomy bump-dependencies`

**Risk: Low** — all copies are nearly identical; the timeout bug gets fixed automatically.

| Step | Detail |
|---|---|
| Starting point | mech-predict's version (most up-to-date, has `TIMEOUT = 30.0`) |
| Changes needed | Parameterize hardcoded GitHub repos; add CLI options matching current `sys.argv` usage |
| Wire into CLI | Add to `autonomy` CLI |
| Bonus | Fixes mech-client and mech-server timeout bug by default |

### Phase 4: Repo-by-repo migration

For each of the 6 mech repos + open-autonomy:
1. Delete the duplicated scripts
2. Update tox.ini to use `autonomy` CLI commands
3. Remove `scripts` from linting targets
4. Delete `scripts/` directory if empty (mech, mech-server)

---

## 6. What This Fixes

| Before | After |
|---|---|
| ~4,500 lines of duplicated code across 7 repos | 3 CLI commands in one place |
| Bug: mech-client/mech-server `bump.py` missing HTTP timeouts | Fixed automatically |
| 4 divergent variants of `check_dependencies.py` | One version supporting Pipfile + pyproject.toml |
| Hardcoded package exclusions per repo (`requests==2.28.2`, popping `solders`) | Configurable `--exclude` flag |
| Scripts linted as repo source code (bandit, black, isort, flake8, mypy, pylint on `scripts/`) | Eliminated — code lives in open-autonomy, linted there once |
| Manual propagation of fixes | `pip install --upgrade open-autonomy` picks up fixes |

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Consolidated `check_dependencies` fails on a repo due to format differences | Medium | CI breaks | Run against all 7 repos before merging; phase the rollout |
| Extra features in open-autonomy's `check_doc_ipfs_hashes` surface new CI failures in downstream repos | Low | New warnings/errors in CI | Extra checks only activate for cross-repo package references, which downstream repos don't use |
| Repos pinned to older open-autonomy version don't get new commands | Low | Migration blocked until version bump | Coordinate with next scheduled open-autonomy release |
| Breaking change in CLI interface if command signatures change later | Low | tox.ini needs updating | Semantic versioning; deprecation warnings before removing flags |

---

## 8. Corrections to Previous Report

| Previous claim | Actual finding |
|---|---|
| `bump.py` should go to tomte | All 3 scripts should go to open-autonomy — tomte doesn't depend on `aea`/`autonomy` and shouldn't start |
| mech-server has a `scripts/` directory | mech-server uses `utils/` |
| 4–6 repos affected | **7 repos** — open-autonomy itself also has the same duplicated scripts |
| mech-interact's `check_dependencies.py` was not highlighted | It's 652 lines with a full Click CLI — the best starting point for consolidation |
| `compare_hashes.py` (mech-interact) is a shared utility | It contains a hardcoded personal filesystem path (`/home/lockhart/work/...`) — it's a one-off dev script, not a shared tool |

---

## Appendix A: Dependency Graph of Duplicated Scripts

All three scripts import from the same packages:

```
bump.py
├── aea.cli.utils.click_utils (PackagesSource, PyPiDependency)
├── aea.configurations.constants (PACKAGES, PACKAGE_TYPE_TO_CONFIG_FILE)
├── aea.configurations.data_types (Dependency)
├── aea.helpers.yaml_utils (yaml_dump, yaml_load, etc.)
├── aea.package_manager.v1 (PackageManagerV1)
└── autonomy.cli.helpers.ipfs_hash (load_configuration)

check_dependencies.py
├── aea.configurations.data_types (Dependency, PackageType)
├── aea.package_manager.base (load_configuration)
└── aea.package_manager.v1 (PackageManagerV1)

check_doc_ipfs_hashes.py
├── aea.cli.packages (get_package_manager)
├── aea.configurations.data_types (PackageId)
└── aea.helpers.base (IPFS_HASH_REGEX, SIMPLE_ID_REGEX)
```

All imports are already available in any environment that has `open-autonomy` installed — which every repo in scope does.

## Appendix B: Repos After Migration

| Repo | `scripts/` directory | Contents remaining |
|---|---|---|
| mech-predict | Kept | `generate_metadata.py`, `publish_metadata.py`, `test_tool.py`, `test_tools.py` |
| mech-agents-fun | Kept | `test_image_gen.py`, `test_recraft_image_gen.py`, `test_short_maker.py`, `test_stabilityai_request.py`, `test_video_gen.py` |
| mech-interact | Removed | (was only duplicated scripts + `compare_hashes.py` which should be removed) |
| mech | **Removed entirely** | (was only duplicated scripts) |
| mech-client | Kept | `whitelist.py`, `benchmark.sh` |
| mech-server | **`utils/` removed entirely** | (was only duplicated scripts) |
| open-autonomy | Kept | Remaining repo-specific scripts (`check_copyright.py`, `generate_api_documentation.py`, etc.) |
