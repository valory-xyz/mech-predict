# Implementation Plan: Script Deduplication — Phase 2

**Date:** 2026-03-25
**Companion doc:** [analysis-report.md](./analysis-report.md)

---

## Overview

Phase 2 has two parts:
1. **Migrate 8 remaining repos** to existing `aea-helpers` commands (bump, check-dependencies, check-doc-hashes)
2. **Add 4 new commands** to `aea-helpers` for deployment/config scripts (`config-replace`, `run-agent`, `run-service`, `make-release`)

**Dependency chain:**

```
Phase 2A ─── Fix aea-helpers plugin (OA) ──────────────┐
  Task 0.1: Handle customs package type                 │
  Task 0.2: Release aea-helpers update                  │
                                                         ▼
Phase 2B ─── Migrate 8 repos (existing commands) ─────────
  Layer 1 (no cross-deps): funds-manager, genai,
    kv-store, market-creator, optimus                    │  (all parallelizable)
  Layer 2 (depends on upstream): IEKit, meme-ooorr,     │
    trader                                               │
                                                         ▼
Phase 2C ─── New aea-helpers commands (OA) ────────────────
  Task C.1: aea-helpers config-replace                   │
  Task C.2: aea-helpers run-agent                        │  (can be parallelized)
  Task C.3: aea-helpers run-service                      │
  Task C.4: aea-helpers make-release                     │
                                                         ▼
Phase 2D ─── Migrate repos to new commands ────────────────
  config-replace: IEKit, market-creator, meme-ooorr,
    optimus, trader                                      │
  run-agent: all 8 agent repos                           │  (all parallelizable)
  run-service: all 8 agent repos                         │
  make-release: mech, mech-agents-fun, mech-predict      │
```

**Note:** IEKit, meme-ooorr, and trader may have cross-dependencies on other upstream repos. Verify dependency chains before starting Layer 2.

---

## Phase 2A: Plugin Fixes Required

Before migrating, the `aea-helpers` plugin needs one fix already identified:

### Task 0.1: Handle `customs` package type

**Branch:** `fix/aea-helpers-customs-package` (already created on OA)

The `Package` class in `check_doc_hashes.py` crashes on `customs` packages (mech tools). Fix: skip with log message, matching original per-repo behavior.

**Status:** Fix committed, PR pending.

### Task 0.2: Release updated `aea-helpers`

After Task 0.1 merges, cut a new OA release to publish the fix to PyPI. All downstream repos with `customs` packages (mech-predict, mech-agents-fun, IEKit, meme-ooorr, trader, optimus, market-creator, genai) depend on this.

---

## Phase 2B: Migrate 8 Repos

Each repo migration follows the same pattern as the Phase 1 mech repo migrations.

### Common migration steps (per repo)

#### Step A — Add `aea-helpers` dependency

Add to `pyproject.toml` (or `Pipfile`):
```
aea-helpers = "==<version>"
```

And to `tox.ini` `[deps-packages]` (or `[testenv]` deps):
```ini
aea-helpers==<version>
```

#### Step B — Update tox.ini: script invocations → CLI commands

Replace:
```ini
[testenv:check-dependencies]
allowlist_externals = {toxinidir}/scripts/check_dependencies.py
commands = {toxinidir}/scripts/check_dependencies.py
```

With:
```ini
[testenv:check-dependencies]
commands = aea-helpers check-dependencies --check <repo-specific-excludes>
```

Replace:
```ini
[testenv:check-doc-hashes]
allowlist_externals = {toxinidir}/scripts/check_doc_ipfs_hashes.py
commands = {toxinidir}/scripts/check_doc_ipfs_hashes.py
```

With:
```ini
[testenv:check-doc-hashes]
commands = aea-helpers check-doc-hashes
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

#### Step C — Remove `scripts` from linting targets (if directory will be empty)

For repos where `scripts/` only contains the 3 duplicated files + `__init__.py` (funds-manager, genai, kv-store), remove `scripts` from all linting tox targets and `application-import-names`.

For repos with remaining scripts (IEKit, market-creator, meme-ooorr, optimus, trader), keep `scripts` in linting targets.

#### Step D — Delete duplicated scripts

Remove: `bump.py`, `check_dependencies.py`, `check_doc_ipfs_hashes.py`
Remove `__init__.py` only if no other scripts remain.

#### Step E — Verify CI passes

---

### Task 1: `funds-manager`

- **Excludes for check-dependencies:** TBD (check for hardcoded hacks in current script)
- **scripts/ directory:** Delete entirely (only has the 3 duplicated files + `__init__.py`)
- **Linting:** Remove `scripts` from all targets

### Task 2: `genai`

- **Excludes for check-dependencies:** TBD
- **scripts/ directory:** Delete entirely
- **Linting:** Remove `scripts` from all targets

### Task 3: `kv-store`

- **Excludes for check-dependencies:** TBD
- **scripts/ directory:** Delete entirely
- **Linting:** Remove `scripts` from all targets

### Task 4: `market-creator`

- **Excludes for check-dependencies:** TBD
- **scripts/ directory:** Keep — still has `aea-config-replace.py`, `list_markets.py`
- **Delete:** `bump.py`, `check_dependencies.py`, `check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in targets

### Task 5: `optimus`

- **Excludes for check-dependencies:** TBD
- **scripts/ directory:** Keep — still has `aea-config-replace.py`, `run_merkle_api.py`
- **Delete:** `bump.py`, `check_dependencies.py`, `check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in targets

### Task 6: `IEKit`

- **Excludes for check-dependencies:** TBD
- **scripts/ directory:** Keep — has 15+ repo-specific scripts
- **Delete:** `bump.py`, `check_dependencies.py`, `check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in targets

### Task 7: `meme-ooorr`

- **Excludes for check-dependencies:** TBD
- **scripts/ directory:** Keep — has 25+ repo-specific scripts
- **Delete:** `bump.py`, `check_dependencies.py`, `check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in targets

### Task 8: `trader`

- **Excludes for check-dependencies:** TBD
- **scripts/ directory:** Keep — still has `aea-config-replace.py`, `propel.py`
- **Delete:** `bump.py`, `check_dependencies.py`, `check_doc_ipfs_hashes.py`
- **Linting:** Keep `scripts` in targets

---

## Execution Order Checklist

```
[ ] Phase 2A — Plugin fixes
    [ ] 0.1  Merge customs package fix into OA
    [ ] 0.2  Release updated aea-helpers to PyPI

[ ] Phase 2B — Repo migrations (all parallelizable within layers)
    Layer 1 (no cross-deps):
    [ ] 1  funds-manager
    [ ] 2  genai
    [ ] 3  kv-store
    [ ] 4  market-creator
    [ ] 5  optimus

    Layer 2 (may have cross-deps):
    [ ] 6  IEKit
    [ ] 7  meme-ooorr
    [ ] 8  trader
```

---

## Phase 2C: New `aea-helpers` Commands

> **Repo:** `open-autonomy`
> These commands are implemented in the plugin, then downstream repos migrate to them.

---

### Task C.1: `aea-helpers config-replace`

**Consolidates:** `aea-config-replace.py` from 5 repos (IEKit, market-creator, meme-ooorr, optimus, trader)

**Starting point:** trader's version (164 lines, most comprehensive)

**Core logic (shared):**
1. Read `.env` file
2. Load a mapping file (`config-mapping.json` or `config-mapping.yaml`) that maps agent config YAML paths to environment variable names
3. For each mapping entry, substitute the env var value into the agent config

**CLI interface:**
```
aea-helpers config-replace \
  --mapping config-mapping.json \
  --env-file .env \
  --agent-dir agent
```

**Per-repo migration:**
- Extract the `PATH_TO_VAR` dict from each repo's `aea-config-replace.py` into a `config-mapping.json` file
- Delete `aea-config-replace.py`
- Update `run_agent.sh` (or `aea-helpers run-agent`) to call `aea-helpers config-replace`

---

### Task C.2: `aea-helpers run-agent`

**Consolidates:** `run_agent.sh` from 8 repos

**CLI interface:**
```
aea-helpers run-agent \
  --name valory/trader \
  --env-file .env \
  --config-replace \
  --config-mapping config-mapping.json
```

**Flags:**
- `--name` (required): agent name for `autonomy fetch --local --agent <name>`
- `--env-file` (optional): env file to source (default: `.env`)
- `--agent-env-file` (optional): env file passed to `aea -s run --env <file>` (for repos using `.agentenv`)
- `--config-replace` (flag): whether to run config-replace after fetch
- `--config-mapping` (optional): path to mapping file for config-replace
- `--skip-tendermint` (flag): skip tendermint init/start

**Core logic:**
1. Cleanup trap
2. Remove previous build, run `make clean`
3. `autonomy packages lock` + `autonomy fetch --local --agent <name> --alias agent`
4. Copy keys and env files
5. `aea -s add-key ethereum` + `aea -s issue-certificates`
6. (optional) `aea-helpers config-replace`
7. Start tendermint + `aea -s run`

---

### Task C.3: `aea-helpers run-service`

**Consolidates:** `run_service.sh` from 8 repos

**CLI interface:**
```
aea-helpers run-service \
  --name valory/trader \
  --agents 4 \
  --env-file .env \
  --cpu-limit 4.0 \
  --memory-limit 8192
```

**Flags:**
- `--name` (required): service name for `autonomy fetch --local --service <name>`
- `--agents` (optional, default: 4): number of agents
- `--env-file` (optional): env file to source
- `--keys-file` (optional, default: `keys.json`): keys file to copy
- `--cpu-limit` (optional): agent CPU limit for deploy build
- `--memory-limit` (optional): agent memory limit for deploy build
- `--memory-request` (optional): agent memory request for deploy build
- `--docker-cleanup` (flag): clean up Docker containers before starting
- `--detach` (flag): run deployment in detached mode

**Core logic:**
1. Clean previous builds
2. `autonomy push-all`
3. `autonomy fetch --local --service <name>`
4. Copy keys, env files
5. `autonomy deploy build -ltm` (with resource flags)
6. `autonomy deploy run` (with optional `--detach`)

**Repo-specific hooks:** Repos with truly unique steps (meme-ooorr database backup, mech-agents-fun Docker cleanup) can either:
- Use `--pre-deploy-cmd` / `--post-deploy-cmd` flags
- Or keep a thin wrapper script that calls `aea-helpers run-service` with extra steps

---

### Task C.4: `aea-helpers make-release`

**Consolidates:** `make_release.sh` from 3 mech repos (identical, 29 lines)

**CLI interface:**
```
aea-helpers make-release \
  --version 1.0.0 \
  --env prod \
  --description "Release description"
```

**Core logic:**
1. Create git tag: `release_<version>_<env>`
2. Push tag to origin
3. Create GitHub release via `gh release create`

---

## Phase 2D: Migrate Repos to New Commands

### `config-replace` migration (5 repos)

For each of IEKit, market-creator, meme-ooorr, optimus, trader:
1. Extract `PATH_TO_VAR` dict from `aea-config-replace.py` into `config-mapping.json`
2. Delete `aea-config-replace.py`
3. Update any scripts/docs that call `aea-config-replace.py` to use `aea-helpers config-replace`

### `run-agent` migration (8 repos)

For each repo with `run_agent.sh`:
1. Replace `run_agent.sh` with a call to `aea-helpers run-agent` with appropriate flags
2. Or keep `run_agent.sh` as a thin wrapper: `aea-helpers run-agent --name valory/trader --env-file .env --config-replace`

### `run-service` migration (8 repos)

For each repo with `run_service.sh`:
1. Replace core logic with `aea-helpers run-service` call
2. Keep repo-specific pre/post steps as wrapper logic if needed

### `make-release` migration (3 repos)

For mech, mech-agents-fun, mech-predict:
1. Delete `make_release.sh`
2. Use `aea-helpers make-release` directly

---

## Execution Order Checklist

```
[ ] Phase 2A — Plugin fixes
    [ ] 0.1  Merge customs package fix into OA
    [ ] 0.2  Release updated aea-helpers to PyPI

[ ] Phase 2B — Migrate 8 repos (existing commands)
    Layer 1 (no cross-deps):
    [ ] 1  funds-manager
    [ ] 2  genai
    [ ] 3  kv-store
    [ ] 4  market-creator
    [ ] 5  optimus

    Layer 2 (may have cross-deps):
    [ ] 6  IEKit
    [ ] 7  meme-ooorr
    [ ] 8  trader

[ ] Phase 2C — New aea-helpers commands (OA plugin)
    [ ] C.1  Implement aea-helpers config-replace
    [ ] C.2  Implement aea-helpers run-agent
    [ ] C.3  Implement aea-helpers run-service
    [ ] C.4  Implement aea-helpers make-release
    [ ] C.x  Release updated aea-helpers to PyPI

[ ] Phase 2D — Migrate repos to new commands
    [ ] D.1  config-replace: 5 repos (IEKit, market-creator, meme-ooorr, optimus, trader)
    [ ] D.2  run-agent: 8 repos
    [ ] D.3  run-service: 8 repos
    [ ] D.4  make-release: 3 mech repos
```
