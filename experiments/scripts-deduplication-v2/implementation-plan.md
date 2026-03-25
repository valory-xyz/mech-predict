# Implementation Plan: Script Deduplication — Phase 2

**Date:** 2026-03-25
**Companion doc:** [analysis-report.md](./analysis-report.md)

---

## Overview

Phase 2 consolidates all remaining duplicated scripts into the `aea-helpers` plugin. This includes:
- The 3 existing CI commands (bump, check-dependencies, check-doc-hashes) for 8 more repos
- 4 new commands (config-replace, run-agent with port management, run-service, make-release)

All plugin work happens on a single OA branch, tested against mech repos first, then rolled out to all repos.

**Dependency chain:**

```
Step 1 ─── Plugin development (single OA branch) ─────────┐
  Fix customs package type (already done)                   │
  Implement config-replace                                  │
  Implement run-agent (with port management)                │
  Implement run-service                                     │
  Implement make-release                                    │
  Thorough testing of all commands                          │
                                                             ▼
Step 2 ─── Release candidate ──────────────────────────────┐
  Release aea-helpers RC to PyPI                            │
                                                             ▼
Step 3 ─── Validate on mech repos ─────────────────────────┐
  Test all commands against mech repos (CI must pass)       │
  Fix any edge cases found                                  │
                                                             ▼
Step 4 ─── Migrate all remaining repos ────────────────────
  Replace all duplicated scripts with aea-helpers commands
  in all 8 remaining repos + mech repos for new commands
```

---

## Step 1: Plugin Development

> **Repo:** `open-autonomy`
> **Branch:** `fix/aea-helpers-customs-package` (extend this existing branch)
> All tasks below are on the same branch.

---

### Task 1.0: Fix customs package type (DONE)

The `Package` class in `check_doc_hashes.py` crashes on `customs` packages. Fix: skip with log message.

**Status:** Already committed on `fix/aea-helpers-customs-package`.

---

### Task 1.1: `aea-helpers config-replace`

**Consolidates:** `aea-config-replace.py` from 5 repos (IEKit, market-creator, meme-ooorr, optimus, trader)

**Starting point:** trader's version (164 lines, most comprehensive)

**Core logic (shared):**
1. Read a `.env` file (or environment variables)
2. Load a mapping file (`config-mapping.json` or `config-mapping.yaml`) that maps agent config YAML paths to environment variable names
3. For each mapping entry, substitute the env var value into the agent config YAML

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

**Edge cases to handle:**
- optimus has a different structure (creates data directory, uses MODE_LEDGER_RPC) — verify this is covered
- Some repos have nested YAML paths with array indices
- Some values need type coercion (bool, int vs string)

---

### Task 1.2: `aea-helpers run-agent`

**Consolidates:** `run_agent.sh` from 8 repos + port management from [trader#874](https://github.com/valory-xyz/trader/pull/874)

**CLI interface:**
```
aea-helpers run-agent \
  --name valory/trader \
  --env-file .env \
  --config-replace \
  --config-mapping config-mapping.json \
  --free-ports
```

**Flags:**
- `--name` (required): agent name for `autonomy fetch --local --agent <name>`
- `--env-file` (optional): env file to source (default: `.env`)
- `--agent-env-file` (optional): env file passed to `aea -s run --env <file>` (for repos using `.agentenv`)
- `--config-replace` (flag): run config-replace after fetch
- `--config-mapping` (optional): path to mapping file for config-replace
- `--skip-tendermint` (flag): skip tendermint init/start
- `--free-ports` (flag): auto-find free ports for tendermint/HTTP (from trader#874)
- `--abci-port`, `--rpc-port`, `--p2p-port`, `--com-port`, `--http-port` (optional): explicit port overrides

**Port management (from trader#874):**

Instead of a separate `scripts/generate_port_env.py` (412 lines) + shell integration, port resolution is built into the command:
- Without `--free-ports`: uses default ports or environment variable overrides
- With `--free-ports`: auto-finds available ports starting from 50000, respects any explicitly set ports
- Ports are validated for availability before starting
- Supports running multiple agents on the same machine without port conflicts

This subsumes the trader#874 PR — all repos get port management for free.

**Core logic:**
1. Cleanup trap
2. Remove previous build, run `make clean`
3. `autonomy packages lock` + `autonomy fetch --local --agent <name> --alias agent`
4. Copy keys and env files
5. `aea -s add-key ethereum` + `aea -s issue-certificates`
6. (optional) `aea-helpers config-replace`
7. (optional) Port resolution and environment setup
8. Start tendermint (with resolved ports) + `aea -s run`

**Edge cases to handle:**
- mech repos use `.agentenv` instead of `.env`
- Some repos call `add-key` twice (ethereum + connection key)
- optimus has optional test API support (`USE_TEST_API` env var)
- Tendermint startup flags vary slightly between repos

---

### Task 1.3: `aea-helpers run-service`

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
- `--pre-deploy-cmd` (optional): command to run before deploy
- `--post-deploy-cmd` (optional): command to run after deploy

**Core logic:**
1. Clean previous builds
2. `autonomy push-all`
3. `autonomy fetch --local --service <name>`
4. Copy keys, env files
5. (optional) Run pre-deploy command
6. `autonomy deploy build -ltm` (with resource flags)
7. `autonomy deploy run` (with optional `--detach`)
8. (optional) Run post-deploy command

**Edge cases to handle:**
- meme-ooorr has database backup logic in stop/start cycle
- mech-agents-fun has Docker container cleanup
- Build directory discovery varies (some use `find`, some use `ls -d`, some hardcode `abci_build/`)
- Some repos have `.1env` typo in env file sourcing — standardize
- market-creator has extensive hardcoded environment setup in the script

---

### Task 1.4: `aea-helpers make-release`

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

## Step 2: Release Candidate

After all 4 new commands + the customs fix are implemented and tested locally:

1. Bump `aea-helpers` version to next RC (e.g. `0.21.16rc1`)
2. Merge the OA branch
3. Cut an OA release to publish the RC to PyPI
4. Downstream repos can test with `aea-helpers==0.21.16rc1`

---

## Step 3: Validate on Mech Repos

Test all commands against the 6 mech repos (already on `chore/de-duplicate-scripts` branches):

1. **Existing commands** (bump, check-dependencies, check-doc-hashes): should already work — verify customs fix resolves mech-predict/mech-agents-fun failures
2. **run-agent**: test against mech, mech-predict, mech-agents-fun (have `run_agent.sh`)
3. **run-service**: test against mech, mech-predict, mech-agents-fun, mech-server (have `run_service.sh`)
4. **make-release**: test against mech, mech-predict, mech-agents-fun (have `make_release.sh`)

Fix any edge cases found. Once mech repo CI is green, proceed to Step 4.

---

## Step 4: Migrate All Remaining Repos

### 4A: Migrate existing commands (8 repos)

Migrate `bump.py`, `check_dependencies.py`, `check_doc_ipfs_hashes.py` in: funds-manager, genai, IEKit, kv-store, market-creator, meme-ooorr, optimus, trader.

Same pattern as Phase 1 mech migrations. Per repo:
1. Add `aea-helpers==<version>` to deps
2. Update tox.ini commands
3. Delete duplicated scripts
4. Remove `scripts` from linting targets if directory empty

**Layer 1 (no cross-deps):** funds-manager, genai, kv-store, market-creator, optimus
**Layer 2 (may have cross-deps):** IEKit, meme-ooorr, trader

### 4B: Migrate config-replace (5 repos)

For IEKit, market-creator, meme-ooorr, optimus, trader:
1. Extract `PATH_TO_VAR` dict into `config-mapping.json`
2. Delete `aea-config-replace.py`
3. Update references to use `aea-helpers config-replace --mapping config-mapping.json`

### 4C: Migrate run-agent (8+ repos)

For all repos with `run_agent.sh`:
- mech, mech-agents-fun, mech-predict (identical scripts)
- IEKit, market-creator, trader, meme-ooorr (similar + config-replace)
- optimus (unique structure)

Replace `run_agent.sh` with `aea-helpers run-agent` call with appropriate flags. Subsumes trader#874 (port management).

### 4D: Migrate run-service (8+ repos)

For all repos with `run_service.sh`:
Replace core logic with `aea-helpers run-service`. Keep thin wrapper scripts for repos with unique pre/post steps.

### 4E: Migrate make-release (3 repos)

For mech, mech-agents-fun, mech-predict:
Delete `make_release.sh`, use `aea-helpers make-release` directly.

---

## Execution Order Checklist

```
[ ] Step 1 — Plugin development (single OA branch)
    [ ] 1.0  Fix customs package type (DONE)
    [ ] 1.1  Implement aea-helpers config-replace + tests
    [ ] 1.2  Implement aea-helpers run-agent (with port mgmt) + tests
    [ ] 1.3  Implement aea-helpers run-service + tests
    [ ] 1.4  Implement aea-helpers make-release + tests
    [ ] 1.x  Run linting + make generators on OA

[ ] Step 2 — Release candidate
    [ ] 2.1  Bump plugin version
    [ ] 2.2  Merge OA branch
    [ ] 2.3  Publish aea-helpers RC to PyPI

[ ] Step 3 — Validate on mech repos
    [ ] 3.1  Test existing commands (customs fix)
    [ ] 3.2  Test run-agent on mech repos
    [ ] 3.3  Test run-service on mech repos
    [ ] 3.4  Test make-release on mech repos
    [ ] 3.x  Fix edge cases, re-release if needed

[ ] Step 4 — Migrate all repos
    [ ] 4A   Migrate existing commands (8 repos, layered)
    [ ] 4B   Migrate config-replace (5 repos)
    [ ] 4C   Migrate run-agent (8+ repos)
    [ ] 4D   Migrate run-service (8+ repos)
    [ ] 4E   Migrate make-release (3 repos)
```
