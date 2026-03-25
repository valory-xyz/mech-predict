# Implementation Plan: Script Deduplication — Phase 2

**Date:** 2026-03-25
**Companion doc:** [analysis-report.md](./analysis-report.md)

---

## Overview

Phase 2 consolidates all remaining duplicated scripts. The work is split by repo type:

- **Open Autonomy:** Fix customs issue + implement 4 new `aea-helpers` commands
- **Mech repos:** Delete outdated run scripts (replaced by `mech` CLI from mech-server), migrate CI scripts
- **Agent repos:** Migrate CI scripts + deployment scripts to `aea-helpers` commands
- **Library repos:** Migrate CI scripts only

---

## Step 1: Plugin Development (Open Autonomy)

> **Branch:** `fix/aea-helpers-customs-package` (extend this existing branch)

### Task 1.0: Fix customs package type (DONE)

Skip `customs` packages in `check_doc_hashes.py` instead of crashing.

**Status:** Already committed on branch.

### Task 1.1: `aea-helpers config-replace`

**Consolidates:** `aea-config-replace.py` from 5 agent repos

**Starting point:** trader's version — generic PATH_TO_VAR dict + regex-based `find_and_replace()`. ~50 lines of core logic.

**CLI:**
```
aea-helpers config-replace \
  --mapping config-mapping.json \
  --env-file .env \
  --agent-dir agent
```

**What goes into the command:** The `find_and_replace()` function, YAML loading, env var resolution, dotenv support.

**What stays per-repo:** A `config-mapping.json` file containing the PATH_TO_VAR dict. Example:
```json
{
  "config/ledger_apis/gnosis/address": "GNOSIS_LEDGER_RPC",
  "models/params/args/setup/all_participants": "ALL_PARTICIPANTS"
}
```

**Prerequisite for optimus:** Refactor optimus's `aea-config-replace.py` to use the standard PATH_TO_VAR pattern. Move directory creation into `run_agent.sh`. Use `--alias agent` in fetch.

**Edge cases to verify:**
- Regex `${type:value}` handles all type prefixes (str, list, bool, int) — yes, trader's version does string replacement preserving the type prefix
- Nested YAML paths with array indices — not used by any repo currently
- Missing env vars are skipped with warning (not error) — consistent across all 4 repos

### Task 1.2: `aea-helpers run-agent`

**Consolidates:** `run_agent.sh` from 5 agent repos + port management from trader#874

After optimus refactoring, all 5 agent repos follow the same pattern:

```
aea-helpers run-agent \
  --name valory/trader \
  --env-file .env \
  --config-replace \
  --config-mapping config-mapping.json \
  --connection-key \
  --free-ports
```

**Flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--name` (required) | — | Agent name for `autonomy fetch` |
| `--env-file` | `.env` | Env file to source |
| `--config-replace` | off | Run config-replace after fetch |
| `--config-mapping` | — | Path to mapping file (required if `--config-replace`) |
| `--connection-key` | off | Add second key for connection |
| `--free-ports` | off | Auto-find free ports (from trader#874) |
| `--abci-port`, `--rpc-port`, `--p2p-port`, `--com-port`, `--http-port` | defaults | Explicit port overrides |
| `--skip-make-clean` | off | Skip `make clean` step |

**Core logic:**
1. Cleanup trap (kill tendermint on exit)
2. Remove previous agent build directory
3. `find . -empty -type d -delete` + `make clean` (unless `--skip-make-clean`)
4. `autonomy packages lock`
5. `autonomy fetch --local --agent <name> --alias agent`
6. Source env file
7. (if `--config-replace`) Run `aea-helpers config-replace`
8. `cd agent`
9. Copy `ethereum_private_key.txt`
10. `aea -s add-key ethereum` + (if `--connection-key`) `aea -s add-key ethereum --connection`
11. `aea -s issue-certificates`
12. (if `--free-ports`) Resolve ports
13. `tendermint init` + `tendermint node` (with resolved ports)
14. `aea -s run`

**Port management (subsumes trader#874):** Instead of a separate `generate_port_env.py` (412 lines), port resolution is built in. `--free-ports` finds available ports starting from 50000. Individual `--abci-port` etc. flags allow explicit overrides.

### Task 1.3: `aea-helpers run-service`

**Consolidates:** `run_service.sh` from 5 agent repos

```
aea-helpers run-service \
  --name valory/trader \
  --env-file .env \
  --agents 4 \
  --cpu-limit 4.0 \
  --memory-limit 8192 \
  --pre-deploy-cmd "bash pre-deploy.sh"
```

**Flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--name` (required) | — | Service name for `autonomy fetch` |
| `--env-file` | `.env` | Env file to source |
| `--keys-file` | `keys.json` | Keys file to copy |
| `--agents` | 4 | Number of agents |
| `--author` | `valory` | Author for `autonomy init` |
| `--cpu-limit` | — | Agent CPU limit |
| `--memory-limit` | — | Agent memory limit |
| `--memory-request` | — | Agent memory request |
| `--detach` | off | Run deployment in detached mode |
| `--docker-cleanup` | off | Clean Docker containers before start |
| `--pre-deploy-cmd` | — | Command to run before deploy |
| `--post-deploy-cmd` | — | Command to run after deploy |

**Core logic:**
1. Remove previous service build directory
2. `autonomy init --reset --author <author> --remote --ipfs`
3. `autonomy push-all`
4. `autonomy fetch --local --service <name>`
5. `autonomy build-image`
6. Copy keys file + env file
7. (if `--pre-deploy-cmd`) Run pre-deploy command
8. `autonomy deploy build -ltm` (with resource flags)
9. `autonomy deploy run` (with optional `--detach`)
10. (if `--post-deploy-cmd`) Run post-deploy command

**Unique per-repo steps via hooks:**
- market-creator: `--pre-deploy-cmd "bash scripts/escape-prompt.sh"` (sed preprocessing)
- meme-ooorr: `--post-deploy-cmd "bash scripts/backup-db.sh"` (database backup)

### Task 1.4: `aea-helpers make-release`

**Consolidates:** `make_release.sh` from 3 mech repos (identical, 29 lines)

```
aea-helpers make-release --version 1.0.0 --env prod --description "Release"
```

**Core logic:** Create git tag `release_<version>_<env>` → push → `gh release create`.

---

## Step 2: Release Candidate

1. Bump `aea-helpers` version to RC
2. Run full linting + `make generators` on OA
3. Merge the OA branch
4. Publish RC to PyPI

---

## Step 3: Validate on Mech Repos

Test against the 6 mech repos (already on `chore/de-duplicate-scripts` branches):

### 3.1 Verify customs fix

`aea-helpers check-doc-hashes` should pass on mech-predict and mech-agents-fun (which have `customs` packages).

### 3.2 Delete outdated run scripts

In mech, mech-agents-fun, mech-predict:

| Delete | Replacement |
|---|---|
| `run_agent.sh` | `mech run -c gnosis --dev` |
| `run_service.sh` | `mech run -c gnosis` |
| `run_tm.sh` | Managed by `mech run` |
| `make_release.sh` | `aea-helpers make-release` |

### 3.3 Update READMEs

The READMEs in `mech` and `mech-predict` reference the deleted run scripts. Update to document the `mech` CLI from mech-server:

**mech README.md** (lines 164, 170, 190): Replace `bash run_agent.sh`, `bash run_tm.sh`, `bash run_service.sh` with:
```
pip install mech-server
mech setup -c gnosis
mech run -c gnosis          # production (Docker)
mech run -c gnosis --dev    # development (host)
```

**mech-predict README.md** (lines 122, 156): Replace `bash run_agent.sh`, `bash run_service.sh` with same `mech` CLI commands.

**mech-agents-fun**: No README changes needed (no script references).

### 3.4 Verify CI passes

All mech repo PRs should pass CI with:
- `aea-helpers` for CI checks (bump, check-deps, check-doc-hashes)
- Outdated run scripts deleted
- `make_release.sh` deleted (or replaced with `aea-helpers make-release`)

---

## Step 4: Migrate Agent Repos

For trader, optimus, IEKit, market-creator, meme-ooorr:

### 4.1 Migrate CI scripts (same as Phase 1 pattern)

Per repo:
1. Add `aea-helpers==<version>` to deps
2. Update tox.ini: replace script invocations with `aea-helpers` commands
3. Delete `bump.py`, `check_dependencies.py`, `check_doc_ipfs_hashes.py`

### 4.2 Refactor optimus (prerequisite)

Before migrating optimus deployment scripts:
1. Rewrite `aea-config-replace.py` to use PATH_TO_VAR dict pattern
2. Change `autonomy fetch` to use `--alias agent`
3. Move directory creation to `run_agent.sh`
4. Extract PATH_TO_VAR into `config-mapping.json`

### 4.3 Migrate deployment scripts

Per repo:
1. Extract PATH_TO_VAR from `aea-config-replace.py` → `config-mapping.json`
2. Delete `aea-config-replace.py`
3. Replace `run_agent.sh` with `aea-helpers run-agent` call (or thin wrapper)
4. Replace `run_service.sh` with `aea-helpers run-service` call (or thin wrapper with pre/post hooks)

**Trader-specific:** Close PR #874 — port management is built into `aea-helpers run-agent --free-ports`.

### 4.4 Per-repo `run-agent` invocations

```bash
# trader
aea-helpers run-agent --name valory/trader --config-replace --config-mapping config-mapping.json --connection-key

# optimus (after refactor)
aea-helpers run-agent --name valory/optimus --config-replace --config-mapping config-mapping.json --connection-key

# IEKit
aea-helpers run-agent --name valory/impact_evaluator --config-replace --config-mapping config-mapping.json --connection-key

# market-creator
aea-helpers run-agent --name valory/market_maker --config-replace --config-mapping config-mapping.json --connection-key

# meme-ooorr
aea-helpers run-agent --name dvilela/memeooorr --config-replace --config-mapping config-mapping.json --connection-key
```

### 4.5 Per-repo `run-service` invocations

```bash
# trader
aea-helpers run-service --name valory/trader --env-file .env

# optimus
aea-helpers run-service --name valory/optimus --env-file .env

# IEKit
aea-helpers run-service --name valory/impact_evaluator --env-file .env

# market-creator
aea-helpers run-service --name valory/market_maker --env-file .env --pre-deploy-cmd "bash scripts/escape-prompt.sh"

# meme-ooorr
aea-helpers run-service --name dvilela/memeooorr --env-file .env --cpu-limit 4.0 --memory-limit 8192 --memory-request 1024 --detach --post-deploy-cmd "bash scripts/backup-db.sh"
```

---

## Step 5: Migrate Library Repos

For funds-manager, genai, kv-store:

1. Add `aea-helpers==<version>` to deps
2. Update tox.ini
3. Delete `bump.py`, `check_dependencies.py`, `check_doc_ipfs_hashes.py`, `__init__.py`
4. Delete `scripts/` directory entirely (no remaining scripts)
5. Remove `scripts` from all linting targets

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
    [ ] 3.1  Verify customs fix (check-doc-hashes passes)
    [ ] 3.2  Delete outdated run scripts (run_agent.sh, run_service.sh, run_tm.sh)
    [ ] 3.3  Replace make_release.sh with aea-helpers make-release
    [ ] 3.4  Update READMEs in mech and mech-predict to reference mech CLI
    [ ] 3.5  CI green on all 6 mech repos

[ ] Step 4 — Migrate agent repos
    [ ] 4.1  Refactor optimus aea-config-replace.py to standard pattern
    [ ] 4.2  Migrate CI scripts (5 repos: trader, optimus, IEKit, market-creator, meme-ooorr)
    [ ] 4.3  Extract config-mapping.json per repo
    [ ] 4.4  Replace run_agent.sh with aea-helpers run-agent
    [ ] 4.5  Replace run_service.sh with aea-helpers run-service
    [ ] 4.6  Close trader PR #874 (port mgmt built into aea-helpers)
    [ ] 4.7  CI green on all 5 agent repos

[ ] Step 5 — Migrate library repos
    [ ] 5.1  funds-manager
    [ ] 5.2  genai
    [ ] 5.3  kv-store
```
