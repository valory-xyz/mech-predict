# Script Deduplication Analysis — Phase 2

**Date:** 2026-03-25
**Status:** Proposal for Review
**Author:** Engineering
**Scope:** 16 repos across the Valory org

---

## Executive Summary

Following the successful Phase 1 consolidation of `bump.py`, `check_dependencies.py`, and `check_doc_ipfs_hashes.py` into the `aea-helpers` plugin, this report analyses remaining duplicated scripts across 16 Valory repositories. We identify additional consolidation candidates — both Python CI scripts and shell deployment scripts — and recommend which should be added to `aea-helpers` and which should remain per-repo.

---

## 1. Repos Analysed

### Agent repos (downstream consumers)

| Repo | URL |
|---|---|
| Trader | https://github.com/valory-xyz/trader |
| Optimus | https://github.com/valory-xyz/optimus |
| Agents Fun (Meme-ooorr) | https://github.com/valory-xyz/meme-ooorr |
| Mech Agents Fun | https://github.com/valory-xyz/mech-agents-fun |
| Mech Predict | https://github.com/valory-xyz/mech-predict |
| Mech Client | https://github.com/valory-xyz/mech-client |
| Mech Server | https://github.com/valory-xyz/mech-server |
| Mech Interact | https://github.com/valory-xyz/mech-interact |
| Mech | https://github.com/valory-xyz/mech |
| Market Creator | https://github.com/valory-xyz/market-creator |

### Upstream repos (frameworks/libraries)

| Repo | URL |
|---|---|
| Open AEA | https://github.com/valory-xyz/open-aea |
| Open Autonomy | https://github.com/valory-xyz/open-autonomy |
| IE Kit | https://github.com/valory-xyz/IEKit |
| Gen AI | https://github.com/valory-xyz/genai |
| Funds Manager | https://github.com/valory-xyz/funds-manager |
| KV Store | https://github.com/valory-xyz/kv-store |

---

## 2. Phase 1 Status (Already Completed)

The following scripts have already been consolidated into the `aea-helpers` plugin (`open-autonomy/plugins/aea-helpers/`):

| Script | Plugin command | Status |
|---|---|---|
| `bump.py` | `aea-helpers bump-dependencies` | Migrated in 6 mech repos |
| `check_dependencies.py` | `aea-helpers check-dependencies` | Migrated in 6 mech repos |
| `check_doc_ipfs_hashes.py` | `aea-helpers check-doc-hashes` | Migrated in 6 mech repos |

**Remaining repos with these scripts (not yet migrated):** funds-manager, genai, IEKit, kv-store, market-creator, meme-ooorr, optimus, trader (8 repos).

---

## 3. Duplicated Python Scripts (New Findings)

### 3.1 `bump.py` — 8 repos still have local copies

| Repo | Lines | Timeout | Auth method |
|---|---|---|---|
| funds-manager | 319 | None (BUG) | `os.getenv()` |
| genai | 318 | 10s | `os.getenv()` |
| IEKit | 318 | 60s | `os.environ.get()` |
| kv-store | 318 | 10s | `os.getenv()` |
| market-creator | 318 | 10s | `os.getenv()` |
| meme-ooorr | 319 | 60s | `os.environ.get()` |
| optimus | 320 | 30s (constant) | `os.environ.get()` |
| trader | 319 | 30s | `os.environ.get()` |

Same drift pattern as Phase 1 — inconsistent timeouts, mixed auth methods. The `aea-helpers bump-dependencies` command fixes all of these.

**Action:** Migrate all 8 repos to `aea-helpers bump-dependencies`.

### 3.2 `check_dependencies.py` — 8 repos, all identical (653 lines)

All 8 repos (funds-manager, genai, IEKit, kv-store, market-creator, meme-ooorr, optimus, trader) have identical copies of the mech-interact version with Click CLI, `Pipfile`/`ToxFile`/`PyProjectToml` classes.

**Action:** Migrate all 8 repos to `aea-helpers check-dependencies`.

### 3.3 `check_doc_ipfs_hashes.py` — 8 repos, minor drift

| Group | Repos | Lines |
|---|---|---|
| Group 1 | funds-manager, genai, kv-store, market-creator, trader | 373 |
| Group 2 | IEKit | 372 |
| Group 3 | meme-ooorr, optimus | 369 |

Minor string formatting differences. All subsets of the OA version.

**Action:** Migrate all 8 repos to `aea-helpers check-doc-hashes`.

### 3.4 `aea-config-replace.py` — 5 repos, REPO-SPECIFIC

| Repo | Lines | Config entries |
|---|---|---|
| IEKit | 146 | 35 (Twitter, OpenAI, Ceramic) |
| market-creator | 140 | 70 (News API, Omen subgraph) |
| meme-ooorr | 152 | 82 (Base chain, memecoin, x402) |
| optimus | 145 | Different structure entirely |
| trader | 164 | 94 (Polygon, multi-chain, Polymarket) |

Each repo has completely different environment variable mappings specific to its service. However, the **core logic is identical**: read a `.env` file, iterate over a path→env_var mapping, and substitute values into agent YAML config files. Only the mapping dict differs per repo.

**Action:** Extract the shared logic into `aea-helpers config-replace --mapping <file>`. Each repo keeps a `config-mapping.json` (or YAML) file defining its specific variable mappings. The script logic (~100 lines) is consolidated; the per-repo config (~50-90 entries) stays local.

### 3.5 `propel.py` — 2 repos, shared core classes

| Repo | Lines | Service |
|---|---|---|
| IEKit | 352 | impact_evaluator (4 agents) |
| trader | 320 | trader_pearl (variable agents) |

The `Agent`, `Service`, and `Propel` classes are ~90% identical. The deployment-specific logic (service names, variables, agent counts) differs.

**Action:** Keep per-repo for now. Low ROI to extract — only 2 repos use it.

### 3.6 `whitelist.py` — 3 repos, REPO-SPECIFIC

Each is a vulture dead-code whitelist unique to its codebase. Cannot be shared.

**Action:** Keep per-repo.

---

## 4. Duplicated Shell Scripts (New Findings)

### 4.1 `run_agent.sh` — 8 repos

| Group | Repos | Lines | Notes |
|---|---|---|---|
| Identical | mech, mech-agents-fun, mech-predict | 44 | Same script, only agent name differs |
| Similar | IEKit, market-creator, trader | 47 | Same pattern + `aea-config-replace.py` call |
| Similar | meme-ooorr | 46 | Same pattern + `aea-config-replace.py` call |
| Unique | optimus | 64 | Different structure, test API support |

**Core pattern (shared across all):**
1. Cleanup trap for tendermint
2. Remove previous build, clean directories
3. `autonomy packages lock` + `autonomy fetch --local --agent <name>`
4. Copy keys/env, add ethereum key, issue certificates
5. Start tendermint + `aea -s run`

**Differences:** Agent name, env file handling (`.agentenv` vs `.env` + `aea-config-replace.py`), number of key-add calls.

**Action:** Extract into `aea-helpers run-agent` command. Differences are parameterizable:

```
aea-helpers run-agent \
  --name valory/trader \
  --env-file .env \
  --config-replace           # calls aea-helpers config-replace internally
```

Flags cover all variations: `--name` (agent name), `--env-file` (`.env` vs `.agentenv`), `--config-replace` (whether to run config substitution), `--extra-key-add` (for repos needing a second key-add call).

### 4.2 `run_service.sh` — 8 repos, all unique but same core pattern

All 8 scripts follow the same pattern with repo-specific flags:
1. Clean previous builds
2. `autonomy push-all` + `autonomy fetch --local --service <name>`
3. Build Docker image
4. Copy keys/env, build deployment, run deployment

**Differences are all parameterizable:**
- Service name → `--name` flag
- Agent count → `--agents` flag
- Deploy flags (CPU/memory limits) → `--cpu-limit`, `--memory-limit` flags
- Env file sourcing → `--env-file` flag
- Docker cleanup → `--docker-cleanup` flag
- Build directory discovery → standardize to one approach

**Action:** Extract into `aea-helpers run-service` command:

```
aea-helpers run-service \
  --name valory/trader \
  --agents 4 \
  --env-file .env \
  --cpu-limit 4.0 \
  --memory-limit 8192
```

Some repos have truly unique steps (meme-ooorr has database backup, mech-agents-fun has Docker container cleanup). These can be handled with pre/post hooks or kept as small wrapper scripts that call `aea-helpers run-service` internally.

### 4.3 `run_tm.sh` — 2 repos, identical (2 lines)

mech and mech-agents-fun have identical 2-line tendermint startup scripts.

**Action:** Trivial — can be removed if `run_agent.sh` is consolidated (it starts tendermint internally).

### 4.4 `make_release.sh` — 3 repos, identical (29 lines)

mech, mech-agents-fun, mech-predict have identical release scripts that create git tags and GitHub releases.

**Action:** Could be added to `aea-helpers` as `aea-helpers make-release` or moved to a shared GitHub Actions workflow.

---

## 5. Consolidation Summary

### Immediate action (add to `aea-helpers` plugin)

These are already implemented in `aea-helpers` — just need migration in the 8 remaining repos:

| Script | Repos to migrate | Plugin command |
|---|---|---|
| `bump.py` | funds-manager, genai, IEKit, kv-store, market-creator, meme-ooorr, optimus, trader | `aea-helpers bump-dependencies` |
| `check_dependencies.py` | Same 8 repos | `aea-helpers check-dependencies` |
| `check_doc_ipfs_hashes.py` | Same 8 repos | `aea-helpers check-doc-hashes` |

### New `aea-helpers` commands to implement

| Script | Repos | Proposed command | Key flags |
|---|---|---|---|
| `aea-config-replace.py` | 5 repos | `aea-helpers config-replace` | `--mapping <file>`, `--env-file <path>` |
| `run_agent.sh` | 8 repos | `aea-helpers run-agent` | `--name <agent>`, `--env-file <path>`, `--config-replace` |
| `run_service.sh` | 8 repos | `aea-helpers run-service` | `--name <service>`, `--agents <n>`, `--env-file <path>`, `--cpu-limit`, `--memory-limit` |
| `make_release.sh` | 3 mech repos | `aea-helpers make-release` | `--version`, `--env`, `--description` |

### Keep per-repo (not candidates for consolidation)

| Script | Reason |
|---|---|
| `whitelist.py` | Vulture config, inherently repo-specific |
| `propel.py` | Only 2 repos, different deployment targets |
| `config-mapping.json` (per-repo) | Variable mappings for `config-replace`, inherently service-specific |
| IEKit ceramic/staking scripts | Domain-specific utilities |
| meme-ooorr analytics/test scripts | Domain-specific utilities |
| mech-predict metadata scripts | Domain-specific utilities |
| mech-agents-fun test scripts | Domain-specific test utilities |
| open-aea framework scripts | Framework-specific CI tooling |
| open-autonomy framework scripts | Framework-specific CI tooling |

---

## 6. Repos After Phase 2 Migration

After migrating the 8 remaining repos to `aea-helpers`:

| Repo | Scripts remaining | Action |
|---|---|---|
| funds-manager | `__init__.py` only | Delete `scripts/` entirely |
| genai | `__init__.py` only | Delete `scripts/` entirely |
| kv-store | `__init__.py` only | Delete `scripts/` entirely |
| IEKit | 15+ repo-specific scripts | Keep `scripts/`, remove 3 duplicated files |
| market-creator | `aea-config-replace.py`, `list_markets.py` | Keep `scripts/`, remove 3 duplicated files |
| meme-ooorr | 25+ repo-specific scripts | Keep `scripts/`, remove 3 duplicated files |
| optimus | `aea-config-replace.py`, `run_merkle_api.py` | Keep `scripts/`, remove 3 duplicated files |
| trader | `aea-config-replace.py`, `propel.py` | Keep `scripts/`, remove 3 duplicated files |

---

## Appendix: Deployment Script Patterns

### `run_agent.sh` common template

```bash
#!/bin/bash
# Cleanup
trap 'kill $PPID' EXIT
# Fetch and configure agent
autonomy packages lock
autonomy fetch --local --agent <AGENT_NAME> --alias agent
# Keys and certificates
aea -s add-key ethereum
aea -s issue-certificates
# Start tendermint and agent
tendermint init && tendermint node &
aea -s run
```

### `make_release.sh` (identical across 3 repos)

```bash
#!/bin/bash
VERSION=$1; ENV=$2; DESCRIPTION=$3
TAG="release_${VERSION}_${ENV}"
git tag -a "$TAG" -m "$DESCRIPTION"
git push origin "$TAG"
gh release create "$TAG" --title "$TAG" --notes "$DESCRIPTION"
```
