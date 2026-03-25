# Script Deduplication Analysis — Phase 2

**Date:** 2026-03-25
**Status:** Proposal for Review
**Author:** Engineering
**Scope:** 16 repos across the Valory org

---

## Executive Summary

Following the successful Phase 1 consolidation of `bump.py`, `check_dependencies.py`, and `check_doc_ipfs_hashes.py` into the `aea-helpers` plugin, this report analyses remaining duplicated scripts across 16 Valory repositories.

Key finding: **mech repos' run scripts (`run_agent.sh`, `run_service.sh`) are outdated** — the `mech-server` package now provides the `mech` CLI (`mech run`, `mech stop`, `mech setup`) which is the proper way to run mech services. These scripts should be deleted, not consolidated.

For **agent repos** (trader, optimus, IEKit, market-creator, meme-ooorr), the deployment scripts (`run_agent.sh`, `run_service.sh`, `aea-config-replace.py`) share ~80% logic and should be consolidated into `aea-helpers` commands.

---

## 1. Repos Analysed

### Mech repos (use `mech` CLI from mech-server)

| Repo | URL |
|---|---|
| Mech | https://github.com/valory-xyz/mech |
| Mech Agents Fun | https://github.com/valory-xyz/mech-agents-fun |
| Mech Predict | https://github.com/valory-xyz/mech-predict |
| Mech Client | https://github.com/valory-xyz/mech-client |
| Mech Server | https://github.com/valory-xyz/mech-server |
| Mech Interact | https://github.com/valory-xyz/mech-interact |

### Agent repos (need `aea-helpers` run commands)

| Repo | URL |
|---|---|
| Trader | https://github.com/valory-xyz/trader |
| Optimus | https://github.com/valory-xyz/optimus |
| Agents Fun (Meme-ooorr) | https://github.com/valory-xyz/meme-ooorr |
| Market Creator | https://github.com/valory-xyz/market-creator |
| IE Kit | https://github.com/valory-xyz/IEKit |

### Upstream repos (frameworks/libraries)

| Repo | URL |
|---|---|
| Open AEA | https://github.com/valory-xyz/open-aea |
| Open Autonomy | https://github.com/valory-xyz/open-autonomy |
| Gen AI | https://github.com/valory-xyz/genai |
| Funds Manager | https://github.com/valory-xyz/funds-manager |
| KV Store | https://github.com/valory-xyz/kv-store |

---

## 2. Phase 1 Status (Already Completed)

The following scripts have already been consolidated into the `aea-helpers` plugin:

| Script | Plugin command | Status |
|---|---|---|
| `bump.py` | `aea-helpers bump-dependencies` | Migrated in 6 mech repos |
| `check_dependencies.py` | `aea-helpers check-dependencies` | Migrated in 6 mech repos |
| `check_doc_ipfs_hashes.py` | `aea-helpers check-doc-hashes` | Migrated in 6 mech repos |

**Remaining repos with these scripts (not yet migrated):** funds-manager, genai, IEKit, kv-store, market-creator, meme-ooorr, optimus, trader (8 repos).

---

## 3. Mech Repos: Outdated Run Scripts

### 3.1 The `mech` CLI (from mech-server)

The `mech-server` package provides a proper CLI via `mech` command:

| Command | What it does |
|---|---|
| `mech setup -c gnosis` | Configure service, deploy metadata |
| `mech run -c gnosis` | Run service via Docker (production) |
| `mech run -c gnosis --dev` | Run on host — pushes packages, starts tendermint + agent |
| `mech stop -c gnosis` | Stop service |
| `mech add-tool` | Scaffold new tools |
| `mech prepare-metadata` | Generate and publish metadata to IPFS |
| `mech update-metadata` | Update metadata hash on-chain |

`mech run --dev` internally does everything that `run_agent.sh` + `run_service.sh` do: pushes packages, resolves service hash, builds deployment, and runs.

### 3.2 Outdated scripts in mech repos

| Script | mech | mech-agents-fun | mech-predict | Action |
|---|---|---|---|---|
| `run_agent.sh` | Yes (44 lines) | Yes (44 lines) | Yes (44 lines) | **Delete** — use `mech run --dev` |
| `run_service.sh` | Yes (31 lines) | Yes (60 lines) | Yes (33 lines) | **Delete** — use `mech run` |
| `run_tm.sh` | Yes (2 lines) | Yes (2 lines) | No | **Delete** — tendermint managed by `mech run` |
| `make_release.sh` | Yes (29 lines) | Yes (29 lines) | Yes (29 lines) | **Consolidate** into `aea-helpers make-release` |

All three `run_agent.sh` files are identical (only agent name differs). All `run_tm.sh` files are identical (2 lines). All `make_release.sh` files are byte-for-byte identical (29 lines).

---

## 4. Duplicated CI Scripts (8 repos still need migration)

### 4.1 `bump.py` — 8 repos still have local copies

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

**Action:** Migrate all 8 repos to `aea-helpers bump-dependencies`.

### 4.2 `check_dependencies.py` — 8 repos, all identical (653 lines)

**Action:** Migrate all 8 repos to `aea-helpers check-dependencies`.

### 4.3 `check_doc_ipfs_hashes.py` — 8 repos, minor drift

**Action:** Migrate all 8 repos to `aea-helpers check-doc-hashes`.

---

## 5. Agent Repo Deployment Scripts

### 5.1 `aea-config-replace.py` — 5 agent repos

**Deep analysis findings:**

| Repo | Lines | Entries | Structure |
|---|---|---|---|
| IEKit | 146 | 41 | Generic: PATH_TO_VAR dict + regex replace |
| market-creator | 140 | 21 | Generic: PATH_TO_VAR dict + regex replace |
| meme-ooorr | 152 | 40 | Generic: PATH_TO_VAR dict + regex replace |
| trader | 164 | 51 | Generic: PATH_TO_VAR dict + regex replace |
| optimus | 145 | ~15 | **Different**: hardcoded logic, dir creation, type coercions |

**4 repos are structurally identical** — only the PATH_TO_VAR mapping differs. The core logic (trader's version, ~50 lines):
1. `load_dotenv(override=True)`
2. Load `aea-config.yaml` with `yaml.safe_load_all()`
3. For each path/var in PATH_TO_VAR: find section by path traversal, extract old value via regex `${type:value}`, replace with env var
4. Write back with `yaml.dump_all()`

**Optimus is genuinely different:**
- Hardcoded logic instead of dict iteration
- Creates `data/` directory (side effect)
- Type coercions: `${list:...}`, `${str:...}` explicit in code
- Searches config by `public_id` and `type` instead of path traversal
- Uses hardcoded "optimus" directory instead of "agent"

**Action:**
- Refactor optimus to use the standard PATH_TO_VAR pattern (move dir creation to run script, use `--alias agent` in fetch)
- Then consolidate all 5 into `aea-helpers config-replace --mapping config-mapping.json`

### 5.2 `run_agent.sh` — 5 agent repos

After excluding mech repos (which should use `mech` CLI), 5 agent repos remain:

| Repo | Lines | Env handling | Config replace | Keys | Unique features |
|---|---|---|---|---|---|
| IEKit | 47 | `.env` + config-replace | Yes | 2 (agent + connection) | — |
| market-creator | 47 | `.env` + config-replace | Yes | 2 (agent + connection) | — |
| trader | 47 | `.env` + config-replace | Yes | 2 (agent + connection) | Port management PR #874 |
| meme-ooorr | 46 | `.env` + config-replace | Yes | 2 (agent + connection) | — |
| optimus | 64 | `.env` + config-replace | Yes (custom) | 2 (agent + connection) | No make clean, no --alias, `run_merkle_api.py` (outdated test mock — delete) |

After migrating optimus to `--alias agent` and standard config-replace, all 5 follow the same flow:
1. Cleanup trap → 2. Remove previous build → 3. `make clean` → 4. `autonomy packages lock` → 5. `autonomy fetch --local --agent <name> --alias agent` → 6. Source `.env` → 7. Run `aea-config-replace.py` → 8. `cd agent` → 9. Copy keys → 10. `add-key ethereum` + `add-key ethereum --connection` → 11. `issue-certificates` → 12. Start tendermint → 13. `aea -s run`

**Action:** Consolidate into `aea-helpers run-agent`:
```
aea-helpers run-agent \
  --name valory/trader \
  --env-file .env \
  --config-replace \
  --config-mapping config-mapping.json \
  --connection-key
```

**Port management from [trader#874](https://github.com/valory-xyz/trader/pull/874):** Built into the command via `--free-ports` flag, giving all repos automatic port resolution without a separate 412-line script.

### 5.3 `run_service.sh` — 5 agent repos

| Repo | Lines | Service name | Agents | Deploy flags | Unique steps |
|---|---|---|---|---|---|
| IEKit | 31 | valory/impact_evaluator | default | `-ltm` | Sources .env after build |
| market-creator | 48 | valory/market_maker | default | `-ltm` | Prompt file sed preprocessing |
| trader | 32 | valory/trader | default | `-ltm` | — |
| meme-ooorr | 42 | dvilela/memeooorr | default | `-ltm --agent-cpu-limit 4.0 --agent-memory-limit 8192 --agent-memory-request 1024` | Database backup |
| optimus | 30 | valory/optimus | default | `-ltm` | — |

Core pattern: clean → `autonomy init` → `autonomy push-all` → `autonomy fetch --local --service <name>` → `autonomy build-image` → copy keys → `autonomy deploy build -ltm` → `autonomy deploy run`

**Truly unique steps that need hooks:**
- market-creator: reads `market_identification_prompt.txt` and escapes it with `sed` before deploy
- meme-ooorr: copies `memeooorr.db` to `persistent_data/logs` for database persistence

**Action:** Consolidate into `aea-helpers run-service`:
```
aea-helpers run-service \
  --name valory/trader \
  --env-file .env \
  --agents 4 \
  --cpu-limit 4.0 \
  --memory-limit 8192
```

With `--pre-deploy-cmd` / `--post-deploy-cmd` for unique steps.

### 5.4 `make_release.sh` — 3 mech repos (identical)

All 3 copies (mech, mech-agents-fun, mech-predict) are byte-for-byte identical. Creates git tag + GitHub release.

**Action:** Consolidate into `aea-helpers make-release`.

---

## 6. Consolidation Summary

### Changes in Open Autonomy (`aea-helpers` plugin)

| Task | Description |
|---|---|
| Fix customs package | Handle `customs` type in check-doc-hashes (already on branch) |
| `aea-helpers config-replace` | New command — generic YAML config substitution with mapping file |
| `aea-helpers run-agent` | New command — fetch, configure, and run agent with port management |
| `aea-helpers run-service` | New command — build and deploy service with resource limits |
| `aea-helpers make-release` | New command — create git tag + GitHub release |

### Changes in Mech Repos

| Repo | Delete | Keep | Notes |
|---|---|---|---|
| mech | `run_agent.sh`, `run_service.sh`, `run_tm.sh`, `make_release.sh` | — | Use `mech` CLI instead; `make_release.sh` → `aea-helpers make-release` |
| mech-agents-fun | `run_agent.sh`, `run_service.sh`, `run_tm.sh`, `make_release.sh` | test scripts | Same as above |
| mech-predict | `run_agent.sh`, `run_service.sh`, `make_release.sh` | metadata scripts, test scripts | Same as above |
| mech-client | — | `whitelist.py`, `benchmark.sh` | No run scripts to delete |
| mech-server | — | — | Provides the `mech` CLI, no scripts to change |
| mech-interact | — | — | No scripts |

### Changes in Agent Repos

| Repo | Migrate CI scripts | Migrate run scripts | Repo-specific actions |
|---|---|---|---|
| trader | bump, check-deps, check-doc-hashes | `run_agent.sh` → `aea-helpers run-agent`, `run_service.sh` → `aea-helpers run-service`, `aea-config-replace.py` → `config-mapping.json` | Close PR #874 (port management built into aea-helpers) |
| optimus | bump, check-deps, check-doc-hashes | Same as trader. Delete `run_merkle_api.py` (outdated test mock) | Use `--alias agent` in fetch, standard config-replace |
| IEKit | bump, check-deps, check-doc-hashes | Same as trader | — |
| market-creator | bump, check-deps, check-doc-hashes | Same as trader. Prompt escaping stays as env var set before calling `aea-helpers run-service` | — |
| meme-ooorr | bump, check-deps, check-doc-hashes | Same as trader, `--post-deploy-cmd` for db backup | — |

### Changes in Library Repos (CI scripts only)

| Repo | Migrate | Notes |
|---|---|---|
| funds-manager | bump, check-deps, check-doc-hashes | Delete `scripts/` entirely after |
| genai | bump, check-deps, check-doc-hashes | Delete `scripts/` entirely after |
| kv-store | bump, check-deps, check-doc-hashes | Delete `scripts/` entirely after |

### Keep per-repo (not consolidation candidates)

| Script | Reason |
|---|---|
| `whitelist.py` | Vulture config, inherently repo-specific |
| `propel.py` | Only 2 repos, different deployment targets |
| `config-mapping.json` (per-repo) | Variable mappings for config-replace |
| IEKit ceramic/staking scripts | Domain-specific utilities |
| meme-ooorr analytics/test scripts | Domain-specific utilities |
| mech-predict metadata scripts | Domain-specific utilities |
| mech-agents-fun test scripts | Domain-specific test utilities |
| open-aea framework scripts | Framework-specific CI tooling |
| open-autonomy framework scripts | Framework-specific CI tooling |

---

## Appendix A: `aea-config-replace.py` Core Logic (from trader)

```python
PATH_TO_VAR = {
    "config/ledger_apis/gnosis/address": "GNOSIS_LEDGER_RPC",
    "models/params/args/setup/all_participants": "ALL_PARTICIPANTS",
    # ... 51 entries
}
CONFIG_REGEX = r"\${.*?:(.*)}"

def find_and_replace(config, path, new_value):
    for i, section in enumerate(config):
        value = section
        try:
            for part in path:
                value = value[part]
        except KeyError:
            continue
        sub_dic = config[i]
        for part in path[:-1]:
            sub_dic = sub_dic[part]
        old_str_value = sub_dic[path[-1]]
        match = re.match(CONFIG_REGEX, old_str_value)
        old_var_value = match.groups()[0]
        sub_dic[path[-1]] = old_str_value.replace(old_var_value, new_value)
    return config
```

This handles `${str:value}`, `${list:value}`, `${bool:value}` etc. because it does string replacement within the existing template — the type prefix is preserved.

## Appendix B: `mech` CLI (from mech-server)

```
$ mech --help
CLI to create, deploy and manage Mechs on the Olas Marketplace.

Commands:
  add-tool          Scaffold new mech tools
  prepare-metadata  Generate and publish metadata to IPFS
  run               Run the mech AI agent (--dev for host mode)
  setup             Configure service and deploy metadata
  stop              Stop the mech agent service
  update-metadata   Update metadata hash on-chain
```

`mech run -c gnosis --dev` replaces both `run_agent.sh` and `run_service.sh` for mech repos.
