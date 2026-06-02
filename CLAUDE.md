# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the **Autonolas AI Mechs** repository for the Predict Agent Economy. It contains autonomous agents (mechs) that execute AI tasks, particularly predictions, using the Open Autonomy framework. The mechs operate as decentralized services that listen for requests on-chain, execute AI tools, and deliver results back on-chain.

**Important:** The [mech-quickstart](https://github.com/valory-xyz/mech-quickstart) repo is now the recommended way to run and extend mechs. This repository is the production mech with all available tools.

## Architecture

### Core Components

**Agent Service Architecture:**
- Built on Open Autonomy (v0.21.16) framework which uses ABCI (Application Blockchain Interface)
- Agents run as autonomous services with consensus via Tendermint
- Services are composed of multiple **skills** that define behaviors and state machines
- Skills communicate through FSM (Finite State Machine) rounds

**Key Skills (fetched via `autonomy packages sync`, not checked in):**
- `mech_abci`: Main FSM orchestration for the mech service
- `task_execution`: Handles receiving, executing, and delivering AI tool tasks
- `task_submission_abci`: Manages task submission rounds
- `transaction_settlement_abci`: Handles on-chain transaction delivery
- `contract_subscription`: Listens for on-chain events (Request events from marketplace)
- `registration_abci`, `reset_pause_abci`, `termination_abci`: Service lifecycle management

**Mech Tools (Custom Packages):**
- Tools live in `packages/{author}/customs/{tool_name}/`
- Each tool implements a `run()` function that takes a prompt and API keys, returns results
- Tools are registered via `TOOLS_TO_PACKAGE_HASH` environment variable mapping tool names to IPFS hashes
- Common tool pattern: use `@with_key_rotation` decorator for automatic API key rotation on rate limits

**Operating Modes:**
- **Polling mode**: Periodically reads blockchain for Request events (default: every 30s)
- **Websocket mode**: Subscribes to events via websocket for real-time notifications
- Mode controlled by `USE_POLLING` environment variable

### Agent vs Service

- **Agent** (`packages/valory/agents/mech_predict/`): Single autonomous agent instance
- **Service** (`packages/valory/services/mech_predict/`): Multi-agent service with consensus (typically 4 agents)
- Services require Tendermint for consensus, agents run with embedded Tendermint

### Tool Architecture

Tools follow a standard structure:
```
packages/{author}/customs/{tool_name}/
├── __init__.py
├── {tool_name}.py       # Main implementation with run() function
└── component.yaml       # Tool metadata (description, callable, dependencies)
```

The `run()` function signature:
```python
def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    # Returns: (result, prompt_used, transaction_data, additional_info)
```

## Common Commands

### Setup and Installation

```bash
# Install dependencies
uv sync
source .venv/bin/activate

# Sync packages from Open Autonomy registry
autonomy packages sync --update-packages
```

### Development

```bash
# Format code
make format
# Or: tomte format-code

# Run all code checks (black, isort, flake8, mypy, pylint, darglint)
make code-checks
# Or: tomte check-code

# Security checks (safety, bandit, gitleaks)
make security
# Or: tomte check-security

# Clean build artifacts
make clean
```

### Testing

```bash
# Run unit tests for the current platform (auto-discovered customs + benchmark/tests)
tomte tox -e py3.10-linux        # or -darwin / -win

# Live-tool integration suite (requires API key secrets)
tomte tox -e integration-tests

# Test a specific tool
python scripts/test_tool.py

# Test multiple tools
python scripts/test_tools.py
```

### Package Management

```bash
# Update package hashes after changes
autonomy packages lock

# Check package hashes are correct
tox -e check-hash

# Check package dependencies
tox -e check-packages
# Or: python scripts/check_dependencies.py

# Generate/update IPFS hashes in documentation
tox -e fix-doc-hashes
```

### ABCI/FSM Development

```bash
# Generate ABCI docstrings
tox -e abci-docstrings

# Check ABCI docstrings
tox -e check-abci-docstrings

# Update FSM specifications
make fix-abci-app-specs
# Or: autonomy analyse fsm-specs --update --app-class MechAbciApp --package packages/valory/skills/mech_abci

# Check FSM specifications
tox -e check-abciapp-specs

# Check handler implementations
tox -e check-handlers
```

### Running the Mech

**As a standalone agent:**
```bash
# Setup keys
autonomy generate-key ethereum

# Configure environment
cp .example.env .1env
# Edit .1env with your API keys
source .1env

# Run agent (starts Tendermint automatically)
bash run_agent.sh
```

**As a service:**
```bash
# Generate keys for service
autonomy generate-key ethereum -n 1

# Configure environment
cp .example.env .1env
# Edit .1env with ALL_PARTICIPANTS addresses from keys.json
source .1env

# Run service (builds Docker images)
bash run_service.sh
```

### Before Creating a PR

Run checks in this order:
```bash
make clean
make format         # or: tomte format-code
make code-checks    # or: tomte check-code
make security       # or: tomte check-security

# If you modified AbciApp definitions:
make abci-docstrings

# If you modified packages/:
make generators
make common-checks-1

# Otherwise:
tomte format-copyright --author valory [with exclusions from Makefile]

# After committing:
make common-checks-2
```

## Key Environment Variables

Configure in `.1env` or `.agentenv`:

- `API_KEYS`: JSON dict mapping service names to API key lists (e.g., `{"openai": ["key1", "key2"], "google_api_key": ["key"]}`)
- `TOOLS_TO_PACKAGE_HASH`: Maps tool names to IPFS package hashes
- `TOOLS_TO_PRICING`: Dynamic pricing configuration per tool
- `MECH_TO_CONFIG`: Mech-specific configuration (dynamic pricing, marketplace settings)
- `MECH_MARKETPLACE_ADDRESS`: Smart contract address for the marketplace
- `SERVICE_REGISTRY_ADDRESS`: Service registry contract
- `COMPLEMENTARY_SERVICE_METADATA_ADDRESS`: Metadata contract for mech registration
- `USE_POLLING`: Boolean to enable polling mode (vs websocket)
- `POLLING_INTERVAL`: Seconds between polls (default: 30.0)
- `TASK_DEADLINE`: Max seconds to execute a task (default: 240.0)
- `DEFAULT_CHAIN_ID`: Default blockchain (e.g., "gnosis")
- `{CHAIN}_LEDGER_RPC_{N}`: RPC endpoints per agent per chain

## Important Patterns

### Tool-improvement housekeeping rules

When modifying an existing prediction tool (e.g. in response to a `tool-improvement`-labelled issue), follow these rules. They are scoped to **housekeeping** only — file paths, naming, the `tool_lineage.json` ledger, side effects. They do NOT cover investigation, hypothesis-forming, or whether a change is warranted (that logic lives in the `tool-improvement-agent` pipeline in the `agent-skills` repo).

#### In place vs new version

Pick exactly ONE based on the size of the change:

| Path | When | What you DO NOT do |
| --- | --- | --- |
| **In place** — edit `packages/{author}/customs/<tool>/<tool>.py` | Small bugfix: output clamp, parsing fix, one-line prompt tweak, typo, dead-branch removal. Pre-lint diff ≤ 30 LOC. Does NOT touch the prompt's `system`/`user` template structure or the tool's high-level reasoning flow. | Do NOT touch `tool_lineage.json`. Do NOT rename. Do NOT create a new sibling directory. |
| **New version** — create `packages/{author}/customs/<base>_v<n+1>/<base>_v<n+1>.py` | Significant change: prompt rewrite, mechanism swap, new business logic, new evidence source, different model class. | Do NOT delete or modify the existing `<tool>` source — the previous variant keeps running alongside the new one. |

If unsure: default to **in place** for smaller changes, **new version** for anything that crosses a paragraph boundary in the prompt or rewires the call graph.

#### Naming convention (new-version path)

To compute the new tool name from the `<tool>` named in the issue:

1. Strip a trailing `-v<digits>` from `<tool>` to get `<base>`. Bare names with no suffix are implicit `-v0`.
2. Find the largest `n` such that `<base>-v<n>` exists in either `benchmark/tools.py:TOOL_REGISTRY` keys or `tool_lineage.json` `tools` keys.
3. New name is `<base>-v<n+1>`.

Examples:

| `<tool>` (from issue) | `<base>` | Existing | `n` | New name |
| --- | --- | --- | --- | --- |
| `superforcaster` | `superforcaster` | `superforcaster` | 0 | `superforcaster-v1` |
| `superforcaster-v2` | `superforcaster` | `superforcaster`, `superforcaster-v2` | 2 | `superforcaster-v3` |
| `superforcaster-polymarket-v1` | `superforcaster-polymarket` | `superforcaster-polymarket-v1` | 1 | `superforcaster-polymarket-v2` |

Directory uses `_` (Python module convention); the registered tool name uses `-` (trader-visible convention).

#### The `tool_lineage.json` ledger

Lives at the repo root. The `tools` field is a **dictionary keyed by tool name**, so lookup is O(1) and "does this name already exist?" is a trivial `name in tools` check. Updated **only** on the new-version path (in-place edits leave it untouched).

```json
{
  "version": 1,
  "tools": {
    "superforcaster-v3": {
      "parent": "superforcaster-v2",
      "reason": "<one-line why, e.g. 'prompt rewrite to add low-p_yes evidence bar'>",
      "pr": "<URL of the PR introducing this variant>",
      "deployed": false
    }
  }
}
```

Field semantics:

- The dictionary key is the tool name. Must match `^[a-z][a-z0-9_-]*$` and equal `<base>-v<n+1>` per the naming convention.
- `parent` — the tool named in the issue (the one that triggered this version), or `null` for a tool introduced from scratch. Lets readers walk back the lineage.
- `reason` — one human-readable line. NOT the hypothesis (that goes in the PR body); just the housekeeping reason.
- `pr` — URL of this PR. Use `"PENDING"` as a placeholder if appending before `gh pr create`; replace with the real URL in the same commit.
- `deployed` — defaults to `false`. `false` = variant ships in source but NOT routed to live traders. Promotion to live traffic is a separate human PR on `agent-deployments`; that PR flips this to `true`. The automation never sets `deployed: true`.

Add exactly ONE entry per new variant. Do not modify or remove existing entries.

#### File side effects, by path

**In-place path:**

| File | Action |
| --- | --- |
| `packages/{author}/customs/<tool>/<tool>.py` | edit |
| `packages/{author}/customs/<tool>/component.yaml` | re-fingerprinted by `autonomy packages lock` |
| `packages/packages.json` | CID bumped by `autonomy packages lock` |
| `benchmark/tournament_tools.json[<tool>]` | bump to new CID (value swap; key unchanged) |
| `tool_lineage.json` | **untouched** |
| `benchmark/tools.py:TOOL_REGISTRY` | **untouched** |

**New-version path:**

| File | Action |
| --- | --- |
| `packages/{author}/customs/<tool>/` | **untouched** — previous variant keeps running |
| `packages/{author}/customs/<base>_v<n+1>/<base>_v<n+1>.py` | NEW — copy previous tool's source as starting point, then apply the change |
| `packages/{author}/customs/<base>_v<n+1>/component.yaml` | NEW — generated by `autonomy packages lock` |
| `packages/{author}/customs/<base>_v<n+1>/__init__.py` | NEW — copy from previous tool's `__init__.py` |
| `packages/packages.json` | CID added by `autonomy packages lock` |
| `benchmark/tools.py:TOOL_REGISTRY` | ADD entry mapping `<base>-v<n+1>` → new module path |
| `benchmark/tournament_tools.json` | ADD entry `<base>-v<n+1>` → new CID. Previous entry left in place. |
| `tool_lineage.json` | ADD one entry to `tools` (schema above) |

#### Adding a new tool from scratch (not an update)

If you are adding a tool that has no predecessor (a brand-new tool, not a variant of an existing one), the flow is the same as the new-version path above, except:

- `tool_lineage.json` gets an entry where `parent` is `null`.
- The new tool's name does not need a `-v<N>` suffix on first introduction — `superforcaster` is fine; the suffix only appears when a variant of it is later spawned.

#### Out-of-scope reminders

- Do NOT run `autonomy push-all`. Publishing bytes to IPFS is a human step before merging.
- Do NOT modify `agent-deployments`. Routing a CID to live traffic is a separate human PR.
- Do NOT delete old variants. The previous version keeps running until a human explicitly retires it.

### API Key Management

The `KeyChain` class (in `packages/valory/skills/task_execution/utils/apis.py`) handles:
- Multiple API keys per service
- Automatic rotation on rate limit errors
- Round-robin distribution across agents

### FSM Development

- FSM apps define state transitions through Round classes
- Each Round has an `end_block()` method determining next state
- Consensus is reached when threshold of agents agree (typically 2/3)
- See: [Open Autonomy FSM documentation](https://stack.olas.network/open-autonomy/key_concepts/fsm_app_introduction/)

### Package Versioning

- Packages use semantic versioning
- IPFS hashes ensure immutable package references
- Use `scripts/bump.py` for version bumps
- Package fingerprints track file changes

## Testing Notes

- Tests use pytest with asyncio mode
- Integration tests marked with `@pytest.mark.integration`
- E2E tests marked with `@pytest.mark.e2e`
- Mock API responses to avoid rate limits during testing
- Use `KeyChain` class for API key management in tests

## Dependencies

- Python >=3.10, <3.15
- uv for dependency management
- Docker and Docker Compose for service deployment
- Tendermint 0.34.19 for consensus
- Open Autonomy framework (all packages in `packages/` directory)

## Mints Directory

Contains NFT metadata (JSON and PNG files) for mech tokens - not relevant for development.
