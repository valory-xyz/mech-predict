# MPP Proposal — Executive Summary

**Date:** 2026-04-13
**Detailed docs:** [Mode Definitions](./MPP_MODE_DEFINITIONS.md) | [Cost Proposal](./MPP_COST_PROPOSAL.md) | [Data Pipeline](./MPP_DATA_PIPELINE.md)

---

## Action Required

**BLOCKER — Wildcard team, before Chrome store submission:**
- Add 3 fields to the prediction request body: `market_url`, `condition_id`, `market_prob_at_prediction`. The extension already computes all three — it's ~5 lines on each side. **If the extension ships without these, adding them later requires a Chrome store review cycle (days to weeks).** See [details below](#urgent-what-the-wildcard-team-needs-to-do-before-chrome-store-submission).

**IMMEDIATE — Wildcard team:**
- Reprice Quick from $0.001 → $0.01 and Deep from $0.01 → $0.03. Both modes are currently priced below API cost. See [Phase 1 pricing](#phase-1-keep-current-hardcoded-prices-fix-the-losses-now).
- Enable Redis AOF persistence (`--appendonly yes` + volume mount). One-line Docker config change. Currently every prediction vanishes after 1 hour with no backup.

**NEXT — mech-predict team:**
- Publish `compute_tier` column in IPFS performance CSV (Layer 3 work)
- Implement BSS + softmax selection logic in MPP server (~100 lines Python)
- Add JSONL append log for persistent prediction storage

---

## What We're Proposing

Three user-facing prediction modes (quick / deep / super), defined by compute intensity, with dynamic tool selection behind each mode driven by benchmark performance data.

| Mode | What Happens | Starting Tool | Break-Even Cost |
|---|---|---|---|
| **Quick** | LLM-only, no external data | `prediction-offline` | $0.008/call |
| **Deep** | LLM + live web search | `prediction-online` | $0.026/call |
| **Super** | LLM + search + embeddings + multi-step reasoning | `prediction-request-reasoning` | $0.045/call |

Each mode has a pool of 4 tools (including Claude and GPT variants). The selection logic picks the best tool dynamically based on benchmark performance.

---

## Tool Selection — How the Right Tool Gets Picked

Today Wildcard hardcodes one tool per mode. We propose replacing this with a dynamic, data-driven selection system:

**How it works:**
1. Each tool declares a `compute_tier` (quick/deep/super) in its IPFS metadata
2. Our benchmark pipeline scores all tools and publishes a per-platform performance CSV to IPFS
3. At request time, the MPP server filters the CSV by `compute_tier == requested_mode`
4. Classifies the question's category (crypto, politics, etc.) via shared `classify_category()`
5. Computes weights using Brier Skill Score + softmax (T=0.5) — better tools get more traffic
6. Weighted random selection from the pool
7. If the selected tool fails, falls back to the next-best tool

**What this enables:**
- **Rotate tools without code changes.** Update IPFS metadata or the CSV — the server picks it up automatically.
- **A/B testing built in.** Multiple tools in a pool get traffic proportional to their performance.
- **Category-aware routing.** A tool that excels on politics questions gets more politics traffic, even if it's weaker on crypto.
- **Self-improving loop.** Production predictions feed the benchmark pipeline → updated scores → better tool selection.

**Tool pools (12 tools across 3 modes):**

| Quick (LLM-only) | Deep (LLM + search) | Super (LLM + search + reasoning) |
|---|---|---|
| `prediction-offline` (GPT-4.1) | `prediction-online` (GPT-4.1) | `prediction-request-reasoning` (GPT-4.1) |
| `claude-prediction-offline` (Claude) | `claude-prediction-online` (Claude) | `prediction-request-reasoning-claude` (Claude) |
| `prediction-offline-sme` (GPT-4o) | `prediction-online-sme` (GPT-4o) | `prediction-request-rag` (GPT-4.1) |
| `gemini-prediction` (Gemini Flash) | `superforcaster` (GPT-4.1) | `prediction-request-rag-claude` (Claude) |

All tools are existing polystrat tools already running on Polymarket with production data.

---

## Pricing — Two Phases

### Phase 1: Keep Current Hardcoded Prices, Fix the Losses (Now)

Wildcard currently hardcodes 3 tools with prices that don't cover API costs on two modes:

| Mode | Current Tool | API Cost | Current Price | Proposed Price |
|---|---|---|---|---|
| Quick | `prediction-offline` | $0.008 | $0.001 (loss) | **$0.01** |
| Deep | `prediction-online` | $0.026 | $0.01 (loss) | **$0.03** |
| Super | `superforcaster` | $0.020 | $0.05 (profitable) | **$0.05** (no change) |

These prices cover the 3 existing hardcoded tools at break-even. No dynamic selection, no Claude variants — just repricing to stop losing money on every Quick and Deep call.

### Phase 2: Dynamic Pool Pricing (Once Selection Logic Ships)

Once the weighted selection logic is live and tools can be rotated dynamically, prices need to cover the most expensive tool in each pool (including Claude Sonnet variants):

| Mode | Break-Even Price | Why Higher |
|---|---|---|
| Quick | **$0.015** | Pool includes `claude-prediction-offline` ($0.014/call) |
| Deep | **$0.04** | Pool includes `claude-prediction-online` ($0.040/call) |
| Super | **$0.08** | Pool includes `prediction-request-reasoning-claude` ($0.074/call) |

This is a real cost increase driven by Claude variants, which have the best benchmark performance but are 1.5-2x more expensive than GPT-4.1. The tradeoff: better predictions cost more.

---

## Urgent: What the Wildcard Team Needs to Do Before Chrome Store Submission

The extension must start sending 3 new fields in the prediction request body:

| Field | Already Available In Extension? | Change Required |
|---|---|---|
| `market_url` | Yes — `window.location.href` | Add to POST body |
| `condition_id` | Yes — `selectedMarket.conditionId` from Gamma API | Add to POST body |
| `market_prob_at_prediction` | Yes — `marketPriceYes` computed from CLOB/Gamma APIs | Add to POST body |

**This is a blocker.** These fields are needed for:
- Edge-over-market calculation (was our prediction better than the market?)
- Outcome resolution (linking predictions to resolved markets via `condition_id`)
- Benchmark pipeline integration (scoring production predictions)

The extension already computes all three values — it's ~5 lines to include them in the request. The server-side Pydantic model needs 3 new optional fields (~3 lines). **If the extension ships without these fields, adding them later requires a Chrome store review cycle (days to weeks).**

---

## Persistent Storage — Phased Approach

**Current state: zero persistent storage.** Every prediction vanishes from Redis after 1 hour. No database, no log files, no exports.

### If Time-Crunched (Minimum Viable)

Increase the Redis TTL from 1 hour to 30 days and enable Redis AOF persistence (one-line Docker config change). This keeps predictions in Redis across restarts and gives us a 30-day window to export data.

```yaml
# docker-compose.yml — one change
redis:
  command: redis-server --appendonly yes
  volumes:
    - redis_data:/data
```

This is not a long-term solution — Redis is an in-memory store and 30 days of predictions will consume RAM. But it buys time.

### End Goal

1. **JSONL append log** — after each prediction, append one JSON line to a Docker-mounted volume. Directly readable by our benchmark pipeline. Simplest persistent layer, ships in ~15 lines of Python.

2. **PostgreSQL** — queryable database with a `predictions` table. Enables dashboards, aggregation, outcome tracking. Add `postgres:16-alpine` to docker-compose + `asyncpg` dependency. Async writes don't block predictions.

Both coexist: JSONL for benchmark pipeline ingestion, PostgreSQL for querying and future features (outcome resolution, per-category analysis, cost tracking).

---

## Decisions Locked

| Decision | Chosen |
|---|---|
| Integration architecture | Option A — MPP reads IPFS CSV directly, no mech-interact dependency |
| Selection algorithm | BSS + softmax, T=0.5 launch default |
| Pricing (Phase 1) | $0.01 / $0.03 / $0.05 for current 3 hardcoded tools |
| Pricing (Phase 2) | $0.015 / $0.04 / $0.08 once dynamic pools ship |
| Extension fields | BLOCKER — must ship before Chrome store submission |
| Storage (immediate) | Redis AOF + longer TTL |
| Storage (end goal) | JSONL + PostgreSQL |
