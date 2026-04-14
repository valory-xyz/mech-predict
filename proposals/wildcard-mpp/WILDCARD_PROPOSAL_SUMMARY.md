# MPP Proposal — Executive Summary

**Date:** 2026-04-13
**Detailed docs:** [Mode Definitions](./MPP_MODE_DEFINITIONS.md) | [Cost Proposal](./MPP_COST_PROPOSAL.md) | [Data Pipeline](./MPP_DATA_PIPELINE.md)

---

## Action Items

### For the Wildcard Team

**a. Add metadata to prediction requests (BLOCKER — before Chrome store submission)**

The extension must include 3 new fields in the prediction request POST body:

| Field | Already Available? | Source in Extension |
|---|---|---|
| `market_url` | Yes | `window.location.href` |
| `condition_id` | Yes | `selectedMarket.conditionId` from Gamma API |
| `market_prob_at_prediction` | Yes | `marketPriceYes` computed in `PredictionPanel.tsx` |

The extension already computes all three values — it's ~5 lines to include them in the POST body. Without these, we cannot compute edge-over-market, resolve outcomes, or feed the benchmark pipeline. Changing the request schema after Chrome store launch requires a new review cycle (days to weeks). Full details: [MPP_DATA_PIPELINE.md](./MPP_DATA_PIPELINE.md).

**b. Update mode pricing**

Current prices don't cover API costs on Quick and Deep — Valory is losing money on every call. We propose **Phase 2 pricing** directly (covers the full dynamic tool pool including better-performing Claude variants):

| Mode | Current | Proposed |
|---|---|---|
| Quick | $0.001 | **$0.015** |
| Deep | $0.01 | **$0.04** |
| Super | $0.05 | **$0.08** |

**Why Phase 2 directly, not Phase 1:** Phase 1 prices ($0.01 / $0.03 / $0.05) cover only the 3 tools Wildcard hardcodes today. Once we enable dynamic tool selection on the server side, the price must cover any tool that may be selected (including Claude Sonnet variants at ~2x cost). Setting Phase 2 prices now avoids a second pricing change in a few months when server-side dynamic selection ships. Full cost breakdown per tool: [MPP_COST_PROPOSAL.md](./MPP_COST_PROPOSAL.md).

**c. Nothing else required from Wildcard team.** All other changes (tool selection logic, persistent storage, benchmark integration) are server-side.

---

### Server-Side Changes

**a. Dynamic tool selection and routing**

Replace the hardcoded one-tool-per-mode mapping with a data-driven selection system:

- Each tool declares a `compute_tier` (quick/deep/super) in IPFS metadata
- Benchmark pipeline publishes a per-platform performance CSV to IPFS with BSS scores per `(tool, category)` cell
- MPP server filters the CSV by `compute_tier == requested_mode`, classifies the question's category via shared `classify_category()`, computes weights using Brier Skill Score + softmax (T=0.5)
- Weighted random selection picks the best tool dynamically; fallback chain handles tool failures

This enables tool rotation without code changes, category-aware routing, and a self-improving loop where production predictions feed back into the benchmark scoring. Full architecture: [MPP_MODE_DEFINITIONS.md](./MPP_MODE_DEFINITIONS.md).

**Tool pools (12 existing polystrat tools across 3 modes):**

| Quick (LLM-only) | Deep (LLM + search) | Super (LLM + search + reasoning) |
|---|---|---|
| `prediction-offline` (GPT-4.1) | `prediction-online` (GPT-4.1) | `prediction-request-reasoning` (GPT-4.1) |
| `claude-prediction-offline` (Claude) | `claude-prediction-online` (Claude) | `prediction-request-reasoning-claude` (Claude) |
| `prediction-offline-sme` (GPT-4o) | `prediction-online-sme` (GPT-4o) | `prediction-request-rag` (GPT-4.1) |
| `gemini-prediction` (Gemini Flash) | `superforcaster` (GPT-4.1) | `prediction-request-rag-claude` (Claude) |

**b. Persistent storage**

This is a **server-side change** — the MPP server (owned by Valory, not the Chrome extension) currently has zero persistent storage. Every prediction vanishes from Redis after 1 hour. No database, no log files, no exports.

Phased approach:

| Phase | What | Why |
|---|---|---|
| **Immediate** | Enable Redis AOF persistence (`--appendonly yes` + volume mount). One-line Docker config change. | Survives server restarts. Minimum viable persistence. |
| **Before launch** | JSONL append log — after each prediction, append one JSON line to a mounted volume. ~15 lines of Python. | Directly readable by the benchmark pipeline. Simplest queryable persistent layer. |
| **Post-launch** | PostgreSQL — `predictions` table with async writes. Add `postgres:16-alpine` + `asyncpg`. | Queryable database for dashboards, outcome tracking, per-category analysis. |

Full storage proposal and schema: [MPP_DATA_PIPELINE.md](./MPP_DATA_PIPELINE.md).

**c. Supporting Layer 3 work in the benchmark pipeline**

- Add `compute_tier` column to the IPFS performance CSV
- Publish per-platform CSVs from benchmark CI
- Implement BSS + softmax selection logic in MPP server (~100 lines of Python)
- Capture tool output metadata currently ignored by the server (`cost_dict`, `prompt_used`, `latency_ms`, derived `category`)

---

## What We're Proposing (Overview)

Three user-facing prediction modes (quick / deep / super), defined by compute intensity, with dynamic tool selection behind each mode driven by benchmark performance data.

| Mode | What Happens | Starting Tool | Break-Even Cost |
|---|---|---|---|
| **Quick** | LLM-only, no external data | `prediction-offline` | $0.008/call |
| **Deep** | LLM + live web search | `prediction-online` | $0.026/call |
| **Super** | LLM + search + embeddings + multi-step reasoning | `prediction-request-reasoning` | $0.045/call |

Each mode has a pool of 4 tools (including Claude and GPT variants). The selection logic picks the best tool dynamically based on benchmark performance.

---

## Decisions Locked

| Decision | Chosen |
|---|---|
| Integration architecture | Option A — MPP reads IPFS CSV directly, no mech-interact dependency |
| Selection algorithm | BSS + softmax, T=0.5 launch default |
| Proposed pricing | Phase 2 directly: $0.015 / $0.04 / $0.08 |
| Extension fields | BLOCKER — must ship before Chrome store submission |
| Storage (immediate) | Redis AOF + longer TTL |
| Storage (end goal) | JSONL + PostgreSQL |
