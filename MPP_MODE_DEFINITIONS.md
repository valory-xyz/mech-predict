# MPP Mode Definitions — Proposal

**Status:** Draft
**Date:** 2026-04-13
**Scope:** Defines user-facing prediction modes for the Wildcard Chrome extension (MPP server), tool pool membership, and selection logic.

---

## Table of Contents

1. [Context: What Is MPP and Why Does This Matter](#1-context)
2. [Background: The Existing Tool Selection Pipeline](#2-background)
3. [Mode Definitions: Quick / Deep / Super](#3-mode-definitions)
4. [Dynamic Pool Membership via `compute_tier` Metadata](#4-dynamic-pool-membership)
5. [Selection Logic Within a Pool](#5-selection-logic)
6. [Integration Architecture](#6-integration-architecture)

---

## 1. Context: What Is MPP and Why Does This Matter <a id="1-context"></a>

### The Product

[Wildcard](https://github.com/valory-xyz/wildcard) is a Chrome extension that brings AI-powered predictions to Polymarket. Users see a floating panel on Polymarket event pages, choose a prediction mode (quick, deep, or super), and receive calibrated probability estimates. Payments are handled via MPP (Machine Payments Protocol) — a micropayment channel on the Tempo chain.

### The Problem

The Wildcard team currently hardcodes which mech-predict tools power each mode:

| Mode | Tool (hardcoded) | Price |
|---|---|---|
| quick | `prediction-offline` | $0.001 |
| deep | `prediction-online` | $0.01 |
| super | `superforcaster` | $0.05 |

This is brittle. We (the mech-predict team) own the tools and the benchmark data showing which tools perform best — but we don't control which tool runs behind each mode. If we improve a tool, discover a better one, or need to rotate due to API issues, Wildcard must change their server code.

### The Goal

**Define three user-facing modes with clear semantics, make tool-to-mode assignment dynamic and data-driven, and align the selection logic with the existing tool selection pipeline already designed for polystrat/omenstrat.**

The user sees "quick / deep / super" and pays a fixed price. Behind the label, we control what runs — and we can rotate tools, A/B test, and improve without touching the Wildcard server.

---

## 2. Background: The Existing Tool Selection Pipeline <a id="2-background"></a>

Before defining MPP modes, it's important to understand the tool selection infrastructure that already exists (or is being built) across three repos: **mech-interact**, **trader**, and **mech-predict**. The MPP mode system should extend this pipeline, not reinvent it.

### 2.1 The Three-Layer Pipeline (from [TOOL_SELECTION_SPEC.md](https://github.com/valory-xyz/trader/blob/main/TOOL_SELECTION_SPEC.md))

A comprehensive tool selection spec has been designed spanning three repos:

```
ELIGIBILITY (catalog gates)  →  PERFORMANCE (segment routing)  →  FEEDBACK (benchmarks)
        in mech-interact              in trader / EGreedyPolicy        in mech-predict
```

#### Layer 1 — Catalog Gates (mech-interact)

[mech-interact](https://github.com/valory-xyz/mech-interact) is an Open Autonomy skill that manages interactions with AI mechs on the Autonolas marketplace. Today, every agent sees a **flat pool of all registered tools** from all mechs — prediction tools, image generation tools, broken tools, spam tools. The only filter is a manually-maintained `irrelevant_tools` blocklist.

The [PROPOSALS-tool-filtering.md](https://github.com/valory-xyz/mech-interact/blob/main/PROPOSALS-tool-filtering.md) and [TOOL_SELECTION_SPEC.md](https://github.com/valory-xyz/trader/blob/main/TOOL_SELECTION_SPEC.md) propose four gates that run at pool construction time, filtering tools **before** any selection logic runs:

| Gate | What It Filters | How |
|---|---|---|
| **Cold-start grace** | New mechs with < 20 global requests bypass the reputation gate only (not other gates) | Prevents death spiral where new mechs can't earn reputation |
| **Category** | Mech operators self-declare `category` (e.g., `"prediction"`) in IPFS metadata. Agents set `task_type` to filter. | A prediction agent only sees prediction tools |
| **Capability** | Filter mechs whose payment type the caller cannot satisfy (e.g., NVM-only mechs filtered from NATIVE-paying agents) | Prevents FSM deadlocks from incompatible payment flows |
| **Reputation** | `gate_score = 0.6 * wilson_reliability + 0.4 * liveness`. Mechs below `min_mech_score` (recommended: 0.3) are excluded. | Spam mechs, dead mechs, and unreliable mechs are removed from the pool |

Wilson lower bound (95% confidence) is used instead of Laplace smoothing for the reputation gate because it returns 0.0 for zero observations — a spam mech with no history can't pass.

The `irrelevant_tools` blocklist is kept as a fourth gate (name-level, per-tool) for within-category hard exclusions that shouldn't depend on measured performance.

**Implementation status:** Proposed, not yet merged. See [PROPOSALS-tool-filtering.md](https://github.com/valory-xyz/mech-interact/blob/main/PROPOSALS-tool-filtering.md) for full details.

#### Layer 2 — Category-Aware Tool Selection (trader)

The [trader](https://github.com/valory-xyz/trader) (which powers polystrat and omenstrat) uses an `EGreedyPolicy` to select which tool handles a given market question. Today it's a single-key policy — one accuracy score per tool, no segmentation.

The spec extends this to a **nested `(tool, category)` routing table**:

- Each service points `TOOLS_ACCURACY_HASH` at a per-platform IPFS CSV (Polymarket CSV for polystrat, Omen CSV for omenstrat)
- The CSV has rows per `(tool, category)` with accuracy scores and sample counts
- At selection time: if a specific `(tool, category)` cell has `n >= 30`, use it; otherwise fall back to the tool's aggregate row
- Quarantine (fast-feedback for API failures) stays tool-level, independent of category routing

**Current CSV format (single-key, in production today):**
```
tool,tool_accuracy,total_requests,min,max
superforcaster,72.58,485,2026-01-22,2026-03-17
prediction-request-reasoning,62.33,6605,2026-01-26,2026-03-17
```

**Proposed CSV format (with category column, backward compatible):**
```
tool,category,tool_accuracy,total_requests,min,max
prediction-request-reasoning,,62.33,6605,2026-01-26,2026-03-17
prediction-request-reasoning,politics,71.20,891,2026-01-26,2026-03-17
prediction-request-reasoning,crypto,55.10,402,2026-01-28,2026-03-17
```

An empty `category` cell is the aggregate — exactly what today's CSV already represents.

A shared `classify_category()` function (canonical source: mech-predict's `benchmark/datasets/fetch_production.py`) ensures both repos classify market questions into the same buckets. 14 categories: `business, politics, science, technology, health, entertainment, weather, finance, international, travel, sports, sustainability, curiosities, pets`.

**Implementation status:** Spec complete ([TOOL_SELECTION_SPEC.md](https://github.com/valory-xyz/trader/blob/main/TOOL_SELECTION_SPEC.md)), trader-side policy.py bug fixes merged ([PR #902](https://github.com/valory-xyz/trader/pull/902)), category-aware routing not yet implemented.

#### Layer 3 — Benchmark Evaluation Loop (mech-predict)

[mech-predict](https://github.com/valory-xyz/mech-predict) owns the benchmark pipeline that produces the performance data consumed by Layer 2.

**What the scorer outputs today** (per `benchmark/scorer.py`):

| Metric | Description | Used For |
|---|---|---|
| `brier` | Mean squared error of probability estimates (lower = better) | **Primary ranking metric** |
| `directional_accuracy` | % of predictions where the tool's lean matched the outcome | Diagnostic |
| `reliability` | `valid_outputs / attempted_runs` — hard gate at 80% | Gate (tools below 80% excluded) |
| `brier_skill_score` | `1 - (brier / baseline_brier)` — positive = beats naive predictor | Diagnostic |
| `edge` | `market_brier - tool_brier` — does the tool beat the market? | **System diagnostic only, NOT a ranking signal** |
| `calibration` (ECE, slope, intercept) | How well predicted probabilities match realized frequencies | Diagnostic |

Scores are stratified by: `by_tool`, `by_platform`, `by_category`, `by_horizon`, `by_difficulty`, `by_liquidity`, and cross-breakdowns.

**Key design principle: Brier first.** Edge over market is a *consequence* of accuracy, not a goal. Optimising for edge incentivises contrarianism. Market probability is never fed into tool prompts to avoid anchoring bias.

**What Layer 3 needs to add for the full pipeline:**
1. CSV export step — flatten `scores.json` into the row format Layer 2 consumes
2. IPFS publish step — pin per-platform CSVs from CI
3. `by_tool_category` cross-breakdown (already partially exists)

**Implementation status:** Scorer and benchmark pipeline are production-ready. CSV export and IPFS publish are net-new work.

### 2.2 How polystrat/omenstrat Use This Today

```
polystrat (Polymarket trader service)
    │
    ├─ mech-interact: discovers mechs, filters by irrelevant_tools blocklist
    │  └─ returns flat pool of tool names
    │
    ├─ trader (EGreedyPolicy): reads IPFS accuracy CSV (TOOLS_ACCURACY_HASH)
    │  └─ selects best tool from pool (epsilon-exploration 25%, exploit 75%)
    │
    └─ mech-interact: routes selected tool to best-ranked mech for execution
```

The key insight: **mech-interact handles mech discovery and filtering; the trader handles tool selection; mech-predict produces the performance data.** Each layer has a clear owner.

### 2.3 What This Means for MPP

The MPP server (Wildcard) is a **new consumer** of this pipeline. It doesn't use mech-interact's FSM (it calls tool `run()` functions directly), but it needs the same selection intelligence — specifically, the performance data from Layer 3.

The question is: how much of the existing pipeline does MPP reuse vs. build its own?

---

## 3. Mode Definitions: Quick / Deep / Super <a id="3-mode-definitions"></a>

Each mode is defined by its **compute budget** (what the tool is allowed to do) and contains a **pool of eligible tools**. Within a pool, the tool with the best benchmark performance gets the most traffic.

### 3.1 Quick — LLM-Only, No External Data

**Compute budget:** 1–2 LLM calls. No web search, no embeddings, no external API calls beyond the LLM provider.

**User expectation:** Fast gut-check. The LLM uses only its training knowledge. No real-time information. Seconds, not minutes.

**Tool pool:**

| Tool | Package | Model | What It Does |
|---|---|---|---|
| `prediction-offline` | `valory/prediction_request` | GPT-4.1 | Superforecaster-style prompt, LLM knowledge only |
| `claude-prediction-offline` | `valory/prediction_request` | Claude Sonnet | Same tool, Claude model variant |
| `prediction-offline-sme` | `nickcom007/prediction_request_sme` | GPT-4o | Subject matter expert role-prompting, no search |
| `gemini-prediction` | `dvilela/gemini_prediction` | Gemini 2.0 Flash | Single Gemini call, minimal prompt |

**Starting default** (used until benchmark data drives selection): `prediction-offline`

**Benchmark evidence:** `prediction-offline` has a Brier score of 0.2303 with 61% accuracy on the production dataset (n=129). Among offline tools, it's the strongest performer.

### 3.2 Deep — LLM + Live Web Search

**Compute budget:** 1–2 LLM calls + web search API calls + optional page scraping. The tool fetches current information from the web before making a prediction.

**User expectation:** Informed prediction backed by real-time data. Slower than Quick, but the prediction accounts for recent events and news.

**Tool pool:**

| Tool | Package | Model | What It Does |
|---|---|---|---|
| `prediction-online` | `valory/prediction_request` | GPT-4.1 | LLM generates search queries, Serper/Google search, scrapes top pages, LLM predicts with evidence |
| `claude-prediction-online` | `valory/prediction_request` | Claude Sonnet | Same tool, Claude model variant |
| `prediction-online-sme` | `nickcom007/prediction_request_sme` | GPT-4o | SME role-prompting with web search |
| `superforcaster` | `valory/superforcaster` | GPT-4.1 | Serper search snippets (no full page scrape) + calibrated superforecaster prompt with structured reasoning |

**Starting default:** `prediction-online` (with v5 improved prompt — holdout Brier 0.2211, a 24.8% improvement over baseline)

**Benchmark evidence:** `prediction-online` after prompt improvements shows strong holdout performance (Brier 0.2211, 76.7% accuracy on 60-market holdout). `superforcaster` has the highest production volume (24,519 predictions on Omen) but weaker Brier (0.3444 pre-improvement, ~0.28 post-v4 prompt fix).

### 3.3 Super — LLM + Search + Structured Reasoning / RAG

**Compute budget:** 3+ LLM calls + web search + embeddings (FAISS vector retrieval) or multi-step reasoning chains. The tool performs deep analysis: search, retrieve, embed, reason through evidence step-by-step, then predict.

**User expectation:** Most thorough analysis. Highest confidence predictions. The tool doesn't just search — it reasons through the evidence in multiple passes.

**Tool pool:**

| Tool | Package | Model | What It Does |
|---|---|---|---|
| `prediction-request-reasoning` | `napthaai/prediction_request_reasoning` | GPT-4.1 | 3-stage: search → embed → reason → predict. Uses FAISS vector retrieval + explicit reasoning chain |
| `prediction-request-reasoning-claude` | `napthaai/prediction_request_reasoning` | Claude Sonnet | Same tool, Claude model variant |
| `prediction-request-rag` | `napthaai/prediction_request_rag` | GPT-4.1 | Search + RAG (FAISS embeddings, semantic retrieval of relevant passages) → predict |
| `prediction-request-rag-claude` | `napthaai/prediction_request_rag` | Claude Sonnet | Same tool, Claude model variant |

**Starting default:** `prediction-request-reasoning`

**Benchmark evidence:** `prediction-request-reasoning` is the **only tool with positive Brier Skill Score on Polymarket** (+0.12 BSS, meaning it beats the naive base-rate predictor). Brier of 0.2568 with 68.9% accuracy (n=349). After prompt improvements: training Brier 0.1985 (15.3% improvement), holdout Brier 0.2473 (8.1% improvement), and 85% reduction in overconfident-wrong predictions.

### 3.4 Mode Assignment Rationale

The boundary between modes is defined by **data source and compute intensity**, not by measured quality. Quality drives *which tool within a pool gets the most traffic* (via the selection logic in Section 5), but the pool boundary is structural:

| | Quick | Deep | Super |
|---|---|---|---|
| **LLM calls** | 1–2 | 1–2 | 3+ |
| **Web search** | None | Yes | Yes |
| **Embeddings / RAG** | None | None | Yes (FAISS) |
| **Reasoning chains** | None | None | Yes (multi-step) |
| **Real-time data** | No | Yes | Yes |

A tool that does web search will never be in the Quick pool regardless of how fast it is. A tool that uses embeddings/RAG will never be in the Deep pool. This keeps the mode semantics clean and predictable for users.

---

## 4. Dynamic Pool Membership via `compute_tier` Metadata <a id="4-dynamic-pool-membership"></a>

### 4.1 The Problem With Static Pools

If mode-to-tool mappings are hardcoded in the MPP server config, every change requires a server-side code or config update:
- Adding a new tool to a pool
- Moving a tool between pools (e.g., a simplified version of `superforcaster` that drops search → moves from Deep to Quick)
- Removing a broken tool

### 4.2 Solution: Extend IPFS Tool Metadata

The Layer 1 spec already introduces `category` as a self-declared field in IPFS tool metadata. We extend this with a `compute_tier` field:

```json
{
  "tools": [
    {
      "name": "prediction-online",
      "category": "prediction",
      "compute_tier": "deep"
    },
    {
      "name": "prediction-request-reasoning",
      "category": "prediction",
      "compute_tier": "super"
    }
  ]
}
```

**Vocabulary:**

| Value | Meaning |
|---|---|
| `quick` | LLM-only, no external data fetching |
| `deep` | LLM + web search |
| `super` | LLM + search + embeddings/reasoning chains |

### 4.3 Extend the Performance CSV

The Layer 2 spec already proposes adding a `category` column to the IPFS performance CSV. We add `compute_tier` as another optional column:

```
tool,category,compute_tier,tool_accuracy,total_requests,min,max
prediction-offline,,quick,64.67,184,2026-02-04,2026-03-16
prediction-offline,politics,quick,68.20,42,2026-02-10,2026-03-16
prediction-online,,deep,66.30,95,2026-01-28,2026-03-17
prediction-online,politics,deep,71.20,891,2026-01-26,2026-03-17
prediction-request-reasoning,,super,68.90,349,2026-01-26,2026-03-17
prediction-request-reasoning,crypto,super,55.10,402,2026-01-28,2026-03-17
superforcaster,,deep,72.58,485,2026-01-22,2026-03-17
```

**Backward compatibility:** CSVs without a `compute_tier` column are still valid. Consumers that don't need compute_tier (polystrat, omenstrat) ignore it. The MPP server filters by it.

### 4.4 How Pool Membership Becomes Dynamic

With `compute_tier` in the CSV:

1. **Adding a tool to a pool:** Declare `compute_tier` in the tool's IPFS metadata → benchmark pipeline picks it up → next CSV publish includes it with scores → MPP server sees it in the filtered pool automatically.
2. **Moving a tool between pools:** Update the tool's `compute_tier` in IPFS metadata → propagates through the pipeline.
3. **Removing a tool:** Remove from IPFS metadata or let its reliability drop below the 80% gate → benchmark excludes it → disappears from the CSV.

No MPP server code change needed for any of these operations.

---

## 5. Selection Logic Within a Pool <a id="5-selection-logic"></a>

### 5.1 Overview

At request time, the MPP server receives a mode (`quick`, `deep`, or `super`) and a question with two outcomes. The selection logic:

```
1. Filter the performance CSV to tools where compute_tier == requested_mode
2. Classify the question's category via classify_category(question)
3. For each tool in the filtered set:
   a. If a specific (tool, category) cell exists with n >= 30 → use that accuracy
   b. Else → use the tool's aggregate row (tool, "") accuracy
4. Convert accuracies to weights: weight(tool) = accuracy / sum(all accuracies)
5. Weighted random selection
6. Execute the selected tool
7. If tool fails → fallback to next-highest-weight tool in pool
8. Log: {request_id, mode, selected_tool, category, fallback_used, latency, success}
```

### 5.2 Category-Aware Routing

Different tools perform differently on different question categories. For example, `prediction-request-reasoning` has +0.12 BSS on Polymarket overall but only +0.08 on crypto and +0.15 on politics. Category-aware routing lets the MPP server pick the best tool *for the specific type of question being asked*.

The MPP server uses the same `classify_category()` function from mech-predict's benchmark pipeline to classify the incoming question. This ensures consistency — the category a question is classified into at prediction time matches the category used during benchmarking.

**Fallback chain (matches Layer 2 spec):**

```
Specific (tool, category) cell with n >= 30
    ↓ if not available
Aggregate (tool, "") row
    ↓ if no CSV data at all
Starting default for the mode
```

### 5.3 Weight Calculation

Weights are derived from the CSV's accuracy metric (or Brier score once the CSV schema evolves):

```python
def compute_weights(csv_rows: list, mode: str, category: str | None) -> dict[str, float]:
    """Compute selection weights for tools in a given mode."""
    pool = [row for row in csv_rows if row.compute_tier == mode]

    scores = {}
    for tool_name in {row.tool for row in pool}:
        # Prefer category-specific cell if available
        specific = find_cell(pool, tool_name, category, min_n=30)
        aggregate = find_cell(pool, tool_name, category="", min_n=0)
        cell = specific or aggregate
        if cell:
            scores[tool_name] = cell.tool_accuracy

    if not scores:
        return {}  # fallback to starting default

    total = sum(scores.values())
    return {tool: score / total for tool, score in scores.items()}
```

### 5.4 Weight Refresh

The MPP server fetches the performance CSV from IPFS:
- **On startup:** read `TOOLS_ACCURACY_HASH` env var, fetch CSV, cache in memory
- **Periodically:** re-fetch every N hours (matching the benchmark pipeline's publish cadence — e.g., every 6 hours)
- **No Redis needed for weights** — in-memory cache with periodic refresh from IPFS is sufficient

### 5.5 Fallback on Tool Failure

If the selected tool fails at runtime (API error, timeout, invalid response):

1. Remove the failed tool from the candidate set for this request
2. Select the next-highest-weight tool from the remaining pool
3. If all tools in the pool fail → return error to user
4. Log the fallback event for diagnostics

This is analogous to quarantine in the trader spec, but per-request rather than persistent — the MPP server doesn't maintain cross-request quarantine state. A tool that failed once may succeed on the next request (transient API error). Persistent quarantine can be added later if failure patterns warrant it.

### 5.6 Logging Requirements

Every prediction request must log:

| Field | Purpose |
|---|---|
| `request_id` | Trace the request end-to-end |
| `mode` | Which mode the user selected |
| `selected_tool` | Which tool was picked by the selection logic |
| `category` | Classified category of the question |
| `fallback_used` | Whether the primary tool failed and a fallback was used |
| `fallback_tool` | Which tool was used as fallback (if any) |
| `latency_ms` | End-to-end execution time |
| `success` | Whether the tool returned a valid prediction |
| `tool_weights` | The weight distribution at selection time (for debugging) |

This data feeds back into the benchmark pipeline (Layer 3) and is essential for measuring real production performance of the selection logic.

---

## 6. Integration Architecture <a id="6-integration-architecture"></a>

The MPP server currently calls tool `run()` functions directly — it imports them from vendored mech-predict packages. It does not use mech-interact's FSM or mech marketplace discovery. The question is: should it?

### Option A: MPP Server Consumes Layer 3 Directly (No mech-interact)

```
mech-predict (Layer 3)              MPP server (Wildcard)
┌────────────────────────┐     ┌─────────────────────────────────┐
│ Benchmark scorer       │     │ Reads IPFS CSV (TOOLS_ACCURACY_ │
│ → scores.json          │     │   HASH env var)                 │
│ → per-platform CSV     │────▶│ Filters by compute_tier = mode  │
│ → pins to IPFS         │     │ Weighted random selection       │
│                        │     │ Calls run() directly on tool    │
└────────────────────────┘     │ Fallback on failure             │
                               └─────────────────────────────────┘
```

**Advantages:**
- Simplest architecture. MPP server is already calling `run()` directly; adding CSV-based selection is a small change.
- No dependency on mech-interact's FSM, Tendermint consensus, or mech marketplace discovery.
- Fastest to ship. Selection logic is ~100 lines of Python in the MPP server.
- The MPP server runs our own tools — it doesn't need to discover mechs from the open marketplace.

**Disadvantages:**
- MPP server maintains its own selection logic, separate from the trader's `EGreedyPolicy`. Two selection implementations to maintain.
- If Layer 1 gates (reputation, category, capability) later become relevant for MPP, they'd need to be reimplemented.
- No mech-level filtering — the MPP server trusts that all tools in its vendored packages are healthy. This is fine today (it runs its own tools), but doesn't scale if MPP ever routes to third-party mechs.

### Option B: MPP Server Uses mech-interact for Selection

```
mech-interact (Layer 1)         MPP server (Wildcard)
┌────────────────────┐     ┌──────────────────────────────┐
│ Catalog gates      │     │ Calls mech-interact API:     │
│ (category,         │────▶│   get_tools(task_type=       │
│  reputation,       │     │     "prediction",            │
│  capability)       │     │     compute_tier="deep")     │
└────────────────────┘     │                              │
                           │ Reads IPFS CSV for weights   │
mech-predict (Layer 3)     │ Weighted random selection    │
┌────────────────────┐     │ Calls run() on selected tool │
│ Performance CSV    │────▶│                              │
└────────────────────┘     └──────────────────────────────┘
```

**Advantages:**
- Single source of truth for tool eligibility. The same gates that protect polystrat/omenstrat protect MPP.
- If a mech goes down, the reputation gate catches it for all consumers simultaneously.
- If MPP later routes to third-party mechs (not just our own vendored tools), the infrastructure is already there.
- Consistency — one pipeline, multiple consumers.

**Disadvantages:**
- mech-interact is an Open Autonomy skill designed for FSM-based agents. Using it from a FastAPI server requires either: (a) extracting the gate logic into a standalone library, or (b) running mech-interact as a sidecar service. Both are non-trivial.
- Adds operational complexity — the MPP server now depends on mech-interact's subgraph queries, IPFS fetches, and gate configuration.
- Over-engineered for the current use case. The MPP server runs 12 known tools from 4 packages. It doesn't need marketplace discovery or mech reputation scoring.
- Slower to ship. Requires mech-interact changes before MPP can adopt.

### Recommendation

**Start with Option A. Migrate to Option B if/when MPP needs to route to third-party mechs.**

Option A is sufficient for the current architecture (MPP runs its own vendored tools) and can ship immediately. The selection logic is simple enough that maintaining it separately from the trader's `EGreedyPolicy` is acceptable — especially since the MPP server has a different routing dimension (`compute_tier`) that the trader doesn't use.

The migration path to Option B is clean: extract mech-interact's gate logic into a reusable library that both the FSM skill and the MPP server can import. This refactor is worth doing when mech-interact's gates are production-ready (currently proposed, not merged).

