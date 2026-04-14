# MPP Cost Proposal — Mode Pricing for Wildcard

**Status:** Draft
**Date:** 2026-04-13
**Companion doc:** [MPP_MODE_DEFINITIONS.md](./MPP_MODE_DEFINITIONS.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State: Wildcard's Pricing vs. Our Actual Costs](#2-current-state)
3. [API Pricing Reference (Sources)](#3-api-pricing)
4. [Per-Tool Cost Breakdown With Proof](#4-per-tool-cost)
5. [Per-Mode Cost Range](#5-per-mode-cost)
6. [Proposed Pricing: Two Scenarios](#6-proposed-pricing)

---

## 1. Executive Summary <a id="1-executive-summary"></a>

Valory runs the MPP server and pays for all API keys (OpenAI, Anthropic, Google, Serper). The current Wildcard pricing ($0.001 / $0.01 / $0.05) was set before understanding real API costs.

**Two of the three modes are priced below break-even.** This document proves it with traced API calls from source code and published provider rates, then proposes break-even pricing under two scenarios:

1. **Scenario A (Current hardcoded tools):** Wildcard keeps using only `prediction-offline`, `prediction-online`, and `superforcaster` — the 3 tools hardcoded today.
2. **Scenario B (Dynamic tool pools) — CHOSEN:** Wildcard adopts the pool-based selection from [MPP_MODE_DEFINITIONS.md](./MPP_MODE_DEFINITIONS.md), where any tool in the pool may be selected. Price must cover the most expensive tool that could run.

> **Decision:** We propose **Scenario B pricing ($0.015 / $0.04 / $0.08)** directly to Wildcard. Rationale: setting Phase 2 prices now avoids a second pricing change when server-side dynamic selection ships. Scenario A is retained below as reference for understanding the cost floor of current hardcoded tools.

---

## 2. Current State <a id="2-current-state"></a>

**Source:** Wildcard server — `server/src/tools/registry.py` (lines 39-70):

```python
MODE_CONFIG = {
    "quick": {"tool": "prediction-offline", "price": 1000, "price_display": "0.001"},
    "deep":  {"tool": "prediction-online",  "price": 10000, "price_display": "0.01"},
    "super": {"tool": "superforcaster",     "price": 50000, "price_display": "0.05"},
}
```

Prices are in USDC base units (6 decimals). `1000` = $0.001.

---

## 3. API Pricing Reference (Sources) <a id="3-api-pricing"></a>

All prices below are published rates as of April 2026. These are the rates Valory pays when the MPP server makes API calls.

### 3.1 LLM Providers

| Model | Input ($/1M tokens) | Output ($/1M tokens) | Per 1K tokens (in / out) | Source |
|---|---|---|---|---|
| **gpt-4.1-2025-04-14** | $2.00 | $8.00 | $0.002 / $0.008 | [OpenAI API Pricing](https://openai.com/api/pricing/) |
| **gpt-4o-2024-08-06** | $2.50 | $10.00 | $0.0025 / $0.010 | [OpenAI API Pricing](https://openai.com/api/pricing/) |
| **claude-4-sonnet-20250514** | $3.00 | $15.00 | $0.003 / $0.015 | [Anthropic Models & Pricing](https://docs.anthropic.com/en/docs/about-claude/models) — verified: $3/MTok input, $15/MTok output |
| **claude-3-haiku-20240307** | $0.25 | $1.25 | $0.00025 / $0.00125 | [Anthropic Models & Pricing](https://docs.anthropic.com/en/docs/about-claude/models) — verified: $0.25/MTok input, $1.25/MTok output |
| **gemini-2.0-flash** | $0.10 | $0.40 | $0.0001 / $0.0004 | [Google AI Pricing](https://ai.google.dev/pricing) — verified: $0.10/MTok input, $0.40/MTok output. **Note: deprecated June 1, 2026.** |

**Cross-reference:** These rates match our codebase's `TOKEN_PRICES` dict at `packages/valory/skills/task_execution/utils/benchmarks.py` (lines 30-51).

### 3.2 Embedding API

| Model | Input ($/1M tokens) | Per 1K tokens | Source |
|---|---|---|---|
| **text-embedding-3-large** (3072 dims) | $0.13 | $0.00013 | [OpenAI API Pricing](https://openai.com/api/pricing/) |

### 3.3 Search APIs

| Service | Estimated Cost Per Query | Source |
|---|---|---|
| **Serper API** | ~$0.001 (at scale, $50/50K queries) | [Serper Pricing](https://serper.dev) — free tier: 2,500 queries; paid plans vary |
| **Google Custom Search API** | ~$0.005 ($5/1K queries after free tier) | [Google Custom Search Pricing](https://developers.google.com/custom-search/v1/overview#pricing) |

**Note on Serper:** Exact per-query pricing depends on plan tier. $0.001/query is an estimate based on the $50/50K queries tier. Actual cost may be higher at lower volumes or lower at higher volumes.

---

## 4. Per-Tool Cost Breakdown With Proof <a id="4-per-tool-cost"></a>

For each tool, we trace the exact API calls from the source code and calculate cost using the rates above. Token estimates come from measuring prompt templates in the codebase plus typical response sizes.

### 4.1 prediction-offline (GPT-4.1) — Quick pool

**Source:** `packages/valory/customs/prediction_request/prediction_request.py`

| Step | API | Tokens (in) | Tokens (out) | Evidence |
|---|---|---|---|---|
| Prediction | OpenAI GPT-4.1 | ~2,200 | ~500 | `OFFLINE_PREDICTION_PROMPT` (~1,600 tokens, line 526) + system prompt (~20 tokens, line 593) + user prompt (~580). Output: JSON with p_yes/p_no/confidence/info_utility. |

```
Input:  2,200 × $0.002/1K = $0.0044
Output:   500 × $0.008/1K = $0.0040
────────────────────────────────────
TOTAL:                      $0.0084
```

### 4.2 claude-prediction-offline (Claude Sonnet) — Quick pool

Same flow as 4.1, Claude pricing.

```
Input:  2,200 × $0.003/1K = $0.0066
Output:   500 × $0.015/1K = $0.0075
────────────────────────────────────
TOTAL:                      $0.0141
```

### 4.3 prediction-offline-sme (GPT-4o) — Quick pool

**Source:** `packages/nickcom007/customs/prediction_request_sme/prediction_request_sme.py`
Uses GPT-4o (line 93: `TOOL_TO_ENGINE["prediction-offline-sme"] = "gpt-4o-2024-08-06"`).

```
Input:  2,500 × $0.0025/1K = $0.00625
Output:   500 × $0.010/1K  = $0.00500
──────────────────────────────────────
TOTAL:                        $0.01125
```

### 4.4 gemini-prediction (Gemini 2.0 Flash) — Quick pool

**Source:** `packages/dvilela/customs/gemini_prediction/gemini_prediction.py`
Single Gemini call (line 188). No search, no embeddings.

```
Input:  2,000 × $0.0001/1K = $0.0002
Output:   500 × $0.0004/1K = $0.0002
─────────────────────────────────────
TOTAL:                       $0.0004
```

**Caveat:** No Polymarket benchmark data. Quality is unproven. Model deprecated June 1, 2026.

### 4.5 prediction-online (GPT-4.1) — Deep pool

**Source:** `packages/valory/customs/prediction_request/prediction_request.py`

| Step | API | Tokens (in) | Tokens (out) | Evidence |
|---|---|---|---|---|
| 1. Query generation | OpenAI GPT-4.1 | ~700 | ~200 | `URL_QUERY_PROMPT` (~600 tokens, line 563). `DEFAULT_NUM_QUERIES = 2` (line 160). |
| 2. Search | Serper API | — | — | 2 queries. |
| 3. URL scraping | HTTP GET | — | — | Up to 6 URLs (3/query, line 467). `DEFAULT_NUM_WORDS = 300`/URL (line 470). |
| 4. Prediction | OpenAI GPT-4.1 | ~8,500 | ~500 | `PREDICTION_PROMPT` (~1,500 tokens, line 482) + scraped content (~6,000 tokens) + user prompt. Truncated by `adjust_additional_information` (line 1138). |

```
LLM Call 1:   700 in × $0.002/1K = $0.0014
              200 out × $0.008/1K = $0.0016
Serper:       2 queries × $0.001  = $0.0020
LLM Call 2: 8,500 in × $0.002/1K = $0.0170
              500 out × $0.008/1K = $0.0040
────────────────────────────────────────────
TOTAL:                              $0.0260
```

### 4.6 claude-prediction-online (Claude Sonnet) — Deep pool

Same flow as 4.5, Claude pricing.

```
LLM Call 1:   700 in × $0.003/1K = $0.0021
              200 out × $0.015/1K = $0.0030
Serper:       2 queries × $0.001  = $0.0020
LLM Call 2: 8,500 in × $0.003/1K = $0.0255
              500 out × $0.015/1K = $0.0075
────────────────────────────────────────────
TOTAL:                              $0.0401
```

### 4.7 superforcaster (GPT-4.1) — Deep pool

**Source:** `packages/valory/customs/superforcaster/superforcaster.py`

| Step | API | Tokens (in) | Tokens (out) | Evidence |
|---|---|---|---|---|
| 1. Search | Serper API | — | — | 1 query (line 436). Returns organic + "People Also Ask". |
| 2. Prediction | OpenAI GPT-4.1 | ~5,500 | ~1,000 | `PREDICTION_PROMPT` (~2,000 tokens, line 179) + formatted snippets (~1,500 tokens). Structured reasoning output (facts/yes/no/thinking/answer) produces ~1,000 tokens typical, up to 2,000-3,000. |

```
Serper:       1 query × $0.001    = $0.0010
LLM Call:   5,500 in × $0.002/1K = $0.0110
            1,000 out × $0.008/1K = $0.0080
────────────────────────────────────────────
TOTAL:                              $0.0200
```

**Upper bound:** With longer reasoning output (~2,500 tokens): $0.032.

### 4.8 prediction-request-reasoning (GPT-4.1) — Super pool

**Source:** `packages/napthaai/customs/prediction_request_reasoning/prediction_request_reasoning.py`

| Step | API | Tokens (in) | Tokens (out) | Evidence |
|---|---|---|---|---|
| 1. Query gen | OpenAI GPT-4.1 | ~700 | ~200 | `URL_QUERY_PROMPT`. `DEFAULT_NUM_QUERIES = 2` (line 311). |
| 2. Search | Serper API | — | — | 2 queries. |
| 3. URL scraping | HTTP GET | — | — | Up to 6 URLs. |
| 4. Embedding | OpenAI text-embedding-3-large | ~8,000 | — | Chunks at 1,800 tokens (line 315). `EMBEDDING_MODEL = "text-embedding-3-large"` (line 314). |
| 5. Reasoning | OpenAI GPT-4.1 | ~6,500 | ~2,000 | `REASONING_PROMPT` (~400 tokens, line 399) + RAG-retrieved chunks. |
| 6. Prediction | OpenAI GPT-4.1 | ~3,000 | ~500 | `PREDICTION_PROMPT` (~500 tokens, line 365) + reasoning output. |

```
LLM Call 1:     700 in × $0.002/1K   = $0.0014
                200 out × $0.008/1K   = $0.0016
Serper:         2 queries × $0.001    = $0.0020
Embedding:    8,000 in × $0.00013/1K  = $0.0010
LLM Call 2:   6,500 in × $0.002/1K   = $0.0130
              2,000 out × $0.008/1K   = $0.0160
LLM Call 3:   3,000 in × $0.002/1K   = $0.0060
                500 out × $0.008/1K   = $0.0040
──────────────────────────────────────────────────
TOTAL:                                  $0.0450
```

**Upper bound:** With max reasoning output (4,096 tokens) and more embedded content: ~$0.068.

### 4.9 prediction-request-reasoning-claude (Claude Sonnet) — Super pool

Same flow as 4.8, Claude pricing for LLM calls, OpenAI for embeddings.

```
LLM Call 1:     700 in × $0.003/1K   = $0.0021
                200 out × $0.015/1K   = $0.0030
Serper:         2 queries × $0.001    = $0.0020
Embedding:    8,000 in × $0.00013/1K  = $0.0010
LLM Call 2:   6,500 in × $0.003/1K   = $0.0195
              2,000 out × $0.015/1K   = $0.0300
LLM Call 3:   3,000 in × $0.003/1K   = $0.0090
                500 out × $0.015/1K   = $0.0075
──────────────────────────────────────────────────
TOTAL:                                  $0.0741
```

### 4.10 prediction-request-rag (GPT-4.1) — Super pool

**Source:** `packages/napthaai/customs/prediction_request_rag/prediction_request_rag.py`

| Step | API | Tokens (in) | Tokens (out) | Evidence |
|---|---|---|---|---|
| 1. Query gen | OpenAI GPT-4.1 | ~700 | ~200 | Same as online tools. |
| 2. Search | Serper API | — | — | 2 queries. |
| 3. URL scraping | HTTP GET | — | — | Up to 6 URLs. |
| 4. Embedding | OpenAI text-embedding-3-large | ~8,000 | — | Chunks at 1,800 tokens (line 308). `NUM_NEIGHBORS = 4` (line 312). |
| 5. Prediction | OpenAI GPT-4.1 | ~5,500 | ~500 | `PREDICTION_PROMPT` (~800 tokens, line 325) + RAG chunks (~4,000 tokens). |

```
LLM Call 1:     700 in × $0.002/1K   = $0.0014
                200 out × $0.008/1K   = $0.0016
Serper:         2 queries × $0.001    = $0.0020
Embedding:    8,000 in × $0.00013/1K  = $0.0010
LLM Call 2:   5,500 in × $0.002/1K   = $0.0110
                500 out × $0.008/1K   = $0.0040
──────────────────────────────────────────────────
TOTAL:                                  $0.0210
```

### 4.11 prediction-request-rag-claude (Claude Sonnet) — Super pool

Same flow as 4.10, Claude pricing for LLM.

```
LLM Call 1:     700 in × $0.003/1K   = $0.0021
                200 out × $0.015/1K   = $0.0030
Serper:         2 queries × $0.001    = $0.0020
Embedding:    8,000 in × $0.00013/1K  = $0.0010
LLM Call 2:   5,500 in × $0.003/1K   = $0.0165
                500 out × $0.015/1K   = $0.0075
──────────────────────────────────────────────────
TOTAL:                                  $0.0321
```

---

## 5. Per-Mode Cost Range <a id="5-per-mode-cost"></a>

### 5.1 Quick Mode — All Tools

| Tool | Est. Cost | Notes |
|---|---|---|
| `gemini-prediction` | **$0.0004** | No benchmark data. Model deprecated June 2026. |
| `prediction-offline` (GPT-4.1) | **$0.008** | Brier 0.2303, accuracy 61% (n=129). Strongest offline tool. |
| `prediction-offline-sme` (GPT-4o) | **$0.011** | Brier 0.5536, accuracy 33% (n=10). Unreliable tiny sample. |
| `claude-prediction-offline` (Claude) | **$0.014** | Brier 0.3126, accuracy 56% (n=77). |

**Range:** $0.0004 – $0.014

### 5.2 Deep Mode — All Tools

| Tool | Est. Cost | Notes |
|---|---|---|
| `superforcaster` (GPT-4.1) | **$0.020** | Brier 0.3444 pre-improvement, ~0.28 post-v4 (n=904). |
| `prediction-online` (GPT-4.1) | **$0.026** | Holdout Brier 0.2211 post-v5 (n=60). |
| `prediction-online-sme` (GPT-4o) | **~$0.030** | No Polymarket benchmark data. |
| `claude-prediction-online` (Claude) | **$0.040** | Brier 0.3565, accuracy 50% (n=12). Tiny sample. |

**Range:** $0.020 – $0.040

### 5.3 Super Mode — All Tools

| Tool | Est. Cost | Notes |
|---|---|---|
| `prediction-request-rag` (GPT-4.1) | **$0.021** | Brier 0.2507, BSS +0.08 on Polymarket (n=45). |
| `prediction-request-rag-claude` (Claude) | **$0.032** | Brier 0.2821, accuracy 59% (n=17). |
| `prediction-request-reasoning` (GPT-4.1) | **$0.045** | Brier 0.2568, BSS +0.12 on Polymarket (n=349). Best Polymarket BSS. |
| `prediction-request-reasoning-claude` (Claude) | **$0.074** | Brier 0.2058, accuracy 71% (n=181). Best overall Brier. |

**Range:** $0.021 – $0.074

---

## 6. Proposed Pricing: Two Scenarios <a id="6-proposed-pricing"></a>

Both scenarios target **break-even** — covering API costs without adding margin. Infrastructure costs (server, Redis, monitoring) are not included and would need to be accounted for separately.

### Scenario A: Current Hardcoded Tools (No Change to Selection Logic)

Wildcard keeps using exactly the 3 tools currently hardcoded in `registry.py`. No pool selection, no dynamic routing.

| Mode | Tool (fixed) | API Cost | Current Price | Proposed Break-Even Price | Status |
|---|---|---|---|---|---|
| Quick | `prediction-offline` (GPT-4.1) | $0.008 | $0.001 | **$0.01** | Currently losing ~$0.007/call |
| Deep | `prediction-online` (GPT-4.1) | $0.026 | $0.01 | **$0.03** | Currently losing ~$0.016/call |
| Super | `superforcaster` (GPT-4.1) | $0.020 | $0.05 | **$0.02** (floor) | Currently profitable at $0.03/call |

**Key finding:** Quick and Deep are priced below API cost and must be raised. Super is currently profitable — the break-even floor is $0.02, but the current price of $0.05 generates $0.03/call margin. **Lowering Super to $0.02 is the break-even floor, not a recommendation.** There is no reason to reduce a profitable price. The $0.02 figure is shown for completeness to illustrate the true cost floor.

**Note on Super (superforcaster reassignment):** Under the proposed dynamic pool model (Scenario B / [MPP_MODE_DEFINITIONS.md](./MPP_MODE_DEFINITIONS.md)), superforcaster moves from the Super pool to the **Deep pool** because it performs web search but no embeddings/reasoning. This means Scenario A's Super pricing ($0.02 floor) and Scenario B's Super pricing ($0.08 floor) reflect different tools entirely.

**Note on tool cost vs quality:** Superforcaster costs less than prediction-online because it makes only 1 LLM call + 1 Serper call (search snippets, no page scraping), while prediction-online makes 2 LLM calls + 2 Serper calls + URL scraping. But superforcaster has worse benchmark performance (Brier 0.34 vs 0.26). Under the current hardcoded setup, the user pays the most for Super but gets a cheaper, weaker tool.

### Scenario B: Dynamic Tool Pools (Per MPP_MODE_DEFINITIONS.md)

Wildcard adopts pool-based weighted selection. Any tool in a mode's pool may be selected based on benchmark performance. Price must cover the **most expensive tool in the pool** to ensure no selection produces a loss.

| Mode | Cheapest Tool | Most Expensive Tool | Weighted Average (est.) | Proposed Break-Even Price |
|---|---|---|---|---|
| Quick | `gemini-prediction` ($0.0004) | `claude-prediction-offline` ($0.014) | ~$0.009 | **$0.015** |
| Deep | `superforcaster` ($0.020) | `claude-prediction-online` ($0.040) | ~$0.028 | **$0.04** |
| Super | `prediction-request-rag` ($0.021) | `prediction-request-reasoning-claude` ($0.074) | ~$0.048 | **$0.08** |

**How "Weighted Average" is estimated:** Assumes starting defaults (`prediction-offline`, `prediction-online`, `prediction-request-reasoning`) get ~60% of traffic, with remaining 40% distributed across other pool tools proportional to benchmark performance. This is approximate — real weighted averages depend on the IPFS CSV scores.

**Why price at max, not average:** If the best-performing tool in the pool turns out to be the most expensive (e.g., `prediction-request-reasoning-claude` has the best Brier in Super), the selection logic should be free to route heavy traffic there. Pricing at the average would mean losing money every time Claude gets selected. Pricing at the max ensures we never lose regardless of routing.

### Summary Comparison

| Mode | Current Price | Scenario A (break-even, hardcoded) | Scenario B (break-even, dynamic) |
|---|---|---|---|
| **Quick** | $0.001 | **$0.01** (+10x) | **$0.015** (+15x) |
| **Deep** | $0.01 | **$0.03** (+3x) | **$0.04** (+4x) |
| **Super** | $0.05 | **$0.02** (-60%) | **$0.08** (+60%) |

**Scenario A** reveals that Quick and Deep are underpriced and must be raised. Super's break-even floor is $0.02, but its current price ($0.05) is profitable — lowering it is a separate policy choice, not a cost requirement.

**Scenario B is a real pricing tradeoff, not just added flexibility.** The pool includes Claude Sonnet variants, which are ~1.5-2x more expensive than GPT-4.1 for the same task. The best-performing tool (`prediction-request-reasoning-claude`, Brier 0.2058) is also the most expensive ($0.074/call). Dynamic pools mean either:
- **Higher user prices** (as proposed: $0.015 / $0.04 / $0.08) to cover worst-case tool costs, or
- **An intentional subsidy** where we absorb the cost difference when expensive tools are selected, pricing at the weighted average instead of the max.

The break-even prices above assume no subsidy — every possible tool selection must be cost-covered. If Valory is willing to subsidize Claude routing to drive better predictions, the prices can be lower but the expected loss per Claude selection should be quantified and budgeted.
