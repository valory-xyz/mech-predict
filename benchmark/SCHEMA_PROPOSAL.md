# IPFS Schema Proposal

## Current State

### Request payload (trader → mech, stored on IPFS)

```json
{
  "prompt": "Will BTC hit $100k by June?",
  "tool": "prediction-online"
}
```

Minimal — just the question and which tool to use.

### Response payload (mech → IPFS)

```json
{
  "requestId": 12345,
  "result": "p_yes: 0.72, p_no: 0.28, confidence: 0.8, ...",
  "tool": "prediction-online",
  "prompt": "You are an LLM... ARTICLE 0, URL: https://... CONTENT: ...",
  "cost_dict": {
    "input_tokens": 4200,
    "output_tokens": 850,
    "total_tokens": 5050,
    "input_cost": 0.0084,
    "output_cost": 0.0068,
    "total_cost": 0.0152
  },
  "metadata": {
    "model": "gpt-4.1-2025-04-14",
    "tool": "prediction-online",
    "params": {}
  },
  "is_offchain": false
}
```

---

## Proposed Changes

### Request payload — add `market_context`

All market-related metadata goes in a dedicated `market_context` object. Tools can ignore it. Benchmark reads it.

```json
{
  "schema_version": "2.0",
  "prompt": "Will BTC hit $100k by June?",
  "tool": "prediction-online",
  "market_context": {
    "market_id": "polymarket_xyz789",
    "platform": "polymarket",
    "market_prob": 0.65,
    "market_liquidity_usd": 450000,
    "market_close_at": "2026-06-30T00:00:00Z"
  }
}
```

**Design decisions:**
- `market_context` is a separate object, not top-level fields — keeps it clean, tools don't need to know about it
- `schema_version` at the top level so consumers know what to expect
- All fields in `market_context` are optional — old requests without it still work, benchmark marks them as lower provenance grade
- `market_prob`: the market price at request time, standardized as mid-price. This is what the benchmark compares against for edge-over-market (the most important metric). Fetching this later from a subgraph for every historical prediction would be expensive and unreliable — the trader already has it.
- `market_liquidity_usd`: needed for stratification (beating a $500k market is meaningful, beating a $500 market might be noise). Cheaper to embed at request time than to fetch retroactively for thousands of predictions.
- `market_close_at`: needed to calculate prediction lead time (how far before resolution was the prediction made). Static field the trader already knows.

### Response payload — add `execution`, `source_content`, `tool_hash`

```json
{
  "schema_version": "2.0",
  "requestId": 12345,
  "result": "p_yes: 0.72, p_no: 0.28, confidence: 0.8, ...",
  "tool": "prediction-online",
  "prompt": "You are an LLM... the full prompt sent to the model...",
  "cost_dict": {
    "input_tokens": 4200,
    "output_tokens": 850,
    "total_tokens": 5050,
    "input_cost": 0.0084,
    "output_cost": 0.0068,
    "total_cost": 0.0152
  },
  "metadata": {
    "model": "gpt-4.1-2025-04-14",
    "tool": "prediction-online",
    "tool_hash": "bafybei...",
    "params": {}
  },
  "execution": {
    "latency_ms": 12300
  },
  "source_content": {
    "https://reuters.com/btc-forecast": "Bitcoin analysts predict...",
    "https://bbc.com/crypto-market": "Cryptocurrency markets showed..."
  },
  "is_offchain": false
}
```

**New fields explained:**

`execution.latency_ms` — how long the `run()` function took end-to-end. Currently only in Prometheus as aggregates. Needed per-request for benchmark cost-performance analysis and production timeout parity.

`source_content` — the web content the tool used, stored as a dict of URL → scraped text. Currently baked into the `prompt` field as formatted text (fragile to parse back out). Stored separately so cached replay can feed it directly to another tool/prompt variant. Optional — tools that don't do web search won't have this.

`metadata.tool_hash` — the IPFS hash of the tool package that was executed (from `TOOLS_TO_PACKAGE_HASH`). This is the version identifier — different hash means different code, different prompt template, different behavior. Without this in the response, you'd have to cross-reference the mech's deployment config at the time of prediction to know which version ran, which is unreliable since configs change between deployments.

Note: `execution.status` and `execution.error_reason` are NOT included — the mech already pushes error responses to IPFS (e.g., "Invalid response"), so success/failure is derivable from the `result` field.

---

## Backward Compatibility

- `schema_version` field lets consumers distinguish old vs new payloads
- All new fields are additive — old consumers that don't know about `market_context`, `execution`, or `source_content` just ignore them
- Old requests/responses without `schema_version` are treated as `"1.0"`
- The `prompt` field continues to exist unchanged — `source_content` is an additional field, not a replacement

---

## What This Enables

| Field | What it unlocks |
|-------|----------------|
| `market_context.market_id` | Direct question-to-market matching (eliminates string prefix hack) |
| `market_context.market_prob` | Edge-over-market calculation without expensive subgraph lookups |
| `market_context.market_liquidity_usd` | Market efficiency stratification without subgraph lookups |
| `market_context.market_close_at` | Prediction lead time calculation without API calls |
| `execution.latency_ms` | Per-request latency for cost-performance analysis |
| `source_content` | Clean cached replay without parsing the prompt field |
| `metadata.tool_hash` | Know exactly which tool version produced each prediction |

---

## What Changes Where

| Component | Change | Effort |
|-----------|--------|--------|
| **Trader** | Add `schema_version` + `market_context` to request payload | Medium — trader already has this data |
| **Mech (`behaviours.py`)** | Add `schema_version`, `execution.latency_ms`, `metadata.tool_hash` to response | Small — data already available in code |
| **Tools** | Return `source_content` separately (in addition to current behavior) | Medium — each tool needs to return scraped content alongside the result |
| **Benchmark** | Read new fields from IPFS, fall back gracefully for old `"1.0"` payloads | Built into benchmark code from the start |
