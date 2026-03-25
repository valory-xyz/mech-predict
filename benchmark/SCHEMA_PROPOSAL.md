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

### Request payload — add `request_context`

All request-related metadata goes in a dedicated `request_context` object. Tools can ignore it. Benchmark reads it. The naming is intentionally generic — `request_context` works for prediction mechs today and can accommodate other mech types (image generation, code execution, etc.) in the future.

The `type` field inside `request_context` acts as a discriminator for which platform-specific fields are present — `request_context` contains both common fields (applicable to all platforms) and platform-specific fields (e.g., `market_spread` for Polymarket only). Consumers use `type` to know which fields to expect.

> **Trust note:** All values in `request_context` are provided by the trader and should be treated as **untrusted input**. The benchmark must not assume these values are accurate without verification. Currently, natural incentive alignment provides reasonable assurance (traders bet real money, so fake data hurts them), but the schema is designed to accommodate a `proof` or `attestation` field in the future if cryptographic verification is needed.

> **Timing note:** `market_prob` is captured at request time by the trader. There can be a delay of 1-5 minutes before the mech executes the tool (polling interval + execution time). For most markets (political, long-horizon) this is negligible. For fast-moving markets, the benchmark can stratify by market volatility to account for staleness. Request time is still the correct anchor — it captures what the market thought *before* the mech was asked, which is the fair baseline to beat. Capturing at execution time would require coupling the mech to market-specific APIs.

#### Polymarket example

```json
{
  "schema_version": "2.0",
  "prompt": "Will BTC hit $100k by June?",
  "tool": "prediction-online",
  "request_context": {
    "market_id": "0xdef456...",
    "type": "polymarket",
    "market_prob": 0.65,
    "market_liquidity_usd": 450000,
    "market_close_at": "2026-06-30T00:00:00Z",
    "market_spread": 0.02
  }
}
```

#### Omen example

```json
{
  "schema_version": "2.0",
  "prompt": "Will ETH hit $5k by March?",
  "tool": "prediction-online",
  "request_context": {
    "market_id": "0xabc123...",
    "type": "omen",
    "market_prob": 0.40,
    "market_liquidity_usd": 12000,
    "market_close_at": "2026-03-31T00:00:00Z"
  }
}
```

**Field reference:**

| Field | Platforms | Description |
|-------|----------|-------------|
| `market_id` | All | Platform-specific market identifier (Polymarket condition ID, Omen FPMM contract address) |
| `type` | All | `"polymarket"` or `"omen"` — discriminator that tells consumers which platform-specific fields to expect |
| `market_prob` | All | Market price at request time (mid-price). Used for edge-over-market calculation. See timing note above |
| `market_liquidity_usd` | All | Market liquidity in USD. Used for stratification (high-liquidity markets are harder to beat) |
| `market_close_at` | All | Market close/resolution date. Used to calculate prediction lead time |
| `market_spread` | Polymarket | Bid-ask spread from the CLOB order book. Not applicable to Omen (AMM has no order book) |

**Design decisions:**
- `request_context` is a separate object, not top-level fields — keeps it clean, tools don't need to know about it
- Named `request_context` (not `market_context`) so it's generic enough for non-prediction mechs in the future
- `schema_version` at the top level so consumers know what to expect
- All fields in `request_context` are optional — old requests without it still work, benchmark marks them as lower provenance grade
- `type` inside `request_context` acts as a discriminator for platform-specific fields — this allows Polymarket-specific fields (like `market_spread`) without forcing them onto Omen
- Common fields (`market_prob`, `market_liquidity_usd`, `market_close_at`) exist on both platforms. Cheaper to embed at request time than to fetch retroactively from subgraphs for thousands of predictions
- Platform-specific fields can be added over time without breaking the schema — just check `type` before reading them
- All values are untrusted — see trust note above

### Response payload — add `tool_hash`, `execution_latency_ms`, `source_content`, runtime params to `metadata`

```json
{
  "schema_version": "2.0",
  "requestId": 12345,
  "executed_at": "2026-03-15T14:23:00Z",
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
    "execution_latency_ms": 12300,
    "params": {
      "default_model": "gpt-4.1-2025-04-14",
      "temperature": 0,
      "max_tokens": 500,
      "num_urls": 3,
      "num_words": 300
    },
    "source_content": {
      "https://reuters.com/btc-forecast": "Bitcoin analysts predict...",
      "https://bbc.com/crypto-market": "Cryptocurrency markets showed..."
    }
  },
  "is_offchain": false
}
```

**New fields explained:**

`executed_at` — ISO 8601 UTC timestamp of when the tool execution completed. Currently the only timing signal is the block timestamp of the on-chain Deliver event, which reflects *delivery* time (after consensus), not *execution* time. `executed_at` captures the precise moment the tool finished, enabling accurate prediction lead time calculation and temporal integrity verification.

`metadata.execution_latency_ms` — how long the `run()` function took end-to-end. Currently only in Prometheus as aggregates. Needed per-request for benchmark cost-performance analysis and production timeout parity.

`metadata.tool_hash` — the IPFS hash of the tool package that was executed (from `TOOLS_TO_PACKAGE_HASH`). This is the version identifier — different hash means different code, different prompt template, different behavior. Without this in the response, you'd have to cross-reference the mech's deployment config at the time of prediction to know which version ran, which is unreliable since configs change between deployments.

`metadata.params` — the actual runtime configuration used for this prediction. Currently `params` only stores static defaults from `component.yaml` (just `default_model`). Runtime values like `temperature`, `max_tokens`, `num_urls`, `num_words`, embedding model for RAG tools etc. are passed via kwargs but never recorded. Capturing the full runtime config lets the benchmark know exactly what configuration produced each prediction, which is essential for parameter sweeps and reproducibility.

`metadata.source_content` — the web content the tool used, stored as a dict of URL → scraped text. Currently baked into the `prompt` field as formatted text (fragile to parse back out). Stored separately inside `metadata` (alongside other execution artifacts) so cached replay can feed it directly to another tool/prompt variant. Optional — tools that don't do web search won't have this.

---

## Backward Compatibility

- `schema_version` field lets consumers distinguish old vs new payloads
- All new fields are additive — old consumers that don't know about `request_context` or new `metadata` fields just ignore them
- Old requests/responses without `schema_version` are treated as `"1.0"`
- The `prompt` field continues to exist unchanged — `metadata.source_content` is an additional field, not a replacement
- Platform-specific fields in `request_context` can be added over time — consumers check `type` before reading them

---

## What This Enables

| Field | What it unlocks |
|-------|----------------|
| `executed_at` | Precise execution timestamp for prediction lead time and temporal integrity |
| `request_context.market_id` | Direct question-to-market matching (eliminates string prefix hack) |
| `request_context.type` | Platform-aware evaluation and platform-specific field handling |
| `request_context.market_prob` | Edge-over-market calculation without expensive subgraph lookups |
| `request_context.market_liquidity_usd` | Market efficiency stratification without subgraph lookups |
| `request_context.market_close_at` | Prediction lead time calculation without API calls |
| `request_context.market_spread` | Polymarket spread analysis for PnL simulation |
| `metadata.execution_latency_ms` | Per-request latency for cost-performance analysis |
| `metadata.tool_hash` | Know exactly which tool version produced each prediction |
| `metadata.params` | Full runtime config for reproducibility and parameter sweep analysis |
| `metadata.source_content` | Clean cached replay without parsing the prompt field |

---

## What Changes Where

| Component | Change | Effort |
|-----------|--------|--------|
| **Trader** | Add `schema_version`, `request_context` to request payload | Medium — trader already has this data |
| **Mech (`behaviours.py`)** | Add `schema_version`, `executed_at`, `metadata.execution_latency_ms`, `metadata.tool_hash` to response. Populate `metadata.params` with runtime config | Small — data already available in code |
| **Tools** | Return `source_content` in metadata (in addition to current behavior) | Medium — each tool needs to return scraped content alongside the result |
| **Benchmark** | Read new fields from IPFS, fall back gracefully for old `"1.0"` payloads | Built into benchmark code from the start |

---

## Open Questions

### A. Order book data in `request_context`

Adding order book depth (beyond just spread) could be useful for Polymarket PnL analysis. However, order books are Polymarket-specific — Omen uses an AMM with no order book.

The schema already supports this via the platform-specific fields pattern — any Polymarket-specific fields can be added without affecting Omen. The question is whether to include them now or defer.

**Status:** Not blocking for initial implementation. Can add later without breaking the schema.

---

### B. `extract_question` regex mismatch in superforcaster (pre-existing)

Production prompts use `repr()` to format the question, which wraps it in **single quotes**: `With the given question 'Will BTC...'`. The `extract_question()` regex in superforcaster expects **double quotes** and fails to match. The fallback returns the full prompt as the question — works but makes Serper search queries less targeted.

This is a pre-existing issue, not introduced by the `source_links` change. Affects both live and cached paths equally.

**Suggestion:** Fix the regex to handle both quote styles: `r'question\s+["\'](.+?)["\']\s+and\s+the\s+` `` `yes` `` `'`

**Status:** Low priority. Existing fallback works. Can fix separately.
