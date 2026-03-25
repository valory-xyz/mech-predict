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

> **Timing note:** `market_prob` is captured at request time by the trader. There can be a delay of 1-5 minutes before the mech executes the tool (polling interval + execution time). For most markets (political, long-horizon) this is negligible. For fast-moving markets, the benchmark can stratify by difficulty (`abs(market_prob - 0.5)`) and prediction lead time as proxies. Request time is still the correct anchor — it captures what the market thought *before* the mech was asked, which is the fair baseline to beat. Capturing at execution time would require coupling the mech to market-specific APIs.
>
> **Volatility (deferred):** Neither platform provides a pre-computed volatility metric. Polymarket's Gamma API has `oneHourPriceChange` but Omen has no equivalent, so there is no cross-platform field to include. If per-request volatility is needed in the future, the benchmark can compute it retroactively from trade history (Polymarket CLOB `/prices-history` at 1-min fidelity, Omen subgraph `FpmmTrade` entity) without requiring it in the request payload.

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

| Field | Platforms | Source | Description |
|-------|----------|--------|-------------|
| `market_id` | All | Polymarket: `conditionId`. Omen: FPMM contract address | Platform-specific market identifier. Enables direct question-to-market matching (eliminates fragile string prefix hack) |
| `type` | All | Trader knows which platform the market is on | `"polymarket"` or `"omen"` — discriminator that tells consumers which platform-specific fields to expect |
| `market_prob` | All | Polymarket: mid-price `(bestBid + bestAsk) / 2` from CLOB. Omen: `outcomeTokenMarginalPrices[0]` from subgraph | Market consensus probability at request time. Used for **edge-over-market** (Stage 2) — measures prediction quality vs market consensus, not execution price. See notes below |
| `market_liquidity_usd` | All | Polymarket: `liquidityNum` from Gamma API. Omen: `usdLiquidityParameter` from subgraph | Market liquidity in USD. Used for **market efficiency stratification** and as a **slippage proxy for PnL simulation** (Stage 5) |
| `market_close_at` | All | Polymarket: `endDateIso` from Gamma API. Omen: `openingTimestamp` from subgraph | Market close/resolution date. Used to calculate **prediction lead time** and **time horizon stratification** |
| `market_spread` | Polymarket | `spread` from Gamma API (or `bestAsk - bestBid` from CLOB) | Bid-ask spread from the CLOB order book. Used for **PnL simulation** (Stage 5) as transaction cost. Not applicable to Omen — AMM has no order book, friction comes from the AMM fee (~2%) which is a known constant the benchmark can hardcode |

> **`market_prob` is mid-price, not execution price — by design.** Edge-over-market (the primary selection metric) measures whether the tool's prediction is better than the market's *consensus fair value*. Mid-price is the correct representation of consensus for both platforms. Execution realism (spread, slippage) is handled separately in PnL simulation (Stage 5) using `market_spread` + `market_liquidity_usd`.
>
> **Platform-specific friction costs.** Polymarket (CLOB): traders pay the bid-ask `market_spread`. Omen (AMM): traders pay a pool fee (~2%, a known constant) plus slippage determined by `market_liquidity_usd` and trade size. The AMM fee is not included in the request because it's a protocol-level constant, not a per-market variable — the benchmark hardcodes it.
>
> **Order book depth (deferred).** Full order book data (multiple price/size levels) would improve Polymarket PnL simulation precision but adds significant payload (20+ numbers per request). The `market_spread` + `market_liquidity_usd` proxy is sufficient for V1 — for typical trade sizes on markets with reasonable liquidity, the difference is negligible. Can be added as a Polymarket-specific field later if PnL simulation needs higher fidelity.

**Which benchmark metric uses which field:**

| Benchmark metric | Fields used from `request_context` |
|-----------------|-----------------------------------|
| Reliability (Stage 1) | None — uses tool output status only |
| Edge over market (Stage 2) | `market_prob` |
| Brier score (Stage 3) | None — uses tool output + resolution |
| Calibration / ECE (Stage 4) | None — uses tool output + resolution |
| PnL simulation (Stage 5 Tier 1) | `market_prob` + `market_spread` + `market_liquidity_usd` |
| PnL realized (Stage 5 Tier 2) | None from request — uses trader execution data |
| Platform stratification | `type` |
| Market efficiency stratification | `market_liquidity_usd` |
| Time horizon stratification | `market_close_at` |
| Difficulty stratification | `market_prob` (via `abs(market_prob - 0.5)`) |
| Market matching | `market_id` |

**Design decisions:**
- `request_context` is a separate object, not top-level fields — keeps it clean, tools don't need to know about it
- Named `request_context` (not `market_context`) so it's generic enough for non-prediction mechs in the future
- `schema_version` at the top level so consumers know what to expect
- All fields in `request_context` are optional — old requests without it still work, benchmark marks them as lower provenance grade
- `type` inside `request_context` acts as a discriminator for platform-specific fields — this allows Polymarket-specific fields (like `market_spread`) without forcing them onto Omen
- Common fields (`market_id`, `market_prob`, `market_liquidity_usd`, `market_close_at`) exist on both platforms. Cheaper to embed at request time than to fetch retroactively from subgraphs for thousands of predictions
- `market_spread` is Polymarket-only because Omen's AMM has no order book — Omen's friction (pool fee + slippage) is derivable from `market_liquidity_usd` and the protocol's constant fee rate
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
