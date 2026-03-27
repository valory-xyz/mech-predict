# Benchmark System — Identified Gaps

Changes needed in existing systems before the benchmark pipeline can fully function.

---

### 1. Add `source_links` support to superforcaster and gemini_prediction

**Status:** Blocking

**Why:** The benchmark's cached replay mode feeds tools pre-fetched web content so they don't hit the live web. Without this, running these tools on resolved questions means they Google the answer — that's testing web search, not forecasting ability. All other prediction tools (`prediction_request`, `prediction_request_rag`, `prediction_url_cot`, `prediction_request_reasoning`, `prediction_request_sme`) already support this.

**What exists today:**

`prediction_request.py` already does this (line ~935):
```python
if not source_links:
    urls = get_urls_from_queries_serper(queries, api_key)  # live web
    docs = extract_texts(urls)
else:
    for url, content in source_links.items():              # cached content
        doc = extract_text(html=content)
```

`superforcaster.py` always hits the live web (line ~404):
```python
serper_response = fetch_additional_sources(question, serper_api_key)  # always live
```

**What to implement:**

In `superforcaster.py`, add `source_links` kwarg support:
```python
source_links = kwargs.get("source_links", None)
if source_links:
    # Format cached content into the same structure the tool expects
    # Skip the Serper API call entirely
else:
    # Existing behavior: call Serper API
    serper_response = fetch_additional_sources(question, serper_api_key)
```

Same pattern for `gemini_prediction`.

**Files to change:**
- `packages/valory/customs/superforcaster/superforcaster.py`
- `packages/dvilela/customs/gemini_prediction/gemini_prediction.py`

**Effort:** Small — follow existing pattern from `prediction_request.py`

---

### 2. Market metadata in trader requests

**Status:** Blocking — needs discussion on format with trader team

**Why:** Edge-over-market measures whether the tool's prediction is better than the market consensus — it's the most important benchmark metric. To calculate it, we need the market probability at the time the prediction was requested. The trader already knows this — it's looking at the market to decide whether to request a prediction. Currently the trader only sends `{"prompt": "...", "tool": "..."}`.

**What the trader should send:**
```json
{
  "prompt": "Will BTC hit $100k by June?",
  "tool": "prediction-online",
  "market_id": "polymarket_xyz",
  "platform": "polymarket",
  "market_prob_at_request": 0.65,
  "market_prob_type": "mid",
  "market_liquidity": 450000,
  "market_volume": 1200000
}
```

This lands on IPFS automatically since requests are stored on-chain. Tools ignore fields they don't use — no tool changes needed. The benchmark's `fetch_production.py` reads the request from IPFS and gets contemporaneous market state for free.

**Without this:** We'd have to reconstruct historical market prices from trade history APIs (Polymarket CLOB, Omen subgraph), which is lossy and unreliable.

**Also enables:**
- Direct question-to-market matching via `market_id` (eliminates the fragile string prefix matching currently used for Omen accuracy calculation)
- Future tools that want to incorporate market price or liquidity into their reasoning

**Owner:** Trader team (not this repo)

---

### 3. Per-request latency in IPFS response

**Status:** Nice to have

**Why:** Tool execution time is currently tracked in Prometheus as aggregate histograms but not stored per-request in the IPFS response. For the benchmark we want per-prediction latency to:
- Enforce production timeout parity (`TASK_DEADLINE` = 240s)
- Correlate speed with accuracy (do slower predictions = better predictions?)
- Track cost-performance tradeoffs (a tool that's 2x better but 10x slower may not be practical)

The benchmark runner can measure latency itself for new runs — this only matters for historical production data.

**What to implement:**

In `packages/valory/skills/task_execution/behaviours.py`, add one field to the response dict (~line 779):
```python
response["latency_ms"] = int(tool_exec_time_duration * 1000)
```

**Effort:** 1 line

---

### 4. Web content as separate IPFS field

**Status:** Nice to have — current workaround is functional

**Current state:** Scraped articles are embedded inside the `prompt` field on IPFS as formatted text (`ARTICLE 0, URL: ..., CONTENT: ...`). Parseable but fragile — if a tool formats differently the parsing breaks.

**Ideal:** Store as a separate field in the IPFS response (e.g., `"source_content": {"url": "html_content", ...}`).

**Workaround:** Parse from `prompt` field for historical data. Benchmark runner captures snapshots cleanly as separate files for new data going forward.

---

## Summary

| # | Gap | Status | Owner |
|---|-----|--------|-------|
| 1 | `source_links` in superforcaster + gemini_prediction | **Blocking** | Tools (this repo) |
| 2 | Market metadata in trader requests | **Blocking** (needs format discussion) | Trader team |
| 3 | Per-request latency in IPFS response | Nice to have | Mech (this repo) |
| 4 | Web content as separate IPFS field | Nice to have | Mech (this repo) |
