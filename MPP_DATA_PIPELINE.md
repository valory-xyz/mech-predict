# MPP Data Pipeline — Metadata Capture & Persistent Storage Proposal

**Status:** Draft
**Date:** 2026-04-13
**Urgency:** Chrome store submission is imminent. Extension-side changes (new request fields) must be agreed before submission. Server-side storage can follow, but the extension API contract is harder to change after launch.
**Companion docs:** [MPP_MODE_DEFINITIONS.md](./MPP_MODE_DEFINITIONS.md), [MPP_COST_PROPOSAL.md](./MPP_COST_PROPOSAL.md)

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Current State: What's Captured and Where It Goes](#2-current-state)
3. [What We Need to Capture (Minimum Viable Schema)](#3-schema)
4. [Extension Changes: What the Plugin Must Send Before Chrome Store Submission](#4-extension-changes)
5. [Server Changes: What the Server Must Capture](#5-server-changes)
6. [Persistent Storage Options](#6-storage-options)
7. [How This Feeds the Data Loop](#7-data-loop)

---

## 1. The Problem <a id="1-the-problem"></a>

**There is no persistent storage for prediction data.** Every prediction the MPP server produces vanishes from Redis after 1 hour. There is no database, no log file, no export mechanism.

This means:

- **We cannot measure tool quality in production.** We have no record of what any tool predicted, for which market, at what market odds.
- **We cannot compute edge over market.** Without knowing the market probability at prediction time, we can't measure whether our tools add value vs. the market consensus.
- **We cannot validate benchmark scores against real usage.** Our benchmark pipeline scores tools on historical data, but we have no way to compare those scores against live MPP predictions.
- **We cannot debug production failures.** When a tool fails or produces a bad prediction, there is no audit trail — the error expires from Redis and is gone.
- **We cannot track actual API costs.** Tools return `cost_dict` with real token counts, but the server ignores it. We're estimating costs from source code (see [MPP_COST_PROPOSAL.md](./MPP_COST_PROPOSAL.md)) when we could be measuring them.

**The Chrome store submission deadline makes the extension-side changes urgent.** Once the extension ships, changing the request schema (adding new fields) requires a Chrome store update review cycle. Server-side changes have no such constraint.

---

## 2. Current State: What's Captured and Where It Goes <a id="2-current-state"></a>

### 2.1 What the Extension Sends (PredictionRequest)

**Source:** `wildcard/server/src/models/request.py` (lines 8-11)

```python
class PredictionRequest(BaseModel):
    mode: Literal["quick", "deep", "super"] = "deep"
    question: str = Field(..., min_length=1, max_length=5000)
    outcomes: list[str] = Field(..., min_length=2, max_length=2)
```

Three fields. No market context, no probabilities, no market identifier.

### 2.2 What the Extension Already Has But Doesn't Send

The extension fetches rich market data that never reaches the server:

**Gamma API** (`https://gamma-api.polymarket.com/events?slug=<slug>`):
- `conditionId` — unique market identifier
- `question` — market question
- `outcomes` — ["Yes", "No"]
- `outcomePrices` — **current market probabilities** (e.g., ["0.75", "0.25"])
- `endDate` — market resolution deadline
- `volume` — total trading volume

**CLOB API** (`https://clob.polymarket.com/markets/<conditionId>`):
- `tokens[].price` — **live trading prices** (more current than Gamma)

**Source:** `wildcard/src/core/polymarket/market-data.ts` (lines 52-136)

The content-UI already computes `marketPriceYes` from these APIs and uses it to display edge in the UI (`wildcard/src/components/prediction/PredictionResult.tsx`, line 34). It just doesn't include it in the POST body to the server.

### 2.3 What the Server Stores

**Source:** `wildcard/server/src/store.py` (lines 1-71)

All prediction data is stored in Redis with a **1-hour TTL** (`config.py` line 33: `prediction_ttl: int = 3600`):

```
Key:    prediction:{request_id}
TTL:    3600 seconds (1 hour)
Value:  {
    "status": "processing|complete|failed",
    "mode": "quick|deep|super",
    "tool": "prediction-offline",
    "model": "gpt-4.1-2025-04-14",
    "cost": "1000",
    "result": {"p_yes": 0.65, "p_no": 0.35, "confidence": 0.8, "info_utility": 0.42},
    "error": null
}
```

After 1 hour, this key expires and the prediction is gone forever.

### 2.4 What the Server Ignores From Tool Output

Each tool's `run()` function returns a tuple:

```python
(deliver_msg, prompt_used, transaction_data, counter_callback)
```

The server (`tools/runner.py` lines 98-109) only uses `deliver_msg` (element 0, the JSON prediction). It **ignores**:

| Element | Contains | Currently |
|---|---|---|
| `prompt_used` (element 1) | The full prompt sent to the LLM | Ignored |
| `transaction_data` (element 2) | Optional metadata dict | Ignored |
| `counter_callback` (element 3) | `TokenCounterCallback` with `cost_dict`: actual input/output tokens, per-model costs | Ignored |

The `cost_dict` is particularly valuable — it contains the **real API cost** of each prediction, not our estimates.

### 2.5 Verified: No Persistent Storage Exists

We verified exhaustively that the wildcard server has **zero persistent storage beyond Redis**:

- **No database dependencies** — `pyproject.toml` has no SQLAlchemy, psycopg2, pymongo, sqlite3, or any DB library
- **No database services** — `docker-compose.yml` only defines `redis:7-alpine`, no other services
- **No Docker volumes** — no volume mounts of any kind; Redis data is purely in-memory
- **No file writing** — no code path writes to disk (no FileHandler in logging, no JSONL exports, no data dumps)
- **No telemetry** — no Prometheus, Datadog, Sentry, or any metrics/analytics integration
- **No background exports** — the only background task is the MPP settlement loop

If Redis restarts, all in-flight predictions and their results are lost entirely.

---

## 3. What We Need to Capture (Minimum Viable Schema) <a id="3-schema"></a>

Every completed prediction should be persisted with these fields:

### 3.1 Fields From the Extension (Must be added to PredictionRequest before Chrome store submission)

| Field | Type | Example | Why |
|---|---|---|---|
| `market_url` | `string` | `"https://polymarket.com/event/btc-100k-july"` | Link prediction to specific market. Enables outcome resolution. |
| `condition_id` | `string` | `"0x1234abcd..."` | Polymarket's unique market identifier. Stable across URL changes. Needed for API lookups. |
| `market_prob_at_prediction` | `float` | `0.65` | Market's Yes probability when the user requested the prediction. **Critical for edge-over-market calculation.** Without this, we can never measure if our tools beat the market. |

**Implementation effort:** The extension already computes all three values. `market_url` is the current page URL. `condition_id` comes from the Gamma API response (`Market.conditionId`). `marketPriceYes` is already computed in `PredictionPanel.tsx` (line 135-140) from CLOB/Gamma APIs.

The change is: add these 3 fields to the `requestPrediction()` call (`App.tsx` line 125) and to the server's `PredictionRequest` Pydantic model. Approximately 5 lines of code on each side.

### 3.2 Fields the Server Already Has (No extension change needed)

| Field | Type | Source | Why |
|---|---|---|---|
| `request_id` | `string` | Server generates (`req_{token}`) | Primary key |
| `predicted_at` | `datetime (UTC)` | Server timestamps | When the prediction was made |
| `question` | `string` | Extension sends (already) | The prediction question |
| `outcomes` | `list[str]` | Extension sends (already) | Yes/No labels |
| `mode` | `string` | Extension sends (already) | quick/deep/super |
| `tool` | `string` | Server (from mode config or selection logic) | Which tool actually ran |
| `model` | `string` | Server (from mode config) | Which LLM model was used |
| `p_yes` | `float` | Tool output | Predicted probability |
| `p_no` | `float` | Tool output | 1 - p_yes |
| `confidence` | `float` | Tool output | Tool's self-assessed confidence |
| `info_utility` | `float` | Tool output | Usefulness of source information |
| `success` | `bool` | Server | Did the tool return a valid result? |
| `error` | `string \| null` | Server | Error message if failed |

### 3.3 Fields the Server Should Start Capturing (Currently ignored from tool output)

| Field | Type | Source | Why |
|---|---|---|---|
| `cost_dict` | `object` | `counter_callback.cost_dict` (tool tuple element 3) | Real token usage: `{input_tokens, output_tokens, total_tokens, input_cost, output_cost, total_cost}`. Actual API spend per prediction. |
| `prompt_used` | `string` | Tool tuple element 1 | Full prompt sent to LLM. Enables debugging, prompt comparison, reproducibility. |
| `latency_ms` | `int` | Server measures (`time.monotonic()` around tool execution) | Execution time. Needed for latency analysis and SLA monitoring. |

### 3.4 Fields the Server Can Derive

| Field | Type | How | Why |
|---|---|---|---|
| `category` | `string` | `classify_category(question)` from mech-predict's shared classifier | Per-category performance analysis. Same classifier used by trader and benchmark pipeline. |

### 3.5 Full Persisted Record

Combining all of the above, each prediction record looks like:

```json
{
    "request_id": "req_abc123...",
    "predicted_at": "2026-04-13T14:30:00Z",
    "question": "Will BTC hit 100k by July 2026?",
    "outcomes": ["Yes", "No"],
    "mode": "deep",
    "tool": "prediction-online",
    "model": "gpt-4.1-2025-04-14",
    "market_url": "https://polymarket.com/event/btc-100k-july",
    "condition_id": "0x1234abcd...",
    "market_prob_at_prediction": 0.65,
    "category": "crypto",
    "p_yes": 0.72,
    "p_no": 0.28,
    "confidence": 0.8,
    "info_utility": 0.7,
    "success": true,
    "error": null,
    "latency_ms": 12340,
    "cost_dict": {
        "input_tokens": 9200,
        "output_tokens": 500,
        "total_tokens": 9700,
        "input_cost": 0.0184,
        "output_cost": 0.004,
        "total_cost": 0.0224
    },
    "prompt_used": "You are an LLM inside a multi-agent system..."
}
```

---

## 4. Extension Changes: What the Plugin Must Send Before Chrome Store Submission <a id="4-extension-changes"></a>

### 4.1 New Fields in PredictionRequest

The server's `PredictionRequest` model (`server/src/models/request.py`) needs 3 new optional fields:

```python
class PredictionRequest(BaseModel):
    mode: Literal["quick", "deep", "super"] = "deep"
    question: str = Field(..., min_length=1, max_length=5000)
    outcomes: list[str] = Field(..., min_length=2, max_length=2)
    # NEW — market context for the data loop
    market_url: str | None = None
    condition_id: str | None = None
    market_prob_at_prediction: float | None = None
```

Fields are **optional** (`None` default) for backward compatibility. Free predictions and any existing clients continue to work without sending them.

### 4.2 Extension-Side Change

In the extension's prediction request builder, the data is already available. The change is to include it in the POST body:

**Current** (`App.tsx` / `prediction.ts`):
```typescript
body: { question, outcomes, mode }
```

**Proposed:**
```typescript
body: {
    question,
    outcomes,
    mode,
    market_url: window.location.href,
    condition_id: selectedMarket.conditionId,
    market_prob_at_prediction: marketPriceYes,
}
```

`marketPriceYes` is already computed in `PredictionPanel.tsx` (lines 135-140) by preferring the live CLOB price, falling back to Gamma's `outcomePrices[0]`, and defaulting to 0.5:

```typescript
const marketPriceYes = parseFloat(
    clobData?.tokens?.find((t) => t.outcome === 'Yes')?.price?.toString() ??
        selectedMarket.outcomePrices[0] ??
        '0.5',
);
```

### 4.3 Why This Can't Wait

Once the extension ships to the Chrome store:
- **Changing the request body** requires a new extension version → Chrome store review (days to weeks)
- **Adding server-side storage** requires only a server deploy (minutes)

If we ship the extension without these fields, we'll have a persistent storage layer with no market context to store. We'll be able to record `{tool: prediction-online, p_yes: 0.72}` but not *which market* it was for or *what the market thought* at the time. That data is useless for measuring edge.

---

## 5. Server Changes: What the Server Must Capture <a id="5-server-changes"></a>

### 5.1 Capture `cost_dict` From Tool Output

**Current** (`tools/runner.py` lines 98-104):
```python
result = await asyncio.wait_for(coro, timeout=TOOL_TIMEOUT)
deliver_msg = result[0]
# Elements 1-3 are IGNORED
parsed = json.loads(deliver_msg)
await store.update_complete(request_id, parsed)
```

**Proposed:** Extract and persist the additional tuple elements:
```python
result = await asyncio.wait_for(coro, timeout=TOOL_TIMEOUT)
deliver_msg = result[0]
prompt_used = result[1] if len(result) > 1 else None
counter_callback = result[3] if len(result) > 3 else None
cost_dict = counter_callback.cost_dict if counter_callback else None

parsed = json.loads(deliver_msg)
# Pass metadata to persistent storage (not to Redis TTL store)
await persist_prediction(request_id, mode, question, outcomes, parsed, cost_dict, prompt_used, latency_ms, ...)
await store.update_complete(request_id, parsed)  # Redis still serves polling
```

### 5.2 Measure Latency

Wrap tool execution with timing:

```python
t0 = time.monotonic()
result = await asyncio.wait_for(coro, timeout=TOOL_TIMEOUT)
latency_ms = int((time.monotonic() - t0) * 1000)
```

### 5.3 Classify Category

Import the shared `classify_category()` function (same one used by the benchmark pipeline and trader):

```python
from benchmark.datasets.fetch_production import classify_category

category = classify_category(question)
```

This ensures the category assigned at prediction time matches what the benchmark pipeline uses for scoring.

---

## 6. Persistent Storage Options <a id="6-storage-options"></a>

We propose three options, from simplest to most robust. They are **not mutually exclusive** — Option 1 can ship immediately while Option 2 is built, and Option 3 provides crash resilience regardless.

### Option 1: Append-Only JSONL File

**What:** After each prediction completes, append one JSON line to a file on a Docker-mounted volume.

**Implementation:**
- Add a Docker volume mount to `docker-compose.yml`:
  ```yaml
  app:
    volumes:
      - prediction_data:/app/data
  volumes:
    prediction_data:
  ```
- Add ~15 lines of Python: an async function that appends to `/app/data/predictions.jsonl`
- Call it from `runner.py` after tool execution completes (success or failure)

**Advantages:**
- Simplest possible change. No new dependencies, no new services.
- JSONL is directly loadable by our benchmark pipeline (`fetch_production.py` already reads JSONL).
- Append-only = no corruption risk from concurrent writes (single writer, `asyncio` serialized).
- Human-readable, `jq`-queryable, `grep`-searchable.

**Disadvantages:**
- No indexing — querying "all predictions for tool X in category Y" requires full file scan.
- No concurrent read access guarantees while writing.
- Need manual log rotation (daily rotation with date suffix is straightforward).
- Not a real database — no aggregation, no joins, no schema enforcement.

**Data volume estimate:** At ~1KB per prediction record, 1,000 predictions/day = ~1MB/day = ~365MB/year. Trivially small.

**When to use:** Ship immediately as the first persistent layer. Can be replaced or supplemented by Option 2 later.

### Option 2: PostgreSQL

**What:** Add `postgres:16-alpine` to `docker-compose.yml`. Create a `predictions` table. Write prediction records asynchronously after tool completion.

**Implementation:**
- Add PostgreSQL service to `docker-compose.yml`:
  ```yaml
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: wildcard
      POSTGRES_USER: wildcard
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  ```
- Add `asyncpg` to `pyproject.toml` dependencies
- Create `predictions` table:
  ```sql
  CREATE TABLE predictions (
      request_id       TEXT PRIMARY KEY,
      predicted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      question         TEXT NOT NULL,
      outcomes         JSONB NOT NULL,
      mode             TEXT NOT NULL,
      tool             TEXT NOT NULL,
      model            TEXT NOT NULL,
      market_url       TEXT,
      condition_id     TEXT,
      market_prob      FLOAT,
      category         TEXT,
      p_yes            FLOAT,
      p_no             FLOAT,
      confidence       FLOAT,
      info_utility     FLOAT,
      success          BOOLEAN NOT NULL,
      error            TEXT,
      latency_ms       INTEGER,
      cost_dict        JSONB,
      prompt_used      TEXT
  );

  CREATE INDEX idx_predictions_tool ON predictions(tool);
  CREATE INDEX idx_predictions_category ON predictions(category);
  CREATE INDEX idx_predictions_predicted_at ON predictions(predicted_at);
  CREATE INDEX idx_predictions_condition_id ON predictions(condition_id);
  ```
- Add async write function using `asyncpg` connection pool
- Async insert after tool completion — does not block the prediction response

**Advantages:**
- Proper queryable database. Aggregation, filtering, joins.
- Indexed lookups: "all predictions by tool X in category Y for the last 30 days" is fast.
- Schema enforcement — invalid records fail at insert, not silently.
- Standard operational tooling (pg_dump, pg_restore, WAL archiving, replication).
- Enables future features: outcome resolution tracking, user feedback, dashboards.

**Disadvantages:**
- Adds a new service to operate (backups, monitoring, connection pooling, migrations).
- Adds a dependency (`asyncpg`) to the server.
- Schema migrations needed when fields change.
- Overkill if we only need append + periodic batch reads.

**When to use:** Build as the long-term storage layer. Can coexist with Option 1 (JSONL for immediate benchmark pipeline integration, PostgreSQL for querying and future features).

### Option 3: Redis Persistence (AOF)

**What:** Enable Redis Append-Only File (AOF) so Redis data survives restarts.

**Implementation:**
- One-line change in `docker-compose.yml`:
  ```yaml
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
  ```

**Advantages:**
- Zero code changes. Just Docker config.
- All existing Redis data (predictions, channels, rate limits) survives restarts.
- AOF provides durability — at most 1 second of data loss on crash (with `appendfsync everysec`).

**Disadvantages:**
- Predictions still expire after 1 hour (TTL is application-level, not Redis-level). AOF keeps the data on disk but Redis still evicts expired keys. This does **not** make predictions persistent beyond their TTL.
- Primarily useful for **channel state durability** (payment channels that shouldn't be lost on restart), not for prediction archiving.
- Increased disk usage from AOF log (needs periodic `BGREWRITEAOF`).

**When to use:** Enable immediately regardless of other options. It protects payment channel state from Redis crashes — this is a separate concern from prediction archiving but equally important.

### Recommendation: All Three, Phased

| Phase | What | When | Why |
|---|---|---|---|
| **Phase 0** | Enable Redis AOF (Option 3) | Now | Protects payment channel state from crashes. Zero code changes. |
| **Phase 1** | Add JSONL append (Option 1) + extension fields | Before Chrome store submission | Minimum viable persistence. Immediately usable by benchmark pipeline. |
| **Phase 2** | Add PostgreSQL (Option 2) | After launch, when query patterns are known | Long-term queryable storage. Enables dashboards, outcome tracking, feedback loops. |

---

## 7. How This Feeds the Data Loop <a id="7-data-loop"></a>

With persistent storage in place, the following data loop becomes possible:

```
User requests prediction (extension sends market context)
    ↓
MPP server runs tool, captures full metadata
    ↓
Prediction record persisted (JSONL → PostgreSQL)
    ↓
Benchmark pipeline reads production predictions
    ↓
Outcome resolution (Polymarket API, via condition_id)
    ↓
Score predictions: Brier, edge, calibration, per-tool, per-category
    ↓
Update IPFS performance CSV (Layer 3 of tool selection pipeline)
    ↓
MPP server reads updated CSV → better tool selection
    ↓
Better predictions → repeat
```

### 7.1 Outcome Resolution

With `condition_id` persisted, resolving outcomes is straightforward:

1. Query Polymarket's API: `GET https://gamma-api.polymarket.com/markets?conditionId={id}`
2. Check if `closed == true` and read `resolutionSource`
3. Map resolution to binary outcome (Yes = 1, No = 0)
4. Match back to our prediction record by `condition_id`
5. Compute Brier score: `(p_yes - outcome)^2`

Our benchmark pipeline already does this for Omen markets via subgraph queries. Extending it to Polymarket via the Gamma API is incremental work, not a new system.

### 7.2 What This Enables

| Capability | Requires | Status |
|---|---|---|
| **Production Brier scores per tool** | Predictions + outcomes | Enabled once outcomes are resolved |
| **Edge over market** | `market_prob_at_prediction` + outcomes | Enabled by extension sending market_prob |
| **Per-category tool performance** | `category` field | Server derives via `classify_category()` |
| **Actual API cost tracking** | `cost_dict` from tool output | Server captures from `counter_callback` |
| **Cost estimate validation** | `cost_dict` vs [MPP_COST_PROPOSAL.md](./MPP_COST_PROPOSAL.md) estimates | Compare real costs against our projections |
| **Prompt debugging** | `prompt_used` from tool output | Server captures from tool tuple element 1 |
| **Latency monitoring** | `latency_ms` | Server measures around tool execution |
| **Feed IPFS performance CSV** | All of the above | Benchmark pipeline reads JSONL/PostgreSQL → scores → publishes CSV |
| **Dynamic tool selection improvement** | Updated CSV → MPP server reads → better routing | Full closed loop |
