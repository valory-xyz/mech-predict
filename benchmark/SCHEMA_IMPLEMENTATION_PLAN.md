# Schema Implementation Plan

Based on [SCHEMA_PROPOSAL.md](SCHEMA_PROPOSAL.md). This document breaks the schema changes into shipping phases, identifies where each field's data comes from, and lists the files that need modification.

---

## Phase 1: Mech Response Enrichment

**Goal:** Add `schema_version`, `executed_at`, `metadata.tool_hash`, `metadata.execution_latency_ms` to the mech response payload.

**Risk:** Low — purely additive, self-contained in mech-predict, no cross-repo dependency.

### Resolved decisions

1. **Failure path enrichment:** Yes — `schema_version` and `executed_at` appear on both success and failure responses. `metadata.tool_hash` and `metadata.execution_latency_ms` only appear in the success path (where the metadata dict exists).
2. **`executed_at` capture point:** Captured at the start of `_handle_done_task()` — reflects "task execution is done", not IPFS upload time.
3. **Schema version constant:** Module-level constant `RESPONSE_SCHEMA_VERSION = "2.0"` in `behaviours.py`, not inline string.
4. **`datetime` API:** Use `datetime.now(timezone.utc)` (modern form), not deprecated `datetime.utcnow()`.
5. **Test location:** Add tests to existing `packages/valory/skills/task_execution/tests/test_behaviours.py` using the established `capture_store` / `stored_payloads` pattern from conftest fixtures.

### Data sources

| Field | Source | Already available? | Location |
|-------|--------|--------------------|----------|
| `schema_version` | Constant `RESPONSE_SCHEMA_VERSION = "2.0"` | — | New constant in `behaviours.py` |
| `executed_at` | `datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")` at start of `_handle_done_task()` | Yes (timing already calculated) | `behaviours.py:~740` |
| `metadata.tool_hash` | `self._tools_to_package_hash.get(tool)` | Yes (dict already loaded) | `behaviours.py:231`, `models.py:80-82` |
| `metadata.execution_latency_ms` | `int(tool_exec_time_duration * 1000)` (reuse existing computation) | Yes (duration already computed at L797-799, just not serialized) | `behaviours.py:797-799` |

### Files to modify

| Repo | File | Change |
|------|------|--------|
| mech | `packages/valory/skills/task_execution/behaviours.py` | Add `from datetime import datetime, timezone` import. Add `RESPONSE_SCHEMA_VERSION = "2.0"` constant. In `_handle_done_task()`: capture `executed_at` at method entry, add `schema_version` + `executed_at` to base response dict (L755), add `tool_hash` + `execution_latency_ms` to metadata dict (L774-778). Move `tool_exec_time_duration` computation before metadata dict construction. |
| mech | `packages/valory/skills/task_execution/tests/test_behaviours.py` | Add tests for Phase 1 schema fields using existing `capture_store`/`stored_payloads` pattern |

**Note:** `task_execution` is a third-party synced package in mech-predict. Changes must be made in the **mech** repo and synced via `autonomy packages sync`.

### Implementation detail

**Current response structure (success path):**
```json
{
  "requestId": 42,
  "result": "prediction text",
  "tool": "prediction-online",
  "prompt": "Will BTC hit $100k?",
  "cost_dict": {"input": 10},
  "metadata": {"model": "gpt-4o", "tool": "prediction-online", "params": {}},
  "is_offchain": false
}
```

**After Phase 1 (success path):**
```json
{
  "schema_version": "2.0",
  "requestId": 42,
  "result": "prediction text",
  "tool": "prediction-online",
  "prompt": "Will BTC hit $100k?",
  "cost_dict": {"input": 10},
  "metadata": {
    "model": "gpt-4o",
    "tool": "prediction-online",
    "params": {},
    "tool_hash": "bafybei...",
    "execution_latency_ms": 3450
  },
  "is_offchain": false,
  "executed_at": "2026-03-26T14:35:42.123456Z"
}
```

**After Phase 1 (failure path):**
```json
{
  "schema_version": "2.0",
  "requestId": 42,
  "result": "Invalid response",
  "tool": "prediction-online",
  "executed_at": "2026-03-26T14:35:42.123456Z"
}
```

### Code change walkthrough

In `_handle_done_task()`:

1. **At method entry (before any other logic):** Capture `executed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")`
2. **At base response construction (L755):** Change from `{"requestId": req_id, "result": result_msg, "tool": tool}` to include `"schema_version": RESPONSE_SCHEMA_VERSION` and `"executed_at": executed_at`
3. **Move `tool_exec_time_duration` computation (currently L797-799) up** before the metadata dict, so `execution_latency_ms` is available
4. **In metadata dict (L774-778):** Add `"tool_hash": self._tools_to_package_hash.get(tool)` and `"execution_latency_ms": int(tool_exec_time_duration * 1000)`
5. **After L805 (reset timing):** No change needed — timing reset stays where it is

### Edge cases

- Tool hash lookup: tools not in `_tools_to_package_hash` don't execute (already guarded), so `.get(tool)` always returns a value; using `.get()` as belt-and-suspenders
- `executed_at` must be UTC ISO 8601 with `Z` suffix — enforced by `.replace("+00:00", "Z")`
- `tool_execution_start_time` can be `0.0` if `_prepare_task` was not called — existing `or time.perf_counter()` fallback yields ~0ms, which is acceptable (task didn't truly execute)
- Failure path: `schema_version` and `executed_at` always present; no metadata dict so no `tool_hash`/`execution_latency_ms`

### Tests

Tests go in `packages/valory/skills/task_execution/tests/test_behaviours.py`, using the existing `capture_store` + `stored_payloads` pattern (see e.g. `test_happy_path_executes_and_stores` at L52).

| Test | What it catches |
|------|-----------------|
| `test_handle_done_task_success_response_contains_schema_version` | Success response has `schema_version: "2.0"` |
| `test_handle_done_task_success_response_contains_executed_at` | Success response has `executed_at` in ISO 8601 UTC with `Z` suffix |
| `test_handle_done_task_success_metadata_contains_tool_hash` | `metadata.tool_hash` matches the hash from `_tools_to_package_hash` |
| `test_handle_done_task_success_metadata_contains_execution_latency_ms` | `metadata.execution_latency_ms` is a positive int |
| `test_handle_done_task_failure_response_contains_schema_version_and_executed_at` | Failure path still has `schema_version` + `executed_at` |
| `test_handle_done_task_failure_response_has_no_metadata` | Failure path does NOT have metadata (no spurious `tool_hash`/`latency`) |
| Mutation: change `* 1000` to `* 1` → `execution_latency_ms` test fails | Wrong unit conversion |
| Mutation: remove `tool_hash` from metadata → test fails | Regression detection |
| Mutation: change `[0]` index to `[1]` on `outcomeTokenMarginalPrices` → Phase 2 test fails | Wrong probability index (Phase 2 scope, listed for completeness) |

---

## Phase 2: Trader Request Enrichment (Detailed)

**Goal:** Add `schema_version` and `request_context` to the trader's IPFS request payload.

**Risk:** Medium — cross-repo change (trader), but most data is already on the `Bet` object.

### How the current request flow works

1. `DecisionRequestBehaviour.setup()` (decision_request.py:64-71) reads `sampled_bet` and builds a `MechMetadata(prompt, tool, nonce)`.
2. `MechMetadata` is a 3-field dataclass (mech_interact_abci/states/base.py:71-77): `prompt`, `tool`, `nonce`.
3. In `async_act()` (decision_request.py:98-99), `MechMetadata` is serialized via `dataclasses.asdict()` → JSON → sent to IPFS (request.py:582).
4. `sampled_bet` is a `Bet` object (bets.py:139-167) retrieved by index from the bets list (behaviours/base.py:277-281).
5. The `Bet.market` field is set at construction time: `Bet(**raw_bet, market=self._current_market)` — in update_bets.py:176 for Omen, polymarket_fetch_market.py:332 for Polymarket. The value is the subgraph/client key from `creator_per_subgraph` config (e.g., `"omen_subgraph"`, or whatever key is configured for Polymarket).

### Field-by-field analysis

#### `market_id` — VIABLE, zero effort

| Platform | Value | Source |
|----------|-------|--------|
| Omen | `bet.id` — the FPMM contract address (e.g., `"0xabc..."`) | Subgraph `id` field (omen.py:38) |
| Polymarket | `bet.id` — set from `market.get("id")` in the Gamma API response | polymarket_fetch_market.py:385,448 |

Both platforms already populate `bet.id`. Direct read, no transformation needed.

**Note:** For Polymarket, `bet.id` is the Gamma API market ID (a numeric string like `"504911"`), not the `conditionId`. The schema proposal says `market_id` should be `conditionId` for Polymarket. The `conditionId` is available as `bet.condition_id` (bets.py:165, set at polymarket_fetch_market.py:451). **Decision needed:** use `bet.id` or `bet.condition_id` for Polymarket? The `conditionId` is the canonical on-chain identifier and matches what the proposal says. Recommend using `bet.condition_id` for Polymarket and `bet.id` for Omen.

#### `type` — VIABLE, trivial mapping

| Platform | `bet.market` value | Maps to |
|----------|-------------------|---------|
| Omen | `"omen_subgraph"` (from skill.yaml:163-164 `creator_per_subgraph`) | `"omen"` |
| Polymarket | The key used in `creator_per_subgraph` for Polymarket (need to verify exact value) | `"polymarket"` |

Simple string mapping. The `bet.market` value comes from the iterator key in `creator_per_subgraph` config. For Omen it's `"omen_subgraph"`. For Polymarket, `_current_market` is set the same way — need to check the Polymarket service config to confirm the key name.

**Implementation:** A dict mapping `{"omen_subgraph": "omen", "polymarket_client": "polymarket"}` or similar. Alternatively, a method on `Bet` that derives the platform type.

#### `market_prob` — VIABLE, already available

| Platform | Value | Source |
|----------|-------|--------|
| Omen | `bet.outcomeTokenMarginalPrices[0]` | Subgraph field (omen.py:46) |
| Polymarket | `bet.outcomeTokenMarginalPrices[0]` | Parsed from Gamma API `outcomePrices` (polymarket_fetch_market.py:394,458) |

Both platforms populate `outcomeTokenMarginalPrices` as `List[float]`. Index `[0]` is the "Yes" probability.

**When is this price from, and how stale is it?** The `Bet` object's prices come from the **market manager refresh** — the `UpdateBetsBehaviour` (Omen) or `PolymarketFetchMarketBehaviour` (Polymarket) that runs at the start of each period. The composed FSM flow from market refresh to profitability decision is (from composition.py:166-169):

```
MarketManager refresh (prices fetched here)
  → SamplingRound (bet picked)
    → ToolSelectionRound
      → DecisionRequestRound (MechMetadata built — request_context attached here)
        → MechRequestRound (upload to IPFS, build on-chain tx)
          → PreTxSettlementRound (prepare safe tx)
            → TransactionSettlement (submit on-chain)
              → MechResponseRound (poll/wait for mech delivery — timeout: 300s per skill.yaml:202)
                → DecisionReceiveRound (profitability check — uses SAME cached prices)
```

**The real gap is NOT sub-minute.** Between market refresh and `DecisionReceive`, the trader must: go through sampling/tool selection, upload to IPFS, submit an on-chain transaction, wait for the mech to poll/receive/execute/deliver, then poll for the mech's response. The `response_timeout` is **300 seconds** (5 minutes) per mech_interact_abci/skill.yaml:202. Real-world total gap is likely **2-10 minutes** depending on blockchain confirmation times, mech execution time, and polling intervals.

**The profitability calculation also uses stale prices.** `DecisionReceiveBehaviour._is_profitable()` (decision_receive.py:523-524) uses the same cached `bet.outcomeTokenMarginalPrices` — it does NOT re-fetch. This means the entire trading pipeline currently operates on stale prices. This is arguably a separate issue — should the profitability check re-fetch? Maybe. But that's outside the scope of the schema change.

**Could we fetch fresh prices at DecisionRequest time?**

| Option | Feasibility | Effort | Notes |
|--------|-------------|--------|-------|
| Fetch in `setup()` | No | — | `setup()` is synchronous — network calls need `yield from` in `async_act()` |
| Fetch in `async_act()` | Possible | Medium | Would need to restructure the behaviour to do an async API call before building metadata |
| Add a new round before DecisionRequest | Possible | High | New round in the FSM to fetch latest prices, store in synchronized data |
| Fetch in DecisionReceive (for profitability) | Possible | Medium | More impactful — would fix the stale price problem for the actual trading decision too |

**Verdict for request_context:** Use the cached price. Reasons:

1. **Consistency** — `request_context.market_prob` should reflect what the trader knew when it asked the mech. The cached price IS what the trader used to decide to ask. If we fetch fresh prices just for the schema but the trading decision was made on stale prices, we'd be lying to the benchmark about the trader's information state.
2. **Correctness for edge-over-market** — The benchmark's edge-over-market metric asks "was the tool better than what the market said when the trader decided to bet?" The cached price is the right answer to that question.
3. **The staleness problem is real but separate** — If the trader should re-fetch prices before checking profitability, that's a trading improvement worth discussing, but it's independent of the schema change. The schema should capture what actually happened, not what should have happened.
4. **Retroactive precision available** — If the benchmark needs higher fidelity, it can look up the actual price at `executed_at` time from subgraph/CLOB history. The cached price is a good-enough anchor for V1.

**Worth flagging as a future improvement:** The trader's profitability check (DecisionReceive) should probably re-fetch prices before deciding to bet. The market can move significantly during the 2-10 minute mech interaction window. This is a real trading risk that's currently unaddressed — but it's a trader improvement, not a schema concern.

#### `market_liquidity_usd` — VIABLE, but units need verification

| Platform | Value | Source | Unit |
|----------|-------|--------|------|
| Omen | `bet.scaledLiquidityMeasure` | Subgraph field (omen.py:48) | xDai (≈ USD) |
| Polymarket | `bet.scaledLiquidityMeasure` | Gamma API `liquidity` field parsed as float (polymarket_fetch_market.py:419,462) | USD |

Both platforms store this in `scaledLiquidityMeasure`. For Omen, this is in xDai (a stablecoin pegged to USD). For Polymarket, the Gamma API `liquidity` field is in USD. Both are approximately USD-denominated, so no conversion needed.

**Gotcha:** Omen's `scaledLiquidityMeasure` represents the liquidity parameter of the AMM, not total volume. It's a reasonable proxy for market depth but not identical to "total liquidity in the pool."

#### `market_close_at` — VIABLE, needs timestamp normalization

| Platform | Value | Source | Format |
|----------|-------|--------|--------|
| Omen | `bet.openingTimestamp` | Subgraph field (omen.py:43) | Unix epoch (int) |
| Polymarket | `bet.openingTimestamp` | Parsed from Gamma API `endDate` via `date_parser.isoparse(end_date).timestamp()` (polymarket_fetch_market.py:412-416) | Unix epoch (int, converted from ISO) |

Both platforms store as Unix epoch int in `bet.openingTimestamp`. Need to convert to ISO 8601 UTC string:
```python
datetime.fromtimestamp(bet.openingTimestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")
```

**Naming confusion:** On Omen, `openingTimestamp` is the market close/resolution time (when answers can be submitted), not when the market was created. On Polymarket, it's derived from `endDate`. Both represent the same concept — when the market resolves. The field name in the Bet dataclass is misleading but the data is correct for our purposes.

#### `market_spread` — NOT VIABLE without additional work

**Current state:** The trader does NOT fetch or store spread data for Polymarket. The Gamma API `/markets` endpoint is called (connection.py:538) but only the following fields are used from the response: `id`, `question`, `outcomes`, `outcomePrices`, `clobTokenIds`, `liquidity`, `endDate`, `conditionId`, `submitted_by`, `closed`, `createdAt`. No `spread`, `bestBid`, or `bestAsk` fields are read.

**What the Gamma API provides:** The Gamma API `/markets` response does include a `spread` field (decimal value like `0.02`). However:
1. The trader doesn't currently read it
2. The `Bet` dataclass has no field for it
3. Adding it means modifying the Polymarket connection, the fetch behaviour, and the Bet dataclass

**Options:**

| Option | Effort | Risk |
|--------|--------|------|
| A. Add `spread` to Polymarket fetch + `Bet` dataclass | Medium — 3 files, but touches data pipeline | Low — additive field |
| B. Compute from `outcomePrices`: `spread ≈ abs(price_yes + price_no - 1)` | Zero — already have the data | Inaccurate — `outcomePrices` are mid-prices, not bid/ask |
| C. Defer to Phase 2b | Zero | None — benchmark can hardcode a typical spread for V1 |

**Recommendation: Defer (Option C).** The `market_spread` is only used in Stage 5 PnL simulation, which is the last benchmark stage. The benchmark can use a reasonable default (e.g., 2 cents) or compute it retroactively from CLOB history. Adding `spread` to the data pipeline is a real change — new field on `Bet`, new field read in the connection, new field carried through serialization — and it's not worth blocking the core `request_context` implementation for it.

### What we ship in Phase 2

| Field | Ship? | Complexity |
|-------|-------|------------|
| `schema_version` | Yes | Hardcoded `"2.0"` |
| `request_context.market_id` | Yes | `bet.condition_id or bet.id` |
| `request_context.type` | Yes | Mapping from `bet.market` |
| `request_context.market_prob` | Yes | `bet.outcomeTokenMarginalPrices[0]` |
| `request_context.market_liquidity_usd` | Yes | `bet.scaledLiquidityMeasure` |
| `request_context.market_close_at` | Yes | `datetime.fromtimestamp(bet.openingTimestamp, UTC)` → ISO 8601 |
| `request_context.market_spread` | **Deferred** | Requires new data pipeline work |

### Files to modify

| Repo | File | Change | Complexity |
|------|------|--------|------------|
| trader | `packages/valory/skills/mech_interact_abci/states/base.py:71-77` | Add optional `schema_version: str = "2.0"` and `request_context: Optional[Dict] = None` fields to `MechMetadata` dataclass | Low |
| trader | `packages/valory/skills/market_manager_abci/bets.py` | Add `to_request_context() -> Dict` method on `Bet` class | Low |
| trader | `packages/valory/skills/decision_maker_abci/behaviours/decision_request.py:64-71` | Call `sampled_bet.to_request_context()` and pass to `MechMetadata` | Low |

### Detailed implementation notes

#### 1. `MechMetadata` dataclass change (states/base.py)

```python
@dataclass
class MechMetadata:
    """A Mech's metadata."""
    prompt: str
    tool: str
    nonce: str
    schema_version: str = "2.0"
    request_context: Optional[Dict[str, Any]] = None
```

**Key consideration:** `MechMetadata` is serialized via `dataclasses.asdict()` at request.py:582. Adding optional fields with defaults means:
- Old code that constructs `MechMetadata(prompt, tool, nonce)` still works (backward compatible)
- `asdict()` will include `schema_version` and `request_context` in the output
- If `request_context` is `None`, it will serialize as `null` in JSON — we should either exclude it or accept `null`

**Decision:** Use a custom `to_dict()` method or post-process `asdict()` to exclude `None` values, so old-format requests don't carry a `"request_context": null` field. Or accept `null` — the mech and benchmark should handle both `null` and missing.

**Impact on `SynchronizedData.mech_requests`** (states/base.py:353-358): This property deserializes `MechMetadata` from JSON via `MechMetadata(**metadata_item)`. The new optional fields with defaults mean deserialization of old payloads (without `schema_version`/`request_context`) will still work. But the reverse is also true — if a newer payload has extra fields, `MechMetadata(**metadata_item)` will accept them because of the default values. This is clean.

**Impact on `BetsDecoder`** (bets.py:415-447): The decoder uses exact field matching to distinguish `Bet` from `PredictionResponse`. Adding `to_request_context()` as a method (not a field) doesn't affect serialization.

#### 2. `Bet.to_request_context()` method (bets.py)

```python
# Mapping from bet.market config key to platform type
MARKET_TO_PLATFORM = {
    "omen_subgraph": "omen",
    # Add Polymarket key when confirmed
}

def to_request_context(self) -> Optional[Dict[str, Any]]:
    """Build request_context for the benchmark schema."""
    platform = MARKET_TO_PLATFORM.get(self.market)
    if platform is None:
        return None

    context = {
        "market_id": self.condition_id or self.id,
        "type": platform,
        "market_prob": self.outcomeTokenMarginalPrices[0] if self.outcomeTokenMarginalPrices else None,
        "market_liquidity_usd": self.scaledLiquidityMeasure,
        "market_close_at": datetime.fromtimestamp(self.openingTimestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    # Strip None values
    return {k: v for k, v in context.items() if v is not None}
```

**Why on `Bet` and not in `DecisionRequestBehaviour`:** The `Bet` object has all the data. Putting the method there keeps it testable in isolation and avoids leaking market structure knowledge into the decision maker.

**Why `condition_id or id`:** For Omen, `condition_id` is `None` (not set in the subgraph query) and `id` is the FPMM address — the canonical market identifier. For Polymarket, `condition_id` is the on-chain condition ID (set at polymarket_fetch_market.py:451), and `id` is the Gamma API numeric ID. The schema wants the on-chain identifier, so `condition_id` is preferred when available, falling back to `id`.

#### 3. `DecisionRequestBehaviour.setup()` change (decision_request.py)

```python
def setup(self) -> None:
    sampled_bet = self.sampled_bet
    prompt_params = dict(question=sampled_bet.title, yes=sampled_bet.yes, no=sampled_bet.no)
    prompt = self.params.prompt_template.substitute(prompt_params)
    tool = self.synchronized_data.mech_tool
    nonce = str(uuid4())
    request_context = sampled_bet.to_request_context()
    self._metadata = MechMetadata(prompt, tool, nonce, request_context=request_context)
```

Minimal change — one line to build context, one kwarg to pass it.

### What adds too much complexity (excluded)

1. **`market_spread` from CLOB** — Requires modifying the Polymarket connection to read `spread` from Gamma API response, adding a field to `Bet`, threading it through serialization. Three files across two packages for one optional benchmark field. Deferred.

2. **Real-time `market_prob` at request time** — Would require a fresh subgraph/API call at request time instead of using cached market data. Adds latency and complexity to the request path for marginal accuracy improvement. Not worth it.

3. **Custom serialization for `MechMetadata`** — Could implement `__json__` or a custom encoder to strip `None` fields. But `asdict()` + JSON already works, and consumers should handle `null`. Not worth the complexity.

4. **Adding `request_context` to `MechInteractionResponse`** — The mech response doesn't need to echo back the request context. The benchmark can join request and response by `requestId`.

### What's not viable

1. **Getting `market_spread` from Omen** — AMMs don't have a bid-ask spread. The Omen fee (~2%) is a protocol constant. This is correctly excluded from Omen's `request_context`.

2. **Getting real-time order book depth** — The Polymarket CLOB has depth data, but fetching it per-request would add significant latency and payload size. Deferred per schema proposal.

### Edge cases

- **Blacklisted bets** (`outcomes is None`): `to_request_context()` still works — `outcomeTokenMarginalPrices` may be stale but non-null. However, blacklisted bets shouldn't reach `DecisionRequestBehaviour` (they're filtered in sampling). Belt-and-suspenders: check `self.outcomes is not None` in `to_request_context()`.
- **Zero liquidity bets** (`scaledLiquidityMeasure == 0`): These are blacklisted in `_check_usefulness()` (bets.py:281-284) and won't be sampled. Safe to ignore.
- **Missing `outcomeTokenMarginalPrices`**: Would only happen if the bet was blacklisted during validation. Guard with a length check.
- **`openingTimestamp` of 0**: Would produce `"1970-01-01T00:00:00Z"`. This would indicate bad data — should return `None` for `market_close_at` in this case.
- **Old mechs receiving new requests**: Mechs only read `prompt` and `tool` from the request payload. Extra fields are ignored. Verified in mech-predict's `behaviours.py` — request parsing destructures only known keys.
- **Consensus**: `MechMetadata` is serialized to JSON and shared across agents via `participant_to_requests`. All agents must produce the same JSON for consensus. Since `request_context` is deterministic from `sampled_bet` (which is agreed upon via `sampled_bet_index` in synchronized data), all agents will produce identical payloads. No consensus risk.

### Tests

| Test | What it catches |
|------|-----------------|
| `to_request_context()` returns correct keys for an Omen bet | Missing or wrong field mapping |
| `to_request_context()` returns correct keys for a Polymarket bet (with `condition_id`) | Wrong market ID source |
| `to_request_context()` returns `None` for unknown platform | Crash on unexpected `bet.market` value |
| `to_request_context()` omits `None` values | Serialization of null fields |
| `MechMetadata` with `request_context` serializes correctly via `asdict()` | Broken serialization |
| `MechMetadata` without `request_context` serializes without the field (or with `null`) | Backward compatibility |
| `market_close_at` is ISO 8601 UTC for both platforms | Timestamp format mismatch |
| `market_close_at` handles `openingTimestamp=0` gracefully | Bad data edge case |
| Mutation: flip `outcomeTokenMarginalPrices[0]` to `[1]` → test should fail | Wrong probability index |

### Concrete example: end-to-end flow

**Scenario:** Trader is running Polymarket strategy. Market manager just refreshed and fetched a BTC market.

#### Step 1: Market Manager refreshes (PolymarketFetchMarketBehaviour)

Gamma API returns (among many fields):
```json
{
  "id": "504911",
  "question": "Will BTC hit $100k by June?",
  "conditionId": "0xdef456abc...",
  "outcomes": "[\"Yes\",\"No\"]",
  "outcomePrices": "[\"0.65\",\"0.35\"]",
  "liquidity": "450000",
  "endDate": "2026-06-30T00:00:00Z",
  "clobTokenIds": "[\"71321...\",\"71322...\"]",
  "closed": false
}
```

This gets transformed into a `Bet` object (polymarket_fetch_market.py:447-473):
```python
Bet(
    id="504911",                                    # Gamma API numeric ID
    market="polymarket_client",                     # from _current_market
    title="Will BTC hit $100k by June?",
    condition_id="0xdef456abc...",                   # on-chain condition ID
    outcomeTokenMarginalPrices=[0.65, 0.35],        # parsed from outcomePrices
    scaledLiquidityMeasure=450000.0,                # from liquidity field (USD)
    openingTimestamp=1751241600,                     # parsed from endDate → Unix epoch
    outcomeTokenAmounts=[292500000, 157500000],      # liquidity * price * 10^6
    outcomes=["Yes", "No"],
    collateralToken="0x2791Bcc1...",                 # USDC.e on Polygon
    ...
)
```

#### Step 2: SamplingRound picks this bet

`sampled_bet_index` is set in synchronized data. All agents agree on this index.

#### Step 3: DecisionRequestBehaviour.setup() builds the mech request

Currently (before our change):
```python
sampled_bet = self.sampled_bet  # Bet object from step 1
prompt = "Will BTC hit $100k by June? Yes or No?"
tool = "prediction-online"
nonce = "a1b2c3d4-..."
self._metadata = MechMetadata(prompt, tool, nonce)
```

Serialized via `asdict()` → uploaded to IPFS:
```json
{"nonce": "a1b2c3d4-...", "prompt": "Will BTC hit $100k by June? Yes or No?", "tool": "prediction-online"}
```

After our change:
```python
sampled_bet = self.sampled_bet
request_context = sampled_bet.to_request_context()
# Returns:
# {
#     "market_id": "0xdef456abc...",    ← bet.condition_id (preferred for Polymarket)
#     "type": "polymarket",             ← mapped from bet.market="polymarket_client"
#     "market_prob": 0.65,              ← bet.outcomeTokenMarginalPrices[0]
#     "market_liquidity_usd": 450000.0, ← bet.scaledLiquidityMeasure
#     "market_close_at": "2026-06-30T00:00:00Z"  ← from bet.openingTimestamp
# }
self._metadata = MechMetadata(prompt, tool, nonce, request_context=request_context)
```

Serialized → uploaded to IPFS:
```json
{
  "schema_version": "2.0",
  "prompt": "Will BTC hit $100k by June? Yes or No?",
  "tool": "prediction-online",
  "nonce": "a1b2c3d4-...",
  "request_context": {
    "market_id": "0xdef456abc...",
    "type": "polymarket",
    "market_prob": 0.65,
    "market_liquidity_usd": 450000.0,
    "market_close_at": "2026-06-30T00:00:00Z"
  }
}
```

#### Step 4: Mech receives request, executes tool, returns prediction

The mech reads `prompt` and `tool` from the IPFS payload. It ignores `request_context` entirely — the extra fields are transparent to the mech's request parsing.

#### Step 5: DecisionReceiveBehaviour checks profitability

Uses `bet.outcomeTokenMarginalPrices[predicted_vote_side]` (decision_receive.py:523-524) — the **same cached prices** that went into `request_context.market_prob`. This is consistent: the benchmark sees what the trader saw.

#### Step 6: Benchmark reads the IPFS payload later

```python
request = fetch_from_ipfs(request_hash)
version = request.get("schema_version", "1.0")

if version == "2.0" and "request_context" in request:
    ctx = request["request_context"]
    market_prob = ctx["market_prob"]        # 0.65
    mech_prob = parse_result(response)      # e.g., 0.72
    edge = mech_prob - market_prob          # +0.07 → mech was more bullish than market
```

#### Same flow for Omen

Key differences:
- `bet.id` = `"0xabc123..."` (FPMM contract address) — used as `market_id` (no `condition_id` on Omen)
- `bet.market` = `"omen_subgraph"` → `type: "omen"`
- `bet.outcomeTokenMarginalPrices` from subgraph, same field
- `bet.scaledLiquidityMeasure` in xDai (≈ USD)
- `bet.openingTimestamp` directly from subgraph (already Unix epoch)
- No `market_spread` field

### Cross-check: verified claims

| Claim | Verified? | How |
|-------|-----------|-----|
| `MechMetadata` has 3 fields: `prompt`, `tool`, `nonce` | Yes | Read states/base.py:71-77 |
| Serialized via `dataclasses.asdict()` | Yes | Read decision_request.py:52, request.py:582 |
| `sampled_bet` is a `Bet` object from `self.bets[index]` | Yes | Read behaviours/base.py:277-281 |
| `Bet.market` set from `self._current_market` | Yes | Read update_bets.py:176, polymarket_fetch_market.py:332 |
| Omen subgraph queries `scaledLiquidityMeasure` | Yes | Read omen.py:48 |
| Polymarket maps `liquidity` → `scaledLiquidityMeasure` | Yes | Read polymarket_fetch_market.py:419,462 |
| Polymarket maps `outcomePrices` → `outcomeTokenMarginalPrices` | Yes | Read polymarket_fetch_market.py:394,458 |
| Polymarket stores `conditionId` as `condition_id` on Bet | Yes | Read polymarket_fetch_market.py:451 |
| Omen does NOT query `condition_id` (field is `None`) | Yes | Read omen.py:24-51 — not in query fields |
| Gamma API response has no `spread`/`bestBid`/`bestAsk` used | Yes | Grep found zero matches in polymarket_client/ |
| Prices are NOT re-fetched between market refresh and decision | Yes | Composed FSM (composition.py:166-169): MarketManager → Sampling → DecisionRequest → MechRequest → TxSettlement → MechResponse → DecisionReceive — no re-fetch anywhere in chain |
| `_is_profitable` uses same cached `bet.outcomeTokenMarginalPrices` | Yes | Read decision_receive.py:523-524 — uses `bet.outcomeTokenMarginalPrices[predicted_vote_side]` directly |
| Gap between market refresh and profitability check is 2-10 min, not sub-minute | Yes | Includes on-chain tx + mech execution + mech response polling. `response_timeout` alone is 300s (skill.yaml:202) |

### Open questions to resolve before implementation

1. **What is the exact `bet.market` value for Polymarket?** The config key in `creator_per_subgraph` — need to check the Polymarket service YAML. Code uses `self._current_market` which comes from iterating `creator_per_subgraph` keys. Likely `"polymarket_client"` but needs confirmation.
2. **Should `request_context: null` be serialized or omitted?** If we use `asdict()` as-is, `None` becomes `null` in JSON. Cleaner to omit it, but requires post-processing. Recommend: accept `null` for simplicity — consumers check `if "request_context" in payload and payload["request_context"]`.
3. **Should `market_id` use `bet.condition_id` (preferred, on-chain) or `bet.id` (always present)?** Recommendation: `condition_id` when available, fall back to `id`. For Omen, `condition_id` is `None` so we use `id` (FPMM address). For Polymarket, `condition_id` is the on-chain condition — use it.

---

## Phase 3: Tool `metadata.params` Enrichment

**Goal:** Populate `metadata.params` with actual runtime kwargs (`temperature`, `max_tokens`, `num_urls`, `num_queries`) instead of static defaults from `component.yaml`.

**Risk:** Medium — touches how params flow through execution, but no external contract changes.

### Data sources

Each tool's `run()` function receives runtime kwargs. The mech's `_prepare_task()` builds `task_data` with these values. Currently, `metadata.params` is populated from `component.yaml` static config (`tool_params`), not from what was actually used.

### Files to modify

| Repo | File | Change |
|------|------|--------|
| mech | `packages/valory/skills/task_execution/behaviours.py` (~L774) | Merge actual `task_data` kwargs into `metadata.params` instead of only using static `tool_params` |

### Edge cases

- **Security:** Some kwargs contain API keys — must filter to an allowlist of safe param names (`temperature`, `max_tokens`, `num_urls`, `num_queries`, `default_model`) before serializing
- Different tools use different params — the set varies per tool, which is fine (schema says "tools record whichever kwargs they actually used")

### Tests

- `metadata.params` contains runtime values, not just static defaults
- API keys and sensitive kwargs are excluded from `metadata.params`

---

## Phase 4: Tool `source_content` Return

**Goal:** Have tools return scraped web content separately as `metadata.source_content` (URL → text dict) instead of only embedding it in the prompt string.

**Risk:** Higher — changes the tool return contract, requires modifying every tool that does web search.

### Data sources

Inside each tool's `run()` function. Tools already fetch URLs and have the raw content — they format it into the prompt and discard the mapping. The change is to preserve and return it.

### Design decision: how to pass `source_content` back

The tool return type is currently a 5-tuple: `(deliver_msg, prompt, transaction, counter_callback, keychain)`. Options:

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A | Add 6th tuple element | Clean separation | Breaks all existing tools that return 5 |
| B | Embed in `transaction` dict | No tuple change | Overloads `transaction` semantics |
| C | Attach to `counter_callback` object | No tuple change | Hacky, mixes concerns |
| D | Return a dict/dataclass instead of tuple | Future-proof | Largest refactor |

**Recommendation:** Start with option A for 1-2 pilot tools. Guard the unpacking in `behaviours.py` with a length check for backward compatibility (`if len(result) >= 6`). Migrate remaining tools incrementally.

### Files to modify

| Repo | File | Change |
|------|------|--------|
| mech-predict | Each tool in `packages/*/customs/*/` that does web search | Preserve URL → content mapping and return as 6th element |
| mech | `packages/valory/skills/task_execution/behaviours.py` `_handle_done_task()` | Extract `source_content` from result tuple, add to `metadata` |

### Rollout strategy

1. Pilot with `prediction-online` and `prediction-offline` tools
2. Validate the pipeline end-to-end
3. Migrate remaining tools

### Tests

- Tool returns `source_content` dict with correct URL keys
- `behaviours.py` handles both 5-tuple (old tools) and 6-tuple (new tools) gracefully
- `metadata.source_content` is present in response when tool provides it, absent when it doesn't

---

## Phase 5: Benchmark Consumer

**Goal:** Update benchmark code to read new schema fields with graceful `"1.0"` fallback.

**Risk:** None to production — benchmark is read-only analysis.

### Files to modify

| Repo | File | Change |
|------|------|--------|
| mech-predict | `benchmark/` directory | Read `schema_version`, branch logic for `"1.0"` vs `"2.0"` payloads, use new fields for metrics |

### Fallback behavior

- Missing `schema_version` → treat as `"1.0"`
- Missing `request_context` → skip edge-over-market, mark as lower provenance
- Missing `metadata.tool_hash` → skip version tracking for that prediction
- Missing `metadata.source_content` → fall back to prompt parsing (current behavior)

---

## Execution Order

```
Phase 1 (mech response)
  → Phase 2 (trader request)
    → Phase 3 (params enrichment)
      → Phase 4 (source_content — pilot then rollout)
        → Phase 5 (benchmark consumer)
```

Phase 1 is self-contained and can ship independently. Phase 2 requires a trader release. Phases 3-4 are progressively more invasive. Phase 5 can be developed in parallel with any phase once the schema is agreed.

---

## Cross-Repo Dependency Map

```
Phase 1: mech only (task_execution skill) → sync to mech-predict
Phase 2: trader (produces) → mech-predict (ignores gracefully, benchmark reads)
Phase 3: mech only (task_execution skill) → sync to mech-predict
Phase 4: mech (behaviours) + mech-predict (tools in customs/)
Phase 5: mech-predict only (benchmark/)
```
