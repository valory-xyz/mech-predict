# Benchmark Implementation Notes

Living document tracking what's been built, decisions made, and divergences from `PROPOSAL.md`.

## What's Built

### `benchmark/datasets/fetch_open.py`

Script to fetch open markets and snapshot web content for cached replay.

**Platforms:**
- `--platform omen` — queries `omen.subgraph.autonolas.tech` for `fixedProductMarketMakers`, filtered to 2 creators (Pearl + QS)
- `--platform polymarket` — queries Gamma API (`gamma-api.polymarket.com/markets`), filtered to 10 categories, binary, non-resolved, non-neg-risk, non-zero liquidity
- `--platform all` — both

**Tool-specific search groups:**

The proposal assumed a single `source_links` per market. We found that tools search differently, so we capture per-group data:

| Group | Tools | What's captured |
|-------|-------|----------------|
| A (superforcaster) | Raw question → 1 Serper call | `serper_response.json` (snippets + PAA) |
| B (factual_research) | LLM → 3-6 sub-questions → N Serper + scrape | `source_links.json` (extracted text, 400 words/page) |
| C (rag, reasoning, sme, url_cot) | LLM → 5 queries → N Serper + scrape | `source_links.json` (raw HTML — tools call `extract_text(html=)` internally) |

Groups B and C use LLM (GPT-4.1) to generate queries matching exact tool prompts. Group A uses raw question.

**Output structure:**
```
benchmark/datasets/
  open_markets.jsonl
  snapshots/{market_id}/
    metadata.json
    group_a/serper_response.json, queries.json
    group_b/serper_responses.json, queries.json, source_links.json
    group_c/serper_responses.json, queries.json, source_links.json
```

**Resilience:**
- Incremental — skips markets with existing snapshots (matching requested groups)
- JSONL preserved across interrupted runs
- Empty snapshots not written on error
- Serper rate limit detection with exponential backoff (10s/20s/40s), clean stop on quota exhaustion

## Decisions & Divergences from Proposal

### 1. Per-tool-group snapshots (not in proposal)

The proposal specified a single `source_links` per market. We split into groups because:
- Tools generate **different search queries** for the same market question
- factual_research expects **extracted text** in `source_links`; Group C tools expect **raw HTML**
- superforcaster doesn't accept `source_links` at all — needs raw Serper snippets

### 2. Omen market source

The proposal mentions "Omen subgraph" generically. We use `omen.subgraph.autonolas.tech` (Autonolas proxy to the full Omen FPMM subgraph on Gnosis), filtered to the 2 market creators that the trader bets on. This gives ~135 markets with rich data (prices, volume, liquidity, category).

The `predict-omen` subgraph (`api.subgraph.staging.autonolas.tech/api/proxy/predict-omen`) was considered first but only indexes bets (not markets directly) and lacks liquidity/volume fields.

### 3. Polymarket volume

Polymarket has 10k+ open markets even in a 4-day window (unfiltered). The `--min-liquidity` flag is essential for Polymarket (e.g., `--min-liquidity 1000`). With $1k+ filter, ~342 binary markets remain. For Omen, liquidity is in single-digit USD so the filter isn't useful — creator filtering handles quality instead.

### 4. Tools that can't use cached replay

These tools don't accept `source_links` and can only be evaluated via tournament mode (live search):
- **superforcaster** — Group A captures its Serper snippets, but the tool itself doesn't have a `source_links` parameter
- **prediction_langchain** — uses Tavily (different search API), agent-driven queries
- **corcel_request**, **gemini_prediction** — no web search at all (LLM-only)

### 5. Order book for Polymarket (planned, not yet built)

The plan includes capturing CLOB order book data (bid/ask/spread) per Polymarket market via `py-clob-client`. This enables `market_prob_type: bid/ask` in the proposal schema. Not yet implemented.

## What's Not Built Yet

From the proposal's architecture:
- `fetch_resolved.py` — pull resolved markets for ground truth
- `fetch_production.py` — pull production predictions from on-chain data
- `runner.py` — cached replay benchmark runner
- `tournament.py` — forward-looking tournament runner
- `scorer.py` — Brier, edge, calibration scoring
- `compare.py`, `sweep.py`, `search.py`, `promote.py` — iteration tools

## API Costs Per Run (Omen, 135 markets, all groups)

| Resource | Calls | Notes |
|----------|-------|-------|
| Serper | ~1,755 | 1 (group A) + 6 (group B) + 6 (group C) per market |
| OpenAI GPT-4.1 | ~270 | 2 per market (groups B + C query generation) |
| Page scrapes | ~1,350 | ~10 unique URLs per market (deduped across groups) |

Serper free tier: 2,500 queries/month — barely fits one full run.
