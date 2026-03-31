# MVP Benchmark Pipeline — Implementation Plan

## What We're Building

```
Daily (automated via GitHub Actions):
  fetch_production.py → production_log.jsonl → scorer.py → scores.json → analyze.py → report.md

On-demand (human triggers when there's a fix to test):
  runner.py (replay with fix) → scorer.py (compare) → did the fix help?
```

---

## Scripts

### 1. `fetch_production.py` (~2 hours)

Incremental production data fetcher. Runs daily, appends new rows.

**What it does:**
- Queries predict-omen + predict-polymarket-agents subgraphs (adapts existing queries from `tool_accruacy.py` and `get_polymarket_agents_accuracy_and_roi.py`)
- For each bet on a resolved market, finds the last mech request before the bet (existing attribution logic from both platforms)
- Downloads mech response from IPFS gateway, parses `result` field for `p_yes`, `p_no`
- Extracts URLs and source content from the `prompt` field:
  - For tools using `ARTICLE N, URL: ..., CONTENT: ...` format → extracts URL + article text
  - For superforcaster using `**Title:** ... **Snippet:** ...` format → extracts Serper-style snippets
- Extracts `tool`, `model`, `cost_dict` from response metadata
- Records block timestamp as `predicted_at`
- Classifies question category via keyword matching (see below)
- Tracks state in `.fetch_state.json` (last processed bet ID per platform)
- Appends new rows to `production_log.jsonl`

**Row schema:**
```json
{
  "row_id": "prod_001",
  "platform": "omen",
  "question_text": "Will BTC hit $100k by June?",
  "tool_name": "prediction-online",
  "model": "gpt-4.1-2025-04-14",
  "p_yes": 0.72,
  "p_no": 0.28,
  "prediction_parse_status": "valid",
  "final_outcome": true,
  "predicted_at": "2026-01-10T14:23:00Z",
  "resolved_at": "2026-02-15T00:00:00Z",
  "prediction_lead_time_days": 36,
  "category": "crypto",
  "input_tokens": 4200,
  "output_tokens": 850,
  "cost_usd": 0.042,
  "match_confidence": 0.95,
  "source_urls": ["https://reuters.com/...", "https://bbc.com/..."],
  "source_format": "article" | "serper_snippet"
}
```

**Category classification (built-in, keyword-based):**
```python
CATEGORY_KEYWORDS = {
    "crypto": ["bitcoin", "btc", "eth", "ethereum", "crypto", "token", "defi", "blockchain"],
    "politics": ["president", "election", "vote", "congress", "senate", "parliament", "minister"],
    "sports": ["win", "championship", "league", "cup", "match", "tournament", "team"],
    "economics": ["gdp", "inflation", "fed", "interest rate", "unemployment", "recession"],
    "tech": ["ai", "openai", "google", "apple", "microsoft", "launch", "release"],
    "geopolitics": ["war", "conflict", "sanctions", "treaty", "nato", "invasion"],
}
# Falls back to "other" if no keywords match
```

**Dependencies:**
- `THE_GRAPH_API_KEY` (already in GitHub secrets)
- IPFS gateway (public: `gateway.autonolas.tech`)
- Mech safe address(es) (need from team)

---

### 2. `scorer.py` (~1 hour)

Reads `production_log.jsonl`, computes metrics, outputs `scores.json`.

**Metrics:**
- Brier score: `mean((p_yes - outcome)²)` — lower is better
- Reliability: `valid_outputs / total_rows` — hard gate at 80%
- Per-tool breakdown
- Per-platform breakdown (Polymarket vs Omen)
- Per-category breakdown (crypto, politics, sports, etc.)
- Per-time-horizon breakdown (short <7d, medium 7-30d, long >30d)
- Trend: Brier score over time (monthly buckets)

**Output:**
```json
{
  "generated_at": "2026-03-31T06:00:00Z",
  "total_rows": 500,
  "valid_rows": 470,
  "overall": {"brier": 0.231, "reliability": 0.94},
  "by_tool": {
    "prediction-online": {"brier": 0.228, "reliability": 0.95, "n": 200},
    "superforcaster": {"brier": 0.218, "reliability": 0.96, "n": 150}
  },
  "by_platform": { ... },
  "by_category": { ... },
  "by_horizon": { ... },
  "trend": [ ... ]
}
```

---

### 3. `analyze.py` (~1 hour)

Reads `scores.json` + `production_log.jsonl`, generates human-readable `report.md`.

**What it highlights:**
- Overall health: reliability, Brier, sample size
- Top 10 worst predictions (highest Brier) with question text, tool, and what went wrong
- Top 10 best predictions where tool got it right and market was uncertain
- Tool ranking: which tool is best overall, per platform, per category
- Weak spots: categories or platforms where Brier > 0.30 (worse than random)
- Reliability issues: tools with >10% malformed/timeout/error rate
- Trend alerts: if Brier worsened by >0.02 in the last month vs prior month
- Sample size warnings: categories with <20 questions flagged as low-confidence

**Output format** (`report.md`):
```markdown
# Benchmark Report — 2026-03-31

## Overall
- Predictions scored: 470 / 500 (94% reliability)
- Overall Brier: 0.231

## Tool Ranking
1. superforcaster — Brier: 0.218 (n=150)
2. prediction-online — Brier: 0.228 (n=200)
3. prediction-rag — Brier: 0.235 (n=120)

## Weak Spots
⚠ politics: Brier 0.312 (n=73) — all tools struggle here
⚠ prediction-online on omen: Brier 0.289 (n=45)

## Worst Predictions
1. "Will X happen?" — prediction-online predicted 0.85, outcome: No (Brier: 0.72)
   Category: politics, Platform: omen, Lead time: 5 days
2. ...

## Trend
- Jan 2026: 0.225 → Feb 2026: 0.229 → Mar 2026: 0.238 ↑ (degrading)
```

---

### 4. `runner.py` — Replay (~1.5 hours)

Replays a subset of production questions with a different tool/prompt/model config. Uses `source_links` to inject cached content so tools don't hit the live web.

**What it does:**
- Takes a config file (tool, model, temperature, prompt template, etc.)
- Takes a subset of questions from `production_log.jsonl` (or all of them)
- For each question:
  - If `source_format == "article"`: re-fetches the `source_urls`, builds `source_links = {url: html_content}`
  - If `source_format == "serper_snippet"`: parses stored snippets from original prompt, builds `source_links` in Serper format
  - Runs the tool with `source_links` injected
  - Enforces 240s timeout (matching production `TASK_DEADLINE`)
- Records new `p_yes`, `p_no`, latency, cost
- Outputs `replay_results.jsonl`
- Runs `scorer.py` on replay results
- Prints side-by-side comparison: production baseline vs replay

**Usage:**
```bash
# Test a new prompt template
python benchmark/runner.py \
  --config '{"tool": "prediction-online", "model": "gpt-4.1", "temperature": 0.2}' \
  --dataset benchmark/datasets/production_log.jsonl \
  --limit 50

# Output:
#                  Production    Replay      Delta
# Brier            0.228         0.195       -0.033 ↓ (better)
# Reliability      95%           96%         +1%
# n                50            50
```

---

### 5. GitHub Actions Workflow (~30 min)

```yaml
# .github/workflows/benchmark_flywheel.yaml
name: benchmark-flywheel
on:
  schedule:
    - cron: '0 6 * * *'  # daily at 6am UTC
  workflow_dispatch: {}   # manual trigger

jobs:
  benchmark:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install requests
      - name: Fetch production data
        env:
          THE_GRAPH_API_KEY: ${{ secrets.THE_GRAPH_API_KEY }}
        run: python benchmark/datasets/fetch_production.py
      - name: Score
        run: python benchmark/scorer.py
      - name: Analyze
        run: python benchmark/analyze.py
      - name: Commit results
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add benchmark/results/ benchmark/datasets/production_log.jsonl benchmark/datasets/.fetch_state.json
          git diff --staged --quiet || git commit -m "chore: daily benchmark update"
          git push
```

---

## File Structure

```
benchmark/
├── datasets/
│   ├── fetch_production.py
│   ├── .fetch_state.json          # gitignored — tracks last processed state
│   └── production_log.jsonl       # gitignored — append-only prediction log
├── scorer.py
├── analyze.py
├── runner.py
├── results/                       # gitignored
│   ├── scores.json
│   └── report.md
└── prompts/
    └── templates.py               # prompt variants for replay testing
```

---

## Known Shortcomings We're Accepting

### 1. No edge-over-market metric

Can't calculate "does our tool beat the market consensus" because the trader doesn't send `market_prob` at request time. We only have Brier score (absolute accuracy) and reliability.

**Impact:** Brier alone can't tell us if a tool is profitable — a tool that always agrees with the market scores well on Brier but generates zero trading edge.

**When it's fixed:** When trader request schema changes land (`request_context.market_prob`).

### 2. Replay uses re-fetched URLs, not original content

~10-20% of URLs may return different content than what the tool originally saw. Some will be dead links.

**Impact:** Replay comparisons have noise. A tool might score differently because the content changed, not because the config is better/worse.

**Mitigation built in:** `runner.py` tracks URL fetch success rate per question. Questions where >30% of URLs fail are flagged in results. Scorer can filter them out.

**When it's fixed:** When `source_content` is added to the IPFS response schema (stores the actual scraped content at prediction time).

### 3. Question-to-market matching is fragile for Omen

String prefix matching between mech subgraph (truncated titles) and predict-omen subgraph (full titles). Can produce false matches.

**Impact:** Some predictions may be matched to the wrong market outcome, corrupting Brier scores.

**Mitigation built in:** `match_confidence` field — exact match = 1.0, prefix match = 0.8. Scorer can filter to high-confidence matches only.

**When it's fixed:** When trader sends `market_id` in request payload.

### 4. No statistical significance testing

When comparing replay vs baseline, we just compare numbers. Small improvements might be noise.

**Impact:** Risk of promoting a change that isn't actually better.

**Mitigation:** Report sample size alongside every number. With 200+ predictions, improvements >0.03 Brier are likely real. Below that, treat with caution.

**When it's fixed:** Add `compare.py` with paired bootstrap testing (proposal Part 5).

### 5. IPFS response parsing may fail for old formats

Older mech responses may have different JSON structure or unparseable result strings.

**Impact:** Some historical predictions can't be scored — reduces dataset size.

**Mitigation built in:** Classify parse status: `valid` / `malformed` / `missing_fields`. Malformed rows count against reliability. Log parse failure rate.

**When it's fixed:** `schema_version` in responses (landing in current PR).

---

## Build Order

1. `fetch_production.py` — the data foundation, everything depends on this
2. `scorer.py` — produces the numbers
3. `analyze.py` — makes numbers human-readable, highlights issues
4. GitHub Actions workflow — automates steps 1-3 daily
5. `runner.py` — replay for testing fixes (can come a few days later)

---

## What's NOT in MVP (Proposal Parts Deferred)

| Proposal Part | What | Why deferred |
|--------------|------|-------------|
| Part 2: Tournament | Forward-looking predictions on open markets | Requires waiting for resolution. Add in week 2. |
| Part 5: Statistical testing | Bootstrap significance tests | Manual number comparison is fine for MVP |
| Part 7: Automated search | Parameter sweeps, prompt evolution | Needs replay working first |
| Part 8: Ablation testing | Component-by-component testing | Needs search working first |
| Part 9: Human review tool | Structured review sets with cause taxonomy | `analyze.py` report covers 80% of this |
| Part 10: Promotion pipeline | Canary deployment, rollback gates | Way downstream |
| Part 12: IPFS accuracy publication | Publish accuracy hashes for traders | After pipeline is proven |
