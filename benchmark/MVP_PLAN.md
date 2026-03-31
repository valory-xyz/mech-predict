# Production Replay Benchmark Pipeline — MVP Plan

## Goal

Build a first Production Replay Benchmark Pipeline focused on:
- **Forecasting metrics**, starting with Brier score
  - For a binary outcome, Brier = `(p_yes - final_outcome)²`
  - Measures how close the predicted probability was to what actually happened
  - Lower is better (0 = perfect, 0.25 = random guessing)
- **Per-tool and per-platform aggregates**

## Why This Approach

We are starting with a narrower production replay benchmark because:
- It is already feasible with the current on-chain data and gives us useful signal quickly
- This lets us deliver a reliable first benchmark layer before expanding into the broader replay/search/promotion loop proposed in the full benchmark proposal
- We removed the focus on cached replay for now because we need better planning given the concerns raised around content storage, trust model, and schema changes

After this work, we can define how to feed the report back into AI-assisted improvement loops.

---

## What We're Building

```
Daily (automated via GitHub Actions):
  fetch_production.py → production_log.jsonl → scorer.py → scores.json → analyze.py → report.md
```

Three scripts, one GitHub Actions workflow. All automated, no manual runs.

---

## Scripts

### 1. `benchmark/datasets/fetch_production.py`

Associates request → delivery → market → final outcome.

**What it does:**
- Queries predict-omen + predict-polymarket-agents subgraphs (adapts existing queries from `mech-interact/tool_accruacy.py` and `random-valory-scripts/polymarket/get_polymarket_agents_accuracy_and_roi.py`)
- For each bet on a resolved market, finds the last mech request before the bet (existing "last mech request before bet placement" attribution logic used by both platforms)
- Downloads mech response from IPFS gateway, parses `result` field for `p_yes`, `p_no`
- Extracts `tool`, `model`, `cost_dict` from response metadata
- Records block timestamp as `predicted_at`
- Classifies question category via keyword matching
- Tracks state in `.fetch_state.json` (last processed bet ID per platform) for incremental runs
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
  "match_confidence": 0.95
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

### 2. `benchmark/scorer.py`

Reads the joined data and computes the score for each row.

**Metrics:**
- Brier score: `mean((p_yes - outcome)²)` — lower is better, 0.25 = random guessing
- Reliability: `valid_outputs / total_rows` — hard gate at 80%
- Per-tool breakdown
- Per-platform breakdown (Polymarket vs Omen)
- Per-category breakdown (crypto, politics, sports, etc.)
- Per-time-horizon breakdown (short <7d, medium 7-30d, long >30d)
- Trend: Brier score over time (monthly buckets)

**Output (`scores.json`):**
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
  "by_platform": {
    "omen": {"brier": 0.240, "reliability": 0.93, "n": 300},
    "polymarket": {"brier": 0.221, "reliability": 0.95, "n": 170}
  },
  "by_category": { "...": "..." },
  "by_horizon": {
    "short_lt_7d": {"brier": 0.142, "n": 45},
    "medium_7_30d": {"brier": 0.201, "n": 89},
    "long_gt_30d": {"brier": 0.247, "n": 66}
  },
  "trend": [
    {"month": "2026-01", "brier": 0.225, "n": 65},
    {"month": "2026-02", "brier": 0.229, "n": 78},
    {"month": "2026-03", "brier": 0.238, "n": 92}
  ]
}
```

---

### 3. `benchmark/analyze.py`

Generates a report of the results. Human-readable markdown.

**What it highlights:**
- Overall health: reliability, Brier, sample size
- Tool ranking: which tool is best overall, per platform, per category
- Top 10 worst predictions (highest Brier) with question text, tool, what was predicted vs what happened
- Top 10 best predictions where tool got it right
- Weak spots: categories or platforms where Brier > 0.30 (worse than random guessing)
- Reliability issues: tools with >10% malformed/timeout/error rate
- Trend alerts: if Brier worsened by >0.02 in the last month vs prior month
- Sample size warnings: categories with <20 questions flagged as low-confidence

**Output (`report.md`):**
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
- politics: Brier 0.312 (n=73) — all tools struggle here
- prediction-online on omen: Brier 0.289 (n=45)

## Worst Predictions
1. "Will X happen?" — prediction-online predicted 0.85, outcome: No (Brier: 0.72)
   Category: politics, Platform: omen, Lead time: 5 days
2. ...

## Trend
- Jan 2026: 0.225 → Feb 2026: 0.229 → Mar 2026: 0.238 (degrading)
```

---

### 4. GitHub Actions Workflow

Embedded in CI, runs daily.

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
├── results/                       # gitignored
│   ├── scores.json
│   └── report.md
```

---

## Known Shortcomings

### 1. No edge-over-market metric

Can't calculate "does our tool beat the market consensus" because the trader doesn't send `market_prob` at request time. We only have Brier score (absolute accuracy) and reliability.

**Impact:** Brier alone can't tell us if a tool is profitable — a tool that always agrees with the market scores well on Brier but generates zero trading edge.

**When it's fixed:** When trader request schema changes land (`request_context.market_prob`).

### 2. Question-to-market matching is fragile for Omen

String prefix matching between mech subgraph (truncated titles) and predict-omen subgraph (full titles). Can produce false matches.

**Impact:** Some predictions may be matched to the wrong market outcome, corrupting Brier scores.

**Mitigation built in:** `match_confidence` field — exact match = 1.0, prefix match = 0.8. Scorer can filter to high-confidence matches only.

**When it's fixed:** When trader sends `market_id` in request payload.

### 3. IPFS response parsing may fail for old formats

Older mech responses may have different JSON structure or unparseable result strings.

**Impact:** Some historical predictions can't be scored — reduces dataset size.

**Mitigation built in:** Classify parse status: `valid` / `malformed` / `missing_fields`. Malformed rows count against reliability. Parse failure rate is logged.

**When it's fixed:** `schema_version` in responses (landing in current PR).

### 4. No statistical significance testing

When comparing numbers across tools, small differences might be noise.

**Impact:** Risk of drawing conclusions from random variation.

**Mitigation:** Report sample size alongside every number. With 200+ predictions, differences >0.03 Brier are likely real. Below that, treat with caution.

**When it's fixed:** Add `compare.py` with paired bootstrap testing.

---

## What Comes Next (After MVP)

Once the production replay pipeline is running and producing daily reports, the next steps are:

1. **Cached replay for testing fixes** — replay production questions with a different tool/prompt/model config using cached web content, compare Brier against baseline. Requires resolving the content storage approach (push scraped content to IPFS at prediction time, reference by CID at replay time).

2. **Tournament** — forward-looking predictions on open markets. No content storage problem (tools hit live web, markets haven't resolved yet). Gives temporally clean out-of-sample evaluation.

3. **AI-assisted improvement loops** — feed the analyze.py report into automated search (parameter sweeps, prompt evolution). The report highlights where tools are weakest, the search loop tries to fix those weaknesses.

4. **Edge-over-market** — once the trader sends `market_prob` in requests, add edge calculation to the scorer. This is the most important metric for trading value.

5. **Promotion pipeline** — canary deployment, rollback gates, production monitoring. Requires all of the above working first.
