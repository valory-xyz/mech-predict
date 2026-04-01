# Benchmark Pipeline

Tools for evaluating and comparing prediction tool performance.

## Pipeline overview

```
                         ┌─────────────────────────────┐
                         │  Production (daily CI job)   │
                         │                              │
  Subgraphs ──────────►  │  fetch_production.py         │
  (Omen, Polymarket)     │  → production_log.jsonl      │
                         │  → scorer.py → scores.json   │
                         │  → analyze.py → report.md    │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │  Cached Replay (local dev)   │
                         │                              │
  production_log.jsonl   │  sweep.py                    │
  ──────────────────►    │  → enrich with source_content│
                         │  → replay_dataset.jsonl      │
                         │  → runner.py (baseline)      │
                         │  → runner.py (candidate)     │
                         │  → compare.py → delta table  │
                         └─────────────────────────────┘
```

## Scripts

| Script | What it does | When to use |
|--------|-------------|-------------|
| `datasets/fetch_production.py` | Pulls predictions from subgraphs, matches to resolved markets | Daily CI or manual data refresh |
| `scorer.py` | Computes Brier, accuracy, sharpness from production_log | After fetch, or on replay results |
| `analyze.py` | Generates markdown report from scores | After scoring |
| `runner.py` | Replays questions through tools with cached web content | Testing model/prompt/code changes |
| `compare.py` | Diffs two scores.json files into a delta table | After two scoring runs |
| `sweep.py` | Orchestrates the full replay pipeline in one command | Main developer workflow |

## How fetch_production.py works

```
1. Query marketplace subgraphs for recent deliveries
   → Each delivery has: question, tool, model, prediction (p_yes/p_no), timestamps

2. Query prediction subgraphs for resolved markets
   → Each market has: question, outcome (true/false), resolution timestamp

3. Match deliveries ↔ markets by question text or market_id
   → Produces scored rows with prediction + outcome

4. Append to production_log.jsonl (incremental, dedup by row_id)
```

**Key fields in production_log.jsonl:**

```json
{
  "row_id": "prod_omen_abc123",
  "deliver_id": "0xabc...",
  "question_text": "Will Bitcoin exceed $100,000?",
  "tool_name": "prediction-online",
  "model": "gpt-4.1-2025-04-14",
  "p_yes": 0.72,
  "p_no": 0.28,
  "prediction_parse_status": "valid",
  "final_outcome": true,
  "platform": "omen",
  "category": "crypto"
}
```

**CLI flags:**

```bash
# Standard daily fetch
python benchmark/datasets/fetch_production.py

# Fetch last 30 days
python benchmark/datasets/fetch_production.py --lookback-days 30

# Only keep last 100 rows in output
python benchmark/datasets/fetch_production.py --last-n 100

# Custom output path
python benchmark/datasets/fetch_production.py --output my_log.jsonl
```

## How cached replay works

The core idea: prediction tools accept a `source_content` parameter. When provided, the tool skips live web search and uses the cached content instead. This means two runs with different models see **identical evidence** — the only variable is the model's reasoning.

```
source_content provided?
  ├─ Yes → tool uses cached pages/PDFs, skips Google/Serper
  └─ No  → tool fetches live web content (normal production behavior)
```

**What gets cached:**
- Web-fetching tools (prediction-online, etc.): `{"pages": {"url": "html"}, "pdfs": {"url": "text"}}`
- Superforcaster: `{"serper_response": {"organic": [...], "peopleAlsoAsk": [...]}}`

## Developer workflow

### Prerequisites

1. **Get production_log.jsonl** — download from CI artifacts (daily benchmark workflow) or generate:
   ```bash
   python benchmark/datasets/fetch_production.py --lookback-days 30
   ```

2. **Set API keys** (needed for replay — tools make real LLM calls):
   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

### Workflow 1: Compare two models (sweep)

```bash
python benchmark/sweep.py \
  --last-n 50 \
  --baseline-model gpt-4o \
  --candidate-model gpt-4.1-2025-04-14 \
  --tools prediction-online,superforcaster
```

What happens:
1. Reads last 50 rows from `datasets/production_log.jsonl`
2. Looks up IPFS hashes for each row's `deliver_id`
3. Fetches `source_content` from IPFS gateway
4. Writes `results/sweep_replay_dataset.jsonl` (reusable)
5. Runs baseline model → scores
6. Runs candidate model → scores
7. Prints comparison table

### Workflow 2: Reuse existing replay dataset

After the first sweep, reuse the dataset (skips IPFS fetching):

```bash
python benchmark/sweep.py \
  --dataset benchmark/results/sweep_replay_dataset.jsonl \
  --baseline-model gpt-4o \
  --candidate-model gpt-4.1-2025-04-14
```

### Workflow 3: Test tool code changes

```bash
# 1. Run baseline with current code
python benchmark/runner.py \
  --dataset replay_dataset.jsonl \
  --tools superforcaster \
  --model gpt-4.1-2025-04-14 \
  --output results/before.jsonl

python benchmark/scorer.py --input results/before.jsonl --output results/before_scores.json

# 2. Make your code change to the tool

# 3. Run candidate with modified code
python benchmark/runner.py \
  --dataset replay_dataset.jsonl \
  --tools superforcaster \
  --model gpt-4.1-2025-04-14 \
  --output results/after.jsonl

python benchmark/scorer.py --input results/after.jsonl --output results/after_scores.json

# 4. Compare
python benchmark/compare.py --baseline results/before_scores.json --candidate results/after_scores.json
```

### Workflow 4: Just compare two existing score files

```bash
python benchmark/compare.py --baseline scores_old.json --candidate scores_new.json
```

## What you can and can't test

| Change | Cached replay? | Why |
|--------|---------------|-----|
| Switch model | Yes | Same evidence, different reasoning |
| Change prompt | Yes | Same evidence, different prompt |
| Change reasoning logic | Yes | Same evidence, different processing |
| Compare two tools | Yes | Same questions, different tools |
| Change search queries | No | Search is bypassed |
| Change num_words (cleaned mode) | No | Truncation baked in at capture time |

## Example output

```
## Overall
|                     |  B.Brier |  C.Brier |    Delta |    B.Acc |    C.Acc |    Delta |     N | Direction  |
|---------------------|----------|----------|----------|----------|----------|----------|-------|------------|
| Overall             |   0.2217 |   0.0358 |  -0.1859 |   0.6667 |   1.0000 |  +0.3333 |     3 | improved   |

## By Tool
|                     |  B.Brier |  C.Brier |    Delta |    B.Acc |    C.Acc |    Delta |     N | Direction  |
|---------------------|----------|----------|----------|----------|----------|----------|-------|------------|
| prediction-online   |   0.3100 |   0.2200 |  -0.0900 |   0.5600 |   0.6500 |  +0.0900 |   120 | improved   |
| superforcaster      |   0.2300 |   0.2400 |  +0.0100 |   0.7300 |   0.7200 |  -0.0100 |    85 | regressed  |
```

Lower Brier is better (0 = perfect). Higher accuracy is better. Delta shows candidate minus baseline.

## File locations

```
benchmark/
├── datasets/
│   ├── fetch_production.py      # pulls data from subgraphs
│   ├── production_log.jsonl     # scored predictions (gitignored)
│   ├── sample_replay.jsonl      # sample data for testing
│   └── .fetch_state.json        # incremental fetch cursor (gitignored)
├── results/                     # output directory (gitignored)
│   ├── sweep_replay_dataset.jsonl
│   ├── sweep_baseline_*.jsonl
│   ├── sweep_candidate_*.jsonl
│   └── *_scores.json
├── runner.py                    # cached replay runner
├── scorer.py                    # Brier/accuracy computation
├── analyze.py                   # markdown report generator
├── compare.py                   # score delta comparison
└── sweep.py                     # orchestrator
```
