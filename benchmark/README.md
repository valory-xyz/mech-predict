# Benchmark Pipeline

Tools for evaluating and comparing prediction tool performance.

## How it works — the big picture

Prediction mechs receive questions from on-chain markets (Omen, Polymarket), search the web
for evidence, and output a probability (`p_yes`). Once the market resolves, we know the
actual outcome and can score each prediction with the **Brier score** (0 = perfect, 1 = worst).

This pipeline collects those predictions, matches them to outcomes, and lets you compare
different tool code, models, or prompts against the same evidence.

```
  On-chain markets                    Subgraphs                    Benchmark pipeline
  ┌──────────────┐                ┌──────────────────┐          ┌─────────────────────┐
  │ Mech receives│  delivers +    │ Omen, Polymarket, │  fetch   │ production_log.jsonl│
  │ request,     │──predictions──>│ Marketplace       │────────>│ (question, p_yes,   │
  │ predicts,    │                │ subgraphs         │         │  outcome, tool, ...) │
  │ delivers     │                └──────────────────┘          └────────┬────────────┘
  └──────────────┘                                                      │
                                                                        ▼
                                                              ┌─────────────────────┐
                                                              │ scorer.py           │
                                                              │ → Brier, accuracy,  │
                                                              │   calibration       │
                                                              └────────┬────────────┘
                                                                       │
                                                                       ▼
                                                              ┌─────────────────────┐
                                                              │ analyze.py          │
                                                              │ → markdown report   │
                                                              └─────────────────────┘
```

## Two modes of evaluation

### 1. Production scoring (daily CI)

Fetches real predictions and outcomes from subgraphs, scores them. This measures how
the mech is performing in production right now.

```bash
python benchmark/datasets/fetch_production.py    # fetch + match + score
python benchmark/analyze.py                       # generate report
```

### 2. Cached replay (local dev — sweep.py)

**This is the developer workflow.** You change tool code, switch a model, or tweak a prompt.
Then you replay the same questions with the same web evidence, and compare your change
against what production actually predicted.

```
  production_log.jsonl
  (past predictions with known outcomes)
          │
          ▼
  ┌─ sweep.py ─────────────────────────────────────────────┐
  │                                                         │
  │  1. Filter: last N rows matching your tool              │
  │  2. Enrich: fetch source_content from IPFS              │
  │     (the cached web pages/search results used           │
  │      by the mech when it originally predicted)          │
  │  3. Score baseline: production predictions on            │
  │     the enriched rows (same questions as candidate)     │
  │  4. Replay candidate: run your modified tool code       │
  │     with cached source_content (no live web fetch)      │
  │  5. Score candidate                                     │
  │  6. Compare: print delta table                          │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

**Key insight:** Both baseline and candidate see **identical evidence** (cached web content).
The only variable is what you changed — code, model, or prompt.

### 3. Tournament mode (forward-looking)

Runs predictions on **currently open markets** with live web search. Evidence is captured
and stored for future cached replay. Markets are scored later when they resolve.

```bash
python benchmark/tournament.py --tools superforcaster --model gpt-4.1-2025-04-14
# ... wait for markets to resolve ...
python benchmark/score_tournament.py
```

Tournament mode is useful for building replay datasets with `source_content` for future
sweep comparisons.

## source_content: cached web evidence

When a tool predicts, it searches the web (Google/Serper) and extracts text from pages.
The `source_content` field captures this evidence so it can be replayed deterministically.

**Two storage modes:**

| Mode | What's stored | Size | Replay tests | Default |
|------|--------------|------|--------------|---------|
| `cleaned` | Extracted text only | ~2 KB/page | LLM + prompt only | Yes (production) |
| `raw` | Full HTML | ~100-300 KB/page | Full pipeline (including extraction) | No (tournament only) |

**Cleaned mode** (production default): 98.9% smaller, but extraction is frozen at capture time.
You can test model changes, prompt changes, and reasoning logic — but not search query
changes or text extraction changes.

**Raw mode** (tournament): Full HTML stored, so replay re-extracts text. Tests the complete
pipeline including extraction logic. Higher storage cost.

The `source_content` dict always includes a `mode` marker:
```json
{
  "mode": "cleaned",
  "pages": {"https://example.com": "extracted text here..."},
  "pdfs": {"https://example.com/doc.pdf": "extracted text from PDF"}
}
```

For superforcaster (which uses Serper API instead of web scraping):
```json
{
  "mode": "cleaned",
  "serper_response": {"organic": [...], "peopleAlsoAsk": [...]}
}
```

## Developer workflow

### Prerequisites

1. **Get production_log.jsonl** — download from CI artifacts or generate:
   ```bash
   python benchmark/datasets/fetch_production.py --lookback-days 30
   ```

2. **Set API keys** (tools make real LLM calls during replay):
   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...   # only for claude-* tools
   ```

### Test a tool code change

```bash
# 1. Make your change
vim packages/valory/customs/superforcaster/superforcaster.py

# 2. Run sweep
python benchmark/sweep.py \
  --last-n 500 \
  --tools superforcaster \
  --candidate-model gpt-4.1-2025-04-14

# 3. Read the comparison table
```

### Test a model change

```bash
# Same workflow — just change the model
python benchmark/sweep.py \
  --last-n 500 \
  --tools superforcaster \
  --candidate-model claude-4-sonnet-20250514
```

The baseline comes from the production log (whatever model was running). The candidate
uses your specified model. Both see the same cached evidence.

### Test both code + model change

```bash
# Change the tool code, then:
python benchmark/sweep.py \
  --last-n 500 \
  --tools prediction-request-rag \
  --candidate-model gpt-4.1-2025-04-14
```

The comparison shows the combined effect of code + model change vs production.

### Reuse an existing replay dataset

After the first run, the enriched dataset is saved. Skip IPFS fetching on subsequent runs:

```bash
python benchmark/sweep.py \
  --dataset benchmark/results/sweep_replay_dataset.jsonl \
  --tools superforcaster \
  --candidate-model gpt-4.1-2025-04-14
```

### Compare specific tools

```bash
python benchmark/sweep.py \
  --last-n 500 \
  --tools prediction-online,superforcaster \
  --candidate-model gpt-4.1-2025-04-14
```

## What you can and can't test with cached replay

| Change | Testable? | Why |
|--------|-----------|-----|
| Switch model | Yes | Same evidence, different reasoning |
| Change prompt | Yes | Same evidence, different prompt |
| Change reasoning logic | Yes | Same evidence, different processing |
| Change temperature | Yes | Edit tool code, same evidence |
| Compare two tools | Yes | Same questions, different tools |
| Change search queries | No | Search is bypassed in replay |
| Change num_words (cleaned mode) | No | Truncation baked in at capture time |
| Change text extraction (cleaned mode) | No | Extraction baked in at capture time |
| Change text extraction (raw mode) | Yes | Raw HTML re-extracted during replay |

## Scripts reference

| Script | What it does | When to use |
|--------|-------------|-------------|
| `sweep.py` | Full pipeline: filter → enrich → baseline → replay → compare | Main developer workflow |
| `runner.py` | Replays questions through tools with cached web content | Called by sweep, or standalone |
| `scorer.py` | Computes Brier, accuracy, sharpness from prediction logs | Called by sweep, or standalone |
| `compare.py` | Diffs two scores.json files into a delta table | Called by sweep, or standalone |
| `analyze.py` | Generates markdown report from scores | After scoring |
| `tournament.py` | Live predictions on open markets, captures evidence | Building replay datasets |
| `score_tournament.py` | Scores tournament predictions after resolution | After markets resolve |
| `datasets/fetch_production.py` | Pulls predictions + outcomes from subgraphs | Daily CI or manual refresh |
| `datasets/fetch_open.py` | Fetches currently open markets | For tournament mode |

## Example output

```
## Overall
|                     |  B.Brier |  C.Brier |    Delta |    B.Acc |    C.Acc |    Delta |   B.N |   C.N | Direction  |
|---------------------|----------|----------|----------|----------|----------|----------|-------|-------|------------|
| Overall             |   0.2217 |   0.0358 |  -0.1859 |   0.6667 |   1.0000 |  +0.3333 |     3 |     3 | improved   |

## By Tool
|                     |  B.Brier |  C.Brier |    Delta |    B.Acc |    C.Acc |    Delta |   B.N |   C.N | Direction  |
|---------------------|----------|----------|----------|----------|----------|----------|-------|-------|------------|
| superforcaster      |   0.2300 |   0.2400 |  +0.0100 |   0.7300 |   0.7200 |  -0.0100 |    85 |    85 | regressed  |
```

- **B.Brier / C.Brier**: Baseline / Candidate Brier score (lower is better, 0 = perfect)
- **B.Acc / C.Acc**: Baseline / Candidate accuracy (higher is better)
- **B.N / C.N**: Sample sizes (should match for fair comparison)
- **Delta**: Candidate minus baseline (negative Brier delta = improvement)
- **Direction**: `improved` / `regressed` / `unchanged`

## Scoring metrics — formulas

All formulas below match the implementations in `scorer.py` and `ci_replay.py`.

### Primary metrics (used for tool ranking)

**Brier score** — measures probabilistic forecast accuracy. Lower is better.

```
Per prediction:  brier_i = (p_yes - outcome)²
Aggregate:       Brier   = mean(brier_i)  over all valid predictions

outcome = 1.0 if Yes, 0.0 if No
```

| Value | Meaning |
|-------|---------|
| 0.0 | Perfect — predicted exactly the outcome |
| yes_rate × (1 - yes_rate) | No skill — equivalent to always predicting the base rate. Only equals 0.25 when outcomes are balanced (50/50). See Baseline Brier below. |
| 1.0 | Worst — maximally wrong on every prediction |

**Reliability** — fraction of attempted runs that produced a valid, parseable prediction.

```
Reliability = valid_outputs / attempted_runs
```

A row is "valid" when `prediction_parse_status == "valid"`, `final_outcome` is not null, and `p_yes` is not null. Tools below 80% reliability are flagged with a warning in the report. Exclusion from ranking is not currently enforced — it is planned as part of the staged pipeline gates.

**Accuracy** — directional correctness. Predictions at exactly 0.5 count as incorrect (no directional signal).

```
correct_i = 1  if (p_yes > 0.5) == outcome
            0  otherwise
Accuracy  = sum(correct_i) / n_valid
```

**Sharpness** — how decisive the predictions are. A tool that always predicts 0.5 has zero sharpness.

```
Sharpness = mean(|p_yes - 0.5|)
```

| Value | Meaning |
|-------|---------|
| 0.0 | All predictions at 50/50 — no conviction |
| 0.5 | Maximally decisive — every prediction near 0 or 1 |

High sharpness is only good if calibration is also good. A tool that confidently predicts 0.95 on everything is sharp but badly calibrated.

**Baseline Brier** — the Brier score of a naive predictor that always outputs the observed base rate.

```
yes_rate       = count(outcome == Yes) / n_valid
Baseline Brier = yes_rate × (1 - yes_rate)
```

If 70% of outcomes are Yes, the naive predictor always says 0.7 and gets Brier = 0.7 × 0.3 = 0.21. Any tool worth using should beat this.

**Brier Skill Score (BSS)** — improvement over the baseline predictor. Positive = better than base rate.

```
BSS = 1 - (Brier / Baseline Brier)
```

| Value | Meaning |
|-------|---------|
| > 0 | Better than predicting the base rate |
| 0 | Same as predicting the base rate |
| < 0 | Worse than predicting the base rate — actively harmful |

### Calibration

Predictions are binned into 10 decile ranges (0.0–0.1, 0.1–0.2, ... 0.9–1.0). For each bin:

```
avg_predicted = mean(p_yes)  for predictions in this bin
realized_rate = count(outcome == Yes) / n  for predictions in this bin
gap           = avg_predicted - realized_rate
```

| Gap sign | Meaning |
|----------|---------|
| Positive | Overconfident — predicts higher than reality |
| Negative | Underconfident — predicts lower than reality |

A perfectly calibrated tool has gap ≈ 0 in every bin. The calibration plot (reliability diagram) shows avg_predicted vs realized_rate; the diagonal line is perfect calibration.

**Note:** The current implementation uses fixed equispaced decile bins (0.0–0.1, 0.1–0.2, ... 0.9–1.0). This is provisional — binning should be monitored and adjusted over time based on sample size and bin stability. With fewer than 200 total predictions, coarser bins (e.g., 5 instead of 10) may be more appropriate to avoid empty or low-count bins.


### Edge over market (diagnostic — not for ranking)

Edge measures whether the tool's prediction was closer to the truth than the market's price. Positive = tool beat market. This is a system diagnostic, not a ranking metric (see PROPOSAL.md for rationale).

```
Per prediction:  edge_i = (market_prob - outcome)² - (p_yes - outcome)²
Aggregate:       Edge   = mean(edge_i)  over edge-eligible predictions
```

Expanding: `edge_i = market_brier_i - tool_brier_i`. When the tool has lower Brier than the market on a question, edge is positive for that question.

**Eligibility:** A row is edge-eligible when it has a valid prediction, a resolved outcome, and `market_prob_at_prediction` is not null.

**Edge positive rate** — fraction of edge-eligible predictions where the tool beat the market:

```
Edge positive rate = count(edge_i > 0) / n_edge_eligible
```

A tool can have negative aggregate edge but > 50% positive rate. This means it beats the market on most questions but loses bigger when it loses — the magnitude of losses exceeds the magnitude of wins.

### Overconfident-wrong (ci_replay.py only)

Used in PR-comment replay comparisons. Counts predictions where the tool was confident and wrong:

```
overconf_wrong_i = 1  if max(p_yes, 1 - p_yes) > 0.80 AND predicted direction ≠ outcome
                   0  otherwise
Overconfident wrong count = sum(overconf_wrong_i)
```

These are the most expensive mistakes — high-conviction wrong predictions that would trigger large Kelly bets in the wrong direction.

### Stratification dimensions

All primary and diagnostic metrics are computed per group across these dimensions:

| Dimension | Buckets | How it's computed |
|-----------|---------|-------------------|
| **Tool** | One bucket per `tool_name` | Direct field grouping |
| **Platform** | `polymarket`, `omen` | Direct field grouping |
| **Category** | `crypto`, `politics`, `sports`, etc. | Direct field grouping |
| **Horizon** | `short_lt_7d`, `medium_7_30d`, `long_gt_30d` | `prediction_lead_time_days`: < 7, 7–30, > 30 |
| **Difficulty** | `hard`, `medium`, `easy` | `\|market_prob - 0.5\|`: < 0.15 = hard, 0.15–0.30 = medium, > 0.30 = easy |
| **Liquidity** | `low`, `medium`, `high` | `market_liquidity_at_prediction` in USD: < 500 = low, 500–5000 = medium, > 5000 = high |
| **Monthly trend** | `YYYY-MM` | Extracted from `predicted_at` |

Cross-dimensions are also computed: platform × difficulty, platform × liquidity, tool × platform, tool × platform × horizon.

Rows where the grouping field is null go into an `unknown` bucket.

## File locations

```
benchmark/
├── datasets/
│   ├── fetch_production.py         # pulls data from subgraphs
│   ├── fetch_open.py               # fetches open markets
│   ├── production_log.jsonl        # scored predictions (gitignored)
│   ├── open_markets.jsonl          # current markets (gitignored)
│   └── .fetch_state.json           # incremental fetch cursor (gitignored)
├── results/                        # output directory (gitignored)
│   ├── sweep_filtered.jsonl        # rows matching tool filter
│   ├── sweep_replay_dataset.jsonl  # enriched with source_content
│   ├── sweep_baseline_scores.json  # production baseline scores
│   ├── sweep_candidate_*.jsonl     # candidate predictions
│   └── sweep_candidate_*_scores.json
├── sweep.py                        # main developer workflow
├── runner.py                       # cached replay engine
├── scorer.py                       # Brier/accuracy computation
├── compare.py                      # score delta comparison
├── analyze.py                      # markdown report generator
├── tournament.py                   # forward-looking predictions
└── score_tournament.py             # score resolved tournaments
```
