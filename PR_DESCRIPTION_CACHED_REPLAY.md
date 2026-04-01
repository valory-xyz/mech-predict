# feat: cached replay pipeline — compare models and tool changes offline

## Summary

Adds a benchmark pipeline that replays resolved prediction market questions through tools with cached web content (`source_content`), enabling controlled A/B comparison of models, prompts, and tool code changes without deploying to production.

- `benchmark/runner.py` — replays questions through tools with cached `source_content`
- `benchmark/compare.py` — compares two scorer outputs, produces markdown delta table
- `benchmark/sweep.py` — single-command orchestrator: enrich production_log → replay → score → compare
- `benchmark/datasets/sample_replay.jsonl` — sample dataset for testing
- `benchmark/datasets/fetch_production.py` — adds `deliver_id` to output rows, `--last-n` flag, IPFS utilities

## How it works

```
production_log.jsonl (download from CI artifacts)
        │
        ▼
sweep.py --last-n 50
        │
        ├─ Read last 50 rows (each has deliver_id)
        ├─ Query subgraph for IPFS hashes by deliver_id
        ├─ Fetch source_content from IPFS gateway
        ├─ Write → replay_dataset.jsonl (reusable)
        │
        ├─ Baseline: runner.py → scorer.py → baseline_scores.json
        ├─ Candidate: runner.py → scorer.py → candidate_scores.json
        └─ compare.py → comparison table
```

The key insight: when `source_content` is injected into a tool's `run()`, the tool skips live web search and uses the cached evidence instead. This means two runs on the same dataset with different models see **identical web content** — the only variable is the model's reasoning.

## Developer guide

### Prerequisites

1. Download `production_log.jsonl` from CI artifacts (daily benchmark workflow) or generate it locally:
   ```bash
   python benchmark/datasets/fetch_production.py --lookback-days 30
   ```
2. Set LLM API keys in your environment:
   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

### Quick start: compare two models

```bash
# Using sample data (no API keys needed for this — tools aren't called)
python benchmark/compare.py --baseline scores_a.json --candidate scores_b.json
```

### Full sweep: model comparison

```bash
# Compare gpt-4o vs gpt-4.1 on last 50 production predictions
python benchmark/sweep.py \
  --last-n 50 \
  --baseline-model gpt-4o \
  --candidate-model gpt-4.1-2025-04-14
```

This will:
1. Read last 50 rows from `benchmark/datasets/production_log.jsonl`
2. Fetch their IPFS payloads to get `source_content`
3. Write `benchmark/results/sweep_replay_dataset.jsonl` (reusable)
4. Run each model through all tools
5. Score both runs
6. Print comparison table

### Reuse an existing replay dataset

Once you have a `replay_dataset.jsonl`, skip the enrichment step:

```bash
python benchmark/sweep.py \
  --dataset benchmark/results/sweep_replay_dataset.jsonl \
  --baseline-model gpt-4o \
  --candidate-model gpt-4.1-2025-04-14
```

### Compare specific tools only

```bash
python benchmark/sweep.py \
  --dataset replay_dataset.jsonl \
  --tools prediction-online,superforcaster \
  --baseline-model gpt-4o \
  --candidate-model gpt-4.1-2025-04-14
```

### Test tool code changes (same model, different code)

```bash
# 1. Run baseline BEFORE your code change
python benchmark/runner.py \
  --dataset replay_dataset.jsonl \
  --tools superforcaster \
  --model gpt-4.1-2025-04-14 \
  --output benchmark/results/before.jsonl

python benchmark/scorer.py \
  --input benchmark/results/before.jsonl \
  --output benchmark/results/before_scores.json

# 2. Make your code change to the tool

# 3. Run candidate AFTER your code change
python benchmark/runner.py \
  --dataset replay_dataset.jsonl \
  --tools superforcaster \
  --model gpt-4.1-2025-04-14 \
  --output benchmark/results/after.jsonl

python benchmark/scorer.py \
  --input benchmark/results/after.jsonl \
  --output benchmark/results/after_scores.json

# 4. Compare
python benchmark/compare.py \
  --baseline benchmark/results/before_scores.json \
  --candidate benchmark/results/after_scores.json
```

### Example output

```
## Overall

|                                     |  B.Brier |  C.Brier |    Delta |    B.Acc |    C.Acc |    Delta |     N | Direction  |
|------------------------------------|---------|---------|---------|---------|---------|---------|------|-----------|
| Overall                             |   0.2217 |   0.0358 |  -0.1859 |   0.6667 |   1.0000 |  +0.3333 |     3 | improved   |

## By Tool

|                                     |  B.Brier |  C.Brier |    Delta |    B.Acc |    C.Acc |    Delta |     N | Direction  |
|------------------------------------|---------|---------|---------|---------|---------|---------|------|-----------|
| prediction-online                   |   0.2217 |   0.0358 |  -0.1859 |   0.6667 |   1.0000 |  +0.3333 |     3 | improved   |
```

## What can and can't be tested with cached replay

| Change type | Works with cached replay? | Why |
|------------|---------------------------|-----|
| Switch model (gpt-4o → gpt-4.1) | Yes | Same evidence, different reasoning |
| Change prompt template | Yes | Same evidence, different prompt |
| Change reasoning logic / add calibration | Yes | Same evidence, different post-processing |
| Compare tools (prediction-online vs superforcaster) | Yes | Same questions, different tools |
| Change search queries / URL selection | No | Search is bypassed — use tournament mode |
| Change search provider (Google → Serper) | No | Search is bypassed |
| Change `num_words` with cleaned-mode snapshots | No | Truncation baked in at capture time |

## Source content modes

PR #174 introduces `source_content_mode`. The runner is mode-transparent:

| Mode | `pages` contains | Replay behavior |
|------|-----------------|-----------------|
| `"raw"` | Raw HTML | Tool calls `extract_text(html)` during replay |
| `"cleaned"` | Pre-extracted text | Tool uses text directly, skips extraction |

The runner logs which mode each row uses: `[1/50] superforcaster (source:raw) | Will Bitcoin...`

## Files changed

### New

| File | Lines | Purpose |
|------|-------|---------|
| `benchmark/runner.py` | 504 | Cached replay runner — loads tools via importlib, calls `run()` with `source_content`, parses results, writes production_log-compatible JSONL |
| `benchmark/compare.py` | 282 | Compares two scores.json files — computes deltas for Brier, accuracy, sharpness across all dimensions |
| `benchmark/sweep.py` | 383 | Orchestrator — reads production_log, enriches with source_content from IPFS, runs baseline + candidate, scores, compares |
| `benchmark/datasets/sample_replay.jsonl` | 3 | Sample dataset with 2 raw-mode and 1 cleaned-mode rows |

### Modified

| File | Change |
|------|--------|
| `benchmark/datasets/fetch_production.py` | Added `deliver_id` to `build_row()` output. Added `--last-n` CLI flag. Added IPFS utilities (`_hex_cid_to_base32`, `fetch_ipfs_source_content`, query templates). No existing code removed. |

## Known limitations

- **`mechDelivery.ipfsHash` is null for recent deliveries** (~after March 13). This is a subgraph indexing issue. The enrichment step skips rows without hashes gracefully. Source content enrichment will work once this is resolved.
- **LLM calls are not cached** — only web content is. Each replay run makes real API calls (costs money). The query-generation LLM call also runs during replay.
- **LLM predictions are non-deterministic** — running the same replay twice produces slightly different p_yes/p_no values.

## Test plan

- [x] All 13 tools load via importlib
- [x] Runner output feeds into scorer — Brier matches manual calculation
- [x] Compare correctly identifies improved/regressed/unchanged
- [x] Sweep orchestration: enrich → replay → score → compare end-to-end
- [x] IPFS hex CID → base32 conversion verified against live subgraph
- [x] Live IPFS gateway fetch works (directory parse + file extraction)
- [x] Empty dataset, empty tools, None source_content handled gracefully
- [x] Both raw and cleaned source_content modes logged correctly
- [x] Deduplication prevents duplicate replay rows
- [x] SIGALRM timeout disabled safely in non-main threads
