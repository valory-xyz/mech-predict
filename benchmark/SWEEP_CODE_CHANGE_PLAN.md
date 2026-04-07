# Sweep Pipeline: Design & Implementation

## Problem

Developers change tool code, models, or prompts and need a fast way to verify
if the change improved predictions. The production log already has baseline
predictions with known outcomes — no need to re-run the baseline model.

## Solution

`sweep.py` is a single pipeline that:

1. **Filters** last N rows from production_log matching your tool
2. **Enriches** with `source_content` from IPFS (cached web evidence)
3. **Scores baseline** using only the enriched rows (production predictions)
4. **Replays candidate** with your modified code + cached source_content
5. **Scores candidate** and prints a comparison table

Baseline scoring happens **after** enrichment so both baseline and candidate
cover exactly the same set of questions — fair comparison.

## Usage

```bash
# Test a code change
python benchmark/sweep.py --last-n 500 --tools superforcaster --candidate-model gpt-4.1-2025-04-14

# Test a model change
python benchmark/sweep.py --last-n 500 --tools superforcaster --candidate-model claude-4-sonnet-20250514

# Reuse existing dataset (skip IPFS fetching)
python benchmark/sweep.py --dataset results/sweep_replay_dataset.jsonl --tools superforcaster --candidate-model gpt-4.1-2025-04-14
```

## Requirements

- `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` for claude tools) in environment
- `production_log.jsonl` with `deliver_id` fields (for IPFS lookup)
- The mech that generated the deliveries must have `return_source_content=true`

## CLI flags

| Flag | Required | Description |
|------|----------|-------------|
| `--tools` | Yes | Comma-separated tool names to test |
| `--candidate-model` | No | Model for candidate (default: gpt-4.1-2025-04-14) |
| `--last-n` | No | Filter last N rows (default: 100) |
| `--production-log` | No | Path to production log (default: datasets/production_log.jsonl) |
| `--dataset` | No | Existing replay dataset (skips filter + enrichment) |
| `--baseline-scores` | No | Existing baseline scores (skips baseline scoring) |
| `--output-dir` | No | Output directory (default: results/) |
| `--timeout` | No | Per-tool timeout seconds (default: 240) |
