# Benchmark Tool Improvement Guidelines

## Tool Registry

Tool names map to Python modules via `TOOL_REGISTRY` in `benchmark/runner.py`. Key mappings:

| Tool Name | Module |
|-----------|--------|
| `prediction-online` | `packages.valory.customs.prediction_request.prediction_request` |
| `prediction-offline` | same module, different prompt path |
| `claude-prediction-online` | same module, Claude model variant |
| `claude-prediction-offline` | same module, Claude model variant |
| `superforcaster` | `packages.valory.customs.superforcaster.superforcaster` |
| `prediction-request-reasoning` | `packages.napthaai.customs.prediction_request_reasoning.prediction_request_reasoning` |
| `prediction-request-reasoning-claude` | same module, Claude model variant |
| `prediction-request-rag` | `packages.napthaai.customs.prediction_request_rag.prediction_request_rag` |
| `prediction-request-rag-claude` | same module, Claude model variant |
| `prediction-url-cot` | `packages.napthaai.customs.prediction_url_cot.prediction_url_cot` |
| `prediction-url-cot-claude` | same module, Claude model variant |
| `prediction-offline-sme` | `packages.nickcom007.customs.prediction_request_sme.prediction_request_sme` |
| `prediction-online-sme` | same module, online variant |

Multiple tool names can share a module — they differ by the `tool` kwarg passed to `run()`, which selects between online/offline prompts and OpenAI/Claude models.

## Brier Score Reference

- `0.0` = perfect prediction
- `0.25` = random guessing (coin flip)
- `< 0.25` = better than random (good)
- `0.25 - 0.30` = acceptable
- `> 0.30` = underperforming, needs improvement
- `1.0` = maximally wrong

## What You Can Change (validated via cached replay)

- Prompt templates (the string constants used in LLM calls)
- Temperature, max_tokens, and other LLM parameters
- Prompt structure (adding chain-of-thought, reasoning steps, bias correction)

## What You Can Change (NOT validated — flag in PR as "needs tournament validation")

- Search query formulation (how the tool builds queries for Serper/Google)
- Source filtering/ranking logic
- Number of search queries or retrieved chunks
- Embedding parameters or retrieval strategy

These changes affect the retrieval pipeline, which cached replay bypasses. They must be validated through tournament mode (Mode 2) after merge. Flag them clearly in the PR description.

## What You Must NOT Change

- Function signatures (`run(**kwargs)` interface)
- Return format (`Tuple[str, Optional[str], Optional[Dict], Any]`)
- Package structure or `component.yaml` files
- Import paths or module names
- The `@with_key_rotation` decorator pattern

## Examples of Good Prompts

Study these well-performing tools for patterns:

- **`superforcaster`** (`packages/valory/customs/superforcaster/superforcaster.py`): 6-step methodology with explicit bias correction, weighted reasoning for YES/NO, sanity check step. Brier ~0.27.
- **`prediction-offline`** (OFFLINE_PREDICTION_PROMPT in `packages/valory/customs/prediction_request/prediction_request.py`): Superforecaster persona, structured analysis steps, strict JSON output. Brier ~0.21.
- **`prediction-request-reasoning`** (`packages/napthaai/customs/prediction_request_reasoning/prediction_request_reasoning.py`): Two-stage pipeline — reasoning first, then prediction. Multi-query expansion with reciprocal rank fusion. Brier ~0.26.

## Running Cached Replay

To validate changes against historical data:

```bash
# Run tool on cached questions (source_content injected, no live web)
python benchmark/runner.py --dataset <replay_dataset.jsonl> --tools <tool_name>

# Score the results
python benchmark/scorer.py --input benchmark/results/replay_results.jsonl --output benchmark/results/replay_scores.json
```

**Important:** Cached replay only tests prompt/reasoning changes. It cannot evaluate search/retrieval improvements since web queries are bypassed.

## Common Improvement Patterns

1. **Add reasoning scaffolding:** Break the prediction into explicit steps (gather info, weigh evidence, estimate probability, sanity check)
2. **Add bias correction:** Instruct the model to account for negativity bias, sensationalism, and anchoring
3. **Use two-stage approach:** First call for reasoning/analysis, second call for probability estimation
4. **Improve output format:** Strict JSON with explicit field requirements reduces parsing failures
5. **Add calibration guidance:** "Probabilities of 0.95+ should be reserved for near-certainties"
