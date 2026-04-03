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

## Known Problems (from benchmark data)

- **Severe overconfidence:** ~70% of high-confidence predictions (p_yes >= 0.95) resolve No. The gap between predicted and realized probability grows with confidence.
- **Overall base rate is ~15%:** Most prediction market questions resolve "No", especially "on or before DATE" questions (~90% of questions). Tools systematically overestimate p_yes.
- **Superforcaster clusters at specific values:** 0.93, 0.97, 0.99 appear repeatedly — the model is pattern-matching, not reasoning from evidence.
- **Conflicting output formats:** Some tools ask for both XML-tagged reasoning steps AND JSON output, causing the model to skip reasoning and jump to JSON.

## Required Prompt Fixes (apply when improving any tool)

### 1. Base-rate anchoring (MANDATORY for all tools)

Add before the output format section:
```
CALIBRATION RULES (mandatory):
1. Before estimating, identify the event category and state a base-rate probability.
   Historical data shows ~15% of prediction market questions resolve "Yes".
2. Adjust from the base rate based on specific evidence, not general plausibility.
3. "Sounds likely" is not evidence. Only concrete, verifiable facts should move
   your estimate above 0.70.
```

### 2. Tail discipline (MANDATORY for all tools)

```
TAIL DISCIPLINE:
- Never output p_yes = 1.0 or p_yes = 0.0. No real-world event is 100% certain.
- p_yes above 0.90 requires evidence that the event has ALREADY occurred or is
  imminent with no plausible failure mode remaining.
- p_yes above 0.80 requires strong, specific evidence — not just "this seems likely."
- When the question has a tight deadline ("on or before [date]"), most events do NOT
  happen within the deadline. Weight the time constraint heavily.
- Absence of evidence that the event has occurred IS evidence it hasn't.
- p_yes must be between 0.03 and 0.97. Never output values closer to 0 or 1.
```

### 3. Information barrier (for tools with web search)

The estimating LLM should not see raw web content directly. A synthesis step should produce a factual briefing first. This prevents anchoring on phrases like "analysts predict 80%" found in scraped articles. See `factual_research` tool (PR #118) for the reference implementation.

### 4. Absence-of-evidence reasoning

```
Absence of expected evidence is a NO signal, not neutral.
```

If web search returns nothing relevant about the event occurring, that is evidence against it — not a reason to fall back on "sounds plausible."

### 5. Skepticism in multi-stage tools

**Note:** Multi-stage tools (e.g., prediction-request-reasoning) cannot be partially tested via cached replay yet — both stages re-run together. Changes to stage 2 prompts require tournament validation.

For tools with a reasoning stage followed by a prediction stage, add to the prediction prompt:
```
IMPORTANT: The reasoning AI tends to be overconfident. Apply skepticism.
Before accepting its conclusion, check:
1. Does the reasoning cite specific, verifiable evidence, or just general plausibility?
2. Has the event already occurred, or is the reasoning about future possibility?
3. What is the base rate for this type of event? (~15% of prediction market questions resolve Yes)
If the reasoning concludes "very likely" but cites no evidence it has already happened,
your p_yes should generally not exceed 0.75.
```

## Common Improvement Patterns

1. **Add reasoning scaffolding:** Break the prediction into explicit steps (gather info, weigh evidence, estimate probability, sanity check)
2. **Add bias correction:** Instruct the model to account for negativity bias, sensationalism, and anchoring
3. **Use two-stage approach:** First call for reasoning/analysis, second call for probability estimation
4. **Improve output format:** Strict JSON with explicit field requirements reduces parsing failures. Consider Pydantic Structured Outputs with `reasoning` field first (forces model to think before outputting numbers)
5. **Fix conflicting formats:** If a prompt asks for both XML-tagged steps AND JSON output, pick one. Either keep the reasoning chain and convert to JSON at the end, or use Structured Outputs
6. **Check stale knowledge cutoffs:** Some prompts hardcode outdated cutoff dates (e.g., "October 2023" for gpt-4.1). Remove or dynamically set these

## Further Reading

See `benchmark/PROMPT_IMPROVEMENT_PLAN.md` for detailed per-tool analysis and proposed fixes.
