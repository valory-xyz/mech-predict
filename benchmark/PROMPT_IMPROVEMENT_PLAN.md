# Prompt Improvement Plan

Based on analysis of two CI benchmark runs (Mar 31: 22,755 rows, Apr 2: 36,241 rows) and the approach taken by the `factual_research` tool in PR #118.

---

## Part 1: What the data shows

### Overall stats

| Metric | Value |
|--------|-------|
| Overall base rate (outcome=Yes) | 14.86% |
| Overall avg predicted p_yes | 0.38 |
| Overall Brier score | 0.2555 (random guessing = 0.25) |
| Questions matching "on or before DATE" | 90% |

### The overconfidence problem

| Metric | Count |
|--------|-------|
| Predictions with p_yes >= 0.95 that resolved No | 3,434 |
| Predictions with p_yes = 1.00 that resolved No | 484 |
| Predictions with p_yes <= 0.05 that resolved Yes | 447 |

Overconfident-wrong (p_yes >= 0.95, outcome=No) by tool:

| Tool | Wrong / Total high-conf | Error rate |
|------|------------------------|------------|
| superforcaster | 1,950 / 2,771 | 70% |
| prediction-request-reasoning | 1,164 / 1,730 | 67% |
| prediction-request-rag | 171 / 263 | 65% |
| prediction-request-reasoning-claude | 24 / 39 | 62% |
| prediction-online | 90 / 156 | 58% |
| prediction-offline | 34 / 35 | 97% |

### Superforcaster clusters at specific values

| p_yes value | Count | Wrong | Error rate |
|-------------|-------|-------|------------|
| 0.97 | 1,358 | 1,046 | 77% |
| 0.93 | 1,491 | 1,094 | 73% |
| 0.99 | 667 | 388 | 58% |

These are suspiciously round and repetitive — the model is pattern-matching, not reasoning from the structured chain.

### Calibration is severely off

| Predicted range | Avg predicted | Realized yes-rate | Gap |
|-----------------|---------------|-------------------|-----|
| 0.0-0.1 | 0.04 | 0.06 | -0.02 |
| 0.3-0.4 | 0.34 | 0.13 | +0.22 |
| 0.6-0.7 | 0.65 | 0.20 | +0.45 |
| 0.9-1.0 | 0.97 | 0.30 | +0.67 |

Above 0.3, predictions massively overestimate. The gap grows with confidence.

### 100% malformed tools

| Tool | Malformed | Total |
|------|-----------|-------|
| prediction_request_reasoning-claude | 8 | 8 |
| prediction_request_reasoning | 6 | 6 |
| prediction_request_reasoning-5.2.mini | 7 | 7 |
| resolve-market-jury-v1 | 55 | 55 |

These are likely tool name variants that hit the wrong code path or use a model that doesn't follow the expected output format. Not a prompt issue — needs routing/config investigation.

---

## Part 2: What PR #118 (`factual_research`) does differently

The `factual_research` tool in PR #118 addresses the exact problems we see in the data. Key design decisions:

### 1. Information barrier

The estimating LLM never sees raw web content. A synthesis step produces a factual briefing first. This prevents anchoring on phrases like "analysts predict 80%" found in scraped articles.

**Our tools:** Raw web content goes directly into the prediction prompt. The LLM sees headlines, betting odds, and opinion pieces alongside facts.

### 2. Mandatory base-rate anchoring

The estimation prompt forces:
```
STEP 1 — Base rate anchor (MANDATORY)
a. Identify the event category
b. State an explicit base-rate probability for events of this type
c. This base rate is your starting point
```

**Our tools:** No prompt mentions base rates. The LLM has no anchor and drifts toward "this sounds plausible, so p_yes=0.95."

### 3. Tail discipline rules

```
• Probabilities above 90% require evidence that major failure modes are effectively eliminated.
• Probabilities above 80% require strong historical precedents under similar conditions.
• If confidence < 0.5, probabilities above 70% or below 30% require strong justification.
• If confidence < 0.3, probabilities should remain within [0.2, 0.8].
```

**Our tools:** No constraints on extreme probabilities. The LLM freely outputs 0.97 and 1.00.

### 4. Absence-of-evidence reasoning

```
• Absence of expected evidence is a NO signal, not neutral.
```

**Our tools:** If web search returns nothing relevant, the LLM falls back to "this sounds plausible" reasoning and still outputs high confidence.

### 5. Structured Outputs (Pydantic)

Uses `client.beta.chat.completions.parse()` with Pydantic models. Guaranteed schema conformance — no fragile JSON-in-prompt parsing, no `json.loads()` failures.

The `reasoning` field is positioned first in the schema, forcing the model to explain before committing to numbers.

### 6. Blocked domains

Hard-filters prediction markets, social media, odds sites from search results. Prevents anchoring on market prices.

---

## Part 3: Proposed prompt fixes

### Fix 1: Add base-rate anchoring (ALL tools)

Add to every prediction prompt, before the output format section:

```
CALIBRATION RULES (mandatory):
1. Before estimating, identify the event category and state a base-rate probability.
   Historical data shows ~15% of prediction market questions resolve "Yes".
2. Adjust from the base rate based on specific evidence, not general plausibility.
3. "Sounds likely" is not evidence. Only concrete, verifiable facts should move
   your estimate above 0.70.
```

**Expected impact:** Anchors all predictions around a realistic starting point. Should dramatically reduce the 3,434 overconfident failures.

### Fix 2: Add tail discipline (ALL tools)

Add to every prediction prompt:

```
TAIL DISCIPLINE:
- Never output p_yes = 1.0 or p_yes = 0.0. No real-world event is 100% certain.
- p_yes above 0.90 requires evidence that the event has ALREADY occurred or is
  imminent with no plausible failure mode remaining.
- p_yes above 0.80 requires strong, specific evidence — not just "this seems likely."
- When the question has a tight deadline ("on or before [date]"), most events do NOT
  happen within the deadline. Weight the time constraint heavily.
- Absence of evidence that the event has occurred IS evidence it hasn't.
```

**Expected impact:** Directly addresses the 0.93/0.97 clustering in superforcaster and the p_yes=1.00 outputs in reasoning.

### Fix 3: Superforcaster — fix the contradictory output format

**Current problem:** The prompt has a 7-step structured reasoning chain using XML tags (`<facts>`, `<yes>`, `<no>`, `<thinking>`, `<tentative>`, `<answer>`) but then OUTPUT_FORMAT asks for JSON. The model appears to skip the reasoning chain and jump to JSON.

**Fix:** Remove the XML-tag instructions (steps 1-7) OR remove the JSON output format. Not both.

**Recommended approach:** Keep the 7-step reasoning chain but change the output to use it:
- After step 7, add: "Now convert your `<answer>` value into the JSON format below."
- Remove the duplicate "Output only the JSON object" instruction that conflicts with the reasoning steps.

Alternatively, adopt the `factual_research` approach: use Pydantic Structured Outputs with a `reasoning` field first, so the model is forced to reason before outputting numbers.

### Fix 4: Superforcaster — update stale knowledge cutoff

Line 192: `Your pretraining knowledge cutoff: October 2023`

This is wrong for gpt-4.1-2025-04-14. Should be dynamically set or removed. A wrong cutoff date may cause the model to discount recent information it actually has.

### Fix 5: prediction-request-reasoning — add skepticism to stage 2

**Current PREDICTION_PROMPT (stage 2):**
```
The reasoning from the other AI is: {REASONING}
Carefully consider the user's question and the provided reasoning.
```

This defers to stage 1's reasoning without questioning it. When stage 1 says "very likely", stage 2 rubber-stamps it.

**Fix — add to PREDICTION_PROMPT:**
```
IMPORTANT: The reasoning AI tends to be overconfident. Apply skepticism.
Before accepting its conclusion, check:
1. Does the reasoning cite specific, verifiable evidence, or just general plausibility?
2. Has the event already occurred, or is the reasoning about future possibility?
3. What is the base rate for this type of event? (~15% of prediction market questions resolve Yes)
If the reasoning concludes "very likely" but cites no evidence it has already happened,
your p_yes should generally not exceed 0.75.
```

### Fix 6: Enforce probability range in output format (ALL tools)

Change in all OUTPUT_FORMAT sections:
```
* p_yes must be between 0.03 and 0.97. Never output values closer to 0 or 1.
```

This is a hard floor/ceiling that prevents the worst-case Brier scores (1.0 when wrong at the extremes).

---

## Part 4: Tool changes for cached prediction_prompt testing

### Goal

Allow tools to accept a pre-built `prediction_prompt` via kwargs, skip all web fetch / search / RAG logic, and go directly to LLM call. This lets us iterate on prompt templates using the 36k IPFS deliveries without waiting for cache replay infrastructure.

### Why this works

Every tool stores its full `prediction_prompt` (result[1]) in the IPFS delivery payload under `response["prompt"]`. For all pre-source_content deliveries, this is the only way to know what the LLM saw.

### Changes per tool

#### superforcaster (`superforcaster.py`)

Current flow:
```
source_content? → format_sources_data() → PREDICTION_PROMPT.format() → LLM call
```

Add early exit at line ~402:
```python
cached_prediction_prompt = kwargs.get("prediction_prompt", None)
if cached_prediction_prompt is not None:
    prediction_prompt = cached_prediction_prompt
else:
    # ... existing source_content / Serper logic (lines 409-436)
    prediction_prompt = PREDICTION_PROMPT.format(...)

# Continue with messages, LLM call, return (line 438+)
```

#### prediction_request (`prediction_request.py`)

Current flow:
```
fetch_additional_information() → adjust_additional_information() → PREDICTION_PROMPT.format() → LLM call
```

Add early exit at line ~1145:
```python
cached_prediction_prompt = kwargs.get("prediction_prompt", None)
if cached_prediction_prompt is not None:
    prediction_prompt = cached_prediction_prompt
else:
    # ... existing logic (lines 1145-1186)
    prediction_prompt = active_prompt.format(...)

# Continue with messages, LLM call, return (line 1187+)
```

#### prediction_request_reasoning (`prediction_request_reasoning.py`)

This has two prompts: REASONING_PROMPT → PREDICTION_PROMPT, stored concatenated with `////`.

Two options:
- **Option A (simple):** Accept full concatenated string, split on `////`, skip reasoning, re-run only prediction stage with new PREDICTION_PROMPT.
- **Option B (flexible):** Accept `reasoning_output` kwarg (text from stage 1). Re-run only stage 2 with modified PREDICTION_PROMPT template.

Recommend **Option B** — it lets us test PREDICTION_PROMPT changes while keeping original reasoning.

```python
cached_reasoning = kwargs.get("reasoning_output", None)
if cached_reasoning is not None:
    reasoning = cached_reasoning
    # Skip stage 1 entirely
else:
    # ... existing reasoning stage

# Stage 2 uses (possibly modified) PREDICTION_PROMPT with the reasoning
```

#### prediction_request_rag (`prediction_request_rag.py`)

Same pattern as prediction_request:
```python
cached_prediction_prompt = kwargs.get("prediction_prompt", None)
if cached_prediction_prompt is not None:
    prediction_prompt = cached_prediction_prompt
else:
    # ... existing fetch + RAG logic
```

### Supporting infrastructure

#### 1. Dataset builder script (`benchmark/build_prompt_dataset.py`)

New script that:
1. Reads `production_log.jsonl`
2. For each `deliver_id`, fetches the IPFS payload
3. Extracts `response["prompt"]` (the `prediction_prompt` from IPFS)
4. Outputs JSONL: `{question_text, tool_name, prediction_prompt, final_outcome, original_p_yes}`

#### 2. Runner modification (`benchmark/runner.py`)

When dataset row has `prediction_prompt` field:
- Pass it as `prediction_prompt` kwarg to the tool
- Tool skips web fetch and uses it directly
- Everything else (scoring, comparing) works unchanged

### Testing workflow

```bash
# 1. Build prompt dataset from IPFS deliveries
python benchmark/build_prompt_dataset.py --last-n 200 --output datasets/prompt_dataset.jsonl

# 2. Run baseline (current prompts, cached prediction_prompt)
python benchmark/runner.py \
    --dataset datasets/prompt_dataset.jsonl \
    --tools superforcaster \
    --output results/baseline.jsonl
python benchmark/scorer.py --input results/baseline.jsonl --output results/baseline_scores.json

# 3. Edit prompt template in superforcaster.py (add calibration rules, tail discipline, etc.)

# 4. Run candidate (modified prompts, same cached prediction_prompt... wait)
```

**Important caveat:** If we pass the old `prediction_prompt` directly, we're testing the LLM on the *old* prompt — the template changes have no effect. There are two approaches:

**Approach A — Re-template:** Extract `additional_information` / `sources` from the old `prediction_prompt` via parsing, then re-format using the new template. Works for superforcaster (parse `<background>` tags) and prediction-online (parse ARTICLE blocks). Does NOT work for reasoning (the prompt includes stage 1 output).

**Approach B — Two-phase:** For the first iteration, just test whether the same tool with the same evidence produces better numbers when the prompt has calibration rules. This means:
1. Extract the evidence/sources portion from the old prediction_prompt
2. Inject it into the NEW prompt template
3. Send the new prompt to the LLM

This is what we should implement. The tool change is: accept `cached_sources` (the evidence text), skip web fetch, inject into the new template.

### Revised tool change (superforcaster example)

```python
cached_sources = kwargs.get("cached_sources", None)
if cached_sources is not None:
    sources = cached_sources
else:
    # ... existing Serper/source_content logic
    sources = format_sources_data(organic_data, misc_data)

prediction_prompt = PREDICTION_PROMPT.format(
    question=question, today=d, sources=sources
)
```

This way, the evidence stays the same but the PREDICTION_PROMPT template (with new calibration rules) is what changes.

---

## Part 5: Priority order

1. **Fix the superforcaster prompt** — biggest tool (16k rows), highest overconfidence count (1,950). Add base-rate anchoring, tail discipline, fix knowledge cutoff date, resolve XML/JSON conflict.
2. **Fix the prediction-request-reasoning prompt** — second biggest (10.5k rows). Add skepticism to stage 2.
3. **Build the testing harness** — `cached_sources` kwarg support + dataset builder.
4. **Run A/B comparison** — baseline vs modified prompt on 200 rows.
5. **Investigate 100% malformed tools** — routing/config issue, not prompt.

---

## Part 6: What we explicitly are NOT changing

- No changes to web search logic, URL selection, or scraping
- No changes to RAG chunking/embedding pipeline
- No changes to source_content capture or replay infrastructure
- No changes to the scoring or reporting pipeline (except the calibration label bug already fixed)
- No new tools — improving existing ones first
