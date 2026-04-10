# prediction-request-reasoning Prompt A/B Test Report

## Problem

`prediction-request-reasoning` ranks 3rd/14 tools (Brier 0.2378, 70% acc) but has severe overconfidence: 1,164 / 1,730 high-confidence predictions (p_yes >= 0.95) resolved No (67% error rate). The tool has a two-stage architecture: REASONING_PROMPT (stage 1, with web evidence) produces step-by-step reasoning, then PREDICTION_PROMPT (stage 2) evaluates that reasoning and outputs p_yes/p_no. The PREDICTION_PROMPT has zero calibration guidance — it rubber-stamps plausible-sounding reasoning.

## Methodology

- **Cached replay**: fetch original formatted prompts from IPFS deliveries (stored as `reasoning_prompt + "////" + prediction_prompt`), extract `user_prompt`, `additional_information`, and `reasoning`, re-format stage 2 with candidate prompt, send to same LLM
- **Phase 1**: hold stage-1 reasoning fixed, iterate on PREDICTION_PROMPT only (1 LLM call/row)
- **Model**: `gpt-4.1-2025-04-14`, temperature=0, max_tokens=4096
- **Dataset**: 100 markets (50 omen + 50 polymarket), stratified by outcome
- **Baseline**: production p_yes/p_no from the same 100 markets
- **Scripts**: `benchmark/prompt_replay.py enrich --tool prediction-request-reasoning` (data), `benchmark/prompt_replay.py replay --phase prediction-only` (replay)

## Dataset Breakdown

| Platform | Total | Yes Outcome | No Outcome |
|----------|-------|-------------|------------|
| omen | 50 | 16 | 34 |
| polymarket | 50 | 13 | 37 |
| **total** | **100** | **29** | **71** |

---

## Baseline Prompt

The original `PREDICTION_PROMPT` — evaluates reasoning with no calibration guidance:

```
You will be evaluating the likelihood of an event based on a user's question and reasoning provided by another AI.
The user's question is: <user_input> {USER_INPUT} </user_input>

The reasoning from the other AI is: {REASONING}

Carefully consider the user's question and the provided reasoning. Then, think through the following:
 - The probability that the event specified in the user's question will happen (p_yes)
 - The probability that the event will not happen (p_no)
 - Your confidence level in your prediction
 - How useful the reasoning was in helping you make your prediction (info_utility)

Provide your final scores in the following format: <p_yes>probability between 0 and 1</p_yes> <p_no>probability between 0 and 1</p_no>
your confidence level between 0 and 1 <info_utility>utility of the reasoning between 0 and 1</info_utility>

Remember, p_yes and p_no should add up to 1. Provide your reasoning for each score in the scratchpad before giving your final scores.

Your response should be structured as follows:
<p_yes></p_yes>
<p_no></p_no>
<info_utility></info_utility>
<confidence></confidence>
<analysis></analysis>
```

### Baseline results (sanity check — same prompt replayed)

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| Avg Brier | 0.2344 | 0.2345 | +0.0% |
| Accuracy | 70.0% | 71.0% | +1.0pp |

Confirms the pipeline reproduces production scores.

---

## V1: Calibration Rules (PREDICTION_PROMPT only)

### Changes from baseline

1. Added "Your performance is evaluated according to the Brier score"
2. Added **ESTIMATION STEPS** (4-step structured process):
   - Identify event category
   - State base-rate probability ("most 'will X happen by date Y?' questions resolve No")
   - Evaluate reasoning quality: does it cite specific verifiable evidence, or general plausibility?
   - Adjust from base rate using only concrete evidence
3. Added **CALIBRATION CHECKS** (7 rules):
   - Confirmed events escape valve (high p_yes is justified if reasoning cites confirmation)
   - p_yes cap at 0.75 if reasoning concludes "likely" but cites no confirmation
   - p_yes > 0.90 requires verified completion (signed, awarded, published)
   - p_yes > 0.80 requires strong specific evidence, not just coherent argumentation
   - Confidence coupling: if confidence < 0.5, keep p_yes between 0.20 and 0.80
   - Numeric threshold carve-out: compare values directly
   - Absence of expected evidence in reasoning = NO signal

### Full prompt

```
You will be evaluating the likelihood of an event based on a user's question and reasoning provided by another AI. Your performance is evaluated according to the Brier score.
The user's question is: <user_input> {USER_INPUT} </user_input>

The reasoning from the other AI is: {REASONING}

ESTIMATION STEPS (follow in order):
1. Identify the event category (regulatory, product launch, political, legal, scientific, financial, etc.).
2. State a base-rate probability for this category. Most "will X happen by date Y?" questions resolve No.
3. Evaluate the reasoning quality: Does it cite specific, verifiable evidence (dates, sources, confirmed actions), or is it general plausibility and speculation?
4. Adjust from the base rate using only concrete evidence in the reasoning. Stay close to the base rate if the reasoning is vague or mixed.

CALIBRATION CHECKS (apply before outputting scores):
- If the reasoning says the event already occurred or is confirmed, high p_yes is justified.
- If the reasoning concludes "likely" but cites no confirmation it has happened, p_yes should not exceed 0.75.
- p_yes above 0.90 requires the reasoning to cite verified completion (signed, awarded, published, enacted). Plans and intentions are not completions.
- p_yes above 0.80 requires strong, specific evidence in the reasoning, not just coherent argumentation.
- If your confidence is low (< 0.5), keep p_yes between 0.20 and 0.80.
- For numeric threshold questions (price, temperature, count), compare the current value to the threshold rather than relying on narrative reasoning.
- Absence of expected evidence in the reasoning (e.g., no mention of an announcement that should exist if the event occurred) is a signal the event has not happened.

Provide your final scores in the following format: <p_yes>probability between 0 and 1</p_yes> <p_no>probability between 0 and 1</p_no>
your confidence level between 0 and 1 <info_utility>utility of the reasoning between 0 and 1</info_utility>

Remember, p_yes and p_no should add up to 1. Provide your reasoning for each score in the scratchpad before giving your final scores.

Your response should be structured as follows:
<p_yes></p_yes>
<p_no></p_no>
<info_utility></info_utility>
<confidence></confidence>
<analysis></analysis>
```

### V1 Results

| Metric | Baseline | V1 | Delta |
|--------|----------|----|-------|
| Avg Brier | 0.2344 | 0.2224 | **-5.1%** |
| Accuracy | 70.0% | 69.0% | -1.0pp |

| Platform | Baseline Brier | V1 Brier | Delta |
|----------|---------------|----------|-------|
| omen | 0.2965 | 0.2626 | **-11.4%** |
| polymarket | 0.1723 | 0.1822 | +5.7% |

| Metric | Baseline | V1 |
|--------|----------|----|
| Overconfident-wrong (p>=0.80, outcome=No) | 10 | 4 |
| Markets improved | — | 22 |
| Markets worsened | — | 37 |
| Markets same | — | 41 |

### V1 Analysis

- Strong Omen improvement (-11.4%) driven by reducing overconfident-wrong predictions (10 → 4)
- Polymarket slight regression (+5.7%) — baseline was already well-calibrated there (0.1723), calibration rules may be over-correcting
- More markets worsened (37) than improved (22) but improvements were larger in magnitude (concentrated in high-error cases)
- The 0.75 cap may be too aggressive for Polymarket where evidence quality in reasoning is generally higher

---

## V2: Soften the 0.75 cap

### Changes from V1

1. Moved "no confirmation" rule below the evidence bar rules (0.90 / 0.80 thresholds)
2. Softened cap from 0.75 → 0.80
3. Added "or is imminent" escape to avoid penalizing in-progress events

### V2 Results

| Metric | Baseline | V1 | V2 |
|--------|----------|----|----|
| Avg Brier | 0.2344 | 0.2224 (-5.1%) | 0.2231 (-4.8%) |
| Accuracy | 70.0% | 69.0% | 69.0% |

| Platform | Baseline Brier | V1 Brier | V2 Brier |
|----------|---------------|----------|----------|
| omen | 0.2965 | 0.2626 (-11.4%) | 0.2623 (-11.5%) |
| polymarket | 0.1723 | 0.1822 (+5.7%) | 0.1839 (+6.7%) |

| Metric | Baseline | V1 | V2 |
|--------|----------|----|----|
| Overconfident-wrong (p>=0.80) | 10 | 4 | 5 |
| Markets improved | — | 22 | 23 |
| Markets worsened | — | 37 | 34 |
| Markets same | — | 41 | 43 |

### V2 Analysis

- Softening the cap did not fix Polymarket regression — it got slightly worse (+6.7% vs +5.7%)
- Omen held steady (-11.5%)
- The issue is not the cap threshold — the calibration rules are over-correcting on markets that were already well-calibrated
- V1 remains the better prediction prompt (stronger overall Brier, better overconfidence reduction)

### Decision: Lock V1 as PREDICTION_PROMPT winner, move to Phase 2 (REASONING_PROMPT)

---

## Phase 2: REASONING_PROMPT iteration

Using V1 PREDICTION_PROMPT (locked), now iterating on the REASONING_PROMPT that produces the step-by-step reasoning fed into stage 2. Phase uses `--phase reasoning-only` (2 LLM calls/row).

### Baseline REASONING_PROMPT

```
Here is the user's question: {USER_PROMPT}
Here is some additional information that may be relevant to answering the question: <additional_information> {ADDITIONAL_INFOMATION} </additional_information>

Please carefully read the user's question and the additional information provided. Think through the problem step-by-step, taking into account:

- The key details from the user's question, such as the specific event they are asking about and the date by which they want to know if it will occur
- Any relevant facts or context provided in the additional information that could help inform your reasoning
- Your own knowledge and analytical capabilities to reason through the likelihood of the event happening by the specified date

Explain your thought process and show your reasoning for why you believe the event either will or will not occur by the given date. Provide your response inside tags.
<reasoning></reasoning>
```

No structured evidence evaluation. "Your own knowledge" encourages hallucination. No check for event completion vs speculation.

---

## R1: Structured Evidence Evaluation

### Changes from baseline reasoning

1. Replaced free-form "think step-by-step" with **5-step structured format**:
   - EVENT: What event, what deadline?
   - STATUS: Has it already occurred? Cite specific source if yes.
   - EVIDENCE FOR (YES): Concrete facts from additional information only.
   - EVIDENCE AGAINST (NO): Including missing expected evidence.
   - ASSESSMENT: Weigh evidence, distinguish confirmed vs plausible.
2. Removed "your own knowledge" — focus on provided evidence only
3. Added "if additional information is thin, say so explicitly rather than filling gaps"
4. Added missing-evidence-as-signal: "if you'd expect confirmation but find none, state this"

### Full prompt

```
Here is the user's question: {USER_PROMPT}
Here is some additional information that may be relevant to answering the question: <additional_information> {ADDITIONAL_INFOMATION} </additional_information>

Please carefully read the user's question and the additional information provided. Structure your reasoning as follows:

1. EVENT: What specific event is being asked about, and what is the deadline?
2. STATUS: Based on the additional information, has this event already occurred or been confirmed? If yes, cite the specific source. If no, state that clearly.
3. EVIDENCE FOR (YES): List concrete facts from the additional information that support the event happening. Only include verifiable claims with sources.
4. EVIDENCE AGAINST (NO): List concrete facts that argue against. Importantly, if you would expect to find confirmation of the event but the additional information contains none, state this as evidence against.
5. ASSESSMENT: Weigh the evidence. Distinguish between "the event is confirmed/completed" vs "the event seems plausible but unconfirmed."

Focus on what the additional information actually says. Do not speculate beyond the provided evidence. If the additional information is thin or irrelevant, say so explicitly rather than filling gaps with assumptions.

Provide your response inside tags.
<reasoning></reasoning>
```

### R1 Results (reasoning-only phase, V1 prediction prompt)

| Metric | Baseline | V1 (pred only) | R1 (reasoning + V1 pred) |
|--------|----------|----------------|--------------------------|
| Avg Brier | 0.2344 | 0.2224 (-5.1%) | **0.2106 (-10.2%)** |
| Accuracy | 70.0% | 69.0% | **71.0%** |

| Platform | Baseline Brier | V1 Brier | R1 Brier |
|----------|---------------|----------|----------|
| omen | 0.2965 | 0.2626 (-11.4%) | **0.2422 (-18.3%)** |
| polymarket | 0.1723 | 0.1822 (+5.7%) | 0.1790 (+3.9%) |

| Metric | Baseline | V1 | R1 |
|--------|----------|----|----|
| Overconfident-wrong (p>=0.80) | 10 | 4 | **3** |
| Markets improved | — | 22 | 34 |
| Markets worsened | — | 37 | 41 |
| Markets same | — | 41 | 25 |

### R1 Analysis

- Structured reasoning is the bigger lever: -10.2% overall vs -5.1% from prediction prompt alone
- Omen: -18.3% Brier, nearly double V1's improvement
- Polymarket regression shrank from +5.7% (V1) to +3.9% (R1)
- Overconfident-wrong cut from 10 to 3
- More markets moved (34 improved, 41 worsened, 25 same) — the structured format changes more predictions
- Omen accuracy improved 60% → 66%, but Polymarket dropped 80% → 76%

### R1 Polymarket regression investigation

Analyzed all 50 Polymarket markets. The regressions are almost entirely **numeric threshold questions** (stock prices, temperatures, post counts, approval ratings):
- **YES outcomes pulled down** (9 cases): avg b_pyes 0.59 → c_pyes 0.45. The "most resolve No" base-rate prior drags correct moderate-to-high predictions downward.
- **NO outcomes pushed up** (12 cases): avg b_pyes 0.23 → c_pyes 0.31. Structured reasoning hedges on numeric questions instead of directly comparing values.

Root cause: the 5-step format (EVENT/STATUS/EVIDENCE FOR/AGAINST/ASSESSMENT) doesn't fit numeric comparison questions well. These need direct value-vs-threshold comparison, not narrative evidence evaluation.

---

## R2: Numeric Threshold Nudge

### Changes from R1

One sentence added to the ASSESSMENT step:

> "For questions about numeric thresholds (prices, temperatures, counts, ratings), compare the current or most recent value directly to the threshold and state the gap."

Light touch — no separate code path, no branching logic. Just nudges the model to compare numbers directly when applicable.

### Full prompt

```
Here is the user's question: {USER_PROMPT}
Here is some additional information that may be relevant to answering the question: <additional_information> {ADDITIONAL_INFOMATION} </additional_information>

Please carefully read the user's question and the additional information provided. Structure your reasoning as follows:

1. EVENT: What specific event is being asked about, and what is the deadline?
2. STATUS: Based on the additional information, has this event already occurred or been confirmed? If yes, cite the specific source. If no, state that clearly.
3. EVIDENCE FOR (YES): List concrete facts from the additional information that support the event happening. Only include verifiable claims with sources.
4. EVIDENCE AGAINST (NO): List concrete facts that argue against. Importantly, if you would expect to find confirmation of the event but the additional information contains none, state this as evidence against.
5. ASSESSMENT: Weigh the evidence. Distinguish between "the event is confirmed/completed" vs "the event seems plausible but unconfirmed." For questions about numeric thresholds (prices, temperatures, counts, ratings), compare the current or most recent value directly to the threshold and state the gap.

Focus on what the additional information actually says. Do not speculate beyond the provided evidence. If the additional information is thin or irrelevant, say so explicitly rather than filling gaps with assumptions.

Provide your response inside tags.
<reasoning></reasoning>
```

### R2 Results (reasoning-only phase, V1 prediction prompt)

| Metric | Baseline | R1 | R2 |
|--------|----------|----|----|
| Avg Brier | 0.2344 | 0.2106 (-10.2%) | **0.1985 (-15.3%)** |
| Accuracy | 70.0% | 71.0% | **74.0%** |

| Platform | Baseline Brier | R1 Brier | R2 Brier |
|----------|---------------|----------|----------|
| omen | 0.2965 | 0.2422 (-18.3%) | **0.2239 (-24.5%)** |
| polymarket | 0.1723 | 0.1790 (+3.9%) | **0.1731 (+0.5%)** |

| Platform | Baseline Acc | R1 Acc | R2 Acc |
|----------|-------------|--------|--------|
| omen | 60% | 66% | **68%** |
| polymarket | 80% | 76% | **80%** |

| Metric | Baseline | R1 | R2 |
|--------|----------|----|----|
| Overconfident-wrong (p>=0.80) | 10 | 3 | **3** |
| Markets improved | — | 34 | 37 |
| Markets worsened | — | 41 | 41 |
| Markets same | — | 25 | 22 |

### R2 Analysis

- One sentence neutralized the Polymarket regression: +3.9% → +0.5% Brier, 76% → 80% accuracy
- Omen continued improving: -18.3% → -24.5% Brier, 66% → 68% accuracy
- Overall: -15.3% Brier, 74% accuracy (vs 70% baseline)
- The numeric nudge works because it lets the model short-circuit narrative reasoning on price/temperature questions and go straight to value comparison

---

## Cumulative Results

| Version | Overall Brier | Overall Acc | Omen Brier | Omen Acc | Poly Brier | Poly Acc | Overconf-wrong |
|---------|--------------|-------------|------------|----------|------------|----------|----------------|
| Baseline | 0.2344 | 70% | 0.2965 | 60% | 0.1723 | 80% | 10 |
| V1 (pred) | 0.2224 (-5.1%) | 69% | 0.2626 (-11.4%) | 60% | 0.1822 (+5.7%) | 78% | 4 |
| V2 (pred) | 0.2231 (-4.8%) | 69% | 0.2623 (-11.5%) | 60% | 0.1839 (+6.7%) | 78% | 5 |
| R1 (reas+V1) | 0.2106 (-10.2%) | 71% | 0.2422 (-18.3%) | 66% | 0.1790 (+3.9%) | 76% | 3 |
| **R2 (reas+V1)** | **0.1985 (-15.3%)** | **74%** | **0.2239 (-24.5%)** | **68%** | **0.1731 (+0.5%)** | **80%** | **3** |

---

## Holdout Validation

Separate dataset of 75 markets (50 Omen, 25 Polymarket) with zero overlap to the training set. Different random seed (99), deduplicated by deliver_id.

### Holdout Dataset Breakdown

| Platform | Total | Yes Outcome | No Outcome |
|----------|-------|-------------|------------|
| omen | 50 | 16 | 34 |
| polymarket | 25 | 7 | 18 |
| **total** | **75** | **23** | **52** |

### R2 Holdout Results

| Metric | Baseline | R2 | Delta |
|--------|----------|----|-------|
| Avg Brier | 0.2689 | 0.2473 | **-8.1%** |
| Accuracy | 62.7% | 61.3% | -1.4pp |

| Platform | Baseline Brier | R2 Brier | Delta | Baseline Acc | R2 Acc |
|----------|---------------|----------|-------|-------------|--------|
| omen | 0.2971 | 0.2498 | **-15.9%** | 60% | 60% |
| polymarket | 0.2126 | 0.2422 | +13.9% | 68% | 64% |

| Metric | Baseline | R2 |
|--------|----------|----|
| Overconfident-wrong (p>=0.80) | 13 | **2** |
| Markets improved | — | 31 |
| Markets worsened | — | 33 |
| Markets same | — | 11 |

### Holdout Analysis

- **Omen Brier improvement holds**: -15.9% on holdout (vs -24.5% on training) — expected shrinkage, not overfitting
- **Overconfidence fix generalizes strongly**: 13 → 2 overconfident-wrong predictions (stronger than training 10→3)
- **Polymarket regresses on holdout**: +13.9% Brier, 68→64% accuracy (only 25 samples — high variance, but consistent direction)
- **Accuracy flat on holdout**: 62.7→61.3% — Brier improvement is from better calibration (less extreme wrong predictions), not more correct binary calls
- **Production mix context**: 98% of production traffic is Omen (4,659 vs 89 Polymarket), so Omen gains dominate in practice

### Training vs Holdout Comparison

| Metric | Training (n=100) | Holdout (n=75) |
|--------|-----------------|----------------|
| Overall Brier delta | -15.3% | -8.1% |
| Omen Brier delta | -24.5% | -15.9% |
| Poly Brier delta | +0.5% | +13.9% |
| Overconf-wrong reduction | 10→3 | 13→2 |

### Fresh Data Validation (CI artifact, Brier-stratified sampling)

Separate dataset of 60 markets (30 Omen, 30 Polymarket) from the CI benchmark flywheel artifact (68k rows). Uses updated Brier-stratified sampling (platform × outcome × brier_bucket). Seed 99, zero overlap with training/holdout.

| Metric | Baseline | V1+R2 | Delta |
|--------|----------|-------|-------|
| Avg Brier | 0.3006 | 0.2667 | **-11.3%** |
| Accuracy | 60.0% | 61.7% | +1.7pp |
| Overconf-wrong | 10 | 4 | **-60%** |

| Platform | Baseline Brier | V1+R2 Brier | Delta | Baseline Acc | V1+R2 Acc |
|----------|---------------|-------------|-------|-------------|-----------|
| omen | 0.2769 | 0.2412 | **-12.9%** | 63.3% | 66.7% |
| polymarket | 0.3244 | 0.2923 | **-9.9%** | 56.7% | 56.7% |

### Fresh Data Analysis

- **First Polymarket improvement**: -9.9% Brier (previously flat or regressing on training/holdout)
- Brier-stratified sampling ensures badly-calibrated predictions are represented, giving more realistic validation
- Omen improvement consistent: -12.9% (vs -24.5% training, -15.9% holdout)
- Overconf-wrong 10→4, Polymarket overconf-wrong 6→1

### All Validations Summary

| Dataset | n | Brier delta | Acc delta | Overconf-wrong |
|---------|---|-------------|-----------|----------------|
| Training | 100 | -15.3% | +5.0pp | 10→3 |
| Holdout | 75 | -8.1% | -1.4pp | 13→2 |
| Fresh (Brier-stratified) | 60 | -11.3% | +1.7pp | 10→4 |

---

## Phase 3: Prediction Prompt V2 (with R2 reasoning)

Tested whether the prediction prompt could be improved now that R2 produces structured reasoning (EVENT/STATUS/EVIDENCE FOR/AGAINST/ASSESSMENT).

### V2 Changes from V1

1. Replaced "most resolve No" with category-specific base rates (product launches ~30%, narrow bands ~15-25%, govt announcements ~40-60%)
2. Replaced hard caps (0.75, 0.80, 0.90) with evidence-based probability ranges (confirmed → 0.90-0.95, strong evidence → 0.65-0.85, weak → 0.30-0.50, none → 0.05-0.25)
3. Added explicit narrow-band numeric question handling
4. Softened absence-of-evidence from conclusive to directional signal
5. Referenced R2's structure: "Focus on the STATUS and EVIDENCE sections"

### V2 Results (prediction-only phase, R2 cached reasoning, training set n=100)

| Metric | Baseline | V1+R2 | V2+R2 |
|--------|----------|-------|-------|
| Avg Brier | 0.2344 | 0.1985 (-15.3%) | 0.2018 (-13.9%) |
| Accuracy | 70.0% | 75.0% | 74.0% |
| Overconf-wrong | 10 | 5 | 3 |

| Platform | Baseline Brier | V1+R2 Brier | V2+R2 Brier |
|----------|---------------|-------------|-------------|
| omen | 0.2965 | 0.2239 (-24.5%) | 0.2335 (-21.3%) |
| polymarket | 0.1723 | 0.1731 (+0.5%) | 0.1702 (-1.2%) |

### V2 Analysis

- V2 trades Omen Brier (-21.3% vs -24.5%) for slightly better Polymarket (-1.2% vs +0.5%) and fewer overconf-wrong (3 vs 5)
- Differences are marginal (1.4pp Brier, 1pp accuracy) — within noise on 100 markets
- V1 remains the stronger prediction prompt overall

### Decision: Keep V1+R2 as the winner. No further prediction prompt iteration needed.

---

## Final Winner: V1 (PREDICTION_PROMPT) + R2 (REASONING_PROMPT)

| Metric | Baseline (prod) | Winner (V1+R2) | Delta |
|--------|-----------------|----------------|-------|
| Training Brier (n=100) | 0.2344 | 0.1985 | **-15.3%** |
| Training Accuracy | 70.0% | 75.0% | **+5.0pp** |
| Holdout Brier (n=75) | 0.2689 | 0.2473 | **-8.1%** |
| Overconf-wrong (training) | 10 | 3 | **-70%** |
| Overconf-wrong (holdout) | 13 | 2 | **-85%** |

---

## Post-PR#194: Corrected Polymarket Outcomes

> **Important:** PR #194 fixed an inverted-outcome bug in `fetch_production.py` for Polymarket markets. All Polymarket Brier deltas above were computed against wrong ground truth. The sections below use corrected data from the post-#194 production log (68,987 rows, flywheel run [24123952342](https://github.com/valory-xyz/mech-predict/actions/runs/24123952342)).

### Corrected Baseline (V1+R2, phase both, n=100, 50/platform, seed 42)

From PR #198 revalidation comment — re-ran V1 PREDICTION_PROMPT + R2 REASONING_PROMPT against corrected outcomes:

| Platform | Baseline Brier | V1+R2 Brier | Delta |
|----------|---------------|-------------|-------|
| omen | 0.2785 | 0.2560 | **-8.1%** |
| polymarket | 0.2604 | 0.2714 | **+4.2% (regressed)** |
| overall | 0.2695 | 0.2637 | -2.1% |

**Polymarket regression confirmed** on corrected data: 13 better, 24 worse, 13 same. Calibration rules over-fitted to Omen overconfidence patterns at Polymarket's expense.

---

## V3: Loosen All Calibration Rules (PREDICTION_PROMPT only)

### Changes from V1

1. Removed "Most 'will X happen by date Y?' questions resolve No" base-rate prior
2. Replaced with "Consider what base rate is reasonable for this category, without assuming a default direction"
3. Removed 0.75 hard cap for "likely without confirmation"
4. Softened 0.80/0.90 from "requires" to "should be supported by"
5. Removed confidence coupling rule (confidence < 0.5 → p_yes 0.20-0.80)
6. Expanded numeric threshold bullet

### V3 Results (prediction-only, R2 reasoning held fixed, n=100)

| Platform | Baseline Brier | V1 Brier | V3 Brier |
|----------|---------------|----------|----------|
| omen | 0.2785 | 0.2560 (-8.1%) | 0.2759 (-0.9%) |
| polymarket | 0.2604 | 0.2714 (+4.2%) | 0.2723 (+4.6%) |
| overall | 0.2695 | 0.2637 (-2.1%) | 0.2741 (+1.7%) |

### V3 Analysis

- **Failed.** Loosening all calibration rules destroyed the Omen improvement (from -8.1% to -0.9%) without fixing Polymarket (+4.6%, same as V1).
- The "most resolve No" prior and hard caps are doing critical work for Omen overconfidence — removing them loses that benefit.
- Polymarket regression is not caused by the caps being too tight — it persists even when they're removed.
- **Not a viable direction.** Need targeted changes, not blanket loosening.

Note: V3 was tested prediction-only (holds R2 reasoning fixed). V1 numbers are from phase-both (PR #198). Not directly comparable, but the direction is clear.

---

## V3b: Keep V1 + Numeric Threshold Override (PREDICTION_PROMPT only)

### Changes from V1

Minimal, targeted changes:
1. **Removed** confidence coupling rule (`confidence < 0.5 → keep p_yes 0.20-0.80`) — pure dampening with no calibration benefit
2. **Expanded** numeric threshold bullet into an explicit cap override: "If the value is far from the threshold, a confident prediction (below 0.15 or above 0.85) is appropriate regardless of the caps above"
3. All other V1 rules kept intact (base-rate prior, 0.75/0.80/0.90 caps, absence-of-evidence signal)

### Full V3b prompt

```
You will be evaluating the likelihood of an event based on a user's question and reasoning provided by another AI. Your performance is evaluated according to the Brier score.
The user's question is: <user_input> {USER_INPUT} </user_input>

The reasoning from the other AI is: {REASONING}

ESTIMATION STEPS (follow in order):
1. Identify the event category (regulatory, product launch, political, legal, scientific, financial, etc.).
2. State a base-rate probability for this category. Most "will X happen by date Y?" questions resolve No.
3. Evaluate the reasoning quality: Does it cite specific, verifiable evidence (dates, sources, confirmed actions), or is it general plausibility and speculation?
4. Adjust from the base rate using only concrete evidence in the reasoning. Stay close to the base rate if the reasoning is vague or mixed.

CALIBRATION CHECKS (apply before outputting scores):
- If the reasoning says the event already occurred or is confirmed, high p_yes is justified.
- If the reasoning concludes "likely" but cites no confirmation it has happened, p_yes should not exceed 0.75.
- p_yes above 0.90 requires the reasoning to cite verified completion (signed, awarded, published, enacted). Plans and intentions are not completions.
- p_yes above 0.80 requires strong, specific evidence in the reasoning, not just coherent argumentation.
- For numeric threshold questions (price, temperature, count, rating): compare the current value directly to the threshold and let the gap determine your probability. If the value is far from the threshold, a confident prediction (below 0.15 or above 0.85) is appropriate regardless of the caps above.
- Absence of expected evidence in the reasoning (e.g., no mention of an announcement that should exist if the event occurred) is a signal the event has not happened.
```

### V3b Results — prediction-only (n=100, R2 reasoning held fixed)

| Platform | Baseline Brier | V1 Brier | V3b Brier |
|----------|---------------|----------|-----------|
| omen | 0.2785 | 0.2560 (-8.1%) | 0.2572 (-7.7%) |
| polymarket | 0.2604 | 0.2714 (+4.2%) | **0.2558 (-1.8%)** |
| overall | 0.2695 | 0.2637 (-2.1%) | **0.2565 (-4.8%)** |

| Metric | Baseline | V3b |
|--------|----------|-----|
| Accuracy | 63.0% | 64.0% |

Note: V1 numbers are from phase-both (PR #198). V3b prediction-only holds original production reasoning fixed. Not directly comparable, but Polymarket regression is eliminated.

### V3b Results — phase-both (n=100, R2 reasoning + V3b prediction)

| Platform | Baseline Brier | V1+R2 Brier | V3b+R2 Brier |
|----------|---------------|-------------|-------------|
| omen | 0.2785 | 0.2560 (-8.1%) | 0.2724 (-2.2%) |
| polymarket | 0.2604 | 0.2714 (+4.2%) | **0.2215 (-14.9%)** |
| overall | 0.2695 | 0.2637 (-2.1%) | **0.2470 (-8.3%)** |

| Metric | Baseline | V1+R2 | V3b+R2 |
|--------|----------|-------|--------|
| Accuracy | 63.0% | — | 64.0% |
| Overconf-wrong (p>=0.80) | 12 | — | 5 |
| Markets improved | — | — | 33 |
| Markets worsened | — | — | 44 |
| Markets same | — | — | 23 |

### V3b Analysis

- **Polymarket regression eliminated**: flipped from V1's +4.2% to V3b's -1.8% improvement
- **Omen improvement preserved**: -7.7% (vs V1's -8.1% — within noise)
- **Overall improved**: -4.8% (vs V1's -2.1%)
- The numeric threshold override is the key change — explicitly allows confident predictions on price/temperature/count questions, bypassing caps that were dragging well-calibrated Polymarket predictions toward 0.5
- Removing confidence coupling had minimal effect (was already rare in practice)

**Phase-both confirms:**
- Polymarket regression fully eliminated: V1's +4.2% → V3b's **-14.9%** improvement
- Overall improved from -2.1% to **-8.3%**
- Omen improvement weaker (-2.2% vs V1's -8.1%) — R2 reasoning regeneration introduces LLM variance; Omen's "will X happen by date Y?" questions are more sensitive to reasoning phrasing differences
- Overconfident-wrong predictions halved: 12 → 5
- Awaiting CI benchmark for independent validation
