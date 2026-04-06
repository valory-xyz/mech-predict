# prediction-online Prompt A/B Test Report

## Problem

`prediction-online` ranks 9th/14 tools (Brier 0.3119, 57% acc). Calibration is broken — when it predicts 0.8-0.9, only 25% of events actually happen (gap +0.59). Root cause: `PREDICTION_PROMPT` has zero calibration guidance, no base-rate anchoring, no evidence evaluation structure.

## Methodology

- **Cached replay**: fetch original formatted prompts from IPFS deliveries, extract `user_prompt` + `additional_information`, re-format with candidate prompt, send to same LLM
- **Model**: `gpt-4.1-2025-04-14`, temperature=0, max_tokens=4096
- **Dataset**: 87 markets from last 7 days (50 omen + 37 polymarket), stratified by outcome
- **Baseline**: production p_yes/p_no from the same 87 markets (apples-to-apples)
- **Scripts**: `benchmark/datasets/fetch_replay.py` (data), `benchmark/prompt_replay.py` (replay)

## Dataset Breakdown

| Platform | Total | Yes Outcome | No Outcome |
|----------|-------|-------------|------------|
| omen | 50 | 14 | 36 |
| polymarket | 37 | 17 | 20 |
| **total** | **87** | **31** | **56** |

---

## Baseline Prompt

The original `PREDICTION_PROMPT` — a flat prompt with no calibration guidance:

```
You are an LLM inside a multi-agent system that takes in a prompt of a user
requesting a probability estimation for a given event...

INSTRUCTIONS
* Read the input under the label "USER_PROMPT"...
* You must provide a probability estimation of the event happening, based on your training data.
* You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION"...
* You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data.
* If an item in "ADDITIONAL_INFORMATION" is not relevant, you must ignore that item...
```

No structured reasoning. No base rates. No overconfidence checks.

---

## V1: Calibration Rules

### Changes from baseline

1. Added "Your performance is evaluated according to the Brier score"
2. Added **ESTIMATION STEPS** (4-step structured process):
   - Categorize the event
   - Consider base-rate probability
   - Evaluate evidence for YES / NO / missing evidence (missing = NO signal)
   - Adjust from base rate
3. Added **CALIBRATION CHECKS**:
   - p_yes > 0.90 requires verified completion (signed, awarded, published)
   - p_yes > 0.80 requires strong specific evidence
   - Low confidence (< 0.5) → keep p_yes between 0.20 and 0.80
   - Numeric thresholds: compare current value to target
4. Removed "provide a probability estimation based on your training data"

### Full prompt

```
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a given event. Your performance is evaluated according to the Brier score.
You are provided with an input under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies an event.
* The event will only have two possible outcomes: either the event will happen or the event will not happen.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION" delimited by three backticks.
* You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data.
* If an item in "ADDITIONAL_INFORMATION" is not relevant, you must ignore that item for the estimation.

ESTIMATION STEPS
1. Identify what the event is and what category it falls into (e.g. regulatory action, product launch, political event, legal decision).
2. Consider a base-rate probability for events of this type. How often do similar events actually occur?
3. Evaluate the ADDITIONAL_INFORMATION for concrete evidence:
   - What specific facts support YES?
   - What specific facts support NO?
   - Is expected evidence missing (no announcement found, no official confirmation)? Missing expected evidence is a NO signal.
4. Adjust from the base rate using the evidence. If evidence is thin or mixed, stay close to the base rate.

CALIBRATION CHECKS (apply before outputting your answer)
* p_yes > 0.90 requires verified completion or binding commitment (signed, awarded, published). Plans and intentions are not completions.
* p_yes > 0.80 requires strong, specific evidence — not just plausibility or reputation.
* If your confidence is low (< 0.5), keep p_yes between 0.20 and 0.80.
* For numeric threshold questions (prices, temperatures, counts), find the current value in the sources and compare to the threshold.

USER_PROMPT:
...
ADDITIONAL_INFORMATION:
...
OUTPUT_FORMAT:
(same as baseline)
```

### Results

| Platform | Baseline Brier | V1 Brier | Delta | Delta % |
|----------|---------------|----------|-------|---------|
| omen | 0.2784 | 0.2745 | -0.0039 | -1.4% |
| polymarket | 0.3094 | 0.2653 | -0.0441 | -14.3% |
| **overall** | **0.2916** | **0.2706** | **-0.0210** | **-7.2%** |

| Platform | Better | Worse | Same |
|----------|--------|-------|------|
| omen | 17 | 25 | 8 |
| polymarket | 15 | 10 | 12 |
| **total** | **32** | **35** | **20** |

### Analysis

**Polymarket** (-14.3%): Strong improvement. Calibration checks work well on stock price and earnings questions — pulling overconfident predictions toward 0.50 when evidence is thin.

**Omen** (-1.4%): Marginal Brier improvement but more individual predictions got worse (25) than better (17). The calibration rules are too conservative on YES outcomes — pulling down correct high-confidence predictions. Big wins come from overconfident-wrong cases (e.g. 0.85→0.75 on NO outcomes), but those are offset by losses on correct YES predictions.

### Diagnosis

The V1 prompt's one-directional bias: it has rules to pull DOWN high p_yes, but no rules to pull UP low p_yes when evidence supports YES. It penalizes confidence rather than penalizing miscalibration symmetrically.

---

## V2: Symmetric Calibration

### Changes from V1

1. Added escape valve: "If sources confirm the event already occurred, high p_yes is correct"
2. Added symmetric low-end check: "p_yes < 0.10 also requires strong evidence"
3. Softened high-end rules with "If NOT confirmed" qualifier

### Full prompt diff from V1

```
CALIBRATION CHECKS (apply before outputting your answer)
+ * If sources confirm the event already occurred or was completed, high p_yes (> 0.90) is correct.
- * p_yes > 0.90 requires verified completion or binding commitment (signed, awarded, published). Plans and intentions are not completions.
- * p_yes > 0.80 requires strong, specific evidence — not just plausibility or reputation.
+ * If NOT confirmed: p_yes > 0.90 requires verified commitment (signed, awarded, published). Plans and intentions are not completions.
+ * If NOT confirmed: p_yes > 0.80 requires strong, specific evidence — not just plausibility or reputation.
+ * p_yes < 0.10 also requires strong evidence that the event is nearly impossible — do not default to low probabilities without justification.
  * If your confidence is low (< 0.5), keep p_yes between 0.20 and 0.80.
  * For numeric threshold questions (prices, temperatures, counts), find the current value in the sources and compare to the threshold.
```

### Results

| Platform | Baseline Brier | V2 Brier | Delta | Delta % |
|----------|---------------|----------|-------|---------|
| omen | 0.2656 | 0.2725 | +0.0069 | +2.6% |
| polymarket | 0.3255 | 0.3011 | -0.0244 | -7.5% |
| **overall** | **0.2916** | **0.2847** | **-0.0069** | **-2.4%** |

| Platform | Better | Worse | Same |
|----------|--------|-------|------|
| omen | 13 | 18 | 19 |
| polymarket | 16 | 11 | 10 |
| **total** | **29** | **29** | **29** |

### Analysis

**Regressed vs V1.** The escape valve ("if confirmed, high p_yes is correct") and the symmetric low-end check made the model more confident again on omen, undoing the V1 gains. The model interprets "sources confirm" too liberally — news articles discussing an event's likelihood get treated as confirmation.

V1 remains the better candidate. The problem isn't asymmetry in the rules — it's that omen questions are inherently harder (novel events with short deadlines) and the model's base confidence on them is already close to correct for YES outcomes but badly overconfident for NO outcomes.

---

## V3: Explicit Base-Rate Prior (best so far)

### Changes from V1

1. Added explicit base-rate prior: "Most 'will X happen by date Y?' questions resolve NO"
2. Kept escape valve for confirmed events: "If sources confirm the event already occurred, high p_yes is justified"
3. Gated the evidence bar rules with "Otherwise:" so they don't fire on confirmed events
4. Removed the symmetric low-end check from V2 (that made model too confident)

### Full prompt

```
You are an LLM inside a multi-agent system that takes in a prompt of a user requesting a probability estimation
for a given event. Your performance is evaluated according to the Brier score.
You are provided with an input under the label "USER_PROMPT". You must follow the instructions
under the label "INSTRUCTIONS". You must provide your response in the format specified under "OUTPUT_FORMAT".

INSTRUCTIONS
* Read the input under the label "USER_PROMPT" delimited by three backticks.
* The "USER_PROMPT" specifies an event.
* The event will only have two possible outcomes: either the event will happen or the event will not happen.
* If the event has more than two possible outcomes, you must ignore the rest of the instructions and output the response "Error".
* You are provided an itemized list of information under the label "ADDITIONAL_INFORMATION" delimited by three backticks.
* You can use any item in "ADDITIONAL_INFORMATION" in addition to your training data.
* If an item in "ADDITIONAL_INFORMATION" is not relevant, you must ignore that item for the estimation.

ESTIMATION STEPS
1. Identify what the event is and what category it falls into (e.g. regulatory action, product launch, political event, legal decision).
2. Consider a base-rate probability for events of this type. How often do similar events actually occur?
3. Evaluate the ADDITIONAL_INFORMATION for concrete evidence:
   - What specific facts support YES?
   - What specific facts support NO?
   - Is expected evidence missing (no announcement found, no official confirmation)? Missing expected evidence is a NO signal.
4. Adjust from the base rate using the evidence. If evidence is thin or mixed, stay close to the base rate.

CALIBRATION CHECKS (apply before outputting your answer)
* Most "will X happen by date Y?" questions resolve NO. The base rate for a specific announced action happening within a short deadline is low unless there is direct evidence it already occurred.
* If sources confirm the event already occurred or was completed, high p_yes is justified.
* Otherwise: p_yes > 0.90 requires verified commitment (signed, awarded, published). Plans and intentions are not completions.
* Otherwise: p_yes > 0.80 requires strong, specific evidence — not just plausibility or reputation.
* If your confidence is low (< 0.5), keep p_yes between 0.20 and 0.80.
* For numeric threshold questions (prices, temperatures, counts), find the current value in the sources and compare to the threshold.

USER_PROMPT:
...
ADDITIONAL_INFORMATION:
...
OUTPUT_FORMAT:
(same as baseline)
```

### Results

| Platform | Baseline Brier | V3 Brier | Delta | Delta % |
|----------|---------------|----------|-------|---------|
| omen | 0.2656 | 0.2482 | -0.0174 | **-6.5%** |
| polymarket | 0.3255 | 0.2755 | -0.0500 | **-15.4%** |
| **overall** | **0.2916** | **0.2598** | **-0.0318** | **-10.9%** |

| Platform | Better | Worse | Same |
|----------|--------|-------|------|
| omen | 22 | 19 | 9 |
| polymarket | 18 | 9 | 10 |
| **total** | **40** | **28** | **19** |

### Analysis

Best result so far. The explicit "most will-X-by-Y resolve NO" prior is the key differentiator — it addresses overconfidence on NO outcomes (the core problem) without hurting correct YES predictions. The escape valve for confirmed events prevents the model from being too conservative when evidence clearly supports YES.

Both platforms improved. Omen flipped from more-worse-than-better (V1: 17B/25W) to more-better-than-worse (V3: 22B/19W). Polymarket remains strong.

## Comparison Table

| Version | Overall Brier | Delta % | Omen Delta % | Polymarket Delta % | Better | Worse | Same |
|---------|--------------|---------|-------------|-------------------|--------|-------|------|
| Baseline | 0.2916 | — | — | — | — | — | — |
| V1 | 0.2706 | -7.2% | -1.4% | -14.3% | 32 | 35 | 20 |
| V2 | 0.2847 | -2.4% | +2.6% | -7.5% | 29 | 29 | 29 |
| **V3** | **0.2598** | **-10.9%** | **-6.5%** | **-15.4%** | **40** | **28** | **19** |

## Validation: V3 on Week 2 (unseen data)

Week 2 dataset: 100 markets from days 8-14, no overlap with week 1.

| Platform | n | Yes | No | Baseline Brier | V3 Brier | Delta % | Better | Worse | Same |
|----------|---|-----|-----|---------------|----------|---------|--------|-------|------|
| omen | 50 | 8 | 42 | 0.2804 | 0.2271 | **-19.0%** | 22 | 16 | 12 |
| polymarket | 50 | 13 | 37 | 0.3037 | 0.2678 | **-11.8%** | 22 | 10 | 18 |
| **overall** | **100** | **21** | **79** | **0.2919** | **0.2475** | **-15.2%** | **44** | **26** | **30** |

**No overfitting.** Week 2 improved more than week 1 (-15.2% vs -10.9%). The NO-heavy omen distribution (42 no / 8 yes) confirms the base-rate prior works well — it's the core improvement driver.

---

## V4: Narrowed Base-Rate Prior

### Changes from V3

1. Narrowed the "most resolve NO" prior to only apply to "announce/launch/complete a novel action by deadline" questions
2. Added separate rule for stock prices, earnings, weather: "use data directly, compare to thresholds, use trends"
3. Merged old numeric threshold check into the new continuous-outcomes rule

### Full prompt diff from V3

```
CALIBRATION CHECKS (apply before outputting your answer)
- * Most "will X happen by date Y?" questions resolve NO. The base rate for a specific announced action happening within a short deadline is low unless there is direct evidence it already occurred.
+ * For questions about whether an organization will announce, launch, or complete a specific novel action by a deadline: the base rate is low. Most such questions resolve NO unless there is direct evidence the action already occurred.
  * If sources confirm the event already occurred or was completed, high p_yes is justified.
  * Otherwise: p_yes > 0.90 requires verified commitment (signed, awarded, published). Plans and intentions are not completions.
  * Otherwise: p_yes > 0.80 requires strong, specific evidence — not just plausibility or reputation.
- * If your confidence is low (< 0.5), keep p_yes between 0.20 and 0.80.
- * For numeric threshold questions (prices, temperatures, counts), find the current value in the sources and compare to the threshold.
+ * For stock prices, earnings, weather, and other questions about continuous/measurable outcomes: use the data in ADDITIONAL_INFORMATION directly. Compare current values to thresholds and estimate based on recent trends and volatility.
+ * If your confidence is low (< 0.5), keep p_yes between 0.20 and 0.80.
```

### Results

| Dataset | Baseline | V4 Brier | Delta % | Better | Worse | Same |
|---------|----------|----------|---------|--------|-------|------|
| Week 1 overall | 0.2916 | 0.2587 | -11.3% | 36 | 33 | 18 |
| Week 1 omen | 0.2656 | 0.2427 | -8.6% | 20 | 19 | 11 |
| Week 1 polymarket | 0.3255 | 0.2804 | -13.9% | 16 | 14 | 7 |
| Week 2 overall | 0.2919 | 0.2447 | **-16.2%** | 45 | 30 | 25 |
| Week 2 omen | 0.2804 | 0.2102 | **-25.0%** | 25 | 18 | 7 |
| Week 2 polymarket | 0.3037 | 0.2792 | -8.1% | 20 | 12 | 18 |

### Analysis

V4 is better than V3 on omen (narrower prior avoids false conservatism on legitimate YES events) but worse on polymarket (the separate stock/weather rule is too permissive). The tradeoff:

- V3: strong polymarket, weak omen
- V4: strong omen, weaker polymarket

## Updated Comparison Table

| Version | W1 Overall | W1 Omen | W1 Poly | W2 Overall | W2 Omen | W2 Poly |
|---------|-----------|---------|---------|-----------|---------|---------|
| Baseline | 0.2916 | 0.2656 | 0.3255 | 0.2919 | 0.2804 | 0.3037 |
| V1 | -7.2% | -1.4% | -14.3% | — | — | — |
| V2 | -2.4% | +2.6% | -7.5% | — | — | — |
| V3 | -10.9% | -6.5% | **-15.4%** | -15.2% | -19.0% | **-11.8%** |
| V4 | -11.3% | -8.6% | -13.9% | **-16.2%** | **-25.0%** | -8.1% |

---

## V5: Combined (V3 broad prior + V4 measurable-outcomes carve-out)

### Changes from V3

1. Kept V3's broad "most will-X-by-Y resolve NO" prior
2. Strengthened confirmed-event escape valve: "do not second-guess confirmed facts"
3. Added V4's measurable-outcomes rule for stock/earnings/weather with explicit note: "the 'most resolve NO' prior does not apply to these"

### Full prompt diff from V3

```
CALIBRATION CHECKS (apply before outputting your answer)
  * Most "will X happen by date Y?" questions resolve NO. The base rate for a specific announced action happening within a short deadline is low unless there is direct evidence it already occurred.
- * If sources confirm the event already occurred or was completed, high p_yes is justified.
+ * If sources confirm the event already occurred or was completed, high p_yes is justified — do not second-guess confirmed facts.
  * Otherwise: p_yes > 0.90 requires verified commitment (signed, awarded, published). Plans and intentions are not completions.
  * Otherwise: p_yes > 0.80 requires strong, specific evidence — not just plausibility or reputation.
+ * For stock prices, earnings, weather, and other measurable outcomes: use the data in ADDITIONAL_INFORMATION directly. Compare current values to thresholds and estimate based on recent trends and volatility. The "most resolve NO" prior does not apply to these.
  * If your confidence is low (< 0.5), keep p_yes between 0.20 and 0.80.
- * For numeric threshold questions (prices, temperatures, counts), find the current value in the sources and compare to the threshold.
```

### Results

| Dataset | Baseline Brier | V5 Brier | Delta % | Baseline Acc | V5 Acc | Better | Worse | Same |
|---------|---------------|----------|---------|-------------|--------|--------|-------|------|
| W1 overall | 0.2916 | 0.2571 | **-11.8%** | 59.8% | 52.9% | 37 | 32 | 18 |
| W1 omen | 0.2656 | 0.2465 | -7.2% | 66.0% | 56.0% | 22 | 22 | 6 |
| W1 polymarket | 0.3255 | 0.2713 | **-16.7%** | 54.1% | 48.6% | 15 | 10 | 12 |
| W2 overall | 0.2919 | 0.2475 | **-15.2%** | 63.0% | **68.0%** | 39 | 27 | 34 |
| W2 omen | 0.2804 | 0.2330 | -16.9% | 64.0% | 68.0% | 20 | 18 | 12 |
| W2 polymarket | 0.3037 | 0.2619 | **-13.7%** | 62.0% | 68.0% | 19 | 9 | 22 |

### Analysis

Best balanced version. Combines V3's strength on polymarket (broad prior) with V4's measurable-outcomes carve-out. Key properties:
- Best polymarket Brier on both weeks
- Solid omen improvement on both weeks
- Best W2 accuracy (68.0% vs 63.0% baseline)
- W1 accuracy drops (52.9%) — same pattern as V3/V4 where predictions cross the 0.5 line due to calibration pulling toward center

## Updated Comparison Table

| Version | W1 Brier (Δ%) | W1 Acc | W2 Brier (Δ%) | W2 Acc | W1 B/W/S | W2 B/W/S |
|---------|--------------|--------|--------------|--------|----------|----------|
| Baseline | 0.2916 | 59.8% | 0.2919 | 63.0% | — | — |
| V1 | -7.2% | — | — | — | 32/35/20 | — |
| V2 | -2.4% | — | — | — | 29/29/29 | — |
| V3 | -10.9% | 54.0% | -15.2% | 67.0% | 40/28/19 | 44/26/30 |
| V4 | -11.3% | 55.2% | **-16.2%** | 66.0% | 36/33/18 | 45/30/25 |
| **V5** | **-11.8%** | 52.9% | -15.2% | **68.0%** | 37/32/18 | 39/27/34 |
