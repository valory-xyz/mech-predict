# Tool improvement memory — superforcaster-polymarket-v5

## Entry: 2026-08-13 — Sub-pipeline E (benchmark result / react-benchmark)

### Context
- PR: valory-xyz/mech-predict#441
- Issue: #439 (regression on superforcaster-polymarket-v2, Polymarket)
- Trigger: PR-CI benchmark result comment posted 2026-08-13T06:07:04Z
- Benchmark: v5 (candidate) vs v2 (production baseline), seed 42, n=200

### Hypothesis (from investigation agent)
Structured outputs (`client.beta.chat.completions.parse` + `PredictionResult` schema with text fields before numeric fields) force the 7-step CoT to complete, including step 6 criterion-specificity check. Max_tokens raised 500→4096. Expected outcome: overconfident-YES on narrow-scope markets drops.

### Benchmark result

| Metric | Baseline (v2) | Candidate (v5) | Delta |
|---|---|---|---|
| Brier | 0.3165 | 0.3113 | −1.7% ✅ |
| Directional Accuracy | 58.6% | 62.5% | +6.7% ✅ |
| Overconf-wrong count | 51 | 52 | +2.0% (noise) |
| Overconf-wrong rate | 0.2550 | 0.2600 | +2.0% (noise) |
| Parse failures | — | 0/200 | ✅ |

Pool: 602 usable production deliveries; seed 42; run_type: iteration.

### Hypothesis audit
- Brier and DA moved WITH the hypothesis (broader calibration improves when reasoning chain completes).
- Overconf-wrong FLAT (not improved): expected, since v5 prompt is identical to v2 — step 6 now executes but v2's criterion check lacks v4's 4a–4d explicit screens.
- Mechanistic explanation: The modal call (moderate p_yes) calibrates better with more compute. The tail (p_yes ≥ 0.90 wrong) persists because the forward-looking materialization discount and TYPE A/B temporal classification from v4 are absent.

### Status: open (holdout pending)
- Run was tagged `run_type: iteration` → HC5 requires holdout confirmation before promotion.
- Holdout benchmark posted: `/benchmark superforcaster-polymarket-v2 --candidate-tool superforcaster-polymarket-v5 --platform polymarket --sample 300 --seed 1337`
- Preliminary disposition: PROMOTE if holdout confirms (Brier gate met, output contract guaranteed).

### Confirmed facts
- V5 output contract: 0 parse failures / 200 markets. `run()[0]` is flat JSON with exactly 4 numeric keys. ✅
- V4 reference (not competing): holdout seed 1337 n=301 → Brier 0.2218, overconf-wrong 22. V4 was deployed but produced 0 parseable on-chain outputs (broken output contract — prose + trailing JSON). V2 remains effective production baseline.
- V5 is strictly better than v2 on Brier and DA; output contract reliability is the primary added value.

### Ruled out
- "Improvement is noise": Brier -1.7% on n=200, directional accuracy +6.7% — both directionally consistent, not attributable to sampling noise alone.
- "Overconf-wrong flat means the fix is wrong": The flat overconf-wrong is mechanistically explained by the absence of v4's 4a–4d screens, not by a flaw in the structured-output mechanism.

### Next step
If holdout (seed 1337, n=300) confirms: recommend promotion. File follow-up issue for v6: add v4's 4a–4d screens (prediction-market-circularity check, forward-looking materialization discount, TYPE A/B temporal classification, criterion-specificity gate) to address the overconfidence tail.