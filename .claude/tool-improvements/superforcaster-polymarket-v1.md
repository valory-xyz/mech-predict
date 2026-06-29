## Issue #332 -- 2026-06-09

- **Trigger:** regression on `polymarket` W-1 (`2026-06-02T04:17:54Z` -> `2026-06-09T04:17:54Z`); headline Brier `0.2323` (n=656) vs W-2 `0.1470` (n=222) (delta +0.0853); trigger=`regression`.
- **PR:** branch `tool-improvement/superforcaster-polymarket-v2-criterion-specificity`.
- **Status:** draft PR opened (path a).
- **Revision 2026-06-10:** CI-fix — added `family="superforcaster"` to `superforcaster-polymarket-v2` `ToolSpec` in `benchmark/tools.py`. `ToolSpec.family` became a required field (no default) after a repo update; the PR was missing it, causing a `TypeError` at import time in the Enrich dataset step (run 27252299875 / job 80479168011). Fix was present on branch as commit `aad9d158` before agent run; agent posted explanatory comment #4666451627.

### Confirmed hypothesis

At the **prediction-LLM-call stage**, the tool over-estimates `p_yes` for narrow-criterion questions where the resolution condition requires an exact word/phrase, an exact threshold on a specific date, or any binary test narrower than "the topic is being discussed". The evidence retrieved is topically relevant but does not confirm the narrow criterion; the 7-step CoT in `PREDICTION_PROMPT` has no explicit step differentiating topical-relevance from criterion-satisfaction. Outcome: systematic overconfidence -> inflated `p_yes` -> high Brier contribution.

### Stage map (v1 run())

| Stage | Gate-visible? |
|-------|--------------|
| in-code retrieval (fetch_additional_sources) | NO |
| format_sources_data() | YES |
| PREDICTION_PROMPT construction | YES |
| generate_prediction_with_retry() -> GPT-4.1 | YES |
| JSON output parsing | YES |

### Evidence sample (worst-miss rows, W-1 window)

All 20 inspected IPFS deliveries showed good-evidence/bad-reasoning pattern: relevant search evidence retrieved; model conflated topical relevance with criterion satisfaction.

### Localised cells

| cell | n W-1 | Brier W-1 | delta |
|------|--------|-----------|-------|
| category=other | 81 | 0.3267 | +0.0944 |
| category=politics | 481 | 0.2165 | +0.0673 |

Both cells are same CID in both windows; mix-shift ruled out.

### Mechanism and fix

- **Mechanism (v1):** step 6 of PREDICTION_PROMPT: "Consider priors and base rates" - too abstract; no instruction to check whether evidence literally satisfies the resolution criterion.
- **Fix (v2):** Expanded step 6 with explicit criterion-specificity check: identify the exact condition for YES; assess whether evidence directly confirms it vs only establishes topical relevance; apply base-rate correction for narrow-scope criteria.
- **Pre-lint creative LOC:** +7 lines.
- **New tool:** `superforcaster-polymarket-v2`.

### Pre-PR sanity

- autonomy packages lock --check -> Verification successful
- import ok
- tournament_tools.json entry added
- Non-trivial change: YES
- ASCII-only: PASS
- LOC: +7 (well under 150 soft / 300 hard)


## Issue #374 -> PR #375 -- 2026-06-29

- **Trigger:** Issue #374 chronic-bad overconfident-YES regression on polymarket.
- **PR:** #375 `feat(superforcaster-polymarket-v4): step-4 evidence-reliability screen for overconfident-YES`
- **Branch:** `tool-improvement/superforcaster-polymarket-v5-temporal-criterion-screen`
- **Status:** holdout confirmed 2026-06-29 -- promotion recommended (comment #4835611333)

### Hypothesis (from PR body, investigation context not separately recorded)
At the prediction-LLM-call stage (gate-visible), superforcaster-polymarket-v1 produces overconfident YES predictions (~53% of W-1 Brier mass). The step-4 evidence-reliability screen in PREDICTION_PROMPT addresses four sub-checks:
- **4a** prediction-market-odds filter (discard circular self-referential odds)
- **4b** forward-looking-intent discount (40-60% materialization discount)
- **4c** TYPE A/B temporal-evidence classification (base-rate fallback on all-TYPE-B) -- targets "X in headlines this week" p_yes=0.99 that resolves NO
- **4d** criterion-specificity check (require TYPE A evidence for the exact criterion)

Plus `max_tokens` 500 -> 1500 for full chain-of-thought execution. New version: `superforcaster-polymarket-v4`.

### Benchmark ledger
- **Benchmark 2026-06-29:** SHA `19c818e5c991901987b9f0e1567d9d68abc08391`, seed 42, n=50, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- n=50 too noisy (delta=-0.024, 2*SE~=0.090 > delta); Overconf-wrong -18.2% positive fingerprint signal; growing to n=300
- **Benchmark 2026-06-29:** SHA `19c818e5c991901987b9f0e1567d9d68abc08391`, seed 42, n=300->301, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- result arrived late (comment #4834249768; previously thought cancelled): Brier 0.2625->0.2412 (-8.1%), DA 64.5%->63.7% (-1.2%), Overconf-wrong 50->28 (-44.0%); parse 301/301; consistent with n=100 E2 result; dev-seed, holdout still pending
- **Benchmark 2026-06-29:** SHA `cd5fc0d3e265b3a960a3d31614b4e4b6b42c6c35`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (new SHA from revert-restore commit; all CI green)
- **Benchmark 2026-06-29:** SHA `6e3f11cca3871f0e620140a6a5f3f2c98c4945fc`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (memory-only chore commit; tool code identical to cd5fc0d3; all CI green; comment #4834755919)
- **E2 diagnosis 2026-06-29:** seed 42 n=100 result (comment #4834829972): Brier 0.2734->0.2227 (-18.5%), DA 63%->72% (+14.3%), Overconf-wrong 19->11 (-42.1%). Both aggregate Brier and targeted fingerprint improved -> E3 promotion path. All prior runs used seed 42 (dev); posting holdout-confirmation at seed 1337, n=300.
- **Benchmark 2026-06-29:** SHA `237173e5c4f28442a5607640c7e3582272436900`, seed 1337, n=300, holdout, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (holdout-confirmation; E3 path; comment #4834886768)
- **Benchmark 2026-06-29:** SHA `a57a0253a547cf17ad894bc3a49a8bb1a77c93ff`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (memory-only chore commit; tool code identical to holdout SHA 237173e5; all CI green; comment #4835068378)
- **E2/E3 diagnosis 2026-06-29:** seed 42 n=100 result (comment #4835268684): Brier 0.2734->0.2154 (-21.2%), DA 63%->70% (+11.1%), Overconf-wrong 19->7 (-63.2%). Parse 100/100. E2: aggregate Brier and targeted fingerprint both improved -> E3 path confirmed. Holdout (seed 1337, n=300, SHA 237173e5, comment #4834886768) already triggered; awaiting result for promotion decision.
- **Benchmark 2026-06-29:** SHA `9242f769a7a5ef0cbd7d5c1a0a3ecce51d63d80e`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (memory-only chore commit; tool code identical to holdout SHA 237173e5; all CI green; trigger comment #4835328995, result comment #4835516704: Brier 0.2734->0.2320 (-15.1%), DA 63%->68%, Overconf-wrong 19->9 (-52.6%))
- **Holdout result 2026-06-29:** SHA `237173e5c4f28442a5607640c7e3582272436900`, seed 1337, n=301, holdout, baseline=superforcaster-polymarket-v1, platform=polymarket -- result (comment #4835535374): Brier 0.2610->0.2218 (-15.0%), DA 63.8%->67.1% (+5.2%), Overconf-wrong 52->22 (-57.7%); parse 301/301 (100%). WIN: all three primary metrics improved; Overconf-wrong fingerprint strongly validated (-57.7%, consistent with 5 dev-seed runs at -18% to -63%). Promotion recommended (comment #4835611333).

- **Benchmark 2026-06-29:** SHA `01c875f36d14dfb2bb345466d4dd7e226a838d13`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- result (memory-only chore commit; tool code identical to holdout SHA 237173e5; trigger comment #4835578188, result comment #4835759925): Brier 0.2734->0.2283 (-16.5%), DA 63%->68% (+7.9%), Overconf-wrong 19->9 (-52.6%); parse 100/100. E2: aggregate Brier and fingerprint both improved -> E3. Promotion already recommended (comment #4835611333); result is consistent.
- **Benchmark 2026-06-29:** SHA `20185df20f4d2e69f1f85ee9e8297c8c87448f26`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (memory-only chore commit; tool code identical to holdout SHA 237173e5; all CI green; comment #4835773767)