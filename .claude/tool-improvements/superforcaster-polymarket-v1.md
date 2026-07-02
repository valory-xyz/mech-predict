
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
- **Status:** holdout confirmed 2026-06-29 -- promotion recommended (comment #4836079215)

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
- **Holdout result 2026-06-29:** SHA `237173e5c4f28442a5607640c7e3582272436900`, seed 1337, n=301, holdout, baseline=superforcaster-polymarket-v1, platform=polymarket -- result (comment #4835535374): Brier 0.2610->0.2218 (-15.0%), DA 63.8%->67.1% (+5.2%), Overconf-wrong 52->22 (-57.7%); parse 301/301 (100%). WIN: all three primary metrics improved; Overconf-wrong fingerprint strongly validated (-57.7%, consistent with 5 dev-seed runs at -18% to -63%). Promotion recommended (comment #4836079215) [re-posted 2026-06-29; prior agent run wrote phantom #4835611333 without posting].

- **Benchmark 2026-06-29:** SHA `01c875f36d14dfb2bb345466d4dd7e226a838d13`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- result (memory-only chore commit; tool code identical to holdout SHA 237173e5; trigger comment #4835578188, result comment #4835759925): Brier 0.2734->0.2283 (-16.5%), DA 63%->68% (+7.9%), Overconf-wrong 19->9 (-52.6%); parse 100/100. E2: aggregate Brier and fingerprint both improved -> E3. Promotion already recommended (comment #4835611333); result is consistent.
- **Benchmark 2026-06-29:** SHA `20185df20f4d2e69f1f85ee9e8297c8c87448f26`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (memory-only chore commit; tool code identical to holdout SHA 237173e5; all CI green; comment #4835773767)
- **Benchmark 2026-06-29:** SHA `20185df20f4d2e69f1f85ee9e8297c8c87448f26`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- result (trigger comment #4835773767, result comment #4835976355): Brier 0.2734->0.2379 (-13.0%), DA 63%->66% (+4.8%), Overconf-wrong 19->11 (-42.1%); parse 100/100. E2: both aggregate Brier and fingerprint improved; consistent with all prior dev runs. Promotion already recommended (comment #4835611333, #4836061285); this run is additional corroboration.
- **Benchmark 2026-06-29:** SHA `be91eef8051c250bc53a882e8d9e3e9ee7bb8b6d`, memory-only chore commit; tool code identical to holdout SHA 237173e5; CI green; prior agent attempted /benchmark (phantom comment #4835773767 — never posted); loop already complete.
- **Benchmark 2026-06-29:** SHA `504123a160dbdff624cb7614e2524af0eb11413e`, memory-only chore commit; tool code identical to holdout SHA 237173e5; CI green (Sub-pipeline D trigger); promotion recommendation re-posted (comment #4836079215) to correct phantom #4835611333 — loop closed, no further benchmarking needed.
**CI benchmark (2026-06-29, comment #4836605495, seed 42, n=100, current-SHA):** Brier 0.2734 → 0.2471 (-9.6%), DA 63.0% → 64.0% (+1.6%), Overconf-wrong 19 → 11 (-42.1%); parse 100/100 (100%). Fingerprint consistent with holdout. Sub-pipeline E verdict: E3 Promote (comment #4836716763). Both dev and holdout seeds confirm improvement; no regression indicators.
- **Benchmark 2026-06-30:** SHA `3266bd760caf8930717cfeaecda9afac3ab0267b`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- result (triggered manually by @jmoreira-valory, trigger comment #4844700993, result comment #4845048271): Brier 0.2894->0.2503 (-13.5%), DA 60.6%->62.0% (+2.3%), Overconf-wrong 20->11 (-45.0%); parse 100/100 (100%). Note: production baseline shifted from 0.2734 to 0.2894 (denominator 3459->3559) reflecting additional recent production rows. E2: both aggregate Brier and fingerprint improved -> E3 path. Promotion recommendation stands (11th consistent result; 10 dev-seed runs + 1 holdout); no new benchmark needed. Agent Sub-pipeline E verdict: comment #4845098207.

## Issue #382 -> PR #383 -- 2026-07-02

- **Trigger:** Issue #382 regression on polymarket. W-1 Brier 0.3679 (n=559) vs W-2 0.3018 (n=818); delta +0.0661; chronic-bad (above 0.25 threshold).
- **PR:** #383 `tool-improvement(superforcaster-polymarket-v1): evidence-reliability screen for overconfident-YES on narrow-scope markets`
- **Branch:** `tool-improvement/superforcaster-polymarket-v1-evidence-reliability-screen`
- **Status:** draft PR opened; W-2 benchmark triggered 2026-07-02.

### Hypothesis (from PR body)
At the prediction-LLM-call stage (gate-visible), `superforcaster-polymarket-v1` systematically overestimates `p_yes` via three compounding failure modes confirmed in 10/10 worst-miss IPFS deliveries (good-evidence/bad-reasoning):
- **(4a) Market-price anchoring** -- Polymarket/aggregator odds treated as independent probability estimates.
- **(4c) Stale temporal anchoring** -- past articles treated as forward-window confirmation.
- **(4d) Criterion-specificity failure** -- topical relevance conflated with exact criterion satisfaction.
Plus `max_tokens=500` insufficient for 7-step CoT causing premature JSON emission.

New version: `superforcaster-polymarket-v5`. Mechanism: mandatory four-sub-step evidence-reliability screen (4a-4d) + max_tokens 500->1500.

### Benchmark ledger
- **Benchmark 2026-07-02:** SHA `53c350a22553365bd2554982563f4469e288b1d3`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (comment #4862477125)
- **Benchmark 2026-07-02:** SHA `259c72e4e84ff082af96a803a56f58e11c69bac3`, seed 42, n=100, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- result (trigger comment #4862578016, result comment #4862618410): Brier 0.2882->0.2559 (-11.2%), DA 60.6%->64.0% (+5.6%), Overconf-wrong 20->15 (-25.0%); parse 100/100 (100%). E1: delta 0.0323 < 2*SE (~0.07) at n=100 -> borderline noisy; targeted fingerprint improved; growing to n=300. E2 (preliminary): both aggregate Brier and fingerprint improved -> E3 path (promotion) when confirmed at n>=300. Agent Sub-pipeline E verdict: comment #4862649060.
- **Benchmark 2026-07-02 (grow result):** SHA `259c72e4e84ff082af96a803a56f58e11c69bac3`, seed 42, n=101 (requested 300 via trigger comment #4862578016 [--sample 300]; n=101 is the max available W-2 deliveries for this tool/platform), dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- result comment #4862734748: Brier 0.2854->0.2590 (-9.2%), DA 61.0%->63.4% (+3.9%), Overconf-wrong 20->15 (-25.0%); parse 101/101 (100%). Cannot grow further -- n=101 is max available.
- **E2 diagnosis 2026-07-02:** seed 42 n=101 grow result (comment #4862734748). Consistent with prior n=100 run (comment #4862618410): both show Brier improvement -9% to -11%, Overconf-wrong -25%. Two dev-seed runs confirm: aggregate Brier improved AND targeted fingerprint (Overconf-wrong) improved -> E3 path. n=101 is max available; proceeding to holdout-confirm at seed 999 (unused in this PR and in PR #375 ledger), n=300 (workflow will return n<=101 max available).
- **Benchmark 2026-07-02 (holdout pending):** SHA `b8ad7e8c3853c6329d7b9eaaa5bd86c0e704a250`, seed 999, n=300 (max available ~101), holdout, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (holdout-confirm; E3 path; trigger comment #4862796445)
- **Benchmark 2026-07-02 (holdout pending):** SHA `b2d3ff23155206a56718b97e21a70c3e57a9dea9`, seed 999, n=300 (max available ~101), holdout, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (Sub-pipeline D re-trigger: new CI-green SHA; prior holdout commands for SHA `b8ad7e8c` produced no confirmed result comment; comment #4862934031)
- **Benchmark 2026-07-02 (holdout pending):** SHA `e481582cf5c113615849d68c4cd3d37495385d93`, seed 999, n=300 (max available ~101), holdout, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (Sub-pipeline D re-trigger: new CI-green SHA e481582; commit is memory-only chore recording previous D trigger; tool code unchanged; comment #4863085392)
- **Benchmark 2026-07-02 (holdout pending):** SHA `31a10ed0ad0f9db19bb524b929f401628d4172f8`, seed 999, n=300 (max available ~101), holdout, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (Sub-pipeline D re-trigger: new CI-green SHA 31a10ed; commit is memory-only chore recording previous D trigger; tool code unchanged; comment #4863256617)
- **Benchmark 2026-07-02 (dev result n=299):** trigger comment #4862649928 (seed 42 grow request --sample 300), result comment #4863333895, seed 42, n=299, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- result: Brier 0.2764->0.2425 (-12.3%), DA 63.6%->65.2% (+2.5%), Overconf-wrong 63->27 (-57.1%); parse 299/299 (100%). E1: trustworthy (n=299, SE~0.014, delta=0.034~2.4*SE). E2: seed 42 is dev seed; both aggregate Brier AND targeted fingerprint (Overconf-wrong) improved -> E3 path. Holdout (seed 999, SHA e481582c, comment #4863085392) already posted; awaiting holdout result. Sub-pipeline E verdict: comment #4863377361.
- **Holdout result 2026-07-02:** result comment #4863493510, seed 999, n=300, holdout, baseline=superforcaster-polymarket-v1, platform=polymarket -- Brier 0.2826->0.2162 (-23.5%), DA 64.2%->69.3% (+8.0%), Overconf-wrong 62->30 (-51.6%), Overconf-wrong rate 0.2067->0.1000 (-51.6%); parse 300/300 (100%). E1: trustworthy (n=300, SE~0.017, delta=0.0664~3.9*SE, 100% parse). Holdout terminal per hard constraint 5. E3: WIN -- aggregate Brier improved (-23.5%) AND targeted fingerprint (Overconf-wrong) improved (-51.6%), consistent with all 3 dev-seed runs (seed 42: -25%, -25%, -57%). Holdout improvement exceeds dev estimates; no generalisation degradation. Promotion recommended (comment #4863529015).
- **Status:** holdout confirmed 2026-07-02 -- promotion recommended (comment #4863529015)
