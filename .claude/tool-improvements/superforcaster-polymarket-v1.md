## Issue #332 -- 2026-06-09

- **Trigger:** regression on `polymarket` W-1 (`2026-06-02T04:17:54Z` -> `2026-06-09T04:17:54Z`); headline Brier `0.2323` (n=656) vs W-2 `0.1470` (n=222) (delta +0.0853); trigger=`regression`.
- **PR:** branch `tool-improvement/superforcaster-polymarket-v2-criterion-specificity`.
- **Status:** draft PR opened (path a).

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
