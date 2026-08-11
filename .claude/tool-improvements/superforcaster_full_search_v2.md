# superforcaster_full_search_v2 -- Investigation Memory

Parent tool: superforcaster_full_search (untouched; v2 is a new sibling)
Issue: #423 (Polymarket Brier regression)
PR: #424

## Baseline (C-1 window, from issue #423)

| Window | n | Brier | WM fraction |
|---|---|---|---|
| C-2 (2026-07-13-2026-07-21) | 105 | 0.2300 | 58% |
| C-1 (2026-07-22-2026-08-03) | 105 | 0.2978 | 91% |

Delta +0.068; within-WM Brier 0.2462->0.3048 (+0.0586) confirms code-level regression, not mix artefact.

## Hypotheses

### H1 -- WM screen applied unconditionally (CONFIRMED -> IMPLEMENTATION REVISED)

Source: 9 IPFS delivery inspections from worst C-1 WM rows (Brier > 0.45).

Finding: In 8/9 deliveries:
1. Tool retrieved Polymarket/Kalshi market pages and treated embedded odds (~65% YES) as independent evidence (circular reasoning).
2. Evidence showed topical relevance ("Speaker X often uses word Y") but question was event-specific ("Will X say Y during EVENT Z?"); tool failed to apply event-specific base rates (30-60%), instead predicting 75-90%.

Fix attempted (initial PR, commit d2cb22a): Added word_mention_check field to PredictionResult schema -- model must complete WM screen before committing to p_yes. Switched to beta.chat.completions.parse (Structured Outputs).

First benchmark result (2026-08-10, comment 5253217784, human-triggered, seed 42, n=300):
- Brier: 0.2829 -> 0.3203 (+13.2%)
- DA: 60.0% -> 55.4% (-7.7%)
- Parse: 100%
- VERDICT: NEGATIVE

Diagnosis of benchmark failure (H1-R1): word_mention_check in PredictionResult fired unconditionally for ALL market types. The W-2 replay dataset (~58% WM) had a larger share of non-WM markets. For those, the WM-specific conservative base-rate logic (30-60% suppression) applied incorrectly, pulling p_yes down on markets where high-confidence YES predictions were warranted. Result: universal conservative suppression -> DA drop on non-WM markets.

Implementation revision (H1-R1, commit 239f321, 2026-08-11):
- Added _is_word_mention_market(question) regex classifier
- Added StandardPredictionResult (same schema, no word_mention_check field)
- run() dispatches to PredictionResult (WM screen) for WM markets and StandardPredictionResult (no screen) for all others
- Tests updated: 28/28 pass, new TestIsWordMentionMarket (7+6 cases)

Status: CI pending on commit 239f321. Benchmark to be posted once all checks green.

## Ruled-out hypotheses

(none -- H1 confirmed from evidence, H1-R1 is the implementation revision)

## Benchmark ledger

- Benchmark 2026-08-04 (errored): SHA d2cb22a, seed 42, n=300 -- infrastructure failure at replay step (run 30888339768); no scoring result.
- Benchmark 2026-08-10 (human-triggered, seed 42, n=300): SHA d2cb22a, baseline superforcaster_full_search vs superforcaster_full_search_v2; Brier 0.2829->0.3203 (+13.2%), DA 60.0%->55.4% (-7.7%), parse 100%. NEGATIVE. Root cause: unconditional WM screen on non-WM markets.
- Next benchmark pending: SHA 239f321 (H1-R1 revision); /allow-benchmark granted 2026-08-11 by jmoreira-valory; awaiting CI green before posting command.
