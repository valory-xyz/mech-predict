# Family ledger — superforcaster (Polymarket)

## Version history

| Version | Status | Key change | Brier (holdout/benchmark) | Overconf-wrong |
|---|---|---|---|---|
| v1 | retired | baseline | ~0.147 (W-2) | — |
| v2 | production (active) | Step 6 criterion check added | 0.2964 (W-2, n=255) | 51/200 |
| v3 | — | (no v3 in lineage) | — | — |
| v4 | broken | 4a–4d evidence-reliability screens | 0.2218 (holdout seed 1337 n=301) | 22/200 |
| v5 | pending-holdout | Structured outputs (v2+) | 0.3113 (seed 42 n=200 vs v2) | 52/200 |

## V4 incident
V4 introduced forward-looking discount, TYPE A/B temporal classification, prediction-market-circularity filter, and criterion-specificity gate. Holdout showed large improvement (−15% Brier, −57.7% overconf-wrong). Deployed but output contract was violated (prose + trailing JSON → 38 deliveries, 0 parseable). V2 remains effective production tool.

## V5 (2026-08-13)
Structured outputs fix applied to v2 (not v4). One mechanism: `client.beta.chat.completions.parse` + `PredictionResult` schema, text reasoning fields before numeric. Max_tokens 500→4096. Benchmark vs v2: Brier −1.7%, DA +6.7%, overconf-wrong flat. Holdout pending (seed 1337 n=300).

## Known overconfidence mechanism (open)
Tool systematically produces p_yes ≥ 0.90 on narrow-criterion questions when evidence is topically relevant but doesn't directly confirm criterion satisfaction (TYPE A vs TYPE B evidence confusion; forward-looking vs backward-looking failure). V4's 4a–4d screens addressed this; v5 does not include them. Next version (v6) should integrate v4's reasoning improvements with v5's output-contract compliance.

## Outstanding hypothesis (carry forward)
The correct fix for overconfident-YES tail is: add v4's 4a–4d explicit evidence-reliability screens to v5's code. This is a gate-visible change (prompt construction / LLM call stage). Single mechanism, single file. Warrants a follow-up tool-improvement issue once v5 is promoted.