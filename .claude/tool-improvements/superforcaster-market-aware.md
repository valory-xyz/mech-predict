# superforcaster-market-aware -- investigation log

Per-tool investigation memory for the `tool-improvement` pipeline. Append a new
section per issue/PR; never overwrite prior content (oldest-first).

## Issue #493 -- 2026-09-18 -- path (a)

- **Trigger:** `level_floor` on `polymarket`, TOURNAMENT count-window T-1 (`2026-09-09T03:42:21Z` -> `2026-09-17T04:19:15Z`); headline Brier `0.2681` (n=105), BSS vs base rate `-0.1170`. T-2 = preceding 75 rows only (Brier `0.2125`), below the 105 floor -> Step 1.5 branch 2: **level-only sufficiency**, regression framing unevaluable.
- **Reproduced exactly:** `n=105, brier=0.2681, BSS=-0.1170` (|delta| = 0.0000).
- **Source caveat:** tournament arm, not production. Production rows for this tool within 45d = **0** (it has never been deployed); the tournament harness passes **no** `request_context`, so every scored row is the tool's **blind mode** -- `market_prob_seen=None`, `market_reconciliation` inert. PR #467 (open) is the fix for that and is NOT this agent's to make.
- **PR:** branch `tool-improvement/superforcaster-market-aware-symmetric-disconfirmation`, new version `superforcaster-market-aware-v1`.
- **Top survivor / confirmed hypothesis:** the `evidence_reliability_screen` stage is a **one-directional skeptic** -- all four sub-checks push the estimate DOWN, and (d) ("no TYPE A evidence directly confirms the criterion -> add uncertainty toward the base rate") fires on essentially every row because a forward-looking criterion cannot be confirmed before it resolves. The stage executes it as "absence of confirming evidence => NO", so the NO tail is unguarded while the YES tail (which must overcome the screen) is not. (`calibration` lens generated; verified by delivery reads; gate-visible per Step 3a.)
- **Stage + code site:** prompt construction / prediction-LLM call -- `PredictionResult.evidence_reliability_screen` .. `p_independent` region and the matching numbered step in `PREDICTION_PROMPT`.
- **Localized cells:** none met the Step-2 bar (`brier > agg + 0.05 AND n >= 30`); closest was `difficulty=hard` (n=23, brier 0.4138, +0.1458). Per Step 2's `level_floor` exit, a global-property hypothesis was formed from the raw-row sample instead.
- **The asymmetry (T-1 + T-2, n=180, single CID both windows):**
  | slice | n | stated p_yes | realized YES | Brier | share of Brier mass |
  |---|---|---|---|---|---|
  | confident-NO (`p_yes <= 0.20`) | 65 | 0.057 | **0.354** (95% CI [0.246, 0.462]) | 0.3044 | **44.9%** |
  | confident-YES (`p_yes >= 0.80`) | 54 | 0.938 | 0.833 | 0.1556 | 19.1% |
  Error ratio 6.2x on the NO side vs 2.7x on the YES side.
- **Evidence sample (worst-miss rows; `source_content` read from `tournament_predictions.jsonl`, 180/180 readable):**
  | question (truncated) | p_yes | outcome | evidence_finding |
  |---|---|---|---|
  | WTI Crude Oil closes above $95 on September 9? | 0.005 | YES | good-evidence/bad-reasoning (evidence: "Crude Oil rose to 94.36 on September 9", bullish tanker-strike catalyst) |
  | Will Micron (MU) hit (HIGH) $960 Week of September 14? | 0.01 | YES | good-evidence/bad-reasoning (evidence: closed $932.97 17h earlier; 5.83% one-day move the day before) |
  | WTI Crude Oil closes above $96 on September 9? | 0.01 | YES | good-evidence/bad-reasoning ("prices remained elevated near $96 a barrel") |
  | Will Trump say "Make America Healthy Again" ...? | 0.02 | YES | good-evidence/bad-reasoning (evidence is dominated by MAHA as an administration slogan) |
  | Will South Korea ETF (EWY) hit (LOW) $187 Week of September 14? | 0.07 | YES | good-evidence/bad-reasoning (evidence implies spot ~$188.8; threshold 1% away; touch not close) |
  | Will Silver (XAGUSD) hit (LOW) $64 Week of September 7? | 0.18 | YES | good-evidence/bad-reasoning ("$65.43 per ounce, continuing their downward trend") |
  | Will Rocket Lab (RKLB) hit (LOW) $62 Week of September 14? | 0.03 | YES | bad/empty-evidence (only stale/wrong spot "$72.95"; current level absent) |
  | Will S&P 500 (SPY) hit (LOW) $750 Week of September 14? | 0.001 | YES | bad/empty-evidence (no current SPY level in the delivery) |
- **Adversarial critic (all three attacks survived):**
  - *selection* -- holds on the fully disjoint T-2 window (realized 0.208 vs stated 0.056) and within all four question families independently (price-barrier 0.400, close-threshold 0.364, utterance 0.320, other 0.250).
  - *few rows* -- strip the 5 worst confident-NO rows and it is still realized 0.300 vs stated 0.062.
  - *metric artifact* -- one CID in both windows (no version mix); the bin discriminates correctly against each window's base rate (lift -0.161 T-1, -0.232 T-2), so the defect is the magnitude of stated confidence, not the direction.
- **4b non-researchable screen:** the `utterance` family (n=77) is substantively non-researchable, but it does NOT carry the finding -- excluding it, T-1 Brier is still `0.2529` (> 0.25) and the confident-NO defect is *worse* (realized 0.500 vs stated 0.058). Path (a) is therefore not blocked by 4b.
- **Economic value: HIGH.** On the confident-NO slice, edge vs the market price is `-0.1653`; on its large-disagreement subset (n=37, the bet-eligible zone) `-0.2798` with `market_brier=0.1827` -- a soft, beatable price, i.e. the adverse-selection zone the trader actually bets.
- **Refuted / not-pursued hypotheses (so the next run de-dups):**
  - `question-semantics: the tool misreads "hit (LOW/HIGH) $X" barrier semantics` -- **refuted as the primary mechanism**: killed by the family table, the confident-NO defect is the same size in `utterance` (0.320) and `other` (0.250) rows that carry no barrier semantics at all. Barrier handling is a *sub-case* of the NO-tail defect, addressed by sub-answer (2) of the fix, not a mechanism of its own. Critic: artifact.
  - `version-mix / composition artifact` -- **ruled out**: a single CID (`bafybeiecvkdx2...`) accounts for 100% of rows in BOTH windows.
  - `calibration: uniformly compressed confidence across every question type (clamp/temperature carve-out)` -- **ruled out**: the compression is one-sided (6.2x on NO vs 2.7x on YES), so Step 5 condition (v)'s output-distribution carve-out does not apply and a clamp would be a symptom fix.
  - `retrieval: bad/empty evidence` -- **partial, not the top survivor**: 2 of 8 independently-sampled worst-miss rows (RKLB, SPY) had no current price in the delivery. Real, but gate-invisible (cached replay injects `source_content`) -> retrieval owner / tournament mode, filed as a follow-up.
- **Typed action emitted:** none. The blind-mode / market-context finding is already owned by open PR #467; a duplicate would be noise.
- **Mechanism disrupted:** the screen gets a mandatory NO-side counterpart -- a `disconfirmation_check` the stage must fill (name the disconfirming TYPE A source, or quantify the gap to threshold / the repeat-event frequency) plus a committed `base_rate_floor`, both declared before any probability, so a tail NO must be *earned* rather than defaulted into.
- **Pre-PR sanity (Step 6.5):** import OK; `autonomy packages lock --check` -> Verification successful; trader-contract parse check OK (on-chain key set byte-identical to the parent's); 164/164 package tests and 1860/1860 benchmark tests pass; +57 creative LOC pre-lint in the tool source (the other 23 changed lines are the mechanical `superforcaster-market-aware` -> `-v1` rename). **W-2 is the only scored gate.**
- **Status:** path (a): opened draft PR; PR-CI cached-replay pending.
