# Family ledger: superforcaster

Standing `open` / `ruled-out` / `confirmed-fixed` list, shared across every
`superforcaster*` version and sibling. Step 3 reads this FIRST and de-dups
against it; Step 5.7 writes it on every exit.

## open

- [#493] `superforcaster-market-aware` -- the `evidence_reliability_screen` is a one-directional skeptic: every sub-check pushes the estimate down and (d) fires on essentially every forward-looking row, so the NO tail is unguarded (confident-NO 45% of Brier mass, realized 0.354 vs stated 0.057) -- `calibration` -- status: proposed PR (`tool-improvement/superforcaster-market-aware-symmetric-disconfirmation`, new version `superforcaster-market-aware-v1`)
- [#493] `superforcaster-market-aware` -- 2 of 8 independently-sampled worst-miss rows carried no current price for the asset the threshold is on (RKLB spot listed as a stale $72.95; SPY absent entirely) -- `retrieval` -- status: partial, gate-invisible, hand-off to retrieval / tournament-mode
- [#493] `superforcaster-market-aware` -- the tournament harness passes no `request_context`, so the *market-aware* half of this tool has never been exercised by any scored row -- status: already owned by open PR #467, no duplicate filed

## ruled-out

- [#493] `superforcaster-market-aware` -- *the tool misreads "hit (LOW/HIGH) $X" barrier/touch semantics* -- `question-semantics` -- **killed by:** the per-family table; the confident-NO defect is the same size in `utterance` (realized 0.320) and `other` (0.250) rows that contain no barrier at all (critic: artifact). Barrier handling is a sub-case of the NO-tail defect, not a mechanism of its own.
- [#493] `superforcaster-market-aware` -- *version-rollover / composition artifact* -- `calibration` -- **killed by:** one CID (`bafybeiecvkdx2...`) covers 100% of rows in both T-1 and T-2 (critic: artifact).
- [#493] `superforcaster-market-aware` -- *uniformly compressed confidence across every question type (would unlock a clamp / temperature scale under Step 5 (v))* -- `calibration` -- **killed by:** the compression is one-sided, 6.2x error ratio on the NO tail vs 2.7x on the YES tail; the output distribution is not itself the named failure, so a clamp is a symptom fix here.

## confirmed-fixed

- [#374] `superforcaster-polymarket-v1` -- overconfident **YES** on narrow-criterion questions (topical relevance read as criterion satisfaction) -- fixed in PR #375 as `superforcaster-polymarket-v4`'s four-part `evidence_reliability_screen` (prediction-LLM-call stage); holdout seed 1337 n=301: Brier 0.2610 -> 0.2218, overconfident-wrong -57.7%. **Note for future runs:** that screen is the direct ancestor of the #493 finding above -- it corrected the YES tail and left the NO tail unguarded.
