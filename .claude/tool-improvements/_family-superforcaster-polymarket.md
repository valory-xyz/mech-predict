# Family ledger: superforcaster-polymarket

Shared across all versions/siblings of the `superforcaster-polymarket` family.
Reconciled on read (Step 3): entries referencing a PR are moved to
`confirmed-fixed` on merge, or to `ruled-out` with `abandoned:` when the PR is
closed unmerged.

Reconstructed 2026-09-24 from the repo copy plus the `memory ledger delta`
blocks on closed issues #501, #500, #495 (the comment-embedded deltas that
path-(b)/(c) runs leave behind). The closed-issue listing returned 20 rows,
which EQUALS the query limit -- older ruled-out entries may exist beyond that
window.

## open

- [#503] `superforcaster-polymarket-v4` -- prediction-LLM-call stage evaluates an
  "at least once during a window" criterion (`hit (HIGH/LOW) $X Week of ...`,
  `... by <date>`, `... during <event>`, `... this week`) as a POINT-IN-TIME test:
  it reports whether the condition holds now / is already scheduled and never
  prices the chance it is met at any point over the remaining window.
  `question-semantics`, gate-visible. CONFIRMED (cross-lens verify on the
  production arm + all three critic attacks survived; 44/44 evidence coverage).
  Fingerprint: T-1 at-least-once ex-utterance n=44 Brier 0.4201 vs point-in-time
  n=41 Brier 0.2086 (same code, same window); 20/20 of the rows priced >20pp
  below market resolved YES; barrier-ladder inversion rate 31.2% (v4) vs 5.0%
  (market) vs 0.0% (outcomes) over 80 ordered pairs -- an OUTCOME-FREE coherence
  violation. Gate: the family is 83.0% of the production replay corpus (89.6% of
  its Brier mass), of which the researchable core this PR claims is 30/264 rows
  (11.4%, 10.8% of mass) -- visible to the gate, unlike #501's 6.8% cell, but
  thin -- status: proposed PR #<PR>, ships `superforcaster-polymarket-v5`.
- [#501] `superforcaster-polymarket-v4` -- prediction-LLM-call stage treats an
  asset-price threshold question as a deterministic spot-vs-threshold comparison
  and emits near-certain `p_yes`, never pricing the residual move to
  `market_close_at`. `question-semantics`, gate-visible. CONFIRMED -- status:
  **BLOCKED on gate attributability, NOT refuted** (PR-CI would score a corpus
  that is 6.8% price-threshold vs 71.7% in the tournament arm). Revisit when the
  replay corpus can carry it (deployment-review #502, `widen-sample`).
  **Related to but distinct from #503**: #501 is about pricing the residual move
  to a single close; #503 is about the criterion's temporal structure (any point
  in a span vs one instant). #503's cell is gate-visible where #501's was not.
- [#483, re-confirmed #495, #500] `superforcaster-polymarket-v4` -- C-1 level
  carried by a non-researchable market cluster (`Will "<word>" be in the
  headlines this week?` / `Will Trump say|post "<word>" ...`): 58-69% of rows,
  70-79% of Brier mass, tool 0.4383 vs market 0.1824. `question-semantics` /
  market-selection -- status: deployment-review **#486** open, action `deselect`.
  Re-confirmed on #503's production pool: 189/264 rows (71.6%), 78.8% of mass.
- [#501, #503] `superforcaster-polymarket-v4` -- the tournament count-window
  trigger fires near its own noise budget at `COUNT_WINDOW_N = 105`
  -- status: evidence attached to deployment-review **#502**, `widen-sample`.
- [#483, #495, #500] CROSS-CUTTING PRODUCER DEFECT -- the count-window triage
  tier reads the frozen `datasets/logs/` dir while the daily writer is skipped
  under `USE_MECH_ANALYTICS_ROWS=true` -- status: recorded on #486; producer owns
  it. #503 observes the downstream symptom: production v4 rows stop at
  2026-08-24 (30d before the issue) while the tournament arm is current to
  2026-09-23.
- [#501] METHOD NOTE (every tournament-sourced issue, any tool): tournament rows
  carry no `deliver_id`, but `tournament_predictions.jsonl` (the
  `tournament-predictions` CI artifact) carries `source_content` inline, joined
  to `tournament_scored.jsonl` by `row_id`. #503 re-used this for 44/44 coverage
  -- status: agent-skills draft PR valory-xyz/agent-skills#270.

## ruled-out

- [#503] `superforcaster-polymarket-v4` -- uniform output miscalibration
  ("a clamp / temperature scale would fix it") -- `calibration` -- **killed by:**
  T-1 point-in-time rows are MORE extreme (61.0%) yet score BETTER (0.2086) than
  at-least-once rows (50.0% extreme, 0.4201); extremity does not track loss
  (critic: artifact). Independently ruled out at #501.
- [#503] `superforcaster-polymarket-v4` -- bad/empty/off-topic evidence --
  `retrieval` -- **killed by:** 44/44 tournament deliveries readable with
  populated, on-point, hours-old `organic` results; several quote the
  at-any-point criterion verbatim; plus the outcome-free ladder-inversion
  argument (critic: artifact).
- [#503] `superforcaster-polymarket-v4` -- stale prior-period figure anchoring --
  `evidence-staleness` -- **killed by:** misses are same-day, same-underlying
  ladder rungs on hours-old sources; the figures were current, the question was
  misread.
- [#501, #503] `superforcaster-polymarket-v4` -- T-1 regression is a
  version-rollover / question-mix composition artifact -- `calibration`/mix --
  **killed by:** single CID and single model in BOTH windows; at #503 the
  at-least-once share FELL 65% -> 57% while its Brier rose 0.2398 -> 0.3818
  (critic: artifact).
- [#501] `superforcaster-polymarket-v4` -- T-1 loss carried by the
  non-researchable utterance cluster -- `question-semantics` -- **killed by:**
  Step 4b gives only 13.3% of T-1 rows on the tournament arm vs 58-69% on the
  production windows (critic: artifact). NB this refutes it *for the tournament
  T-windows only*; it remains CONFIRMED on the production windows (#486).
- [#422] `superforcaster-polymarket-v4` -- utterance-screen for TYPE A/B
  misclassification -- `question-semantics` -- abandoned: PR #425 closed,
  merged=false (reconciled #495).
- [#452] `superforcaster-polymarket-v4` -- outlet-name keyword matches scored as
  TYPE A; TYPE B base-rate not mechanically bound -- `question-semantics` --
  abandoned: PR #453 closed, merged=false (reconciled #495) **despite a positive
  gate (-24.5% Brier, -84.2% overconfident-wrong). Do not re-derive; revive
  #453.**
- [#455] `superforcaster-polymarket-v4` -- free-text prompts silently disable
  retrieval -- `retrieval` -- abandoned: PR #456 closed, merged=false, but the
  fix merged in-place onto v4 (`58990786`, 2026-09-01/02).
- [#439] `superforcaster-polymarket-v2` -- structured outputs to enforce the
  7-step CoT -- abandoned: PR #441 closed, merged=false (reconciled #495).

## confirmed-fixed

- [#374] `superforcaster-polymarket-v1` -- systematic overconfident-YES at the
  prediction-LLM-call stage -- fixed in PR #375 as
  `superforcaster-polymarket-v4` (step-4 evidence-reliability screen: odds
  filter, forward-looking-intent discount, TYPE A/B temporal classification,
  criterion-specificity). Holdout confirmed 2026-06-29 (seed 1337, n=301):
  Brier 0.2610 -> 0.2218 (-15.0%), overconfident-wrong -57.7%.
- [#332] `superforcaster-polymarket-v1` -- prediction stage conflated topical
  relevance with satisfaction of a narrow resolution criterion -- fixed as
  `superforcaster-polymarket-v2` (PREDICTION_PROMPT step-6
  criterion-specificity check).
