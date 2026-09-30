# Investigation memory: `superforcaster-polymarket-v4`

Per-tool ledger. Append a new section per investigation; never overwrite.
Family ledger: `_family-superforcaster-polymarket.md`.

## Issue #503 -- 2026-09-24 -- path (a)

- **Trigger:** `regression` on `polymarket`, TOURNAMENT count-window T-1
  (`2026-09-17` -> `2026-09-23`, n=105); headline Brier `0.3146` vs T-2 `0.2207`
  (n=105, `2026-09-14` -> `2026-09-17`), delta `+0.0939`.
  Reproduced exactly: `n=105, brier=0.3146`, delta `+0.0940`.
  Chronic-bad band (`0.3146 >= BRIER_LEVEL_THRESHOLD 0.25`, `n >= 105`)
  -> default PROPOSE.
- **Production cross-check (tournament-parity caveat):** production v4 rows in
  the same 45d period: n=91, newest `predicted_at` 2026-08-24 (30d stale), zero
  rows inside the T-1 span. Reported side-by-side, never blended. All-time
  production pool n=264, Brier 0.3155.
- **PR:** #<PR> (branch `tool-improvement/superforcaster-polymarket-v4-window-criterion-screen`),
  ships new version `superforcaster-polymarket-v5`.
- **Top survivor / confirmed hypothesis:** at the **prediction-LLM-call stage**,
  v4 evaluates an "at least once during a window" resolution criterion (`hit
  (HIGH/LOW) $X Week of ...`, `... by <date>`, `... during <event>`, `... this
  week`) as a **point-in-time** test -- it reports whether the condition holds
  *now* / is *already scheduled*, and never prices the chance it is met at any
  point over the remaining window. (`question-semantics` lens, gate-visible per
  Step 3a.)
- **Stage + code site:** prediction-LLM call + structured-output schema --
  `PredictionResult` (no field forces the temporal structure of the criterion to
  be resolved) and the ordered reasoning list in `PREDICTION_PROMPT`.
  `evidence_reliability_screen` (the v4 step-4 screen from #374/PR #375) screens
  evidence *reliability* but never the criterion's *temporal shape*.
- **Localized cells (T-1, aggregate Brier 0.3146):**
  `difficulty=medium (n=33, brier=0.4436, +0.1290)`;
  `disagree_bucket=large (n=70, brier=0.4203, +0.1056)`;
  `horizon<=1d (n=30, brier=0.3886, +0.0739)`.
  Question-template cut (the cell that actually localizes):
  `at-least-once (n=60, brier=0.3818, 73% of T-1 Brier mass)` vs
  `point-in-time (n=41, brier=0.2086)`; excluding the non-researchable utterance
  markets (Step 4b / #486), `at-least-once ex-utterance n=44, brier=0.4201,
  market 0.1054, edge -0.3148, 56% of T-1 Brier mass`.
- **Step 2.5 composition control:** NOT a composition artifact. Single CID
  (`bafybeiefu5cetn...h5roq`, the registered current version) at share 1.00 in
  BOTH windows; single model. Within matched strata the Brier genuinely rose:
  `difficulty=medium 0.2442 -> 0.4436` at identical share 0.31;
  `disagree=large 0.3260 -> 0.4203` at identical share 0.67;
  `at-least-once 0.2398 -> 0.3818` while its SHARE FELL 65% -> 57%
  (mix moved toward the better stratum, yet Brier rose).
- **Evidence sample (worst-miss rows, T-1; coverage 44/44 and 24/24 worst):**

  | question (truncated) | p_yes | mkt | outcome | evidence_finding |
  |---|---|---|---|---|
  | Will South Korea ETF (EWY) hit (LOW) $190 Week of Sep 21 2026? | 0.030 | 0.94 | 1 | good-evidence/bad-reasoning -- evidence gives NAV **$189.78** on Sep 21: the barrier was ALREADY breached |
  | Will the Greenland agreement text be released by September 23, 2026? | 0.010 | 0.525 | 1 | good-evidence/bad-reasoning -- 5h-old source "No text of the agreement has been released" read as a verdict, not a starting state |
  | Will Trump meet with Mette Frederiksen by September 30, 2026? | 0.030 | 0.84 | 1 | good-evidence/bad-reasoning -- "No confirmed schedule exists" read as ~0 for a 9-day window |
  | Will Meta (META) hit (HIGH) $750 Week of Sep 21 2026? | 0.030 | 0.745 | 1 | good-evidence/bad-reasoning -- evidence: META at $741 (+11%), target raised to $796; 1.2% away, 5 sessions |
  | Will Gold (XAUUSD) hit (LOW) $4,300 Week of Sep 21 2026? | 0.030 | 0.53 | 1 | good-evidence/bad-reasoning -- evidence quotes the criterion verbatim ("at any point ... any 1-minute candle") and spot $4,361 |
  | Will Tesla (TSLA) hit (HIGH) $382.50 Week of Sep 21 2026? | 0.010 | 0.33 | 1 | good-evidence/bad-reasoning -- evidence carries spot $369.51 AND the options expected move +/- $6.01 |

  **METHOD (from #501, re-used):** tournament rows carry no `deliver_id`, but the
  `tournament-predictions` artifact's `tournament_predictions.jsonl` carries
  `source_content` inline, joined to `tournament_scored.jsonl` on `row_id`.
  Coverage was 44/44 -- NOT `delivery-unreadable`.
- **Decisive outcome-free evidence (barrier-ladder coherence).** For a fixed
  underlying + direction + window, P(touch $K) must be monotone in K. Over 15
  ladders / 80 ordered pairs on the T-1+T-2 touch set:
  v4 **31.2%** inversions, market price **5.0%**, realised outcomes **0.0%**.
  Worked example (Silver XAGUSD HIGH, week of Sep 14): $65 -> 0.010,
  $66 -> 0.980, $67 -> 0.020, $68 -> 0.970, $69 -> 0.005. No evidence set makes
  those jointly coherent, so the defect is intrinsic to the reasoning step, not
  to retrieval.
- **Directional confirmation:** of the T-1 at-least-once ex-utterance rows priced
  >20pp BELOW the market, **20/20 resolved YES**.
- **Adversarial critic (all three survived):**
  - *selection* -- holds on T-2 (disjoint window) and on the **production arm**
    (different question mix, never used to generate): production at-least-once
    ex-utterance n=30, Brier 0.3003, market 0.1454, edge -0.1549.
  - *few rows* -- stripping the 1/2/3/5 worst T-1 rows leaves
    0.4071 / 0.3935 / 0.3796 / 0.3509 vs the point-in-time comparator 0.2086.
  - *metric artifact* -- the monotonicity test uses no outcomes at all; Step 2.5
    rules out version/mix; the cell's share fell while its Brier rose.
- **Refuted hypotheses (with killing rows -- so the next run de-dups):**
  - `calibration`: "uniform output miscalibration; a clamp / temperature scale
    would fix it" -- **killed by** the within-window contrast: T-1 point-in-time
    rows are MORE extreme (61.0% at p_yes<=0.05 or >=0.95) yet score BETTER
    (0.2086) than at-least-once rows (50.0% extreme, 0.4201). Extremity does not
    track loss, so the distribution is not the failure. (critic: artifact.)
    Also independently ruled out at #501.
  - `retrieval`: "bad/empty/off-topic evidence" -- **killed by** 44/44 readable
    tournament deliveries with populated, on-point, freshly-dated `organic`
    results (and 10/10 on the production sample); plus the ladder-inversion
    argument, which is evidence-independent. (critic: artifact.)
  - `evidence-staleness`: "anchors a stale prior-period figure" -- **killed by**
    the misses being same-day, same-underlying ladder rungs with hours-old
    sources; the figures the tool used were current, it used them for the wrong
    question.
  - `retrieval`/mix: "T-1 rise is a version-rollover or question-mix composition
    artifact" -- **killed by** Step 2.5 (single CID share 1.00 both windows;
    within-stratum rise; at-least-once share FELL 65% -> 57%).
- **Known-blocked, deliberately NOT re-derived (from the family ledger):**
  #501's price-threshold hypothesis (BLOCKED on gate attributability, 6.8%
  of the replay corpus), #486's non-researchable utterance cluster
  (`deselect`, path (b) by Step 4b), #453's TYPE-A outlet-name fix
  ("do not re-derive; revive #453").
- **Typed action emitted:** none -- the two that apply (`deselect` #486,
  `widen-sample` #502) are ALREADY OPEN for this tool; per the Step-5 dedup rule
  they are referenced, not duplicated.
- **Mechanism disrupted:** the prediction stage must now classify the resolution
  criterion as WINDOW (met at any point in a stated span) or POINT (decided at a
  single stated instant) and, when WINDOW, derive the probability from the number
  of chances the span contains instead of reporting whether the condition holds
  at prediction time.
- **Pre-PR sanity (Step 6.5):** see PR body.
- **Status:** path (a) -- opened draft PR; PR-CI cached-replay pending on W-2.
  **Gate-visibility note carried to the PR:** the at-least-once family is 83.0%
  of the production replay corpus by rows and 89.6% by Brier mass, but 71.6% of
  rows are the #486 non-researchable utterance cluster; the *researchable* core
  this PR claims is 30/264 rows (11.4%) and 10.8% of Brier mass. The gate can see
  this fix (unlike #501's 6.8% price-threshold cell), but the claimed core is
  thin -- read the W-2 delta with that in mind.
