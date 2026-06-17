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


---

## Issue #366 -- 2026-06-17

- **Trigger:** level_floor on `polymarket` W-1 (`2026-06-10T04:26:56Z` -> `2026-06-17T04:26:56Z`); headline Brier `0.2565` vs W-2 `0.3253` (Delta `-0.0688`).
- **PR:** branch `tool-improvement/superforcaster-polymarket-v1-forward-looking-evidence-discount`.
- **Confirmed hypothesis:** At the prediction-LLM-call stage, the tool anchors on (a) prediction-market odds embedded in Serper search results (e.g. "Leading. 92%. Elon Musk" from polymarket.com pages) treating them as independent factual evidence, and (b) forward-looking intent language ("is set to", "expected to", "plans to", "confirmed to attend") treating announced intentions as near-certain outcomes, producing extreme YES predictions (p_yes=0.93-0.97) that resolve NO.
- **Stage + code site:** `prediction-LLM-call` stage at `PREDICTION_PROMPT` step 4 aggregation (gate-visible per Step 3a map).
- **Localized cells:** `by_tool_category=business (n=104, brier=0.3847)`, `by_outcome=True (n=95, brier=0.3386)`.
- **Evidence sample (worst-miss rows):**
  | question (truncated 80 char) | p_yes | outcome | evidence_finding |
  |---|---|---|---|
  | Will Elon Musk be on-stage at a bell ceremony commemorating SpaceX IPO? | 0.960 | 0 | good-evidence/bad-reasoning (market odds: "Leading. 92%. Elon Musk") |
  | SpaceX IPO: Trading Halted for Volatility? | 0.930 | 0 | good-evidence/bad-reasoning (speculative: "SPCX is expected to experience trading halts") |
  | Will Zohran Mamdani stream on Twitch again by June 12? | 0.970 | 0 | good-evidence/bad-reasoning (intent: "plans to launch a livestream series") |
  | Will Elon Musk attend UFC Freedom 250? | 0.970 | 0 | good-evidence/bad-reasoning (intent: "Elon Musk is set to attend UFC Freedom 250") |
  | Will Trump say "Memory" this week? | 0.960 | 0 | good-evidence/bad-reasoning (market odds: "Will Trump say Memory this week? 97%") |
  | Will Trump post "China" on Truth Social this week? | 0.930 | 0 | good-evidence/bad-reasoning (past-extrapolation: prior Trump posts mentioning China) |
  | Will Cam Skattebo attend UFC Freedom 250? | 0.960 | 0 | good-evidence/bad-reasoning (intent: "UFC Freedom 250 is scheduled") |
  | Will Trump say "Six Seven" during G7 events? | 0.030 | 1 | bad/empty-evidence (generic snippets, not G7-specific) |
- **Mechanism disrupted:** Added mandatory evidence-reliability screen to PREDICTION_PROMPT step 4 that (a) discards prediction-market-odds from polymarket.com/metaculus/etc. as circular self-referential evidence and (b) applies 40-60% materialization discount to forward-looking intent language; max_tokens raised 500->1500 so the full reasoning chain executes.
- **Pre-PR sanity (Step 6.5):** import OK, `autonomy packages lock --check` green, +57/-15 LOC pre-lint. **W-2 is the only scored gate** -- recorded by PR-CI on the PR after hand-off.
- **Status:** opened draft PR; PR-CI cached-replay pending on W-2.
- **Benchmark 2026-06-17:** SHA `1930289a8956d677edd2544ecafc34f1bdae9133`, seed=default (workflow-assigned), n=100, dev, baseline=superforcaster-polymarket-v1 (n=1028 W-2 rows), platform=polymarket — posted
- **Benchmark 2026-06-17:** SHA `870750ce3a7540aac6a08c53a2f07b069c635036`, seed=default (workflow-assigned), n=100, dev, baseline=superforcaster-polymarket-v1 (n=1028 W-2 rows), platform=polymarket — posted
- **Benchmark 2026-06-17:** SHA `11bf1b065e1587f2a08bb95704e910b57b1efca1`, seed=default (workflow-assigned), n=100, dev, baseline=superforcaster-polymarket-v1 (n=1028 W-2 rows), platform=polymarket — posted
- **Benchmark result 2026-06-17:** SHA `870750ce3a7540aac6a08c53a2f07b069c635036`, seed=42, n=100, dev -- Brier 0.2677 (baseline) vs 0.2891 (candidate, +8.0%), DA 59%->53% (-10.2%), overconf-wrong 16->14 (-12.5%). E1: delta (+0.0214) is below 2*SE (0.06-0.08) at n=100 -- verdict is noisy. Directional signal: targeted fingerprint (overconf-wrong) improved; DA dropped (consistent with over-broad screen). Action: grow to n=300.
- **Benchmark 2026-06-17:** SHA `a2687b381e43bf9ca5dcde02a150e0a25680441a`, seed=workflow-assigned, n=300, dev, baseline=superforcaster-polymarket-v1, platform=polymarket -- posted (grow-sample from noisy n=100; comment #4730163087)
