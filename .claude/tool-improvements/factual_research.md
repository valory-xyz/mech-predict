# factual_research — investigation log

Per-tool investigation memory for the `tool-improvement` pipeline. Append a new
section per issue/PR; never overwrite prior content (oldest-first).

## Issue #314 — 2026-06-05

- **Trigger:** regression on `polymarket` W-1 (`2026-05-26T11:45:34Z` → `2026-06-02T11:45:34Z`); headline Brier `0.3979` (n=180) vs W-2 `0.3131` (n=227) (Δ `+0.0848`). Reproduced exactly from raw rows (`repro.py`: `n=180, brier=0.3979`).
- **PR:** opened from branch `tool-improvement/factual_research-asof-date-anchoring` (new version `factual_research-v1`).
- **Confirmed hypothesis:** on earnings/revenue-threshold markets the estimate stage anchors `p_yes` on a STALE prior-period figure carried in the briefing (Dell ISG `$10.3B` / HPE Cloud&AI `$6.3B`, both Q1 a fiscal year earlier) and collapses to a confident NO on "above $X" markets, ignoring the forward/current evidence (raised guidance, YoY growth, segment expansion, backlog) present in the SAME briefing.
- **Stage + code site:** estimate stage at `ESTIMATE_SYSTEM` / `ESTIMATE_USER` (the `PredictionResult` LLM call, where `p_yes` is produced), reinforced at `SYNTHESIS_USER` — gate-visible per Step 3a map (runs on injected `source_content`; live retrieval is bypassed by the cached replay).
- **Localized cells (Step 2):** `by_tool_category=business (n=99, brier=0.4942, Δ +0.16)`. The `by_tool_version_mode=dcxmqe… (n=80, brier=0.3926)` cell was dropped by the version-currency check (non-registered CID). Step 2.5: version `eczrqs…` is present in BOTH windows and its own Brier rose `0.3131 → 0.4021` (ΔBrier +0.089 ≫ 0.03) on matched strata — a genuine within-version rise, not a pure composition artifact; W-1 Brier 0.3979 ≥ 0.25 (chronic-bad) with n=180 ≥ 105.
- **Evidence sample (worst-miss rows; deliveries read from IPFS, 10/10 readable):**
  | question (truncated 80 char) | p_yes | outcome | evidence_finding |
  |---|---|---|---|
  | Will Dell Q1 Infrastructure Solutions Group revenue be above $23.5B? | 0.00 | YES | good-evidence/bad-reasoning |
  | Will Dell Q1 Infrastructure Solutions Group revenue be above $22.5B? | 0.01 | YES | good-evidence/bad-reasoning |
  | Will Hewlett Packard Enterprise Q2 Cloud and AI revenue be above $7.0B? | 0.01 | YES | good-evidence/bad-reasoning |
  | Will Hewlett Packard Enterprise Q2 Cloud and AI revenue be above $6.5B? | 0.01 | YES | good-evidence/bad-reasoning |
  | Will Donald Trump sign an executive order on May 29, 2026? | 0.025 | YES | good-evidence/bad-reasoning (base-rate neglect; secondary) |
- **Mechanism disrupted:** as-of-date freshness guard — `ESTIMATE_USER` STEP 0 "resolution-period freshness check" + `ESTIMATE_SYSTEM` figure-freshness rule + `SYNTHESIS_USER` figure-period preservation: date-stamp every figure, label prior-period figures STALE, and project a not-yet-reported resolving metric forward from the latest baseline using the briefing's growth signals instead of anchoring on the stale figure. Structural reasoning-step change (NOT a clamp/temperature) — passes Step 5 condition (v).
- **Pre-PR sanity (Step 6.5):** import OK; `autonomy packages lock --check` green; `+22/-1` LOC pre-lint added at the named estimate/synthesis stages; new-version CID `bafybeicg52wve2ytam3eisks7gumvrswm2v6bkuafburdlmh6d553nrvee`. **W-2 is the only scored gate** — recorded by PR-CI on the PR after hand-off, not by the agent.
- **Follow-ups (named, not fixed — distinct mechanisms, hard-constraint 2):** (1) politics deadline/event cluster (over-predicts deadline events); (2) Trump-utterance / high-frequency base-rate-neglect markets — extreme-tail collapse on inherently noisy events.
- **Status:** opened draft PR; PR-CI cached-replay pending on W-2 (`2026-05-19T11:45:34Z` → `2026-05-26T11:45:34Z`).
