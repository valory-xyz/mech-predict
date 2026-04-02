# Daily Plan — 3 April 2026

Tracking progress on the [Prediction Tool Benchmark & Continuous Improvement System](https://github.com/valory-xyz/mech-predict/blob/main/benchmark/PROPOSAL.md).

---

## Yesterday's progress (2 April)

### Tournament Mode (Jenslee)
- Implemented forward-looking tournament predictions ([#178](https://github.com/valory-xyz/mech-predict/pull/178))
  - `fetch_open.py` — fetches open markets from Omen subgraph and Polymarket Gamma API
  - `tournament.py` — runs prediction tools on live markets, captures source_content snapshots for future cached replay
  - `score_tournament.py` — resolves stored predictions, computes Brier scores, deduplicates per-market queries
  - Added tournament jobs to the benchmark CI workflow
  - **Impact:** enables continuous forward-looking evaluation of prediction tools against live markets

### Cached Replay Pipeline (Divya)
- Cached replay pipeline ([#176](https://github.com/valory-xyz/mech-predict/pull/176)) — addressed review feedback (8 items), added lint fixes for CI, added prompt improvement plan. PR is approved; merge pending source_content flowing in production (Monday)

### Benchmark Analysis & Prompt Calibration (Divya)
- Superforecaster prompt calibration fix ([#179](https://github.com/valory-xyz/mech-predict/pull/179))
  - **Impact:** 43% Brier reduction on superforecaster (0.52 → 0.29, 20-delivery test set)
- Benchmark reports covering Mar 31 baseline and Apr 2 runs ([#180](https://github.com/valory-xyz/mech-predict/pull/180)) — structured tracking of findings and actions per CI run

---

## What we're doing today (3 April)

| Owner | Task |
|-------|------|
| Jenslee | Review the existing accuracy report and decide which tools to remove based on persistently low performance |
| Divya | Run the cached replay pipeline using already-collected IPFS final prompt data |
| Divya | Optimise prompts for the underperforming tools identified in the accuracy report |
| Jenslee | Automate sharing of benchmark reports to a Slack channel for visibility |

---

## Where we need input

| Item | From whom | Status |
|------|-----------|--------|
| **`source_content_mode` production enablement on Monday** | Production team | Green light received for Monday. |
| **Benchmark workflow migration to infra cron** | Production / Infra | Production confirmed support for early next week. |
