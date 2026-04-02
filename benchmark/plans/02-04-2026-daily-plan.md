# Daily Plan — 2 April 2026

Tracking progress on the [Prediction Tool Benchmark & Continuous Improvement System](https://github.com/valory-xyz/mech-predict/blob/main/benchmark/PROPOSAL.md).

---

## Yesterday's progress (1 April)

### Benchmarking Pipeline (Divya)
- Created the cached replay pipeline ([#176](https://github.com/valory-xyz/mech-predict/pull/176))
  - Replay runner — replays questions through prediction tools with cached evidence
  - Comparison tool — generates Brier/accuracy/sharpness delta tables between two runs
  - Sweep orchestrator — runs the full pipeline in one command
  - End-to-end sweep: reads production log, fetches source content from IPFS, builds replay dataset, runs baseline vs candidate, outputs comparison
  - Developer documentation covering pipeline overview and workflows

### Tool Improvement (Jenslee)
- Implemented `source_content_mode` across all tools ([#174](https://github.com/valory-xyz/mech-predict/pull/174), [#175](https://github.com/valory-xyz/mech-predict/pull/175))
  - Cleaned source content can now be stored per-tool
  - Ran [storage cost analysis](https://github.com/LOCKhart07/random-valory-scripts/blob/main/mech/source_content_cleaned_only_report.md) across tools
  - **Impact:** 9.6 GB/day (raw) → 96 MB/day (cleaned), 99.4% reduction
- Deployed to one mech with flag set to `false`

### CI Optimization (Jenslee)
- Split unit and integration tests ([#172](https://github.com/valory-xyz/mech-predict/pull/172))
  - Integration tests gate on unit test success — no wasted API calls on broken builds
  - Relaxed test matrix from 15 → 9 cells
  - Added multi-API-key support
- **Impact:** Google Search API calls per CI run: 1,380 → 92 (93% reduction)

---

## What we're doing today (2 April)

| Owner | Task |
|-------|------|
| Jenslee | Enable `source_content_mode` in production — **pending production team green light** |
| Jenslee | Implement tournament mode |
| Divya | Sync with Production on migrating daily benchmark workflow from GitHub Actions → infra cron jobs |
| Divya | Once source content is flowing, run first real sweep comparing models on cached production data to validate pipeline end-to-end |

---

## Where we need input

| Item | From whom | Why it matters |
|------|-----------|----------------|
| **Go-ahead to enable `source_content_mode`** in production | Production team | First real sweep and tournament mode depend on this. Flag is deployed but set to `false`. |
| **Benchmark workflow migration** to infra cron | Production / Infra | Required for canary deployment and A/B validation path described in the proposal. Divya syncing today. |
