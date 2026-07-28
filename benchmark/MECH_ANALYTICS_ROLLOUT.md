# Mech-analytics rollout notes for the daily benchmark

## Flag flip: what changes

`USE_MECH_ANALYTICS_ROWS=true` routes the nightly benchmark's data source
from the marketplace subgraph + IPFS pull to mech-analytics's
`/v1/data/scored-rows` endpoint. Default off; the legacy path continues to
run until the flag is flipped in the workflow inputs or via the repo
variable.

## Steps skipped when the flag is on

- Backfill (`benchmark.datasets.backfill_responses`) — the flag-on path
  doesn't write to `logs/`, so nothing to repair.
- Fetch production data (`benchmark.datasets.fetch_production`) — same.
- Score current / previous rolling window — `analyze.py` builds these
  in-memory and writes them to disk for the triage step.
- Simulate trader ROI (`benchmark.roi_sim`) — runs against `logs/`, which
  is frozen; outputs are cleared each run so the Slack section degrades
  gracefully. Note: the standalone `benchmark_roi.yaml` workflow keeps
  running against the frozen `logs/` from the artifact while the flag is
  on. Its output is a job summary + artifact only, no Slack post, so
  low-stakes — but readers of that job should ignore the numbers while
  the flag is on.

## Rolling back to the legacy path

Flip `USE_MECH_ANALYTICS_ROWS=false` and rerun. Two known gotchas:

1. **`logs/` grew stale during the flag-on window.** The fetch step is
   skipped while the flag is on, so deliveries during that period never
   entered `logs/`. Any subsequent legacy `--rebuild` sees a hole for the
   flag-on window. **Mitigation:** on the first flag-off run after any
   flag-on period, bump `lookback_days` to cover the full flag-on window
   so the fetch backfills the hole.

2. **`scores.json` gets rebuilt from cache.** The mech-analytics rebuild
   path unlinks `scores.json` before each rebuild. The legacy incremental
   `update()` path reads it. First legacy-mode run after a flag-on period
   sees whatever the last flag-on write produced (~month-to-date).
   `update()` detects the mech-analytics source stamp and auto-rebuilds
   from `logs/` before merging incoming rows, so the request_id /
   platform:deliver_id dedup namespaces never overlap. The auto-rebuild
   only sees rows the fetch step has already backfilled into `logs/`, so
   still bump `lookback_days` to cover the flag-on window (see item 1).

## First-time setup (fresh benchmark repo)

The mech-analytics rebuild refuses to run when `scores_history.jsonl` is
missing (would silently start from empty and lose the "trend" section
forever). To bootstrap a truly-fresh setup, set
`MECH_ANALYTICS_ALLOW_EMPTY_HISTORY=true` for the first run only. The
next month's rollover will land the first real history row.
