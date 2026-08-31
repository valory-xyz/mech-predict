# Mech-analytics rollout notes for the daily benchmark

## Flag flip: what changes

`USE_MECH_ANALYTICS_ROWS=true` routes the nightly benchmark's data source
from the marketplace subgraph + IPFS pull to mech-analytics's
`/v1/data/scored-rows` endpoint. Default off; the legacy path continues to
run until the flag is flipped in the workflow inputs or via the repo
variable.

## Timestamp semantics

The endpoint filters `since` / `until` on `requested_at`. Rows carry
`predicted_at` derived locally from `delivered_at` (falls back to
`requested_at`), which matches the legacy log-row semantic.

The rebuild fetches `[start_of_month - MECH_ANALYTICS_RESOLUTION_LAG_DAYS,
until]` and re-derives every prior month in the window into
`scores_history.jsonl` via a per-month upsert. Default lag is 90 days;
override with `MECH_ANALYTICS_RESOLUTION_LAG_DAYS`. Without the lag the
month-to-date window plus `resolved=True` filter would drop every
prediction requested in month M that only resolved in M+1 (excluded on
M's last run because still unresolved, out of window from M+1's first run).

## Steps skipped when the flag is on

- Backfill (`benchmark.datasets.backfill_responses`) — the flag-on path
  doesn't write to `logs/`, so nothing to repair.
- Fetch production data (`benchmark.datasets.fetch_production`) — same.
- Score current / previous rolling window — `analyze.py` rebuilds these
  in-memory from mech-analytics rows and writes them to
  `rolling_scores_<platform>.json` / `prev_rolling_scores_<platform>.json`
  itself via `_write_rolling_json` (analyze.py:3144-3157). Running the
  Score steps under the flag would produce combined-file outputs that
  analyze then unconditionally overwrites and no reader consumes.

## Steps that route via mech-analytics when the flag is on

- Score trailing 90d window — no other producer writes
  `trailing_scores_<platform>.json`, so this step runs under both modes.
  Under the flag it fetches `[now-90d, now]` from `/v1/data/scored-rows`
  via `score_period_split_by_platform_from_mech_analytics`; under
  legacy it scans `logs/`. Same output shape either side.
- Simulate trader ROI (`benchmark.roi_sim`) — under the flag,
  production rows come from mech-analytics via
  `load_input_rows_from_mech_analytics`; tournament rows still come
  from the local `tournament_scored.jsonl` (produced by the
  tournament-run job, which is not flag-gated). The two streams merge
  in `main()` so no rows are dropped from the ROI report.
- The standalone `benchmark_roi.yaml` workflow does not check the flag
  and always runs against the frozen `logs/` from the artifact. Its
  output is a job summary + artifact only, no Slack post, so
  low-stakes — but readers of that job should ignore the numbers
  while the flag is on.

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
