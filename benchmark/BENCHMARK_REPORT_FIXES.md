# Benchmark Report — Audit Fixes

Findings from Pi's review of the 2026-04-15 benchmark report
(run `24440270163`, artifact `6445429063`) and follow-up proposal on
report cadence.

Each finding was verified by downloading the artifact and tracing the
responsible code path. Fixes in this branch cover issues 1-4. Issue 5
is a design change that needs Pi's buy-in before building.

---

## 1. `scores.json` inflated by dropped state files (P1)

### Symptom

- `scores.json` reports `total_rows = 197,334`, `valid_rows = 191,173`.
- `scored_row_ids.json` contains only `23,206` IDs.
- Those 23,206 IDs overlap 100% with this run's input log
  (`production_log_2026_04_15.jsonl`, covering 2026-04-08 to
  2026-04-15, i.e. exactly the 7-day fetch lookback).

### Root cause

Not a force rebuild. Workflow log shows `FORCE_REBUILD:` empty.
Two latent bugs in the workflow, each harmless on its own, compounded:

**Bug A: `.fetch_state.json` is a hidden file and `upload-artifact@v4`
drops it by default.** The workflow lists it under `path:` but v4
introduced `include-hidden-files: false` as the default. Files that
start with a dot are silently filtered out of the uploaded zip. Every
cron run starts with no fetch cursor, logs
`Loaded 0 existing row IDs for deduplication (state_loss=True)`, and
re-fetches the full 7-day lookback from the subgraph.

**Bug B: `scored_row_ids.json` was not in the upload list.** Commit
`7cd019ef` (2026-04-09 18:30 IST) split the dedup set out of
`scores.json` into its own file but did not update the workflow's
upload step. From Apr 9 through Apr 14, `scored_row_ids.json` was
created on disk every run but never persisted across runs. Commit
`5ba99629` (2026-04-14 19:16 IST) added it to the upload list. Apr 15
06:39 UTC was the first cron that actually uploaded it (3 files in
the artifact instead of 2).

Alone, either bug is benign:

- If `.fetch_state.json` survived but dedup did not, the fetch would
  use the cursor and not fetch duplicates, so no double-count.
- If dedup survived but `.fetch_state.json` did not, `update()` would
  drop the duplicates the refetch dragged in.

Combined, every run refetches the same ~21K rows and merges them onto
the accumulators with no memory of what it already saw. Five days of
that compounds into the observed 197,334.

### Why the Apr 15 artifact "looks fixed"

Apr 15 was the first run where `scored_row_ids.json` was uploaded, so
the dedup file going forward will persist. But:

- `scores.json` is still the same file that accumulated five days of
  double-counts before Bug B was fixed. `total_rows` is still polluted.
- `.fetch_state.json` is still a hidden file, so Bug A is still active.
  The dedup now saves us from new double-counting, but every run
  continues to refetch ~21K rows from the subgraph unnecessarily. If
  dedup ever fails again (schema change, partial artifact, etc.) the
  double-counting resumes immediately.

### Fix (applied in this branch)

1. `.github/workflows/benchmark_flywheel.yaml`: add
   `include-hidden-files: true` to the `benchmark-data` upload step.
   Unblocks `.fetch_state.json`.
2. Same workflow: add `benchmark/datasets/.fetch_state.json` to the
   `rm -f` list inside the "Clear cached scores (force rebuild)" step.
   Before fix #1, `.fetch_state.json` never persisted, so deleting it
   on force_rebuild was a no-op. After fix #1 it persists — and without
   this additional delete, `force_rebuild=true` would inherit a stale
   cursor and fetch ~0 new rows, producing an empty rebuild.

### Recovery procedure (manual, requires triggering CI)

Run these steps in order:

1. **Land this branch.** This deploys fixes #1 and #2 above.
2. **Wait for one normal cron run** to execute with the new workflow.
   That run will be the first one to successfully upload
   `.fetch_state.json` as part of the `benchmark-data` artifact.
   Confirm by downloading the artifact and checking the file is
   present. Skipping this step means the next force_rebuild would
   still run without a starting cursor in place.
3. **Trigger a `workflow_dispatch` run with `force_rebuild=true`.**
   The Clear step now wipes `scores.json`, `scores_history.jsonl`,
   `scored_row_ids.json`, **and** `.fetch_state.json`. Fetch will
   refetch the full 7-day lookback from the subgraph; `scorer.rebuild()`
   will reconstruct `scores.json` and a fresh dedup set from that fetch.

### What recovery can and cannot restore

**Can restore:** a clean `scores.json` / `scored_row_ids.json` covering
the last 7 days of production data, with counts matching the fetch.

**Cannot restore:** pre-recovery monthly snapshots. Raw production log
files are uploaded per run under `benchmark-log-${run_id}` artifacts
but are **never re-downloaded** by any workflow step. `scorer.rebuild()`
therefore only ever sees the log file produced by the current run's
fetch (i.e. at most the lookback window). All historical month
snapshots that lived in `scores_history.jsonl` before recovery are
permanently lost. This is an acceptable cost here because those
snapshots were themselves polluted by the double-counting bug — but
it is a one-way operation.

### How to verify the workflow fix

Next cron run's upload step log should say:

```
With the provided path, there will be 4 files uploaded
```

(5 if the month rollover has created `scores_history.jsonl`.) Download
the artifact and confirm `.fetch_state.json` is present. The cron run
after that should log `state_loss=False` and the "Loaded N existing
row IDs" line should show N close to the cumulative count, not zero.

### How to verify the recovery rebuild

After the `force_rebuild=true` run:

```bash
python3 -c "
import json
s = json.load(open('benchmark/results/scores.json'))
ids = json.load(open('benchmark/results/scored_row_ids.json'))
print('scores total :', s['total_rows'])
print('dedup count  :', len(ids))
print('match        :', s['total_rows'] == len(ids))
"
```

Expect the two numbers to match (modulo rows without `row_id`, which
should be zero since `fetch_production._make_row_id` guarantees one
per row). A drift of more than ~10 means something still persists
stale state.

**Files touched:**

- `.github/workflows/benchmark_flywheel.yaml`

---

## 2. "Low sample" label misapplied to tools with all-malformed output (P1)

### Symptom

Tools with a large `n` but zero valid parses render as `⚠ low sample`,
same label as tools with e.g. `n=5`. Example from a prior report:

```
14. resolve-market-jury-v1 — N/A (n=55) ⚠ low sample
```

The real issue here is 100% malformed output (a pipeline failure),
not insufficient volume.

### Root cause

`_sample_label()` in `benchmark/analyze.py` (before fix, line ~125)
branched only on `decision_worthy`, which `scorer.py:1058` defines as
`valid_n >= MIN_SAMPLE_SIZE`. Any tool with `valid_n == 0` flunks that
check regardless of total `n`.

### Fix (applied in this branch)

Split the label into two cases:

```python
def _sample_label(stats):
    n = stats.get("n", 0)
    valid_n = stats.get("valid_n", 0)
    if n >= MIN_SAMPLE_SIZE and valid_n == 0:
        return " ⚠ all malformed"
    if valid_n < MIN_SAMPLE_SIZE:
        return " ⚠ low sample"
    return ""
```

### How to verify

Unit tests added in `benchmark/tests/test_analyze.py::TestSampleLabel`:

- `n=5, valid_n=5` renders ` ⚠ low sample`.
- `n=55, valid_n=0` renders ` ⚠ all malformed`.
- `n=100, valid_n=80` renders empty.
- `n=3, valid_n=0` still renders ` ⚠ low sample` (not enough volume
  to confidently call "all malformed").

**Files touched:**

- `benchmark/analyze.py`
- `benchmark/tests/test_analyze.py`

---

## 3. Inconsistent sample-size messaging (P2)

### Symptoms

**3a.** "Since Last Report" section lists tools with `n=1`, `n=8` with
no low-sample flag, even though other sections flag similar cases.

**3b.** The final sample-size-warnings line says "All categories have
sufficient sample size" while the directional-bias subsection above
lists several categories as "insufficient data". Reads as a
contradiction.

### Root cause

**3a.** `section_period()` in `benchmark/analyze.py` rendered per-tool
bullets without calling `_sample_label`. Tool ranking and
tool-version-breakdown sections both flag low samples; this one
did not.

**3b.** The two messages use different thresholds and different
denominators, both correct:

- `section_sample_size_warnings()` uses `SAMPLE_SIZE_WARNING = 20` on
  **total category `n`** (the reporting gate for including a category
  at all).
- The directional-bias subsection uses `MIN_SAMPLE_SIZE = 30` on
  **`n_losses` within that category** (a narrower denominator: rows
  where the tool and market disagreed AND the market was closer to
  truth).

Both checks answer different questions but the old wording treated
the category-level gate as if it spoke for the whole report.

### Fix (applied in this branch)

**3a.** Add `_sample_label(stats)` to the per-tool bullet format
string in `section_period`. Same marker logic as everywhere else.

**3b.** Reword the default line to be explicit about which gate it
checks and that stricter subsection gates exist:

```
All categories have at least 20 total questions (the category
reporting gate). Subsections that use stricter gates on narrower
denominators (e.g. n_losses in directional bias) may still flag
specific categories as insufficient.
```

### How to verify

Unit tests:

- `TestSectionPeriod::test_tiny_tool_flagged_as_low_sample` — period
  with a `n=1` tool renders `⚠ low sample`.
- `TestSectionPeriod::test_sufficient_tool_not_flagged` — period with a
  `n=95, valid_n=95` tool renders no marker.
- `TestSectionPeriod::test_mixed_population_flags_only_small_ones` —
  only the small tool's line carries the marker.
- `TestSectionPeriod::test_all_malformed_tool_gets_distinct_label` —
  `n=55, valid_n=0` renders `⚠ all malformed`.
- `TestSectionSampleSizeWarnings::test_large_category_not_warned` —
  updated to assert the new wording mentions the category reporting
  gate and directional bias explicitly.
- `TestSectionSampleSizeWarnings::test_threshold_value_embedded_in_copy`
  — guards against the threshold drifting out of the user-facing copy.

**Files touched:**

- `benchmark/analyze.py`
- `benchmark/tests/test_analyze.py`

---

## 4. No-signal rate rendered as 0% (P3)

### Symptom

Report shows `No-signal rate: 0%` while the count next to it shows
positive entries (e.g. 142). Underlying rate is 0.00072 ≈ 0.07%.

### Root cause

`benchmark/analyze.py` (before fix, line ~85):

```python
no_sig_str = f"{no_sig:.0%}" if no_sig is not None else "N/A"
```

`:.0%` rounds to whole percentage points. No-signal rate typically
lives in the 0.01%–1% range, so coarse formatting prints "0%".

### Fix (applied in this branch)

Switch to `:.2%`. Two decimals cover the expected range while leaving
the count visible next to it for precise values.

### How to verify

Unit test
`TestSectionOverall::test_no_signal_rate_small_value_renders_two_decimals`:
feed `no_signal_rate=0.00072, no_signal_count=142`, assert the rendered
section contains the literal `0.07%` and does not contain
`No-signal rate: 0%`.

**Files touched:**

- `benchmark/analyze.py`
- `benchmark/tests/test_analyze.py`

---

## 5. Report cadence (not fixed in this branch)

Pi proposed: daily report shows only tools with "significant new data";
full report weekly on Mondays.

Direction is right (daily deltas on tools with `n=1`/`n=8` are pure
noise), but "tools with new data" alone isn't the right filter:
superforcaster's 95 new rows barely move a 22K-row aggregate, and a
tool that drops to zero deliveries would disappear silently.

Counter-proposal for discussion with Pi:

1. Drop "Since Last Report" from the daily Slack digest. Use the
   existing `rolling_scores` (last 7 days) as the baseline.
2. In the digest, only surface tools that pass both
   `n_new >= 30 in the rolling window` AND
   `|brier_rolling - brier_alltime| > material_threshold`. Everything
   else collapses to one line ("5 tools unchanged within noise"), so
   nothing disappears silently.
3. On Mondays, add a "vs last Monday" block on top of the same rolling
   view. No second pipeline.
4. Keep `report.md` unfiltered daily. The digest is the only surface
   that filters.

Wait on Pi's buy-in before building.

---

## Priority order

1. Issue #1 — workflow fix is in this branch. The manual
   `force_rebuild=true` run needs to happen after merge so the
   existing polluted `scores.json` is reset.
2. Issues #2-4 — in this branch with tests.
3. Issue #5 — pending design discussion.
