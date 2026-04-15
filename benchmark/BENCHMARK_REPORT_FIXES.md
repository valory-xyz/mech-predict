# Benchmark Report — Audit Fixes

Findings from Pi's review of the 2026-04-15 benchmark report
(run `24440270163`, artifact `6445429063`) and follow-up proposal on
report cadence.

Verified by downloading the artifact and tracing each claim to the
responsible code path.

---

## 1. Artifact mismatch — inflated scores from lost dedup state (P1)

**Symptom**

- `scores.json` reports `total_rows = 197,334`, `valid_rows = 191,173`
- `scored_row_ids.json` contains only `23,206` IDs
- The 23,206 IDs overlap 100% with this run's input log
  (`production_log_2026_04_15.jsonl`, covering 2026-04-08 to 2026-04-15)

**Root cause**

Not a force rebuild. Workflow log shows `FORCE_REBUILD:` empty. The
actual cause: the **previous** `benchmark-data` artifact contained only
`scores.json` and `report.md`. The "Extracting" step at 07:25:55 shows:

```
inflating: benchmark/results/scores.json
inflating: benchmark/results/report.md
##[endgroup]
```

Missing from the downloaded artifact:

- `benchmark/results/scored_row_ids.json`
- `benchmark/results/scores_history.jsonl`
- `benchmark/datasets/.fetch_state.json`

Consequences:

1. `fetch_production` logged
   `Loaded 0 existing row IDs for deduplication (state_loss=True)` and
   re-fetched the full 7-day lookback.
2. `scorer.update()` called `_load_dedup_ids()` on a missing file →
   returned an empty set → every one of the 23,206 rows was merged onto
   the existing ~174K accumulators in `scores.json` with **zero
   deduplication**.
3. Previous runs had the same problem, so `scores.json` has been
   compounding double-counts from overlapping 7-day windows.
4. `scores_history.jsonl` (monthly snapshots) is lost each time the
   artifact is incomplete.

`actions/upload-artifact@v4` silently skips missing files in its `path:`
list, so once any of these files fails to exist at upload time, every
subsequent run downloads a partial artifact, omits the same files on
upload, and the corruption self-perpetuates.

**Fix**

1. In `.github/workflows/benchmark_flywheel.yaml`, make the upload step
   fail (or loudly warn) when any of the tracked state files is missing.
   Options:
   - `actions/upload-artifact@v4` has `if-no-files-found: error` — set
     it. But that applies to the whole path glob, not per-file. Need to
     check explicit file existence with a pre-upload shell step.
   - Add a step before upload that `test -f` each required file and
     fails fast with a clear message when one is missing.
2. Investigate the prior run(s) where the three files stopped being
   uploaded. Find the root cause there (disk cleanup, a crashed step,
   or a path change) and fix at source.
3. One-shot recovery: trigger a `force_rebuild=true` run so
   `scores.json`, `scores_history.jsonl`, and `scored_row_ids.json` are
   regenerated from raw logs. Until this runs, all reported numbers
   since the corruption began are unreliable.
4. Add a self-check inside `scorer.update()`: if
   `len(scored_row_ids) < scores["total_rows"]` by more than
   `rows_without_row_id`, log a warning. Today the mismatch is silent.

**Files**

- `.github/workflows/benchmark_flywheel.yaml`
- `benchmark/scorer.py` (`update()` self-check, ~line 1555)
- `benchmark/datasets/fetch_production.py` (optional: cross-check
  `scored_row_ids` size vs scores total)

---

## 2. Misclassified "low sample" warning (P1)

**Symptom**

Tools with a large `n` but zero valid parses render as
`⚠ low sample`, same label as tools with e.g. `n=5`. Example from a
prior run: `resolve-market-jury-v1 (n=55) ⚠ low sample` — the real
issue is 100% malformed output.

**Root cause**

`_sample_label()` in `benchmark/analyze.py:125-129` keys off
`decision_worthy`, which `scorer.py:1058` defines as
`valid_n >= MIN_SAMPLE_SIZE`. Any tool with `valid_n == 0` (all
malformed) fails this check regardless of total `n`.

**Fix**

Split the label into two cases:

```python
def _sample_label(stats: dict[str, Any]) -> str:
    n = stats.get("n", 0)
    valid_n = stats.get("valid_n", 0)
    if n >= MIN_SAMPLE_SIZE and valid_n == 0:
        return " ⚠ all malformed"
    if valid_n < MIN_SAMPLE_SIZE:
        return " ⚠ low sample"
    return ""
```

Also consider a third case (`valid_n > 0` but `< MIN_SAMPLE_SIZE` while
`n >= MIN_SAMPLE_SIZE`): that's "mostly malformed", different from both.

**Files**

- `benchmark/analyze.py` (`_sample_label`, ~line 125)

---

## 3. Inconsistent sample-size messaging (P2)

**Symptom**

- "Since Last Report" section lists tools with `n=1`, `n=8` with no
  low-sample flag.
- The final sample-size-warnings line says "All categories have
  sufficient sample size" while the directional-bias subsection above it
  lists several categories as "insufficient data".

**Root cause**

Two separate issues:

a. `section_period()` in `benchmark/analyze.py:916-932` renders per-tool
   bullets as `**{tool}**: {brier} ... (n={stats['n']})` with no
   sample-size gate. Tool ranking and version-breakdown sections both
   flag low samples; this one doesn't.

b. The two messages use different thresholds and denominators but read
   as contradictory. `section_sample_size_warnings()` uses
   `SAMPLE_SIZE_WARNING = 20` on **total category `n`**. The directional
   bias subsection uses `MIN_SAMPLE_SIZE = 30` on **`n_losses` within
   that category**. Both are technically correct but the wording doesn't
   make the distinction obvious.

**Fix**

a. Add a low-sample flag to `section_period()` per-tool bullets. Use
   the same `_sample_label` logic as the ranking section (after it's
   fixed per issue #2). Alternatively, skip rendering tools below a
   threshold entirely — preferable if paired with the rolling-window
   digest proposal (section 5).

b. Reword the final sample-size line to be explicit about what it
   checks. For example:

   ```
   All categories have at least 20 scored predictions (sample size
   gate for reporting). Categories may still appear as "insufficient
   data" in subsections that use stricter thresholds on a narrower
   denominator (e.g. n_losses in directional bias).
   ```

   Or: consolidate both checks into a single table showing each
   category's `n`, `n_valid`, `n_losses`, with a row-level flag when any
   cell fails its gate.

**Files**

- `benchmark/analyze.py` (`section_period`, ~line 881;
  `section_sample_size_warnings`, ~line 365;
  `_render_directional_bias`, ~line 615)

---

## 4. No-signal rate display rounds to 0% (P3)

**Symptom**

Report shows `No-signal rate: 0%` even when the underlying rate is
0.0007 (about 0.07%) and `no_signal_count > 0`.

**Root cause**

`benchmark/analyze.py:85`:

```python
no_sig_str = f"{no_sig:.0%}" if no_sig is not None else "N/A"
```

`:.0%` rounds to whole percentage points.

**Fix**

Switch to `:.2%` (shows two decimals of percentage — 0.07%) or
`:.1%` (one decimal — 0.1%). `:.2%` preserves granularity while keeping
the label consistent with other percentage fields.

Matching fields elsewhere in the file use `:.0%` deliberately for
coarse rates like `reliability` and `directional_accuracy`. No-signal
rate is usually near zero, so it needs finer resolution.

**Files**

- `benchmark/analyze.py` (~line 85)

---

## 5. Report cadence — Pi's proposal (P2, design change)

**Pi's suggestion**

Daily report shows only tools with "significant new data" since last
report. Full per-tool × per-platform report on Mondays.

**Why it's in the right direction**

Today's "Since Last Report" section is mostly noise. Only superforcaster
had meaningful new volume (95 rows); the others had 1, 8, 1. Deltas on
`n=1` or `n=8` carry no statistical signal.

**Where "new volume" alone isn't enough**

- Superforcaster's 95 new rows is meaningful by volume but barely moves
  its 22K-row all-time aggregate. Volume filter still lets in noise.
- A tool that silently drops to 0 deliveries would disappear from the
  daily view. That's an incident signal we'd want to catch.
- Weekly rollup doesn't fix the `n=1/n=8` problem either — over 7 days
  those tools might have 7/56 rows, still below the statistical gate.

**Counter-proposal**

1. Drop "Since Last Report" from the daily Slack digest. Replace it
   with the 7-day rolling window already computed as `rolling_scores`.
   Daily deltas are too noisy; 7-day gives a stable baseline.
2. In the Slack digest, only surface tools that pass **both** gates:
   - `n_new >= 30` in the rolling window
   - `|brier_rolling - brier_alltime| > material_threshold`
     (tune empirically, e.g. 0.02)

   Everything else collapses into one line: "5 tools unchanged within
   noise". Nothing disappears silently — tools with zero new data get
   flagged as "no new deliveries" explicitly.
3. On Mondays, add a "vs last Monday" block on top of the same rolling
   view. No second pipeline — just an extra section that only renders
   when `today.weekday() == Monday`.
4. Keep `report.md` generated daily with all tools, no filtering.
   Anyone wanting the full breakdown opens the artifact. The Slack
   digest is the only surface that filters.

**Files**

- `benchmark/notify_slack.py` (filtering logic for the digest)
- `benchmark/analyze.py` (`section_period` — either drop or keep for
  the full `report.md` but not the Slack digest)
- `.github/workflows/benchmark_flywheel.yaml` (no change to cadence,
  the Monday block is driven in-code)

---

## Priority order

1. Issue #1 — actively corrupting `scores.json`. Fix the workflow
   artifact upload first, then trigger a force rebuild.
2. Issue #2 — wrong label on visible output, one-line fix.
3. Issue #3a — missing low-sample flag in `section_period`, one-line
   fix. #3b (wording) can wait.
4. Issue #4 — formatter fix, one-line.
5. Issue #5 — design change, needs Pi's buy-in on the counter-proposal
   before building.
