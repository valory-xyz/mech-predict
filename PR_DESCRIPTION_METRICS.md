## Implement full metric taxonomy from PR #202 review

Implements all scoring metric changes agreed during Pi's review on [PR #202](https://github.com/valory-xyz/mech-predict/pull/202), plus the proposal metric taxonomy update from PROPOSAL_EVOLUTION.md discussions. Merges `docs/proposal-metrics-reorder` (Brier-primary reorder) and builds all new metrics on top of the edge-over-market branch.

### Review comments addressed

All points from Pi's review on PR #202. Full plan in `benchmark/METRICS_CHANGES_PLAN.md`.

| # | Review comment | Status | What was done |
|---|---------------|--------|---------------|
| **1** | Fix accuracy at `p_yes == 0.5` — asymmetric handling counts 0.5 as correct for No but incorrect for Yes | **Done** | Renamed `accuracy` → `directional_accuracy`, excluded `p_yes == 0.5` entirely, added `n_directional`. Fixed in scorer.py (batch + incremental), ci_replay.py, compare.py, analyze.py, notify_slack.py, tests. |
| **2** | Add no-signal rate — count how often tools output 0.5 | **Done** | `no_signal_rate = count(p_yes == 0.5) / n_valid` and `no_signal_count` in both paths. |
| **3** | *(Already fixed in PR #202)* | N/A | — |
| **4** | Derive thresholds from data instead of hardcoding | **Deferred** | Hardcoded with TODO comments. Needs data inspection first. |
| **5** | *(Already fixed in PR #202)* | N/A | — |
| **6** | Add ECE scalar — single number for overall calibration quality | **Done** | `compute_ece()` from calibration bins. Added to `score()` output, `_finalize_scores()`, and analyze.py calibration section. |
| **7** | Add calibration intercept and slope — diagnose *how* calibration is off | **Done** | `compute_calibration_regression()` via weighted linear regression on bins. Intercept = bias direction, slope = dispersion. Returns None if < 3 populated bins. |
| **8** | Normalize overconfident-wrong — raw count is meaningless across different dataset sizes | **Done** | Added `overconf_wrong_rate = overconf_wrong / n` in ci_replay.py. |
| **9** | Add log loss — punishes confidently wrong predictions harder than Brier | **Done** | `log_loss_score()` with epsilon clamping. Both batch and incremental paths. Added to compare.py, analyze.py rankings, notify_slack.py prompt. |
| **10** | Update metric taxonomy in PROPOSAL.md and README — reflect agreed categories | **Done** | PROPOSAL.md rewritten with Core ranking / Eligibility / Core diagnostic / Secondary diagnostic taxonomy. README.md updated with all new formulas. |
| **11** | Statistical protections | **Deferred** | Separate effort, on the roadmap. |

Additionally implemented from CEO feedback:

| Item | Status | What was done |
|------|--------|---------------|
| Period-aware reporting | **Done** | "Since Last Report" and "Last 7 Days Rolling" sections with deltas vs all-time. Flywheel workflow updated. Slack prompt leads with diffs. |

Deferred from PROPOSAL_EVOLUTION.md discussions (not in this PR):

| Item | Status |
|------|--------|
| Conditional accuracy when disagreeing | Deferred |
| Disagreement-stratified Brier | Deferred |
| Directional bias | Deferred |

### What changed

#### New metrics (both batch and incremental paths)

| Metric | Type | Description |
|--------|------|-------------|
| **Directional accuracy** | Core diagnostic | Replaces `accuracy`. Excludes `p_yes == 0.5` (no signal) — fixes the asymmetry where 0.5 was counted correct for No outcomes but incorrect for Yes. |
| **No-signal rate** | Core diagnostic | `count(p_yes == 0.5) / n_valid` — how often the tool says "I don't know." |
| **Log loss** | Core ranking | `-mean(outcome * log(p_yes) + (1-outcome) * log(1-p_yes))`. Punishes confidently wrong predictions exponentially harder than Brier. Clamped to `[1e-15, 1-1e-15]` to avoid `log(0)`. |
| **ECE** | Core diagnostic | Expected Calibration Error — weighted average of per-bin absolute gaps. Single scalar for overall calibration quality. |
| **Calibration intercept** | Core diagnostic | Weighted linear regression of realized vs predicted on calibration bins. Positive = systematic underestimate. |
| **Calibration slope** | Core diagnostic | `< 1.0` = overconfident, `> 1.0` = underconfident, `1.0` = perfect. Returns None if < 3 populated bins. |
| **Overconfident-wrong rate** | ci_replay | Normalizes existing count by `n` for cross-dataset comparison. |

#### Period-aware reporting

- New `score_period(logs_dir, days)` function scores the most recent N daily log files
- New `--period-days` CLI flag for scorer
- Report now leads with **"Since Last Report"** and **"Last 7 Days Rolling"** sections showing deltas vs all-time
- Flywheel workflow updated with period scoring steps before analyze
- Slack prompt updated to lead with diffs, not all-time numbers

#### Metric taxonomy update (docs)

PROPOSAL.md and README.md reorganized under the agreed taxonomy:

| Category | Metrics |
|----------|---------|
| **Core ranking** | Brier, Log Loss, BSS |
| **Eligibility/gating** | Reliability (< 80% = excluded) |
| **Core diagnostic** | ECE, Calibration intercept/slope, Sharpness, Directional Accuracy, No-signal rate |
| **Secondary diagnostic** | Edge, Overconfident-wrong rate |

### Files changed

| File | What changed |
|------|-------------|
| `benchmark/scorer.py` | All new metrics in both batch (`compute_group_stats`) and incremental (`_accumulate_group` / `_derive_group`) paths. `_compute_edge_diagnostics()` extracted. `score_period()` added. `_ACCUM_KEYS` extended. `_restore_group` handles backward compat. |
| `benchmark/analyze.py` | `section_period()` for delta reporting. `section_diagnostic_metrics()` for edge diagnostics. ECE/calibration regression in calibration section. All `accuracy` → `directional_accuracy`. Log loss in rankings. `generate_report()` accepts period/rolling scores. CLI `--period`/`--rolling` flags. |
| `benchmark/compare.py` | `accuracy` → `directional_accuracy`, `log_loss` added to metric definitions and comparison table. |
| `benchmark/ci_replay.py` | Fixed 0.5 accuracy bug (same as scorer). Added `overconf_wrong_rate`, `n_directional`. |
| `benchmark/notify_slack.py` | Log loss in tool/platform bullets. Prompt leads with diffs. ECE/no-signal rate guidance. |
| `benchmark/tests/test_scorer.py` | New test classes: `TestDirectionalAccuracy`, `TestLogLoss`, `TestECE`, `TestCalibrationRegression` — each with batch/incremental parity assertions. |
| `benchmark/PROPOSAL.md` | Metric taxonomy rewritten per agreed categories. Log loss promoted to core ranking. |
| `benchmark/README.md` | All new metric formulas. Directional accuracy replaces accuracy. Diagnostic edge metrics section. |
| `.github/workflows/benchmark_flywheel.yaml` | Period scoring steps (1d and 7d) before analyze. `--period`/`--rolling` flags passed to analyze. |

### How to verify

```bash
# All tests pass
pytest benchmark/tests/test_scorer.py -q

# Linters clean
black --check benchmark/
isort --check-only benchmark/
flake8 benchmark/scorer.py benchmark/analyze.py benchmark/compare.py benchmark/ci_replay.py
mypy benchmark/scorer.py benchmark/analyze.py benchmark/compare.py benchmark/ci_replay.py benchmark/tests/test_scorer.py --disallow-untyped-defs --config-file tox.ini
pylint --disable=C0103,R0801,R0912,C0301,C0201,C0204,C0209,W1203,C0302,R1735,R1729,W0511,W0603,W0703,R0913,W0613,R0914,R1702,W0102 benchmark/scorer.py benchmark/analyze.py benchmark/compare.py benchmark/ci_replay.py

# Batch vs incremental parity (covered by tests, but can also verify manually)
python -m benchmark.scorer --input benchmark/datasets/production_log.jsonl --output /tmp/batch.json
# Compare batch output keys with incremental update() output
```

### Deferred

- **Point 4**: Derive thresholds from data (hardcoded for now, marked with TODO)
- **Point 11**: Statistical protections (separate effort)
- **AUC-ROC / Discrimination**: Not yet implemented
- **Simulated PnL**: Blocked on Kelly vs fixed-fraction decision
