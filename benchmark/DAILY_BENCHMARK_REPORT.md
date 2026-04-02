# Daily Benchmark Report

Prediction tool performance tracking from automated CI benchmark runs. Each entry summarises findings from the benchmark pipeline artifacts and actions taken.

---

## Report Format

Each entry follows this structure:

- **Date:** When the benchmark ran
- **Artifact:** Link to GitHub Actions artifact
- **Summary:** Key metrics and findings
- **Actions:** What was done or planned in response

---

## 2026-04-02

**Artifact:** [benchmark-results](https://github.com/valory-xyz/mech-predict/actions/runs/23887276687)
**Data period:** 2026-03-31 to 2026-04-02 (incremental from where Mar 31 run left off, +13,486 new rows)

**Summary:**
- 36,241 predictions scored, 98% reliability
- Overall Brier: 0.2555 (0.25 = random guessing, lower is better)
- Top tool: prediction-request-reasoning-claude (Brier: 0.20, 1,648 predictions)
- Worst tool by volume impact: superforcaster (Brier: 0.26, 16,163 predictions) — contributes most total error due to high volume
- 4 tools with 100% malformed output: prediction_request_reasoning-claude, prediction_request_reasoning, prediction_request_reasoning-5.2.mini, resolve-market-jury-v1 (these are tools we are not hosting — external mech operators)
- Calibration: predictions above 0.3 are severely overconfident. The 0.9-1.0 bucket predicts avg 0.97 but only 30% resolve Yes (gap: +0.67)
- Note: the report's calibration summary incorrectly labelled overconfident predictions as "underconfident" due to a bug in analyze.py

**Actions:**
- Fixed calibration label bug in analyze.py and dedup leak in fetch_production.py: [PR #177](https://github.com/valory-xyz/mech-predict/pull/177) (merged)
- Identified superforcaster as highest-priority tool for improvement (69% of its Brier error from the 0.9-1.0 confidence bucket)
- Ran cached prompt replay experiment on superforcaster using 20 production deliveries fetched from IPFS
- Tested prompt modifications: base-rate anchoring + evidence-aware tail discipline
- Results: **43% Brier reduction** (0.5151 -> 0.2937 on the test set) — 12/20 improved, 3/20 worsened, 5/20 unchanged
- Created branch with prompt edits: `fix/superforcaster-prompt-calibration`
- Full test log: `benchmark/BENCHMARK_TEST_LOG.md`
- Next: run on larger dataset (50-200 rows) to validate on representative sample
- Next: run tournament mode (baseline vs candidate) once available
- Next: analyse cost impact of longer prompts before production deployment

---

## 2026-03-31

**Artifact:** [benchmark-results](https://github.com/valory-xyz/mech-predict/actions/runs/23814551383)
**Data period:** 2026-03-24 to 2026-03-31 (7 days lookback, first run)

**Summary:**
- 22,755 predictions scored, 99% reliability
- Overall Brier: 0.2406
- Top tool: prediction-request-reasoning-claude (Brier: 0.20, 960 predictions)
- Superforcaster: Brier 0.2265 (10,113 predictions) — best Brier at scale in this run
- 3 tools with 100% malformed output (same as Apr 2 minus resolve-market-jury-v1) — external mech operators, not tools we host
- Platform comparison: Omen Brier 0.2398 (20,996), Polymarket Brier 0.2502 (1,759)
- Weak categories: internet (Brier 0.81), music (Brier 0.82) — anti-predictive, worse than coin flip

**Actions:**
- First benchmark run — established baseline metrics
- No immediate action taken; data collection phase
