# Benchmark — Next Steps Plan

**Date:** 2026-04-01
**Status:**
- **Benchmark reporting pipeline** (fetch → score → analyze) is implemented and tested locally. Ran manually and produced reports on 22,755 real predictions. PRs open but cannot merge due to CI failures.
- **Cached replay mode** is not yet built. Requires PR #166 (source content capture) to be deployed first.

---

## Where We Are

The production reporting pipeline is complete:

```
fetch_production.py → production_log.jsonl → scorer.py → scores.json → analyze.py → report.md
```

**First report highlights (22,755 predictions):**
- Overall Brier: 0.24, Accuracy: 69%, Sharpness: 0.37
- Best tools: `prediction-request-reasoning-claude` (0.20, 70% acc), `prediction-offline` (0.20, 70% acc), `superforcaster` (0.23, 73% acc)
- Worst tools: `prediction-request-rag` (0.32, 56% acc), `prediction-online` (0.31, 56% acc)
- 3 tool variants have 100% failure rate (malformed outputs)
- Two categories are anti-predictive: internet (0.81), music (0.82)
- Omen (0.24) and Polymarket (0.25) perform similarly

**What we can do now:** See which tools work, which don't, and where the weak spots are.
**What we can't do yet:** Test fixes without deploying them to production.

---

## How We Fix Issues

The proposal (Part 7: Automated Tool Improvement) describes a 4-level approach, all powered by **cached replay mode**:

| Level | What | How | Example |
|-------|------|-----|---------|
| **1. Parameter Sweep** | Try different models, temperatures, num_urls | Grid search over existing tool parameters using cached replay | Switch `prediction-online` from gpt-4.1 to claude-sonnet → does Brier drop? |
| **2. Prompt Evolution** | LLM-generated prompt variants | Evolve prompts on a dev set, validate on held-out eval set | Generate 50 prompt variants for superforcaster, keep the best |
| **3. Tool Code Modification** | Change reasoning pipeline, add calibration | LLM analyzes failures, proposes code changes, benchmarks them | Add post-hoc calibration to shrink extreme predictions toward 0.5 |
| **4. Ensemble/Routing** | Combine tools, route by category | Average predictions, cascade cheap→expensive, route by question type | Use `prediction-offline` for politics, `superforcaster` for crypto |

**All 4 levels require cached replay mode** — you can't run 50 prompt variants through production. You need to replay the same questions with cached web content so results are comparable.

---

## What's Blocking Us

Two separate workstreams are blocked by CI test failures (Google API quota limits, not related to our code):

### 1. Source content capture (PR #166) — blocks cached replay

**PR:** [#166 — feat: add return_source_content flag with structured source capture](https://github.com/valory-xyz/mech-predict/pull/166)

**What it does:** When `return_source_content=true`, tools capture raw search results and scraped pages into `used_params` alongside the prediction. This gives us contemporaneous content snapshots — the raw material for cached replay.

**Status:** Cannot be merged due to failing CI tests (pre-existing quota issues). Once merged, needs to be deployed to all production mechs. Until deployed, no new predictions will carry source content, and cached replay cannot start.

### 2. Production replay pipeline (PRs #164, #168, #169) — blocks daily reports

**PRs:**
- [#164 — fetch_production.py](https://github.com/valory-xyz/mech-predict/pull/164) (fetches data from subgraphs, scores, generates report)
- [#168 — unit tests](https://github.com/valory-xyz/mech-predict/pull/168) (115 tests covering all three scripts)
- [#169 — daily benchmark workflow](https://github.com/valory-xyz/mech-predict/pull/169) (GitHub Actions workflow with artifact persistence)

**What they do:** The full pipeline that produced the report attached to this PR. Fetches production predictions from on-chain subgraphs, matches to resolved markets, scores with Brier/accuracy/sharpness, generates a breakdown report.

**Status:** Cannot be merged due to same failing CI tests. We ran the pipeline locally and it produced the attached report on 22,755 real predictions. All 115 benchmark-specific unit tests pass.

## What Needs to Happen Next

### 1. Fix CI to unblock merges

The pre-existing integration test failures (Google API quota limits) are blocking all benchmark PRs. This needs to be resolved before any of the above can be merged.

### 2. Deploy PR #166 to production mechs

Once merged, deploy to all mechs. This starts the clock on accumulating contemporaneous content snapshots for cached replay. ~1-2 weeks of predictions needed before we have enough for meaningful replay.

### 3. Build cached replay mode

Once PR #166 is deployed and predictions start carrying source content, build the replay runner (Proposal Part 6) and the automated improvement pipeline (Proposal Part 7, Levels 1-4).

### 4. Tournament mode

Forward-looking predictions on open markets. Less urgent than cached replay but the only path for evaluating retrieval improvements and new tools.

---

## Suggested Order

| # | What | Depends on |
|---|------|-----------|
| 1 | Fix CI (quota limits) to unblock PR merges | Team / infra |
| 2 | Merge + deploy PR #166 (source content capture) | CI fixed |
| 3 | Merge benchmark reporting PRs (#164, #168, #169) | CI fixed |
| 4 | Build cached replay runner + comparator | PR #166 deployed, snapshots accumulating |
| 5 | First parameter sweep (Level 1) on worst tools | Cached replay working |
| 6 | Tournament mode | Cached replay validated |
