# Benchmark — Next Steps Plan

**Date:** 2026-04-01
**Status:** Production replay pipeline is implemented and tested locally. PRs are open but CI is failing due to pre-existing integration test quota limits (Google API rate limits, not related to benchmark code). Benchmark unit tests (115) all pass. We ran the pipeline manually and it produced valuable reports on 22,895 real predictions.
**Context:** This plan covers what to build next, informed by the first benchmark report and team discussion.

---

## Where We Are

The production replay pipeline is complete and running:

```
fetch_production.py → production_log.jsonl → scorer.py → scores.json → analyze.py → report.md
```

**First report highlights (22,895 predictions scored):**
- Overall Brier: 0.54 (worse than random guessing at 0.25)
- Best tools: `prediction-request-rag-claude` (0.22), `claude-prediction-online` (0.23)
- Worst tools: `superforcaster` (0.60), `prediction-request-reasoning` (0.55)
- Polymarket (0.25) dramatically outperforms Omen (0.56)
- Multiple categories are anti-predictive: social (0.81), economics (0.82), pets (0.90)

**What we can do now:** See how bad things are.
**What we can't do yet:** Test fixes, compare alternative configs, or evaluate new tools without deploying them to production.

---

## What's Next (Priority Order)

### 1. Get mech tool changes deployed to all mechs

**Owner:** Requires mech-predict PRs to be merged first.

**What:** The benchmark uncovered that `superforcaster` (10k predictions, Brier 0.60) and `prediction-request-reasoning` (6.7k predictions, Brier 0.55) are the two highest-volume tools and both perform worse than random. Any tool-level fixes (prompt changes, model upgrades, parameter tuning) need to go through the normal mech-predict PR → deploy flow.

**Blockers:** Open mech-predict PRs need review and merge.

---

### 2. Cached Replay Mode

**Owner:** Divya
**Goal:** Re-run production questions through tools locally with different configs, without hitting live web or waiting for markets to resolve.

#### Why this is the priority

The production report tells us *what's broken* but we can't iterate on fixes without replay:
- "superforcaster has Brier 0.60" → want to test: does switching to claude-4-sonnet improve it?
- "politics category is anti-predictive" → want to test: does a different prompt template help?
- "Omen is 0.56 but Polymarket is 0.25" → want to test: is this a tool issue or a question distribution issue?

Cached replay lets us answer these questions in minutes instead of waiting days for production data.

#### How it works

```
Production question: "Will BTC hit $100k by June?"
Production web content: [cached search results + pages from when the tool originally ran]
Tool under test: superforcaster with claude-4-sonnet instead of gpt-4.1
                                    ↓
                 run() with source_content=cached_pages
                                    ↓
                 Compare new p_yes vs original p_yes vs actual outcome
```

#### Prerequisites

1. **Content snapshots from production** — PR #166 (`feat: add return_source_content flag with structured source capture`) already adds this. When `return_source_content=true` is set in `API_KEYS`, tools capture raw source content into `used_params` alongside the prediction. This gives contemporaneous snapshots with perfect temporal alignment — no retroactive scraping needed.

   **Status:** PR #166 is merged but not yet deployed to mechs. Once deployed, new predictions will carry their source content. Older predictions (before deployment) won't have snapshots.

   Structured format per tool type:
   - Web-fetching tools: `{"pages": {url: html}, "pdfs": {url: text}}`
   - Superforcaster: `{"serper_response": <json>}`

2. **Tool `source_content` injection support** (for replaying cached content through tools):
   - ✅ `prediction_request` (all variants: reasoning, rag, online, offline, claude)
   - ✅ `superforcaster`
   - ✅ `prediction_request_rag` (added in PR #166)
   - ✅ `prediction_request_reasoning` (added in PR #166)
   - ✅ `prediction_url_cot` (added in PR #166)
   - ✅ `prediction_request_sme` (added in PR #166)
   - ❌ `prediction_langchain`
   - ❌ `gemini_prediction`
   - ❌ `corcel_request`

   Tools with `source_content` support cover **95%+ of production volume**.

#### Implementation Plan

**Phase A: Snapshot extraction (~2 days)**

Once PR #166 is deployed and predictions start carrying `source_content` in `used_params`, we need to extract and store those snapshots for replay.

1. **`benchmark/datasets/extract_snapshots.py`** — pull source content from production deliveries
   - Query marketplace subgraph for deliveries that have `used_params` with source content
   - Extract and store locally:
   ```
   snapshots/
     {market_id_or_hash}/
       metadata.json      # question, snapshot_at, resolved_at, snapshot_origin
       source_content.json # the raw {pages: {url: html}, pdfs: {url: text}} or {serper_response: ...}
   ```
   - Tag all with `snapshot_origin: "contemporaneous"` (captured at prediction time)

2. **Blocker: deploy PR #166 to mechs** — until this is deployed, no new predictions will carry source content. Older predictions won't have snapshots. The pipeline only works for predictions made after deployment.

**Phase B: Benchmark runner (~2 days)**

4. **`benchmark/runner.py`** — the core replay engine
   ```python
   def run_benchmark(
       tools: list[str],
       dataset: str,          # path to production_log.jsonl or custom dataset
       models: list[str],
       mode: str = "cached_replay",
       snapshot_dir: str = "benchmark/datasets/snapshots/",
   ) -> str:  # returns path to results JSONL
   ```
   - Load questions from dataset
   - For each question × tool × model:
     - Load content snapshot
     - Call `tool.run()` with `source_content=snapshot`
     - Record p_yes, p_no, latency, cost
   - Output results JSONL in the same schema as production_log

5. **`benchmark/compare.py`** — compare two runs
   ```
   $ python benchmark/compare.py baseline.jsonl candidate.jsonl
   
   Tool: superforcaster
     Baseline (gpt-4.1):     Brier=0.60
     Candidate (claude-4-sonnet): Brier=0.42  ← improvement
     Delta: -0.18 on 200 questions
   ```

**Phase C: Integration (~1 day)**

6. Score and analyze replay results using the existing `scorer.py` and `analyze.py` (they're mode-agnostic — `mode: "cached_replay"` just flows through)

7. Add `--mode cached_replay` to the scorer/analyzer to filter by mode if both production and replay data are in the same log

#### What this unblocks

- Test prompt changes without deploying
- Compare models (gpt-4.1 vs claude-4-sonnet vs gemini) on the same questions
- Identify if Omen's poor performance is a question difficulty issue or a tool issue
- Run parameter sweeps (temperature, num_urls, num_queries)
- CI regression tests for tool PRs

---

### 3. Tournament Mode

**Owner:** Jens / Divya
**Goal:** Forward-looking predictions on open (unresolved) markets — the only way to evaluate new tools on unseen questions with zero temporal contamination.

#### How it differs from cached replay

| | Cached Replay | Tournament |
|---|---|---|
| Questions | Already resolved | Currently open |
| Web content | Cached/historical | Live |
| Temporal integrity | Depends on snapshot quality | Perfect |
| Speed | Instant | Wait days/weeks for resolution |
| Use for | Fast iteration, CI | Final validation, new tool eval |

#### Implementation Plan

1. **`benchmark/datasets/fetch_open.py`** — fetch currently open markets from Polymarket/Omen
2. **`benchmark/tournament.py`** — run all tools on open markets, store predictions with timestamps
3. **`benchmark/score_tournament.py`** — scheduled job that matches stored predictions against resolutions as markets close
4. Content snapshots saved during tournament runs (free data for future cached replay)

#### Timeline

Tournament mode is less urgent than cached replay because:
- It requires waiting for markets to resolve (days/weeks latency)
- Cached replay gives faster feedback for the immediate need (testing fixes)
- But tournament is the only path for evaluating retrieval improvements and new tools

Suggest starting after cached replay is validated and producing useful results.

---

### 4. Infra Team Confirmation

**Owner:** Jens (requires rough numbers)

**What's needed:**
- Storage for content snapshots: ~10KB per question × ~1000 questions/month = ~10MB/month
- GitHub Actions minutes: ~5 min/day for production replay, ~30 min for replay runs
- If snapshots go to IPFS instead of artifacts: pin costs

**Not urgent** — current volumes are small. Becomes relevant when we have thousands of snapshots or run sweeps with hundreds of tool variants.

---

## Suggested Sprint Plan

| Week | What | Who | Blocker |
|------|------|-----|---------|
| **This week** | Deploy PR #166 to mechs | Team | Merge open mech-predict PRs |
| **This week** | Cached replay Phase B (runner + compare) | Divya | Can start without deployment — use retroactive snapshots for dev/testing |
| **Next week** | Cached replay Phase A (extract production snapshots) | Divya | PR #166 deployed, predictions with source content flowing |
| **Next week** | Start tournament mode design | Jens + Divya | — |
| **Week 3** | Tournament mode implementation | Divya | — |
| **Week 3** | Infra numbers for long-term storage | Jens | — |

---

## Open Questions

1. **Snapshot storage:** GitHub Actions artifacts (90 day TTL) vs IPFS (permanent) vs S3? Artifacts are simplest but snapshots are more valuable if they persist forever.

2. **Which tools to prioritize for replay?** `superforcaster` + `prediction-request-reasoning` are 75% of volume and the worst performers. Both already support `source_content`. Start there?

3. **How long until enough contemporaneous snapshots?** After PR #166 is deployed, we need ~1-2 weeks of predictions to accumulate enough snapshots for meaningful replay. In the meantime, we can build and test the runner using a few manually-constructed test snapshots.

4. **`used_params` availability in subgraph:** Does the marketplace subgraph expose `used_params` from deliveries, or do we need to fetch from IPFS? This determines how `extract_snapshots.py` works.
