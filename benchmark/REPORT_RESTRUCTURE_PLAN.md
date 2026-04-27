# Benchmark Report Restructure — Scoping

Origin: Slack restructure ask (Divya). Step 1 is the per-platform split; the rest is layered on top as separate PRs.

## Requirements

| # | Requirement | Status |
|---|---|---|
| R1 | Two Slack messages per day, one per platform (omenstrat, polystrat) | **Done** — Phase 1 (PR #237) |
| R2 | Each per-platform report includes its own tournament slice | **Done** — Phase 1 (PR #237) |
| R3 | Surface Tool × Category in Slack summary | Open — Phase 3 |
| R4 | Remove "Edge by Difficulty" from daily report | **Done (side effect)** — Phase 1 (PR #237) |
| R5 | Invert deployment status — show **active** tools per platform | Open — Phase 3 |
| R6 | Reduce window 7 → 3 days | Open — Phase 2 |
| R7 | Annotate each metric with ideal/reference value | Open — Phase 2 |
| R8 | Collapse the report to a single window. All-time sections go away; the report and Slack summary both use the rolling window only (David's ask — see Phase 2) | Open — Phase 2 |

## Phases

- **Phase 1** — Per-platform split (R1, R2, R4). **Done** in PR #237. R4 landed here as a side effect of dropping `by_platform_difficulty` / `by_platform_liquidity` sub-blocks in the platform-scoped renderer. Original plan had R4 in Phase 2; restored below.
- **Phase 2** — Window unification + rendering fixes (R6, R7, R8). Depends on Phase 1.
- **Phase 3** — Post-split UX (R3, R5). Depends on Phase 1.

Phase 2 and Phase 3 can run in parallel after Phase 1 lands.

## Phase 1 shipping notes (PR #237)

- **Edge by Difficulty / Liquidity removed.** The `section_edge_analysis` platform gate drops the `### By Platform`, `### By Platform × Difficulty`, and `### By Platform × Liquidity` sub-blocks in every per-platform report. The `by_platform` one-row table is genuinely degenerate in platform-scoped mode; the difficulty / liquidity sub-blocks are multi-row but land under R4's remove-Edge-by-Difficulty ask anyway, so Phase 1 absorbs R4. The scorer still populates these fields in the JSON so future callers can render them if the team changes course.
- **LLM prompt patched for the fleet-wide Tool Deployment Status section.** Phase 3 will invert that section in `analyze.py`. In the meantime, the Slack prompt is tightened to include only deployments whose name starts with the lowercase of `{platform_label}` (e.g. `omenstrat Pearl` + `omenstrat QS` for Omenstrat, `polystrat Pearl` for Polystrat), preventing cross-platform deployment leakage in the summary before Phase 3 lands.
- **Migration after merge:** the first post-merge scheduled run starts with no `scores_<platform>.json` in the downloaded artifact, so the per-platform `## Overall (All-Time)` sections show day-1 data only until rows accumulate. `Since Last Report` and `Last 7 Days Rolling` are fine (they re-read logs). Mitigation: run `python -m benchmark.scorer --rebuild` once against the full log archive (or via `workflow_dispatch` on main after pulling the archived logs) to seed the per-platform accumulators. Skipping this leaves the all-time view thin for ~30 days until catch-up.

---

## Phase 1 — Per-platform split

### Goal

Generate two independent reports (omenstrat, polystrat), each with its own tournament slice, post each as a separate Slack message.

### Architecture

**Single-pass multi-aggregation.** Log reads and row scoring are the expensive work; aggregation is cheap. Load + score rows once, partition into `{all, omen, polymarket}`, run the same aggregation on each partition, emit three JSON files. Same pattern applies to period, rolling, and tournament scoring.

`scores.json` is retained as the canonical cumulative state (used by `scored_row_ids.json` dedup and `fetch_production.py` incremental update) but is no longer rendered into a report. Two per-platform reports are the only rendered artifacts.

### Steps

#### 1.1 Scorer: platform partitioning

**Files:** `benchmark/scorer.py`, `benchmark/tests/test_scorer.py`

**Changes:**
- Add `score_by_platform(rows) -> dict[str, dict]`: partitions on `row.get("platform")`, calls existing `score()` per partition.
- Teach `--rebuild`, `--period-days`, and `--update` paths to emit per-platform files alongside the combined file. New CLI flag `--platform-outputs-dir DIR` writes `DIR/<base>_omen.json` and `DIR/<base>_polymarket.json` where `<base>` matches the existing output filename (`scores`, `period_scores`, `rolling_scores`, `scores_tournament`).
- Incremental `--update` path: accumulator map grows to `{platform: accumulator}` so each row increments its platform slot and the combined slot in one pass. No extra log reads.

**Tests:**
- Property: `score_by_platform(rows)["omen"]` equals `score([r for r in rows if r["platform"] == "omen"])` for all fields in the output dict.
- Integration: fixture with 10 omen + 5 polymarket rows produces three files with correct `n` values.
- `--update` incremental: apply a single batch, assert both combined and per-platform accumulators advance consistently.

**Done when:** scorer invocation from every entry point (rebuild, period-days, update) emits per-platform files and property test is green.

#### 1.2 Analyze: platform-scoped rendering

**Files:** `benchmark/analyze.py`, `benchmark/tests/test_analyze.py`

**Changes:**
- `generate_report()` takes required `platform: Literal["omen","polymarket"]`.
- Header becomes `# Benchmark Report (Omenstrat) — <date>` or `(Polystrat)`.
- Drop sections that are meaningless for a single platform: `section_platform`, `section_tool_platform`, Platform × Difficulty sub-block of `section_edge_analysis`, Platform × Liquidity sub-block.
- Remove the combined/fleet-wide code path entirely. No `platform=None` mode, no combined `report.md` output.
- CLI: `--platform {omen,polymarket}` required; loads `scores_<platform>.json`, `rolling_scores_<platform>.json`, etc.

**Tests:**
- Render with `platform="omen"`: Platform Comparison / Tool × Platform sections absent; header contains `(Omenstrat)`.
- Render with `platform="polymarket"`: header contains `(Polystrat)`; no omen-specific content leaks.
- Empty-platform branch: render when the platform's `by_tool` is empty, assert explicit "no data for this platform" messaging rather than silent empty sections.

**Done when:** two per-platform reports render cleanly and the combined-rendering code path is fully removed.

#### 1.3 Workflow: dual analyze + dual upload

**Files:** `.github/workflows/benchmark_flywheel.yaml`

**Changes:**
- Scoring step: pass `--platform-outputs-dir benchmark/results/` so every scoring invocation emits per-platform files.
- Replace single `analyze` run with two: `--platform omen` and `--platform polymarket`.
- Artifact upload: `report_omen.md` + `report_polymarket.md` + the per-platform scores files. Remove `report.md` from the upload list.

**Tests:** yaml lint + one end-to-end CI run after merge confirms the expected set of uploaded files.

**Done when:** a green CI run uploads exactly two report files and no combined report.

#### 1.4 notify_slack: dual post with platform-scoped prompt

**Files:** `benchmark/notify_slack.py`, `benchmark/tests/test_notify_slack.py`

**Changes:**
- New `--platform-label` CLI arg (e.g. `Omenstrat`).
- `SUMMARY_SYSTEM_PROMPT` parameterized on the label; drop "list all platforms" / "one line per platform" phrasing since input is now single-platform.
- Workflow invokes notify_slack twice (once per report + label).

**Tests:**
- Dry-run: two invocations produce two distinct summaries; neither references the other platform's deployments or markets.
- Smoke: rendered prompt string does not contain residual "list all platforms" phrasing.

**Done when:** CI dry-run produces two platform-scoped Slack-ready messages.

### Risks

- **Prompt regression:** LLM with a new system prompt can produce garbage. Mitigation: include a dry-run summary in the PR description.
- **Fleet-wide sections become per-platform:** Base Rates, Weak Spots, Category Performance etc. are already driven by keys inside each per-platform scores file, so they naturally render platform-scoped once the scorer is partitioned. No extra work — but every section has to be verified against the partitioned data to confirm this holds (self-review rendering pass).
- **Empty-platform branch:** if one platform has zero rows in the window, the per-platform report must render explicit "no data" messaging, not silent empty output.
- **Legacy `scores.json` consumers:** anything downstream that reads `scores.json` for rendering has to be pointed at the partitioned files. Grep the repo for `scores.json` references outside the scorer before Phase 1 lands.

---

## Phase 2 — Window unification + rendering fixes

Depends on Phase 1. All edits touch the Slack summary prompt and analyze sections that Phase 1 already modifies; stacking Phase 2 on top avoids double-editing.

### Steps

#### 2.1 ~~Remove "Edge by Difficulty"~~ — done in Phase 1 (PR #237)

R4 landed as a side effect of the `section_edge_analysis` platform gate. The scorer still populates `by_platform_difficulty` / `by_platform_liquidity` (zero marginal cost) so the data is recoverable if the team changes course. Nothing left to do in Phase 2 for this item.

#### 2.2 Reduce window 7 → 3 days

**Files:** `.github/workflows/benchmark_flywheel.yaml:166`, `benchmark/analyze.py` rolling heading, `benchmark/notify_slack.py:29`

**Changes:**
- Workflow: `--period-days 7` → `--period-days 3`.
- Analyze heading: `"Last 7 Days Rolling"` → `"Last 3 Days Rolling"`.
- Prompt: `"in the last 7 days"` → `"in the last 3 days"`.
- Sweep repo for any remaining `"7 days"`, `period-days 7`, `Last 7 Days` references (comments, docstrings, tests) and reconcile.

**Tests:** existing rolling-section unit tests updated to new heading; grep for `"7 days"` in `benchmark/` returns only historical/test contexts.

**Done when:** no active (non-fixture, non-historical) reference to the 7-day window remains.

#### 2.3 Metric reference legend

**Files:** `benchmark/analyze.py`

**Changes:** render a legend block as the second section of the report (right after header):

```
Metric references:
- Brier (ideal 0.00, coin-flip 0.25; lower is better)
- Log Loss (ideal 0.00; lower is better)
- BSS (ideal > 0; negative = worse than base-rate predictor)
- Edge (ideal > 0; positive = tool beats market)
```

Single legend at top rather than inline clutter per metric row.

**Tests:** unit test that legend text appears once per report.

**Done when:** every rendered report opens with the reference legend.

#### 2.4 Collapse report to single window (the big one)

**Files:** `benchmark/analyze.py`, `benchmark/tests/test_analyze.py`

**Changes:**
- `generate_report()` drops `scores` (all-time) as a content input. It retains `scores` only for historical/trend purposes that read the `scores_history.jsonl` file.
- Remove all-time-driven sections from the render: Overall, Tool Ranking (all-time), Category Performance (all-time), Tool × Category (all-time), Weak Spots (all-time), Edge Over Market (all-time), Diagnostic Metrics (all-time), Calibration (all-time), Latency, Worst/Best Predictions.
- Re-render the rolling-sourced equivalents where the information is still useful: rolling Tool Ranking, rolling Category Performance, rolling Tool × Category, rolling Weak Spots, rolling Diagnostics. These use `rolling_scores_<platform>.json`, which already carries the full schema.
- "Since Last Report" stays but compares against the prior rolling window, not all-time.
- `section_trend(history, ...)` is a long-range direction-of-travel view driven by monthly history; keep as-is (it's not a point-in-time metric and doesn't conflict with the single-window rule).

**Tests:**
- Assert `## Overall` is absent from rendered report.
- Assert rolling Tool Ranking / Category / Weak Spots sections are present and sourced from `rolling_scores`.
- Trend section present and unchanged.

**Done when:** every point-in-time section in the report is driven by `rolling_scores`; no all-time sections remain.

#### 2.5 Slack summary prompt: single-window rewrite

**Files:** `benchmark/notify_slack.py` `SUMMARY_SYSTEM_PROMPT`

**Changes:**
- Rewrite the prompt to assume single-platform + single-window input (stacks on top of Phase 1's prompt edit).
- Remove every reference to "all-time" / "cumulative" / "since the beginning of the month". Every per-tool and per-category bullet is scoped to the rolling window.
- Reorder prompt sections to match the simpler rendered report.

**Tests:**
- Dry-run summary on a fixture: output contains `"last 3 days"` scoping, does not contain `"all-time"` / `"cumulative"`.
- Sanity: summary is still valid Slack mrkdwn (no rogue `**double asterisks**`).

**Done when:** dry-run summary is internally consistent on window scoping.

#### 2.6 Explicit window label on every n=

**Files:** `benchmark/analyze.py` (all retained sections)

**Changes:** every `n=N` in row text carries `(last 3 days)`. A one-line note in the legend block reinforces the default. Any section that deliberately reports a different window (e.g. trend, tournament) carries its own explicit label.

**Tests:** per-section render test that `n=` labels carry the window qualifier.

**Done when:** no unqualified `n=` remains in the rendered report.

### Rendering review for Phase 2

3-day window pushes more sections below `MIN_SAMPLE_SIZE`. Every "insufficient data" branch must emit a line per dimension — not silently skip. Run the rendering review pass (self-review-rendering rule) before calling Phase 2 done.

---

## Phase 3 — Post-split UX

Depends on Phase 1. Steps are independent of each other and can ship as one PR or two.

### Steps

#### 3.1 Invert deployment status (active tools per platform)

**Files:** `benchmark/analyze.py` (`section_tool_deployment_status`), `benchmark/tool_usage.py`, `benchmark/tests/`

**Context from Phase 1:** The Slack summary prompt already filters the fleet-wide `Tool Deployment Status` section to deployments matching the platform prefix (see Phase 1 shipping notes). That is a stopgap — the analyze-side source still lists every deployment. 3.1 moves the filter into the report itself and flips the lens from "disabled" to "active". Once that lands, the prompt bullet can drop the filtering instruction and revert to "one line per deployment".

**Changes:**
- Compute `active = benchmarked_tools \ disabled_for_deployment` per deployment.
- In omen report: render active lists for `omenstrat Pearl` and `omenstrat QS`.
- In polymarket report: render active list for `polystrat Pearl`.
- Failed-fetch branch renders `⚠️ unavailable` — never an empty-list false negative.
- Per-platform gating so omen-deployment sections don't leak into the polymarket report and vice versa.
- Simplify the Slack summary prompt bullet back to unfiltered "one line per deployment" once the source section is already platform-scoped.

**Tests:**
- Inversion math unit test (registry + disabled list → active list).
- Failed-fetch branch emits warning, not empty list.
- Per-platform gating: `polystrat Pearl` section absent from omen report; omen deployment sections absent from polymarket report.

**Done when:** per-platform reports show only their own deployments' active lists and the prompt's filtering instruction has been removed.

#### 3.2 Surface Tool × Category in Slack summary

**Files:** `benchmark/analyze.py` (section ordering), `benchmark/notify_slack.py` `SUMMARY_SYSTEM_PROMPT`

**Changes:**
- Move `section_tool_category` earlier in the rendered markdown so the LLM summarizer weights it more heavily.
- Update prompt: replace `"pick 2–4 standout combinations"` with `"list every cell clearing MIN_SAMPLE_SIZE for this platform"`. Per-platform reports have smaller tables so listing all fits.
- Keep the fallback branch: when no cells clear the threshold, output exactly `"insufficient tool × category data"`.

**Tests:**
- Dry-run summary on a data-rich fixture: output lists every qualifying Tool × Category cell.
- Dry-run summary on a data-sparse fixture: output includes exactly the fallback string.

**Done when:** Tool × Category block reliably appears in every Slack summary where data exists.

---

## Decisions (from Divya, 2026-04-21)

1. Fleet-wide sections → rendered **per-platform** in each report (naturally scoped once the scorer is partitioned).
2. No combined `report.md`. Two reports only — `report_omen.md` and `report_polymarket.md`.
3. Header labels: **Omenstrat** and **Polystrat**.
4. R8: all-time sections are **removed** from the rendered report. Single window (rolling, 3 days post-R6).
5. Tournament: **full tool list** in each per-platform report's tournament slice (tool selection is a tournament-policy question, not a reporting one).

## Code verification (done 2026-04-21)

- **Tournament rows carry platform.** `tournament.py:258` emits `"platform"` on every row; `score_tournament.py:302` already reads it.
- **Rolling scores schema is fully populated.** `scorer.py:821 score()` returns `by_tool` / `by_category` / `by_tool_category` / `by_tool_platform` etc. Rolling scorer calls the same function on a filtered row set, so `rolling_scores.json` has the same schema as `scores.json`. R8a is feasible.
- **`scores.json` non-render consumers:** `fetch_production.py` (incremental update) and `release_map.py` (independent CID-coverage CLI). Neither renders a report. Keeping `scores.json` as cumulative state and retiring its rendered report is safe.
