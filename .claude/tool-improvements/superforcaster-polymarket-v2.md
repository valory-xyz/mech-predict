## Issue #412 -- 2026-07-30

- **Trigger:** regression (count-window fallback) on `polymarket` C-1 (`2026-07-14` -> `2026-07-29`); headline Brier `0.3585` vs C-2 `0.2865` (delta `+0.0720`).
- **PR:** #417 (branch `tool-improvement/superforcaster-polymarket-v2-max-tokens-chain-of-thought`).
- **Confirmed hypothesis:** At the prediction-LLM-call stage, `OUTPUT_FORMAT` instructs the model to emit only JSON, causing it to skip the 7-step chain-of-thought (including step 6s resolution-criterion analysis); with `max_tokens=500` any attempt at chain-of-thought is also truncated; the model therefore assigns high p_yes based on topical relevance of retrieved snippets without checking whether the evidence directly confirms the resolution criterion.
- **Stage + code site:** `prediction-LLM-call` stage -- `PREDICTION_PROMPT` OUTPUT_FORMAT block + `DEFAULT_OPENAI_SETTINGS["max_tokens"]` in `superforcaster_polymarket_v2.py` (gate-visible: downstream of source_content injection).
- **Localized cells:** `by_tool_category=other (n=76, brier=0.4127)` from scores_polymarket.json; C-1 per-category: `politics (n=60, brier=0.3307)`, `other (n=22, brier=0.4655)`, `business (n=7, brier=0.4856)`.
- **Evidence sample (worst-miss rows):**
  | question (truncated 80 char) | p_yes | outcome | evidence_finding |
  |---|---|---|---|
  | Will Trump say "Forever" during tribute to Lindsey Graham? | 0.99 | 0 | good-evidence/bad-reasoning |
  | Will "Hormuz" be in the headlines this week? | 0.98 | 0 | good-evidence/bad-reasoning |
  | Will Tesla say "Cybertruck" during earnings call? | 0.98 | 0 | good-evidence/bad-reasoning |
  | Will Trump say "Spain" this week? | 0.98 | 0 | good-evidence/bad-reasoning |
  | Will Trump say "Middle East" during Speech to the Nation? | 0.97 | 0 | good-evidence/bad-reasoning |
  | Will T-Mobile (TMUS) Q2 total service revenues above $19.2B? | 0.97 | 0 | good-evidence/bad-reasoning |
  | Will Trump say "Make America Great Again" during Speech? | 0.93 | 0 | good-evidence/bad-reasoning |
  | Will Netflix say "NFL" / "Football" during earnings call? | 0.93 | 0 | good-evidence/bad-reasoning |
- **Mechanism disrupted:** Require the model to emit the full 7-step chain-of-thought (including step 6 resolution-criterion analysis) BEFORE the JSON output, enabled by raising `max_tokens` from 500 to 2000; in superforcaster-polymarket-v5, `OUTPUT_FORMAT` says to append the JSON after the analysis chain, forcing criterion-check execution before p_yes is formed.
- **Pre-PR sanity (Step 6.5):** import OK, `autonomy packages lock --check` green. **W-2 is the only scored gate** -- recorded by PR-CI on the PR after hand-off, not by the agent.
- **Status:** opened draft PR; PR-CI cached-replay pending on W-2.
- **Benchmark 2026-07-30:** SHA `b506836fb5dd0b5cc1524a419f580020a447dad3`, seed workflow-default, n=100, dev, baseline=superforcaster-polymarket-v2, platform=polymarket -- posted (comment #5127634652).
- **Benchmark result 2026-07-30 (dev, n=100, seed 42):** Brier 0.3233->0.3277 (+0.0044, +1.4%), DA 54%->56% (+3.7%), overconf-wrong 23->24 (+1), parse 100% vs 96.7% prod. **Verdict: inconclusive** -- delta (0.0044) well within noise (SE approx +/-0.03-0.04 at n=100); fingerprint (+1 overconf-wrong) within noise. E1 action: grow sample.
- **Benchmark 2026-07-30 (dev, n=300, grow-sample):** SHA `262878e296db63b19227abb9cdb4ab03def78bc8`, seed workflow-default, n=300, dev, baseline=superforcaster-polymarket-v2, platform=polymarket -- posted (comment #5127938655). Note: 262878e is a memory-only commit; tool code unchanged from b506836. Awaiting result.
- **Benchmark result 2026-07-30 (dev, n=301, seed 42):** Brier 0.3188->0.3147 (-0.0041, -1.3%), DA 58.0%->57.1% (-1.5%), overconf-wrong 75->71 (-5.3%, rate 0.2492->0.2359), parse 301/301 (100%) vs 555/574 (96.7%) prod. **E1 verdict: within noise** -- absolute delta (0.0041) << 2*SE (~0.054 at n=301 for Brier~0.32); contradicts run-1 direction (run-1: +1.4%). Targeted fingerprint (overconf-wrong) moved in expected direction (-5.3%). Parse rate improvement unambiguous across both runs. E1 action: grow to n=500 (cap).
- **Benchmark 2026-07-30 (dev, n=500, grow-sample E1 cap):** SHA `6b9abc8a3ad40ee0895444842d6bed60f93f4b13`, seed workflow-default, n=500, dev, baseline=superforcaster-polymarket-v2, platform=polymarket -- posted (comment #5128542509). Awaiting result.
