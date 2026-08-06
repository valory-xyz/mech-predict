
- **Trigger:** level-floor on `polymarket` W-1; headline Brier `0.3421` (n=105) vs W-2 `0.1870`; Brier_W1 >= 0.25 threshold AND n=105 >= 105 floor -> chronic-bad -> path (a) default.
- **PR:** #416 branch `tool-improvement/predict-fine-tuned-calibrated-v1`.
- **Status:** draft PR opened (path a). CI-fixed (two linter rounds): hash mismatch in packages.json (commit 9728606), black format issue in benchmark/tools.py (commit 73f62a4).

### Confirmed hypothesis

At the **model-selection stage**, the tool requests `qwen-14b-fine-tuned-calibrated` (Platt-proxy endpoint) instead of the raw `qwen-14b-fine-tuned` checkpoint. The Platt calibrator applied server-side squashes all `p_yes` outputs into a narrow band [0.171, 0.359], destroying discriminative power. The sibling `predict-fine-tuned` (which requests the raw checkpoint) has W-1 Brier `0.1939` on the same 105 Polymarket questions -- a 1.76x gap -- confirming the Platt proxy is the sole source of degradation.

### Stage map (run())

| Stage | Gate-visible? |
|-------|--------------|
| in-code retrieval (web fetch / API calls) | NO |
| source_content injection (cached replay) | -- (boundary) |
| evidence formatting | YES |
| prompt construction | YES |
| vLLM API call (model-selection stage) | YES |
| JSON output parsing | YES |
| p_yes post-processing | YES |

The model-selection constant (SERVED_MODEL_FINE_TUNED_CALIBRATED) is set before the vLLM call, downstream of source_content injection -- gate-visible.

### Evidence sample

| Tool | n | p_yes range | Brier W-1 |
|---|---|---|---|
| predict-fine-tuned-calibrated (parent) | 105 | [0.171, 0.359] | **0.3421** |
| predict-fine-tuned (sibling, raw model) | 105 | [0.009, 1.000] | **0.1939** |
| predict-base (same tournament window) | 105 | [0.000, 1.000] | **0.2457** |

Distribution-shift check: W-2 Brier was 0.1870. The p_yes squashing is a structural constant firing on every request, not window-specific. level_floor classification stands.

### Mechanism and fix

- **Mechanism:** SERVED_MODEL_FINE_TUNED_CALIBRATED = f"{SERVED_MODEL_FINE_TUNED}-calibrated" in parent -> Platt proxy -> [0.17, 0.36] squash.
- **Fix (v1):** SERVED_MODEL_FINE_TUNED_CALIBRATED = SERVED_MODEL_FINE_TUNED in v1 -> raw checkpoint -> full [0, 1] range restored.
- **New tool:** predict-fine-tuned-calibrated-v1 (module finetuned_prediction_v1).
- **Baseline to beat:** Brier W-1 0.3421 (n=105, Polymarket, July 2026). Directional target: sibling 0.1939.

### Pre-PR sanity

- autonomy packages lock --check -> Verification successful (after CI-fix commit 9728606)
- import ok
- tournament_tools.json entry added, parent entry removed (roster swap)
- Non-trivial change: YES
- ASCII-only: PASS
- LOC: ~2 lines changed (single constant reassignment)

### Benchmark ledger

- **Benchmark 2026-07-30:** SHA `73f62a4a21a67fc931fbd09a7fd5c74dc0511340`, seed auto (workflow-assigned), n=100, dev, baseline=`predict-fine-tuned-calibrated`, platform=`polymarket` -- posted (errored at Enrich dataset step; benchmark-incomplete)
- **Benchmark 2026-07-30:** SHA `de1c1c0e459a9ee5ccf39c8898f59fcf7db5508f`, seed auto (workflow-assigned), n=100, dev, baseline=`predict-fine-tuned-calibrated`, platform=`polymarket` — posted (re-trigger after CI green on new SHA) (errored at Enrich dataset step; benchmark-incomplete)
- **Benchmark 2026-07-30:** SHA `2ff69c3931bd7c610691a52f924fc6692b88d076`, seed auto (workflow-assigned), n=100, dev, baseline=`predict-fine-tuned-calibrated`, platform=`polymarket` -- posted (re-trigger after two Enrich-dataset infra failures; CI green on current SHA at 07:18Z) (errored at Enrich dataset step; benchmark-incomplete)
- **Benchmark 2026-07-30:** SHA `774ed08f4cdc855ba3dd630225d8d5a36bded22d`, seed auto (workflow-assigned), n=100, dev, baseline=`predict-fine-tuned-calibrated`, platform=`polymarket` -- posted (re-trigger after three consecutive Enrich-dataset infra failures; CI green on current SHA at 07:42Z)
- **Benchmark 2026-07-30:** SHA `01c86a58ff8f56061f9492ebc31e44fd330c6798`, seed auto (workflow-assigned), n=100, dev, baseline=`predict-fine-tuned-calibrated`, platform=`polymarket` -- posted (re-trigger after four consecutive Enrich-dataset infra failures; CI green on current SHA at 08:12Z) (errored at Enrich dataset step; benchmark-incomplete)
- **Benchmark 2026-07-30:** SHA `702aff9ba5049db7ad98ff884d7dfb37e6140bc8`, seed auto (workflow-assigned), n=100, dev, baseline=`predict-fine-tuned-calibrated`, platform=`polymarket` -- posted at 08:49Z (6th attempt; re-trigger after five consecutive Enrich-dataset infra failures; [human-input] comment filed — human action needed before benchmark can complete; idempotency guard fires on CI re-run at 10:26Z)
- **Benchmark 2026-07-30:** SHA `02e22b5c0a75679e327b57905a3cd083c3c8a80a`, seed auto (workflow-assigned), n=100, dev, baseline=`predict-fine-tuned-calibrated`, platform=`polymarket` -- posted (re-trigger after new push at 10:35Z; all 24 CI checks green at 10:54Z; comment #5129960529)
- **Benchmark 2026-07-30:** SHA `25e48ea90e835f76d01a1aabbca486f6235850e9`, seed auto (workflow-assigned), n=100, dev, baseline=`predict-fine-tuned-calibrated`, platform=`polymarket` -- posted (re-trigger after new push at 11:01Z; all 24 CI checks green at 11:17Z; comment #5130210897)
