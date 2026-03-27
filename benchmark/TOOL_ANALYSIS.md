# Prediction Tool Analysis

Qualitative comparison of the prediction tools registered in `TOOLS_TO_PACKAGE_HASH`, evaluated on reasoning logic and expected calibration quality. No empirical data — this is a code/logic-level review.

## Tools covered

| Tool name(s) | Package |
|---|---|
| `prediction-offline`, `prediction-online`, `claude-prediction-online`, `claude-prediction-offline` | `napthaai/customs/prediction_request_rag` |
| `prediction-online-sme`, `prediction-offline-sme` | `nickcom007/customs/prediction_request_sme` |
| `resolve-market-reasoning-gpt-4.1` | `napthaai/customs/resolve_market_reasoning` |
| `prediction-request-rag`, `prediction-request-rag-claude` | `napthaai/customs/prediction_request_rag` |
| `prediction-request-reasoning`, `prediction-request-reasoning-claude` | `napthaai/customs/prediction_request_reasoning` |
| `superforcaster` | `valory/customs/superforcaster` |
| `factual-research` (PR #118) | `valory/customs/factual_research` |

---

## Pipeline summary

| Tool | LLM calls | Search strategy | Scraping | Calibration guidance |
|---|---|---|---|---|
| `prediction_request_rag` | 1 | LLM generates queries | Yes + FAISS embeddings | None |
| `prediction_request_sme` | 2 (SME persona + predict) | LLM generates queries | Yes | None |
| `prediction_request_reasoning` | 2 (reason + predict) | LLM generates queries | Yes + FAISS embeddings | None |
| `resolve_market_reasoning` | 2 (reason + predict) | LLM generates queries | Yes | None (outcome verifier, not forecaster) |
| `superforcaster` | 1 | Raw question, snippets only | No | Strong — structured 7-step chain |
| `factual_research` | 3 (reframe + synthesize + estimate) | Sub-question decomposition | Yes, top 6 pages | Strong — base-rate anchoring + tail constraints |

---

## Calibration discipline

This is the most important differentiator for prediction quality.

**`prediction_request_rag` / `prediction_request_sme` / `prediction_request_reasoning`**

The prediction prompt amounts to: "here's the question and some web content, output p_yes/p_no/confidence/info_utility." No structure for *how* to reason about probability. The model is free to anchor on whatever it encounters in search results — including phrases like "analysts give it 70% odds" or "betting markets favor X."

`prediction_request_sme` adds a persona step (e.g. "you are a political scientist") which could improve domain relevance but adds no calibration structure. `prediction_request_reasoning` adds a generic "reason step-by-step" call before the prediction, which improves coherence but still gives no guidance on converting reasoning to a well-calibrated number.

**`superforcaster`**

Uses a 7-step structured chain in a single call:
1. Extract key factual points (no probability conclusions yet)
2. List reasons the answer is NO with strength ratings (1-10)
3. List reasons the answer is YES with strength ratings (1-10)
4. Aggregate competing factors — explicitly warns about negativity bias and sensationalism bias
5. Output tentative probability
6. Reflect: check for over/underconfidence, base rates, conjunctive/disjunctive conditions
7. Final answer

Also explicitly reminds the model that Brier score evaluation penalises both overconfidence and underconfidence, and that 0.5% and 5% are "markedly different" odds that should not be treated similarly.

**`factual_research`**

Four-stage pipeline with an information barrier:
- Stage 1 (Reframe): Decomposes question into 3-6 sub-questions covering different angles
- Stage 2 (Search): Parallel web searches + page scraping
- Stage 3 (Synthesize): Converts raw evidence into a structured factual briefing — explicitly prohibited from outputting any probabilities or predictions
- Stage 4 (Estimate): Sees *only* the sanitized briefing, not raw web content

The estimation prompt includes: mandatory base-rate anchoring before evidence review, explicit YES/NO signal listing, tail discipline rules (>90% requires identifying specific failure modes that have been eliminated), and confidence-probability coupling constraints.

---

## Evidence quality

**Sub-question decomposition (`factual_research`)** should surface more diverse evidence than a single rephrased query. For a question like "Will X pass Congress?", other tools might return five articles covering the same angle. `factual_research` explicitly asks about competing bills, remaining procedural steps, historical passage rates, expert signals, and obstacles as separate searches.

**FAISS embeddings (`prediction_request_rag`, `prediction_request_reasoning`)** select semantically relevant chunks from scraped pages. `factual_research` uses token-budget truncation instead. For long, information-dense documents, embedding-based retrieval may surface more relevant content. For typical prediction market questions (short news articles), the difference is likely negligible.

**`superforcaster`** only uses Serper snippets (~150 chars per result, top 5 organic + "People Also Ask"). No page scraping. For recent or niche events where the model's training data is thin, this is a weakness. The strong prompt partially compensates by leveraging parametric knowledge, but it can't substitute for actual current reporting.

---

## Information leakage

Only `factual_research` prevents the estimating LLM from seeing raw web content. Every other tool feeds search snippets or scraped pages directly into the call that outputs the probability. This means a single sentence like "prediction markets give it 80%" in a scraped article can heavily anchor the output.

`factual_research`'s synthesis stage explicitly filters this out — the model is instructed never to reference prediction markets, odds, or likelihood language.

---

## `resolve_market_reasoning`

This tool is designed for post-resolution fact-checking, not forecasting. Given a market that has already closed, it verifies whether the event actually occurred. It has additional checks for question validity and determinability. It should not be compared directly to the forecasting tools.

---

## Weaknesses of `factual_research`

- **No embedding-based retrieval** — Relies on token-budget truncation. Could miss relevant content in long documents.
- **3 LLM calls** — Higher cost and latency than single-call tools. The parallelized I/O (search + scrape) reduces wall time but doesn't eliminate the serial LLM pipeline.
- **No explicit debiasing** — Unlike `superforcaster`, it doesn't warn about negativity bias or sensationalism bias. Base-rate anchoring partially compensates but they address different failure modes.
- **Blocked domains** — Aggressively filtering prediction-market domains prevents anchoring but also blocks factual reporting hosted on those domains.

---

## Summary

`factual_research` and `superforcaster` are the only tools with meaningful calibration discipline. The `prediction_request_rag/sme/reasoning` family are essentially "smart LLM + web search" with no structure for how to think about probability — their outputs are likely to be overconfident and poorly calibrated.

Between `factual_research` and `superforcaster`: they address different failure modes. `superforcaster` is better at debiasing and has a lighter computational footprint (1 LLM call, no scraping). `factual_research` is better at evidence diversity (sub-question decomposition), anchoring prevention (information barrier), and calibration discipline (base-rate anchoring, tail constraints). They would complement each other well in an ensemble.

Empirical Brier score data is needed to determine which performs better in practice.
