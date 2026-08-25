# Actions to improve ROI, accuracy and Brier — ranked

> **Status:** action list, 2026-08-25. Ranked by impact vs effort. The displayed metrics (website) and the real trading metrics are improved by different actions — both matter. Deeper background: [ROI_LOOP_PLAN.md](ROI_LOOP_PLAN.md).

| # | Action | What it improves | Impact | Effort |
|---|--------|------------------|--------|--------|
| 1 | **Fix the daily report pipeline** (broken ~10 days: one platform's data outage stops the whole run) | Website numbers update at all | High | 1–2 days |
| 2 | **Fix the website ROI calculation** — it shows worse ROI than reality (stale payout data; newer bets not counted; some traders show no ROI at all) | Displayed ROI, immediately | High | Low (with the website maintainer) |
| 3 | **Claim unredeemed winnings** — some agents never collect winning bets | Real + displayed ROI | Medium | Low |
| 4 | **Stop betting the loss-pocket market types** (linked "above $N" ladders + weather favourites — ~23% of stake, ~83% of the loss; validated on held-back data: −7.3% → +0.3%) | Real ROI | Highest validated | Medium (trader config gate) |
| 5 | **Bet only researchable markets** — drop "will word X be said"-type questions where research cannot help (~62% of forecast spend) | Real ROI + accuracy | High | Medium (productionize the classifier first) |
| 6 | **Fix the fallback that bets when the main strategy declines** — a "no edge" verdict can still become a bet today | Real ROI | Medium | Low |
| 7 | **Verify the trader's edge-cap setting** — a July widening was silently reverted in a merge; one config line to confirm or restore | Real ROI | Small–medium | Minutes |
| 8 | **Promote the tool version that passed two independent benchmarks** (v5 vs v2: better Brier and directional accuracy, twice) — after one final check against the market price | Accuracy + Brier | Medium | Low |
| 9 | **Make the improvement loop judge tools on money, not only Brier** (keep the market price in benchmarks; profit-based verdicts and issues) | Every future improvement targets ROI | Compounding | ~1 week |

**Reading order:** actions 1–3 fix the *displayed* numbers this week · actions 4–7 improve *real* ROI · actions 8–9 make accuracy gains translate into money.
