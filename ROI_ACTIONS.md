# Actions to improve ROI, accuracy and Brier — ranked

> **Status:** action list, 2026-08-25. Ranked by impact vs effort. Some actions fix the numbers we *display*, others improve the *real* trading results — both matter. Deeper background: [ROI_LOOP_PLAN.md](ROI_LOOP_PLAN.md).

| # | Action | What it improves | Impact | Effort |
|---|--------|------------------|--------|--------|
| 1 | **Fix the daily report pipeline** — broken for ~10 days: when one platform's data source fails, the whole daily run stops | Website numbers update at all | High | 1–2 days |
| 2 | **Fix the website ROI calculation** — it currently shows a worse ROI than reality: stale payout data, newer bets not counted, some traders showing no ROI at all | Displayed ROI, immediately | High | Low (with the website maintainer) |
| 3 | **Collect unclaimed winnings** — some agents win bets and never collect the payout | Real + displayed ROI | Medium | Low |
| 4 | **Stop betting the market types where most of the loss is** — families of linked "above $N" questions and weather markets: ~23% of the money bet, ~83% of the loss. Replaying past bets without them turns −7.3% into +0.3% | Real ROI | Highest proven | Medium (trader configuration) |
| 5 | **Only bet questions research can help with** — drop "will word X be said"-type markets, where searching the web cannot improve a guess (~62% of forecast spend goes there) | Real ROI + accuracy | High | Medium (build the market classifier first) |
| 6 | **Fix the backup betting rule** — when the main strategy says "no advantage here, don't bet", a fallback rule can still place the bet. It shouldn't | Real ROI | Medium | Low |
| 7 | **Double-check one trader setting** — a betting limit agreed in July was accidentally undone in a code merge; one line to confirm or restore | Real ROI | Small–medium | Minutes |
| 8 | **Promote the tool version that already won two independent benchmarks** (v5 vs v2: more accurate, twice) — after one final check that it also beats the market price | Accuracy + Brier | Medium | Low |
| 9 | **Make the improvement loop judge tools on money, not only accuracy** — keep the market price in benchmarks, and base verdicts and issues on simulated profit | Every future improvement targets ROI | Compounding | ~1 week |

**Reading order:** actions 1–3 fix the *displayed* numbers this week · actions 4–7 improve *real* ROI · actions 8–9 make accuracy gains translate into money.
