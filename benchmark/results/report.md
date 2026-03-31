# Benchmark Report — 2026-03-31

## Overall

- Predictions scored: 22456 / 22755 (99% reliability)
- Overall Brier: 0.2406
  - 0.0 = perfect, 0.25 = random guessing, 1.0 = maximally wrong
- Accuracy: 69%
- Sharpness: 0.3659
  - 0.0 = all predictions at 50/50, 0.5 = maximally decisive

## Tool Ranking

1. **prediction-request-reasoning-claude** — Brier: 0.2, Acc: 70%, Sharp: 0.3044 (n=960)
2. **prediction-offline** — Brier: 0.2049, Acc: 70%, Sharp: 0.2511 (n=2094)
3. **superforcaster** — Brier: 0.2265, Acc: 73%, Sharp: 0.4138 (n=10113)
4. **claude-prediction-online** — Brier: 0.2301, Acc: 68%, Sharp: 0.2350 (n=37)
5. **prediction-request-rag-claude** — Brier: 0.2397, Acc: 64%, Sharp: 0.2512 (n=28)
6. **claude-prediction-offline** — Brier: 0.2432, Acc: 63%, Sharp: 0.2795 (n=980)
7. **prediction-request-reasoning** — Brier: 0.258, Acc: 66%, Sharp: 0.3764 (n=6685)
8. **prediction-online-sme** — Brier: 0.2745, Acc: 52%, Sharp: 0.2095 (n=23)
9. **prediction-online** — Brier: 0.3096, Acc: 56%, Sharp: 0.2663 (n=926)
10. **prediction-request-rag** — Brier: 0.3181, Acc: 56%, Sharp: 0.3045 (n=888)
11. **prediction_request_reasoning-claude** — Brier: N/A, Acc: N/A, Sharp: N/A (n=8) — 0% reliability
12. **prediction_request_reasoning** — Brier: N/A, Acc: N/A, Sharp: N/A (n=6) — 0% reliability
13. **prediction_request_reasoning-5.2.mini** — Brier: N/A, Acc: N/A, Sharp: N/A (n=7) — 0% reliability

## Platform Comparison

- **omen**: Brier: 0.2398 (n=20996)
- **polymarket**: Brier: 0.2502 (n=1759)

## Weak Spots

- **internet** (category): Brier 0.8106 (n=91) — anti-predictive (worse than coin flip)
- **music** (category): Brier 0.8199 (n=140) — anti-predictive (worse than coin flip)

## Reliability Issues

- **prediction_request_reasoning-claude**: 100.0% error/malformed rate
- **prediction_request_reasoning**: 100.0% error/malformed rate
- **prediction_request_reasoning-5.2.mini**: 100.0% error/malformed rate

## Worst Predictions

1. "Will OpenAI publicly announce, on or before March 30, 2026, the launch of a n..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: No (Brier: 1.0000)
   Category: tech, Platform: omen
2. "Will the City of Port Arthur or Jefferson County officials publicly announce,..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: No (Brier: 1.0000)
   Category: other, Platform: omen
3. "Will any national government publicly announce, on or before March 29, 2026, ..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: No (Brier: 1.0000)
   Category: internet, Platform: omen
4. "Will Nvidia, OpenAI, Google, or Anthropic publicly announce, on or before Mar..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: No (Brier: 1.0000)
   Category: business, Platform: omen
5. "Will any major U.S. city publicly announce, on or before March 28, 2026, the ..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: No (Brier: 1.0000)
   Category: other, Platform: omen
6. "Will any additional statue or monument previously removed during the 2020 rac..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: No (Brier: 1.0000)
   Category: other, Platform: omen
7. "Will any major consumer router manufacturer (such as TP-Link, Netgear, or Cis..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: No (Brier: 1.0000)
   Category: business, Platform: omen
8. "Will Uber, in partnership with Pony AI and Verne, publicly announce the comme..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: No (Brier: 1.0000)
   Category: business, Platform: omen
9. "Will the UK government publicly announce the specific locations of at least o..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: No (Brier: 1.0000)
   Category: other, Platform: omen
10. "Will TotalEnergies publicly announce, on or before March 29, 2026, the commen..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: No (Brier: 1.0000)
   Category: business, Platform: omen

## Best Predictions

1. "Will Elon Musk or any of his companies (Tesla, SpaceX, or xAI) publicly annou..."
   prediction-request-rag predicted p_yes=1.00, outcome: Yes (Brier: 0.0000)
   Category: business, Platform: omen
2. "Will Markwayne Mullin be publicly sworn in and officially assume the role of ..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: Yes (Brier: 0.0000)
   Category: other, Platform: omen
3. "Will any major U.S. government official publicly announce, on or before March..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: Yes (Brier: 0.0000)
   Category: tech, Platform: omen
4. "Will any major U.S. social media company (Meta, Google/YouTube, TikTok, or Sn..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: Yes (Brier: 0.0000)
   Category: business, Platform: omen
5. "Will the state of Hawaii publicly announce, on or before March 28, 2026, that..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: Yes (Brier: 0.0000)
   Category: other, Platform: omen
6. "Will the Cuban government publicly announce, on or before March 27, 2026, the..."
   prediction-request-reasoning predicted p_yes=0.00, outcome: No (Brier: 0.0000)
   Category: other, Platform: omen
7. "Will Cuba experience another nationwide blackout, as confirmed by major news ..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: Yes (Brier: 0.0000)
   Category: other, Platform: omen
8. "Will any of the class action investors in the lawsuit against Elon Musk regar..."
   prediction-request-reasoning predicted p_yes=0.00, outcome: No (Brier: 0.0000)
   Category: social, Platform: omen
9. "Will the Federal Reserve publicly announce at least one increase to the feder..."
   prediction-request-reasoning predicted p_yes=0.00, outcome: No (Brier: 0.0000)
   Category: economics, Platform: omen
10. "Will Kodiak AI publicly announce the start of fully driverless (no safety dri..."
   prediction-request-reasoning predicted p_yes=1.00, outcome: Yes (Brier: 0.0000)
   Category: business, Platform: omen

## Trend

- 2026-03: Brier 0.2406 (n=22755)

## Sample Size Warnings

- **crypto**: only 9 questions — treat with caution
