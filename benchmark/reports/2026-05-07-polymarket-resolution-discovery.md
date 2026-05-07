# Polymarket Resolution Discovery Investigation

Date: 2026-05-07

Run under investigation: GitHub Actions run `25474907796`

Artifact under investigation: `benchmark-data`, artifact id `6846974035`

## Summary

The latest benchmark workflow did not fail at the CI level. The May 7 run completed successfully, produced artifacts, and executed the fetch, score, rolling score, and Polystrat analysis steps successfully.

The observed Polymarket data gap is better explained by the Polymarket resolution discovery query than by a general subgraph outage.

The fetched Polymarket deliveries contain valid `market_id`s. The current bet-based resolved-market query does not discover the resolved questions for those `market_id`s. A direct question-based query using those same `market_id`s does discover resolutions for many of them.

The failure is therefore not in fetching Polymarket deliveries. It is in constructing the resolved-market set used to join deliveries to outcomes. Because the resolved-market set is incomplete, many Polymarket deliveries stay in `pending_deliveries` instead of being written as scored production rows.

Suggested fix: discover Polymarket resolved markets directly from `questions(where: { resolution_: { blockTimestamp_gt: ... } })`, then keep the existing deterministic `question.id == request_context.market_id` match.

## Problem Statement

The benchmark pipeline has three separate stages for this data path:

```text
1. Fetch Polymarket deliveries
   marketplace subgraph -> delivery records with request_context.market_id

2. Fetch resolved Polymarket markets
   prediction subgraph -> question id + resolution outcome

3. Join them
   delivery.market_id == resolved_question.id
```

The evidence points to stage 2, not stage 1.

Observed:

- Stage 1 works: the reproduction fetched `3,892` Polymarket deliveries into `pending_deliveries`.
- Those deliveries include `189` unique `market_id`s.
- Direct lookup of those exact `market_id`s as `question(id: market_id)` found all `189` question entities.
- Direct lookup found non-null `question.resolution` for `97` of them.
- Current production discovery through `bets -> question -> resolution` found `0` overlap with the resolved pending delivery IDs.

More precise formulation:

```text
The fetched Polymarket deliveries contain valid market_ids.
The current bet-based resolved-market query does not discover the resolved questions for those market_ids.
A direct question-based query using those same market_ids does discover resolutions for many of them.
```

This is also why "closed market" needs careful wording. A closed market is not always resolved. Some pending Polymarket markets have `resolution: null` and should remain pending. The bug is that many already-resolved markets also remain pending because the current resolved-market query does not discover them.

## Scope And Assumptions

Evidence is split into three categories:

1. GitHub metadata: run status, job steps, artifact sizes, and commit.
2. Local reproduction: running the repository fetch pipeline against temporary files.
3. Subgraph probes: direct GraphQL checks against the Polymarket subgraph.

The actual `benchmark-data` artifact ZIP was not downloaded because unauthenticated artifact download returned `401 Requires authentication`. A local attempted download at `/private/tmp/mech_report_6846974035.zip` was not a ZIP; it contained the text `Not Found`. Workflow log download returned `403 Must have admin rights to Repository`.

Therefore, this report does not claim to have inspected the exact uploaded artifact contents. It does show that the same production fetch path can reproduce a Polymarket-specific near-empty result, and it identifies a concrete data source mismatch.

## Run-Level Evidence

May 7 run:

- Run id: `25474907796`
- Event: `schedule`
- Branch: `main`
- Commit: `b9ce0e4319aa8309e8e1f5e730c5b1a9e9e1ddf7`
- Status/conclusion: `completed/success`
- Started: `2026-05-07T03:45:04Z`
- Updated: `2026-05-07T04:26:51Z`

May 7 artifacts:

| Artifact | Id | Size | Created |
| --- | ---: | ---: | --- |
| `benchmark-data` | `6846974035` | 12,832,834 bytes | `2026-05-07T04:26:25Z` |
| `benchmark-log-25474907796` | `6846977890` | 6,650,894 bytes | `2026-05-07T04:26:49Z` |
| `tournament-predictions` | `6846926436` | 84,268,784 bytes | `2026-05-07T04:21:44Z` |
| `tournament-markets` | `6846559639` | 39,648 bytes | `2026-05-07T03:45:18Z` |

Relevant May 7 benchmark job steps:

| Step | Result | Time |
| --- | --- | --- |
| Download previous benchmark data | success | `2026-05-07T04:22:10Z..04:22:17Z` |
| Fetch production data | success | `2026-05-07T04:22:19Z..04:26:10Z` |
| Score | success | `2026-05-07T04:26:10Z..04:26:10Z` |
| Merge tournament into cumulative scores | success | `2026-05-07T04:26:10Z..04:26:12Z` |
| Score current rolling window | success | `2026-05-07T04:26:12Z..04:26:15Z` |
| Score previous rolling window | success | `2026-05-07T04:26:15Z..04:26:21Z` |
| Analyze Omenstrat | success | `2026-05-07T04:26:21Z..04:26:22Z` |
| Analyze Polystrat | success | `2026-05-07T04:26:22Z..04:26:23Z` |
| Upload benchmark data | success | `2026-05-07T04:26:23Z..04:26:25Z` |

Interpretation: this is not consistent with a total CI failure or a total artifact upload failure.

## Yesterday Comparison

May 6 run:

- Run id: `25415304673`
- Same commit: `b9ce0e4319aa8309e8e1f5e730c5b1a9e9e1ddf7`
- Benchmark job conclusion: success
- `benchmark-data` size: 12,137,872 bytes

May 7 run:

- Run id: `25474907796`
- Same commit: `b9ce0e4319aa8309e8e1f5e730c5b1a9e9e1ddf7`
- Benchmark job conclusion: success
- `benchmark-data` size: 12,832,834 bytes

Both runs followed the same workflow path and used the same code. That means "yesterday looked fine" does not imply the resolver was correct yesterday. It is more likely that the issue was latent and became visible because of the current-window market mix or report aggregation.

A timing probe on the current pending Polymarket set found 83 unique Polymarket market IDs, representing 1,523 deliveries, that were delivered and closed before the May 6 run and already had non-null `question.resolution` via direct lookup. Those would still be missed by the current bet-based discovery if they were absent from the discovered `bets` set.

## Local Reproduction Evidence

Command used against temporary files:

```bash
python3 -m benchmark.datasets.fetch_production \
  --lookback-days 7 \
  --logs-dir /private/tmp/mech_probe_logs \
  --state-file /private/tmp/mech_probe_fetch_state.json \
  --scores /private/tmp/mech_probe_scores.json \
  --history /private/tmp/mech_probe_scores_history.jsonl
```

Reproduction summary:

| Metric | Omen | Polymarket |
| --- | ---: | ---: |
| Rows written to production log | 9,418 | 1 |
| Pending deliveries after run | 20,618 | 3,892 |

Overall:

- Total rows written: 9,419
- Valid predictions: 9,181
- Rows by platform: `omen=9418`, `polymarket=1`

Interpretation: the fetch pipeline is not globally empty, and Polymarket delivery fetching is not empty. The failure is specifically that fetched Polymarket deliveries are not being matched to resolved Polymarket questions, so they remain pending instead of becoming scored rows.

## Code Path Under Investigation

Run commit: `b9ce0e4319aa8309e8e1f5e730c5b1a9e9e1ddf7`.

Relevant code locations in `benchmark/datasets/fetch_production.py` at that commit:

- `POLYMARKET_BETS_QUERY`: line 727
- `fetch_polymarket_resolved`: line 1085
- `POLYMARKET_BETS_QUERY` used inside `fetch_polymarket_resolved`: line 1100
- Comment stating `Polymarket question ID matches request_context.market_id`: line 1137
- `_match_delivery`: line 1156
- Deterministic `market_id in markets.by_id` match: line 1171

Current resolved-market discovery mechanism:

1. Fetch Polymarket deliveries from the marketplace subgraph. This produces delivery records with `market_id`.
2. Query recent Polymarket `bets` from the prediction subgraph.
3. For each bet, inspect `bet.question.resolution`.
4. Keep only questions whose `resolution.blockTimestamp > resolved_after`.
5. Build `ResolvedMarkets` indexed by `question.id`.
6. Match deliveries by `delivery.market_id in resolved_markets.by_id`.

The deterministic match itself is reasonable: `delivery.market_id` is intended to match `question.id`. The weak part is using `bets` as the discovery index for resolved questions. The evidence shows the `question` entities exist and have resolutions, while the bet-based discovery path does not include them.

## Subgraph Evidence

The Polymarket subgraph responds and returns data. A hard endpoint outage is therefore not the best explanation.

The key comparison is between two queries against the same Polymarket prediction subgraph:

```text
Current production discovery:
  bets -> bet.question -> question.resolution

Direct diagnostic discovery:
  question(id: delivery.market_id) -> question.resolution
```

The second query uses the exact `market_id`s that came from the fetched Polymarket deliveries.

Current pending Polymarket deliveries:

- Pending deliveries: 3,892
- Unique pending `market_id`s: 189
- All 189 direct `question(id)` lookups existed.
- 97 of those questions had non-null `resolution`.
- 92 had `resolution: null`.

By close date:

| Close date | Unique market ids | Delivery count | Non-null resolution | Null resolution |
| --- | ---: | ---: | ---: | ---: |
| 2026-05-01 | 24 | 171 | 24 | 0 |
| 2026-05-03 | 38 | 1,275 | 38 | 0 |
| 2026-05-04 | 10 | 24 | 10 | 0 |
| 2026-05-05 | 4 | 39 | 4 | 0 |
| 2026-05-06 | 19 | 38 | 19 | 0 |
| 2026-05-07 | 66 | 1,853 | 1 | 65 |
| 2026-05-08 | 7 | 47 | 0 | 7 |
| 2026-05-10 | 21 | 356 | 1 | 20 |

Interpretation:

- Some pending markets are legitimately unresolved, especially May 7 or later close dates.
- But many older pending markets already have non-null resolution and should be matchable.
- Therefore the problem is not that all pending deliveries are unresolved; the problem is that the current resolved-market query misses many resolved questions for those delivery IDs.

Comparison of resolved-market discovery sources for cutoff `1777522939` (`2026-04-30T04:22:19Z`, the May 7 run window):

| Metric | Count |
| --- | ---: |
| Pending delivery rows | 3,892 |
| Pending unique market ids | 189 |
| Resolved markets discovered by current bet-based logic | 69 |
| Resolved questions discovered by direct question query | 22,471 |
| Pending ids resolved via current bet-based logic | 0 |
| Pending ids resolved via direct question query | 96 |
| Pending resolved questions missed by current discovery | 96 |

Sample missed resolved pending questions:

| Market id | Title | Resolution |
| --- | --- | --- |
| `0x0bd7152cc36d115bf9b3f40c5d61ec41b240a6781defd49fb52af372865b108a` | Will Apple (AAPL) close above $285 on May 6? | `winningIndex=0`, `blockTimestamp=1778102111` |
| `0x15ca9b8a291a205bded7a1d9e4e25a1351a50fefc9dc6d4f02a4f65a6c20a3cc` | Will Tesla (TSLA) close above $390 on May 6? | `winningIndex=0`, `blockTimestamp=1778102043` |
| `0xa5ed7c5f0c37897af7593ef0d67fed2013e8e25d9508c08e209dee90f7bee733` | Will Tesla (TSLA) close above $400 on May 6? | `winningIndex=1`, `blockTimestamp=1778102113` |

One direct check showed a resolved question where querying `bets(where: { question_: { id: ... } })` returned no rows. That means a `question` entity can be resolvable while the `bets` entity is not a complete discovery source for the benchmark's needs.

## Interpretation

Reasonable interpretations:

1. Not a hard subgraph outage: the endpoint is reachable and `question(id)` returns populated data, including resolutions.
2. Possibly a Polymarket subgraph entity-model/indexing limitation: the `bets` entity does not enumerate all resolved questions relevant to marketplace deliveries.
3. Application-level bug: `fetch_polymarket_resolved()` relies on `bets` as a proxy for resolved-market discovery even though the `questions` entity directly exposes `resolution`.

The third interpretation is the actionable one. Even if the underlying reason is a subgraph indexing limitation, the benchmark code should use the entity that contains the needed resolution data.

## Suggested Fix

Replace the Polymarket resolved-market discovery query with direct `questions` pagination.

Suggested GraphQL shape:

```graphql
{
  questions(
    first: %(first)s
    skip: %(skip)s
    orderBy: id
    orderDirection: asc
    where: {
      resolution_: {
        blockTimestamp_gt: %(resolved_after)s
      }
    }
  ) {
    id
    metadata {
      title
      outcomes
    }
    resolution {
      winningIndex
      blockTimestamp
    }
  }
}
```

Suggested implementation behavior:

1. Add `POLYMARKET_RESOLVED_QUESTIONS_QUERY`.
2. Change `fetch_polymarket_resolved(resolved_after)` to call `_paginated_fetch(..., "questions", {"resolved_after": resolved_after})`.
3. For each returned question:
   - require `question.id`
   - require non-empty `metadata.title`
   - require `resolution.blockTimestamp`
   - require `resolution.winningIndex`
4. Keep the current outcome mapping:
   - `winningIndex == 0` means `Yes`
   - `winningIndex == 1` means `No`
5. Add each market as:
   - `market_id = question["id"]`
   - `title = question["metadata"]["title"].strip()`
   - `data = {"outcome": winning_index == 0, "resolved_at_ts": int(blockTimestamp)}`
6. Keep `_match_delivery()` unchanged unless tests reveal a separate issue.

This fix avoids using recent bets as a proxy for recently resolved questions. It also removes the 30-day bet candidate window as a failure mode.

## Suggested Tests

Unit-level regression:

- Mock `_paginated_fetch` for Polymarket to return `questions`, not `bets`.
- Include a question with `resolution` and no corresponding bet fixture.
- Assert `fetch_polymarket_resolved()` returns that question in `markets.by_id`.

Matching-level regression:

- Build a delivery with `market_id` equal to the returned `question.id`.
- Assert `_match_delivery()` returns confidence `1.0`.

Integration-style diagnostic:

- Run `scripts/compare_polymarket_resolution_sources.py` against a state file containing pending Polymarket deliveries.
- Compare:
  - pending ids resolved by the current production function
  - pending ids resolved by direct `questions`
- Expected after the code fix: pending ids with non-null `question.resolution` should be discoverable through the production function.

Operational validation:

- Re-run `fetch_production` into temporary files.
- Expected change: Polymarket rows should increase materially from the reproduced value of `1`, while legitimately unresolved markets remain pending.

## Open Questions

1. Exact uploaded artifact row counts are not confirmed because artifact download requires GitHub authentication.
2. The May 6 report appearance cannot be reconstructed exactly without the May 6 `benchmark-data` artifact.
3. Some pending Polymarket markets are expected to remain pending because their direct `question.resolution` is null.
4. The direct `questions` query returns a much larger resolved universe. The fix should rely on `resolved_after` and existing deduplication to keep output controlled.
