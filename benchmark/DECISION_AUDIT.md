# Reproducing a benchmark deployment decision

The daily workflow captures the deployment roster once, computes a shared
per-platform decision record, and uploads the evidence before posting to Slack.
Slack and the Markdown **Deployment decision** section use that same record.
The existing promotion and demotion thresholds are unchanged.

## What to review

1. Check **roster coverage**. Every distinct deployed manifest tool is accounted
   for, including tools absent from the execution registry. A usable prediction
   establishes forecasting capability. Registry membership also identifies a
   forecaster that has no scores. The explicit component-role mapping in
   `tool_usage.NON_PREDICTION_TOOLS` identifies `propose-question` (question
   creation) and `resolve-market-jury-v1` (resolution). Their source paths and
   classification are frozen in the deployment snapshot; they remain visible
   as confirmed non-prediction manifest members, outside forecasting counts.
   Otherwise the tool is explicitly unclassified; a zero score count is never
   proof that it is a non-prediction tool.
2. Read the actual trigger. A sustained-baseline demotion shows the Brier,
   baseline, difference, and scored count in both production windows. Conditional
   accuracy shows its disagreement count; the Edge lower bound and priced count
   explain the other prerequisite. Counts represent predictions, not independent
   markets. The 90d and recent windows overlap.
3. Distinguish **remaining deployed** from **assessed and retained**. An unassessed
   tool remains deployed unless a separate action removes it, but it cannot
   justify removing every assessed forecaster. Confirmed non-prediction tools are
   accounted for separately. Roster lookup failure prevents deployment proposals.
   Tools whose demotion is blocked are counted as **flagged but retained**.
4. Review each tool's evidence in `report.md`, including unassessed tools and
   tournament candidates. Classification and alias failures are explicit.

For example, a synthetic roster containing seven forecasters, four proposed
removals, and three assessed keep verdicts must say **three remain deployed**,
regardless of whether only six tools have execution-registry entries. This is a
coverage regression fixture, not a re-evaluation of the September 29 production
recommendations: the previously omitted tool must itself pass through the gate.

## Download and replay

Download the `benchmark-decision-audit-<attempt>` artifact linked from Slack.
Each platform directory contains:

| File | Purpose |
|---|---|
| `manifest.json` | Run/attempt, source commit, configuration, registry snapshot, file hashes, input acceptance status |
| `deployment_snapshot.json` | Captured release reference, deployment service URL, mech addresses, manifest identifiers, tool names, lookup failures |
| `trailing_scores_<platform>.json` | Exact trailing 90d input |
| `rolling_scores_<platform>.json` | Exact current rolling input |
| `prev_rolling_scores_<platform>.json` | Exact preceding rolling input |
| `scores_tournament_<platform>.json` | Exact cumulative candidate input |
| `analysis_scores_<platform>.json` | Actual main-window input used by the analyzer; MTD on the HTTP path |
| `decision.json` | Shared verdicts, thresholds, evidence, roster counts, query/observed window metadata |
| `source_report.md`, `report.md` | Original analysis and analysis with the complete decision evidence appended |
| `summary.txt`, `slack_payload.json` | Saved narrative and the prepared Slack content |

Missing inputs have an explicit status and no file. Stale or malformed inputs
are retained for diagnosis but rejected from decision calculation and category
rendering. Requested bounds describe the actual query, with an inclusive start
and exclusive end; observed bounds describe the first/last scored prediction.
The tournament's cumulative pool has observed bounds rather than a rolling query.

Use the source commit recorded in `manifest.json`, then run from that checkout:

```sh
python -m benchmark.report_audit replay --bundle /path/to/download/omen
python -m benchmark.report_audit replay --bundle /path/to/download/polymarket
```

Replay verifies every retained file's hash and recomputes the decision, roster
counts, Markdown decision section, and Slack content. It makes no network or LLM
calls. A different policy/source revision can legitimately fail replay. The
artifact reconstructs decisions from scored aggregates; it does not re-score raw
predictions or recover historical upstream API state. Older report artifacts
without these inputs cannot provide this guarantee.

## Workflow and local dry runs

The workflow order is **restore state → initialize audit → score → analyze →
prepare → upload → post**. Each run/attempt gets a new directory. Initialization
removes only derived rolling/trailing outputs and reports, retaining cumulative
resume state. Both analyzers receive the same `--deployment-snapshot` file.
Preparation and posting are independent per platform.

For a local capture, initialize before generating the scores:

```sh
python -m benchmark.report_audit init --output /tmp/report-audit-example
# Run the scorer and analyzers; pass this snapshot to each analyzer:
# --deployment-snapshot /tmp/report-audit-example/deployment_snapshot.json
python -m benchmark.report_audit prepare --output /tmp/report-audit-example --platform omen
python -m benchmark.report_audit replay --bundle /tmp/report-audit-example/omen
python -m benchmark.report_audit post --bundle /tmp/report-audit-example/omen --dry-run
```

`--results` selects an alternate scorer output directory for `init` and `prepare`.
`init --snapshot FILE` accepts a frozen deployment snapshot for offline fixtures.
`prepare --summary FILE` accepts a saved narrative and skips LLM generation.
Absent a usable LLM summary, preparation preserves the computed evidence with a
summary-unavailable notice. Secrets are not written to the bundle.

`post` first verifies replay, then adds the uploaded artifact URL to the saved
payload. It neither refreshes data nor recomputes a new narrative. The workflow
requires successful artifact upload before posting. `--dry-run` prints the
message without sending it; `notify_slack=false` still disables workflow posts.

## Failure behavior

- Missing, malformed, or stale inputs only disable dependent evaluations:
  90d scores are required for production verdicts; Current 7d scores are required
  for the sustained-baseline gate; tournament scores gate candidates; Prev 7d
  scores only affect comparisons. Rejected candidate inputs mean **candidate
  evaluation unavailable**, not that no candidate qualifies. A failed weekly
  input cannot turn an unevaluated baseline gate into an assessed survivor;
  a valid empty or thin week retains the existing policy behavior.
- A rejected analysis/comparison input suppresses the saved trend narrative in
  Slack, while preserving it in the artifact. Production evidence remains usable.
- Ambiguous aliases are excluded only from the affected tool/window evaluation.
  Aggregates are never merged or picked arbitrarily; original inputs remain saved.
- Partial deployment lookup: **DECISION UNAVAILABLE**; no claim that a proposed
  removal leaves adequate coverage. Available findings remain in the full report.
- A deployed tool with no usable scores: explicitly unassessed, retained in the
  roster and remaining-deployed count.
- Every assessed forecaster fails: preserve the existing replace-first/platform
  escalation rule; unassessed tools do not satisfy the survivor guard.
- Upload or replay failure: the prepared content is not posted as a verified
  decision. Normal workflow failure notifications remain in place.

Conditional-accuracy confidence tests, ROI-based policy changes, and category
selection changes require separate policy review. This change makes the existing
rules inspectable and reproducible.
