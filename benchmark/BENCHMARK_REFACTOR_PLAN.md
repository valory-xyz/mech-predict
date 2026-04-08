# Benchmark Refactoring Plan

## Context

The benchmark scripts (`runner.py`, `tournament.py`, `score_tournament.py`, `fetch_production.py`, `prompt_replay.py`, etc.) have grown organically and now contain significant code duplication. `tournament.py` even has a comment on line 126: *"duplicated from runner.py -- will be extracted later"*. Tests live at `tests/benchmark/` but should be colocated with the benchmark package.

### Recent additions (since initial plan)

- **`benchmark/analyze.py`** (513 lines) — Generate markdown reports from `scores.json` + `scores_history.jsonl`
- **`benchmark/compare.py`** (294 lines) — Delta tables comparing two `scores.json` files
- **`benchmark/prompt_replay.py`** (910 lines) — Prompt-only replay for A/B testing prompts (supports OpenAI + Claude)
- **`benchmark/datasets/fetch_replay.py`** (668 lines) — Fetch replay-ready datasets from on-chain subgraphs
- **`benchmark/scorer.py`** now has incremental scoring (`update()` method, monthly snapshots, `--rebuild` CLI)
- **`benchmark/datasets/fetch_production.py`** now uses daily log rotation (`logs/production_log_{YYYY-MM-DD}.jsonl`), IPFS metadata enrichment, and inline scorer integration
- **New tests**: `test_analyze.py`, `test_fetch_production.py`, `test_score_tournament.py`, `test_tournament.py`, `test_fetch_open.py`

## Commit 1: `benchmark/tools.py` — shared tool infrastructure

**New file:** `benchmark/tools.py`

Extract these identical blocks from `runner.py` + `tournament.py`:

| What | runner.py lines | tournament.py lines |
|------|----------------|---------------------|
| `ToolSpec` dataclass | 60-64 | 130-134 |
| `TOOL_REGISTRY` dict (13 tools) | 67-113 | 137-177 |
| `build_keychain()` | 121-142 | 185-204 |
| `load_tool_run()` + `_tool_cache` | 152-166 | 214-228 |
| Timeout utils (`_ToolTimeout`, `_alarm_handler`, `_can_use_sigalrm`) | 173-186 | 235-248 |

`build_keychain` gets a `return_source_content: bool = False` param (the only diff between the two copies).

**Update imports in:**
- `runner.py` — remove extracted code, import from `benchmark.tools`
- `tournament.py` — remove extracted code, import from `benchmark.tools`
- `sweep.py` — change `from benchmark.runner import TOOL_REGISTRY` to `from benchmark.tools import TOOL_REGISTRY`
- `notify_slack.py` — replace hardcoded `OUR_TOOLS` set with `OUR_TOOLS = set(TOOL_REGISTRY)`

**Add TODO comments to `_make_row_id` in:**
- `runner.py:194` — `# TODO: unify _make_row_id across runner, tournament, prompt_replay & fetch_production into benchmark/tools.py`
- `tournament.py:256` — same TODO
- `prompt_replay.py:561` — same TODO (4th copy, different signature: `prefix, tool_name, question_text, model`)
- `datasets/fetch_production.py:1333` — same TODO (different signature: `platform, deliver_id`)

## Commit 2: `benchmark/io.py` — shared JSONL I/O

**New file:** `benchmark/io.py`

Extract:

| Function | Duplicated in |
|----------|---------------|
| `load_jsonl(path) -> list[dict]` | `scorer.py:41` (`load_rows`), `score_tournament.py:276` (`load_predictions`), `analyze.py:43` (`load_history`), inline in `runner.py`, `tournament.py`, `prompt_replay.py` (3 loops: lines 278, 605, 677), `fetch_replay.py:522` |
| `load_existing_ids(path, key="row_id") -> set[str]` | `runner.py`, `tournament.py`, `score_tournament.py:287`, `fetch_open.py:413` (`key="market_id"`), `fetch_production.py:1483` (specialized daily-log variant) |
| `append_jsonl(path, rows) -> int` | `fetch_open.py:428`, inline in `tournament.py`, `fetch_production.py` |
| `write_jsonl(path, rows)` (full-write, not append) | `fetch_replay.py:430` (`_write_jsonl`) |

**Update imports in:**
- `scorer.py` — remove `load_rows`, use `from benchmark.io import load_jsonl as load_rows`
- `score_tournament.py` — remove `load_predictions` and `load_existing_row_ids`
- `runner.py` — remove `load_existing_row_ids`
- `tournament.py` — remove `load_existing_row_ids`
- `fetch_open.py` — remove `load_existing_ids` and `append_jsonl`
- `fetch_production.py` — remove `load_existing_row_ids`
- `analyze.py` — remove `load_history`, use `from benchmark.io import load_jsonl as load_history`
- `prompt_replay.py` — replace 3 inline JSONL read loops with `load_jsonl`
- `datasets/fetch_replay.py` — remove `_write_jsonl`, use `from benchmark.io import write_jsonl`; replace inline read loop with `load_jsonl`

## Commit 3: Move tests + update tox

- Move `tests/benchmark/*.py` → `benchmark/tests/`
- Create `benchmark/tests/__init__.py`
- Remove empty `tests/benchmark/` directory
- Update `tox.ini`:
  - Line 285: `pytest tests/benchmark` → `pytest benchmark/tests`
  - Line 291: `pytest tests --ignore=tests/benchmark` → `pytest tests`
- Update test imports for symbols that moved (e.g. `TOOL_REGISTRY` now from `benchmark.tools`)

## CI impact

- `.github/workflows/common_checks.yaml` — runs `tox -e py*-linux` etc., which delegate to `[testenv:unit-tests]`. **No changes needed** — only `tox.ini` line 285 needs updating.
- `.github/workflows/benchmark_flywheel.yaml` — runs benchmark scripts directly (`fetch_production.py`, `analyze.py`, `notify_slack.py`, `fetch_open.py`, `tournament.py`, `score_tournament.py`), no test paths referenced. **No changes needed** — import paths stay the same since we're adding `benchmark/tools.py` and `benchmark/io.py` as new modules, not moving existing ones.
- All platform tox envs (`py3.10-linux`, `py3.12-darwin`, etc.) inherit from `[testenv:unit-tests]` commands — updating line 285 covers everything.

## Files to modify

| File | Commit |
|------|--------|
| `benchmark/tools.py` (new) | 1 |
| `benchmark/io.py` (new) | 2 |
| `benchmark/runner.py` | 1, 2 |
| `benchmark/tournament.py` | 1, 2 |
| `benchmark/prompt_replay.py` | 1 (TODO for `_make_row_id`), 2 |
| `benchmark/analyze.py` | 2 |
| `benchmark/datasets/fetch_replay.py` | 2 |
| `benchmark/notify_slack.py` | 1 |
| `benchmark/sweep.py` | 1 |
| `benchmark/scorer.py` | 2 |
| `benchmark/score_tournament.py` | 2 |
| `benchmark/datasets/fetch_production.py` | 1 (TODO), 2 |
| `benchmark/datasets/fetch_open.py` | 2 |
| `benchmark/tests/__init__.py` (new) | 3 |
| `tests/benchmark/*` (move) | 3 |
| `tox.ini` | 3 |

### Files not touched by this refactor (no duplication)

| File | Notes |
|------|-------|
| `benchmark/compare.py` | Self-contained delta logic, no JSONL I/O |

## Verification

After each commit, run the full benchmark test suite to catch import breakage:
```bash
pytest tests/benchmark -v    # commits 1-2
pytest benchmark/tests -v    # commit 3 (after move)
```

Also verify the scripts still run end-to-end:
```bash
# notify_slack (dry-run, no Slack post)
set -a && source .env && set +a && python benchmark/notify_slack.py --report "benchmark-results (7)/report.md" --dry-run

# scorer (smoke test — just check it imports and parses args)
python -m benchmark.scorer --help

# runner
python -m benchmark.runner --help

# tournament
python -m benchmark.tournament --help

# score_tournament
python -m benchmark.score_tournament --help

# sweep
python -m benchmark.sweep --help

# analyze
python -m benchmark.analyze --help

# compare
python -m benchmark.compare --help

# prompt_replay
python -m benchmark.prompt_replay --help

# fetch_replay
python -m benchmark.datasets.fetch_replay --help
```

After commit 3, also verify tox still finds the tests:
```bash
tox -e unit-tests -- benchmark/tests -v --no-header
```
