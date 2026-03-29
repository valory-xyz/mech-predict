# Dependency & Testing Tooling Analysis

**Date:** 2026-03-30
**Status:** Analysis for Review
**Author:** Engineering
**Scope:** All Valory repos (14 downstream + 2 upstream)

---

## Executive Summary

All Valory repos use **Poetry + tox** together, with significant dependency duplication between `pyproject.toml` and `tox.ini`. This report analyses why both tools exist, whether one can be eliminated, and evaluates modern alternatives (`uv`, `nox`, `hatch`, `pdm`).

**Key finding:** `uv` is the strongest candidate to unify dependency management and replace Poetry. For test orchestration, either `tox-uv` (zero migration) or `uv` + `nox` (full migration) can replace the current tox setup with 5-10x faster CI times.

---

## 1. Current State: Why Both Poetry AND Tox?

### What each tool does

| Responsibility | Poetry | Tox |
|---|---|---|
| Declare project metadata | Yes (`[tool.poetry]`) | No |
| Manage dependencies | Yes (`poetry.lock`) | Yes (`[deps-packages]` in tox.ini) |
| Create virtual environments | Yes (`~/.cache/pypoetry/virtualenvs/`) | Yes (`.tox/` per environment) |
| Run tests | `poetry run pytest` (single env) | Yes (multi-env, multi-Python, multi-OS) |
| Run linters in isolation | No | Yes (each linter gets its own venv) |
| Test across Python versions | No | Yes (py3.10, 3.11, 3.12, 3.13, 3.14) |
| Publish to PyPI | Yes (`poetry publish`) | No |
| Lock dependencies | Yes (`poetry.lock`) | No (uses `[deps-packages]` as pseudo-lock) |

### Why dependencies are duplicated

Both `pyproject.toml` and `tox.ini [deps-packages]` list the same packages because:

1. **Different purposes**: `pyproject.toml` defines what gets published to PyPI. `tox.ini` defines what runs in CI.
2. **Different pinning strategies**: `pyproject.toml` uses ranges (`^2.1.1`) for compatibility. `tox.ini` pins strictly (`==2.3.3`) for reproducible CI.
3. **No integration**: Poetry doesn't manage tox. Tox doesn't read `pyproject.toml`. They're independent tools that happen to list overlapping deps.
4. **Historical**: tox.ini `[deps-packages]` predates poetry adoption. When poetry was added, the tox deps weren't removed because tox environments need their own isolated deps.

### The overlap problem

In `mech-predict`, for example:

| Package | pyproject.toml | tox.ini [deps-packages] |
|---|---|---|
| open-autonomy | ==0.21.16 | ==0.21.16 |
| requests | ==2.32.5 | ==2.32.5 |
| pytest | >=8.2,<10 | ==8.4.2 |
| pandas | ^2.1.1 | ==2.3.3 |

The versions often match but sometimes drift — tox pins more aggressively, creating a second pseudo-lockfile that must be kept in sync manually.

### What tox does that Poetry cannot

1. **Multi-environment isolation**: ~20 environments per repo (bandit, black, isort, flake8, mypy, pylint, darglint, check-hash, check-packages, check-dependencies, py3.10-linux, py3.11-linux, etc.)
2. **Per-env dependency control**: Linters only install linting deps (`skipsdist = True, skip_install = True`). Test envs install the full app.
3. **Multi-Python testing**: `py{3.10,3.11,3.12,3.13,3.14}-{win,linux,darwin}` matrix
4. **Conditional install**: `usedevelop = True` for some envs, `skip_install = True` for others

`poetry run pytest` is a single-env, single-Python, no-isolation approach — insufficient for CI.

---

## 2. Can Tox Be Removed?

**Not directly.** Tox provides environment isolation and multi-Python testing that Poetry cannot replicate. However, tox can be **replaced** by:

1. **`tox-uv` plugin** — keeps tox.ini, swaps pip/virtualenv for uv (zero migration, 5-10x faster)
2. **`nox` + `uv`** — Python-programmable test matrix runner with uv backend (full migration)
3. **Shell scripts / Makefile / `justfile`** with `uv run --python=3.X` calls (lightweight migration)

---

## 3. Can Poetry Be Replaced?

**Yes.** Poetry's `[tool.poetry]` is a non-standard format. The Python ecosystem is moving toward PEP 621 (`[project]`) as the standard. Poetry can be replaced by:

| Tool | PEP 621 | Lock file | Speed | Monorepo | Maturity |
|---|---|---|---|---|---|
| **uv** | Native | Yes (`uv.lock`) | 10-100x faster | Yes (workspaces) | Young but production-ready |
| **pdm** | Native | Yes (`pdm.lock`) | ~2x faster than pip | No | Mature |
| **hatch** | Native | No (uses pip) | Same as pip (can use uv) | No | Mature |
| **pip-tools** | N/A | Yes (`requirements.txt`) | Same as pip | No | Mature |

---

## 4. Tool-by-Tool Evaluation

### 4.1 uv (Astral) — Recommended

**What it is:** An extremely fast Python package and project manager written in Rust. Single binary that replaces pip, virtualenv, pyenv, poetry, and pip-tools.

**Key features:**
- `uv lock` — generates universal cross-platform lockfile (works across OSes and Python versions)
- `uv sync` — installs exactly what the lockfile specifies
- `uv run --python=3.X` — runs commands under a specific Python version, auto-downloading if needed
- `uv run --with 'dep==X.Y'` — test against different dep versions without changing lockfile
- PEP 621 native (`[project]` table, not `[tool.poetry]`)
- Workspaces for monorepo support
- Dependency groups (PEP 735) for dev/test/lint separation

**Performance:**
| Operation | Poetry/pip | uv | Speedup |
|---|---|---|---|
| Lock file generation | 10+ min | ~1 min | 10x |
| Install (cold) | 21s | 2.6s | 8x |
| Venv creation | ~1s | 12ms | 80x |
| Install (cached) | seconds | milliseconds | 10-100x |

**Dependency model:**
- Ranges in `[project.dependencies]` (flexible for library repos like OA/open-aea)
- Exact pins in `uv.lock` (reproducible for downstream repos)
- Resolution strategies: `highest` (default), `lowest`, `lowest-direct`
- Override mechanism for erroneous upstream constraints

**Migration path:**
1. **Phase 1 (1 hour):** Install `tox-uv` plugin — keeps existing tox.ini, uses uv for env creation. Immediate 5-10x speedup.
2. **Phase 2 (half day):** Convert `pyproject.toml` from `[tool.poetry]` to `[project]` using `uvx migrate-to-uv`.
3. **Phase 3 (1-2 days):** Replace tox.ini with `noxfile.py` or `justfile` using `uv run`.
4. **Phase 4 (cleanup):** Delete `poetry.lock`, `tox.ini`. CI uses `uv sync` + `uv run`.

**Production adoption:** Rippling (386 deps), LlamaIndex (600+ packages monorepo), Broad Institute.

### 4.2 nox — Best tox replacement

**What it is:** A Python-programmable test automation tool (like tox but with Python instead of INI).

**Key features:**
- `noxfile.py` defines sessions (equivalent to tox environments)
- Full Python logic for conditional deps, matrix testing, etc.
- First-class uv support: `session.install(..., uv=True)`
- Can parametrize across Python versions, deps, OSes

**Example replacing tox.ini:**
```python
import nox

@nox.session(python=["3.10", "3.11", "3.12"])
def tests(session):
    session.install(".", "pytest")
    session.run("pytest", "tests/")

@nox.session
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", ".")

@nox.session
def mypy(session):
    session.install(".", "mypy")
    session.run("mypy", "packages/")
```

**Advantage over tox:** Python logic instead of INI config. Can express complex conditions that tox.ini struggles with.

### 4.3 hatch — Alternative project manager

**What it is:** A Python project manager with built-in matrix environment support.

**Key features:**
- PEP 621 native
- Built-in environment matrices (like tox but in `pyproject.toml`)
- Can use uv as backend (`[tool.hatch.envs.default] installer = "uv"`)
- No separate lockfile (uses pip under the hood)

**Drawback:** No lockfile means less reproducibility. For the Valory repos that need strict pinning, this is a significant gap.

### 4.4 pdm — Standards-compliant alternative

**What it is:** A Python package manager that strictly follows PEP standards.

**Key features:**
- PEP 621 native
- Lock file (`pdm.lock`)
- Can use uv as resolver (`pdm config use_uv true`)
- Standards-compliant — no custom `[tool.X]` sections

**Drawback:** No workspace/monorepo support. Slower than uv. Less momentum.

### 4.5 tox-uv — Zero-migration option

**What it is:** A tox plugin that swaps pip and virtualenv for uv.

**Usage:**
```bash
pip install tox tox-uv
tox -e py3.10-linux  # Same tox.ini, uv creates venvs and installs deps
```

**Result:** One project saw non-test CI time drop from ~155s to ~16s. Zero changes to tox.ini required.

**Best for:** Immediate speedup without any migration risk.

---

## 5. Recommendation

### For Valory's dependency model

| Repo type | Current | Recommended |
|---|---|---|
| **OA / open-aea** (upstream libraries) | Poetry with ranges | uv with ranges in `[project.dependencies]` |
| **Downstream repos** (trader, optimus, etc.) | Poetry with strict pins + tox with same pins | uv with ranges in `[project.dependencies]`, strict pins in `uv.lock` |

**Key insight:** With uv, `[project.dependencies]` can use ranges (like OA wants) while `uv.lock` provides the exact reproducible pins (like downstream repos need). This eliminates the need for duplicating deps in tox.ini.

### Migration strategy

**Phase 1 — Immediate win (this week):**
- Install `tox-uv` plugin across all repos
- Zero changes to tox.ini or pyproject.toml
- CI gets 5-10x faster env creation

**Phase 2 — Convert Poetry to uv (per repo, 1 day each):**
- Run `uvx migrate-to-uv` to convert `[tool.poetry]` to `[project]`
- Generate `uv.lock`
- Delete `poetry.lock`
- Update CI to use `uv sync` instead of `poetry install`

**Phase 3 — Replace tox with nox or justfile (per repo, 1-2 days each):**
- Convert tox environments to nox sessions or justfile tasks
- Delete `tox.ini`
- Remove duplicate deps from tox (they now live only in `pyproject.toml` + `uv.lock`)

**Phase 4 — Remove dependency duplication:**
- `pyproject.toml` has ranges
- `uv.lock` has exact pins
- No more tox.ini `[deps-packages]`
- Single source of truth

### What this eliminates

| Problem | Current | After migration |
|---|---|---|
| Deps duplicated in pyproject.toml + tox.ini | Yes (manual sync) | No — `uv.lock` is the single lock |
| Slow CI env creation | 20-60s per env (pip) | 2-5s per env (uv) |
| Non-standard pyproject.toml format | `[tool.poetry]` | `[project]` (PEP 621) |
| Poetry resolver hangs on large deps | Yes (known issue) | No — PubGrub resolver in Rust |
| Pipenv holdouts (mech-interact) | Yes (now migrated to poetry) | Would use uv directly |

---

## 6. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| uv is young (v0.x) | Low — backed by Astral, production-proven at Rippling/LlamaIndex scale | Start with `tox-uv` plugin (zero risk), convert gradually |
| `uv.lock` format changes | Low — format is stabilizing | Lock to specific uv version in CI |
| nox learning curve | Low — Python is more familiar than tox INI | Port one repo first as proof of concept |
| tomte (Valory's linter wrapper) depends on tox | Medium | `tox-uv` keeps tox compatibility; or update tomte to support uv |
| aea CLI tools assume pip/poetry | Medium | Test thoroughly before removing poetry |

---

## Appendix: Current tox environments per repo

Typical tox.ini has ~20 environments:

```
Security:     bandit, safety
Code style:   black, black-check, isort, isort-check, flake8
Type check:   mypy
Linting:      pylint, darglint
Autonomy:     check-hash, check-packages, check-dependencies,
              check-generate-all-protocols, check-abciapp-specs,
              check-abci-docstrings, check-handlers
Testing:      py{3.10,3.11,3.12,3.13,3.14}-{win,linux,darwin}
```

Each creates an isolated venv. With `tox-uv`, each venv is created 80x faster.
