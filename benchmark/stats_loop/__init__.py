# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""Stats-loop: regression-triage and tool-improvement issue dispatcher.

Companion to the daily benchmark flywheel. Reads the per-platform
rolling-window scores produced by ``benchmark.scorer`` + ``benchmark.analyze``
and applies a deterministic gate cascade. When a tool meets all gates on a
Polymarket regression for two consecutive daily runs, opens a GitHub issue
labelled ``tool-improvement`` on this repo. The label routes the issue to
``tool-improvement-agent`` in the agent-skills monorepo, which is responsible
for investigating the regression and (when warranted) proposing a draft PR.

Module layout:

- ``triage``: pure gate logic over two rolling-window score files.
  Produces a ``TriageDecision`` per tool. No LLM, no network.
- ``open_issue``: given a triage decision and the prior-day state, builds the
  issue body and shells out to ``gh issue create``. Updates the cross-day
  state file.

There is no LLM call in this module. The opened issue is data only; the
investigating agent (tool-improvement-agent on the agent-skills VPS stack)
does all the analysis after the GitHub webhook fires.
"""
