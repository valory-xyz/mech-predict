# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2026 Valory AG
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

"""Superforcaster Full Search tool served by Olas-Predict-R1-14B.

A sibling of superforcaster_full_search that keeps its full-page search
evidence pipeline and forecasting prompt and swaps the forecaster for
Olas-Predict-R1-14B, a fine-tuned DeepSeek-R1-Distill-Qwen-14B served from a
self-hosted, OpenAI-compatible vLLM endpoint.
"""
