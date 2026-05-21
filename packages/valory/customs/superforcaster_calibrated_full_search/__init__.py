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

"""Superforcaster Calibrated Full Search.

Sibling of the calibration-ON superforcaster build that is live on Omen
production (Structured Outputs via Pydantic, max_tokens=4096, CALIBRATION
block + EVIDENCE BAR / CONFIDENCE COUPLING / NUMERIC QUESTIONS checks in
the prompt). Augments evidence-gathering: after the Serper search, the
top organic results are scraped, the main article text is extracted via
readability + markdownify, and the cleaned page body is rendered into
the forecasting prompt alongside the Serper snippet. The prediction
architecture and prompt are otherwise unchanged from the v0.18.1 build.

The non-calibrated sibling (`superforcaster_full_search`) mirrors the
v0.16.5 reverted build that is live on Polymarket; the two are intended
to be scored in tournament mode against their respective live incumbents.
"""
