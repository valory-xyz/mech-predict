# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""Superforcaster Full Search tool.

A sibling of the original superforcaster that augments evidence by fetching
the top search-result pages, extracting the main article text via
readability + markdownify, and feeding the cleaned page body into the
forecasting prompt alongside the Serper snippet. The prompt and prediction
architecture are unchanged from superforcaster.
"""
