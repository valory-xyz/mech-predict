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

"""Unit tests for superforcaster-polymarket-v4's output-contract JSON extraction."""

import json
from typing import Optional

import pytest

from packages.valory.customs.superforcaster_polymarket_v4.superforcaster_polymarket_v4 import (
    extract_result_json,
)

NULL_PREDICTION = {"p_yes": 0.5, "p_no": 0.5, "confidence": 0.0, "info_utility": 0.0}


class TestExtractResultJson:
    """extract_result_json must always return a flat json.loads-parseable string."""

    def test_prose_prefixed_output_is_recovered(self) -> None:
        """v4's prose+trailing-JSON output (which the trader's json.loads rejects) is recovered."""
        completion = (
            "<facts>Fed cut priced in; polls stable.</facts>\n"
            "<thinking>Base rate favors NO.</thinking>\n"
            "<answer>0.32</answer>\n"
            '{"p_yes": 0.32, "p_no": 0.68, "confidence": 0.7, "info_utility": 0.5}'
        )
        # The raw completion is exactly what broke production: JSONDecodeError at char 0.
        with pytest.raises(json.JSONDecodeError):
            json.loads(completion)
        out = extract_result_json(completion)
        assert json.loads(out) == {
            "p_yes": 0.32,
            "p_no": 0.68,
            "confidence": 0.7,
            "info_utility": 0.5,
        }

    def test_bare_json_value_is_preserved(self) -> None:
        """A clean bare-JSON completion round-trips unchanged in value."""
        completion = (
            '{"p_yes": 0.9, "p_no": 0.1, "confidence": 0.8, "info_utility": 0.2}'
        )
        assert json.loads(extract_result_json(completion)) == json.loads(completion)

    def test_trailing_prediction_wins_over_embedded_json(self) -> None:
        """When the prose contains earlier JSON, the trailing prediction object is chosen."""
        completion = (
            'context {"x": 1}\n<answer>0.4</answer>\n'
            '{"p_yes": 0.4, "p_no": 0.6, "confidence": 0.5, "info_utility": 0.3}'
        )
        assert json.loads(extract_result_json(completion))["p_yes"] == 0.4

    @pytest.mark.parametrize(
        "bad", [None, "", "no json here", "prose\n{p_yes: broken}"]
    )
    def test_unparseable_falls_back_to_null_prediction(
        self, bad: Optional[str]
    ) -> None:
        """None/empty/no-brace/malformed all yield a parseable null-prediction, never raw prose."""
        assert json.loads(extract_result_json(bad)) == NULL_PREDICTION
