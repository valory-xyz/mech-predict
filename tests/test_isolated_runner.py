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

"""Unit tests for how the isolated runner judges a tool's delivery."""

import json

from tests.isolated_runner import validate_response

FORECAST = {"p_yes": 0.3, "p_no": 0.7, "confidence": 0.6, "info_utility": 0.5}


def _response(result: str) -> tuple:
    """Build a well-typed 5-tuple response carrying ``result`` as the delivery.

    :param result: the delivered result string.
    :return: the response tuple a tool's ``run()`` returns.
    """
    return (result, "prompt", None, None, None)


def test_a_bare_json_forecast_passes() -> None:
    """A forecast object is what a strict consumer can use."""
    assert validate_response(_response(json.dumps(FORECAST))) == []


def test_a_scaffold_that_only_contains_the_field_names_fails() -> None:
    """The substring check this replaced passed on exactly this delivery."""
    scaffold = (
        "<facts>\n- the committee meets in December\n</facts>\n"
        "<answer>*0.3*</answer>\np_yes p_no confidence info_utility"
    )
    errors = validate_response(_response(scaffold))
    assert len(errors) == 1
    assert "not JSON" in errors[0]


def test_a_typed_error_null_fails_and_surfaces_the_tool_error() -> None:
    """A null p_yes carries every field name, so the old check passed it too."""
    null = json.dumps(
        {
            "p_yes": None,
            "p_no": None,
            "confidence": 0.0,
            "info_utility": 0.0,
            "error": "Response truncated (finish_reason='length', max_tokens=4096)",
            "error_type": "TruncatedCompletionError",
        }
    )
    errors = validate_response(_response(null))
    assert [e.split("'")[1] for e in errors] == ["p_yes", "p_no"]
    assert all("not a probability" in error for error in errors)
    assert all("Response truncated" in error for error in errors)


def test_an_out_of_range_p_yes_fails() -> None:
    """A p_yes outside [0, 1] is not a probability."""
    errors = validate_response(_response(json.dumps(dict(FORECAST, p_yes=1.7))))
    assert len(errors) == 1
    assert "not a probability" in errors[0]


def test_a_boolean_p_yes_fails() -> None:
    """True is an int in Python, but it is not a forecast."""
    errors = validate_response(_response(json.dumps(dict(FORECAST, p_yes=True))))
    assert len(errors) == 1
    assert "not a probability" in errors[0]


def test_a_missing_field_fails() -> None:
    """Every prediction field must be present in the parsed object."""
    partial = {key: value for key, value in FORECAST.items() if key != "info_utility"}
    errors = validate_response(_response(json.dumps(partial)))
    assert errors == ["Missing 'info_utility' in delivered message."]


def test_a_json_array_fails() -> None:
    """Valid JSON that is not an object is not a forecast."""
    errors = validate_response(_response(json.dumps([FORECAST])))
    assert len(errors) == 1
    assert "not a JSON object" in errors[0]


def test_non_prediction_tools_skip_the_payload_check() -> None:
    """An image tool's delivery is not JSON, and validate_prediction=False allows it."""
    delivery = _response("https://example.com/cityscape.png")
    assert validate_response(delivery, validate_prediction=False) == []


def test_the_range_bounds_are_inclusive() -> None:
    """0.0 and 1.0 are probabilities: a tool certain either way must pass."""
    certain_no = dict(FORECAST, p_yes=0.0, p_no=1.0, confidence=1.0, info_utility=0.0)
    assert validate_response(_response(json.dumps(certain_no))) == []


def test_integer_probabilities_pass() -> None:
    """JSON `0` and `1` decode as int, which is as usable as 0.0 and 1.0."""
    integral = {"p_yes": 0, "p_no": 1, "confidence": 1, "info_utility": 0}
    assert validate_response(_response(json.dumps(integral))) == []


def test_a_non_numeric_field_other_than_p_yes_fails() -> None:
    """Every prediction field is checked, not only p_yes."""
    errors = validate_response(_response(json.dumps(dict(FORECAST, confidence="n/a"))))
    assert len(errors) == 1
    assert "'confidence' is not a probability" in errors[0]


def test_a_null_field_other_than_p_yes_fails() -> None:
    """A null info_utility is not a number, so it cannot be scored."""
    errors = validate_response(
        _response(json.dumps(dict(FORECAST, info_utility=None)))
    )
    assert len(errors) == 1
    assert "'info_utility' is not a probability" in errors[0]
