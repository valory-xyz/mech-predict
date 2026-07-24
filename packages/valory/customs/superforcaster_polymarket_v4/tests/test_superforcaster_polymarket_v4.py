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

"""Unit tests for superforcaster-polymarket-v4's structured-output contract."""

import inspect
import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

from packages.valory.customs.superforcaster_polymarket_v4.superforcaster_polymarket_v4 import (
    PredictionResult,
    _parse_completion,
    run,
)

V4_MODULE = (
    "packages.valory.customs.superforcaster_polymarket_v4."
    "superforcaster_polymarket_v4"
)

FAKE_SERPER_RESPONSE = {
    "organic": [{"title": "T", "link": "https://example.test", "snippet": "S"}],
    "peopleAlsoAsk": [{"question": "Q?", "snippet": "A."}],
}

FAKE_PREDICTION = PredictionResult(
    facts="Fact 1. Fact 2.",
    reasons_no="No 1 (strength 6).",
    reasons_yes="Yes 1 (strength 5).",
    evidence_reliability_screen=(
        "(a) no market odds. (b) no intent language. (c) 1 TYPE A, 0 TYPE B. "
        "(d) criterion directly confirmed."
    ),
    aggregation="Base rate 0.5. Tentative: 0.32.",
    reflection="Passes the sanity checks.",
    p_yes=0.32,
    p_no=0.68,
    confidence=0.7,
    info_utility=0.4,
)

PROMPT = "Will X happen? p_yes and p_no?"


def _make_mock_api_keys() -> MagicMock:
    """Create a mock KeyChain-like api_keys object."""
    services = {"openai": ["sk-test"], "serperapi": ["serper-test"]}
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key][0]
    mock.get = lambda key, default="": services.get(key, [default])[0]
    return mock


def _mock_parse_response() -> MagicMock:
    """Fake response matching the beta.chat.completions.parse shape."""
    return MagicMock(
        choices=[MagicMock(message=MagicMock(parsed=FAKE_PREDICTION, refusal=None))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )


class TestStructuredOutputContract:
    """v4 uses OpenAI Structured Outputs; the on-chain result is always clean JSON."""

    @patch(f"{V4_MODULE}.OpenAIClientManager")
    @patch(f"{V4_MODULE}.fetch_additional_sources")
    def test_uses_structured_parse_not_raw_create(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """run() calls beta.chat.completions.parse with response_format=PredictionResult."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_client = MagicMock()
        mock_client.client.beta.chat.completions.parse.return_value = (
            _mock_parse_response()
        )
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        run(
            tool="superforcaster-polymarket-v4",
            model="gpt-4.1-2025-04-14",
            prompt=PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        parse = mock_client.client.beta.chat.completions.parse
        parse.assert_called_once()
        assert parse.call_args.kwargs["response_format"] is PredictionResult
        # The raw (prose-leaking) completion path must NOT be used.
        mock_client.client.chat.completions.create.assert_not_called()

    @patch(f"{V4_MODULE}.OpenAIClientManager")
    @patch(f"{V4_MODULE}.fetch_additional_sources")
    def test_on_chain_result_is_flat_json_loads_parseable(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """The on-chain result flat-json.loads-parses to exactly the four mech fields."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_client = MagicMock()
        mock_client.client.beta.chat.completions.parse.return_value = (
            _mock_parse_response()
        )
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster-polymarket-v4",
            model="gpt-4.1-2025-04-14",
            prompt=PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        on_chain = result[0]
        # This is exactly the trader's consumer path -- it must NOT raise.
        parsed = json.loads(on_chain)
        assert on_chain.startswith("{")
        assert set(parsed.keys()) == {"p_yes", "p_no", "confidence", "info_utility"}
        # No reasoning field leaks on-chain.
        assert "facts" not in parsed and "evidence_reliability_screen" not in parsed
        assert parsed["p_yes"] == 0.32 and parsed["p_no"] == 0.68


class TestPredictionResultSchema:
    """The Pydantic schema enforces the mech numeric contract."""

    def test_valid_prediction_carries_the_numbers(self) -> None:
        """A well-formed prediction constructs and carries the four numbers."""
        assert FAKE_PREDICTION.p_yes == 0.32
        assert FAKE_PREDICTION.p_no == 0.68

    def test_validator_rejects_mismatched_sum(self) -> None:
        """p_yes + p_no must equal 1 (guards against an inconsistent forecast)."""
        with pytest.raises(ValidationError):
            PredictionResult(
                facts="f",
                reasons_no="n",
                reasons_yes="y",
                evidence_reliability_screen="s",
                aggregation="a",
                reflection="r",
                p_yes=0.7,
                p_no=0.7,
                confidence=0.5,
                info_utility=0.5,
            )

    def test_parse_completion_takes_client_and_schema(self) -> None:
        """_parse_completion takes the client and a response_format schema."""
        params = list(inspect.signature(_parse_completion).parameters)
        assert "client" in params and "response_format" in params

    def test_numeric_fields_are_declared_last(self) -> None:
        """The four numeric fields stay last in the schema.

        Structured outputs emit fields in declaration order, so keeping the
        numbers after the reasoning chain is what preserves v4's calibration.
        A future reorder that breaks this must fail here.
        """
        assert list(PredictionResult.model_fields)[-4:] == [
            "p_yes",
            "p_no",
            "confidence",
            "info_utility",
        ]


class TestFailurePathContract:
    """Any failure still yields a flat-json.loads-parseable null prediction."""

    @patch(f"{V4_MODULE}._parse_completion", side_effect=RuntimeError("boom"))
    @patch(f"{V4_MODULE}.OpenAIClientManager")
    @patch(f"{V4_MODULE}.fetch_additional_sources")
    def test_run_failure_returns_parseable_null_prediction(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_parse: MagicMock,
    ) -> None:
        """A raising run() still yields flat-json.loads null JSON, p_yes None."""
        mock_fetch.return_value = MagicMock(json=lambda: FAKE_SERPER_RESPONSE)
        mock_client = MagicMock()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster-polymarket-v4",
            model="gpt-4.1-2025-04-14",
            prompt=PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )

        on_chain = result[0]
        parsed = json.loads(on_chain)  # trader consumer path -- must NOT raise
        assert on_chain.startswith("{")
        assert parsed["p_yes"] is None and parsed["p_no"] is None
        assert parsed["confidence"] == 0.0 and parsed["info_utility"] == 0.0
        # error + error_type let ops tell a systemic failure from a one-off.
        assert parsed["error"] and parsed["error_type"] == "RuntimeError"

    @patch(f"{V4_MODULE}.OpenAIClientManager")
    @patch(f"{V4_MODULE}.fetch_additional_sources")
    def test_serper_http_error_surfaces_as_null_prediction(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A Serper 4xx/5xx (raise_for_status) yields the null-prediction JSON."""
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError("402 no credits")
        mock_fetch.return_value = resp
        mock_client = MagicMock()
        mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)

        result = run(
            tool="superforcaster-polymarket-v4",
            model="gpt-4.1-2025-04-14",
            prompt=PROMPT,
            api_keys=_make_mock_api_keys(),
            counter_callback=None,
        )
        parsed = json.loads(result[0])
        assert parsed["p_yes"] is None
        assert parsed["error_type"] == "HTTPError"


class TestParseCompletion:
    """_parse_completion's own retry / refusal / success behaviour."""

    def test_returns_parsed_on_success(self) -> None:
        """A successful parse returns the model instance (schema is honored)."""
        client = MagicMock()
        client.beta.chat.completions.parse.return_value = _mock_parse_response()
        parsed, _ = _parse_completion(
            client=client,
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": "x"}],
            response_format=PredictionResult,
            counter_callback=None,
        )
        assert parsed is FAKE_PREDICTION
        assert (
            client.beta.chat.completions.parse.call_args.kwargs["response_format"]
            is PredictionResult
        )

    def test_refusal_retries_then_raises_chained_runtimeerror(self) -> None:
        """A refusal (parsed=None) retries and raises RuntimeError with the cause."""
        client = MagicMock()
        client.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(parsed=None, refusal="policy"))]
        )
        with pytest.raises(RuntimeError, match="after 2 attempts"):
            _parse_completion(
                client=client,
                model="gpt-4.1-2025-04-14",
                messages=[{"role": "user", "content": "x"}],
                response_format=PredictionResult,
                retries=2,
                delay=0,
            )
        assert client.beta.chat.completions.parse.call_count == 2
