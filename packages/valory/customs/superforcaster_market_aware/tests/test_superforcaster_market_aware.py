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

"""Unit tests for superforcaster_market_aware: page scrape, capture/replay, fallbacks."""

import inspect
import json
import re
from pathlib import Path
from typing import Any, Optional, get_args
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

import packages.valory.customs.superforcaster_market_aware.superforcaster_market_aware as module
from packages.valory.customs.superforcaster_market_aware.superforcaster_market_aware import (
    OpenAIClientManager,
    PredictionResult,
    _parse_completion,
    fetch_additional_sources,
    run,
)


class TestOpenAIClientManager:
    """Verify OpenAIClientManager creates per-context clients without globals."""

    def test_context_manager_returns_client_instance(self) -> None:
        """__enter__ returns a fresh OpenAI client, __exit__ closes it."""
        mgr = OpenAIClientManager(api_key="sk-test")
        with patch(
            "packages.valory.customs.superforcaster_market_aware.superforcaster_market_aware.OpenAI"
        ) as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance

            with mgr as client:
                assert client is mock_instance
                MockClient.assert_called_once_with(api_key="sk-test")

            mock_instance.close.assert_called_once()

    def test_no_global_client_variable(self) -> None:
        """The module must not define a module-level 'client' variable."""
        source = Path(module.__file__).read_text(encoding="utf-8")
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("client:") or stripped.startswith("client ="):
                if not line.startswith(" ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level 'client' variable found at line {i}: {line}"
                    )

    def test_parse_completion_requires_client_param(self) -> None:
        """_parse_completion requires client as its first param."""
        params = list(inspect.signature(_parse_completion).parameters)
        assert params[0] == "client"


SF_MODULE = (
    "packages.valory.customs.superforcaster_market_aware.superforcaster_market_aware"
)

FAKE_SERPER_RESPONSE = {
    "searchParameters": {"q": "test query", "type": "search"},
    "organic": [
        {
            "title": "Test Result",
            "link": "http://example.com/result",
            "snippet": "Test snippet content",
            "position": 1,
        },
        {
            "title": "Second Result",
            "link": "http://example.com/second",
            "snippet": "Second snippet",
            "position": 2,
        },
    ],
    "peopleAlsoAsk": [
        {"question": "What is test?", "snippet": "A test answer."},
    ],
}

# (cleaned_text, capture_payload) tuples - matches _fetch_page_content's return
FAKE_PAGE_CONTENT = "Extracted main article body about the test topic."
FAKE_FETCH_RESULTS = {
    "http://example.com/result": (FAKE_PAGE_CONTENT, FAKE_PAGE_CONTENT),
    "http://example.com/second": ("Second page body.", "Second page body."),
}

# Real HTML that readability + markdownify extract into non-empty article text.
_HTML_PAGE = (
    "<html><head><title>Fed decision</title></head><body><article>"
    "<h1>Federal Reserve holds rates</h1>"
    "<p>The Federal Reserve held interest rates steady on Wednesday, citing "
    "persistent inflation concerns and a resilient labor market. Officials "
    "signaled they expect two more cuts before the end of the year.</p>"
    "<p>The decision was widely expected by economists surveyed beforehand. "
    "Markets moved modestly higher following the announcement as investors "
    "digested the updated projections.</p>"
    "</article></body></html>"
)


def _fake_fetch(
    url: str, mode: str = "cleaned", **_: object
) -> tuple[Optional[str], Optional[str]]:
    """Stand-in for _fetch_page_content that never touches the network."""
    return FAKE_FETCH_RESULTS.get(url, (None, None))


PREDICTION_JSON = json.dumps(
    {"p_yes": 0.5, "p_no": 0.5, "confidence": 0.5, "info_utility": 0.5}
)

PREDICTION_PROMPT = (
    'With the given question "Will X happen?" '
    "and the `yes` option represented by `Yes` and the `no` option represented by `No`, "
    "what are the respective probabilities of `p_yes` and `p_no` occurring?"
)


def _make_mock_api_keys(
    return_source_content: str = "false", source_content_mode: str = "cleaned"
) -> MagicMock:
    """Create a mock KeyChain-like api_keys object."""
    services = {
        "openai": ["sk-test"],
        "serperapi": ["serper-test"],
        "return_source_content": [return_source_content],
        "source_content_mode": [source_content_mode],
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: services[key][0]
    mock.get = lambda key, default="": services.get(key, [default])[0]
    return mock


def _make_prediction_stub() -> PredictionResult:
    """Build a valid PredictionResult for use as a parse() return value.

    :return: a PredictionResult whose numbers match PREDICTION_JSON.
    """
    numbers = json.loads(PREDICTION_JSON)
    return PredictionResult(
        facts="some facts",
        researchability="R",
        evidence_quality=0.6,
        reasons_no="reasons no",
        reasons_yes="reasons yes",
        evidence_reliability_screen="screen block",
        aggregation="aggregation block",
        reflection="reflection block",
        **numbers,
    )


def _stub_openai(mock_client_mgr: MagicMock) -> MagicMock:
    """Wire OpenAIClientManager to a client whose parse() returns a PredictionResult.

    :param mock_client_mgr: the patched OpenAIClientManager.
    :return: the stubbed OpenAI client.
    """
    mock_client = MagicMock()
    parsed_msg = MagicMock(parsed=_make_prediction_stub(), refusal=None)
    mock_client.beta.chat.completions.parse.return_value = MagicMock(
        choices=[MagicMock(message=parsed_msg)],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )
    mock_client_mgr.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_mgr.return_value.__exit__ = MagicMock(return_value=False)
    return mock_client


class TestPredictionResultSchema:
    """Guard the structured-output schema this tool depends on."""

    def test_numeric_fields_are_declared_last(self) -> None:
        """The four numerics must be the last fields, in order.

        Structured outputs generate fields in declaration order, so the
        numbers being last is what conditions them on the reasoning chain.
        Reordering still validates and still passes every other test, which
        is exactly why it needs an explicit assertion.
        """
        assert list(PredictionResult.model_fields)[-4:] == [
            "p_yes",
            "p_no",
            "confidence",
            "info_utility",
        ]

    def test_reasoning_fields_precede_the_numbers(self) -> None:
        """Every reasoning field is declared before the first number."""
        order = list(PredictionResult.model_fields)
        first_number = order.index("p_yes")
        for name in (
            "facts",
            "researchability",
            "evidence_quality",
            "reasons_no",
            "reasons_yes",
            "evidence_reliability_screen",
            "aggregation",
            "reflection",
        ):
            assert order.index(name) < first_number

    def test_validator_rejects_probabilities_that_do_not_sum_to_one(self) -> None:
        """p_yes + p_no far from 1 is refused by the model validator."""
        with pytest.raises(ValidationError):
            PredictionResult(
                facts="f",
                researchability="R",
                evidence_quality=0.5,
                reasons_no="n",
                reasons_yes="y",
                evidence_reliability_screen="s",
                aggregation="a",
                reflection="r",
                p_yes=0.7,
                p_no=0.5,
                confidence=0.5,
                info_utility=0.5,
            )

    def test_validator_accepts_probabilities_summing_to_one(self) -> None:
        """A well-formed instance validates."""
        parsed = PredictionResult(
            facts="f",
            researchability="R",
            evidence_quality=0.5,
            reasons_no="n",
            reasons_yes="y",
            evidence_reliability_screen="s",
            aggregation="a",
            reflection="r",
            p_yes=0.7,
            p_no=0.3,
            confidence=0.5,
            info_utility=0.5,
        )
        assert parsed.p_yes + parsed.p_no == 1.0

    def test_every_numeric_field_is_bounded_to_the_unit_interval(self) -> None:
        """Each of the four numerics declares ge=0 and le=1.

        Asserted structurally rather than by feeding one bad value: with
        p_yes + p_no forced to 1, an out-of-range p_yes implies an
        out-of-range p_no, so a single-value probe passes even when one
        field has lost its bounds.
        """
        for name in ("p_yes", "p_no", "confidence", "info_utility", "evidence_quality"):
            meta = PredictionResult.model_fields[name].metadata
            bounds = {
                type(c).__name__: getattr(c, "ge", getattr(c, "le", None)) for c in meta
            }
            assert bounds.get("Ge") == 0.0, f"{name} lost its ge=0 bound"
            assert bounds.get("Le") == 1.0, f"{name} lost its le=1 bound"

    def test_out_of_range_probability_is_refused(self) -> None:
        """Field bounds ge=0 / le=1 are enforced."""
        with pytest.raises(ValidationError):
            PredictionResult(
                facts="f",
                researchability="R",
                evidence_quality=0.5,
                reasons_no="n",
                reasons_yes="y",
                evidence_reliability_screen="s",
                aggregation="a",
                reflection="r",
                p_yes=1.4,
                p_no=-0.4,
                confidence=0.5,
                info_utility=0.5,
            )


# A real production request_context, captured from the Polygon marketplace
# subgraph. Kept verbatim so the tests exercise the shape the trader actually
# sends -- including the keys this tool deliberately ignores.
REAL_REQUEST_CONTEXT = {
    "market_id": "0x36d9e405ab5a9313c42daa29045ae8ee771f14c980d3325020d74b2bfabd3787",
    "type": "polymarket",
    "market_prob": 0.765,
    "market_liquidity_usd": 1311.4949,
    "market_spread": 0.01,
    "market_close_at": "2026-08-28T23:59:00Z",
    "description": "Kevin Warsh is scheduled to deliver a speech at the Jackson "
    "Hole Economic Policy Symposium on August 28, 2026. This market will "
    'resolve to "Yes" if Warsh says the listed term during his appearance.',
}


class TestCoerceMarketProb:
    """A supplied price is only used when it is a real number in [0, 1]."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.765, 0.765),
            (0, 0.0),
            (1, 1.0),
            (0.0, 0.0),
            (1.0, 1.0),
        ],
    )
    def test_usable_values(self, value: Any, expected: float) -> None:
        """Real numbers inside the unit interval pass through."""
        assert module._coerce_market_prob(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "0.5",
            None,
            [0.5],
            {"p": 0.5},
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_unusable_values_are_rejected(self, value: Any) -> None:
        """Anything else is refused rather than rendered into the prompt."""
        assert module._coerce_market_prob(value) is None

    @pytest.mark.parametrize("value", [True, False])
    def test_booleans_are_rejected(self, value: bool) -> None:
        """Booleans are a subclass of int, so True must be excluded explicitly.

        Without the explicit check, `market_prob: true` would silently become
        the probability 1.0 -- a maximally confident, entirely fabricated
        prior.

        :param value: the boolean under test.
        """
        assert module._coerce_market_prob(value) is None


class TestExtractMarketContext:
    """Market context is optional; malformed input degrades to blind mode."""

    def test_real_production_context_yields_the_three_read_keys(self) -> None:
        """The captured request gives price, close time and rules."""
        ctx = module._extract_market_context(REAL_REQUEST_CONTEXT)
        assert ctx["market_prob"] == 0.765
        assert ctx["market_close_at"] == "2026-08-28T23:59:00Z"
        assert ctx["resolution_rules"].startswith("Kevin Warsh is scheduled")

    def test_execution_cost_keys_are_never_read(self) -> None:
        """Liquidity, spread and fee belong to the trading engine, not here."""
        ctx = module._extract_market_context(REAL_REQUEST_CONTEXT)
        for ignored in (
            "market_liquidity_usd",
            "market_spread",
            "amm_fee",
            "market_id",
            "type",
        ):
            assert ignored not in ctx

    @pytest.mark.parametrize(
        "bad", [None, "oops", 42, [1, 2, 3], (), {}, {"unrelated": "x"}]
    )
    def test_absent_or_malformed_context_is_blind_mode(self, bad: Any) -> None:
        """Never raises; a live mech request must not fail on a bad field."""
        assert module._extract_market_context(bad) == {}

    @pytest.mark.parametrize(
        "prob", ["0.7", 1.4, -0.2, True, float("nan"), None, [0.7]]
    )
    def test_unusable_price_drops_only_the_price(self, prob: Any) -> None:
        """A bad price must not take the resolution rules down with it."""
        ctx = module._extract_market_context(
            {**REAL_REQUEST_CONTEXT, "market_prob": prob}
        )
        assert "market_prob" not in ctx
        assert "resolution_rules" in ctx

    def test_blank_strings_are_treated_as_absent(self) -> None:
        """Whitespace-only close time / rules do not render empty blocks."""
        ctx = module._extract_market_context(
            {**REAL_REQUEST_CONTEXT, "market_close_at": "   ", "description": "  "}
        )
        assert "market_close_at" not in ctx and "resolution_rules" not in ctx

    def test_resolution_rules_are_length_capped(self) -> None:
        """A hostile requester cannot blow up the prompt with rules."""
        ctx = module._extract_market_context(
            {"description": "x" * (module.MAX_RESOLUTION_RULES_CHARS + 5000)}
        )
        assert len(ctx["resolution_rules"]) == module.MAX_RESOLUTION_RULES_CHARS

    def test_omen_shaped_context_works(self) -> None:
        """Omen requests carry no description or spread; that is fine."""
        ctx = module._extract_market_context(
            {"market_id": "0xfpmm", "type": "omen", "market_prob": 0.3, "amm_fee": 0.02}
        )
        assert ctx == {"market_prob": 0.3}


class TestRenderMarketBlocks:
    """The rendered suffix, and what it must never contain."""

    def test_blind_mode_renders_nothing(self) -> None:
        """No context -> empty suffix -> prompt identical to blind mode."""
        assert module._render_market_blocks({}) == ""

    def test_price_and_close_time_are_rendered(self) -> None:
        """Both appear when both were supplied."""
        block = module._render_market_blocks(
            module._extract_market_context(REAL_REQUEST_CONTEXT)
        )
        assert "0.765" in block
        assert "2026-08-28T23:59:00Z" in block

    def test_rules_render_without_a_price(self) -> None:
        """Rules alone must not produce a 'prices P(Yes) at None' line."""
        block = module._render_market_blocks({"resolution_rules": "settle strictly"})
        assert "settle strictly" in block
        assert "P(Yes)" not in block

    def test_no_none_or_nan_ever_reaches_the_prompt(self) -> None:
        """Guard against formatting a missing value into the text."""
        for ctx in (
            {"market_prob": 0.5},
            {"resolution_rules": "r"},
            module._extract_market_context(REAL_REQUEST_CONTEXT),
        ):
            block = module._render_market_blocks(ctx)
            assert "None" not in block and "nan" not in block

    def test_block_states_the_odds_filter_does_not_apply(self) -> None:
        """The supplied price must not be swept up by the scraped-odds filter.

        The block sits next to the price for a reason: the instruction that
        exempts it has to be adjacent to the datum it exempts, or the screen's
        blanket-sounding clause wins.
        """
        block = module._render_market_blocks({"market_prob": 0.5})
        assert "does NOT apply to the price above" in block

    def test_block_does_not_instruct_copying_the_price(self) -> None:
        """The price is a prior to update from, not an anchor to copy."""
        block = module._render_market_blocks({"market_prob": 0.5})
        assert "Do not copy it" in block


class TestBlindVersusMarketAwareRun:
    """End-to-end through run(), both modes."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_blind_mode_prompt_has_no_market_block(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """With no request_context the prompt carries no market reasoning."""
        _stub_openai(mock_client_mgr)
        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )
        assert "Market context for this question" not in result[1]
        assert json.loads(result[0])["market_prob_seen"] is None

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_market_aware_prompt_carries_the_price(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """With a real request_context the price reaches the prompt."""
        _stub_openai(mock_client_mgr)
        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
            request_context=REAL_REQUEST_CONTEXT,
        )
        assert "Market context for this question" in result[1]
        assert "0.765" in result[1]
        assert json.loads(result[0])["market_prob_seen"] == 0.765

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_malformed_context_runs_as_blind_mode_without_raising(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """A garbage request_context must not fail a live mech request."""
        _stub_openai(mock_client_mgr)
        for bad in ("oops", 42, [1, 2], {"market_prob": "not-a-number"}):
            result = run(
                tool="superforcaster-market-aware",
                model="gpt-4o",
                prompt=PREDICTION_PROMPT,
                api_keys=_make_mock_api_keys("false"),
                counter_callback=None,
                source_content={"serper_response": FAKE_SERPER_RESPONSE},
                request_context=bad,
            )
            payload = json.loads(result[0])
            assert payload["market_prob_seen"] is None
            assert payload["p_yes"] + payload["p_no"] == 1.0

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_extras_are_present_in_both_modes(self, mock_client_mgr: MagicMock) -> None:
        """Rows stay schema-comparable so an A/B can pair them."""
        _stub_openai(mock_client_mgr)
        common = dict(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )
        blind = json.loads(run(**common)[0])
        aware = json.loads(run(**common, request_context=REAL_REQUEST_CONTEXT)[0])
        assert set(blind) == set(aware)
        for key in ("researchability", "evidence_quality"):
            assert blind[key] is not None and aware[key] is not None

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_market_prob_seen_is_echoed_not_generated(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """The echo comes from the request, never from the model.

        The stub always returns the same PredictionResult, so if the value
        tracked the model it would be constant across these two calls.

        :param mock_client_mgr: the patched OpenAIClientManager.
        """
        _stub_openai(mock_client_mgr)
        common = dict(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )
        a = json.loads(run(**common, request_context={"market_prob": 0.11})[0])
        b = json.loads(run(**common, request_context={"market_prob": 0.92})[0])
        assert (a["market_prob_seen"], b["market_prob_seen"]) == (0.11, 0.92)

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_p_no_is_derived_so_the_sum_is_exact(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """The schema tolerates 0.01 drift; the trader tests exact equality.

        A model answering 0.7 / 0.31 validates against the schema and would
        then be rejected by the trader as an invalid response. Deriving the
        complement removes that whole failure class.

        :param mock_client_mgr: the patched OpenAIClientManager.
        """
        mock_client = _stub_openai(mock_client_mgr)
        drifting = _make_prediction_stub().model_copy(
            update={"p_yes": 0.7, "p_no": 0.31}
        )
        mock_client.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(parsed=drifting, refusal=None))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )
        payload = json.loads(result[0])
        assert payload["p_yes"] + payload["p_no"] == 1.0


class TestReplayCompatibility:
    """The replay harness formats this template with exactly three kwargs."""

    def test_prompt_has_exactly_the_three_replayable_slots(self) -> None:
        """A fourth slot would KeyError on every replayed row.

        benchmark/prompt_replay.py's superforcaster branch calls
        PREDICTION_PROMPT.format(question=, today=, sources=). The guard that
        would name a mismatch by name is wired only to the factual_research
        branch, so this assertion stands in for it.
        """
        slots = set(re.findall(r"\{(\w+)\}", module.PREDICTION_PROMPT))
        assert slots == {"question", "today", "sources"}

    def test_market_blocks_live_outside_the_replayed_template(self) -> None:
        """Market context is appended by run(), not a template slot."""
        assert "market_prob" not in module.PREDICTION_PROMPT
        assert "{market_prob}" in module.MARKET_CONTEXT_BLOCK

    def test_the_replayed_prompt_renders(self) -> None:
        """Format it the way the harness does; it must not raise."""
        rendered = module.PREDICTION_PROMPT.format(
            question="Will X?", today="01/01/2026", sources="<background/>"
        )
        assert "Will X?" in rendered


class TestResearchabilityField:
    """The researchability class is an epistemic property, not a trade signal."""

    def test_taxonomy_is_the_frozen_eight_classes(self) -> None:
        """The module constant matches the frozen rubric, order included."""
        assert module.RESEARCHABILITY_CLASSES == (
            "NR-sports",
            "NR-utterance",
            "NR-price",
            "NR-numeric",
            "NR-headline",
            "NR-behavior",
            "R",
            "REVIEW",
        )

    def test_schema_enum_matches_the_taxonomy_constant(self) -> None:
        """The schema's Literal and the module constant cannot drift apart."""
        literal = get_args(PredictionResult.model_fields["researchability"].annotation)
        assert set(literal) == set(module.RESEARCHABILITY_CLASSES)

    def test_unknown_class_is_refused(self) -> None:
        """A class outside the taxonomy fails validation."""
        with pytest.raises(ValidationError):
            _make_prediction_stub().model_copy(update={"researchability": "NR-weather"})
            PredictionResult.model_validate(
                {
                    **_make_prediction_stub().model_dump(),
                    "researchability": "NR-weather",
                }
            )

    def test_sports_exception_is_stated_in_the_field_description(self) -> None:
        """The border rule must never promote sports to R.

        Without the exception spelled out, the 'when torn answer R' rule
        pulls sports questions into R, which is the single largest class in
        the labelled corpus this field is meant to be comparable with.
        """
        desc = PredictionResult.model_fields["researchability"].description
        assert "ALWAYS NR-sports" in desc
        assert "never promotes them" in desc


class TestForecastSignalOnly:
    """Rule 1: the tool emits forecast signals, never trade advice."""

    def test_no_trade_advice_vocabulary_anywhere_model_facing(self) -> None:
        """No trade-action vocabulary in the prompt or any field description.

        Matched on word boundaries rather than exact phrases. An earlier
        exact-phrase list let "you should not bet" through: it contains
        neither "should bet" nor "do not bet". Any standalone occurrence of
        the trade verbs is a violation regardless of the surrounding words,
        so the token itself is what gets banned.

        Asserted over everything the model reads -- the prompt, the system
        prompt, and every schema field description -- because that is where
        such wording would actually change behaviour.
        """
        model_facing = [module.PREDICTION_PROMPT, module.SYSTEM_PROMPT]
        model_facing += [
            f.description or "" for f in PredictionResult.model_fields.values()
        ]
        blob = " ".join(model_facing)
        banned = re.compile(
            r"\b(bets?|betting|wager(?:s|ed|ing)?|stake(?:s|d)?|"
            r"eligib(?:le|ility)|abstain(?:s|ed|ing)?|"
            r"buy|sell|long|short\s+the)\b",
            re.IGNORECASE,
        )
        hits = sorted({m.group(0).lower() for m in banned.finditer(blob)})
        assert not hits, f"trade-advice vocabulary in model-facing text: {hits}"

    def test_the_ban_regex_actually_fires(self) -> None:
        """Negative control: the guard is not vacuously green."""
        banned = re.compile(
            r"\b(bets?|betting|wager(?:s|ed|ing)?|stake(?:s|d)?|"
            r"eligib(?:le|ility)|abstain(?:s|ed|ing)?|"
            r"buy|sell|long|short\s+the)\b",
            re.IGNORECASE,
        )
        for phrase in (
            "you should not bet",
            "do not bet on this",
            "the market is not eligible",
            "abstain from trading",
            "place a wager",
        ):
            assert banned.search(phrase), f"guard misses: {phrase!r}"
        assert not banned.search("a better forecast between two options")

    def test_researchability_is_framed_as_a_property_not_a_recommendation(self) -> None:
        """The field description says so explicitly."""
        desc = PredictionResult.model_fields["researchability"].description
        assert "objective property" in desc
        assert "NOT a recommendation" in desc


class TestEvidenceReliabilityScreen:
    """The ported screen, with the market-odds clause inverted."""

    def test_screen_is_declared_before_the_aggregation(self) -> None:
        """The screen must precede any probability forming."""
        order = list(PredictionResult.model_fields)
        assert order.index("evidence_reliability_screen") < order.index("aggregation")

    def test_all_four_clauses_survived_the_port(self) -> None:
        """Clauses (a)-(d) are all present."""
        desc = PredictionResult.model_fields["evidence_reliability_screen"].description
        for clause in (
            "Prediction-market-odds filter",
            "Forward-looking-intent discount",
            "Temporal-evidence filter",
            "Criterion-specificity check",
        ):
            assert clause in desc

    def test_odds_clause_distinguishes_scraped_from_supplied(self) -> None:
        """Odds found in sources stay discarded; supplied market context does not.

        This inversion is the whole point of the tool. The parent screen
        discards every prediction-market price as circular, which is right for
        a price scraped off a web page and wrong for the price of the very
        market being forecast when it is handed over as an input.
        """
        desc = PredictionResult.model_fields["evidence_reliability_screen"].description
        assert "found in the SOURCES" in desc
        assert "does NOT apply to a market price supplied to you directly" in desc


class TestTokenBudget:
    """The completion budget has to fit the whole structured object."""

    def test_max_tokens_is_large_enough_for_the_schema(self) -> None:
        """A 500-token budget truncates the 12-field object mid-parse.

        Found the hard way: inherited from the parent, which emits a short
        prose JSON object. This tool emits eight reasoning fields plus four
        numbers, so `.parse()` raised "Could not parse response content as the
        length limit was reached" on EVERY live call, and the decorator turned
        that into {"p_yes": null, ...} -- a delivery the trader rejects. The
        failure is invisible to a stubbed test, which is why it needs an
        explicit floor.
        """
        assert module.DEFAULT_OPENAI_SETTINGS["max_tokens"] >= 4096

    def test_budget_matches_the_structured_output_siblings(self) -> None:
        """Every structured-output tool in the family uses the same budget."""
        import importlib

        for sibling in (
            "superforcaster",
            "superforcaster_calibrated_full_search",
            "superforcaster_polymarket_v4",
        ):
            mod = importlib.import_module(
                f"packages.valory.customs.{sibling}.{sibling}"
            )
            assert (
                mod.DEFAULT_OPENAI_SETTINGS["max_tokens"]
                == module.DEFAULT_OPENAI_SETTINGS["max_tokens"]
            ), f"{sibling} disagrees on the structured-output token budget"

    def test_temperature_is_deterministic(self) -> None:
        """Replay and A/B comparisons depend on temperature 0."""
        assert module.DEFAULT_OPENAI_SETTINGS["temperature"] == 0


class TestDeliveredPayload:
    """The bytes the trader receives must stay a flat, strict-parseable object."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_payload_is_flat_json_with_the_four_required_plus_extras(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """run() delivers strict-json.loads-parseable JSON, four keys, sum 1."""
        _stub_openai(mock_client_mgr)

        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )

        raw = result[0]
        assert raw.lstrip()[:1] == "{", "a leading non-brace char means NO BET"
        payload = json.loads(raw)
        assert set(payload) == {
            "p_yes",
            "p_no",
            "confidence",
            "info_utility",
            "researchability",
            "evidence_quality",
            "market_prob_seen",
        }
        for required in ("p_yes", "p_no", "confidence", "info_utility"):
            assert isinstance(payload[required], float)
        assert payload["p_yes"] + payload["p_no"] == 1.0
        for leaked in ("facts", "aggregation", "evidence_reliability_screen"):
            assert leaked not in raw


class TestSuperforcasterSourceContent:
    """Verify superforcaster_market_aware captures and replays source_content correctly."""

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_live_capture_includes_serper_and_pages(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
    ) -> None:
        """Live run captures Serper response AND scraped page texts."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        _stub_openai(mock_client_mgr)

        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
        )

        captured = result[4]["source_content"]
        assert captured["mode"] == "cleaned"
        assert captured["serper_response"] == FAKE_SERPER_RESPONSE
        # Both organic URLs were scraped -> both in pages capture
        assert captured["pages"] == {
            "http://example.com/result": FAKE_PAGE_CONTENT,
            "http://example.com/second": "Second page body.",
        }
        # Scraped page text reaches the prediction prompt under "Content:"
        prediction_prompt = result[1]
        assert FAKE_PAGE_CONTENT in prediction_prompt
        assert "**Content:**" in prediction_prompt
        # result[0] is the LLM completion content (via the OpenAIClient
        # wrapper), not an auto-MagicMock - so this JSON assertion is real.
        payload = json.loads(result[0])
        for key, value in json.loads(PREDICTION_JSON).items():
            assert payload[key] == value

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_live_scrape_failure_falls_back_to_snippet(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        mock_page_fetch: MagicMock,
    ) -> None:
        """Scrape returning (None, None) for every URL is non-fatal."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        mock_page_fetch.side_effect = lambda *a, **kw: (None, None)
        _stub_openai(mock_client_mgr)

        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
        )

        captured = result[4]["source_content"]
        assert captured["pages"] == {}
        prediction_prompt = result[1]
        # Still has Serper-tier evidence; no Content line was rendered
        assert "Test snippet content" in prediction_prompt
        assert "**Content:**" not in prediction_prompt

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_replay_with_pages_hydrates_content_into_prompt(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Replay format with `pages` injects cached content into the prompt."""
        _stub_openai(mock_client_mgr)

        source_content = {
            "mode": "cleaned",
            "serper_response": FAKE_SERPER_RESPONSE,
            "pages": {
                "http://example.com/result": "Cached cleaned article text.",
            },
        }
        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        prediction_prompt = result[1]
        assert "Cached cleaned article text." in prediction_prompt
        assert "Test snippet content" in prediction_prompt  # snippet preserved

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_replay_raw_mode_runs_clean_html_on_cached_html(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """mode='raw' replay re-extracts cleaned text from cached HTML."""
        _stub_openai(mock_client_mgr)

        source_content = {
            "mode": "raw",
            "serper_response": FAKE_SERPER_RESPONSE,
            "pages": {"http://example.com/result": _HTML_PAGE},
        }
        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        prediction_prompt = result[1]
        # raw HTML was run back through _clean_html -> extracted article text
        assert "Federal Reserve" in prediction_prompt
        assert "<html>" not in prediction_prompt  # raw markup not dumped verbatim

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_replay_legacy_format_without_pages_still_works(
        self, mock_client_mgr: MagicMock
    ) -> None:
        """Captures produced before evidence-gathering replay cleanly."""
        _stub_openai(mock_client_mgr)

        # Old format: no `pages` key, no `mode` key.
        source_content = {"serper_response": FAKE_SERPER_RESPONSE}
        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("true"),
            counter_callback=None,
            source_content=source_content,
        )

        prediction_prompt = result[1]
        assert "Test Result" in prediction_prompt
        assert "Test snippet content" in prediction_prompt
        assert "What is test?" in prediction_prompt
        # No Content line because there were no cached pages
        assert "**Content:**" not in prediction_prompt

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_flag_off_no_source_content(
        self,
        mock_fetch: MagicMock,
        mock_client_mgr: MagicMock,
        _mock_page_fetch: MagicMock,
    ) -> None:
        """When return_source_content is false, source_content is not in used_params."""
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_SERPER_RESPONSE
        mock_fetch.return_value = mock_response
        _stub_openai(mock_client_mgr)

        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        used_params = result[4]
        assert "source_content" not in used_params


class TestScrapePages:
    """Unit-level coverage for the scrape helper that runs in-process."""

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    def test_scrape_pages_attaches_content_and_captures(
        self, _mock_page_fetch: MagicMock
    ) -> None:
        """Successful scrapes mutate items and return the capture dict."""
        from packages.valory.customs.superforcaster_market_aware.superforcaster_market_aware import (
            _scrape_pages,
        )

        organic = [
            {"link": "http://example.com/result", "title": "T1", "snippet": "s1"},
            {"link": "http://example.com/second", "title": "T2", "snippet": "s2"},
        ]
        captured = _scrape_pages(organic, mode="cleaned")
        assert organic[0]["content"] == FAKE_PAGE_CONTENT
        assert organic[1]["content"] == "Second page body."
        assert captured == {
            "http://example.com/result": FAKE_PAGE_CONTENT,
            "http://example.com/second": "Second page body.",
        }

    @patch(
        f"{SF_MODULE}._fetch_page_content",
        side_effect=lambda *a, **kw: (None, None),
    )
    def test_scrape_pages_failure_returns_empty(
        self, _mock_page_fetch: MagicMock
    ) -> None:
        """When every fetch fails, items are untouched and capture is empty."""
        from packages.valory.customs.superforcaster_market_aware.superforcaster_market_aware import (
            _scrape_pages,
        )

        organic = [{"link": "http://example.com/x", "title": "T", "snippet": "s"}]
        captured = _scrape_pages(organic, mode="cleaned")
        assert "content" not in organic[0]
        assert captured == {}

    @patch(f"{SF_MODULE}._fetch_page_content", side_effect=_fake_fetch)
    def test_scrape_pages_mixed_success(self, _mock_page_fetch: MagicMock) -> None:
        """One URL succeeds, one fails: only the success gets content + capture."""
        from packages.valory.customs.superforcaster_market_aware.superforcaster_market_aware import (
            _scrape_pages,
        )

        organic = [
            {"link": "http://example.com/result", "title": "T1", "snippet": "s1"},
            {"link": "http://example.com/unknown", "title": "T2", "snippet": "s2"},
        ]
        captured = _scrape_pages(organic, mode="cleaned")
        # success -> content attached + in capture; failure -> neither (exercises
        # the `if text:` / `if capture:` guards).
        assert organic[0]["content"] == FAKE_PAGE_CONTENT
        assert "content" not in organic[1]
        assert captured == {"http://example.com/result": FAKE_PAGE_CONTENT}


class TestEvidenceBlockCap:
    """The cap drops trailing organic items until the rendered block fits."""

    def test_small_evidence_unchanged(self) -> None:
        """Below-budget evidence is returned without truncation marker."""
        from packages.valory.customs.superforcaster_market_aware.superforcaster_market_aware import (
            _cap_evidence_block,
        )

        organic = [
            {"title": "T", "link": "http://x", "snippet": "s", "position": 1},
        ]
        rendered = _cap_evidence_block(organic, [], model="gpt-4.1")
        assert "[... evidence truncated ...]" not in rendered
        assert "T" in rendered

    def test_oversize_evidence_is_trimmed_with_marker(self) -> None:
        """When over budget, trailing items are dropped and a marker is appended."""
        from packages.valory.customs.superforcaster_market_aware.superforcaster_market_aware import (
            MAX_EVIDENCE_TOKENS,
            _cap_evidence_block,
            count_tokens,
        )

        huge = "lorem ipsum " * 800  # ~1600 tokens per item
        organic = [
            {
                "title": f"T{i}",
                "link": f"http://x/{i}",
                "snippet": f"s{i}",
                "position": i + 1,
                "content": huge,
            }
            for i in range(5)
        ]
        rendered = _cap_evidence_block(organic, [], model="gpt-4.1")
        assert "[... evidence truncated ...]" in rendered
        assert count_tokens(rendered, "gpt-4.1") <= MAX_EVIDENCE_TOKENS + 100
        # Trailing items are dropped, leading (most-relevant) kept: a
        # leading-drop mutation would keep T4 and drop T0, failing this.
        assert "T0" in rendered
        assert "T4" not in rendered

    def test_paa_only_overflow_returns_without_loop(self) -> None:
        """With no organic items the cap returns as-is (no marker, no infinite loop)."""
        from packages.valory.customs.superforcaster_market_aware.superforcaster_market_aware import (
            _cap_evidence_block,
        )

        huge_paa = [
            {"question": "lorem ipsum " * 800, "link": "http://x", "snippet": "s"}
        ]
        rendered = _cap_evidence_block([], huge_paa, model="gpt-4.1")
        # organic is empty -> early return, no trailing-drop marker added
        assert "[... evidence truncated ...]" not in rendered
        assert "lorem ipsum" in rendered


class TestFetchPageContent:
    """Direct coverage of _fetch_page_content's four early-return paths."""

    @staticmethod
    def _resp(
        status: int = 200,
        content_type: str = "text/html; charset=utf-8",
        text: str = "",
    ) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.headers = {"Content-Type": content_type}
        resp.text = text
        return resp

    @patch(f"{SF_MODULE}.requests.get")
    def test_happy_path_cleaned(self, mock_get: MagicMock) -> None:
        """200 + HTML -> (cleaned_text, cleaned_text) in cleaned mode."""
        mock_get.return_value = self._resp(text=_HTML_PAGE)
        text, capture = module._fetch_page_content("http://x", mode="cleaned")
        assert text is not None and "Federal Reserve" in text
        assert capture == text  # cleaned mode stores the cleaned text

    @patch(f"{SF_MODULE}.requests.get")
    def test_happy_path_raw_stores_html(self, mock_get: MagicMock) -> None:
        """200 + HTML -> capture is the raw HTML in raw mode."""
        mock_get.return_value = self._resp(text=_HTML_PAGE)
        text, capture = module._fetch_page_content("http://x", mode="raw")
        assert text is not None and "Federal Reserve" in text
        assert capture == _HTML_PAGE  # raw mode stores the raw html

    @patch(f"{SF_MODULE}.requests.get")
    def test_non_200_returns_none(self, mock_get: MagicMock) -> None:
        """A 404 yields (None, None)."""
        mock_get.return_value = self._resp(status=404, text=_HTML_PAGE)
        assert module._fetch_page_content("http://x") == (None, None)

    @patch(f"{SF_MODULE}.requests.get")
    def test_non_html_content_type_returns_none(self, mock_get: MagicMock) -> None:
        """A non-HTML content-type (JSON) yields (None, None)."""
        mock_get.return_value = self._resp(
            content_type="application/json", text='{"a": 1}'
        )
        assert module._fetch_page_content("http://x") == (None, None)

    @patch(f"{SF_MODULE}.requests.get")
    def test_request_exception_returns_none(self, mock_get: MagicMock) -> None:
        """A network exception is swallowed -> (None, None)."""
        mock_get.side_effect = requests.Timeout("slow")
        assert module._fetch_page_content("http://x") == (None, None)

    @patch(f"{SF_MODULE}._clean_html", return_value=None)
    @patch(f"{SF_MODULE}.requests.get")
    def test_unextractable_html_returns_none(
        self, mock_get: MagicMock, _mock_clean: MagicMock
    ) -> None:
        """200 + HTML but readability extracts nothing -> (None, None)."""
        mock_get.return_value = self._resp(text="<html></html>")
        assert module._fetch_page_content("http://x") == (None, None)


class TestErrorHandling:
    """with_key_rotation's catch-all returns parseable null-prediction JSON."""

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_unexpected_error_returns_parseable_error_json(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """An unexpected exception yields {p_yes:None,...,error:...}, not a raw string."""
        _stub_openai(mock_client_mgr)
        mock_fetch.side_effect = RuntimeError("boom")

        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        payload = json.loads(result[0])  # must be valid JSON, not a bare str
        assert payload["p_yes"] is None
        assert payload["p_no"] is None
        assert payload["error"] == "boom"

    @patch(f"{SF_MODULE}.time.sleep", return_value=None)
    @patch(f"{SF_MODULE}.OpenAIClientManager")
    def test_null_content_surfaces_as_error_json(
        self, mock_client_mgr: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        """An LLM refusal (parsed=None) becomes error JSON, not a None prediction."""
        mock_client = _stub_openai(mock_client_mgr)
        mock_client.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(parsed=None, refusal="policy"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )

        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
            source_content={"serper_response": FAKE_SERPER_RESPONSE},
        )

        payload = json.loads(result[0])  # not None -> no downstream json.loads(None)
        assert payload["p_yes"] is None
        assert "refus" in payload["error"].lower()


class TestSerperRequest:
    """Serper call carries a timeout and surfaces HTTP errors."""

    @patch(f"{SF_MODULE}.requests.request")
    def test_fetch_additional_sources_passes_timeout(
        self, mock_request: MagicMock
    ) -> None:
        """The Serper request forwards timeout=30 (fleet standard)."""
        fetch_additional_sources("question?", "serper-key")
        _, kwargs = mock_request.call_args
        assert kwargs["timeout"] == 30

    @patch(f"{SF_MODULE}.OpenAIClientManager")
    @patch(f"{SF_MODULE}.fetch_additional_sources")
    def test_serper_http_error_surfaces_as_error_json(
        self, mock_fetch: MagicMock, mock_client_mgr: MagicMock
    ) -> None:
        """A 4xx/5xx Serper response raises via raise_for_status -> error JSON."""
        _stub_openai(mock_client_mgr)
        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many")
        mock_fetch.return_value = bad_response

        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=None,
        )

        bad_response.raise_for_status.assert_called_once()
        bad_response.json.assert_not_called()  # never reached on HTTP error
        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert "429" in payload["error"]


class TestSourceContentModeValidation:
    """An invalid source_content_mode surfaces as a recognisable error."""

    def test_invalid_mode_returns_error_json(self) -> None:
        """A bad mode yields error JSON (not a silent string) via the catch-all."""
        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys(source_content_mode="bogus"),
            counter_callback=None,
        )

        payload = json.loads(result[0])
        assert payload["p_yes"] is None
        assert "Invalid source_content_mode" in payload["error"]


class TestMaxCostPath:
    """delivery_rate=0 returns the float max_cost untouched (float guard)."""

    def test_max_cost_returns_float_not_wrapped_tuple(self) -> None:
        """Without the isinstance(result, float) guard this raises TypeError."""
        result = run(
            tool="superforcaster-market-aware",
            model="gpt-4o",
            prompt=PREDICTION_PROMPT,
            api_keys=_make_mock_api_keys("false"),
            counter_callback=lambda **_: 0.0123,
            delivery_rate=0,
        )
        assert result == 0.0123
