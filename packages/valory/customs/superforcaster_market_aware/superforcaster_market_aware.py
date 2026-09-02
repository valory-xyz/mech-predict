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
"""Contains the job definitions"""

import functools
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import openai
import requests
from markdownify import markdownify as md
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from readability import Document as ReadabilityDocument
from tiktoken import encoding_for_model

MechResponseWithKeys = Tuple[
    str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]], Any
]
MechResponse = Tuple[
    str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]]
]
MaxCostResponse = float

N_MODEL_CALLS = 1
DEFAULT_DELIVERY_RATE = 100
# Serper degrades sharply on prompt-shaped queries (instruction boilerplate,
# JSON-format text), in the worst case to zero organic results (issue #455).
_MAX_SEARCH_QUERY_LEN = 150


def with_key_rotation(func: Callable) -> Callable:
    """
    Decorator that retries a function with API key rotation on failure.

    :param func: The function to be decorated.
    :type func: Callable
    :returns: Callable -- the wrapped function that handles retries with key rotation.
    """

    @functools.wraps(func)
    def wrapper(
        *args: Any, **kwargs: Any
    ) -> Union[MaxCostResponse, MechResponseWithKeys]:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> Union[MaxCostResponse, MechResponseWithKeys]:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                # Max-cost path returns a float; pass through without
                # appending api_keys (tuple concatenation would fail).
                if isinstance(result, float):
                    return result
                return result + (api_keys,)
            except openai.RateLimitError as e:
                # try with a new key again
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
            except Exception as e:  # noqa: BLE001
                # Return a parseable null-prediction JSON (matches
                # factual_research) so downstream tournament scoring sees
                # an explicit error rather than treating a raw exception
                # string as a prediction.
                # Same key set as a normal delivery and the flagged null,
                # so every exit path is schema-comparable downstream.
                error_json = json.dumps(
                    {
                        "p_yes": None,
                        "p_no": None,
                        "confidence": 0.0,
                        "info_utility": 0.0,
                        "researchability": None,
                        "research_class": None,
                        "research_reason": None,
                        "evidence_quality": None,
                        "market_prob_seen": None,
                        "p_independent": None,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                return error_json, "", None, None, None, api_keys

        return execute()

    return wrapper


class OpenAIClientManager:
    """Context manager that creates and closes a local OpenAI client."""

    def __init__(self, api_key: str):
        """Initializes with API key."""
        self.api_key = api_key
        self._client: Optional[OpenAI] = None

    def __enter__(self) -> OpenAI:
        """Initializes and returns the OpenAI client."""
        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Closes the OpenAI client."""
        if self._client is not None:
            self._client.close()
            self._client = None


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    try:
        enc = encoding_for_model(model)
    except KeyError:
        from tiktoken import get_encoding  # pylint: disable=import-outside-toplevel

        enc = get_encoding("o200k_base")
    return len(enc.encode(text))


# max_tokens is 4096, not the parent's 500. The parent emits a short prose
# JSON object; this tool emits a 17-field structured object whose reasoning
# fields carry the whole chain of thought. At 500 the completion is
# truncated mid-object and `.parse()` raises "Could not parse response content
# as the length limit was reached", which the decorator converts into
# {"p_yes": null, ...} -- a delivery the trader rejects, i.e. silently no bet
# on every single request. Every structured-output tool in this family
# (superforcaster, superforcaster_calibrated_full_search,
# superforcaster-polymarket-v4) uses 4096 for the same reason.
DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 4096,
    "temperature": 0,
}
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"
ALLOWED_TOOLS = ["superforcaster-market-aware"]
MAX_SOURCES = 5
COMPLETION_RETRIES = 3
COMPLETION_DELAY = 2

# Evidence-gathering: fetch full page content for the top organic results so
# the forecaster reasons over article text, not just Serper snippets.
MAX_PAGES_TO_SCRAPE = 5
_MAX_PAGE_WORDS = 400
_PAGE_FETCH_TIMEOUT_S = 10
_SCRAPE_POOL_WORKERS = 6
_IMG_TAG_PATTERN = re.compile(r"<img[^>]*>", re.IGNORECASE)
_SCRIPT_STYLE_PATTERN = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)

# Cap on the rendered <background> evidence block to bound prompt size and
# avoid lost-in-the-middle degradation when an outlier page returns a very
# long body. Trailing organic items are dropped (Serper orders by relevance)
# until the rendered block fits. Same trailing-drop pattern as
# factual_research (which caps at 3000); budget set to 4000 here to fit
# observed evidence sizes with headroom. Not load-bearing for gpt-4.1's
# 1M context but bounds cost and guards against outlier pages.
MAX_EVIDENCE_TOKENS = 4000

# Researchability taxonomy. Frozen wording, reused verbatim rather than
# reinvented: the same eight tokens and the same border rule already back a
# large corpus of hand/LLM-labelled questions, so this tool's judgements can be
# scored against existing labels instead of needing a fresh ground truth.
# Order matters -- the prompt says first match wins, and NR-sports is first
# because sports questions otherwise get promoted to R by the border rule.
RESEARCHABILITY_CLASSES = (
    "NR-sports",
    "NR-utterance",
    "NR-price",
    "NR-numeric",
    "NR-headline",
    "NR-behavior",
    "R",
    "REVIEW",
)


class PredictionResult(BaseModel):
    """Superforecaster structured output.

    The text fields carry the reasoning chain the prompt asks for (facts ->
    reasons against -> reasons for -> aggregation + tentative -> reflection).
    Only the four numeric fields are returned on-chain per the mech protocol.

    Using structured outputs means the completion is guaranteed to match this
    schema, so the prompt carries no JSON-format instructions and no regex
    extraction is needed on the way out.
    """

    facts: str = Field(
        ...,
        description=(
            "Core factual points compiled from the sources and from relevant "
            "background that may not be in the sources. Specific and relevant, "
            "covering the core considerations for the forecast. No conclusions "
            "about how a fact influences the answer."
        ),
    )
    research_class: Literal[
        "NR-sports",
        "NR-utterance",
        "NR-price",
        "NR-numeric",
        "NR-headline",
        "NR-behavior",
        "R",
        "REVIEW",
    ] = Field(
        ...,
        description=(
            "Whether pre-resolution research can genuinely move a rational "
            "forecast on THIS question. Answer R only if you can NAME a "
            "specific, findable kind of source that would shift your "
            "probability by more than 5 points if you had it (a scheduled "
            "filing, a court docket, a poll, a published record). If you "
            "cannot name one, choose the closest NR class rather than "
            "defaulting to R. Classes, first match wins. NR-sports: the "
            "outcome of a contest undecided at question time - research into "
            "form, injuries and lineups is real, but the residual on-field "
            "randomness dominates what it can add. NR-utterance: hinges on "
            "whether a person says, posts or names specific words, or how "
            "many times. NR-price: a short-horizon asset-price tick or "
            "threshold. NR-numeric: a narrow numeric band effectively random "
            "at question time. NR-headline: hinges on the exact phrasing of a "
            "future media headline. NR-behavior: a trivial personal behaviour "
            "of an individual. R: researchable - focused research before "
            "resolution could meaningfully move a rational forecast "
            "(elections, court rulings, product launches, geopolitical "
            "events, scheduled announcements). REVIEW only when the question "
            "is too ambiguous to classify. This is an objective property of "
            "the question. It is NOT a recommendation to act or not act."
        ),
    )
    research_reason: str = Field(
        ...,
        description=(
            "One sentence naming the specific findable source that justifies "
            "the class above (for R: the filing, docket, poll or record that "
            "would move the forecast; for NR-*: why no such source can exist "
            "for this question)."
        ),
    )
    researchability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How much pre-resolution research can move a rational forecast on "
            "this question, as a number consistent with the class above. "
            "Anchor bands: NR-price and NR-numeric 0.0-0.15; NR-utterance, "
            "NR-headline and NR-behavior 0.05-0.25; NR-sports 0.15-0.35 "
            "(research is real but on-field randomness dominates); REVIEW "
            "0.4-0.6; R 0.6-1.0, higher when the nameable source is scheduled "
            "or certain to exist, lower when it is speculative. Pick a value "
            "inside the band, not a boundary. This is an objective property "
            "of the question, emitted for continuous downstream weighting - "
            "never an instruction to act and never a hard filter."
        ),
    )
    evidence_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How well the retrieved evidence actually bears on the resolution "
            "criterion (0 = the sources say nothing that discriminates between "
            "Yes and No, 1 = a source directly establishes the outcome). Judge "
            "the evidence, not your confidence in the forecast: a question can "
            "be answered confidently from base rates while the retrieved "
            "evidence is worthless, and that case scores low here."
        ),
    )
    reasons_no: str = Field(
        ...,
        description=(
            "Reasons why the answer might be NO. Rate the strength of each "
            "reason on a scale of 1-10."
        ),
    )
    reasons_yes: str = Field(
        ...,
        description=(
            "Reasons why the answer might be YES. Rate the strength of each "
            "reason on a scale of 1-10."
        ),
    )
    evidence_reliability_screen: str = Field(
        ...,
        description=(
            "MANDATORY evidence-reliability screen, completed BEFORE forming a "
            "tentative probability. (a) Prediction-market-odds filter: discard "
            "any prediction-market trading price you found in the SOURCES "
            "(polymarket / metaculus / manifold / predictit / kalshi) as "
            "circular self-referential evidence. This applies to odds scraped "
            "from a web page. It does NOT apply to a market price supplied to "
            "you directly as market context, which is a legitimate input and is "
            "handled where that context is given. (b) Forward-looking-intent "
            "discount: for intent or expectation language ('is set to', 'is "
            "expected to', 'plans to', 'scheduled to', 'is poised to'), treat "
            "the outcome as only 40-60% likely to materialize absent strong "
            "specific evidence. (c) Temporal-evidence filter: classify each "
            "source TYPE A (dated within the resolution window, or directly "
            "states the criterion was met) vs TYPE B (undated, outside the "
            "window, or a standing page); state the TYPE A and TYPE B counts; "
            "if ALL sources are TYPE B, anchor on the base rate FOR THIS "
            "CATEGORY OF QUESTION, which you must state explicitly before "
            "using it. Do not carry a single default across categories: "
            "'will someone say a specific word' markets resolve YES far "
            "less often than 'will this asset close above a level it is "
            "already near' markets, and a threshold already crossed at "
            "question time is likelier still. When a market price has been "
            "supplied, it is a better estimate of the base rate than any "
            "figure you can recall, and you should say so rather than "
            "anchoring below it. (d) "
            "Criterion-specificity check: does any TYPE A evidence directly "
            "confirm the exact resolution condition (not merely that the topic "
            "is active)? If not, add uncertainty toward the base rate."
        ),
    )
    independent_reasoning: str = Field(
        ...,
        description=(
            "Reason to a probability from YOUR evidence alone. Do not mention "
            "or use any supplied market price in this field. If no market "
            "price was supplied, reason normally."
        ),
    )
    p_independent: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Your probability from your own evidence alone, committed BEFORE "
            "you consider any supplied market price. When no price is "
            "supplied this is simply your estimate."
        ),
    )
    market_reconciliation: str = Field(
        ...,
        description=(
            "If a market price was supplied, state where it sits relative to "
            "p_independent and what other forecasters might know that your "
            "sources do not. The price aggregates the judgement of people who "
            "have seen public information too, so it is evidence about the "
            "base rate and about facts you may lack -- for a question whose "
            "answer turns on a current value you cannot observe, it may be "
            "the ONLY evidence of that value you have. Update toward it where "
            "it plausibly reflects something your sources do not, and stand "
            "your ground where you hold a specific TYPE A source it cannot "
            "have priced. Write 'no market context supplied' when none was "
            "given."
        ),
    )
    aggregation: str = Field(
        ...,
        description=(
            "Aggregate the considerations. Do not summarize or repeat previous "
            "points; investigate how the competing factors and mechanisms "
            "interact and weigh against each other. Factorize across "
            "exhaustive, mutually exclusive cases only when that helps. Adjust "
            "for the negativity and sensationalism bias of news sources. Think "
            "like a superforecaster. End by stating an initial tentative "
            "probability as a single number between 0 and 1."
        ),
    )
    reflection: str = Field(
        ...,
        description=(
            "Sanity checks and finalisation. Check for over/underconfidence, "
            "improper treatment of conjunctive or disjunctive conditions, and "
            "other forecasting biases. Consider priors and base rates, and how "
            "far case-specific information justifies deviating from them. Be "
            "precise with tail probabilities. Never change the forecast for the "
            "sake of modesty or balance alone. Highlight the key factors that "
            "inform the final forecast."
        ),
    )
    # IMPORTANT: the four numeric fields below MUST stay LAST in this schema.
    # Structured outputs generate fields in declaration order, so keeping the
    # numbers after the reasoning is what conditions them on the chain of
    # thought. Do not reorder or alphabetize: pydantic will not complain and
    # the schema still validates, but the calibration silently degrades.
    p_yes: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Estimated probability that the event in the Question occurs. "
            "Your final answer after market_reconciliation: it may differ "
            "from p_independent, and should where the supplied price carries "
            "information your own sources lack."
        ),
    )
    p_no: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that the event does NOT occur.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the prediction (0 = lowest, 1 = highest).",
    )
    info_utility: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Utility of the information in the sources to inform the "
            "prediction (0 = lowest, 1 = highest)."
        ),
    )

    @model_validator(mode="after")
    def _check_p_yes_p_no_sum(self) -> "PredictionResult":
        """Validate that p_yes and p_no sum to 1.

        :return: the validated model.
        :raises ValueError: when the two probabilities do not sum to 1.
        """
        if abs(self.p_yes + self.p_no - 1.0) > 0.01:
            raise ValueError(
                f"p_yes + p_no must equal 1 (got {self.p_yes} + {self.p_no} = "
                f"{self.p_yes + self.p_no})"
            )
        return self


# Maximum characters of resolution rules echoed into the prompt. The trader
# already caps its `description` at 5000 chars, so this is a defensive second
# bound for requesters that are not the trader.
MAX_RESOLUTION_RULES_CHARS = 5000

# Appended to PREDICTION_PROMPT by run() when the request supplies market
# context. Kept OUT of PREDICTION_PROMPT itself and out of its format() slots
# on purpose: the replay harness formats the superforcaster template with
# exactly {question}, {today} and {sources}, so a fourth slot would raise
# KeyError on every replayed row, and the guard that names that failure by
# name is wired only to the factual_research branch. Appending instead keeps
# the tool replayable, and the prompt the harness renders is exactly this
# tool's blind mode -- which is the control arm an A/B needs anyway.
MARKET_CONTEXT_BLOCK = """

Market context for this question. This was supplied with the request. It was NOT
retrieved from the web:
- The market currently prices P(Yes) at {market_prob}.{close_line}

ORDER OF WORK. Reason to `p_independent` from YOUR evidence alone and commit to it first, so
that your own reading is recorded before you see where the crowd sits. Then, in
`market_reconciliation`, weigh the price: it aggregates the judgement of others who have also
seen public information, so treat it as evidence about the base rate and about facts your
sources may not contain. Your final `p_yes` may differ from `p_independent`, and SHOULD where
the price plausibly carries something you lack - a current value you cannot observe, or a base
rate your sources never state. Hold your own estimate where you have a specific TYPE A source
the price cannot have absorbed. Coinciding with the price is a correct answer, not a failure.

Note on the prediction-market-odds filter in `evidence_reliability_screen`: that filter
concerns odds you found in the retrieved sources. It does NOT apply to the price above,
which is a supplied input about the very question you are forecasting.
"""

MARKET_CLOSE_LINE = """
- The market closes at {market_close_at}."""

RESOLUTION_RULES_BLOCK = """

Resolution rules for this market. These were supplied with the request and are
authoritative for how it settles:
<resolution_rules>{resolution_rules}</resolution_rules>

Judge the question strictly against these rules rather than against your own reading of
the title.
"""

SYSTEM_PROMPT = "You are a helpful assistant."

PREDICTION_PROMPT = """
You are an advanced AI system which has been finetuned to provide calibrated probabilistic
forecasts under uncertainty, with your performance evaluated according to the Brier score. When
forecasting, do not treat 0.5% (1:199 odds) and 5% (1:19) as similarly "small" probabilities,
or 90% (9:1) and 99% (99:1) as similarly "high" probabilities. As the odds show, they are
markedly different, so output your probabilities accordingly.

Question:
{question}

Today's date: {today}
Your pretraining knowledge cutoff: October 2023

We have retrieved the following information for this question:
<background>{sources}</background>

Recall the question you are forecasting:
{question}

Instructions. Return a structured PredictionResult whose fields carry the following
reasoning chain:

1. `facts` - Compress key factual information from the sources, as well as useful background
information which may not be in the sources, into a list of core factual points to reference.
Aim for information which is specific, relevant, and covers the core considerations you'll use
to make your forecast. For this step, do not draw any conclusions about how a fact will
influence your answer or forecast.

2. `research_class` - Classify, in one token, whether pre-resolution web research can
genuinely inform this question. Follow that field's class list exactly; first match wins, and
sports outcomes are always NR-sports. This is an objective property of the question, not a
recommendation to act. Then `research_reason` - one sentence naming the findable source that
justifies the class (or why none can exist), and `researchability` - a 0-1 number inside that
class's anchor band, per the field description.

3. `evidence_quality` - Rate 0 to 1 how well the retrieved sources actually bear on the
resolution criterion. Judge the evidence, not your confidence in the answer.

4. `reasons_no` - Provide a few reasons why the answer might be no. Rate the strength of each
reason on a scale of 1-10.

5. `reasons_yes` - Provide a few reasons why the answer might be yes. Rate the strength of each
reason on a scale of 1-10.

6. `evidence_reliability_screen` - MANDATORY, and completed BEFORE you form any probability.
Follow every part of that field's instructions: the prediction-market-odds filter, the
forward-looking-intent discount, the temporal-evidence TYPE A / TYPE B classification (state
both counts), and the criterion-specificity check.

7. `independent_reasoning` and `p_independent` - Reason to a probability from YOUR evidence
alone and commit to it here, BEFORE considering any supplied market price.

8. `market_reconciliation` - If a market price was supplied, say where it sits relative to
`p_independent`. Moving away from it requires naming a specific TYPE A source.

9. `aggregation` - Aggregate your considerations. Do not summarize or repeat previous points;
instead, investigate how the competing factors and mechanisms interact and weigh against each
other. Factorize your thinking across (exhaustive, mutually exclusive) cases if and only if it
would be beneficial to your reasoning. We have detected that you overestimate world conflict,
drama, violence, and crises due to news' negativity bias, which doesn't necessarily represent
overall trends or base rates. Similarly, we also have detected you overestimate dramatic,
shocking, or emotionally charged news due to news' sensationalism bias. Therefore adjust for
news' negativity bias and sensationalism bias by considering reasons to why your provided
sources might be biased or exaggerated. Think like a superforecaster. End this field by
stating an initial tentative probability as a single number between 0 and 1 given the steps above.

10. `reflection` - Reflect on your tentative answer, performing sanity checks and mentioning any
additional knowledge or background information which may be relevant. Check for
over/underconfidence, improper treatment of conjunctive or disjunctive conditions (only if
applicable), and other forecasting biases when reviewing your reasoning. Consider priors/base
rates, and the extent to which case-specific information justifies the deviation between your
tentative forecast and the prior. Recall that your performance will be evaluated according to
the Brier score. Be precise with tail probabilities. Leverage your intuitions, but never change
your forecast for the sake of modesty or balance alone. Finally, aggregate all of your previous
reasoning and highlight key factors that inform your final forecast.

11. `p_yes`, `p_no`, `confidence`, `info_utility` - Output your final prediction. `p_yes` is the
probability that the event in the Question occurs and `p_no` that it does not; the two must sum
to 1. `confidence` is how confident you are in the prediction and `info_utility` how useful the
retrieved information was in making it. All four are numbers between 0 and 1.
"""


def _parse_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Any,
    temperature: float = 0,
    max_tokens: int = 4096,
    retries: int = COMPLETION_RETRIES,
    delay: int = COMPLETION_DELAY,
    counter_callback: Optional[Callable] = None,
) -> Tuple[Any, Optional[Callable]]:
    """Call OpenAI Structured Outputs and parse into a Pydantic model.

    ``client.beta.chat.completions.parse()`` guarantees the completion is
    well-formed JSON matching the schema's field names and types, so the
    prompt carries no JSON-format instructions and no output extraction is
    needed. It does NOT enforce the custom ``model_validator``
    (``p_yes + p_no ~= 1``): that raises ``pydantic.ValidationError`` (a
    ``ValueError``) inside ``.parse()``, which is why ``ValueError`` is in the
    retry tuple below.

    :param client: an initialised OpenAI client.
    :param model: OpenAI model identifier.
    :param messages: chat messages list (role + content dicts).
    :param response_format: Pydantic model class used as the structured-output schema.
    :param temperature: sampling temperature (0 = deterministic).
    :param max_tokens: maximum tokens to generate.
    :param retries: number of retry attempts on transient / validation failure.
    :param delay: delay in seconds between retries.
    :param counter_callback: optional callback tracking token usage.
    :return: tuple of (parsed model instance, counter_callback).
    :raises RuntimeError: if all retries are exhausted without a successful parse.
    """
    attempt = 0
    last_error: Optional[Exception] = None
    while attempt < retries:
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=150,
            )

            parsed = response.choices[0].message.parsed

            if parsed is None:
                refusal = response.choices[0].message.refusal
                raise ValueError(
                    f"Model refused or returned unparseable output: {refusal}"
                )

            if counter_callback is not None:
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=model,
                    token_counter=count_tokens,
                )

            return parsed, counter_callback
        except openai.LengthFinishReasonError as e:
            # Truncation is deterministic at temperature 0: the same prompt and
            # the same budget truncate again, so the retries below would burn
            # three more paid calls to reproduce the failure. Fail immediately
            # with a message naming the actual cause.
            #
            # This exception subclasses OpenAIError, NOT ValueError, so without
            # this branch it bypasses the retry tuple entirely and lands in
            # with_key_rotation's catch-all as an opaque string -- which is
            # exactly how a 500-token budget shipped unnoticed through 97
            # stubbed tests.
            raise RuntimeError(
                f"Structured completion truncated: max_tokens={max_tokens} is "
                f"too small for the {response_format.__name__} schema "
                f"({len(response_format.model_fields)} fields). Raise "
                "DEFAULT_OPENAI_SETTINGS['max_tokens']."
            ) from e
        except (
            openai.APIConnectionError,
            openai.InternalServerError,
            ValueError,
        ) as e:
            # NB: openai.RateLimitError is deliberately NOT caught here.
            # Letting it propagate to the with_key_rotation decorator lets the
            # decorator rotate API keys on a rate-limit hit; retrying in-place
            # on the same throttled key never rotates. Transient connection /
            # server / validation failures stay here and retry on the same key.
            print(f"[superforcaster-market-aware] Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
            attempt += 1
            last_error = e

    raise RuntimeError(
        f"Failed to get structured LLM completion after retries: {last_error}"
    ) from last_error


def _clean_html(html: str, max_words: int = _MAX_PAGE_WORDS) -> Optional[str]:
    """Extract main article text from HTML via readability + markdownify."""
    cleaned = _SCRIPT_STYLE_PATTERN.sub("", html)
    cleaned = _IMG_TAG_PATTERN.sub("", cleaned)
    article_html = ReadabilityDocument(cleaned).summary()
    text = md(article_html, heading_style="ATX", strip=["img", "figure"])
    if not text or not text.strip():
        return None
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + " [...]"
    return text.strip()


def _fetch_page_content(
    url: str,
    mode: str = "cleaned",
    max_words: int = _MAX_PAGE_WORDS,
    timeout: int = _PAGE_FETCH_TIMEOUT_S,
) -> Tuple[Optional[str], Optional[str]]:
    """Fetch a URL and return (cleaned_text, capture_payload).

    `capture_payload` is the raw HTML when mode=="raw" (for full-fidelity
    replay) and the cleaned text otherwise. Returns (None, None) on any
    fetch / parse failure - the caller falls back to the Serper snippet.

    :param url: The URL to fetch.
    :param mode: ``"cleaned"`` stores extracted text; ``"raw"`` stores HTML.
    :param max_words: Maximum number of words to keep in the cleaned text.
    :param timeout: Request timeout in seconds.
    :return: Tuple of (cleaned text for the LLM prompt, payload to store
        for replay).
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MechBot/1.0)"},
        )
        if resp.status_code != 200:
            return None, None
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return None, None
        text = _clean_html(resp.text, max_words=max_words)
        if not text:
            return None, None
        capture = resp.text if mode == "raw" else text
        return text, capture
    except Exception as e:  # noqa: BLE001 -- best-effort scrape, never raise
        print(f"[superforcaster-market-aware] Failed to fetch {url}: {e}")
        return None, None


def _scrape_pages(
    organic_data: List[Dict[str, Any]],
    mode: str,
    max_pages: int = MAX_PAGES_TO_SCRAPE,
) -> Dict[str, str]:
    """Concurrently scrape the top organic links and attach `content` in place.

    Returns the capture dict {url: cleaned_text_or_raw_html} for replay. The
    organic items themselves are mutated to add a `content` key when the
    scrape succeeds, so format_sources_data() can render it alongside the
    snippet without other plumbing.

    :param organic_data: Serper organic-result dicts (mutated in place to
        add a ``content`` key on successful scrapes).
    :param mode: ``"cleaned"`` stores extracted text in the capture dict;
        ``"raw"`` stores raw HTML.
    :param max_pages: Cap on how many top results to scrape.
    :return: Capture dict ``{url: cleaned_text_or_raw_html}`` for replay.
    """
    captured: Dict[str, str] = {}
    items_to_scrape = [it for it in organic_data[:max_pages] if it.get("link")]
    if not items_to_scrape:
        return captured

    with ThreadPoolExecutor(max_workers=_SCRAPE_POOL_WORKERS) as pool:
        future_to_item = {
            pool.submit(_fetch_page_content, item["link"], mode): item
            for item in items_to_scrape
        }
        for fut in as_completed(future_to_item):
            item = future_to_item[fut]
            try:
                text, capture = fut.result()
            except Exception as e:  # noqa: BLE001
                print(
                    f"[superforcaster-market-aware] Scrape error for {item['link']}: {e}"
                )
                continue
            if text:
                item["content"] = text
            if capture:
                captured[item["link"]] = capture
    return captured


def _hydrate_organic_from_pages(
    organic_data: List[Dict[str, Any]],
    pages: Dict[str, str],
    mode: str,
) -> None:
    """Replay path: re-attach cached page content to organic items in place."""
    if not pages:
        return
    for item in organic_data:
        cached = pages.get(item.get("link", ""))
        if cached is None:
            continue
        if mode == "raw":
            # Same log-and-skip contract as the live scrape: one bad cached
            # page degrades that item to its Serper snippet, never the whole
            # replayed request (readability raises Unparseable on empty HTML).
            try:
                text = _clean_html(cached)
            except Exception as e:  # noqa: BLE001 -- best-effort, never raise
                print(
                    "[superforcaster-market-aware] Failed to clean cached "
                    f"page {item.get('link', '')!r}: {e}"
                )
                continue
            if text:
                item["content"] = text
        else:
            item["content"] = cached


def fetch_additional_sources(question: str, serper_api_key: str) -> requests.Response:
    """Fetches additional sources for the given question using the Serper API."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": question})
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json",
    }
    # timeout matches the fleet's other Serper callers (factual_research,
    # prediction_request, ...); without it a hung connection blocks the run.
    return requests.request("POST", url, headers=headers, data=payload, timeout=30)


def format_sources_data(organic_data: Any, misc_data: Any) -> str:
    """Formats organic search results and "People Also Ask" data into a human-readable string."""
    sources = ""

    if len(organic_data) > 0:
        print("Adding organic data...")

        sources = """
        Organic Results:
        """

        for item in organic_data:
            sources += f"""{item.get('position', 'N/A')}. **Title:** {item.get("title", 'N/A')}
            - **Link:** [{item.get("link", '#')}]({item.get("link", '#')})
            - **Snippet:** {item.get("snippet", 'N/A')}
            """
            content = item.get("content")
            if content:
                sources += f"            - **Content:** {content}\n"

    if len(misc_data) > 0:
        print("Adding misc data...")

        sources += "People Also Ask:\n"

        counter = 1
        for item in misc_data:
            sources += f"""{counter}. **Question:** {item.get("question", 'N/A')}
            - **Link:** [{item.get("link", '#')}]({item.get("link", '#')})
            - **Snippet:** {item.get("snippet", 'N/A')}
            """
            counter += 1

    return sources


def _cap_evidence_block(
    organic_data: List[Dict[str, Any]],
    misc_data: List[Dict[str, Any]],
    model: str,
    max_tokens: int = MAX_EVIDENCE_TOKENS,
) -> str:
    """Render the evidence block, dropping trailing organic items until it fits.

    Same trailing-drop pattern as factual_research (which caps at 3000;
    4000 here): Serper orders organic results by relevance so trailing
    drops are cheapest. If the block still exceeds the budget once all
    organic items are gone, the result is returned as-is (peopleAlsoAsk is
    small and not separately trimmed).

    :param organic_data: Serper organic results (already capped to MAX_SOURCES).
    :param misc_data: Serper peopleAlsoAsk items.
    :param model: model name for tokeniser selection.
    :param max_tokens: target ceiling on the rendered block.
    :return: rendered evidence string, with a truncation marker if items were dropped.
    """
    rendered = format_sources_data(organic_data, misc_data)
    if count_tokens(rendered, model) <= max_tokens or not organic_data:
        return rendered

    trimmed = list(organic_data)
    while (
        trimmed
        and count_tokens(format_sources_data(trimmed, misc_data), model) > max_tokens
    ):
        trimmed.pop()
    rendered = format_sources_data(trimmed, misc_data)
    rendered += "\n[... evidence truncated ...]\n"
    return rendered


def _coerce_market_prob(value: Any) -> Optional[float]:
    """Return a usable P(Yes) from a request_context value, or None.

    Rejects anything that is not a real number in [0, 1]. ``bool`` is excluded
    explicitly because ``isinstance(True, int)`` is True in Python, so a stray
    boolean would otherwise become the probability 1.0.

    ``math.isfinite`` is belt-and-braces rather than load-bearing: the range
    check alone already rejects NaN (every comparison against NaN is False)
    and both infinities (they fail a bound). It is kept because relying on
    NaN comparison semantics to reject NaN is the kind of implicit behaviour
    that a later edit to the bounds would silently remove.

    :param value: the raw ``market_prob`` from the request context.
    :return: the probability as a float, or None when unusable.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        return None
    return numeric


def _extract_market_context(request_context: Any) -> Dict[str, Any]:
    """Pull the three fields this tool reads out of a mech request_context.

    Market context is an OPTIONAL input: a request that carries none, or
    carries a malformed one, must leave the tool behaving exactly as it does
    without it. So every failure here degrades to an empty dict rather than
    raising -- a live mech request must never fail because a market field was
    the wrong shape.

    Only three keys are read. ``market_liquidity_usd``, ``market_spread`` and
    ``amm_fee`` are deliberately ignored: they are execution-cost signals that
    belong to the trading engine, not inputs to a forecast.

    :param request_context: the ``request_context`` kwarg from the mech request.
    :return: dict with any of ``market_prob``, ``market_close_at``,
        ``resolution_rules``; empty when nothing usable was supplied.
    """
    if not isinstance(request_context, dict):
        return {}

    context: Dict[str, Any] = {}

    market_prob = _coerce_market_prob(request_context.get("market_prob"))
    if market_prob is not None:
        context["market_prob"] = market_prob

    close_at = request_context.get("market_close_at")
    if isinstance(close_at, str) and close_at.strip():
        context["market_close_at"] = close_at.strip()

    rules = request_context.get("description")
    if isinstance(rules, str) and rules.strip():
        context["resolution_rules"] = rules.strip()[:MAX_RESOLUTION_RULES_CHARS]

    # A field that was SUPPLIED but rejected (wrong shape) is a supply-side
    # bug worth its own log line: without it, a caller whose market_prob
    # starts arriving as a string degrades to blind mode permanently and
    # looks identical to a caller that never sent a price.
    rejected = [
        key
        for key, kept in (
            ("market_prob", "market_prob" in context),
            ("market_close_at", "market_close_at" in context),
            ("description", "resolution_rules" in context),
        )
        if request_context.get(key) is not None and not kept
    ]
    if rejected:
        print(
            "[superforcaster-market-aware] Market context fields supplied "
            f"but rejected (wrong shape): {rejected}"
        )

    return context


def _format_market_prob(market_prob: float) -> str:
    """Render a price with trailing zeros trimmed but never as a bare integer.

    :param market_prob: the coerced price in [0, 1].
    :return: e.g. 0.765 -> "0.765", 0.5 -> "0.5", 1.0 -> "1.0".
    """
    text = f"{market_prob:.4f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _render_market_blocks(context: Dict[str, Any]) -> str:
    """Render the prompt suffix for a supplied market context.

    The price block is gated on the price alone: a close timestamp or a set of
    resolution rules without a price still renders the rules, but never an
    empty "prices P(Yes) at None" line.

    :param context: the dict from :func:`_extract_market_context`.
    :return: the suffix to append to the prediction prompt; "" when the
        request carried no usable market context.
    """
    suffix = ""

    market_prob = context.get("market_prob")
    if market_prob is not None:
        close_at = context.get("market_close_at")
        close_line = (
            MARKET_CLOSE_LINE.format(market_close_at=close_at) if close_at else ""
        )
        suffix += MARKET_CONTEXT_BLOCK.format(
            market_prob=_format_market_prob(market_prob),
            close_line=close_line,
        )

    rules = context.get("resolution_rules")
    if rules:
        suffix += RESOLUTION_RULES_BLOCK.format(resolution_rules=rules)

    return suffix


# Matches from 'question "' to '" and the `yes`' to handle nested quotes.
_TRADER_TEMPLATE_RE = re.compile(r'question\s+"(.+?)"\s+and\s+the\s+`yes`', re.DOTALL)
# Question-clause candidates: every question-word occurrence starts one, running
# to the FIRST '?' after it (via str.find; tolerates embedded dots --
# abbreviations, decimals, market ids -- which sentence-boundary splitting
# would cut on).
# Candidates may overlap; a feature score selects the market question among
# them (see _score_clause).
_QUESTION_WORD_RE = re.compile(
    r"(?:will|is|are|was|were|does|do|did|can|could|who|what|when|where|which"
    r"|how|whether)\b",
    re.IGNORECASE,
)
# Meta/instruction stems: a question addressed at the RESPONDER ("Can you
# estimate...", "What is your probability...") or prompt scaffolding ("What
# follows is..."), never the market question itself. Second-person only:
# first-person clauses ("Will we...", "Do I...") occur in real market wording.
_META_STEM_RE = re.compile(
    r"^(?:(?:can|could|would|will|do|does|did|is|are)\s+(?:you|your)\b"
    r"|what\s+(?:is|are)\s+(?:your|the\s+(?:respective\s+)?probabilit)"
    r"|what\s+follows\b)",
    re.IGNORECASE,
)
_MARKET_VERB_RE = re.compile(
    r"^(?:Will|Is|Are|Was|Were|Does|Do|Did|Which|Who|When|Whether)\b"
)
# Chars that may directly precede a sentence-initial question word: whitespace,
# sentence punctuation, ASCII quotes/paren, and typographic quotes.
_CLAUSE_BOUNDARY = " \t\n.!?:\"'(\u201c\u201d\u2018\u2019"
# Candidate scanning is bounded to the prompt head: every question-word
# occurrence starts a candidate and each candidate scans forward for '?', so
# an unbounded scan is quadratic (a ~100KB prompt -- the mech's
# MAX_PROMPT_BYTES cap -- costs seconds; benchmark/direct calls are not even
# capped). Market questions sit in the prompt head in practice (the longest
# observed production prompt is under 1KB), so a 10KB window loses nothing.
_MAX_SCAN_CHARS = 10_000
# Near-best window for the last-market-verb tiebreaker. Equals the largest
# single-feature weight (the digit bonus in _score_clause) so a market clause
# can never be pushed out of contention by one feature alone.
_NEAR_BEST_WINDOW = 3


def _score_clause(prompt: str, start: int, clause: str) -> int:
    """Score a question-clause candidate; the market question should win.

    Features: digits (market questions carry deadlines/quantities; instruction
    and clarifying questions rarely do), a market-shaped opening verb, a
    sentence-initial capitalized start, a penalty for responder-addressed /
    scaffolding stems, and a penalty for sweeping across a sentence boundary.

    :param prompt: the full prompt (for boundary context).
    :param start: the clause's start offset in the prompt.
    :param clause: the candidate clause text.
    :return: the feature score (higher = more market-question-shaped).
    """
    score = 0
    if any(ch.isdigit() for ch in clause):
        score += 3
    if _MARKET_VERB_RE.match(clause):
        score += 1
    if clause[0].isupper() and (start == 0 or prompt[start - 1] in _CLAUSE_BOUNDARY):
        score += 2
    if _META_STEM_RE.match(clause):
        score -= 3
    if ". " in clause:
        score -= 1
    return score


def _shape_serper_sources(
    raw: Dict[str, Any], context: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate a serper_response body and slice it into (organic, misc).

    A body without the organic key is a broken or reshaped integration (a
    quota-error body, a renamed key, a corrupted cache entry), not a genuine
    zero-hit -- raise so it surfaces as an error null with error_type instead
    of collapsing into the flagged null.

    :param raw: the serper_response dict (live or cached).
    :param context: short label for the error message (live vs cached replay).
    :return: the (organic, peopleAlsoAsk) lists, organic capped at MAX_SOURCES.
    """
    if "organic" not in raw:
        raise ValueError(
            f"{context}: Serper response missing 'organic' key; "
            f"got keys: {sorted(raw)[:8]}"
        )
    return raw.get("organic", [])[:MAX_SOURCES], raw.get("peopleAlsoAsk", [])


def _truncate_query(query: str) -> str:
    """Cap the query at _MAX_SEARCH_QUERY_LEN, cutting on a word boundary.

    :param query: the derived search query.
    :return: the query, truncated without a dangling partial word.
    """
    if len(query) <= _MAX_SEARCH_QUERY_LEN:
        return query
    cut = query[:_MAX_SEARCH_QUERY_LEN]
    if not query[_MAX_SEARCH_QUERY_LEN].isspace():
        cut = cut.rsplit(None, 1)[0] if " " in cut.strip() else cut
    return cut.rstrip()


class ParsedPrompt(NamedTuple):
    """parse_prompt's result: the LLM question, the Serper query, the tier."""

    question: str
    query: str
    tier: Literal["template", "clause", "raw"]


def parse_prompt(prompt: str) -> ParsedPrompt:
    """Split a request prompt into the LLM question and the Serper search query.

    Trader-template prompts carry the bare market question between known
    delimiters: it serves as both values, keeping that path byte-identical to
    previous releases. Any other prompt is free text under the advertised
    input contract (issue #455): the LLM receives the WHOLE prompt (resolution
    criteria, source, and deadline stay in context) while the search query is
    the best-scoring question clause (see _score_clause), with double quotes
    dropped (Serper treats quoted spans as exact-match terms) and the length
    capped on a word boundary.

    :param prompt: the raw prompt passed to run().
    :return: a ParsedPrompt -- tier is 'template' (trader regex matched),
        'clause' (a scored question clause), or 'raw' (no clause found;
        capped prompt head).
    """
    match = _TRADER_TEMPLATE_RE.findall(prompt)
    if match:
        question = match[0]
        return ParsedPrompt(question, question, "template")
    scan = prompt[:_MAX_SCAN_CHARS]
    candidates = []
    for word in _QUESTION_WORD_RE.finditer(scan):
        start = word.start()
        if start > 0 and scan[start - 1].isalnum():
            continue
        end = scan.find("?", start)
        if end == -1:
            continue
        clause = scan[start : end + 1]
        candidates.append(
            (_score_clause(scan, start, clause), len(clause), -start, clause)
        )
    tier: Literal["template", "clause", "raw"]
    if candidates:
        # Clarifying questions (inside resolution criteria) often carry the
        # dates/counts that outscore a digit-free market question. In free
        # text the market question is reliably the LAST market-verb-shaped
        # question -- clarifiers and instructions precede it -- so among
        # candidates near the best score, prefer the last market-verb one.
        best_score = max(candidates)[0]
        market_shaped = [
            c
            for c in candidates
            if c[0] >= best_score - _NEAR_BEST_WINDOW
            and _MARKET_VERB_RE.match(c[3])
            and not _META_STEM_RE.match(c[3])
        ]
        chosen = (
            min(market_shaped, key=lambda c: c[2]) if market_shaped else max(candidates)
        )
        query, tier = chosen[3], "clause"
    else:
        query, tier = scan, "raw"
    query = _truncate_query(query.replace('"', "").strip())
    if not query:
        # Degenerate prompts (only quotes/whitespace) must not strip down to
        # an empty Serper query -- fall back to the unstripped prompt head.
        query = _truncate_query(prompt.strip())
    return ParsedPrompt(prompt, query, tier)


def _flagged_null_result(
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    captured_source_content: Optional[Dict[str, Any]],
    return_source_content: bool,
    counter_callback: Optional[Callable[..., Any]],
    context: str,
    tier: str,
    market_context: Dict[str, Any],
) -> MechResponse:
    """Build the flagged null prediction returned on empty retrieval.

    A VALID prediction (p_yes = p_no = 0.5) with zero confidence and
    info_utility, so a requester can detect and discount it while the strict
    trader consumer still parses it (issue #455).

    :param model: the model name recorded in used_params.
    :param temperature: the temperature recorded in used_params.
    :param max_tokens: the max_tokens recorded in used_params.
    :param captured_source_content: the (empty) retrieval capture.
    :param return_source_content: whether to attach the capture to used_params.
    :param counter_callback: the cost callback, threaded back unchanged.
    :param context: short label for the log line (live vs cached replay).
    :param tier: the parse_prompt tier that produced the search query.
    :param market_context: the extracted market context; only
        ``market_prob`` is echoed, as ``market_prob_seen``.
    :return: the flagged-null MechResponse tuple.
    """
    print(
        f"[superforcaster-market-aware] {context}: empty retrieval"
        " -- returning null prediction"
    )
    null_result = json.dumps(
        {
            "p_yes": 0.5,
            "p_no": 0.5,
            "confidence": 0.0,
            "info_utility": 0.0,
            # Model-derived extras are null (no model ran), but the payload
            # keeps the full key set so null deliveries stay
            # schema-comparable, and market_prob_seen still records whether
            # this was a market-aware request.
            "researchability": None,
            "research_class": None,
            "research_reason": None,
            "evidence_quality": None,
            "market_prob_seen": market_context.get("market_prob"),
            "p_independent": None,
        }
    )
    used_params: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Off-chain markers distinguishing a flagged null from a genuine
        # max-uncertainty forecast and recording the derivation tier
        # (matches superforcaster-polymarket-v4).
        "empty_retrieval": True,
        "parse_tier": tier,
    }
    if return_source_content:
        used_params["source_content"] = captured_source_content
    return null_result, "", None, counter_callback, used_params


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """Run the task"""
    tool = kwargs["tool"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    model = kwargs.get("model")
    if model is None:
        raise ValueError("Model not supplied.")

    delivery_rate = int(kwargs.get("delivery_rate", DEFAULT_DELIVERY_RATE))
    counter_callback: Optional[Callable[..., Any]] = kwargs.get(
        "counter_callback", None
    )
    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback was given to calculate the max cost with."
            )

        max_cost = counter_callback(
            max_cost=True,
            models_calls=(model,) * N_MODEL_CALLS,
        )
        return max_cost

    openai_api_key = kwargs["api_keys"]["openai"]
    source_content = kwargs.get("source_content", None)
    return_source_content = (
        kwargs["api_keys"].get("return_source_content", "false") == "true"
    )
    source_content_mode = kwargs["api_keys"].get("source_content_mode", "cleaned")
    if source_content_mode not in ("cleaned", "raw"):
        raise ValueError(
            f"Invalid source_content_mode: {source_content_mode!r}. Must be 'cleaned' or 'raw'."
        )
    with OpenAIClientManager(openai_api_key) as llm_client:
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        prompt = kwargs["prompt"]

        today = date.today()
        d = today.strftime("%d/%m/%Y")

        # Optional market context, extracted BEFORE the retrieval guards so
        # an empty-retrieval delivery still echoes market_prob_seen and stays
        # schema-comparable with normal deliveries.
        market_context = _extract_market_context(kwargs.get("request_context"))

        question, search_query, tier = parse_prompt(prompt)
        if tier == "raw":
            print(
                "[superforcaster-market-aware] No question clause found; "
                f"using capped prompt head as the search query: {search_query!r}"
            )
        elif tier != "template":
            print(
                f"[superforcaster-market-aware] Free-text prompt (tier={tier}); "
                f"derived search query: {search_query!r}"
            )

        if source_content is not None:
            print("Using provided source content (cached replay)...")
            captured_source_content = source_content
            serper_data = source_content.get("serper_response", source_content)
            organic_data, misc_data = _shape_serper_sources(
                serper_data, "cached replay"
            )
            # Shallow-copy each organic item so attaching `content` does not
            # mutate the caller's cached source_content payload.
            organic_data = [dict(it) for it in organic_data]
            if not organic_data and not misc_data:
                return _flagged_null_result(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    captured_source_content=captured_source_content,
                    return_source_content=return_source_content,
                    counter_callback=counter_callback,
                    context="cached replay",
                    tier=tier,
                    market_context=market_context,
                )
            cached_pages = source_content.get("pages", {})
            # Legacy captures (written before "mode" existed) stored cleaned
            # text; defaulting to the operator's current mode would run
            # readability over plain text under raw mode and silently drop
            # every cached page.
            cached_mode = source_content.get("mode", "cleaned")
            _hydrate_organic_from_pages(organic_data, cached_pages, cached_mode)
            sources = _cap_evidence_block(organic_data, misc_data, model)
        else:
            if not search_query.strip('"').strip():
                # Nothing searchable (empty, whitespace-only, or quotes-only
                # query): skip the wasted Serper call and return the flagged
                # null directly.
                return _flagged_null_result(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    captured_source_content=None,
                    return_source_content=return_source_content,
                    counter_callback=counter_callback,
                    context="empty query",
                    tier=tier,
                    market_context=market_context,
                )
            serper_api_key = kwargs["api_keys"]["serperapi"]
            print("Fetching additional sources...")
            # Use the compressed search_query instead of the full prompt so
            # Serper returns organic results for free-text callers (issue #455).
            serper_response = fetch_additional_sources(search_query, serper_api_key)
            # Surface HTTP errors with a real status code instead of crashing
            # .json() on a non-JSON 4xx/5xx body (matches the fleet pattern).
            serper_response.raise_for_status()
            sources_data = serper_response.json()
            print(f"Additional sources fetched: {sources_data}")
            organic_data, misc_data = _shape_serper_sources(sources_data, "live search")
            # Shallow-copy organic items: _scrape_pages attaches `content`,
            # and we don't want that leaking into the captured serper_response.
            organic_data = [dict(it) for it in organic_data]
            # Empty-retrieval guard: even a correct short query can fail
            # (e.g. very niche or recent market). Placed BEFORE the scraping
            # step so empty retrieval wastes no page fetches (issue #455).
            if not organic_data and not misc_data:
                return _flagged_null_result(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    captured_source_content={
                        "mode": source_content_mode,
                        "serper_response": sources_data,
                    },
                    return_source_content=return_source_content,
                    counter_callback=counter_callback,
                    context="live search",
                    tier=tier,
                    market_context=market_context,
                )
            print("Scraping page content for top organic results...")
            captured_pages = _scrape_pages(organic_data, source_content_mode)
            print(
                f"Scraped {len(captured_pages)}/{min(MAX_SOURCES, len(organic_data))} pages."
            )
            captured_source_content = {
                "mode": source_content_mode,
                "serper_response": sources_data,
                "pages": captured_pages,
            }
            print("Formatting sources...")
            sources = _cap_evidence_block(organic_data, misc_data, model)

        print("Updating prompt...")
        prediction_prompt = PREDICTION_PROMPT.format(
            question=question, today=d, sources=sources
        )
        # Absent or malformed context -> blind mode, in which the rendered
        # prompt is byte-identical to the parent's plus this tool's own
        # reasoning-field instructions, and no market-derived reasoning
        # happens at all.
        prediction_prompt += _render_market_blocks(market_context)
        if market_context:
            print(
                f"[superforcaster-market-aware] Market context supplied: "
                f"{sorted(market_context)}"
            )
        else:
            print("[superforcaster-market-aware] No market context; blind mode.")
        print(f"\n{prediction_prompt=}\n")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prediction_prompt},
        ]
        print("Getting prompt response...")
        prediction: PredictionResult
        prediction, counter_callback = _parse_completion(
            client=llm_client,
            model=model,
            messages=messages,
            response_format=PredictionResult,
            temperature=temperature,
            max_tokens=max_tokens,
            counter_callback=counter_callback,
        )

        print(f"[superforcaster-market-aware] === FACTS ===\n{prediction.facts}")
        print(
            f"[superforcaster-market-aware] === REASONS_NO ===\n{prediction.reasons_no}"
        )
        print(
            f"[superforcaster-market-aware] === REASONS_YES ===\n"
            f"{prediction.reasons_yes}"
        )
        print(
            f"[superforcaster-market-aware] === AGGREGATION ===\n"
            f"{prediction.aggregation}"
        )
        print(
            f"[superforcaster-market-aware] === REFLECTION ===\n"
            f"{prediction.reflection}"
        )
        print(
            f"[superforcaster-market-aware] Result: p_yes={prediction.p_yes}, "
            f"p_no={prediction.p_no}, confidence={prediction.confidence}, "
            f"info_utility={prediction.info_utility}"
        )

        # Delivered payload: the four standard mech fields first, extras
        # additive and non-breaking. p_no is derived, not model-supplied:
        # strict consumers require an exact p_yes + p_no sum.
        p_yes = round(prediction.p_yes, 4)
        result = json.dumps(
            {
                "p_yes": p_yes,
                "p_no": round(1.0 - p_yes, 4),
                "confidence": prediction.confidence,
                "info_utility": prediction.info_utility,
                # Optional extras. Epistemic properties of the question and of
                # the forecast -- never a trade recommendation. Unknown keys
                # are dropped by the trader and by every benchmark parser
                # today, so these cost nothing downstream until something is
                # taught to read them. researchability is the 0-1 sizing
                # signal; research_class/research_reason preserve the
                # categorical reasoning it was derived from.
                "researchability": round(prediction.researchability, 4),
                "research_class": prediction.research_class,
                "research_reason": prediction.research_reason,
                "evidence_quality": prediction.evidence_quality,
                # Echoed from the request by this code, never generated by the
                # model, so it cannot be hallucinated. Without it nothing
                # downstream can tell a market-aware delivery from a blind one
                # after the fact.
                "market_prob_seen": market_context.get("market_prob"),
                # The pre-market commitment, emitted so drift toward a
                # supplied price is measurable from the delivered payload
                # rather than inferred. On 50 screening questions the final
                # probability equalled this on 50/50 rows.
                "p_independent": round(prediction.p_independent, 4),
            }
        )

        used_params: Dict[str, Any] = {
            "parse_tier": tier,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if return_source_content:
            used_params["source_content"] = captured_source_content
        return result, prediction_prompt, None, counter_callback, used_params
