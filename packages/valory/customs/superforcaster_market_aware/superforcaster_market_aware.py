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
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

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
                error_json = json.dumps(
                    {
                        "p_yes": None,
                        "p_no": None,
                        "confidence": 0.0,
                        "info_utility": 0.0,
                        "error": str(e),
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


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
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
    researchability: Literal[
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
            "Whether pre-resolution web research can genuinely inform this "
            "question. Exactly one class, first match wins. NR-sports: the "
            "outcome of a sports match, game, race or tournament, or a player's "
            "in-game performance - ALWAYS this class, even though form and news "
            "research exist. NR-utterance: hinges on whether a person says or "
            "tweets specific words a specific number of times. NR-price: a "
            "short-horizon asset-price tick or threshold. NR-numeric: a narrow "
            "numeric band effectively random at question time. NR-headline: "
            "hinges on the exact phrasing of a future media headline. "
            "NR-behavior: a trivial personal behaviour of an individual. R: "
            "researchable - focused web research before the market resolves "
            "could meaningfully move a rational forecast (elections, court "
            "rulings, product launches, geopolitical events, scheduled "
            "announcements). Border rule: if strong research could move a "
            "rational forecast by more than 5 points, answer R; when genuinely "
            "torn, answer R. EXCEPTION: sports outcomes are ALWAYS NR-sports - "
            "the border rule never promotes them. REVIEW only when the question "
            "is too ambiguous to classify. This is an objective property of the "
            "question. It is NOT a recommendation to act or not act."
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
            "if ALL sources are TYPE B, anchor on the category base rate "
            "(20-40% YES for 'X in headlines this week'-style markets). (d) "
            "Criterion-specificity check: does any TYPE A evidence directly "
            "confirm the exact resolution condition (not merely that the topic "
            "is active)? If not, add uncertainty toward the base rate."
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
        description="Estimated probability that the event in the Question occurs.",
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

How to use it. Treat this price as a prior held by other forecasters, and as a question
to answer: what might those forecasters know that your sources do not? Do not copy it.
Reason from your own evidence and state where you end up. If your evidence genuinely
adds nothing beyond what the price already reflects, your honest estimate may coincide
with the price - that is a correct answer, not a failure. If your evidence contradicts
the price, say so in `aggregation` and let your own estimate stand.

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

2. `researchability` - Classify, in one token, whether pre-resolution web research can
genuinely inform this question. Follow that field's class list exactly; first match wins, and
sports outcomes are always NR-sports. This is an objective property of the question, not a
recommendation to act.

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

7. `aggregation` - Aggregate your considerations. Do not summarize or repeat previous points;
instead, investigate how the competing factors and mechanisms interact and weigh against each
other. Factorize your thinking across (exhaustive, mutually exclusive) cases if and only if it
would be beneficial to your reasoning. We have detected that you overestimate world conflict,
drama, violence, and crises due to news' negativity bias, which doesn't necessarily represent
overall trends or base rates. Similarly, we also have detected you overestimate dramatic,
shocking, or emotionally charged news due to news' sensationalism bias. Therefore adjust for
news' negativity bias and sensationalism bias by considering reasons to why your provided
sources might be biased or exaggerated. Think like a superforecaster. End this field by
stating an initial tentative probability as a single number between 0 and 1 given the steps above.

8. `reflection` - Reflect on your tentative answer, performing sanity checks and mentioning any
additional knowledge or background information which may be relevant. Check for
over/underconfidence, improper treatment of conjunctive or disjunctive conditions (only if
applicable), and other forecasting biases when reviewing your reasoning. Consider priors/base
rates, and the extent to which case-specific information justifies the deviation between your
tentative forecast and the prior. Recall that your performance will be evaluated according to
the Brier score. Be precise with tail probabilities. Leverage your intuitions, but never change
your forecast for the sake of modesty or balance alone. Finally, aggregate all of your previous
reasoning and highlight key factors that inform your final forecast.

9. `p_yes`, `p_no`, `confidence`, `info_utility` - Output your final prediction. `p_yes` is the
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
            text = _clean_html(cached)
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

    return context


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
            market_prob=f"{market_prob:.4f}".rstrip("0").rstrip("."),
            close_line=close_line,
        )

    rules = context.get("resolution_rules")
    if rules:
        suffix += RESOLUTION_RULES_BLOCK.format(resolution_rules=rules)

    return suffix


def extract_question(prompt: str) -> str:
    """Uses regexp to extract question from the prompt"""
    # Match from 'question "' to '" and the `yes`' to handle nested quotes
    pattern = r'question\s+"(.+?)"\s+and\s+the\s+`yes`'
    try:
        question = re.findall(pattern, prompt, re.DOTALL)[0]
    except Exception as e:  # noqa: BLE001
        print(f"Error extracting question: {e}")
        question = prompt
    return question


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

        question = extract_question(prompt)

        if source_content is not None:
            print("Using provided source content (cached replay)...")
            captured_source_content = source_content
            serper_data = source_content.get("serper_response", source_content)
            # Shallow-copy each organic item so attaching `content` does not
            # mutate the caller's cached source_content payload.
            organic_data = [
                dict(it) for it in serper_data.get("organic", [])[:MAX_SOURCES]
            ]
            misc_data = serper_data.get("peopleAlsoAsk", [])
            cached_pages = source_content.get("pages", {})
            cached_mode = source_content.get("mode", source_content_mode)
            _hydrate_organic_from_pages(organic_data, cached_pages, cached_mode)
            sources = _cap_evidence_block(organic_data, misc_data, model)
        else:
            serper_api_key = kwargs["api_keys"]["serperapi"]
            print("Fetching additional sources...")
            serper_response = fetch_additional_sources(question, serper_api_key)
            # Surface HTTP errors with a real status code instead of crashing
            # .json() on a non-JSON 4xx/5xx body (matches the fleet pattern).
            serper_response.raise_for_status()
            sources_data = serper_response.json()
            print(f"Additional sources fetched: {sources_data}")
            # Shallow-copy organic items: _scrape_pages attaches `content`,
            # and we don't want that leaking into the captured serper_response.
            organic_data = [
                dict(it) for it in sources_data.get("organic", [])[:MAX_SOURCES]
            ]
            misc_data = sources_data.get("peopleAlsoAsk", [])
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
        # Optional market context. Absent or malformed -> blind mode, in which
        # the rendered prompt is byte-identical to the parent's plus this
        # tool's own reasoning-field instructions, and no market-derived
        # reasoning happens at all.
        market_context = _extract_market_context(kwargs.get("request_context"))
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

        # Delivered payload. The four standard mech fields are unchanged and
        # stay first: the trader pops exactly those and ignores everything
        # else, so the extras below are additive and non-breaking. Structured
        # outputs guarantee the object is flat and strict-json.loads-parseable,
        # so no reasoning prose can leak in.
        #
        # p_no is derived rather than taken from the model. The schema
        # validator tolerates 0.01 of drift, but the trader tests
        # `p_yes + p_no != 1` with exact float equality and treats a failure as
        # an invalid response, so a model that answers 0.7 / 0.31 would be
        # dropped. Deriving the complement makes the sum exact by construction.
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
                # taught to read them.
                "researchability": prediction.researchability,
                "evidence_quality": prediction.evidence_quality,
                # Echoed from the request by this code, never generated by the
                # model, so it cannot be hallucinated. Without it nothing
                # downstream can tell a market-aware delivery from a blind one
                # after the fact.
                "market_prob_seen": market_context.get("market_prob"),
            }
        )

        used_params: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if return_source_content:
            used_params["source_content"] = captured_source_content
        return result, prediction_prompt, None, counter_callback, used_params
