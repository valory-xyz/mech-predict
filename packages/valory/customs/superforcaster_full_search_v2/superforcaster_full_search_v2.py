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
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai
import requests
from markdownify import markdownify as md
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


class PredictionResult(BaseModel):
    """Superforecaster structured output.

    The text fields carry the reasoning chain (facts, pros/cons, word-mention
    screen, aggregation, reflection). Only the four numeric fields are returned
    on-chain per the mech protocol.
    """

    facts: str = Field(
        ...,
        description=(
            "Core factual points compiled from the sources and relevant "
            "background. Specific, relevant, no conclusions about how a "
            "fact influences the forecast."
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
    word_mention_check: str = Field(
        ...,
        description=(
            "WORD-MENTION MARKET SCREEN. Write 'Not applicable.' if the question "
            "does not ask whether a specific person will say/use a specific "
            "word/phrase during a named event or time window. If it IS a "
            "word-mention market: (a) identify and exclude circular prediction-market "
            "evidence; (b) flag whether evidence is event-specific or from a past "
            "occasion; (c) apply event-specific base rates; (d) check for event "
            "disruption signals. Synthesise into an adjusted probability estimate."
        ),
    )
    aggregation: str = Field(
        ...,
        description=(
            "Aggregate considerations. Weigh competing factors, apply the "
            "CALIBRATION block (state a base rate, adjust using specific evidence, "
            "treat missing confirmation as NO), adjust for news negativity and "
            "sensationalism bias. If word_mention_check is applicable, anchor on its "
            "adjusted estimate. End by stating a tentative probability in [0,1]."
        ),
    )
    reflection: str = Field(
        ...,
        description=(
            "Sanity checks and finalisation. Apply: EVIDENCE BAR (p_yes > 0.80 "
            "needs strong specific evidence; plans/proposals are not actions), "
            "MARKET CIRCULARITY (prediction-market prices are circular; form your "
            "own view), COUNT THRESHOLDS (for N+ requirements, apply count-specific "
            "base rates). Check for over/underconfidence; highlight key factors."
        ),
    )
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
            "Utility of the information in the sources to inform the prediction "
            "(0 = lowest, 1 = highest)."
        ),
    )

    @model_validator(mode="after")
    def _check_p_yes_p_no_sum(self) -> "PredictionResult":
        """Validate that p_yes + p_no == 1."""
        if abs(self.p_yes + self.p_no - 1.0) > 0.01:
            raise ValueError(
                f"p_yes + p_no must equal 1 (got {self.p_yes} + {self.p_no} = "
                f"{self.p_yes + self.p_no})"
            )
        return self


class StandardPredictionResult(BaseModel):
    """Superforecaster structured output for non-word-mention markets.

    Used when the market question does NOT ask whether a specific person will
    say/use a specific word or phrase during a named event. Identical to
    PredictionResult but without the word_mention_check field, so the model
    is not forced to apply a WM screen that is inapplicable and suppresses
    p_yes on ordinary markets.
    """

    facts: str = Field(
        ...,
        description=(
            "Core factual points compiled from the sources and relevant "
            "background. Specific, relevant, no conclusions about how a "
            "fact influences the forecast."
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
    aggregation: str = Field(
        ...,
        description=(
            "Aggregate considerations. Weigh competing factors, apply the "
            "CALIBRATION block (state a base rate, adjust using specific evidence, "
            "treat missing confirmation as NO), adjust for news negativity and "
            "sensationalism bias. End by stating a tentative probability in [0,1]."
        ),
    )
    reflection: str = Field(
        ...,
        description=(
            "Sanity checks and finalisation. Apply: EVIDENCE BAR (p_yes > 0.80 "
            "needs strong specific evidence; plans/proposals are not actions), "
            "MARKET CIRCULARITY (prediction-market prices are circular; form your "
            "own view), COUNT THRESHOLDS (for N+ requirements, apply count-specific "
            "base rates). Check for over/underconfidence; highlight key factors."
        ),
    )
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
            "Utility of the information in the sources to inform the prediction "
            "(0 = lowest, 1 = highest)."
        ),
    )

    @model_validator(mode="after")
    def _check_p_yes_p_no_sum(self) -> "StandardPredictionResult":
        """Validate that p_yes + p_no == 1."""
        if abs(self.p_yes + self.p_no - 1.0) > 0.01:
            raise ValueError(
                f"p_yes + p_no must equal 1 (got {self.p_yes} + {self.p_no} = "
                f"{self.p_yes + self.p_no})"
            )
        return self


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
                # Return a parseable null-prediction JSON so downstream
                # tournament scoring sees an explicit error rather than
                # treating a raw exception string as a prediction.
                error_json = json.dumps(
                    {
                        "p_yes": None,
                        "p_no": None,
                        "confidence": 0.0,
                        "info_utility": 0.0,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                return error_json, "", None, None, None, api_keys

        return execute()

    return wrapper


class OpenAIClientManager:
    """Client context manager for OpenAI."""

    def __init__(self, api_key: str):
        """Initializes with API keys"""
        self.api_key = api_key
        self._client: Optional["openai.OpenAI"] = None

    @property
    def client(self) -> "openai.OpenAI":
        """Return the underlying OpenAI client."""
        if self._client is None:
            raise RuntimeError("OpenAIClientManager not entered")
        return self._client

    def __enter__(self) -> "OpenAIClientManager":
        """Initializes the LLM client and returns self."""
        self._client = openai.OpenAI(api_key=self.api_key)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Closes the LLM client"""
        if self._client is not None:
            self._client.close()
            self._client = None


class Usage:
    """Usage class."""

    def __init__(
        self,
        prompt_tokens: Optional[Any] = None,
        completion_tokens: Optional[Any] = None,
    ):
        """Initializes with prompt tokens and completion tokens."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    try:
        enc = encoding_for_model(model)
    except KeyError:
        from tiktoken import get_encoding  # pylint: disable=import-outside-toplevel

        enc = get_encoding("o200k_base")
    return len(enc.encode(text))


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 2000,
    "temperature": 0,
}
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"
ALLOWED_TOOLS = ["superforcaster_full_search_v2"]
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

# Cap on the rendered <background> evidence block to bound prompt size.
MAX_EVIDENCE_TOKENS = 4000

# Word-mention market detection: matches questions asking whether a specific
# person will say/use/mention a specific quoted word or phrase at an event.
# Used to dispatch between PredictionResult (WM screen included) and
# StandardPredictionResult (WM screen omitted).
_WM_PATTERN = re.compile(
    r"""
    (?:
        # Explicit "the word/phrase/term" language
        \b(?:say|use|mention|utter)s?\s+(?:the\s+)?(?:word|phrase|term)s?\b
        |
        # Quoted word/phrase directly after say/mention/use/utter
        \b(?:say|says|mention|mentions|use|uses|utter|utters)\s+['"][^'"]{1,80}['"]
        |
        # "N times" pattern with a specific word: "say X at least N times"
        \b(?:say|mention|use|utter)\s+\S+\s+(?:at\s+least\s+)?\d+\s+times?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_word_mention_market(question: str) -> bool:
    """Return True if the question is a word-mention market.

    Word-mention markets ask whether a specific person will say, use, or
    mention a specific quoted word or phrase during a named event or time
    window (e.g. "Will X say 'tariff' at the debate?"). These require a
    dedicated base-rate reasoning step; regular markets do not.

    :param question: the market question string.
    :return: True if the question matches the word-mention pattern.
    """
    return bool(_WM_PATTERN.search(question))


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

Return a structured PredictionResult whose fields capture the following reasoning chain:

1. `facts` - Compress key factual information from the sources, as well as useful background
information which may not be in the sources, into a list of core factual points. Aim for
information which is specific, relevant, and covers the core considerations for your forecast.
For this step, do not draw any conclusions about how a fact will influence your forecast.

2. `reasons_no` - Provide a few reasons why the answer might be no. Rate the strength of each
reason on a scale of 1-10.

3. `reasons_yes` - Provide a few reasons why the answer might be yes. Rate the strength of each
reason on a scale of 1-10.

4. `word_mention_check` - WORD-MENTION MARKET SCREEN:

First, determine whether this question asks whether a specific speaker will say or use a
specific word or phrase during a named event or time window (e.g., "Will X say [WORD] during
[EVENT]?", "Will [COMPANY] say [PHRASE] on the [DATE] earnings call?", "Will [WORD] be in
the headlines this week?").

If NOT a word-mention market: write "Not applicable."

If YES, it IS a word-mention market, complete all four sub-steps:

(a) CIRCULAR EVIDENCE FILTER: Identify any sources that are prediction-market pages
(Polymarket, Kalshi, Metaculus, PredictIt, or similar). These pages show current market
odds -- they are circular self-reference, not independent forecasting evidence. List which
sources (by title/URL) are circular and exclude their implied probabilities from your
estimate.

(b) EVIDENCE CURRENCY: For each article in the background, determine whether it describes
THIS SPECIFIC target event (evidence IS event-specific) or a different or past occasion
(evidence IS NOT event-specific). A speaker mentioning a word at a past speech or interview
is weak evidence that they will mention it at THIS specific event.

(c) EVENT-SPECIFIC BASE RATE: Topical relevance ("X is in the news") is not the same as
literal mention at this specific event. Apply:
- Common word in speaker's core topic area (e.g., "economy" for a Fed chair): ~55-70%.
- Specific but frequent phrase from the speaker's known rhetorical style: ~30-50%.
- Niche or unusual word/phrase, or one tied to a specific context absent here: ~10-30%.
- Any count threshold of N+2 or more (e.g., "3+ times"): halve the rate for each N
  above 1 (so "3+ times" gets 0.5x of the single-mention base rate).
State which category applies and give the resulting base rate.

(d) EVENT DISRUPTION: If evidence indicates the target event was cancelled, disrupted, or
that the speaker did not give the expected remarks, treat this as a strong NO signal unless
another source specifically confirms remarks were delivered.

Synthesise (a)-(d) into an adjusted probability estimate for this step.

5. `aggregation` - Aggregate your considerations. Do not summarize or repeat previous points;
instead, investigate how the competing factors and mechanisms interact and weigh against each
other. Adjust for news negativity bias and sensationalism bias.

CALIBRATION (mandatory before stating a tentative probability):
- State a base-rate probability for this event category and justify it.
- If word_mention_check is applicable, start from the adjusted estimate computed in step 4.
- Adjust from the base rate using specific evidence only.
- Missing expected evidence (no announcement found, no confirmation found) is a NO signal.

End the `aggregation` field by stating an initial tentative probability (a single number
between 0 and 1).

6. `reflection` - Reflect on your tentative answer, performing sanity checks. Check for
over/underconfidence, improper treatment of conjunctive or disjunctive conditions, and other
forecasting biases. Be precise with tail probabilities.

BEFORE FINAL ANSWER -- apply all three checks:

1. EVIDENCE BAR: If sources confirm the event already occurred, high p_yes is fine.
   If not: p_yes > 0.90 needs verified commitment (signed, awarded, published, confirmed).
   p_yes > 0.80 needs strong specific evidence, not plausibility or reputation alone.
   Plans, proposals, and intentions are not completed actions.

2. MARKET CIRCULARITY: If your best independent evidence is actually a prediction-market
   price (Polymarket, Kalshi, etc.), do not let it anchor your p_yes. Those prices embed
   others' estimates; form your own view from the underlying evidence.

3. COUNT THRESHOLDS: For "N+ times" requirements, verify that the speaker routinely reaches
   that count at this type of event. A large gap between typical frequency and N overrides
   general sentiment-based forecasts.

Highlight the key factors that inform the final forecast.
"""


STANDARD_PREDICTION_PROMPT = """
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

Return a structured response whose fields capture the following reasoning chain:

1. `facts` - Compress key factual information from the sources, as well as useful background
information which may not be in the sources, into a list of core factual points. Aim for
information which is specific, relevant, and covers the core considerations for your forecast.
For this step, do not draw any conclusions about how a fact will influence your forecast.

2. `reasons_no` - Provide a few reasons why the answer might be no. Rate the strength of each
reason on a scale of 1-10.

3. `reasons_yes` - Provide a few reasons why the answer might be yes. Rate the strength of each
reason on a scale of 1-10.

4. `aggregation` - Aggregate your considerations. Do not summarize or repeat previous points;
instead, investigate how the competing factors and mechanisms interact and weigh against each
other. Adjust for news negativity bias and sensationalism bias.

CALIBRATION (mandatory before stating a tentative probability):
- State a base-rate probability for this event category and justify it.
- Adjust from the base rate using specific evidence only.
- Missing expected evidence (no announcement found, no confirmation found) is a NO signal.

End the `aggregation` field by stating an initial tentative probability (a single number
between 0 and 1).

5. `reflection` - Reflect on your tentative answer, performing sanity checks. Check for
over/underconfidence, improper treatment of conjunctive or disjunctive conditions, and other
forecasting biases. Be precise with tail probabilities.

BEFORE FINAL ANSWER -- apply all three checks:

1. EVIDENCE BAR: If sources confirm the event already occurred, high p_yes is fine.
   If not: p_yes > 0.90 needs verified commitment (signed, awarded, published, confirmed).
   p_yes > 0.80 needs strong specific evidence, not plausibility or reputation alone.
   Plans, proposals, and intentions are not completed actions.

2. MARKET CIRCULARITY: If your best independent evidence is actually a prediction-market
   price (Polymarket, Kalshi, etc.), do not let it anchor your p_yes. Those prices embed
   others' estimates; form your own view from the underlying evidence.

3. COUNT THRESHOLDS: For "N+ times" requirements, verify that the speaker routinely reaches
   that count at this type of event. A large gap between typical frequency and N overrides
   general sentiment-based forecasts.

Highlight the key factors that inform the final forecast.
"""


def _parse_completion(
    client: "openai.OpenAI",
    model: str,
    messages: List[Dict[str, str]],
    response_format: Any,
    temperature: float = 0,
    max_tokens: int = 2000,
    retries: int = COMPLETION_RETRIES,
    delay: int = COMPLETION_DELAY,
    counter_callback: Optional[Callable] = None,
) -> Tuple[Any, Optional[Callable]]:
    """Call OpenAI Structured Outputs and parse into a Pydantic model.

    Uses client.beta.chat.completions.parse which guarantees the response
    conforms to the supplied Pydantic schema.

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
    :raises RuntimeError: if all retries exhausted without a successful parse.
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
            )
            choice = response.choices[0]
            if choice.message.refusal:
                raise ValueError(f"Model refused to answer: {choice.message.refusal}")
            parsed = choice.message.parsed
            if parsed is None:
                raise ValueError("Model returned no structured content")

            if counter_callback is not None:
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=model,
                    token_counter=count_tokens,
                )

            return parsed, counter_callback
        except Exception as e:  # noqa: BLE001
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(delay)
            attempt += 1
            last_error = e
    raise RuntimeError(
        f"Failed to generate prediction after {retries} attempts: {last_error}"
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
        print(f"[superforcaster_full_search_v2] Failed to fetch {url}: {e}")
        return None, None


def _scrape_pages(
    organic_data: List[Dict[str, Any]],
    mode: str,
    max_pages: int = MAX_PAGES_TO_SCRAPE,
) -> Dict[str, str]:
    """Concurrently scrape the top organic links and attach `content` in place.

    :param organic_data: Serper organic-result dicts (mutated in place to
        add a ``content`` key on successful scrapes).
    :param mode: ``"cleaned"`` stores extracted text; ``"raw"`` stores HTML.
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
                    f"[superforcaster_full_search_v2] Scrape error for {item['link']}: {e}"
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


SYSTEM_PROMPT = "You are a helpful assistant."


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
                "A delivery rate of `0` was passed, but no counter callback "
                "was given to calculate the max cost with."
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

        # Dispatch: word-mention markets get the WM screen (PredictionResult);
        # all other markets use the leaner schema (StandardPredictionResult).
        is_wm_market = _is_word_mention_market(question)
        if is_wm_market:
            prediction_prompt = PREDICTION_PROMPT.format(
                question=question, today=d, sources=sources
            )
            response_format: Any = PredictionResult
        else:
            prediction_prompt = STANDARD_PREDICTION_PROMPT.format(
                question=question, today=d, sources=sources
            )
            response_format = StandardPredictionResult

        print(
            f"[superforcaster_full_search_v2] market_type={'wm' if is_wm_market else 'standard'}"
        )
        print(f"\n{prediction_prompt=}\n")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prediction_prompt},
        ]
        print("Getting structured prediction response...")
        prediction: Union[PredictionResult, StandardPredictionResult]
        prediction, counter_callback = _parse_completion(
            client=llm_client.client,
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            counter_callback=counter_callback,
        )

        print(
            f"[superforcaster_full_search_v2] p_yes={prediction.p_yes}, "
            f"p_no={prediction.p_no}, confidence={prediction.confidence}, "
            f"info_utility={prediction.info_utility}"
        )
        if is_wm_market and hasattr(prediction, "word_mention_check"):
            wm_preview = prediction.word_mention_check[:120]  # type: ignore[union-attr]
            print(f"[superforcaster_full_search_v2] word_mention_check={wm_preview}")

        result = json.dumps(
            {
                "p_yes": prediction.p_yes,
                "p_no": prediction.p_no,
                "confidence": prediction.confidence,
                "info_utility": prediction.info_utility,
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
