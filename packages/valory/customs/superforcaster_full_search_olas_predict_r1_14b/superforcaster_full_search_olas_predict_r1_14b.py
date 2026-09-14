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
"""Olas-Predict-R1-14B forecasting tool.

A sibling of `superforcaster_full_search` that keeps its evidence pipeline and
forecasting prompt and swaps the forecaster for Olas-Predict-R1-14B, a
fine-tuned DeepSeek-R1-Distill-Qwen-14B served from a self-hosted vLLM endpoint
(OpenAI-compatible, so the client differs only by `base_url`).

What differs from the parent: the client is pointed at the vLLM endpoint from
the KeyChain, the served model is fixed per tool name rather than taken from the
request, and the completion is stripped of the model's `<think>` block before
the JSON is parsed. Rationale and evaluation results are in the pull request.
"""

import functools
import json
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
                if retries_left[VLLM_SERVER_API_KEY] <= 0:
                    raise e
                retries_left[VLLM_SERVER_API_KEY] -= 1
                api_keys.rotate(VLLM_SERVER_API_KEY)
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
                        "error_type": type(e).__name__,
                    }
                )
                return error_json, "", None, None, None, api_keys

        return execute()

    return wrapper


# KeyChain services carrying the vLLM endpoint and its key. The KeyChain is the
# only config channel that reaches a component running as bytes published from
# IPFS, so the endpoint rides it alongside the key.
#
# Named for what they ARE -- a vLLM server URL and its key -- not for the
# prototype tool that happens to share the machine today. These names are the
# deployment contract: they must match the entries in each mech's 1Password
# `api-keys` item exactly, and a rename after deployment fails every delivery.
VLLM_SERVER_API_KEY = "vllm_server_api_key"
VLLM_SERVER_URL = "vllm_server_url"


class OpenAIClientManager:
    """Client context manager for OpenAI."""

    def __init__(self, api_key: str, base_url: str):  # noqa: DAR101
        """Initializes with the vLLM key and base URL"""
        self.api_key = api_key
        self.base_url = base_url
        self._client: Optional["OpenAIClient"] = None

    def __enter__(self) -> "OpenAIClient":
        """Initializes and returns LLM client."""
        self._client = OpenAIClient(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Closes the LLM client"""
        if self._client is not None:
            self._client.client.close()
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


class OpenAIResponse:
    """Response class."""

    def __init__(self, content: Optional[str] = None, usage: Optional[Usage] = None):
        """Initializes with content and usage class."""
        self.content = content
        self.usage = usage if usage is not None else Usage()


class OpenAIClient:
    """OpenAI Client"""

    def __init__(self, api_key: str, base_url: str):
        """Initializes a client bound to the authenticated vLLM endpoint.

        :param api_key: API key for the vLLM gateway.
        :param base_url: OpenAI-compatible vLLM endpoint URL.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def completions(
        self,
        model: str,
        messages: List = [],  # noqa: B006
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Any = None,
        max_tokens: Optional[float] = None,
    ) -> Optional[OpenAIResponse]:
        """Generate a completion from the specified LLM provider using the given model and messages."""
        response_provider = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            timeout=timeout if timeout is not None else 150,
            stop=None,
        )
        response = OpenAIResponse()
        response.content = response_provider.choices[0].message.content
        response.usage.prompt_tokens = response_provider.usage.prompt_tokens
        response.usage.completion_tokens = response_provider.usage.completion_tokens
        return response


def budget_tokens(text: str, model: str) -> int:
    """Token count for window budgeting, scaled for the tokeniser mismatch.

    :param text: the text to size.
    :param model: model name for tokeniser selection.
    :return: a deliberately conservative token count.
    """
    return int(count_tokens(text, model) * TOKENIZER_SAFETY_FACTOR) + 1


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    try:
        enc = encoding_for_model(model)
    except KeyError:
        from tiktoken import get_encoding  # pylint: disable=import-outside-toplevel

        enc = get_encoding("o200k_base")
    return len(enc.encode(text))


# The served model's context window (vLLM `--max-model-len`). The tools this
# package descends from target GPT-4.1, where prompt length is a non-issue; here
# the prompt AND the completion must fit in 8k together, so the two are budgeted
# against each other rather than capped independently.
MODEL_CONTEXT_WINDOW = 8192
# The chat template wraps the messages in tokens this tokeniser never sees.
CONTEXT_SAFETY_MARGIN = 256
# `count_tokens` uses tiktoken, which has no Qwen encoding and falls back to
# o200k_base -- so every budget here is an ESTIMATE in the wrong tokeniser.
# Measured against the endpoint's reported prompt_tokens on four prompt shapes:
#   english news 1.058 | unicode es+jp 1.078 | bare template 1.033
#   numeric/URL-heavy 1.158   <- market evidence is full of prices, dates, URLs
# At 1.158 a full prompt under-counts by ~740 tokens, far past the 256 margin,
# and the request is rejected with a 400. Scale the estimate instead of
# trusting it; 1.25 leaves headroom above the worst shape measured.
TOKENIZER_SAFETY_FACTOR = 1.25
# The requester supplies `max_tokens` (the mech forwards task_data), so it is
# untrusted in the same way `model` is. Left unclamped, a value near the window
# drives the evidence budget to zero and the tool forecasts on no evidence at
# all while still returning a normal-looking answer. The prompt keeps this
# floor; a request asking for more completion than that is capped.
MIN_PROMPT_BUDGET = 3000
# 2048, not the fleet's 4096. Issue #455 raised the fleet value so free-text
# completions are not truncated before the JSON, and the same reasoning applies
# to a <think> block -- but at 4096 only 4096 remain for the prompt, which the
# evidence block alone exceeds. Observed completions from this model run
# 394-523 tokens, so 2048 is roughly 4x the longest seen while leaving 6144 for
# the prompt.
DEFAULT_MODEL_SETTINGS = {
    "max_tokens": 2048,
    "temperature": 0,
}
# The Olas-Predict served model emits reasoning followed by JSON, often inside
# a markdown fence. The delivery must contain only the validated JSON object.
# Reasoning models emit a think block before the answer. Two shapes occur:
# `<think>...</think>{json}` when the model writes both tags, and a BARE
# `...</think>{json}` when the chat template already supplied the opener. The
# bare shape is the common one for DeepSeek-R1 templates, so matching only the
# paired form leaves the whole reasoning in place -- and the reasoning contains
# draft probabilities. Everything before the LAST `</think>` is dropped.
# Case-insensitive throughout: the tags come from the chat template, which is a
# deployment artifact we do not control. A template emitting <THINK> would make
# a case-sensitive guard silently no-op and put draft probabilities back in play.
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"^.*</think>\s*", re.DOTALL | re.IGNORECASE)
# NB no regex: a character class cannot match a nested object, so
# `{"p_yes": 0.8, ..., "meta": {"a": 1}}` -- a perfectly good forecast with one
# extra key -- would be skipped and the whole delivery lost to a null.
# raw_decode handles nesting and gives the object's true extent.
# Two wire names, one package: the code is identical for both platforms (the
# evaluation ran this same pipeline on Omen and Polymarket), and separate names
# let the two be scored, promoted and served independently.
TOOL_OMEN = "superforcaster_full_search_olas_predict_r1_14b_omen"
TOOL_POLYMARKET = "superforcaster_full_search_olas_predict_r1_14b_polymarket"
ALLOWED_TOOLS = [TOOL_OMEN, TOOL_POLYMARKET]

# vLLM --served-model-name (the SFT warm-start checkpoint; the server renamed it
# from `qwen-14b-sft` on 2026-09-14 and the old name now 404s), resolved from the
# tool rather than the request. The
# mech's `model` kwarg is requester-controlled (`task_data.get("model",
# params.default_model)`), and the benchmark tournament passes its own default,
# so honouring it would send this endpoint a checkpoint it does not serve. The
# requester picks the tool; the tool picks the model.
SERVED_MODEL = "olas-predict-r1-14b"
# Derived, not hand-listed: a third wire name added to ALLOWED_TOOLS without a
# matching entry would otherwise KeyError at delivery time rather than here.
MODEL_BY_TOOL = {tool: SERVED_MODEL for tool in ALLOWED_TOOLS}


def resolve_model(tool: str) -> str:
    """Return the vLLM served-model name for `tool`.

    :param tool: One of `ALLOWED_TOOLS`.
    :return: The served-model name to request from the vLLM endpoint.
    """
    return MODEL_BY_TOOL[tool]


# The question is interpolated TWICE into the prompt, so a long free-text
# prompt (Pearl sends the user's message verbatim) costs double. Capped so it
# can never crowd out the evidence entirely.
_MAX_QUESTION_TOKENS = 1200

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
# observed evidence sizes with headroom. Binding at 8k alongside the
# 1M context but bounds cost and guards against outlier pages.
MAX_EVIDENCE_TOKENS = 4000


PREDICTION_PROMPT = """
You are an advanced AI system which has been finetuned to provide calibrated probabilistic
forecasts under uncertainty, with your performance evaluated according to the Brier score. When
forecasting, do not treat 0.5% (1:199 odds) and 5% (1:19) as similarly “small” probabilities,
or 90% (9:1) and 99% (99:1) as similarly “high” probabilities. As the odds show, they are
markedly different, so output your probabilities accordingly.

Question:
{question}

Today's date: {today}
Your pretraining knowledge cutoff: October 2023

We have retrieved the following information for this question:
<background>{sources}</background>

Recall the question you are forecasting:
{question}

Instructions:
1. Compress key factual information from the sources, as well as useful background information
which may not be in the sources, into a list of core factual points to reference. Aim for
information which is specific, relevant, and covers the core considerations you'll use to make
your forecast. For this step, do not draw any conclusions about how a fact will influence your
answer or forecast. Place this section of your response in <facts></facts> tags.

2. Provide a few reasons why the answer might be no. Rate the strength of each reason on a
scale of 1-10. Use <no></no> tags.

3. Provide a few reasons why the answer might be yes. Rate the strength of each reason on a
scale of 1-10. Use <yes></yes> tags.

4. Aggregate your considerations. Do not summarize or repeat previous points; instead,
investigate how the competing factors and mechanisms interact and weigh against each other.
Factorize your thinking across (exhaustive, mutually exclusive) cases if and only if it would be
beneficial to your reasoning. We have detected that you overestimate world conflict, drama,
violence, and crises due to news' negativity bias, which doesn't necessarily represent overall
trends or base rates. Similarly, we also have detected you overestimate dramatic, shocking,
or emotionally charged news due to news' sensationalism bias. Therefore adjust for news'
negativity bias and sensationalism bias by considering reasons to why your provided sources
might be biased or exaggerated. Think like a superforecaster. Use <thinking></thinking> tags
for this section of your response.

5. Output an initial probability (prediction) as a single number between 0 and 1 given steps 1-4.
Use <tentative></tentative> tags.

6. Reflect on your answer, performing sanity checks and mentioning any additional knowledge
or background information which may be relevant. Check for over/underconfidence, improper
treatment of conjunctive or disjunctive conditions (only if applicable), and other forecasting
biases when reviewing your reasoning. Consider priors/base rates, and the extent to which
case-specific information justifies the deviation between your tentative forecast and the prior.
Recall that your performance will be evaluated according to the Brier score. Be precise with tail
probabilities. Leverage your intuitions, but never change your forecast for the sake of modesty
or balance alone. Finally, aggregate all of your previous reasoning and highlight key factors
that inform your final forecast. Use <thinking></thinking> tags for this portion of your response.

7. Output your final prediction (a number between 0 and 1 with an asterisk at the beginning and
end of the decimal) in <answer></answer> tags.


OUTPUT_FORMAT
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility".
* Each item in the JSON must have a value between 0 and 1.
   - "p_yes": Estimated probability that the event in the "Question" occurs.
   - "p_no": Estimated probability that the event in the "Question" does not occur.
   - "confidence": A value between 0 and 1 indicating the confidence in the prediction. 0 indicates lowest
     confidence value; 1 maximum confidence value.
   - "info_utility": Utility of the information provided in "sources" to help you make the prediction.
     0 indicates lowest utility; 1 maximum utility.
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object. Do not include any other contents in your response.
* This is incorrect:"```json{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}```"
* This is incorrect:```json"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"```
* This is correct:"{{\n  \"p_yes\": 0.2,\n  \"p_no\": 0.8,\n  \"confidence\": 0.7,\n  \"info_utility\": 0.5\n}}"
"""


def generate_prediction_with_retry(
    client: "OpenAIClient",
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    retries: int = COMPLETION_RETRIES,
    delay: int = COMPLETION_DELAY,
    counter_callback: Optional[Callable] = None,
) -> Tuple[Any, Optional[Callable]]:
    """Attempt to generate a prediction with retries on failure."""
    attempt = 0
    last_error: Optional[Exception] = None
    while attempt < retries:
        try:
            response = client.completions(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                timeout=90,
                stop=None,
            )

            # A refusal / empty completion yields content=None. Surface it as
            # an error (mirrors the calibrated sibling's `parsed is None`
            # guard) so the decorator's error JSON carries the real reason
            # instead of returning None as the prediction and tripping
            # json.loads(None) downstream in tournament scoring.
            if response is None or response.content is None:
                raise ValueError("Model returned no content (possible refusal)")

            if counter_callback is not None:
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=model,
                    token_counter=count_tokens,
                )

            return response.content, counter_callback
        except openai.RateLimitError as e:
            # Retry HERE first -- re-raising immediately would send every 429 to
            # with_key_rotation, which re-runs all of run() including the Serper
            # search and the page scrapes. But on exhaustion re-raise the
            # ORIGINAL RateLimitError rather than wrapping it: with_key_rotation
            # dispatches on that exact type, so a RuntimeError would silently
            # disable key rotation for the one case it exists to handle.
            print(f"Attempt {attempt + 1} rate-limited: {e}")
            time.sleep(delay)
            attempt += 1
            last_error = e
        except Exception as e:  # noqa: BLE001
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(delay)
            attempt += 1
            last_error = e
    if isinstance(last_error, openai.RateLimitError):
        raise last_error
    raise RuntimeError(
        f"Failed to generate prediction after retries: {last_error}"
    ) from last_error


def _coerce_unit_interval(value: Any, default: float = 0.5) -> float:
    """Coerce a value to a float in [0, 1], falling back to `default`."""
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if not 0.0 <= coerced <= 1.0:
        return default
    return coerced


def _json_objects(text: str) -> List[Dict[str, Any]]:
    """Every top-level JSON object in `text`, in order of appearance.

    :param text: text that may contain JSON objects among prose.
    :return: the decoded objects, nested values included.
    """
    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            return found
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        idx = end


def canonical_prediction(completion: Optional[str]) -> Optional[str]:
    """Build the delivery JSON from an Olas-Predict reasoning completion.

    :param completion: Raw completion from the Olas-Predict endpoint.
    :return: Clean prediction JSON, or None when no valid p_yes exists.
    """
    if not completion:
        return None
    # An opener with no closer means the completion was cut off mid-reasoning
    # (the token budget ran out). Everything present is therefore a DRAFT, and
    # harvesting one would deliver a working estimate as the final answer --
    # the same failure the think strip exists to prevent. There is no answer to
    # recover here, so return None and let the caller surface the error.
    if THINK_OPEN_RE.search(completion) and not THINK_CLOSE_RE.search(completion):
        return None
    # Walk the candidates from the END: if any reasoning survives the think
    # strip it carries draft probabilities, and the real answer is last.
    candidates = _json_objects(THINK_BLOCK_RE.sub("", completion))
    prediction = p_yes = None
    for parsed in reversed(candidates):
        try:
            p_yes = float(parsed["p_yes"])
        except (KeyError, TypeError, ValueError):
            continue
        prediction = parsed
        break
    if prediction is None or p_yes is None:
        return None
    if not 0.0 <= p_yes <= 1.0:
        return None
    return json.dumps(
        {
            "p_yes": p_yes,
            "p_no": round(1.0 - p_yes, 6),
            "confidence": _coerce_unit_interval(prediction.get("confidence")),
            "info_utility": _coerce_unit_interval(prediction.get("info_utility")),
        }
    )


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
        text = " ".join(words[:max_words]) + " […]"
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
    fetch / parse failure -- the caller falls back to the Serper snippet.

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
        print(
            f"[superforcaster_full_search_olas_predict_r1_14b] Failed to fetch {url}: {e}"
        )
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
                    f"[superforcaster_full_search_olas_predict_r1_14b] Scrape error for {item['link']}: {e}"
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
) -> "_CappedEvidence":
    """Render the evidence block, dropping trailing organic items until it fits.

    peopleAlsoAsk is dropped FIRST, then trailing organic items: Serper orders
    organic by relevance, and a scraped page is better evidence than a PAA
    snippet, so the cheaper material goes first. (The parent trims only organic
    and leaves peopleAlsoAsk alone -- safe on a 1M-token window, not on 8k.)
    The effective ceiling is min(MAX_EVIDENCE_TOKENS, the caller's window
    budget). If even an empty block overflows, the caller detects it via
    _evidence_is_exhausted rather than sending a doomed prompt. If the block still exceeds the budget once all
    organic items are gone, the result is returned as-is (peopleAlsoAsk is
    small and not separately trimmed).

    :param organic_data: Serper organic results (already capped to MAX_SOURCES).
    :param misc_data: Serper peopleAlsoAsk items.
    :param model: model name for tokeniser selection.
    :param max_tokens: target ceiling on the rendered block.
    :return: the rendered block plus how many organic / peopleAlsoAsk items survived.
    """
    # MAX_EVIDENCE_TOKENS still binds: the lost-in-the-middle rationale is about
    # how much evidence the model reads well, independent of how much the window
    # physically allows. The window budget is the other ceiling, whichever is
    # tighter.
    max_tokens = min(max_tokens, MAX_EVIDENCE_TOKENS)
    rendered = format_sources_data(organic_data, misc_data)
    if budget_tokens(rendered, model) <= max_tokens or not organic_data:
        return _CappedEvidence(rendered, len(organic_data), len(misc_data))

    # peopleAlsoAsk goes first. The parent trimmed only organic items, which is
    # safe on a 1M-token window but not on 8k: a large peopleAlsoAsk block would
    # otherwise evict every scraped page -- the better evidence -- and could
    # still overflow, leaving the request to be rejected outright.
    misc = list(misc_data)
    while (
        misc
        and budget_tokens(format_sources_data(organic_data, misc), model) > max_tokens
    ):
        misc.pop()
    trimmed = list(organic_data)
    while (
        trimmed
        and budget_tokens(format_sources_data(trimmed, misc), model) > max_tokens
    ):
        trimmed.pop()
    rendered = format_sources_data(trimmed, misc)
    rendered += "\n[… evidence truncated …]\n"
    return _CappedEvidence(rendered, len(trimmed), len(misc))


def _truncate_to_tokens(text: str, limit: int, model: str) -> str:
    """Cut `text` down to at most `limit` tokens, on a word boundary.

    :param text: the text to shorten.
    :param limit: maximum tokens to keep.
    :param model: model name for tokeniser selection.
    :return: the text, shortened if it was over the limit.
    """
    if budget_tokens(text, model) <= limit:
        return text
    words = text.split()
    while words and budget_tokens(" ".join(words), model) > limit:
        words = words[: int(len(words) * 0.9)] or words[:-1]
    return " ".join(words)


class _CappedEvidence(NamedTuple):
    """The capped evidence block plus what survived the cap.

    Counting here rather than string-matching the render: the markers are
    `format_sources_data`'s template, so a reword would silently flip an
    exhaustion check either way -- always-null or never-null, with no test
    coupling the two.
    """

    rendered: str
    organic_kept: int
    misc_kept: int

    @property
    def is_empty(self) -> bool:
        """Whether the cap left no evidence at all.

        :return: True when neither an organic nor a peopleAlsoAsk item survived.
        """
        return self.organic_kept == 0 and self.misc_kept == 0


def _evidence_budget(question: str, today: str, model: str, max_tokens: int) -> int:
    """Tokens left for the evidence block once everything else is placed.

    Rendering the prompt with empty sources prices the template and both copies
    of the question in one step, so the budget cannot drift from the template.

    :param question: the question as it will be interpolated.
    :param today: the date string as it will be interpolated.
    :param model: model name for tokeniser selection.
    :param max_tokens: completion budget reserved from the same window.
    :return: tokens available for evidence; may be zero.
    """
    skeleton = PREDICTION_PROMPT.format(question=question, today=today, sources="")
    return max(
        0,
        MODEL_CONTEXT_WINDOW
        - max_tokens
        - CONTEXT_SAFETY_MARGIN
        - budget_tokens(skeleton, model),
    )


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
# Deliberately case-sensitive (unlike the IGNORECASE _QUESTION_WORD_RE): a
# capitalized market verb marks a sentence-initial market question, and adding
# IGNORECASE here would double-count lowercase occurrences via the +1 bonus.
_MARKET_VERB_RE = re.compile(
    r"^(?:Will|Is|Are|Was|Were|Does|Do|Did|Which|Who|When|Whether)\b"
)
# Chars that may directly precede a sentence-initial question word: whitespace,
# sentence punctuation, ASCII quotes/paren, and typographic quotes.
_CLAUSE_BOUNDARY = " \t\n.!?:\"'(\u201c\u201d\u2018\u2019"
# Candidate scanning is bounded to the prompt head: every question-word
# occurrence starts a candidate and each candidate scans forward for '?', so
# an unbounded scan is quadratic. Measured cost is small at the mech's cap
# (~6.6ms unbounded at 100KB, the MAX_PROMPT_BYTES limit in the mech repo's
# valory/task_execution skill) but grows ~4x per 2x and benchmark/direct
# calls are not capped at all (multi-MB prompts reach seconds) -- the window
# is defence-in-depth for those paths. Market questions sit in the prompt
# head in practice (the longest observed production prompt is under 1KB), so
# a 10KB window loses nothing on real traffic.
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
    if not isinstance(raw.get("organic"), list):
        raise ValueError(
            f"{context}: Serper response missing or malformed 'organic' key; "
            f"got keys: {sorted(raw)[:8]}"
        )
    misc = raw.get("peopleAlsoAsk", [])
    if not isinstance(misc, list):
        raise ValueError(
            f"{context}: Serper response has a malformed 'peopleAlsoAsk' key; "
            f"got {type(misc).__name__}"
        )
    return raw["organic"][:MAX_SOURCES], misc


def _truncate_query(query: str) -> str:
    """Cap the query at _MAX_SEARCH_QUERY_LEN, cutting on a word boundary.

    :param query: the derived search query.
    :return: the query, truncated without a dangling partial word.
    """
    if len(query) <= _MAX_SEARCH_QUERY_LEN:
        return query
    cut = query[:_MAX_SEARCH_QUERY_LEN]
    if not query[_MAX_SEARCH_QUERY_LEN].isspace() and not cut.endswith(" "):
        cut = cut.rsplit(None, 1)[0] if " " in cut else cut
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
    scan_truncated: bool = False,
) -> MechResponse:
    """Build the flagged null prediction returned on empty retrieval.

    Unlike the with_key_rotation error null this is a VALID prediction
    (p_yes = p_no = 0.5) with zero confidence and info_utility, so the strict
    trader consumer still parses it (issue #455). The on-chain JSON carries
    only the four standard fields; the explicit marker for requesters lives in
    used_params["empty_retrieval"] (off-chain metadata.params), intended for
    off-chain consumers (not yet wired up -- the benchmark scorer's
    null-vs-forecast branch is a follow-up).

    :param model: the model name recorded in used_params.
    :param temperature: the temperature recorded in used_params.
    :param max_tokens: the max_tokens recorded in used_params.
    :param captured_source_content: the (empty) retrieval capture.
    :param return_source_content: whether to attach the capture to used_params.
    :param counter_callback: the cost callback, threaded back unchanged.
    :param context: why the null was produced; recorded unconditionally in
        used_params["null_reason"] so a skipped Serper call ("empty query")
        stays distinguishable from a genuine zero-hit ("live search").
    :param tier: the parse_prompt tier that produced the search query.
    :param scan_truncated: whether the scan window did not cover the whole
        prompt (any non-template tier; a template match returns before the
        window can matter).
    :return: the flagged-null MechResponse tuple.
    """
    print(
        f"[superforcaster_full_search_olas_predict_r1_14b] {context}: empty retrieval"
        " -- returning null prediction"
    )
    null_result = json.dumps(
        {"p_yes": 0.5, "p_no": 0.5, "confidence": 0.0, "info_utility": 0.0}
    )
    used_params: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "empty_retrieval": True,
        "null_reason": context,
        "parse_tier": tier,
        "scan_truncated": scan_truncated,
    }
    if return_source_content:
        used_params["source_content"] = captured_source_content
    return null_result, "", None, counter_callback, used_params


def _optional_key(api_keys: Any, service: str) -> Optional[str]:
    """Return the key for `service`, or None if the KeyChain lacks it.

    :param api_keys: KeyChain-like mapping of service names to values.
    :param service: Service name to retrieve.
    :return: The configured value, or None when absent.
    """
    try:
        return api_keys[service]
    except Exception:  # noqa: BLE001 - KeyChain raises various types when absent
        return None


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """Run the task"""
    tool = kwargs["tool"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    model = resolve_model(tool)

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

    api_keys = kwargs["api_keys"]
    # Key and endpoint are both REQUIRED: a default would hide a misconfigured
    # deployment behind a 401 or a connection error to the wrong host.
    llm_api_key = _optional_key(api_keys, VLLM_SERVER_API_KEY)
    if not llm_api_key:
        raise ValueError(
            f"No API key for the forecasting endpoint: set "
            f"'{VLLM_SERVER_API_KEY}' in the mech's API_KEYS."
        )
    endpoint = _optional_key(api_keys, VLLM_SERVER_URL)
    if not endpoint:
        raise ValueError(
            "No endpoint for the forecasting service: set "
            f"'{VLLM_SERVER_URL}' in the mech's API_KEYS."
        )
    source_content = kwargs.get("source_content", None)
    return_source_content = (
        kwargs["api_keys"].get("return_source_content", "false") == "true"
    )
    source_content_mode = kwargs["api_keys"].get("source_content_mode", "cleaned")
    if source_content_mode not in ("cleaned", "raw"):
        raise ValueError(
            f"Invalid source_content_mode: {source_content_mode!r}. Must be 'cleaned' or 'raw'."
        )
    with OpenAIClientManager(llm_api_key, endpoint) as llm_client:
        # Clamped, not trusted -- see MIN_PROMPT_BUDGET. `or` rather than a
        # get() default so an explicit null (the key present, value None) falls
        # back too instead of raising TypeError; max(1, ...) so 0 or a negative
        # is corrected here rather than burning three retries on a 400.
        _ceiling = MODEL_CONTEXT_WINDOW - MIN_PROMPT_BUDGET
        _requested = int(
            kwargs.get("max_tokens") or DEFAULT_MODEL_SETTINGS["max_tokens"]
        )
        max_tokens = max(1, min(_requested, _ceiling))
        if _requested != max_tokens:
            print(
                f"[{TOOL_OMEN.rsplit('_', 1)[0]}] max_tokens {_requested} "
                f"clamped to {max_tokens} (window {MODEL_CONTEXT_WINDOW}, "
                f"prompt floor {MIN_PROMPT_BUDGET})"
            )
        temperature = kwargs.get("temperature", DEFAULT_MODEL_SETTINGS["temperature"])
        prompt = kwargs["prompt"]

        today = date.today()
        d = today.strftime("%d/%m/%Y")

        question, search_query, tier = parse_prompt(prompt)
        # The scan window not covering the whole prompt is observable on its
        # own: even a clause-tier pick may have missed the real question
        # sitting past the window (not only the raw-tier no-clause case).
        # A template match is exempt: it returns the exact question before
        # the window plays any role, so nothing can have been missed.
        scan_truncated = tier != "template" and len(prompt) > _MAX_SCAN_CHARS
        if scan_truncated:
            print(
                f"[superforcaster_full_search_olas_predict_r1_14b] Scan window exhausted: "
                f"prompt is {len(prompt)} chars, scanned the first "
                f"{_MAX_SCAN_CHARS}; tier={tier}, query: {search_query!r}"
            )
        elif tier == "raw":
            print(
                "[superforcaster_full_search_olas_predict_r1_14b] No question clause found; "
                f"using capped prompt head as the search query: {search_query!r}"
            )
        elif tier == "clause":
            print(
                f"[superforcaster_full_search_olas_predict_r1_14b] Free-text prompt (tier={tier}); "
                f"derived search query: {search_query!r}"
            )

        # Free-text puts the whole user prompt in the question slot, twice. When
        # that does not fit, prefer the clause parse_prompt already identified as
        # the question over a truncation: in free text the question usually comes
        # last, so cutting the tail is what removes it.
        if budget_tokens(question, model) > _MAX_QUESTION_TOKENS:
            if tier != "raw":
                question = search_query
            question = _truncate_to_tokens(question, _MAX_QUESTION_TOKENS, model)
            print(
                f"[{TOOL_OMEN.rsplit('_', 1)[0]}] Question too long for the "
                f"{MODEL_CONTEXT_WINDOW}-token window; using tier={tier} "
                f"question: {question[:120]!r}"
            )
        evidence_budget = _evidence_budget(question, d, model, max_tokens)

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
                    scan_truncated=scan_truncated,
                )
            cached_pages = source_content.get("pages", {})
            cached_mode = source_content.get("mode", source_content_mode)
            _hydrate_organic_from_pages(organic_data, cached_pages, cached_mode)
            capped = _cap_evidence_block(
                organic_data, misc_data, model, evidence_budget
            )
            sources = capped.rendered
        else:
            if not any(ch.isalnum() for ch in search_query):
                # Nothing searchable: no alphanumeric character at all (empty,
                # whitespace, quotes, or bare punctuation) -- skip the wasted
                # Serper call and return the flagged null directly.
                return _flagged_null_result(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    captured_source_content=None,
                    return_source_content=return_source_content,
                    counter_callback=counter_callback,
                    context="empty query",
                    tier=tier,
                    scan_truncated=scan_truncated,
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
                    scan_truncated=scan_truncated,
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
            capped = _cap_evidence_block(
                organic_data, misc_data, model, evidence_budget
            )
            sources = capped.rendered

        # The budget can trim every item away -- a requester-supplied
        # `max_tokens`, or a question long enough to crowd out the block. The
        # empty-retrieval guard above cannot see this: it runs BEFORE trimming.
        # Without this the tool forecasts on nothing and returns a
        # normal-looking answer, which is the exact gap _flagged_null_result
        # exists to close.
        if (organic_data or misc_data) and capped.is_empty:
            return _flagged_null_result(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                captured_source_content=captured_source_content,
                return_source_content=return_source_content,
                counter_callback=counter_callback,
                context="evidence budget exhausted",
                tier=tier,
                scan_truncated=scan_truncated,
            )

        print("Updating prompt...")
        prediction_prompt = PREDICTION_PROMPT.format(
            question=question, today=d, sources=sources
        )
        print(f"\n{prediction_prompt=}\n")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prediction_prompt},
        ]
        print("Getting prompt response...")
        completion, counter_callback = generate_prediction_with_retry(
            client=llm_client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=COMPLETION_RETRIES,
            delay=COMPLETION_DELAY,
            counter_callback=counter_callback,
        )
        extracted_block = canonical_prediction(completion)
        if extracted_block is None:
            raise ValueError("Model output did not contain a parseable p_yes.")

        # How much evidence the model actually saw. At a 1M window trimming was
        # rare and the parent could omit this; at 8k under a 4000-token ceiling
        # it is routine, and a one-source forecast is a different thing from a
        # five-source one. The only other trace is the truncation marker inside
        # the prompt, which never reaches the delivery.
        used_params = {
            "sources_used": capped.organic_kept + capped.misc_kept,
            "sources_dropped": (
                (len(organic_data) + len(misc_data))
                - (capped.organic_kept + capped.misc_kept)
            ),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "parse_tier": tier,
            "scan_truncated": scan_truncated,
        }
        if return_source_content:
            used_params["source_content"] = captured_source_content
        return extracted_block, prediction_prompt, None, counter_callback, used_params
