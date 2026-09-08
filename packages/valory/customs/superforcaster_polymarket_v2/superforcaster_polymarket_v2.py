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
    def wrapper(*args: Any, **kwargs: Any) -> MechResponseWithKeys:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponseWithKeys:
            """Retry the function with a new key."""
            try:
                result: MechResponse = func(*args, **kwargs)
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
            except Exception as e:
                return str(e), "", None, None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


class OpenAIClientManager:
    """Client context manager for OpenAI."""

    def __init__(self, api_key: str):
        """Initializes with API keys"""
        self.api_key = api_key
        self._client: Optional["OpenAIClient"] = None

    def __enter__(self) -> "OpenAIClient":
        """Initializes and returns LLM client."""
        self._client = OpenAIClient(api_key=self.api_key)
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
        self.usage = Usage()


class OpenAIClient:
    """OpenAI Client"""

    def __init__(self, api_key: str):
        """Initializes with API keys and client."""
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)

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
            timeout=150,
            stop=None,
        )
        response = OpenAIResponse()
        response.content = response_provider.choices[0].message.content
        response.usage.prompt_tokens = response_provider.usage.prompt_tokens
        response.usage.completion_tokens = response_provider.usage.completion_tokens
        return response


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "limit_max_tokens": 4096,
    "temperature": 0,
}
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"
ALLOWED_TOOLS = ["superforcaster-polymarket-v2"]
ALLOWED_MODELS = [DEFAULT_OPENAI_MODEL]
MAX_SOURCES = 5
COMPLETION_RETRIES = 3
COMPLETION_DELAY = 2


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
Critically: identify the exact resolution criterion -- the specific condition that must literally
be true for this market to resolve YES. For each piece of evidence, ask: does this evidence
bear on whether the criterion itself will be satisfied, or does it only establish that the topic
is relevant? Topical relevance is not criterion evidence. Derive p_yes by reasoning explicitly
about the probability that the criterion is satisfied: consider the base rate for this type of
condition, any case-specific factors that raise or lower that probability, and whether any
evidence directly confirms or denies criterion satisfaction. Ground your estimate in that
reasoning chain -- do not substitute topical signal strength for a criterion probability.
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

            if (
                response
                and response.content is not None
                and counter_callback is not None
            ):
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=model,
                    token_counter=count_tokens,
                )

            content = response.content if response else None
            return content, counter_callback
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(delay)
            attempt += 1
    raise Exception("Failed to generate prediction after retries")


def fetch_additional_sources(question: Any, serper_api_key: Any) -> requests.Response:
    """Fetches additional sources for the given question using the Serper API."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": question})
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response


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

    A VALID prediction (p_yes = p_no = 0.5) with zero confidence and
    info_utility, so a requester can detect and discount it while the strict
    trader consumer still parses it (issue #455). The on-chain JSON carries
    only the four standard fields; the explicit marker for requesters lives
    in used_params["empty_retrieval"] (off-chain metadata.params).

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
        f"[superforcaster-polymarket-v2] {context}: empty retrieval"
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

        question, search_query, tier = parse_prompt(prompt)
        # The scan window not covering the whole prompt is observable on its
        # own: even a clause-tier pick may have missed the real question
        # sitting past the window (not only the raw-tier no-clause case).
        # A template match is exempt: it returns the exact question before
        # the window plays any role, so nothing can have been missed.
        scan_truncated = tier != "template" and len(prompt) > _MAX_SCAN_CHARS
        if scan_truncated:
            print(
                f"[superforcaster-polymarket-v2] Scan window exhausted: "
                f"prompt is {len(prompt)} chars, scanned the first "
                f"{_MAX_SCAN_CHARS}; tier={tier}, query: {search_query!r}"
            )
        elif tier == "raw":
            print(
                "[superforcaster-polymarket-v2] No question clause found; "
                f"using capped prompt head as the search query: {search_query!r}"
            )
        elif tier == "clause":
            print(
                f"[superforcaster-polymarket-v2] Free-text prompt (tier={tier}); "
                f"derived search query: {search_query!r}"
            )

        if source_content is not None:
            print("Using provided source content (cached replay)...")
            captured_source_content = source_content
            serper_data = source_content.get("serper_response", source_content)
            organic_data, misc_data = _shape_serper_sources(
                serper_data, "cached replay"
            )
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
            sources = format_sources_data(organic_data, misc_data)
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
            serper_response = fetch_additional_sources(search_query, serper_api_key)
            # Raise on a 4xx/5xx error body (credit / auth error) instead of
            # calling .json() on it and feeding the model an empty <background>
            # block that looks like a healthy run (matches the fleet pattern).
            serper_response.raise_for_status()
            sources_data = serper_response.json()
            # mode tag included for consistency across tools; content is identical
            # regardless of mode since Serper returns structured JSON, not HTML
            captured_source_content = {
                "mode": source_content_mode,
                "serper_response": sources_data,
            }
            print(f"Additional sources fetched: {sources_data}")
            organic_data, misc_data = _shape_serper_sources(sources_data, "live search")
            if not organic_data and not misc_data:
                return _flagged_null_result(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    captured_source_content=captured_source_content,
                    return_source_content=return_source_content,
                    counter_callback=counter_callback,
                    context="live search",
                    tier=tier,
                    scan_truncated=scan_truncated,
                )
            print("Formating sources...")
            sources = format_sources_data(organic_data, misc_data)

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
        extracted_block, counter_callback = generate_prediction_with_retry(
            client=llm_client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=COMPLETION_RETRIES,
            delay=COMPLETION_DELAY,
            counter_callback=counter_callback,
        )

        used_params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "parse_tier": tier,
            "scan_truncated": scan_truncated,
        }
        if return_source_content:
            used_params["source_content"] = captured_source_content
        return extracted_block, prediction_prompt, None, counter_callback, used_params
