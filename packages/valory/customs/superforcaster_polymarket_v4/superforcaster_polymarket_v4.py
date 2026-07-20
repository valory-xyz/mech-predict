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
"""Contains the job definitions.

What superforcaster-polymarket-v4 does (vs superforcaster-polymarket-v1)
-----------------------------------------------------------------------
v4 adds a mandatory evidence-reliability screen at PREDICTION_PROMPT step 4
(gate-visible: downstream of source_content injection, exercised by PR-CI's
cached replay) and raises max_tokens from 500 to 1500 so the full
chain-of-thought executes before the JSON is emitted. It targets systematic
overconfident-YES on Polymarket (high Brier in the >=0.9 p_yes bucket). One
coherent mechanism -- evidence classification before probability formation --
via four sub-steps:

(4a) Prediction-market-odds filter: discard circular self-referential odds
     embedded in sources (polymarket.com, metaculus, manifold, predictit,
     kalshi); they are the price of the market being resolved, not evidence.

(4b) Forward-looking intent discount: apply a 40-60% materialization discount
     to intent/expectation language ("is set to", "is expected to", "plans to",
     "scheduled to") -- announced intent, not a completed fact.

(4c) Temporal-evidence classification: TYPE A (within-window dated) vs TYPE B
     (perennial/undated standing pages), with a base-rate fallback when all
     evidence is TYPE B. Fixes "X in headlines this week" markets that scored
     p_yes=0.99 on standing pages and resolved NO (20.9% of issue #374 W-1
     Brier mass).

(4d) Criterion-specificity check: require TYPE A evidence that directly confirms
     the exact resolution criterion (an exact phrase said at a named event, a
     word in a named outlet's headline), not mere topic salience (32.2% of
     issue #374 W-1 Brier mass).

Consolidates the abandoned superforcaster-polymarket v4/v5 PR (#375) -- based
off a stale branch, never merged -- into a single v4 off v1.
"""

import functools
import json
import re
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    # Raised from 500 to 1500 (in v4) to allow the full chain-of-thought reasoning
    # (steps 1-6) to execute before the final JSON is emitted.
    "max_tokens": 1500,
    "limit_max_tokens": 4096,
    "temperature": 0,
}
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"
ALLOWED_TOOLS = ["superforcaster-polymarket-v4"]
ALLOWED_MODELS = [DEFAULT_OPENAI_MODEL]
MAX_SOURCES = 5
COMPLETION_RETRIES = 3
COMPLETION_DELAY = 2


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

4. Before aggregating, apply the following evidence-reliability screen -- this step is
mandatory and must be completed before you form your tentative probability:

  (a) Prediction-market-odds filter: check whether any snippet or URL in your sources
  comes from a prediction market platform (e.g. polymarket.com, metaculus.com,
  manifold.markets, predictit.org, kalshi.com). If so, the number shown (e.g. "92%",
  "Leading 90%") is that market's current trading price -- it is circular self-referential
  evidence for the very market being resolved here. Discard it entirely from your
  reasoning; do not treat it as an independent probability estimate.

  (b) Forward-looking intent discount: identify any evidence that describes a future
  action using intent or expectation language ("is set to", "is expected to", "plans to",
  "will reportedly", "confirmed to attend", "scheduled to", "is poised to", "is due to").
  Such language describes an announced intention or external expectation, not a completed
  fact. Apply a materialization discount: treat the described outcome as having only a
  40-60% base-rate probability of actually occurring as planned, unless you have strong
  specific evidence of higher or lower rates for this category of event.

  (c) Temporal-evidence filter: note today's date ({today}) and the resolution window
  implied by the question (any explicit deadline, "this week", "by [date]", or similar
  phrasing). For each piece of evidence, check its publication date if visible. Classify
  each source as one of:
    - TYPE A (within-window): explicitly dated within the resolution window, OR directly
      states that the resolution criterion was met or not met during that window.
    - TYPE B (background): undated, dated outside the window, or a standing page
      describing permanent attributes of the subject (stock tickers, official newsrooms,
      category pages, search-engine result pages, aggregators).
  State how many sources are TYPE A and how many are TYPE B. If ALL sources are TYPE B
  with no TYPE A evidence, information utility is low. In that case anchor your estimate
  on the historical base rate for this question category rather than on the subject's
  general prominence. For "X in headlines this week"-style markets, the base rate for
  YES resolution is 20-40% even for prominent subjects, because headline-aggregator
  resolution criteria are narrower than general web presence.

  (d) Criterion-specificity check: re-read the exact resolution condition stated in the
  question. Ask explicitly: does any TYPE A evidence directly confirm that the resolution
  condition was satisfied (e.g., the named person said the exact phrase, the specific
  word appeared in a named outlet's headline, a numerical threshold was crossed on a
  specific date), as opposed to merely confirming the topic is active or the subject is
  newsworthy? If no TYPE A evidence directly confirms criterion satisfaction, apply an
  additional uncertainty factor toward the base rate.

  After completing (a), (b), (c), and (d), aggregate your remaining considerations.
  Investigate how the competing factors and mechanisms interact and weigh against each
  other. We have detected that you overestimate world conflict, drama, violence, and
  crises due to news' negativity bias, which doesn't necessarily represent overall trends
  or base rates. Similarly, we also have detected you overestimate dramatic, shocking,
  or emotionally charged news due to news' sensationalism bias. Therefore adjust for
  news' negativity bias and sensationalism bias by considering reasons to why your
  provided sources might be biased or exaggerated. Think like a superforecaster. Use
  <thinking></thinking> tags for this section of your response.

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
After completing ALL reasoning steps 1-6 above, output a JSON object as the final part
of your response. The JSON must be parseable by Python's "json.loads()".
* The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility".
* Each item in the JSON must have a value between 0 and 1.
   - "p_yes": Estimated probability that the event in the "Question" occurs.
   - "p_no": Estimated probability that the event in the "Question" does not occur.
   - "confidence": A value between 0 and 1 indicating the confidence in the prediction. 0 indicates lowest
     confidence value; 1 maximum confidence value.
   - "info_utility": Utility of the information provided in "sources" to help you make the prediction.
     0 indicates lowest utility; 1 maximum utility.
* The sum of "p_yes" and "p_no" must equal 1.
* The JSON must be the last thing in your response (after all reasoning tags).
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


def extract_question(prompt: str) -> str:
    """Uses regexp to extract question from the prompt"""
    # Match from 'question "' to '" and the `yes`' to handle nested quotes
    pattern = r'question\s+"(.+?)"\s+and\s+the\s+`yes`'
    try:
        question = re.findall(pattern, prompt, re.DOTALL)[0]
    except Exception as e:
        print(f"Error extracting question: {e}")
        question = prompt
    return question


def extract_result_json(content: Optional[str]) -> str:
    """Return the trailing JSON object as a compact, json.loads-parseable string.

    v4's prompt lets the model emit reasoning prose (facts/thinking/answer tags)
    before the final JSON object, so the raw completion is not itself
    json.loads-parseable and the strict trader consumer rejects it. Extract the
    last balanced ``{...}`` block (the prediction) and re-serialize it compact,
    mirroring ``finetuned_prediction.canonical_prediction``. On any failure
    return a parseable null-prediction JSON so the on-chain result never breaks
    the trader's flat ``json.loads`` (mirrors
    ``superforcaster_calibrated_full_search``).

    :param content: the raw model completion (prose + trailing JSON), or None.
    :return: a compact JSON string that ``json.loads`` parses to the prediction.
    """
    null_prediction = json.dumps(
        {"p_yes": 0.5, "p_no": 0.5, "confidence": 0.0, "info_utility": 0.0}
    )
    if not content:
        return null_prediction
    end = content.rfind("}")
    if end == -1:
        return null_prediction
    depth = 0
    for i in range(end, -1, -1):
        if content[i] == "}":
            depth += 1
        elif content[i] == "{":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(content[i : end + 1])
                except ValueError:
                    return null_prediction
                return json.dumps(obj, separators=(",", ":"))
    return null_prediction


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
            organic_data = serper_data.get("organic", [])[:MAX_SOURCES]
            misc_data = serper_data.get("peopleAlsoAsk", [])
            sources = format_sources_data(organic_data, misc_data)
        else:
            serper_api_key = kwargs["api_keys"]["serperapi"]
            print("Fetching additional sources...")
            serper_response = fetch_additional_sources(question, serper_api_key)
            sources_data = serper_response.json()
            # mode tag included for consistency across tools; content is identical
            # regardless of mode since Serper returns structured JSON, not HTML
            captured_source_content = {
                "mode": source_content_mode,
                "serper_response": sources_data,
            }
            print(f"Additional sources fetched: {sources_data}")
            organic_data = sources_data.get("organic", [])[:MAX_SOURCES]
            misc_data = sources_data.get("peopleAlsoAsk", [])
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
        raw_completion, counter_callback = generate_prediction_with_retry(
            client=llm_client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=COMPLETION_RETRIES,
            delay=COMPLETION_DELAY,
            counter_callback=counter_callback,
        )
        # v4's prompt permits reasoning prose before the final JSON object, so the
        # raw completion is not flat-``json.loads``-parseable; extract the trailing
        # JSON before returning so the strict trader consumer can parse the result.
        extracted_block = extract_result_json(raw_completion)

        used_params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if return_source_content:
            used_params["source_content"] = captured_source_content
        return extracted_block, prediction_prompt, None, counter_callback, used_params
