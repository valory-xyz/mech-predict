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
from pydantic import BaseModel, Field, model_validator
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
    """superforcaster-polymarket-v4 structured output.

    The text fields carry v4's reasoning chain (facts -> reasons -> the
    mandatory evidence-reliability screen -> aggregation -> reflection). Only
    the four numeric fields at the bottom are returned on-chain per the mech
    protocol. Using OpenAI structured outputs guarantees the completion parses,
    so no prompt-side JSON-format instruction or output extraction is needed.
    """

    facts: str = Field(
        ...,
        description=(
            "Core factual points compiled from the sources and relevant "
            "background. Specific, relevant, no conclusions about how a fact "
            "influences the forecast."
        ),
    )
    reasons_no: str = Field(
        ...,
        description="Reasons the answer might be NO, each rated 1-10 for strength.",
    )
    reasons_yes: str = Field(
        ...,
        description="Reasons the answer might be YES, each rated 1-10 for strength.",
    )
    evidence_reliability_screen: str = Field(
        ...,
        description=(
            "MANDATORY evidence-reliability screen, completed BEFORE forming a "
            "tentative probability. (a) Prediction-market-odds filter: discard "
            "any prediction-market trading price (polymarket / metaculus / "
            "manifold / predictit / kalshi) as circular self-referential "
            "evidence. (b) Forward-looking-intent discount: for intent or "
            "expectation language ('is set to', 'is expected to', 'plans to', "
            "'scheduled to', 'is poised to'), treat the outcome as only 40-60% "
            "likely to materialize absent strong specific evidence. (c) "
            "Temporal-evidence filter: classify each source TYPE A (dated within "
            "the resolution window, or directly states the criterion was met) vs "
            "TYPE B (undated, outside the window, or a standing page); state the "
            "TYPE A and TYPE B counts; if ALL sources are TYPE B, anchor on the "
            "category base rate (20-40% YES for 'X in headlines this week'-style "
            "markets). (d) Criterion-specificity check: does any TYPE A evidence "
            "directly confirm the exact resolution condition (not merely that the "
            "topic is active)? If not, add uncertainty toward the base rate."
        ),
    )
    aggregation: str = Field(
        ...,
        description=(
            "Aggregate the remaining considerations after the screen. Weigh how "
            "competing factors interact; adjust for news negativity and "
            "sensationalism bias. End by stating a tentative probability in [0,1]."
        ),
    )
    reflection: str = Field(
        ...,
        description=(
            "Sanity checks and finalisation: over/underconfidence, conjunctive "
            "or disjunctive conditions, priors vs case-specific evidence. Be "
            "precise with tail probabilities; never change the forecast for "
            "modesty or balance alone. Highlight the key factors informing the "
            "final forecast."
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
        """Validate that p_yes + p_no is approximately 1."""
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
            except Exception as e:  # noqa: BLE001
                # Return a parseable null-prediction JSON (matches
                # superforcaster_calibrated_full_search) so the strict trader
                # consumer sees an explicit no-prediction rather than a raw
                # exception string that its flat ``json.loads`` would reject.
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


class OpenAIClient:
    """OpenAI Client"""

    def __init__(self, api_key: str):
        """Initializes with API keys and client."""
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)


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

Produce a structured forecast by filling every field of the required output schema,
reasoning in this order:

- facts: compress the sources and useful background into specific, relevant core factual
  points. Do NOT draw conclusions about how a fact influences the answer here.
- reasons_no: a few reasons the answer might be NO, each rated 1-10 for strength.
- reasons_yes: a few reasons the answer might be YES, each rated 1-10 for strength.
- evidence_reliability_screen: MANDATORY, and completed BEFORE you form any probability.
  Follow every part of that field's instructions: the prediction-market-odds filter, the
  forward-looking-intent discount, the temporal-evidence TYPE A / TYPE B classification
  (state both counts), and the criterion-specificity check.
- aggregation: after the screen, weigh how the competing factors interact. We have detected
  that you overestimate conflict, drama, violence and crises (news negativity bias) and
  dramatic or emotionally charged news (sensationalism bias); adjust for both. Think like a
  superforecaster and end by stating a tentative probability in [0,1].
- reflection: sanity checks -- over/underconfidence, conjunctive or disjunctive conditions,
  priors vs case-specific evidence. Be precise with tail probabilities; never change the
  forecast for modesty or balance alone. Highlight the key factors informing the final
  forecast.
- p_yes, p_no, confidence, info_utility: your final numbers. Each must be in [0,1] and
  p_yes + p_no must equal 1. p_yes is the probability the event occurs; confidence is your
  confidence in the prediction; info_utility is how useful the sources were.
"""


def _parse_completion(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Any,
    temperature: float = 0,
    max_tokens: int = 1500,
    retries: int = COMPLETION_RETRIES,
    delay: int = COMPLETION_DELAY,
    counter_callback: Optional[Callable] = None,
) -> Tuple[Any, Optional[Callable]]:
    """Call OpenAI Structured Outputs and parse into a Pydantic model.

    ``client.beta.chat.completions.parse()`` guarantees the response conforms to
    the supplied Pydantic schema, so no prompt-side JSON-format instruction or
    output extraction is required -- the on-chain result is always a clean,
    flat-``json.loads``-parseable object.

    :param client: an initialised ``openai.OpenAI`` client.
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
            openai.RateLimitError,
            openai.InternalServerError,
            ValueError,
        ) as e:
            print(f"[superforcaster-polymarket-v4] Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
            attempt += 1

    raise RuntimeError("Failed to get structured LLM completion after retries")


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
        # OpenAI structured outputs: the model reasons INTO the PredictionResult
        # schema fields and the SDK returns a validated object, so the on-chain
        # result is a clean, flat-``json.loads``-parseable JSON object -- no
        # reasoning prose can leak into it (the bug v4 previously shipped).
        prediction: PredictionResult
        prediction, counter_callback = _parse_completion(
            client=llm_client.client,
            model=model,
            messages=messages,
            response_format=PredictionResult,
            temperature=temperature,
            max_tokens=max_tokens,
            counter_callback=counter_callback,
        )
        print(
            f"[superforcaster-polymarket-v4] Result: p_yes={prediction.p_yes}, "
            f"p_no={prediction.p_no}, confidence={prediction.confidence}, "
            f"info_utility={prediction.info_utility}"
        )

        # On-chain result -- only the four standard mech fields.
        result = json.dumps(
            {
                "p_yes": prediction.p_yes,
                "p_no": prediction.p_no,
                "confidence": prediction.confidence,
                "info_utility": prediction.info_utility,
            }
        )

        used_params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if return_source_content:
            used_params["source_content"] = captured_source_content
        return result, prediction_prompt, None, counter_callback, used_params
