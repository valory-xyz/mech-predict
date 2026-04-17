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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai
import requests
from openai import OpenAI
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


# ---------------------------------------------------------------------------
# Pydantic schema for OpenAI Structured Outputs
# ---------------------------------------------------------------------------


class PredictionResult(BaseModel):
    """Superforecaster structured output.

    The text fields carry the 7-step reasoning chain (facts → pros/cons →
    aggregation + tentative → reflection → final). Only the four numeric
    fields at the bottom are returned on-chain per the mech protocol.
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
            "CALIBRATION block (state a base rate and justify, adjust using "
            "specific evidence, treat missing expected evidence as a NO signal), "
            "adjust for news negativity / sensationalism bias. End by stating a "
            "tentative probability in [0,1]."
        ),
    )
    reflection: str = Field(
        ...,
        description=(
            "Sanity checks and finalisation. Apply the three checks: "
            "EVIDENCE BAR, CONFIDENCE COUPLING, NUMERIC QUESTIONS. Check for "
            "over/underconfidence and forecasting biases. Highlight the key "
            "factors informing the final forecast."
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
        """Validate that p_yes + p_no ≈ 1."""
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
                # Max-cost path returns a float; pass through without appending
                # api_keys (tuple concatenation would fail).
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
            except Exception as e:
                return str(e), "", None, None, None, api_keys

        mech_response = execute()
        return mech_response

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
        from tiktoken import get_encoding

        enc = get_encoding("o200k_base")
    return len(enc.encode(text))


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 4096,
    "limit_max_tokens": 4096,
    "temperature": 0,
}
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"
ALLOWED_TOOLS = ["superforcaster"]
ALLOWED_MODELS = [DEFAULT_OPENAI_MODEL]
MAX_SOURCES = 5
COMPLETION_RETRIES = 3
COMPLETION_DELAY = 2


SYSTEM_PROMPT = "You are a helpful assistant."

PREDICTION_PROMPT = """
You are an advanced AI system which has been finetuned to provide calibrated probabilistic
forecasts under uncertainty, with your performance evaluated according to the Brier score. When
forecasting, do not treat 0.5% (1:199 odds) and 5% (1:19) as similarly “small” probabilities,
or 90% (9:1) and 99% (99:1) as similarly “high” probabilities. As the odds show, they are
markedly different, so output your probabilities accordingly.

Question:
{question}

Today's date: {today}

We have retrieved the following information for this question:
<background>{sources}</background>

Recall the question you are forecasting:
{question}

Return a structured PredictionResult whose fields capture the following reasoning chain:

1. `facts` — Compress key factual information from the sources, as well as useful background information
which may not be in the sources, into a list of core factual points to reference. Aim for
information which is specific, relevant, and covers the core considerations you'll use to make
your forecast. For this step, do not draw any conclusions about how a fact will influence your
answer or forecast.

2. `reasons_no` — Provide a few reasons why the answer might be no. Rate the strength of each reason on a
scale of 1-10.

3. `reasons_yes` — Provide a few reasons why the answer might be yes. Rate the strength of each reason on a
scale of 1-10.

4. `aggregation` — Aggregate your considerations. Do not summarize or repeat previous points; instead,
investigate how the competing factors and mechanisms interact and weigh against each other.
Factorize your thinking across (exhaustive, mutually exclusive) cases if and only if it would be
beneficial to your reasoning. We have detected that you overestimate world conflict, drama,
violence, and crises due to news' negativity bias, which doesn't necessarily represent overall
trends or base rates. Similarly, we also have detected you overestimate dramatic, shocking,
or emotionally charged news due to news' sensationalism bias. Therefore adjust for news'
negativity bias and sensationalism bias by considering reasons to why your provided sources
might be biased or exaggerated. Think like a superforecaster.

CALIBRATION (mandatory before any probability):
- State a base-rate probability for this event category and justify it.
- Adjust from the base rate using specific evidence only.
- Missing expected evidence (no announcement found, no confirmation) is a NO signal.

End the `aggregation` field by stating an initial tentative probability (a single number between 0 and 1)
given steps 1-4.

5. `reflection` — Reflect on your tentative answer, performing sanity checks and mentioning any additional knowledge
or background information which may be relevant. Check for over/underconfidence, improper
treatment of conjunctive or disjunctive conditions (only if applicable), and other forecasting
biases when reviewing your reasoning. Consider priors/base rates, and the extent to which
case-specific information justifies the deviation between your tentative forecast and the prior.
Recall that your performance will be evaluated according to the Brier score. Be precise with tail
probabilities. Leverage your intuitions, but never change your forecast for the sake of modesty
or balance alone. Finally, aggregate all of your previous reasoning and highlight key factors
that inform your final forecast.

BEFORE FINAL ANSWER — apply all three checks:

1. EVIDENCE BAR: If sources confirm the event already occurred, high p_yes is fine.
   If not: p_yes > 0.90 needs verified commitment (signed, awarded, published).
   p_yes > 0.80 needs strong specific evidence, not plausibility or reputation.
   Plans, proposals, and intentions are not completed actions.

2. CONFIDENCE COUPLING: If confidence < 0.3, keep p_yes between 0.30-0.70.
   If confidence < 0.5, keep p_yes between 0.20-0.80.

3. NUMERIC QUESTIONS: For price/temperature/count thresholds, find the current
   value and compare to the threshold. A large gap overrides sentiment or forecasts.

6. `p_yes`, `p_no`, `confidence`, `info_utility` — the four numeric fields:
   - "p_yes": Estimated probability that the event in the "Question" occurs.
   - "p_no": Estimated probability that the event in the "Question" does not occur.
   - "confidence": A value between 0 and 1 indicating the confidence in the prediction. 0 indicates lowest
     confidence value; 1 maximum confidence value.
   - "info_utility": Utility of the information provided in "sources" to help you make the prediction.
     0 indicates lowest utility; 1 maximum utility.
   - Each value must be between 0 and 1.
   - The sum of "p_yes" and "p_no" must equal 1.
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

    ``client.beta.chat.completions.parse()`` guarantees the response conforms
    to the supplied Pydantic schema — no prompt-side JSON format instructions
    or regex extraction required.

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
            # Retry only transient failures and model-side issues:
            #   - APIConnectionError (includes APITimeoutError) — network blips
            #   - RateLimitError / InternalServerError — OpenAI-side retryable
            #   - ValueError — covers pydantic ValidationError (e.g. p_yes+p_no
            #     sum check) and the inline "Model refused…" raise above
            print(f"[superforcaster] Attempt {attempt + 1} failed: {e}")
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

        print(f"[superforcaster] === FACTS ===\n{prediction.facts}")
        print(f"[superforcaster] === REASONS_NO ===\n{prediction.reasons_no}")
        print(f"[superforcaster] === REASONS_YES ===\n{prediction.reasons_yes}")
        print(f"[superforcaster] === AGGREGATION ===\n{prediction.aggregation}")
        print(f"[superforcaster] === REFLECTION ===\n{prediction.reflection}")
        print(
            f"[superforcaster] Result: p_yes={prediction.p_yes}, "
            f"p_no={prediction.p_no}, confidence={prediction.confidence}, "
            f"info_utility={prediction.info_utility}"
        )

        # On-chain result — only the four standard mech fields.
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
