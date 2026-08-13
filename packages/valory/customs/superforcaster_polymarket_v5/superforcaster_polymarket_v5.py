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

What superforcaster-polymarket-v5 does (vs superforcaster-polymarket-v2)
------------------------------------------------------------------------
v2 added a resolution-criterion specificity check in step 6 of the prompt,
but its OUTPUT_FORMAT instructed the model to emit ONLY JSON, causing the
7-step chain-of-thought (including the criterion-specificity check) to be
skipped entirely. Additionally, max_tokens=500 would truncate any CoT attempt
before step 6 completed. The result was systematic p_yes >= 0.93 on
"Will X say Y?" and "Will X be in headlines?" questions that resolved NO
(confirmed in Issue #439 C-1 worst-miss rows: 10/10 deliveries had
good evidence but bad reasoning -- step 6 never executed).

v5 fix (single mechanism -- Issue #439): switch to OpenAI structured outputs
(client.beta.chat.completions.parse with PredictionResult) so the model
fills the reasoning fields BEFORE the four numeric fields. This makes the
7-step CoT -- including the criterion-specificity check already in the v2
prompt -- execute as intended. max_tokens raised from 500 to 4096 so the
chain-of-thought fits. No new prompt content is added; only the output
contract changes so the existing reasoning steps actually run.
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
    """superforcaster-polymarket-v5 structured output.

    The text fields carry the 7-step reasoning chain from v2's PREDICTION_PROMPT
    (facts -> reasons_no -> reasons_yes -> aggregation -> tentative_probability ->
    reflection). The four numeric fields are declared LAST so the model conditions
    them on the completed chain-of-thought (Issue #439 fix: the criterion-
    specificity check in step 6 now executes before p_yes is formed).
    Using OpenAI structured outputs guarantees the completion parses without
    any JSON-format instruction or output extraction.
    """

    facts: str = Field(
        ...,
        description=(
            "Step 1: compress key factual information from the sources, "
            "as well as useful background information not in the sources, "
            "into a list of core factual points. Specific, relevant, covers "
            "the core considerations. Do NOT draw conclusions about how a "
            "fact influences the forecast here."
        ),
    )
    reasons_no: str = Field(
        ...,
        description=(
            "Step 2: a few reasons the answer might be NO. "
            "Rate the strength of each reason on a scale of 1-10."
        ),
    )
    reasons_yes: str = Field(
        ...,
        description=(
            "Step 3: a few reasons the answer might be YES. "
            "Rate the strength of each reason on a scale of 1-10."
        ),
    )
    aggregation: str = Field(
        ...,
        description=(
            "Step 4: aggregate the considerations. Do not summarize or repeat "
            "previous points; investigate how competing factors and mechanisms "
            "interact and weigh against each other. We have detected that you "
            "overestimate world conflict, drama, violence, and crises (news "
            "negativity bias) and dramatic or emotionally charged news "
            "(sensationalism bias); adjust for both. Think like a superforecaster."
        ),
    )
    tentative_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Step 5: output an initial probability (prediction) as a single "
            "number between 0 and 1 given steps 1-4."
        ),
    )
    reflection: str = Field(
        ...,
        description=(
            "Step 6: reflect on the answer, performing sanity checks and "
            "mentioning any additional relevant knowledge. Check for over/"
            "underconfidence, improper treatment of conjunctive or disjunctive "
            "conditions, and other forecasting biases. Consider priors/base rates "
            "and the extent to which case-specific information justifies the "
            "deviation between the tentative forecast and the prior. "
            "CRITICALLY: identify the exact resolution criterion -- the specific "
            "condition that must literally be true for this market to resolve YES. "
            "For each piece of evidence, ask: does this evidence bear on whether "
            "the criterion itself will be satisfied, or does it only establish that "
            "the topic is relevant? Topical relevance is not criterion evidence. "
            "Derive p_yes by reasoning explicitly about the probability that the "
            "criterion is satisfied: consider the base rate for this type of "
            "condition, any case-specific factors that raise or lower that "
            "probability, and whether any evidence directly confirms or denies "
            "criterion satisfaction. Ground the estimate in that reasoning chain -- "
            "do not substitute topical signal strength for a criterion probability. "
            "Be precise with tail probabilities. Leverage intuitions but never "
            "change the forecast for modesty or balance alone. Aggregate all "
            "previous reasoning and highlight key factors informing the final "
            "forecast."
        ),
    )
    # IMPORTANT: the four numeric fields below MUST stay LAST in this schema.
    # Structured outputs generate JSON fields in declaration order, so keeping
    # the numbers after the reasoning fields conditions them on the completed
    # chain-of-thought -- this is the core of the Issue #439 fix.
    # Do not reorder or alphabetize (Pydantic will not complain but the fix
    # depends on this ordering to enforce CoT before probability formation).
    p_yes: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Step 7: final probability the event occurs, derived from step 6's "
            "criterion-specificity reasoning. Must be in [0, 1]."
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


def _null_prediction_response(api_keys: Any) -> MechResponseWithKeys:
    """Build the parseable null-prediction tuple for any failure path.

    The strict trader consumer flat-``json.loads`` the delivery, so every
    failure must return this shape rather than a raw exception string.

    :param api_keys: the KeyChain, threaded back to the caller unchanged.
    :return: the null-prediction MechResponseWithKeys tuple.
    """
    error_json = json.dumps(
        {
            "p_yes": 0.5,
            "p_no": 0.5,
            "confidence": 0.0,
            "info_utility": 0.0,
        }
    )
    return error_json, "", None, None, None, api_keys


def with_key_rotation(func: Callable) -> Callable:
    """Decorator that retries a function with API key rotation on failure.

    :param func: The function to be decorated.
    :type func: Callable
    :returns: Callable -- the wrapped function that handles retries with
        key rotation.
    """

    @functools.wraps(func)
    def wrapper(
        *args: Any, **kwargs: Any
    ) -> Union[MaxCostResponse, MechResponseWithKeys]:
        """Retry with key rotation on RateLimitError."""
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> Union[MaxCostResponse, MechResponseWithKeys]:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                if isinstance(result, float):
                    return result
                return result + (api_keys,)
            except openai.RateLimitError as e:
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    print(f"[superforcaster-polymarket-v5] Rate limit exhausted: {e}")
                    return _null_prediction_response(api_keys)
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
            except Exception as e:  # pylint: disable=broad-except
                print(f"[superforcaster-polymarket-v5] Unhandled error: {e}")
                return _null_prediction_response(api_keys)

        return execute()

    return wrapper


class OpenAIClientManager:
    """Client context manager for OpenAI."""

    def __init__(self, api_key: str):
        """Initializes with API keys."""
        self.api_key = api_key
        self._client: Optional["openai.OpenAI"] = None

    def __enter__(self) -> "openai.OpenAI":
        """Initializes and returns LLM client."""
        self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Closes the LLM client."""
        if self._client is not None:
            self._client.close()
            self._client = None


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


DEFAULT_OPENAI_SETTINGS = {
    # Raised from 500 to 4096 (Issue #439 fix): the 7-step reasoning chain
    # -- especially step 6's criterion-specificity check -- needs headroom
    # to complete before the numeric fields are filled. At 500 tokens the
    # model truncates before step 6, which was the root cause of systematic
    # overconfident-YES on "Will X say Y?" and headline questions.
    "max_tokens": 4096,
    "limit_max_tokens": 4096,
    "temperature": 0,
}
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"
ALLOWED_TOOLS = ["superforcaster-polymarket-v5"]
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

- facts: compress key factual information from the sources, as well as useful background
  information not in the sources, into a list of core factual points. Specific, relevant, and
  covering the core considerations. Do NOT draw conclusions about how a fact influences the
  answer here.
- reasons_no: a few reasons the answer might be NO, each rated 1-10 for strength.
- reasons_yes: a few reasons the answer might be YES, each rated 1-10 for strength.
- aggregation: aggregate the considerations. Do not summarize or repeat previous points;
  investigate how competing factors and mechanisms interact and weigh against each other.
  We have detected that you overestimate world conflict, drama, violence and crises (news
  negativity bias) and dramatic or emotionally charged news (sensationalism bias); adjust for
  both. Think like a superforecaster.
- tentative_probability: output an initial probability between 0 and 1 given steps 1-4.
- reflection: reflect on the answer, performing sanity checks. Check for over/underconfidence,
  improper treatment of conjunctive or disjunctive conditions, and other forecasting biases.
  Consider priors/base rates and the extent to which case-specific information justifies the
  deviation from the prior. CRITICALLY: identify the exact resolution criterion -- the specific
  condition that must literally be true for this market to resolve YES. For each piece of
  evidence, ask: does this evidence bear on whether the criterion itself will be satisfied, or
  does it only establish that the topic is relevant? Topical relevance is not criterion evidence.
  Derive p_yes by reasoning explicitly about the probability that the criterion is satisfied:
  consider the base rate for this type of condition, any case-specific factors that raise or
  lower that probability, and whether any evidence directly confirms or denies criterion
  satisfaction. Ground the estimate in that reasoning chain -- do not substitute topical signal
  strength for a criterion probability. Be precise with tail probabilities. Leverage intuitions
  but never change the forecast for modesty or balance alone.
- p_yes, p_no, confidence, info_utility: final numbers derived from the reflection above.
  Each must be in [0,1] and p_yes + p_no must equal 1.
"""


def _parse_completion(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    client: Any,
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
    well-formed JSON matching the schema's field names and types, so no
    prompt-side JSON-format instruction or output extraction is required.
    It does NOT enforce the custom ``model_validator`` (``p_yes + p_no ~= 1``):
    that raises ``pydantic.ValidationError`` (a ``ValueError``) inside
    ``.parse()``, which is why ``ValueError`` is in the retry tuple below.

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
            # decorator rotate API keys on a rate-limit hit -- retrying in-place
            # on the same throttled key never rotates. Transient connection /
            # server / validation failures stay here and retry on the same key.
            print(f"[superforcaster-polymarket-v5] Attempt {attempt + 1} failed: {e}")
            last_error = e
            time.sleep(delay)
            attempt += 1

    raise RuntimeError(
        f"Failed to get structured LLM completion after {retries} attempts: "
        f"{last_error}"
    ) from last_error


def fetch_additional_sources(question: Any, serper_api_key: Any) -> requests.Response:
    """Fetches additional sources for the given question using the Serper API."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": question})
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload, timeout=30)

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
    """Uses regexp to extract question from the prompt."""
    # Match from 'question "' to '" and the `yes`' to handle nested quotes
    pattern = r'question\s+"(.+?)"\s+and\s+the\s+`yes`'
    try:
        question = re.findall(pattern, prompt, re.DOTALL)[0]
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error extracting question: {e}")
        question = prompt
    return question


@with_key_rotation
def run(  # pylint: disable=too-many-locals,too-many-statements
    **kwargs: Any,
) -> Union[MaxCostResponse, MechResponse]:
    """Run the task."""
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
                "A delivery rate of `0` was passed, but no counter callback was given "
                "to calculate the max cost with."
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
        # Issue #439 fix: OpenAI structured outputs force the model to fill
        # the reasoning fields (including step 6's criterion-specificity check)
        # BEFORE emitting the four numeric fields. At max_tokens=500 with
        # "output ONLY JSON", the v2 model skipped the CoT and set p_yes from
        # topical relevance alone. The PredictionResult schema + max_tokens=4096
        # makes the full chain-of-thought execute on every call.
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
        print(
            f"[superforcaster-polymarket-v5] Result: p_yes={prediction.p_yes}, "
            f"p_no={prediction.p_no}, confidence={prediction.confidence}, "
            f"info_utility={prediction.info_utility}"
        )

        # On-chain result -- only the four standard mech fields, flat json.loads-
        # parseable as required by the trader (decision_receive.py:216).
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
