# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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

"""The implementation of the prediction_websearch tool."""

import functools
import re
import traceback as traceback_
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai
from pydantic import BaseModel
from tiktoken import encoding_for_model, get_encoding


N_MODEL_CALLS = 2
NON_ZERO_DELIVERY_RATE = (
    1  # Default delivery rate changed to 1 to avoid the breaking change
)
ALLOWED_TOOLS = [
    "prediction-websearch",
]


LLM_SETTINGS = {
    "gpt-4.1-2025-04-14": {
        "default_max_tokens": 4096,
        "limit_max_tokens": 1_047_576,
        "temperature": 0,
    },
}
ALLOWED_MODELS = list(LLM_SETTINGS.keys())


MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]


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
                if retries_left["openai"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                api_keys.rotate("openai")
                return execute()
            except Exception as e:
                return str(e), traceback_.format_exc(), None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


# pylint: disable=too-few-public-methods
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


# pylint: disable=too-few-public-methods
class LLMResponse:
    """Response class."""

    def __init__(
        self, content: Optional[BaseModel] = None, usage: Optional[Usage] = None
    ):
        """Initializes with content and usage class."""
        self.content = content
        self.usage = Usage()


class SearchContextSize(Enum):
    """Enum for search context size."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PredictionResult(BaseModel):
    """Prediction results model."""

    p_yes: float
    p_no: float
    info_utility: float
    confidence: float
    analysis: str


class LLMClient:
    """Client for LLMs."""

    def __init__(self, api_keys: Dict):
        """Initializes with API keys, model. Sets the LLM provider based on the model."""
        self.api_keys = api_keys
        self.client = openai.OpenAI(api_key=self.api_keys["openai"])  # type: ignore

    def completions_with_web_search(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float] = None,
        search_context_size: Optional[SearchContextSize] = SearchContextSize.MEDIUM,
        output_format: Optional[BaseModel] = None,
    ) -> Optional[LLMResponse]:
        """Generate a completion from the specified LLM provider using the given model and messages."""

        response_provider = self.client.responses.parse(
            model=model,
            input=prompt,
            temperature=temperature,
            timeout=150,
            tools=[
                {
                    "type": "web_search_preview",
                    "search_context_size": search_context_size.value,
                }
            ],
            text_format=output_format,
        )

        response = LLMResponse()
        response.content = response_provider.output_parsed

        response.usage.prompt_tokens = response_provider.usage.input_tokens
        response.usage.completion_tokens = response_provider.usage.output_tokens
        return response


client: Optional[LLMClient] = None


class LLMClientManager:
    """Client context manager for LLMs."""

    def __init__(self, api_keys: Dict, model: str):
        """Initializes with API keys, model. Sets the LLM provider based on the model."""
        self.api_keys = api_keys

    def __enter__(self) -> List:
        """Initializes and returns LLM clients."""
        clients = []
        global client
        if client is None:
            client = LLMClient(self.api_keys)
            clients.append(client)
        return clients

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Closes the LLM client"""
        global client
        if client is not None:
            client.client.close()
            client = None


def extract_question(prompt: str) -> str:
    """Uses regexp to extract question from the prompt"""
    pattern = r"\"(.*?)\""
    try:
        question = re.findall(pattern, prompt)[0]
    except Exception as e:
        print(f"Error extracting question: {e}")
        question = prompt

    return question


PREDICTION_PROMPT = """You will be evaluating the likelihood of an event based on a user's question and additional information from search results.
User Prompt: {USER_PROMPT}

Carefully consider the user's question and the additional information provided. Then, think through the following:
- `p_yes`: Probability that the event will occur (float between 0 and 1)
- `p_no`: Probability that the event will not occur (float between 0 and 1)
- `confidence`: Your confidence in this prediction (float between 0 and 1)
- `info_utility`: How useful the additional information was in making your prediction (float between 0 and 1)
- `analysis`: A brief explanation of your reasoning

Remember, p_yes and p_no should add up to 1.
"""


# Utility: count tokens using model-specific tokenizer
def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    # Workaround since tiktoken does not have support yet for gpt4.1
    # https://github.com/openai/tiktoken/issues/395
    if model == "gpt-4.1-2025-04-14":
        enc = get_encoding("o200k_base")
    else:
        enc = encoding_for_model(model)

    # TODO: Handle GPT5
    return len(enc.encode(text))


@with_key_rotation
def run(*args: Any, **kwargs: Any) -> None:
    """The callable for the prediction_websearch tool."""

    print(f"Running prediction_websearch with {args=} and {kwargs=}.")
    tool = kwargs["tool"]
    model = kwargs.get("model")
    api_keys = kwargs.get("api_keys", {})
    if model is None:
        raise ValueError("Model must be specified in kwargs")

    delivery_rate = int(kwargs.get("delivery_rate", NON_ZERO_DELIVERY_RATE))
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

    with LLMClientManager(api_keys, model):
        event = extract_question(kwargs["prompt"])
        temperature = kwargs.get("temperature", LLM_SETTINGS[model]["temperature"])

        if not client:
            raise RuntimeError("Client not initialized")

        # Make sure the model is supported
        if model not in ALLOWED_MODELS:
            raise ValueError(f"Model {model} not supported.")

        # make sure the tool is supported
        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"Tool {tool} not supported.")

        # Generate the prediction prompt
        prediction_prompt = PREDICTION_PROMPT.format(
            USER_PROMPT=event,
        )

        response = client.completions_with_web_search(
            model=model,
            prompt=prediction_prompt,
            temperature=temperature,
            search_context_size=SearchContextSize.MEDIUM,
            output_format=PredictionResult,
        )
        if not response or response.content is None:
            return (
                "Response Not Valid",
                prediction_prompt,
                None,
                counter_callback,
            )

        if counter_callback:
            counter_callback(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=model,
                token_counter=count_tokens,
            )

        return (
            response.content.model_dump_json(),
            prediction_prompt,
            None,
            counter_callback,
        )
