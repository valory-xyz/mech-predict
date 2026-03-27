# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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
from typing import Any, Callable, Dict, Optional, Tuple

import google.generativeai as genai
import openai

MechResponseWithKeys = Tuple[
    str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]], Any
]
MechResponse = Tuple[
    str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]]
]

DEFAULT_GEMINI_SETTINGS = {
    "candidate_count": 1,
    "stop_sequences": None,
    "max_output_tokens": 500,
    "temperature": 0.7,
}
PREFIX = "gemini-"
ENGINES = {
    "chat": ["2.0-flash", "2.0-flash-lite"],
}

ALLOWED_TOOLS = [PREFIX + value for value in ENGINES["chat"]]


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
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
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


@with_key_rotation
def run(**kwargs: Any) -> MechResponse:
    """Run the task"""

    api_key = kwargs["api_keys"]["gemini"]
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]

    if tool not in ALLOWED_TOOLS:
        return (
            f"Model {tool} is not in the list of supported models.",
            None,
            None,
            None,
            None,
        )

    candidate_count = kwargs.get(
        "candidate_count", DEFAULT_GEMINI_SETTINGS["candidate_count"]
    )
    stop_sequences = kwargs.get(
        "stop_sequences", DEFAULT_GEMINI_SETTINGS["stop_sequences"]
    )
    max_output_tokens = kwargs.get(
        "max_output_tokens", DEFAULT_GEMINI_SETTINGS["max_output_tokens"]
    )
    temperature = kwargs.get("temperature", DEFAULT_GEMINI_SETTINGS["temperature"])

    counter_callback = kwargs.get("counter_callback", None)

    genai.configure(api_key=api_key)
    engine = genai.GenerativeModel(tool)

    try:
        response = engine.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=candidate_count,
                stop_sequences=stop_sequences,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            ),
        )

        # Ensure response has a .text attribute
        response_text = getattr(  # noqa: F841 # pylint: disable=unused-variable
            response, "text", None
        )

    except Exception as e:
        return f"An error occurred: {str(e)}", None, None, None, None

    used_params = {
        "model": tool,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    return response.text, prompt, None, counter_callback, used_params
