# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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
"""Simple OpenAI API wrapper with dynamic model support."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import openai
from openai import OpenAI


client: Optional[OpenAI] = None


MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]


def with_key_rotation(func: Callable) -> Callable:
    """Retry function with key rotation on rate limits."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponseWithKeys:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponseWithKeys:
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
            except Exception as e:  # pragma: no cover
                return str(e), "", None, None, api_keys

        return execute()

    return wrapper


class OpenAIClientManager:
    """Client context manager for OpenAI."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __enter__(self) -> OpenAI:
        global client
        if client is None:
            client = OpenAI(api_key=self.api_key)
        return client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global client
        if client is not None:
            client.close()
            client = None


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 1024,
    "temperature": 1.0,
}
PREFIX = "openai-"


def _uses_max_completion_tokens(model: str) -> bool:
    """Whether model expects `max_completion_tokens` instead of `max_tokens`."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _supports_custom_temperature(model: str) -> bool:
    """Whether model family supports arbitrary temperature values."""
    return not model.startswith(("gpt-5",))


@with_key_rotation
def run(
    **kwargs: Any,
) -> Union[float, Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]]:
    """Run the task."""
    tool = kwargs["tool"]
    if not tool.startswith(PREFIX):
        return (
            f"Tool {tool} is not in the list of supported tools.",
            None,
            None,
            None,
        )

    engine_from_tool = tool.replace(PREFIX, "", 1)
    if not engine_from_tool:
        return (
            "No OpenAI model was provided in tool name.",
            None,
            None,
            None,
        )

    engine = kwargs.get("model") or engine_from_tool
    delivery_rate = int(kwargs.get("delivery_rate", 1))
    counter_callback: Optional[Callable] = kwargs.get("counter_callback", None)

    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback was given to calculate the max cost with."
            )

        max_cost = counter_callback(
            max_cost=True,
            models_calls=(engine,),
        )
        return max_cost

    with OpenAIClientManager(kwargs["api_keys"]["openai"]):
        max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        prompt = kwargs["prompt"]

        if not client:
            raise RuntimeError("Client not initialized")

        request_kwargs: Dict[str, Any] = {
            "model": engine,
            "messages": [{"role": "user", "content": prompt}],
            "n": 1,
            "timeout": 120,
        }
        if _supports_custom_temperature(engine):
            request_kwargs["temperature"] = temperature
        if _uses_max_completion_tokens(engine):
            request_kwargs["max_completion_tokens"] = max_tokens
        else:
            request_kwargs["max_tokens"] = max_tokens

        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        return content or "", prompt, None, None
