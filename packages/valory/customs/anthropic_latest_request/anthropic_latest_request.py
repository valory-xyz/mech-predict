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
"""Simple Anthropic API wrapper with dynamic model support."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import anthropic


client: Optional[anthropic.Anthropic] = None


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
            except anthropic.RateLimitError as e:
                service = "anthropic"
                if retries_left[service] <= 0:
                    raise e
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except Exception as e:  # pragma: no cover
                return str(e), "", None, None, api_keys

        return execute()

    return wrapper


class AnthropicClientManager:
    """Client context manager for Anthropic."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __enter__(self) -> anthropic.Anthropic:
        global client
        if client is None:
            client = anthropic.Anthropic(api_key=self.api_key)
        return client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global client
        client = None


DEFAULT_ANTHROPIC_SETTINGS = {
    "max_tokens": 1024,
    "temperature": 0.7,
}
PREFIX = "anthropic-"


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
            "No Anthropic model was provided in tool name.",
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

    with AnthropicClientManager(kwargs["api_keys"]["anthropic"]):
        max_tokens = kwargs.get("max_tokens", DEFAULT_ANTHROPIC_SETTINGS["max_tokens"])
        temperature = kwargs.get(
            "temperature", DEFAULT_ANTHROPIC_SETTINGS["temperature"]
        )
        prompt = kwargs["prompt"]

        if not client:
            raise RuntimeError("Client not initialized")

        try:
            response = client.messages.create(
                model=engine,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:  # pragma: no cover
            return (
                f"{e}. Hint: use an Anthropic model id enabled for your account/region.",
                None,
                None,
                None,
            )

        output_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text" and getattr(block, "text", None)
        ]
        if not output_parts:
            return "No text content returned by Anthropic.", prompt, None, None

        return "".join(output_parts), prompt, None, None
