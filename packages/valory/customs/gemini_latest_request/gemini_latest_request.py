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
"""Simple Google Gemini API wrapper with dynamic model support."""

from typing import Any, Dict, Optional, Tuple

import google.generativeai as genai


DEFAULT_GEMINI_SETTINGS = {
    "candidate_count": 1,
    "stop_sequences": None,
    "max_output_tokens": 1024,
    "temperature": 0.7,
}
PREFIX = "gemini-"


def run(**kwargs: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task."""
    api_key = kwargs["api_keys"]["gemini"]
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]

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
            "No Gemini model was provided in tool name.",
            None,
            None,
            None,
        )

    engine = kwargs.get("model") or engine_from_tool
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
    model = genai.GenerativeModel(engine)

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=candidate_count,
                stop_sequences=stop_sequences,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            ),
        )
    except Exception as e:  # pragma: no cover
        return f"An error occurred: {str(e)}", None, None, None

    return getattr(response, "text", ""), prompt, None, counter_callback
