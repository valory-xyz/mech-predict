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

import requests
from typing import Any, Dict, Optional, Tuple

DEFAULT_PERPLEXITY_SETTINGS = {
    "model": "llama-3.1-sonar-small-128k-online",
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.9,
    "search_recency_filter": "month",
    "presence_penalty": 0,
    "frequency_penalty": 1,
    "stream": False,
    "return_images": False,
    "return_related_questions": False,
    "search_domain_filter": ["perplexity.ai"],
}

PREFIX = "llama-"
ENGINES = {
    "chat": ["3.1-sonar-small-128k-online", "3.1-sonar-large-128k-online", "3.1-sonar-huge-128k-online"],
}

ALLOWED_TOOLS = [PREFIX + value for value in ENGINES["chat"]]
API_URL = "https://api.perplexity.ai/chat/completions"

def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""

    api_key = kwargs["api_keys"]["perplexity"]
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]

    if tool not in ALLOWED_TOOLS:
        return (
            f"Model {tool} is not in the list of supported models.",
            None,
            None,
            None,
        )
    
    counter_callback = kwargs.get("counter_callback", None)

    payload = {
        "model": tool,
        "messages": [
            {"role": "system", "content": kwargs.get("system_prompt", "Be precise and concise.")},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": kwargs.get("max_tokens", DEFAULT_PERPLEXITY_SETTINGS["max_tokens"]),
        "temperature": kwargs.get("temperature", DEFAULT_PERPLEXITY_SETTINGS["temperature"]),
        "top_p": kwargs.get("top_p", DEFAULT_PERPLEXITY_SETTINGS["top_p"]),
        "search_recency_filter": kwargs.get("search_recency_filter", DEFAULT_PERPLEXITY_SETTINGS["search_recency_filter"]),
        "presence_penalty": kwargs.get("presence_penalty", DEFAULT_PERPLEXITY_SETTINGS["presence_penalty"]),
        "frequency_penalty": kwargs.get("frequency_penalty", DEFAULT_PERPLEXITY_SETTINGS["frequency_penalty"]),
        "stream": kwargs.get("stream", DEFAULT_PERPLEXITY_SETTINGS["stream"]),
        "return_images": kwargs.get("return_images", DEFAULT_PERPLEXITY_SETTINGS["return_images"]),
        "return_related_questions": kwargs.get("return_related_questions", DEFAULT_PERPLEXITY_SETTINGS["return_related_questions"]),
        "search_domain_filter": kwargs.get("search_domain_filter", DEFAULT_PERPLEXITY_SETTINGS["search_domain_filter"]),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {}).get("content", None)
        if not message:
            return "No content received from the assistant.", None, None, None
    except Exception as e:
        return f"An error occurred: {str(e)}", None, None, None

    return message, prompt, None, counter_callback
