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
r"""Contains the job definitions.

What v3 changes vs superforcaster-polymarket-v1
-----------------------------------------------
v3 is a sibling of v2 — both depart from v1:

  superforcaster-polymarket-v1
      |-- superforcaster-polymarket-v2  (criterion-specificity prompt)
      \-- superforcaster-polymarket-v3  (Anthropic claude-fable-5 swap)

v3 swaps the default LLM model from OpenAI ``gpt-4.1-2025-04-14`` to
Anthropic ``claude-fable-5`` while keeping v1's prompt, output format,
retry behaviour and Polymarket pipeline byte-identical.

Dispatch follows the convention every other claude-using tool in this
repo uses (``prediction_request``, ``prediction_request_rag``,
``prediction_request_reasoning``, ``prediction_url_cot``):

* ``LLMClientManager`` picks the SDK by model name (``"claude" in
  model`` -> anthropic, else openai).
* ``LLMClient.completions()`` branches by provider — the OpenAI path
  is unchanged from v1; the Anthropic path uses ``messages.create()``
  with the system messages extracted out, ``temperature`` dropped when
  zero (claude-fable-5 rejects the parameter), and the first
  ``TextBlock`` picked from the response (adaptive-thinking models
  emit a ``ThinkingBlock`` first which we skip).
* ``with_key_rotation`` catches ``openai.RateLimitError`` and
  ``anthropic.RateLimitError`` only and rotates the failing provider's
  pool (provider-gated so an error on one SDK doesn't burn the other
  pool's budget). Auth / permission / bad-request errors and the
  ``KeyError: 'anthropic'`` raised when a v1-era keychain lacks the
  anthropic key fall into the bare ``except Exception`` and are wrapped
  into the result tuple — see the decorator's docstring for the
  deployment note.
* ``component.yaml`` declares BOTH ``openai==1.93.0`` AND
  ``anthropic==0.23.1`` (no ``pyproject.toml`` bump).
"""

import functools
import json
import re
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anthropic
import openai
import requests
from anthropic import Anthropic
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


# Anthropic exception classes the rotation branch treats as recoverable.
# Used only by ``isinstance(e, _ANTHROPIC_ERRORS)`` to decide which pool to
# rotate; the ``except`` clause itself lists all classes inline so mypy can
# verify they're BaseException subclasses (a tuple alias triggers misc/B030).
_ANTHROPIC_ERRORS = (anthropic.RateLimitError,)


def with_key_rotation(func: Callable) -> Callable:
    """
    Decorator that retries on rate limits and wraps anything else as a result.

    Catches ``openai.RateLimitError`` and ``anthropic.RateLimitError`` ONLY and
    rotates the failing provider's key pool. Any other exception (including
    ``openai.AuthenticationError`` / ``PermissionDeniedError``,
    ``anthropic.AuthenticationError`` / ``BadRequestError``, and
    ``KeyError: 'anthropic'`` from a keychain that pre-dates v3) is wrapped
    into a result tuple by the bare ``except Exception`` branch — see the
    deployment note below.

    Deployment note: this tool requires both ``openai`` and ``anthropic``
    keys to be present in the KeyChain. A keychain provisioned only for v1
    will raise ``KeyError: 'anthropic'`` on the dispatch path and produce a
    result whose first element is the string ``"'anthropic'"``, NOT a valid
    prediction JSON. Mech deployments updating from v1 must add the
    anthropic key.

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
        # Older KeyChain.max_retries() implementations may not include an
        # ``anthropic`` entry; default to 0 so we don't crash on lookup but
        # still allow OpenAI-side rotation to proceed normally.
        retries_left.setdefault("anthropic", 0)
        retries_left.setdefault("openai", 0)
        retries_left.setdefault("openrouter", 0)

        def execute() -> MechResponseWithKeys:
            """Retry the function with a new key."""
            try:
                result: MechResponse = func(*args, **kwargs)
                return result + (api_keys,)
            except (
                openai.RateLimitError,
                anthropic.RateLimitError,
            ) as e:
                # Provider-gated rotation: rotate ONLY the pool that actually
                # failed. Rotating the other provider's keys on an error it
                # didn't cause would burn its retry budget for nothing and
                # trigger an early "all exhausted" re-raise under sustained
                # one-provider rate limiting.
                if isinstance(e, _ANTHROPIC_ERRORS):
                    if retries_left["anthropic"] <= 0:
                        raise e
                    retries_left["anthropic"] -= 1
                    api_keys.rotate("anthropic")
                    return execute()
                # OpenAI / OpenRouter branch.
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                if retries_left["openai"] > 0:
                    retries_left["openai"] -= 1
                    api_keys.rotate("openai")
                if retries_left["openrouter"] > 0:
                    retries_left["openrouter"] -= 1
                    api_keys.rotate("openrouter")
                return execute()
            except Exception as e:
                return str(e), "", None, None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


def _provider_for(model: str) -> str:
    """Return ``"anthropic"`` for claude-shaped models, ``"openai"`` otherwise."""
    return "anthropic" if "claude" in model else "openai"


class LLMClientManager:
    """Context manager that picks the SDK by model name.

    Mirrors the convention used by every other claude-using tool in this
    repo (``prediction_request``, ``prediction_request_rag``, etc.):
    ``"claude" in model`` routes to the Anthropic SDK, else to OpenAI.
    """

    def __init__(self, api_keys: Any, model: str):
        """Initializes with the keychain + model (provider derived from model)."""
        self.api_keys = api_keys
        self.model = model
        self.provider = _provider_for(model)
        self._client: Optional["LLMClient"] = None

    def __enter__(self) -> "LLMClient":
        """Instantiate the LLM client for the selected provider."""
        self._client = LLMClient(api_keys=self.api_keys, model=self.model)
        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the underlying HTTP client."""
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


class LLMClient:
    """Provider-agnostic LLM client that dispatches by model name.

    The legacy class name was ``OpenAIClient``; the response wrapper is still
    called ``OpenAIResponse`` because its ``.content`` / ``.usage`` shape is
    preserved across both providers, so downstream code is unchanged.
    """

    def __init__(self, api_keys: Any, model: str):
        """Initializes the SDK client for the provider derived from *model*."""
        self.api_keys = api_keys
        self.model = model
        self.provider = _provider_for(model)
        if self.provider == "anthropic":
            self.client = Anthropic(api_key=self.api_keys["anthropic"])
        else:
            self.client = openai.OpenAI(api_key=self.api_keys["openai"])

    def completions(
        self,
        model: str,
        messages: List = [],  # noqa: B006
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Any = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[OpenAIResponse]:
        """Generate a completion.

        Returns the same ``OpenAIResponse`` shape on both providers so the
        caller is provider-agnostic.

        :param model: provider-specific model identifier.
        :param messages: ordered list of ``{"role", "content"}`` dicts;
            ``system`` entries are extracted out for the Anthropic branch.
        :param timeout: kept for signature parity; timeout is hard-coded
            to 150s inside the call.
        :param temperature: sampling temperature; on the Anthropic branch
            it's dropped when zero because adaptive-thinking models like
            claude-fable-5 reject the parameter with HTTP 400.
        :param top_p: nucleus-sampling cap (OpenAI branch only).
        :param n: number of completions (OpenAI branch only; pinned to 1).
        :param stop: stop sequence (OpenAI branch only).
        :param max_tokens: cap on generated tokens. On the Anthropic
            branch this defaults to 4096 (not the v1 OpenAI default of
            500) because claude-fable-5's ``ThinkingBlock`` shares the
            ``max_tokens`` budget with the JSON output — small caps
            routinely truncate the response.
        :raises ValueError: on the Anthropic branch when the response is
            truncated (``stop_reason == "max_tokens"``) or has no
            ``TextBlock``. Both conditions previously returned ``None``
            content (or partial JSON) silently and bypassed the caller's
            retry loop.
        :return: ``OpenAIResponse`` carrying ``.content`` (text) and
            ``.usage`` (prompt/completion-token counts).
        """
        response = OpenAIResponse()
        if self.provider == "anthropic":
            # Anthropic separates ``system`` out of the messages list; extract
            # any system entries (joined) and pass the rest as ``messages=``.
            system_parts: List[str] = []
            user_assistant: List[Dict[str, str]] = []
            for msg in messages:
                if msg["role"] == "system":
                    system_parts.append(msg["content"])
                else:
                    user_assistant.append(msg)
            # ``max_tokens`` default lifted from 500 -> 4096 on the
            # Anthropic branch. claude-fable-5 is adaptive-thinking; the
            # ``ThinkingBlock`` shares the ``max_tokens`` budget with the
            # JSON output, so v1's OpenAI default of 500 truncates almost
            # every response (the truncation is invisible: either
            # ``text_block is None`` or partial JSON like ``{"p_yes": 0.6``).
            anthropic_max_tokens = max_tokens or 4096
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": user_assistant,
                "max_tokens": anthropic_max_tokens,
                "timeout": 150,
            }
            if system_parts:
                kwargs["system"] = "\n\n".join(system_parts)
            # ``temperature`` is deprecated on adaptive-thinking models like
            # claude-fable-5 (HTTP 400 "temperature is deprecated for this
            # model"). Pass it only when explicitly non-zero for legacy
            # callers using older claude models that still accept it.
            if temperature is not None and temperature != 0:
                kwargs["temperature"] = temperature
            resp = self.client.messages.create(**kwargs)
            # Truncation guard: if the model hit max_tokens we MUST raise
            # so the caller's retry loop engages. Without this a truncation
            # that lands on syntactically valid JSON parses as success with
            # incomplete reasoning, and one that doesn't returns None text
            # silently. Both flow on-chain with no error signal.
            if resp.stop_reason == "max_tokens":
                text_len = sum(
                    len(b.text)
                    for b in resp.content
                    if getattr(b, "type", None) == "text"
                )
                raise ValueError(
                    f"Response truncated (stop_reason='max_tokens', "
                    f"max_tokens={anthropic_max_tokens}, text_len={text_len}); "
                    f"raise max_tokens for this call site"
                )
            # Pick the first TextBlock; adaptive-thinking models emit a
            # ``ThinkingBlock`` before the ``TextBlock`` which we skip.
            text_block = next(
                (b for b in resp.content if getattr(b, "type", None) == "text"),
                None,
            )
            # If there's no TextBlock at all (thinking-only response, or
            # an unexpected content shape), we must raise — returning None
            # would silently propagate to the caller and bypass retry.
            if text_block is None:
                raise ValueError(
                    f"Model emitted no text block; stop_reason="
                    f"{resp.stop_reason!r}, content_types="
                    f"{[getattr(b, 'type', None) for b in resp.content]!r}"
                )
            response.content = text_block.text
            response.usage.prompt_tokens = resp.usage.input_tokens
            response.usage.completion_tokens = resp.usage.output_tokens
            return response

        response_provider = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            timeout=150,
            stop=None,
        )
        response.content = response_provider.choices[0].message.content
        response.usage.prompt_tokens = response_provider.usage.prompt_tokens
        response.usage.completion_tokens = response_provider.usage.completion_tokens
        return response


# Back-compat alias: any caller (including downstream tests) that imported
# the old class name still works.
OpenAIClient = LLMClient


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text."""
    enc = encoding_for_model(model)
    return len(enc.encode(text))


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "limit_max_tokens": 4096,
    "temperature": 0,
}
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_ANTHROPIC_MODEL = "claude-fable-5"
ALLOWED_TOOLS = ["superforcaster-polymarket-v3"]
ALLOWED_MODELS = [DEFAULT_OPENAI_MODEL, DEFAULT_ANTHROPIC_MODEL]
MAX_SOURCES = 5
COMPLETION_RETRIES = 3
COMPLETION_DELAY = 2


PREDICTION_PROMPT = """
You are an advanced AI system which has been finetuned to provide calibrated probabilistic
forecasts under uncertainty, with your performance evaluated according to the Brier score. When
forecasting, do not treat 0.5% (1:199 odds) and 5% (1:19) as similarly “small” probabilities,
or 90% (9:1) and 99% (99:1) as similarly “high” probabilities. As the odds show, they are
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

4. Aggregate your considerations. Do not summarize or repeat previous points; instead,
investigate how the competing factors and mechanisms interact and weigh against each other.
Factorize your thinking across (exhaustive, mutually exclusive) cases if and only if it would be
beneficial to your reasoning. We have detected that you overestimate world conflict, drama,
violence, and crises due to news' negativity bias, which doesn't necessarily represent overall
trends or base rates. Similarly, we also have detected you overestimate dramatic, shocking,
or emotionally charged news due to news' sensationalism bias. Therefore adjust for news'
negativity bias and sensationalism bias by considering reasons to why your provided sources
might be biased or exaggerated. Think like a superforecaster. Use <thinking></thinking> tags
for this section of your response.

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
* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* The JSON must contain four fields: "p_yes", "p_no", "confidence", and "info_utility".
* Each item in the JSON must have a value between 0 and 1.
   - "p_yes": Estimated probability that the event in the "Question" occurs.
   - "p_no": Estimated probability that the event in the "Question" does not occur.
   - "confidence": A value between 0 and 1 indicating the confidence in the prediction. 0 indicates lowest
     confidence value; 1 maximum confidence value.
   - "info_utility": Utility of the information provided in "sources" to help you make the prediction.
     0 indicates lowest utility; 1 maximum utility.
* The sum of "p_yes" and "p_no" must equal 1.
* Output only the JSON object. Do not include any other contents in your response.
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
            # Empty content must engage the retry loop, NOT return None
            # as the prediction. The Anthropic branch can produce this
            # state if a future code path stops raising on missing-text
            # (today the LLMClient raises in that case); guard here too.
            if content is None:
                raise ValueError("LLM returned empty content")
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


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """Run the task"""
    tool = kwargs["tool"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    model = kwargs.get("model")
    if model is None:
        raise ValueError("Model not supplied.")
    if model not in ALLOWED_MODELS:
        # With two SDKs a typo'd model string routes through
        # ``_provider_for()`` to the wrong client and fails deep in the
        # retry loop with an opaque SDK error. Enforce the allow-list
        # here so the wire-name error is unambiguous.
        raise ValueError(
            f"Model {model!r} is not in ALLOWED_MODELS={ALLOWED_MODELS!r}."
        )

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

    # LLMClientManager picks the SDK by model name; the keychain carries
    # both ``openai`` and ``anthropic`` keys (matches the prediction_request,
    # prediction_request_rag, etc. tools' convention).
    source_content = kwargs.get("source_content", None)
    return_source_content = (
        kwargs["api_keys"].get("return_source_content", "false") == "true"
    )
    source_content_mode = kwargs["api_keys"].get("source_content_mode", "cleaned")
    if source_content_mode not in ("cleaned", "raw"):
        raise ValueError(
            f"Invalid source_content_mode: {source_content_mode!r}. Must be 'cleaned' or 'raw'."
        )
    with LLMClientManager(kwargs["api_keys"], model) as llm_client:
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
        extracted_block, counter_callback = generate_prediction_with_retry(
            client=llm_client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=COMPLETION_RETRIES,
            delay=COMPLETION_DELAY,
            counter_callback=counter_callback,
        )

        used_params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if return_source_content:
            used_params["source_content"] = captured_source_content
        return extracted_block, prediction_prompt, None, counter_callback, used_params
