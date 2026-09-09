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
"""Mech tool that serves a fine-tuned 14B Qwen prediction model via vLLM.

This tool is the deployment counterpart of the `fine_tuning` pipeline.
That pipeline fine-tunes `unsloth/DeepSeek-R1-Distill-Qwen-14B` with
GRPO to emit a calibrated `p_yes` for a binary prediction-market question. This
tool runs the fine-tuned checkpoints behind a vLLM OpenAI-compatible endpoint
and returns the same JSON schema the other prediction tools deliver.

Three modes (base / fine-tuned / fine-tuned-calibrated)
-------------------------------------------------------
The package is registered under three tool names — `predict-base`,
`predict-fine-tuned` and `predict-fine-tuned-calibrated` — and the tool NAME is
the only selector: each maps to a fixed vLLM served-model name (MODEL_BY_TOOL).
All are DeepSeek-R1-Distill-Qwen-14B underneath; whether the fine-tuned one is a
merged checkpoint or a runtime LoRA adapter is a serving detail invisible to
this tool (it just sends a `model` name). The calibrated mode targets a third
served name that ft-serve fronts vLLM with — the fine-tuned model with a Platt
calibrator applied to `p_yes` — so `predict-fine-tuned` stays the honest raw
model and `predict-fine-tuned-calibrated` is opt-in by name. The requester
picks only the tool, so there is no untrusted model input. The production
analogue of fine_tuning test.py's base-vs-fine-tuned(-calibrated) comparison.

Why a dedicated tool (not a parametrised `superforcaster`)
---------------------------------------------------------
The reason is the BACKEND and lifecycle, not the prompt: the model runs on a
self-hosted vLLM endpoint (OpenAI client + `base_url`), needs a reasoning-sized
token budget, has its own pricing, and must not be entangled with the
production `superforcaster`. `superforcaster`'s tiktoken token-counting also has
no Qwen encoding.

Training-parity framing (see fine_tuning prompting.py `to_chat_format`):
the model was trained on a SINGLE user message (NO system message) whose content
is the full deliver-side forecaster prompt, with retrieved evidence embedded in
`<background>` / `<additional_information>` XML tags, and it emits a reasoning
output (`<think>…</think>` then a flat JSON object). build_messages mirrors that
framing exactly; the constant-system-message "mech-parity" shape was a
benchmark-only artifact and is intentionally NOT used here.

Parsing parity
--------------
`extract_json` / `parse_p_yes` are vendored from
`fine_tuning/src/fine_tuning/training/reward.py` (the single source of truth
there). They MUST stay behaviourally equivalent — same regexes and parsing
logic — so the reward used to train the model and the parser used to score its
deliveries agree; otherwise production parsing silently diverges from the
benchmark. (Only cosmetics differ here: type-hint syntax and docstrings.) If the
upstream parser changes, mirror the change here. See the pinned commit on the
parsing block below.
"""

import functools
import json
import re
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai
import requests

MechResponseWithKeys = Tuple[
    str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]], Any
]
MechResponse = Tuple[
    str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]]
]
MaxCostResponse = float

# ---------------------------------------------------------------------------
# Tool + model configuration
# ---------------------------------------------------------------------------

N_MODEL_CALLS = 1
DEFAULT_DELIVERY_RATE = 100

# Three modes, each a fixed vLLM served-model name. The tool NAME is the only
# selector — `predict-base` calls the base model, `predict-fine-tuned` calls the
# raw fine-tuned model, and `predict-fine-tuned-calibrated` calls the calibrated
# served name (the fine-tuned model with a Platt calibrator applied to p_yes,
# which ft-serve exposes as a third vLLM served name). All are
# DeepSeek-R1-Distill-Qwen-14B underneath; how the fine-tuned one is produced
# (LoRA weights merged into a standalone checkpoint, or a runtime adapter) is a
# SERVING detail invisible to this tool — either way vLLM exposes it under the
# name below. The requester does not choose the model; it only picks the tool,
# so there is no untrusted model input.
TOOL_BASE = "predict-base"
TOOL_FINE_TUNED = "predict-fine-tuned"
TOOL_FINE_TUNED_CALIBRATED = f"{TOOL_FINE_TUNED}-calibrated"
ALLOWED_TOOLS = [TOOL_BASE, TOOL_FINE_TUNED, TOOL_FINE_TUNED_CALIBRATED]

# vLLM --served-model-name for each mode. Edit to match your vLLM deployment.
# The calibrated name is a VIRTUAL served name ft-serve's proxy adds in front of
# vLLM (the fine-tuned model with Platt calibration applied to p_yes); it MUST
# match the SERVED_MODEL_* constants in investigation_ml serve.py, pinned in
# lockstep (a mismatch → the tool requests a name vLLM doesn't expose → 404).
# Adding a different base model later means a follow-up design (model becomes a
# second axis); today there is exactly one base.
SERVED_MODEL_BASE = "qwen-14b-base"
SERVED_MODEL_FINE_TUNED = "qwen-14b-fine-tuned"
SERVED_MODEL_FINE_TUNED_CALIBRATED = f"{SERVED_MODEL_FINE_TUNED}-calibrated"

MODEL_BY_TOOL = {
    TOOL_BASE: SERVED_MODEL_BASE,
    TOOL_FINE_TUNED: SERVED_MODEL_FINE_TUNED,
    TOOL_FINE_TUNED_CALIBRATED: SERVED_MODEL_FINE_TUNED_CALIBRATED,
}

# Self-hosted vLLM OpenAI-compatible endpoint. One server hosts both models, so
# the two modes share it. This is only the DEFAULT: a deployment points the tool
# at its own server via api_keys[API_KEYS_ENDPOINT_SERVICE] (the KeyChain is the
# only config channel that reaches run() in production — env vars do not, since
# the component runs as published-from-IPFS bytes). The endpoint rides the same
# channel as the key rather than an env var for that reason.
VLLM_ENDPOINT = "http://localhost:8000/v1"

# vLLM does not require a real key by default; the OpenAI client still demands a
# non-empty string. A secured gateway can override via api_keys["finetuned"].
DUMMY_API_KEY = "EMPTY"
API_KEYS_SERVICE = "finetuned"
# KeyChain service carrying the vLLM base_url. Absent/empty → VLLM_ENDPOINT.
API_KEYS_ENDPOINT_SERVICE = "finetuned_endpoint"


def resolve_model(tool: str) -> str:
    """The vLLM served-model name for `tool` (the mode)."""
    return MODEL_BY_TOOL[tool]


# Generation settings mirror fine_tuning's mech-parity evaluation:
# temperature 0.0 (deterministic, matches the GPT-4.1 deliver-time setting the
# model was compared against) and a 1024-token budget — a reasoning model needs
# headroom for its <think> block, so superforcaster's 500-token cap is wrong here.
DEFAULT_SETTINGS = {
    "temperature": 0.0,
    "max_tokens": 1024,
}

MAX_SOURCES = 5
# Single inference attempt: with the 150s per-attempt timeout, one Serper call
# (≤60s) + one attempt stays under the 240s task deadline. Raise this (and lower
# the timeout to keep the product under 240s) only if transient vLLM failures
# warrant retries.
COMPLETION_RETRIES = 1
COMPLETION_DELAY = 2
COMPLETION_TIMEOUT = 150

# Placeholder tokens substituted into PREDICTION_TEMPLATE. Sentinels (not
# str.format) are used because the vendored template contains literal JSON
# braces that would break str.format.
QUESTION_PLACEHOLDER = "__QUESTION__"
TODAY_PLACEHOLDER = "__TODAY__"
SOURCES_PLACEHOLDER = "__SOURCES__"
DATE_FORMAT = "%d/%m/%Y"

# The in-distribution forecaster prompt. Vendored verbatim from the dominant
# <background> template in fine_tuning's out_evidence/train.parquet (the
# research-rich corpus the deployed checkpoint trained on) — extracted and
# round-trip-verified against that data, with the question / date / sources
# fields replaced by the sentinel tokens above. The model is robust to the
# corpus's template variation (GRPO over a heterogeneous mix), so the dominant
# variant is a faithful representative; we do NOT chase byte-exact parity with
# every variant. Regenerate via scripts if the corpus template shifts.
PREDICTION_TEMPLATE = """
You are an advanced AI system which has been finetuned to provide calibrated probabilistic
forecasts under uncertainty, with your performance evaluated according to the Brier score. When
forecasting, do not treat 0.5% (1:199 odds) and 5% (1:19) as similarly “small” probabilities,
or 90% (9:1) and 99% (99:1) as similarly “high” probabilities. As the odds show, they are
markedly different, so output your probabilities accordingly.

Question:
__QUESTION__

Today's date: __TODAY__

We have retrieved the following information for this question:
<background>
__SOURCES__</background>

Recall the question you are forecasting:
__QUESTION__

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
* This is incorrect:"```json{
  "p_yes": 0.2,
  "p_no": 0.8,
  "confidence": 0.7,
  "info_utility": 0.5
}```"
* This is incorrect:```json"{
  "p_yes": 0.2,
  "p_no": 0.8,
  "confidence": 0.7,
  "info_utility": 0.5
}"```
* This is correct:"{
  "p_yes": 0.2,
  "p_no": 0.8,
  "confidence": 0.7,
  "info_utility": 0.5
}"
"""


# ---------------------------------------------------------------------------
# Output parsing — vendored from fine_tuning reward.py (keep in sync)
# ---------------------------------------------------------------------------
# Vendored from valory-xyz/fine-tuning @ 5551073
# (src/fine_tuning/training/reward.py). Re-sync on every upstream parser change
# so production parsing cannot drift from the training-time reward. Behaviourally
# identical to that source; only the type-hint syntax and docstrings differ (the
# regexes and parsing logic are the same).

# Strip the <think>...</think> block (non-greedy, multiline).
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Match a flat JSON object: from the first '{' to the first '}'.
JSON_RE = re.compile(r"\{[^}]*\}")


def _to_text(completion: Union[str, List[Dict[str, str]]]) -> str:
    """Normalise a completion (string or chat-message list) to plain text."""
    if isinstance(completion, list):
        return " ".join(
            msg.get("content", "") for msg in completion if isinstance(msg, dict)
        )
    return completion or ""


def extract_json(
    completion: Union[str, List[Dict[str, str]]],
) -> Optional[Dict[str, Any]]:
    """Strip the <think> block and parse the remaining flat JSON object.

    Returns None if the response has no parseable JSON object.

    :param completion: the model output (string or chat-message list).
    :return: the parsed JSON object, or None if not parseable.
    """
    text = _to_text(completion)
    if not text:
        return None
    stripped = THINK_BLOCK_RE.sub("", text).strip()
    match = JSON_RE.search(stripped)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def parse_p_yes(completion: Union[str, List[Dict[str, str]]]) -> Optional[float]:
    """Extract `p_yes` from a model response. Returns None if invalid.

    Invalid if: no parseable JSON object, no `p_yes` key, or `p_yes` is not a
    float in [0, 1].

    :param completion: the model output (string or chat-message list).
    :return: the parsed p_yes in [0, 1], or None if invalid.
    """
    obj = extract_json(completion)
    if not isinstance(obj, dict):
        return None
    if "p_yes" not in obj:
        return None
    try:
        p_yes = float(obj["p_yes"])
    except (TypeError, ValueError):
        return None
    if not 0.0 <= p_yes <= 1.0:
        return None
    return p_yes


def canonical_prediction(completion: Optional[str]) -> Optional[str]:
    """Build the canonical delivery JSON from a raw model completion.

    The model emits `<think>…</think>{json}`. Mech consumers expect a clean JSON
    object (no reasoning block) carrying at least `p_yes`/`p_no`. We re-derive a
    normalised object from the parsed completion so `p_no` is always present and
    consistent with `p_yes`, defaulting confidence/info_utility when the model
    omitted them. Returns None when `p_yes` could not be parsed.

    :param completion: the raw model completion (or None).
    :return: the canonical delivery JSON string, or None if p_yes is unparseable.
    """
    if completion is None:
        return None
    p_yes = parse_p_yes(completion)
    if p_yes is None:
        return None
    obj = extract_json(completion) or {}
    result = {
        "p_yes": p_yes,
        "p_no": round(1.0 - p_yes, 6),
        "confidence": _coerce_unit_interval(obj.get("confidence")),
        "info_utility": _coerce_unit_interval(obj.get("info_utility")),
    }
    return json.dumps(result)


def _coerce_unit_interval(value: Any, default: float = 0.5) -> float:
    """Coerce a value to a float in [0, 1], falling back to `default`."""
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if not 0.0 <= coerced <= 1.0:
        return default
    return coerced


# ---------------------------------------------------------------------------
# API key rotation (framework contract: return value must end with api_keys)
# ---------------------------------------------------------------------------


def with_key_rotation(func: Callable) -> Callable:
    """Retry on rate limits, rotating any configured key services.

    Unlike superforcaster's variant, this does not hard-code openai/openrouter:
    it rotates whatever services the KeyChain exposes (e.g. `finetuned`,
    `serperapi`), so a self-hosted deployment with no OpenAI key does not
    KeyError. The wrapper always returns the tool result with `api_keys`
    appended, as the mech task-execution layer requires.

    :param func: the tool entrypoint to wrap.
    :return: the wrapped function that retries with key rotation.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponseWithKeys:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponseWithKeys:
            """Run the tool, rotating keys on rate-limit errors."""
            try:
                result = func(*args, **kwargs)
                # The delivery_rate==0 cost-estimation path returns a bare
                # MaxCostResponse (float), which carries no api_keys to append.
                # Only the normal MechResponse tuple gets the keychain appended.
                if not isinstance(result, tuple):
                    return result
                return result + (api_keys,)
            except openai.RateLimitError as e:
                if all(remaining <= 0 for remaining in retries_left.values()):
                    raise e
                for service, remaining in retries_left.items():
                    if remaining > 0:
                        retries_left[service] -= 1
                        api_keys.rotate(service)
                return execute()
            except Exception as e:  # noqa: BLE001 — surface any error as a result
                return str(e), "", None, None, None, api_keys

        return execute()

    return wrapper


# ---------------------------------------------------------------------------
# vLLM (OpenAI-compatible) client
# ---------------------------------------------------------------------------


class Usage:
    """Token usage container."""

    def __init__(
        self,
        prompt_tokens: Optional[Any] = None,
        completion_tokens: Optional[Any] = None,
    ):
        """Initialise with prompt and completion token counts."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class LLMResponse:
    """Normalised LLM response (content + usage)."""

    def __init__(self, content: Optional[str] = None):
        """Initialise with content and an empty usage record."""
        self.content = content
        self.usage = Usage()


class VLLMClient:
    """OpenAI-compatible client pointed at a self-hosted vLLM endpoint."""

    def __init__(self, api_key: str, base_url: str):
        """Initialise the OpenAI client against the vLLM `base_url`."""
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: int = COMPLETION_TIMEOUT,
    ) -> LLMResponse:
        """Generate one chat completion from the vLLM-served model."""
        provider_response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            timeout=timeout,
            stop=None,
        )
        response = LLMResponse()
        response.content = provider_response.choices[0].message.content
        usage = provider_response.usage
        if usage is not None:
            response.usage.prompt_tokens = usage.prompt_tokens
            response.usage.completion_tokens = usage.completion_tokens
        return response


class VLLMClientManager:
    """Context manager that opens and closes a `VLLMClient`."""

    def __init__(self, api_key: str, base_url: str):
        """Store the key and endpoint for lazy client creation."""
        self.api_key = api_key
        self.base_url = base_url
        self._client: Optional[VLLMClient] = None

    def __enter__(self) -> VLLMClient:
        """Open the client."""
        self._client = VLLMClient(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the underlying OpenAI client."""
        if self._client is not None:
            self._client.client.close()
            self._client = None


def generate_prediction_with_retry(
    client: VLLMClient,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    retries: int = COMPLETION_RETRIES,
    delay: int = COMPLETION_DELAY,
    counter_callback: Optional[Callable] = None,
) -> Tuple[Optional[str], Optional[Callable]]:
    """Generate a completion, retrying transient failures with a backoff."""
    attempt = 0
    while attempt < retries:
        try:
            response = client.completions(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response.content is not None and counter_callback is not None:
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=model,
                )
            return response.content, counter_callback
        except Exception as e:  # noqa: BLE001 — retry any transient inference error
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(delay)
            attempt += 1
    raise Exception("Failed to generate prediction after retries")


# ---------------------------------------------------------------------------
# Web research (borrowed from superforcaster)
# ---------------------------------------------------------------------------


def fetch_additional_sources(question: str, serper_api_key: str) -> requests.Response:
    """Fetch web results for `question` via the Serper API."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": question})
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json",
    }
    return requests.request("POST", url, headers=headers, data=payload, timeout=60)


def format_sources_data(organic_data: Any, misc_data: Any) -> str:
    """Format organic + 'People Also Ask' results into the <background> body.

    Reproduced VERBATIM from superforcaster.format_sources_data (indentation and
    markdown included) because this exact string shape is what populated the
    <background> blocks the model trained on — see fine_tuning
    out_evidence/train.parquet. Changing the formatting would drift the inner
    evidence text away from the training distribution.

    :param organic_data: Serper organic results.
    :param misc_data: Serper 'People Also Ask' results.
    :return: the formatted evidence block.
    """
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
    """Extract the market question from the mech prompt via regex."""
    pattern = r'question\s+"(.+?)"\s+and\s+the\s+`yes`'
    try:
        return re.findall(pattern, prompt, re.DOTALL)[0]
    except Exception as e:  # noqa: BLE001 — fall back to the whole prompt
        print(f"Error extracting question: {e}")
        return prompt


def gather_sources(question: str, serper_api_key: str) -> str:
    """Run the question through Serper and format the top results.

    Fails (raises, so with_key_rotation returns the error for the mech to
    handle) on a Serper request error OR when Serper returns no usable results:
    the model was trained only on research-backed prompts, so an empty
    `<background>` block is out-of-distribution — we fail the prediction with an
    explanation rather than forecast on zero web context.

    :param question: the market question to search.
    :param serper_api_key: the Serper API key.
    :return: the formatted `<background>` evidence block.
    """
    try:
        response = fetch_additional_sources(question, serper_api_key)
        data = response.json()
    except Exception as exc:  # noqa: BLE001 — surface as an explanatory failure
        raise RuntimeError(f"Web search (Serper) request failed: {exc}") from exc

    organic = data.get("organic", [])[:MAX_SOURCES]
    misc = data.get("peopleAlsoAsk", [])
    if not organic and not misc:
        raise RuntimeError(
            "Web search (Serper) returned no results; this model requires web "
            "context and cannot forecast without it."
        )
    return format_sources_data(organic, misc)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def build_forecaster_prompt(question: str, today: str, sources: str) -> str:
    """Fill the vendored <background> forecaster template.

    Sentinel substitution (not str.format) because PREDICTION_TEMPLATE contains
    literal JSON braces. `question` is inserted at both the `Question:` header
    and the trailing `Recall the question…` echo.

    :param question: the market question.
    :param today: the current date string.
    :param sources: the formatted <background> evidence block.
    :return: the filled forecaster prompt.
    """
    return (
        PREDICTION_TEMPLATE.replace(SOURCES_PLACEHOLDER, sources)
        .replace(TODAY_PLACEHOLDER, today)
        .replace(QUESTION_PLACEHOLDER, question)
    )


def build_messages(content: str) -> List[Dict[str, str]]:
    """Wrap the prompt content in training-parity chat framing.

    The model was trained via fine_tuning's `to_chat_format`: a SINGLE
    user message, NO system message. We mirror that exactly — adding a system
    message (as the mech-parity *benchmark* did) would push the model out of
    distribution relative to how it was trained.

    :param content: the user-message content (the forecaster prompt).
    :return: the single-user-message chat list.
    """
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """Run the fine-tuned prediction tool.

    Expected kwargs (supplied by the mech task-execution layer):
      - tool:        one of ALLOWED_TOOLS — selects base vs fine-tuned MODE.
      - prompt:      the bare prediction-market request prompt.
      - api_keys:    KeyChain; `serperapi` (required), `finetuned` (optional),
                     `finetuned_endpoint` (optional vLLM base_url override).
      - delivery_rate, counter_callback: mech cost-accounting plumbing.

    The served model is fixed per mode (MODEL_BY_TOOL[tool]); the endpoint comes
    from api_keys[`finetuned_endpoint`], falling back to the VLLM_ENDPOINT
    constant. The requester chooses neither.

    :param kwargs: the mech task kwargs described above.
    :return: the mech response tuple, or the max-cost float for delivery_rate=0.
    """
    tool = kwargs["tool"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    model = resolve_model(tool)

    delivery_rate = int(kwargs.get("delivery_rate", DEFAULT_DELIVERY_RATE))
    counter_callback: Optional[Callable[..., Any]] = kwargs.get("counter_callback")
    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback was "
                "given to calculate the max cost with."
            )
        return counter_callback(max_cost=True, models_calls=(model,) * N_MODEL_CALLS)

    api_keys = kwargs["api_keys"]
    llm_api_key = _optional_key(api_keys, API_KEYS_SERVICE) or DUMMY_API_KEY
    base_url = _optional_key(api_keys, API_KEYS_ENDPOINT_SERVICE) or VLLM_ENDPOINT

    prompt = kwargs["prompt"]
    temperature = float(kwargs.get("temperature", DEFAULT_SETTINGS["temperature"]))
    max_tokens = int(kwargs.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]))

    # Reproduce the production pipeline: the tool receives the bare-question
    # prompt, pulls the question, runs web search, and builds the <background>
    # forecaster prompt the model trained on.
    serper_api_key = api_keys["serperapi"]
    question = extract_question(prompt)
    sources = gather_sources(question, serper_api_key)
    today = date.today().strftime(DATE_FORMAT)
    content = build_forecaster_prompt(question, today, sources)

    messages = build_messages(content)

    with VLLMClientManager(llm_api_key, base_url) as llm_client:
        completion, counter_callback = generate_prediction_with_retry(
            client=llm_client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            counter_callback=counter_callback,
        )

    result = canonical_prediction(completion)
    if result is None:
        raise ValueError(
            "Model output did not contain a parseable p_yes. Raw completion: "
            f"{completion!r}"
        )

    used_params = {
        "tool": tool,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return result, completion, None, counter_callback, used_params


def _optional_key(api_keys: Any, service: str) -> Optional[str]:
    """Return the key for `service`, or None if the KeyChain lacks it."""
    try:
        return api_keys[service]
    except Exception:  # noqa: BLE001 — KeyChain raises various types when absent
        return None
