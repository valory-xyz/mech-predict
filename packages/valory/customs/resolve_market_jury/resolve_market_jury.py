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
"""Multi-model jury tool for resolving prediction markets.

Fans out a market question to N independent AI voters (each with native web search),
then an Anthropic Claude judge synthesizes the final verdict.

Drop-in replacement for resolve_market_reasoning -- same input kwargs, same output
tuple shape, same JSON result schema ({is_valid, is_determinable, has_occurred}).
"""

import functools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai

# ---------------------------------------------------------------------------
# Types (mech tool contract)
# ---------------------------------------------------------------------------

MechResponseWithKeys = Tuple[
    str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]], Any
]
MechResponse = Tuple[
    str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]]
]
MaxCostResponse = float

DEFAULT_DELIVERY_RATE = 100


# ---------------------------------------------------------------------------
# Voter / Judge configuration
# ---------------------------------------------------------------------------

VOTER_MODEL_OPENAI = "openai/gpt-4.1:online"
VOTER_MODEL_GROK = "x-ai/grok-4.1-fast:online"
VOTER_MODEL_GEMINI = "google/gemini-2.5-flash:online"
VOTER_MODEL_CLAUDE = "anthropic/claude-haiku-4.5:online"
JUDGE_MODEL_CLAUDE = "anthropic/claude-sonnet-4:online"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


VOTER_MAX_RETRIES = 1
VOTER_MAX_TOKENS = 1024
VOTER_RETRY_DELAY = 5
VOTER_TIMEOUT = 120
JUDGE_MAX_RETRIES = 3
JUDGE_MAX_TOKENS = 4096
JUDGE_RETRY_DELAY = 5
JUDGE_TIMEOUT = 120

ALLOWED_TOOLS = [
    "resolve-market-jury-v1",
]

TOOL_TO_ENGINE = {
    "resolve-market-jury-v1": JUDGE_MODEL_CLAUDE,
}


def _noop_token_counter(*_args: Any, **_kwargs: Any) -> int:
    """No-op token counter passed to TokenCounterCallback."""
    return 0


# ---------------------------------------------------------------------------
# Shared voter prompt
# ---------------------------------------------------------------------------

VOTER_PROMPT = """You are an expert fact checker. You have access to web search. \
A prediction market question asked whether an event would happen before a given \
date. That date has now passed. Your role is to determine whether the event \
actually happened before the date.

INSTRUCTIONS:
* Search the web for recent, reliable information about the question below.
* Think through the problem step by step, showing your reasoning:
  1. Identify the key event described and the deadline date.
  2. Search for credible news articles, official statements, or records.
  3. Pay attention to dates -- an article dated BEFORE the deadline reporting the \
event happened is strong evidence it occurred. An article dated AFTER the deadline \
discussing whether it WILL happen suggests it did not.
  4. Consider the intent and spirit of the question, not just literal keywords. \
For example, legislation "addressing AI's impact on the workforce" reasonably \
covers white-collar employment even without that exact phrase.
* There are only two possible outcomes: the event happened (true) or it did not \
(false). If your confidence is below 0.7, set is_determinable to false -- do \
NOT guess when evidence is insufficient.

VALIDITY RULES:
* Questions with relative dates ("in 6 months") are invalid.
* Questions about opinions rather than facts are invalid.

Question: "{question}"

CRITICAL: Respond with ONLY valid JSON. No markdown, no text before or after.
CONSISTENCY RULES:
- If has_occurred is true or false, then is_determinable MUST be true and confidence \
should be >= 0.7.
- If has_occurred is null, then is_determinable MUST be false and confidence should \
be < 0.7.
- confidence reflects how sure you are of your answer (0.0 = no idea, 1.0 = certain).
{{
    "is_valid": true,
    "is_determinable": true or false,
    "has_occurred": true or false or null,
    "confidence": 0.0 to 1.0,
    "reasoning": "Step-by-step explanation, 200 words max. What you found, why you reached this verdict.",
    "sources": ["url1", "url2"]
}}"""

JUDGE_PROMPT = """You are a senior analyst synthesizing independent fact-checker \
assessments of a prediction market question. You have access to web search -- \
use it to verify disputed claims when the voters disagree.

Question: "{question}"

Voter assessments:
{votes}

DECISION PROCESS:
1. Review each voter's evidence and sources, not just their verdict.
2. If all voters with a definitive answer agree, follow their consensus.
3. If voters disagree:
   a. Count definitive votes (ignore indeterminate ones).
   b. Search the web to verify the specific claims in dispute.
   c. Follow the majority UNLESS your own research or the minority's sources \
show a clear factual error in the majority's reasoning.
   d. When evidence quality is similar on both sides, follow the majority.
4. If no clear majority exists, or evidence is too weak, set is_determinable \
to false.
5. If a voter flags the question as invalid with sound reasoning, mark invalid.

Respond in JSON only (no markdown fences, no text before or after):
{{
    "is_valid": true or false,
    "is_determinable": true or false,
    "has_occurred": true or false or null,
    "judge_reasoning": "Which voters you agreed with and why. Cite evidence."
}}"""


# ---------------------------------------------------------------------------
# VoterResult
# ---------------------------------------------------------------------------


@dataclass
class VoterResult:
    """Uniform output from every voter."""

    voter: str
    model: str
    is_valid: Optional[bool] = None
    is_determinable: Optional[bool] = None
    has_occurred: Optional[bool] = None
    confidence: float = 0.0
    reasoning: str = ""
    sources: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Voter registry
# ---------------------------------------------------------------------------


@dataclass
class VoterConfig:
    """Configuration for a single voter."""

    adapter: str  # "_adapter_openai" or "_adapter_openrouter"
    model: str  # model name / slug
    key_name: str  # KeyChain service name


VOTER_CONFIG: Dict[str, VoterConfig] = {
    "openai": VoterConfig(
        adapter="_adapter_openrouter",
        model=VOTER_MODEL_OPENAI,
        key_name="openrouter",
    ),
    "grok": VoterConfig(
        adapter="_adapter_openrouter",
        model=VOTER_MODEL_GROK,
        key_name="openrouter",
    ),
    "gemini": VoterConfig(
        adapter="_adapter_openrouter",
        model=VOTER_MODEL_GEMINI,
        key_name="openrouter",
    ),
    "claude": VoterConfig(
        adapter="_adapter_openrouter",
        model=VOTER_MODEL_CLAUDE,
        key_name="openrouter",
    ),
}

DEFAULT_VOTERS: List[str] = list(VOTER_CONFIG.keys())


# ---------------------------------------------------------------------------
# JSON parsing (shared across all adapters)
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from a response that may contain markdown fences or extra text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    # Find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _parse_vote(raw: str, voter: str, model: str) -> VoterResult:
    """Parse raw LLM text into a VoterResult."""
    data = _extract_json(raw)
    if data is None:
        return VoterResult(
            voter=voter,
            model=model,
            error=f"Unparseable JSON: {raw[:200]}",
        )
    has_occurred = data.get("has_occurred")
    is_determinable = data.get("is_determinable")
    confidence = float(data.get("confidence", 0.0))

    # Enforce consistency between fields
    if has_occurred is None:
        is_determinable = False
        confidence = min(confidence, 0.5)
    if is_determinable is False:
        has_occurred = None
        confidence = min(confidence, 0.5)

    return VoterResult(
        voter=voter,
        model=model,
        is_valid=data.get("is_valid"),
        is_determinable=is_determinable,
        has_occurred=has_occurred,
        confidence=confidence,
        reasoning=data.get("reasoning", ""),
        sources=data.get("sources", []),
    )


# ---------------------------------------------------------------------------
# Adapter: OpenRouter
# ---------------------------------------------------------------------------


def _adapter_openrouter(
    model: str,
    prompt: str,
    api_key: str,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    retry_delay: int,
    counter_callback: Optional[Callable] = None,
) -> str:
    """Make an OpenRouter chat completion call and return the raw text.

    Records token usage + per-call surcharge to ``counter_callback``.
    Retries on 529 (overloaded) errors up to ``max_retries`` attempts.

    :param model: OpenRouter model slug.
    :param prompt: prompt to send.
    :param api_key: OpenRouter API key.
    :param max_tokens: max output tokens.
    :param timeout: per-request timeout in seconds.
    :param max_retries: total attempts (>=1). Only 529 errors trigger a retry.
    :param retry_delay: seconds to sleep between retry attempts.
    :param counter_callback: optional token/cost accounting callback.
    :return: raw text from the response (may be empty string).
    """
    client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    for attempt in range(max_retries):  # pragma: no branch
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            break
        except openai.APIStatusError as err:
            if err.status_code == 529 and attempt < max_retries - 1:
                print(
                    f"  {model} overloaded, retrying in {retry_delay}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(retry_delay)
            else:
                raise
    raw = response.choices[0].message.content or ""

    # Forward token usage and real call cost to the callback so that
    # total_cost matches what OpenRouter billed (including any
    # web-search surcharge or routing markup).
    usage = getattr(response, "usage", None)
    if counter_callback is not None and usage is not None:
        try:
            counter_callback(
                model=model,
                token_counter=_noop_token_counter,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                call_cost=getattr(usage, "cost", None),
            )
        except Exception as e:  # pylint: disable=broad-except
            print(f"  Warning: counter_callback failed for {model}: {e}")

    return raw


_ADAPTERS: Dict[str, Callable] = {
    "_adapter_openrouter": _adapter_openrouter,
}


# ---------------------------------------------------------------------------
# Voting facade
# ---------------------------------------------------------------------------


def cast_vote(
    voter_name: str,
    question: str,
    api_keys: Any,
    counter_callback: Optional[Callable] = None,
) -> VoterResult:
    """Uniform entry point -- delegates to provider-specific adapter.

    :param voter_name: registry key for this voter.
    :param question: market question.
    :param api_keys: KeyChain object.
    :param counter_callback: optional token/cost accounting callback.
    :return: parsed vote result.
    """
    config = VOTER_CONFIG[voter_name]
    api_key = api_keys[config.key_name]
    prompt = VOTER_PROMPT.format(question=question)
    model = config.model
    adapter_fn = _ADAPTERS[config.adapter]
    raw = adapter_fn(
        model=model,
        prompt=prompt,
        api_key=api_key,
        max_tokens=VOTER_MAX_TOKENS,
        timeout=VOTER_TIMEOUT,
        max_retries=VOTER_MAX_RETRIES,
        retry_delay=VOTER_RETRY_DELAY,
        counter_callback=counter_callback,
    )
    return _parse_vote(raw, voter_name, model)


def collect_votes(
    question: str,
    voter_names: List[str],
    api_keys: Any,
    counter_callback: Optional[Callable] = None,
) -> List[VoterResult]:
    """Fan out to all voters in parallel, collect results.

    :param question: market question.
    :param voter_names: list of voter registry keys to use.
    :param api_keys: KeyChain object.
    :param counter_callback: optional token/cost accounting callback.
    :return: list of voter results.
    """
    results: List[VoterResult] = []
    with ThreadPoolExecutor(max_workers=len(voter_names)) as pool:
        futures = {
            pool.submit(cast_vote, name, question, api_keys, counter_callback): name
            for name in voter_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result(timeout=VOTER_TIMEOUT + 30)
                results.append(result)
                print(
                    f"  Voter [{name}]: "
                    f"has_occurred={result.has_occurred}, "
                    f"is_determinable={result.is_determinable}, "
                    f"confidence={result.confidence}"
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"  Voter [{name}] failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def _run_judge(
    question: str,
    votes: List[VoterResult],
    api_key: str,
    counter_callback: Optional[Callable] = None,
) -> dict:
    """Judge -- synthesizes voter results into final verdict via OpenRouter.

    :param question: market question.
    :param votes: voter results to synthesize.
    :param api_key: OpenRouter API key.
    :param counter_callback: optional token/cost accounting callback.
    :return: judge verdict dict.
    """
    formatted_votes = ""
    for i, v in enumerate(votes, 1):
        vote_data = {
            "is_valid": v.is_valid,
            "is_determinable": v.is_determinable,
            "has_occurred": v.has_occurred,
            "confidence": v.confidence,
            "reasoning": v.reasoning,
            "sources": v.sources,
        }
        formatted_votes += f"\nVoter {i}:\n{json.dumps(vote_data, indent=2)}\n"

    prompt = JUDGE_PROMPT.format(question=question, votes=formatted_votes)
    raw = _adapter_openrouter(
        model=JUDGE_MODEL_CLAUDE,
        prompt=prompt,
        api_key=api_key,
        max_tokens=JUDGE_MAX_TOKENS,
        timeout=JUDGE_TIMEOUT,
        max_retries=JUDGE_MAX_RETRIES,
        retry_delay=JUDGE_RETRY_DELAY,
        counter_callback=counter_callback,
    )
    data = _extract_json(raw)
    if data is None:
        return {
            "is_valid": False,
            "is_determinable": False,
            "has_occurred": None,
            "judge_reasoning": f"Unparseable judge response: {raw[:200]}",
        }
    return data


# ---------------------------------------------------------------------------
# Consensus helpers
# ---------------------------------------------------------------------------


def _decided_votes(votes: List[VoterResult]) -> List[VoterResult]:
    """Filter to votes that are valid and determinable."""
    return [
        v
        for v in votes
        if v.is_determinable is not False
        and v.is_valid is not False
        and v.error is None
    ]


def _has_consensus(votes: List[VoterResult]) -> bool:
    """Check if a majority of all voters unanimously agree on has_occurred."""
    decided = _decided_votes(votes)
    # Need at least a majority of all voters to have decided
    if len(decided) < 2 or len(decided) <= len(votes) / 2:
        return False
    return all(v.has_occurred == decided[0].has_occurred for v in decided)


def _build_consensus_result(votes: List[VoterResult]) -> dict:
    """Build result from unanimous votes (skip judge)."""
    decided = _decided_votes(votes)
    return {
        "is_valid": True,
        "is_determinable": True,
        "has_occurred": decided[0].has_occurred,
        "votes": [asdict(v) for v in votes],
        "judge_reasoning": "Voter majority consensus -- judge skipped.",
        "agreement_ratio": 1.0,
        "n_voters": len(votes),
        "n_successful": len(votes),
        "n_decided": len(decided),
    }


def _compute_agreement(votes: List[VoterResult]) -> float:
    """Compute agreement ratio among decided votes."""
    decided = _decided_votes(votes)
    if not decided:
        return 0.0
    yes_count = sum(1 for v in decided if v.has_occurred is True)
    no_count = sum(1 for v in decided if v.has_occurred is False)
    return max(yes_count, no_count) / len(decided)


# ---------------------------------------------------------------------------
# @with_key_rotation decorator
# ---------------------------------------------------------------------------


def with_key_rotation(func: Callable) -> Callable:
    """Decorator that retries a function with API key rotation on failure."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponseWithKeys:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponseWithKeys:
            try:
                result: MechResponse = func(*args, **kwargs)
                return result + (api_keys,)
            except openai.RateLimitError:
                rotated = False
                for service in ("openai", "openrouter"):
                    if retries_left.get(service, 0) > 0:
                        retries_left[service] -= 1
                        api_keys.rotate(service)
                        rotated = True
                if not rotated:
                    raise
                return execute()
            except Exception as e:  # pylint: disable=broad-except
                print(f"Unexpected error in run(): {e}")
                return str(e), "", None, None, None, api_keys

        return execute()

    return wrapper


# ---------------------------------------------------------------------------
# run() -- mech tool entry point
# ---------------------------------------------------------------------------


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """Run the resolve_market_jury tool.

    :param kwargs: keyword arguments including prompt, tool, api_keys,
        delivery_rate, and counter_callback.
    :return: max cost float (if delivery_rate==0) or MechResponse tuple.
    """
    tool = kwargs["tool"]
    delivery_rate = int(kwargs.get("delivery_rate", DEFAULT_DELIVERY_RATE))
    counter_callback: Optional[Callable] = kwargs.get("counter_callback", None)
    api_keys = kwargs["api_keys"]
    prompt = kwargs["prompt"]

    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported. Allowed: {ALLOWED_TOOLS}")

    voters = DEFAULT_VOTERS

    # Cost calculation mode
    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback was given."
            )
        voter_models = tuple(VOTER_CONFIG[name].model for name in voters)
        max_cost = counter_callback(
            max_cost=True,
            models_calls=voter_models + (JUDGE_MODEL_CLAUDE,),
        )
        return max_cost

    # 1. Fan out to voters (parallel)
    print(f"Collecting votes from {voters}...")
    votes = collect_votes(prompt, voters, api_keys, counter_callback)

    # 2. Early exit: no successful votes
    if not votes:
        result: Dict[str, Any] = {
            "is_valid": False,
            "is_determinable": False,
            "has_occurred": None,
            "votes": [],
            "judge_reasoning": "All voters failed.",
            "agreement_ratio": 0.0,
            "n_voters": len(voters),
            "n_successful": 0,
            "n_decided": 0,
        }
        return json.dumps(result), "All voters failed.", None, counter_callback, None

    voter_models = [VOTER_CONFIG[v].model for v in voters]
    used_params = {
        "model": JUDGE_MODEL_CLAUDE,
        "voter_models": voter_models,
        "n_voters": len(voters),
    }

    # 3. Majority consensus early exit (cost saving -- skip judge)
    if _has_consensus(votes):
        print("  Voter majority consensus -- skipping judge.")
        result = _build_consensus_result(votes)
        used_params["model"] = voter_models[0]  # judge was not called
        return (
            json.dumps(result),
            result["judge_reasoning"],
            None,
            counter_callback,
            used_params,
        )

    # 4. Judge synthesizes (only when voters disagree or partial)
    print("  Voters disagree -- running judge...")
    verdict = _run_judge(prompt, votes, api_keys["openrouter"], counter_callback)

    # 5. Build result with vote metadata
    judge_reasoning = verdict.get("judge_reasoning", "")
    result = {
        "is_valid": verdict.get("is_valid", True),
        "is_determinable": verdict.get("is_determinable", True),
        "has_occurred": verdict.get("has_occurred"),
        "votes": [asdict(v) for v in votes],
        "judge_reasoning": judge_reasoning,
        "agreement_ratio": _compute_agreement(votes),
        "n_voters": len(voters),
        "n_successful": len(votes),
        "n_decided": len(_decided_votes(votes)),
    }

    return json.dumps(result), judge_reasoning, None, counter_callback, used_params
