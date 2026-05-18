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

Fans out a market question to N independent AI voters (each with web search),
then an AI judge synthesizes the final verdict.

Drop-in replacement for resolve_market_reasoning -- same input kwargs, same output
tuple shape, same JSON result schema ({is_valid, is_determinable, has_occurred}).
"""

import functools
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai

COUNTER_CALLBACK_LOCK = threading.Lock()

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
VOTER_MODEL_GROK = "x-ai/grok-4.3:online"
VOTER_MODEL_GEMINI = "google/gemini-2.5-flash:online"
VOTER_MODEL_CLAUDE = "anthropic/claude-haiku-4.5:online"
JUDGE_MODEL_CLAUDE = "anthropic/claude-sonnet-4:online"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


VOTER_MAX_ATTEMPTS = 1
VOTER_MAX_TOKENS = 1024
VOTER_RETRY_DELAY = 5
VOTER_TIMEOUT = 120
JUDGE_MAX_ATTEMPTS = 3
JUDGE_MAX_TOKENS = 4096
JUDGE_RETRY_DELAY = 5
JUDGE_TIMEOUT = 120

ALLOWED_TOOLS = [
    "resolve-market-jury-v1",
]


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
  5. SEMANTIC CHECKS before setting has_occurred:
     (a) NEGATION TRAP -- if the question uses negative framing ("still pending", \
"still in effect", "remain at", "not yet"), first answer it in plain English \
("Is X still pending? Yes/No"), then set has_occurred to the plain-English answer. \
"Still pending = true" means YES, the pending state holds. Cross-check that your \
final boolean matches your reasoning text.
     (b) NUMERIC RANGE -- for "at or above X" / "below X", a guidance \
RANGE [A, B] satisfies "at or above X" if A >= X (the lower endpoint \
already reaches the threshold) OR a confirmed point value >= X. \
Equality at the lower bound COUNTS as satisfying: a "2.5% to 5%" \
range satisfies "at or above 2.5%" because A == X == 2.5%. \
The rule only FAILS when the upper bound *alone* equals X with the \
lower bound strictly below it (e.g. a "1.5% to 2.5%" range does NOT \
reliably satisfy "at or above 2.5%" because most of the range is below). \
Symmetric rule for "below X" (B < X, or A < X with the whole range \
beneath the threshold).
     (c) VERB MATCH -- announce != complete != deploy != ratify. Evidence of an \
ANNOUNCEMENT OF INTENT to do X does not count as evidence that X has been \
COMPLETED. Match the verb in the question precisely.
There are ONLY FOUR valid output shapes (mapped to the downstream
resolver's contract). Pick exactly one:

  (A) INVALID -- the question is malformed (relative date, opinion, etc.):
        is_valid=false, is_determinable=null, has_occurred=null
  (B) UNDETERMINABLE -- valid question, but evidence is insufficient:
        is_valid=true, is_determinable=false, has_occurred=null, confidence<0.7
  (C1) YES -- the event occurred:
        is_valid=true, is_determinable=true, has_occurred=true, confidence>=0.7
  (C2) NO -- the event did NOT occur:
        is_valid=true, is_determinable=true, has_occurred=false, confidence>=0.7

VALIDITY RULES (when to choose A -- INVALID):
* Questions with relative dates ("in 6 months") are invalid.
* Questions about opinions rather than facts are invalid.

Question: "{question}"

CRITICAL: Respond with ONLY valid JSON. No markdown, no text before or after.
CONSISTENCY RULES:
- If is_valid is false  -> is_determinable AND has_occurred MUST both be null.
- If is_valid is true and is_determinable is true -> has_occurred MUST be true \
or false (never null), confidence >= 0.7.
- If is_valid is true and is_determinable is false -> has_occurred MUST be null, \
confidence < 0.7.
- confidence reflects how sure you are of your answer (0.0 = no idea, 1.0 = certain).
{{
    "is_valid": true or false,
    "is_determinable": true, false, or null,
    "has_occurred": true, false, or null,
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

There are ONLY FOUR valid output shapes (mapped to the downstream
resolver's contract). Pick exactly one:

  (A) INVALID -- the question is malformed (relative date, opinion, etc.):
        is_valid=false, is_determinable=null, has_occurred=null
  (B) UNDETERMINABLE -- valid question, but evidence is insufficient:
        is_valid=true, is_determinable=false, has_occurred=null
  (C1) YES -- the event occurred:
        is_valid=true, is_determinable=true, has_occurred=true
  (C2) NO -- the event did NOT occur:
        is_valid=true, is_determinable=true, has_occurred=false

DECISION PROCESS:
1. Review each voter's evidence and sources, not just their verdict.
2. If all voters with a definitive answer agree, follow their consensus.
3. If voters disagree:
   a. Count definitive votes (ignore indeterminate ones).
   b. Search the web to verify the specific claims in dispute.
   c. Follow the majority UNLESS your own research or the minority's sources \
show a clear factual error in the majority's reasoning.
   d. When evidence quality is similar on both sides, follow the majority.
4. If no clear majority exists, or evidence is too weak, pick (B) UNDETERMINABLE.
5. If a voter flags the question as invalid with sound reasoning, pick (A) INVALID.
6. If the majority of decided voters say UNDETERMINABLE or INVALID, do NOT \
override them with your own affirmative answer -- pick (B) or (A) respectively. \
Reserve C1/C2 for cases where decided voters actually support that verdict. \
The same rule applies when two or more decided voters with confidence >= 0.9 \
agree on YES or NO: you MUST follow them unless you can cite a specific \
factual error in their reasoning text -- a different INTERPRETATION of the \
question (e.g. of VERB MATCH, NUMERIC RANGE, NEGATION) is NOT a factual error \
and does not justify override.

SEMANTIC CHECKS (apply before producing has_occurred):
(a) NEGATION TRAP -- "still pending", "still in effect", "remain at", "not yet" \
frame the QUESTION negatively. Restate it positively in your judge_reasoning \
("Has X been completed?") and answer that, then map: "X has happened" -> \
has_occurred=false (the pending state is over) / "X has not happened" -> \
has_occurred=true (the pending state holds). Cross-check that your has_occurred \
matches the plain-English reading of your reasoning text.
(b) NUMERIC RANGE -- "at or above X" is satisfied if A >= X (lower endpoint \
reaches the threshold) OR a confirmed point value >= X. Equality at the lower \
bound COUNTS: a "2.5% to 5%" range satisfies "at or above 2.5%" because the \
lower endpoint already equals X. The rule only fails when only the upper bound \
equals X with the lower bound strictly below (e.g. "1.5% to 2.5%" does NOT \
reliably satisfy "at or above 2.5%"). Symmetric for "below X".
(c) VERB MATCH -- announce != complete != deploy != ratify. An announcement of \
intent does NOT satisfy a question asking for completion. Match the verb in the \
question literally.
CONSISTENCY RULES:
- If is_valid is false  -> is_determinable AND has_occurred MUST both be null.
- If is_valid is true and is_determinable is true -> has_occurred MUST be true or false (never null).
- If is_valid is true and is_determinable is false -> has_occurred MUST be null.

Respond in JSON only (no markdown fences, no text before or after):
{{
    "is_valid": true or false,
    "is_determinable": true, false, or null,
    "has_occurred": true, false, or null,
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

    adapter: str
    model: str
    api_key_id: str


VOTER_CONFIG: Dict[str, VoterConfig] = {
    "openai": VoterConfig(
        adapter="_adapter_openrouter",
        model=VOTER_MODEL_OPENAI,
        api_key_id="openrouter",
    ),
    "grok": VoterConfig(
        adapter="_adapter_openrouter",
        model=VOTER_MODEL_GROK,
        api_key_id="openrouter",
    ),
    "gemini": VoterConfig(
        adapter="_adapter_openrouter",
        model=VOTER_MODEL_GEMINI,
        api_key_id="openrouter",
    ),
    "claude": VoterConfig(
        adapter="_adapter_openrouter",
        model=VOTER_MODEL_CLAUDE,
        api_key_id="openrouter",
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
    is_valid = data.get("is_valid")
    has_occurred = data.get("has_occurred")
    is_determinable = data.get("is_determinable")
    # The prompt schema asks for ``confidence: 0.0 to 1.0`` (no null) but
    # LLMs sometimes emit ``null`` under Case A, or strings like ``"high"``
    # / ``"0.8 (high)"`` / ``"~0.75"`` against the schema. Defensively
    # default to 0.0 instead of crashing -- ``float("high")`` would raise
    # ValueError and silently turn a real vote into an error stub.
    raw_conf = data.get("confidence")
    try:
        confidence = float(raw_conf) if raw_conf is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0

    # Canonicalize to the 4 downstream contract cases.
    # Case A (INVALID): is_valid=False  -> is_determinable=None, has_occurred=None
    # Case B (UNDET):   is_valid=True, is_determinable=False -> has_occurred=None
    # Case C1/C2 (YES/NO): is_valid=True, is_determinable=True, has_occurred=True/False
    if is_valid is False:
        is_determinable = None
        has_occurred = None
        confidence = min(confidence, 0.5)
    else:
        if has_occurred is None:
            is_determinable = False
            confidence = min(confidence, 0.5)
        if is_determinable is False:
            has_occurred = None
            confidence = min(confidence, 0.5)

    return VoterResult(
        voter=voter,
        model=model,
        is_valid=is_valid,
        is_determinable=is_determinable,
        has_occurred=has_occurred,
        confidence=confidence,
        reasoning=data.get("reasoning", ""),
        sources=data.get("sources", []),
    )


# ---------------------------------------------------------------------------
# OpenRouter adapter
# ---------------------------------------------------------------------------


def _adapter_openrouter(
    model: str,
    prompt: str,
    api_key: str,
    max_tokens: int,
    timeout: int,
    max_attempts: int,
    retry_delay: int,
    counter_callback: Optional[Callable] = None,
) -> str:
    """Make an OpenRouter chat completion call and return the raw text.

    Records token usage + per-call surcharge to ``counter_callback``.
    Retries on 529 (overloaded) errors up to ``max_attempts`` attempts.

    :param model: OpenRouter model slug.
    :param prompt: prompt to send.
    :param api_key: OpenRouter API key.
    :param max_tokens: max output tokens.
    :param timeout: per-request timeout in seconds.
    :param max_attempts: total attempts (>=1). Only 529 errors trigger a retry.
    :param retry_delay: seconds to sleep between retry attempts.
    :param counter_callback: optional token/cost accounting callback.
    :return: raw text from the response (may be empty string).
    """
    client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    for attempt in range(max_attempts):  # pragma: no branch
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            break
        except openai.APIStatusError as err:
            if err.status_code == 529 and attempt < max_attempts - 1:
                print(
                    f"  {model} overloaded, retrying in {retry_delay}s "
                    f"(attempt {attempt + 1}/{max_attempts})..."
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
        with COUNTER_CALLBACK_LOCK:
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
    voter_id: str,
    question: str,
    api_keys: Any,
    counter_callback: Optional[Callable] = None,
) -> VoterResult:
    """Uniform entry point -- delegates to provider-specific adapter.

    :param voter_id: registry key for this voter.
    :param question: market question.
    :param api_keys: KeyChain object.
    :param counter_callback: optional token/cost accounting callback.
    :return: parsed vote result.
    """
    config = VOTER_CONFIG[voter_id]
    api_key = api_keys[config.api_key_id]
    prompt = VOTER_PROMPT.format(question=question)
    model = config.model
    adapter_fn = _ADAPTERS[config.adapter]
    raw = adapter_fn(
        model=model,
        prompt=prompt,
        api_key=api_key,
        max_tokens=VOTER_MAX_TOKENS,
        timeout=VOTER_TIMEOUT,
        max_attempts=VOTER_MAX_ATTEMPTS,
        retry_delay=VOTER_RETRY_DELAY,
        counter_callback=counter_callback,
    )
    return _parse_vote(raw, voter_id, model)


def collect_votes(
    question: str,
    voter_ids: List[str],
    api_keys: Any,
    counter_callback: Optional[Callable] = None,
) -> List[VoterResult]:
    """Fan out to all voters in parallel, collect results.

    :param question: market question.
    :param voter_ids: list of voter registry keys to use.
    :param api_keys: KeyChain object.
    :param counter_callback: optional token/cost accounting callback.
    :return: list of voter results.
    """
    results: List[VoterResult] = []
    with ThreadPoolExecutor(max_workers=len(voter_ids)) as pool:
        futures = {
            pool.submit(
                cast_vote, voter_id, question, api_keys, counter_callback
            ): voter_id
            for voter_id in voter_ids
        }
        for future in as_completed(futures):
            voter_id = futures[future]
            try:
                result = future.result(timeout=VOTER_TIMEOUT + 30)
                print(
                    f"  Voter [{voter_id}]: "
                    f"has_occurred={result.has_occurred}, "
                    f"is_determinable={result.is_determinable}, "
                    f"confidence={result.confidence}"
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"  Voter [{voter_id}] failed: {e}")
                result = VoterResult(
                    voter=voter_id,
                    model=VOTER_CONFIG[voter_id].model,
                    error=str(e),
                )

            results.append(result)

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
    successful = _successful_votes(votes)
    formatted_votes = ""
    for i, v in enumerate(successful, 1):
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
    try:
        raw = _adapter_openrouter(
            model=JUDGE_MODEL_CLAUDE,
            prompt=prompt,
            api_key=api_key,
            max_tokens=JUDGE_MAX_TOKENS,
            timeout=JUDGE_TIMEOUT,
            max_attempts=JUDGE_MAX_ATTEMPTS,
            retry_delay=JUDGE_RETRY_DELAY,
            counter_callback=counter_callback,
        )
    except Exception as exc:  # pylint: disable=broad-except
        # Adapter-level failure (402 credit-exhausted, network error,
        # timeout, etc.). Without this catch the exception bubbles up
        # to ``with_key_rotation``'s broad-except, which returns a raw
        # ``str(e)`` in tuple[0] -- violating the JSON-shape contract
        # every other failure path obeys. Emit a proper discriminator
        # here so downstream parsers + operator logs see a consistent
        # ``error="judge_api_error"`` payload alongside
        # ``judge_unparseable`` and ``all_voters_failed``.
        return {
            "is_valid": None,
            "is_determinable": None,
            "has_occurred": None,
            "error": "judge_api_error",
            "judge_reasoning": f"Judge adapter raised: {exc!r}"[:300],
        }
    data = _extract_json(raw)
    if data is None:
        # Judge LLM produced unparseable output. Mark verdict as unknown
        # (``is_valid=None``) rather than ``False`` so downstream consumers
        # don't mistake a judge-side LLM failure for a genuine "the
        # question is invalid" verdict and submit ANSWER_INVALID
        # (0xff...ff) on Realitio. See the all-voters-failed branch in
        # ``run()`` for the same rationale.
        return {
            "is_valid": None,
            "is_determinable": None,
            "has_occurred": None,
            "error": "judge_unparseable",
            "judge_reasoning": f"Unparseable judge response: {raw[:200]}",
        }
    return data


# ---------------------------------------------------------------------------
# Consensus helpers
# ---------------------------------------------------------------------------


def _successful_votes(votes: List[VoterResult]) -> List[VoterResult]:
    """Filter to votes whose voter did not error out.

    :param votes: all voter results (including error stubs).
    :return: voter results with ``error is None``.
    """
    return [v for v in votes if v.error is None]


def _decided_votes(votes: List[VoterResult]) -> List[VoterResult]:
    """Voters that reached YES / NO / INVALID (excludes undet + errored)."""
    return [
        v
        for v in votes
        if v.error is None and (v.is_valid is False or v.is_determinable is True)
    ]


# The three valid actionable verdict labels. A decided vote always maps
# to exactly one of these; the (is_valid, is_determinable, has_occurred)
# tuple in the public output is derived from the label.
_LABEL_OUTPUT: Dict[str, Tuple[Optional[bool], Optional[bool], Optional[bool]]] = {
    "yes": (True, True, True),  # Case C1
    "no": (True, True, False),  # Case C2
    "invalid": (False, None, None),  # Case A
}


def _verdict_label(v: VoterResult) -> Optional[str]:
    """``"yes"`` / ``"no"`` / ``"invalid"`` for decided votes; else ``None``."""
    if v.error is not None:
        return None
    if v.is_valid is False:
        return "invalid"
    if v.is_determinable is True and v.has_occurred is True:
        return "yes"
    if v.is_determinable is True and v.has_occurred is False:
        return "no"
    return None


def _label_counts(votes: List[VoterResult]) -> Counter:
    """Tally of decided votes per verdict label (undet/errored skipped)."""
    return Counter(
        label for label in (_verdict_label(v) for v in votes) if label is not None
    )


def _has_consensus(votes: List[VoterResult]) -> bool:
    """True iff strictly >50% of TOTAL voters back one actionable verdict."""
    counts = _label_counts(votes)
    if not counts:
        return False
    return counts.most_common(1)[0][1] > len(votes) / 2


def _build_consensus_result(votes: List[VoterResult]) -> dict:
    """Build result from consensus votes (skip judge).

    Caller MUST have verified ``_has_consensus`` returns True. The
    winning label's canonical shape (see ``_LABEL_OUTPUT``) is emitted
    directly -- no separate code paths for INVALID vs YES/NO.

    :param votes: all voter results.
    :return: top-level result dict matching the judge-path output schema.
    :raises ValueError: if no decided votes exist (caller skipped the
        ``_has_consensus`` check).
    """
    counts = _label_counts(votes)
    if not counts:
        raise ValueError(
            "_build_consensus_result called without consensus -- "
            "no decided votes. Caller must verify _has_consensus() first."
        )
    winner, _ = counts.most_common(1)[0]
    is_valid, is_determinable, has_occurred = _LABEL_OUTPUT[winner]
    reason = (
        "Voter majority consensus on INVALID -- judge skipped."
        if winner == "invalid"
        else "Voter majority consensus -- judge skipped."
    )
    return {
        "is_valid": is_valid,
        "is_determinable": is_determinable,
        "has_occurred": has_occurred,
        "judge_reasoning": reason,
        "votes": [asdict(v) for v in votes],
        "agreement_ratio": _compute_agreement(votes),
        "n_voters": len(votes),
        "n_successful": len(_successful_votes(votes)),
        "n_decided": len(_decided_votes(votes)),
    }


def _compute_agreement(votes: List[VoterResult]) -> float:
    """Most-voted label's share over decided votes (0.0 if none)."""
    counts = _label_counts(votes)
    n_decided = sum(counts.values())
    if n_decided == 0:
        return 0.0
    return counts.most_common(1)[0][1] / n_decided


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
    voter_models = [VOTER_CONFIG[v].model for v in voters]

    # Cost calculation mode
    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback was given."
            )
        max_cost = counter_callback(
            max_cost=True,
            models_calls=tuple(voter_models) + (JUDGE_MODEL_CLAUDE,),
        )
        return max_cost

    # 1. Fan out to voters (parallel)
    print(f"Collecting votes from {voters}...")
    votes = collect_votes(prompt, voters, api_keys, counter_callback)
    successful = _successful_votes(votes)

    # 2. Early exit: no successful votes.
    #
    # IMPORTANT: ``is_valid`` MUST be ``None`` here (not ``False``). Emitting
    # ``is_valid=False`` is reserved for the case where the jury actually
    # determines the question is invalid. Conflating "all voter APIs errored"
    # with "the question is invalid" causes downstream consumers (e.g.
    # market-resolver's ``parse_mech_response``) to submit
    # ``ANSWER_INVALID`` (0xff...ff) to Realitio for what is really an API
    # outage -- burning bonds on a meaningless verdict.
    #
    # ``is_valid=None`` is the natural "we couldn't determine validity"
    # marker; downstream strict parsers already reject it as garbage and
    # retry with cooldown.
    if not successful:
        result: Dict[str, Any] = {
            "is_valid": None,
            "is_determinable": None,
            "has_occurred": None,
            "votes": [asdict(v) for v in votes],
            "judge_reasoning": "All voters failed (API errors / empty responses).",
            "error": "all_voters_failed",
            "agreement_ratio": 0.0,
            "n_voters": len(voters),
            "n_successful": 0,
            "n_decided": 0,
        }
        return (
            json.dumps(result),
            result["judge_reasoning"],
            None,
            counter_callback,
            None,
        )

    used_params: Dict[str, Any] = {
        "model": JUDGE_MODEL_CLAUDE,
        "voter_models": voter_models,
        "n_voters": len(voters),
    }

    # 3. Majority consensus early exit (cost saving -- skip judge)
    if _has_consensus(votes):
        print("  Voter majority consensus -- skipping judge.")
        result = _build_consensus_result(votes)
        # Judge was not called -- record the voters that actually contributed
        # to the verdict instead of pinning a single misleading model id.
        used_params["model"] = None
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

    # 5. Canonicalize judge verdict to one of the four contract cases.
    #
    # Contract (parse_mech_response docstring, market-resolver
    # ``behaviours/base.py``):
    #   Case A:  (False, None, None)   -> INVALID
    #   Case B:  (True,  False, None)  -> undeterminable
    #   Case C1: (True,  True,  True)  -> YES
    #   Case C2: (True,  True,  False) -> NO
    #
    # Anything outside these four shapes is consolidated to the error
    # discriminator ``(None, None, None) + error="malformed_verdict"``,
    # which downstream parsers reject as garbage and retry.
    iv = verdict.get("is_valid")
    id_ = verdict.get("is_determinable")
    ho = verdict.get("has_occurred")
    judge_reasoning = verdict.get("judge_reasoning", "")

    canon_is_valid: Optional[bool]
    canon_is_det: Optional[bool]
    canon_has_occ: Optional[bool]
    canon_error: Optional[str] = None
    if iv is False:
        # Case A
        canon_is_valid, canon_is_det, canon_has_occ = False, None, None
    elif iv is True and id_ is False:
        # Case B
        canon_is_valid, canon_is_det, canon_has_occ = True, False, None
    elif iv is True and id_ is True and ho is True:
        # Case C1
        canon_is_valid, canon_is_det, canon_has_occ = True, True, True
    elif iv is True and id_ is True and ho is False:
        # Case C2
        canon_is_valid, canon_is_det, canon_has_occ = True, True, False
    else:
        # Judge returned a shape outside the contract (e.g. is_valid=None
        # after partial parse, is_determinable=True with has_occurred=None,
        # etc.). Route through the error discriminator so downstream
        # consumers retry instead of acting on an ambiguous verdict.
        # Preserve the upstream error (e.g. ``judge_unparseable``) when
        # present so the operator can tell parser failures apart from
        # off-contract verdicts; default to ``malformed_verdict`` otherwise.
        canon_is_valid = None
        canon_is_det = None
        canon_has_occ = None
        canon_error = verdict.get("error") or "malformed_verdict"

    result = {
        "is_valid": canon_is_valid,
        "is_determinable": canon_is_det,
        "has_occurred": canon_has_occ,
        "votes": [asdict(v) for v in votes],
        "judge_reasoning": judge_reasoning,
        "agreement_ratio": _compute_agreement(votes),
        "n_voters": len(voters),
        "n_successful": len(successful),
        "n_decided": len(_decided_votes(votes)),
    }
    if canon_error is not None:
        result["error"] = canon_error

    return json.dumps(result), judge_reasoning, None, counter_callback, used_params
