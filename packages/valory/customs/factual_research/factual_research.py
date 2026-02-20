# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024-2026 Valory AG
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
"""Factual Research tool — edge generation via narrow factual inquiry.

Philosophy
----------
Mechs produce *edges*, not predictions.  This tool answers narrow, factual
questions by gathering verifiable evidence from the open web and returning
cited, state-of-the-world facts.  A downstream (non-mech) layer converts
facts into a probability — the mech itself never sees prices, odds, or
"likelihood".

Pipeline (3 structured-output LLM calls):
    1. **Reframe** — decompose the incoming prompt into narrow, verifiable,
       factual sub-questions with date anchors.
    2. **Gather** — web-search each sub-question via Serper; blocked domains
       (Polymarket, Twitter/X, prediction sites) are excluded.
    3. **Synthesise** — produce a factual briefing that answers each
       sub-question with citations.
    4. **Estimate** — a *separate* LLM call converts the factual briefing
       into the standard ``{p_yes, p_no, confidence, info_utility}`` output
       required by the mech protocol.

All LLM calls use OpenAI **Structured Outputs**
(``client.beta.chat.completions.parse``) with Pydantic models, so responses
are guaranteed to conform to the expected schema — no fragile JSON-in-prompt
parsing needed.

Hard guardrails (enforced in every system prompt for steps 1-3):
    • Never output a probability in the research phase.
    • Never reference prices, odds, or betting lines.
    • Never search Polymarket, Twitter/X, or prediction sites.
    • Only report verifiable, cited facts.
"""
import functools
import json
import re
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai
import requests
from markdownify import markdownify as md
from openai import OpenAI
from pydantic import BaseModel, Field
from readability import Document as ReadabilityDocument
from tiktoken import encoding_for_model


# ---------------------------------------------------------------------------
# Pydantic schemas for OpenAI Structured Outputs
# ---------------------------------------------------------------------------


class SubQuestions(BaseModel):
    """Output of the reframing step — narrow factual sub-questions."""

    sub_questions: List[str] = Field(
        ...,
        description=(
            "3-6 narrow, verifiable, factual sub-questions derived from the "
            "original prompt.  Cover current status, competitors, historical "
            "base rates, expert signals, obstacles, and timelines where relevant.  "
            "No predictions or probabilities."
        ),
    )


class SourceReference(BaseModel):
    """A single cited source."""

    title: str = Field(..., description="Title of the source article or page.")
    url: str = Field(..., description="URL of the source.")


class SubAnswer(BaseModel):
    """A factual answer to one sub-question, with sources."""

    question: str = Field(..., description="The factual sub-question that was asked.")
    answer: str = Field(
        ...,
        description=(
            "The factual answer supported by evidence.  "
            "Say 'Insufficient evidence' if the evidence is lacking."
        ),
    )
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Up to 2 sources that support the answer.",
        max_length=2,
    )


class FactualBriefing(BaseModel):
    """Output of the synthesis step — cited factual answers + summary."""

    sub_answers: List[SubAnswer] = Field(
        ...,
        description="Concise factual answers for each sub-question (1-3 sentences each).",
    )
    summary: str = Field(
        ...,
        description=(
            "A 3-6 sentence factual summary synthesising the key "
            "state-of-the-world facts relevant to the original question.  "
            "No predictions or probabilities."
        ),
    )
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Most relevant sources relied upon (max 6).",
        max_length=6,
    )
    info_utility: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How useful the gathered evidence was (0 = useless, 1 = highly useful).",
    )


class PredictionResult(BaseModel):
    """Standard mech prediction output — p_yes, p_no, confidence, info_utility + reasoning."""

    reasoning: str = Field(
        ...,
        description=(
            "Step-by-step reasoning that explains the probability estimate.  "
            "Reference specific facts from the briefing.  Mention base rates, "
            "key risk factors, and competing signals.  2-4 paragraphs."
        ),
    )
    p_yes: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that the event occurs.",
    )
    p_no: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that the event does NOT occur.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the prediction (0 = lowest, 1 = highest).",
    )
    info_utility: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Utility of the information gathered to inform the prediction.",
    )


# ---------------------------------------------------------------------------
# Mech-protocol types
# ---------------------------------------------------------------------------

client: Optional[OpenAI] = None
MechResponseWithKeys = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]
MaxCostResponse = float

# 3 LLM calls: reframe → synthesise → estimate
N_MODEL_CALLS = 3
DEFAULT_DELIVERY_RATE = 100


# ---------------------------------------------------------------------------
# Key-rotation decorator (same pattern as other mech tools)
# ---------------------------------------------------------------------------


def with_key_rotation(func: Callable) -> Callable:
    """Decorator that retries a function with API key rotation on failure."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> MechResponseWithKeys:
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
            except Exception as e:
                return str(e), "", None, None, api_keys

        return execute()

    return wrapper


# ---------------------------------------------------------------------------
# OpenAI client helpers
# ---------------------------------------------------------------------------


class OpenAIClientManager:
    """Context manager that creates / closes the module-level OpenAI client."""

    def __init__(self, api_key: str):
        """Initializes with API key."""
        self.api_key = api_key

    def __enter__(self) -> OpenAI:
        """Initializes and returns the OpenAI client."""
        global client
        if client is None:
            client = OpenAI(api_key=self.api_key)
        return client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Closes the OpenAI client."""
        global client
        if client is not None:
            client.close()
            client = None


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in *text* for the given *model*."""
    try:
        enc = encoding_for_model(model)
    except KeyError:
        from tiktoken import get_encoding

        enc = get_encoding("o200k_base")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 2000,
    "limit_max_tokens": 8192,
    "temperature": 0,
}
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"
ALLOWED_TOOLS = ["factual_research"]
ALLOWED_MODELS = [DEFAULT_OPENAI_MODEL]
COMPLETION_RETRIES = 3
COMPLETION_DELAY = 2

# Domains that must never appear in search queries or results.
BLOCKED_DOMAINS = [
    "polymarket.com",
    "twitter.com",
    "x.com",
    "predictit.org",
    "metaculus.com",
    "manifold.markets",
    "kalshi.com",
    "betfair.com",
    "smarkets.com",
    "oddschecker.com",
    "bovada.lv",
]
BLOCKED_DOMAINS_FILTER = " ".join(f"-site:{d}" for d in BLOCKED_DOMAINS)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

REFRAME_SYSTEM = (
    "You are a factual-research assistant. You decompose questions into "
    "narrow, verifiable, factual sub-questions. You NEVER predict, estimate "
    "probabilities, or reference prediction markets, odds, or prices."
)

REFRAME_USER = """Decompose the following question into narrow, verifiable, factual sub-questions.

RULES
1. Strip any "will … happen?" phrasing. Replace it with objective status checks:
   - "Has X completed milestone Y as of today?"
   - "What are the remaining steps / failure modes for X?"
   - "List objective milestones remaining before date D."
2. Each sub-question must be answerable with publicly verifiable facts.
3. Add a date anchor ("as of {today}") wherever useful.
4. Cover DIVERSE angles so the downstream estimator sees a full picture.
   Consider these categories (use whichever are relevant):
   A. Current status — what has already happened or been announced?
   B. Competing alternatives / rival actors — who else could win or block this?
   C. Historical base rates — how often do events like this succeed/fail?
   D. Expert or official signals — what have credible authorities said?
   E. Remaining obstacles or risk factors — what could still derail or enable this?
   F. Timeline & deadlines — what key dates constrain the outcome?
5. Output between 3 and 6 sub-questions. Prefer more when the topic is complex.

INPUT QUESTION:
\"\"\"{question}\"\"\"

Today's date: {today}
"""

SYNTHESIS_SYSTEM = (
    "You are a factual-research analyst. You answer questions using ONLY "
    "verifiable, cited evidence. You NEVER output probabilities, predictions, "
    "or reference prediction markets, odds, or prices."
)

SYNTHESIS_USER = """Produce a factual briefing that answers each sub-question below using the gathered evidence.

ORIGINAL QUESTION:
\"\"\"{question}\"\"\"

SUB-QUESTIONS:
{sub_questions}

Today's date: {today}

EVIDENCE GATHERED:
{evidence}

INSTRUCTIONS:
1. For each sub-question, write a CONCISE factual answer (1-3 sentences max).
   - If evidence is insufficient, say "Insufficient evidence".
   - Explicitly note when *expected evidence is missing* (e.g. no official filing, no confirmation, no action).
   - Cite at most 2 sources per sub-answer.
2. Write a Factual Summary (2-4 sentences) synthesising the key state-of-the-world facts.
   - Report what is verifiably true *and what has not yet happened*.
   - Do NOT predict outcomes or imply likelihood.
3. List only the most relevant sources (max 6 total).
4. Rate how useful the evidence was (info_utility, 0-1).
   - High info_utility means the evidence is informative, not that it supports success.
5. Be concise throughout — every token counts.
"""

ESTIMATE_SYSTEM = """You are an expert probability estimator. You receive a factual briefing and must convert it into a calibrated probability estimate.
Your performance is evaluated by the Brier score.

Calibration rules:
• Absence of expected evidence is a NO signal, not neutral.
• High info_utility does NOT imply high probability.
• Multi-step processes involving institutions, politics, or coordination rarely justify extreme probabilities.

Tail discipline:
• Probabilities above 90% require evidence that major failure modes are effectively eliminated.
• Probabilities above 80% require strong historical precedents under similar conditions.
• When confidence is low, extreme probabilities are rarely justified.
"""

ESTIMATE_USER = """Relying exclusively on the factual briefing provided below, assess the probability that the event in the original question will occur.

ORIGINAL QUESTION:
\"\"\"{question}\"\"\"

Today's date: {today}

FACTUAL BRIEFING:
{briefing}

INSTRUCTIONS:

STEP 1 — Base rate anchor (MANDATORY)
a. Identify the event category (e.g. regulatory approval, election outcome, product launch, treaty, court decision).
b. State an explicit base-rate probability for events of this type.
   - Use historical data if available.
   - If not, use a conservative implied base rate and justify it.
c. This base rate is your starting point.

STEP 2 — Evidence evaluation
a. List the key YES signals (facts that materially increase probability).
b. List the key NO signals, including:
   - Explicit negative evidence
   - Missing evidence that would normally be expected by this stage
c. Weigh signal strength relative to the base rate.
   - Weak, procedural, or intention-based evidence should cause only small updates.

STEP 3 — Synthesis
a. Update from the base rate to a final probability.
b. If evidence is mixed or thin, remain close to the base rate.
c. Do NOT default away from uncertainty without justification.

STEP 4 — Output constraints
• p_yes + p_no must equal 1.
• If confidence < 0.5, probabilities above 70% or below 30% require strong justification.
• If confidence < 0.3, probabilities should remain within [0.2, 0.8].
• It is valid for info_utility to be high while p_yes is low.

OUTPUT:
1. Provide detailed reasoning (2-4 paragraphs).
2. Then output p_yes, p_no, confidence, and info_utility.
"""


# ---------------------------------------------------------------------------
# Structured-output completion helper
# ---------------------------------------------------------------------------


def _parse_completion(
    model: str,
    messages: List[Dict[str, str]],
    response_format: Any,
    temperature: float = 0,
    max_tokens: int = 2000,
    retries: int = COMPLETION_RETRIES,
    delay: int = COMPLETION_DELAY,
    counter_callback: Optional[Callable] = None,
) -> Tuple[Any, Optional[Callable]]:
    """Call OpenAI with Structured Outputs and parse into a Pydantic model.

    Uses ``client.beta.chat.completions.parse()`` which guarantees the
    response conforms to the supplied Pydantic schema — no fragile
    JSON-in-prompt parsing or regex extraction needed.

    :param model: The OpenAI model name to use.
    :type model: str
    :param messages: List of message dicts with 'role' and 'content' keys.
    :type messages: List[Dict[str, str]]
    :param response_format: Pydantic model class for structured output parsing.
    :type response_format: Any
    :param temperature: Sampling temperature (0 = deterministic).
    :type temperature: float
    :param max_tokens: Maximum tokens to generate in the completion.
    :type max_tokens: int
    :param retries: Number of retry attempts on failure.
    :type retries: int
    :param delay: Delay in seconds between retries.
    :type delay: int
    :param counter_callback: Optional callback for tracking token usage.
    :type counter_callback: Optional[Callable]

    :return: Tuple of (parsed_model_instance, counter_callback).
    :rtype: Tuple[Any, Optional[Callable]]
    """
    if not client:
        raise RuntimeError("OpenAI client not initialised")

    attempt = 0
    while attempt < retries:
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=150,
            )

            parsed = response.choices[0].message.parsed

            if parsed is None:
                refusal = response.choices[0].message.refusal
                raise ValueError(
                    f"Model refused or returned unparseable output: {refusal}"
                )

            if counter_callback is not None:
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=model,
                    token_counter=count_tokens,
                )

            return parsed, counter_callback
        except (openai.LengthFinishReasonError, ValueError) as e:
            print(f"[factual_research] Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
            attempt += 1
        except Exception as e:
            print(f"[factual_research] Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
            attempt += 1

    raise RuntimeError("Failed to get structured LLM completion after retries")


# ---------------------------------------------------------------------------
# Web-search helpers (Serper API)
# ---------------------------------------------------------------------------


def _search_serper(
    query: str, api_key: str, num_results: int = 5
) -> List[Dict[str, str]]:
    """Run a Google search via Serper; results from blocked domains are dropped."""
    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    results: List[Dict[str, str]] = []
    for item in data.get("organic", []):
        link = item.get("link", "")
        if any(blocked in link for blocked in BLOCKED_DOMAINS):
            continue
        results.append(
            {
                "title": item.get("title", ""),
                "link": link,
                "snippet": item.get("snippet", ""),
            }
        )
        if len(results) >= num_results:
            break

    # Also grab "People Also Ask" snippets if present
    for paa in data.get("peopleAlsoAsk", []):
        link = paa.get("link", "")
        if any(blocked in link for blocked in BLOCKED_DOMAINS):
            continue
        results.append(
            {
                "title": paa.get("question", ""),
                "link": link,
                "snippet": paa.get("snippet", ""),
            }
        )

    return results


# Maximum words to keep per scraped page
_MAX_PAGE_WORDS = 400
# Image / junk patterns to strip from HTML before extraction
_IMG_TAG_PATTERN = re.compile(r"<img[^>]*>", re.IGNORECASE)
_SCRIPT_STYLE_PATTERN = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)


def _fetch_page_content(
    url: str, max_words: int = _MAX_PAGE_WORDS, timeout: int = 10
) -> Optional[str]:
    """Fetch a URL and extract its main text content.

    Uses ``readability-lxml`` to isolate the main article content, then
    ``markdownify`` to convert it to clean Markdown text.

    :param url: The URL to fetch.
    :type url: str
    :param max_words: Maximum number of words to keep.
    :type max_words: int
    :param timeout: Request timeout in seconds.
    :type timeout: int

    :return: Extracted text content, or None on failure.
    :rtype: Optional[str]
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MechBot/1.0)"},
        )
        if resp.status_code != 200:
            return None
        # Skip non-HTML responses (PDFs, images, etc.)
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return None

        html = resp.text
        # Strip scripts/styles/images before readability
        html = _SCRIPT_STYLE_PATTERN.sub("", html)
        html = _IMG_TAG_PATTERN.sub("", html)

        # Extract main content with readability
        article_html = ReadabilityDocument(html).summary()
        # Convert to clean markdown text
        text = md(article_html, heading_style="ATX", strip=["img", "figure"])
        if not text or not text.strip():
            return None

        # Truncate to max_words
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + " […]"

        return text.strip()
    except Exception as e:
        print(f"[factual_research] Failed to fetch {url}: {e}")
        return None


def _format_evidence(all_results: List[Dict[str, str]]) -> str:
    """Turn search-result dicts into a numbered evidence block."""
    if not all_results:
        return "(no evidence gathered)"
    lines: List[str] = []
    for idx, item in enumerate(all_results, 1):
        content = item.get("content", "")
        if content:
            lines.append(
                f"{idx}. [{item['title']}]({item['link']})\n"
                f"   Snippet: {item['snippet']}\n"
                f"   Content: {content}"
            )
        else:
            lines.append(
                f"{idx}. [{item['title']}]({item['link']})\n" f"   {item['snippet']}"
            )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt extraction helper
# ---------------------------------------------------------------------------


def _extract_question(prompt: str) -> str:
    """Extract the core question from the mech prompt envelope."""
    pattern = r'question\s+"(.+?)"\s+and\s+the\s+`yes`'
    try:
        return re.findall(pattern, prompt, re.DOTALL)[0]
    except (IndexError, TypeError):
        return prompt.strip()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """Run the Factual Research tool.

    Pipeline
    --------
    1. Reframe the incoming prompt into narrow factual sub-questions
       (structured output → ``SubQuestions``).
    2. Search the web for each sub-question via Serper (no prediction sites).
    3. Synthesise a cited factual briefing from the evidence
       (structured output → ``FactualBriefing``).
    4. Convert the factual briefing into the standard mech prediction format
       (structured output → ``PredictionResult``).

    :param kwargs: Keyword arguments including 'tool', 'model', 'prompt',
        'api_keys', 'delivery_rate', 'counter_callback', 'temperature'.
    :type kwargs: Any

    :return: Either max_cost (float) if delivery_rate==0, or MechResponse tuple
        (result_json, full_prompt, None, counter_callback).
    :rtype: Union[MaxCostResponse, MechResponse]
    """
    tool = kwargs["tool"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    model = kwargs.get("model")
    if model is None:
        raise ValueError("Model not supplied.")

    delivery_rate = int(kwargs.get("delivery_rate", DEFAULT_DELIVERY_RATE))
    counter_callback: Optional[Callable[..., Any]] = kwargs.get(
        "counter_callback", None
    )

    if delivery_rate == 0:
        if not counter_callback:
            raise ValueError(
                "A delivery rate of `0` was passed, but no counter callback "
                "was given to calculate the max cost with."
            )
        max_cost = counter_callback(
            max_cost=True,
            models_calls=(model,) * N_MODEL_CALLS,
        )
        return max_cost

    # -- API keys --
    openai_api_key = kwargs["api_keys"]["openai"]
    serper_api_key = kwargs["api_keys"]["serperapi"]

    with OpenAIClientManager(openai_api_key):
        temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
        prompt = kwargs["prompt"]

        today = date.today().strftime("%Y-%m-%d")
        question = _extract_question(prompt)

        # ---------------------------------------------------------------
        # Step 1 — Reframe into factual sub-questions  (Structured Output)
        # ---------------------------------------------------------------
        print("[factual_research] Step 1: Reframing prompt into factual sub-questions…")
        reframe_messages = [
            {"role": "system", "content": REFRAME_SYSTEM},
            {
                "role": "user",
                "content": REFRAME_USER.format(question=question, today=today),
            },
        ]

        sub_q_result: SubQuestions
        sub_q_result, counter_callback = _parse_completion(
            model=model,
            messages=reframe_messages,
            response_format=SubQuestions,
            temperature=0,
            max_tokens=600,
            counter_callback=counter_callback,
        )

        sub_questions = sub_q_result.sub_questions
        if not sub_questions:
            sub_questions = [question]

        print(f"[factual_research] Sub-questions: {sub_questions}")

        # ---------------------------------------------------------------
        # Step 2 — Search the web for each sub-question
        # ---------------------------------------------------------------
        print("[factual_research] Step 2: Searching for evidence…")
        all_evidence: List[Dict[str, str]] = []
        seen_links: set = set()

        for sq in sub_questions:
            try:
                results = _search_serper(sq, serper_api_key, num_results=3)
                for r in results:
                    if r["link"] not in seen_links:
                        seen_links.add(r["link"])
                        all_evidence.append(r)
            except Exception as e:
                print(f"[factual_research] Search failed for '{sq}': {e}")

        # Scrape actual page content for the top results
        MAX_PAGES_TO_SCRAPE = 6
        scraped = 0
        for item in all_evidence:
            if scraped >= MAX_PAGES_TO_SCRAPE:
                break
            content = _fetch_page_content(item["link"])
            if content:
                item["content"] = content
                scraped += 1
        print(f"[factual_research] Scraped {scraped}/{len(all_evidence)} pages.")

        # Cap evidence so we don't blow the synthesis context window
        MAX_EVIDENCE_ITEMS = 20
        if len(all_evidence) > MAX_EVIDENCE_ITEMS:
            all_evidence = all_evidence[:MAX_EVIDENCE_ITEMS]

        evidence_text = _format_evidence(all_evidence)

        # Hard-trim evidence text to stay within a reasonable token budget
        MAX_EVIDENCE_TOKENS = 3000
        ev_tokens = count_tokens(evidence_text, model)
        if ev_tokens > MAX_EVIDENCE_TOKENS:
            # Truncate to roughly MAX_EVIDENCE_TOKENS by character ratio
            ratio = MAX_EVIDENCE_TOKENS / ev_tokens
            truncated_len = int(len(evidence_text) * ratio)
            evidence_text = (
                evidence_text[:truncated_len] + "\n\n[… evidence truncated …]"
            )

        print(
            f"[factual_research] Gathered {len(all_evidence)} evidence items ({count_tokens(evidence_text, model)} tokens)."
        )

        # ---------------------------------------------------------------
        # Step 3 — Synthesise factual briefing  (Structured Output)
        # ---------------------------------------------------------------
        print("[factual_research] Step 3: Synthesising factual briefing…")
        numbered_sqs = "\n".join(
            f"  {i}. {sq}" for i, sq in enumerate(sub_questions, 1)
        )
        synthesis_messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": SYNTHESIS_USER.format(
                    question=question,
                    sub_questions=numbered_sqs,
                    today=today,
                    evidence=evidence_text,
                ),
            },
        ]

        briefing: FactualBriefing
        briefing, counter_callback = _parse_completion(
            model=model,
            messages=synthesis_messages,
            response_format=FactualBriefing,
            temperature=temperature,
            max_tokens=3000,
            counter_callback=counter_callback,
        )

        briefing_text = briefing.model_dump_json(indent=2)
        print(f"[factual_research] Briefing summary: {briefing.summary[:200]}…")
        print(f"[factual_research] === FULL BRIEFING ===\n{briefing_text}")
        print("[factual_research] === END BRIEFING ===")

        # ---------------------------------------------------------------
        # Step 4 — Convert briefing → prediction  (Structured Output)
        # ---------------------------------------------------------------
        print(
            "[factual_research] Step 4: Estimating probability from factual briefing…"
        )

        estimate_messages = [
            {"role": "system", "content": ESTIMATE_SYSTEM},
            {
                "role": "user",
                "content": ESTIMATE_USER.format(
                    question=question, today=today, briefing=briefing_text
                ),
            },
        ]

        prediction: PredictionResult
        prediction, counter_callback = _parse_completion(
            model=model,
            messages=estimate_messages,
            response_format=PredictionResult,
            temperature=temperature,
            max_tokens=1500,
            counter_callback=counter_callback,
        )

        print(f"[factual_research] === REASONING ===\n{prediction.reasoning}")
        print("[factual_research] === END REASONING ===")
        print(
            f"[factual_research] Result: p_yes={prediction.p_yes}, "
            f"p_no={prediction.p_no}, confidence={prediction.confidence}, "
            f"info_utility={prediction.info_utility}"
        )

        # Final JSON string in the standard mech format
        result = json.dumps(
            {
                "p_yes": prediction.p_yes,
                "p_no": prediction.p_no,
                "confidence": prediction.confidence,
                "info_utility": prediction.info_utility,
            }
        )

        # Full prompt trail for audit / transparency
        full_prompt_used = (
            f"--- REFRAME ---\n{json.dumps(reframe_messages, indent=2)}\n\n"
            f"--- SUB-QUESTIONS ---\n{sub_q_result.model_dump_json(indent=2)}\n\n"
            f"--- EVIDENCE ---\n{evidence_text}\n\n"
            f"--- SYNTHESIS ---\n{json.dumps(synthesis_messages, indent=2)}\n\n"
            f"--- BRIEFING ---\n{briefing_text}\n\n"
            f"--- ESTIMATE ---\n{json.dumps(estimate_messages, indent=2)}\n\n"
            f"--- REASONING ---\n{prediction.reasoning}"
        )

        return result, full_prompt_used, None, counter_callback
