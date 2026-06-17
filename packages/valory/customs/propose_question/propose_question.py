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
"""Mech tool that proposes prediction-market questions from recent news."""

import functools
import json
import random
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai
import requests
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from openai import OpenAI
from pydantic import BaseModel

NEWSAPI_TOP_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"
NEWSAPI_DEFAULT_NEWS_SOURCES = [
    "bbc-news",
    "bbc-sport",
    "abc-news",
    "cnn",
    "reuters",
    "usa-today",
    "breitbart-news",
    "the-verge",
    "techradar",
    "associated-press",
    "bloomberg",
    "business-insider",
    "ars-technica",
    "national-geographic",
    "new-scientist",
]

OMEN_SUBGRAPH_URL = "https://gateway-arbitrum.network.thegraph.com/api/{subgraph_api_key}/subgraphs/id/9fUVQpFwzpdWS9bq5WkAnmKbNNcoBwatMR4yZq81pbbz"
HTTP_OK = 200
MAX_ARTICLES = 60
MAX_LATEST_QUESTIONS = 100
# NewsAPI top-headlines endpoint caps a single request at 100 results; we
# request the max and downsample to MAX_ARTICLES later.
NEWSAPI_MAX_PAGE_SIZE = 100
# Max chars of scraped article body passed to the LLM. Used in BOTH the
# question-generation prompt and the self-review prompt so the reviewer
# scores against the same text the generator saw.
ARTICLE_TEXT_MAX_CHARS = 6000

FPMM_CREATORS = [
    "0x89c5cc945dd550bcffb72fe42bff002429f46fec",
    "0xffc8029154ecd55abed15bd428ba596e7d23f557",
]
FPMMS_QUERY = """
    query fpmms_query($creator_in: [Bytes!], $first: Int) {
      fixedProductMarketMakers(
        where: {creator_in: $creator_in}
        orderBy: creationTimestamp
        orderDirection: desc
        first: $first
      ) {
        question {
          title
        }
      }
    }
    """

DEFAULT_TOPICS = [
    "business",
    "cryptocurrency",
    "politics",
    "science",
    "technology",
    "trending",
    "social",
    "health",
    "sustainability",
    "internet",
    "food",
    "pets",
    "animals",
    "curiosities",
    "economy",
    "arts",
    "entertainment",
    "weather",
    "sports",
    "finance",
    "international",
]

SELECT_STORY_PROMPT = """You are provided a numbered list of recent news article
    snippets under ARTICLES. You are provided a list of existing questions under
    EXISTING_QUESTIONS. Your task is to choose one article or story which is suitable
    to create questions for prediction markets. The chosen article should be that the
    questions created are of public interest.

    HARD REJECTION: do NOT choose an article if EXISTING_QUESTIONS already
    contains a question about the SAME UNDERLYING TOPIC (same metric, same
    institution, same event, same named entity), even when the wording or
    numeric threshold is slightly different. Treat these as duplicates:
    - "2-year mortgage rate below 5.80%" and "2-year mortgage rate below 5.85%"
    - "S&P 500 above 7,100" and "S&P 500 above 7,200"
    - "Bank of England rate above X%" and "Bank of England rate at or above Y%"
    - any two questions naming the same institution reporting on the same metric
    When in doubt, prefer a different article with a NEW topic not present in
    EXISTING_QUESTIONS.

    PREFER articles that support MEASUREMENT or CONTINUATION questions, with a
    particular bias toward continuation-friendly topics. Good signals for
    continuation-friendly articles:
    - Economic indicators (interest rates, inflation targets, unemployment,
      stock indices, commodity prices, exchange rates).
    - Political incumbencies (leaders, officials, judges currently in position).
    - Existing moratoriums, sanctions, bans, regulations, policies.
    - Institutional positions (ratings, league standings, memberships,
      subscriptions, listings).
    - Ongoing conflicts, negotiations, strikes, or standoffs.
    These articles produce questions about status-quo persistence, which
    balance the pipeline's natural No-bias from short-deadline announcements.

    Also acceptable:
    - Articles mentioning a measurable quantity that can be checked on a
      future date (measurement framing).

    AVOID articles whose only angle is "will authority X announce Y?" unless
    a scheduled announcement is specifically anticipated in the article.

    You must output the article ID a topic from TOPICS and a brief reasoning.

    ARTICLES
    {articles}

    EXISTING_QUESTIONS (these topics are ALREADY COVERED -- do NOT pick an
    article about any of these topics, even with a different threshold or
    phrasing. Read every line carefully before selecting.)
    {latest_questions}

    FINAL CHECK before you output:
    - Does any existing question mention the same institution/metric/event
      as your chosen article?
    - If YES, pick a DIFFERENT article. Do not output an article_id for a
      topic that already appears above.

    TOPICS
    {topics}
    """

EXTRACT_STATE_PROMPT = """You are analysing a news article to find MEASURABLE
    STATES that can be turned into prediction-market questions. A measurable
    state is something that can be checked on a specific future date by looking
    up a published value, verifying an ongoing condition, or confirming a
    status-quo persists.

    For each measurable state you find, output:
    - "state": a short description of what can be measured
    - "source": who publishes or confirms this (e.g. "Freddie Mac weekly report",
      "SEC 10-Q filing", "NWS flood gauge", "official company statement")
    - "framing": one of "measurement" (numeric value check), "continuation"
      (will status-quo persist?), or "announcement" (specific event expected)

    PREFER "measurement" and "continuation" framings. Only use "announcement"
    if the article specifically describes an imminent scheduled event.

    If the article has no measurable states at all, return an empty list.

    Return at most 5 states.

    ARTICLE
    {article}
    """

PROPOSE_QUESTION_PROMPT = """You are provided a recent news article
    under ARTICLE, and a list of MEASURABLE STATES extracted from it.
    Your task is to formulate {num_questions} novel prediction market question(s)
    based on these states. Use the measurable states to guide your framing.

    CRITICAL CONSTRAINT: You have a WINDOW of only {window_days} days
    (today -> EVENT_DAY). Every date, threshold, and authority action in
    your question must be achievable within {window_days} days. Do NOT
    reference dates outside this window or in the past.

    RULES:
    - TARGET MIX: when multiple candidate framings are possible for the same
      article, aim for a mix where at least one third of the generated
      questions use "continuation" framing. Continuation questions are the
      main counterweight to the No-bias of announcement framings on short
      windows, so they are actively preferred when an article supports them.

    FRAMING TEMPLATES (use exactly these grammatical shapes -- they avoid the
    past-participle ambiguity of "as confirmed by X" which can be misread as
    "already confirmed by X"):
    - For "measurement" states: frame as
      "Will [metric] be above/below [threshold] on EVENT_DAY, according to
      [source]?"
    - For "continuation" states: frame as
      "Will [condition] still hold on EVENT_DAY, according to [source]?"
    - For "announcement" states: frame as
      "Will [entity] [action] between TODAY ({today}) and EVENT_DAY, according
      to [source]?"
      Do NOT use "on or before EVENT_DAY" -- that admits historical events
      from any point in the past and makes the question trivially true from
      the moment it's asked. The question must resolve on something that
      occurs during the market's lifetime (TODAY to EVENT_DAY).
    - In all templates, use "according to [source]" to name the jury's
      verification channel. Do NOT use "as confirmed by" / "as reported by" /
      "as announced by" -- the past-participle phrasing can be misread as
      "already confirmed/reported/announced at the time the question is asked".
      "According to [source]" is future-tense relative to EVENT_DAY and
      unambiguously means "the jury will check [source] on that date".
    - If MEASURABLE_STATES is empty or shows the sentinel "(none found)",
      you may create an announcement-style question, but it must pass ALL
      the checks below.
    - Must be of public interest, semantically different, different from
      EXISTING_QUESTIONS.
    - The answer must be 'yes' or 'no', verifiable, not an opinion, unambiguous,
      and known after EVENT_DAY.
    - Must not encourage unethical behavior or violence.
    - Must not include unmeasurable statements like "significant increase".
    - ALL dates in the question must be between today and EVENT_DAY. Never
      reference past dates or dates beyond EVENT_DAY.
    - DATE FORMAT: every date in the question MUST be written as
      "Month D, YYYY" (e.g. "April 22, 2026"). Do NOT use "22 April 2026",
      "April 22 2026" (no comma), or numeric formats. Copy EVENT_DAY verbatim.
    - SOURCE PUBLICATION CADENCE: for continuation/measurement questions
      whose source publishes weekly, monthly, quarterly, or irregularly
      (e.g. OECD economic outlooks, Moneyfacts rate tables, Nikkei Asia
      special reports, IMF projections), do NOT require the source to
      publish "on EVENT_DAY" / "published on that date". Frame as
      "as of EVENT_DAY, according to the most recent [source]" or
      "still at/above/below X on EVENT_DAY, per the latest [source]"
      instead. A question that demands a publication action on a specific
      day from a non-daily publisher resolves No by default, regardless of
      the underlying state.

    SPECIFIC FRAMING CHECKS (apply to every question):
    1. DEADLINE FEASIBILITY -- Can the criterion physically/procedurally occur
       within {window_days} days? If not, reframe to something that can.
    2. PROCESS-STAGE CLARITY -- If multi-stage process, name the exact stage.
       Never use "formal passage" or "official approval" without a stage qualifier.
    3. DIRECTLY-PUBLISHED FIGURE -- Thresholds must be figures a source publishes
       directly, not derived by arithmetic on separate figures.
    4. AUTHORITY RESPONSE TIME -- The deadline must be realistic for the named
       authority. Government reviews, regulatory investigations take weeks/months.
       If the authority cannot plausibly act within {window_days} days, do NOT
       frame the question around that authority's action.
    5. RESOLUTION SOURCE -- Name WHO confirms and WHAT document/channel.

    MEASURABLE_STATES
    {measurable_states}

    EXISTING_QUESTIONS
    {latest_questions}

    TODAY
    {today}

    EVENT_DAY
    {event_day}

    ARTICLE
    {article}
    """

SELF_REVIEW_PROMPT = """You are auditing prediction-market questions for
    quality. For EACH question, you must perform explicit step-by-step
    reasoning before deciding accept/reject. Do not skip steps.

    STEP-BY-STEP PROCESS for each question:

    A. STATE THE DEADLINE: Write the exact deadline date from the question.

    B. DEADLINE FEASIBILITY: What specific outcome does the question ask
       for? What is the EARLIEST DATE this could physically/procedurally
       happen? Compare: is earliest_date <= deadline_date?
       - Example: "BBC to complete 1,000 job cuts" -- large-scale layoffs take
         weeks/months to execute. Earliest realistic completion: months away.
         If deadline is 5 days away -> REJECT.
       - Example: "Hurricane landfall in April" -- Atlantic hurricane season
         starts June. Earliest possible: June. April deadline -> REJECT.
       - Example: "FAA suspend laser weapon approval" -- regulatory suspensions
         require review processes taking weeks. 5-day deadline -> REJECT.

    C. PROCESS-STAGE CLARITY: Does the question use ambiguous terms like
       "formal passage", "official approval", "formal review"? If yes -> REJECT
       unless a specific stage is named.

    D. FIGURE DERIVABILITY: Does the question ask for a figure that requires
       arithmetic on two separately-published numbers? If yes -> REJECT.

    E. AUTHORITY RESPONSE TIME: Does the question require a government agency,
       regulator, court, or large organization to complete an action? How long
       does that type of action typically take? Is the deadline realistic?
       - Antitrust reviews: weeks to months -> 5-day deadline = REJECT
       - Regulatory investigations (NHTSA, FAA): weeks -> 5-day deadline = REJECT
       - Large-scale layoffs (1000+ jobs): weeks/months -> 5-day deadline = REJECT
       - Company press release about existing decision: days -> OK

    F. DATE FORMAT: Every date in the question must be written as
       "Month D, YYYY" (e.g. "April 22, 2026"). Formats like "22 April 2026",
       "April 22 2026" (no comma), or numeric dates are INVALID -> REJECT.

    G. NOT A DUPLICATE: Compare the candidate against EXISTING_QUESTIONS. If
       any existing question covers the SAME underlying claim -- same metric
       on the same source, same institution's position, same ongoing event --
       REJECT even if wording differs or the numeric threshold is slightly
       different. Examples of duplicates to reject:
       - candidate "...rate below 5.85%..." when an existing has "...rate below 5.80%..."
       - candidate "S&P 500 above 7,150" when an existing has "S&P 500 above 7,100"
       - candidate on the same institution's report with a slightly different
         numeric threshold than an existing question

    H. WINDOW-BOUND EVENT: For announcement questions, check that the event
       must occur between TODAY and EVENT_DAY -- not "on or before EVENT_DAY"
       or any phrasing that would be satisfied by a historical instance. A
       question like "Will OpenAI publicly announce, on or before EVENT_DAY,
       the signing of a nine-figure enterprise deal?" is INVALID because
       OpenAI has already done this many times in the past. REJECT if the
       literal text would be made True by any pre-TODAY event.

    I. DECISION: Accept ONLY if ALL of B, C, D, E, F, G, H pass.

    Return JSON with your explicit reasoning at each step.

    QUESTIONS
    {questions}

    EXISTING_QUESTIONS
    {latest_questions}

    SOURCE ARTICLE
    {article}

    TODAY
    {today}

    EVENT_DAY
    {event_day}
    """

SOURCE_VERIFY_JUDGE_PROMPT = """You are auditing whether a prediction-market
question can be objectively resolved.

SOURCE: {source}
METRIC: {metric}

GOOGLE EVIDENCE (top organic results querying ``{query}``):
{snippets}

Question: Based on these snippets, does this source publish this specific
metric publicly so a researcher could look up its current value within a
few days?

Reply JSON: {{"answer": "YES" | "NO", "reason": "<one sentence>"}}.

Rules:
- YES only if a snippet clearly shows the source publishing this figure
  (or a strongly equivalent figure on the same cadence).
- NO if snippets are only news commentary, speculation, expert opinion,
  or about a related-but-different topic.
- NO if no snippet contains a specific number / value / measurement tied
  to this metric.
"""

MechResponseWithKeys = Tuple[
    str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]], Any
]
MechResponse = Tuple[
    str, Optional[str], Optional[Dict[str, Any]], Any, Optional[Dict[str, Any]]
]
MaxCostResponse = float


class MeasurableState(BaseModel):
    """One measurable state extracted from an article."""

    state: str
    source: str
    framing: str


class LLMExtractStateSchema(BaseModel):
    """Schema for the extract-state step output."""

    states: List[MeasurableState]


class LLMQuestionProposalSchema(BaseModel):
    """Schema for proposed questions."""

    questions: List[str]


class SelfReviewItem(BaseModel):
    """One question's self-review result with chain-of-thought reasoning."""

    question: str
    deadline_date: str
    earliest_plausible_date: str
    deadline_is_feasible: bool
    process_stage_named: bool
    figure_is_directly_published: bool
    authority_can_act_in_time: bool
    date_format_valid: bool
    not_a_duplicate: bool
    window_bound_event: bool
    reasoning: str
    accept: bool


class LLMSelfReviewSchema(BaseModel):
    """Schema for the self-review pass output."""

    reviews: List[SelfReviewItem]


class LLMStorySelectionSchema(BaseModel):
    """Schema for story selection."""

    topic: str
    article_id: int
    reasoning: str


def validate_question_dates(question: str, resolution_ts: int) -> Optional[str]:
    """Check dates in a question: format, not in the past, not too far ahead.

    Also rejects announcement-style phrasings that admit historical instances
    (e.g. "on or before April 22, 2026") because pre-existing events would
    then make the question trivially true.

    Dates must be written as "Month D, YYYY" (e.g. "April 22, 2026"). Other
    orderings cause ApproveMarketsBehaviour to silently drop the market.

    :param question: the question text to scan for date references.
    :param resolution_ts: the market resolution timestamp (Unix epoch).
    :return: None if OK, else a human-readable rejection reason.
    """
    now = datetime.now(tz=timezone.utc)
    deadline = datetime.fromtimestamp(resolution_ts, tz=timezone.utc)

    if re.search(r"\bon\s+or\s+before\b", question, re.IGNORECASE):
        return (
            "Question uses 'on or before' phrasing which admits historical "
            "instances. Announcement questions must be window-bound "
            "(e.g. 'between TODAY and EVENT_DAY')."
        )

    required_fmt_pattern = r"[A-Z][a-z]+ \d{1,2}, \d{4}"
    any_date_pattern = r"(\w+ \d{1,2},? \d{4}|\d{1,2} \w+ \d{4})"
    for match in re.findall(any_date_pattern, question):
        if not re.fullmatch(required_fmt_pattern, match):
            return (
                f"Date '{match}' is not in required 'Month D, YYYY' format "
                "(e.g. 'April 22, 2026')"
            )
        try:
            d = datetime.strptime(match, "%B %d, %Y").replace(tzinfo=timezone.utc)
        except ValueError:
            return f"Date '{match}' could not be parsed"
        if d < now - timedelta(days=1):
            return f"Date '{match}' is in the past"
        if d > deadline + timedelta(days=365):
            return f"Date '{match}' is too far beyond the deadline"
    return None


class KeyChain:
    """Manages a pool of API keys per service with round-robin rotation."""

    def __init__(self, services: Dict[str, List[str]]) -> None:
        """Initialize with a dictionary of service names to API key lists."""
        if not isinstance(services, dict):
            raise ValueError(
                "Services must be a dictionary with service names as keys and lists of API keys as values."
            )
        self.services = services
        self.current_index = {service: 0 for service in services}

    def max_retries(self) -> Dict[str, int]:
        """Return the maximum number of retries for each service."""
        return {service: len(keys) for service, keys in self.services.items()}

    def rotate(self, service_name: str) -> None:
        """Advance the current key index for a service."""
        if service_name not in self.services:
            raise KeyError(f"Service '{service_name!r}' not found in KeyChain.")
        self.current_index[service_name] += 1
        if self.current_index[service_name] >= len(self.services[service_name]):
            self.current_index[service_name] = 0

    def get(self, service_name: str, default_value: str) -> str:
        """Return the current key for a service, or a default if absent."""
        if service_name not in self.services:
            return default_value
        return self.__getitem__(service_name)

    def __getitem__(self, service_name: str) -> str:
        """Return the current key for a service."""
        if service_name not in self.services:
            raise KeyError(f"Service '{service_name!r}' not found in KeyChain.")
        index = self.current_index[service_name]
        return self.services[service_name][index]


def with_key_rotation(func: Callable) -> Callable:  # noqa
    """Retry func with API-key rotation on RateLimitError."""

    @functools.wraps(func)
    def wrapper(
        *args: Any, **kwargs: Any
    ) -> Union[MaxCostResponse, MechResponseWithKeys]:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> Union[MaxCostResponse, MechResponseWithKeys]:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                # Max-cost path returns a float; pass through without
                # appending api_keys (tuple concatenation would fail).
                if isinstance(result, float):
                    return result
                return result + (api_keys,)
            except openai.RateLimitError as e:
                # try with a new key again; this tool only uses the openai key
                if retries_left["openai"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                api_keys.rotate("openai")
                return execute()
            except Exception as e:
                return str(e), "", None, None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


class OpenAIClientManager:
    """Context manager that creates and closes a local OpenAI client."""

    def __init__(self, api_key: str):
        """Initialize with an API key."""
        self.api_key = api_key
        self._client: Optional[OpenAI] = None

    def __enter__(self) -> OpenAI:
        """Create and return an OpenAI client."""
        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the OpenAI client."""
        if self._client is not None:
            self._client.close()
            self._client = None


def _noop_token_counter(*_args: Any, **_kwargs: Any) -> int:
    """No-op token counter; exact counts are always forwarded from response.usage."""
    return 0


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 4096,
    "temperature": 0.7,
}
ALLOWED_TOOLS = ["propose-question"]
TOOL_TO_ENGINE = {tool: "gpt-4.1-2025-04-14" for tool in ALLOWED_TOOLS}
# Cheaper model for classification/extraction steps (story selection,
# measurable-state extraction). Cuts cost by ~50% with no observable
# quality loss on classification tasks.
LIGHT_MODEL = "gpt-4.1-mini-2025-04-14"
# Representative number of full-model LLM calls per invocation (story
# selection + state extraction + question proposal + self-review). The
# verify-state calls are additional but use the same engine; 4 gives a
# conservative upper bound for the max-cost estimate.
N_MODEL_CALLS = 4
DEFAULT_DELIVERY_RATE = 100

_VERIFY_CACHE: Dict[Tuple[str, str], Tuple[bool, str]] = {}
_VERIFY_CACHE_MAX = 1024


def format_utc_timestamp(utc_timestamp: int) -> str:
    """Format UTC timestamp as 'Month D, YYYY' (US format)."""
    dt = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"


def gather_articles(
    news_sources: List[str], newsapi_api_key: str
) -> Optional[List[Dict[str, Any]]]:
    """Gather news from NewsAPI top-headlines endpoint."""
    headers = {"X-Api-Key": newsapi_api_key}
    parameters = {
        "sources": ",".join(news_sources),
        "pageSize": str(NEWSAPI_MAX_PAGE_SIZE),
    }
    response = requests.get(
        url=NEWSAPI_TOP_HEADLINES_URL,
        headers=headers,
        params=parameters,
        timeout=60,
    )
    if response.status_code != HTTP_OK:
        print(
            f"Could not retrieve response from {NEWSAPI_TOP_HEADLINES_URL}."
            f"Received status code {response.status_code}."
            f"{response}"
        )
        return None
    response_data = json.loads(response.content.decode("utf-8"))
    return response_data["articles"]


def gather_latest_questions(subgraph_api_key: str) -> Optional[List[str]]:
    """Gather latest questions opened on Omen exchange."""
    transport = RequestsHTTPTransport(
        url=OMEN_SUBGRAPH_URL.format(subgraph_api_key=subgraph_api_key)
    )
    # Static query -- skip the per-call GraphQL introspection round-trip.
    gql_client = Client(transport=transport, fetch_schema_from_transport=False)
    variables = {
        "creator_in": FPMM_CREATORS,
        "first": MAX_LATEST_QUESTIONS,
    }
    response = gql_client.execute(gql(FPMMS_QUERY), variable_values=variables)
    items = response.get("fixedProductMarketMakers", [])
    return [q["question"]["title"] for q in items]


DEDUP_STOPWORDS = frozenset(
    "a an and any are as at be been being between by did do does for from had "
    "has have if in into is it its of on once or over per still such than that "
    "the their them there these they this to up was were what when where which "
    "who will with would according announce announced announcement confirm "
    "confirmed confirms continue continues hold holds major news occur occurs "
    "official outlet outlets publish published report reports reported "
    "reporting remain remains statement statements yes no maintain maintains "
    "through above below exceed exceeds please".split()
)

# Jaccard similarity at or above this threshold counts as a near-duplicate.
# Tuned against real pool data: 0.64 on the classic "retail sales / retail
# sales excluding gas prices" pair; ~0.30 on genuinely different questions.
QUESTION_NEAR_DUP_THRESHOLD = 0.55


def _dedup_tokens(text: str) -> set:
    """Normalise text to a content-token set for Jaccard dedup comparisons."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9% ]", " ", text)
    return {
        t
        for t in re.sub(r"\s+", " ", text).strip().split()
        if t and t not in DEDUP_STOPWORDS and len(t) > 2
    }


def filter_duplicate_articles(
    articles: List[Dict[str, Any]],
    existing: List[str],
    threshold: float = 0.40,
    min_keep_fraction: float = 0.50,
) -> List[Dict[str, Any]]:
    """Drop articles whose title+description overlap too much with existing questions.

    Cheap first-stage filter that runs before any LLM call. Moderate threshold
    (0.40) so articles with a distinct sub-angle survive; the stricter
    question-level filter (0.55) at the end catches what slips through. Safety
    valve keeps at least ``min_keep_fraction`` of the pool to prevent
    empty-pool errors when news is extremely topic-concentrated.

    :param articles: raw article dicts from gather_articles.
    :param existing: list of recent question titles (EXISTING_QUESTIONS pool).
    :param threshold: Jaccard cutoff for dropping an article.
    :param min_keep_fraction: minimum fraction of input to keep regardless.
    :return: filtered article list (length >= min_keep_fraction * len(articles)).
    """
    if not articles or not existing:
        return articles
    existing_token_sets = [t for t in (_dedup_tokens(q) for q in existing) if t]
    if not existing_token_sets:
        return articles
    scored = []
    for a in articles:
        text = f"{a.get('title', '')} {a.get('description', '')}"
        atok = _dedup_tokens(text)
        if not atok:
            scored.append((0.0, a))
            continue
        max_sim = 0.0
        for qtok in existing_token_sets:
            denom = len(atok | qtok)
            if denom == 0:
                continue
            sim = len(atok & qtok) / denom
            if sim > max_sim:
                max_sim = sim
        scored.append((max_sim, a))
    kept = [a for sim, a in scored if sim < threshold]
    n_in = len(articles)
    min_keep = max(1, int(n_in * min_keep_fraction))
    if len(kept) < min_keep:
        scored.sort(key=lambda t: t[0])
        kept = [a for _, a in scored[:min_keep]]
        print(
            f"Article dedup: safety valve triggered, keeping "
            f"{len(kept)}/{n_in} least-similar articles"
        )
    else:
        print(
            f"Article dedup: dropped {n_in - len(kept)}/{n_in} articles "
            f"(jaccard >= {threshold})"
        )
    return kept


def find_near_duplicate(
    question: str,
    existing: List[str],
    threshold: float = QUESTION_NEAR_DUP_THRESHOLD,
) -> Optional[Tuple[str, float]]:
    """Return (matching_existing, jaccard) if question is a near-duplicate.

    The LLM-based ``not_a_duplicate`` self-review misses paraphrases like
    "retail sales % change" vs "monthly retail sales % change excluding gas
    prices" -- both test the same metric from the same source, Jaccard ~0.64.
    This programmatic filter catches them deterministically.

    :param question: candidate question text.
    :param existing: list of already-accepted question titles.
    :param threshold: Jaccard cutoff; defaults to QUESTION_NEAR_DUP_THRESHOLD.
    :return: (matched_existing, jaccard_score) if a duplicate is found, else None.
    """
    new_tok = _dedup_tokens(question)
    if not new_tok:
        return None
    for old in existing:
        old_tok = _dedup_tokens(old)
        denom = len(new_tok | old_tok)
        if denom == 0:
            continue
        score = len(new_tok & old_tok) / denom
        if score >= threshold:
            return old, score
    return None


def scrape_url(serper_api_key: str, url: str) -> Optional[dict]:
    """Scrape the contents of a URL via the Serper scrape endpoint."""
    serper_url = "https://scrape.serper.dev"
    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    payload = json.dumps({"url": url})
    try:
        response = requests.post(serper_url, headers=headers, data=payload, timeout=60)
        response.raise_for_status()
        scraped_data = response.json()
        print(f"Successfully scraped URL: {url}")
        return scraped_data
    except requests.RequestException:
        return None
    except json.JSONDecodeError:
        return None


def _cache_put(key: Tuple[str, str], result: Tuple[bool, str]) -> None:
    """Insert a verifier result into the cache, evicting oldest on overflow.

    :param key: (source, metric) tuple identifying the verified claim.
    :param result: (is_resolvable, reason) tuple to cache.
    """
    if len(_VERIFY_CACHE) >= _VERIFY_CACHE_MAX:
        oldest = next(iter(_VERIFY_CACHE), None)
        if oldest is not None:
            _VERIFY_CACHE.pop(oldest, None)
    _VERIFY_CACHE[key] = result


def verify_state_is_resolvable(
    client: OpenAI, serper_api_key: str, source: str, metric: str
) -> Tuple[bool, str]:
    """Decide whether a (source, metric) tuple is publicly resolvable.

    Issues one Google search via Serper and passes the top organic snippets
    to a cheap LLM judge, which decides whether the source publishes the
    specific metric on a checkable cadence.

    Results are cached in a process-local dict keyed by (source, metric) so
    the same article re-picked across cycles does not re-burn Serper + LLM
    calls. Fail-open: Serper or OpenAI errors return (True, ...) so a
    transient outage cannot starve generation.

    :param client: an active OpenAI client instance.
    :param serper_api_key: Serper API key for the Google query.
    :param source: source-of-truth name from the extracted measurable state.
    :param metric: metric/state name being asked about.
    :return: (is_resolvable, reason).
    """
    cache_key = (source, metric)
    cached = _VERIFY_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], f"{cached[1]} [cached]"
    query = f"{source} {metric}".strip()
    if not query:
        return True, "empty_query"
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            data=payload,
            timeout=30,
        )
        if response.status_code != 200:
            return True, f"fail_open_serper_status_{response.status_code}"
        body = response.json()
        if "organic" not in body:
            return True, "fail_open_serper_unexpected_shape"
        organic = body["organic"][:5]
    except (requests.RequestException, json.JSONDecodeError):
        return True, "fail_open_serper_error"

    if not organic:
        result = (False, "no_hits")
        _cache_put(cache_key, result)
        return result

    snippets = "\n".join(
        f"- TITLE: {o.get('title', '')}\n  SNIPPET: {o.get('snippet', '')}"
        for o in organic
    )
    try:
        judge_response = client.chat.completions.create(
            model=LIGHT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": SOURCE_VERIFY_JUDGE_PROMPT.format(
                        source=source,
                        metric=metric,
                        query=query,
                        snippets=snippets,
                    ),
                }
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
            timeout=30,
        )
        content = judge_response.choices[0].message.content or ""
        out = json.loads(content)
    except (
        openai.OpenAIError,
        json.JSONDecodeError,
        AttributeError,
        KeyError,
        TypeError,
    ):
        return True, "fail_open_llm_error"
    result = (out.get("answer") == "YES", str(out.get("reason") or "")[:120])
    _cache_put(cache_key, result)
    return result


@with_key_rotation
def run(**kwargs: Any) -> Union[MaxCostResponse, MechResponse]:
    """Propose prediction-market questions from recent news.

    Accepts kwargs from the mech executor. The ``prompt`` field may be a
    JSON-encoded dict with ``resolution_time`` and optionally
    ``num_questions``; plain-string prompts fall back to reading those
    values from kwargs directly.

    When ``delivery_rate == 0`` the function returns the estimated maximum
    cost as a bare ``float`` (via ``counter_callback(max_cost=True, ...)``),
    bypassing all API calls.

    :param kwargs: Keyword arguments including 'tool', 'prompt', 'api_keys',
        'counter_callback', 'resolution_time', 'num_questions',
        'delivery_rate'.
    :type kwargs: Any

    :return: estimated max cost float when delivery_rate==0, else 5-tuple
        (result_json, prompt_echo, None, counter_callback, used_params).
    :rtype: Union[MaxCostResponse, MechResponse]
    """
    try:
        counter_callback = kwargs.get("counter_callback", None)
        tool = kwargs.get("tool")
        delivery_rate = int(kwargs.get("delivery_rate", DEFAULT_DELIVERY_RATE))
        if delivery_rate == 0:
            if not counter_callback:
                raise ValueError(
                    "A delivery rate of `0` was passed, but no counter callback was "
                    "given to calculate the max cost with."
                )
            tool_engine = TOOL_TO_ENGINE.get(
                kwargs.get("tool", ""), list(TOOL_TO_ENGINE.values())[0]
            )
            max_cost: MaxCostResponse = counter_callback(
                max_cost=True,
                models_calls=(tool_engine,) * N_MODEL_CALLS,
            )
            return max_cost

        if not tool or tool not in ALLOWED_TOOLS:
            return (
                json.dumps(
                    {
                        "error": f"Tool {tool} is not in the list of supported tools.",
                        "tool": tool,
                    }
                ),
                None,
                None,
                counter_callback,
                None,
            )

        # Decode resolution_time and num_questions from prompt JSON if present.
        prompt_raw = kwargs.get("prompt", "")
        resolution_time = kwargs.get("resolution_time")
        num_questions = kwargs.get("num_questions")
        try:
            prompt_dict = json.loads(prompt_raw) if prompt_raw else {}
            if isinstance(prompt_dict, dict):
                if resolution_time is None:
                    resolution_time = prompt_dict.get("resolution_time")
                if num_questions is None:
                    num_questions = prompt_dict.get("num_questions")
        except (json.JSONDecodeError, TypeError):
            pass

        if resolution_time is None:
            return (
                json.dumps(
                    {
                        "error": "'resolution_time' is not defined.",
                        "tool": tool,
                    }
                ),
                None,
                None,
                counter_callback,
                None,
            )
        if num_questions is None:
            num_questions = 1

        # Gather latest opened questions from input or from TheGraph.
        latest_questions = kwargs.get("latest_questions")
        if latest_questions is None:
            latest_questions = gather_latest_questions(kwargs["api_keys"]["subgraph"])
        if latest_questions is None:
            return (
                json.dumps(
                    {
                        "error": "Failed to retrieve latest questions.",
                        "tool": tool,
                    }
                ),
                None,
                None,
                counter_callback,
                None,
            )

        # Keep the MAX_LATEST_QUESTIONS most recent (subgraph already returns
        # them newest-first).
        latest_questions = latest_questions[:MAX_LATEST_QUESTIONS]
        latest_questions_string = "\n".join(latest_questions)

        # Gather recent news articles from NewsAPI.
        news_sources = kwargs.get("news_sources", NEWSAPI_DEFAULT_NEWS_SOURCES)
        articles = gather_articles(news_sources, kwargs["api_keys"]["newsapi"])
        if articles is None:
            return (
                json.dumps(
                    {
                        "error": "Failed to retrieve articles from NewsAPI.",
                        "tool": tool,
                    }
                ),
                None,
                None,
                counter_callback,
                None,
            )

        print(
            f"{len(articles)} articles collected from {len(news_sources)} news sources\n"
        )
        articles = random.sample(
            articles, min(MAX_ARTICLES, len(articles))
        )  # nosec: B311

        # Early article-level dedup: drop articles whose topic already appears
        # in EXISTING_QUESTIONS. Moderate threshold (0.40) to keep articles with
        # distinct sub-angles; safety valve prevents empty-pool errors.
        articles = filter_duplicate_articles(articles, latest_questions)

        # Pre-filter: drop articles that individually trigger content moderation
        # before building the selection prompt, avoiding flagging the whole prompt
        # due to a single violent-news snippet.
        with OpenAIClientManager(kwargs["api_keys"]["openai"]) as openai_client:
            clean_articles = []
            for article in articles:
                text = f"{article.get('title', '')}: {article.get('content', '')}"
                try:
                    mod = openai_client.moderations.create(input=text)
                    if not mod.results[0].flagged:
                        clean_articles.append(article)
                except Exception:
                    clean_articles.append(article)
            n_dropped = len(articles) - len(clean_articles)
            if n_dropped > 0:
                print(f"Moderation pre-filter: dropped {n_dropped} flagged article(s)")
            articles = clean_articles

        if not articles:
            return (
                json.dumps(
                    {
                        "error": "All articles were flagged by content moderation.",
                        "tool": tool,
                    }
                ),
                None,
                None,
                counter_callback,
                None,
            )

        articles_string = ""
        for i, article in enumerate(articles, start=0):
            articles_string += f"{i} - {article.get('title', '')} ({article.get('publishedAt', '')}): {article.get('content', '')}\n"

        topics = kwargs.get("topics", DEFAULT_TOPICS)
        topics_string = ", ".join(topics)

        # Story selection -- classification task, uses the cheaper light model.
        model = LIGHT_MODEL
        prompt_values = {
            "articles": articles_string,
            "topics": topics_string,
            "latest_questions": latest_questions_string,
        }
        prompt = SELECT_STORY_PROMPT.format(**prompt_values)

        with OpenAIClientManager(kwargs["api_keys"]["openai"]) as openai_client:
            max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
            temperature = kwargs.get(
                "temperature", DEFAULT_OPENAI_SETTINGS["temperature"]
            )

            moderation_result = openai_client.moderations.create(input=prompt)
            if moderation_result.results[0].flagged:
                return (
                    json.dumps(
                        {
                            "error": "Moderation flagged the prompt as in violation of terms.",
                            "tool": tool,
                        }
                    ),
                    None,
                    None,
                    counter_callback,
                    None,
                )

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            response = openai_client.chat.completions.create(  # type: ignore[call-overload]
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                timeout=120,
                stop=None,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "LLMStorySelectionSchema",
                        "schema": LLMStorySelectionSchema.model_json_schema(),
                    },
                },
            )
            if counter_callback:
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=model,
                    token_counter=_noop_token_counter,
                )
            response_data = json.loads(response.choices[0].message.content)
            article_id = response_data["article_id"]
            topic = response_data["topic"]
            if not isinstance(article_id, int) or not 0 <= article_id < len(articles):
                return (
                    json.dumps(
                        {
                            "error": (
                                f"LLM returned invalid article_id {article_id!r} "
                                f"(have {len(articles)} articles)."
                            ),
                            "tool": tool,
                        }
                    ),
                    None,
                    None,
                    counter_callback,
                    None,
                )
            article = articles[article_id]
            reasoning = (
                f"The article {article['title']!r} "
                f"({article.get('author', '')!r}) has been selected to generate "
                f"prediction market questions because: {response_data['reasoning']}"
            )

        # Scrape the selected article for full body text.
        scrape_result = scrape_url(
            kwargs["api_keys"]["serperapi"], article.get("url", "")
        )
        if scrape_result is None:
            return (
                json.dumps(
                    {
                        "error": f"Failed to scrape url {article.get('url', '')}",
                        "tool": tool,
                    }
                ),
                None,
                None,
                counter_callback,
                None,
            )

        # Extract measurable states -- constrains the LLM to identify what CAN
        # be measured before framing a question, breaking the default
        # "Will X announce Y?" prior.
        article_text = scrape_result["text"][:ARTICLE_TEXT_MAX_CHARS]
        with OpenAIClientManager(kwargs["api_keys"]["openai"]) as openai_client:
            model = LIGHT_MODEL
            extract_prompt = EXTRACT_STATE_PROMPT.format(article=article_text)
            extract_messages = [
                {
                    "role": "system",
                    "content": "You are an analyst extracting measurable facts from news articles.",
                },
                {"role": "user", "content": extract_prompt},
            ]
            extract_response = openai_client.chat.completions.create(
                model=model,
                messages=extract_messages,
                temperature=0.3,
                max_tokens=1024,
                n=1,
                timeout=120,
                stop=None,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "LLMExtractStateSchema",
                        "schema": LLMExtractStateSchema.model_json_schema(),
                    },
                },
            )
            if counter_callback:
                counter_callback(
                    input_tokens=extract_response.usage.prompt_tokens,
                    output_tokens=extract_response.usage.completion_tokens,
                    model=model,
                    token_counter=_noop_token_counter,
                )
            extract_data = json.loads(extract_response.choices[0].message.content)
            states = extract_data.get("states", [])
            print(f"Extracted {len(states)} measurable states:")
            for s in states:
                print(
                    f"  [{s.get('framing', '?')}] {s.get('state', '?')} -- source: {s.get('source', '?')}"
                )

        # Verify each extracted (source, metric) is actually resolvable via
        # a Serper search + cheap LLM judge. Filters fictional sources before
        # they reach question generation.
        with OpenAIClientManager(kwargs["api_keys"]["openai"]) as openai_client:
            verified_states = []
            for s in states:
                src = s.get("source", "") or ""
                metric = s.get("state", "") or ""
                is_verified, reason = verify_state_is_resolvable(
                    openai_client, kwargs["api_keys"]["serperapi"], src, metric
                )
                if is_verified:
                    verified_states.append(s)
                else:
                    print(
                        f"  DROPPED unverifiable state ({reason}): "
                        f"[{s.get('framing', '?')}] {metric} -- source: {src}"
                    )
            print(
                f"Source verification: kept {len(verified_states)}/{len(states)} states"
            )
            if not verified_states and states:
                # Soft fall-through: every extracted state failed verification.
                # Let the question-generator work from the article body alone --
                # the self-review pass is the next gate that rejects un-resolvable
                # framings.
                print(
                    f"WARN: 0/{len(states)} states verified; falling back to "
                    f"article-only generation (no measurable_states context)."
                )
            states = verified_states
            states_string = json.dumps(states, indent=2) if states else "(none found)"

        # Generate candidate questions using the extracted states.
        with OpenAIClientManager(kwargs["api_keys"]["openai"]) as openai_client:
            max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
            temperature = kwargs.get(
                "temperature", DEFAULT_OPENAI_SETTINGS["temperature"]
            )
            model = kwargs.get("model", TOOL_TO_ENGINE[tool])

            # Generate 3x candidates (min 3) so self-review can act as a
            # selector. Reduced from 5x to keep self-review token cost
            # proportional (review prompt scales linearly with candidates).
            n_candidates = max(num_questions * 3, 3)
            window_days = max(
                1,
                (int(resolution_time) - int(datetime.now(tz=timezone.utc).timestamp()))
                // 86400,
            )
            prompt_values = {
                "article": f"{article_text}",
                "today": format_utc_timestamp(
                    int(datetime.now(tz=timezone.utc).timestamp())
                ),
                "event_day": format_utc_timestamp(int(resolution_time)),
                "latest_questions": latest_questions_string,
                "measurable_states": states_string,
                "num_questions": f"{n_candidates}",
                "window_days": str(window_days),
            }
            propose_prompt = PROPOSE_QUESTION_PROMPT.format(**prompt_values)

            moderation_result = openai_client.moderations.create(input=propose_prompt)
            if moderation_result.results[0].flagged:
                return (
                    json.dumps(
                        {
                            "error": "Moderation flagged the prompt as in violation of terms.",
                            "tool": tool,
                        }
                    ),
                    None,
                    None,
                    counter_callback,
                    None,
                )

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": propose_prompt},
            ]
            response = openai_client.chat.completions.create(  # type: ignore[call-overload]
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                timeout=120,
                stop=None,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "LLMQuestionProposalSchema",
                        "schema": LLMQuestionProposalSchema.model_json_schema(),
                    },
                },
            )
            if counter_callback:
                counter_callback(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    model=model,
                    token_counter=_noop_token_counter,
                )
            response_data = json.loads(response.choices[0].message.content)

        # Self-review: audit proposed questions against quality checks.
        questions = response_data["questions"]
        review_model = TOOL_TO_ENGINE[tool]

        with OpenAIClientManager(kwargs["api_keys"]["openai"]) as openai_client:
            review_prompt = SELF_REVIEW_PROMPT.format(
                questions=json.dumps(questions, indent=2),
                latest_questions=latest_questions_string,
                article=f"{scrape_result['text'][:ARTICLE_TEXT_MAX_CHARS]}",
                today=format_utc_timestamp(
                    int(datetime.now(tz=timezone.utc).timestamp())
                ),
                event_day=format_utc_timestamp(int(resolution_time)),
            )
            review_messages = [
                {
                    "role": "system",
                    "content": "You are a prediction-market question auditor.",
                },
                {"role": "user", "content": review_prompt},
            ]
            review_response = openai_client.chat.completions.create(
                model=review_model,
                messages=review_messages,
                temperature=0.0,
                max_tokens=2048,
                n=1,
                timeout=120,
                stop=None,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "LLMSelfReviewSchema",
                        "schema": LLMSelfReviewSchema.model_json_schema(),
                    },
                },
            )
            if counter_callback:
                counter_callback(
                    input_tokens=review_response.usage.prompt_tokens,
                    output_tokens=review_response.usage.completion_tokens,
                    model=review_model,
                    token_counter=_noop_token_counter,
                )
            review_data = json.loads(review_response.choices[0].message.content)
            reviews = review_data.get("reviews", [])

        # Accept questions only when ALL seven self-review checks pass.
        accepted_questions = []
        rejected_questions = []
        for rev in reviews:
            checks = [
                rev.get("deadline_is_feasible", True),
                rev.get("process_stage_named", True),
                rev.get("figure_is_directly_published", True),
                rev.get("authority_can_act_in_time", True),
            ]
            passes = sum(1 for c in checks if c)
            date_ok = rev.get("date_format_valid", True)
            not_dup = rev.get("not_a_duplicate", True)
            window_ok = rev.get("window_bound_event", True)
            if passes == len(checks) and date_ok and not_dup and window_ok:
                accepted_questions.append(rev["question"])
            else:
                rejected_questions.append(
                    {
                        "question": rev["question"],
                        "reason": rev.get(
                            "reasoning",
                            rev.get("rejection_reason", "failed self-review"),
                        ),
                    }
                )

        # Programmatic date validation -- catches past dates and out-of-range
        # dates that the LLM self-review misses.
        date_validated = []
        for q in accepted_questions:
            date_issue = validate_question_dates(q, int(resolution_time))
            if date_issue:
                rejected_questions.append(
                    {"question": q, "reason": f"DATE CHECK: {date_issue}"}
                )
            else:
                date_validated.append(q)

        print(
            f"Self-review: {len(accepted_questions)} accepted, "
            f"{len(rejected_questions)} rejected out of {n_candidates} proposed"
        )
        for rej in rejected_questions:
            print(f"  REJECTED: {rej['question'][:80]}... -- {rej['reason'][:60]}")

        # Programmatic near-duplicate filter: last-stage deterministic check.
        nondup_validated = []
        for q in date_validated:
            hit = find_near_duplicate(q, latest_questions)
            if hit is None:
                nondup_validated.append(q)
            else:
                match, score = hit
                print(
                    f"  PROGRAMMATIC DEDUP REJECT (jaccard {score:.2f}): {q[:80]}...\n"
                    f"    matches existing: {match[:80]}..."
                )

        if not nondup_validated:
            err_payload = {
                "error": (
                    f"All {n_candidates} proposed questions were rejected by "
                    "self-review, date validation, or programmatic dedup."
                ),
                "tool": tool,
            }
            return json.dumps(err_payload), None, None, counter_callback, None

        questions = nondup_validated[:num_questions]

        answers = ["Yes", "No"]
        language = "en_US"
        questions_dict = {}
        for q in questions:
            question_id = str(uuid.uuid4())
            questions_dict[question_id] = {
                "answers": answers,
                "id": question_id,
                "language": language,
                "question": q,
                "resolution_time": resolution_time,
                "topic": topic,
                "article": article,
            }

        output = {
            "questions": questions_dict,
            "reasoning": reasoning,
            "article": article,
        }
        used_params = {"model": model}
        return (
            json.dumps(output, sort_keys=True),
            prompt_raw,
            None,
            counter_callback,
            used_params,
        )
    except openai.RateLimitError:
        # Let @with_key_rotation handle rate limits (rotate key + retry);
        # swallowing it here would make that decorator dead code.
        raise
    except Exception as e:
        return (
            json.dumps(
                {
                    "error": f"An exception has occurred: {e}.",
                    "tool": tool,
                }
            ),
            None,
            None,
            counter_callback,
            None,
        )
