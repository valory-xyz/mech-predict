"""Fetch open Omen prediction markets and snapshot web content per tool group.

Queries the Omen subgraph for unresolved binary markets, then for each market:
  - Group A (superforcaster): 1 Serper call with raw question (snippets only)
  - Group B (factual_research): LLM decomposes into 3-6 sub-questions → Serper each
  - Group C (rag, reasoning, sme, url_cot): LLM generates 2-5 queries → Serper each

Scraped pages are deduplicated across groups. Each group gets its own
``source_links`` mapping for cached replay.

Usage:
    python benchmark/datasets/fetch_open.py --dry-run
    python benchmark/datasets/fetch_open.py --max-markets 5
    python benchmark/datasets/fetch_open.py --skip-search
    python benchmark/datasets/fetch_open.py --groups a      # snippet-only, no LLM
    python benchmark/datasets/fetch_open.py --groups a,b,c  # all groups (default)

Environment variables:
    SERPER_API_KEY   Required unless --skip-search is set.
    OPENAI_API_KEY   Required for groups B and C (LLM query generation).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests
from dotenv import load_dotenv
from markdownify import markdownify as md
from readability import Document as ReadabilityDocument

load_dotenv()

# Unbuffered print for real-time output when piped
print = functools.partial(__builtins__.__dict__["print"], flush=True)  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OMEN_SUBGRAPH_URL = "https://omen.subgraph.autonolas.tech"
SERPER_URL = "https://google.serper.dev/search"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"

# Market creators that the trader bets on (Omen)
OMEN_CREATORS = [
    "0xFfc8029154ECD55ABED15BD428bA596E7D23f557",  # Pearl
    "0x89c5cc945dd550BcFfb72Fe42BfF002429F46Fec",  # Quickstart (QS)
]

# Polymarket categories (same as trader)
POLYMARKET_CATEGORIES = [
    "business", "politics", "science", "technology", "health",
    "travel", "entertainment", "weather", "finance", "international",
]
POLYMARKET_WINDOW_DAYS = 14

SEP = "\u241f"
INVALID_ANSWER = "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

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

USER_AGENT = "Mozilla/5.0 (compatible; MechBot/1.0)"

_IMG_TAG_PATTERN = re.compile(r"<img[^>]*>", re.IGNORECASE)
_SCRIPT_STYLE_PATTERN = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)

MAX_HTML_BYTES = 1_000_000  # 1 MB cap per page

# LLM query generation defaults
GROUP_B_MODEL = "gpt-4.1-2025-04-14"  # matches factual_research
GROUP_C_MODEL = "gpt-4.1-2025-04-14"
GROUP_C_NUM_QUERIES = 5  # matches reasoning/url_cot

# Serper concurrency (keep low for free tier)
SERPER_WORKERS = 3


# ---------------------------------------------------------------------------
# LLM prompt templates (copied from tools to match their exact behavior)
# ---------------------------------------------------------------------------

# Group B — factual_research reframe prompt
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

# Group C — RAG/reasoning/sme/url_cot query generation prompt
URL_QUERY_PROMPT = """
Here is the user prompt: {user_prompt}

Please read the prompt carefully and identify the key pieces of information that need to be searched for in order to comprehensively address the topic.

Brainstorm a list of {num_queries} different search queries that cover various aspects of the user prompt. Each query should be focused on a specific sub-topic or question related to the overarching prompt.

Please write each search query inside its own tags, like this: <query>example search query here</query>

The queries should be concise while still containing enough information to return relevant search results. Focus the queries on gathering factual information to address the prompt rather than opinions.

After you have written all {num_queries} search queries, please submit your final response.

<queries></queries>
"""


# ---------------------------------------------------------------------------
# GraphQL helper
# ---------------------------------------------------------------------------


def _post_graphql(
    url: str,
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    retries: int = 4,
) -> Dict[str, Any]:
    """POST a GraphQL query with retry and exponential backoff."""
    payload: Dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables
    for attempt in range(retries):
        try:
            r = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=90,
            )
            r.raise_for_status()
            d = r.json()
            if "errors" in d:
                raise RuntimeError(d["errors"])
            return d["data"]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (2**attempt))
    raise RuntimeError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Step 1 — Fetch open markets from Omen subgraph
# ---------------------------------------------------------------------------


def fetch_open_markets(max_markets: int = 500) -> List[Dict[str, Any]]:
    """Fetch open (unresolved) binary markets from the Omen subgraph."""
    markets: List[Dict[str, Any]] = []
    skip = 0

    creators_filter = json.dumps([c.lower() for c in OMEN_CREATORS])

    while len(markets) < max_markets:
        data = _post_graphql(
            OMEN_SUBGRAPH_URL,
            f"""
            {{
              fixedProductMarketMakers(
                first: 1000
                skip: {skip}
                orderBy: creationTimestamp
                orderDirection: desc
                where: {{
                  currentAnswer: null
                  outcomeSlotCount: 2
                  creator_in: {creators_filter}
                }}
              ) {{
                id
                title
                outcomes
                outcomeTokenAmounts
                outcomeTokenMarginalPrices
                collateralVolume
                usdVolume
                liquidityMeasure
                usdLiquidityMeasure
                creationTimestamp
                openingTimestamp
                category
              }}
            }}
            """,
        )
        batch = data.get("fixedProductMarketMakers", [])
        if not batch:
            break

        for fpmm in batch:
            market_id = fpmm.get("id", "")
            if not market_id:
                continue

            outcomes = fpmm.get("outcomes") or []
            if len(outcomes) != 2:
                continue

            title = (fpmm.get("title") or "").strip()
            if not title:
                continue

            prices = fpmm.get("outcomeTokenMarginalPrices") or []
            current_prob = None
            if len(prices) == 2:
                try:
                    current_prob = round(float(prices[0]), 4)
                except (ValueError, TypeError):
                    pass

            try:
                usd_volume = round(float(fpmm.get("usdVolume", 0)), 2)
            except (ValueError, TypeError):
                usd_volume = 0.0
            try:
                usd_liquidity = round(float(fpmm.get("usdLiquidityMeasure", 0)), 2)
            except (ValueError, TypeError):
                usd_liquidity = 0.0

            markets.append(
                {
                    "id": f"omen_{market_id}",
                    "market_address": market_id,
                    "question": title,
                    "platform": "omen",
                    "outcomes": outcomes,
                    "current_prob": current_prob,
                    "usd_volume": usd_volume,
                    "usd_liquidity": usd_liquidity,
                    "category": fpmm.get("category") or "",
                    "creation_timestamp": int(fpmm.get("creationTimestamp", 0)),
                    "opening_timestamp": int(fpmm.get("openingTimestamp", 0)),
                }
            )
            if len(markets) >= max_markets:
                break

        if len(batch) < 1000:
            break
        skip += 1000

    return markets


# ---------------------------------------------------------------------------
# Step 1b — Fetch open markets from Polymarket (Gamma API)
# ---------------------------------------------------------------------------


def _fetch_polymarket_tag_id(category: str) -> Optional[int]:
    """Fetch the tag ID for a Polymarket category slug."""
    try:
        resp = requests.get(
            f"{POLYMARKET_GAMMA_URL}/tags/slug/{category}", timeout=10
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("id")
    except Exception:
        return None


def fetch_open_polymarket(
    max_markets: int = 500, window_days: int = POLYMARKET_WINDOW_DAYS
) -> List[Dict[str, Any]]:
    """Fetch open binary markets from Polymarket via the Gamma API."""
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    end_date_min = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date_max = (now + timedelta(days=window_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    seen_ids: set = set()
    markets: List[Dict[str, Any]] = []

    for category in POLYMARKET_CATEGORIES:
        if len(markets) >= max_markets:
            break

        tag_id = _fetch_polymarket_tag_id(category)
        if tag_id is None:
            print(f"  [polymarket] Skipping category '{category}' (no tag ID)")
            continue

        offset = 0
        while len(markets) < max_markets:
            try:
                resp = requests.get(
                    f"{POLYMARKET_GAMMA_URL}/markets",
                    params={
                        "tag_id": tag_id,
                        "end_date_min": end_date_min,
                        "end_date_max": end_date_max,
                        "limit": 300,
                        "offset": offset,
                        "closed": "false",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                batch = resp.json()
            except Exception as e:
                print(f"  [polymarket] Fetch failed for '{category}': {e}")
                break

            if not batch:
                break

            for m in batch:
                market_id = m.get("conditionId") or m.get("id", "")
                if not market_id or market_id in seen_ids:
                    continue
                seen_ids.add(market_id)

                # Binary filter — outcomes is a JSON string
                outcomes_raw = m.get("outcomes", "[]")
                try:
                    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if len(outcomes) != 2:
                    continue
                if not all(
                    o.lower() in ("yes", "no") for o in outcomes
                ):
                    continue

                # Skip resolved (any price >= 0.99)
                prices_raw = m.get("outcomePrices", "[]")
                try:
                    prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                    prices = [float(p) for p in prices]
                except (json.JSONDecodeError, TypeError, ValueError):
                    prices = []
                if any(p >= 0.99 for p in prices):
                    continue

                # Skip zero liquidity
                try:
                    liquidity = float(m.get("liquidity", 0))
                except (ValueError, TypeError):
                    liquidity = 0.0
                if liquidity <= 0:
                    continue

                # Skip negative risk
                if m.get("negRisk", False):
                    continue

                question = (m.get("question") or "").strip()
                if not question:
                    continue

                current_prob = prices[0] if len(prices) >= 2 else None

                try:
                    volume = round(float(m.get("volume", 0)), 2)
                except (ValueError, TypeError):
                    volume = 0.0

                # Parse timestamps
                end_date = m.get("endDate", "")
                created_at = m.get("createdAt", "")
                try:
                    opening_ts = int(
                        datetime.fromisoformat(
                            end_date.replace("Z", "+00:00")
                        ).timestamp()
                    )
                except (ValueError, TypeError):
                    opening_ts = 0
                try:
                    creation_ts = int(
                        datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        ).timestamp()
                    )
                except (ValueError, TypeError):
                    creation_ts = 0

                markets.append(
                    {
                        "id": f"poly_{market_id}",
                        "market_address": market_id,
                        "question": question,
                        "platform": "polymarket",
                        "outcomes": outcomes,
                        "current_prob": round(current_prob, 4) if current_prob is not None else None,
                        "usd_volume": volume,
                        "usd_liquidity": round(liquidity, 2),
                        "category": category,
                        "creation_timestamp": creation_ts,
                        "opening_timestamp": opening_ts,
                    }
                )
                if len(markets) >= max_markets:
                    break

            if len(batch) < 300:
                break
            offset += 300

    return markets


# ---------------------------------------------------------------------------
# Step 2 — Serper web search
# ---------------------------------------------------------------------------


class SerperRateLimitError(Exception):
    """Raised when Serper returns a rate limit error."""


def search_serper(
    query: str, api_key: str, num_results: int = 10, retries: int = 3
) -> Dict[str, Any]:
    """Run a Google search via Serper with retry on rate limit."""
    payload = {"q": query, "num": num_results}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    for attempt in range(retries):
        response = requests.post(SERPER_URL, headers=headers, json=payload, timeout=30)
        if response.status_code in (429, 400):
            wait = 10 * (2**attempt)
            print(f"    [serper] Rate limited (HTTP {response.status_code}), waiting {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise SerperRateLimitError(
        f"Serper rate limit after {retries} retries — quota may be exhausted"
    )


def search_serper_batch(
    queries: List[str],
    api_key: str,
    num_results: int = 5,
    delay: float = 0.5,
) -> List[Dict[str, Any]]:
    """Search multiple queries via Serper with controlled concurrency."""
    results: List[Optional[Dict[str, Any]]] = [None] * len(queries)

    with ThreadPoolExecutor(max_workers=SERPER_WORKERS) as pool:
        future_to_idx = {}
        for idx, q in enumerate(queries):
            # Stagger submissions to respect rate limits
            if idx > 0:
                time.sleep(delay)
            future_to_idx[pool.submit(search_serper, q, api_key, num_results)] = idx

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"    [serper] Query failed: {e}")
                results[idx] = {"organic": [], "peopleAlsoAsk": []}

    return [r for r in results if r is not None]


def extract_urls_from_serper(
    serper_response: Dict[str, Any], max_urls: int = 10
) -> List[str]:
    """Extract URLs from a Serper response, filtering blocked domains."""
    urls: List[str] = []
    for item in serper_response.get("organic", []):
        link = item.get("link", "")
        if not link:
            continue
        if any(blocked in link for blocked in BLOCKED_DOMAINS):
            continue
        urls.append(link)

    for paa in serper_response.get("peopleAlsoAsk", []):
        link = paa.get("link", "")
        if not link:
            continue
        if any(blocked in link for blocked in BLOCKED_DOMAINS):
            continue
        if link not in urls:
            urls.append(link)

    return urls[:max_urls]


# ---------------------------------------------------------------------------
# Step 3 — LLM query generation
# ---------------------------------------------------------------------------


def _openai_chat(
    messages: List[Dict[str, str]],
    model: str,
    api_key: str,
    temperature: float = 0,
    max_tokens: int = 600,
) -> str:
    """Simple OpenAI chat completion call. Returns assistant content."""
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_group_b_queries(question: str, openai_api_key: str) -> List[str]:
    """Generate factual sub-questions (Group B / factual_research style)."""
    today = date.today().strftime("%Y-%m-%d")
    messages = [
        {"role": "system", "content": REFRAME_SYSTEM},
        {
            "role": "user",
            "content": REFRAME_USER.format(question=question, today=today),
        },
    ]
    content = _openai_chat(
        messages,
        GROUP_B_MODEL,
        openai_api_key,
        temperature=0,
        max_tokens=600,
    )
    # Parse numbered list from response
    queries = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading number + punctuation: "1. ", "1) ", "- ", etc.
        cleaned = re.sub(r"^[\d]+[.)]\s*", "", line)
        cleaned = re.sub(r"^[-*]\s*", "", cleaned)
        if cleaned and len(cleaned) > 10:
            queries.append(cleaned)
    return queries[:6]


def generate_group_c_queries(
    question: str, openai_api_key: str, num_queries: int = GROUP_C_NUM_QUERIES
) -> List[str]:
    """Generate search queries (Group C / RAG-reasoning style)."""
    prompt = URL_QUERY_PROMPT.format(user_prompt=question, num_queries=num_queries)
    messages = [
        {
            "role": "system",
            "content": "You are a world class algorithm for generating structured output from a given input.",
        },
        {"role": "user", "content": prompt},
    ]
    content = _openai_chat(
        messages,
        GROUP_C_MODEL,
        openai_api_key,
        temperature=0,
        max_tokens=800,
    )
    # Parse <query>...</query> tags
    queries = re.findall(r"<query>(.*?)</query>", content, re.DOTALL)
    queries = [q.strip() for q in queries if q.strip()]
    # Also include the raw question (matches RAG tool behavior)
    queries.append(question)
    return queries


# ---------------------------------------------------------------------------
# Step 4 — Page scraping
# ---------------------------------------------------------------------------


def fetch_page_html(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch raw HTML from a URL. Returns None on failure."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return None
        html = resp.text
        if len(html.encode("utf-8", errors="replace")) > MAX_HTML_BYTES:
            html = html[:MAX_HTML_BYTES]
        return html
    except Exception as e:
        print(f"    [scrape] Failed {url}: {e}")
        return None


def extract_text_from_html(html: str, max_words: int = 300) -> Optional[str]:
    """Extract main article text from HTML using readability + markdownify."""
    try:
        cleaned = _SCRIPT_STYLE_PATTERN.sub("", html)
        cleaned = _IMG_TAG_PATTERN.sub("", cleaned)
        article_html = ReadabilityDocument(cleaned).summary()
        text = md(article_html, heading_style="ATX", strip=["img", "figure"])
        if not text or not text.strip():
            return None
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + " [...]"
        return text.strip()
    except Exception:
        return None


def scrape_pages_dedup(
    urls: List[str],
    html_cache: Dict[str, Optional[str]],
    max_workers: int = 6,
) -> Dict[str, str]:
    """Fetch raw HTML for URLs, skipping those already in html_cache.

    Returns only successfully fetched pages. Updates html_cache in place.
    """
    to_fetch = [u for u in urls if u not in html_cache]
    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_url = {pool.submit(fetch_page_html, url): url for url in to_fetch}
            for fut in as_completed(future_to_url):
                url = future_to_url[fut]
                try:
                    html_cache[url] = fut.result()
                except Exception as e:
                    print(f"    [scrape] Error {url}: {e}")
                    html_cache[url] = None

    return {u: html_cache[u] for u in urls if html_cache.get(u)}


# ---------------------------------------------------------------------------
# Step 5 — Per-market pipeline (parallelized)
# ---------------------------------------------------------------------------


def process_market(
    question: str,
    groups: Set[str],
    serper_api_key: str,
    openai_api_key: str,
    max_pages: int,
    delay: float,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Run all groups for a single market. Returns (group_results, stats)."""
    group_results: Dict[str, Dict[str, Any]] = {}
    html_cache: Dict[str, Optional[str]] = {}
    stats = {"serper_calls": 0, "llm_calls": 0, "pages_scraped": 0}

    # Phase 1: Fire group A Serper + groups B/C LLM calls in parallel
    group_a_result = None
    group_b_queries = None
    group_c_queries = None

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}

        if "a" in groups:
            futures["a_serper"] = pool.submit(
                search_serper, question, serper_api_key, 10
            )

        if "b" in groups:
            futures["b_llm"] = pool.submit(
                generate_group_b_queries, question, openai_api_key
            )

        if "c" in groups:
            futures["c_llm"] = pool.submit(
                generate_group_c_queries, question, openai_api_key
            )

        for key, fut in futures.items():
            result = fut.result()
            if key == "a_serper":
                group_a_result = result
                stats["serper_calls"] += 1
            elif key == "b_llm":
                group_b_queries = result
                stats["llm_calls"] += 1
            elif key == "c_llm":
                group_c_queries = result
                stats["llm_calls"] += 1

    # Finalize group A (no scraping needed)
    if group_a_result is not None:
        n_organic = len(group_a_result.get("organic", []))
        n_paa = len(group_a_result.get("peopleAlsoAsk", []))
        print(f"    Group A: {n_organic} organic, {n_paa} PAA")
        group_results["group_a"] = {
            "serper_response": group_a_result,
            "queries": [question],
        }

    # Phase 2: Batch all Serper calls from B + C queries together
    all_serper_queries: List[tuple[str, str, int]] = []  # (group, query, num_results)
    if group_b_queries is not None:
        print(f"    Group B: {len(group_b_queries)} sub-questions")
        for q in group_b_queries:
            all_serper_queries.append(("b", q, 3))
    if group_c_queries is not None:
        print(f"    Group C: {len(group_c_queries)} queries")
        for q in group_c_queries:
            all_serper_queries.append(("c", q, 5))

    if all_serper_queries:
        # Run all Serper calls with controlled concurrency
        just_queries = [q for _, q, _ in all_serper_queries]
        num_results_list = [n for _, _, n in all_serper_queries]

        serper_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=SERPER_WORKERS) as pool:
            future_to_idx = {}
            for idx, (_, q, num_r) in enumerate(all_serper_queries):
                if idx > 0:
                    time.sleep(delay)
                future_to_idx[pool.submit(search_serper, q, serper_api_key, num_r)] = idx

            indexed_results: Dict[int, Dict[str, Any]] = {}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    indexed_results[idx] = fut.result()
                except Exception as e:
                    print(f"    [serper] Query failed: {e}")
                    indexed_results[idx] = {"organic": [], "peopleAlsoAsk": []}

            serper_results = [indexed_results[i] for i in range(len(all_serper_queries))]

        stats["serper_calls"] += len(serper_results)

        # Split results back into groups B and C
        b_responses = []
        c_responses = []
        b_urls: List[str] = []
        c_urls: List[str] = []
        b_seen: Set[str] = set()
        c_seen: Set[str] = set()

        for (group, q, num_r), resp in zip(all_serper_queries, serper_results):
            if group == "b":
                b_responses.append({"query": q, "response": resp})
                for url in extract_urls_from_serper(resp, max_urls=3):
                    if url not in b_seen:
                        b_seen.add(url)
                        b_urls.append(url)
            else:
                c_responses.append({"query": q, "response": resp})
                for url in extract_urls_from_serper(resp, max_urls=5):
                    if url not in c_seen:
                        c_seen.add(url)
                        c_urls.append(url)

        b_urls = b_urls[:max_pages]
        c_urls = c_urls[:max_pages]

        # Phase 3: Scrape all unique URLs (deduped across groups)
        all_urls = list(dict.fromkeys(b_urls + c_urls))  # preserve order, dedup
        scrape_pages_dedup(all_urls, html_cache)
        stats["pages_scraped"] = len([v for v in html_cache.values() if v])

        # Build group B source_links (extracted text)
        if group_b_queries is not None:
            b_html_pages = {u: html_cache[u] for u in b_urls if html_cache.get(u)}
            b_source_links: Dict[str, str] = {}
            for url, html in b_html_pages.items():
                text = extract_text_from_html(html, max_words=400)
                if text:
                    b_source_links[url] = text
            print(f"    Group B: {len(b_html_pages)} pages, {len(b_source_links)} extracted")
            group_results["group_b"] = {
                "queries": group_b_queries,
                "serper_responses": b_responses,
                "source_links": b_source_links,
            }

        # Build group C source_links (raw HTML)
        if group_c_queries is not None:
            c_html_pages = {u: html_cache[u] for u in c_urls if html_cache.get(u)}
            print(f"    Group C: {len(c_html_pages)} pages scraped")
            group_results["group_c"] = {
                "queries": group_c_queries,
                "serper_responses": c_responses,
                "source_links": c_html_pages,
            }

    return group_results, stats


# ---------------------------------------------------------------------------
# Step 6 — Snapshot writing
# ---------------------------------------------------------------------------


def write_snapshot(
    market: Dict[str, Any],
    group_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write per-market snapshot with per-group subdirectories."""
    snapshot_id = market["id"]
    snap_dir = output_dir / "snapshots" / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat()

    metadata = {
        "question": market["question"],
        "market_address": market["market_address"],
        "platform": market["platform"],
        "outcomes": market["outcomes"],
        "current_prob": market.get("current_prob"),
        "usd_volume": market.get("usd_volume"),
        "usd_liquidity": market.get("usd_liquidity"),
        "category": market.get("category", ""),
        "snapshot_at": now_iso,
        "snapshot_origin": "contemporaneous",
        "groups": list(group_results.keys()),
    }

    (snap_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    for group_name, data in group_results.items():
        gdir = snap_dir / group_name
        gdir.mkdir(exist_ok=True)

        if "queries" in data:
            (gdir / "queries.json").write_text(
                json.dumps(data["queries"], indent=2, ensure_ascii=False)
            )
        if "serper_response" in data:
            (gdir / "serper_response.json").write_text(
                json.dumps(data["serper_response"], indent=2, ensure_ascii=False)
            )
        if "serper_responses" in data:
            (gdir / "serper_responses.json").write_text(
                json.dumps(data["serper_responses"], indent=2, ensure_ascii=False)
            )
        if "source_links" in data:
            (gdir / "source_links.json").write_text(
                json.dumps(data["source_links"], ensure_ascii=False)
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch open Omen markets and snapshot web data per tool group."
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=500,
        help="Maximum markets to fetch (default: 500)",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=0.0,
        help="Minimum USD liquidity to include (default: 0)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Max pages to scrape per group per market (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/datasets",
        help="Output directory (default: benchmark/datasets)",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="a,b,c",
        help="Comma-separated groups to run: a,b,c (default: a,b,c)",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Fetch markets only, no web search/scraping",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch markets and print stats, don't write files",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between Serper calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="omen",
        choices=["omen", "polymarket", "all"],
        help="Platform to fetch markets from (default: omen)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=POLYMARKET_WINDOW_DAYS,
        help=f"Polymarket: markets closing within N days (default: {POLYMARKET_WINDOW_DAYS})",
    )
    args = parser.parse_args()

    groups = {g.strip().lower() for g in args.groups.split(",")}
    needs_llm = bool(groups & {"b", "c"})

    serper_api_key = os.environ.get("SERPER_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    if not args.skip_search and not args.dry_run:
        if not serper_api_key:
            parser.error(
                "SERPER_API_KEY is required unless --skip-search or --dry-run."
            )
        if needs_llm and not openai_api_key:
            parser.error(
                "OPENAI_API_KEY is required for groups B/C. "
                "Use --groups a to skip LLM query generation."
            )

    output_dir = Path(args.output_dir)

    # ----- Fetch markets -----
    markets: List[Dict[str, Any]] = []

    if args.platform in ("omen", "all"):
        print("Fetching open markets from Omen subgraph...")
        omen_markets = fetch_open_markets(max_markets=args.max_markets)
        print(f"  Omen: {len(omen_markets)} open binary markets")
        markets.extend(omen_markets)

    if args.platform in ("polymarket", "all"):
        print("Fetching open markets from Polymarket Gamma API...")
        poly_markets = fetch_open_polymarket(
            max_markets=args.max_markets, window_days=args.window_days
        )
        print(f"  Polymarket: {len(poly_markets)} open binary markets")
        markets.extend(poly_markets)

    print(f"Total: {len(markets)} markets.")

    if args.min_liquidity > 0:
        before = len(markets)
        markets = [m for m in markets if m["usd_liquidity"] >= args.min_liquidity]
        print(
            f"Filtered to {len(markets)} markets "
            f"(dropped {before - len(markets)} below "
            f"${args.min_liquidity:.2f} liquidity)"
        )
    print()

    if not markets:
        print("No markets found. Exiting.")
        return

    # Summary stats
    volumes = [m["usd_volume"] for m in markets]
    liquidity = [m["usd_liquidity"] for m in markets]
    with_prob = [m for m in markets if m["current_prob"] is not None]
    categories: Dict[str, int] = {}
    for m in markets:
        cat = m.get("category") or "unknown"
        categories[cat] = categories.get(cat, 0) + 1

    print(f"  USD volume range: {min(volumes):.2f} – {max(volumes):.2f}")
    print(f"  USD liquidity range: {min(liquidity):.2f} – {max(liquidity):.2f}")
    print(f"  Markets with prob data: {len(with_prob)}/{len(markets)}")
    print(f"  Categories: {categories}")
    print(f"  Groups to run: {sorted(groups)}")
    print(f"\n  Sample questions:")
    for m in markets[:5]:
        prob_str = f" (p={m['current_prob']:.2f})" if m["current_prob"] else ""
        print(f"    - {m['question'][:90]}{prob_str}")

    if args.dry_run:
        print("\n--dry-run set. Exiting without writing files.")
        return

    # ----- Write markets JSONL + snapshots -----
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "snapshots").mkdir(exist_ok=True)

    jsonl_path = output_dir / "open_markets.jsonl"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Load existing JSONL entries (keyed by market id) to preserve progress
    existing_rows: Dict[str, str] = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text().strip().split("\n"):
            if line:
                try:
                    row = json.loads(line)
                    existing_rows[row["id"]] = line
                except (json.JSONDecodeError, KeyError):
                    pass

    totals = {"skipped": 0, "errors": 0, "serper_calls": 0, "llm_calls": 0, "pages_scraped": 0}

    for i, market in enumerate(markets, 1):
        snapshot_id = market["id"]
        snap_dir = output_dir / "snapshots" / snapshot_id

        # Skip if snapshot already exists with all requested groups
        if (snap_dir / "metadata.json").exists():
            try:
                existing = json.loads((snap_dir / "metadata.json").read_text())
                existing_groups = set(existing.get("groups", []))
                requested_groups = {f"group_{g}" for g in groups}
                if requested_groups <= existing_groups:
                    print(
                        f"[{i}/{len(markets)}] Skipping {snapshot_id} "
                        f"(snapshot exists with groups {existing_groups})"
                    )
                    totals["skipped"] += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                pass  # re-fetch if metadata is corrupt

        print(f"[{i}/{len(markets)}] {market['question'][:80]}...")

        if not args.skip_search:
            try:
                group_results, stats = process_market(
                    market["question"],
                    groups,
                    serper_api_key,
                    openai_api_key,
                    args.max_pages,
                    args.delay,
                )
                for k in ("serper_calls", "llm_calls", "pages_scraped"):
                    totals[k] += stats[k]
            except SerperRateLimitError:
                print("  FATAL: Serper quota exhausted. Stopping.")
                break
            except Exception as e:
                print(f"  ERROR: {e}")
                totals["errors"] += 1
                continue  # skip snapshot — don't write empty metadata
        else:
            group_results = {}

        # Only write snapshot if we have at least one group result
        if group_results or args.skip_search:
            write_snapshot(market, group_results, output_dir)

        # Update JSONL row and flush immediately
        row = {**market, "fetched_at": now_iso, "snapshot_id": snapshot_id}
        existing_rows[market["id"]] = json.dumps(row, ensure_ascii=False)
        jsonl_path.write_text("\n".join(existing_rows.values()) + "\n")

    # ----- Summary -----
    print(f"\nDone.")
    print(
        f"  Markets: {len(markets)} ({totals['skipped']} skipped, {totals['errors']} errors)"
    )
    print(f"  Serper calls: {totals['serper_calls']}")
    print(f"  LLM calls: {totals['llm_calls']}")
    print(f"  Pages scraped: {totals['pages_scraped']} (deduplicated)")
    print(f"  Output: {jsonl_path}")
    print(f"  Snapshots: {output_dir / 'snapshots'}/")


if __name__ == "__main__":
    main()
