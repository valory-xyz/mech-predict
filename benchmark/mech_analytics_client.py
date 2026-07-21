"""HTTP client for the mech-analytics scored-rows endpoint.

Post off-chain migration, mech requests stop emitting per-request events
on the marketplace subgraph and their payloads no longer land on IPFS.
The benchmark daily report's row-fetch layer (subgraph + IPFS pull in
``fetch_production.py``) can't reach either after the switch. mech-analytics
is the new public read path: it ingests from the predict-api data lake,
scores each row once, and serves the results over HTTP.

This module pages ``/v1/data/scored-rows`` for a report window and yields
rows in the shape ``scorer._accumulate_row`` already expects — so the
substrate refactor is confined to how rows arrive; the accumulator and
everything downstream (``analyze.py``, Slack rendering) is untouched.

Feature-flagged. If ``USE_MECH_ANALYTICS_ROWS`` is unset or false, the
current subgraph + IPFS path stays live and this module is dormant.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import requests

log = logging.getLogger(__name__)


DEFAULT_PAGE_SIZE = 5000  # matches the endpoint's DEFAULT_LIMIT
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_PAGES = 10_000  # runaway guard (~50M rows at 5k/page)


class MechAnalyticsError(RuntimeError):
    """Raised when the endpoint response is unusable."""


def _base_url() -> str:
    url = os.getenv("MECH_ANALYTICS_URL")
    if not url:
        raise MechAnalyticsError(
            "MECH_ANALYTICS_URL is not set; cannot fetch rows from mech-analytics"
        )
    return url.rstrip("/")


def _to_iso_z(dt: datetime) -> str:
    """ISO 8601 with a trailing Z suffix, matching the endpoint's format."""
    if dt.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _map_row(api_row: dict[str, Any]) -> dict[str, Any]:
    """Map a scored-rows response entry onto the keys ``_accumulate_row`` reads.

    Field alignment worth calling out:

    - ``tool`` (endpoint) → ``tool_name`` (accumulator).
    - ``question_title`` → ``question_text``; ``_accumulate_row`` also reads it
      to key worst/best-10 lists, and the caller applies ``classify_category``
      on the same value to fill the ``category`` grouping.
    - ``resolved_outcome`` (0.0 / 1.0 / None) → ``final_outcome``; the accumulator
      treats None as unresolved.
    - ``market_liquidity_usd`` → ``market_liquidity_at_prediction`` (accumulator's
      difficulty/liquidity classifiers read the ``_at_prediction`` name).
    - ``latency_s`` is derived from ``delivered_at - requested_at`` — the
      endpoint doesn't materialise it as a column today.

    Fields not present on the endpoint stay ``None`` (or default in
    ``_accumulate_row``) so the report keeps working while grouping-dimension
    decisions land: ``mode``, ``config_hash``, ``prediction_lead_time_days``.

    :param api_row: one entry from the endpoint's ``rows`` array.
    :return: a dict shaped like the row ``_accumulate_row`` expects.
    """
    delivered_at = _parse_iso(api_row.get("delivered_at"))
    requested_at = _parse_iso(api_row.get("requested_at"))
    latency_s: float | None = None
    if delivered_at is not None and requested_at is not None:
        latency_s = (delivered_at - requested_at).total_seconds()
        if latency_s < 0:
            latency_s = None

    return {
        # Identity / grouping keys.
        "request_id": api_row.get("request_id"),
        "tool_name": api_row.get("tool"),
        "tool_version": api_row.get("tool_version"),
        "platform": api_row.get("platform"),
        "question_text": api_row.get("question_title"),
        # Prediction fields.
        "p_yes": api_row.get("p_yes"),
        "p_no": api_row.get("p_no"),
        "confidence": api_row.get("confidence"),
        "prediction_parse_status": api_row.get("prediction_parse_status"),
        # Market context.
        "market_prob_at_prediction": api_row.get("market_prob_at_prediction"),
        "market_liquidity_at_prediction": api_row.get("market_liquidity_usd"),
        # Resolution outcome. ``_accumulate_row`` expects final_outcome ∈ {0, 1, None}.
        "final_outcome": _outcome_bool(api_row.get("resolved_outcome")),
        # Derived / passthrough fields the accumulator reads.
        "latency_s": latency_s,
        "requested_at": api_row.get("requested_at"),
        "delivered_at": api_row.get("delivered_at"),
        # Fields the endpoint doesn't carry today — left None so the
        # accumulator uses its own defaults ("unknown" / "production_replay").
        # See PR #11 on mech-analytics for the grouping-dimension sign-off.
        "mode": None,
        "config_hash": None,
        "prediction_lead_time_days": None,
    }


def _outcome_bool(resolved_outcome: Any) -> bool | None:
    """Map the endpoint's numeric outcome to the True/False/None the scorer uses."""
    if resolved_outcome is None:
        return None
    try:
        return bool(int(round(float(resolved_outcome))))
    except (TypeError, ValueError):
        return None


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        # ``fromisoformat`` accepts "Z" via a trailing offset only from 3.11;
        # normalise defensively for compatibility with the endpoint's format.
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def iter_scored_rows(
    since: datetime,
    until: Optional[datetime] = None,
    *,
    platform: Optional[str] = None,
    chain_id: Optional[int] = None,
    resolved: Optional[bool] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> Iterator[dict[str, Any]]:
    """Page through ``/v1/data/scored-rows`` and yield rows one at a time.

    Rows come back in the shape ``_accumulate_row`` expects (see ``_map_row``).
    Pagination uses the endpoint's opaque keyset cursor.

    :param since: timezone-aware datetime for the ``since`` filter (inclusive).
    :param until: optional timezone-aware datetime for the ``until`` filter
        (exclusive; endpoint semantics).
    :param platform: optional platform filter ("omen" | "polymarket").
    :param chain_id: optional chain filter (100 for Gnosis, 137 for Polygon).
    :param resolved: optional filter to include only resolved / unresolved rows.
    :param page_size: batch size per HTTP call.
    :param timeout_s: per-request timeout in seconds.
    :param max_pages: runaway guard.
    :yield: one accumulator-shaped row per yielded value.
    :raises MechAnalyticsError: if the endpoint is unreachable, the config
        (``MECH_ANALYTICS_URL``) is missing, or the paginator hits ``max_pages``.
    """
    base = _base_url()
    session = requests.Session()

    params: dict[str, Any] = {
        "since": _to_iso_z(since),
        "limit": page_size,
    }
    if until is not None:
        params["until"] = _to_iso_z(until)
    if platform is not None:
        params["platform"] = platform
    if chain_id is not None:
        params["chain_id"] = chain_id
    if resolved is not None:
        params["resolved"] = "true" if resolved else "false"

    url = f"{base}/v1/data/scored-rows"
    pages = 0
    total_rows = 0
    while pages < max_pages:
        pages += 1
        try:
            resp = session.get(url, params=params, timeout=timeout_s)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            raise MechAnalyticsError(
                f"mech-analytics scored-rows fetch failed (page {pages}): {exc}"
            ) from exc

        rows = payload.get("rows") or []
        for api_row in rows:
            yield _map_row(api_row)
        total_rows += len(rows)

        cursor = payload.get("next_cursor")
        if not cursor:
            break
        # Replace since with a cursor param on the next request.
        params.pop("since", None)
        params["cursor"] = cursor

    log.info(
        "mech-analytics: fetched %d rows across %d page(s) since=%s",
        total_rows,
        pages,
        _to_iso_z(since),
    )
