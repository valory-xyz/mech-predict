"""HTTP client for the mech-analytics scored-rows endpoint.

Post off-chain migration, mech requests stop emitting per-request events
on the marketplace subgraph and their payloads no longer land on IPFS.
The benchmark daily report's row-fetch layer (subgraph + IPFS pull in
``fetch_production.py``) can't reach either after the switch. mech-analytics
is the new public read path: it ingests from the predict-api data lake,
scores each row once, and serves the results over HTTP.

This module pages ``/v1/data/scored-rows`` for a report window and yields
rows in the shape ``grouping.accumulate_row`` already expects — so the
substrate refactor is confined to how rows arrive; the accumulator and
everything downstream (``analyze.py``, Slack rendering) is untouched.

Feature-flagged. If ``USE_MECH_ANALYTICS_ROWS`` is unset or false, the
current subgraph + IPFS path stays live and this module is dormant.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import requests

log = logging.getLogger(__name__)


DEFAULT_PAGE_SIZE = 5000  # matches the endpoint's DEFAULT_LIMIT
DEFAULT_TIMEOUT_S = 60.0
# Runaway guard on the pagination loop. 2000 pages at 5000 rows/page is
# ~10M rows, which comfortably covers a 7d report window at peak Omen
# volume while catching a cursor cycle within minutes rather than the
# ~1h a 10_000-page cap would allow. Trader's client caps lower (200)
# because it pages one Safe; the benchmark pulls the whole window.
DEFAULT_MAX_PAGES = 2_000


class MechAnalyticsError(RuntimeError):
    """Raised when the endpoint response is unusable."""


# Production URL for mech-analytics. Hardcoded rather than read from an
# env var so a merged consumer PR is the whole flip — no separate repo
# secret change needed to make the switch land. If the endpoint moves,
# update this constant in a follow-up PR.
MECH_ANALYTICS_URL = "https://mech-analytics-api.autonolas.tech"


def _base_url() -> str:
    return MECH_ANALYTICS_URL


def _to_iso_z(dt: datetime) -> str:
    """ISO 8601 with a trailing Z suffix, matching the endpoint's format."""
    if dt.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _map_row(api_row: dict[str, Any]) -> dict[str, Any]:
    """Map a scored-rows response entry onto the keys ``accumulate_row`` reads.

    Field alignment worth calling out:

    - ``tool`` (endpoint) → ``tool_name`` (accumulator).
    - ``question_title`` → ``question_text``; ``accumulate_row`` also reads it
      to key worst/best-10 lists, and the caller applies ``classify_category``
      on the same value to fill the ``category`` grouping.
    - ``resolved_outcome`` (0.0 / 1.0 / None) → ``final_outcome``; the accumulator
      treats None as unresolved.
    - ``market_liquidity_usd`` → ``market_liquidity_at_prediction`` (accumulator's
      difficulty/liquidity classifiers read the ``_at_prediction`` name).
    - ``latency_s`` is derived from ``delivered_at - requested_at`` — the
      endpoint doesn't materialise it as a column today.
    - Extra pass-through fields (``market_id``, ``market_spread_at_prediction``,
      ``resolved_at``, ``brier``, ``log_loss``, ``edge``, ``directional_correct``)
      are not read by ``accumulate_row`` but are surfaced for the parity
      script (``verify_migration_swap.py``) and any downstream that wants
      the endpoint's pre-computed per-row scores.

    Fields not present on the endpoint stay ``None`` (or default in
    ``accumulate_row``) so the report keeps working while grouping-dimension
    decisions land: ``mode``, ``config_hash``, ``prediction_lead_time_days``.

    :param api_row: one entry from the endpoint's ``rows`` array.
    :return: a dict shaped like the row ``accumulate_row`` expects.
    """
    delivered_at = _parse_iso(api_row.get("delivered_at"))
    requested_at = _parse_iso(api_row.get("requested_at"))
    latency_s: float | None = None
    if delivered_at is not None and requested_at is not None:
        latency_s = (delivered_at - requested_at).total_seconds()
        if latency_s < 0:
            # ``requested_at`` and ``delivered_at`` land through separate
            # lake write paths, so a negative delta means clock skew or
            # an ingest bug. Warn on the row so the source shows up in
            # logs instead of silently thinning the latency stats.
            log.warning(
                "mech-analytics row %s has negative latency "
                "(requested_at=%s > delivered_at=%s); dropping value",
                api_row.get("request_id"),
                api_row.get("requested_at"),
                api_row.get("delivered_at"),
            )
            latency_s = None

    # Re-validate probability ranges: the local producer (fetch_production)
    # only tags rows "valid" after range-checking, and every downstream
    # consumer relies on that invariant without re-checking. If the
    # externally-versioned endpoint ever ships an out-of-range value on
    # a "valid" row, demote to "malformed" so accumulators skip it.
    p_yes, p_yes_ok = _validated_probability(api_row.get("p_yes"))
    p_no, p_no_ok = _validated_probability(api_row.get("p_no"))
    # Note: market_prob only feeds the edge-over-market metrics, which
    # already skip None gracefully via a `market_prob is not None`
    # guard in _accumulate_group. Do NOT let a bad market_prob demote
    # prediction_parse_status — that would remove the row from valid_n /
    # Brier / log-loss / directional accuracy / sharpness (the headline
    # metrics), even when p_yes/p_no are fine. Just null the field and
    # let the edge-metric guard handle it downstream.
    market_prob, _market_prob_ok = _validated_probability(
        api_row.get("market_prob_at_prediction")
    )
    parse_status = api_row.get("prediction_parse_status")
    if parse_status == "valid" and not (p_yes_ok and p_no_ok):
        log.warning(
            "mech-analytics row %s tagged 'valid' but has out-of-range "
            "p_yes=%r p_no=%r; demoting to 'malformed'",
            api_row.get("request_id"),
            api_row.get("p_yes"),
            api_row.get("p_no"),
        )
        parse_status = "malformed"

    return {
        # Identity / grouping keys.
        "request_id": api_row.get("request_id"),
        "tool_name": api_row.get("tool"),
        "tool_version": api_row.get("tool_version"),
        "platform": api_row.get("platform"),
        "question_text": api_row.get("question_title"),
        # Prediction fields.
        "p_yes": p_yes,
        "p_no": p_no,
        "confidence": api_row.get("confidence"),
        "prediction_parse_status": parse_status,
        # Market context.
        "market_prob_at_prediction": market_prob,
        "market_liquidity_at_prediction": api_row.get("market_liquidity_usd"),
        "market_spread_at_prediction": api_row.get("market_spread_at_prediction"),
        "market_id": api_row.get("market_id"),
        # Resolution outcome. ``accumulate_row`` expects final_outcome ∈ {0, 1, None}.
        "final_outcome": _outcome_bool(api_row.get("resolved_outcome")),
        "resolved_at": api_row.get("resolved_at"),
        # Pre-computed per-row scores from mech-analytics. Not read by
        # accumulate_row (which recomputes from p_yes + final_outcome),
        # but the parity script diffs them to catch scoring drift.
        "brier": api_row.get("brier"),
        "log_loss": api_row.get("log_loss"),
        "edge": api_row.get("edge"),
        "directional_correct": api_row.get("directional_correct"),
        # Derived / passthrough fields the accumulator reads.
        "latency_s": latency_s,
        "requested_at": api_row.get("requested_at"),
        "delivered_at": api_row.get("delivered_at"),
        # ``predicted_at`` alias so consumers that bucket rows by
        # prediction time (``rebuild_from_mech_analytics`` month
        # attribution) work against mech-analytics rows without an
        # extra mapping layer. Delivery time matches how
        # ``fetch_production`` stamps ``predicted_at`` on legacy log
        # rows; fall back to ``requested_at`` when delivery is missing.
        "predicted_at": api_row.get("delivered_at") or api_row.get("requested_at"),
        # Fields the endpoint doesn't carry today — left None so the
        # accumulator uses its own defaults ("unknown" / "production_replay").
        # See PR #11 on mech-analytics for the grouping-dimension sign-off.
        "mode": None,
        "config_hash": None,
        "prediction_lead_time_days": None,
    }


def _validated_probability(value: Any) -> tuple[float | None, bool]:
    """Coerce a probability to float in [0, 1]; report whether coercion succeeded.

    Returns ``(value, ok)``. ``ok`` is False when the input was neither
    None nor a finite float in [0, 1]. Callers use ``ok`` to decide
    whether to demote ``prediction_parse_status`` on the row.

    :param value: raw p_yes / p_no / market_prob_at_prediction from the endpoint.
    :return: coerced float (or None if input was None) and an ok flag.
    """
    if value is None:
        return None, True
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None, False
    if math.isnan(as_float):
        return None, False
    if not 0.0 <= as_float <= 1.0:
        return None, False
    return as_float, True


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

    Rows come back in the shape ``accumulate_row`` expects (see ``_map_row``).
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
    :raises MechAnalyticsError: if the endpoint is unreachable
        or the paginator hits ``max_pages``.
    """
    base = _base_url()

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
    cursor: str | None = None
    # Context-managed session so a caller that abandons iteration (break
    # / exception in the consuming loop) doesn't leak the connection
    # pool. Retry adapter absorbs transient 5xx / connection resets so
    # a single blip on page N out of thousands doesn't abort the pull.
    from requests.adapters import HTTPAdapter  # pylint: disable=import-outside-toplevel
    from urllib3.util.retry import Retry  # pylint: disable=import-outside-toplevel

    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(502, 503, 504),
        allowed_methods=("GET",),
    )
    with requests.Session() as session:
        session.mount(base, HTTPAdapter(max_retries=retry))
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

            if not isinstance(payload, dict):
                raise MechAnalyticsError(
                    f"mech-analytics scored-rows page {pages}: "
                    f"payload is {type(payload).__name__}, expected dict"
                )
            if "rows" not in payload:
                raise MechAnalyticsError(
                    f"mech-analytics scored-rows page {pages}: "
                    f"payload missing 'rows' key (got keys: {sorted(payload)})"
                )
            rows = payload["rows"]
            if not isinstance(rows, list):
                raise MechAnalyticsError(
                    f"mech-analytics scored-rows page {pages}: "
                    f"'rows' has unexpected type {type(rows).__name__}"
                )
            for api_row in rows:
                yield _map_row(api_row)
            total_rows += len(rows)

            cursor = payload.get("next_cursor")
            if not cursor:
                break
            # Replace since with a cursor param on the next request.
            params.pop("since", None)
            params["cursor"] = cursor
        else:
            if cursor:
                raise MechAnalyticsError(
                    f"mech-analytics scored-rows paginator hit max_pages={max_pages} "
                    f"with cursor still pending; likely a cursor cycle or "
                    f"unexpectedly large window (fetched {total_rows} rows so far)"
                )

    log.info(
        "mech-analytics: fetched %d rows across %d page(s) since=%s",
        total_rows,
        pages,
        _to_iso_z(since),
    )


def iter_unscored_row_ids(
    since: datetime,
    until: Optional[datetime] = None,
    *,
    chain_id: Optional[int] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> Iterator[str]:
    """Page through ``/v1/data/unscored-rows`` and yield ``request_id`` values.

    Shell rows (predict-api's ``resolution_status='unparseable'`` writes)
    are partitioned onto ``/v1/data/unscored-rows`` — the scored endpoint
    excludes them because every score column is NULL. The coverage check
    needs the union of both endpoints so a shell row present in the lake
    is not reported as "missing from lake". This helper is intentionally
    request-id-only: the coverage check keys on request_id set membership
    and never reads the score columns from an unscored row.

    :param since: timezone-aware datetime for the ``since`` filter (inclusive).
    :param until: optional timezone-aware datetime for the ``until`` filter
        (exclusive; endpoint semantics).
    :param chain_id: optional chain filter (100 for Gnosis, 137 for Polygon).
    :param page_size: batch size per HTTP call.
    :param timeout_s: per-request timeout in seconds.
    :param max_pages: runaway guard.
    :yield: one request_id per yielded value.
    :raises MechAnalyticsError: if the endpoint is unreachable
        or the paginator hits ``max_pages``.
    """
    base = _base_url()

    params: dict[str, Any] = {
        "since": _to_iso_z(since),
        "limit": page_size,
    }
    if until is not None:
        params["until"] = _to_iso_z(until)
    if chain_id is not None:
        params["chain_id"] = chain_id

    url = f"{base}/v1/data/unscored-rows"
    pages = 0
    total_rows = 0
    total_yielded = 0
    total_skipped_falsy_rid = 0
    cursor: str | None = None
    from requests.adapters import HTTPAdapter  # pylint: disable=import-outside-toplevel
    from urllib3.util.retry import Retry  # pylint: disable=import-outside-toplevel

    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(502, 503, 504),
        allowed_methods=("GET",),
    )
    with requests.Session() as session:
        session.mount(base, HTTPAdapter(max_retries=retry))
        while pages < max_pages:
            pages += 1
            try:
                resp = session.get(url, params=params, timeout=timeout_s)
                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException as exc:
                raise MechAnalyticsError(
                    f"mech-analytics unscored-rows fetch failed (page {pages}): {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise MechAnalyticsError(
                    f"mech-analytics unscored-rows page {pages}: "
                    f"payload is {type(payload).__name__}, expected dict"
                )
            if "rows" not in payload:
                raise MechAnalyticsError(
                    f"mech-analytics unscored-rows page {pages}: "
                    f"payload missing 'rows' key (got keys: {sorted(payload)})"
                )
            rows = payload["rows"]
            if not isinstance(rows, list):
                raise MechAnalyticsError(
                    f"mech-analytics unscored-rows page {pages}: "
                    f"'rows' has unexpected type {type(rows).__name__}"
                )
            for api_row in rows:
                rid = api_row.get("request_id")
                if rid:
                    yield rid
                    total_yielded += 1
                else:
                    total_skipped_falsy_rid += 1
            total_rows += len(rows)

            cursor = payload.get("next_cursor")
            if not cursor:
                break
            params.pop("since", None)
            params["cursor"] = cursor
        else:
            if cursor:
                raise MechAnalyticsError(
                    f"mech-analytics unscored-rows paginator hit max_pages={max_pages} "
                    f"with cursor still pending; likely a cursor cycle or "
                    f"unexpectedly large window (fetched {total_rows} rows so far)"
                )

    if total_skipped_falsy_rid:
        # Mirrors the negative-latency warning in ``_map_row``: an unscored
        # row without a request_id is a schema regression upstream (the
        # endpoint's contract is that request_id is the PK) or an ingest
        # bug on the lake side. Yielding it would silently thin the
        # coverage union, so we skip it — but the schema-regression
        # signal has to reach the operator, not just die in the loop.
        log.warning(
            "mech-analytics unscored-rows: skipped %d row(s) with a falsy "
            "request_id — likely a schema regression on the endpoint or an "
            "ingest bug on the lake",
            total_skipped_falsy_rid,
        )

    log.info(
        "mech-analytics: fetched %d unscored row(s) across %d page(s) since=%s "
        "(yielded %d request_id(s))",
        total_rows,
        pages,
        _to_iso_z(since),
        total_yielded,
    )
