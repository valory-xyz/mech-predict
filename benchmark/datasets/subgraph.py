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
"""Shared subgraph GraphQL POST helper with transient-failure retry.

The benchmark dataset fetchers (production, open-market, replay) all talk to
the same Autonolas/Omen/Polymarket subgraphs, which intermittently return
transient 5xx/429 responses under load or maintenance. This module centralises
the retry policy so every fetcher rides out those blips identically instead of
crashing the whole run on a single hiccup.
"""

import logging
import time
from typing import Any, Callable, Optional

import requests

log = logging.getLogger(__name__)

DEFAULT_HTTP_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10
# Transient upstream statuses worth retrying: the subgraph proxy returns 5xx
# under load/maintenance and 429 when rate-limited. Other 4xx (e.g. 400/404)
# are client errors that won't recover on retry, so they propagate immediately.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Predicate over a GraphQL ``errors`` payload deciding whether it is transient
# and worth retrying (e.g. a chain reorganisation).
ShouldRetryGraphqlError = Callable[[Any], bool]


def _is_retryable_http_error(exc: requests.exceptions.HTTPError) -> bool:
    """Return whether an HTTP error is a transient status worth retrying.

    :param exc: the raised HTTP error carrying the response.
    :return: True if the response status is in ``RETRYABLE_STATUS_CODES``.
    """
    response = exc.response
    return response is not None and response.status_code in RETRYABLE_STATUS_CODES


def _backoff_before_retry(url: str, attempt: int, reason: str) -> None:
    """Log a transient failure and sleep with linear backoff before retrying.

    :param url: subgraph endpoint that failed.
    :param attempt: 1-based attempt number that just failed.
    :param reason: short description of the failure (e.g. ``HTTP 503``).
    """
    wait = attempt * RETRY_BACKOFF_SECONDS
    log.warning(
        "%s on %s (attempt %d/%d), retrying in %ds",
        reason,
        url,
        attempt,
        MAX_RETRIES,
        wait,
    )
    time.sleep(wait)


def post_graphql(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int = DEFAULT_HTTP_TIMEOUT,
    should_retry_graphql_error: Optional[ShouldRetryGraphqlError] = None,
) -> dict[str, Any]:
    """POST a GraphQL query and return the response ``data``, retrying transients.

    Retries on read timeouts, connection errors, and transient upstream HTTP
    statuses (see ``RETRYABLE_STATUS_CODES``) with linear backoff. GraphQL-level
    errors raise immediately unless ``should_retry_graphql_error`` returns True
    for them (used by the replay fetcher to ride out chain reorganisations).
    Non-retryable HTTP errors (e.g. 400/404) propagate at once.

    :param url: subgraph endpoint URL.
    :param payload: GraphQL request body (``{"query": ...}``).
    :param timeout: per-request timeout in seconds.
    :param should_retry_graphql_error: optional predicate over the GraphQL
        ``errors`` payload; when it returns True the request is retried.
    :return: the ``data`` object from the GraphQL response.
    :raises RuntimeError: on a non-retryable GraphQL-level error.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            if "errors" in body:
                errors = body["errors"]
                if (
                    should_retry_graphql_error is not None
                    and attempt < MAX_RETRIES
                    and should_retry_graphql_error(errors)
                ):
                    _backoff_before_retry(url, attempt, "retryable GraphQL error")
                    continue
                raise RuntimeError(f"GraphQL errors from {url}: {errors}")
            return body.get("data", {})
        except requests.exceptions.HTTPError as exc:
            if attempt >= MAX_RETRIES or not _is_retryable_http_error(exc):
                raise
            _backoff_before_retry(url, attempt, f"HTTP {exc.response.status_code}")
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            if attempt >= MAX_RETRIES:
                raise
            _backoff_before_retry(url, attempt, type(exc).__name__)
    return {}  # unreachable, but satisfies mypy
