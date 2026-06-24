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
"""Tests for benchmark/datasets/subgraph.py — shared GraphQL retry helper."""

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import requests
from benchmark.datasets import subgraph

# Non-resolvable per RFC 2606: requests.post is always mocked here, so this URL
# is never contacted — and if a mock ever broke, it fails harmlessly instead of
# hitting a real subgraph.
_URL = "https://subgraph.invalid/graphql"


def _http_response(
    status_code: int, json_body: Optional[dict[str, Any]] = None
) -> MagicMock:
    """Build a fake ``requests`` response with the given status.

    ``raise_for_status`` raises an :class:`HTTPError` carrying the response for
    any 4xx/5xx, mirroring real ``requests`` behaviour; otherwise ``json``
    returns ``json_body``.

    :param status_code: HTTP status to simulate.
    :param json_body: decoded JSON body for a successful response.
    :return: the fake response.
    """
    resp = MagicMock()
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
        resp.json.return_value = json_body if json_body is not None else {"data": {}}
    return resp


def _silence_sleep(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the backoff sleep with a recorder so tests don't actually wait.

    :param monkeypatch: pytest patching fixture.
    :return: list that accumulates each slept duration.
    """
    slept: list[int] = []
    monkeypatch.setattr(subgraph.time, "sleep", slept.append)
    return slept


class TestPostGraphqlHttpRetries:
    """Transient upstream HTTP/network failures retry; client errors don't."""

    def test_retries_then_succeeds_on_transient_503(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 503 then a 200 yields the data after one backoff sleep."""
        slept = _silence_sleep(monkeypatch)
        responses = [_http_response(503), _http_response(200, {"data": {"ok": 1}})]
        monkeypatch.setattr(subgraph.requests, "post", lambda *a, **k: responses.pop(0))

        data = subgraph.post_graphql(_URL, {"query": "{ x }"})

        assert data == {"ok": 1}
        assert slept == [subgraph.RETRY_BACKOFF_SECONDS]  # one retry, linear backoff

    def test_connection_error_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ConnectionError on the first attempt is retried, then succeeds."""
        slept = _silence_sleep(monkeypatch)
        calls: list[int] = []

        def _post(*_a: Any, **_k: Any) -> MagicMock:
            calls.append(1)
            if len(calls) == 1:
                raise requests.exceptions.ConnectionError("boom")
            return _http_response(200, {"data": {"ok": 2}})

        monkeypatch.setattr(subgraph.requests, "post", _post)

        assert subgraph.post_graphql(_URL, {"query": "{ x }"}) == {"ok": 2}
        assert len(slept) == 1

    @pytest.mark.parametrize(
        ("status", "retried"),
        [
            (503, True),  # service unavailable — the failure that crashed CI
            (429, True),  # rate limited
            (502, True),  # bad gateway
            (404, False),  # client error — won't recover on retry
            (400, False),  # bad request — won't recover on retry
        ],
        ids=["503", "429", "502", "404", "400"],
    )
    def test_status_code_retry_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        retried: bool,
    ) -> None:
        """Only RETRYABLE_STATUS_CODES are retried; others raise at once."""
        slept = _silence_sleep(monkeypatch)
        attempts: list[int] = []

        def _post(*_a: Any, **_k: Any) -> MagicMock:
            attempts.append(1)
            return _http_response(status)

        monkeypatch.setattr(subgraph.requests, "post", _post)

        with pytest.raises(requests.exceptions.HTTPError):
            subgraph.post_graphql(_URL, {"query": "{ x }"})

        if retried:
            assert len(attempts) == subgraph.MAX_RETRIES
            assert len(slept) == subgraph.MAX_RETRIES - 1
        else:
            assert len(attempts) == 1
            assert not slept


class TestPostGraphqlErrors:
    """GraphQL-level ``errors`` only retry when the predicate opts in."""

    def test_graphql_error_raises_without_predicate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A GraphQL error with no retry predicate raises immediately."""
        slept = _silence_sleep(monkeypatch)
        monkeypatch.setattr(
            subgraph.requests,
            "post",
            lambda *a, **k: _http_response(200, {"errors": ["bad query"]}),
        )

        with pytest.raises(RuntimeError, match="GraphQL errors"):
            subgraph.post_graphql(_URL, {"query": "{ x }"})
        assert not slept

    def test_reorg_error_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reorg GraphQL error is retried when the predicate matches."""
        slept = _silence_sleep(monkeypatch)
        responses = [
            _http_response(200, {"errors": ["block reorganized"]}),
            _http_response(200, {"data": {"ok": 3}}),
        ]
        monkeypatch.setattr(subgraph.requests, "post", lambda *a, **k: responses.pop(0))

        data = subgraph.post_graphql(
            _URL,
            {"query": "{ x }"},
            should_retry_graphql_error=lambda errs: "reorganized" in str(errs),
        )

        assert data == {"ok": 3}
        assert len(slept) == 1

    def test_non_matching_graphql_error_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A GraphQL error the predicate rejects raises without retry."""
        slept = _silence_sleep(monkeypatch)
        monkeypatch.setattr(
            subgraph.requests,
            "post",
            lambda *a, **k: _http_response(200, {"errors": ["syntax error"]}),
        )

        with pytest.raises(RuntimeError, match="GraphQL errors"):
            subgraph.post_graphql(
                _URL,
                {"query": "{ x }"},
                should_retry_graphql_error=lambda errs: "reorganized" in str(errs),
            )
        assert not slept
