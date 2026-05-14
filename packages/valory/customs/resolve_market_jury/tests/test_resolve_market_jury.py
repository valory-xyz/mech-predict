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

"""Unit tests for resolve_market_jury."""

import json
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
    VoterResult,
    _build_consensus_result,
    _compute_agreement,
    _decided_votes,
    _extract_json,
    _has_consensus,
    _noop_token_counter,
    _parse_vote,
    _successful_votes,
    run,
)

MODULE = "packages.valory.customs.resolve_market_jury.resolve_market_jury"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vote(
    voter: str = "v1",
    has_occurred: bool = True,
    is_determinable: bool = True,
    is_valid: bool = True,
    confidence: float = 0.9,
    reasoning: str = "test",
    error: Optional[str] = None,
) -> VoterResult:
    """Build a VoterResult for testing."""
    return VoterResult(
        voter=voter,
        model="test-model",
        is_valid=is_valid,
        is_determinable=is_determinable,
        has_occurred=has_occurred,
        confidence=confidence,
        reasoning=reasoning,
        sources=["http://example.com"],
        error=error,
    )


def _mock_api_keys() -> MagicMock:
    """Build a mock KeyChain."""
    keys = MagicMock()
    keys.__getitem__ = MagicMock(return_value="fake-key")
    keys.max_retries.return_value = {"openai": 1, "openrouter": 1}
    keys.rotate = MagicMock()
    return keys


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    """Tests for JSON extraction from LLM responses."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ('{"a": 1}', {"a": 1}),
            ('  {"a": 1}  ', {"a": 1}),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('```\n{"a": 1}\n```', {"a": 1}),
            ('Here is the result: {"a": 1} and more text.', {"a": 1}),
            ("no json here", None),
            ("{not: valid json}", None),
            ("```json\n{bad}\n```", None),
        ],
        ids=[
            "plain_json",
            "whitespace",
            "markdown_fenced",
            "markdown_no_lang",
            "embedded_in_text",
            "no_json",
            "invalid_braces",
            "invalid_fences",
        ],
    )
    def test_extract(self, text: str, expected: Optional[dict]) -> None:
        """Extract JSON from various formats."""
        assert _extract_json(text) == expected


# ---------------------------------------------------------------------------
# _parse_vote
# ---------------------------------------------------------------------------


class TestParseVote:
    """Tests for parsing LLM text into VoterResult."""

    def test_valid_response(self) -> None:
        """Parse a well-formed voter response."""
        raw = json.dumps(
            {
                "is_valid": True,
                "is_determinable": True,
                "has_occurred": False,
                "confidence": 0.85,
                "reasoning": "Found evidence",
                "sources": ["http://a.com"],
            }
        )
        result = _parse_vote(raw, "test", "model")
        assert result.has_occurred is False
        assert result.confidence == 0.85
        assert result.error is None

    def test_unparseable(self) -> None:
        """Return error VoterResult for garbage input."""
        result = _parse_vote("garbage", "test", "model")
        assert result.error is not None
        assert "Unparseable" in result.error

    def test_missing_fields_use_defaults(self) -> None:
        """Missing fields should get defaults."""
        result = _parse_vote('{"is_valid": true}', "test", "model")
        assert result.is_valid is True
        assert result.has_occurred is None
        assert result.confidence == 0.0

    def test_indeterminable_forces_null_answer(self) -> None:
        """Voter saying is_determinable=false must have has_occurred=None."""
        raw = json.dumps(
            {
                "is_determinable": False,
                "has_occurred": True,
                "confidence": 0.9,
            }
        )
        result = _parse_vote(raw, "test", "model")
        assert result.is_determinable is False
        assert result.has_occurred is None
        assert result.confidence == 0.5

    def test_null_confidence_does_not_crash(self) -> None:
        """``confidence: null`` is tolerated (LLMs emit it for Case A).

        Regression: surfaced from a live scenario run where Gemini/Claude
        emitted ``{"is_valid": false, ..., "confidence": null}``; the parser
        used to do ``float(None)`` and crash, dropping the vote.
        """
        raw = json.dumps(
            {
                "is_valid": False,
                "is_determinable": None,
                "has_occurred": None,
                "confidence": None,
            }
        )
        result = _parse_vote(raw, "test", "model")
        assert result.error is None
        assert result.is_valid is False
        assert result.confidence == 0.0

    def test_invalid_canonicalizes_to_case_a(self) -> None:
        """``is_valid=False`` forces ``(False, None, None)`` per the parser contract.

        Even if the LLM emits contradictory fields (is_determinable=true,
        has_occurred=true) alongside is_valid=false, the parser canonicalizes
        the output to the Case A shape ``(False, None, None)``.
        """
        raw = json.dumps(
            {
                "is_valid": False,
                "is_determinable": True,
                "has_occurred": True,
                "confidence": 0.9,
            }
        )
        result = _parse_vote(raw, "test", "model")
        assert result.is_valid is False
        assert result.is_determinable is None
        assert result.has_occurred is None
        assert result.confidence == 0.5


# ---------------------------------------------------------------------------
# Consensus helpers
# ---------------------------------------------------------------------------


class TestDecidedVotes:
    """Tests for ``_decided_votes`` (broad: YES/NO/INVALID, used by ``n_decided``).

    Decided = the voter reached a verdict the resolver can act on. INVALID
    counts because "this question is invalid" is itself a decision. Only
    undeterminable and errored voters are filtered out.
    """

    @pytest.mark.parametrize(
        "bad_vote, expected_count, expected_kept",
        [
            # Undeterminable -- not decided.
            (_vote(is_determinable=False), 1, "good_vote_only"),
            # INVALID is itself a decided verdict -- kept.
            (_vote(is_valid=False), 2, "both"),
            # Errored stub -- not decided.
            (_vote(error="failed"), 1, "good_vote_only"),
        ],
        ids=["indeterminate", "invalid_kept", "error"],
    )
    def test_filter(
        self,
        bad_vote: VoterResult,
        expected_count: int,
        expected_kept: str,
    ) -> None:
        """Undet + errored are filtered; INVALID and YES/NO are kept."""
        votes = [bad_vote, _vote()]
        assert len(_decided_votes(votes)) == expected_count

    def test_all_valid(self) -> None:
        """All YES/NO votes pass through."""
        votes = [_vote(), _vote(), _vote()]
        assert len(_decided_votes(votes)) == 3


class TestAllAgree:
    """Tests for _has_consensus consensus check."""

    @pytest.mark.parametrize(
        "votes, expected",
        [
            ([_vote(has_occurred=True), _vote(has_occurred=True)], True),
            ([_vote(has_occurred=False), _vote(has_occurred=False)], True),
            ([_vote(has_occurred=True), _vote(has_occurred=False)], False),
            # Single voter: 1/1 > 0.5 -> trivial consensus. Degenerate in
            # production (always 4 voters) but the symmetric rule treats it
            # consistently with all other strict-majority cases.
            ([_vote()], True),
            (
                [
                    _vote(has_occurred=True),
                    _vote(has_occurred=True),
                    _vote(is_determinable=False),
                ],
                True,
            ),
            (
                [
                    _vote(has_occurred=True),
                    _vote(is_determinable=False),
                    _vote(is_determinable=False),
                    _vote(is_determinable=False),
                ],
                False,
            ),
            (
                [
                    _vote(has_occurred=True),
                    _vote(has_occurred=True),
                    _vote(is_determinable=False),
                    _vote(is_determinable=False),
                ],
                False,
            ),
        ],
        ids=[
            "unanimous_yes",
            "unanimous_no",
            "disagreement",
            "single",
            "ignores_indet_minority",
            "minority_decided_rejected",
            "half_decided_rejected",
        ],
    )
    def test_has_consensus(self, votes: list, expected: bool) -> None:
        """Check consensus detection."""
        assert _has_consensus(votes) is expected

    @pytest.mark.parametrize(
        "votes, expected",
        [
            # Unanimous invalid -- strict majority (4/4 > 2)
            ([_vote(is_valid=False, is_determinable=None, has_occurred=None)] * 4, True),
            # 3/4 invalid -- strict majority (3 > 2)
            (
                [_vote(is_valid=False, is_determinable=None, has_occurred=None)] * 3
                + [_vote(has_occurred=True)],
                True,
            ),
            # 2/4 invalid -- not a strict majority
            (
                [_vote(is_valid=False, is_determinable=None, has_occurred=None)] * 2
                + [_vote(has_occurred=True)] * 2,
                False,
            ),
            # 3/4 invalid, 1 errored -- 3 > 2 still strict majority of 4
            (
                [_vote(is_valid=False, is_determinable=None, has_occurred=None)] * 3
                + [_vote(error="boom")],
                True,
            ),
            # 2/4 invalid, 2 errored -- only 2/4 are invalid, not strict majority
            (
                [_vote(is_valid=False, is_determinable=None, has_occurred=None)] * 2
                + [_vote(error="boom")] * 2,
                False,
            ),
        ],
        ids=[
            "unanimous_invalid",
            "majority_invalid_one_yes",
            "half_invalid_half_yes",
            "majority_invalid_one_errored",
            "minority_invalid_half_errored",
        ],
    )
    def test_invalid_consensus(self, votes: list, expected: bool) -> None:
        """``is_valid=False`` strict majority counts as consensus."""
        assert _has_consensus(votes) is expected


class TestBuildConsensusResult:
    """Tests for _build_consensus_result."""

    def test_builds_result_decided(self) -> None:
        """Decided consensus result has correct fields."""
        votes = [_vote(has_occurred=True), _vote(has_occurred=True)]
        result = _build_consensus_result(votes)
        assert result["has_occurred"] is True
        assert result["is_valid"] is True
        assert result["is_determinable"] is True
        assert result["agreement_ratio"] == 1.0
        assert "judge skipped" in result["judge_reasoning"]
        assert "INVALID" not in result["judge_reasoning"]

    def test_builds_result_invalid_consensus(self) -> None:
        """All voters say is_valid=False -> consensus on INVALID.

        Output MUST match the canonical Case A shape ``(False, None, None)``
        per the ``parse_mech_response`` docstring contract in market-resolver's
        ``behaviours/base.py``. ``is_determinable`` and ``has_occurred`` are
        ``None`` because they lose semantic meaning when the question is
        invalid.

        ``n_decided`` must be 4 (NOT 0): INVALID is itself a definitive
        verdict, so all 4 voters count as "decided". Only undeterminable
        (Case B) and errored voters fail to contribute to ``n_decided``.

        Regression: previously this would crash on ``decided[0]`` (empty
        list) and/or emit the wrong ``is_valid=True, is_determinable=True``
        hardcoded shape.
        """
        votes = [
            _vote(is_valid=False, is_determinable=None, has_occurred=None)
        ] * 4
        result = _build_consensus_result(votes)
        assert result["is_valid"] is False
        assert result["is_determinable"] is None
        assert result["has_occurred"] is None
        assert result["agreement_ratio"] == 1.0
        assert result["n_voters"] == 4
        # All 4 voters returned successfully -- this is the key assertion the
        # production catalog calls out: n_successful must be 4, not 0.
        assert result["n_successful"] == 4
        # All 4 voters reached a definitive verdict (INVALID) -- "invalid"
        # is a decided option, so n_decided is 4, not 0.
        assert result["n_decided"] == 4
        assert "INVALID" in result["judge_reasoning"]
        assert "judge skipped" in result["judge_reasoning"]

    def test_builds_result_invalid_consensus_with_one_dissenter(self) -> None:
        """3/4 invalid + 1 yes -> still consensus on INVALID.

        Voter shapes vary (the dissenter has different fields), but the
        canonical output is the same Case A ``(False, None, None)`` regardless.

        ``n_decided`` here is 4: 3 INVALID voters + 1 YES voter all reached
        a definitive verdict.
        """
        votes = [
            _vote(is_valid=False, is_determinable=None, has_occurred=None)
        ] * 3 + [_vote(has_occurred=True)]
        result = _build_consensus_result(votes)
        assert result["is_valid"] is False
        assert result["is_determinable"] is None
        assert result["has_occurred"] is None
        assert result["n_voters"] == 4
        assert result["n_successful"] == 4  # all returned, none errored
        assert result["n_decided"] == 4  # 3 INVALID + 1 YES, all definitive
        assert "INVALID" in result["judge_reasoning"]


class TestComputeAgreement:
    """Tests for _compute_agreement."""

    def test_full_agreement(self) -> None:
        """All votes same gives 1.0."""
        votes = [_vote(has_occurred=True)] * 3
        assert _compute_agreement(votes) == 1.0

    def test_majority(self) -> None:
        """2-1 split gives 2/3."""
        votes = [
            _vote(has_occurred=True),
            _vote(has_occurred=True),
            _vote(has_occurred=False),
        ]
        assert abs(_compute_agreement(votes) - 2 / 3) < 0.01

    def test_empty(self) -> None:
        """No decided votes gives 0.0."""
        votes = [_vote(is_determinable=False)]
        assert _compute_agreement(votes) == 0.0


# ---------------------------------------------------------------------------
# _adapter_openrouter usage recording
# ---------------------------------------------------------------------------


class TestAdapterOpenrouterUsageRecording:
    """Tests for the counter_callback recording inside _adapter_openrouter."""

    def _make_client(
        self, *, usage: Any, content: str = '{"is_valid": true}'
    ) -> MagicMock:
        """Build a mocked openai client returning a single response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=content))
        ]
        mock_client.chat.completions.create.return_value.usage = usage
        return mock_client

    def _call(self, mock_client: MagicMock, counter_callback: Any) -> None:
        from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
            _adapter_openrouter,
        )

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            _adapter_openrouter(
                model="model-x",
                prompt="p",
                api_key="k",
                max_tokens=100,
                timeout=1,
                max_attempts=1,
                retry_delay=0,
                counter_callback=counter_callback,
            )

    def test_no_callback_is_noop(self) -> None:
        """When no callback is provided, nothing is recorded."""
        usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "cost"])
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.cost = 0.01
        self._call(self._make_client(usage=usage), counter_callback=None)
        # no assertion needed -- if anything tried to call the callback, this would crash

    def test_no_usage_is_noop(self) -> None:
        """When response has no usage, callback is not called."""
        cb = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"is_valid": true}'))
        ]
        mock_client.chat.completions.create.return_value.usage = None
        self._call(mock_client, counter_callback=cb)
        cb.assert_not_called()

    def test_forwards_tokens_and_call_cost(self) -> None:
        """Forwards token counts and usage.cost as call_cost."""
        cb = MagicMock()
        usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "cost"])
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.cost = 0.025
        self._call(self._make_client(usage=usage), counter_callback=cb)
        cb.assert_called_once()
        kwargs = cb.call_args.kwargs
        assert kwargs["model"] == "model-x"
        assert kwargs["input_tokens"] == 100
        assert kwargs["output_tokens"] == 50
        assert kwargs["call_cost"] == 0.025

    def test_call_cost_none_when_usage_has_no_cost(self) -> None:
        """When usage.cost is missing, call_cost is None."""
        cb = MagicMock()
        usage = MagicMock(spec=["prompt_tokens", "completion_tokens"])
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        self._call(self._make_client(usage=usage), counter_callback=cb)
        kwargs = cb.call_args.kwargs
        assert kwargs["call_cost"] is None

    def test_callback_exception_is_swallowed(self) -> None:
        """Exceptions in callback are caught and logged."""
        cb = MagicMock(side_effect=ValueError("bad model"))
        usage = MagicMock(spec=["prompt_tokens", "completion_tokens"])
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        self._call(
            self._make_client(usage=usage), counter_callback=cb
        )  # should not raise

    def test_noop_token_counter_returns_zero(self) -> None:
        """The placeholder counter ignores arguments and returns 0."""
        assert _noop_token_counter() == 0
        assert _noop_token_counter("any text", "any model") == 0
        assert _noop_token_counter(text="x", model="y") == 0


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class TestAdapterOpenrouter:
    """Tests for OpenRouter adapter."""

    def test_passes_model_through(self) -> None:
        """Passes the full model slug (including :online) through as-is."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(
                message=MagicMock(content='{"is_valid": true, "has_occurred": true}')
            )
        ]
        mock_client.chat.completions.create.return_value.usage = None

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _adapter_openrouter,
            )

            raw = _adapter_openrouter(
                model="x-ai/grok-4.1-fast:online",
                prompt="prompt",
                api_key="key",
                max_tokens=100,
                timeout=1,
                max_attempts=1,
                retry_delay=0,
            )
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "x-ai/grok-4.1-fast:online"
            assert '"is_valid": true' in raw


# ---------------------------------------------------------------------------
# _run_judge
# ---------------------------------------------------------------------------


class TestRunJudge:
    """Tests for the judge function."""

    def test_judge_returns_verdict(self) -> None:
        """Judge parses valid JSON response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"is_valid": true, "is_determinable": true, '
                    '"has_occurred": false, "judge_reasoning": "majority"}'
                )
            )
        ]

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _run_judge,
            )

            votes = [_vote(has_occurred=True), _vote(has_occurred=False)]
            result = _run_judge("question?", votes, "key")
            assert result["has_occurred"] is False

    def test_judge_unparseable(self) -> None:
        """Judge returns fallback on garbage response.

        ``is_valid`` MUST be ``None`` (not ``False``) so downstream
        consumers don't mistake a judge-side LLM failure for a real
        "question is invalid" verdict.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="not json at all"))
        ]

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _run_judge,
            )

            result = _run_judge("q?", [_vote()], "key")
            assert result["is_valid"] is None
            assert result["is_determinable"] is None
            assert result["has_occurred"] is None
            assert result["error"] == "judge_unparseable"
            assert "Unparseable" in result["judge_reasoning"]

    def test_judge_retries_on_529(self) -> None:
        """Judge retries on 529 overloaded error."""
        mock_client = MagicMock()
        err = MagicMock()
        err.status_code = 529
        mock_client.chat.completions.create.side_effect = [
            __import__("openai").APIStatusError(
                message="overloaded", response=err, body=None
            ),
            MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content='{"is_valid": true, "has_occurred": true}'
                        )
                    )
                ]
            ),
        ]

        with (
            patch(f"{MODULE}.openai.OpenAI", return_value=mock_client),
            patch(f"{MODULE}.time.sleep"),
        ):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _run_judge,
            )

            result = _run_judge("q?", [_vote()], "key")
            assert result["has_occurred"] is True

    def test_judge_retries_exhausted_raises(self) -> None:
        """Judge raises when all retries are exhausted."""
        mock_client = MagicMock()
        err = MagicMock()
        err.status_code = 529
        api_err = __import__("openai").APIStatusError(
            message="overloaded", response=err, body=None
        )
        mock_client.chat.completions.create.side_effect = api_err

        with (
            patch(f"{MODULE}.openai.OpenAI", return_value=mock_client),
            patch(f"{MODULE}.time.sleep"),
            pytest.raises(__import__("openai").APIStatusError),
        ):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                _run_judge,
            )

            _run_judge("q?", [_vote()], "key")


# ---------------------------------------------------------------------------
# collect_votes / cast_vote
# ---------------------------------------------------------------------------


class TestCollectVotes:
    """Tests for vote collection."""

    def test_collects_from_all_voters(self) -> None:
        """Collects votes from all registered voters."""
        mock_result = _vote()

        with patch(f"{MODULE}.cast_vote", return_value=mock_result):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                collect_votes,
            )

            results = collect_votes("question?", ["openai", "grok"], _mock_api_keys())
            assert len(results) == 2

    def test_handles_voter_failure(self) -> None:
        """Failed voters are recorded as error stubs, others still collected."""
        call_count = 0

        def side_effect(*args: str, **kwargs: str) -> VoterResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API down")
            return _vote()

        with patch(f"{MODULE}.cast_vote", side_effect=side_effect):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                collect_votes,
            )

            results = collect_votes("q?", ["openai", "grok"], _mock_api_keys())
            # Both voters appear in results: one error stub, one real vote.
            assert len(results) == 2
            errors = [r for r in results if r.error is not None]
            successes = [r for r in results if r.error is None]
            assert len(errors) == 1
            assert len(successes) == 1
            assert errors[0].error is not None
            assert "API down" in errors[0].error


class TestCastVote:
    """Tests for the cast_vote facade."""

    def test_delegates_to_adapter(self) -> None:
        """cast_vote looks up registry, calls adapter, and parses raw text."""
        raw_json = (
            '{"is_valid": true, "is_determinable": true, "has_occurred": true, '
            '"confidence": 0.9, "reasoning": "test", "sources": ["http://x"]}'
        )
        mock_adapter = MagicMock(return_value=raw_json)

        with patch(f"{MODULE}._ADAPTERS", {"_adapter_openrouter": mock_adapter}):
            from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
                cast_vote,
            )

            keys = _mock_api_keys()
            result = cast_vote("openai", "question?", keys)
            assert isinstance(result, VoterResult)
            assert result.has_occurred is True
            mock_adapter.assert_called_once()


# ---------------------------------------------------------------------------
# with_key_rotation
# ---------------------------------------------------------------------------


class TestWithKeyRotation:
    """Tests for the key rotation decorator."""

    def test_success_appends_api_keys(self) -> None:
        """Successful call appends api_keys to result tuple."""
        keys = _mock_api_keys()

        @__import__(
            "packages.valory.customs.resolve_market_jury.resolve_market_jury",
            fromlist=["with_key_rotation"],
        ).with_key_rotation
        def fake_run(**kwargs: str) -> tuple:
            return ("result", None, None, None)

        result = fake_run(api_keys=keys, tool="resolve-market-jury-v1")
        assert len(result) == 5
        assert result[0] == "result"
        assert result[4] is keys

    def test_broad_exception_returns_error(self) -> None:
        """Unexpected exceptions return error string."""
        keys = _mock_api_keys()

        @__import__(
            "packages.valory.customs.resolve_market_jury.resolve_market_jury",
            fromlist=["with_key_rotation"],
        ).with_key_rotation
        def fake_run(**kwargs: str) -> tuple:
            raise TypeError("bad")

        result = fake_run(api_keys=keys)
        assert "bad" in result[0]

    def test_rate_limit_rotates_keys(self) -> None:
        """Verify RateLimitError triggers key rotation and retry."""
        keys = _mock_api_keys()
        keys.max_retries.return_value = {"openai": 2, "openrouter": 2}
        call_count = 0

        @__import__(
            "packages.valory.customs.resolve_market_jury.resolve_market_jury",
            fromlist=["with_key_rotation"],
        ).with_key_rotation
        def fake_run(**kwargs: str) -> tuple:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise __import__("openai").RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            return ("ok", None, None, None)

        result = fake_run(api_keys=keys)
        assert result[0] == "ok"
        assert keys.rotate.called

    def test_rate_limit_exhausted_raises(self) -> None:
        """Verify RateLimitError raises when retries exhausted."""
        keys = _mock_api_keys()
        keys.max_retries.return_value = {"openai": 0, "openrouter": 0}

        @__import__(
            "packages.valory.customs.resolve_market_jury.resolve_market_jury",
            fromlist=["with_key_rotation"],
        ).with_key_rotation
        def fake_run(**kwargs: str) -> tuple:
            raise __import__("openai").RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body=None,
            )

        with pytest.raises(__import__("openai").RateLimitError):
            fake_run(api_keys=keys)


# ---------------------------------------------------------------------------
# run() -- the main entry point
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for the run() entry point."""

    def test_invalid_tool_returns_error(self) -> None:
        """Invalid tool name returns error string (caught by decorator)."""
        keys = _mock_api_keys()
        result = run(prompt="q?", tool="bad-tool", api_keys=keys)
        assert "not supported" in result[0]

    def test_cost_mode(self) -> None:
        """delivery_rate=0 returns cost via callback."""
        keys = _mock_api_keys()
        cb = MagicMock(return_value=0.05)
        # The decorator tries result + (api_keys,) but cost mode returns
        # a float. TypeError is caught by the decorator's broad except.
        # We verify the callback was invoked and the error message mentions
        # the type issue (confirming cost mode was reached).
        result = run(
            prompt="q?",
            tool="resolve-market-jury-v1",
            api_keys=keys,
            delivery_rate=0,
            counter_callback=cb,
        )
        cb.assert_called_once()
        # Decorator catches TypeError from float + tuple, returns error string
        assert "unsupported operand" in result[0]

    def test_cost_mode_no_callback_returns_error(self) -> None:
        """delivery_rate=0 without callback returns error (caught by decorator)."""
        keys = _mock_api_keys()
        result = run(
            prompt="q?",
            tool="resolve-market-jury-v1",
            api_keys=keys,
            delivery_rate=0,
        )
        assert "counter callback" in result[0]

    def test_unanimous_skips_judge(self) -> None:
        """Unanimous votes skip the judge."""
        keys = _mock_api_keys()
        unanimous_votes = [_vote(has_occurred=True)] * 3

        with (
            patch(f"{MODULE}.collect_votes", return_value=unanimous_votes),
            patch(f"{MODULE}._run_judge") as mock_judge,
        ):
            result = run(
                prompt="q?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )
            mock_judge.assert_not_called()
            parsed = json.loads(result[0])
            assert parsed["has_occurred"] is True

    def test_disagreement_calls_judge(self) -> None:
        """Disagreeing votes invoke the judge."""
        keys = _mock_api_keys()
        mixed_votes = [_vote(has_occurred=True), _vote(has_occurred=False)]
        judge_verdict = {
            "is_valid": True,
            "is_determinable": True,
            "has_occurred": False,
            "judge_reasoning": "majority wins",
        }

        with (
            patch(f"{MODULE}.collect_votes", return_value=mixed_votes),
            patch(f"{MODULE}._run_judge", return_value=judge_verdict),
        ):
            result = run(
                prompt="q?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )
            parsed = json.loads(result[0])
            assert parsed["has_occurred"] is False
            assert parsed["judge_reasoning"] == "majority wins"

    def test_no_votes_returns_failure(self) -> None:
        """No successful votes returns failure result.

        ``is_valid`` MUST be ``None`` (not ``False``) so downstream consumers
        can distinguish an API outage from a real "the question is invalid"
        verdict. ``is_valid=False`` is reserved for genuine invalid verdicts.
        """
        keys = _mock_api_keys()

        with patch(f"{MODULE}.collect_votes", return_value=[]):
            result = run(
                prompt="q?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )
            parsed = json.loads(result[0])
            assert parsed["is_valid"] is None
            assert parsed["is_determinable"] is None
            assert parsed["has_occurred"] is None
            assert parsed["error"] == "all_voters_failed"
            assert parsed["n_successful"] == 0

    def test_all_error_stubs_trigger_failure(self) -> None:
        """If every voter errors, the failure path is taken and the judge is skipped.

        ``is_valid`` MUST be ``None`` (not ``False``) -- see
        ``test_no_successful_votes_returns_failure`` for rationale.
        """
        keys = _mock_api_keys()
        all_errored = [
            _vote(voter="v1", error="boom"),
            _vote(voter="v2", error="timeout"),
        ]

        with (
            patch(f"{MODULE}.collect_votes", return_value=all_errored),
            patch(f"{MODULE}._run_judge") as mock_judge,
        ):
            result = run(
                prompt="q?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )
            mock_judge.assert_not_called()
            parsed = json.loads(result[0])
            assert parsed["is_valid"] is None
            assert parsed["is_determinable"] is None
            assert parsed["has_occurred"] is None
            assert parsed["error"] == "all_voters_failed"
            assert parsed["n_successful"] == 0
            assert parsed["judge_reasoning"] == (
                "All voters failed (API errors / empty responses)."
            )
            # Error stubs are preserved in the response for debuggability.
            assert len(parsed["votes"]) == 2

    def test_n_successful_excludes_error_stubs_consensus(self) -> None:
        """n_successful counts only voters whose error is None (consensus path)."""
        keys = _mock_api_keys()
        # 3 successful + 1 errored -- successful voters unanimously agree, so
        # _has_consensus returns True and the judge is skipped.
        votes = [
            _vote(voter="v1", has_occurred=True),
            _vote(voter="v2", has_occurred=True),
            _vote(voter="v3", has_occurred=True),
            _vote(voter="v4", error="crashed"),
        ]

        with (
            patch(f"{MODULE}.collect_votes", return_value=votes),
            patch(f"{MODULE}._run_judge") as mock_judge,
        ):
            result = run(
                prompt="q?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )
            mock_judge.assert_not_called()
            parsed = json.loads(result[0])
            assert parsed["n_successful"] == 3  # not 4 -- error stub excluded
            assert len(parsed["votes"]) == 4  # all voters still in result

    def test_n_successful_excludes_error_stubs_judge(self) -> None:
        """n_successful counts only voters whose error is None (judge path)."""
        keys = _mock_api_keys()
        # 2 disagreeing successful + 1 errored -- triggers judge.
        votes = [
            _vote(voter="v1", has_occurred=True),
            _vote(voter="v2", has_occurred=False),
            _vote(voter="v3", error="crashed"),
        ]
        judge_verdict = {
            "is_valid": True,
            "is_determinable": True,
            "has_occurred": True,
            "judge_reasoning": "sided with v1",
        }

        with (
            patch(f"{MODULE}.collect_votes", return_value=votes),
            patch(f"{MODULE}._run_judge", return_value=judge_verdict),
        ):
            result = run(
                prompt="q?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )
            parsed = json.loads(result[0])
            assert parsed["n_successful"] == 2  # not 3 -- error stub excluded
            assert len(parsed["votes"]) == 3


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestCounterCallbackConcurrency:
    """Tests for the thread-safety guarantee around counter_callback."""

    def test_counter_callback_lock_is_held_during_invocation(self) -> None:
        """_adapter_openrouter holds COUNTER_CALLBACK_LOCK while invoking the callback.

        We verify this by asserting the lock is locked from inside the callback.
        This is a direct contract check -- no threads needed.
        """
        from packages.valory.customs.resolve_market_jury.resolve_market_jury import (
            COUNTER_CALLBACK_LOCK,
            _adapter_openrouter,
        )

        held_during_call = {"value": False}

        def inspecting_callback(**kwargs: Any) -> None:
            # acquire(blocking=False) returns False iff the lock is already held.
            held_during_call["value"] = not COUNTER_CALLBACK_LOCK.acquire(
                blocking=False
            )
            if not held_during_call["value"]:
                COUNTER_CALLBACK_LOCK.release()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="{}"))
        ]
        mock_client.chat.completions.create.return_value.usage = MagicMock(
            prompt_tokens=1, completion_tokens=1, cost=0.001
        )

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            _adapter_openrouter(
                model="anthropic/claude-haiku-4.5:online",
                prompt="p",
                api_key="k",
                max_tokens=10,
                timeout=1,
                max_attempts=1,
                retry_delay=0,
                counter_callback=inspecting_callback,
            )

        assert held_during_call["value"] is True


# ---------------------------------------------------------------------------
# Integration tests: OpenRouter HTTP 402 ("Insufficient credits") propagates
# correctly through collect_votes -> _successful_votes -> run().
# ---------------------------------------------------------------------------


class TestQuotaExhaustionIntegration:
    """End-to-end integration tests for the OpenRouter quota-exhaustion path.

    Replicates the production-observed scenario from market-resolver Safe
    ``0xa592085e...c7433a9d`` on the Gnosis Mech subgraph: every OpenRouter
    voter call returns ``Error code: 402 - {'error': {'message': 'Insufficient
    credits. Add more using https://openrouter.ai/settings/credits',
    'code': 402}}``. Verifies:

    1. ``collect_votes`` records the error on each voter's ``VoterResult``.
    2. ``_successful_votes`` filters them all out (empty list).
    3. ``run()`` takes the all-voters-failed branch and emits
       ``is_valid=None`` -- NOT ``is_valid=False`` -- so downstream
       consumers don't mistake a billing-quota outage for a "market is
       invalid" verdict.
    """

    def _make_402_error(self) -> Any:
        """Construct an ``openai.APIStatusError`` matching production shape."""
        import openai

        err_response = MagicMock()
        err_response.status_code = 402
        body = {
            "error": {
                "message": (
                    "Insufficient credits. Add more using "
                    "https://openrouter.ai/settings/credits"
                ),
                "code": 402,
            }
        }
        return openai.APIStatusError(
            message=f"Error code: 402 - {body}",
            response=err_response,
            body=body,
        )

    def test_all_voters_402_emits_is_valid_none(self) -> None:
        """All 4 voters get HTTP 402; pipeline emits is_valid=None."""
        keys = _mock_api_keys()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = self._make_402_error()

        with patch(f"{MODULE}.openai.OpenAI", return_value=mock_client):
            result = run(
                prompt="Will the number of immigration hearings be above 500?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )

        parsed = json.loads(result[0])

        # === Pipeline-level assertions ===
        assert parsed["n_voters"] == 4, "should attempt all 4 default voters"
        assert parsed["n_successful"] == 0, "0 voters succeeded"

        # The critical assertion: top-level is_valid is None, NOT False.
        # If this regresses to False, downstream parse_mech_response could
        # interpret the result as a genuine 'market is invalid' verdict and
        # submit ANSWER_INVALID (0xff...ff) to Realitio -- burning bonds on
        # what is really just an API quota outage.
        assert parsed["is_valid"] is None
        assert parsed["is_determinable"] is None
        assert parsed["has_occurred"] is None
        assert parsed["error"] == "all_voters_failed"
        assert "All voters failed" in parsed["judge_reasoning"]
        assert parsed["agreement_ratio"] == 0.0
        assert parsed["n_decided"] == 0

        # === Per-voter assertions ===
        # collect_votes must have produced one VoterResult per attempted
        # voter, each with the 402 error recorded and is_valid=None.
        assert len(parsed["votes"]) == 4
        for vote in parsed["votes"]:
            assert vote["is_valid"] is None
            assert vote["is_determinable"] is None
            assert vote["has_occurred"] is None
            assert vote["confidence"] == 0.0
            assert vote["error"] is not None
            assert "402" in vote["error"], (
                f"voter error should mention HTTP 402, got: {vote['error']!r}"
            )

        # === Filter behaviour ===
        # Reconstruct VoterResult objects from the parsed output and confirm
        # _successful_votes filters them ALL out -- this is the gate that
        # routes into the all-voters-failed branch in run().
        votes = [VoterResult(**v) for v in parsed["votes"]]
        assert _successful_votes(votes) == [], (
            "_successful_votes should drop all error-stubbed voters"
        )
        assert len(votes) == 4
        assert all(v.error is not None for v in votes)

    def test_partial_quota_exhaustion_one_voter_survives(self) -> None:
        """3 voters fail with 402, 1 succeeds -- run() must NOT take the
        all-voters-failed branch; the surviving voter should drive the verdict
        (judge or consensus path)."""
        keys = _mock_api_keys()

        # 1st call (whichever voter pool item runs first) succeeds; the
        # remaining 3 fail with 402.
        good_response = MagicMock()
        good_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "is_valid": True,
                            "is_determinable": True,
                            "has_occurred": True,
                            "confidence": 0.8,
                            "reasoning": "verified via search",
                            "sources": ["http://example.com"],
                        }
                    )
                )
            )
        ]
        good_response.usage = None

        bad_err = self._make_402_error()

        mock_client = MagicMock()
        # side_effect: first call returns the good response, subsequent
        # calls raise 402. Note: thread ordering in collect_votes is
        # non-deterministic but the side_effect queue is process-global, so
        # exactly 1 of the 4 voters will get the success and the other 3
        # will get errors regardless of scheduling.
        mock_client.chat.completions.create.side_effect = [
            good_response, bad_err, bad_err, bad_err,
        ]

        # Mock the judge too -- with only 1 successful vote there is no
        # consensus, so _run_judge will be called. We short-circuit it.
        judge_verdict = {
            "is_valid": True,
            "is_determinable": True,
            "has_occurred": True,
            "judge_reasoning": "Single surviving voter said YES; agreed.",
        }
        with (
            patch(f"{MODULE}.openai.OpenAI", return_value=mock_client),
            patch(f"{MODULE}._run_judge", return_value=judge_verdict),
        ):
            result = run(
                prompt="Will X happen?",
                tool="resolve-market-jury-v1",
                api_keys=keys,
            )

        parsed = json.loads(result[0])

        # NOT all-voters-failed -- one survived, so we DO get a verdict.
        assert parsed.get("error") != "all_voters_failed"
        assert parsed["n_successful"] == 1
        assert len(parsed["votes"]) == 4
        # Three voters errored
        errored = [v for v in parsed["votes"] if v.get("error") is not None]
        assert len(errored) == 3
        for v in errored:
            assert "402" in v["error"]
