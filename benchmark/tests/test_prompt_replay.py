# -*- coding: utf-8 -*-
"""Tests for prompt_replay filter + sidecar plumbing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from benchmark.prompt_replay import (
    _baseline_family,
    _extract_factual_research_prompt_components,
    _load_and_filter_rows,
    _log_replay_summary,
    _parse_vllm_candidate,
    _prepare_output_dir,
    _replay_vllm_candidate,
    extract_prompt_components,
    replay,
)
from benchmark.tools import TOOL_REGISTRY

from packages.valory.customs.factual_research.factual_research import REFRAME_USER
from packages.valory.customs.finetuned_prediction import finetuned_prediction


def _row(
    *,
    tool: str = "superforcaster",
    deliver_id: str = "0xabc",
    status: str = "valid",
    outcome: Any = True,
    predicted_at: str | None = None,
) -> dict:
    r: dict = {
        "tool_name": tool,
        "deliver_id": deliver_id,
        "prediction_parse_status": status,
        "final_outcome": outcome,
    }
    if predicted_at is not None:
        r["predicted_at"] = predicted_at
    return r


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


class TestLoadAndFilterRows:
    """Rejection counters on `_load_and_filter_rows`."""

    def test_all_accepted_zero_rejections(self, tmp_path: Path) -> None:
        """Rows matching every filter land in rows; counters stay at zero."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(), _row()])
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert len(rows) == 2
        assert sum(rejected.values()) == 0

    def test_wrong_tool_counted(self, tmp_path: Path) -> None:
        """Rows with a different tool_name hit the wrong_tool bucket only."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(tool="other"), _row(tool="another")])
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert not rows
        assert rejected["wrong_tool"] == 2
        assert rejected["not_valid_parse"] == 0

    def test_not_valid_parse_counted(self, tmp_path: Path) -> None:
        """The load-bearing bucket — a leaked non-valid row lands here."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(
            path,
            [
                _row(status="malformed"),
                _row(status="missing_fields"),
                _row(status="error"),
                _row(status="valid"),
            ],
        )
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert len(rows) == 1
        assert rejected["not_valid_parse"] == 3

    def test_no_outcome_counted(self, tmp_path: Path) -> None:
        """Rows without final_outcome hit no_outcome, not not_valid_parse."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(outcome=None)])
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert not rows
        assert rejected["no_outcome"] == 1
        assert rejected["not_valid_parse"] == 0

    def test_no_deliver_id_counted(self, tmp_path: Path) -> None:
        """Rows without deliver_id hit no_deliver_id."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(deliver_id="")])
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert not rows
        assert rejected["no_deliver_id"] == 1

    def test_filter_order_gives_first_failing_reason(self, tmp_path: Path) -> None:
        """A row failing multiple predicates counts in the first one only.

        Order: wrong_tool → no_deliver_id → not_valid_parse → no_outcome → cutoff.
        A row with wrong tool AND bad parse AND no outcome increments wrong_tool
        only — so the total rejection count matches row count exactly.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = tmp_path / "log.jsonl"
        _write_jsonl(
            path,
            [_row(tool="other", status="malformed", outcome=None)],
        )
        rows, rejected = _load_and_filter_rows(path, "superforcaster", None)
        assert not rows
        assert rejected["wrong_tool"] == 1
        assert sum(rejected.values()) == 1


class TestLogReplaySummaryFilterStats:
    """`_log_replay_summary` renders the Pre-filter block when stats are given."""

    def _sampled(self) -> list[dict]:
        return [
            {"p_yes": 0.7, "p_no": 0.3, "final_outcome": True, "tool_name": "x"},
            {"p_yes": 0.4, "p_no": 0.6, "final_outcome": False, "tool_name": "x"},
        ]

    def _write_candidate(self, path: Path) -> None:
        rows = [
            {"p_yes": 0.8, "p_no": 0.2, "final_outcome": True},
            {"p_yes": 0.3, "p_no": 0.7, "final_outcome": False},
        ]
        _write_jsonl(path, rows)

    def test_block_rendered_when_filter_stats_given(
        self, tmp_path: Path, caplog: Any
    ) -> None:
        """The Pre-filter line appears in the log when filter_stats is not None."""
        candidate = tmp_path / "candidate.jsonl"
        self._write_candidate(candidate)
        stats = {
            "accepted": 2,
            "rejected": {
                "wrong_tool": 4,
                "no_deliver_id": 0,
                "not_valid_parse": 1,
                "no_outcome": 2,
                "older_than_cutoff": 0,
            },
        }
        with caplog.at_level("INFO"):
            _log_replay_summary(
                sampled=self._sampled(),
                candidate_path=candidate,
                baseline_brier_sum=0.1,
                candidate_brier_sum=0.2,
                total=2,
                n_scored=2,
                baseline_path=candidate,
                status_counts={
                    "valid": 2,
                    "missing_fields": 0,
                    "malformed": 0,
                    "error": 0,
                },
                filter_stats=stats,
            )
        text = caplog.text
        assert "Pre-filter" in text
        assert "not_valid_parse=1" in text

    def test_block_omitted_when_stats_none(self, tmp_path: Path, caplog: Any) -> None:
        """When no sidecar is provided the summary logs exactly as before."""
        candidate = tmp_path / "candidate.jsonl"
        self._write_candidate(candidate)
        with caplog.at_level("INFO"):
            _log_replay_summary(
                sampled=self._sampled(),
                candidate_path=candidate,
                baseline_brier_sum=0.1,
                candidate_brier_sum=0.2,
                total=2,
                n_scored=2,
                baseline_path=candidate,
                status_counts={
                    "valid": 2,
                    "missing_fields": 0,
                    "malformed": 0,
                    "error": 0,
                },
            )
        assert "Pre-filter" not in caplog.text


class TestPrepareOutputDir:
    """`_prepare_output_dir` must isolate runs that share an output_dir.

    Regression coverage for stale-sidecar leak (see PR #231 review): both
    `candidate_failures.jsonl` and `filter_stats.json` are written
    conditionally (only when failures / stats exist), so a clean run in a
    reused directory would otherwise inherit the previous run's sidecars
    and silently mislead ci_replay.
    """

    def test_stale_candidate_failures_purged_before_fresh_run(
        self, tmp_path: Path
    ) -> None:
        """A prior run's candidate_failures.jsonl must not survive prep."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        stale = output_dir / "candidate_failures.jsonl"
        stale.write_text(
            json.dumps({"row_id": "stale", "raw_response": "from prior run"}) + "\n",
            encoding="utf-8",
        )

        _prepare_output_dir(output_dir)

        assert not stale.exists()

    def test_stale_filter_stats_purged_before_fresh_run(self, tmp_path: Path) -> None:
        """A prior run's filter_stats.json must not survive prep."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        stale = output_dir / "filter_stats.json"
        stale.write_text(
            json.dumps({"accepted": 999, "rejected": {"not_valid_parse": 42}}),
            encoding="utf-8",
        )

        _prepare_output_dir(output_dir)

        assert not stale.exists()

    def test_prep_preserves_unrelated_artifacts(self, tmp_path: Path) -> None:
        """Prep must only purge the two sidecars, not e.g. chained enriched output."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        keep = output_dir / "enriched_with_new_reasoning.jsonl"
        keep.write_text("keep me\n", encoding="utf-8")

        _prepare_output_dir(output_dir)

        assert keep.read_text(encoding="utf-8") == "keep me\n"

    def test_prep_is_noop_when_sidecars_absent(self, tmp_path: Path) -> None:
        """Fresh output_dir: prep must not error when there's nothing to purge."""
        output_dir = tmp_path / "out"
        _prepare_output_dir(output_dir)
        assert output_dir.is_dir()


def _fr_prompt(
    *,
    question: str = "Will X happen by 2026-12-31?",
    today: str = "2026-04-22",
    briefing: str = "Key facts:\n- Fact one.\n- Fact two.",
    include_briefing: bool = True,
    reframe_body: str | None = None,
) -> str:
    # Mirrors full_prompt_used in factual_research.py. We format the user
    # content via the real REFRAME_USER template so a reword there breaks
    # these tests (which is what would break prod extraction).
    if reframe_body is None:
        reframe_messages = [
            {"role": "system", "content": "REFRAME_SYSTEM"},
            {
                "role": "user",
                "content": REFRAME_USER.format(question=question, today=today),
            },
        ]
        reframe_body = json.dumps(reframe_messages, indent=2)

    parts = [
        f"--- REFRAME ---\n{reframe_body}\n",
        "--- SUB-QUESTIONS ---\n{}\n",
        "--- EVIDENCE ---\n[1] snippet\n",
        "--- SYNTHESIS ---\n[]\n",
    ]
    if include_briefing:
        parts.append(f"--- BRIEFING ---\n{briefing}\n")
    parts.append("--- ESTIMATE ---\n[]\n")
    parts.append("--- REASONING ---\nblah")
    return "\n".join(parts)


class TestExtractFactualResearchPromptComponents:
    """Extraction for `factual_research` multi-section audit dumps."""

    def test_happy_path_returns_question_briefing_today(self) -> None:
        """Well-formed dump yields question, briefing, and today, stripped."""
        prompt = _fr_prompt(
            question="  Will Artemis II launch by 2026-12-31?  ",
            today="2026-04-22",
            briefing="  Factual summary line 1.\nLine 2.  ",
        )
        out = _extract_factual_research_prompt_components(prompt)
        assert out is not None
        assert out["user_prompt"] == "Will Artemis II launch by 2026-12-31?"
        assert out["today"] == "2026-04-22"
        assert out["additional_information"] == "Factual summary line 1.\nLine 2."

    def test_missing_briefing_returns_none(self) -> None:
        """No BRIEFING section → None (don't mis-route to default regexes)."""
        prompt = _fr_prompt(include_briefing=False)
        assert _extract_factual_research_prompt_components(prompt) is None

    def test_malformed_reframe_json_returns_none(self) -> None:
        """Invalid JSON in REFRAME block → None (no raise)."""
        prompt = _fr_prompt(reframe_body="{this is not json")
        assert _extract_factual_research_prompt_components(prompt) is None

    def test_reframe_missing_user_message_returns_none(self) -> None:
        """REFRAME JSON without a user message → None."""
        reframe_body = json.dumps(
            [{"role": "system", "content": "only system"}], indent=2
        )
        prompt = _fr_prompt(reframe_body=reframe_body)
        assert _extract_factual_research_prompt_components(prompt) is None

    def test_reframe_non_list_json_returns_none(self) -> None:
        """REFRAME JSON that parses but isn't a list → None (don't raise)."""
        prompt = _fr_prompt(reframe_body=json.dumps({"role": "user"}))
        assert _extract_factual_research_prompt_components(prompt) is None

    def test_reframe_list_of_non_dicts_returns_none(self) -> None:
        """REFRAME JSON list with non-dict elements → None (don't raise)."""
        prompt = _fr_prompt(reframe_body=json.dumps(["just a string"]))
        assert _extract_factual_research_prompt_components(prompt) is None

    def test_dispatch_via_extract_prompt_components(self) -> None:
        """`extract_prompt_components` routes `factual_research` to the helper."""
        prompt = _fr_prompt(question="Q?", today="2026-04-22", briefing="B")
        out = extract_prompt_components(prompt, tool_name="factual_research")
        assert out is not None
        assert out["user_prompt"] == "Q?"
        assert out["today"] == "2026-04-22"
        assert out["additional_information"] == "B"


class TestBaselineFamily:
    """`_baseline_family` classifies a tool name to its prompt-attribute schema.

    Direct unit test of the small pure helper the replay path uses to decide
    which symbols (PREDICTION_PROMPT / SYSTEM_PROMPT / ESTIMATE_USER / ...)
    to read from the candidate module. The widening from exact-name match
    fixes the gap the #321 review flagged: previously several registered
    tools fell into the `else` branch and crashed when their module didn't
    export ``SYSTEM_PROMPT_FORECASTER``.
    """

    @pytest.mark.parametrize(
        "tool_name,expected",
        [
            # Reasoning family (two-stage).
            ("prediction-request-reasoning", "reasoning"),
            ("prediction-request-reasoning-claude", "reasoning"),
            # Rag family (PREDICTION_PROMPT + SYSTEM_PROMPT).
            ("prediction-request-rag", "rag"),
            ("prediction-request-rag-claude", "rag"),
            # prediction-url-cot is rag-shaped — #321 review fix.
            ("prediction-url-cot", "rag"),
            ("prediction-url-cot-claude", "rag"),
            # Superforcaster family (PREDICTION_PROMPT only).
            ("superforcaster", "superforcaster"),
            # Existing -polymarket-v1 sibling — #321 review fix; would
            # previously fall to `else` and crash on SYSTEM_PROMPT_FORECASTER.
            ("superforcaster-polymarket-v1", "superforcaster"),
            # Hypothetical new-version siblings the housekeeping default
            # produces: superforcaster -> superforcaster-v1 etc.
            ("superforcaster-v1", "superforcaster"),
            ("superforcaster-polymarket-v2", "superforcaster"),
            # SME tools also export only PREDICTION_PROMPT — #321 review fix.
            ("prediction-offline-sme", "superforcaster"),
            ("prediction-online-sme", "superforcaster"),
            # factual_research family (ESTIMATE_USER + ESTIMATE_SYSTEM).
            ("factual_research", "factual_research"),
            ("factual_research-v1", "factual_research"),
            # Default family (PREDICTION_PROMPT + SYSTEM_PROMPT_FORECASTER).
            ("prediction-online", "default"),
            ("prediction-offline", "default"),
            ("claude-prediction-online", "default"),
            ("claude-prediction-offline", "default"),
        ],
    )
    def test_classifies_registered_tools_to_their_actual_schema(
        self, tool_name: str, expected: str
    ) -> None:
        """Registered tools route to the schema their module actually exports.

        Covers the three tools the #321 review caught falling into the wrong
        branch (superforcaster-polymarket-v1, prediction-url-cot, *-sme) plus
        hypothetical ``-v<n+1>`` siblings the housekeeping default produces.

        :param tool_name: tool name from the parametrize matrix.
        :param expected: family the helper should classify ``tool_name`` to.
        """
        assert _baseline_family(tool_name) == expected

    def test_unknown_tool_falls_to_default(self) -> None:
        """Unknown names fall to the default family.

        The existing ``else`` branch in replay() then handles them (or raises
        on missing attributes downstream).
        """
        assert _baseline_family("totally-made-up-name") == "default"


class TestVllmCandidateRegistry:
    """The vLLM-served tools are registered with the ``vllm`` backend tag.

    The replay path keys off ``ToolSpec.backend`` to route a candidate through
    the self-hosted vLLM helper instead of the hosted OpenAI/Anthropic SDK, so
    the tag must be present (and default to ``openai`` for every other tool).
    """

    @pytest.mark.parametrize("tool_name", ["predict-base", "predict-fine-tuned"])
    def test_finetuned_tools_use_vllm_backend(self, tool_name: str) -> None:
        """Both fine-tuned modes are registered against the vLLM backend.

        :param tool_name: the registered vLLM tool name under test.
        """
        assert TOOL_REGISTRY[tool_name].backend == "vllm"

    def test_hosted_tools_default_to_openai_backend(self) -> None:
        """A hosted-API tool keeps the default ``openai`` backend."""
        assert TOOL_REGISTRY["superforcaster"].backend == "openai"


class TestParseVllmCandidate:
    """`_parse_vllm_candidate` maps a vLLM completion to the scoring dict.

    Parsing is delegated to the candidate tool's own ``canonical_prediction``
    (strip ``<think>``, validate ``p_yes``, derive ``p_no``) so it matches what
    the deployed mech delivers. The wrapper only translates that into the
    harness ``{p_yes, p_no, prediction_parse_status, confidence}`` contract.
    """

    def test_none_response_is_error_status(self) -> None:
        """A failed call (None completion) yields the ``error`` status."""
        parsed = _parse_vllm_candidate(None, finetuned_prediction)
        assert parsed == {
            "p_yes": None,
            "p_no": None,
            "prediction_parse_status": "error",
            "confidence": None,
        }

    def test_unparseable_completion_is_malformed(self) -> None:
        """A delivered-but-unparseable completion yields ``malformed``."""
        parsed = _parse_vllm_candidate(
            "<think>no number here</think> sorry", finetuned_prediction
        )
        assert parsed["prediction_parse_status"] == "malformed"
        assert parsed["p_yes"] is None
        assert parsed["p_no"] is None

    def test_valid_think_json_completion(self) -> None:
        """A ``<think>…</think>{json}`` completion parses to valid p_yes/p_no.

        ``p_no`` is derived from ``p_yes`` and ``confidence`` is carried through,
        exactly as ``canonical_prediction`` normalises them.
        """
        completion = '<think>weighing evidence</think>{"p_yes": 0.7, "confidence": 0.8}'
        parsed = _parse_vllm_candidate(completion, finetuned_prediction)
        assert parsed["prediction_parse_status"] == "valid"
        assert parsed["p_yes"] == 0.7
        assert parsed["p_no"] == 0.3
        assert parsed["confidence"] == 0.8


class TestReplayVllmCandidate:
    """`_replay_vllm_candidate` calls the vLLM server with production framing.

    It must drive the call through the candidate tool's OWN client path
    (``VLLMClientManager`` + ``generate_prediction_with_retry``) so the request
    is a single source of truth with the tool's ``run()`` — same prompt builder,
    same single-user-message framing, same n=1 / stop=None / settings — target
    the supplied base_url, and degrade to None (not raise) on a call failure.
    """

    @staticmethod
    def _row() -> dict:
        return {
            "extracted_user_prompt": "Will it rain tomorrow?",
            "extracted_today": "01/06/2026",
            "extracted_additional_information": "<background>forecast</background>",
        }

    def test_calls_vllm_with_production_framing(self, monkeypatch: Any) -> None:
        """The reused client path targets base_url with production parameters.

        Patches the shared ``openai.OpenAI`` the tool's ``VLLMClient`` builds, so
        the assertions cover the exact request that path emits (n=1, stop=None,
        single user message, deterministic settings).

        :param monkeypatch: pytest fixture used to stub ``openai.OpenAI``.
        """
        completion = '<think>r</think>{"p_yes": 0.6}'
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=completion))
        ]
        captured: dict = {}

        def _factory(**kwargs: Any) -> Any:
            captured["init"] = kwargs
            return fake_client

        monkeypatch.setattr("benchmark.prompt_replay.openai.OpenAI", _factory)

        result = _replay_vllm_candidate(
            row=self._row(),
            candidate_module=finetuned_prediction,
            base_url="https://vllm.example/v1",
            api_key="secret-key",
            model="qwen-14b-fine-tuned",
        )

        assert result == completion
        assert captured["init"]["base_url"] == "https://vllm.example/v1"
        assert captured["init"]["api_key"] == "secret-key"

        create_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert create_kwargs["model"] == "qwen-14b-fine-tuned"
        assert create_kwargs["temperature"] == 0.0
        assert create_kwargs["max_tokens"] == 1024
        # Reused client path locks production single-completion semantics.
        assert create_kwargs["n"] == 1
        assert create_kwargs["stop"] is None
        # Training-parity framing: exactly one user message, no system message.
        messages = create_kwargs["messages"]
        assert [m["role"] for m in messages] == ["user"]
        assert "<background>forecast</background>" in messages[0]["content"]
        # VLLMClientManager closes the underlying client on context exit.
        fake_client.close.assert_called_once()

    def test_call_failure_returns_none(self, monkeypatch: Any) -> None:
        """A server/SDK error degrades to None so the replay run continues.

        The tool's ``generate_prediction_with_retry`` raises after exhausting
        its retries; the helper must catch that and return None. ``time.sleep``
        is stubbed so the retry backoff doesn't slow the test.

        :param monkeypatch: pytest fixture used to stub ``openai.OpenAI``.
        """
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        monkeypatch.setattr(
            "benchmark.prompt_replay.openai.OpenAI", lambda **kwargs: fake_client
        )
        monkeypatch.setattr(finetuned_prediction.time, "sleep", lambda *_a: None)

        result = _replay_vllm_candidate(
            row=self._row(),
            candidate_module=finetuned_prediction,
            base_url="https://vllm.example/v1",
            api_key="secret-key",
            model="qwen-14b-fine-tuned",
        )

        assert result is None
        # The client is still closed (VLLMClientManager __exit__) on failure.
        fake_client.close.assert_called_once()


class TestVllmCandidateBaselineGuard:
    """`replay` rejects a vLLM candidate against a non-superforcaster baseline.

    The vLLM path renders a superforcaster-shaped ``<background>`` prompt from
    the baseline's extracted fields, so it must fail fast (not silently feed a
    malformed prompt) when the baseline belongs to another family.
    """

    def test_non_superforcaster_baseline_rejected(self, tmp_path: Path) -> None:
        """A reasoning-family baseline + vLLM candidate raises ValueError.

        :param tmp_path: pytest tmp_path fixture for the enriched dataset.
        """
        dataset = tmp_path / "dataset.jsonl"
        _write_jsonl(dataset, [{"tool_name": "prediction-request-reasoning"}])
        with pytest.raises(ValueError, match="superforcaster-family baseline"):
            replay(
                dataset=dataset,
                output_dir=tmp_path / "out",
                model="qwen-14b-fine-tuned",
                candidate_tool="predict-fine-tuned",
            )
