# -*- coding: utf-8 -*-
"""Tests for prompt_replay filter + sidecar plumbing."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from benchmark.prompt_replay import (
    DEFAULT_REPLAY_SYSTEM_PROMPT,
    _baseline_family,
    _default_family_system_prompt,
    _extract_factual_research_prompt_components,
    _load_and_filter_rows,
    _log_replay_summary,
    _prepare_output_dir,
    enrich,
    extract_prompt_components,
    main,
    stratified_sample,
)
from benchmark.tools import TOOL_REGISTRY

from packages.valory.customs.factual_research.factual_research import REFRAME_USER


def _row(
    *,
    tool: str = "superforcaster",
    deliver_id: str = "0xabc",
    status: str = "valid",
    outcome: Any = True,
    predicted_at: str | None = None,
    platform: str | None = None,
    row_id: str | None = None,
) -> dict:
    r: dict = {
        "tool_name": tool,
        "deliver_id": deliver_id,
        "prediction_parse_status": status,
        "final_outcome": outcome,
    }
    if predicted_at is not None:
        r["predicted_at"] = predicted_at
    if platform is not None:
        r["platform"] = platform
    if row_id is not None:
        r["row_id"] = row_id
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
        rows, rejected, _ = _load_and_filter_rows(path, "superforcaster", None)
        assert len(rows) == 2
        assert sum(rejected.values()) == 0

    def test_wrong_tool_counted(self, tmp_path: Path) -> None:
        """Rows with a different tool_name hit the wrong_tool bucket only."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(tool="other"), _row(tool="another")])
        rows, rejected, _ = _load_and_filter_rows(path, "superforcaster", None)
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
        rows, rejected, _ = _load_and_filter_rows(path, "superforcaster", None)
        assert len(rows) == 1
        assert rejected["not_valid_parse"] == 3

    def test_no_outcome_counted(self, tmp_path: Path) -> None:
        """Rows without final_outcome hit no_outcome, not not_valid_parse."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(outcome=None)])
        rows, rejected, _ = _load_and_filter_rows(path, "superforcaster", None)
        assert not rows
        assert rejected["no_outcome"] == 1
        assert rejected["not_valid_parse"] == 0

    def test_no_deliver_id_counted(self, tmp_path: Path) -> None:
        """Rows without deliver_id hit no_deliver_id."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(deliver_id="")])
        rows, rejected, _ = _load_and_filter_rows(path, "superforcaster", None)
        assert not rows
        assert rejected["no_deliver_id"] == 1

    def test_filter_order_gives_first_failing_reason(self, tmp_path: Path) -> None:
        """A row failing multiple predicates counts in the first one only.

        Order: duplicate → wrong_tool → wrong_platform → no_deliver_id →
        not_valid_parse → no_outcome → cutoff. A row with wrong tool AND bad
        parse AND no outcome increments wrong_tool only — so the total rejection
        count matches row count exactly.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = tmp_path / "log.jsonl"
        _write_jsonl(
            path,
            [_row(tool="other", status="malformed", outcome=None)],
        )
        rows, rejected, _ = _load_and_filter_rows(path, "superforcaster", None)
        assert not rows
        assert rejected["wrong_tool"] == 1
        assert sum(rejected.values()) == 1

    def test_wrong_platform_counted(self, tmp_path: Path) -> None:
        """With a platform filter, off-platform rows hit wrong_platform only."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(
            path,
            [
                _row(platform="omen"),
                _row(platform="omen"),
                _row(platform="polymarket"),
            ],
        )
        rows, rejected, _ = _load_and_filter_rows(
            path, "superforcaster", None, "polymarket"
        )
        assert len(rows) == 1
        assert rows[0]["platform"] == "polymarket"
        assert rejected["wrong_platform"] == 2
        assert rejected["wrong_tool"] == 0

    def test_platform_filter_none_keeps_all_platforms(self, tmp_path: Path) -> None:
        """Default (no platform filter) keeps every platform — additive contract."""
        path = tmp_path / "log.jsonl"
        _write_jsonl(
            path,
            [_row(platform="omen"), _row(platform="polymarket")],
        )
        rows, rejected, _ = _load_and_filter_rows(path, "superforcaster", None)
        assert len(rows) == 2
        assert rejected["wrong_platform"] == 0

    def test_wrong_platform_ordered_after_wrong_tool(self, tmp_path: Path) -> None:
        """A row failing both tool and platform counts as wrong_tool only.

        Pins wrong_platform's position in the filter order (right after
        wrong_tool): a foreign-tool row never reaches the platform check.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(tool="other", platform="omen")])
        rows, rejected, _ = _load_and_filter_rows(
            path, "superforcaster", None, "polymarket"
        )
        assert not rows
        assert rejected["wrong_tool"] == 1
        assert rejected["wrong_platform"] == 0

    def test_duplicate_row_ids_collapsed_keeping_first(self, tmp_path: Path) -> None:
        """Cross-shard repeats (same row_id) collapse to the first occurrence.

        Mirrors the flywheel emitting one delivery into multiple daily shards,
        sometimes with a flipped outcome in the later shard. With shards fed
        oldest-first, keep-first wins, matching the scorer's keep-oldest dedup.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = tmp_path / "log.jsonl"
        _write_jsonl(
            path,
            [
                _row(row_id="prod_x", outcome=True),
                _row(row_id="prod_x", outcome=False),  # later shard, flipped
                _row(row_id="prod_y", outcome=True),
            ],
        )
        rows, rejected, _ = _load_and_filter_rows(path, "superforcaster", None)
        assert len(rows) == 2  # prod_x once + prod_y
        assert rejected["duplicate"] == 1
        kept = {r["row_id"]: r["final_outcome"] for r in rows}
        assert kept == {"prod_x": True, "prod_y": True}  # first prod_x wins

    def test_rows_without_row_id_not_deduped_but_counted(self, tmp_path: Path) -> None:
        """Rows lacking a row_id are never collapsed, but counted as no_row_id.

        Pins the diagnostic that surfaces a flywheel row_id-schema regression:
        such rows bypass dedup (kept), so the count is the only signal.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, [_row(), _row()])  # identical, no row_id
        rows, rejected, no_row_id = _load_and_filter_rows(path, "superforcaster", None)
        assert len(rows) == 2
        assert rejected["duplicate"] == 0
        assert no_row_id == 2

    def test_duplicate_counted_before_tool_filter(self, tmp_path: Path) -> None:
        """Dedup precedes the tool check: a repeat counts as duplicate, not wrong_tool.

        Two rows share a row_id but carry a foreign tool. The first is rejected
        as wrong_tool (and its row_id remembered); the second is caught by the
        dedup pass first, so it lands in ``duplicate`` rather than wrong_tool.

        :param tmp_path: pytest tmp_path fixture.
        """
        path = tmp_path / "log.jsonl"
        _write_jsonl(
            path,
            [_row(tool="other", row_id="prod_z"), _row(tool="other", row_id="prod_z")],
        )
        rows, rejected, _ = _load_and_filter_rows(path, "superforcaster", None)
        assert not rows
        assert rejected["duplicate"] == 1
        assert rejected["wrong_tool"] == 1


class TestEnrichZeroRows:
    """`enrich` fails loudly when the filtered pool is empty.

    The guard fires after filtering but before the IPFS fetch, so these tests
    need no network — a baseline with no qualifying rows must abort with a
    clear message rather than silently skipping the dataset write and letting
    the downstream replay step die on a FileNotFoundError.
    """

    def test_raises_systemexit_naming_the_tool(self, tmp_path: Path) -> None:
        """0 qualifying rows → SystemExit mentioning the baseline tool.

        The filter_stats sidecar must still be written (enrich writes it before
        the guard) so the zero-row run stays diagnosable — pin that here so a
        future refactor moving the write below the guard can't silently drop it.

        :param tmp_path: pytest tmp_path fixture.
        """
        log_path = tmp_path / "log.jsonl"
        _write_jsonl(log_path, [_row(tool="other"), _row(tool="another")])
        output = tmp_path / "out" / "dataset.jsonl"
        with pytest.raises(SystemExit) as exc:
            enrich(log_path, "superforcaster", output)
        assert "superforcaster" in str(exc.value)
        assert not output.exists()
        assert (tmp_path / "out" / "dataset.jsonl.filter_stats.json").exists()

    def test_message_includes_platform_scope(self, tmp_path: Path) -> None:
        """When a platform filter empties the pool, the scope is in the message."""
        log_path = tmp_path / "log.jsonl"
        _write_jsonl(log_path, [_row(platform="omen"), _row(platform="omen")])
        output = tmp_path / "out" / "dataset.jsonl"
        with pytest.raises(SystemExit) as exc:
            enrich(log_path, "superforcaster", output, platform_filter="polymarket")
        assert "platform=polymarket" in str(exc.value)
        assert not output.exists()
        assert (tmp_path / "out" / "dataset.jsonl.filter_stats.json").exists()


class TestStratifiedSampleSinglePlatform:
    """`stratified_sample` applies the per-platform budget once for one platform."""

    @staticmethod
    def _pm(p_yes: float, outcome: bool, i: int) -> dict:
        r = _row(platform="polymarket", outcome=outcome, row_id=f"r{i}")
        r["p_yes"] = p_yes
        return r

    def test_budget_applied_once_not_doubled(self) -> None:
        """One platform, single stratum → exactly sample_per_platform (not ~2x).

        Pins the --platform semantics: the budget is applied once. A single
        stratum avoids the per-stratum floor and proportional-rounding overshoot
        (both of which can push the count above the budget — see the sibling
        test), isolating the cap so a future two-platform floor (e.g.
        ``sample_per_platform * 2``) that drew 10 here would trip it.
        """
        # One stratum only: all (outcome=yes, brier=good) via p_yes 0.9.
        rows = [self._pm(0.9, True, i) for i in range(20)]
        sampled = stratified_sample(rows, 5, seed=42)
        assert len(sampled) == 5
        assert {r["platform"] for r in sampled} == {"polymarket"}

    def test_per_stratum_floor_can_exceed_budget(self) -> None:
        """Documented edge: more non-empty strata than budget → floor wins.

        With 6 strata (2 outcomes x 3 brier buckets) and budget 3, the ">=1 per
        stratum" floor yields 6 rows — pinning the behaviour the docstring calls
        out (so the count slightly exceeding sample_per_platform is intentional,
        not a regression).
        """
        rows = []
        i = 0
        for outcome, ys in ((True, (0.95, 0.5, 0.1)), (False, (0.05, 0.5, 0.9))):
            for p_yes in ys:  # good, moderate, poor for each outcome
                rows += [self._pm(p_yes, outcome, i), self._pm(p_yes, outcome, i + 1)]
                i += 2
        sampled = stratified_sample(rows, 3, seed=42)
        assert len(sampled) == 6  # 6 strata x floor 1, budget 3 overridden


class TestEnrichCli:
    """The ``main()`` argparse seam: --platform wires to enrich + is validated."""

    def test_platform_arg_wires_to_enrich(self, monkeypatch: Any) -> None:
        """`--platform polymarket` reaches enrich() as platform_filter.

        All other tests call enrich() directly, so this is the only thing
        pinning the args.platform → platform_filter kwarg in the dispatch.

        :param monkeypatch: pytest monkeypatch fixture.
        """
        argv = [
            "prompt_replay",
            "enrich",
            "--production-log",
            "x.jsonl",
            "--tool",
            "superforcaster",
            "--platform",
            "polymarket",
        ]
        monkeypatch.setattr("sys.argv", argv)
        with mock.patch("benchmark.prompt_replay.enrich") as enrich_mock:
            main()
        assert enrich_mock.call_args.kwargs["platform_filter"] == "polymarket"
        assert enrich_mock.call_args.kwargs["tool_filter"] == "superforcaster"

    def test_invalid_platform_rejected_by_choices(self, monkeypatch: Any) -> None:
        """An unknown --platform value is rejected by argparse choices (exit 2)."""
        argv = [
            "prompt_replay",
            "enrich",
            "--production-log",
            "x.jsonl",
            "--platform",
            "ethereum",
        ]
        monkeypatch.setattr("sys.argv", argv)
        with mock.patch("benchmark.prompt_replay.enrich") as enrich_mock:
            with pytest.raises(SystemExit):
                main()
        enrich_mock.assert_not_called()


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
                "duplicate": 5,
                "wrong_tool": 4,
                "wrong_platform": 3,
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
        # New full-pool/--platform buckets must render, else the breakdown no
        # longer accounts for the rejected total.
        assert "duplicate=5" in text
        assert "wrong_platform=3" in text

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
    """`_baseline_family` classifies a tool name to its prompt-schema family.

    Direct unit test of the helper both ``prompt_replay`` dispatch sites use to
    decide which symbols (PREDICTION_PROMPT / SYSTEM_PROMPT / ESTIMATE_USER /
    ...) the replay path reads from a module and which regex extractor the
    enrich path uses. Family is read from ``TOOL_REGISTRY`` (source of truth)
    for registered tools, with a name heuristic only as a fallback for
    not-yet-registered ``-v<n+1>`` siblings.

    Note on history: the prior defect differed by branch. Tools misrouted into
    the old ``else``/default branch hard-imported from ``prediction_request``
    (which *does* export ``SYSTEM_PROMPT_FORECASTER``), so they ran with the
    *wrong prompt* — silent mis-prompting, no exception. The superforcaster
    branch was harsher: SME routed there hit a format-time ``KeyError`` (see
    ``TestRegistryFamilyFormatRoundTrip``), crashing the whole replay.
    """

    @pytest.mark.parametrize(
        "tool_name,expected",
        [
            # Reasoning family (two-stage).
            ("prediction-request-reasoning", "reasoning"),
            ("prediction-request-reasoning-claude", "reasoning"),
            # Not-yet-registered reasoning sibling → heuristic fallback.
            ("prediction-request-reasoning-v2", "reasoning"),
            # Rag family (PREDICTION_PROMPT + SYSTEM_PROMPT).
            ("prediction-request-rag", "rag"),
            ("prediction-request-rag-claude", "rag"),
            # Not-yet-registered rag sibling → heuristic fallback.
            ("prediction-request-rag-v2", "rag"),
            # prediction-url-cot is rag-shaped (<user_prompt> XML).
            ("prediction-url-cot", "rag"),
            ("prediction-url-cot-claude", "rag"),
            # Superforcaster family (PREDICTION_PROMPT + Question/<background>).
            ("superforcaster", "superforcaster"),
            ("superforcaster-polymarket-v1", "superforcaster"),
            # Not-yet-registered new-version siblings route via the heuristic
            # fallback to their parent's family.
            ("superforcaster-v1", "superforcaster"),
            ("superforcaster-polymarket-v2", "superforcaster"),
            # SME exports only PREDICTION_PROMPT but it uses
            # {user_prompt}/{additional_information} and ships no
            # SYSTEM_PROMPT_FORECASTER — that is the *default* schema, not
            # superforcaster (Ojus #321 review, comment 3364080538).
            ("prediction-offline-sme", "default"),
            ("prediction-online-sme", "default"),
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

        Pins the families the #321 review flagged as mis-routed — SME to
        ``default`` (not superforcaster), prediction-url-cot to ``rag`` — plus
        not-yet-registered ``-v<n+1>`` siblings handled by the fallback.

        :param tool_name: tool name from the parametrize matrix.
        :param expected: family the helper should classify ``tool_name`` to.
        """
        assert _baseline_family(tool_name) == expected

    def test_registry_is_source_of_truth_over_heuristic(self) -> None:
        """A registered name returns its registry family, not the heuristic.

        ``prediction-online-sme`` ends in ``-sme`` and the old heuristic mapped
        that to ``superforcaster``; the registry entry says ``default`` and
        must win. Guards against re-introducing the name heuristic as primary.
        """
        assert TOOL_REGISTRY["prediction-online-sme"].family == "default"
        assert _baseline_family("prediction-online-sme") == "default"

    def test_unknown_tool_falls_to_default(self) -> None:
        """Unknown, unregistered names fall to the default family."""
        assert _baseline_family("totally-made-up-name") == "default"

    def test_unregistered_name_warns(self, caplog: Any) -> None:
        """An unregistered tool warns that its family is being guessed.

        Makes a misnamed / not-yet-registered tool visible at classify time
        rather than only as a silent ``Enriched 0/N`` drop downstream.

        :param caplog: pytest log-capture fixture.
        """
        with caplog.at_level("WARNING"):
            _baseline_family("totally-made-up-name")
        assert "not in TOOL_REGISTRY" in caplog.text

    def test_registered_name_does_not_warn(self, caplog: Any) -> None:
        """A registered tool resolves from the registry with no warning.

        :param caplog: pytest log-capture fixture.
        """
        with caplog.at_level("WARNING"):
            _baseline_family("superforcaster")
        assert "not in TOOL_REGISTRY" not in caplog.text


def _sf_prompt(question: str = "Will X?", today: str = "2026-04-22") -> str:
    """Minimal superforcaster IPFS prompt (Question/<background> layout)."""
    return (
        f"Question:\n{question}\n\nToday's date: {today}\n"
        f"<background>Some background.</background>"
    )


def _rag_prompt(question: str = "Will X?") -> str:
    """Minimal RAG-family IPFS prompt (XML-tagged, not backtick-fenced)."""
    return (
        f"<user_prompt>{question}</user_prompt>\n"
        f"<additional_information>Some info.</additional_information>"
    )


def _default_prompt(question: str = "Will X?") -> str:
    """Minimal prediction-online / SME IPFS prompt (backtick-fenced)."""
    return (
        f"USER_PROMPT:\n```\n{question}\n```\n\n"
        f"ADDITIONAL_INFORMATION:\n```\nSome info.\n```"
    )


def _reasoning_prompt(question: str = "Will X?") -> str:
    """Minimal two-stage reasoning IPFS prompt (reasoning ////  prediction)."""
    reasoning_half = (
        f"Here is the user's question: {question}\n"
        f"Here is some additional information "
        f"<additional_information>Some info.</additional_information>"
    )
    prediction_half = f"<user_input>{question}</user_input>"
    return f"{reasoning_half}////{prediction_half}"


class TestExtractPromptComponentsDispatch:
    """`extract_prompt_components` routes sibling baselines to the right extractor.

    Regression guard for the #321 review's unresolved 🔴 (comment 3364080553):
    the enrich-time dispatch used exact-name matches (`== "superforcaster"`,
    `== "factual_research"`), so a sibling baseline like
    ``superforcaster-polymarket-v1`` or ``factual_research-v2`` fell through to
    the default backtick regex, returned None, and every row was silently
    dropped (``Enriched 0/N``). Now both this and the replay path share
    `_baseline_family`, so siblings parse.
    """

    def test_superforcaster_sibling_parses_not_none(self) -> None:
        """A ``superforcaster-*`` baseline parses via the superforcaster extractor.

        Under the old exact-match dispatch this hit the default backtick regex
        and returned None — the 0/140 enrichment failure.
        """
        out = extract_prompt_components(
            _sf_prompt(question="Will it rain?"),
            tool_name="superforcaster-polymarket-v1",
        )
        assert out is not None
        assert out["user_prompt"] == "Will it rain?"

    def test_factual_research_sibling_parses_not_none(self) -> None:
        """A ``factual_research-*`` baseline parses via the FR extractor."""
        out = extract_prompt_components(
            _fr_prompt(question="Will it snow?", today="2026-04-22", briefing="B"),
            tool_name="factual_research-v2",
        )
        assert out is not None
        assert out["user_prompt"] == "Will it snow?"

    def test_url_cot_parses_via_rag_extractor(self) -> None:
        """``prediction-url-cot`` is rag-shaped, not default backtick format."""
        out = extract_prompt_components(
            _rag_prompt(question="Will it hail?"),
            tool_name="prediction-url-cot",
        )
        assert out is not None
        assert out["user_prompt"] == "Will it hail?"

    def test_sme_parses_via_default_extractor(self) -> None:
        """``*-sme`` is default-schema: backtick USER_PROMPT, not superforcaster."""
        out = extract_prompt_components(
            _default_prompt(question="Will it freeze?"),
            tool_name="prediction-online-sme",
        )
        assert out is not None
        assert out["user_prompt"] == "Will it freeze?"

    def test_reasoning_parses_via_two_stage_extractor(self) -> None:
        """Reasoning baselines parse via the ``////``-split extractor."""
        out = extract_prompt_components(
            _reasoning_prompt(question="Will it thaw?"),
            tool_name="prediction-request-reasoning",
        )
        assert out is not None
        assert out["user_prompt"] == "Will it thaw?"
        assert out["user_input"] == "Will it thaw?"


# Per-family contract the replay path relies on: which module attribute(s) hold
# the user-facing template(s) and which kwargs format() them. Each family maps
# to a list so multi-template families (reasoning: PREDICTION_PROMPT +
# REASONING_PROMPT) are covered. Mirrors the dispatch in replay() — keep in sync.
_FAMILY_TEMPLATES = {
    "reasoning": [
        ("PREDICTION_PROMPT", {"USER_INPUT": "q", "REASONING": "r"}),
        ("REASONING_PROMPT", {"USER_PROMPT": "q", "ADDITIONAL_INFOMATION": "a"}),
    ],
    "rag": [("PREDICTION_PROMPT", {"USER_PROMPT": "q", "ADDITIONAL_INFORMATION": "a"})],
    "superforcaster": [
        # ``market_prior`` is required by superforcaster-polymarket-v4 (the
        # market-price anchor block, filled by the replay path via the module's
        # format_market_prior) and tolerated as an unused placeholder by v1/v2/v3,
        # so one fixture covers all four — mirroring how the replay path passes
        # market_prior= to every superforcaster tool.
        (
            "PREDICTION_PROMPT",
            {"question": "q", "today": "t", "sources": "s", "market_prior": "m"},
        ),
    ],
    "factual_research": [
        # ``resolution_rules`` is required by factual_research-v2 (Polymarket
        # rules block) and tolerated as an unused placeholder by the parent
        # + v1 templates, so one fixture covers all three.
        (
            "ESTIMATE_USER",
            {"question": "q", "today": "t", "briefing": "b", "resolution_rules": "r"},
        ),
    ],
    "default": [
        ("PREDICTION_PROMPT", {"user_prompt": "q", "additional_information": "a"}),
    ],
}

# Non-template symbols the replay path reads per family (beyond the templates
# above). Checked for existence so a missing symbol fails loudly here.
_FAMILY_EXTRA_SYMBOLS = {
    "reasoning": ("parser_reasoning_response", "SYSTEM_PROMPT"),
    "rag": ("SYSTEM_PROMPT",),
}


class TestRegistryFamilyFormatRoundTrip:
    """Every registered tool's module satisfies its declared family's contract.

    Closes the #321 review test-gap (comment 3364080586): ``TestBaselineFamily``
    only checked the string→family mapping, never that the chosen family's
    ``.format(**kwargs)`` actually succeeds against the real module. That gap is
    why the SME ``KeyError`` (comment 3364080538) stayed green — SME was routed
    to superforcaster, whose ``format(question=, today=, sources=)`` blows up on
    SME's ``{user_prompt}``/``{additional_information}`` template. This test
    imports each module and runs the round-trip, so a mis-declared family fails.
    """

    @pytest.mark.parametrize("tool_name", sorted(TOOL_REGISTRY))
    def test_family_template_formats_with_family_kwargs(self, tool_name: str) -> None:
        """The family's template(s) exist and format with the family kwargs.

        Every template the replay path formats is exercised — including both
        stages of the reasoning family — so a placeholder added to a template
        without a matching family kwarg fails here instead of at replay time.

        :param tool_name: a registered tool name.
        """
        spec = TOOL_REGISTRY[tool_name]
        module = importlib.import_module(spec.module)

        for attr, kwargs in _FAMILY_TEMPLATES[spec.family]:
            template = getattr(module, attr)
            # Must not raise KeyError/IndexError on the family's kwargs.
            template.format(**kwargs)

        for attr in _FAMILY_EXTRA_SYMBOLS.get(spec.family, ()):
            assert hasattr(module, attr), f"{tool_name} missing {attr}"

    def test_default_family_system_prompt_is_optional(self) -> None:
        """SME (default family) ships no SYSTEM_PROMPT_FORECASTER; replay tolerates it.

        Pins the getattr fallback in replay() so default-family tools without
        the forecaster system prompt don't AttributeError on import.
        """
        module = importlib.import_module(TOOL_REGISTRY["prediction-online-sme"].module)
        assert not hasattr(module, "SYSTEM_PROMPT_FORECASTER")


class TestDefaultFamilySystemPrompt:
    """`_default_family_system_prompt` — the testable system-prompt selector.

    Positive coverage for the getattr fallback (Ojus #321 review, comment on
    prompt_replay.py:1399): asserting the attribute is *absent* on SME doesn't
    prove the fallback string is actually returned. If someone reverts the
    helper to a hard ``candidate_module.SYSTEM_PROMPT_FORECASTER`` access, the
    second case below raises AttributeError and this test fails.
    """

    def test_uses_forecaster_prompt_when_present(self) -> None:
        """When the module exports SYSTEM_PROMPT_FORECASTER, that value is used."""
        module = SimpleNamespace(SYSTEM_PROMPT_FORECASTER="Custom forecaster prompt.")
        assert _default_family_system_prompt(module) == "Custom forecaster prompt."

    def test_falls_back_when_absent(self, caplog: Any) -> None:
        """When the symbol is missing (e.g. SME), the generic fallback is used.

        The substitution is logged at WARNING so a missing/typo'd symbol on a
        non-SME default-family tool is visible.

        :param caplog: pytest log-capture fixture.
        """
        module = SimpleNamespace()
        with caplog.at_level("WARNING"):
            result = _default_family_system_prompt(module)
        assert result == DEFAULT_REPLAY_SYSTEM_PROMPT
        assert "SYSTEM_PROMPT_FORECASTER" in caplog.text
