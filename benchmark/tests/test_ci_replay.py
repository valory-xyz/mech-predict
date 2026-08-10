# -*- coding: utf-8 -*-
"""Tests for parse-reliability metrics and rendering in ci_replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from benchmark import ci_replay
from benchmark.ci_replay import (
    PARSE_STATUS_BUCKETS,
    _compute_parse_reliability,
    _format_reliability_block,
    _load_filter_stats,
    _metrics_table,
    compute_metrics,
    format_report,
    pair_arms,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL. Same construction as the sibling test module.

    :param path: destination.
    :param rows: rows to serialise.
    """
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _row(status: str = "valid", p_yes: float | None = 0.6) -> dict:
    return {
        "platform": "omen",
        "tool_name": "superforcaster",
        "p_yes": p_yes,
        "p_no": None if p_yes is None else 1 - p_yes,
        "prediction_parse_status": status,
        "final_outcome": True,
    }


class TestParseReliability:
    """Bucket counts and rates from ``prediction_parse_status`` values."""

    def test_breakdown_keys_are_always_present(self) -> None:
        """All four buckets are present even when only 'valid' has entries."""
        rel = _compute_parse_reliability([_row("valid")])
        assert set(rel["breakdown"]) == set(PARSE_STATUS_BUCKETS)

    def test_mixed_statuses_counted(self) -> None:
        """Mixed statuses (valid, malformed, missing_fields) land in the right buckets."""
        rows = [
            _row("valid"),
            _row("valid"),
            _row("malformed", None),
            _row("missing_fields", None),
        ]
        rel = _compute_parse_reliability(rows)
        assert rel == {
            "total": 4,
            "valid": 2,
            "parse_rate": 0.5,
            "breakdown": {
                "valid": 2,
                "missing_fields": 1,
                "malformed": 1,
                "error": 0,
            },
        }

    def test_unknown_status_bucketed_as_error(self) -> None:
        """An unexpected status value must not lose the row."""
        rel = _compute_parse_reliability([_row("weird_new_status", None)])
        assert rel["breakdown"]["error"] == 1
        assert rel["total"] == 1

    def test_compute_metrics_embeds_parse_reliability(self) -> None:
        """compute_metrics must surface parse_reliability alongside Brier."""
        metrics = compute_metrics([_row("valid"), _row("malformed", None)])
        assert "parse_reliability" in metrics
        assert metrics["parse_reliability"]["parse_rate"] == 0.5


def _stats(
    *,
    source: str = "production",
    accepted: int = 100,
    not_valid_parse: int = 0,
    duplicate: int = 0,
    wrong_tool: int = 0,
    wrong_platform: int = 0,
    no_deliver_id: int = 0,
    no_outcome: int = 0,
    older_than_cutoff: int = 0,
) -> dict:
    return {
        "source": source,
        "accepted": accepted,
        "rejected": {
            "duplicate": duplicate,
            "wrong_tool": wrong_tool,
            "wrong_platform": wrong_platform,
            "no_deliver_id": no_deliver_id,
            "not_valid_parse": not_valid_parse,
            "no_outcome": no_outcome,
            "older_than_cutoff": older_than_cutoff,
        },
    }


class TestReliabilityBlock:
    """Markdown rendering of the Reliability section (candidate + pre-filter)."""

    def _metrics(self, rows: list[dict]) -> dict:
        return compute_metrics(rows)

    def test_green_happy_path_is_two_lines(self) -> None:
        """All-valid candidate + clean pre-filter → two bullets, no scoping noise."""
        candidate = self._metrics([_row("valid")] * 5)
        lines = _format_reliability_block(candidate, [], _stats(accepted=5))
        text = "\n".join(lines)
        assert "Candidate returned no usable prediction on 0 of 5 markets ✅" in text
        assert "Drawn from 5 usable production deliveries." in text
        # Breakdown is hidden when candidate is at 100%.
        assert "Breakdown:" not in text
        # Scoping rejections are expected noise and never reported.
        assert "Scoping:" not in text
        assert "wrong_tool" not in text
        assert "⚠️" not in text

    def test_candidate_drift_flags_and_shows_breakdown(self) -> None:
        """Candidate < 100% → ⚠️ + breakdown line exposes which buckets failed."""
        candidate = self._metrics([_row("valid")] * 7 + [_row("malformed", None)] * 3)
        lines = _format_reliability_block(candidate, [], _stats(accepted=10))
        text = "\n".join(lines)
        assert "Candidate returned no usable prediction on 3 of 10 markets ⚠️" in text
        assert "malformed=3" in text

    def test_not_valid_parse_is_not_a_warning(self) -> None:
        """Production deliveries enrich dropped for not parsing are normal noise.

        They land in the *rejected* bucket because the filter correctly
        excluded them — that is what keeps the scored baseline 100% valid — so
        a non-zero count must not raise a warning. It only shrinks the pool
        the sample was drawn from.
        """
        candidate = self._metrics([_row("valid")] * 5)
        lines = _format_reliability_block(
            candidate, [], _stats(accepted=5, not_valid_parse=2)
        )
        text = "\n".join(lines)
        assert "⚠️" not in text
        assert "2 of 7 production deliveries were left out" in text

    def test_scoping_rejections_are_not_reported(self) -> None:
        """Scoping buckets (wrong_tool etc.) are rows for other tools — pure noise."""
        candidate = self._metrics([_row("valid")] * 5)
        lines = _format_reliability_block(
            candidate,
            [],
            _stats(
                accepted=5,
                duplicate=8,
                wrong_tool=12,
                wrong_platform=9,
                no_outcome=4,
                older_than_cutoff=7,
            ),
        )
        text = "\n".join(lines)
        # None of the scoping buckets leak into the report.
        for bucket in (
            "duplicate",
            "wrong_tool",
            "wrong_platform",
            "older_than_cutoff",
        ):
            assert bucket not in text
        assert "⚠️" not in text
        # Scoping rejections do not shrink the reported pool either.
        assert "Drawn from 5 usable production deliveries." in text

    def test_left_out_count_rendered_with_rejections(self) -> None:
        """Report how many deliveries were left out, and out of how many.

        The scored baseline is 100% valid by construction (enrich drops the
        non-parseable rows). The pre-drop count is what tells reviewers how
        noisy production actually was.
        """
        candidate = self._metrics([_row("valid")] * 100)
        lines = _format_reliability_block(
            candidate, [], _stats(accepted=100, not_valid_parse=35)
        )
        text = "\n".join(lines)
        assert "35 of 135 production deliveries were left out" in text

    def test_pool_size_rendered_when_no_parse_rejections(self) -> None:
        """Zero not_valid_parse rejections: the pool size still renders.

        Keeping it on the happy path means a later regression (non-zero
        ``not_valid_parse``) isn't a surprise line appearing out of nowhere.
        """
        candidate = self._metrics([_row("valid")] * 50)
        lines = _format_reliability_block(
            candidate,
            [],
            _stats(accepted=50, wrong_tool=3, no_outcome=1),
        )
        text = "\n".join(lines)
        assert "Drawn from 50 usable production deliveries." in text

    def test_sample_block_omitted_when_no_sidecar(self) -> None:
        """Older pipelines (no filter_stats) render no Sample block at all."""
        candidate = self._metrics([_row("valid")] * 5)
        text = "\n".join(_format_reliability_block(candidate, [], None))
        assert "**Sample**" not in text

    def test_prefilter_omitted_when_stats_none(self) -> None:
        """Older pipelines (no sidecar) render without the Sample block."""
        candidate = self._metrics([_row("valid")] * 3)
        text = "\n".join(_format_reliability_block(candidate, [], None))
        assert "**Sample**" not in text
        assert "Candidate returned no usable prediction" in text

    def test_failure_bodies_rendered_in_collapsed_details(self) -> None:
        """Failure bodies inline under a <details> so the PR comment stays tidy."""
        candidate = self._metrics([_row("valid"), _row("malformed", None)])
        failures = [
            {
                "row_id": "c2",
                "question_text": "Q2",
                "prediction_parse_status": "malformed",
                "raw_response": "<facts> leaked content here",
            }
        ]
        text = "\n".join(_format_reliability_block(candidate, failures, None))
        assert "<details>" in text
        assert "<facts> leaked content here" in text
        assert "malformed" in text

    def test_body_with_backticks_is_escaped(self) -> None:
        """Markdown code fences in the body can't break out of the outer ``` block."""
        candidate = self._metrics([_row("malformed", None)])
        failures = [
            {
                "row_id": "c1",
                "question_text": "Q",
                "prediction_parse_status": "malformed",
                "raw_response": 'some ```json{"p_yes":0.5}``` leak',
            }
        ]
        text = "\n".join(_format_reliability_block(candidate, failures, None))
        assert text.count("```") == 2


class TestLoadFilterStats:
    """Sidecar loading alongside candidate.jsonl."""

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        """Older pipelines without the sidecar return cleanly."""
        candidate = tmp_path / "candidate.jsonl"
        candidate.write_text("", encoding="utf-8")
        assert _load_filter_stats(candidate) is None

    def test_parses_sidecar_when_present(self, tmp_path: Path) -> None:
        """filter_stats.json in the same dir as candidate.jsonl is loaded."""
        candidate = tmp_path / "candidate.jsonl"
        candidate.write_text("", encoding="utf-8")
        stats = {"accepted": 5, "rejected": {"not_valid_parse": 1}}
        (tmp_path / "filter_stats.json").write_text(json.dumps(stats), encoding="utf-8")
        assert _load_filter_stats(candidate) == stats

    def test_returns_none_on_malformed_json(self, tmp_path: Path) -> None:
        """A corrupt sidecar must not crash the whole report."""
        candidate = tmp_path / "candidate.jsonl"
        candidate.write_text("", encoding="utf-8")
        (tmp_path / "filter_stats.json").write_text("{not json", encoding="utf-8")
        assert _load_filter_stats(candidate) is None


class TestFormatReportEndToEnd:
    """End-to-end ``format_report`` shape: table first, Reliability below."""

    def test_metrics_table_precedes_reliability_section(self) -> None:
        """Primary comparison (Brier etc.) must come before reliability checks."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(baseline, candidate, {"tool": "superforcaster"})
        assert "**Candidate health**" in report
        assert "| Metric |" in report
        assert report.index("| Metric |") < report.index("**Candidate health**")

    def test_report_renders_without_failures_or_filter_stats(self) -> None:
        """format_report works on its minimal arg set (older pipelines)."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(baseline, candidate, {"tool": "superforcaster"})
        assert "Candidate returned no usable prediction on 0 of 3 markets ✅" in report
        assert "Pre-filter" not in report
        assert "<details><summary>Candidate parse failures" not in report

    def test_report_embeds_failure_bodies_when_provided(self) -> None:
        """format_report threads the failure_rows arg into the reliability section."""
        baseline = compute_metrics([_row("valid")] * 2)
        candidate = compute_metrics([_row("valid"), _row("error", None)])
        failures = [
            {
                "row_id": "c2",
                "question_text": "Q2",
                "prediction_parse_status": "error",
                "raw_response": "Boom.",
            }
        ]
        report = format_report(
            baseline,
            candidate,
            {"tool": "superforcaster"},
            failure_rows=failures,
        )
        assert "Boom." in report
        assert "⚠️" in report

    def test_report_renders_sample_block_when_stats_provided(self) -> None:
        """A filter_stats dict adds the Sample provenance block to the report."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(
            baseline,
            candidate,
            {"tool": "superforcaster"},
            filter_stats=_stats(accepted=3, wrong_tool=10, no_outcome=2),
        )
        # Scoping rejections don't shrink the pool or raise a warning.
        assert "Drawn from 3 usable production deliveries." in report
        assert "⚠️" not in report

    def test_not_valid_parse_lowers_rate_without_warning_in_full_report(self) -> None:
        """not_valid_parse rows shrink the pool but never raise ⚠️."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(
            baseline,
            candidate,
            {"tool": "superforcaster"},
            filter_stats=_stats(accepted=3, not_valid_parse=1),
        )
        assert "1 of 4 production deliveries were left out" in report
        assert "⚠️" not in report


class TestFormatReportHeader:
    """Header must name the platform so a scoped run's scope is visible.

    A run scoped to one platform (``--benchmark ... --platform polymarket``)
    suppresses the per-platform breakdown, so the header is the only place the
    scope appears.
    """

    @staticmethod
    def _prow(platform: str) -> dict:
        return {**_row("valid"), "platform": platform}

    def test_single_platform_named_in_header(self) -> None:
        """A run scoped to one platform names it in the header."""
        rows = [self._prow("polymarket")] * 3
        report = format_report(
            compute_metrics(rows), compute_metrics(rows), {"tool": "superforcaster"}
        )
        assert "## Benchmark: superforcaster — Polymarket" in report

    def test_multiple_platforms_render_all_platforms(self) -> None:
        """An unscoped run spanning platforms reads 'All platforms'."""
        rows = [self._prow("omen"), self._prow("polymarket")]
        report = format_report(
            compute_metrics(rows), compute_metrics(rows), {"tool": "superforcaster"}
        )
        assert "## Benchmark: superforcaster — All platforms" in report

    def test_header_platform_set_unions_baseline_and_candidate(self) -> None:
        """The platform set is the union of baseline and candidate platforms.

        A platform present on only one side still counts, so a baseline/
        candidate platform mismatch reads 'All platforms', not a single name.
        """
        cand = [self._prow("polymarket")] * 3
        report = format_report(
            compute_metrics([_row("valid")] * 3),  # omen baseline
            compute_metrics(cand),
            {"tool": "superforcaster"},
        )
        assert "## Benchmark: superforcaster — All platforms" in report


class TestFormatReportFooter:
    """Footer must make multi-seed runs distinguishable and auditable.

    Reviewers will trigger ``/benchmark`` repeatedly with different seeds;
    without seed + trigger-comment attribution in the footer, the resulting
    PR comments are visually indistinguishable and the parameters that
    produced each one are invisible. Per PR #231 review (#233).
    """

    def _report(self, **meta: str) -> str:
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        full_meta: dict[str, str] = {"tool": "superforcaster", **meta}
        return format_report(baseline, candidate, full_meta)

    def test_footer_includes_seed_when_provided(self) -> None:
        """`meta["seed"]` must appear as `seed <N>` in the footer."""
        report = self._report(seed="1337")
        assert "seed 1337" in report.splitlines()[-1]

    def test_footer_omits_seed_when_absent(self) -> None:
        """Without a seed, no seed label — otherwise older callers emit 'seed None'."""
        report = self._report()
        assert "seed" not in report.splitlines()[-1]

    def test_footer_links_triggered_by_when_comment_url_provided(self) -> None:
        """trigger_comment_url turns the `@user` mention into a markdown link.

        Reviewers need to jump from a benchmark comment to the exact
        ``/benchmark`` request that produced it (params, author, time).
        """
        report = self._report(
            triggered_by="LOCKhart07",
            trigger_comment_url="https://github.com/valory-xyz/mech-predict/pull/231#issuecomment-4263150118",
        )
        expected = (
            "triggered by [@LOCKhart07]"
            "(https://github.com/valory-xyz/mech-predict/pull/231#issuecomment-4263150118)"
        )
        assert expected in report.splitlines()[-1]

    def test_footer_plain_triggered_by_without_comment_url(self) -> None:
        """No URL → plain `@user` mention (backwards compatible)."""
        report = self._report(triggered_by="LOCKhart07")
        footer = report.splitlines()[-1]
        assert "@LOCKhart07" in footer
        assert "[@LOCKhart07]" not in footer

    def test_footer_order_deliveries_seed_triggered_by(self) -> None:
        """Footer parts stay in the existing order: scored → seed → triggered-by."""
        report = self._report(
            seed="1337",
            triggered_by="LOCKhart07",
            trigger_comment_url="https://example.com/c",
        )
        footer = report.splitlines()[-1]
        assert footer.index("scored") < footer.index("seed 1337")
        assert footer.index("seed 1337") < footer.index("LOCKhart07")


class TestTournamentSourcedComment:
    """The PR comment must not present tournament rows as production data.

    The sidecar already records which arm the rows came from; before this,
    ci_replay never read it, so a fallback run rendered "Baseline (prod)" and
    a "Production parse rate" computed entirely from tournament rows -- the
    one artifact a reviewer actually acts on.
    """

    def _metrics(self, rows: list[dict]) -> dict:
        """Metrics for a row list.

        :param rows: replay rows.
        :return: computed metrics.
        """
        return compute_metrics(rows)

    def test_tournament_run_is_labelled_not_prod(self) -> None:
        """The metrics table header must say tournament, not prod."""
        m = self._metrics([_row("valid")] * 5)
        assert "Baseline (prod)" in _metrics_table(m, m, "production")
        assert "Baseline (tourn)" in _metrics_table(m, m, "tournament")

    def test_tournament_run_reports_no_production_parse_rate(self) -> None:
        """A production parse rate off tournament rows is meaningless."""
        candidate = self._metrics([_row("valid")] * 5)
        stats = _stats(source="tournament", accepted=5)
        stats["production_attempt"] = {"accepted": 0, "rejected": {}}
        stats["fallback_reason"] = "0 production rows"
        text = "\n".join(_format_reliability_block(candidate, [], stats))
        assert (
            "Production parse rate" not in text
        ), "captioned tournament rows as a production parse rate"
        assert "tournament arm" in text
        assert "not a production delivery" in text

    def test_production_run_is_unchanged(self) -> None:
        """The production path must keep its existing rendering."""
        candidate = self._metrics([_row("valid")] * 5)
        text = "\n".join(_format_reliability_block(candidate, [], _stats(accepted=5)))
        assert "Drawn from 5 usable production deliveries." in text
        assert "tournament arm" not in text


def _arm_pair(n: int, candidate_fails_on: set[int]) -> tuple[list[dict], list[dict]]:
    """Build index-aligned baseline/candidate arms.

    The baseline is always usable (replay stamps it valid before the
    candidate LLM runs); the candidate returns nothing on the given
    indices, and is deliberately WORSE elsewhere so a drop-driven "win"
    is unambiguous.

    :param n: number of markets.
    :param candidate_fails_on: indices the candidate cannot answer.
    :return: (baseline rows, candidate rows).
    """
    base, cand = [], []
    for i in range(n):
        # The markets the candidate drops are the ones the baseline got
        # WORST (p_yes 0.0 against a True outcome).
        worst = i in candidate_fails_on
        base.append(
            {
                "platform": "omen",
                "tool_name": "t",
                "question_text": f"Q{i}?",
                "p_yes": 0.0 if worst else 0.6,
                "p_no": 1.0 if worst else 0.4,
                "prediction_parse_status": "valid",
                "final_outcome": True,
            }
        )
        cand.append(
            {
                "platform": "omen",
                "tool_name": "t",
                "question_text": f"Q{i}?",
                "p_yes": None if worst else 0.59,
                "p_no": None if worst else 0.41,
                "prediction_parse_status": "malformed" if worst else "valid",
                "final_outcome": True,
            }
        )
    return base, cand


class TestArmPairing:
    """Both arms must be scored over the SAME markets."""

    def test_baseline_is_not_scored_on_markets_the_candidate_dropped(self) -> None:
        """The dropped markets must leave the BASELINE average too.

        Otherwise the candidate wins by failing: it is scored on its own
        subset while the baseline carries the markets it could not answer.
        """
        base, cand = _arm_pair(100, candidate_fails_on={0, 1, 2})
        pb, pc = pair_arms(base, cand)
        assert len(pb) == len(pc) == 97
        # Every surviving baseline row is one the candidate also answered.
        assert all(r["p_yes"] == 0.6 for r in pb), "a dropped market survived"
        assert compute_metrics(pb)["n"] == compute_metrics(pc)["n"] == 97

    def test_dropping_the_hardest_markets_no_longer_buys_a_win(self) -> None:
        """The reported delta must not flip sign purely from candidate drops.

        Unpaired, the baseline carries three 0.0-vs-True markets the candidate
        never answered, so the candidate "improves" Brier. Paired, the
        candidate is strictly worse on identical markets and must show it.
        """
        base, cand = _arm_pair(100, candidate_fails_on={0, 1, 2})
        unpaired_delta = compute_metrics(cand)["brier"] - compute_metrics(base)["brier"]
        pb, pc = pair_arms(base, cand)
        paired_delta = compute_metrics(pc)["brier"] - compute_metrics(pb)["brier"]
        assert unpaired_delta < 0, "fixture no longer reproduces the false win"
        assert paired_delta > 0, "pairing must expose the candidate as worse"

    def test_caption_and_footer_report_the_paired_count(self) -> None:
        """`Computed on N` and the footer must both mean markets scored by both."""
        base, cand = _arm_pair(100, candidate_fails_on={0, 1, 2})
        pb, pc = pair_arms(base, cand)
        cand_metrics = compute_metrics(pc)
        cand_metrics["parse_reliability"] = compute_metrics(cand)["parse_reliability"]
        report = format_report(
            compute_metrics(pb),
            cand_metrics,
            {"tool": "t", "seed": "42", "triggered_by": "bot"},
            failure_rows=[],
            filter_stats=None,
        )
        assert "Computed on 97 markets" in report
        assert report.splitlines()[-1].strip("*").startswith("97 scored")
        # Health still reports against the FULL arm, not the paired subset.
        assert "no usable prediction on 3 of 100 markets" in report

    def test_truncated_arm_refuses_to_render(self) -> None:
        """A partial candidate flush must fail, not score a truncated pair.

        `zip` stops at the shorter arm, so without this a replay that died
        after writing some — but not all — candidate rows would silently drop
        the baseline tail and publish a plausible verdict over whatever
        happened to be flushed. The empty-candidate guard in `main()` only
        catches the zero case; this is the 1..n-1 case.
        """
        base, cand = _arm_pair(10, candidate_fails_on=set())
        with pytest.raises(SystemExit) as exc:
            pair_arms(base, cand[:4])
        msg = str(exc.value)
        assert "different lengths" in msg
        assert "10 vs 4" in msg
        assert "Refusing to compare" in msg

    def test_misaligned_arms_refuse_to_render(self) -> None:
        """Different markets at the same position must fail loudly, not silently."""
        base, cand = _arm_pair(3, candidate_fails_on=set())
        cand[1]["question_text"] = "a completely different market"
        with pytest.raises(SystemExit) as exc:
            pair_arms(base, cand)
        msg = str(exc.value)
        assert "misaligned" in msg
        assert "Refusing to compare" in msg
        assert "a completely different market" in msg, "name the conflict"


class TestSampleBlockNegativeCases:
    """Lines that must NOT render on a clean run."""

    def test_left_out_line_absent_when_nothing_was_left_out(self) -> None:
        """A clean pool must not render `0 of N ... left out`.

        The line is conditional; dropping the guard would print it on every
        healthy run, and every existing test passes a non-zero count.
        """
        candidate = compute_metrics([_row("valid")] * 5)
        text = "\n".join(_format_reliability_block(candidate, [], _stats(accepted=5)))
        assert "Drawn from 5 usable production deliveries." in text
        assert "left out" not in text

    def test_enrichment_losses_reach_the_reader(self) -> None:
        """Rows lost during IPFS enrichment must be attributable in the comment."""
        candidate = compute_metrics([_row("valid")] * 5)
        stats = _stats(accepted=50)
        stats["enrichment_failed"] = 7
        text = "\n".join(_format_reliability_block(candidate, [], stats))
        assert "7 sampled row(s) dropped during IPFS enrichment" in text

    def test_enrichment_line_absent_when_nothing_was_lost(self) -> None:
        """No enrichment loss, no line."""
        candidate = compute_metrics([_row("valid")] * 5)
        text = "\n".join(_format_reliability_block(candidate, [], _stats(accepted=50)))
        assert "IPFS enrichment" not in text


class TestTournamentRenderingCoverage:
    """Tournament-specific strings that previously had no test."""

    @staticmethod
    def _tstats(**rejected: int) -> dict:
        """Tournament sidecar with the given non-zero drop reasons.

        :param rejected: per-reason drop counts.
        :return: sidecar dict.
        """
        return {
            "source": "tournament",
            "accepted": 5,
            "rejected": dict(rejected),
            "no_row_id": 0,
            "production_attempt": {"accepted": 500, "rejected": {}},
            "fallback_reason": "0 production rows",
        }

    def test_discarded_line_renders_with_real_drop_counts(self) -> None:
        """`Tournament rows discarded` must render, sorted, when drops exist.

        Every prior tournament fixture defaulted each bucket to 0, so this
        branch was never entered under test.
        """
        candidate = compute_metrics([_row("valid")] * 5)
        text = "\n".join(
            _format_reliability_block(
                candidate,
                [],
                self._tstats(no_evidence=950, unrenderable=3, no_row_id=2),
            )
        )
        assert (
            "- Tournament rows discarded: no_evidence=950, no_row_id=2, "
            "unrenderable=3" in text
        )

    def test_discarded_line_absent_when_nothing_was_discarded(self) -> None:
        """No drops, no line."""
        candidate = compute_metrics([_row("valid")] * 5)
        text = "\n".join(_format_reliability_block(candidate, [], self._tstats()))
        assert "Tournament rows discarded" not in text

    def test_footer_unit_says_tournament_rows_scored(self) -> None:
        """The tournament footer unit had no test at all.

        `grep "tournament rows scored" benchmark/tests/` was empty, so the
        production/tournament split in the footer was unpinned.
        """
        rows = [
            {**_row("valid"), "question_text": f"Q{i}?", "platform": "polymarket"}
            for i in range(4)
        ]
        report = format_report(
            compute_metrics(rows),
            compute_metrics(rows),
            {"tool": "t", "seed": "42", "triggered_by": "bot"},
            failure_rows=[],
            filter_stats=self._tstats(),
        )
        assert report.splitlines()[-1].strip("*").startswith("4 tournament rows scored")

    def test_footer_count_is_the_candidate_denominator(self) -> None:
        """Footer must track the candidate's N, not the baseline's.

        With equal-sized arms in every prior fixture, reverting this to
        `baseline['n']` passed the whole suite.
        """
        base = [{**_row("valid"), "question_text": f"Q{i}?"} for i in range(9)]
        cand = [
            {
                **_row("valid" if i < 4 else "malformed", 0.6 if i < 4 else None),
                "question_text": f"Q{i}?",
            }
            for i in range(9)
        ]
        report = format_report(
            compute_metrics(base),
            compute_metrics(cand),
            {"tool": "t", "seed": "42", "triggered_by": "bot"},
            failure_rows=[],
            filter_stats=None,
        )
        assert (
            report.splitlines()[-1].strip("*").startswith("4 scored")
        ), "footer used the baseline denominator"


class TestMainRowGuards:
    """`main()` must refuse to render a verdict from a missing arm."""

    def _run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cand: list[dict]
    ) -> int:
        """Run ``main()`` with a full baseline and the given candidate rows.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        :param cand: candidate rows to write.
        :return: the SystemExit code (0 when it ran to completion).
        """

        base_p, cand_p = tmp_path / "baseline.jsonl", tmp_path / "candidate.jsonl"
        _write_jsonl(
            base_p,
            [{**_row("valid"), "question_text": f"Q{i}?"} for i in range(3)],
        )
        _write_jsonl(cand_p, cand)
        monkeypatch.setattr(
            sys,
            "argv",
            ["ci_replay", "--baseline", str(base_p), "--candidate", str(cand_p)],
        )
        try:
            ci_replay.main()
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    def test_empty_candidate_file_is_an_error_not_a_green_verdict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """A candidate that flushed no rows must fail, not render a tick.

        Reachable when replay dies after opening candidate.jsonl but before
        writing to it. Previously this rendered "no usable prediction on 0 of
        0 markets ✅" -- a clean pass over nothing.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        """
        code = self._run(tmp_path, monkeypatch, [])
        assert code == 1
        assert "No candidate rows" in capsys.readouterr().err

    def test_all_malformed_candidate_fails_the_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """A run that scored nothing must exit nonzero, not publish a verdict.

        The 2026-08-04 v5 run had 300 candidate rows -- every one malformed --
        and exited green with a posted report. The rendering was already made
        honest ("Computed on 0 markets"); this pins the exit code, which is
        what makes the run go red and the incomplete-benchmark notice fire.
        Keyed on the PAIRED count so a tournament-fallback run (zero
        production rows, real scored pairs) is untouched.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        """
        code = self._run(
            tmp_path,
            monkeypatch,
            [
                {
                    **_row("malformed", p_yes=None),
                    "question_text": f"Q{i}?",
                }
                for i in range(3)
            ],
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "0 of 3 candidate responses were usable" in err
        assert "malformed=3" in err
        assert "refusing to publish" in err.lower()

    def test_populated_candidate_renders(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """The guard must not reject a healthy run.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        """
        code = self._run(
            tmp_path,
            monkeypatch,
            [{**_row("valid"), "question_text": f"Q{i}?"} for i in range(3)],
        )
        assert code == 0
        assert "Computed on 3 markets" in capsys.readouterr().out

    def test_main_scores_both_arms_over_the_paired_markets(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """`main()` must actually USE the pairing, not merely have it available.

        Driven end-to-end rather than by calling ``pair_arms`` directly: a
        test that exercises the helper alone still passes when ``main()``
        scores each file independently, which is the bug.

        The fixture is the false win itself -- the candidate cannot answer the
        3 markets the baseline got worst, and is marginally worse everywhere
        else. Unpaired it reports an improvement; paired it must report a
        regression, over 97 markets rather than 100.

        :param tmp_path: pytest tmp_path fixture.
        :param monkeypatch: pytest monkeypatch fixture.
        :param capsys: pytest capsys fixture.
        """
        base, cand = _arm_pair(100, candidate_fails_on={0, 1, 2})
        base_p, cand_p = tmp_path / "baseline.jsonl", tmp_path / "candidate.jsonl"
        _write_jsonl(base_p, base)
        _write_jsonl(cand_p, cand)

        monkeypatch.setattr(
            sys,
            "argv",
            ["ci_replay", "--baseline", str(base_p), "--candidate", str(cand_p)],
        )
        ci_replay.main()
        out = capsys.readouterr().out
        assert "Computed on 97 markets" in out, "main() scored the arms unpaired"
        # Health must still describe the FULL arm. Sourcing it from the
        # paired subset would read "0 of 97" and delete the signal.
        assert "no usable prediction on 3 of 100 markets" in out
        # Locate the Brier columns via the table HEADER rather than fixed
        # offsets: hardcoding them makes an added column raise ValueError
        # (a table-formatting error) instead of failing as a regression.
        lines = out.splitlines()
        header = next(line for line in lines if "| Metric" in line)
        cols = [c.strip() for c in header.split("|")]
        # Match by PREFIX: the baseline label carries the row source
        # ("Baseline (prod)" / "Baseline (tourn)") and the candidate is
        # "Candidate (PR)".
        b_i = next(i for i, c in enumerate(cols) if c.startswith("Baseline"))
        c_i = next(i for i, c in enumerate(cols) if c.startswith("Candidate"))
        brier_row = next(line for line in lines if "Brier score" in line)
        cells = [c.strip() for c in brier_row.split("|")]
        b_val, c_val = float(cells[b_i]), float(cells[c_i])
        assert c_val > b_val, f"paired candidate must be worse, got {brier_row}"
