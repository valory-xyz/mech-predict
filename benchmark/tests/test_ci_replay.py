# -*- coding: utf-8 -*-
"""Tests for parse-reliability metrics and rendering in ci_replay."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.ci_replay import (
    PARSE_STATUS_BUCKETS,
    _compute_parse_reliability,
    _format_market_diagnostics_block,
    _format_reliability_block,
    _load_filter_stats,
    compute_market_diagnostics,
    compute_metrics,
    format_report,
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
        assert "Candidate parse rate: 5/5 (100.0%) ✅" in text
        assert "Production parse rate: 5/5 (100.0%)" in text
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
        assert "Candidate parse rate: 7/10 (70.0%) ⚠️" in text
        assert "malformed=3" in text

    def test_not_valid_parse_is_not_a_warning(self) -> None:
        """Production deliveries enrich dropped for not parsing are normal noise.

        They land in the *rejected* bucket because the filter correctly
        excluded them — that is what keeps the scored baseline 100% valid — so
        a non-zero count must not raise a warning. It only lowers the reported
        production parse rate.
        """
        candidate = self._metrics([_row("valid")] * 5)
        lines = _format_reliability_block(
            candidate, [], _stats(accepted=5, not_valid_parse=2)
        )
        text = "\n".join(lines)
        assert "⚠️" not in text
        # 5 / (5 + 2) = 71.4%
        assert "Production parse rate: 5/7 (71.4%)" in text

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
        # Scoping rejections do not affect the production parse rate either.
        assert "Production parse rate: 5/5 (100.0%)" in text

    def test_production_parse_rate_rendered_with_rejections(self) -> None:
        """Render ``accepted/(accepted+not_valid_parse)`` as a percentage.

        The scored baseline is 100% valid by construction (enrich drops the
        non-parseable rows). The pre-drop ratio is what tells reviewers how
        noisy production actually was.
        """
        candidate = self._metrics([_row("valid")] * 100)
        lines = _format_reliability_block(
            candidate, [], _stats(accepted=100, not_valid_parse=35)
        )
        text = "\n".join(lines)
        assert "Production parse rate: 100/135 (74.1%)" in text

    def test_production_parse_rate_at_100_when_no_parse_rejections(self) -> None:
        """Zero not_valid_parse rejections: rate is 100%, but still rendered.

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
        assert "Production parse rate: 50/50 (100.0%)" in text

    def test_production_parse_rate_omitted_when_no_sidecar(self) -> None:
        """Older pipelines (no filter_stats) don't render the production rate line."""
        candidate = self._metrics([_row("valid")] * 5)
        text = "\n".join(_format_reliability_block(candidate, [], None))
        assert "Production parse rate" not in text

    def test_prefilter_omitted_when_stats_none(self) -> None:
        """Older pipelines (no sidecar) render without the production parse line."""
        candidate = self._metrics([_row("valid")] * 3)
        text = "\n".join(_format_reliability_block(candidate, [], None))
        assert "Production parse rate" not in text
        assert "Candidate parse rate" in text

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
        assert "**Reliability**" in report
        assert "| Metric |" in report
        assert report.index("| Metric |") < report.index("**Reliability**")

    def test_report_renders_without_failures_or_filter_stats(self) -> None:
        """format_report works on its minimal arg set (older pipelines)."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(baseline, candidate, {"tool": "superforcaster"})
        assert "Candidate parse rate: 3/3 (100.0%) ✅" in report
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

    def test_report_renders_production_parse_rate_when_stats_provided(self) -> None:
        """A filter_stats dict adds the Production parse rate to the reliability block."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(
            baseline,
            candidate,
            {"tool": "superforcaster"},
            filter_stats=_stats(accepted=3, wrong_tool=10, no_outcome=2),
        )
        # Scoping rejections don't enter the ratio or raise a warning.
        assert "Production parse rate: 3/3 (100.0%)" in report
        assert "⚠️" not in report

    def test_not_valid_parse_lowers_rate_without_warning_in_full_report(self) -> None:
        """not_valid_parse rows lower the production rate but never raise ⚠️."""
        baseline = compute_metrics([_row("valid")] * 3)
        candidate = compute_metrics([_row("valid")] * 3)
        report = format_report(
            baseline,
            candidate,
            {"tool": "superforcaster"},
            filter_stats=_stats(accepted=3, not_valid_parse=1),
        )
        assert "Production parse rate: 3/4 (75.0%)" in report
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
        """Footer parts stay in the existing order: deliveries → seed → triggered-by."""
        report = self._report(
            seed="1337",
            triggered_by="LOCKhart07",
            trigger_comment_url="https://example.com/c",
        )
        footer = report.splitlines()[-1]
        assert footer.index("deliveries") < footer.index("seed 1337")
        assert footer.index("seed 1337") < footer.index("LOCKhart07")


def _pair(
    base_p: float | None,
    cand_p: float | None,
    market: float | None,
    outcome: bool,
) -> tuple[dict, dict]:
    """Build a paired (baseline_row, candidate_row) for diagnostics tests."""
    base = {
        "platform": "polymarket",
        "tool_name": "superforcaster-polymarket-v1",
        "p_yes": base_p,
        "p_no": None if base_p is None else 1 - base_p,
        "prediction_parse_status": "valid",
        "market_prob": market,
        "final_outcome": outcome,
    }
    cand = {
        "platform": "polymarket",
        "tool_name": "superforcaster-polymarket-v1",
        "p_yes": cand_p,
        "p_no": None if cand_p is None else 1 - cand_p,
        "prediction_parse_status": "valid" if cand_p is not None else "malformed",
        "market_prob": market,
        "final_outcome": outcome,
    }
    return base, cand


def _diag(b_rows: list[dict], c_rows: list[dict]) -> dict:
    """compute_market_diagnostics with a non-None assertion (mypy narrowing)."""
    result = compute_market_diagnostics(b_rows, c_rows)
    assert result is not None
    return result


class TestMarketDiagnostics:
    """`compute_market_diagnostics` — blend arm and edge-given-up arm."""

    def test_none_when_no_market_price(self) -> None:
        """Rows without any market_prob yield None (block is omitted)."""
        b, c = _pair(0.6, 0.5, None, True)
        assert compute_market_diagnostics([b], [c]) is None

    def test_blend_briers_hand_computed(self) -> None:
        """Baseline/blend/candidate Briers match a hand calculation."""
        # base=0.2, market=0.6 -> blend=0.4; outcome=YES(1).
        b, c = _pair(0.2, 0.5, 0.6, True)
        diag = compute_market_diagnostics([b], [c])
        assert diag is not None
        blend = diag["blend"]
        assert blend["n"] == 1
        assert abs(blend["baseline_brier"] - 0.64) < 1e-9  # (0.2-1)^2
        assert abs(blend["blend_brier"] - 0.36) < 1e-9  # (0.4-1)^2
        assert abs(blend["candidate_brier"] - 0.25) < 1e-9  # (0.5-1)^2

    def test_candidate_parse_fail_excluded_from_blend(self) -> None:
        """A candidate with p_yes=None contributes nothing to the blend arm."""
        b, c = _pair(0.2, None, 0.6, True)
        diag = compute_market_diagnostics([b], [c])
        assert diag is not None
        assert diag["blend"]["n"] == 0
        assert diag["blend"]["candidate_brier"] is None

    def test_edge_subset_lost(self) -> None:
        """Baseline right vs market, candidate flips to wrong -> edge lost."""
        # base=0.7 (YES, right), market=0.3 (NO), outcome=YES; cand=0.4 (NO).
        b, c = _pair(0.7, 0.4, 0.3, True)
        edge = _diag([b], [c])["edge"]
        assert edge["n_disagree_right"] == 1
        assert edge["n_scored"] == 1
        assert edge["n_lost"] == 1
        assert edge["lost_rate"] == 1.0
        assert abs(edge["baseline_brier"] - 0.09) < 1e-9  # (0.7-1)^2
        assert abs(edge["candidate_brier"] - 0.36) < 1e-9  # (0.4-1)^2

    def test_edge_subset_kept(self) -> None:
        """Candidate that stays on the winning side does not lose the edge."""
        # base=0.7 right vs market 0.3; cand=0.8 still YES -> kept.
        b, c = _pair(0.7, 0.8, 0.3, True)
        edge = _diag([b], [c])["edge"]
        assert edge["n_disagree_right"] == 1
        assert edge["n_lost"] == 0
        assert edge["lost_rate"] == 0.0

    def test_baseline_wrong_disagreement_not_in_edge_subset(self) -> None:
        """Disagreement where the baseline was wrong is not edge to give up."""
        # base=0.7 (YES) vs market 0.3 (NO), but outcome=NO -> baseline wrong.
        b, c = _pair(0.7, 0.4, 0.3, False)
        edge = _diag([b], [c])["edge"]
        assert edge["n_disagree_right"] == 0

    def test_agreement_not_in_edge_subset(self) -> None:
        """When baseline agrees with the market, there is no edge to give up."""
        # base=0.7 and market=0.6 both YES -> agreement.
        b, c = _pair(0.7, 0.4, 0.6, True)
        edge = _diag([b], [c])["edge"]
        assert edge["n_disagree_right"] == 0

    def test_tie_excluded_from_edge_subset(self) -> None:
        """A baseline or market at exactly 0.5 carries no direction."""
        b, c = _pair(0.5, 0.4, 0.3, True)
        assert _diag([b], [c])["edge"]["n_disagree_right"] == 0
        b2, c2 = _pair(0.7, 0.4, 0.5, True)
        assert _diag([b2], [c2])["edge"]["n_disagree_right"] == 0

    def test_rows_paired_by_index(self) -> None:
        """Index i on both sides is the same market; mixed rows aggregate."""
        rows = [
            _pair(0.2, 0.5, 0.6, True),  # blend row
            _pair(0.7, 0.4, 0.3, True),  # edge-lost row
        ]
        b_rows = [b for b, _ in rows]
        c_rows = [c for _, c in rows]
        diag = _diag(b_rows, c_rows)
        assert diag["blend"]["n"] == 2
        assert diag["edge"]["n_disagree_right"] == 1
        assert diag["edge"]["n_lost"] == 1


class TestMarketDiagnosticsRendering:
    """`_format_market_diagnostics_block` verdict text reflects the numbers."""

    def test_beats_blend_marked(self) -> None:
        """Candidate Brier below the blend renders the ✅ beats verdict."""
        b, c = _pair(0.2, 0.5, 0.6, True)  # candidate 0.25 < blend 0.36
        block = "\n".join(_format_market_diagnostics_block(_diag([b], [c])))
        assert "✅" in block and "beats the blend" in block

    def test_does_not_beat_blend_marked(self) -> None:
        """Candidate Brier above the blend renders the ⚠️ warning."""
        # base=0.2, market=0.6 -> blend 0.36; cand=0.1 -> (0.1-1)^2=0.81 > blend.
        b, c = _pair(0.2, 0.1, 0.6, True)
        block = "\n".join(_format_market_diagnostics_block(_diag([b], [c])))
        assert "⚠️" in block and "does **not** beat" in block

    def test_edge_lost_count_rendered(self) -> None:
        """The edge-given-up line shows the surrendered count."""
        b, c = _pair(0.7, 0.4, 0.3, True)
        block = "\n".join(_format_market_diagnostics_block(_diag([b], [c])))
        assert "Edge given up" in block and "1/1" in block

    def test_no_edge_rows_message(self) -> None:
        """With no correct-disagreement rows, the 'nothing to surrender' note shows."""
        b, c = _pair(0.7, 0.4, 0.6, True)  # agreement, so empty edge subset
        block = "\n".join(_format_market_diagnostics_block(_diag([b], [c])))
        assert "nothing to surrender" in block

    def test_block_present_in_full_report(self) -> None:
        """format_report includes the block only when diagnostics are passed."""
        b, c = _pair(0.2, 0.5, 0.6, True)
        diag = compute_market_diagnostics([b], [c])
        base_m = compute_metrics([b])
        cand_m = compute_metrics([c])
        with_block = format_report(
            base_m, cand_m, {"tool": "t"}, market_diagnostics=diag
        )
        without = format_report(base_m, cand_m, {"tool": "t"})
        assert "Market-anchor diagnostics" in with_block
        assert "Market-anchor diagnostics" not in without
