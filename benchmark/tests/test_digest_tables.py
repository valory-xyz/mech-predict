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
"""Tests for the computed benchmark digest tables.

Each test asserts a specific cell against a specific input field.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from benchmark.digest_tables import (
    TITLE_RULE_CHAR,
    build_digest_messages,
)


def _stats(**overrides: Any) -> dict[str, Any]:
    """Build a per-tool stats dict shaped like scorer output.

    :param overrides: fields to override on the default group.
    :return: stats dict.
    """
    base = {
        "n": 100,
        "valid_n": 100,
        "reliability": 0.98,
        "brier": 0.3000,
        "baseline_brier": 0.2400,
        "market_brier": 0.2000,
        "edge": -0.1000,
        "edge_n": 100,
    }
    base.update(overrides)
    return base


def _write(results: Path, name: str, by_tool: dict[str, Any]) -> None:
    """Write a scores file into the results directory.

    :param results: results directory.
    :param name: file name.
    :param by_tool: mapping of tool name to stats.
    """
    (results / name).write_text(json.dumps({"by_tool": by_tool}), encoding="utf-8")


def _results_dir(
    tmp_path: Path,
    at: dict[str, Any] | None = None,
    w1: dict[str, Any] | None = None,
    w2: dict[str, Any] | None = None,
    tournament: dict[str, Any] | None = None,
) -> Path:
    """Create a results directory, writing only the windows passed as non-None.

    :param tmp_path: pytest temp dir.
    :param at: all-time by_tool mapping.
    :param w1: rolling (W-1) by_tool mapping.
    :param w2: previous rolling (W-2) by_tool mapping.
    :param tournament: tournament by_tool mapping (default empty).
    :return: the results directory.
    """
    results = tmp_path / "r"
    results.mkdir()
    windows = {
        "scores_polymarket.json": at,
        "rolling_scores_polymarket.json": w1,
        "prev_rolling_scores_polymarket.json": w2,
    }
    for name, by_tool in windows.items():
        if by_tool is not None:
            _write(results, name, by_tool)
    _write(results, "scores_tournament_polymarket.json", tournament or {})
    return results


@pytest.fixture(name="results")
def _results(tmp_path: Path) -> Path:
    """Create a results directory with one production tool in all windows.

    :param tmp_path: pytest temp dir.
    :return: the results directory.
    """
    results = tmp_path / "results"
    results.mkdir()
    _write(results, "scores_polymarket.json", {"alpha": _stats()})
    _write(
        results,
        "rolling_scores_polymarket.json",
        {"alpha": _stats(brier=0.3500, market_brier=0.1800, edge=-0.1700)},
    )
    _write(
        results,
        "prev_rolling_scores_polymarket.json",
        {"alpha": _stats(brier=0.2800, market_brier=0.2500, edge=-0.0300)},
    )
    _write(results, "scores_tournament_polymarket.json", {})
    (results / "roi_results.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "tool_name": "alpha",
                        "mode": "production",
                        "platform": "polymarket",
                        "roi_mid": 15.3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return results


def _body(results: Path, **kwargs: Any) -> str:
    """Flatten every message's cells and prose into one searchable string.

    The builder returns Block Kit payloads, one per table; tests assert over
    the flattened text so they stay readable.

    :param results: results directory.
    :param kwargs: forwarded to build_digest_messages.
    :return: all cell and prose text, newline-joined.
    """
    return "\n".join(
        _flatten(m) for m in build_digest_messages(results, "polymarket", **kwargs)
    )


def _flatten(payload: dict[str, Any]) -> str:
    """Render one payload's text content as lines, one table row per line.

    :param payload: a webhook payload from the builder.
    :return: the payload's readable text.
    """
    lines = []
    for block in payload["blocks"]:
        if block["type"] == "table":
            for row in block["rows"]:
                lines.append(" | ".join(c["text"] for c in row))
        elif block["type"] == "section":
            lines.append(block["text"]["text"])
        elif block["type"] == "context":
            lines.extend(e["text"] for e in block["elements"])
    return "\n".join(lines)


def _cells(body: str, tool: str, after: str = "") -> list[str]:
    """Extract one tool's cells from the first table containing it.

    :param body: rendered digest body.
    :param tool: tool name.
    :param after: only search past this marker line, to pick a later table.
    :return: stripped cell strings.
    """
    text = body.split(after, 1)[-1] if after else body
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        # The decision tables lead with a rank cell, so the tool is not always
        # the first column.
        if cells and cells[0] == tool or (len(cells) > 1 and cells[1] == tool):
            return cells
    raise AssertionError(f"no row for {tool} after {after!r} in:\n{body}")


class TestMarketBrier:
    """``mkt`` must come from the stored field, never be reconstructed."""

    def test_mkt_differs_from_brier_plus_edge(self, tmp_path: Path) -> None:
        """When the edge pool differs from the valid pool, mkt is not brier+edge.

        ``brier`` averages over all valid rows while ``edge`` averages over the
        edge-eligible subset, so the identity only holds when the pools
        coincide. This fixture makes them differ and pins the stored value.

        :param tmp_path: pytest temp dir.
        """
        stats = _stats(brier=0.3000, edge=-0.1000, market_brier=0.2500, edge_n=80)
        one = {"alpha": stats}
        results = _results_dir(tmp_path, at=one, w1=one, w2=one)
        body = _body(results)
        cells = _cells(body, "alpha")
        assert "0.2500" in cells, "must render the stored market_brier"
        assert "0.2000" not in cells, "must NOT render brier + edge"


class TestNoSkillScore:
    """A skill-score ratio must never reach the digest."""

    def test_bss_is_not_rendered(self, results: Path) -> None:
        """Neither the label nor a computed ratio appears anywhere.

        BSS stays in the JSON artifacts; a ratio whose denominator sits
        off-screen is what let a wrong denominator go unnoticed for weeks.

        :param results: results-directory fixture.
        """
        body = _body(results)
        assert "BSS" not in body
        assert "brier_skill" not in body
        # "no-skill" in the alerts prose is the floor being named, not a ratio.
        assert "skill score" not in body.lower()

    def test_base_is_rendered_instead(self, results: Path) -> None:
        """The no-skill floor is shown in Brier units, as a column."""
        body = _body(results)
        assert "base cum" in body
        assert "0.2400" in _cells(
            body, "alpha", after="1b. PRODUCTION - W-1 vs CUMULATIVE"
        )


class TestDeltas:
    """Deltas must be suppressed, not guessed, below the sample floor."""

    def test_delta_names_its_direction(self, results: Path) -> None:
        """A worse Brier and a worse Edge both say "worse" despite opposite signs."""
        body = _body(results)
        cells = _cells(body, "alpha")
        assert "+0.0700 worse" in cells, cells  # Brier 0.3500 vs 0.2800
        assert "-0.1400 worse" in cells, cells  # Edge -0.1700 vs -0.0300

    def test_low_sample_suppresses_delta(self, tmp_path: Path) -> None:
        """A window under the floor renders `insufficient`, not a number."""
        results = tmp_path / "r"
        results.mkdir()
        thin = _stats(valid_n=10, edge_n=10)
        _write(results, "scores_polymarket.json", {"alpha": thin})
        _write(results, "rolling_scores_polymarket.json", {"alpha": thin})
        _write(results, "prev_rolling_scores_polymarket.json", {"alpha": thin})
        _write(results, "scores_tournament_polymarket.json", {})
        body = _body(results)
        cells = _cells(body, "alpha")
        assert "insufficient" in cells
        assert "10 *" in cells, "under-floor counts carry the marker"


class TestTournament:
    """Tournament windows do not exist and must render as such."""

    def test_missing_windows_render_na(self, tmp_path: Path) -> None:
        """Every W-1 cell is the literal `n/a`, never zero or blank."""
        cand = _stats(brier=0.1900, market_brier=0.0890, edge=-0.1010)
        results = _results_dir(tmp_path, at={}, w1={}, w2={}, tournament={"cand": cand})
        body = _body(results)
        cells = _cells(body, "cand")
        assert cells.count("n/a") >= 6, cells
        assert "0.1900" in cells and "0.0890" in cells


class TestRoi:
    """ROI is already a percentage in roi_results.json."""

    def test_roi_is_not_rescaled(self, results: Path) -> None:
        """roi_mid 15.3 renders as +15.3%, not +1530.0%."""
        body = _body(results)
        assert "+15.3%" in _cells(body, "alpha")


class TestAlerts:
    """Alerts flag blocking conditions, not week-to-week wobble."""

    def test_starved_sample_is_flagged(self, tmp_path: Path) -> None:
        """A W-1 window under the floor blocks any keep/demote call."""
        results = _results_dir(
            tmp_path, at={"alpha": _stats()}, w1={"alpha": _stats(valid_n=13)}, w2={}
        )
        body = _body(results)
        assert "sample starved" in body
        assert "13 scored rows in W-1" in body

    def test_small_reliability_dip_is_not_alerted(self, results: Path) -> None:
        """96% vs 98% is one or two rows and must not raise an alert.

        The previous digest raised three such "worse" bullets in one message;
        at these sample sizes two of the three were a single extra failed row.
        Only a breach of the gate that also excludes the tool from ROI fires.

        :param results: results-directory fixture.
        """
        _write(
            results,
            "rolling_scores_polymarket.json",
            {"alpha": _stats(reliability=0.96)},
        )
        body = _body(results)
        assert "reliability breach" not in body
        assert "96%" in _cells(body, "alpha"), "still visible as a column"

    def test_gate_breach_is_alerted(self, results: Path) -> None:
        """Below the 80% gate the tool is excluded from ROI, so it is flagged."""
        _write(
            results,
            "rolling_scores_polymarket.json",
            {"alpha": _stats(reliability=0.55)},
        )
        body = _body(results)
        assert "reliability breach" in body


class TestRendering:
    """Pin the presentation choices a reader depends on."""

    def test_numeric_columns_right_align_and_tool_left_aligns(
        self, tmp_path: Path
    ) -> None:
        """Numbers right-align; the tool name left-aligns.

        The suite renders through Block Kit column_settings, so a flipped
        default would misalign every numeric column at once -- caught here,
        not by eye in Slack.

        :param tmp_path: pytest temp dir.
        """
        results = _results_dir(tmp_path, at={"alpha": _stats()})
        payloads = build_digest_messages(results, "polymarket")
        tables = [b for m in payloads for b in m["blocks"] if b["type"] == "table"]
        assert tables, "no tables rendered"
        for table in tables:
            headers = [c["text"] for c in table["rows"][0]]
            aligns = [s["align"] for s in table["column_settings"]]
            for name, align in zip(headers, aligns):
                if name == "tool":
                    assert align == "left", f"{name} must left-align"
                elif name.startswith(("n ", "Brier", "Edge", "mkt", "base")):
                    assert align == "right", f"{name} must right-align"

    def test_edge_renders_with_explicit_sign(self, tmp_path: Path) -> None:
        """A positive Edge carries a leading plus; Brier never does.

        The sign is the semantics -- Edge is a delta against the market,
        Brier an absolute score. Dropping the "+" makes a winning edge
        indistinguishable from a raw score at a glance.

        :param tmp_path: pytest temp dir.
        """
        results = _results_dir(
            tmp_path, at={"alpha": _stats(edge=0.1234, brier=0.2000)}
        )
        body = _body(results)
        cells = _cells(body, "alpha", after="1b. PRODUCTION")
        assert "+0.1234" in cells
        assert "0.2000" in cells and "+0.2000" not in cells


class TestRobustness:
    """A broken digest must never break the daily post."""

    def test_missing_files_yield_no_messages(self, tmp_path: Path) -> None:
        """An empty results directory returns no messages rather than raising."""
        results = tmp_path / "empty"
        results.mkdir()
        assert not build_digest_messages(results, "polymarket")

    def test_corrupt_window_degrades_to_na(self, results: Path) -> None:
        """An unparseable window file leaves its cells n/a, not the whole post."""
        (results / "prev_rolling_scores_polymarket.json").write_text(
            "{not json", encoding="utf-8"
        )
        body = _body(results)
        assert body, "a corrupt window must not empty the whole digest"
        assert "n/a" in _cells(body, "alpha")

    def test_render_is_deterministic(self, results: Path) -> None:
        """Same inputs, same bytes -- the property golden files depend on."""
        assert _body(results) == (_body(results))


class TestRoiKeying:
    """roi_sim groups by model too, so (tool, mode) is not unique."""

    def test_duplicate_keys_keep_the_group_with_a_real_roi(
        self, tmp_path: Path
    ) -> None:
        """A None-roi group must not shadow a real one.

        A live artifact's colliding (platform, tool, mode) keys made plain
        last-wins drop a real ROI behind a roi_mid-None group.

        :param tmp_path: pytest temp dir.
        """
        one = {"alpha": _stats()}
        results = _results_dir(tmp_path, at=one, w1=one, w2=one)
        common = {
            "tool_name": "alpha",
            "mode": "production",
            "platform": "polymarket",
        }
        (results / "roi_results.json").write_text(
            json.dumps(
                {
                    "groups": [
                        {**common, "model": "gpt-4.1", "roi_mid": 15.8, "n_bets": 31},
                        {**common, "model": "unknown", "roi_mid": None},
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert "+15.8%" in _cells(_body(results), "alpha")


class TestAlertComparators:
    """Below-market and below-no-skill are different claims."""

    def test_beating_the_floor_but_losing_to_market_is_not_no_skill(
        self, tmp_path: Path
    ) -> None:
        """Edge < 0 alone must never be reported as below no-skill.

        Above-floor Brier with a negative Edge is below MARKET, not no-skill.

        :param tmp_path: pytest temp dir.
        """
        results = tmp_path / "r"
        results.mkdir()
        # Two tools: a platform-wide claim needs more than one.
        stats = _stats(brier=0.2234, baseline_brier=0.2330, edge=-0.0266)
        both = {"alpha": stats, "beta": stats}
        _write(results, "scores_polymarket.json", both)
        _write(results, "rolling_scores_polymarket.json", both)
        _write(results, "prev_rolling_scores_polymarket.json", both)
        _write(results, "scores_tournament_polymarket.json", {})
        body = _body(results)
        assert "platform below market" in body
        assert "platform below no-skill" not in body


class TestVisibility:
    """Only PREDICTION tools appear -- but a failed one must not vanish."""

    def test_permitted_tool_that_scored_nothing_is_listed(self, tmp_path: Path) -> None:
        """A prediction tool at 100% parse failure stays visible.

        Selecting on Brier alone would hide the tool most needing attention.

        :param tmp_path: pytest temp dir.
        """
        broken = _stats(n=684, valid_n=0, reliability=0.0, brier=None, edge=None)
        one = {"broken": broken}
        results = _results_dir(tmp_path, at=one, w1=one, w2=one)
        body = _body(results, allowed_tools={"broken"})
        assert "broken" in body
        assert "reliability breach" in body

    def test_non_prediction_tool_is_never_listed(self, tmp_path: Path) -> None:
        """A tool outside the allowlist stays out, however many rows it has.

        A question proposer emits no p_yes, so valid_n=0 is its correct
        reading, not a failure to alarm on.

        :param tmp_path: pytest temp dir.
        """
        proposer = _stats(n=684, valid_n=0, reliability=0.0, brier=None, edge=None)
        both = {"propose-question": proposer, "alpha": _stats()}
        results = _results_dir(tmp_path, at=both, w1=both, w2=both)
        body = _body(results, allowed_tools={"alpha"})
        assert "alpha" in body
        assert "propose-question" not in body


class TestLoudDegradation:
    """A window that did not run says so; it does not read as unchanged."""

    def test_missing_rolling_window_is_announced(self, tmp_path: Path) -> None:
        """Absent rolling files produce a warning, not a silent wall of n/a.

        :param tmp_path: pytest temp dir.
        """
        results = _results_dir(tmp_path, at={"alpha": _stats()})
        body = _body(results)
        assert "unavailable" in body
        assert "unmeasured" in body


class TestOrderStability:
    """Row order must be reproducible across processes, not just calls."""

    def test_order_is_identical_under_different_hash_seeds(
        self, tmp_path: Path
    ) -> None:
        """Rendering under several PYTHONHASHSEEDs yields identical bytes.

        Set-iteration order is stable WITHIN one process and differs BETWEEN
        them, so only subprocesses with distinct hash seeds can catch it.
        Two redundant guards provide the property (``_ordered`` sorts its
        input, and ``_sort_key`` breaks ties on the name). This asserts the
        PROPERTY, so it fails only when both are removed -- verified by
        mutation.

        :param tmp_path: pytest temp dir.
        """
        names = ["echo", "alpha", "delta", "bravo", "charlie"]
        edgeless = {n: _stats(edge=None, brier=0.3) for n in names}
        results = _results_dir(tmp_path, at=edgeless, w1=edgeless, w2=edgeless)

        script = (
            "import json, sys;"
            "from pathlib import Path;"
            "from benchmark.digest_tables import build_digest_messages;"
            f"print(json.dumps(build_digest_messages(Path({str(results)!r}),"
            "'polymarket')))"
        )
        renders = set()
        for seed in ("0", "1", "17", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(Path(__file__).resolve().parents[2]),
                check=True,
            )
            renders.add(proc.stdout)
        assert len(renders) == 1, f"{len(renders)} distinct renders across seeds"
        payloads = json.loads(renders.pop())
        order = [
            row[0]["text"]
            for payload in payloads
            for block in payload["blocks"]
            if block["type"] == "table"
            for row in block["rows"]
            if row[0]["text"] in names
        ]
        assert order[: len(names)] == sorted(names), order


class TestRoiHaircut:
    """`w/costs` is the execution-cost-adjusted ROI, not a rescale of roi_mid."""

    def test_haircut_renders_beside_roi(self, results: Path) -> None:
        """Both ROI columns appear, with their own values.

        :param results: results-directory fixture.
        """
        (results / "roi_results.json").write_text(
            json.dumps(
                {
                    "groups": [
                        {
                            "tool_name": "alpha",
                            "mode": "production",
                            "platform": "polymarket",
                            "roi_mid": 27.9,
                            "roi_haircut": 14.5,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        body = _body(results)
        assert "w/costs" in body
        cells = _cells(body, "alpha")
        assert "+27.9%" in cells and "+14.5%" in cells

    def test_missing_haircut_is_na_not_zero(self, results: Path) -> None:
        """A group without roi_haircut renders n/a, never +0.0%.

        :param results: results-directory fixture.
        """
        body = _body(results)
        assert "+0.0%" not in _cells(body, "alpha")
        assert "n/a" in _cells(body, "alpha")


class TestPoolAwareGating:
    """Edge lives on a different pool than Brier; the floor must follow it."""

    def test_count_shows_both_pools_when_they_differ(self, tmp_path: Path) -> None:
        """An Edge computed on 11 rows cannot print unstarred beside n=40.

        :param tmp_path: pytest temp dir.
        """
        one = {"alpha": _stats(valid_n=40, edge_n=11)}
        results = _results_dir(tmp_path, at=one, w1=one, w2=one)
        cells = _cells(_body(results), "alpha")
        assert "40/11 *" in cells, cells

    def test_edge_delta_is_floored_on_edge_n(self, tmp_path: Path) -> None:
        """A 40-row Brier delta renders while an 11-row Edge delta does not.

        :param tmp_path: pytest temp dir.
        """
        current = _stats(valid_n=40, edge_n=11, brier=0.35, edge=-0.17)
        prior = _stats(valid_n=40, edge_n=11, brier=0.30, edge=-0.03)
        results = _results_dir(
            tmp_path, at={"alpha": prior}, w1={"alpha": current}, w2={"alpha": prior}
        )
        cells = _cells(_body(results), "alpha")
        assert "+0.0500 worse" in cells, cells
        assert cells.count("insufficient") >= 1, cells


class TestPlatformAlertFloor:
    """A fleet-wide claim needs tools that clear the floor, and more than one."""

    def test_thin_window_cannot_assert_a_platform_claim(self, tmp_path: Path) -> None:
        """The same window marked insufficient cannot condemn the fleet.

        :param tmp_path: pytest temp dir.
        """
        thin = _stats(valid_n=2, edge_n=2, edge=-0.10, brier=0.31, baseline_brier=0.24)
        both = {"alpha": thin, "beta": thin}
        results = _results_dir(tmp_path, at=both, w1=both, w2=both)
        body = _body(results)
        assert "platform below market" not in body
        assert "platform below no-skill" not in body

    def test_single_tool_is_not_a_platform(self, tmp_path: Path) -> None:
        """One tool cannot support a claim about the whole fleet.

        :param tmp_path: pytest temp dir.
        """
        one = {"alpha": _stats(edge=-0.10, brier=0.31, baseline_brier=0.24)}
        results = _results_dir(tmp_path, at=one, w1=one, w2=one)
        assert "platform below" not in _body(results)


class TestWindowLabel:
    """The report names the span it shows, or says it cannot."""

    def test_observed_bounds_are_rendered(self, tmp_path: Path) -> None:
        """window_start/window_end reach the section line.

        :param tmp_path: pytest temp dir.
        """
        results = tmp_path / "r"
        results.mkdir()
        (results / "scores_polymarket.json").write_text(
            json.dumps(
                {
                    "current_month": "2026-08",
                    "window_start": "2026-07-01T00:00:00Z",
                    "window_end": "2026-08-11T09:00:00Z",
                    "by_tool": {"alpha": _stats()},
                }
            ),
            encoding="utf-8",
        )
        _write(results, "rolling_scores_polymarket.json", {"alpha": _stats()})
        _write(results, "prev_rolling_scores_polymarket.json", {"alpha": _stats()})
        _write(results, "scores_tournament_polymarket.json", {})
        assert "2026-07-01..2026-08-11" in _body(results)

    def test_missing_bounds_say_so_rather_than_guess(self, results: Path) -> None:
        """Without bounds the label admits the span is unstated.

        The file's own `current_month` key does NOT describe its span -- on
        live data it read 2026-08 over a file holding rows back to 2026-07-01
        -- so it must never be used as a stand-in.

        :param results: results-directory fixture.
        """
        body = _body(results)
        assert "UNSTATED" in body
        assert "2026-08" not in body


class TestCensoringAlert:
    """A week that has not finished filling must not read as good news."""

    def test_thin_recent_week_is_flagged(self, tmp_path: Path) -> None:
        """W-1 far smaller than W-2 raises the still-filling alert.

        Rows exist only once their market resolves, so the newest window lags.

        :param tmp_path: pytest temp dir.
        """
        results = _results_dir(
            tmp_path,
            at={"alpha": _stats(valid_n=3000)},
            w1={"alpha": _stats(valid_n=600)},
            w2={"alpha": _stats(valid_n=2700)},
        )
        body = _body(results)
        assert "week still filling" in body
        assert "22%" in body

    def test_complete_week_is_not_flagged(self, results: Path) -> None:
        """Comparable windows raise nothing.

        :param results: results-directory fixture.
        """
        assert "week still filling" not in _body(results)


class TestBaseInBothTables:
    """The no-skill floor moves per window, so both tables carry it."""

    def test_weekly_table_shows_base_for_both_weeks(self, tmp_path: Path) -> None:
        """Both weekly base columns render, not just the cumulative one.

        base is per-window and moves the bar the demote rule reads, so it
        belongs where a reader can see it move.

        :param tmp_path: pytest temp dir.
        """
        results = _results_dir(
            tmp_path,
            at={"alpha": _stats()},
            w1={"alpha": _stats(baseline_brier=0.1438)},
            w2={"alpha": _stats(baseline_brier=0.2270)},
        )
        body = _body(results)
        assert "base W-1" in body and "base W-2" in body
        cells = _cells(body, "alpha")
        assert "0.1438" in cells and "0.2270" in cells


class TestEdgeOrdering:
    """Rows are ordered by Edge -- how good -- not by how sure we are."""

    def _cohort(self, tmp_path: Path) -> Path:
        """Build a cohort where Edge and the promote bound disagree.

        `thin` has the BEST edge on few markets; `broad` a slightly worse edge
        on many, so ranking on the bound would put `thin` last.

        :param tmp_path: pytest temp dir.
        :return: the results directory.
        """
        cohort = {
            "thin": _stats(edge=-0.15, edge_sd=0.38, edge_n=31),
            "broad": _stats(edge=-0.17, edge_sd=0.34, edge_n=421),
        }
        return _results_dir(tmp_path, at=cohort, w1=cohort, w2=cohort)

    def test_best_edge_ranks_first_even_on_a_small_sample(self, tmp_path: Path) -> None:
        """The better tool leads, regardless of how much data backs it.

        A lower bound is the conservative test for PROMOTING and is backwards
        for demoting.

        :param tmp_path: pytest temp dir.
        """
        body = _body(self._cohort(tmp_path))
        assert _cells(body, "thin", after="1b. PRODUCTION")[0] == "1"
        assert _cells(body, "broad", after="1b. PRODUCTION")[0] == "2"

    def test_unjudged_tools_are_dashed_not_ranked_last(self, tmp_path: Path) -> None:
        """A tool with no Edge at all is unranked, not "worst".

        :param tmp_path: pytest temp dir.
        """
        cohort = {
            "scored": _stats(edge=0.09, edge_sd=0.10, edge_n=100),
            "unscored": _stats(edge=None, edge_sd=None, edge_n=0),
        }
        body = _body(_results_dir(tmp_path, at=cohort, w1=cohort, w2=cohort))
        assert _cells(body, "scored", after="1b. PRODUCTION")[0] == "1"
        assert _cells(body, "unscored", after="1b. PRODUCTION")[0] == "-"


class TestTitle:
    """The report titles itself, at header size, with the data's own date."""

    def test_header_block_carries_platform_and_date(self, tmp_path: Path) -> None:
        """A `header` block, dated from window_end rather than a clock.

        Slack renders `header` at title size and accepts plain_text only --
        a mrkdwn text object is rejected.

        :param tmp_path: pytest temp dir.
        """
        results = tmp_path / "r"
        results.mkdir()
        (results / "scores_polymarket.json").write_text(
            json.dumps(
                {
                    "window_start": "2026-07-01T00:00:00Z",
                    "window_end": "2026-08-11T03:26:49Z",
                    "by_tool": {"alpha": _stats()},
                }
            ),
            encoding="utf-8",
        )
        _write(results, "rolling_scores_polymarket.json", {"alpha": _stats()})
        _write(results, "prev_rolling_scores_polymarket.json", {"alpha": _stats()})
        _write(results, "scores_tournament_polymarket.json", {})
        payload = build_digest_messages(results, "polymarket")[0]
        first = payload["blocks"][0]
        assert first["type"] == "header"
        assert first["text"]["type"] == "plain_text"
        rule, title, closing = first["text"]["text"].split("\n")
        assert rule == closing and set(rule) == {TITLE_RULE_CHAR}
        assert "POLYSTRAT" in title
        assert "REPORT V2" in title
        assert "2026-08-11" in title
        assert len(rule) >= len(title) - 2, "rule spans the title it frames"
        assert "2026-08-11" in payload["text"]


class TestRankingWindow:
    """Table 1b must rank on the column its caption names."""

    def test_1b_ranks_on_cumulative_edge_not_the_weekly_one(
        self, tmp_path: Path
    ) -> None:
        """The `#` column has to agree with the `Edge cum` column beside it.

        _sort_key ranks on the first window that carries an edge, so the
        argument order at the call site IS the ranking metric.

        :param tmp_path: pytest temp dir.
        """
        results = tmp_path / "results"
        results.mkdir()
        # beta wins the week; alpha wins cumulatively. The caption promises
        # cumulative, so alpha must rank first.
        _write(
            results,
            "scores_polymarket.json",
            {
                "alpha": _stats(edge=0.3000, edge_n=500),
                "beta": _stats(edge=0.0100, edge_n=500),
            },
        )
        _write(
            results,
            "rolling_scores_polymarket.json",
            {
                "alpha": _stats(edge=-0.2000, edge_n=60),
                "beta": _stats(edge=0.2500, edge_n=60),
            },
        )
        _write(results, "prev_rolling_scores_polymarket.json", {})

        body = _body(results, allowed_tools={"alpha", "beta"})
        marker = "1b. PRODUCTION"
        assert _cells(body, "alpha", after=marker)[0] == "1"
        assert _cells(body, "beta", after=marker)[0] == "2"


class TestStarvedAlertRobustness:
    """The starved-sample alert must not crash on the field it gates on."""

    def test_missing_valid_n_does_not_take_down_the_digest(
        self, tmp_path: Path
    ) -> None:
        """_has_floor is False when valid_n is ABSENT, not only when it is low.

        The alert then subscripted the missing field and the KeyError escaped.

        :param tmp_path: pytest temp dir.
        """
        starved = _stats()
        del starved["valid_n"]
        results = _results_dir(
            tmp_path, at={"alpha": _stats()}, w1={"alpha": starved}, w2={}
        )

        body = _body(results, allowed_tools={"alpha"})
        assert "sample starved" in body
        assert "n/a scored rows in W-1" in body
