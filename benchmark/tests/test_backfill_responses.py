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
"""Tests for benchmark/datasets/backfill_responses.py"""

import json
import re
from pathlib import Path
from typing import Any, Optional

import pytest
from benchmark.datasets import backfill_responses as br
from benchmark.datasets.fetch_production import (
    DELIVERS_SCHEMA_LEGACY,
    DELIVERS_SCHEMA_PARSED,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


VALID_RESPONSE = '{"p_yes": 0.72, "p_no": 0.28, "confidence": 0.85}'


def _row(
    deliver_id: str,
    platform: str = "omen",
    status: str = "missing_fields",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a minimal production-log row."""
    row: dict[str, Any] = {
        "row_id": f"prod_{platform}_{deliver_id}",
        "deliver_id": deliver_id,
        "schema_version": "1.0",
        "mode": "production_replay",
        "platform": platform,
        "question_text": "Will it rain tomorrow?",
        "tool_name": "factual_research",
        "p_yes": None,
        "p_no": None,
        "prediction_parse_status": status,
        "final_outcome": True,
        "match_confidence": 1.0,
    }
    row.update(overrides)
    return row


def _write_shard(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSONL shard the same way append_jsonl does."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_shard(path: Path) -> list[dict[str, Any]]:
    """Read all rows back from a JSONL shard."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ids_in_query(query: str) -> list[str]:
    """Extract the quoted deliver ids from a delivers-by-ids query."""
    return re.findall(r'"(0x[0-9a-f]+)"', query)


def _make_fake_post_graphql(
    responses: dict[str, Optional[str]],
    calls: Optional[list[tuple[str, list[str]]]] = None,
    schema: str = DELIVERS_SCHEMA_LEGACY,
) -> Any:
    """Build a fake _post_graphql returning canned tool responses.

    Serves the deliver shape matching *schema*: legacy flat
    ``model``/``toolResponse`` fields, or the nested ``parsedDelivery``
    entity (a None response is served as a not-yet-indexed delivery,
    i.e. ``parsedDelivery: null``).

    :param responses: deliver_id -> tool response to serve.
    :param calls: optional list collecting (url, queried ids) per call.
    :param schema: deliver shape to serve (legacy or parsed).
    :return: fake function with the _post_graphql signature.
    """

    def _deliver(did: str) -> dict[str, Any]:
        """Build one canned deliver in the configured schema shape."""
        response = responses.get(did)
        if schema == DELIVERS_SCHEMA_LEGACY:
            return {"id": did, "model": None, "toolResponse": response}
        if response is None:
            return {"id": did, "parsedDelivery": None}
        return {
            "id": did,
            "parsedDelivery": {
                "response": response,
                "model": "gpt-4o",
                "tool": "factual_research",
                "toolHash": None,
            },
        }

    def fake(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Serve canned delivers for the ids found in the query."""
        ids = _ids_in_query(payload["query"])
        if calls is not None:
            calls.append((url, ids))
        return {"delivers": [_deliver(did) for did in ids]}

    return fake


@pytest.fixture(autouse=True)
def _stub_schema_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin schema detection to legacy so no test probes the network.

    Individual tests override this by re-patching
    ``br.detect_delivers_schema`` (schema-routing tests) after the
    fixture ran.

    :param monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr(
        br, "detect_delivers_schema", lambda url: DELIVERS_SCHEMA_LEGACY
    )


# ---------------------------------------------------------------------------
# repair_row
# ---------------------------------------------------------------------------


class TestRepairRow:
    """Tests for single-row repair."""

    def test_valid_response_repairs_in_place(self) -> None:
        """A now-present valid response updates p_yes/p_no/status/confidence."""
        row = _row("0x01")
        assert br.repair_row(row, VALID_RESPONSE) is True
        assert row["p_yes"] == 0.72
        assert row["p_no"] == 0.28
        assert row["prediction_parse_status"] == "valid"
        assert row["confidence"] == 0.85

    def test_no_confidence_key_when_not_parsed(self) -> None:
        """A response without confidence does not add the field."""
        row = _row("0x01")
        assert br.repair_row(row, '{"p_yes": 0.6, "p_no": 0.4}') is True
        assert "confidence" not in row

    def test_still_null_untouched(self) -> None:
        """A still-null response leaves the row untouched."""
        row = _row("0x01")
        before = dict(row)
        assert br.repair_row(row, None) is False
        assert row == before

    def test_still_unparseable_untouched(self) -> None:
        """A response that parses malformed leaves the row untouched."""
        row = _row("0x01")
        before = dict(row)
        assert br.repair_row(row, '{"p_yes": 0.9, "p_no": 0.9}') is False
        assert row == before


# ---------------------------------------------------------------------------
# backfill: repair behaviour
# ---------------------------------------------------------------------------


class TestBackfillRepairs:
    """Tests for the end-to-end shard repair."""

    def test_repairs_missing_fields_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repairs the candidate row; other fields and row order untouched."""
        shard = tmp_path / "production_log_2026_07_01.jsonl"
        rows = [
            _row("0xaa", status="valid", p_yes=0.5, p_no=0.5),
            _row("0xbb"),
            _row("0xcc", status="malformed"),
        ]
        _write_shard(shard, rows)
        monkeypatch.setattr(
            br, "_post_graphql", _make_fake_post_graphql({"0xbb": VALID_RESPONSE})
        )

        summary = br.backfill(tmp_path)

        assert summary["repaired"] == 1
        after = _read_shard(shard)
        # Order preserved, one row per original row
        assert [r["deliver_id"] for r in after] == ["0xaa", "0xbb", "0xcc"]
        repaired = after[1]
        assert repaired["p_yes"] == 0.72
        assert repaired["p_no"] == 0.28
        assert repaired["prediction_parse_status"] == "valid"
        assert repaired["confidence"] == 0.85
        # Untouched fields survive
        assert repaired["row_id"] == rows[1]["row_id"]
        assert repaired["question_text"] == rows[1]["question_text"]
        assert repaired["final_outcome"] is True
        # Neighbour rows byte-identical
        assert after[0] == rows[0]
        assert after[2] == rows[2]

    def test_still_null_rows_stay_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rows whose response is still null are not modified at all."""
        shard = tmp_path / "production_log_2026_07_01.jsonl"
        _write_shard(shard, [_row("0xbb")])
        monkeypatch.setattr(br, "_post_graphql", _make_fake_post_graphql({}))

        summary = br.backfill(tmp_path)

        assert summary["repaired"] == 0
        assert summary["candidates"] == 1
        assert _read_shard(shard) == [_row("0xbb")]

    def test_malformed_rows_are_not_candidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only missing_fields rows are candidates; malformed are skipped."""
        shard = tmp_path / "production_log_2026_07_01.jsonl"
        _write_shard(shard, [_row("0xdd", status="malformed")])
        calls: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(
            br,
            "_post_graphql",
            _make_fake_post_graphql({"0xdd": VALID_RESPONSE}, calls),
        )

        summary = br.backfill(tmp_path)

        assert summary["candidates"] == 0
        assert summary["repaired"] == 0
        assert not calls  # nothing queried at all

    def test_row_without_deliver_id_not_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing_fields row without deliver_id is not a candidate."""
        shard = tmp_path / "production_log_2026_07_01.jsonl"
        _write_shard(shard, [_row("")])
        monkeypatch.setattr(br, "_post_graphql", _make_fake_post_graphql({}))

        summary = br.backfill(tmp_path)

        assert summary["candidates"] == 0

    def test_atomic_rewrite_only_when_changed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shard file is replaced only when a row was repaired."""
        shard = tmp_path / "production_log_2026_07_01.jsonl"
        _write_shard(shard, [_row("0xbb")])
        inode_before = shard.stat().st_ino
        content_before = shard.read_bytes()

        # Run 1: response still null -> file untouched (same inode)
        monkeypatch.setattr(br, "_post_graphql", _make_fake_post_graphql({}))
        br.backfill(tmp_path)
        assert shard.stat().st_ino == inode_before
        assert shard.read_bytes() == content_before

        # Run 2: response present -> file replaced (new inode, new content)
        monkeypatch.setattr(
            br, "_post_graphql", _make_fake_post_graphql({"0xbb": VALID_RESPONSE})
        )
        br.backfill(tmp_path)
        assert shard.stat().st_ino != inode_before
        assert shard.read_bytes() != content_before
        # No temp files left behind
        assert not list(tmp_path.glob("*.tmp"))

    def test_idempotent_second_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second run after repair finds no candidates and changes nothing."""
        shard = tmp_path / "production_log_2026_07_01.jsonl"
        _write_shard(shard, [_row("0xbb")])
        monkeypatch.setattr(
            br, "_post_graphql", _make_fake_post_graphql({"0xbb": VALID_RESPONSE})
        )

        first = br.backfill(tmp_path)
        assert first["repaired"] == 1
        content_after_first = shard.read_bytes()

        second = br.backfill(tmp_path)
        assert second["candidates"] == 0
        assert second["repaired"] == 0
        assert shard.read_bytes() == content_after_first

    def test_summary_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Summary reports shards scanned, candidates, repairs per platform."""
        _write_shard(
            tmp_path / "production_log_2026_07_01.jsonl",
            [_row("0x01"), _row("0x02"), _row("0x03", status="valid")],
        )
        _write_shard(
            tmp_path / "production_log_2026_07_02.jsonl",
            [_row("0x04", platform="polymarket")],
        )
        monkeypatch.setattr(
            br, "_post_graphql", _make_fake_post_graphql({"0x01": VALID_RESPONSE})
        )

        summary = br.backfill(tmp_path)

        assert summary["shards_scanned"] == 2
        assert summary["candidates"] == 3
        assert summary["repaired"] == 1
        assert summary["platforms"]["omen"] == {"candidates": 2, "repaired": 1}
        assert summary["platforms"]["polymarket"] == {"candidates": 1, "repaired": 0}


# ---------------------------------------------------------------------------
# Batching & platform routing
# ---------------------------------------------------------------------------


class TestBatchingAndRouting:
    """Tests for subgraph query batching and per-platform URL routing."""

    def test_batching_splits_large_id_sets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """More than BACKFILL_BATCH_SIZE ids are split across requests."""
        n = br.BACKFILL_BATCH_SIZE * 2 + 50  # 250 with the default of 100
        rows = [_row(f"0x{i:04x}") for i in range(n)]
        _write_shard(tmp_path / "production_log_2026_07_01.jsonl", rows)
        calls: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(br, "_post_graphql", _make_fake_post_graphql({}, calls))

        br.backfill(tmp_path)

        sizes = [len(ids) for _, ids in calls]
        assert sizes == [100, 100, 50]
        assert sorted(did for _, ids in calls for did in ids) == sorted(
            r["deliver_id"] for r in rows
        )

    def test_platform_routing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omen ids go to the Gnosis URL, Polymarket ids to the Polygon URL."""
        _write_shard(
            tmp_path / "production_log_2026_07_01.jsonl",
            [_row("0x01", platform="omen"), _row("0x02", platform="polymarket")],
        )
        calls: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(br, "_post_graphql", _make_fake_post_graphql({}, calls))

        br.backfill(tmp_path)

        by_url = dict(calls)
        assert by_url[br.MECH_MARKETPLACE_GNOSIS_URL] == ["0x01"]
        assert by_url[br.MECH_MARKETPLACE_POLYGON_URL] == ["0x02"]

    def test_unknown_platform_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Candidates on an unknown platform are counted but never queried."""
        _write_shard(
            tmp_path / "production_log_2026_07_01.jsonl",
            [_row("0x01", platform="kalshi")],
        )
        calls: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(br, "_post_graphql", _make_fake_post_graphql({}, calls))

        summary = br.backfill(tmp_path)

        assert summary["candidates"] == 1
        assert summary["repaired"] == 0
        assert not calls

    def test_failed_batch_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batch that raises is skipped; the run continues and exits clean."""
        _write_shard(tmp_path / "production_log_2026_07_01.jsonl", [_row("0x01")])

        def boom(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            """Simulate a subgraph failure."""
            raise RuntimeError("subgraph down")

        monkeypatch.setattr(br, "_post_graphql", boom)

        summary = br.backfill(tmp_path)

        assert summary["repaired"] == 0


# ---------------------------------------------------------------------------
# Schema-aware querying (nested ParsedDelivery vs legacy flat fields)
# ---------------------------------------------------------------------------


class TestSchemaRouting:
    """Tests for per-endpoint schema detection and query-shape selection."""

    def test_parsed_schema_uses_parsed_query_and_repairs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A parsed-schema endpoint is queried for parsedDelivery and repaired."""
        shard = tmp_path / "production_log_2026_07_01.jsonl"
        _write_shard(shard, [_row("0xbb")])
        monkeypatch.setattr(
            br, "detect_delivers_schema", lambda url: DELIVERS_SCHEMA_PARSED
        )
        queries: list[str] = []
        fake = _make_fake_post_graphql(
            {"0xbb": VALID_RESPONSE}, schema=DELIVERS_SCHEMA_PARSED
        )

        def spy(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            """Record the query text, then serve the parsed-schema fake."""
            queries.append(payload["query"])
            return fake(url, payload)

        monkeypatch.setattr(br, "_post_graphql", spy)

        summary = br.backfill(tmp_path)

        assert summary["repaired"] == 1
        assert all("parsedDelivery" in q for q in queries)
        assert all("toolResponse" not in q for q in queries)
        repaired = _read_shard(shard)[0]
        assert repaired["p_yes"] == 0.72
        assert repaired["prediction_parse_status"] == "valid"

    def test_legacy_schema_uses_flat_query_and_repairs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A legacy-schema endpoint is queried for flat fields and repaired."""
        shard = tmp_path / "production_log_2026_07_01.jsonl"
        _write_shard(shard, [_row("0xbb")])
        queries: list[str] = []
        fake = _make_fake_post_graphql(
            {"0xbb": VALID_RESPONSE}, schema=DELIVERS_SCHEMA_LEGACY
        )

        def spy(url: str, payload: dict[str, Any]) -> dict[str, Any]:
            """Record the query text, then serve the legacy-schema fake."""
            queries.append(payload["query"])
            return fake(url, payload)

        monkeypatch.setattr(br, "_post_graphql", spy)

        summary = br.backfill(tmp_path)

        assert summary["repaired"] == 1
        assert all("toolResponse" in q for q in queries)
        assert all("parsedDelivery" not in q for q in queries)
        repaired = _read_shard(shard)[0]
        assert repaired["p_yes"] == 0.72
        assert repaired["prediction_parse_status"] == "valid"

    def test_parsed_delivery_not_indexed_leaves_row_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deliver whose parsedDelivery is still null stays a candidate."""
        shard = tmp_path / "production_log_2026_07_01.jsonl"
        _write_shard(shard, [_row("0xbb")])
        monkeypatch.setattr(
            br, "detect_delivers_schema", lambda url: DELIVERS_SCHEMA_PARSED
        )
        monkeypatch.setattr(
            br,
            "_post_graphql",
            _make_fake_post_graphql({}, schema=DELIVERS_SCHEMA_PARSED),
        )

        summary = br.backfill(tmp_path)

        assert summary["repaired"] == 0
        assert _read_shard(shard) == [_row("0xbb")]

    def test_probe_failure_skips_platform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing schema probe skips the platform without failing the run."""
        _write_shard(tmp_path / "production_log_2026_07_01.jsonl", [_row("0x01")])

        def boom(url: str) -> str:
            """Simulate a probe hitting an unreachable endpoint."""
            raise RuntimeError("probe failed")

        calls: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(br, "detect_delivers_schema", boom)
        monkeypatch.setattr(br, "_post_graphql", _make_fake_post_graphql({}, calls))

        summary = br.backfill(tmp_path)

        assert summary["repaired"] == 0
        assert not calls  # no by-ids query was ever issued

    def test_probe_once_per_platform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each platform's endpoint is probed exactly once per run."""
        _write_shard(
            tmp_path / "production_log_2026_07_01.jsonl",
            [
                _row("0x01", platform="omen"),
                _row("0x02", platform="omen"),
                _row("0x03", platform="polymarket"),
            ],
        )
        probed: list[str] = []

        def probe(url: str) -> str:
            """Record the probed URL and report the legacy shape."""
            probed.append(url)
            return DELIVERS_SCHEMA_LEGACY

        monkeypatch.setattr(br, "detect_delivers_schema", probe)
        monkeypatch.setattr(br, "_post_graphql", _make_fake_post_graphql({}))

        br.backfill(tmp_path)

        assert sorted(probed) == sorted(
            [br.MECH_MARKETPLACE_GNOSIS_URL, br.MECH_MARKETPLACE_POLYGON_URL]
        )

    def test_no_probe_without_candidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Healthy shards trigger neither a schema probe nor any query."""
        _write_shard(
            tmp_path / "production_log_2026_07_01.jsonl",
            [_row("0xaa", status="valid", p_yes=0.5, p_no=0.5)],
        )

        def probe(url: str) -> str:
            """Fail the test if the probe fires on a healthy dataset."""
            raise AssertionError("schema probe must not run")

        calls: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(br, "detect_delivers_schema", probe)
        monkeypatch.setattr(br, "_post_graphql", _make_fake_post_graphql({}, calls))

        summary = br.backfill(tmp_path)

        assert summary["candidates"] == 0
        assert not calls


# ---------------------------------------------------------------------------
# CLI / GitHub output
# ---------------------------------------------------------------------------


class TestMainOutput:
    """Tests for the CLI entry point and workflow output emission."""

    def test_repaired_line_and_github_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() prints repaired=<n> and appends it to $GITHUB_OUTPUT."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        _write_shard(logs_dir / "production_log_2026_07_01.jsonl", [_row("0xbb")])
        gh_output = tmp_path / "gh_output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(gh_output))
        monkeypatch.setattr(
            br, "_post_graphql", _make_fake_post_graphql({"0xbb": VALID_RESPONSE})
        )
        monkeypatch.setattr(
            "sys.argv", ["backfill_responses", "--logs-dir", str(logs_dir)]
        )

        br.main()

        assert "repaired=1" in capsys.readouterr().out
        assert "repaired=1" in gh_output.read_text()

    def test_no_github_output_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without $GITHUB_OUTPUT, main() still prints the repaired line."""
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        monkeypatch.setattr(br, "_post_graphql", _make_fake_post_graphql({}))
        monkeypatch.setattr(
            "sys.argv", ["backfill_responses", "--logs-dir", str(tmp_path)]
        )

        br.main()

        assert "repaired=0" in capsys.readouterr().out

    def test_main_never_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An unexpected failure is swallowed and reported as repaired=0."""

        def boom(logs_dir: Path, batch_size: int = 0) -> dict[str, Any]:
            """Simulate an unexpected crash inside the backfill."""
            raise RuntimeError("unexpected")

        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        monkeypatch.setattr(br, "backfill", boom)
        monkeypatch.setattr(
            "sys.argv", ["backfill_responses", "--logs-dir", str(tmp_path)]
        )

        br.main()  # must not raise

        assert "repaired=0" in capsys.readouterr().out
