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
"""Regression coverage for complete rosters, decisive evidence, and offline replay."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from benchmark import notify_slack, report_audit, tool_usage
from benchmark.digest_tables import WINDOW_FILES, build_concise_digest_message
from benchmark.report_audit import initialize, post, prepare, replay

NEW_TOOL = "superforcaster-market-aware-olas-predict-r1-14b"


def _stats(**changes: Any) -> dict[str, Any]:
    result = {
        "n": 100,
        "valid_n": 100,
        "brier": 0.20,
        "baseline_brier": 0.24,
        "edge": -0.01,
        "edge_n": 80,
        "edge_sd": 0.05,
        "conditional_accuracy_rate": 0.60,
        "disagree_n": 40,
        "reliability": 1.0,
    }
    result.update(changes)
    return result


def _payloads(rows: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"by_tool": copy.deepcopy(rows) if key != "tournament" else {}}
        for key in WINDOW_FILES
    }


@pytest.fixture(name="audit_run")
def _audit_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Create deterministic input structure with real fresh timestamps."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_SHA", "source-sha")
    results, output = tmp_path / "results", tmp_path / "audit"
    results.mkdir()
    snapshot = {
        "release_ref": "v-test",
        "deployments": {
            name: {
                "platform": platform,
                "status": "complete",
                "tools": ["bad", "good", NEW_TOOL],
            }
            for name, platform in tool_usage.DEPLOYMENT_TO_PLATFORM.items()
        },
    }
    initialize(results, output, snapshot)
    for platform in ("omen", "polymarket"):
        payloads = _payloads(
            {"bad": _stats(brier=0.31), "good": _stats(), NEW_TOOL: _stats()}
        )
        for key, payload in payloads.items():
            payload.update(
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                requested_window={
                    "start": "2026-09-01T00:00:00Z",
                    "end": "2026-09-29T00:00:00Z",
                    "timestamp_field": "requested_at",
                },
            )
            (results / WINDOW_FILES[key].format(platform=platform)).write_text(
                json.dumps(payload)
            )
        (results / f"analysis_scores_{platform}.json").write_text(
            json.dumps(payloads["at"])
        )
        (results / f"report_{platform}.md").write_text(
            "# Platform report\n\nBrier summary.\n"
        )
    return results, output


def test_bundle_replays_without_network_llm_or_live_roster(
    audit_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation and replay share exact inputs and saved narrative."""

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("External dependency reached")

    monkeypatch.setattr(notify_slack, "summarize_report", unexpected)
    monkeypatch.setattr(tool_usage, "fetch_valid_tools", unexpected)
    monkeypatch.setattr(tool_usage, "_http_get", unexpected)
    results, output = audit_run
    bundle = prepare(results, output, "polymarket", "*Summary:* Saved narrative.")
    monkeypatch.setenv("BENCHMARK_ROLLING_WINDOW_DAYS", "14")
    monkeypatch.setenv("USE_MECH_ANALYTICS_ROWS", "false")
    payload = replay(bundle)
    record = json.loads((bundle / "decision.json").read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert record["decision"]["token"] == "DEMOTE 1"
    assert manifest["run"]["source_commit"] == "source-sha"
    assert manifest["run"]["run_attempt"] == "2"
    assert manifest["files"]["summary.txt"]
    assert (
        record["windows"]["at"]["requested_window"]["timestamp_field"] == "requested_at"
    )
    assert "Saved narrative" in json.dumps(payload)
    assert "Deployment decision" in (results / "report_polymarket.md").read_text()
    assert all(
        len(block.get("text", {}).get("text", "")) <= 3000
        for block in payload["blocks"]
    )


@pytest.mark.parametrize(
    "failure", ["missing", "stale", "bad-timestamp", "malformed", "bad-shape", "nan"]
)
def test_failed_current_input_is_never_presented_as_current(
    audit_run: tuple[Path, Path], failure: str
) -> None:
    """A rejected current window remains auditable but cannot support actions."""
    results, output = audit_run
    path = results / "rolling_scores_polymarket.json"
    if failure == "missing":
        path.unlink()
    elif failure in ("stale", "bad-timestamp"):
        payload = json.loads(path.read_text())
        payload["generated_at"] = "2000-01-01T00:00:00Z" if failure == "stale" else 1
        path.write_text(json.dumps(payload))
    else:
        path.write_text(
            {"malformed": "{", "bad-shape": "[]", "nan": '{"x": NaN}'}[failure]
        )
    bundle = prepare(results, output, "polymarket", "Saved summary")
    record = json.loads((bundle / "decision.json").read_text())
    assert record["decision"]["token"] == "NO CHANGE"
    assert record["decision"]["demote"] == []
    assert record["windows"]["w1"]["generated_at"] is None
    assert record["input_status"]["w1"]
    assert record["production"]["bad"]["verdict"].startswith("unavailable")
    assert record["production"]["good"]["verdict"] == "keep"
    assert record["counts"]["remaining_unassessed"] == 1
    replay(bundle)


@pytest.mark.parametrize("window", ["at", "w2", "tournament"])
@pytest.mark.parametrize("failure", ["missing", "stale", "malformed"])
def test_rejected_windows_only_disable_dependent_evaluations(
    audit_run: tuple[Path, Path], window: str, failure: str
) -> None:
    """Optional failures retain production findings and exact offline replay."""
    results, output = audit_run
    name = WINDOW_FILES[window].format(platform="polymarket")
    path = results / name
    if failure == "missing":
        path.unlink()
    elif failure == "stale":
        payload = json.loads(path.read_text())
        payload["generated_at"] = "2000-01-01T00:00:00Z"
        path.write_text(json.dumps(payload))
    else:
        path.write_text("{rejected")
    bundle = prepare(results, output, "polymarket", "Original trend narrative")
    record = json.loads((bundle / "decision.json").read_text())
    message = json.dumps(replay(bundle))
    if window == "at":
        assert record["decision"]["token"] == "DECISION UNAVAILABLE"
        assert record["decision"]["demote"] == []
        assert record["counts"]["remaining_assessed"] == 0
    else:
        assert record["decision"]["demote"] == ["bad"]
    if window == "tournament":
        assert "Candidate evaluation unavailable" in message
        assert "No tournament candidate qualifies" not in message
        assert "Original trend narrative" in message
    if window == "w2":
        assert "Prev 7d unavailable" in message
        assert "Original trend narrative" not in message
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["input_status"][name]
    if failure != "missing":
        assert (bundle / name).read_bytes() == path.read_bytes()
        assert name in manifest["files"]


def test_direct_notifier_preserves_production_without_optional_windows(
    audit_run: tuple[Path, Path],
) -> None:
    """The unaudited entry point uses the same scoped input decisions."""
    results, _ = audit_run
    for window in ("w2", "tournament"):
        (results / WINDOW_FILES[window].format(platform="polymarket")).unlink()
    payload = build_concise_digest_message(
        results, "polymarket", "Summary", deployed_tools=["bad", "good", NEW_TOOL]
    )
    assert payload is not None
    text = json.dumps(payload)
    assert "Decision: DEMOTE 1" in text
    assert "Candidate evaluation unavailable" in text


def test_initialize_removes_derived_outputs_and_preserves_resume_state(
    tmp_path: Path,
) -> None:
    """Yesterday's report cannot survive a failed current scoring step."""
    results = tmp_path / "results"
    results.mkdir()
    for name in (
        "report_omen.md",
        "trailing_scores_omen.json",
        "rolling_scores_omen.json",
        "scores_omen.json",
    ):
        (results / name).write_text("old")
    initialize(results, tmp_path / "run", {"deployments": {}})
    assert (results / "scores_omen.json").read_text() == "old"
    assert not (results / "report_omen.md").exists()
    assert not (results / "trailing_scores_omen.json").exists()
    with pytest.raises(FileExistsError):
        initialize(results, tmp_path / "run", {"deployments": {}})


def test_tampering_or_omitting_an_input_hash_fails_replay(
    audit_run: tuple[Path, Path],
) -> None:
    """Hashes cover every captured input, including fields unused by the gate."""
    results, output = audit_run
    bundle = prepare(results, output, "polymarket", "Summary")
    path = bundle / "trailing_scores_polymarket.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay(bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"][path.name]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every consumed file"):
        replay(bundle)


def test_changed_policy_is_detected_even_with_intact_input_hashes(
    audit_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay checks results as well as archive integrity."""
    results, output = audit_run
    bundle = prepare(results, output, "polymarket", "Summary")
    original = report_audit.build_decision_record

    def changed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        record = original(*args, **kwargs)
        record["decision"]["token"] = "WRONG"
        return record

    monkeypatch.setattr(report_audit, "build_decision_record", changed)
    with pytest.raises(ValueError, match="Decision replay differs"):
        replay(bundle)


def test_post_sends_saved_content_with_only_artifact_link_added(
    audit_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Posting never refreshes data or generates another summary."""
    results, output = audit_run
    bundle = prepare(results, output, "polymarket", "Saved summary")
    saved = replay(bundle)
    sent = []
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.invalid/webhook")
    monkeypatch.setenv("REPORT_ARTIFACT_URL", "https://example.invalid/artifact")
    monkeypatch.setattr(
        notify_slack, "post_to_slack", lambda _url, payload: sent.append(payload)
    )
    post(bundle)
    assert sent[0]["blocks"][:-1] == saved["blocks"]
    assert sent[0]["text"] == saved["text"]
    assert "example.invalid/artifact" in json.dumps(sent[0]["blocks"][-1])


def test_roster_snapshot_rejects_partial_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit capture is strict even though the legacy resolver is permissive."""
    monkeypatch.setattr(tool_usage, "fetch_tools_for_metadata", lambda _: ["good"])
    monkeypatch.setattr(
        tool_usage,
        "_post_graphql",
        lambda *_: {
            "meches": [
                {"address": "0xa", "service": {"metadata": [{"metadata": "0x11"}]}},
                {"address": "0xb", "service": None},
            ]
        },
    )
    with pytest.raises(ValueError, match="Missing metadata"):
        tool_usage.resolve_mech_tools(["0xa", "0xb"], "https://example.invalid", {})


@pytest.mark.parametrize("stage", ["release", "deployment"])
@pytest.mark.parametrize("exception", [AttributeError, TypeError, RuntimeError])
def test_unexpected_discovery_error_does_not_abort_audit_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    exception: type[Exception],
) -> None:
    """Unexpected upstream failures become auditable unavailable snapshots."""

    def release() -> str:
        if stage == "release":
            raise exception("unexpected release shape")
        return "v-test"

    def resolve(_addresses: Any, url: str, provenance: dict[str, Any]) -> list[str]:
        if "gnosis" in url:
            provenance["tools"] = ["resolved-before-failure"]
            raise exception("unexpected manifest shape")
        return ["healthy-platform"]

    monkeypatch.setattr(tool_usage, "latest_trader_ref", release)
    monkeypatch.setattr(
        tool_usage, "_http_get", lambda _: 'valid_mechs: ${VALID_MECHS:list:["0xa"]}'
    )
    monkeypatch.setattr(tool_usage, "resolve_mech_tools", resolve)
    output = tmp_path / "audit"
    initialize(tmp_path / "results", output)
    assert (output / "run.json").is_file()
    snapshot = json.loads((output / "deployment_snapshot.json").read_text())
    assert tool_usage.platform_roster(snapshot, "omen")["status"] == "unavailable"
    if stage == "release":
        assert exception.__name__ in snapshot["error"]
        assert (
            tool_usage.platform_roster(snapshot, "polymarket")["status"]
            == "unavailable"
        )
    else:
        assert exception.__name__ in snapshot["deployments"]["omenstrat Pearl"]["error"]
        assert tool_usage.platform_roster(snapshot, "omen")["tools"] == [
            "resolved-before-failure"
        ]
        assert (
            tool_usage.platform_roster(snapshot, "polymarket")["status"] == "complete"
        )
        assert tool_usage.platform_roster(snapshot, "polymarket")["tools"] == [
            "healthy-platform"
        ]


def test_audit_initialization_still_surfaces_file_write_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery recovery does not swallow failures to persist audit evidence."""
    monkeypatch.setattr(tool_usage, "latest_trader_ref", lambda: "v-test")

    def write_failure(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("cannot persist audit")

    monkeypatch.setattr(report_audit, "_write", write_failure)
    with pytest.raises(OSError, match="cannot persist audit"):
        initialize(tmp_path / "results", tmp_path / "audit")


def test_roster_snapshot_retains_manifest_ids_and_partial_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The captured provenance explains both membership and lookup failures."""
    monkeypatch.setattr(tool_usage, "latest_trader_ref", lambda: "v-test")
    monkeypatch.setattr(
        tool_usage, "_http_get", lambda _url: 'valid_mechs: ${VALID_MECHS:list:["0xa"]}'
    )

    def resolve(_addresses: Any, url: str, provenance: dict[str, Any]) -> list[str]:
        if "polygon" in url:
            raise ValueError("missing mech")
        provenance["mechs"] = {"0xa": "0x11"}
        provenance["manifests"] = {"f0170122011": [NEW_TOOL]}
        return [NEW_TOOL]

    monkeypatch.setattr(tool_usage, "resolve_mech_tools", resolve)
    snapshot = tool_usage.fetch_deployment_snapshot()
    assert snapshot["release_ref"] == "v-test"
    assert snapshot["deployments"]["omenstrat Pearl"]["manifests"]
    assert tool_usage.platform_roster(snapshot, "omen")["tools"] == [NEW_TOOL]
    assert tool_usage.platform_roster(snapshot, "polymarket")["status"] == "unavailable"
    assert tool_usage.snapshot_valid_tools(snapshot)["polystrat Pearl"] is None


def test_captured_utility_classifications_flow_through_prepare_and_replay(
    audit_run: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real capture classifies known utilities; replay uses only saved roles."""
    monkeypatch.setattr(tool_usage, "latest_trader_ref", lambda: "v-test")
    monkeypatch.setattr(
        tool_usage, "_http_get", lambda _url: 'valid_mechs: ${VALID_MECHS:list:["0xa"]}'
    )
    monkeypatch.setattr(
        tool_usage,
        "resolve_mech_tools",
        lambda *_: ["good", "propose_question", "resolve-market-jury-v1", "unknown"],
    )
    snapshot = tool_usage.fetch_deployment_snapshot()
    results, output = audit_run
    (output / "deployment_snapshot.json").write_text(json.dumps(snapshot))
    bundle = prepare(results, output, "omen", "Summary")
    record = json.loads((bundle / "decision.json").read_text())
    assert record["counts"]["manifest_tools"] == 4
    assert record["counts"]["non_prediction"] == 2
    assert record["counts"]["forecasting_or_unknown"] == 2
    assert record["counts"]["remaining_unassessed"] == 1
    for tool in ("propose_question", "resolve-market-jury-v1"):
        assert record["production"][tool]["classification"] == "non_prediction"
    assert record["production"]["unknown"]["classification"] == "unknown"
    monkeypatch.setattr(tool_usage, "NON_PREDICTION_TOOLS", {})
    text = json.dumps(replay(bundle))
    assert "2 confirmed non-prediction" in text
    assert "Warning: `propose_question`" not in text
    assert "Warning: `resolve-market-jury-v1`" not in text
    assert snapshot["non_prediction_tools"]["propose-question"].endswith(
        "propose_question.py"
    )


def test_direct_notifier_classifies_manifest_utilities(
    audit_run: tuple[Path, Path],
) -> None:
    """The direct report also distinguishes known utility roles from no data."""
    results, _ = audit_run
    payload = build_concise_digest_message(
        results,
        "omen",
        "Summary",
        deployed_tools=["good", "propose-question", "resolve-market-jury-v1"],
    )
    assert "2 confirmed non-prediction" in json.dumps(payload)
    assert "unknown tool classification" not in json.dumps(payload)


@pytest.mark.parametrize(
    "failure", ["missing-address", "missing-metadata", "manifest-failure"]
)
def test_partial_roster_retains_findings_but_blocks_actions(
    audit_run: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Resolved manifest members survive a failed sibling mech for audit evidence."""
    monkeypatch.setattr(tool_usage, "latest_trader_ref", lambda: "v-test")
    monkeypatch.setattr(
        tool_usage,
        "_http_get",
        lambda _: 'valid_mechs: ${VALID_MECHS:list:["0xa","0xb"]}',
    )
    meches: list[dict[str, Any]] = [
        {"address": "0xa", "service": {"metadata": [{"metadata": "0x11"}]}}
    ]
    if failure != "missing-address":
        meches.append(
            {
                "address": "0xb",
                "service": (
                    None
                    if failure == "missing-metadata"
                    else {"metadata": [{"metadata": "0x22"}]}
                ),
            }
        )
    monkeypatch.setattr(tool_usage, "_post_graphql", lambda *_: {"meches": meches})

    def manifest(metadata_hash: str) -> list[str]:
        if metadata_hash == "0x22":
            raise ValueError("unavailable manifest")
        return ["bad", "good"]

    monkeypatch.setattr(tool_usage, "fetch_tools_for_metadata", manifest)
    snapshot = tool_usage.fetch_deployment_snapshot()
    assert tool_usage.platform_roster(snapshot, "omen")["status"] == "unavailable"
    assert tool_usage.platform_roster(snapshot, "omen")["tools"] == ["bad", "good"]
    results, output = audit_run
    (output / "deployment_snapshot.json").write_text(json.dumps(snapshot))
    bundle = prepare(results, output, "omen", "Summary")
    record = json.loads((bundle / "decision.json").read_text())
    assert record["decision"]["token"] == "DECISION UNAVAILABLE"
    assert record["decision"]["demote"] == []
    assert record["counts"]["remaining_deployed"] is None
    text = json.dumps(replay(bundle))
    assert "finding only; action blocked" in text
    assert "Brier 0.3100" in text
    assert "remain deployed" not in text
