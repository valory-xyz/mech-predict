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
"""Capture, prepare, replay, and post immutable benchmark decision bundles.

Run init after state restore and before scoring, prepare after analysis, upload
its output, then post. Replay never fetches data, deployments, or an LLM summary.
An artifact reconstructs decisions from saved aggregates; it does not claim to
re-run the underlying raw-row scoring or historical external data queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.decision_report import METRICS, build_decision_record, decision_markdown
from benchmark.digest_tables import WINDOW_FILES, build_concise_digest_message
from benchmark.slack_blocks import context
from benchmark.tool_usage import fetch_deployment_snapshot, platform_roster
from benchmark.tools import TOOL_REGISTRY

log = logging.getLogger(__name__)

PLATFORMS = ("omen", "polymarket")
CONFIG_KEYS = ("USE_MECH_ANALYTICS_ROWS", "BENCHMARK_ROLLING_WINDOW_DAYS")


def _read(path: Path) -> dict[str, Any]:
    """Read a JSON object; reject malformed shapes and non-finite numbers."""

    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected an object")
    return value


def _write(path: Path, value: Any) -> None:
    """Write deterministic UTF-8 JSON with no non-finite numeric values."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    """Hash the exact bytes retained in an artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: Any) -> datetime:
    """Parse an explicitly zoned provenance timestamp."""
    if not isinstance(value, str):
        raise ValueError("Provenance timestamp requires a string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Provenance timestamp requires a timezone")
    return result


def initialize(
    results: Path, output: Path, snapshot: dict[str, Any] | None = None
) -> None:
    """Start a fresh run and discard only previously derived report outputs.

    :param results: existing scorer output directory (resume state is retained).
    :param output: new run/attempt directory; must not already exist.
    :param snapshot: optional frozen roster for offline validation.
    """
    output.mkdir(parents=True, exist_ok=False)
    # Scorer generation timestamps have second precision. Match it so an
    # output generated immediately after initialization is not called stale.
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write(
        output / "run.json",
        {
            "schema_version": 1,
            "started_at": started,
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "source_commit": os.environ.get("GITHUB_SHA"),
            "config": {key: os.environ.get(key) for key in CONFIG_KEYS},
            "prediction_tools": sorted(TOOL_REGISTRY),
        },
    )
    for platform in PLATFORMS:
        derived = [
            WINDOW_FILES[key].format(platform=platform) for key in ("at", "w1", "w2")
        ]
        derived += [f"report_{platform}.md", f"analysis_scores_{platform}.json"]
        for name in derived:
            (results / name).unlink(missing_ok=True)
    _write(
        output / "deployment_snapshot.json",
        snapshot if snapshot is not None else fetch_deployment_snapshot(),
    )


def _score_error(payload: dict[str, Any], started_at: str) -> str | None:
    """Validate freshness and the score fields consumed by the report."""
    try:
        if _timestamp(payload.get("generated_at", "")) < _timestamp(started_at):
            return "stale: generated before this run"
        groups = payload["by_tool"]
        category_groups = payload.get("by_tool_category", {})
        if not isinstance(category_groups, dict):
            return "malformed by_tool_category"
        if not isinstance(groups, dict):
            return "malformed by_tool"
        for name, stats in list(groups.items()) + list(category_groups.items()):
            if not isinstance(name, str) or not isinstance(stats, dict):
                return "malformed tool statistics"
            for key in (*METRICS, "directional_accuracy", "outcome_yes_rate"):
                value = stats.get(key)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    return f"malformed {name}.{key}"
                if (
                    value is not None
                    and key in ("valid_n", "edge_n", "disagree_n")
                    and (value < 0 or int(value) != value)
                ):
                    return f"invalid count {name}.{key}"
    except (KeyError, TypeError, ValueError):
        return "missing or invalid generation timestamp/by_tool"
    return None


def _capture_inputs(
    results: Path, bundle: Path, platform: str, run: dict[str, Any]
) -> dict[str, str | None]:
    """Save exact inputs, including rejected files, with explicit status."""
    statuses: dict[str, str | None] = {}
    names = [name.format(platform=platform) for name in WINDOW_FILES.values()]
    names.append(f"analysis_scores_{platform}.json")
    for name in names:
        source = results / name
        if not source.is_file():
            statuses[name] = "missing"
            continue
        shutil.copyfile(source, bundle / name)
        try:
            statuses[name] = _score_error(_read(bundle / name), run["started_at"])
        except (OSError, ValueError):
            statuses[name] = "malformed JSON"
    return statuses


def _decision_inputs(
    bundle: Path, manifest: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str | None]]:
    """Load only accepted scores; rejected snapshots remain auditable."""
    payloads: dict[str, dict[str, Any]] = {}
    statuses = {}
    for key, template in WINDOW_FILES.items():
        name = template.format(platform=manifest["platform"])
        status = manifest["input_status"][name]
        payloads[key] = {} if status else _read(bundle / name)
        statuses[key] = status
    return payloads, statuses


def _rebuild(bundle: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Recompute decisions using saved roster, registry scope, and scores."""
    payloads, statuses = _decision_inputs(bundle, manifest)
    snapshot = _read(bundle / "deployment_snapshot.json")
    return build_decision_record(
        manifest["platform"],
        payloads,
        platform_roster(snapshot, manifest["platform"]),
        manifest["run"]["prediction_tools"],
        input_status=statuses,
    )


def _render(
    bundle: Path, manifest: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Render accepted files only, without reading rejected stale windows."""
    # The digest's category/warning renderers consume paths. An isolated view
    # ensures those cannot accidentally show a rejected file as current data.
    import tempfile  # pylint: disable=import-outside-toplevel

    with tempfile.TemporaryDirectory() as temp:
        accepted = Path(temp)
        for template in WINDOW_FILES.values():
            name = template.format(platform=manifest["platform"])
            if manifest["input_status"][name] is None:
                shutil.copyfile(bundle / name, accepted / name)
        summary = (bundle / "summary.txt").read_text(encoding="utf-8")
        # Saved prose can refer to a rejected comparison. Keep it in the
        # artifact, but do not present that narrative as a verified trend.
        if any(
            manifest["input_status"][name]
            for name in (
                WINDOW_FILES["w1"].format(platform=manifest["platform"]),
                WINDOW_FILES["w2"].format(platform=manifest["platform"]),
                f"analysis_scores_{manifest['platform']}.json",
            )
        ):
            summary = "*Summary:* Trend summary unavailable because an analysis or comparison input was rejected; consult the available decision evidence."
        payload = build_concise_digest_message(
            accepted,
            manifest["platform"],
            summary,
            decision_record=record,
        )
        if payload is None:
            raise ValueError("Decision renderer produced no message")
        return payload


def prepare(
    results: Path, output: Path, platform: str, summary: str | None = None
) -> Path:
    """Freeze a report and its decision evidence before any Slack posting.

    :param results: scorer/analysis output directory.
    :param output: initialized run directory.
    :param platform: platform to prepare independently.
    :param summary: optional saved summary for offline preparation.
    :return: the complete platform bundle directory.
    """
    run = _read(output / "run.json")
    bundle = output / platform
    bundle.mkdir(exist_ok=False)
    shutil.copyfile(
        output / "deployment_snapshot.json", bundle / "deployment_snapshot.json"
    )
    manifest = {
        "schema_version": 1,
        "platform": platform,
        "run": run,
        "input_status": _capture_inputs(results, bundle, platform, run),
    }
    source = results / f"report_{platform}.md"
    report = (
        source.read_text(encoding="utf-8")
        if source.is_file()
        else "Platform analysis unavailable.\n"
    )
    (bundle / "source_report.md").write_text(report, encoding="utf-8")
    if summary is None:
        key = os.environ.get("OPENAI_API_KEY")
        summary = "*Summary:* Platform summary unavailable; consult the full report."
        if key and source.is_file():
            from benchmark.notify_slack import (  # pylint: disable=import-outside-toplevel,protected-access
                _summary_only,
                summarize_report,
            )

            try:
                label = "Omenstrat" if platform == "omen" else "Polystrat"
                summary = _summary_only(summarize_report(report, key, label))
            except Exception as exc:  # pylint: disable=broad-except
                # The statistical evidence remains usable when narrative generation fails.
                log.warning(
                    "Summary generation failed (%s); keeping computed evidence.",
                    type(exc).__name__,
                )
    (bundle / "summary.txt").write_text(summary, encoding="utf-8")
    record = _rebuild(bundle, manifest)
    _write(bundle / "decision.json", record)
    audited_report = report + "\n" + decision_markdown(record)
    (bundle / "report.md").write_text(audited_report, encoding="utf-8")
    if source.is_file():
        source.write_text(audited_report, encoding="utf-8")
    _write(bundle / "slack_payload.json", _render(bundle, manifest, record))
    manifest["files"] = {
        path.name: _sha(path) for path in sorted(bundle.iterdir()) if path.is_file()
    }
    _write(bundle / "manifest.json", manifest)
    replay(bundle)
    return bundle


def replay(bundle: Path) -> dict[str, Any]:
    """Verify hashes and reproduce the decision, report section, and Slack body.

    :param bundle: downloaded platform artifact directory.
    :return: the verified saved Slack payload.
    :raises ValueError: if inputs or outputs differ from the saved decision.
    """
    manifest = _read(bundle / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported audit schema")
    required = {
        "deployment_snapshot.json",
        "source_report.md",
        "summary.txt",
        "decision.json",
        "report.md",
        "slack_payload.json",
    }
    required.update(
        name for name, status in manifest["input_status"].items() if status != "missing"
    )
    if set(manifest["files"]) != required:
        raise ValueError("Audit manifest does not cover every consumed file")
    for name, expected in manifest["files"].items():
        if (
            Path(name).name != name
            or not (bundle / name).is_file()
            or _sha(bundle / name) != expected
        ):
            raise ValueError(f"Audit file missing or hash mismatch: {name}")
    record = _rebuild(bundle, manifest)
    if record != _read(bundle / "decision.json"):
        raise ValueError(
            "Decision replay differs: use the bundle's source commit and configuration"
        )
    report = (
        (bundle / "source_report.md").read_text(encoding="utf-8")
        + "\n"
        + decision_markdown(record)
    )
    if report != (bundle / "report.md").read_text(encoding="utf-8"):
        raise ValueError("Markdown decision replay differs")
    payload = _render(bundle, manifest, record)
    if payload != _read(bundle / "slack_payload.json"):
        raise ValueError("Slack decision replay differs")
    return payload


def post(bundle: Path, *, dry_run: bool = False) -> None:
    """Post only verified saved content, attaching the uploaded artifact URL."""
    from benchmark.notify_slack import (  # pylint: disable=import-outside-toplevel,protected-access
        _build_report_url,
        post_to_slack,
    )

    payload = replay(bundle)
    url = _build_report_url()
    if url:
        payload["blocks"].append(context(f"<{url}|Full report and decision evidence>"))
    if dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print("SLACK_WEBHOOK_URL unset; verified bundle retained without posting.")
        return
    post_to_slack(webhook, payload)


def main() -> None:
    """CLI for the prepare → upload → post workflow and offline replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--results", type=Path, default=Path("benchmark/results"))
    init.add_argument("--output", type=Path, required=True)
    init.add_argument(
        "--snapshot",
        type=Path,
        help="Frozen deployment snapshot for offline validation",
    )
    prep = commands.add_parser("prepare")
    prep.add_argument("--results", type=Path, default=Path("benchmark/results"))
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--platform", choices=PLATFORMS, required=True)
    prep.add_argument(
        "--summary", type=Path, help="Saved summary; skips LLM generation"
    )
    for name in ("replay", "post"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        if name == "post":
            command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        initialize(
            args.results, args.output, _read(args.snapshot) if args.snapshot else None
        )
    elif args.command == "prepare":
        print(
            prepare(
                args.results,
                args.output,
                args.platform,
                args.summary.read_text(encoding="utf-8") if args.summary else None,
            )
        )
    elif args.command == "replay":
        replay(args.bundle)
        print(
            "Verified: inputs, decisions, roster counts, Markdown, and Slack content reproduce offline."
        )
    else:
        post(args.bundle, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
