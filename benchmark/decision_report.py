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
"""Serializable decision facts and shared Slack/Markdown evidence rendering.

The deployment manifest determines membership. The execution registry is only
positive evidence that an unscored tool is a forecaster, never an exclusion list.
No clocks, network calls, or LLM output participate in the decision.
"""

from __future__ import annotations

from typing import Any, Collection, Mapping

from benchmark.decision_policy import (
    COIN_FLIP,
    MIN_SAMPLE_SIZE,
    NO_SKILL_MARGIN,
    PROMOTE_DELTA,
    RELIABILITY_GATE,
    Z_ONE_SIDED_95,
    _below_no_skill,
    _decision_state,
    _edge_lower_bound,
    _num,
    _survivors,
    _verdict,
    _verdict_core,
)
from benchmark.tool_usage import normalize_tool_name

SCHEMA_VERSION = 1
METRICS = (
    "brier",
    "baseline_brier",
    "valid_n",
    "edge",
    "edge_n",
    "edge_sd",
    "conditional_accuracy_rate",
    "disagree_n",
    "reliability",
)
WINDOW_LABELS = {
    "at": "90d",
    "w1": "Current 7d",
    "w2": "Prev 7d",
    "tournament": "Tournament",
}


def policy_parameters() -> dict[str, Any]:
    """Return the exact policy parameters needed to check a saved decision."""
    return {
        "min_sample_size": MIN_SAMPLE_SIZE,
        "no_skill_margin": NO_SKILL_MARGIN,
        "promote_delta": PROMOTE_DELTA,
        "conditional_threshold": COIN_FLIP,
        "one_sided_z": Z_ONE_SIDED_95,
        "reliability_warning_threshold": RELIABILITY_GATE,
    }


def _evidence(stats: dict[str, Any]) -> dict[str, Any]:
    """Retain the gate inputs, derived bound, and baseline difference."""
    evidence = {key: stats.get(key) for key in METRICS}
    brier, base = _num(stats.get("brier")), _num(stats.get("baseline_brier"))
    evidence["brier_minus_baseline"] = (
        None if brier is None or base is None else round(brier - base, 10)
    )
    evidence["edge_lower_bound"] = _edge_lower_bound(stats)
    return evidence


def _rule(verdict: str) -> str:
    """Give each policy result a stable reason identifier."""
    if verdict == "not a prediction tool":
        return "not_applicable"
    if "condAcc" in verdict:
        return "conditional_accuracy"
    if "no-skill" in verdict:
        return "sustained_baseline" if verdict.startswith("demote") else "baseline"
    if verdict.startswith(("PROMOTE", "keep (floor ok")):
        return "promotion_floor"
    if verdict.startswith(
        (
            "no data",
            "n=",
            "no spread",
            "needs --rebuild",
            "unknown",
            "ambiguous",
            "unavailable",
        )
    ):
        return "insufficient_evidence"
    return "no_action_threshold_met"


def _index(payload: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Index aliases without merging possibly overlapping scored pools."""
    result: dict[str, Any] = {}
    ambiguous: set[str] = set()
    for name, stats in sorted((payload.get("by_tool") or {}).items()):
        canonical = normalize_tool_name(name)
        if canonical in result:
            ambiguous.add(canonical)
        result[canonical] = (name, stats)
    for canonical in ambiguous:
        del result[canonical]
    return result, ambiguous


def _production_verdict(
    at: dict[str, Any], week: dict[str, Any], issues: dict[str, str]
) -> str:
    """Reject only inputs reached by this tool's existing policy gates.

    A failed week differs from a valid, empty week. Do not let the policy's
    absent-week fallback turn an unevaluated baseline gate into a survivor.

    :param at: accepted, unambiguous trailing scores for this tool.
    :param week: accepted, unambiguous recent scores for this tool.
    :param issues: unavailable or ambiguous input reasons keyed by window.
    :return: existing policy verdict or the unavailable prerequisite.
    """
    if "at" in issues:
        return issues["at"]
    if "w1" in issues and _below_no_skill(at) and _verdict_core(at, True) == "keep":
        return issues["w1"]
    return _verdict(at, True, week)


def _candidate_records(payload: dict[str, Any], known: set[str]) -> dict[str, Any]:
    """Assess candidates independently, retaining the existing eligibility scope."""
    # Retain the existing tournament eligibility scope; this PR repairs
    # production membership without broadening the candidate policy.
    candidates: dict[str, Any] = {}
    tournament, collisions = _index(payload)
    for canonical in sorted(set(tournament) | collisions):
        candidate_aliases = {
            name: stats
            for name, stats in payload.get("by_tool", {}).items()
            if normalize_tool_name(name) == canonical
        }
        name = sorted(candidate_aliases)[0]
        stats = tournament.get(canonical, (None, {}))[1]
        if canonical not in known or not any(
            _num(row.get("brier")) is not None for row in candidate_aliases.values()
        ):
            continue
        verdict = _verdict(stats, False)
        if canonical in collisions:
            verdict = "ambiguous score aliases"
        candidates[name] = {
            "verdict": verdict,
            "rule": _rule(verdict),
            "evidence": {"tournament": _evidence(stats)},
        }
    return candidates


def _assessment_warnings(record: dict[str, Any]) -> list[str]:
    """Own input and tool assessment warnings for both Slack and Markdown."""
    warnings = []
    for key, status in record["input_status"].items():
        if status:
            label = (
                "Candidate evaluation" if key == "tournament" else WINDOW_LABELS[key]
            )
            warnings.append(f"Warning: {label} unavailable — {status}.")
    for key in ("w2", "tournament"):
        for name in record["ambiguous_inputs"].get(key, []):
            warnings.append(
                f"Warning: `{name}` — ambiguous score aliases in {WINDOW_LABELS[key]}; excluded from that evaluation."
            )
    for name, row in record["production"].items():
        verdict = row["verdict"]
        if verdict == "no data" or verdict.startswith("n="):
            warnings.append(
                f":warning: `{name}` has insufficient data to judge ({verdict}). "
                "Investigate prediction failures and unresolved markets before acting."
            )
        elif (
            not verdict.startswith(("keep", "demote"))
            and row["classification"] != "non_prediction"
        ):
            warnings.append(f"Warning: `{name}` — {verdict}.")
        if (
            verdict.startswith("demote")
            and "condAcc" in verdict
            and "w1" not in row["input_issues"]
        ):
            week = row["evidence"]["w1"]
            counts = [_num(week.get(key)) for key in ("valid_n", "edge_n")]
            weekly_n = min(
                (int(value) for value in counts if value is not None), default=0
            )
            if weekly_n < MIN_SAMPLE_SIZE:
                warnings.append(
                    f":warning: `{name}` has insufficient weekly data (n={weekly_n}). "
                    "Its finding is based on the 90d condAcc result, not the weekly trend."
                )
    return warnings


def _accepted_inputs(
    payloads: dict[str, dict[str, Any]], input_status: Mapping[str, str | None] | None
) -> tuple[dict[str, dict[str, Any]], dict[str, str | None]]:
    """Apply audit rejections and basic shape checks to direct notifier inputs."""
    statuses = {
        key: (input_status or {}).get(key) or (None if payloads.get(key) else "missing")
        for key in WINDOW_LABELS
    }
    for key, payload in payloads.items():
        groups = payload.get("by_tool")
        if (
            key in statuses
            and not statuses[key]
            and (
                not isinstance(groups, dict)
                or any(not isinstance(stats, dict) for stats in groups.values())
            )
        ):
            statuses[key] = "malformed tool statistics"
    payloads = {key: {} if statuses[key] else payloads[key] for key in WINDOW_LABELS}
    return payloads, statuses


def build_decision_record(
    platform: str,
    payloads: dict[str, dict[str, Any]],
    roster: dict[str, Any],
    prediction_tools: Collection[str],
    *,
    input_status: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Account for the full roster and apply the existing policy once.

    :param platform: platform identifier.
    :param payloads: exact scores keyed by at, w1, w2, tournament.
    :param roster: saved platform deployment snapshot with status and tools.
    :param prediction_tools: known forecasting tools, including unscored ones.
    :param input_status: rejection reasons keyed by input window; None is accepted.
    :return: JSON-serializable evidence and actionable decision.
    """
    payloads, statuses = _accepted_inputs(payloads, input_status)
    indexes = {key: _index(value) for key, value in payloads.items()}
    known = {normalize_tool_name(name) for name in prediction_tools}
    names: dict[str, list[str]] = {}
    for name in roster.get("tools", []):
        names.setdefault(normalize_tool_name(name), []).append(name)
    errors = []
    if roster.get("status") != "complete":
        errors.append("Deployment roster is incomplete or unavailable.")
    production: dict[str, Any] = {}
    for canonical, aliases in sorted(names.items()):
        at = indexes.get("at", ({}, set()))[0].get(canonical, (None, {}))[1]
        week = indexes.get("w1", ({}, set()))[0].get(canonical, (None, {}))[1]
        name = min(
            [
                index[canonical][0]
                for key, (index, _) in indexes.items()
                if key != "tournament" and canonical in index
            ]
            or aliases
        )
        issues = {
            key: (
                f"unavailable {WINDOW_LABELS[key]} scores: {statuses[key]}"
                if statuses[key]
                else f"ambiguous score aliases in {WINDOW_LABELS[key]}"
            )
            for key in ("at", "w1")
            if statuses[key] or canonical in indexes[key][1]
        }
        has_prediction = any(
            _num(stats.get("brier")) is not None
            for payload in payloads.values()
            for scored_name, stats in payload.get("by_tool", {}).items()
            if normalize_tool_name(scored_name) == canonical
        )
        classification = (
            "prediction"
            if has_prediction or canonical in known
            else (
                "non_prediction"
                if canonical
                in {
                    normalize_tool_name(tool)
                    for tool in roster.get("non_prediction_tools", [])
                }
                else "unknown"
            )
        )
        verdict = (
            _production_verdict(at, week, issues)
            if classification == "prediction"
            else (
                "not a prediction tool"
                if classification == "non_prediction"
                else "unknown tool classification"
            )
        )
        production[name] = {
            "canonical_name": canonical,
            "manifest_names": sorted(set(aliases)),
            "classification": classification,
            "verdict": verdict,
            "rule": _rule(verdict),
            "input_issues": issues,
            "evidence": {"at": _evidence(at), "w1": _evidence(week)},
        }
    candidates = _candidate_records(payloads["tournament"], known)
    prod = {
        name: row["verdict"]
        for name, row in production.items()
        if row["classification"] != "non_prediction"
    }
    tourn = {name: row["verdict"] for name, row in candidates.items()}
    state, token, promote, demote = _decision_state(prod, tourn)
    if statuses["at"] and not promote:
        errors.append("Production evaluation unavailable: 90d scores were rejected.")
    if errors:
        state, token, promote, demote = "unavailable", "DECISION UNAVAILABLE", [], []
    # Blocked policy results are findings, not proposed removals.
    if state == "blocked":
        demote = []
    remaining = {name: verdict for name, verdict in prod.items() if name not in demote}
    record = {
        "schema_version": SCHEMA_VERSION,
        "platform": platform,
        "policy": policy_parameters(),
        "windows": {
            key: {
                "generated_at": value.get("generated_at"),
                "requested_window": value.get("requested_window"),
                "observed_start": value.get("window_start"),
                "observed_end": value.get("window_end"),
            }
            for key, value in payloads.items()
        },
        "roster_status": roster.get("status"),
        "input_status": statuses,
        "ambiguous_inputs": {
            key: sorted(collisions)
            for key, (_, collisions) in indexes.items()
            if collisions
        },
        "production": production,
        "tournament": candidates,
        "decision": {
            "state": state,
            "token": token,
            "promote": promote,
            "demote": demote,
        },
        "counts": {
            "manifest_tools": len(production),
            "non_prediction": len(production) - len(prod),
            "forecasting_or_unknown": len(prod),
            "assessed": len(_survivors(prod))
            + sum(v.startswith("demote") for v in prod.values()),
            "remaining_deployed": (
                len(remaining) if roster.get("status") == "complete" else None
            ),
            "remaining_assessed": len(_survivors(remaining)),
            "remaining_flagged": sum(
                v.startswith("demote") for v in remaining.values()
            ),
            "remaining_unassessed": sum(
                not v.startswith(("keep", "demote")) for v in remaining.values()
            ),
        },
        "errors": sorted(set(errors)),
    }
    record["warnings"] = _assessment_warnings(record)
    return record


def _number(value: Any, *, signed: bool = False) -> str:
    """Format stored evidence consistently in both reports."""
    if _num(value) is None:
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _count(value: Any) -> str:
    """Preserve explicit zero counts and name unavailable denominators."""
    return "n/a" if value is None else str(value)


def evidence_text(row: dict[str, Any]) -> str:
    """Render the actual rule inputs, including their distinct denominators."""
    evidence = row["evidence"]
    at = evidence.get("at", evidence.get("tournament", {}))
    floor = (
        f"Edge lower bound {_number(at.get('edge_lower_bound'), signed=True)} "
        f"(promotion requires > +{PROMOTE_DELTA:.2f}; "
        f"priced n={_count(at.get('edge_n'))}); "
    )
    conditional = at.get("conditional_accuracy_rate")
    cond_text = "n/a" if conditional is None else f"{conditional:.1%}"
    if row["rule"] in ("sustained_baseline", "baseline"):
        parts = [
            floor
            + f"condAcc {cond_text} (disagreements n={_count(at.get('disagree_n'))}); "
            + f"Brier − baseline must be ≥ {NO_SKILL_MARGIN:.2f}"
        ]
        for key, label in (
            ("at", "90d"),
            ("w1", "Current week"),
            ("tournament", "Tournament"),
        ):
            if key not in evidence:
                continue
            stats = evidence[key]
            parts.append(
                f"{label}: Brier {_number(stats.get('brier'))}; "
                f"baseline {_number(stats.get('baseline_brier'))}; "
                f"gap {_number(stats.get('brier_minus_baseline'), signed=True)} "
                f"(scored n={_count(stats.get('valid_n'))})"
            )
        return ". ".join(parts) + "."
    return (
        floor
        + f"90d condAcc {cond_text}".replace(
            "90d", "Tournament" if "tournament" in evidence else "90d"
        )
        + f" (disagreements n={_count(at.get('disagree_n'))}; conditional gate fails below 50%). "
        + f"Scored n={_count(at.get('valid_n'))}."
    )


def decision_text(record: dict[str, Any]) -> str:
    """Render the action, roster accounting, and decisive evidence for Slack."""
    action = record["decision"]
    candidate_status = record["input_status"]["tournament"]
    ambiguous_candidates = any(
        row["verdict"].startswith("ambiguous") for row in record["tournament"].values()
    )
    candidates_complete = not candidate_status and not ambiguous_candidates
    lines = [f"*Decision: {action['token']}*"]
    if action["state"] == "unavailable":
        lines.extend(record["errors"])
        lines.append("Restore the missing inputs before reviewing deployment changes.")
    elif action["state"] == "blocked":
        lines.append(
            "No production tool has an assessed keep verdict. Resolve unassessed tools and review failures at platform level before removing tools."
        )
    elif action["token"] == "NO CHANGE":
        lines.append("No assessed production tool meets the demotion criteria.")
    for kind, cohort in (("promote", "tournament"), ("demote", "production")):
        for name in action[kind]:
            row = record[cohort][name]
            lines.append(f"• `{name}` — {row['verdict']}\n  {evidence_text(row)}")
        for name, row in record[cohort].items():
            if row["verdict"].lower().startswith(kind) and name not in action[kind]:
                lines.append(
                    f"• `{name}` — {row['verdict']} (*finding only; action blocked*)\n  {evidence_text(row)}"
                )
    if action["token"].endswith(" FIRST"):
        lines.append("Deploy a qualified replacement before reviewing any demotion.")
    counts = record["counts"]
    if record["roster_status"] == "complete":
        lines.append(
            f"Roster: {counts['manifest_tools']} manifest tools; "
            f"{counts['non_prediction']} confirmed non-prediction; "
            f"{counts['forecasting_or_unknown']} forecasting/unknown tools; "
            f"{counts['assessed']} assessed. After proposed demotions: "
            f"{counts['remaining_deployed']} remain deployed; "
            f"{counts['remaining_assessed']} assessed and retained, "
            f"{counts['remaining_flagged']} flagged but retained, "
            f"{counts['remaining_unassessed']} unassessed."
        )
    else:
        lines.append(
            f"Roster incomplete: {counts['manifest_tools']} resolved manifest tools; "
            f"{counts['assessed']} assessed. Total deployment and remaining counts unavailable."
        )
    lines.extend(record["warnings"])
    if action["demote"] and not action["promote"] and candidates_complete:
        lines.append("No tournament candidate qualifies as a replacement.")
    elif action["token"] == "NO CHANGE" and candidates_complete:
        lines.append("No candidate qualifies for promotion.")
    if action["promote"] or action["demote"]:
        lines.append(
            "*Next step:* Review the proposed changes; confirm promotions on an independent window or holdout before deployment."
            if action["promote"]
            else "*Next step:* Review the proposed demotions."
        )
    return "\n\n".join(lines)


def decision_markdown(record: dict[str, Any]) -> str:
    """Render every tool and its prerequisites in the full audit report."""
    lines = [
        "## Deployment decision",
        decision_text(record),
        "### Complete decision evidence",
        "Counts are scored predictions, not independent markets. Conditional accuracy uses the disagreement subset. Statistical thresholds are unchanged; a verdict is a proposal for human review.",
    ]
    lines.append(
        f"Prerequisites: priced n ≥ {MIN_SAMPLE_SIZE}; recent scored n ≥ {MIN_SAMPLE_SIZE} for sustained-baseline demotions. The 90d and recent windows overlap; they are not independent confirmations."
    )
    lines.append("### Input windows")
    for name, window in record["windows"].items():
        lines.append(
            f"- {name}: query bounds {window['requested_window']}; observed rows {window['observed_start']} to {window['observed_end']}; generated {window['generated_at']}."
        )
    for cohort in ("production", "tournament"):
        for name, row in record[cohort].items():
            lines.append(
                f"- **{name}** ({cohort}): {row['verdict']}. {evidence_text(row)}"
            )
    return "\n\n".join(lines) + "\n"
