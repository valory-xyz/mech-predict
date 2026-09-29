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

from typing import Any, Collection

from benchmark.decision_policy import (
    COIN_FLIP,
    MIN_SAMPLE_SIZE,
    NO_SKILL_MARGIN,
    PROMOTE_DELTA,
    RELIABILITY_GATE,
    Z_ONE_SIDED_95,
    _decision_state,
    _edge_lower_bound,
    _num,
    _survivors,
    _verdict,
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
        ("no data", "n=", "no spread", "needs --rebuild", "unknown", "ambiguous")
    ):
        return "insufficient_evidence"
    return "no_action_threshold_met"


def _index(payload: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Index aliases without merging possibly overlapping scored pools."""
    result: dict[str, Any] = {}
    ambiguous: set[str] = set()
    for name, stats in (payload.get("by_tool") or {}).items():
        canonical = normalize_tool_name(name)
        if canonical in result:
            ambiguous.add(canonical)
        result[canonical] = (name, stats)
    return result, ambiguous


def build_decision_record(
    platform: str,
    payloads: dict[str, dict[str, Any]],
    roster: dict[str, Any],
    prediction_tools: Collection[str],
    *,
    input_errors: Collection[str] = (),
) -> dict[str, Any]:
    """Account for the full roster and apply the existing policy once.

    :param platform: platform identifier.
    :param payloads: exact scores keyed by at, w1, w2, tournament.
    :param roster: saved platform deployment snapshot with status and tools.
    :param prediction_tools: known forecasting tools, including unscored ones.
    :param input_errors: missing, malformed, or stale required inputs.
    :return: JSON-serializable evidence and actionable decision.
    """
    indexes = {key: _index(value) for key, value in payloads.items()}
    known = {normalize_tool_name(name) for name in prediction_tools}
    names: dict[str, list[str]] = {}
    for name in roster.get("tools", []):
        names.setdefault(normalize_tool_name(name), []).append(name)
    errors = list(input_errors)
    if roster.get("status") != "complete":
        errors.append("Deployment roster is incomplete or unavailable.")
    production: dict[str, Any] = {}
    candidates: dict[str, Any] = {}
    for canonical, aliases in sorted(names.items()):
        at = indexes.get("at", ({}, set()))[0].get(canonical, (None, {}))[1]
        week = indexes.get("w1", ({}, set()))[0].get(canonical, (None, {}))[1]
        scored_names = [
            index[canonical][0]
            for key, (index, _) in indexes.items()
            if key != "tournament" and canonical in index
        ]
        name = sorted(set(scored_names))[0] if scored_names else sorted(aliases)[0]
        ambiguous = any(canonical in collisions for _, collisions in indexes.values())
        has_prediction = any(
            _num(index.get(canonical, (None, {}))[1].get("brier")) is not None
            for index, _ in indexes.values()
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
            _verdict(at, True, week)
            if classification == "prediction"
            else (
                "not a prediction tool"
                if classification == "non_prediction"
                else "unknown tool classification"
            )
        )
        if ambiguous:
            verdict = "ambiguous score aliases"
            errors.append(f"{name}: multiple score keys normalize to {canonical}.")
        production[name] = {
            "canonical_name": canonical,
            "manifest_names": sorted(set(aliases)),
            "classification": classification,
            "verdict": verdict,
            "rule": _rule(verdict),
            "evidence": {"at": _evidence(at), "w1": _evidence(week)},
        }
    # Retain the existing tournament eligibility scope; this PR repairs
    # production membership without broadening the candidate policy.
    tournament, collisions = indexes.get("tournament", ({}, set()))
    for canonical, (name, stats) in sorted(tournament.items()):
        if canonical not in known or _num(stats.get("brier")) is None:
            continue
        verdict = _verdict(stats, False)
        if canonical in collisions:
            verdict = "ambiguous score aliases"
            errors.append(f"{name}: ambiguous tournament score aliases.")
        candidates[name] = {
            "verdict": verdict,
            "rule": _rule(verdict),
            "evidence": {"tournament": _evidence(stats)},
        }
    prod = {
        name: row["verdict"]
        for name, row in production.items()
        if row["classification"] != "non_prediction"
    }
    tourn = {name: row["verdict"] for name, row in candidates.items()}
    state, token, promote, demote = _decision_state(prod, tourn)
    if errors:
        state, token, promote, demote = "unavailable", "DECISION UNAVAILABLE", [], []
    # Blocked policy results are findings, not proposed removals.
    if state == "blocked":
        demote = []
    remaining = {name: verdict for name, verdict in prod.items() if name not in demote}
    return {
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
    lines = [f"*Decision: {action['token']}*"]
    if action["state"] == "unavailable":
        lines.extend(record["errors"])
        lines.append("Restore the missing inputs before reviewing deployment changes.")
    elif action["state"] == "blocked":
        lines.append(
            "No production tool has an assessed keep verdict. Resolve unassessed tools and review failures at platform level before removing tools."
        )
    elif action["token"] == "NO CHANGE":
        lines.append(
            "No candidate qualifies for promotion, and no assessed production tool meets the demotion criteria."
        )
    for kind, cohort in (("promote", "tournament"), ("demote", "production")):
        for name in action[kind]:
            row = record[cohort][name]
            lines.append(f"• `{name}` — {row['verdict']}\n  {evidence_text(row)}")
    if action["token"].endswith(" FIRST"):
        lines.append("Deploy a qualified replacement before reviewing any demotion.")
    counts = record["counts"]
    if record["roster_status"] == "complete":
        lines.append(
            f"Roster: {counts['forecasting_or_unknown']} forecasting/unknown tools; "
            f"{counts['assessed']} assessed. After proposed demotions: "
            f"{counts['remaining_deployed']} remain deployed; "
            f"{counts['remaining_assessed']} assessed and retained, "
            f"{counts['remaining_flagged']} flagged but retained, "
            f"{counts['remaining_unassessed']} unassessed."
        )
    for name, row in record["production"].items():
        if (
            not row["verdict"].startswith(("keep", "demote"))
            and row["classification"] != "non_prediction"
        ):
            lines.append(f"Warning: `{name}` — {row['verdict']}.")
    if action["demote"] and not action["promote"]:
        lines.append("No tournament candidate qualifies as a replacement.")
    if action["promote"] or action["demote"]:
        lines.append(
            "*Next step:* Review the proposed changes; confirm promotions on an independent window or holdout before deployment."
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
