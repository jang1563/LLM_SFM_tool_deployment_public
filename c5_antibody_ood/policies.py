"""Deterministic C5 policy baselines and compact aggregate evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from llm_sfm_tool_deployment import Action, Trajectory

from .manifest import (
    c5_trust_schema_issues,
    ideal_trajectory_from_c5_row,
    score_c5_trajectory,
    trajectory_from_c5_row,
    validate_c5_manifest,
)


Policy = Callable[[Mapping[str, Any]], Action]


def c5_trust_all_action(row: Mapping[str, Any]) -> Action:
    """Trust every specialist output regardless of calibration metadata."""

    return Action.TRUST_SPECIALIST_OUTPUT


def c5_general_gate_action(row: Mapping[str, Any]) -> Action:
    """Apply a generic protein-interface threshold without C5 regime checks."""

    visible = _visible(row)
    specialist = _mapping(visible["specialist_result"])
    calibration = _mapping(visible["general_calibration"])
    metric = specialist.get("metric_value")
    threshold = calibration.get("threshold_value")
    if _numeric(metric) and _numeric(threshold) and metric >= threshold:
        return Action.TRUST_SPECIALIST_OUTPUT
    return Action.DEFER_OR_REQUEST_MORE_EVIDENCE


def c5_regime_gate_action(row: Mapping[str, Any]) -> Action:
    """Certify trust with C5 metadata, otherwise return no trusted set."""

    visible = _visible(row)
    if not c5_trust_schema_issues(visible):
        return Action.TRUST_SPECIALIST_OUTPUT
    return Action.DEFER_OR_REQUEST_MORE_EVIDENCE


def c5_fail_closed_action(row: Mapping[str, Any]) -> Action:
    """Route strict C5 gate failures to baseline, verification, or defer."""

    visible = _visible(row)
    if not c5_trust_schema_issues(visible):
        return Action.TRUST_SPECIALIST_OUTPUT
    baseline = _mapping(visible["baseline_result"])
    if baseline.get("available") is True and baseline.get("dominates_specialist") is True:
        return Action.USE_CHEAP_BASELINE
    if visible.get("verification_available") is True:
        return Action.VERIFY_WITH_ASSAY_OR_DATABASE
    return Action.DEFER_OR_REQUEST_MORE_EVIDENCE


C5_POLICIES: dict[str, Policy] = {
    "trust_all": c5_trust_all_action,
    "general_gate": c5_general_gate_action,
    "regime_specific_gate": c5_regime_gate_action,
    "fail_closed": c5_fail_closed_action,
}


def trajectory_for_c5_policy(
    row: Mapping[str, Any],
    policy_name: str,
) -> Trajectory:
    """Build a visible-only trajectory for one deterministic policy."""

    try:
        policy = C5_POLICIES[policy_name]
    except KeyError as exc:
        raise ValueError(f"Unknown C5 policy: {policy_name!r}") from exc
    action = policy(row)
    reason = None
    if action != Action.TRUST_SPECIALIST_OUTPUT:
        issues = c5_trust_schema_issues(_visible(row))
        reason = ",".join(issues) if issues else "general_gate_threshold_not_met"
    return trajectory_from_c5_row(row, action, fail_closed_reason=reason)


def build_c5_baseline_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate four deterministic policies without storing row-level outputs."""

    manifest_issues = validate_c5_manifest(rows)
    oracle_pass = sum(
        score_c5_trajectory(row, ideal_trajectory_from_c5_row(row)).passed
        for row in rows
    )
    policy_reports: dict[str, dict[str, Any]] = {}
    for policy_name in C5_POLICIES:
        results = []
        action_counts: Counter[str] = Counter()
        violation_counts: Counter[str] = Counter()
        unsafe_trust = 0
        for row in rows:
            trajectory = trajectory_for_c5_policy(row, policy_name)
            result = score_c5_trajectory(row, trajectory)
            results.append(result)
            action = Action(trajectory.terminal_action)
            action_counts[action.value] += 1
            violation_counts.update(result.violations)
            if action == Action.TRUST_SPECIALIST_OUTPUT and (
                "c5_interface_safety" in result.violations
                or "c5_trust_gate_completeness" in result.violations
                or "baseline_dominates_do_not_trust_specialist" in result.violations
            ):
                unsafe_trust += 1
        exact = sum(result.passed for result in results)
        mean_score = (
            sum(result.score for result in results) / len(results)
            if results
            else 0.0
        )
        policy_reports[policy_name] = {
            "rows": len(results),
            "exact_pass": exact,
            "exact_pass_rate": round(exact / len(results), 6) if results else 0.0,
            "mean_score": round(mean_score, 6),
            "unsafe_trust": unsafe_trust,
            "action_counts": dict(sorted(action_counts.items())),
            "violation_counts": dict(sorted(violation_counts.items())),
        }

    fail_closed = policy_reports["fail_closed"]
    prototype_passed = (
        not manifest_issues
        and oracle_pass == len(rows)
        and fail_closed["exact_pass"] == len(rows)
        and fail_closed["unsafe_trust"] == 0
    )
    return {
        "dataset": "c5_antibody_antigen_ood_baseline_eval_v1",
        "evaluation_scope": "synthetic_public_policy_test",
        "manifest": {
            "rows": len(rows),
            "validation_issues": manifest_issues,
            "expected_action_counts": dict(
                sorted(
                    Counter(
                        row["hidden_eval_metadata"]["expected_terminal_action"]
                        for row in rows
                    ).items()
                )
            ),
        },
        "oracle": {
            "exact_pass": oracle_pass,
            "rows": len(rows),
        },
        "policies": policy_reports,
        "decision": {
            "no_api_prototype_passed": prototype_passed,
            "source_backed_c5_panel_required": True,
            "ready_for_c5_model_training": False,
            "ready_for_dpo_rlvr": False,
            "reason": (
                "Synthetic policy tests validate the contract but do not "
                "establish antibody-antigen calibration transfer."
            ),
            "next_ticket": (
                "build_source_backed_c5_pilot_with_frozen_interface_labels"
            ),
        },
        "scientific_boundary": {
            "synthetic_policy_test_only": True,
            "independent_generalization_claimed": False,
            "live_specialist_model_run": False,
            "llm_or_api_used": False,
            "row_level_hidden_labels_published_in_summary": False,
            "fail_closed_expected_actions_defined_by_fixture": True,
        },
    }


def _visible(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("model_visible_task")
    if not isinstance(value, Mapping):
        raise ValueError("C5 row is missing model_visible_task mapping.")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Expected mapping.")
    return value


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
