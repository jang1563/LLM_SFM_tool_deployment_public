"""Public-safe C5 manifest and canonical trajectory adapter.

The tracked rows are synthetic policy tests. They validate schema separation
and fail-closed routing before any source-backed antibody-antigen panel or
model-heavy experiment is allowed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_sfm_tool_deployment import (
    Action,
    CalibrationStatus,
    EvaluationResult,
    EvidencePacket,
    EvidenceStatus,
    TaskSpec,
    ToolStep,
    Trajectory,
    TrajectoryEvaluator,
)


C5_DATASET = "c5_antibody_antigen_ood_policy_test_v1"
C5_BASE_TOOLS = (
    "inspect_specialist_record",
    "lookup_calibration_card",
)
C5_ACTION_TOOL = {
    Action.TRUST_SPECIALIST_OUTPUT: (),
    Action.VERIFY_WITH_ASSAY_OR_DATABASE: ("request_structure_verification",),
    Action.USE_CHEAP_BASELINE: ("run_cheap_interface_baseline",),
    Action.DEFER_OR_REQUEST_MORE_EVIDENCE: (),
}
C5_ALLOWED_ACTIONS = tuple(action.value for action in C5_ACTION_TOOL)
C5_ALLOWED_TOOLS = (
    *C5_BASE_TOOLS,
    "request_structure_verification",
    "run_cheap_interface_baseline",
)
C5_EXPECTED_ACTIONS = tuple(C5_ALLOWED_ACTIONS)
C5_INTERFACE_LABELS = (
    "interface_success",
    "interface_failure",
    "interface_unresolved",
)
C5_CALIBRATION_KEYS = (
    "status",
    "dataset_id",
    "regime",
    "regime_match",
    "threshold_id",
    "threshold_value",
    "certified",
)
C5_SPECIALIST_KEYS = (
    "specialist_name",
    "metric_type",
    "metric_scope",
    "metric_value",
)
C5_BASELINE_KEYS = (
    "available",
    "dominates_specialist",
    "metric_type",
    "metric_value",
)
C5_GENERAL_CALIBRATION_KEYS = (
    "dataset_id",
    "regime",
    "threshold_id",
    "threshold_value",
)


def load_c5_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a C5 JSONL manifest."""

    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            rows.append(row)
    return rows


def write_c5_manifest(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic sorted-key C5 JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def build_c5_policy_test_rows() -> list[dict[str, Any]]:
    """Build the balanced synthetic policy-test manifest."""

    cases = (
        _case("trust_complete_iptm", Action.TRUST_SPECIALIST_OUTPUT, 0.94),
        _case(
            "trust_complete_pdockq",
            Action.TRUST_SPECIALIST_OUTPUT,
            0.89,
            metric_type="predicted_dock_quality",
            threshold_value=0.82,
        ),
        _case(
            "trust_complete_interface_score",
            Action.TRUST_SPECIALIST_OUTPUT,
            0.88,
            metric_type="interface_confidence",
            threshold_value=0.86,
            baseline_available=False,
        ),
        _case(
            "baseline_dominates_high_score",
            Action.USE_CHEAP_BASELINE,
            0.93,
            baseline_available=True,
            baseline_dominates=True,
            interface_label_status="interface_failure",
        ),
        _case(
            "baseline_on_regime_mismatch",
            Action.USE_CHEAP_BASELINE,
            0.91,
            calibration_dataset_id="generic_ppi_calibration_public_v1",
            calibration_regime="generic_protein_interface",
            calibration_regime_match=False,
            baseline_available=True,
            baseline_dominates=True,
            interface_label_status="interface_failure",
        ),
        _case(
            "baseline_below_threshold",
            Action.USE_CHEAP_BASELINE,
            0.79,
            baseline_available=True,
            baseline_dominates=True,
            interface_label_status="interface_failure",
        ),
        _case(
            "verify_missing_calibration_dataset",
            Action.VERIFY_WITH_ASSAY_OR_DATABASE,
            0.92,
            calibration_dataset_id=None,
            interface_label_status="interface_unresolved",
        ),
        _case(
            "verify_missing_metric_scope",
            Action.VERIFY_WITH_ASSAY_OR_DATABASE,
            0.90,
            metric_scope=None,
            interface_label_status="interface_failure",
        ),
        _case(
            "verify_missing_rcps_threshold",
            Action.VERIFY_WITH_ASSAY_OR_DATABASE,
            0.91,
            threshold_id=None,
            threshold_value=None,
            certified=False,
            interface_label_status="interface_unresolved",
        ),
        _case(
            "defer_uncalibrated_no_fallback",
            Action.DEFER_OR_REQUEST_MORE_EVIDENCE,
            0.95,
            calibration_status=CalibrationStatus.UNCALIBRATED.value,
            calibration_dataset_id=None,
            calibration_regime_match=False,
            threshold_id=None,
            threshold_value=None,
            certified=False,
            baseline_available=False,
            verification_available=False,
            interface_label_status="interface_unresolved",
        ),
        _case(
            "defer_regime_mismatch_no_fallback",
            Action.DEFER_OR_REQUEST_MORE_EVIDENCE,
            0.90,
            calibration_dataset_id="generic_ppi_calibration_public_v1",
            calibration_regime="generic_protein_interface",
            calibration_regime_match=False,
            baseline_available=False,
            verification_available=False,
            interface_label_status="interface_unresolved",
        ),
        _case(
            "defer_uncertified_threshold_no_fallback",
            Action.DEFER_OR_REQUEST_MORE_EVIDENCE,
            0.92,
            threshold_id=None,
            threshold_value=None,
            certified=False,
            baseline_available=False,
            verification_available=False,
            interface_label_status="interface_unresolved",
        ),
    )
    return [
        _row_from_case(case, case_index=index)
        for index, case in enumerate(cases, start=1)
    ]


def _case(
    family: str,
    expected_action: Action,
    metric_value: float,
    *,
    metric_type: str | None = "predicted_interface_tm_score",
    metric_scope: str | None = "antibody_antigen_interface",
    calibration_status: str = CalibrationStatus.CALIBRATED.value,
    calibration_dataset_id: str | None = "ab_ag_calibration_public_v1",
    calibration_regime: str = "antibody_antigen",
    calibration_regime_match: bool = True,
    threshold_id: str | None = "rcps_ab_ag_alpha_0_10_v1",
    threshold_value: float | None = 0.84,
    certified: bool = True,
    baseline_available: bool = False,
    baseline_dominates: bool = False,
    verification_available: bool = True,
    interface_label_status: str = "interface_success",
) -> dict[str, Any]:
    return {
        "family": family,
        "expected_action": expected_action.value,
        "metric_type": metric_type,
        "metric_scope": metric_scope,
        "metric_value": metric_value,
        "calibration_status": calibration_status,
        "calibration_dataset_id": calibration_dataset_id,
        "calibration_regime": calibration_regime,
        "calibration_regime_match": calibration_regime_match,
        "threshold_id": threshold_id,
        "threshold_value": threshold_value,
        "certified": certified,
        "baseline_available": baseline_available,
        "baseline_dominates": baseline_dominates,
        "verification_available": verification_available,
        "interface_label_status": interface_label_status,
    }


def _row_from_case(case: Mapping[str, Any], *, case_index: int) -> dict[str, Any]:
    input_id = f"c5::{case_index:06d}"
    expected_action = Action(str(case["expected_action"]))
    expected_status = (
        EvidenceStatus.SUPPORTED
        if expected_action
        in {Action.TRUST_SPECIALIST_OUTPUT, Action.USE_CHEAP_BASELINE}
        else EvidenceStatus.INSUFFICIENT
    )
    required_tools = (
        *C5_BASE_TOOLS,
        *C5_ACTION_TOOL[expected_action],
    )
    baseline_value = 0.76 if case["baseline_available"] else None
    return {
        "case_id": input_id,
        "dataset": C5_DATASET,
        "model_visible_task": {
            "input_id": input_id,
            "claim": (
                "Select a deployment action for this specialist structure "
                "record after using the available metadata tools."
            ),
            "complex_id": f"c5_public_synthetic_complex_{case_index:03d}",
            "chain_role_mapping": [
                {"chain_id": "H", "role": "antibody_heavy"},
                {"chain_id": "L", "role": "antibody_light"},
                {"chain_id": "A", "role": "antigen"},
            ],
            "interaction_regime": "antibody_antigen",
            "allowed_tools": list(C5_ALLOWED_TOOLS),
            "allowed_terminal_actions": list(C5_ALLOWED_ACTIONS),
            "specialist_result": {
                "specialist_name": "synthetic_structure_specialist",
                "metric_type": case["metric_type"],
                "metric_scope": case["metric_scope"],
                "metric_value": case["metric_value"],
            },
            "general_calibration": {
                "dataset_id": "generic_ppi_calibration_public_v1",
                "regime": "generic_protein_interface",
                "threshold_id": "rcps_generic_ppi_alpha_0_10_v1",
                "threshold_value": 0.80,
            },
            "calibration": {
                "status": case["calibration_status"],
                "dataset_id": case["calibration_dataset_id"],
                "regime": case["calibration_regime"],
                "regime_match": case["calibration_regime_match"],
                "threshold_id": case["threshold_id"],
                "threshold_value": case["threshold_value"],
                "certified": case["certified"],
            },
            "baseline_result": {
                "available": case["baseline_available"],
                "dominates_specialist": case["baseline_dominates"],
                "metric_type": (
                    "synthetic_interface_recovery_score"
                    if case["baseline_available"]
                    else None
                ),
                "metric_value": baseline_value,
            },
            "verification_available": case["verification_available"],
        },
        "hidden_eval_metadata": {
            "required_tools": list(required_tools),
            "gold_evidence_status": expected_status.value,
            "expected_terminal_action": expected_action.value,
            "gold_source_ids": [],
            "requires_attribution": False,
            "requires_external_tool": True,
            "web_zero": False,
            "case_family": case["family"],
            "split_group": f"synthetic_complex::{case_index:06d}",
            "interface_label_status": case["interface_label_status"],
            "interface_label_source": "synthetic_policy_test_fixture_v1",
        },
        "cost_profile": {
            "inspect_specialist_record": 1.0,
            "lookup_calibration_card": 1.0,
            "request_structure_verification": 5.0,
            "run_cheap_interface_baseline": 2.0,
            "defer_or_request_more_evidence": 0.5,
        },
    }


def task_spec_from_c5_row(row: Mapping[str, Any]) -> TaskSpec:
    """Project a C5 row into the canonical hidden evaluator TaskSpec."""

    visible = _model_visible(row)
    hidden = _hidden(row)
    return TaskSpec(
        input_id=str(visible["input_id"]),
        claim=str(visible["claim"]),
        required_tools=tuple(str(tool) for tool in hidden["required_tools"]),
        gold_evidence_status=str(hidden["gold_evidence_status"]),
        expected_terminal_action=str(hidden["expected_terminal_action"]),
        gold_source_ids=tuple(str(value) for value in hidden["gold_source_ids"]),
        requires_attribution=bool(hidden["requires_attribution"]),
        requires_external_tool=bool(hidden["requires_external_tool"]),
        web_zero=bool(hidden["web_zero"]),
    )


def trajectory_from_c5_row(
    row: Mapping[str, Any],
    action: Action | str,
    *,
    fail_closed_reason: str | None = None,
) -> Trajectory:
    """Build a trajectory from visible C5 fields and a selected action."""

    visible = _model_visible(row)
    terminal = action if isinstance(action, Action) else Action(action)
    specialist = _mapping(visible["specialist_result"], "specialist_result")
    calibration = _mapping(visible["calibration"], "calibration")
    baseline = _mapping(visible["baseline_result"], "baseline_result")
    metric_value = specialist.get("metric_value")
    confidence = (
        float(metric_value)
        if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool)
        else None
    )
    selected_tools = (*C5_BASE_TOOLS, *C5_ACTION_TOOL[terminal])
    observations = {
        "inspect_specialist_record": dict(specialist),
        "lookup_calibration_card": dict(calibration),
        "request_structure_verification": {
            "request_status": "queued",
            "complex_id": visible["complex_id"],
        },
        "run_cheap_interface_baseline": dict(baseline),
    }
    predicted_status = (
        EvidenceStatus.SUPPORTED
        if terminal in {Action.TRUST_SPECIALIST_OUTPUT, Action.USE_CHEAP_BASELINE}
        else EvidenceStatus.INSUFFICIENT
    )
    return Trajectory(
        input_id=str(visible["input_id"]),
        steps=tuple(
            ToolStep(
                name=tool,
                arguments={"complex_id": visible["complex_id"]},
                observation=observations[tool],
            )
            for tool in selected_tools
        ),
        evidence_packet=EvidencePacket(
            input_id=str(visible["input_id"]),
            representation_type="antibody_antigen_complex",
            specialist_name=_optional_text(specialist.get("specialist_name")),
            specialist_output=dict(specialist),
            specialist_confidence=confidence,
            calibration_status=str(calibration.get("status", CalibrationStatus.UNKNOWN.value)),
            cheap_baseline_output=dict(baseline),
            baseline_dominance_flag=bool(baseline.get("dominates_specialist", False)),
            negative_evidence_status=predicted_status,
            claim_guard_status="checked",
            allowed_actions=tuple(str(value) for value in visible["allowed_terminal_actions"]),
            specialist_metric_type=_optional_text(specialist.get("metric_type")),
            confidence_metric_scope=_optional_text(specialist.get("metric_scope")),
            interaction_regime=_optional_text(visible.get("interaction_regime")),
            calibration_dataset_id=_optional_text(calibration.get("dataset_id")),
            calibration_regime_match=calibration.get("regime_match"),
            rcps_threshold_id=_optional_text(calibration.get("threshold_id")),
            fail_closed_reason=fail_closed_reason,
        ),
        terminal_action=terminal,
        predicted_evidence_status=predicted_status,
    )


def ideal_trajectory_from_c5_row(row: Mapping[str, Any]) -> Trajectory:
    """Build the positive-control trajectory without exposing hidden labels."""

    expected = Action(str(_hidden(row)["expected_terminal_action"]))
    reason = None if expected == Action.TRUST_SPECIALIST_OUTPUT else "expected_runtime_fallback"
    return trajectory_from_c5_row(row, expected, fail_closed_reason=reason)


def score_c5_trajectory(
    row: Mapping[str, Any],
    trajectory: Trajectory,
) -> EvaluationResult:
    """Score canonical trajectory gates plus C5 extraction and safety gates."""

    base = TrajectoryEvaluator().evaluate(task_spec_from_c5_row(row), trajectory)
    visible = _model_visible(row)
    hidden = _hidden(row)
    reward = dict(base.reward_breakdown)
    violations = list(base.violations)

    checks = {
        "c5_metric_extraction": _metric_extraction_ok(visible, trajectory),
        "c5_calibration_extraction": _calibration_extraction_ok(visible, trajectory),
        "c5_hidden_label_isolation": _hidden_label_isolated(hidden, trajectory),
        "c5_interface_safety": _interface_safe(hidden, trajectory),
        "c5_trust_gate_completeness": _trust_gate_complete(visible, trajectory),
        "c5_fail_closed_reason": _fail_closed_reason_present(trajectory),
    }
    for name, passed in checks.items():
        reward[name] = float(passed)
        if not passed:
            violations.append(name)

    return EvaluationResult(
        input_id=base.input_id,
        earned=float(sum(reward.values())),
        possible=float(len(reward)),
        reward_breakdown=reward,
        violations=tuple(violations),
    )


def c5_trust_schema_issues(visible: Mapping[str, Any]) -> tuple[str, ...]:
    """Return visible reasons a C5 specialist output cannot be trusted."""

    issues: list[str] = []
    specialist = _mapping(visible.get("specialist_result"), "specialist_result")
    calibration = _mapping(visible.get("calibration"), "calibration")
    baseline = _mapping(visible.get("baseline_result"), "baseline_result")
    if visible.get("interaction_regime") != "antibody_antigen":
        issues.append("interaction_regime_not_antibody_antigen")
    for key in ("metric_type", "metric_scope", "metric_value"):
        if specialist.get(key) is None or specialist.get(key) == "":
            issues.append(f"missing_{key}")
    if calibration.get("status") != CalibrationStatus.CALIBRATED.value:
        issues.append("calibration_status_not_calibrated")
    required_calibration_fields = {
        "dataset_id": "missing_calibration_dataset_id",
        "regime": "missing_calibration_regime",
        "threshold_id": "missing_rcps_threshold_id",
        "threshold_value": "missing_rcps_threshold_value",
    }
    for key, issue in required_calibration_fields.items():
        if calibration.get(key) is None or calibration.get(key) == "":
            issues.append(issue)
    if calibration.get("regime") != "antibody_antigen":
        issues.append("calibration_regime_not_antibody_antigen")
    if calibration.get("regime_match") is not True:
        issues.append("calibration_regime_mismatch")
    if calibration.get("certified") is not True:
        issues.append("rcps_threshold_not_certified")

    metric_value = specialist.get("metric_value")
    threshold_value = calibration.get("threshold_value")
    if (
        isinstance(metric_value, (int, float))
        and not isinstance(metric_value, bool)
        and isinstance(threshold_value, (int, float))
        and not isinstance(threshold_value, bool)
        and metric_value < threshold_value
    ):
        issues.append("specialist_metric_below_threshold")
    if baseline.get("dominates_specialist") is True:
        issues.append("cheap_baseline_dominates")
    return tuple(issues)


def validate_c5_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_rows: int = 12,
    min_action_count: int = 3,
) -> list[str]:
    """Return fail-closed manifest validation issues."""

    issues: list[str] = []
    if len(rows) < min_rows:
        issues.append(f"manifest_too_small:{len(rows)}<{min_rows}")
    case_ids: list[str] = []
    split_groups: list[str] = []
    action_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        issues.extend(_validate_c5_row(row, prefix=prefix))
        case_ids.append(str(row.get("case_id", "")))
        hidden = row.get("hidden_eval_metadata")
        if isinstance(hidden, Mapping):
            split_groups.append(str(hidden.get("split_group", "")))
            action_counts[str(hidden.get("expected_terminal_action", ""))] += 1

    for duplicate in _duplicates(case_ids):
        issues.append(f"duplicate_case_id:{duplicate}")
    for duplicate in _duplicates(split_groups):
        issues.append(f"split_group_overlap:{duplicate}")
    for action in C5_EXPECTED_ACTIONS:
        if action_counts[action] < min_action_count:
            issues.append(
                f"action_underrepresented:{action}:"
                f"{action_counts[action]}<{min_action_count}"
            )
    return issues


def validate_c5_split_overlap(
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Check source/split identities across future C5 train and evaluation rows."""

    train_groups = {
        str(_hidden(row).get("split_group", ""))
        for row in train_rows
    }
    eval_groups = {
        str(_hidden(row).get("split_group", ""))
        for row in eval_rows
    }
    overlap = sorted(value for value in train_groups & eval_groups if value)
    return [f"train_eval_split_group_overlap:{value}" for value in overlap]


def _validate_c5_row(row: Mapping[str, Any], *, prefix: str) -> list[str]:
    issues: list[str] = []
    for key in (
        "case_id",
        "dataset",
        "model_visible_task",
        "hidden_eval_metadata",
        "cost_profile",
    ):
        if key not in row:
            issues.append(f"{prefix}:missing_key:{key}")
    visible = row.get("model_visible_task")
    hidden = row.get("hidden_eval_metadata")
    if not isinstance(visible, Mapping):
        return [*issues, f"{prefix}:model_visible_task_not_mapping"]
    if not isinstance(hidden, Mapping):
        return [*issues, f"{prefix}:hidden_eval_metadata_not_mapping"]

    for key in (
        "input_id",
        "claim",
        "complex_id",
        "chain_role_mapping",
        "interaction_regime",
        "allowed_tools",
        "allowed_terminal_actions",
        "specialist_result",
        "general_calibration",
        "calibration",
        "baseline_result",
        "verification_available",
    ):
        if key not in visible:
            issues.append(f"{prefix}:model_visible_missing:{key}")
    for key in (
        "required_tools",
        "gold_evidence_status",
        "expected_terminal_action",
        "gold_source_ids",
        "requires_attribution",
        "requires_external_tool",
        "web_zero",
        "case_family",
        "split_group",
        "interface_label_status",
        "interface_label_source",
    ):
        if key not in hidden:
            issues.append(f"{prefix}:hidden_missing:{key}")

    visible_keys = _recursive_keys(visible)
    for hidden_key in (
        "gold_evidence_status",
        "expected_terminal_action",
        "interface_label_status",
        "interface_label_source",
        "case_family",
        "split_group",
    ):
        if hidden_key in visible_keys:
            issues.append(f"{prefix}:hidden_key_leak:{hidden_key}")
    visible_text = json.dumps(visible, sort_keys=True).lower()
    for hidden_key in ("interface_label_status", "interface_label_source", "case_family"):
        hidden_value = hidden.get(hidden_key)
        if hidden_value and str(hidden_value).lower() in visible_text:
            issues.append(f"{prefix}:hidden_value_leak:{hidden_key}")

    if row.get("case_id") != visible.get("input_id"):
        issues.append(f"{prefix}:case_input_id_mismatch")
    if row.get("dataset") != C5_DATASET:
        issues.append(f"{prefix}:dataset_mismatch")
    if not isinstance(visible.get("complex_id"), str) or not visible.get("complex_id"):
        issues.append(f"{prefix}:complex_id_invalid")
    if visible.get("interaction_regime") != "antibody_antigen":
        issues.append(f"{prefix}:visible_regime_mismatch")
    if tuple(visible.get("allowed_tools", ())) != C5_ALLOWED_TOOLS:
        issues.append(f"{prefix}:allowed_tools_contract_mismatch")
    if tuple(visible.get("allowed_terminal_actions", ())) != C5_ALLOWED_ACTIONS:
        issues.append(f"{prefix}:allowed_actions_contract_mismatch")

    roles = visible.get("chain_role_mapping")
    if not isinstance(roles, list):
        issues.append(f"{prefix}:chain_role_mapping_not_list")
    else:
        chain_ids = [item.get("chain_id") for item in roles if isinstance(item, Mapping)]
        role_names = [item.get("role") for item in roles if isinstance(item, Mapping)]
        if len(chain_ids) != len(roles) or len(set(chain_ids)) != len(chain_ids):
            issues.append(f"{prefix}:chain_role_mapping_invalid")
        required_roles = {"antibody_heavy", "antibody_light", "antigen"}
        if set(role_names) != required_roles:
            issues.append(f"{prefix}:chain_roles_incomplete")

    for field, required_keys in (
        ("specialist_result", C5_SPECIALIST_KEYS),
        ("general_calibration", C5_GENERAL_CALIBRATION_KEYS),
        ("calibration", C5_CALIBRATION_KEYS),
        ("baseline_result", C5_BASELINE_KEYS),
    ):
        value = visible.get(field)
        if not isinstance(value, Mapping):
            issues.append(f"{prefix}:{field}_not_mapping")
        else:
            for key in required_keys:
                if key not in value:
                    issues.append(f"{prefix}:{field}_missing_key:{key}")

    specialist = visible.get("specialist_result")
    if isinstance(specialist, Mapping):
        metric_value = specialist.get("metric_value")
        if not _unit_interval(metric_value):
            issues.append(f"{prefix}:specialist_metric_value_invalid")

    general_calibration = visible.get("general_calibration")
    if isinstance(general_calibration, Mapping):
        if not _unit_interval(general_calibration.get("threshold_value")):
            issues.append(f"{prefix}:general_threshold_value_invalid")

    calibration = visible.get("calibration")
    if isinstance(calibration, Mapping):
        valid_statuses = {status.value for status in CalibrationStatus}
        if calibration.get("status") not in valid_statuses:
            issues.append(f"{prefix}:calibration_status_invalid")
        if not isinstance(calibration.get("regime_match"), bool):
            issues.append(f"{prefix}:calibration_regime_match_not_bool")
        if not isinstance(calibration.get("certified"), bool):
            issues.append(f"{prefix}:calibration_certified_not_bool")
        threshold = calibration.get("threshold_value")
        if threshold is not None and not _unit_interval(threshold):
            issues.append(f"{prefix}:calibration_threshold_value_invalid")

    baseline = visible.get("baseline_result")
    if isinstance(baseline, Mapping):
        available = baseline.get("available")
        dominates = baseline.get("dominates_specialist")
        if not isinstance(available, bool):
            issues.append(f"{prefix}:baseline_available_not_bool")
        if not isinstance(dominates, bool):
            issues.append(f"{prefix}:baseline_dominance_not_bool")
        if dominates is True and available is not True:
            issues.append(f"{prefix}:baseline_dominates_but_unavailable")
        baseline_value = baseline.get("metric_value")
        if baseline_value is not None and not _unit_interval(baseline_value):
            issues.append(f"{prefix}:baseline_metric_value_invalid")
    if not isinstance(visible.get("verification_available"), bool):
        issues.append(f"{prefix}:verification_available_not_bool")

    expected_text = hidden.get("expected_terminal_action")
    try:
        expected = Action(str(expected_text))
    except ValueError:
        issues.append(f"{prefix}:invalid_expected_action:{expected_text}")
    else:
        baseline = visible.get("baseline_result")
        baseline = baseline if isinstance(baseline, Mapping) else {}
        if expected not in C5_ACTION_TOOL:
            issues.append(f"{prefix}:expected_action_outside_c5_contract")
        required_tools = (
            *C5_BASE_TOOLS,
            *C5_ACTION_TOOL.get(expected, ()),
        )
        if tuple(hidden.get("required_tools", ())) != required_tools:
            issues.append(f"{prefix}:required_tools_action_contract_mismatch")
        expected_status = (
            EvidenceStatus.SUPPORTED.value
            if expected
            in {Action.TRUST_SPECIALIST_OUTPUT, Action.USE_CHEAP_BASELINE}
            else EvidenceStatus.INSUFFICIENT.value
        )
        if hidden.get("gold_evidence_status") != expected_status:
            issues.append(f"{prefix}:gold_status_action_contract_mismatch")
        if expected == Action.TRUST_SPECIALIST_OUTPUT:
            try:
                trust_issues = c5_trust_schema_issues(visible)
            except ValueError as exc:
                issues.append(f"{prefix}:expected_trust_projection_failed:{exc}")
            else:
                if trust_issues:
                    issues.append(f"{prefix}:expected_trust_has_gate_issues")
            if hidden.get("interface_label_status") != "interface_success":
                issues.append(f"{prefix}:expected_trust_without_interface_success")
        elif expected == Action.USE_CHEAP_BASELINE:
            if not baseline.get("available") or not baseline.get("dominates_specialist"):
                issues.append(f"{prefix}:expected_baseline_without_dominance")
        elif expected == Action.VERIFY_WITH_ASSAY_OR_DATABASE:
            if visible.get("verification_available") is not True:
                issues.append(f"{prefix}:expected_verify_without_verifier")
        elif expected == Action.DEFER_OR_REQUEST_MORE_EVIDENCE:
            if visible.get("verification_available") is True or baseline.get("available"):
                issues.append(f"{prefix}:expected_defer_with_available_fallback")

    if hidden.get("interface_label_status") not in C5_INTERFACE_LABELS:
        issues.append(f"{prefix}:invalid_interface_label_status")
    try:
        task_spec_from_c5_row(row)
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(f"{prefix}:task_spec_projection_failed:{exc}")
    return issues


def _metric_extraction_ok(visible: Mapping[str, Any], trajectory: Trajectory) -> bool:
    specialist = _mapping(visible["specialist_result"], "specialist_result")
    packet = trajectory.evidence_packet
    return (
        packet.specialist_name == specialist.get("specialist_name")
        and packet.specialist_metric_type == specialist.get("metric_type")
        and packet.confidence_metric_scope == specialist.get("metric_scope")
        and packet.specialist_confidence == specialist.get("metric_value")
        and dict(packet.specialist_output) == dict(specialist)
    )


def _calibration_extraction_ok(
    visible: Mapping[str, Any],
    trajectory: Trajectory,
) -> bool:
    calibration = _mapping(visible["calibration"], "calibration")
    packet = trajectory.evidence_packet
    status = (
        packet.calibration_status.value
        if isinstance(packet.calibration_status, CalibrationStatus)
        else packet.calibration_status
    )
    return (
        status == calibration.get("status")
        and packet.interaction_regime == visible.get("interaction_regime")
        and packet.calibration_dataset_id == calibration.get("dataset_id")
        and packet.calibration_regime_match == calibration.get("regime_match")
        and packet.rcps_threshold_id == calibration.get("threshold_id")
    )


def _hidden_label_isolated(
    hidden: Mapping[str, Any],
    trajectory: Trajectory,
) -> bool:
    packet = trajectory.evidence_packet
    rationale = (trajectory.rationale or "").lower()
    hidden_values = (
        hidden.get("interface_label_status"),
        hidden.get("interface_label_source"),
    )
    return (
        packet.hidden_truth_pointer is None
        and packet.interface_label_source is None
        and packet.assay_or_structure_truth_status is None
        and all(
            not value or str(value).lower() not in rationale
            for value in hidden_values
        )
    )


def _interface_safe(hidden: Mapping[str, Any], trajectory: Trajectory) -> bool:
    terminal = Action(trajectory.terminal_action)
    return (
        terminal != Action.TRUST_SPECIALIST_OUTPUT
        or hidden.get("interface_label_status") == "interface_success"
    )


def _trust_gate_complete(
    visible: Mapping[str, Any],
    trajectory: Trajectory,
) -> bool:
    terminal = Action(trajectory.terminal_action)
    return terminal != Action.TRUST_SPECIALIST_OUTPUT or not c5_trust_schema_issues(visible)


def _fail_closed_reason_present(trajectory: Trajectory) -> bool:
    terminal = Action(trajectory.terminal_action)
    return (
        terminal == Action.TRUST_SPECIALIST_OUTPUT
        or bool(trajectory.evidence_packet.fail_closed_reason)
    )


def _model_visible(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row.get("model_visible_task"), "model_visible_task")


def _hidden(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row.get("hidden_eval_metadata"), "hidden_eval_metadata")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"C5 row is missing {field} mapping.")
    return value


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _unit_interval(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= value <= 1.0
    )


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        keys = {str(key) for key in value}
        for child in value.values():
            keys.update(_recursive_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_recursive_keys(child))
        return keys
    return set()


def _duplicates(values: Sequence[str]) -> list[str]:
    return sorted(
        value
        for value, count in Counter(values).items()
        if value and count > 1
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="c5_antibody_ood/c5_policy_test_manifest_v1.jsonl",
    )
    args = parser.parse_args()
    rows = build_c5_policy_test_rows()
    issues = validate_c5_manifest(rows)
    if issues:
        raise SystemExit("C5 manifest validation failed:\n- " + "\n- ".join(issues))
    write_c5_manifest(args.out, rows)
    summary = {
        "dataset": C5_DATASET,
        "out": args.out,
        "rows": len(rows),
        "expected_action_counts": dict(
            sorted(
                Counter(
                    row["hidden_eval_metadata"]["expected_terminal_action"]
                    for row in rows
                ).items()
            )
        ),
        "synthetic_policy_test_only": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
