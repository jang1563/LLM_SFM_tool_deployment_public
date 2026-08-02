"""Calibration-first DockQ reveal gates for the prospective C5 experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .calibration import (
    fixed_threshold_exact_binomial_metrics,
    fixed_threshold_hoeffding_metrics,
    select_exact_binomial_certificate,
    select_hoeffding_certificate,
)
from .manifest import load_c5_manifest
from .prospective_native_lock import validate_native_structure_lock
from .prospective_panel import (
    SHA256_RE,
    canonical_sha256,
    validate_preregistration,
    write_json,
)
from .prospective_predictions import validate_prediction_lock


CALIBRATION_LOCK_SCHEMA = "c5_prospective_calibration_lock_v1"
PUBLIC_CALIBRATION_SCHEMA = "c5_prospective_calibration_freeze_v1"
EVALUATION_LOCK_SCHEMA = "c5_prospective_evaluation_lock_v1"
PUBLIC_EVALUATION_SCHEMA = "c5_prospective_evaluation_result_v1"
CALIBRATION_WORKFLOW_STATE = "calibration_locked_evaluation_pending"
EVALUATION_WORKFLOW_STATE = "complete"
LABEL_KEYS = frozenset(
    {
        "target_id",
        "selected_output_id",
        "selected_model_sha256",
        "evaluator",
        "evaluator_version",
        "evaluator_commit",
        "metric",
        "metric_scope",
        "native_chain_mapping_sha256",
        "native_chain_mapping_matches_committed_roles",
        "native_structure_sha256",
        "dockq",
    }
)


class ProspectiveRevealError(ValueError):
    """Raised when a staged label reveal violates the preregistered gates."""


def build_calibration_lock(
    *,
    preregistration: Mapping[str, Any],
    input_freeze: Mapping[str, Any],
    retained_rows: Sequence[Mapping[str, Any]],
    prediction_lock: Mapping[str, Any],
    native_structure_lock: Mapping[str, Any],
    calibration_labels: Sequence[Mapping[str, Any]],
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reveal only calibration labels and freeze the resulting policy."""

    _validate_shared_inputs(
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_rows=retained_rows,
        prediction_lock=prediction_lock,
        native_structure_lock=native_structure_lock,
    )
    observations = _validated_label_observations(
        labels=calibration_labels,
        expected_role="calibration",
        preregistration=preregistration,
        retained_rows=retained_rows,
        prediction_lock=prediction_lock,
        native_structure_lock=native_structure_lock,
    )
    risk = preregistration["protocol"]["risk_control"]
    alphas = (
        float(risk["primary_alpha"]),
        *(float(value) for value in risk["secondary_alphas"]),
    )
    certificates = {
        f"alpha_{alpha:.2f}": _calibration_certificate(
            [
                (observation["ranking_score"], observation["success"])
                for observation in observations
            ],
            preregistration=preregistration,
            alpha=alpha,
        )
        for alpha in alphas
    }
    primary_key = f"alpha_{float(risk['primary_alpha']):.2f}"
    primary = certificates[primary_key]
    selected_policy = _selected_policy(preregistration, primary)
    label_set_sha256 = canonical_sha256(list(calibration_labels))
    decision_commitment = {
        "prediction_set_sha256": prediction_lock["commitments"][
            "selected_prediction_set_sha256"
        ],
        "calibration_label_set_sha256": label_set_sha256,
        "certificates": certificates,
        "selected_policy": selected_policy,
    }
    timestamp = created_at_utc or _utc_timestamp()
    successes = sum(observation["success"] for observation in observations)
    private_lock = {
        "schema_version": CALIBRATION_LOCK_SCHEMA,
        "created_at_utc": timestamp,
        "workflow_state": CALIBRATION_WORKFLOW_STATE,
        "preregistration_id": preregistration["preregistration_id"],
        "protocol_sha256": preregistration["commitment"]["protocol_sha256"],
        "input_commitments": {
            "retained_manifest_sha256": input_freeze["retention"][
                "manifest_sha256"
            ],
            "selected_prediction_set_sha256": prediction_lock["commitments"][
                "selected_prediction_set_sha256"
            ],
            "prediction_lock_sha256": canonical_sha256(prediction_lock),
            "native_structure_lock_sha256": canonical_sha256(
                native_structure_lock
            ),
            "native_structure_set_sha256": native_structure_lock[
                "native_structure_set_sha256"
            ],
            "calibration_label_set_sha256": label_set_sha256,
        },
        "counts": {
            "calibration_targets": len(observations),
            "calibration_successes": successes,
            "calibration_failures": len(observations) - successes,
            "evaluation_targets_read": 0,
        },
        "certificates": certificates,
        "selected_policy": selected_policy,
        "calibration_decision_sha256": canonical_sha256(
            decision_commitment
        ),
        "observations": observations,
        "evidence_boundary": {
            "calibration_labels_read": True,
            "evaluation_labels_read": False,
            "per_target_labels_emitted_publicly": False,
            "evaluation_threshold_tuning_allowed": False,
        },
        "decision": {
            "ready_for_evaluation_label_reveal": True,
            "external_specialist_trust_enabled": False,
            "ready_for_model_training": False,
            "ready_for_dpo_rlvr": False,
        },
    }
    lock_issues = validate_calibration_lock(
        private_lock,
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_rows=retained_rows,
        prediction_lock=prediction_lock,
        native_structure_lock=native_structure_lock,
    )
    if lock_issues:
        raise ProspectiveRevealError(
            "calibration_lock_invalid:" + ",".join(lock_issues)
        )
    return private_lock, public_calibration_freeze(private_lock)


def build_evaluation_lock(
    *,
    preregistration: Mapping[str, Any],
    input_freeze: Mapping[str, Any],
    retained_rows: Sequence[Mapping[str, Any]],
    prediction_lock: Mapping[str, Any],
    native_structure_lock: Mapping[str, Any],
    calibration_lock: Mapping[str, Any],
    evaluation_labels: Sequence[Mapping[str, Any]],
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reveal evaluation labels once and apply only the frozen policy."""

    _validate_shared_inputs(
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_rows=retained_rows,
        prediction_lock=prediction_lock,
        native_structure_lock=native_structure_lock,
    )
    calibration_issues = validate_calibration_lock(
        calibration_lock,
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_rows=retained_rows,
        prediction_lock=prediction_lock,
        native_structure_lock=native_structure_lock,
    )
    if calibration_issues:
        raise ProspectiveRevealError(
            "calibration_lock_invalid:" + ",".join(calibration_issues)
        )
    observations = _validated_label_observations(
        labels=evaluation_labels,
        expected_role="evaluation",
        preregistration=preregistration,
        retained_rows=retained_rows,
        prediction_lock=prediction_lock,
        native_structure_lock=native_structure_lock,
    )
    selected_policy = calibration_lock["selected_policy"]
    policies, transfer_supported = _evaluation_policies(
        observations=observations,
        selected_policy=selected_policy,
        preregistration=preregistration,
    )
    label_set_sha256 = canonical_sha256(list(evaluation_labels))
    evaluation_commitment = {
        "calibration_decision_sha256": calibration_lock[
            "calibration_decision_sha256"
        ],
        "evaluation_label_set_sha256": label_set_sha256,
        "policies": policies,
        "transfer_supported": transfer_supported,
    }
    successes = sum(observation["success"] for observation in observations)
    private_lock = {
        "schema_version": EVALUATION_LOCK_SCHEMA,
        "created_at_utc": created_at_utc or _utc_timestamp(),
        "workflow_state": EVALUATION_WORKFLOW_STATE,
        "preregistration_id": preregistration["preregistration_id"],
        "protocol_sha256": preregistration["commitment"]["protocol_sha256"],
        "input_commitments": {
            "selected_prediction_set_sha256": prediction_lock["commitments"][
                "selected_prediction_set_sha256"
            ],
            "calibration_decision_sha256": calibration_lock[
                "calibration_decision_sha256"
            ],
            "calibration_lock_sha256": canonical_sha256(calibration_lock),
            "evaluation_label_set_sha256": label_set_sha256,
        },
        "counts": {
            "evaluation_targets": len(observations),
            "evaluation_successes": successes,
            "evaluation_failures": len(observations) - successes,
        },
        "frozen_policy": dict(selected_policy),
        "policies": policies,
        "evaluation_decision_sha256": canonical_sha256(
            evaluation_commitment
        ),
        "observations": observations,
        "evidence_boundary": {
            "calibration_policy_frozen_before_evaluation": True,
            "evaluation_labels_read_once": True,
            "evaluation_threshold_tuned": False,
            "per_target_labels_emitted_publicly": False,
            "llm_or_api_used": False,
        },
        "decision": {
            "regime_specific_transfer_supported": transfer_supported,
            "external_specialist_trust_enabled": transfer_supported,
            "ready_for_model_training": False,
            "ready_for_dpo_rlvr": False,
        },
    }
    lock_issues = validate_evaluation_lock(
        private_lock,
        preregistration=preregistration,
        retained_rows=retained_rows,
        prediction_lock=prediction_lock,
        native_structure_lock=native_structure_lock,
        calibration_lock=calibration_lock,
    )
    if lock_issues:
        raise ProspectiveRevealError(
            "evaluation_lock_invalid:" + ",".join(lock_issues)
        )
    return private_lock, public_evaluation_result(private_lock)


def public_calibration_freeze(
    private_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a calibration lock without per-target scores or labels."""

    return {
        "schema_version": PUBLIC_CALIBRATION_SCHEMA,
        "created_at_utc": private_lock["created_at_utc"],
        "workflow_state": private_lock["workflow_state"],
        "preregistration_id": private_lock["preregistration_id"],
        "protocol_sha256": private_lock["protocol_sha256"],
        "input_commitments": dict(private_lock["input_commitments"]),
        "counts": dict(private_lock["counts"]),
        "certificates": dict(private_lock["certificates"]),
        "selected_policy": dict(private_lock["selected_policy"]),
        "calibration_decision_sha256": private_lock[
            "calibration_decision_sha256"
        ],
        "evidence_boundary": dict(private_lock["evidence_boundary"]),
        "release_boundary": {
            "target_ids_emitted": False,
            "per_target_confidence_emitted": False,
            "per_target_dockq_emitted": False,
            "prediction_paths_emitted": False,
        },
        "decision": dict(private_lock["decision"]),
    }


def public_evaluation_result(
    private_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the one-time evaluation into an aggregate public result."""

    return {
        "schema_version": PUBLIC_EVALUATION_SCHEMA,
        "created_at_utc": private_lock["created_at_utc"],
        "workflow_state": private_lock["workflow_state"],
        "preregistration_id": private_lock["preregistration_id"],
        "protocol_sha256": private_lock["protocol_sha256"],
        "input_commitments": dict(private_lock["input_commitments"]),
        "counts": dict(private_lock["counts"]),
        "frozen_policy": dict(private_lock["frozen_policy"]),
        "policies": dict(private_lock["policies"]),
        "evaluation_decision_sha256": private_lock[
            "evaluation_decision_sha256"
        ],
        "evidence_boundary": dict(private_lock["evidence_boundary"]),
        "release_boundary": {
            "target_ids_emitted": False,
            "per_target_confidence_emitted": False,
            "per_target_dockq_emitted": False,
            "prediction_paths_emitted": False,
        },
        "decision": dict(private_lock["decision"]),
    }


def validate_calibration_lock(
    lock: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    input_freeze: Mapping[str, Any],
    retained_rows: Sequence[Mapping[str, Any]],
    prediction_lock: Mapping[str, Any],
    native_structure_lock: Mapping[str, Any],
) -> list[str]:
    """Recompute the calibration policy and verify evaluation remains sealed."""

    issues: list[str] = []
    if set(lock) != {
        "schema_version",
        "created_at_utc",
        "workflow_state",
        "preregistration_id",
        "protocol_sha256",
        "input_commitments",
        "counts",
        "certificates",
        "selected_policy",
        "calibration_decision_sha256",
        "observations",
        "evidence_boundary",
        "decision",
    }:
        issues.append("calibration_lock_schema_invalid")
    if lock.get("schema_version") != CALIBRATION_LOCK_SCHEMA:
        issues.append("schema_version_invalid")
    if lock.get("workflow_state") != CALIBRATION_WORKFLOW_STATE:
        issues.append("workflow_state_invalid")
    if lock.get("preregistration_id") != preregistration.get(
        "preregistration_id"
    ):
        issues.append("preregistration_id_mismatch")
    if lock.get("protocol_sha256") != preregistration.get(
        "commitment", {}
    ).get("protocol_sha256"):
        issues.append("protocol_sha256_mismatch")
    if lock.get("input_commitments", {}).get(
        "retained_manifest_sha256"
    ) != input_freeze.get("retention", {}).get("manifest_sha256"):
        issues.append("retained_manifest_sha256_mismatch")
    if lock.get("input_commitments", {}).get(
        "selected_prediction_set_sha256"
    ) != prediction_lock.get("commitments", {}).get(
        "selected_prediction_set_sha256"
    ):
        issues.append("selected_prediction_set_sha256_mismatch")
    if lock.get("input_commitments", {}).get(
        "native_structure_set_sha256"
    ) != native_structure_lock.get("native_structure_set_sha256"):
        issues.append("native_structure_set_sha256_mismatch")
    if not SHA256_RE.fullmatch(
        str(
            lock.get("input_commitments", {}).get(
                "calibration_label_set_sha256", ""
            )
        )
    ):
        issues.append("calibration_label_set_sha256_invalid")

    observations = lock.get("observations")
    if not isinstance(observations, list):
        return [*issues, "observations_invalid"]
    expected_ids = {
        str(row["target_id"])
        for row in retained_rows
        if row["panel_role"] == "calibration"
    }
    actual_ids = {observation.get("target_id") for observation in observations}
    if actual_ids != expected_ids or len(observations) != len(expected_ids):
        issues.append("calibration_observation_set_mismatch")
    if any(
        observation.get("panel_role") != "calibration"
        for observation in observations
    ):
        issues.append("non_calibration_observation")
    issues.extend(
        _observation_binding_issues(
            observations=observations,
            expected_role="calibration",
            preregistration=preregistration,
            prediction_lock=prediction_lock,
            native_structure_lock=native_structure_lock,
        )
    )
    try:
        pairs = [
            (
                float(observation["ranking_score"]),
                observation["success"],
            )
            for observation in observations
        ]
        risk = preregistration["protocol"]["risk_control"]
        alphas = (
            float(risk["primary_alpha"]),
            *(float(value) for value in risk["secondary_alphas"]),
        )
        expected_certificates = {
            f"alpha_{alpha:.2f}": _calibration_certificate(
                pairs,
                preregistration=preregistration,
                alpha=alpha,
            )
            for alpha in alphas
        }
    except (KeyError, TypeError, ValueError):
        issues.append("calibration_observations_invalid")
    else:
        if lock.get("certificates") != expected_certificates:
            issues.append("certificate_recomputation_mismatch")
        primary = expected_certificates[
            f"alpha_{float(risk['primary_alpha']):.2f}"
        ]
        expected_policy = _selected_policy(preregistration, primary)
        if lock.get("selected_policy") != expected_policy:
            issues.append("selected_policy_mismatch")
        expected_decision = canonical_sha256(
            {
                "prediction_set_sha256": prediction_lock["commitments"][
                    "selected_prediction_set_sha256"
                ],
                "calibration_label_set_sha256": lock[
                    "input_commitments"
                ]["calibration_label_set_sha256"],
                "certificates": expected_certificates,
                "selected_policy": expected_policy,
            }
        )
        if lock.get("calibration_decision_sha256") != expected_decision:
            issues.append("calibration_decision_sha256_mismatch")
    boundary = lock.get("evidence_boundary", {})
    if boundary.get("evaluation_labels_read") is not False:
        issues.append("evaluation_label_boundary_invalid")
    if boundary.get("evaluation_threshold_tuning_allowed") is not False:
        issues.append("evaluation_tuning_boundary_invalid")
    if lock.get("counts", {}).get("evaluation_targets_read") != 0:
        issues.append("evaluation_target_count_not_zero")
    if lock.get("decision", {}).get(
        "ready_for_evaluation_label_reveal"
    ) is not True:
        issues.append("evaluation_reveal_gate_invalid")
    return sorted(set(issues))


def validate_evaluation_lock(
    lock: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    retained_rows: Sequence[Mapping[str, Any]],
    prediction_lock: Mapping[str, Any],
    native_structure_lock: Mapping[str, Any],
    calibration_lock: Mapping[str, Any],
) -> list[str]:
    """Recompute the one-time evaluation result from frozen observations."""

    issues: list[str] = []
    if set(lock) != {
        "schema_version",
        "created_at_utc",
        "workflow_state",
        "preregistration_id",
        "protocol_sha256",
        "input_commitments",
        "counts",
        "frozen_policy",
        "policies",
        "evaluation_decision_sha256",
        "observations",
        "evidence_boundary",
        "decision",
    }:
        issues.append("evaluation_lock_schema_invalid")
    if lock.get("schema_version") != EVALUATION_LOCK_SCHEMA:
        issues.append("schema_version_invalid")
    if lock.get("workflow_state") != EVALUATION_WORKFLOW_STATE:
        issues.append("workflow_state_invalid")
    if lock.get("preregistration_id") != preregistration.get(
        "preregistration_id"
    ):
        issues.append("preregistration_id_mismatch")
    if lock.get("protocol_sha256") != preregistration.get(
        "commitment", {}
    ).get("protocol_sha256"):
        issues.append("protocol_sha256_mismatch")
    if lock.get("input_commitments", {}).get(
        "selected_prediction_set_sha256"
    ) != prediction_lock.get("commitments", {}).get(
        "selected_prediction_set_sha256"
    ):
        issues.append("selected_prediction_set_sha256_mismatch")
    if lock.get("input_commitments", {}).get(
        "calibration_decision_sha256"
    ) != calibration_lock.get("calibration_decision_sha256"):
        issues.append("calibration_decision_sha256_mismatch")
    if not SHA256_RE.fullmatch(
        str(
            lock.get("input_commitments", {}).get(
                "evaluation_label_set_sha256", ""
            )
        )
    ):
        issues.append("evaluation_label_set_sha256_invalid")
    if lock.get("frozen_policy") != calibration_lock.get("selected_policy"):
        issues.append("frozen_policy_mismatch")

    observations = lock.get("observations")
    if not isinstance(observations, list):
        return [*issues, "observations_invalid"]
    expected_ids = {
        str(row["target_id"])
        for row in retained_rows
        if row["panel_role"] == "evaluation"
    }
    actual_ids = {observation.get("target_id") for observation in observations}
    if actual_ids != expected_ids or len(observations) != len(expected_ids):
        issues.append("evaluation_observation_set_mismatch")
    if any(
        observation.get("panel_role") != "evaluation"
        for observation in observations
    ):
        issues.append("non_evaluation_observation")
    issues.extend(
        _observation_binding_issues(
            observations=observations,
            expected_role="evaluation",
            preregistration=preregistration,
            prediction_lock=prediction_lock,
            native_structure_lock=native_structure_lock,
        )
    )
    try:
        policies, transfer_supported = _evaluation_policies(
            observations=observations,
            selected_policy=calibration_lock["selected_policy"],
            preregistration=preregistration,
        )
    except (KeyError, TypeError, ValueError):
        issues.append("evaluation_observations_invalid")
    else:
        if lock.get("policies") != policies:
            issues.append("evaluation_policy_recomputation_mismatch")
        expected_decision = canonical_sha256(
            {
                "calibration_decision_sha256": calibration_lock[
                    "calibration_decision_sha256"
                ],
                "evaluation_label_set_sha256": lock[
                    "input_commitments"
                ]["evaluation_label_set_sha256"],
                "policies": policies,
                "transfer_supported": transfer_supported,
            }
        )
        if lock.get("evaluation_decision_sha256") != expected_decision:
            issues.append("evaluation_decision_sha256_mismatch")
        if lock.get("decision", {}).get(
            "regime_specific_transfer_supported"
        ) is not transfer_supported:
            issues.append("transfer_decision_mismatch")
        if lock.get("decision", {}).get(
            "external_specialist_trust_enabled"
        ) is not transfer_supported:
            issues.append("external_trust_decision_mismatch")
    boundary = lock.get("evidence_boundary", {})
    if boundary.get("calibration_policy_frozen_before_evaluation") is not True:
        issues.append("calibration_freeze_boundary_invalid")
    if boundary.get("evaluation_threshold_tuned") is not False:
        issues.append("evaluation_tuning_boundary_invalid")
    return sorted(set(issues))


def _validate_shared_inputs(
    *,
    preregistration: Mapping[str, Any],
    input_freeze: Mapping[str, Any],
    retained_rows: Sequence[Mapping[str, Any]],
    prediction_lock: Mapping[str, Any],
    native_structure_lock: Mapping[str, Any],
) -> None:
    preregistration_issues = validate_preregistration(preregistration)
    if preregistration_issues:
        raise ProspectiveRevealError(
            "preregistration_invalid:" + ",".join(preregistration_issues)
        )
    prediction_issues = validate_prediction_lock(
        prediction_lock,
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_rows=retained_rows,
    )
    if prediction_issues:
        raise ProspectiveRevealError(
            "prediction_lock_invalid:" + ",".join(prediction_issues)
        )
    if prediction_lock.get("workflow_state") != (
        "predictions_locked_label_reveal_pending"
    ):
        raise ProspectiveRevealError("prediction_workflow_state_invalid")
    if prediction_lock.get("evidence_boundary", {}).get(
        "evaluation_labels_read"
    ) is not False:
        raise ProspectiveRevealError("prediction_evaluation_boundary_invalid")
    native_issues = validate_native_structure_lock(
        native_structure_lock,
        retained_rows=retained_rows,
        input_freeze=input_freeze,
    )
    if native_issues:
        raise ProspectiveRevealError(
            "native_structure_lock_invalid:" + ",".join(native_issues)
        )


def _validated_label_observations(
    *,
    labels: Sequence[Mapping[str, Any]],
    expected_role: str,
    preregistration: Mapping[str, Any],
    retained_rows: Sequence[Mapping[str, Any]],
    prediction_lock: Mapping[str, Any],
    native_structure_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows_by_id = {
        str(row["target_id"]): row
        for row in retained_rows
        if row["panel_role"] == expected_role
    }
    targets_by_id = {
        str(target["target_id"]): target
        for target in prediction_lock["targets"]
    }
    native_by_id = {
        str(target["target_id"]): target
        for target in native_structure_lock["targets"]
    }
    label_ids = [str(label.get("target_id")) for label in labels]
    if len(label_ids) != len(set(label_ids)):
        raise ProspectiveRevealError("label_target_duplicate")
    if set(label_ids) != set(rows_by_id):
        issue = (
            "calibration_label_target_set_mismatch"
            if expected_role == "calibration"
            else "evaluation_label_target_set_mismatch"
        )
        raise ProspectiveRevealError(issue)
    expected_count = int(
        preregistration["protocol"]["stopping_rule"][
            f"required_{expected_role}_targets"
        ]
    )
    if len(labels) != expected_count:
        raise ProspectiveRevealError(f"{expected_role}_label_count_mismatch")

    label_protocol = preregistration["protocol"]["label"]
    observations: list[dict[str, Any]] = []
    for label in labels:
        if set(label) != LABEL_KEYS:
            raise ProspectiveRevealError("label_schema_invalid")
        target_id = str(label["target_id"])
        target = targets_by_id[target_id]
        row = rows_by_id[target_id]
        selected = _selected_sample(target)
        expected_values = {
            "selected_output_id": target["selected_output_id"],
            "selected_model_sha256": selected["artifacts"]["model_cif"][
                "sha256"
            ],
            "evaluator": label_protocol["evaluator"],
            "evaluator_version": label_protocol["version"],
            "evaluator_commit": label_protocol["code_commit"],
            "metric": label_protocol["metric"],
            "metric_scope": label_protocol["metric_scope"],
            "native_chain_mapping_sha256": canonical_sha256(
                row["chain_role_mapping"]
            ),
            "native_chain_mapping_matches_committed_roles": True,
            "native_structure_sha256": native_by_id[target_id][
                "native_structure_sha256"
            ],
        }
        for key, expected in expected_values.items():
            if label.get(key) != expected:
                raise ProspectiveRevealError(f"label_{key}_mismatch")
        dockq = label["dockq"]
        if (
            isinstance(dockq, bool)
            or not isinstance(dockq, (int, float))
            or not math.isfinite(dockq)
            or not 0.0 <= float(dockq) <= 1.0
        ):
            raise ProspectiveRevealError("label_dockq_invalid")
        observations.append(
            {
                "target_id": target_id,
                "panel_role": expected_role,
                "selected_output_id": target["selected_output_id"],
                "selected_model_sha256": selected["artifacts"]["model_cif"][
                    "sha256"
                ],
                "native_structure_sha256": native_by_id[target_id][
                    "native_structure_sha256"
                ],
                "ranking_score": float(selected["ranking_score"]),
                "dockq": float(dockq),
                "success": (
                    float(dockq)
                    >= float(label_protocol["interface_success_threshold"])
                ),
            }
        )
    return sorted(observations, key=lambda observation: observation["target_id"])


def _selected_sample(target: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = [
        sample
        for sample in target["samples"]
        if sample["output_id"] == target["selected_output_id"]
    ]
    if len(selected) != 1:
        raise ProspectiveRevealError("selected_prediction_invalid")
    return selected[0]


def _observation_binding_issues(
    *,
    observations: Sequence[Mapping[str, Any]],
    expected_role: str,
    preregistration: Mapping[str, Any],
    prediction_lock: Mapping[str, Any],
    native_structure_lock: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    targets_by_id = {
        str(target["target_id"]): target
        for target in prediction_lock["targets"]
    }
    native_by_id = {
        str(target["target_id"]): target
        for target in native_structure_lock["targets"]
    }
    success_threshold = float(
        preregistration["protocol"]["label"][
            "interface_success_threshold"
        ]
    )
    expected_keys = {
        "target_id",
        "panel_role",
        "selected_output_id",
        "selected_model_sha256",
        "native_structure_sha256",
        "ranking_score",
        "dockq",
        "success",
    }
    for observation in observations:
        if set(observation) != expected_keys:
            issues.append("observation_schema_invalid")
            continue
        target_id = str(observation["target_id"])
        target = targets_by_id.get(target_id)
        native = native_by_id.get(target_id)
        if target is None or native is None:
            issues.append("observation_target_missing")
            continue
        try:
            selected = _selected_sample(target)
        except (KeyError, ProspectiveRevealError):
            issues.append("observation_selected_prediction_invalid")
            continue
        expected_values = {
            "panel_role": expected_role,
            "selected_output_id": target["selected_output_id"],
            "selected_model_sha256": selected["artifacts"]["model_cif"][
                "sha256"
            ],
            "native_structure_sha256": native[
                "native_structure_sha256"
            ],
            "ranking_score": float(selected["ranking_score"]),
        }
        for key, expected in expected_values.items():
            if observation.get(key) != expected:
                issues.append(f"observation_{key}_mismatch")
        dockq = observation.get("dockq")
        if (
            isinstance(dockq, bool)
            or not isinstance(dockq, (int, float))
            or not math.isfinite(dockq)
            or not 0.0 <= float(dockq) <= 1.0
        ):
            issues.append("observation_dockq_invalid")
            continue
        success = observation.get("success")
        if not isinstance(success, bool):
            issues.append("observation_success_invalid")
        elif success is not (float(dockq) >= success_threshold):
            issues.append("observation_success_mismatch")
    return issues


def _selected_policy(
    preregistration: Mapping[str, Any],
    primary_certificate: Mapping[str, Any],
) -> dict[str, Any]:
    risk = preregistration["protocol"]["risk_control"]
    certified = primary_certificate["certified"] is True
    exact_binomial = risk.get("certificate_method") == (
        "exact_binomial_bonferroni"
    )
    policy = {
        "calibration_dataset_id": (
            f"{preregistration['preregistration_id']}::calibration"
        ),
        "calibration_regime": (
            "sabdab2_antibody_antigen_train_paired_chain_"
            "template_free_af3_v3.0.3"
        ),
        "regime_match": True,
        "confidence_metric": risk["confidence_metric"],
        "confidence_metric_scope": "whole_complex_ranking",
        "certificate": risk["certificate"],
        "alpha": float(risk["primary_alpha"]),
        "delta": float(risk["delta"]),
        "certified": certified,
        "threshold_id": (
            (
                "c5_sabdab2_ranking_score_exact_binomial_alpha_0_30_v2"
                if exact_binomial
                else "c5_sabdab2_ranking_score_hoeffding_alpha_0_30_v1"
            )
            if certified
            else None
        ),
        "threshold": primary_certificate["threshold"],
        "action_when_uncertified": risk["no_certificate_action"],
    }
    if exact_binomial:
        policy["certificate_method"] = risk["certificate_method"]
        policy["sampling_unit"] = preregistration["protocol"][
            "target_selection"
        ]["sampling_unit"]
    return policy


def _evaluation_policies(
    *,
    observations: Sequence[Mapping[str, Any]],
    selected_policy: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    pairs = [
        (float(observation["ranking_score"]), observation["success"])
        for observation in observations
    ]
    risk = preregistration["protocol"]["risk_control"]
    certified = selected_policy["certified"] is True
    frozen_threshold = (
        float(selected_policy["threshold"]) if certified else None
    )
    applied_threshold = frozen_threshold if certified else math.inf
    policies = {
        "trust_all": _fixed_threshold_metrics(
            pairs,
            threshold=None,
            preregistration=preregistration,
        ),
        "fixed_ranking_score_0_80": _fixed_threshold_metrics(
            pairs,
            threshold=0.80,
            preregistration=preregistration,
        ),
        "regime_specific_calibrated_gate": {
            **_fixed_threshold_metrics(
                pairs,
                threshold=applied_threshold,
                preregistration=preregistration,
            ),
            "calibration_certified": certified,
            "threshold": frozen_threshold,
            "uncertified_action": (
                None
                if certified
                else selected_policy["action_when_uncertified"]
            ),
        },
        "fail_closed": _fixed_threshold_metrics(
            pairs,
            threshold=math.inf,
            preregistration=preregistration,
        ),
    }
    primary = policies["regime_specific_calibrated_gate"]
    exact_risk_test = primary.get("risk_test_passed")
    risk_supported = (
        exact_risk_test is True
        if isinstance(exact_risk_test, bool)
        else (
            primary["risk_upper_bound"] is not None
            and primary["risk_upper_bound"]
            <= float(risk["primary_alpha"])
        )
    )
    transfer_supported = (
        certified
        and selected_policy.get("regime_match") is True
        and primary["trusted"]
        >= int(risk["minimum_evaluation_trusted_for_transfer_claim"])
        and risk_supported
    )
    return policies, transfer_supported


def _calibration_certificate(
    observations: Sequence[tuple[float, bool]],
    *,
    preregistration: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    risk = preregistration["protocol"]["risk_control"]
    thresholds = tuple(float(value) for value in risk["candidate_thresholds"])
    delta = float(risk["delta"])
    if risk.get("certificate_method") != "exact_binomial_bonferroni":
        return select_hoeffding_certificate(
            observations,
            alpha=alpha,
            delta=delta,
            thresholds=thresholds,
        )
    certificate = select_exact_binomial_certificate(
        observations,
        alpha=alpha,
        delta=delta,
        thresholds=thresholds,
        multiplicity=int(risk["candidate_count"]),
    )
    certificate["hoeffding_sensitivity"] = select_hoeffding_certificate(
        observations,
        alpha=alpha,
        delta=delta,
        thresholds=thresholds,
    )
    return certificate


def _fixed_threshold_metrics(
    observations: Sequence[tuple[float, bool]],
    *,
    threshold: float | None,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    risk = preregistration["protocol"]["risk_control"]
    delta = float(risk["delta"])
    if risk.get("certificate_method") != "exact_binomial_bonferroni":
        return fixed_threshold_hoeffding_metrics(
            observations,
            threshold=threshold,
            delta=delta,
        )
    metrics = fixed_threshold_exact_binomial_metrics(
        observations,
        threshold=threshold,
        alpha=float(risk["primary_alpha"]),
        delta=delta,
    )
    metrics["hoeffding_sensitivity"] = fixed_threshold_hoeffding_metrics(
        observations,
        threshold=threshold,
        delta=delta,
    )
    return metrics


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ProspectiveRevealError("expected_json_object")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProspectiveRevealError(
                    f"expected_json_object:{line_no}"
                )
            rows.append(value)
    return rows


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--preregistration", type=Path, required=True)
    common.add_argument("--input-freeze", type=Path, required=True)
    common.add_argument("--retained-manifest", type=Path, required=True)
    common.add_argument("--prediction-lock", type=Path, required=True)
    common.add_argument("--native-structure-lock", type=Path, required=True)
    common.add_argument("--labels", type=Path, required=True)
    common.add_argument("--private-out", type=Path, required=True)
    common.add_argument("--public-out", type=Path, required=True)

    subparsers.add_parser("calibrate", parents=[common])
    evaluate = subparsers.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--calibration-lock", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    preregistration = _load_json(args.preregistration)
    input_freeze = _load_json(args.input_freeze)
    retained = load_c5_manifest(args.retained_manifest)
    prediction_lock = _load_json(args.prediction_lock)
    native_structure_lock = _load_json(args.native_structure_lock)
    labels = _load_jsonl(args.labels)
    if args.command == "calibrate":
        private, public = build_calibration_lock(
            preregistration=preregistration,
            input_freeze=input_freeze,
            retained_rows=retained,
            prediction_lock=prediction_lock,
            native_structure_lock=native_structure_lock,
            calibration_labels=labels,
        )
    else:
        private, public = build_evaluation_lock(
            preregistration=preregistration,
            input_freeze=input_freeze,
            retained_rows=retained,
            prediction_lock=prediction_lock,
            native_structure_lock=native_structure_lock,
            calibration_lock=_load_json(args.calibration_lock),
            evaluation_labels=labels,
        )
    write_json(args.private_out, private)
    write_json(args.public_out, public)
    print(json.dumps(public["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
