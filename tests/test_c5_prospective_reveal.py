import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from c5_antibody_ood.calibration import (
    fixed_threshold_hoeffding_metrics,
    select_hoeffding_certificate,
)
from c5_antibody_ood.manifest import load_c5_manifest
from c5_antibody_ood.prospective_native_lock import NATIVE_LOCK_SCHEMA
from c5_antibody_ood.prospective_panel import canonical_sha256
from c5_antibody_ood.prospective_predictions import PRIVATE_LOCK_SCHEMA
from c5_antibody_ood.prospective_reveal import (
    ProspectiveRevealError,
    build_calibration_lock,
    build_evaluation_lock,
    validate_calibration_lock,
    validate_evaluation_lock,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "c5_antibody_ood/c5_prospective_panel_preregistration_v1.json"
)
INPUT_FREEZE = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json"
)
RETAINED_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl"
)
CANDIDATE_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_panel_manifest_v1.jsonl"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _prediction_lock(
    rows: list[dict],
    preregistration: dict,
    input_freeze: dict,
    *,
    ranking_score: float = 0.90,
) -> dict:
    targets = []
    for row in sorted(rows, key=lambda value: value["target_id"]):
        samples = []
        for sample_index in range(5):
            output_id = f"seed-20260725_sample-{sample_index}"
            artifacts = {
                "model_cif": {
                    "relative_path": (
                        f"{row['instance_id'].lower()}/{output_id}/model.cif"
                    ),
                    "bytes": 10,
                    "sha256": _sha(
                        f"{row['target_id']}|{output_id}|model"
                    ),
                },
                "confidences_json": {
                    "relative_path": (
                        f"{row['instance_id'].lower()}/{output_id}/full.json"
                    ),
                    "bytes": 10,
                    "sha256": _sha(
                        f"{row['target_id']}|{output_id}|full"
                    ),
                },
                "summary_confidences_json": {
                    "relative_path": (
                        f"{row['instance_id'].lower()}/{output_id}/summary.json"
                    ),
                    "bytes": 10,
                    "sha256": _sha(
                        f"{row['target_id']}|{output_id}|summary"
                    ),
                },
            }
            samples.append(
                {
                    "output_id": output_id,
                    "seed": 20260725,
                    "sample_index": sample_index,
                    "ranking_score": ranking_score,
                    "iptm": 0.90,
                    "ptm": 0.90,
                    "fraction_disordered": 0.0,
                    "has_clash": False,
                    "artifacts": artifacts,
                }
            )
        targets.append(
            {
                "target_id": row["target_id"],
                "instance_id": row["instance_id"],
                "panel_role": row["panel_role"],
                "job_name": row["instance_id"].lower(),
                "samples": samples,
                "selected_output_id": samples[0]["output_id"],
                "selected_model_relative_path": samples[0]["artifacts"][
                    "model_cif"
                ]["relative_path"],
                "job_artifact_set_sha256": _sha(
                    f"{row['target_id']}|job"
                ),
            }
        )
    prediction_artifact_set_sha256 = canonical_sha256(
        [
            {
                "target_id": target["target_id"],
                "job_artifact_set_sha256": target[
                    "job_artifact_set_sha256"
                ],
            }
            for target in targets
        ]
    )
    selected_prediction_set_sha256 = canonical_sha256(
        [
            {
                "target_id": target["target_id"],
                "panel_role": target["panel_role"],
                "output_id": target["selected_output_id"],
                "ranking_score": target["samples"][0]["ranking_score"],
                "iptm": target["samples"][0]["iptm"],
                "model_cif_sha256": target["samples"][0]["artifacts"][
                    "model_cif"
                ]["sha256"],
                "summary_confidences_sha256": target["samples"][0][
                    "artifacts"
                ]["summary_confidences_json"]["sha256"],
            }
            for target in targets
        ]
    )
    return {
        "schema_version": PRIVATE_LOCK_SCHEMA,
        "created_at_utc": "2026-07-25T12:00:00+00:00",
        "workflow_state": "predictions_locked_label_reveal_pending",
        "preregistration_id": preregistration["preregistration_id"],
        "protocol_sha256": preregistration["commitment"]["protocol_sha256"],
        "input_commitments": {
            "retained_manifest_sha256": input_freeze["retention"][
                "manifest_sha256"
            ],
            "af3_input_set_sha256": input_freeze["af3_inputs"][
                "af3_input_set_sha256"
            ],
            "run_attestation_sha256": "a" * 64,
        },
        "selection": {},
        "counts": {
            "targets": 120,
            "targets_by_role": {
                "calibration": 80,
                "evaluation": 40,
            },
            "samples_per_target": 5,
            "samples": 600,
        },
        "commitments": {
            "prediction_artifact_set_sha256": (
                prediction_artifact_set_sha256
            ),
            "selected_prediction_set_sha256": (
                selected_prediction_set_sha256
            ),
        },
        "targets": targets,
        "evidence_boundary": {
            "dockq_or_native_interface_labels_read": False,
            "evaluation_labels_read": False,
            "label_input_accepted_by_this_command": False,
        },
        "decision": {
            "ready_for_calibration_label_reveal": True,
            "ready_for_evaluation_label_reveal": False,
        },
    }


def _labels(
    rows: list[dict],
    prediction_lock: dict,
    native_structure_lock: dict,
    preregistration: dict,
    *,
    role: str,
    dockq: float,
) -> list[dict]:
    targets = {
        target["target_id"]: target
        for target in prediction_lock["targets"]
    }
    native_targets = {
        target["target_id"]: target
        for target in native_structure_lock["targets"]
    }
    label_protocol = preregistration["protocol"]["label"]
    labels = []
    for row in rows:
        if row["panel_role"] != role:
            continue
        target = targets[row["target_id"]]
        selected = target["samples"][0]
        labels.append(
            {
                "target_id": row["target_id"],
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
                "native_structure_sha256": native_targets[row["target_id"]][
                    "native_structure_sha256"
                ],
                "dockq": dockq,
            }
        )
    return labels


def _native_structure_lock(
    candidate_rows: list[dict],
    retained_rows: list[dict],
    input_freeze: dict,
) -> dict:
    targets = [
        {
            "target_id": row["target_id"],
            "panel_role": row["panel_role"],
            "native_structure_sha256": _sha(
                f"{row['target_id']}|native"
            ),
            "chain_role_mapping_sha256": canonical_sha256(
                row["chain_role_mapping"]
            ),
            "qc_passed": True,
            "qc_issues": [],
        }
        for row in sorted(candidate_rows, key=lambda value: value["target_id"])
    ]
    set_sha256 = canonical_sha256(
        {
            target["target_id"]: target["native_structure_sha256"]
            for target in targets
        }
    )
    input_freeze["structure_qc"]["native_structure_set_sha256"] = set_sha256
    return {
        "schema_version": NATIVE_LOCK_SCHEMA,
        "preregistration_id": input_freeze["preregistration_id"],
        "protocol_sha256": input_freeze["protocol_sha256"],
        "native_structure_set_sha256": set_sha256,
        "counts": {
            "candidate_targets": len(targets),
            "retained_targets": len(retained_rows),
            "qc_passed": len(targets),
        },
        "targets": targets,
        "evidence_boundary": {
            "dockq_or_interface_labels_read": False,
            "raw_structures_or_sequences_emitted": False,
            "local_paths_emitted": False,
        },
        "decision": {
            "ready_for_staged_label_reveal": True,
            "external_specialist_trust_enabled": False,
        },
    }


@pytest.fixture
def reveal_fixture() -> dict:
    preregistration = json.loads(PREREGISTRATION.read_text())
    input_freeze = deepcopy(json.loads(INPUT_FREEZE.read_text()))
    rows = load_c5_manifest(RETAINED_MANIFEST)
    candidates = load_c5_manifest(CANDIDATE_MANIFEST)
    native_structure_lock = _native_structure_lock(
        candidates,
        rows,
        input_freeze,
    )
    return {
        "preregistration": preregistration,
        "input_freeze": input_freeze,
        "rows": rows,
        "native_structure_lock": native_structure_lock,
        "prediction_lock": _prediction_lock(
            rows,
            preregistration,
            input_freeze,
        ),
    }


def test_calibration_reveal_certifies_and_keeps_evaluation_sealed(
    reveal_fixture,
):
    fixture = reveal_fixture
    labels = _labels(
        fixture["rows"],
        fixture["prediction_lock"],
        fixture["native_structure_lock"],
        fixture["preregistration"],
        role="calibration",
        dockq=0.90,
    )

    private, public = build_calibration_lock(
        preregistration=fixture["preregistration"],
        input_freeze=fixture["input_freeze"],
        retained_rows=fixture["rows"],
        prediction_lock=fixture["prediction_lock"],
        native_structure_lock=fixture["native_structure_lock"],
        calibration_labels=labels,
        created_at_utc="2026-07-25T13:00:00+00:00",
    )

    assert private["counts"] == {
        "calibration_targets": 80,
        "calibration_successes": 80,
        "calibration_failures": 0,
        "evaluation_targets_read": 0,
    }
    assert private["selected_policy"]["certified"] is True
    assert private["selected_policy"]["threshold"] == 0.50
    assert private["selected_policy"]["calibration_dataset_id"].endswith(
        "::calibration"
    )
    assert private["selected_policy"]["confidence_metric_scope"] == (
        "whole_complex_ranking"
    )
    assert private["evidence_boundary"]["evaluation_labels_read"] is False
    assert private["decision"]["external_specialist_trust_enabled"] is False
    rendered = json.dumps(public, sort_keys=True)
    assert fixture["rows"][0]["target_id"] not in rendered
    assert '"observations"' not in rendered
    assert '"dockq"' not in rendered.lower()


def test_calibration_reveal_rejects_evaluation_rows_and_model_drift(
    reveal_fixture,
):
    fixture = reveal_fixture
    evaluation_labels = _labels(
        fixture["rows"],
        fixture["prediction_lock"],
        fixture["native_structure_lock"],
        fixture["preregistration"],
        role="evaluation",
        dockq=0.90,
    )
    with pytest.raises(
        ProspectiveRevealError,
        match="calibration_label_target_set_mismatch",
    ):
        build_calibration_lock(
            preregistration=fixture["preregistration"],
            input_freeze=fixture["input_freeze"],
            retained_rows=fixture["rows"],
            prediction_lock=fixture["prediction_lock"],
            native_structure_lock=fixture["native_structure_lock"],
            calibration_labels=evaluation_labels,
        )

    calibration_labels = _labels(
        fixture["rows"],
        fixture["prediction_lock"],
        fixture["native_structure_lock"],
        fixture["preregistration"],
        role="calibration",
        dockq=0.90,
    )
    calibration_labels[0]["selected_model_sha256"] = "0" * 64
    with pytest.raises(
        ProspectiveRevealError,
        match="label_selected_model_sha256_mismatch",
    ):
        build_calibration_lock(
            preregistration=fixture["preregistration"],
            input_freeze=fixture["input_freeze"],
            retained_rows=fixture["rows"],
            prediction_lock=fixture["prediction_lock"],
            native_structure_lock=fixture["native_structure_lock"],
            calibration_labels=calibration_labels,
        )
    calibration_labels = _labels(
        fixture["rows"],
        fixture["prediction_lock"],
        fixture["native_structure_lock"],
        fixture["preregistration"],
        role="calibration",
        dockq=0.90,
    )
    calibration_labels[0]["native_structure_sha256"] = "0" * 64
    with pytest.raises(
        ProspectiveRevealError,
        match="label_native_structure_sha256_mismatch",
    ):
        build_calibration_lock(
            preregistration=fixture["preregistration"],
            input_freeze=fixture["input_freeze"],
            retained_rows=fixture["rows"],
            prediction_lock=fixture["prediction_lock"],
            native_structure_lock=fixture["native_structure_lock"],
            calibration_labels=calibration_labels,
        )


def test_uncertified_calibration_forces_verify_all_on_evaluation(
    reveal_fixture,
):
    fixture = reveal_fixture
    calibration_labels = _labels(
        fixture["rows"],
        fixture["prediction_lock"],
        fixture["native_structure_lock"],
        fixture["preregistration"],
        role="calibration",
        dockq=0.10,
    )
    calibration_lock, _ = build_calibration_lock(
        preregistration=fixture["preregistration"],
        input_freeze=fixture["input_freeze"],
        retained_rows=fixture["rows"],
        prediction_lock=fixture["prediction_lock"],
        native_structure_lock=fixture["native_structure_lock"],
        calibration_labels=calibration_labels,
    )
    evaluation_labels = _labels(
        fixture["rows"],
        fixture["prediction_lock"],
        fixture["native_structure_lock"],
        fixture["preregistration"],
        role="evaluation",
        dockq=0.90,
    )

    private, public = build_evaluation_lock(
        preregistration=fixture["preregistration"],
        input_freeze=fixture["input_freeze"],
        retained_rows=fixture["rows"],
        prediction_lock=fixture["prediction_lock"],
        native_structure_lock=fixture["native_structure_lock"],
        calibration_lock=calibration_lock,
        evaluation_labels=evaluation_labels,
    )

    assert private["frozen_policy"]["certified"] is False
    assert private["policies"]["regime_specific_calibrated_gate"][
        "trusted"
    ] == 0
    assert private["policies"]["regime_specific_calibrated_gate"][
        "uncertified_action"
    ] == "verify_all"
    assert private["decision"]["regime_specific_transfer_supported"] is False
    assert public["decision"]["external_specialist_trust_enabled"] is False


def test_certified_policy_is_applied_once_and_can_support_transfer(
    reveal_fixture,
):
    fixture = reveal_fixture
    calibration_labels = _labels(
        fixture["rows"],
        fixture["prediction_lock"],
        fixture["native_structure_lock"],
        fixture["preregistration"],
        role="calibration",
        dockq=0.90,
    )
    calibration_lock, _ = build_calibration_lock(
        preregistration=fixture["preregistration"],
        input_freeze=fixture["input_freeze"],
        retained_rows=fixture["rows"],
        prediction_lock=fixture["prediction_lock"],
        native_structure_lock=fixture["native_structure_lock"],
        calibration_labels=calibration_labels,
    )
    evaluation_labels = _labels(
        fixture["rows"],
        fixture["prediction_lock"],
        fixture["native_structure_lock"],
        fixture["preregistration"],
        role="evaluation",
        dockq=0.90,
    )

    private, public = build_evaluation_lock(
        preregistration=fixture["preregistration"],
        input_freeze=fixture["input_freeze"],
        retained_rows=fixture["rows"],
        prediction_lock=fixture["prediction_lock"],
        native_structure_lock=fixture["native_structure_lock"],
        calibration_lock=calibration_lock,
        evaluation_labels=evaluation_labels,
    )

    gate = private["policies"]["regime_specific_calibrated_gate"]
    assert gate["threshold"] == 0.50
    assert gate["trusted"] == 40
    assert gate["failures_among_trusted"] == 0
    assert gate["risk_upper_bound"] <= 0.30
    assert private["decision"]["regime_specific_transfer_supported"] is True
    assert public["decision"]["external_specialist_trust_enabled"] is True
    assert private["evidence_boundary"]["evaluation_threshold_tuned"] is False
    private["policies"]["trust_all"]["trusted"] = 0
    assert "evaluation_policy_recomputation_mismatch" in (
        validate_evaluation_lock(
            private,
            preregistration=fixture["preregistration"],
            retained_rows=fixture["rows"],
            prediction_lock=fixture["prediction_lock"],
            native_structure_lock=fixture["native_structure_lock"],
            calibration_lock=calibration_lock,
        )
    )


def test_mutated_calibration_policy_blocks_evaluation(reveal_fixture):
    fixture = reveal_fixture
    calibration_labels = _labels(
        fixture["rows"],
        fixture["prediction_lock"],
        fixture["native_structure_lock"],
        fixture["preregistration"],
        role="calibration",
        dockq=0.90,
    )
    calibration_lock, _ = build_calibration_lock(
        preregistration=fixture["preregistration"],
        input_freeze=fixture["input_freeze"],
        retained_rows=fixture["rows"],
        prediction_lock=fixture["prediction_lock"],
        native_structure_lock=fixture["native_structure_lock"],
        calibration_labels=calibration_labels,
    )
    calibration_lock["selected_policy"]["threshold"] = 0.99

    assert "selected_policy_mismatch" in validate_calibration_lock(
        calibration_lock,
        preregistration=fixture["preregistration"],
        input_freeze=fixture["input_freeze"],
        retained_rows=fixture["rows"],
        prediction_lock=fixture["prediction_lock"],
        native_structure_lock=fixture["native_structure_lock"],
    )
    with pytest.raises(
        ProspectiveRevealError,
        match="calibration_lock_invalid",
    ):
        build_evaluation_lock(
            preregistration=fixture["preregistration"],
            input_freeze=fixture["input_freeze"],
            retained_rows=fixture["rows"],
            prediction_lock=fixture["prediction_lock"],
            native_structure_lock=fixture["native_structure_lock"],
            calibration_lock=calibration_lock,
            evaluation_labels=_labels(
                fixture["rows"],
                fixture["prediction_lock"],
                fixture["native_structure_lock"],
                fixture["preregistration"],
                role="evaluation",
                dockq=0.90,
            ),
        )


def test_af3_ranking_score_range_and_fixed_policy_bound():
    certificate = select_hoeffding_certificate(
        [(1.20, True)] * 80 + [(-99.0, False)],
        alpha=0.30,
        delta=0.10,
        thresholds=(0.50, 0.90),
    )
    metrics = fixed_threshold_hoeffding_metrics(
        [(0.90, True)] * 40,
        threshold=0.50,
        delta=0.10,
    )

    assert certificate["certified"] is True
    assert certificate["calibration_trusted"] == 80
    assert metrics["trusted"] == 40
    assert metrics["risk_upper_bound"] <= 0.30
