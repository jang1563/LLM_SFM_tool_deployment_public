from copy import deepcopy
import json
from pathlib import Path

import pytest

from c5_antibody_ood.calibration import (
    fixed_threshold_exact_binomial_metrics,
    select_exact_binomial_certificate,
)
from c5_antibody_ood.prospective_panel import (
    build_preregistration_v2,
    canonical_sha256,
    minimum_zero_failure_trusted_exact,
    validate_preregistration,
    validate_public_panel,
)
from c5_antibody_ood.prospective_source import (
    EligibleTarget,
    ProspectiveSourceError,
    blocked_pdb_ids_from_manifests,
    select_public_panel,
)
from c5_antibody_ood.manifest import load_c5_manifest


ROOT = Path(__file__).resolve().parents[1]
TRACKED_PREREGISTRATION = (
    ROOT / "c5_antibody_ood/c5_prospective_panel_preregistration_v2.json"
)
TRACKED_PANEL = (
    ROOT / "c5_antibody_ood/c5_sabdab2_prospective_panel_manifest_v2.jsonl"
)
TRACKED_AUDIT = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_source_audit_2026-08-02_v2.json"
)
TRACKED_COMMITMENT = (
    ROOT / "c5_antibody_ood/c5_sabdab2_prospective_panel_commitment_v2.json"
)
TRACKED_RETAINED_PANEL = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v2.jsonl"
)
TRACKED_INPUT_FREEZE = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-08-02_v2.json"
)
TRACKED_INPUT_REFREEZE = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_af3_input_refreeze_2026-08-03_v3.json"
)
FROMM_MANIFEST = ROOT / "c5_antibody_ood/c5_source_backed_manifest_v1.jsonl"
GRAY_MANIFEST = (
    ROOT / "c5_antibody_ood/c5_gray_independent_calibration_manifest_v1.jsonl"
)


def _target(
    *,
    split: str,
    cluster: int,
    member: int = 0,
) -> EligibleTarget:
    offset = 0 if split == "train" else 100_000
    value = offset + cluster * 100 + member
    pdb_id = f"{value:08x}"[-4:]
    return EligibleTarget(
        instance_id=f"pdb_0000{pdb_id}_H_L",
        pdb_id=pdb_id,
        sabdab_id=f"H{value:08d}L{value:08d}",
        source_split=split,
        release_date="2025-01-01",
        experimental_method="EM" if cluster % 2 == 0 else "XRAY",
        resolution_angstrom=2.5,
        heavy_chain="H",
        light_chain="L",
        antigen_chains=("A",),
        source_row_sha256=canonical_sha256(
            {"split": split, "cluster": cluster, "member": member}
        ),
        source_cluster_sha256=canonical_sha256(
            {"split": split, "cluster": cluster}
        ),
    )


def _eligible_v2() -> list[EligibleTarget]:
    rows = [
        _target(split="train", cluster=index)
        for index in range(105)
    ]
    rows.extend(
        _target(split="test", cluster=index)
        for index in range(44)
    )
    # A large cluster must still contribute at most one target and must not
    # gain selection priority merely because it has many rows.
    rows.extend(
        _target(split="train", cluster=0, member=index)
        for index in range(1, 31)
    )
    return rows


def test_v2_preregistration_is_append_only_and_exact_binomial():
    record = build_preregistration_v2()

    assert validate_preregistration(record) == []
    assert record["schema_version"] == "c5_prospective_panel_preregistration_v2"
    assert record["amendment"]["predictions_observed"] is False
    assert record["amendment"]["calibration_labels_observed"] is False
    assert record["amendment"]["evaluation_labels_observed"] is False
    selection = record["protocol"]["target_selection"]
    assert selection["unit"] == "one paired-chain instance per ab_ag_cluster"
    assert selection["deduplicate_source_cluster"] is True
    assert selection["evaluation"]["reserve_targets"] == 4
    risk = record["protocol"]["risk_control"]
    assert risk["certificate_method"] == "exact_binomial_bonferroni"
    assert risk["sensitivity_certificate_method"] == "uniform_hoeffding"
    assert record["design_analysis"]["alpha_0.30"][
        "minimum_zero_failure_trusted"
    ] == 18
    assert (
        minimum_zero_failure_trusted_exact(
            alpha=0.30,
            delta=0.10,
            candidate_count=50,
        )
        == 18
    )


def test_tracked_v2_panel_is_reproducible_unique_and_label_free():
    preregistration = json.loads(TRACKED_PREREGISTRATION.read_text())
    rows = load_c5_manifest(TRACKED_PANEL)
    audit = json.loads(TRACKED_AUDIT.read_text())
    commitment = json.loads(TRACKED_COMMITMENT.read_text())
    blocked = blocked_pdb_ids_from_manifests(FROMM_MANIFEST, GRAY_MANIFEST)

    assert preregistration == build_preregistration_v2()
    assert len(rows) == 144
    assert len({row["source_cluster_sha256"] for row in rows}) == 144
    assert validate_public_panel(
        rows,
        preregistration,
        blocked_pdb_ids=blocked,
    ) == []
    assert commitment["panel"]["manifest_sha256"] == canonical_sha256(rows)
    assert commitment["panel"]["unique_source_clusters"] == 144
    assert commitment["panel"]["duplicate_source_clusters"] == 0
    assert commitment["validation"] == {"issues": [], "passed": True}
    assert audit["selection"]["selected_by_role"] == {
        "calibration": 80,
        "calibration_reserve": 20,
        "evaluation": 40,
        "evaluation_reserve": 4,
    }
    assert audit["privacy"]["raw_sequences_emitted"] is False
    assert audit["privacy"]["native_interface_labels_read"] is False
    assert audit["privacy"]["dockq_values_read"] is False


def test_tracked_v2_input_freeze_is_cluster_unique_and_fail_closed():
    preregistration = json.loads(TRACKED_PREREGISTRATION.read_text())
    candidate_rows = load_c5_manifest(TRACKED_PANEL)
    retained_rows = load_c5_manifest(TRACKED_RETAINED_PANEL)
    freeze = json.loads(TRACKED_INPUT_FREEZE.read_text())
    blocked = blocked_pdb_ids_from_manifests(FROMM_MANIFEST, GRAY_MANIFEST)

    assert len(retained_rows) == 120
    assert len({row["source_cluster_sha256"] for row in retained_rows}) == 120
    assert validate_public_panel(
        retained_rows,
        preregistration,
        blocked_pdb_ids=blocked,
        expected_role_counts={"calibration": 80, "evaluation": 40},
    ) == []
    assert freeze["preregistration_id"] == preregistration["preregistration_id"]
    assert freeze["protocol_sha256"] == preregistration["commitment"][
        "protocol_sha256"
    ]
    assert freeze["candidate_manifest"] == {
        "rows": 144,
        "sha256": canonical_sha256(candidate_rows),
    }
    assert freeze["retention"]["manifest_sha256"] == canonical_sha256(
        retained_rows
    )
    assert freeze["retention"]["retained_by_role"] == {
        "calibration": 80,
        "evaluation": 40,
    }
    assert freeze["retention"]["validation_issues"] == []
    assert freeze["structure_qc"]["checked"] == 144
    assert freeze["structure_qc"]["passed"] == 144
    assert freeze["structure_qc"]["issues_by_reason"] == {}
    assert freeze["structure_qc"]["dockq_or_interface_labels_read"] is False
    assert (
        freeze["structure_qc"][
            "raw_structures_or_sequences_emitted_publicly"
        ]
        is False
    )
    assert freeze["af3_inputs"]["files"] == 120
    assert freeze["af3_inputs"]["templates_disabled"] is True
    assert freeze["af3_inputs"]["raw_sequences_emitted_publicly"] is False
    assert freeze["decision"] == {
        "external_specialist_trust_enabled": False,
        "ready_for_af3_prediction": True,
        "ready_for_dpo_rlvr": False,
        "ready_for_model_training": False,
    }


def test_tracked_v3_refreeze_changes_only_runtime_chain_ids():
    old_freeze = json.loads(TRACKED_INPUT_FREEZE.read_text())
    refreeze = json.loads(TRACKED_INPUT_REFREEZE.read_text())

    assert refreeze["schema_version"] == (
        "c5_prospective_af3_input_refreeze_v1"
    )
    amendment = refreeze["amendment"]
    assert amendment["kind"] == (
        "append_only_pre_inference_runtime_compatibility_refreeze"
    )
    assert amendment["only_af3_runtime_chain_ids_changed"] is True
    assert amendment["target_set_changed"] is False
    assert amendment["panel_roles_changed"] is False
    assert amendment["native_sequences_changed"] is False
    assert amendment["templates_or_model_seeds_changed"] is False
    assert amendment["input_dialect_or_version_changed"] is False
    assert amendment["native_chain_mapping_changed"] is False
    assert amendment["prediction_or_inference_outputs_read"] is False
    assert amendment["dockq_or_interface_labels_read"] is False
    assert refreeze["af3_inputs"]["files"] == 120
    assert refreeze["af3_inputs"]["remapped_targets"] == 11
    assert refreeze["af3_inputs"]["remapped_chains"] == 22
    assert refreeze["af3_inputs"]["unchanged_input_files"] == 109
    assert refreeze["af3_inputs"]["sequence_set_sha256"] == (
        old_freeze["af3_inputs"]["sequence_set_sha256"]
    )
    validation = refreeze["refreeze_validation"]
    assert all(
        value
        for key, value in validation.items()
        if key != "raw_sequences_or_chain_ids_emitted_publicly"
    )
    assert validation["raw_sequences_or_chain_ids_emitted_publicly"] is False
    assert not any(refreeze["release_boundary"].values())
    assert refreeze["decision"] == {
        "external_specialist_trust_enabled": False,
        "ready_for_af3_prediction": True,
        "ready_for_dpo_rlvr": False,
        "ready_for_model_training": False,
        "refreeze_validated": True,
    }


def test_cluster_first_selection_is_unique_and_permutation_stable():
    preregistration = build_preregistration_v2()
    eligible = _eligible_v2()

    rows, audit = select_public_panel(
        eligible,
        preregistration,
        blocked_pdb_ids=set(),
    )
    reversed_rows, reversed_audit = select_public_panel(
        list(reversed(eligible)),
        preregistration,
        blocked_pdb_ids=set(),
    )

    assert rows == reversed_rows
    assert len(rows) == 144
    assert audit == reversed_audit
    assert audit["selected_by_role"] == {
        "calibration": 80,
        "calibration_reserve": 20,
        "evaluation": 40,
        "evaluation_reserve": 4,
    }
    assert audit["selected_unique_source_clusters"] == 144
    assert audit["conflict_clusters_skipped"] == 0
    assert len({row["source_cluster_sha256"] for row in rows}) == 144
    assert validate_public_panel(
        rows,
        preregistration,
        blocked_pdb_ids=set(),
    ) == []


def test_v2_panel_rejects_duplicate_cluster_within_one_split():
    preregistration = build_preregistration_v2()
    rows, _ = select_public_panel(
        _eligible_v2(),
        preregistration,
        blocked_pdb_ids=set(),
    )
    tampered = deepcopy(rows)
    tampered[1]["source_cluster_sha256"] = tampered[0][
        "source_cluster_sha256"
    ]

    issues = validate_public_panel(
        tampered,
        preregistration,
        blocked_pdb_ids=set(),
    )

    assert any("source_cluster_duplicate" in issue for issue in issues)


def test_v2_selection_fails_when_unique_test_clusters_are_insufficient():
    preregistration = build_preregistration_v2()
    eligible = [
        _target(split="train", cluster=index)
        for index in range(100)
    ]
    eligible.extend(
        _target(split="test", cluster=index)
        for index in range(43)
    )

    with pytest.raises(
        ProspectiveSourceError,
        match="insufficient_test_clusters:43<44",
    ):
        select_public_panel(
            eligible,
            preregistration,
            blocked_pdb_ids=set(),
        )


def test_exact_binomial_calibration_boundaries_are_locked():
    thresholds = (0.5,)
    twelve_failures = [(0.9, index >= 12) for index in range(80)]
    thirteen_failures = [(0.9, index >= 13) for index in range(80)]
    eighteen_clean = [(0.9, True)] * 18
    seventeen_clean = [(0.9, True)] * 17

    certified = select_exact_binomial_certificate(
        twelve_failures,
        alpha=0.30,
        delta=0.10,
        thresholds=thresholds,
        multiplicity=50,
    )
    rejected = select_exact_binomial_certificate(
        thirteen_failures,
        alpha=0.30,
        delta=0.10,
        thresholds=thresholds,
        multiplicity=50,
    )
    minimum = select_exact_binomial_certificate(
        eighteen_clean,
        alpha=0.30,
        delta=0.10,
        thresholds=thresholds,
        multiplicity=50,
    )
    below_minimum = select_exact_binomial_certificate(
        seventeen_clean,
        alpha=0.30,
        delta=0.10,
        thresholds=thresholds,
        multiplicity=50,
    )

    assert certified["certified"] is True
    assert certified["calibration_failures"] == 12
    assert rejected["certified"] is False
    assert minimum["certified"] is True
    assert below_minimum["certified"] is False


def test_exact_binomial_fixed_evaluation_makes_minimum_coverage_coherent():
    metrics = fixed_threshold_exact_binomial_metrics(
        [(0.9, True)] * 10,
        threshold=0.5,
        alpha=0.30,
        delta=0.10,
    )

    assert metrics["trusted"] == 10
    assert metrics["failures_among_trusted"] == 0
    assert metrics["risk_upper_bound"] <= 0.30
    assert metrics["risk_test_passed"] is True
