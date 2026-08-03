import json
from pathlib import Path

import pytest

from c5_antibody_ood.manifest import load_c5_manifest
from c5_antibody_ood.prospective_inputs import (
    ProspectiveInputError,
    StructureQC,
    af3_runtime_chain_id_compatible,
    assign_af3_chain_ids,
    retain_primary_or_promote_reserve,
    write_private_af3_inputs,
)
from c5_antibody_ood.prospective_panel import (
    canonical_sha256,
    validate_public_panel,
)
from c5_antibody_ood.prospective_source import (
    blocked_pdb_ids_from_manifests,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "c5_antibody_ood/c5_prospective_panel_preregistration_v1.json"
)
CANDIDATE_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_panel_manifest_v1.jsonl"
)
RETAINED_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl"
)
INPUT_FREEZE = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json"
)
FROMM_MANIFEST = (
    ROOT / "c5_antibody_ood/c5_source_backed_manifest_v1.jsonl"
)
GRAY_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_gray_independent_calibration_manifest_v1.jsonl"
)


def _qc_results(
    rows,
    *,
    failed_targets=(),
) -> list[StructureQC]:
    failed = set(failed_targets)
    return [
        StructureQC(
            target_id=row["target_id"],
            panel_role=row["panel_role"],
            passed=row["target_id"] not in failed,
            issues=(
                ("polymer_sequence_empty",)
                if row["target_id"] in failed
                else ()
            ),
            structure_sha256="a" * 64,
            chain_sequences=tuple(
                (
                    mapping["chain_id"],
                    "ACDEFGHIKLMNPQRSTVWY",
                )
                for mapping in row["chain_role_mapping"]
            ),
        )
        for row in rows
    ]


def test_all_passing_primary_targets_are_retained_without_promotion():
    rows = load_c5_manifest(CANDIDATE_MANIFEST)
    preregistration = json.loads(PREREGISTRATION.read_text())

    retained, audit = retain_primary_or_promote_reserve(
        rows,
        _qc_results(rows),
        preregistration,
    )

    assert len(retained) == 120
    assert audit["retained_by_role"] == {
        "calibration": 80,
        "evaluation": 40,
    }
    assert audit["promotions_by_role"] == {
        "calibration": 0,
        "evaluation": 0,
    }
    assert {
        row["target_id"]
        for row in retained
        if row["panel_role"] == "calibration"
    } == {
        row["target_id"]
        for row in rows
        if row["panel_role"] == "calibration"
    }


def test_failed_primary_promotes_only_same_split_frozen_reserve():
    rows = load_c5_manifest(CANDIDATE_MANIFEST)
    preregistration = json.loads(PREREGISTRATION.read_text())
    failed = next(
        row["target_id"]
        for row in rows
        if row["panel_role"] == "calibration"
    )
    first_reserve = next(
        row["target_id"]
        for row in rows
        if row["panel_role"] == "calibration_reserve"
        and row["selection_rank"] == 1
    )

    retained, audit = retain_primary_or_promote_reserve(
        rows,
        _qc_results(rows, failed_targets={failed}),
        preregistration,
    )

    retained_ids = {row["target_id"] for row in retained}
    assert failed not in retained_ids
    assert first_reserve in retained_ids
    assert audit["failed_primary_by_role"]["calibration"] == 1
    assert audit["promotions_by_role"]["calibration"] == 1
    promoted = next(row for row in retained if row["target_id"] == first_reserve)
    assert promoted["panel_role"] == "calibration"
    assert promoted["source_split"] == "train"


def test_exhausted_same_split_reserve_aborts_without_cross_split_fill():
    rows = load_c5_manifest(CANDIDATE_MANIFEST)
    preregistration = json.loads(PREREGISTRATION.read_text())
    failed = {
        row["target_id"]
        for row in rows
        if row["panel_role"]
        in {"calibration", "calibration_reserve"}
    }

    with pytest.raises(
        ProspectiveInputError,
        match="structure_qc_insufficient_calibration",
    ):
        retain_primary_or_promote_reserve(
            rows,
            _qc_results(rows, failed_targets=failed),
            preregistration,
        )


def test_private_af3_inputs_disable_templates_and_return_only_hashes(tmp_path):
    rows = load_c5_manifest(CANDIDATE_MANIFEST)
    preregistration = json.loads(PREREGISTRATION.read_text())
    retained = [rows[0]]
    qc = _qc_results(retained)

    commitment = write_private_af3_inputs(
        retained,
        qc,
        preregistration,
        tmp_path / "inputs",
        tmp_path / "private_chain_mapping.jsonl",
    )

    files = list((tmp_path / "inputs").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["modelSeeds"] == [20260725]
    assert all(
        sequence["protein"]["templates"] == []
        for sequence in payload["sequences"]
    )
    assert commitment["files"] == 1
    assert commitment["private_chain_mapping_manifest_written"] is True
    assert commitment["af3_chain_ids_runtime_compatible"] is True
    assert commitment["raw_sequences_emitted_publicly"] is False
    rendered_commitment = json.dumps(commitment)
    assert "ACDEFGHIKLMNPQRSTVWY" not in rendered_commitment


def test_af3_chain_id_bridge_preserves_valid_and_remaps_only_invalid_ids():
    assigned = assign_af3_chain_ids(("H", "L", "1", "b"))

    assert assigned == ("H", "L", "A", "B")
    assert all(af3_runtime_chain_id_compatible(chain_id) for chain_id in assigned)
    assert assign_af3_chain_ids(("D", "E", "A")) == ("D", "E", "A")


def test_af3_chain_id_bridge_rejects_duplicate_native_ids():
    with pytest.raises(ProspectiveInputError, match="native_chain_ids_invalid"):
        assign_af3_chain_ids(("A", "A"))


def test_private_mapping_collision_fails_before_writing_inputs(tmp_path):
    rows = load_c5_manifest(CANDIDATE_MANIFEST)
    preregistration = json.loads(PREREGISTRATION.read_text())
    retained = [rows[0]]
    mapping_path = tmp_path / "private_chain_mapping.jsonl"
    mapping_path.write_text("occupied\n")
    input_dir = tmp_path / "inputs"

    with pytest.raises(
        ProspectiveInputError,
        match="private_chain_mapping_output_exists",
    ):
        write_private_af3_inputs(
            retained,
            _qc_results(retained),
            preregistration,
            input_dir,
            mapping_path,
        )

    assert not input_dir.exists()


def test_tracked_cayuga_input_freeze_is_public_safe_and_locked():
    preregistration = json.loads(PREREGISTRATION.read_text())
    candidates = load_c5_manifest(CANDIDATE_MANIFEST)
    retained = load_c5_manifest(RETAINED_MANIFEST)
    audit = json.loads(INPUT_FREEZE.read_text())
    blocked = blocked_pdb_ids_from_manifests(
        FROMM_MANIFEST,
        GRAY_MANIFEST,
    )

    assert len(candidates) == 150
    assert len(retained) == 120
    assert validate_public_panel(
        retained,
        preregistration,
        blocked_pdb_ids=blocked,
        expected_role_counts={
            "calibration": 80,
            "evaluation": 40,
        },
    ) == []
    assert audit["candidate_manifest"]["sha256"] == canonical_sha256(
        candidates
    )
    assert audit["retention"]["manifest_sha256"] == canonical_sha256(
        retained
    )
    assert audit["structure_qc"]["checked"] == 150
    assert audit["structure_qc"]["passed"] == 150
    assert audit["structure_qc"]["issues_by_reason"] == {}
    assert audit["retention"]["promotions_by_role"] == {
        "calibration": 0,
        "evaluation": 0,
    }
    assert audit["af3_inputs"]["files"] == 120
    assert audit["af3_inputs"]["templates_disabled"] is True
    assert audit["decision"]["ready_for_af3_prediction"] is True
    assert audit["decision"]["external_specialist_trust_enabled"] is False
    rendered = json.dumps(audit, sort_keys=True).lower()
    assert "raw_sequences_emitted_publicly\": true" not in rendered
    assert "native_path" not in rendered
    assert "output_path" not in rendered
