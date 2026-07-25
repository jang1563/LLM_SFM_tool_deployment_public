import json
from copy import deepcopy
from pathlib import Path

import pytest

from c5_antibody_ood.manifest import load_c5_manifest
from c5_antibody_ood.prospective_inputs import StructureQC
from c5_antibody_ood.prospective_native_lock import (
    ProspectiveNativeLockError,
    build_native_structure_lock_from_qc,
    validate_native_structure_lock,
)
from c5_antibody_ood.prospective_panel import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
INPUT_FREEZE = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json"
)
CANDIDATE_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_panel_manifest_v1.jsonl"
)
RETAINED_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl"
)


def _fixture() -> tuple[list[dict], list[dict], dict, list[StructureQC]]:
    candidates = load_c5_manifest(CANDIDATE_MANIFEST)
    retained = load_c5_manifest(RETAINED_MANIFEST)
    input_freeze = deepcopy(json.loads(INPUT_FREEZE.read_text()))
    qc = [
        StructureQC(
            target_id=row["target_id"],
            panel_role=row["panel_role"],
            passed=True,
            issues=(),
            structure_sha256=canonical_sha256(
                {"target_id": row["target_id"], "kind": "native"}
            ),
            chain_sequences=tuple(
                (mapping["chain_id"], "ACDE")
                for mapping in row["chain_role_mapping"]
            ),
        )
        for row in candidates
    ]
    input_freeze["structure_qc"]["native_structure_set_sha256"] = (
        canonical_sha256(
            {
                result.target_id: result.structure_sha256
                for result in qc
            }
        )
    )
    return candidates, retained, input_freeze, qc


def test_native_structure_lock_recovers_hashes_without_sequences_or_paths():
    candidates, retained, input_freeze, qc = _fixture()

    lock = build_native_structure_lock_from_qc(
        candidate_rows=candidates,
        retained_rows=retained,
        input_freeze=input_freeze,
        qc_results=qc,
    )

    assert lock["counts"] == {
        "candidate_targets": 150,
        "retained_targets": 120,
        "qc_passed": 150,
    }
    assert validate_native_structure_lock(
        lock,
        retained_rows=retained,
        input_freeze=input_freeze,
    ) == []
    rendered = json.dumps(lock, sort_keys=True)
    assert "ACDE" not in rendered
    assert "structure_path" not in rendered
    assert lock["evidence_boundary"][
        "dockq_or_interface_labels_read"
    ] is False


def test_native_structure_drift_fails_before_label_reveal():
    candidates, retained, input_freeze, qc = _fixture()
    qc[0] = StructureQC(
        target_id=qc[0].target_id,
        panel_role=qc[0].panel_role,
        passed=True,
        issues=(),
        structure_sha256="0" * 64,
        chain_sequences=qc[0].chain_sequences,
    )

    with pytest.raises(
        ProspectiveNativeLockError,
        match="native_structure_set_sha256_mismatch",
    ):
        build_native_structure_lock_from_qc(
            candidate_rows=candidates,
            retained_rows=retained,
            input_freeze=input_freeze,
            qc_results=qc,
        )
