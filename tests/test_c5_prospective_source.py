import csv
import json
from pathlib import Path

import pytest

from c5_antibody_ood.prospective_panel import (
    build_panel_commitment,
    build_preregistration,
    validate_public_panel,
)
from c5_antibody_ood.prospective_source import (
    EligibleTarget,
    ProspectiveSourceContract,
    ProspectiveSourceError,
    blocked_pdb_ids_from_manifests,
    intake_sabdab2_split,
    select_public_panel,
)
from c5_antibody_ood.manifest import load_c5_manifest
from c5_antibody_ood.source_pilot import sha256_file


ROOT = Path(__file__).resolve().parents[1]
TRACKED_PREREGISTRATION = (
    ROOT
    / "c5_antibody_ood/c5_prospective_panel_preregistration_v1.json"
)
TRACKED_PANEL = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_panel_manifest_v1.jsonl"
)
TRACKED_AUDIT = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_source_audit_2026-07-25.json"
)
TRACKED_COMMITMENT = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_panel_commitment_v1.json"
)
FROMM_MANIFEST = (
    ROOT / "c5_antibody_ood/c5_source_backed_manifest_v1.jsonl"
)
GRAY_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_gray_independent_calibration_manifest_v1.jsonl"
)


REQUIRED_VALUES = {
    "INSTANCE": "pdb_00009aaa_H_L",
    "PDB_ID": "pdb_00009aaa",
    "SABDAB_ID": "H0001L0001",
    "PDBdepo": "2025-01-01",
    "method": "XRAY",
    "resolution": "2.5",
    "type": "FAB",
    "holo": "True",
    "Hchain": "H",
    "Lchain": "L",
    "agchains": "A/B",
    "agtypes": "PROTEIN/ION",
    "ab_ag_cluster": "cluster_train_1",
    "ab_ag_split": "train",
}


def _write_source_fixture(path: Path) -> ProspectiveSourceContract:
    sequence_columns = [
        "Hseq",
        "Lseq",
        "Hseq_expected",
        "Lseq_expected",
        "VH_numerable_seq",
        "VL_numerable_seq",
        "agresolvedseqs",
        "agexpectedseqs",
    ]
    filler_columns = [
        f"metadata_{index:02d}"
        for index in range(48 - len(REQUIRED_VALUES) - len(sequence_columns))
    ]
    columns = [*REQUIRED_VALUES, *sequence_columns, *filler_columns]
    rows = []
    for index, changes in enumerate(
        (
            {},
            {
                "INSTANCE": "pdb_00009aab_H_L",
                "PDB_ID": "pdb_00009aab",
                "SABDAB_ID": "H0002L0002",
                "ab_ag_cluster": "cluster_test_1",
                "ab_ag_split": "test",
            },
            {
                "INSTANCE": "pdb_00009aac_H_L",
                "PDB_ID": "pdb_00009aac",
                "SABDAB_ID": "H0003L0003",
                "PDBdepo": "2021-09-30",
            },
            {
                "INSTANCE": "pdb_00009aad_H_+",
                "PDB_ID": "pdb_00009aad",
                "SABDAB_ID": "H0004L0000",
                "type": "SD-H",
                "Lchain": "+",
            },
        )
    ):
        row = {
            **REQUIRED_VALUES,
            **changes,
            **{column: "ACDE" for column in sequence_columns},
            **{column: f"value_{index}" for column in filler_columns},
        }
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return ProspectiveSourceContract(
        expected_sha256=sha256_file(path),
        expected_bytes=path.stat().st_size,
        expected_rows=4,
        expected_columns=48,
    )


def _eligible_targets(
    *,
    train: int,
    test: int,
) -> list[EligibleTarget]:
    targets = []
    for source_split, count, offset in (
        ("train", train, 0),
        ("test", test, 1000),
    ):
        for index in range(count):
            value = offset + index
            pdb_id = f"{value:04x}"[-4:]
            targets.append(
                EligibleTarget(
                    instance_id=f"pdb_0000{pdb_id}_H_L",
                    pdb_id=pdb_id,
                    sabdab_id=f"H{value:05d}L{value:05d}",
                    source_split=source_split,
                    release_date="2025-01-01",
                    experimental_method="EM",
                    resolution_angstrom=3.0,
                    heavy_chain="H",
                    light_chain="L",
                    antigen_chains=("A",),
                    source_row_sha256=f"{value + 1:064x}"[-64:],
                    source_cluster_sha256=f"{value + 2000:064x}"[-64:],
                )
            )
    return targets


def test_source_intake_excludes_sequences_and_applies_metadata_qc(tmp_path):
    path = tmp_path / "abag_split.csv"
    contract = _write_source_fixture(path)

    eligible, audit = intake_sabdab2_split(
        path,
        build_preregistration(),
        contract=contract,
    )

    assert len(eligible) == 2
    assert audit["source"]["rows"] == 4
    assert audit["source"]["columns"] == 48
    assert audit["source"]["excluded_column_count"] == 34
    assert audit["eligibility"]["excluded_by_reason"] == {
        "before_release_date_cutoff": 1,
        "not_paired_chain_antibody": 1,
    }
    assert audit["privacy"]["raw_sequences_emitted"] is False
    assert all(not hasattr(target, "Hseq") for target in eligible)
    assert all(target.antigen_chains == ("A",) for target in eligible)


def test_source_intake_fails_closed_on_checksum_and_schema(tmp_path):
    path = tmp_path / "abag_split.csv"
    contract = _write_source_fixture(path)
    bad_hash = ProspectiveSourceContract(
        expected_sha256="0" * 64,
        expected_bytes=contract.expected_bytes,
        expected_rows=contract.expected_rows,
        expected_columns=contract.expected_columns,
    )

    with pytest.raises(ProspectiveSourceError, match="source_sha256_mismatch"):
        intake_sabdab2_split(
            path,
            build_preregistration(),
            contract=bad_hash,
        )


def test_metadata_only_selection_builds_exact_primary_and_reserve_roles():
    eligible = _eligible_targets(train=110, test=60)
    preregistration = build_preregistration()

    rows, audit = select_public_panel(
        eligible,
        preregistration,
        blocked_pdb_ids=set(),
    )

    assert len(rows) == 150
    assert audit["selected_by_role"] == {
        "calibration": 80,
        "calibration_reserve": 20,
        "evaluation": 40,
        "evaluation_reserve": 10,
    }
    assert audit["selected_unique_pdb_ids"] == 150
    assert audit["selected_unique_sabdab_ids"] == 150
    assert audit["selected_source_cluster_overlap"] == 0
    assert validate_public_panel(
        rows,
        preregistration,
        blocked_pdb_ids=set(),
    ) == []
    commitment = build_panel_commitment(
        rows,
        preregistration,
        blocked_pdb_ids=set(),
    )
    assert commitment["decision"]["ready_for_prediction"] is True


def test_selection_excludes_prior_pdb_ids_before_hash_rank():
    eligible = _eligible_targets(train=110, test=60)
    blocked = {eligible[0].pdb_id}

    rows, audit = select_public_panel(
        eligible,
        build_preregistration(),
        blocked_pdb_ids=blocked,
    )

    assert all(row["pdb_id"] not in blocked for row in rows)
    assert audit["blocked_overlap_rows_excluded"] == 1


def test_selection_aborts_when_same_split_reserve_is_insufficient():
    eligible = _eligible_targets(train=99, test=50)

    with pytest.raises(
        ProspectiveSourceError,
        match="insufficient_train_targets:99<100",
    ):
        select_public_panel(
            eligible,
            build_preregistration(),
            blocked_pdb_ids=set(),
        )


def test_tracked_cayuga_source_intake_and_panel_commitment_are_locked():
    preregistration = json.loads(TRACKED_PREREGISTRATION.read_text())
    rows = load_c5_manifest(TRACKED_PANEL)
    audit = json.loads(TRACKED_AUDIT.read_text())
    tracked_commitment = json.loads(TRACKED_COMMITMENT.read_text())
    blocked = blocked_pdb_ids_from_manifests(
        FROMM_MANIFEST,
        GRAY_MANIFEST,
    )

    assert len(blocked) == 197
    assert len(rows) == 150
    assert validate_public_panel(
        rows,
        preregistration,
        blocked_pdb_ids=blocked,
    ) == []
    assert (
        build_panel_commitment(
            rows,
            preregistration,
            blocked_pdb_ids=blocked,
        )
        == tracked_commitment
    )
    assert audit["source"]["rows"] == 15_641
    assert audit["source"]["columns"] == 48
    assert audit["source"]["excluded_column_count"] == 34
    assert audit["eligibility"]["retained_rows"] == 2_417
    assert audit["selection"]["blocked_overlap_rows_excluded"] == 129
    assert audit["selection"]["selected_source_cluster_overlap"] == 0
    assert audit["privacy"]["raw_sequences_emitted"] is False
    assert audit["privacy"]["native_interface_labels_read"] is False
    assert audit["decision"]["ready_for_prediction"] is True
    assert audit["decision"]["external_specialist_trust_enabled"] is False
