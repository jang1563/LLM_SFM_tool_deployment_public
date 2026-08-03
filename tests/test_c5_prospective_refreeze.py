import copy
import hashlib
import json
from pathlib import Path

import pytest

from c5_antibody_ood.manifest import load_c5_manifest
from c5_antibody_ood.prospective_inputs import _safe_job_name
from c5_antibody_ood.prospective_panel import canonical_sha256
from c5_antibody_ood.prospective_refreeze import (
    ProspectiveRefreezeError,
    refreeze_af3_inputs,
)
from c5_antibody_ood.source_pilot import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "c5_antibody_ood/c5_prospective_panel_preregistration_v2.json"
)
RETAINED_MANIFEST = (
    ROOT / "c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v2.jsonl"
)


def _fixture(tmp_path: Path):
    preregistration = json.loads(PREREGISTRATION.read_text())
    row = copy.deepcopy(load_c5_manifest(RETAINED_MANIFEST)[0])
    mappings = row["chain_role_mapping"]
    mappings[0]["chain_id"] = "H"
    mappings[1]["chain_id"] = "L"
    mappings[2]["chain_id"] = "1"
    for index in range(3, len(mappings)):
        mappings[index]["chain_id"] = f"{index + 1}"
    rows = [row]
    job_name = _safe_job_name(str(row["instance_id"]))
    old_dir = tmp_path / "old_inputs"
    old_dir.mkdir()
    sequence = "ACDEFGHIK"
    payload = {
        "name": job_name,
        "modelSeeds": preregistration["protocol"]["prediction"]["model_seeds"],
        "sequences": [
            {
                "protein": {
                    "id": mapping["chain_id"],
                    "sequence": sequence,
                    "templates": [],
                }
            }
            for mapping in mappings
        ],
        "dialect": preregistration["protocol"]["prediction"]["input_dialect"],
        "version": preregistration["protocol"]["prediction"]["input_version"],
    }
    old_path = old_dir / f"{job_name}.json"
    old_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sequence_set_sha256 = canonical_sha256(
        {
            str(row["target_id"]): canonical_sha256(
                [
                    {
                        "chain_id": mapping["chain_id"],
                        "sequence_sha256": hashlib.sha256(
                            sequence.encode()
                        ).hexdigest(),
                    }
                    for mapping in mappings
                ]
            )
        }
    )
    old_input_set_sha256 = canonical_sha256(
        {str(row["target_id"]): sha256_file(old_path)}
    )
    old_freeze = {
        "schema_version": "c5_prospective_af3_input_freeze_v1",
        "preregistration_id": preregistration["preregistration_id"],
        "protocol_sha256": preregistration["commitment"]["protocol_sha256"],
        "candidate_manifest": {},
        "structure_qc": {},
        "retention": {
            "rows": 1,
            "manifest_sha256": canonical_sha256(rows),
        },
        "af3_inputs": {
            "files": 1,
            "af3_input_set_sha256": old_input_set_sha256,
            "sequence_set_sha256": sequence_set_sha256,
            "templates_disabled": True,
            "raw_sequences_emitted_publicly": False,
        },
        "decision": {
            "ready_for_af3_prediction": True,
            "external_specialist_trust_enabled": False,
            "ready_for_model_training": False,
            "ready_for_dpo_rlvr": False,
        },
    }
    chain_audit = {
        "schema_version": "c5_v2_af3_chain_id_compatibility_audit_v1",
        "expected_cases": 1,
        "current_input_set": {
            "runtime_incompatible_cases": 1,
            "runtime_incompatible_chains": len(mappings) - 2,
        },
        "proposed_minimal_bridge": {
            "collision_cases": 0,
            "runtime_incompatible_cases_after_bridge": 0,
        },
        "decision": {
            "deterministic_minimal_bridge_technically_feasible": True,
            "remaining_cpu_array_authorized": False,
        },
        "release_boundary": {
            "target_identifiers_emitted": False,
            "local_paths_emitted": False,
        },
    }
    return preregistration, rows, old_dir, old_freeze, chain_audit


def test_refreeze_changes_only_invalid_chain_ids_and_records_private_bridge(
    tmp_path,
):
    preregistration, rows, old_dir, old_freeze, chain_audit = _fixture(
        tmp_path
    )
    new_dir = tmp_path / "new_inputs"
    mapping_path = tmp_path / "private_mapping.jsonl"

    report = refreeze_af3_inputs(
        old_input_dir=old_dir,
        new_input_dir=new_dir,
        private_chain_mapping_out=mapping_path,
        old_input_freeze=old_freeze,
        old_input_freeze_sha256="a" * 64,
        chain_compatibility_audit=chain_audit,
        chain_compatibility_audit_sha256="b" * 64,
        retained_rows=rows,
        preregistration=preregistration,
    )

    payload = json.loads(next(new_dir.glob("*.json")).read_text())
    observed = [entry["protein"]["id"] for entry in payload["sequences"]]
    assert observed[:3] == ["H", "L", "A"]
    assert len(observed) == len(set(observed))
    assert mapping_path.stat().st_mode & 0o777 == 0o600
    private_mapping = json.loads(mapping_path.read_text().splitlines()[0])
    assert private_mapping["chain_mapping"][2] == {
        "native_chain_id": "1",
        "af3_chain_id": "A",
        "role": rows[0]["chain_role_mapping"][2]["role"],
    }
    assert report["schema_version"] == "c5_prospective_af3_input_refreeze_v1"
    assert report["af3_inputs"]["remapped_targets"] == 1
    assert report["af3_inputs"]["remapped_chains"] >= 1
    assert report["refreeze_validation"]["sequence_set_checksum_unchanged"] is True
    assert report["decision"]["ready_for_af3_prediction"] is True
    assert all(report["release_boundary"].values()) is False
    public_text = json.dumps(report, sort_keys=True)
    assert "ACDEFGHIK" not in public_text
    assert str(tmp_path) not in public_text


def test_refreeze_rejects_stale_old_input_commitment_before_writing(tmp_path):
    preregistration, rows, old_dir, old_freeze, chain_audit = _fixture(
        tmp_path
    )
    old_freeze["af3_inputs"]["af3_input_set_sha256"] = "0" * 64
    new_dir = tmp_path / "new_inputs"

    with pytest.raises(
        ProspectiveRefreezeError,
        match="old_input_set_checksum_mismatch",
    ):
        refreeze_af3_inputs(
            old_input_dir=old_dir,
            new_input_dir=new_dir,
            private_chain_mapping_out=tmp_path / "private_mapping.jsonl",
            old_input_freeze=old_freeze,
            old_input_freeze_sha256="a" * 64,
            chain_compatibility_audit=chain_audit,
            chain_compatibility_audit_sha256="b" * 64,
            retained_rows=rows,
            preregistration=preregistration,
        )

    assert not new_dir.exists()
