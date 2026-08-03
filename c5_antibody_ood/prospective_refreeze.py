"""Append-only AF3 input refreeze with a native-to-runtime chain-ID bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .af3_phase_outputs import _contains_forbidden_label_key
from .manifest import load_c5_manifest
from .prospective_inputs import (
    _safe_job_name,
    af3_runtime_chain_id_compatible,
    assign_af3_chain_ids,
)
from .prospective_panel import (
    canonical_sha256,
    validate_preregistration,
    write_json,
)
from .source_pilot import sha256_file


REFREEZE_SCHEMA = "c5_prospective_af3_input_refreeze_v1"
CHAIN_AUDIT_SCHEMA = "c5_v2_af3_chain_id_compatibility_audit_v1"
FROZEN_TOP_LEVEL_KEYS = {
    "dialect",
    "modelSeeds",
    "name",
    "sequences",
    "version",
}
FROZEN_PROTEIN_KEYS = {"id", "sequence", "templates"}


class ProspectiveRefreezeError(ValueError):
    """Raised when an AF3 input set cannot be refrozen without drift."""


def _load_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveRefreezeError("json_invalid") from exc
    if not isinstance(value, dict):
        raise ProspectiveRefreezeError("json_object_required")
    return value


def _expected_inputs(
    retained_rows: Sequence[Mapping[str, Any]],
    input_dir: Path,
) -> dict[str, Path]:
    expected: dict[str, Path] = {}
    for row in retained_rows:
        target_id = str(row["target_id"])
        if target_id in expected:
            raise ProspectiveRefreezeError("retained_target_duplicate")
        expected[target_id] = input_dir / (
            _safe_job_name(str(row["instance_id"])) + ".json"
        )
    return expected


def _validate_exact_inventory(
    expected: Mapping[str, Path],
    input_dir: Path,
) -> None:
    if not input_dir.is_dir() or input_dir.is_symlink():
        raise ProspectiveRefreezeError("input_directory_invalid")
    entries = set(input_dir.iterdir())
    expected_paths = set(expected.values())
    if entries != expected_paths:
        raise ProspectiveRefreezeError("input_inventory_mismatch")
    if any(
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size <= 0
        for path in expected_paths
    ):
        raise ProspectiveRefreezeError("input_file_invalid")


def _input_set_sha256(expected: Mapping[str, Path]) -> str:
    return canonical_sha256(
        {
            target_id: sha256_file(path)
            for target_id, path in sorted(expected.items())
        }
    )


def _validate_chain_audit(
    audit: Mapping[str, Any],
    expected_count: int,
) -> None:
    decision = audit.get("decision", {})
    bridge = audit.get("proposed_minimal_bridge", {})
    if (
        audit.get("schema_version") != CHAIN_AUDIT_SCHEMA
        or audit.get("expected_cases") != expected_count
        or audit.get("current_input_set", {}).get(
            "runtime_incompatible_cases"
        )
        in (None, 0)
        or bridge.get("collision_cases") != 0
        or bridge.get("runtime_incompatible_cases_after_bridge") != 0
        or decision.get(
            "deterministic_minimal_bridge_technically_feasible"
        )
        is not True
        or decision.get("remaining_cpu_array_authorized") is not False
        or any(audit.get("release_boundary", {}).values())
    ):
        raise ProspectiveRefreezeError("chain_compatibility_audit_invalid")


def refreeze_af3_inputs(
    *,
    old_input_dir: str | Path,
    new_input_dir: str | Path,
    private_chain_mapping_out: str | Path,
    old_input_freeze: Mapping[str, Any],
    old_input_freeze_sha256: str,
    chain_compatibility_audit: Mapping[str, Any],
    chain_compatibility_audit_sha256: str,
    retained_rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    preregistration_issues = validate_preregistration(preregistration)
    if preregistration_issues:
        raise ProspectiveRefreezeError(
            "preregistration_invalid:" + ",".join(preregistration_issues)
        )
    protocol_sha256 = preregistration["commitment"]["protocol_sha256"]
    retained_sha256 = canonical_sha256(list(retained_rows))
    if (
        old_input_freeze.get("preregistration_id")
        != preregistration["preregistration_id"]
        or old_input_freeze.get("protocol_sha256") != protocol_sha256
        or old_input_freeze.get("retention", {}).get("manifest_sha256")
        != retained_sha256
        or old_input_freeze.get("decision", {}).get(
            "ready_for_af3_prediction"
        )
        is not True
    ):
        raise ProspectiveRefreezeError("old_input_freeze_invalid")
    _validate_chain_audit(chain_compatibility_audit, len(retained_rows))

    old_root = Path(old_input_dir)
    new_root = Path(new_input_dir)
    mapping_path = Path(private_chain_mapping_out)
    mapping_partial = mapping_path.with_name(f".{mapping_path.name}.partial")
    stage = new_root.with_name(f".{new_root.name}.partial")
    for path in (new_root, stage, mapping_path, mapping_partial):
        if path.exists() or path.is_symlink():
            raise ProspectiveRefreezeError("refreeze_output_boundary_not_clean")

    old_expected = _expected_inputs(retained_rows, old_root)
    _validate_exact_inventory(old_expected, old_root)
    old_input_set_sha256 = _input_set_sha256(old_expected)
    if old_input_set_sha256 != old_input_freeze["af3_inputs"].get(
        "af3_input_set_sha256"
    ):
        raise ProspectiveRefreezeError("old_input_set_checksum_mismatch")

    prediction = preregistration["protocol"]["prediction"]
    stage.mkdir(parents=True)
    stage.chmod(0o700)
    mapping_partial.parent.mkdir(parents=True, exist_ok=True)
    file_hashes: dict[str, str] = {}
    sequence_hashes: dict[str, str] = {}
    mapping_hashes: dict[str, str] = {}
    mapping_manifest_rows: list[dict[str, Any]] = []
    changed_targets = 0
    changed_chains = 0
    unchanged_files = 0

    for row in retained_rows:
        target_id = str(row["target_id"])
        old_path = old_expected[target_id]
        value = _load_object(old_path)
        if _contains_forbidden_label_key(value):
            raise ProspectiveRefreezeError("old_input_contains_hidden_label_key")
        if set(value) != FROZEN_TOP_LEVEL_KEYS:
            raise ProspectiveRefreezeError("old_input_top_level_schema_invalid")
        job_name = _safe_job_name(str(row["instance_id"]))
        if (
            value.get("name") != job_name
            or value.get("dialect") != prediction["input_dialect"]
            or value.get("version") != prediction["input_version"]
            or value.get("modelSeeds") != prediction["model_seeds"]
        ):
            raise ProspectiveRefreezeError("old_input_identity_mismatch")
        sequences = value.get("sequences")
        role_mapping = row["chain_role_mapping"]
        if not isinstance(sequences, list) or len(sequences) != len(role_mapping):
            raise ProspectiveRefreezeError("old_input_chain_count_mismatch")

        native_ids: list[str] = []
        sequence_commitment: list[dict[str, str]] = []
        for entry, mapping in zip(sequences, role_mapping):
            if not isinstance(entry, dict) or set(entry) != {"protein"}:
                raise ProspectiveRefreezeError("old_input_sequence_entry_invalid")
            protein = entry["protein"]
            if not isinstance(protein, dict) or set(protein) != FROZEN_PROTEIN_KEYS:
                raise ProspectiveRefreezeError("old_input_protein_schema_invalid")
            native_id = str(mapping["chain_id"])
            sequence = protein.get("sequence")
            if protein.get("id") != native_id:
                raise ProspectiveRefreezeError("old_input_native_chain_mismatch")
            if not isinstance(sequence, str) or not sequence:
                raise ProspectiveRefreezeError("old_input_sequence_invalid")
            if protein.get("templates") != []:
                raise ProspectiveRefreezeError("old_input_templates_enabled")
            native_ids.append(native_id)
            sequence_commitment.append(
                {
                    "chain_id": native_id,
                    "sequence_sha256": hashlib.sha256(
                        sequence.encode()
                    ).hexdigest(),
                }
            )

        af3_ids = assign_af3_chain_ids(native_ids)
        rewritten = json.loads(json.dumps(value))
        private_mapping: list[dict[str, str]] = []
        changed = 0
        for index, (native_id, af3_id, mapping) in enumerate(
            zip(native_ids, af3_ids, role_mapping)
        ):
            rewritten["sequences"][index]["protein"]["id"] = af3_id
            changed += int(native_id != af3_id)
            private_mapping.append(
                {
                    "native_chain_id": native_id,
                    "af3_chain_id": af3_id,
                    "role": str(mapping["role"]),
                }
            )
        if (
            len(af3_ids) != len(set(af3_ids))
            or not all(
                af3_runtime_chain_id_compatible(chain_id)
                for chain_id in af3_ids
            )
        ):
            raise ProspectiveRefreezeError("rewritten_chain_ids_invalid")

        new_path = stage / old_path.name
        new_path.write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n")
        new_path.chmod(0o600)
        file_hashes[target_id] = sha256_file(new_path)
        sequence_hashes[target_id] = canonical_sha256(sequence_commitment)
        mapping_hashes[target_id] = canonical_sha256(private_mapping)
        mapping_manifest_rows.append(
            {
                "target_id": target_id,
                "instance_id": str(row["instance_id"]),
                "chain_mapping": private_mapping,
            }
        )
        changed_chains += changed
        changed_targets += int(changed > 0)
        unchanged_files += int(sha256_file(old_path) == sha256_file(new_path))

    sequence_set_sha256 = canonical_sha256(sequence_hashes)
    if sequence_set_sha256 != old_input_freeze["af3_inputs"].get(
        "sequence_set_sha256"
    ):
        raise ProspectiveRefreezeError("sequence_set_drift")
    new_input_set_sha256 = canonical_sha256(file_hashes)
    if changed_targets <= 0 or changed_chains <= 0:
        raise ProspectiveRefreezeError("refreeze_made_no_chain_id_changes")
    audited_current = chain_compatibility_audit["current_input_set"]
    if (
        changed_targets != audited_current.get("runtime_incompatible_cases")
        or changed_chains
        != audited_current.get("runtime_incompatible_chains")
    ):
        raise ProspectiveRefreezeError("chain_audit_count_mismatch")
    if unchanged_files != len(retained_rows) - changed_targets:
        raise ProspectiveRefreezeError("unchanged_input_count_mismatch")
    task0_old_path = min(old_expected.values(), key=lambda path: path.name)
    task0_new_path = stage / task0_old_path.name
    task0_input_unchanged = (
        sha256_file(task0_old_path) == sha256_file(task0_new_path)
    )

    mapping_partial.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in mapping_manifest_rows
        )
    )
    mapping_partial.chmod(0o600)
    mapping_manifest_sha256 = sha256_file(mapping_partial)
    mapping_partial.rename(mapping_path)
    try:
        stage.rename(new_root)
    except Exception:
        mapping_path.rename(mapping_partial)
        raise

    af3_inputs = dict(old_input_freeze["af3_inputs"])
    af3_inputs.update(
        {
            "files": len(file_hashes),
            "af3_input_set_sha256": new_input_set_sha256,
            "sequence_set_sha256": sequence_set_sha256,
            "af3_chain_mapping_set_sha256": canonical_sha256(mapping_hashes),
            "private_chain_mapping_manifest_sha256": mapping_manifest_sha256,
            "private_chain_mapping_manifest_written": True,
            "remapped_targets": changed_targets,
            "remapped_chains": changed_chains,
            "unchanged_input_files": unchanged_files,
            "af3_chain_ids_runtime_compatible": True,
            "native_chain_ids_preserved_when_runtime_compatible": True,
            "native_or_af3_chain_ids_emitted_publicly": False,
        }
    )
    report = {
        **old_input_freeze,
        "schema_version": REFREEZE_SCHEMA,
        "amendment": {
            "kind": "append_only_pre_inference_runtime_compatibility_refreeze",
            "reason": "pinned_af3_chain_id_contract",
            "previous_input_freeze_sha256": old_input_freeze_sha256,
            "previous_af3_input_set_sha256": old_input_set_sha256,
            "chain_id_compatibility_audit_sha256": (
                chain_compatibility_audit_sha256
            ),
            "target_set_changed": False,
            "panel_roles_changed": False,
            "native_sequences_changed": False,
            "templates_or_model_seeds_changed": False,
            "input_dialect_or_version_changed": False,
            "native_chain_mapping_changed": False,
            "only_af3_runtime_chain_ids_changed": True,
            "prediction_or_inference_outputs_read": False,
            "data_pipeline_output_contract_read": True,
            "failed_array_runtime_error_read": True,
            "dockq_or_interface_labels_read": False,
        },
        "af3_inputs": af3_inputs,
        "refreeze_validation": {
            "old_input_set_checksum_matches": True,
            "retained_manifest_checksum_matches": True,
            "sequence_set_checksum_unchanged": True,
            "new_input_count_exact": len(file_hashes) == len(retained_rows),
            "all_new_chain_ids_runtime_compatible": True,
            "all_new_chain_ids_unique_per_target": True,
            "private_mapping_manifest_checksum_recorded": True,
            "task0_input_bytes_unchanged": task0_input_unchanged,
            "raw_sequences_or_chain_ids_emitted_publicly": False,
        },
        "decision": {
            **old_input_freeze["decision"],
            "ready_for_af3_prediction": True,
            "refreeze_validated": True,
            "ready_for_model_training": False,
            "ready_for_dpo_rlvr": False,
        },
        "release_boundary": {
            "target_identifiers_emitted": False,
            "native_or_af3_chain_identifiers_emitted": False,
            "sequence_or_structure_content_emitted": False,
            "label_content_emitted": False,
            "local_paths_emitted": False,
            "scheduler_identifiers_emitted": False,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-input-dir", type=Path, required=True)
    parser.add_argument("--new-input-dir", type=Path, required=True)
    parser.add_argument("--private-chain-mapping-out", type=Path, required=True)
    parser.add_argument("--old-input-freeze", type=Path, required=True)
    parser.add_argument("--chain-compatibility-audit", type=Path, required=True)
    parser.add_argument("--retained-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()
    report = refreeze_af3_inputs(
        old_input_dir=args.old_input_dir,
        new_input_dir=args.new_input_dir,
        private_chain_mapping_out=args.private_chain_mapping_out,
        old_input_freeze=_load_object(args.old_input_freeze),
        old_input_freeze_sha256=sha256_file(args.old_input_freeze),
        chain_compatibility_audit=_load_object(
            args.chain_compatibility_audit
        ),
        chain_compatibility_audit_sha256=sha256_file(
            args.chain_compatibility_audit
        ),
        retained_rows=load_c5_manifest(args.retained_manifest),
        preregistration=_load_object(args.preregistration),
    )
    write_json(args.audit_out, report)
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "files": report["af3_inputs"]["files"],
                "remapped_targets": report["af3_inputs"]["remapped_targets"],
                "remapped_chains": report["af3_inputs"]["remapped_chains"],
                "ready_for_af3_prediction": report["decision"][
                    "ready_for_af3_prediction"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
