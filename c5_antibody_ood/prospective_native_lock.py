"""Reconstruct the private native-structure map committed before C5 labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import load_c5_manifest
from .prospective_inputs import StructureQC, inspect_structures
from .prospective_panel import SHA256_RE, canonical_sha256, write_json


NATIVE_LOCK_SCHEMA = "c5_prospective_native_structure_lock_v1"


class ProspectiveNativeLockError(ValueError):
    """Raised when native structures drift from the input-freeze commitment."""


def build_native_structure_lock(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    retained_rows: Sequence[Mapping[str, Any]],
    input_freeze: Mapping[str, Any],
    structures_dir: str | Path,
) -> dict[str, Any]:
    """Re-run label-free structure QC and recover target-specific hashes."""

    qc_results = inspect_structures(candidate_rows, structures_dir)
    return build_native_structure_lock_from_qc(
        candidate_rows=candidate_rows,
        retained_rows=retained_rows,
        input_freeze=input_freeze,
        qc_results=qc_results,
    )


def build_native_structure_lock_from_qc(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    retained_rows: Sequence[Mapping[str, Any]],
    input_freeze: Mapping[str, Any],
    qc_results: Sequence[StructureQC],
) -> dict[str, Any]:
    """Build the private map from already inspected structures."""

    rows_by_id = {
        str(row["target_id"]): row for row in candidate_rows
    }
    qc_by_id = {result.target_id: result for result in qc_results}
    if (
        len(rows_by_id) != len(candidate_rows)
        or len(qc_by_id) != len(qc_results)
        or set(rows_by_id) != set(qc_by_id)
    ):
        raise ProspectiveNativeLockError("candidate_qc_target_set_mismatch")
    if any(result.structure_sha256 is None for result in qc_results):
        raise ProspectiveNativeLockError("native_structure_checksum_missing")

    structure_hashes = {
        target_id: str(qc_by_id[target_id].structure_sha256)
        for target_id in sorted(qc_by_id)
    }
    set_sha256 = canonical_sha256(structure_hashes)
    expected_qc = input_freeze.get("structure_qc", {})
    if set_sha256 != expected_qc.get("native_structure_set_sha256"):
        raise ProspectiveNativeLockError("native_structure_set_sha256_mismatch")
    issue_counts = Counter(
        issue for result in qc_results for issue in result.issues
    )
    passed_by_role = Counter(
        result.panel_role for result in qc_results if result.passed
    )
    expected_counts = {
        "checked": len(qc_results),
        "passed": sum(result.passed for result in qc_results),
        "passed_by_role": dict(sorted(passed_by_role.items())),
        "issues_by_reason": dict(sorted(issue_counts.items())),
    }
    for key, actual in expected_counts.items():
        if expected_qc.get(key) != actual:
            raise ProspectiveNativeLockError(f"structure_qc_{key}_mismatch")

    targets = [
        {
            "target_id": target_id,
            "panel_role": str(rows_by_id[target_id]["panel_role"]),
            "native_structure_sha256": structure_hashes[target_id],
            "chain_role_mapping_sha256": canonical_sha256(
                rows_by_id[target_id]["chain_role_mapping"]
            ),
            "qc_passed": qc_by_id[target_id].passed,
            "qc_issues": list(qc_by_id[target_id].issues),
        }
        for target_id in sorted(rows_by_id)
    ]
    retained_ids = {str(row["target_id"]) for row in retained_rows}
    retained_entries = [
        target for target in targets if target["target_id"] in retained_ids
    ]
    if (
        len(retained_entries) != len(retained_ids)
        or any(not target["qc_passed"] for target in retained_entries)
    ):
        raise ProspectiveNativeLockError("retained_native_structure_not_ready")
    lock = {
        "schema_version": NATIVE_LOCK_SCHEMA,
        "preregistration_id": input_freeze["preregistration_id"],
        "protocol_sha256": input_freeze["protocol_sha256"],
        "native_structure_set_sha256": set_sha256,
        "counts": {
            "candidate_targets": len(targets),
            "retained_targets": len(retained_entries),
            "qc_passed": sum(target["qc_passed"] for target in targets),
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
    issues = validate_native_structure_lock(
        lock,
        retained_rows=retained_rows,
        input_freeze=input_freeze,
    )
    if issues:
        raise ProspectiveNativeLockError(
            "native_structure_lock_invalid:" + ",".join(issues)
        )
    return lock


def validate_native_structure_lock(
    lock: Mapping[str, Any],
    *,
    retained_rows: Sequence[Mapping[str, Any]],
    input_freeze: Mapping[str, Any],
) -> list[str]:
    """Validate target hashes against the pre-label aggregate commitment."""

    issues: list[str] = []
    if set(lock) != {
        "schema_version",
        "preregistration_id",
        "protocol_sha256",
        "native_structure_set_sha256",
        "counts",
        "targets",
        "evidence_boundary",
        "decision",
    }:
        issues.append("native_structure_lock_schema_invalid")
    if lock.get("schema_version") != NATIVE_LOCK_SCHEMA:
        issues.append("schema_version_invalid")
    if lock.get("preregistration_id") != input_freeze.get(
        "preregistration_id"
    ):
        issues.append("preregistration_id_mismatch")
    if lock.get("protocol_sha256") != input_freeze.get("protocol_sha256"):
        issues.append("protocol_sha256_mismatch")
    targets = lock.get("targets")
    if not isinstance(targets, list):
        return [*issues, "targets_invalid"]
    target_ids = [target.get("target_id") for target in targets]
    if len(target_ids) != len(set(target_ids)):
        issues.append("target_id_duplicate")
    expected_checked = input_freeze.get("structure_qc", {}).get("checked")
    if len(targets) != expected_checked:
        issues.append("candidate_target_count_mismatch")
    if lock.get("counts", {}).get("candidate_targets") != len(targets):
        issues.append("candidate_count_record_mismatch")
    try:
        structure_hashes = {
            str(target["target_id"]): str(
                target["native_structure_sha256"]
            )
            for target in targets
        }
    except (KeyError, TypeError):
        issues.append("target_structure_commitment_invalid")
        structure_hashes = {}
    if any(
        not SHA256_RE.fullmatch(value) for value in structure_hashes.values()
    ):
        issues.append("native_structure_sha256_invalid")
    recomputed = canonical_sha256(structure_hashes)
    expected = input_freeze.get("structure_qc", {}).get(
        "native_structure_set_sha256"
    )
    if recomputed != expected or lock.get(
        "native_structure_set_sha256"
    ) != expected:
        issues.append("native_structure_set_sha256_mismatch")
    targets_by_id = {
        str(target.get("target_id")): target for target in targets
    }
    for target in targets:
        if set(target) != {
            "target_id",
            "panel_role",
            "native_structure_sha256",
            "chain_role_mapping_sha256",
            "qc_passed",
            "qc_issues",
        }:
            issues.append("native_target_schema_invalid")
        if not SHA256_RE.fullmatch(
            str(target.get("chain_role_mapping_sha256", ""))
        ):
            issues.append("chain_role_mapping_sha256_invalid")
        if not isinstance(target.get("qc_passed"), bool):
            issues.append("qc_passed_invalid")
        if not isinstance(target.get("qc_issues"), list):
            issues.append("qc_issues_invalid")
    for row in retained_rows:
        target = targets_by_id.get(str(row["target_id"]))
        if target is None:
            issues.append("retained_target_missing")
            continue
        if target.get("qc_passed") is not True:
            issues.append("retained_target_failed_qc")
        if target.get("chain_role_mapping_sha256") != canonical_sha256(
            row["chain_role_mapping"]
        ):
            issues.append("retained_chain_role_mapping_mismatch")
    boundary = lock.get("evidence_boundary", {})
    if boundary.get("dockq_or_interface_labels_read") is not False:
        issues.append("label_boundary_invalid")
    if lock.get("decision", {}).get(
        "ready_for_staged_label_reveal"
    ) is not True:
        issues.append("label_reveal_gate_invalid")
    if lock.get("counts", {}).get("retained_targets") != len(retained_rows):
        issues.append("retained_count_record_mismatch")
    return sorted(set(issues))


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ProspectiveNativeLockError("expected_json_object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--retained-manifest", type=Path, required=True)
    parser.add_argument("--input-freeze", type=Path, required=True)
    parser.add_argument("--structures-dir", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    args = parser.parse_args()

    lock = build_native_structure_lock(
        candidate_rows=load_c5_manifest(args.candidate_manifest),
        retained_rows=load_c5_manifest(args.retained_manifest),
        input_freeze=_load_json(args.input_freeze),
        structures_dir=args.structures_dir,
    )
    write_json(args.private_out, lock)
    print(
        json.dumps(
            {
                "candidate_targets": lock["counts"]["candidate_targets"],
                "retained_targets": lock["counts"]["retained_targets"],
                "ready_for_staged_label_reveal": lock["decision"][
                    "ready_for_staged_label_reveal"
                ],
                "native_structure_set_sha256": lock[
                    "native_structure_set_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
