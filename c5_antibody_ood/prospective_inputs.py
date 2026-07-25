"""Private structure QC and AlphaFold 3 input freeze for prospective C5."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import load_c5_manifest, write_c5_manifest
from .prospective_panel import (
    canonical_sha256,
    validate_preregistration,
    validate_public_panel,
    write_json,
)
from .prospective_source import blocked_pdb_ids_from_manifests
from .source_pilot import sha256_file


STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
PRIMARY_ROLES = ("calibration", "evaluation")
RESERVE_ROLE = {
    "calibration": "calibration_reserve",
    "evaluation": "evaluation_reserve",
}


class ProspectiveInputError(ValueError):
    """Raised when native QC or input freezing cannot satisfy the protocol."""


@dataclass(frozen=True)
class StructureQC:
    target_id: str
    panel_role: str
    passed: bool
    issues: tuple[str, ...]
    structure_sha256: str | None
    chain_sequences: tuple[tuple[str, str], ...]


def inspect_structures(
    rows: Sequence[Mapping[str, Any]],
    structures_dir: str | Path,
) -> list[StructureQC]:
    """Inspect committed chains without calculating DockQ or interface labels."""

    try:
        import gemmi
    except ImportError as exc:
        raise ProspectiveInputError(
            "gemmi_required_for_structure_qc"
        ) from exc

    root = Path(structures_dir)
    results: list[StructureQC] = []
    for row in rows:
        path = root / f"{row['instance_id']}.cif"
        issues: list[str] = []
        sequences: list[tuple[str, str]] = []
        structure_hash: str | None = None
        if not path.is_file():
            issues.append("structure_missing")
        else:
            structure_hash = sha256_file(path)
            try:
                structure = gemmi.read_structure(str(path))
                model = structure[0]
            except Exception:
                issues.append("structure_parse_failed")
            else:
                for mapping in row["chain_role_mapping"]:
                    chain_id = str(mapping["chain_id"])
                    chain = model.find_chain(chain_id)
                    if chain is None:
                        issues.append("committed_chain_missing")
                        continue
                    sequence = (
                        chain.get_polymer()
                        .make_one_letter_sequence()
                        .replace("-", "")
                        .upper()
                    )
                    if not sequence:
                        issues.append("polymer_sequence_empty")
                        continue
                    if set(sequence) - STANDARD_AMINO_ACIDS:
                        issues.append("nonstandard_amino_acid")
                        continue
                    sequences.append((chain_id, sequence))
        if len(sequences) != len(row["chain_role_mapping"]):
            issues.append("chain_sequence_count_mismatch")
        results.append(
            StructureQC(
                target_id=str(row["target_id"]),
                panel_role=str(row["panel_role"]),
                passed=not issues,
                issues=tuple(sorted(set(issues))),
                structure_sha256=structure_hash,
                chain_sequences=tuple(sequences),
            )
        )
    return results


def retain_primary_or_promote_reserve(
    rows: Sequence[Mapping[str, Any]],
    qc_results: Sequence[StructureQC],
    preregistration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retain passing primaries and promote reserves without label access."""

    qc_by_target = {result.target_id: result for result in qc_results}
    if len(qc_by_target) != len(qc_results):
        raise ProspectiveInputError("duplicate_qc_target")
    if set(qc_by_target) != {str(row["target_id"]) for row in rows}:
        raise ProspectiveInputError("qc_target_set_mismatch")

    retained: list[dict[str, Any]] = []
    promotions: Counter[str] = Counter()
    failed_primary: Counter[str] = Counter()
    failed_reserve: Counter[str] = Counter()
    selection = preregistration["protocol"]["target_selection"]
    for primary_role in PRIMARY_ROLES:
        required = int(selection[primary_role]["primary_targets"])
        primaries = sorted(
            (
                row
                for row in rows
                if row["panel_role"] == primary_role
            ),
            key=lambda row: int(row["selection_rank"]),
        )
        reserves = sorted(
            (
                row
                for row in rows
                if row["panel_role"] == RESERVE_ROLE[primary_role]
            ),
            key=lambda row: int(row["selection_rank"]),
        )
        accepted = [
            row
            for row in primaries
            if qc_by_target[str(row["target_id"])].passed
        ]
        failed_primary[primary_role] = len(primaries) - len(accepted)
        for row in reserves:
            if len(accepted) == required:
                break
            if qc_by_target[str(row["target_id"])].passed:
                accepted.append(row)
                promotions[primary_role] += 1
            else:
                failed_reserve[primary_role] += 1
        if len(accepted) != required:
            raise ProspectiveInputError(
                f"structure_qc_insufficient_{primary_role}:"
                f"{len(accepted)}<{required}"
            )
        for rank, row in enumerate(accepted, start=1):
            retained_row = dict(row)
            retained_row["panel_role"] = primary_role
            retained_row["selection_rank"] = rank
            retained.append(retained_row)

    return retained, {
        "failed_primary_by_role": dict(sorted(failed_primary.items())),
        "failed_reserve_by_role": {
            role: failed_reserve[role] for role in PRIMARY_ROLES
        },
        "promotions_by_role": {
            role: promotions[role] for role in PRIMARY_ROLES
        },
        "retained_by_role": dict(
            sorted(Counter(row["panel_role"] for row in retained).items())
        ),
    }


def write_private_af3_inputs(
    retained_rows: Sequence[Mapping[str, Any]],
    qc_results: Sequence[StructureQC],
    preregistration: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write private template-free AF3 JSON and return hash commitments."""

    prediction = preregistration["protocol"]["prediction"]
    qc_by_target = {result.target_id: result for result in qc_results}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    file_hashes: dict[str, str] = {}
    sequence_hashes: dict[str, str] = {}
    for row in retained_rows:
        target_id = str(row["target_id"])
        qc = qc_by_target[target_id]
        if not qc.passed:
            raise ProspectiveInputError("retained_target_failed_qc")
        sequence_by_chain = dict(qc.chain_sequences)
        proteins = []
        for mapping in row["chain_role_mapping"]:
            chain_id = str(mapping["chain_id"])
            proteins.append(
                {
                    "protein": {
                        "id": chain_id,
                        "sequence": sequence_by_chain[chain_id],
                        "templates": [],
                    }
                }
            )
        af3_input = {
            "name": _safe_job_name(str(row["instance_id"])),
            "modelSeeds": prediction["model_seeds"],
            "sequences": proteins,
            "dialect": prediction["input_dialect"],
            "version": prediction["input_version"],
        }
        path = output / f"{af3_input['name']}.json"
        path.write_text(
            json.dumps(af3_input, indent=2, sort_keys=True) + "\n"
        )
        file_hashes[target_id] = sha256_file(path)
        sequence_hashes[target_id] = canonical_sha256(
            [
                {
                    "chain_id": chain_id,
                    "sequence_sha256": hashlib.sha256(
                        sequence.encode()
                    ).hexdigest(),
                }
                for chain_id, sequence in qc.chain_sequences
            ]
        )
    return {
        "files": len(file_hashes),
        "af3_input_set_sha256": canonical_sha256(file_hashes),
        "sequence_set_sha256": canonical_sha256(sequence_hashes),
        "raw_sequences_emitted_publicly": False,
        "templates_disabled": (
            prediction["templates"] == "disabled_for_every_protein_chain"
        ),
    }


def build_input_freeze(
    candidate_rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    *,
    blocked_pdb_ids: set[str],
    structures_dir: str | Path,
    private_input_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run pre-prediction QC and build a public-safe input commitment."""

    preregistration_issues = validate_preregistration(preregistration)
    if preregistration_issues:
        raise ProspectiveInputError(
            "preregistration_invalid:" + ",".join(preregistration_issues)
        )
    candidate_issues = validate_public_panel(
        candidate_rows,
        preregistration,
        blocked_pdb_ids=blocked_pdb_ids,
    )
    if candidate_issues:
        raise ProspectiveInputError(
            "candidate_manifest_invalid:" + ",".join(candidate_issues)
        )
    qc_results = inspect_structures(candidate_rows, structures_dir)
    retained, retention = retain_primary_or_promote_reserve(
        candidate_rows,
        qc_results,
        preregistration,
    )
    retained_issues = validate_public_panel(
        retained,
        preregistration,
        blocked_pdb_ids=blocked_pdb_ids,
        expected_role_counts={
            "calibration": 80,
            "evaluation": 40,
        },
    )
    if retained_issues:
        raise ProspectiveInputError(
            "retained_manifest_invalid:" + ",".join(retained_issues)
        )
    input_commitment = write_private_af3_inputs(
        retained,
        qc_results,
        preregistration,
        private_input_dir,
    )
    structure_hashes = {
        result.target_id: result.structure_sha256
        for result in qc_results
        if result.structure_sha256 is not None
    }
    issue_counts = Counter(
        issue for result in qc_results for issue in result.issues
    )
    passed_by_role = Counter(
        result.panel_role for result in qc_results if result.passed
    )
    audit = {
        "schema_version": "c5_prospective_af3_input_freeze_v1",
        "preregistration_id": preregistration["preregistration_id"],
        "protocol_sha256": preregistration["commitment"]["protocol_sha256"],
        "candidate_manifest": {
            "rows": len(candidate_rows),
            "sha256": canonical_sha256(list(candidate_rows)),
        },
        "structure_qc": {
            "checked": len(qc_results),
            "passed": sum(result.passed for result in qc_results),
            "passed_by_role": dict(sorted(passed_by_role.items())),
            "issues_by_reason": dict(sorted(issue_counts.items())),
            "native_structure_set_sha256": canonical_sha256(
                structure_hashes
            ),
            "dockq_or_interface_labels_read": False,
            "raw_structures_or_sequences_emitted_publicly": False,
        },
        "retention": {
            **retention,
            "rows": len(retained),
            "manifest_sha256": canonical_sha256(retained),
            "validation_issues": retained_issues,
        },
        "af3_inputs": input_commitment,
        "decision": {
            "ready_for_af3_prediction": (
                not retained_issues
                and len(retained) == 120
                and input_commitment["files"] == 120
            ),
            "external_specialist_trust_enabled": False,
            "ready_for_model_training": False,
            "ready_for_dpo_rlvr": False,
        },
    }
    return retained, audit


def _safe_job_name(instance_id: str) -> str:
    value = instance_id.lower()
    if not re.fullmatch(r"[a-z0-9_]+", value):
        raise ProspectiveInputError("instance_id_not_safe_for_job_name")
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--fromm-manifest", type=Path, required=True)
    parser.add_argument("--gray-manifest", type=Path, required=True)
    parser.add_argument("--structures-dir", type=Path, required=True)
    parser.add_argument("--private-input-dir", type=Path, required=True)
    parser.add_argument("--retained-manifest-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    preregistration = _load_json(args.preregistration)
    candidate_rows = load_c5_manifest(args.candidate_manifest)
    blocked = blocked_pdb_ids_from_manifests(
        args.fromm_manifest,
        args.gray_manifest,
    )
    retained, audit = build_input_freeze(
        candidate_rows,
        preregistration,
        blocked_pdb_ids=blocked,
        structures_dir=args.structures_dir,
        private_input_dir=args.private_input_dir,
    )
    write_c5_manifest(args.retained_manifest_out, retained)
    write_json(args.audit_out, audit)
    print(
        json.dumps(
            {
                "retained_rows": len(retained),
                "ready_for_af3_prediction": audit["decision"][
                    "ready_for_af3_prediction"
                ],
                "structure_qc_issues": audit["structure_qc"][
                    "issues_by_reason"
                ],
            },
            sort_keys=True,
        )
    )
    return int(not audit["decision"]["ready_for_af3_prediction"])


if __name__ == "__main__":
    raise SystemExit(main())
