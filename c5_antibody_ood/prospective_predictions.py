"""Fail-closed AF3 output intake and prediction freeze for prospective C5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .af3_preflight import verify_attestation
from .manifest import load_c5_manifest
from .prospective_inputs import _safe_job_name
from .prospective_panel import (
    SHA256_RE,
    canonical_sha256,
    validate_preregistration,
    write_json,
)
from .source_pilot import sha256_file


PRIVATE_LOCK_SCHEMA = "c5_prospective_af3_prediction_lock_v1"
PUBLIC_FREEZE_SCHEMA = "c5_prospective_af3_prediction_freeze_v1"
EXPECTED_WORKFLOW_STATE = "panel_locked_prediction_pending"
LOCKED_WORKFLOW_STATE = "predictions_locked_label_reveal_pending"
SUMMARY_KEYS = frozenset(
    {
        "ptm",
        "iptm",
        "ranking_score",
        "fraction_disordered",
        "has_clash",
        "chain_pair_pae_min",
        "chain_pair_iptm",
        "chain_ptm",
        "chain_iptm",
    }
)
RANKING_COLUMNS = ("seed", "sample", "ranking_score")
FORBIDDEN_LABEL_KEY_FRAGMENTS = (
    "dockq",
    "interface_label",
    "native_interface",
    "ground_truth",
)


class ProspectivePredictionError(ValueError):
    """Raised when AF3 outputs cannot satisfy the frozen prediction contract."""


def collect_af3_predictions(
    retained_rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    output_root: str | Path,
) -> list[dict[str, Any]]:
    """Validate every official AF3 sample and return a private target lock."""

    root = Path(output_root)
    if not root.is_dir():
        raise ProspectivePredictionError("af3_output_root_missing")
    if root.is_symlink():
        raise ProspectivePredictionError("af3_output_root_symlink_forbidden")

    expected_jobs: dict[str, Mapping[str, Any]] = {}
    for row in retained_rows:
        job_name = _safe_job_name(str(row["instance_id"]))
        if job_name in expected_jobs:
            raise ProspectivePredictionError("duplicate_af3_job_name")
        expected_jobs[job_name] = row
    root_entries = list(root.iterdir())
    if any(path.is_symlink() for path in root_entries):
        raise ProspectivePredictionError("af3_output_root_contains_symlink")
    present_jobs = {path.name for path in root_entries if path.is_dir()}
    if present_jobs != set(expected_jobs):
        raise ProspectivePredictionError("af3_job_directory_set_mismatch")
    unexpected_files = [path for path in root_entries if not path.is_dir()]
    if unexpected_files:
        raise ProspectivePredictionError("af3_output_root_contains_files")

    targets = [
        collect_af3_target_output(
            row=expected_jobs[job_name],
            preregistration=preregistration,
            job_dir=root / job_name,
        )
        for job_name in sorted(expected_jobs)
    ]
    if len(targets) != len(retained_rows):
        raise ProspectivePredictionError("af3_target_count_mismatch")
    return targets


def collect_af3_target_output(
    *,
    row: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    job_dir: str | Path,
) -> dict[str, Any]:
    """Validate one target using the canonical full-output intake contract."""

    prediction = preregistration["protocol"]["prediction"]
    expected_sample_keys = {
        (int(seed), sample)
        for seed in prediction["model_seeds"]
        for sample in range(int(prediction["diffusion_samples_per_seed"]))
    }
    job_name = _safe_job_name(str(row["instance_id"]))
    path = Path(job_dir)
    if path.name != job_name:
        raise ProspectivePredictionError("af3_job_directory_name_mismatch")
    if not path.is_dir() or path.is_symlink():
        raise ProspectivePredictionError("af3_job_directory_invalid")
    return _collect_target(
        row=row,
        job_name=job_name,
        job_dir=path,
        expected_sample_keys=expected_sample_keys,
    )


def build_prediction_lock(
    *,
    preregistration: Mapping[str, Any],
    input_freeze: Mapping[str, Any],
    retained_rows: Sequence[Mapping[str, Any]],
    retained_manifest_path: str | Path,
    private_input_dir: str | Path,
    attestation_path: str | Path,
    expected_attestation_sha256: str,
    output_root: str | Path,
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind complete AF3 outputs to the pre-label method and input locks."""

    issues = validate_preregistration(preregistration)
    if issues:
        raise ProspectivePredictionError(
            "preregistration_invalid:" + ",".join(issues)
        )
    if preregistration.get("workflow_state") != EXPECTED_WORKFLOW_STATE:
        raise ProspectivePredictionError("preregistration_workflow_state_invalid")
    protocol_sha256 = preregistration["commitment"]["protocol_sha256"]
    if input_freeze.get("preregistration_id") != preregistration.get(
        "preregistration_id"
    ):
        raise ProspectivePredictionError("input_freeze_preregistration_mismatch")
    if input_freeze.get("protocol_sha256") != protocol_sha256:
        raise ProspectivePredictionError("input_freeze_protocol_mismatch")
    if input_freeze.get("decision", {}).get(
        "ready_for_af3_prediction"
    ) is not True:
        raise ProspectivePredictionError("input_freeze_not_ready")
    if input_freeze.get("structure_qc", {}).get(
        "dockq_or_interface_labels_read"
    ) is not False:
        raise ProspectivePredictionError("input_freeze_label_boundary_invalid")

    retained_hash = canonical_sha256(list(retained_rows))
    if input_freeze.get("retention", {}).get(
        "manifest_sha256"
    ) != retained_hash:
        raise ProspectivePredictionError("retained_manifest_checksum_mismatch")
    expected_count = int(input_freeze["retention"]["rows"])
    if len(retained_rows) != expected_count:
        raise ProspectivePredictionError("retained_manifest_count_mismatch")
    expected_roles = {
        "calibration": int(
            preregistration["protocol"]["stopping_rule"][
                "required_calibration_targets"
            ]
        ),
        "evaluation": int(
            preregistration["protocol"]["stopping_rule"][
                "required_evaluation_targets"
            ]
        ),
    }
    actual_roles = dict(
        sorted(Counter(str(row["panel_role"]) for row in retained_rows).items())
    )
    if actual_roles != expected_roles:
        raise ProspectivePredictionError("retained_manifest_role_count_mismatch")

    try:
        attestation = verify_attestation(
            attestation_path=attestation_path,
            expected_attestation_sha256=expected_attestation_sha256,
            preregistration=preregistration,
            input_freeze=input_freeze,
            retained_manifest=retained_manifest_path,
            input_dir=private_input_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProspectivePredictionError(
            f"run_attestation_invalid:{exc}"
        ) from exc

    targets = collect_af3_predictions(
        retained_rows,
        preregistration,
        output_root,
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
    selected_records = [_selected_commitment(target) for target in targets]
    selected_prediction_set_sha256 = canonical_sha256(selected_records)
    samples_per_target = (
        len(preregistration["protocol"]["prediction"]["model_seeds"])
        * preregistration["protocol"]["prediction"][
            "diffusion_samples_per_seed"
        ]
    )
    timestamp = created_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    private_lock = {
        "schema_version": PRIVATE_LOCK_SCHEMA,
        "created_at_utc": timestamp,
        "workflow_state": LOCKED_WORKFLOW_STATE,
        "preregistration_id": preregistration["preregistration_id"],
        "protocol_sha256": protocol_sha256,
        "input_commitments": {
            "retained_manifest_sha256": retained_hash,
            "af3_input_set_sha256": input_freeze["af3_inputs"][
                "af3_input_set_sha256"
            ],
            "run_attestation_sha256": attestation["attestation_sha256"],
        },
        "selection": {
            "confidence_metric": "ranking_score",
            "rule": list(
                preregistration["protocol"]["prediction"][
                    "target_sample_selection"
                ]
            ),
            "lexical_output_id_order": "ascending",
            "ranking_score_source": "official_ranking_scores_csv",
            "iptm_tie_break_source": "official_summary_confidences_json",
        },
        "counts": {
            "targets": len(targets),
            "targets_by_role": actual_roles,
            "samples_per_target": samples_per_target,
            "samples": sum(len(target["samples"]) for target in targets),
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
            "external_specialist_trust_enabled": False,
            "ready_for_model_training": False,
            "ready_for_dpo_rlvr": False,
        },
    }
    lock_issues = validate_prediction_lock(
        private_lock,
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_rows=retained_rows,
    )
    if lock_issues:
        raise ProspectivePredictionError(
            "prediction_lock_invalid:" + ",".join(lock_issues)
        )
    return private_lock, public_prediction_freeze(private_lock)


def public_prediction_freeze(
    private_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the private target-level lock into a public-safe aggregate."""

    counts = dict(private_lock["counts"])
    return {
        "schema_version": PUBLIC_FREEZE_SCHEMA,
        "created_at_utc": private_lock["created_at_utc"],
        "workflow_state": private_lock["workflow_state"],
        "preregistration_id": private_lock["preregistration_id"],
        "protocol_sha256": private_lock["protocol_sha256"],
        "input_commitments": dict(private_lock["input_commitments"]),
        "selection": dict(private_lock["selection"]),
        "counts": counts,
        "commitments": dict(private_lock["commitments"]),
        "validation": {
            "af3_output_contract": "alphafold3_v3.0.3",
            "complete_target_set": True,
            "complete_five_sample_set_per_target": (
                counts["samples_per_target"] == 5
                and counts["samples"] == 5 * counts["targets"]
            ),
            "deterministic_target_selection": True,
            "all_prediction_files_checksum_frozen": True,
        },
        "evidence_boundary": dict(private_lock["evidence_boundary"]),
        "release_boundary": {
            "target_ids_emitted": False,
            "per_target_confidence_emitted": False,
            "prediction_paths_emitted": False,
            "structure_or_sequence_content_emitted": False,
            "per_target_labels_emitted": False,
        },
        "decision": dict(private_lock["decision"]),
    }


def validate_prediction_lock(
    lock: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    input_freeze: Mapping[str, Any],
    retained_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Validate a private prediction lock without reopening prediction files."""

    issues: list[str] = []
    if set(lock) != {
        "schema_version",
        "created_at_utc",
        "workflow_state",
        "preregistration_id",
        "protocol_sha256",
        "input_commitments",
        "selection",
        "counts",
        "commitments",
        "targets",
        "evidence_boundary",
        "decision",
    }:
        issues.append("prediction_lock_schema_invalid")
    if lock.get("schema_version") != PRIVATE_LOCK_SCHEMA:
        issues.append("schema_version_invalid")
    if lock.get("workflow_state") != LOCKED_WORKFLOW_STATE:
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
        "af3_input_set_sha256"
    ) != input_freeze.get("af3_inputs", {}).get("af3_input_set_sha256"):
        issues.append("af3_input_set_sha256_mismatch")
    if not SHA256_RE.fullmatch(
        str(
            lock.get("input_commitments", {}).get(
                "run_attestation_sha256", ""
            )
        )
    ):
        issues.append("run_attestation_sha256_invalid")

    targets = lock.get("targets")
    if not isinstance(targets, list):
        return [*issues, "targets_invalid"]
    expected_by_id = {
        str(row["target_id"]): row for row in retained_rows
    }
    prediction = preregistration["protocol"]["prediction"]
    expected_sample_keys = {
        (int(seed), sample_index)
        for seed in prediction["model_seeds"]
        for sample_index in range(
            int(prediction["diffusion_samples_per_seed"])
        )
    }
    if {target.get("target_id") for target in targets} != set(expected_by_id):
        issues.append("target_set_mismatch")
    if len(targets) != len(expected_by_id):
        issues.append("target_count_mismatch")
    for target in targets:
        if set(target) != {
            "target_id",
            "instance_id",
            "panel_role",
            "job_name",
            "samples",
            "selected_output_id",
            "selected_model_relative_path",
            "job_artifact_set_sha256",
        }:
            issues.append("target_schema_invalid")
        target_id = target.get("target_id")
        expected = expected_by_id.get(str(target_id))
        if expected is None:
            continue
        if target.get("panel_role") != expected["panel_role"]:
            issues.append("target_role_mismatch")
        try:
            expected_job_name = _safe_job_name(str(expected["instance_id"]))
        except (ValueError, TypeError):
            issues.append("target_job_name_invalid")
        else:
            if target.get("job_name") != expected_job_name:
                issues.append("target_job_name_mismatch")
        if not SHA256_RE.fullmatch(
            str(target.get("job_artifact_set_sha256", ""))
        ):
            issues.append("job_artifact_set_sha256_invalid")
        samples = target.get("samples")
        if (
            not isinstance(samples, list)
            or len(samples) != len(expected_sample_keys)
        ):
            issues.append("sample_count_invalid")
            continue
        sample_keys: set[tuple[int, int]] = set()
        output_ids: set[str] = set()
        for sample in samples:
            if set(sample) != {
                "output_id",
                "seed",
                "sample_index",
                "ranking_score",
                "iptm",
                "ptm",
                "fraction_disordered",
                "has_clash",
                "artifacts",
            }:
                issues.append("sample_schema_invalid")
                continue
            try:
                seed = int(sample["seed"])
                sample_index = int(sample["sample_index"])
                output_id = str(sample["output_id"])
                ranking_score = float(sample["ranking_score"])
                iptm = float(sample["iptm"])
            except (KeyError, TypeError, ValueError):
                issues.append("sample_schema_invalid")
                continue
            if (
                not isinstance(sample.get("seed"), int)
                or isinstance(sample.get("seed"), bool)
                or not isinstance(sample.get("sample_index"), int)
                or isinstance(sample.get("sample_index"), bool)
            ):
                issues.append("sample_index_invalid")
            if output_id != f"seed-{seed}_sample-{sample_index}":
                issues.append("sample_output_id_mismatch")
            sample_keys.add((seed, sample_index))
            output_ids.add(output_id)
            if (
                not math.isfinite(ranking_score)
                or not -100.0 <= ranking_score <= 1.5
            ):
                issues.append("sample_ranking_score_invalid")
            if not math.isfinite(iptm) or not 0.0 <= iptm <= 1.0:
                issues.append("sample_iptm_invalid")
            artifacts = sample.get("artifacts")
            if not isinstance(artifacts, Mapping) or set(artifacts) != {
                "model_cif",
                "confidences_json",
                "summary_confidences_json",
            }:
                issues.append("sample_artifact_schema_invalid")
                continue
            for artifact in artifacts.values():
                if not isinstance(artifact, Mapping):
                    issues.append("sample_artifact_invalid")
                    continue
                if set(artifact) != {"relative_path", "bytes", "sha256"}:
                    issues.append("sample_artifact_schema_invalid")
                if not SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
                    issues.append("sample_artifact_sha256_invalid")
                relative = artifact.get("relative_path")
                if (
                    not isinstance(relative, str)
                    or not relative
                    or PurePosixPath(relative).is_absolute()
                    or ".." in PurePosixPath(relative).parts
                ):
                    issues.append("sample_artifact_relative_path_invalid")
                size = artifact.get("bytes")
                if (
                    not isinstance(size, int)
                    or isinstance(size, bool)
                    or size <= 0
                ):
                    issues.append("sample_artifact_bytes_invalid")
        if sample_keys != expected_sample_keys:
            issues.append("sample_seed_index_set_mismatch")
        if len(output_ids) != len(samples):
            issues.append("sample_output_id_duplicate")
        selected_id = target.get("selected_output_id")
        selected = [
            sample
            for sample in samples
            if sample.get("output_id") == selected_id
        ]
        if len(selected) != 1:
            issues.append("selected_output_invalid")
            continue
        try:
            expected_selected = min(
                samples,
                key=lambda sample: (
                    -float(sample["ranking_score"]),
                    -float(sample["iptm"]),
                    str(sample["output_id"]),
                ),
            )
        except (KeyError, TypeError, ValueError):
            issues.append("selected_output_rule_unverifiable")
        else:
            if selected_id != expected_selected["output_id"]:
                issues.append("selected_output_rule_mismatch")
        if _selected_commitment(target).get(
            "model_cif_sha256"
        ) != selected[0].get("artifacts", {}).get("model_cif", {}).get(
            "sha256"
        ):
            issues.append("selected_model_checksum_mismatch")
        if target.get("selected_model_relative_path") != selected[0].get(
            "artifacts", {}
        ).get("model_cif", {}).get("relative_path"):
            issues.append("selected_model_relative_path_mismatch")

    try:
        expected_artifact_set = canonical_sha256(
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
    except (KeyError, TypeError):
        issues.append("prediction_artifact_commitment_invalid")
    else:
        if lock.get("commitments", {}).get(
            "prediction_artifact_set_sha256"
        ) != expected_artifact_set:
            issues.append("prediction_artifact_set_sha256_mismatch")
    try:
        expected_selected_set = canonical_sha256(
            [_selected_commitment(target) for target in targets]
        )
    except (KeyError, TypeError):
        issues.append("selected_prediction_commitment_invalid")
    else:
        if lock.get("commitments", {}).get(
            "selected_prediction_set_sha256"
        ) != expected_selected_set:
            issues.append("selected_prediction_set_sha256_mismatch")
    if _contains_forbidden_label_key(lock.get("targets", [])):
        issues.append("prediction_lock_contains_label_key")
    boundary = lock.get("evidence_boundary", {})
    if boundary.get("dockq_or_native_interface_labels_read") is not False:
        issues.append("label_boundary_invalid")
    if lock.get("decision", {}).get(
        "ready_for_calibration_label_reveal"
    ) is not True:
        issues.append("calibration_reveal_gate_invalid")
    if lock.get("decision", {}).get(
        "ready_for_evaluation_label_reveal"
    ) is not False:
        issues.append("evaluation_reveal_gate_invalid")
    return sorted(set(issues))


def _collect_target(
    *,
    row: Mapping[str, Any],
    job_name: str,
    job_dir: Path,
    expected_sample_keys: set[tuple[int, int]],
) -> dict[str, Any]:
    expected_sample_dirs = {
        f"seed-{seed}_sample-{sample}"
        for seed, sample in expected_sample_keys
    }
    expected_top_files = {
        f"{job_name}_model.cif",
        f"{job_name}_confidences.json",
        f"{job_name}_summary_confidences.json",
        f"{job_name}_data.json",
        f"{job_name}_ranking_scores.csv",
        "TERMS_OF_USE.md",
    }
    entry_list = list(job_dir.iterdir())
    if any(path.is_symlink() for path in entry_list):
        raise ProspectivePredictionError(
            f"af3_job_contains_symlink:{job_name}"
        )
    entries = set(entry_list)
    expected_entries = {
        *(job_dir / name for name in expected_sample_dirs),
        *(job_dir / name for name in expected_top_files),
    }
    if entries != expected_entries:
        raise ProspectivePredictionError(
            f"af3_job_entry_set_mismatch:{job_name}"
        )
    ranking_rows = _read_ranking_csv(
        job_dir / f"{job_name}_ranking_scores.csv",
        expected_sample_keys=expected_sample_keys,
    )
    chain_count = len(row["chain_role_mapping"])
    samples: list[dict[str, Any]] = []
    job_artifacts: list[dict[str, Any]] = []
    for seed, sample_index in sorted(expected_sample_keys):
        output_id = f"seed-{seed}_sample-{sample_index}"
        sample_dir = job_dir / output_id
        prefix = f"{job_name}_{output_id}"
        expected_files = {
            sample_dir / f"{prefix}_model.cif",
            sample_dir / f"{prefix}_confidences.json",
            sample_dir / f"{prefix}_summary_confidences.json",
        }
        sample_entries = list(sample_dir.iterdir())
        if any(path.is_symlink() for path in sample_entries):
            raise ProspectivePredictionError(
                f"af3_sample_contains_symlink:{job_name}:{output_id}"
            )
        if set(sample_entries) != expected_files:
            raise ProspectivePredictionError(
                f"af3_sample_file_set_mismatch:{job_name}:{output_id}"
            )
        summary_path = sample_dir / f"{prefix}_summary_confidences.json"
        summary = _read_summary(summary_path, chain_count=chain_count)
        ranking_score = ranking_rows[(seed, sample_index)]
        if abs(summary["ranking_score"] - ranking_score) > 0.005001:
            raise ProspectivePredictionError(
                f"ranking_score_summary_csv_mismatch:{job_name}:{output_id}"
            )
        artifacts = {
            "model_cif": _artifact(
                sample_dir / f"{prefix}_model.cif",
                root=job_dir.parent,
            ),
            "confidences_json": _artifact(
                sample_dir / f"{prefix}_confidences.json",
                root=job_dir.parent,
            ),
            "summary_confidences_json": _artifact(
                summary_path,
                root=job_dir.parent,
            ),
        }
        samples.append(
            {
                "output_id": output_id,
                "seed": seed,
                "sample_index": sample_index,
                "ranking_score": ranking_score,
                "iptm": summary["iptm"],
                "ptm": summary["ptm"],
                "fraction_disordered": summary["fraction_disordered"],
                "has_clash": summary["has_clash"],
                "artifacts": artifacts,
            }
        )
        job_artifacts.extend(artifacts.values())

    max_score = max(sample["ranking_score"] for sample in samples)
    top_summary = _read_summary(
        job_dir / f"{job_name}_summary_confidences.json",
        chain_count=chain_count,
    )
    if abs(top_summary["ranking_score"] - max_score) > 0.005001:
        raise ProspectivePredictionError(
            f"top_level_ranking_score_mismatch:{job_name}"
        )
    for name in sorted(expected_top_files):
        job_artifacts.append(_artifact(job_dir / name, root=job_dir.parent))
    selected = min(
        samples,
        key=lambda sample: (
            -sample["ranking_score"],
            -sample["iptm"],
            sample["output_id"],
        ),
    )
    return {
        "target_id": str(row["target_id"]),
        "instance_id": str(row["instance_id"]),
        "panel_role": str(row["panel_role"]),
        "job_name": job_name,
        "samples": samples,
        "selected_output_id": selected["output_id"],
        "selected_model_relative_path": selected["artifacts"]["model_cif"][
            "relative_path"
        ],
        "job_artifact_set_sha256": canonical_sha256(
            sorted(job_artifacts, key=lambda artifact: artifact["relative_path"])
        ),
    }


def _read_ranking_csv(
    path: Path,
    *,
    expected_sample_keys: set[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RANKING_COLUMNS:
                raise ProspectivePredictionError("ranking_csv_header_invalid")
            values: dict[tuple[int, int], float] = {}
            for row in reader:
                if set(row) != set(RANKING_COLUMNS):
                    raise ProspectivePredictionError(
                        "ranking_csv_columns_invalid"
                    )
                key = (int(row["seed"]), int(row["sample"]))
                score = float(row["ranking_score"])
                if key in values:
                    raise ProspectivePredictionError(
                        "ranking_csv_duplicate_sample"
                    )
                if not math.isfinite(score) or not -100.0 <= score <= 1.5:
                    raise ProspectivePredictionError(
                        "ranking_csv_score_invalid"
                    )
                values[key] = score
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, ProspectivePredictionError):
            raise
        raise ProspectivePredictionError("ranking_csv_parse_failed") from exc
    if set(values) != expected_sample_keys:
        raise ProspectivePredictionError("ranking_csv_sample_set_mismatch")
    return values


def _read_summary(path: Path, *, chain_count: int) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectivePredictionError("summary_json_parse_failed") from exc
    if not isinstance(value, dict) or set(value) != SUMMARY_KEYS:
        raise ProspectivePredictionError("summary_json_schema_invalid")
    normalized = {
        key: _finite_number(value[key], key)
        for key in ("ptm", "iptm", "ranking_score", "fraction_disordered")
    }
    for key in ("ptm", "iptm", "fraction_disordered"):
        if not 0.0 <= normalized[key] <= 1.0:
            raise ProspectivePredictionError(f"summary_{key}_outside_range")
    if not -100.0 <= normalized["ranking_score"] <= 1.5:
        raise ProspectivePredictionError("summary_ranking_score_outside_range")
    normalized["has_clash"] = _binary_clash(value["has_clash"])
    _validate_matrix(
        value["chain_pair_pae_min"],
        chain_count=chain_count,
        key="chain_pair_pae_min",
        minimum=0.0,
    )
    _validate_matrix(
        value["chain_pair_iptm"],
        chain_count=chain_count,
        key="chain_pair_iptm",
        minimum=0.0,
        maximum=1.0,
    )
    _validate_vector(
        value["chain_ptm"],
        chain_count=chain_count,
        key="chain_ptm",
    )
    _validate_vector(
        value["chain_iptm"],
        chain_count=chain_count,
        key="chain_iptm",
    )
    recomputed = (
        0.8 * normalized["iptm"]
        + 0.2 * normalized["ptm"]
        + 0.5 * normalized["fraction_disordered"]
        - 100.0 * int(normalized["has_clash"])
    )
    if abs(normalized["ranking_score"] - recomputed) > 0.015001:
        raise ProspectivePredictionError("summary_ranking_formula_mismatch")
    return normalized


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ProspectivePredictionError("af3_artifact_symlink_forbidden")
    if not path.is_file() or path.stat().st_size == 0:
        raise ProspectivePredictionError("af3_artifact_missing_or_empty")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProspectivePredictionError(
            "af3_artifact_outside_output_root"
        ) from exc
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _selected_commitment(target: Mapping[str, Any]) -> dict[str, Any]:
    selected_id = target.get("selected_output_id")
    selected = next(
        (
            sample
            for sample in target.get("samples", [])
            if sample.get("output_id") == selected_id
        ),
        {},
    )
    artifacts = selected.get("artifacts", {})
    return {
        "target_id": target.get("target_id"),
        "panel_role": target.get("panel_role"),
        "output_id": selected_id,
        "ranking_score": selected.get("ranking_score"),
        "iptm": selected.get("iptm"),
        "model_cif_sha256": artifacts.get("model_cif", {}).get("sha256"),
        "summary_confidences_sha256": artifacts.get(
            "summary_confidences_json", {}
        ).get("sha256"),
    }


def _finite_number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProspectivePredictionError(f"summary_{key}_not_numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ProspectivePredictionError(f"summary_{key}_not_finite")
    return normalized


def _binary_clash(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1, 0.0, 1.0}:
        return bool(value)
    raise ProspectivePredictionError("summary_has_clash_not_binary")


def _validate_matrix(
    value: Any,
    *,
    chain_count: int,
    key: str,
    minimum: float,
    maximum: float | None = None,
) -> None:
    if not isinstance(value, list) or len(value) != chain_count:
        raise ProspectivePredictionError(f"summary_{key}_shape_invalid")
    for row in value:
        if not isinstance(row, list) or len(row) != chain_count:
            raise ProspectivePredictionError(f"summary_{key}_shape_invalid")
        for item in row:
            number = _finite_number(item, key)
            if number < minimum or (
                maximum is not None and number > maximum
            ):
                raise ProspectivePredictionError(
                    f"summary_{key}_outside_range"
                )


def _validate_vector(
    value: Any,
    *,
    chain_count: int,
    key: str,
) -> None:
    if not isinstance(value, list) or len(value) != chain_count:
        raise ProspectivePredictionError(f"summary_{key}_shape_invalid")
    for item in value:
        number = _finite_number(item, key)
        if not 0.0 <= number <= 1.0:
            raise ProspectivePredictionError(f"summary_{key}_outside_range")


def _contains_forbidden_label_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower())
            if any(fragment in normalized for fragment in FORBIDDEN_LABEL_KEY_FRAGMENTS):
                return True
            if _contains_forbidden_label_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_label_key(item) for item in value)
    return False


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ProspectivePredictionError("expected_json_object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--input-freeze", type=Path, required=True)
    parser.add_argument("--retained-manifest", type=Path, required=True)
    parser.add_argument("--private-input-dir", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument(
        "--expected-attestation-sha256",
        required=True,
    )
    parser.add_argument("--af3-output-root", type=Path, required=True)
    parser.add_argument("--private-lock-out", type=Path, required=True)
    parser.add_argument("--public-freeze-out", type=Path, required=True)
    args = parser.parse_args()

    preregistration = _load_json(args.preregistration)
    input_freeze = _load_json(args.input_freeze)
    retained = load_c5_manifest(args.retained_manifest)
    private_lock, public_freeze = build_prediction_lock(
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_rows=retained,
        retained_manifest_path=args.retained_manifest,
        private_input_dir=args.private_input_dir,
        attestation_path=args.attestation,
        expected_attestation_sha256=args.expected_attestation_sha256,
        output_root=args.af3_output_root,
    )
    write_json(args.private_lock_out, private_lock)
    write_json(args.public_freeze_out, public_freeze)
    print(
        json.dumps(
            {
                "targets": public_freeze["counts"]["targets"],
                "samples": public_freeze["counts"]["samples"],
                "ready_for_calibration_label_reveal": public_freeze[
                    "decision"
                ]["ready_for_calibration_label_reveal"],
                "prediction_artifact_set_sha256": public_freeze[
                    "commitments"
                ]["prediction_artifact_set_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
