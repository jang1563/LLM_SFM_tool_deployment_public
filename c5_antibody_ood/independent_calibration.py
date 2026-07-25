"""Independent-source C5 calibration intake and locked transfer replay.

The canonical input is the commit-pinned AF3 result table from Hitawala and
Gray. Only bound antibody/nanobody rows with a complete ranking score,
heavy-antigen ipTM, and DockQ label are retained. Source filenames are hashed
on export, sequences and structures are never loaded, and PDB IDs present in
the Fromm source-backed panel are excluded before calibration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_sfm_tool_deployment import Action, CalibrationStatus, EvidenceStatus

from .calibration import (
    select_hoeffding_certificate,
    threshold_policy_metrics,
)
from .manifest import (
    C5_ACTION_TOOL,
    C5_ALLOWED_ACTIONS,
    C5_ALLOWED_TOOLS,
    C5_BASE_TOOLS,
    score_c5_trajectory,
    trajectory_from_c5_row,
)
from .source_pilot import (
    C5_CANDIDATE_THRESHOLDS,
    C5_DELTA,
    C5_DOCKQ_SUCCESS_THRESHOLD,
    C5_PRIMARY_ALPHA,
    public_artifact_issues,
    sha256_file,
)


GRAY_DATASET = "c5_gray_af3_independent_calibration_v1"
GRAY_PAPER_DOI = "10.1080/19420862.2025.2545601"
GRAY_ZENODO_DOI = "10.5281/zenodo.16426003"
GRAY_ZENODO_RECORD = "https://zenodo.org/records/16426003"
GRAY_ZENODO_LICENSE = "CC-BY-4.0"
GRAY_ZENODO_ARCHIVE = "revised_Compiled_Benchmark_Data.zip"
GRAY_ZENODO_ARCHIVE_MD5 = "6d0c48b4c30b75a6e331c0d3aba4c010"
GRAY_UPSTREAM_REPOSITORY = "NooriFatima/AF3_AbNb_Benchmark"
GRAY_UPSTREAM_COMMIT = "749933edc2b7b5f841f453a667bd2204d3e31e56"
GRAY_REPOSITORY_LICENSE = "MIT"
GRAY_SOURCE_FILE = "datafiles/final_af3_rmsds.csv"
GRAY_SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    f"{GRAY_UPSTREAM_REPOSITORY}/{GRAY_UPSTREAM_COMMIT}/{GRAY_SOURCE_FILE}"
)
GRAY_SOURCE_SHA256 = (
    "c012928f1bd36ac255a43b6a3abc33d4f59033b97f6655d9b7c300850e0c433b"
)
FROMM_LOCKED_MANIFEST_SHA256 = (
    "8de4a31cd927b9dadf2d9bf5bd4f21d4a46f46f4bc4c62c30b2c21b42c426d9e"
)

GRAY_REQUIRED_COLUMNS = frozenset(
    {
        "AF3_PDB",
        "Bound_Unbound",
        "Native_PDB",
        "Protein_type",
        "PDB_short",
        "Seed",
        "Model",
        "DockQ",
        "ipTM_HA",
        "Rank",
    }
)
GRAY_FORMATS = ("antibody", "nanobody")
GRAY_CHAIN_ROLES = {
    "antibody": (
        ("H", "antibody_heavy"),
        ("L", "antibody_light"),
        ("A", "antigen"),
    ),
    "nanobody": (
        ("H", "antibody_heavy"),
        ("A", "antigen"),
    ),
}


@dataclass(frozen=True)
class GraySourceContract:
    """Expected identity and scientific shape of the Gray source table."""

    expected_sha256: str | None
    expected_rows: int
    expected_targets: int
    expected_bound_rows: int
    expected_bound_targets: int


CANONICAL_GRAY_CONTRACT = GraySourceContract(
    expected_sha256=GRAY_SOURCE_SHA256,
    expected_rows=1_900,
    expected_targets=130,
    expected_bound_rows=1_565,
    expected_bound_targets=108,
)


@dataclass(frozen=True)
class GraySample:
    """Allowlisted fields for one bound AF3 prediction."""

    sample_id: str
    complex_id: str
    pdb_id: str
    antibody_format: str
    dockq: float
    iptm_ha: float
    ranking_score: float


@dataclass(frozen=True)
class GraySelectedTarget:
    """One source-protocol-ranked prediction for a target."""

    sample: GraySample
    top_ranking_tie_count: int
    post_iptm_tie_count: int

    @property
    def success(self) -> bool:
        return self.sample.dockq >= C5_DOCKQ_SUCCESS_THRESHOLD


class GraySourceIntakeError(ValueError):
    """Raised when independent source identity or shape fails closed."""

    def __init__(self, issues: Sequence[str]):
        self.issues = tuple(issues)
        super().__init__("; ".join(self.issues))


def intake_gray_scores(
    path: str | Path,
    *,
    contract: GraySourceContract = CANONICAL_GRAY_CONTRACT,
) -> tuple[list[GraySample], dict[str, Any]]:
    """Validate the source table and retain complete bound rows only."""

    source_path = Path(path)
    source_sha256 = sha256_file(source_path)
    issues: list[str] = []
    if contract.expected_sha256 and source_sha256 != contract.expected_sha256:
        issues.append(
            "source_sha256_mismatch:"
            f"{source_sha256}!={contract.expected_sha256}"
        )

    samples: list[GraySample] = []
    source_columns: tuple[str, ...] = ()
    source_rows = 0
    source_targets: set[str] = set()
    unbound_rows = 0
    with source_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        source_columns = tuple(reader.fieldnames or ())
        missing = sorted(GRAY_REQUIRED_COLUMNS - set(source_columns))
        if missing:
            issues.append(f"source_missing_columns:{','.join(missing)}")
        else:
            for row_number, row in enumerate(reader, start=2):
                source_rows += 1
                try:
                    complex_id, pdb_id = _parse_complex_id(row.get("PDB_short"))
                    source_targets.add(complex_id)
                    bound_status = _required_text(row, "Bound_Unbound").lower()
                    if bound_status not in {"bound", "unbound"}:
                        raise ValueError("Bound_Unbound_invalid")
                    if bound_status == "unbound":
                        unbound_rows += 1
                        continue
                    samples.append(
                        _parse_gray_bound_sample(
                            row,
                            complex_id=complex_id,
                            pdb_id=pdb_id,
                        )
                    )
                except ValueError as exc:
                    issues.append(f"source_row_invalid:{row_number}:{exc}")

    if source_rows != contract.expected_rows:
        issues.append(f"source_row_count:{source_rows}!={contract.expected_rows}")
    if len(source_targets) != contract.expected_targets:
        issues.append(
            f"source_target_count:{len(source_targets)}!="
            f"{contract.expected_targets}"
        )
    if len(samples) != contract.expected_bound_rows:
        issues.append(
            f"source_bound_row_count:{len(samples)}!="
            f"{contract.expected_bound_rows}"
        )
    bound_targets = {sample.complex_id for sample in samples}
    if len(bound_targets) != contract.expected_bound_targets:
        issues.append(
            f"source_bound_target_count:{len(bound_targets)}!="
            f"{contract.expected_bound_targets}"
        )

    duplicate_sample_ids = sorted(
        sample_id
        for sample_id, count in Counter(
            sample.sample_id for sample in samples
        ).items()
        if count > 1
    )
    if duplicate_sample_ids:
        issues.append(f"source_duplicate_sample_id:{duplicate_sample_ids[0]}")
    observed_formats = {sample.antibody_format for sample in samples}
    unexpected_formats = sorted(observed_formats - set(GRAY_FORMATS))
    if unexpected_formats:
        issues.append(
            f"source_unexpected_antibody_format:{','.join(unexpected_formats)}"
        )
    missing_formats = sorted(set(GRAY_FORMATS) - observed_formats)
    if missing_formats:
        issues.append(
            f"source_missing_antibody_format:{','.join(missing_formats)}"
        )

    if issues:
        raise GraySourceIntakeError(issues)

    audit = {
        "sha256": source_sha256,
        "bytes": source_path.stat().st_size,
        "rows": source_rows,
        "targets": len(source_targets),
        "bound_rows_retained": len(samples),
        "bound_targets_retained": len(bound_targets),
        "unbound_rows_excluded": unbound_rows,
        "source_column_count": len(source_columns),
        "allowlisted_column_count": len(GRAY_REQUIRED_COLUMNS),
        "excluded_column_count": len(source_columns) - len(GRAY_REQUIRED_COLUMNS),
        "raw_filenames_emitted": False,
        "raw_paths_or_sequences_emitted": False,
    }
    return samples, audit


def select_gray_targets(
    samples: Sequence[GraySample],
    *,
    blocked_pdb_ids: set[str] | frozenset[str],
) -> tuple[list[GraySelectedTarget], dict[str, Any]]:
    """Select one target sample and exclude source-overlapping PDB entries."""

    if not blocked_pdb_ids:
        raise ValueError("blocked_pdb_ids must be non-empty")
    grouped: dict[str, list[GraySample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.complex_id].append(sample)

    selected_before_overlap: list[GraySelectedTarget] = []
    for complex_id in sorted(grouped):
        target_samples = grouped[complex_id]
        maximum_rank = max(sample.ranking_score for sample in target_samples)
        rank_tied = [
            sample
            for sample in target_samples
            if sample.ranking_score == maximum_rank
        ]
        maximum_iptm = max(sample.iptm_ha for sample in rank_tied)
        iptm_tied = [
            sample for sample in rank_tied if sample.iptm_ha == maximum_iptm
        ]
        winner = min(iptm_tied, key=lambda sample: sample.sample_id)
        selected_before_overlap.append(
            GraySelectedTarget(
                sample=winner,
                top_ranking_tie_count=len(rank_tied),
                post_iptm_tie_count=len(iptm_tied),
            )
        )

    overlap = [
        target
        for target in selected_before_overlap
        if target.sample.pdb_id in blocked_pdb_ids
    ]
    selected = [
        target
        for target in selected_before_overlap
        if target.sample.pdb_id not in blocked_pdb_ids
    ]
    if not selected:
        raise ValueError("overlap exclusion removed every target")
    retained_pdb_ids = {target.sample.pdb_id for target in selected}
    residual_overlap = sorted(retained_pdb_ids & set(blocked_pdb_ids))
    if residual_overlap:
        raise ValueError(f"source_overlap_not_removed:{residual_overlap[0]}")

    audit = {
        "selection_rule": (
            "max_ranking_score_then_max_iptm_ha_then_lexical_sample_id"
        ),
        "selected_before_overlap": len(selected_before_overlap),
        "overlapping_pdb_ids_excluded": len(
            {target.sample.pdb_id for target in overlap}
        ),
        "overlapping_complexes_excluded": len(overlap),
        "selected_after_overlap": len(selected),
        "residual_pdb_overlap": 0,
        "targets_with_top_ranking_ties": sum(
            target.top_ranking_tie_count > 1
            for target in selected
        ),
        "targets_with_post_iptm_ties": sum(
            target.post_iptm_tie_count > 1
            for target in selected
        ),
        "selected_by_format": dict(
            sorted(
                Counter(
                    target.sample.antibody_format
                    for target in selected
                ).items()
            )
        ),
        "selected_interface_successes": sum(target.success for target in selected),
        "selected_interface_success_rate": round(
            sum(target.success for target in selected) / len(selected),
            6,
        ),
        "dockq_success_threshold": C5_DOCKQ_SUCCESS_THRESHOLD,
    }
    return selected, audit


def fromm_pdb_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return normalized four-character PDB IDs from canonical Fromm rows."""

    pdb_ids: set[str] = set()
    for index, row in enumerate(rows):
        visible = row.get("model_visible_task")
        if not isinstance(visible, Mapping):
            raise ValueError(f"fromm_row[{index}] model_visible_task missing")
        complex_id = str(visible.get("complex_id", "")).strip().lower()
        if not re.fullmatch(r"[0-9][a-z0-9]{3}", complex_id):
            raise ValueError(f"fromm_row[{index}] complex_id invalid")
        pdb_ids.add(complex_id)
    if len(pdb_ids) != len(rows):
        raise ValueError("Fromm complex IDs must be unique")
    return pdb_ids


def build_gray_manifest(
    selected: Sequence[GraySelectedTarget],
    *,
    certificates: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build canonical fail-closed C5 rows for independent calibration."""

    return [
        _gray_manifest_row(
            target,
            certificate=certificates[target.sample.antibody_format],
        )
        for target in sorted(selected, key=lambda item: item.sample.complex_id)
    ]


def validate_gray_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
    blocked_pdb_ids: set[str] | frozenset[str],
) -> list[str]:
    """Validate schema reuse, label isolation, overlap, and fail-closed action."""

    issues: list[str] = []
    if len(rows) != expected_rows:
        issues.append(f"manifest_row_count:{len(rows)}!={expected_rows}")
    case_ids: list[str] = []
    complex_ids: list[str] = []
    split_groups: list[str] = []
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        visible = row.get("model_visible_task")
        hidden = row.get("hidden_eval_metadata")
        if not isinstance(visible, Mapping):
            issues.append(f"{prefix}:model_visible_task_not_mapping")
            continue
        if not isinstance(hidden, Mapping):
            issues.append(f"{prefix}:hidden_eval_metadata_not_mapping")
            continue

        case_id = str(row.get("case_id", ""))
        complex_id = str(visible.get("complex_id", ""))
        split_group = str(hidden.get("split_group", ""))
        case_ids.append(case_id)
        complex_ids.append(complex_id)
        split_groups.append(split_group)
        if row.get("dataset") != GRAY_DATASET:
            issues.append(f"{prefix}:dataset_mismatch")
        if visible.get("input_id") != case_id:
            issues.append(f"{prefix}:case_input_id_mismatch")
        if tuple(visible.get("allowed_tools", ())) != C5_ALLOWED_TOOLS:
            issues.append(f"{prefix}:allowed_tools_contract_mismatch")
        if tuple(visible.get("allowed_terminal_actions", ())) != C5_ALLOWED_ACTIONS:
            issues.append(f"{prefix}:allowed_actions_contract_mismatch")
        if visible.get("interaction_regime") != "antibody_antigen":
            issues.append(f"{prefix}:interaction_regime_mismatch")
        if hidden.get("split") != "independent_calibration":
            issues.append(f"{prefix}:split_mismatch")

        if not re.fullmatch(r"[0-9][a-z0-9]{3}(?:_[0-9]+)?", complex_id):
            issues.append(f"{prefix}:complex_id_invalid")
        pdb_id = complex_id[:4].lower()
        if pdb_id in blocked_pdb_ids:
            issues.append(f"{prefix}:source_target_overlap:{pdb_id}")

        visible_text = json.dumps(visible, sort_keys=True).lower()
        for hidden_key in (
            "interface_label_status",
            "interface_label_source",
            "expected_terminal_action",
            "split",
            "split_group",
        ):
            if hidden_key in _recursive_keys(visible):
                issues.append(f"{prefix}:hidden_key_leak:{hidden_key}")
        for hidden_key in ("interface_label_status", "interface_label_source"):
            hidden_value = hidden.get(hidden_key)
            if hidden_value and str(hidden_value).lower() in visible_text:
                issues.append(f"{prefix}:hidden_value_leak:{hidden_key}")

        calibration = visible.get("calibration")
        if not isinstance(calibration, Mapping):
            issues.append(f"{prefix}:calibration_not_mapping")
        elif calibration.get("certified") is not True:
            if hidden.get("expected_terminal_action") != (
                Action.VERIFY_WITH_ASSAY_OR_DATABASE.value
            ):
                issues.append(f"{prefix}:uncertified_row_not_fail_closed")

        role_mapping = visible.get("chain_role_mapping")
        if not isinstance(role_mapping, list):
            issues.append(f"{prefix}:chain_role_mapping_not_list")
            role_names: set[Any] = set()
        else:
            role_names = {
                item.get("role")
                for item in role_mapping
                if isinstance(item, Mapping)
            }
        if "antibody_heavy" not in role_names or "antigen" not in role_names:
            issues.append(f"{prefix}:chain_roles_incomplete")
        antibody_format = visible.get("antibody_format")
        if antibody_format not in GRAY_CHAIN_ROLES:
            issues.append(f"{prefix}:antibody_format_invalid")
        elif role_mapping != [
            {"chain_id": chain_id, "role": role}
            for chain_id, role in GRAY_CHAIN_ROLES[str(antibody_format)]
        ]:
            issues.append(f"{prefix}:chain_role_mapping_mismatch")

        specialist = visible.get("specialist_result")
        if not isinstance(specialist, Mapping):
            issues.append(f"{prefix}:specialist_result_not_mapping")
        else:
            expected_fields = {
                "metric_type": "ranking_score",
                "metric_scope": "whole_complex_ranking",
                "interface_metric_type": (
                    "predicted_interface_tm_score_heavy_antigen"
                ),
                "interface_metric_scope": "heavy_chain_antigen_interface",
            }
            for key, expected_value in expected_fields.items():
                if specialist.get(key) != expected_value:
                    issues.append(f"{prefix}:specialist_{key}_mismatch")
            for key in ("metric_value", "interface_metric_value"):
                value = specialist.get(key)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    issues.append(f"{prefix}:specialist_{key}_invalid")

        expected_action = Action.VERIFY_WITH_ASSAY_OR_DATABASE.value
        if isinstance(calibration, Mapping) and calibration.get("certified") is True:
            threshold = calibration.get("threshold_value")
            metric_value = (
                specialist.get("metric_value")
                if isinstance(specialist, Mapping)
                else None
            )
            if (
                isinstance(threshold, (int, float))
                and not isinstance(threshold, bool)
                and isinstance(metric_value, (int, float))
                and not isinstance(metric_value, bool)
                and metric_value >= threshold
            ):
                expected_action = Action.TRUST_SPECIALIST_OUTPUT.value
        if hidden.get("expected_terminal_action") != expected_action:
            issues.append(f"{prefix}:calibration_action_mismatch")

        sample_hash = hidden.get("selected_sample_id_sha256")
        if not isinstance(sample_hash, str) or not re.fullmatch(
            r"[a-f0-9]{64}", sample_hash
        ):
            issues.append(f"{prefix}:selected_sample_hash_invalid")
        provenance = row.get("source_provenance")
        if not isinstance(provenance, Mapping):
            issues.append(f"{prefix}:source_provenance_not_mapping")
        else:
            if provenance.get("source_sha256") != GRAY_SOURCE_SHA256:
                issues.append(f"{prefix}:source_sha256_mismatch")
            if provenance.get("upstream_commit") != GRAY_UPSTREAM_COMMIT:
                issues.append(f"{prefix}:source_commit_mismatch")

        try:
            expected = hidden["expected_terminal_action"]
            trajectory = trajectory_from_c5_row(
                row,
                expected,
                fail_closed_reason=(
                    None
                    if expected == Action.TRUST_SPECIALIST_OUTPUT.value
                    else "independent_calibration_not_certified"
                ),
            )
            result = score_c5_trajectory(row, trajectory)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                f"{prefix}:canonical_projection_failed:{type(exc).__name__}"
            )
        else:
            if not result.passed:
                issues.append(
                    f"{prefix}:canonical_trajectory_failed:"
                    + ",".join(result.violations)
                )
        issues.extend(
            f"{prefix}:{issue}" for issue in public_artifact_issues(row)
        )

    for label, values in (
        ("case_id", case_ids),
        ("complex_id", complex_ids),
        ("split_group", split_groups),
    ):
        duplicates = sorted(
            value
            for value, count in Counter(values).items()
            if value and count > 1
        )
        if duplicates:
            issues.append(f"duplicate_{label}:{duplicates[0]}")
    return issues


def build_independent_calibration(
    gray_source_path: str | Path,
    fromm_rows: Sequence[Mapping[str, Any]],
    *,
    contract: GraySourceContract = CANONICAL_GRAY_CONTRACT,
    fromm_manifest_sha256: str | None = None,
    expected_fromm_manifest_sha256: str | None = FROMM_LOCKED_MANIFEST_SHA256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run source intake, overlap exclusion, calibration, and locked transfer."""

    if (
        expected_fromm_manifest_sha256
        and fromm_manifest_sha256 != expected_fromm_manifest_sha256
    ):
        raise ValueError(
            "Fromm manifest SHA-256 mismatch:"
            f"{fromm_manifest_sha256}!={expected_fromm_manifest_sha256}"
        )
    samples, intake = intake_gray_scores(gray_source_path, contract=contract)
    blocked = fromm_pdb_ids(fromm_rows)
    selected, selection = select_gray_targets(
        samples,
        blocked_pdb_ids=blocked,
    )
    by_format = {
        antibody_format: [
            target
            for target in selected
            if target.sample.antibody_format == antibody_format
        ]
        for antibody_format in GRAY_FORMATS
    }
    certificates_by_alpha = {
        antibody_format: {
            f"alpha_{alpha:.2f}": select_hoeffding_certificate(
                _observations(targets),
                alpha=alpha,
                delta=C5_DELTA,
                thresholds=C5_CANDIDATE_THRESHOLDS,
            )
            for alpha in (0.10, 0.20, C5_PRIMARY_ALPHA)
        }
        for antibody_format, targets in by_format.items()
    }
    primary_certificates = {
        antibody_format: certificates[
            f"alpha_{C5_PRIMARY_ALPHA:.2f}"
        ]
        for antibody_format, certificates in certificates_by_alpha.items()
    }
    rows = build_gray_manifest(
        selected,
        certificates=primary_certificates,
    )
    manifest_issues = validate_gray_manifest(
        rows,
        expected_rows=len(selected),
        blocked_pdb_ids=blocked,
    )

    fromm_evaluation = _fromm_evaluation_observations(fromm_rows)
    antibody_certificate = primary_certificates["antibody"]
    transfer_threshold = antibody_certificate["threshold"]
    report = {
        "dataset": "c5_gray_independent_calibration_eval_v1",
        "evaluation_scope": (
            "independent_source_published_label_calibration_and_locked_"
            "external_replay"
        ),
        "source": {
            "paper_doi": GRAY_PAPER_DOI,
            "archive_doi": GRAY_ZENODO_DOI,
            "archive_url": GRAY_ZENODO_RECORD,
            "archive_license": GRAY_ZENODO_LICENSE,
            "archive_file": GRAY_ZENODO_ARCHIVE,
            "archive_md5": GRAY_ZENODO_ARCHIVE_MD5,
            "repository": GRAY_UPSTREAM_REPOSITORY,
            "repository_license": GRAY_REPOSITORY_LICENSE,
            "upstream_commit": GRAY_UPSTREAM_COMMIT,
            "source_file": GRAY_SOURCE_FILE,
            "source_url": GRAY_SOURCE_URL,
            **intake,
        },
        "overlap_and_selection": selection,
        "calibration": {
            "confidence_metric": "af3_ranking_score",
            "interface_metric": "heavy_antigen_predicted_tm_score",
            "success_label": "DockQ >= 0.23",
            "method": "uniform_hoeffding_union_bound",
            "candidate_thresholds": {
                "minimum": min(C5_CANDIDATE_THRESHOLDS),
                "maximum": max(C5_CANDIDATE_THRESHOLDS),
                "step": 0.01,
                "count": len(C5_CANDIDATE_THRESHOLDS),
            },
            "format_specific_certificates": certificates_by_alpha,
        },
        "source_cohort_policies": {
            antibody_format: {
                "trust_all": threshold_policy_metrics(
                    _observations(targets),
                    threshold=None,
                ),
                "fixed_ranking_score_0_80": threshold_policy_metrics(
                    _observations(targets),
                    threshold=0.80,
                ),
                "certified_gate": {
                    **threshold_policy_metrics(
                        _observations(targets),
                        threshold=(
                            float(
                                primary_certificates[antibody_format][
                                    "threshold"
                                ]
                            )
                            if primary_certificates[antibody_format][
                                "certified"
                            ]
                            else math.inf
                        ),
                    ),
                    "certified": primary_certificates[antibody_format][
                        "certified"
                    ],
                    "threshold": primary_certificates[antibody_format][
                        "threshold"
                    ],
                },
            }
            for antibody_format, targets in by_format.items()
        },
        "locked_fromm_evaluation": {
            "manifest_sha256": fromm_manifest_sha256,
            "targets": len(fromm_evaluation),
            "pdb_overlap_with_gray_calibration": 0,
            "metric_alignment": (
                "af3_score_names_and_ranges_aligned_"
                "exact_model_version_equivalence_unverified"
            ),
            "trust_all": threshold_policy_metrics(
                fromm_evaluation,
                threshold=None,
            ),
            "fixed_ranking_score_0_80": threshold_policy_metrics(
                fromm_evaluation,
                threshold=0.80,
            ),
            "independent_calibration_gate": {
                **threshold_policy_metrics(
                    fromm_evaluation,
                    threshold=(
                        float(transfer_threshold)
                        if antibody_certificate["certified"]
                        else math.inf
                    ),
                ),
                "certified": antibody_certificate["certified"],
                "threshold": transfer_threshold,
            },
        },
        "manifest": {
            "rows": len(rows),
            "validation_issues": manifest_issues,
            "hidden_label_isolation": not any(
                "hidden_" in issue for issue in manifest_issues
            ),
        },
        "decision": {
            "independent_source_adapter_passed": not manifest_issues,
            "antibody_ranking_gate_certified": antibody_certificate[
                "certified"
            ],
            "nanobody_ranking_gate_certified": primary_certificates[
                "nanobody"
            ]["certified"],
            "external_trust_enabled": antibody_certificate["certified"],
            "ready_for_c5_model_training": False,
            "ready_for_dpo_rlvr": False,
            "next_ticket": (
                "expand_independent_antibody_calibration_or_run_"
                "preregistered_cayuga_panel"
            ),
        },
        "scientific_boundary": {
            "independent_source_dataset_used": True,
            "source_target_overlap_after_exclusion": 0,
            "independent_hidden_test_claimed": False,
            "published_labels_used": True,
            "blinded_evaluation_claimed": False,
            "new_structure_prediction_run": False,
            "llm_or_api_used": False,
            "source_sample_ids_published": False,
            "raw_paths_structures_or_sequences_published": False,
            "source_model_version_equivalence_claimed": False,
        },
    }
    report_issues = public_artifact_issues(report)
    if report_issues:
        report["manifest"]["validation_issues"].extend(
            f"report:{issue}" for issue in report_issues
        )
        report["decision"]["independent_source_adapter_passed"] = False
    return rows, report


def _parse_gray_bound_sample(
    row: Mapping[str, str | None],
    *,
    complex_id: str,
    pdb_id: str,
) -> GraySample:
    antibody_format = _required_text(row, "Protein_type").lower()
    if antibody_format not in GRAY_FORMATS:
        raise ValueError("Protein_type_invalid")
    sample_id = _required_text(row, "AF3_PDB")
    if Path(sample_id).name != sample_id:
        raise ValueError("AF3_PDB_not_basename")
    if not sample_id.lower().endswith(".pdb"):
        raise ValueError("AF3_PDB_not_pdb")
    seed = _required_text(row, "Seed")
    model = _required_text(row, "Model")
    if complex_id not in sample_id.lower():
        raise ValueError("AF3_PDB_target_mismatch")
    if not re.search(rf"seed_?{re.escape(seed)}(?:_|\.|$)", sample_id):
        raise ValueError("AF3_PDB_seed_mismatch")
    if f"model_{model}" not in sample_id:
        raise ValueError("AF3_PDB_model_mismatch")
    native_id = _required_text(row, "Native_PDB")
    if Path(native_id).name != native_id:
        raise ValueError("Native_PDB_not_basename")
    if complex_id not in native_id.lower():
        raise ValueError("Native_PDB_target_mismatch")
    return GraySample(
        sample_id=sample_id,
        complex_id=complex_id,
        pdb_id=pdb_id,
        antibody_format=antibody_format,
        dockq=_unit_interval(row, "DockQ"),
        iptm_ha=_unit_interval(row, "ipTM_HA"),
        ranking_score=_unit_interval(row, "Rank"),
    )


def _parse_complex_id(value: str | None) -> tuple[str, str]:
    if value is None:
        raise ValueError("PDB_short_empty")
    complex_id = value.strip().lower()
    if complex_id.endswith(".pdb"):
        complex_id = complex_id[:-4]
    if not re.fullmatch(r"[0-9][a-z0-9]{3}(?:_[0-9]+)?", complex_id):
        raise ValueError("PDB_short_invalid")
    return complex_id, complex_id[:4]


def _gray_manifest_row(
    target: GraySelectedTarget,
    *,
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    sample = target.sample
    certified = certificate.get("certified") is True
    threshold = certificate.get("threshold")
    can_trust = (
        certified
        and isinstance(threshold, (int, float))
        and sample.ranking_score >= float(threshold)
    )
    expected_action = (
        Action.TRUST_SPECIALIST_OUTPUT
        if can_trust
        else Action.VERIFY_WITH_ASSAY_OR_DATABASE
    )
    expected_status = (
        EvidenceStatus.SUPPORTED
        if can_trust
        else EvidenceStatus.INSUFFICIENT
    )
    input_id = f"c5-gray-af3::{sample.complex_id}"
    threshold_id = (
        "hoeffding_gray_abag_ranking_alpha_0_30_v1" if certified else None
    )
    return {
        "case_id": input_id,
        "dataset": GRAY_DATASET,
        "model_visible_task": {
            "input_id": input_id,
            "claim": (
                "Select a deployment action for this antibody-antigen "
                "structure prediction after inspecting its ranking confidence "
                "and calibration metadata."
            ),
            "complex_id": sample.complex_id,
            "chain_role_mapping": [
                {"chain_id": chain_id, "role": role}
                for chain_id, role in GRAY_CHAIN_ROLES[sample.antibody_format]
            ],
            "interaction_regime": "antibody_antigen",
            "antibody_format": sample.antibody_format,
            "allowed_tools": list(C5_ALLOWED_TOOLS),
            "allowed_terminal_actions": list(C5_ALLOWED_ACTIONS),
            "specialist_result": {
                "specialist_name": "alphafold3",
                "metric_type": "ranking_score",
                "metric_scope": "whole_complex_ranking",
                "metric_value": sample.ranking_score,
                "interface_metric_type": (
                    "predicted_interface_tm_score_heavy_antigen"
                ),
                "interface_metric_scope": "heavy_chain_antigen_interface",
                "interface_metric_value": sample.iptm_ha,
            },
            "general_calibration": {
                "dataset_id": "generic_fixed_threshold_not_source_fitted_v1",
                "regime": "generic_protein_interface",
                "threshold_id": "generic_ranking_score_0_80_v1",
                "threshold_value": 0.80,
            },
            "calibration": {
                "status": (
                    CalibrationStatus.CALIBRATED.value
                    if certified
                    else CalibrationStatus.UNCALIBRATED.value
                ),
                "dataset_id": (
                    f"gray_{sample.antibody_format}_af3_calibration_v1"
                    if certified
                    else None
                ),
                "regime": "antibody_antigen",
                "regime_match": True,
                "threshold_id": threshold_id,
                "threshold_value": threshold,
                "certified": certified,
            },
            "baseline_result": {
                "available": False,
                "dominates_specialist": False,
                "metric_type": None,
                "metric_value": None,
            },
            "verification_available": True,
        },
        "hidden_eval_metadata": {
            "required_tools": [
                *C5_BASE_TOOLS,
                *C5_ACTION_TOOL[expected_action],
            ],
            "gold_evidence_status": expected_status.value,
            "expected_terminal_action": expected_action.value,
            "gold_source_ids": [],
            "requires_attribution": False,
            "requires_external_tool": True,
            "web_zero": False,
            "case_family": "independent_source_af3_ranking_calibration",
            "split_group": f"complex::{sample.complex_id}",
            "split": "independent_calibration",
            "interface_label_status": (
                "interface_success"
                if target.success
                else "interface_failure"
            ),
            "interface_label_source": (
                "hitawala_gray_2025_dockq_ge_0_23"
            ),
            "selected_sample_id_sha256": hashlib.sha256(
                f"gray-af3-v1::{sample.sample_id}".encode()
            ).hexdigest(),
            "top_ranking_tie_count": target.top_ranking_tie_count,
            "post_iptm_tie_count": target.post_iptm_tie_count,
        },
        "source_provenance": {
            "paper_doi": GRAY_PAPER_DOI,
            "archive_doi": GRAY_ZENODO_DOI,
            "archive_license": GRAY_ZENODO_LICENSE,
            "repository": GRAY_UPSTREAM_REPOSITORY,
            "repository_license": GRAY_REPOSITORY_LICENSE,
            "source_file": GRAY_SOURCE_FILE,
            "source_sha256": GRAY_SOURCE_SHA256,
            "upstream_commit": GRAY_UPSTREAM_COMMIT,
        },
        "cost_profile": {
            "inspect_specialist_record": 1.0,
            "lookup_calibration_card": 1.0,
            "request_structure_verification": 5.0,
            "run_cheap_interface_baseline": 2.0,
            "defer_or_request_more_evidence": 0.5,
        },
    }


def _observations(
    targets: Sequence[GraySelectedTarget],
) -> list[tuple[float, bool]]:
    return [
        (target.sample.ranking_score, target.success)
        for target in targets
    ]


def _fromm_evaluation_observations(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[float, bool]]:
    observations: list[tuple[float, bool]] = []
    for index, row in enumerate(rows):
        hidden = row.get("hidden_eval_metadata")
        visible = row.get("model_visible_task")
        if not isinstance(hidden, Mapping) or not isinstance(visible, Mapping):
            raise ValueError(f"fromm_row[{index}] malformed")
        if hidden.get("split") != "evaluation":
            continue
        specialist = visible.get("specialist_result")
        if not isinstance(specialist, Mapping):
            raise ValueError(f"fromm_row[{index}] specialist_result malformed")
        score = specialist.get("ranking_metric_value")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError(f"fromm_row[{index}] ranking metric invalid")
        label = hidden.get("interface_label_status")
        if label not in {"interface_success", "interface_failure"}:
            raise ValueError(f"fromm_row[{index}] interface label invalid")
        observations.append((float(score), label == "interface_success"))
    if len(observations) != 55:
        raise ValueError(f"Fromm evaluation rows must equal 55, got {len(observations)}")
    return observations


def _required_text(row: Mapping[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"{key}_empty")
    return value.strip()


def _unit_interval(row: Mapping[str, str | None], key: str) -> float:
    raw = _required_text(row, key)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key}_not_numeric") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{key}_outside_unit_interval")
    return value


def _recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_recursive_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_recursive_keys(item))
    return keys
