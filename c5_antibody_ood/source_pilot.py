"""Source-backed C5 intake, frozen splitting, and calibration baselines.

The canonical input is the AlphaFold 3 score table archived with Fromm et al.
Only an allowlisted scientific subset is parsed. Raw structures, sequences,
absolute compute paths, and source sample identifiers are never emitted.
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
from typing import Any, Iterable, Mapping, Sequence

from llm_sfm_tool_deployment import Action, CalibrationStatus, EvidenceStatus

from .manifest import (
    C5_ACTION_TOOL,
    C5_ALLOWED_ACTIONS,
    C5_ALLOWED_TOOLS,
    C5_BASE_TOOLS,
    score_c5_trajectory,
    trajectory_from_c5_row,
)


C5_SOURCE_DATASET = "c5_abag_af3_source_backed_pilot_v1"
C5_SPLIT_SALT = "c5-abag-af3-v1"
C5_DOCKQ_SUCCESS_THRESHOLD = 0.23
C5_GENERIC_IPTM_THRESHOLD = 0.80
C5_PRIMARY_ALPHA = 0.30
C5_DELTA = 0.10
C5_CANDIDATE_THRESHOLDS = tuple(value / 100 for value in range(50, 100))

FROMM_PAPER_DOI = "10.1093/bioinformatics/btag136"
FROMM_ZENODO_DOI = "10.5281/zenodo.17978681"
FROMM_ZENODO_RECORD = "https://zenodo.org/records/17978681"
FROMM_LICENSE = "CC-BY-4.0"
FROMM_UPSTREAM_COMMIT = "06b21927bcacf6fb0612e56cdc110c206d9eebdc"
FROMM_ARCHIVE_FILE = "abag-benchmark-set-main.zip"
FROMM_ARCHIVE_MD5 = "2933fb1bb66903469d02c64a9d21f5d9"
FROMM_SOURCE_MEMBER = (
    "abag-benchmark-set-main/data/scores/alphafold3/"
    "scores_alphafold3.csv"
)
FROMM_AF3_SHA256 = (
    "56259a84f1e8cc216e5ee91a96584f824ca46f062ef4f2c06aa4674472daf1c8"
)

SOURCE_REQUIRED_COLUMNS = frozenset(
    {
        "sample_id",
        "pdbid",
        "Achain",
        "Hchain",
        "Lchain",
        "abag_dockq",
        "iptm",
        "ranking_confidence",
        "preset",
    }
)
SOURCE_ALLOWLISTED_COLUMNS = SOURCE_REQUIRED_COLUMNS
PUBLIC_FORBIDDEN_FRAGMENTS = (
    "/users/",
    "/home/",
    "/" + "proj" + "/",
    "/" + "scratch" + "/",
    "reference_pdb",
    "query_pdb",
    "query_af_features",
    "query_af_data",
    "prediction_path",
)


@dataclass(frozen=True)
class SourceContract:
    """Expected shape and identity for one source score table."""

    expected_sha256: str | None
    expected_rows: int
    expected_targets: int
    expected_samples_per_target: int
    expected_preset: str = "alphafold3"


CANONICAL_SOURCE_CONTRACT = SourceContract(
    expected_sha256=FROMM_AF3_SHA256,
    expected_rows=22_000,
    expected_targets=110,
    expected_samples_per_target=200,
)


@dataclass(frozen=True)
class SourceSample:
    """Allowlisted source fields needed for target-level C5 evaluation."""

    sample_id: str
    complex_id: str
    antigen_chains: tuple[str, ...]
    heavy_chains: tuple[str, ...]
    light_chains: tuple[str, ...]
    dockq: float
    iptm: float
    ranking_confidence: float
    preset: str


@dataclass(frozen=True)
class SelectedTarget:
    """One confidence-ranked sample selected for one antibody-antigen target."""

    sample: SourceSample
    top_ranking_tie_count: int
    split: str

    @property
    def success(self) -> bool:
        return self.sample.dockq >= C5_DOCKQ_SUCCESS_THRESHOLD


class SourceIntakeError(ValueError):
    """Raised when source identity, schema, or scientific values fail closed."""

    def __init__(self, issues: Sequence[str]):
        self.issues = tuple(issues)
        super().__init__("; ".join(self.issues))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def intake_source_scores(
    path: str | Path,
    *,
    contract: SourceContract = CANONICAL_SOURCE_CONTRACT,
) -> tuple[list[SourceSample], dict[str, Any]]:
    """Load and validate a source CSV while retaining only allowlisted fields."""

    source_path = Path(path)
    source_sha256 = sha256_file(source_path)
    issues: list[str] = []
    if contract.expected_sha256 and source_sha256 != contract.expected_sha256:
        issues.append(
            "source_sha256_mismatch:"
            f"{source_sha256}!={contract.expected_sha256}"
        )

    samples: list[SourceSample] = []
    source_columns: tuple[str, ...] = ()
    excluded_absolute_path_cells = 0
    with source_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        source_columns = tuple(reader.fieldnames or ())
        missing = sorted(SOURCE_REQUIRED_COLUMNS - set(source_columns))
        if missing:
            issues.append(f"source_missing_columns:{','.join(missing)}")
        else:
            for row_number, row in enumerate(reader, start=2):
                excluded_absolute_path_cells += sum(
                    _looks_like_absolute_path(value)
                    for key, value in row.items()
                    if key not in SOURCE_ALLOWLISTED_COLUMNS
                )
                try:
                    samples.append(_parse_source_sample(row))
                except ValueError as exc:
                    issues.append(f"source_row_invalid:{row_number}:{exc}")

    if len(samples) != contract.expected_rows:
        issues.append(
            f"source_row_count:{len(samples)}!={contract.expected_rows}"
        )

    target_counts = Counter(sample.complex_id for sample in samples)
    if len(target_counts) != contract.expected_targets:
        issues.append(
            f"source_target_count:{len(target_counts)}!="
            f"{contract.expected_targets}"
        )
    bad_target_counts = sorted(
        target
        for target, count in target_counts.items()
        if count != contract.expected_samples_per_target
    )
    if bad_target_counts:
        issues.append(
            "source_samples_per_target_mismatch:"
            + ",".join(bad_target_counts[:10])
        )

    sample_ids = [sample.sample_id for sample in samples]
    duplicates = sorted(
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    )
    if duplicates:
        issues.append(f"source_duplicate_sample_id:{duplicates[0]}")

    bad_presets = sorted(
        {sample.preset for sample in samples}
        - {contract.expected_preset}
    )
    if bad_presets:
        issues.append(f"source_preset_mismatch:{','.join(bad_presets)}")

    if issues:
        raise SourceIntakeError(issues)

    audit = {
        "sha256": source_sha256,
        "bytes": source_path.stat().st_size,
        "rows": len(samples),
        "targets": len(target_counts),
        "samples_per_target": contract.expected_samples_per_target,
        "source_column_count": len(source_columns),
        "allowlisted_column_count": len(SOURCE_ALLOWLISTED_COLUMNS),
        "excluded_column_count": (
            len(source_columns) - len(SOURCE_ALLOWLISTED_COLUMNS)
        ),
        "excluded_absolute_path_cells": excluded_absolute_path_cells,
        "absolute_path_values_emitted": False,
    }
    return samples, audit


def _parse_source_sample(row: Mapping[str, str | None]) -> SourceSample:
    sample_id = _required_text(row, "sample_id")
    complex_id = _required_text(row, "pdbid").lower()
    if not re.fullmatch(r"[0-9][a-z0-9]{3}", complex_id):
        raise ValueError("pdbid_invalid")
    antigen_chains = parse_chain_ids(_required_text(row, "Achain"))
    heavy_chains = parse_chain_ids(_required_text(row, "Hchain"))
    light_chains = parse_chain_ids(row.get("Lchain"))
    if not antigen_chains:
        raise ValueError("Achain_empty")
    if not heavy_chains:
        raise ValueError("Hchain_empty")
    all_chains = (*antigen_chains, *heavy_chains, *light_chains)
    if len(set(all_chains)) != len(all_chains):
        raise ValueError("chain_role_overlap")
    return SourceSample(
        sample_id=sample_id,
        complex_id=complex_id,
        antigen_chains=antigen_chains,
        heavy_chains=heavy_chains,
        light_chains=light_chains,
        dockq=_unit_interval(row, "abag_dockq"),
        iptm=_unit_interval(row, "iptm"),
        ranking_confidence=_unit_interval(row, "ranking_confidence"),
        preset=_required_text(row, "preset"),
    )


def parse_chain_ids(value: str | None) -> tuple[str, ...]:
    """Parse source chain lists such as ``1 | 2 | 3`` and optional NaNs."""

    if value is None or value.strip().lower() in {"", "nan", "none", "null"}:
        return ()
    chains = tuple(part.strip() for part in value.split("|") if part.strip())
    if len(set(chains)) != len(chains):
        raise ValueError("duplicate_chain_id")
    return chains


def select_target_samples(
    samples: Sequence[SourceSample],
    *,
    calibration_targets: int,
    split_salt: str = C5_SPLIT_SALT,
) -> list[SelectedTarget]:
    """Select one ranked sample per target and assign target-grouped splits."""

    grouped: dict[str, list[SourceSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.complex_id].append(sample)
    if not 0 < calibration_targets < len(grouped):
        raise ValueError("calibration_targets must leave non-empty splits")

    ordered_targets = sorted(
        grouped,
        key=lambda target: (
            hashlib.sha256(f"{split_salt}::{target}".encode()).hexdigest(),
            target,
        ),
    )
    calibration_ids = set(ordered_targets[:calibration_targets])
    selected: list[SelectedTarget] = []
    for target in sorted(grouped):
        target_samples = grouped[target]
        maximum = max(sample.ranking_confidence for sample in target_samples)
        tied = [
            sample
            for sample in target_samples
            if sample.ranking_confidence == maximum
        ]
        winner = min(tied, key=lambda sample: sample.sample_id)
        selected.append(
            SelectedTarget(
                sample=winner,
                top_ranking_tie_count=len(tied),
                split=(
                    "calibration"
                    if target in calibration_ids
                    else "evaluation"
                ),
            )
        )
    return selected


def select_hoeffding_threshold(
    calibration_rows: Sequence[SelectedTarget],
    *,
    alpha: float,
    delta: float = C5_DELTA,
    thresholds: Sequence[float] = C5_CANDIDATE_THRESHOLDS,
) -> dict[str, Any]:
    """Select the highest-coverage threshold with a union-bound risk certificate."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    candidates = tuple(thresholds)
    if not candidates:
        raise ValueError("thresholds must be non-empty")

    eligible: list[dict[str, Any]] = []
    for threshold in candidates:
        trusted = [
            row for row in calibration_rows if row.sample.iptm >= threshold
        ]
        if not trusted:
            continue
        failures = sum(not row.success for row in trusted)
        empirical_risk = failures / len(trusted)
        radius = math.sqrt(
            math.log(len(candidates) / delta) / (2 * len(trusted))
        )
        upper_bound = min(1.0, empirical_risk + radius)
        if upper_bound <= alpha:
            eligible.append(
                {
                    "threshold": threshold,
                    "trusted": len(trusted),
                    "failures": failures,
                    "empirical_risk": empirical_risk,
                    "risk_upper_bound": upper_bound,
                }
            )

    if not eligible:
        return {
            "alpha": alpha,
            "delta": delta,
            "candidate_count": len(candidates),
            "certified": False,
            "threshold": None,
            "calibration_trusted": 0,
            "calibration_failures": 0,
            "calibration_empirical_risk": None,
            "calibration_risk_upper_bound": None,
        }

    best = min(
        eligible,
        key=lambda candidate: (
            -candidate["trusted"],
            candidate["threshold"],
        ),
    )
    return {
        "alpha": alpha,
        "delta": delta,
        "candidate_count": len(candidates),
        "certified": True,
        "threshold": best["threshold"],
        "calibration_trusted": best["trusted"],
        "calibration_failures": best["failures"],
        "calibration_empirical_risk": round(
            best["empirical_risk"], 6
        ),
        "calibration_risk_upper_bound": round(
            best["risk_upper_bound"], 6
        ),
    }


def policy_metrics(
    rows: Sequence[SelectedTarget],
    *,
    threshold: float | None,
) -> dict[str, Any]:
    """Return target-level trust risk and coverage for one threshold policy."""

    trusted = (
        list(rows)
        if threshold is None
        else [row for row in rows if row.sample.iptm >= threshold]
    )
    failures = sum(not row.success for row in trusted)
    count = len(rows)
    return {
        "targets": count,
        "trusted": len(trusted),
        "verify_or_defer": count - len(trusted),
        "failures_among_trusted": failures,
        "failure_rate_among_trusted": (
            round(failures / len(trusted), 6) if trusted else 0.0
        ),
        "coverage": round(len(trusted) / count, 6) if count else 0.0,
    }


def build_source_manifest(
    selected: Sequence[SelectedTarget],
    *,
    primary_certificate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build canonical C5 rows with published labels isolated from prompts."""

    certified = primary_certificate.get("certified") is True
    threshold = primary_certificate.get("threshold")
    rows = [
        _source_manifest_row(
            target,
            certified=certified,
            threshold=threshold if isinstance(threshold, (int, float)) else None,
        )
        for target in sorted(selected, key=lambda item: item.sample.complex_id)
    ]
    return rows


def _source_manifest_row(
    target: SelectedTarget,
    *,
    certified: bool,
    threshold: float | None,
) -> dict[str, Any]:
    sample = target.sample
    can_trust = certified and threshold is not None and sample.iptm >= threshold
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
    calibration_status = (
        CalibrationStatus.CALIBRATED
        if certified
        else CalibrationStatus.UNCALIBRATED
    )
    input_id = f"c5-source-af3::{sample.complex_id}"
    threshold_id = (
        f"hoeffding_abag_alpha_{C5_PRIMARY_ALPHA:.2f}_v1"
        if certified
        else None
    )
    return {
        "case_id": input_id,
        "dataset": C5_SOURCE_DATASET,
        "model_visible_task": {
            "input_id": input_id,
            "claim": (
                "Select a deployment action for this antibody-antigen "
                "structure prediction after inspecting its confidence and "
                "calibration metadata."
            ),
            "complex_id": sample.complex_id,
            "chain_role_mapping": _chain_role_mapping(sample),
            "interaction_regime": "antibody_antigen",
            "allowed_tools": list(C5_ALLOWED_TOOLS),
            "allowed_terminal_actions": list(C5_ALLOWED_ACTIONS),
            "specialist_result": {
                "specialist_name": "alphafold3",
                "metric_type": "predicted_interface_tm_score",
                "metric_scope": "antibody_antigen_interface",
                "metric_value": sample.iptm,
                "ranking_metric_type": "ranking_confidence",
                "ranking_metric_value": sample.ranking_confidence,
            },
            "general_calibration": {
                "dataset_id": "generic_fixed_threshold_not_source_fitted_v1",
                "regime": "generic_protein_interface",
                "threshold_id": "generic_iptm_0_80_v1",
                "threshold_value": C5_GENERIC_IPTM_THRESHOLD,
            },
            "calibration": {
                "status": calibration_status.value,
                "dataset_id": (
                    "fromm_abag_af3_target_grouped_calibration_v1"
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
            "case_family": "source_backed_af3_confidence_routing",
            "split_group": f"complex::{sample.complex_id}",
            "split": target.split,
            "interface_label_status": (
                "interface_success"
                if target.success
                else "interface_failure"
            ),
            "interface_label_source": "fromm_2026_abag_dockq_ge_0_23",
            "selected_sample_id_sha256": hashlib.sha256(
                f"fromm-af3-v1::{sample.sample_id}".encode()
            ).hexdigest(),
            "top_ranking_tie_count": target.top_ranking_tie_count,
        },
        "source_provenance": {
            "paper_doi": FROMM_PAPER_DOI,
            "archive_doi": FROMM_ZENODO_DOI,
            "license": FROMM_LICENSE,
            "archive_file": FROMM_ARCHIVE_FILE,
            "source_member": FROMM_SOURCE_MEMBER,
            "source_sha256": FROMM_AF3_SHA256,
            "upstream_commit": FROMM_UPSTREAM_COMMIT,
        },
        "cost_profile": {
            "inspect_specialist_record": 1.0,
            "lookup_calibration_card": 1.0,
            "request_structure_verification": 5.0,
            "run_cheap_interface_baseline": 2.0,
            "defer_or_request_more_evidence": 0.5,
        },
    }


def validate_source_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int = 110,
    expected_calibration_rows: int = 55,
) -> list[str]:
    """Validate source-backed rows without requiring synthetic action balance."""

    issues: list[str] = []
    if len(rows) != expected_rows:
        issues.append(f"manifest_row_count:{len(rows)}!={expected_rows}")

    case_ids: list[str] = []
    complex_ids: list[str] = []
    split_groups: list[str] = []
    split_counts: Counter[str] = Counter()
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
        split = str(hidden.get("split", ""))
        case_ids.append(case_id)
        complex_ids.append(complex_id)
        split_groups.append(split_group)
        split_counts[split] += 1

        if row.get("dataset") != C5_SOURCE_DATASET:
            issues.append(f"{prefix}:dataset_mismatch")
        if case_id != visible.get("input_id"):
            issues.append(f"{prefix}:case_input_id_mismatch")
        if not complex_id:
            issues.append(f"{prefix}:complex_id_missing")
        if visible.get("interaction_regime") != "antibody_antigen":
            issues.append(f"{prefix}:interaction_regime_mismatch")
        if tuple(visible.get("allowed_tools", ())) != C5_ALLOWED_TOOLS:
            issues.append(f"{prefix}:allowed_tools_contract_mismatch")
        if tuple(visible.get("allowed_terminal_actions", ())) != C5_ALLOWED_ACTIONS:
            issues.append(f"{prefix}:allowed_actions_contract_mismatch")

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

        roles = visible.get("chain_role_mapping")
        if not isinstance(roles, list):
            issues.append(f"{prefix}:chain_role_mapping_not_list")
        else:
            role_names = {
                item.get("role")
                for item in roles
                if isinstance(item, Mapping)
            }
            if "antibody_heavy" not in role_names or "antigen" not in role_names:
                issues.append(f"{prefix}:chain_roles_incomplete")

        calibration = visible.get("calibration")
        if not isinstance(calibration, Mapping):
            issues.append(f"{prefix}:calibration_not_mapping")
        elif calibration.get("certified") is not True:
            if hidden.get("expected_terminal_action") != (
                Action.VERIFY_WITH_ASSAY_OR_DATABASE.value
            ):
                issues.append(f"{prefix}:uncertified_row_not_fail_closed")

        try:
            trajectory = trajectory_from_c5_row(
                row,
                hidden["expected_terminal_action"],
                fail_closed_reason=(
                    None
                    if hidden["expected_terminal_action"]
                    == Action.TRUST_SPECIALIST_OUTPUT.value
                    else "source_calibration_not_certified"
                ),
            )
            result = score_c5_trajectory(row, trajectory)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{prefix}:canonical_projection_failed:{type(exc).__name__}")
        else:
            if not result.passed:
                issues.append(
                    f"{prefix}:canonical_trajectory_failed:"
                    + ",".join(result.violations)
                )

        privacy_issues = public_artifact_issues(row)
        issues.extend(f"{prefix}:{issue}" for issue in privacy_issues)

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

    expected_evaluation_rows = expected_rows - expected_calibration_rows
    if split_counts["calibration"] != expected_calibration_rows:
        issues.append(
            "calibration_split_count:"
            f"{split_counts['calibration']}!={expected_calibration_rows}"
        )
    if split_counts["evaluation"] != expected_evaluation_rows:
        issues.append(
            "evaluation_split_count:"
            f"{split_counts['evaluation']}!={expected_evaluation_rows}"
        )
    unexpected_splits = sorted(set(split_counts) - {"calibration", "evaluation"})
    if unexpected_splits:
        issues.append(f"unexpected_splits:{','.join(unexpected_splits)}")
    return issues


def build_source_pilot(
    source_path: str | Path,
    *,
    contract: SourceContract = CANONICAL_SOURCE_CONTRACT,
    calibration_targets: int = 55,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the no-model C5 source intake and deterministic policy pilot."""

    samples, intake = intake_source_scores(source_path, contract=contract)
    selected = select_target_samples(
        samples,
        calibration_targets=calibration_targets,
    )
    calibration = [row for row in selected if row.split == "calibration"]
    evaluation = [row for row in selected if row.split == "evaluation"]

    certificates = {
        f"alpha_{alpha:.2f}": select_hoeffding_threshold(
            calibration,
            alpha=alpha,
        )
        for alpha in (0.10, 0.20, C5_PRIMARY_ALPHA)
    }
    primary_certificate = certificates[f"alpha_{C5_PRIMARY_ALPHA:.2f}"]
    rows = build_source_manifest(
        selected,
        primary_certificate=primary_certificate,
    )
    manifest_issues = validate_source_manifest(
        rows,
        expected_rows=contract.expected_targets,
        expected_calibration_rows=calibration_targets,
    )

    primary_threshold = primary_certificate["threshold"]
    report = {
        "dataset": "c5_abag_af3_source_backed_pilot_eval_v1",
        "evaluation_scope": "published_label_replay_target_grouped",
        "source": {
            "paper_doi": FROMM_PAPER_DOI,
            "archive_doi": FROMM_ZENODO_DOI,
            "archive_url": FROMM_ZENODO_RECORD,
            "license": FROMM_LICENSE,
            "archive_file": FROMM_ARCHIVE_FILE,
            "archive_md5": FROMM_ARCHIVE_MD5,
            "archive_member_byte_identity_verified": True,
            "archive_member_verification_date": "2026-07-25",
            "upstream_commit": FROMM_UPSTREAM_COMMIT,
            "source_member": FROMM_SOURCE_MEMBER,
            **intake,
        },
        "selection": {
            "rule": (
                "max_ranking_confidence_then_lexical_sample_id_tiebreak_"
                "with_id_hashed_on_export"
            ),
            "selected_targets": len(selected),
            "targets_with_top_score_ties": sum(
                row.top_ranking_tie_count > 1 for row in selected
            ),
            "selected_interface_successes": sum(row.success for row in selected),
            "selected_interface_success_rate": round(
                sum(row.success for row in selected) / len(selected),
                6,
            ),
            "dockq_success_threshold": C5_DOCKQ_SUCCESS_THRESHOLD,
        },
        "split": {
            "method": "sha256_target_group_sort",
            "salt": C5_SPLIT_SALT,
            "calibration_targets": len(calibration),
            "evaluation_targets": len(evaluation),
            "target_overlap": 0,
            "calibration_successes": sum(row.success for row in calibration),
            "evaluation_successes": sum(row.success for row in evaluation),
        },
        "policies": {
            "trust_all": policy_metrics(evaluation, threshold=None),
            "generic_fixed_iptm_0_80": policy_metrics(
                evaluation,
                threshold=C5_GENERIC_IPTM_THRESHOLD,
            ),
            "regime_specific_hoeffding": {
                **policy_metrics(
                    evaluation,
                    threshold=(
                        float(primary_threshold)
                        if isinstance(primary_threshold, (int, float))
                        else math.inf
                    ),
                ),
                "alpha": C5_PRIMARY_ALPHA,
                "delta": C5_DELTA,
                "certified": primary_certificate["certified"],
                "threshold": primary_threshold,
            },
            "fail_closed": {
                **policy_metrics(evaluation, threshold=math.inf),
                "action": Action.VERIFY_WITH_ASSAY_OR_DATABASE.value,
            },
            "free_form_llm": {
                "status": "not_run",
                "reason": "no_saved_source_backed_model_outputs",
            },
            "general_ppi_transfer": {
                "status": "deferred",
                "reason": (
                    "no_public_full_per_sample_ppi_and_abag_confidence_table_"
                    "validated_for_intake"
                ),
            },
        },
        "calibration": {
            "method": "uniform_hoeffding_union_bound",
            "candidate_thresholds": {
                "minimum": min(C5_CANDIDATE_THRESHOLDS),
                "maximum": max(C5_CANDIDATE_THRESHOLDS),
                "step": 0.01,
                "count": len(C5_CANDIDATE_THRESHOLDS),
            },
            "certificates": certificates,
        },
        "manifest": {
            "rows": len(rows),
            "validation_issues": manifest_issues,
            "hidden_label_isolation": not any(
                "hidden_" in issue for issue in manifest_issues
            ),
        },
        "decision": {
            "source_backed_pilot_passed": not manifest_issues,
            "regime_specific_trust_certified": primary_certificate["certified"],
            "ready_for_c5_model_training": False,
            "ready_for_dpo_rlvr": False,
            "next_ticket": (
                "obtain_validated_general_ppi_per_sample_confidence_or_expand_"
                "independent_abag_calibration"
            ),
        },
        "scientific_boundary": {
            "independent_hidden_test_claimed": False,
            "published_labels_used": True,
            "new_structure_prediction_run": False,
            "llm_or_api_used": False,
            "general_ppi_transfer_claimed": False,
            "source_sample_ids_published": False,
            "raw_paths_or_sequences_published": False,
        },
    }
    report_issues = public_artifact_issues(report)
    if report_issues:
        report["manifest"]["validation_issues"].extend(
            f"report:{issue}" for issue in report_issues
        )
        report["decision"]["source_backed_pilot_passed"] = False
    return rows, report


def public_artifact_issues(value: Any) -> list[str]:
    """Find path breadcrumbs or prohibited raw-source fields recursively."""

    issues: list[str] = []
    for text in _iter_strings(value):
        lowered = text.lower()
        for fragment in PUBLIC_FORBIDDEN_FRAGMENTS:
            if fragment in lowered:
                issues.append(f"public_forbidden_fragment:{fragment}")
    return sorted(set(issues))


def _chain_role_mapping(sample: SourceSample) -> list[dict[str, str]]:
    mapping: list[dict[str, str]] = []
    for chain in sample.heavy_chains:
        mapping.append({"chain_id": chain, "role": "antibody_heavy"})
    for chain in sample.light_chains:
        mapping.append({"chain_id": chain, "role": "antibody_light"})
    for chain in sample.antigen_chains:
        mapping.append({"chain_id": chain, "role": "antigen"})
    return mapping


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


def _looks_like_absolute_path(value: str | None) -> int:
    if not isinstance(value, str):
        return 0
    stripped = value.strip()
    return int(
        stripped.startswith(("/", "~/"))
        or (
            len(stripped) > 2
            and stripped[1] == ":"
            and stripped[2] in {"/", "\\"}
        )
    )


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


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)
