"""Fail-closed preregistration contract for the prospective Stage B C5 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .source_pilot import C5_CANDIDATE_THRESHOLDS


PREREGISTRATION_SCHEMA = "c5_prospective_panel_preregistration_v1"
PREREGISTRATION_ID = "c5-sabdab2-af3-prospective-2026-07-25-v1"
PREREGISTRATION_SCHEMA_V2 = "c5_prospective_panel_preregistration_v2"
PREREGISTRATION_ID_V2 = "c5-sabdab2-af3-prospective-2026-08-02-v2"
SABDAB2_ARCHIVE_BYTES = 876_381_859
SABDAB2_ARCHIVE_MD5 = "0dbb4cc499e9eb77f14008b232f2c38c"
SABDAB2_SPLIT_BYTES = 140_769_855
SABDAB2_SPLIT_SHA256 = (
    "01414df16af9d3343994f527e13e81a25bf8da08198858018493a96f173a8cfb"
)
AF3_COMMIT = "7b197fe859790fc3e04d03ea70dd0b9ba48881c9"
DOCKQ_COMMIT = "d9cbb1940bb0f42db3257f7da3b0e96f162b94d9"
PANEL_PUBLIC_KEYS = frozenset(
    {
        "target_id",
        "pdb_id",
        "sabdab_id",
        "instance_id",
        "source_split",
        "panel_role",
        "selection_rank",
        "release_date",
        "experimental_method",
        "resolution_angstrom",
        "antibody_format",
        "chain_role_mapping",
        "source_row_sha256",
        "source_cluster_sha256",
    }
)
PANEL_ROLES = (
    "calibration",
    "evaluation",
    "calibration_reserve",
    "evaluation_reserve",
)
PDB_ID_RE = re.compile(r"^[a-z0-9]{4,12}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable canonical representation."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def minimum_zero_failure_trusted(
    *,
    alpha: float,
    delta: float,
    candidate_count: int,
) -> int:
    """Return the minimum trusted count for a zero-failure Hoeffding certificate."""

    return math.ceil(
        math.log(candidate_count / delta) / (2 * alpha * alpha)
    )


def minimum_zero_failure_trusted_exact(
    *,
    alpha: float,
    delta: float,
    candidate_count: int,
) -> int:
    """Return the exact-binomial zero-failure minimum after Bonferroni."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 1
    ):
        raise ValueError("candidate_count must be a positive integer")
    return math.ceil(
        math.log(delta / candidate_count) / math.log(1 - alpha)
    )


def build_preregistration() -> dict[str, Any]:
    """Build and seal the prospective C5 method contract."""

    thresholds = list(C5_CANDIDATE_THRESHOLDS)
    protocol: dict[str, Any] = {
        "research_question": (
            "Can an AlphaFold 3 ranking-score gate earn regime-matched "
            "antibody-antigen trust on a sequence-separated prospective panel?"
        ),
        "evidence_boundary": {
            "independent_hidden_test_claimed": False,
            "prospective_method_lock_claimed": True,
            "native_structures_are_public": True,
            "labels_must_remain_unread_until_prediction_freeze": True,
            "model_training_on_panel_allowed": False,
            "threshold_tuning_on_evaluation_allowed": False,
            "llm_api_or_judge_used": False,
        },
        "source": {
            "dataset": "SAbDab2 Machine Learning Dataset",
            "version": "0.1.0",
            "doi": "10.5281/zenodo.20083995",
            "license": "CC-BY-4.0",
            "archive_file": "splits.tar.gz",
            "archive_bytes": SABDAB2_ARCHIVE_BYTES,
            "archive_md5": SABDAB2_ARCHIVE_MD5,
            "split_file": "abag_split.csv",
            "split_bytes": SABDAB2_SPLIT_BYTES,
            "split_sha256": SABDAB2_SPLIT_SHA256,
            "split_column": "ab_ag_split",
            "split_semantics": (
                "official antibody-and-antigen sequence-aware split"
            ),
        },
        "target_selection": {
            "unit": "one paired-chain antibody-antigen instance per PDB",
            "antibody_format": "paired_chain",
            "antigen_requirement": "one_or_more_protein_antigen_chains",
            "minimum_release_date": "2021-10-01",
            "allowed_experimental_methods": [
                "EM",
                "XRAY",
            ],
            "maximum_resolution_angstrom": 4.0,
            "exclude_all_fromm_pdb_ids": True,
            "exclude_all_gray_pdb_ids": True,
            "deduplicate_pdb_id": True,
            "deduplicate_sabdab_id": True,
            "selection_algorithm": (
                "sha256(public_seed + '|' + source_instance_id), "
                "then lexical source_instance_id"
            ),
            "public_seed": "c5-sabdab2-prospective-v1-2026-07-25",
            "calibration": {
                "source_split": "train",
                "primary_targets": 80,
                "reserve_targets": 20,
            },
            "evaluation": {
                "source_split": "test",
                "primary_targets": 40,
                "reserve_targets": 10,
            },
            "replacement_rule": (
                "promote the next same-split reserve only for a preregistered "
                "metadata, sequence, structure, or infrastructure QC failure"
            ),
        },
        "prediction": {
            "engine": "AlphaFold 3",
            "code_tag": "v3.0.3",
            "code_commit": AF3_COMMIT,
            "model_parameters": "official AlphaFold 3 3.0.x parameters",
            "model_parameter_checksum_required_in_private_run_attestation": True,
            "input_dialect": "alphafold3",
            "input_version": 1,
            "templates": "disabled_for_every_protein_chain",
            "msa_mode": "official_data_pipeline_default",
            "model_seeds": [20260725],
            "diffusion_samples_per_seed": 5,
            "target_sample_selection": [
                "maximum_ranking_score",
                "maximum_iptm_for_exact_ranking_score_ties",
                "lexical_output_id",
            ],
            "compute_order": ["Cayuga", "Expanse"],
            "local_model_compute_allowed": False,
        },
        "label": {
            "evaluator": "DockQ",
            "version": "v2.1.3",
            "code_commit": DOCKQ_COMMIT,
            "metric": "DockQ",
            "metric_scope": "paired_antibody_chains_vs_all_antigen_chains",
            "interface_success_rule": "DockQ >= 0.23",
            "interface_success_threshold": 0.23,
            "native_chain_mapping_must_match_committed_roles": True,
        },
        "risk_control": {
            "confidence_metric": "ranking_score",
            "candidate_thresholds": thresholds,
            "candidate_count": len(thresholds),
            "primary_alpha": 0.30,
            "secondary_alphas": [0.20, 0.10],
            "delta": 0.10,
            "certificate": "uniform Hoeffding upper bound with union correction",
            "threshold_selection": (
                "highest coverage certified threshold; lower threshold wins "
                "coverage ties"
            ),
            "no_certificate_action": "verify_all",
            "minimum_evaluation_trusted_for_transfer_claim": 10,
            "evaluation_upper_bound": (
                "fixed-threshold Hoeffding bound without threshold-search "
                "multiplicity"
            ),
        },
        "phase_gates": [
            {
                "phase": "source_intake",
                "requires": [
                    "archive byte count and MD5 pass",
                    "official split file present",
                    "zero Fromm/Gray PDB overlap",
                    "public panel manifest passes privacy validator",
                ],
            },
            {
                "phase": "prediction",
                "requires": [
                    "80 calibration and 40 evaluation targets are frozen",
                    "source splits and target IDs are disjoint",
                    "no DockQ or native-interface labels were read",
                    "AF3 code and private parameter checksums are recorded",
                ],
            },
            {
                "phase": "label_reveal",
                "requires": [
                    "five AF3 outputs exist for every retained target",
                    "one target-level output is selected and checksum frozen",
                    "prediction artifacts are immutable",
                ],
            },
            {
                "phase": "evaluation_reveal",
                "requires": [
                    "calibration certificate and threshold are checksum frozen",
                    "evaluation labels have not been used for selection",
                ],
            },
        ],
        "stopping_rule": {
            "required_calibration_targets": 80,
            "required_evaluation_targets": 40,
            "maximum_prediction_attempts_per_target": 2,
            "retry_scope": "infrastructure_failure_only_same_locked_input",
            "adaptive_seed_extension_allowed": False,
            "score_or_label_dependent_extension_allowed": False,
            "insufficient_targets_action": "abort_without_trust_claim",
            "completion": (
                "stop after all 120 retained targets have five outputs or "
                "abort when a same-split reserve is exhausted"
            ),
        },
        "release_boundary": {
            "public": [
                "preregistration and protocol digest",
                "PDB/SAbDab target IDs and chain roles",
                "source and panel checksums",
                "aggregate QC, calibration, and evaluation summaries",
            ],
            "private": [
                "raw structures and sequences",
                "AlphaFold parameter files and private paths",
                "raw AF3 outputs and scheduler logs",
                "per-target DockQ labels until the reveal gate",
            ],
        },
    }
    design_analysis = {
        f"alpha_{alpha:.2f}": {
            "alpha": alpha,
            "minimum_zero_failure_trusted": minimum_zero_failure_trusted(
                alpha=alpha,
                delta=protocol["risk_control"]["delta"],
                candidate_count=len(thresholds),
            ),
        }
        for alpha in (
            protocol["risk_control"]["primary_alpha"],
            *protocol["risk_control"]["secondary_alphas"],
        )
    }
    result = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "preregistration_id": PREREGISTRATION_ID,
        "created_date": "2026-07-25",
        "workflow_state": "method_locked_source_intake_pending",
        "protocol": protocol,
        "design_analysis": design_analysis,
    }
    result["commitment"] = {
        "algorithm": "sha256-canonical-json",
        "scope": "protocol",
        "protocol_sha256": canonical_sha256(protocol),
    }
    return result


def build_preregistration_v2() -> dict[str, Any]:
    """Build the label-free, cluster-balanced pre-prediction amendment."""

    v1 = build_preregistration()
    protocol = deepcopy(v1["protocol"])
    selection = protocol["target_selection"]
    selection.update(
        {
            "unit": "one paired-chain instance per ab_ag_cluster",
            "sampling_unit": "official_ab_ag_cluster",
            "cluster_column": "ab_ag_cluster",
            "deduplicate_source_cluster": True,
            "selection_algorithm": (
                "sha256(public_seed + '|cluster|' + source_cluster_sha256), "
                "then sha256(public_seed + '|target|' + "
                "source_cluster_sha256 + '|' + source_instance_id), then "
                "lexical source_instance_id"
            ),
        }
    )
    selection["evaluation"]["reserve_targets"] = 4

    risk = protocol["risk_control"]
    risk.update(
        {
            "certificate": (
                "exact one-sided binomial test with Bonferroni correction"
            ),
            "certificate_method": "exact_binomial_bonferroni",
            "calibration_test": (
                "lower-tail Binomial(n, alpha) test at delta/candidate_count"
            ),
            "evaluation_upper_bound": (
                "fixed-threshold exact one-sided binomial bound without "
                "threshold-search multiplicity"
            ),
            "sensitivity_certificate": (
                "uniform Hoeffding upper bound with union correction"
            ),
            "sensitivity_certificate_method": "uniform_hoeffding",
        }
    )
    protocol["phase_gates"][0]["requires"].append(
        "one target per official ab_ag_cluster across primary and reserves"
    )
    protocol["phase_gates"][1]["requires"].append(
        "80 calibration and 40 evaluation source clusters are unique"
    )

    thresholds = risk["candidate_thresholds"]
    alphas = (
        risk["primary_alpha"],
        *risk["secondary_alphas"],
    )
    design_analysis = {
        "sampling_audit": {
            "v1_calibration_targets": 80,
            "v1_calibration_unique_clusters": 34,
            "v1_calibration_largest_cluster": 35,
            "v1_evaluation_targets": 40,
            "v1_evaluation_unique_clusters": 17,
            "v1_evaluation_largest_cluster": 16,
            "v2_calibration_unique_clusters": 80,
            "v2_evaluation_unique_clusters": 40,
        },
        **{
            f"alpha_{alpha:.2f}": {
                "alpha": alpha,
                "minimum_zero_failure_trusted": (
                    minimum_zero_failure_trusted_exact(
                        alpha=alpha,
                        delta=risk["delta"],
                        candidate_count=len(thresholds),
                    )
                ),
                "hoeffding_sensitivity_minimum_zero_failure_trusted": (
                    minimum_zero_failure_trusted(
                        alpha=alpha,
                        delta=risk["delta"],
                        candidate_count=len(thresholds),
                    )
                ),
            }
            for alpha in alphas
        },
    }
    amendment = {
        "kind": "append_only_pre_prediction_method_amendment",
        "supersedes_preregistration_id": v1["preregistration_id"],
        "supersedes_protocol_sha256": v1["commitment"]["protocol_sha256"],
        "reason": (
            "target-level selection concentrated repeated observations "
            "within official sequence clusters and did not support the "
            "independence interpretation required by the risk certificate"
        ),
        "evidence_used": "public_safe_metadata_and_method_power_only",
        "predictions_observed": False,
        "calibration_labels_observed": False,
        "evaluation_labels_observed": False,
    }
    result = {
        "schema_version": PREREGISTRATION_SCHEMA_V2,
        "preregistration_id": PREREGISTRATION_ID_V2,
        "created_date": "2026-08-02",
        "workflow_state": "method_locked_source_intake_pending",
        "amendment": amendment,
        "protocol": protocol,
        "design_analysis": design_analysis,
    }
    result["commitment"] = {
        "algorithm": "sha256-canonical-json",
        "scope": "protocol",
        "protocol_sha256": canonical_sha256(protocol),
    }
    return result


def validate_preregistration(record: Mapping[str, Any]) -> list[str]:
    """Validate both schema completeness and immutable method choices."""

    issues: list[str] = []
    is_v2 = (
        record.get("schema_version") == PREREGISTRATION_SCHEMA_V2
        or record.get("preregistration_id") == PREREGISTRATION_ID_V2
    )
    expected = build_preregistration_v2() if is_v2 else build_preregistration()
    for key in (
        "schema_version",
        "preregistration_id",
        "created_date",
    ):
        if record.get(key) != expected[key]:
            issues.append(f"{key}_mismatch")
    if record.get("workflow_state") not in {
        "method_locked_source_intake_pending",
        "panel_locked_prediction_pending",
        "predictions_locked_label_reveal_pending",
        "calibration_locked_evaluation_pending",
        "complete",
        "aborted",
    }:
        issues.append("workflow_state_invalid")

    protocol = record.get("protocol")
    if not isinstance(protocol, Mapping):
        return [*issues, "protocol_missing"]
    if protocol != expected["protocol"]:
        issues.append("protocol_drift")

    commitment = record.get("commitment")
    if not isinstance(commitment, Mapping):
        issues.append("commitment_missing")
    else:
        digest = commitment.get("protocol_sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            issues.append("protocol_sha256_invalid")
        elif digest != canonical_sha256(protocol):
            issues.append("protocol_sha256_mismatch")
        if commitment.get("algorithm") != "sha256-canonical-json":
            issues.append("commitment_algorithm_invalid")
        if commitment.get("scope") != "protocol":
            issues.append("commitment_scope_invalid")

    design = record.get("design_analysis")
    if design != expected["design_analysis"]:
        issues.append("design_analysis_drift")
    if is_v2 and record.get("amendment") != expected["amendment"]:
        issues.append("amendment_drift")
    return issues


def validate_public_panel(
    rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    *,
    blocked_pdb_ids: set[str],
    expected_role_counts: Mapping[str, int] | None = None,
) -> list[str]:
    """Validate a public-safe target commitment before any prediction."""

    issues = [
        f"preregistration:{issue}"
        for issue in validate_preregistration(preregistration)
    ]
    if not rows:
        return [*issues, "panel_empty"]

    counts: Counter[str] = Counter()
    seen_targets: set[str] = set()
    seen_pdb: set[str] = set()
    seen_sabdab: set[str] = set()
    seen_rank: set[tuple[str, int]] = set()
    seen_source_clusters: set[str] = set()
    calibration_clusters: set[str] = set()
    evaluation_clusters: set[str] = set()
    blocked = {value.lower() for value in blocked_pdb_ids}
    protocol = preregistration["protocol"]
    selection = protocol["target_selection"]
    expected_counts = (
        dict(expected_role_counts)
        if expected_role_counts is not None
        else {
            "calibration": selection["calibration"]["primary_targets"],
            "calibration_reserve": selection["calibration"][
                "reserve_targets"
            ],
            "evaluation": selection["evaluation"]["primary_targets"],
            "evaluation_reserve": selection["evaluation"]["reserve_targets"],
        }
    )
    split_by_role = {
        "calibration": selection["calibration"]["source_split"],
        "calibration_reserve": selection["calibration"]["source_split"],
        "evaluation": selection["evaluation"]["source_split"],
        "evaluation_reserve": selection["evaluation"]["source_split"],
    }

    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        extra = set(row) - PANEL_PUBLIC_KEYS
        missing = PANEL_PUBLIC_KEYS - set(row)
        if extra:
            issues.append(f"{prefix}:unexpected_keys:{','.join(sorted(extra))}")
        if missing:
            issues.append(f"{prefix}:missing_keys:{','.join(sorted(missing))}")
            continue
        serialized = json.dumps(row, sort_keys=True).lower()
        for forbidden in (
            "sequence",
            "dockq",
            "interface_label",
            "confidence",
            "ranking_score",
            "native_path",
            "output_path",
        ):
            if forbidden in serialized:
                issues.append(f"{prefix}:forbidden_public_field:{forbidden}")

        target_id = row["target_id"]
        pdb_id = row["pdb_id"]
        sabdab_id = row["sabdab_id"]
        role = row["panel_role"]
        source_split = row["source_split"]
        rank = row["selection_rank"]
        if not isinstance(target_id, str) or not target_id:
            issues.append(f"{prefix}:target_id_invalid")
        elif target_id in seen_targets:
            issues.append(f"{prefix}:target_id_duplicate")
        else:
            seen_targets.add(target_id)
        if not isinstance(pdb_id, str) or not PDB_ID_RE.fullmatch(pdb_id):
            issues.append(f"{prefix}:pdb_id_invalid")
        else:
            pdb_id = pdb_id.lower()
            if pdb_id in seen_pdb:
                issues.append(f"{prefix}:pdb_id_duplicate")
            if pdb_id in blocked:
                issues.append(f"{prefix}:blocked_pdb_overlap")
            seen_pdb.add(pdb_id)
        if not isinstance(sabdab_id, str) or not sabdab_id:
            issues.append(f"{prefix}:sabdab_id_invalid")
        elif sabdab_id in seen_sabdab:
            issues.append(f"{prefix}:sabdab_id_duplicate")
        else:
            seen_sabdab.add(sabdab_id)
        if role not in PANEL_ROLES:
            issues.append(f"{prefix}:panel_role_invalid")
        else:
            counts[role] += 1
            if source_split != split_by_role[role]:
                issues.append(f"{prefix}:source_split_role_mismatch")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 1
        ):
            issues.append(f"{prefix}:selection_rank_invalid")
        elif role in PANEL_ROLES and (role, rank) in seen_rank:
            issues.append(f"{prefix}:selection_rank_duplicate")
        else:
            seen_rank.add((role, rank))

        release_date = row["release_date"]
        if (
            not isinstance(release_date, str)
            or release_date < selection["minimum_release_date"]
        ):
            issues.append(f"{prefix}:release_date_before_cutoff")
        if row["experimental_method"] not in selection[
            "allowed_experimental_methods"
        ]:
            issues.append(f"{prefix}:experimental_method_invalid")
        resolution = row["resolution_angstrom"]
        if (
            not isinstance(resolution, (int, float))
            or isinstance(resolution, bool)
            or not math.isfinite(resolution)
            or resolution <= 0
            or resolution > selection["maximum_resolution_angstrom"]
        ):
            issues.append(f"{prefix}:resolution_invalid")
        if row["antibody_format"] != "paired_chain":
            issues.append(f"{prefix}:antibody_format_invalid")
        issues.extend(_chain_role_issues(row["chain_role_mapping"], prefix))
        source_hash = row["source_row_sha256"]
        if not isinstance(source_hash, str) or not SHA256_RE.fullmatch(
            source_hash
        ):
            issues.append(f"{prefix}:source_row_sha256_invalid")
        cluster_hash = row["source_cluster_sha256"]
        if not isinstance(cluster_hash, str) or not SHA256_RE.fullmatch(
            cluster_hash
        ):
            issues.append(f"{prefix}:source_cluster_sha256_invalid")
        else:
            if selection.get("deduplicate_source_cluster") is True:
                if cluster_hash in seen_source_clusters:
                    issues.append(f"{prefix}:source_cluster_duplicate")
                seen_source_clusters.add(cluster_hash)
            if role in {"calibration", "calibration_reserve"}:
                calibration_clusters.add(cluster_hash)
            elif role in {"evaluation", "evaluation_reserve"}:
                evaluation_clusters.add(cluster_hash)

    for role in PANEL_ROLES:
        expected_count = expected_counts.get(role, 0)
        if counts[role] != expected_count:
            issues.append(
                f"panel_role_count:{role}:{counts[role]}!={expected_count}"
            )
    cluster_overlap = calibration_clusters & evaluation_clusters
    if cluster_overlap:
        issues.append(
            f"source_cluster_overlap_between_splits:{len(cluster_overlap)}"
        )
    return issues


def build_panel_commitment(
    rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    *,
    blocked_pdb_ids: set[str],
) -> dict[str, Any]:
    """Build a compact source-intake commitment; never silently accept issues."""

    validation_issues = validate_public_panel(
        rows,
        preregistration,
        blocked_pdb_ids=blocked_pdb_ids,
    )
    counts = Counter(str(row.get("panel_role")) for row in rows)
    panel = {
        "rows": len(rows),
        "roles": dict(sorted(counts.items())),
        "manifest_sha256": canonical_sha256(list(rows)),
        "blocked_pdb_overlap": sum(
            str(row.get("pdb_id", "")).lower()
            in {value.lower() for value in blocked_pdb_ids}
            for row in rows
        ),
        "source_cluster_overlap_between_splits": len(
            {
                str(row.get("source_cluster_sha256"))
                for row in rows
                if row.get("panel_role")
                in {"calibration", "calibration_reserve"}
            }
            & {
                str(row.get("source_cluster_sha256"))
                for row in rows
                if row.get("panel_role")
                in {"evaluation", "evaluation_reserve"}
            }
        ),
    }
    is_v2 = preregistration.get("schema_version") == PREREGISTRATION_SCHEMA_V2
    if is_v2:
        clusters = [
            str(row.get("source_cluster_sha256"))
            for row in rows
        ]
        panel.update(
            {
                "unique_source_clusters": len(set(clusters)),
                "duplicate_source_clusters": len(clusters) - len(set(clusters)),
            }
        )
    return {
        "schema_version": (
            "c5_prospective_panel_commitment_v2"
            if is_v2
            else "c5_prospective_panel_commitment_v1"
        ),
        "preregistration_id": preregistration.get("preregistration_id"),
        "protocol_sha256": preregistration.get("commitment", {}).get(
            "protocol_sha256"
        ),
        "source": {
            "doi": preregistration.get("protocol", {})
            .get("source", {})
            .get("doi"),
            "archive_md5": preregistration.get("protocol", {})
            .get("source", {})
            .get("archive_md5"),
        },
        "panel": panel,
        "validation": {
            "passed": not validation_issues,
            "issues": validation_issues,
        },
        "decision": {
            "ready_for_prediction": not validation_issues,
            "external_specialist_trust_enabled": False,
            "ready_for_model_training": False,
            "ready_for_dpo_rlvr": False,
        },
    }


def _chain_role_issues(value: Any, prefix: str) -> list[str]:
    if not isinstance(value, list) or len(value) < 3:
        return [f"{prefix}:chain_role_mapping_invalid"]
    roles: Counter[str] = Counter()
    chains: set[str] = set()
    issues: list[str] = []
    for mapping in value:
        if not isinstance(mapping, Mapping):
            return [f"{prefix}:chain_role_mapping_invalid"]
        chain_id = mapping.get("chain_id")
        role = mapping.get("role")
        if not isinstance(chain_id, str) or not chain_id:
            issues.append(f"{prefix}:chain_id_invalid")
        elif chain_id in chains:
            issues.append(f"{prefix}:chain_id_duplicate")
        else:
            chains.add(chain_id)
        if role not in {
            "antibody_heavy",
            "antibody_light",
            "antigen",
        }:
            issues.append(f"{prefix}:chain_role_invalid")
        else:
            roles[str(role)] += 1
    if roles["antibody_heavy"] != 1:
        issues.append(f"{prefix}:heavy_chain_count_invalid")
    if roles["antibody_light"] != 1:
        issues.append(f"{prefix}:light_chain_count_invalid")
    if roles["antigen"] < 1:
        issues.append(f"{prefix}:antigen_chain_count_invalid")
    return issues


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-default", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    if bool(args.write_default) == bool(args.check):
        parser.error("choose exactly one of --write-default or --check")
    if args.write_default:
        record = (
            build_preregistration_v2()
            if args.version == "v2"
            else build_preregistration()
        )
        write_json(args.write_default, record)
        print(f"wrote {args.write_default}")
        print(f"protocol_sha256={record['commitment']['protocol_sha256']}")
        return 0
    record = _load_json(args.check)
    issues = validate_preregistration(record)
    print(json.dumps({"issues": issues, "passed": not issues}, sort_keys=True))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
