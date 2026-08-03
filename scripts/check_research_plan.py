#!/usr/bin/env python3
"""Validate the active research plan against drift-critical checkpoints."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research" / "2026-06-25_posttrain_tool_use_landscape" / "LONG_TERM_RESEARCH_PLAN_2026-07-04.md"
ROADMAP = ROOT / "ROADMAP.md"
README = ROOT / "README.md"
PUBLIC_STATUS = ROOT / "STATUS.md"
C5_SOURCE_REPORT = (
    ROOT
    / "c5_antibody_ood"
    / "c5_source_backed_pilot_result_2026-07-25.json"
)
C5_PROVENANCE = (
    ROOT / "c5_antibody_ood" / "SOURCE_BACKED_PILOT_PROVENANCE.md"
)
C5_INDEPENDENT_REPORT = (
    ROOT
    / "c5_antibody_ood"
    / "c5_gray_independent_calibration_result_2026-07-25.json"
)
C5_INDEPENDENT_PROVENANCE = (
    ROOT / "c5_antibody_ood" / "INDEPENDENT_CALIBRATION_PROVENANCE.md"
)
C5_PROSPECTIVE_PREREGISTRATION = (
    ROOT
    / "c5_antibody_ood"
    / "c5_prospective_panel_preregistration_v1.json"
)
C5_PROSPECTIVE_SOURCE_AUDIT = (
    ROOT
    / "c5_antibody_ood"
    / "c5_sabdab2_prospective_source_audit_2026-07-25.json"
)
C5_PROSPECTIVE_PANEL_COMMITMENT = (
    ROOT
    / "c5_antibody_ood"
    / "c5_sabdab2_prospective_panel_commitment_v1.json"
)
C5_PROSPECTIVE_INPUT_FREEZE = (
    ROOT
    / "c5_antibody_ood"
    / "c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json"
)
C5_PROSPECTIVE_PREREGISTRATION_V2 = (
    ROOT
    / "c5_antibody_ood"
    / "c5_prospective_panel_preregistration_v2.json"
)
C5_PROSPECTIVE_SOURCE_AUDIT_V2 = (
    ROOT
    / "c5_antibody_ood"
    / "c5_sabdab2_prospective_source_audit_2026-08-02_v2.json"
)
C5_PROSPECTIVE_PANEL_COMMITMENT_V2 = (
    ROOT
    / "c5_antibody_ood"
    / "c5_sabdab2_prospective_panel_commitment_v2.json"
)
C5_PROSPECTIVE_INPUT_FREEZE_V2 = (
    ROOT
    / "c5_antibody_ood"
    / "c5_sabdab2_prospective_af3_input_freeze_2026-08-02_v2.json"
)
C5_AF3_READINESS = (
    ROOT
    / "c5_antibody_ood"
    / "c5_af3_environment_readiness_2026-07-25.json"
)
C5_AF3_CONTAINER_READINESS = (
    ROOT
    / "c5_antibody_ood"
    / "c5_af3_container_readiness_2026-07-25.json"
)
C5_AF3_DATABASE_READINESS = (
    ROOT
    / "c5_antibody_ood"
    / "c5_af3_database_readiness_2026-07-26.json"
)
C5_AF3_ARRAY = ROOT / "c5_antibody_ood" / "run_c5_af3_cayuga.sbatch"
C5_AF3_DATA_PIPELINE_ARRAY = (
    ROOT / "c5_antibody_ood" / "run_c5_af3_data_pipeline_cayuga.sbatch"
)
C5_AF3_INFERENCE_ARRAY = (
    ROOT / "c5_antibody_ood" / "run_c5_af3_inference_cayuga.sbatch"
)
C5_AF3_PARAMETER_JOB = (
    ROOT / "c5_antibody_ood" / "provision_c5_af3_parameters_cayuga.sbatch"
)
C5_AF3_OFFICIAL_PARAMETER_JOB = (
    ROOT / "c5_antibody_ood" / "fetch_c5_af3_parameters_cayuga.sbatch"
)
C5_AF3_ATTESTATION_JOB = (
    ROOT / "c5_antibody_ood" / "attest_c5_af3_runtime_cayuga.sbatch"
)
C5_AF3_PHASE_OUTPUTS = (
    ROOT / "c5_antibody_ood" / "af3_phase_outputs.py"
)
C5_PROSPECTIVE_PREDICTIONS = (
    ROOT / "c5_antibody_ood" / "prospective_predictions.py"
)
C5_PROSPECTIVE_NATIVE_LOCK = (
    ROOT / "c5_antibody_ood" / "prospective_native_lock.py"
)
C5_PROSPECTIVE_REVEAL = (
    ROOT / "c5_antibody_ood" / "prospective_reveal.py"
)
PROSPECTIVE_PROTOCOL_SHA256 = (
    "9c3fd6784fecef3b8971daedb8bfbfc3a1ca725f0353e05f0b7420a30f06e17a"
)
PROSPECTIVE_INPUT_SET_SHA256 = (
    "3569fc8641613c5328a05c991942a576f9ba1f9ad24daf135ea2b62806a52b18"
)


def read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise RuntimeError(f"Could not read {path.relative_to(ROOT)}: {exc}") from exc


def require_contains(issues: list[str], text: str, needle: str, label: str) -> None:
    if needle not in text:
        issues.append(f"missing {label}: {needle!r}")


def require_pattern(issues: list[str], text: str, pattern: str, label: str) -> None:
    if not re.search(pattern, text, flags=re.DOTALL):
        issues.append(f"missing {label}: /{pattern}/")


def main() -> int:
    issues: list[str] = []
    plan = read(PLAN)
    roadmap = read(ROADMAP)
    readme = read(README)
    public_status = read(PUBLIC_STATUS)
    c5_provenance = read(C5_PROVENANCE)
    c5_report = json.loads(read(C5_SOURCE_REPORT))
    c5_independent_provenance = read(C5_INDEPENDENT_PROVENANCE)
    c5_independent_report = json.loads(read(C5_INDEPENDENT_REPORT))
    c5_preregistration = json.loads(read(C5_PROSPECTIVE_PREREGISTRATION))
    c5_source_audit = json.loads(read(C5_PROSPECTIVE_SOURCE_AUDIT))
    c5_panel_commitment = json.loads(read(C5_PROSPECTIVE_PANEL_COMMITMENT))
    c5_input_freeze = json.loads(read(C5_PROSPECTIVE_INPUT_FREEZE))
    c5_preregistration_v2 = json.loads(
        read(C5_PROSPECTIVE_PREREGISTRATION_V2)
    )
    c5_source_audit_v2 = json.loads(read(C5_PROSPECTIVE_SOURCE_AUDIT_V2))
    c5_panel_commitment_v2 = json.loads(
        read(C5_PROSPECTIVE_PANEL_COMMITMENT_V2)
    )
    c5_input_freeze_v2 = json.loads(read(C5_PROSPECTIVE_INPUT_FREEZE_V2))
    c5_af3_readiness = json.loads(read(C5_AF3_READINESS))
    c5_af3_container_readiness = json.loads(
        read(C5_AF3_CONTAINER_READINESS)
    )
    c5_af3_database_readiness = json.loads(
        read(C5_AF3_DATABASE_READINESS)
    )
    c5_af3_array = read(C5_AF3_ARRAY)
    c5_af3_data_pipeline_array = read(C5_AF3_DATA_PIPELINE_ARRAY)
    c5_af3_inference_array = read(C5_AF3_INFERENCE_ARRAY)
    c5_af3_parameter_job = read(C5_AF3_PARAMETER_JOB)
    c5_af3_official_parameter_job = read(C5_AF3_OFFICIAL_PARAMETER_JOB)
    c5_af3_attestation_job = read(C5_AF3_ATTESTATION_JOB)
    c5_af3_phase_outputs = read(C5_AF3_PHASE_OUTPUTS)
    c5_predictions = read(C5_PROSPECTIVE_PREDICTIONS)
    c5_native_lock = read(C5_PROSPECTIVE_NATIVE_LOCK)
    c5_reveal = read(C5_PROSPECTIVE_REVEAL)

    require_contains(
        issues,
        plan,
        "## Research-First 6-8 Week Execution Board",
        "research-first execution board",
    )
    require_pattern(
        issues,
        plan,
        r"Week 1.*enum_action.*Week 2.*tool_query.*routing_after_loop",
        "component experiment order",
    )
    require_contains(
        issues,
        plan,
        "DPO/RLVR remains gated until all three slices have held-out reports.",
        "DPO/RLVR component gate",
    )
    require_contains(
        issues,
        plan,
        "--decode-mode enum_candidate_score",
        "enum candidate-scoring repair path",
    )
    require_pattern(
        issues,
        plan,
        r"Do not repeat or tune on the completed 25-row sealed set\. Keep DPO, RLVR, and\s+Hugging Face publication closed until a learned routing repair beats static\s+baselines, adds useful decisive coverage, and survives independent evaluation\.",
        "post-sealed method and publication gate",
    )
    for needle in (
        "12 rows are balanced across `trust`, `baseline`, `verify`, and `defer`",
        "oracle and fail-closed trajectories pass 12/12",
        "`trust_all` produces 9 unsafe trusts",
        "groups every sampled prediction by `complex_id`",
    ):
        require_contains(issues, plan, needle, "completed C5 prototype")
    for needle in (
        "22,000 AF3 samples over 110 targets",
        "frozen 55/55 target-group split",
        "trust-all has 28 failures",
        "certifies no trusted set",
    ):
        require_contains(issues, plan, needle, "source-backed C5 checkpoint")
    for needle in (
        "1,565 complete bound predictions",
        "9 PDB IDs representing 11 complex copies",
        "44 antibodies and 53 nanobodies",
        "0/55 trusted and 55/55 routed",
        "The next evidence panel is now locked",
    ):
        require_contains(issues, plan, needle, "independent C5 checkpoint")
    for needle in (
        "`complex_id`",
        "metric type, scope, and value",
        "calibration dataset ID",
        "hidden interface label status",
        "expected terminal action",
    ):
        require_contains(issues, plan, needle, "C5 required field")

    require_contains(
        issues,
        plan,
        "A repeatedly inspected held-out slice must be frozen as development data.",
        "sealed evaluation drift gate",
    )
    require_contains(
        issues,
        plan,
        "source-separated sealed extension with private row-level labels",
        "sealed extension privacy gate",
    )

    for needle in (
        "### 1. Stage A Component Smoke Results",
        "### 4. Audited RLVR Gate",
        "### 6. Release v0.1 And Hugging Face Package",
        "python scripts/check_research_plan.py",
    ):
        require_contains(issues, roadmap, needle, "roadmap research-first milestone")

    require_contains(
        issues,
        readme,
        "python scripts/check_research_plan.py",
        "README quickstart research-plan check",
    )
    require_contains(
        issues,
        public_status,
        "stage_b_c5_v3_reattestation_and_remaining_cpu_pipeline",
        "public STATUS C5 research decision",
    )
    require_contains(
        issues,
        public_status,
        "v3_inputs_refrozen_runtime_reattestation_pending",
        "public STATUS v3 workflow state",
    )
    require_pattern(
        issues,
        public_status,
        r"[Dd]o not tune on or rescore these\s+25\s+sealed\s+rows",
        "public STATUS sealed-set reuse prohibition",
    )
    require_contains(
        issues,
        public_status,
        "stage_a_sealed_extension_commitment_2026-07-10.json",
        "public STATUS sealed commitment artifact",
    )
    require_contains(
        issues,
        roadmap,
        "aggregate balance/overlap counts plus cryptographic commitments",
        "roadmap sealed commitment boundary",
    )
    for needle in (
        "official antibody-and-antigen sequence-aware split",
        "150/150 native structures pass",
        "120 template-free AF3 inputs",
        "cannot support 0.10",
        "generation-pinned object in private Cayuga storage",
        "submit the 120-target Cayuga CPU data-pipeline array",
    ):
        require_contains(issues, plan, needle, "prospective C5 freeze/next gate")
    for needle in (
        "10.5281/zenodo.17978681",
        "CC-BY-4.0",
        "56259a84f1e8cc216e5ee91a96584f824ca46f062ef4f2c06aa4674472daf1c8",
        "132,000 absolute compute-path",
    ):
        require_contains(issues, c5_provenance, needle, "C5 provenance")

    expected_report_values = {
        "source.rows": c5_report["source"].get("rows") == 22_000,
        "source.targets": c5_report["source"].get("targets") == 110,
        "split.calibration": c5_report["split"].get("calibration_targets") == 55,
        "split.evaluation": c5_report["split"].get("evaluation_targets") == 55,
        "trust_all.failures": (
            c5_report["policies"]["trust_all"].get("failures_among_trusted")
            == 28
        ),
        "fixed_gate.failures": (
            c5_report["policies"]["generic_fixed_iptm_0_80"].get(
                "failures_among_trusted"
            )
            == 3
        ),
        "regime_gate.not_certified": (
            c5_report["decision"].get("regime_specific_trust_certified")
            is False
        ),
        "pilot.passed": (
            c5_report["decision"].get("source_backed_pilot_passed") is True
        ),
    }
    for label, passed in expected_report_values.items():
        if not passed:
            issues.append(f"C5 source report invariant failed: {label}")

    for needle in (
        "10.5281/zenodo.16426003",
        "749933edc2b7b5f841f453a667bd2204d3e31e56",
        "c012928f1bd36ac255a43b6a3abc33d4f59033b97f6655d9b7c300850e0c433b",
        "9 overlapping PDB",
        "11 Gray complex",
        "44 antibodies",
        "53 nanobodies",
    ):
        require_contains(
            issues,
            c5_independent_provenance,
            needle,
            "independent C5 provenance",
        )

    independent_selection = c5_independent_report["overlap_and_selection"]
    independent_decision = c5_independent_report["decision"]
    independent_transfer = c5_independent_report["locked_fromm_evaluation"]
    expected_independent_values = {
        "source.rows": c5_independent_report["source"].get("rows") == 1_900,
        "source.bound_rows": (
            c5_independent_report["source"].get("bound_rows_retained") == 1_565
        ),
        "source.bound_targets": (
            c5_independent_report["source"].get("bound_targets_retained") == 108
        ),
        "overlap.pdb_ids": (
            independent_selection.get("overlapping_pdb_ids_excluded") == 9
        ),
        "overlap.complexes": (
            independent_selection.get("overlapping_complexes_excluded") == 11
        ),
        "selection.retained": (
            independent_selection.get("selected_after_overlap") == 97
        ),
        "antibody.not_certified": (
            independent_decision.get("antibody_ranking_gate_certified") is False
        ),
        "nanobody.not_certified": (
            independent_decision.get("nanobody_ranking_gate_certified") is False
        ),
        "transfer.disabled": (
            independent_decision.get("external_trust_enabled") is False
        ),
        "transfer.trusted_zero": (
            independent_transfer["independent_calibration_gate"].get("trusted")
            == 0
        ),
    }
    for label, passed in expected_independent_values.items():
        if not passed:
            issues.append(f"C5 independent report invariant failed: {label}")

    prospective_expected_values = {
        "prereg.protocol_sha": (
            c5_preregistration["commitment"].get("protocol_sha256")
            == PROSPECTIVE_PROTOCOL_SHA256
        ),
        "prereg.workflow_state": (
            c5_preregistration.get("workflow_state")
            == "panel_locked_prediction_pending"
        ),
        "prereg.no_hidden_test_claim": (
            c5_preregistration["protocol"]["evidence_boundary"].get(
                "independent_hidden_test_claimed"
            )
            is False
        ),
        "prereg.no_evaluation_tuning": (
            c5_preregistration["protocol"]["evidence_boundary"].get(
                "threshold_tuning_on_evaluation_allowed"
            )
            is False
        ),
        "prereg.no_certificate_action": (
            c5_preregistration["protocol"]["risk_control"].get(
                "no_certificate_action"
            )
            == "verify_all"
        ),
        "source.rows": c5_source_audit["source"].get("rows") == 15_641,
        "source.eligible": (
            c5_source_audit["eligibility"].get("retained_rows") == 2_417
        ),
        "source.selected": (
            c5_source_audit["selection"].get("selected_rows") == 150
        ),
        "source.no_cluster_overlap": (
            c5_source_audit["selection"].get(
                "selected_source_cluster_overlap"
            )
            == 0
        ),
        "source.no_labels_read": (
            c5_source_audit["privacy"].get("dockq_values_read") is False
            and c5_source_audit["privacy"].get(
                "native_interface_labels_read"
            )
            is False
        ),
        "panel.no_prior_overlap": (
            c5_panel_commitment["panel"].get("blocked_pdb_overlap") == 0
        ),
        "panel.roles": (
            c5_panel_commitment["panel"].get("roles")
            == {
                "calibration": 80,
                "calibration_reserve": 20,
                "evaluation": 40,
                "evaluation_reserve": 10,
            }
        ),
        "input.qc_passed": (
            c5_input_freeze["structure_qc"].get("passed") == 150
        ),
        "input.retained": (
            c5_input_freeze["retention"].get("rows") == 120
        ),
        "input.set_sha": (
            c5_input_freeze["af3_inputs"].get("af3_input_set_sha256")
            == PROSPECTIVE_INPUT_SET_SHA256
        ),
        "input.no_training": (
            c5_input_freeze["decision"].get("ready_for_model_training")
            is False
        ),
        "environment.not_ready": (
            c5_af3_readiness.get("ready_for_af3_prediction") is False
        ),
        "environment.source_ready": (
            c5_af3_readiness["components"].get("source_commit_matches")
            is True
            and c5_af3_readiness["components"].get("source_tag_matches")
            is True
        ),
        "environment.inputs_ready": (
            c5_af3_readiness["components"].get("input_set_complete") is True
            and c5_af3_readiness["components"].get(
                "input_set_checksum_matches"
            )
            is True
        ),
        "environment.dependencies_blocked": (
            c5_af3_readiness["components"].get("container_present") is False
            and c5_af3_readiness["components"].get(
                "model_parameters_present"
            )
            is False
            and c5_af3_readiness["components"].get(
                "database_entries_complete"
            )
            is False
        ),
        "environment.no_paths": (
            c5_af3_readiness["release_boundary"].get(
                "local_paths_emitted"
            )
            is False
        ),
        "container.ready": (
            c5_af3_container_readiness["decision"].get("container_ready")
            is True
        ),
        "container.not_prediction_ready": (
            c5_af3_container_readiness["decision"].get(
                "ready_for_af3_prediction"
            )
            is False
        ),
        "container.gpu_smoke": (
            c5_af3_container_readiness["verification"]
            .get("jax_gpu_device_smoke", {})
            .get("status")
            == "pass"
        ),
        "container.no_paths": (
            c5_af3_container_readiness["release_boundary"].get(
                "local_paths_emitted"
            )
            is False
        ),
        "database.ready": (
            c5_af3_database_readiness["decision"].get("database_ready")
            is True
        ),
        "database.not_prediction_ready": (
            c5_af3_database_readiness["decision"].get(
                "ready_for_af3_prediction"
            )
            is False
        ),
        "database.only_parameters_blocked": (
            c5_af3_database_readiness["decision"].get("remaining_blockers")
            == ["authorized_model_parameters_missing"]
        ),
        "database.content_hashed": (
            c5_af3_database_readiness["verification"].get(
                "per_file_content_sha256"
            )
            is True
            and c5_af3_database_readiness["verification"].get(
                "sidecar_checksum_recheck"
            )
            is True
        ),
        "database.complete": (
            c5_af3_database_readiness["summary"].get("required_entries") == 9
            and c5_af3_database_readiness["summary"].get("files") == 195_867
        ),
        "database.no_paths": (
            c5_af3_database_readiness["release_boundary"].get(
                "local_paths_emitted"
            )
            is False
        ),
    }
    for label, passed in prospective_expected_values.items():
        if not passed:
            issues.append(f"C5 prospective invariant failed: {label}")

    prospective_v2_expected_values = {
        "prereg.schema": (
            c5_preregistration_v2.get("schema_version")
            == "c5_prospective_panel_preregistration_v2"
        ),
        "prereg.append_only": (
            c5_preregistration_v2.get("amendment", {}).get("kind")
            == "append_only_pre_prediction_method_amendment"
            and c5_preregistration_v2["amendment"].get(
                "predictions_observed"
            )
            is False
            and c5_preregistration_v2["amendment"].get(
                "calibration_labels_observed"
            )
            is False
            and c5_preregistration_v2["amendment"].get(
                "evaluation_labels_observed"
            )
            is False
        ),
        "prereg.cluster_sampling": (
            c5_preregistration_v2["protocol"]["target_selection"].get(
                "sampling_unit"
            )
            == "official_ab_ag_cluster"
            and c5_preregistration_v2["protocol"]["target_selection"].get(
                "deduplicate_source_cluster"
            )
            is True
        ),
        "prereg.exact_binomial": (
            c5_preregistration_v2["protocol"]["risk_control"].get(
                "certificate_method"
            )
            == "exact_binomial_bonferroni"
            and c5_preregistration_v2["protocol"]["risk_control"].get(
                "sensitivity_certificate_method"
            )
            == "uniform_hoeffding"
        ),
        "source.selected": (
            c5_source_audit_v2["selection"].get("selected_rows") == 144
        ),
        "source.unique_clusters": (
            c5_source_audit_v2["selection"].get(
                "selected_unique_source_clusters"
            )
            == 144
        ),
        "source.roles": (
            c5_source_audit_v2["selection"].get("selected_by_role")
            == {
                "calibration": 80,
                "calibration_reserve": 20,
                "evaluation": 40,
                "evaluation_reserve": 4,
            }
        ),
        "source.no_labels": (
            c5_source_audit_v2["privacy"].get("dockq_values_read") is False
            and c5_source_audit_v2["privacy"].get(
                "native_interface_labels_read"
            )
            is False
        ),
        "panel.valid": (
            c5_panel_commitment_v2["validation"].get("passed") is True
            and c5_panel_commitment_v2["panel"].get(
                "unique_source_clusters"
            )
            == 144
            and c5_panel_commitment_v2["panel"].get(
                "duplicate_source_clusters"
            )
            == 0
        ),
        "input.qc": (
            c5_input_freeze_v2["structure_qc"].get("checked") == 144
            and c5_input_freeze_v2["structure_qc"].get("passed") == 144
            and c5_input_freeze_v2["structure_qc"].get("issues_by_reason")
            == {}
        ),
        "input.retained": (
            c5_input_freeze_v2["retention"].get("rows") == 120
            and c5_input_freeze_v2["retention"].get("retained_by_role")
            == {"calibration": 80, "evaluation": 40}
        ),
        "input.private_boundary": (
            c5_input_freeze_v2["af3_inputs"].get("files") == 120
            and c5_input_freeze_v2["af3_inputs"].get(
                "templates_disabled"
            )
            is True
            and c5_input_freeze_v2["af3_inputs"].get(
                "raw_sequences_emitted_publicly"
            )
            is False
            and c5_input_freeze_v2["structure_qc"].get(
                "dockq_or_interface_labels_read"
            )
            is False
        ),
        "input.fail_closed": (
            c5_input_freeze_v2["decision"].get(
                "ready_for_af3_prediction"
            )
            is True
            and c5_input_freeze_v2["decision"].get(
                "external_specialist_trust_enabled"
            )
            is False
            and c5_input_freeze_v2["decision"].get(
                "ready_for_dpo_rlvr"
            )
            is False
        ),
    }
    for label, passed in prospective_v2_expected_values.items():
        if not passed:
            issues.append(f"C5 prospective v2 invariant failed: {label}")
    for needle in (
        "--output_dir=/root/af_output",
        'TARGET_OUTPUT="${AF3_OUTPUT_DIR}/${JOB_NAME}"',
        "AF3_MODEL_MANIFEST",
        "AF3_DB_MANIFEST",
        "af3_preflight verify-runtime",
        "--benchmark-dir /root/benchmark",
        "--model-manifest /root/c5_model_manifest.json",
        "uv run --no-sync python3",
        "--pwd /app/alphafold",
    ):
        require_contains(
            issues,
            c5_af3_array,
            needle,
            "C5 AF3 official output-root contract",
        )
    require_pattern(
        issues,
        c5_af3_array,
        r"--output_dir=/root/af_output(?!/\$\{JOB_NAME\})",
        "C5 AF3 non-nested output root",
    )
    for needle in (
        "--run_data_pipeline=true",
        "--run_inference=false",
        "c5_antibody_ood.af3_phase_outputs",
        'mv "${STAGE_TARGET}" "${TARGET_OUTPUT}"',
    ):
        require_contains(
            issues,
            c5_af3_data_pipeline_array,
            needle,
            "C5 AF3 CPU data-pipeline contract",
        )
    for needle in (
        "--run_data_pipeline=false",
        "--run_inference=true",
        "--force_output_dir=true",
        "c5_antibody_ood.af3_phase_outputs",
        'mv "${TARGET_OUTPUT}" "${BACKUP_OUTPUT}"',
        'mv "${STAGE_TARGET}" "${TARGET_OUTPUT}"',
    ):
        require_contains(
            issues,
            c5_af3_inference_array,
            needle,
            "C5 AF3 GPU inference contract",
        )
    for needle in (
        "pipeline_job_entry_set_mismatch",
        "pipeline_hidden_label_key_detected",
        "collect_af3_target_output",
        "prediction_scores_emitted",
    ):
        require_contains(
            issues,
            c5_af3_phase_outputs,
            needle,
            "C5 AF3 split-phase output validation",
        )
    for needle in (
        "AF3_AUTHORIZED_SOURCE_CONFIRMED",
        "AF3_SIF_SHA256",
        "provision-model",
        "--authorized-source-confirmed",
        'mv "${STAGE}" "${AF3_MODEL_OUT}"',
    ):
        require_contains(
            issues,
            c5_af3_parameter_job,
            needle,
            "C5 AF3 authorized-parameter provisioning",
        )
    for needle in (
        "AF3_MODEL_TERMS_ACCEPTED",
        "https://storage.googleapis.com/alphafold3/af3.bin.zst",
        "1780568696389861",
        "1020545840",
        "dd1a7badb62cbb0d4571666002159842c8c578c5",
        "--proto '=https'",
        "--proto-redir '=https'",
        "--official-google-storage-download-confirmed",
        "--model-terms-accepted",
        'mv "${SIDECAR_PARTIAL}" "${AF3_MODEL_MANIFEST_OUT}.sha256"',
        'mv "${MANIFEST_PARTIAL}" "${AF3_MODEL_MANIFEST_OUT}"',
    ):
        require_contains(
            issues,
            c5_af3_official_parameter_job,
            needle,
            "C5 AF3 official-parameter download",
        )
    for needle in (
        "AF3_MODEL_MANIFEST_SHA256",
        "AF3_DB_MANIFEST_SHA256",
        "--benchmark-dir /root/benchmark",
        "--model-manifest /root/c5_model_manifest.json",
        "--expected-model-manifest-sha256",
        "--runtime-command uv",
        "af3_preflight verify-runtime",
        "--mode quick",
        'mv "${SIDECAR_PARTIAL}" "${AF3_ATTESTATION_OUT}.sha256"',
        'mv "${ATTESTATION_PARTIAL}" "${AF3_ATTESTATION_OUT}"',
    ):
        require_contains(
            issues,
            c5_af3_attestation_job,
            needle,
            "C5 AF3 runtime-attestation binding",
        )
    for needle in (
        "ranking_score_summary_csv_mismatch",
        "selected_output_rule_mismatch",
        "complete_five_sample_set_per_target",
        "dockq_or_native_interface_labels_read",
        "--benchmark-dir",
    ):
        require_contains(
            issues,
            c5_predictions,
            needle,
            "C5 prediction freeze gate",
        )
    for needle in (
        "native_structure_set_sha256_mismatch",
        "ready_for_staged_label_reveal",
        "dockq_or_interface_labels_read",
    ):
        require_contains(
            issues,
            c5_native_lock,
            needle,
            "C5 native-structure lock gate",
        )
    for needle in (
        "calibration_label_target_set_mismatch",
        "evaluation_label_target_set_mismatch",
        "evaluation_threshold_tuned",
        "calibration_dataset_id",
        "confidence_metric_scope",
        "evaluation_policy_recomputation_mismatch",
    ):
        require_contains(
            issues,
            c5_reveal,
            needle,
            "C5 staged reveal gate",
        )

    if issues:
        print(f"FAIL research plan check found {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("OK research plan check passed")
    print("- component order: enum_action -> tool_query -> routing_after_loop")
    print("- Stage A checkpoint: tool_query 0/5; sealed routing 5/25; runtime oracle 25/25")
    print("- prospective Stage A: routing 35/180; hybrid 115/180; compiler 25/25 clean")
    print("- C5 prototype: fail-closed 12/12; trust-all 9 unsafe trusts")
    print("- C5 v2 freeze: 144/144 unique candidate clusters; 120 retained")
    print("- C5 source replay: trust-all 28/55 failures; fixed gate 3/20")
    print("- C5 certification: no trusted set at alpha <= 0.30")
    print("- C5 independent calibration: 97 targets; 0 residual PDB overlap")
    print("- C5 independent certificates: antibody and nanobody both fail")
    print("- C5 v1 target concentration: prediction stopped before label access")
    print("- C5 phase gates: 600-sample prediction lock and staged 80/40 reveal implemented")
    print("- C5 execution path: staged CPU data pipeline and GPU inference implemented")
    print("- C5 v3 execution gate: fresh input-bound re-attestation pending")
    print("- C5 next gate: resume audit, then remaining CPU data pipeline")
    print("- DPO/RLVR/HF gate: useful routing coverage plus independent evaluation required")
    print("- sealed evaluation gate: completed rows cannot be tuned on or rescored")
    print("- C5 gate: calibration metadata required before trust")
    return 0


if __name__ == "__main__":
    sys.exit(main())
