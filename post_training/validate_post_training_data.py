#!/usr/bin/env python3
"""Validate tracked post-training JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

STAGE_A_SFT_DATASET = "negbiodb_ct_stage_a_sft_v1"
STAGE_A_PREFERENCE_DATASET = "negbiodb_ct_stage_a_preferences_v1"
STAGE_A_PROCESS_DATASET = "negbiodb_ct_stage_a_process_v1"
STAGE_A_SPLIT_DATASET = "negbiodb_ct_stage_a_split_v1"
STAGE_A_STRICT_SFT_DATASET = "negbiodb_ct_stage_a_strict_contract_sft_v1"
STAGE_A_STRICT_PREFERENCE_DATASET = "negbiodb_ct_stage_a_strict_contract_preferences_v1"
STAGE_A_STRICT_PROCESS_DATASET = "negbiodb_ct_stage_a_strict_contract_process_v1"
STAGE_A_STRICT_MANIFEST_DATASET = "negbiodb_ct_stage_a_strict_contract_export_v1"
STAGE_A_STRICT_COMPONENT_DATASET = "negbiodb_ct_stage_a_strict_component_targets_v1"
STAGE_A_STRICT_COMPONENT_MANIFEST_DATASET = "negbiodb_ct_stage_a_strict_component_targets_manifest_v1"
STAGE_A_EVIDENCE_COMPONENT_DATASET = "negbiodb_ct_stage_a_evidence_conditioned_component_targets_v1"
STAGE_A_EVIDENCE_COMPONENT_MANIFEST_DATASET = (
    "negbiodb_ct_stage_a_evidence_conditioned_component_targets_manifest_v1"
)
STAGE_A_EVIDENCE_COMPONENT_PROMPT_CONTRACT = "stage_a_v2_evidence_conditioned_component"
STAGE_A_EVIDENCE_COMPONENTS = ("enum_action", "routing_after_loop")
STAGE_A_ENUM_CORRECTIVE_PAIR_DATASET = "negbiodb_ct_stage_a_enum_corrective_pairs_v1"
STAGE_A_ENUM_CORRECTIVE_MANIFEST_DATASET = "negbiodb_ct_stage_a_enum_corrective_pairs_manifest_v1"
STAGE_A_ENUM_ACTION_CONTRAST_PAIR_DATASET = "negbiodb_ct_stage_a_enum_action_contrast_pairs_v1"
STAGE_A_ENUM_ACTION_CONTRAST_MANIFEST_DATASET = "negbiodb_ct_stage_a_enum_action_contrast_pairs_manifest_v1"
STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_PAIR_DATASET = (
    "negbiodb_ct_stage_a_routing_action_status_contrast_pairs_v1"
)
STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_MANIFEST_DATASET = (
    "negbiodb_ct_stage_a_routing_action_status_contrast_pairs_manifest_v1"
)
STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_PAIR_DATASET = (
    "negbiodb_ct_stage_a_routing_defer_verify_contrast_pairs_v1"
)
STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_MANIFEST_DATASET = (
    "negbiodb_ct_stage_a_routing_defer_verify_contrast_pairs_manifest_v1"
)
STAGE_A_SAVED_OUTPUT_CALIBRATION_PROBE_PAIR_DATASET = (
    "negbiodb_ct_stage_a_saved_output_calibration_probe_v1"
)
STAGE_A_SAVED_OUTPUT_CALIBRATION_PROBE_MANIFEST_DATASET = (
    "negbiodb_ct_stage_a_saved_output_calibration_probe_manifest_v1"
)
STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_DATASET = (
    "negbiodb_ct_stage_a_saved_output_evidence_candidate_routing_v1"
)
STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_MANIFEST_DATASET = (
    "negbiodb_ct_stage_a_saved_output_evidence_candidate_routing_manifest_v1"
)
STAGE_A_STRICT_PROMPT_CONTRACT = "stage_a_v2_strict"
STAGE_A_STRICT_COMPONENTS = (
    "enum_action",
    "tool_query",
    "routing_after_loop",
)
STAGE_A_STRICT_FAILURE_MODES = (
    "full_loop_verify_supported",
    "observed_single_tool_verify_supported",
)
STAGE_A_STRICT_COMPONENT_TARGET_KEYS = {
    "enum_action": ("action", "evidence_status"),
    "tool_query": ("tool_calls",),
    "routing_after_loop": ("action", "evidence_status", "cited_source_ids"),
}
STAGE_A_ENUM_CORRECTIVE_FAILURE_MODE = "ground_supported_collapse"
STAGE_A_ENUM_CORRECTIVE_REJECTED_OUTPUT = {
    "action": "ground",
    "evidence_status": "supported",
}
STAGE_A_ENUM_ACTION_CONTRAST_FAILURE_MODE = "same_status_wrong_action_contrast"
STAGE_A_ENUM_ACTION_CONTRAST_CANDIDATE_POLICY = "same_status_action_contrast"
STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_FAILURE_MODE = "routing_action_status_confusion"
STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_CANDIDATE_POLICY = "observed_constrained_routing_confusion"
STAGE_A_ROUTING_ACTION_STATUS_REJECTED_BY_CHOSEN_PAIR = {
    "defer/insufficient": {"action": "reject", "evidence_status": "contradicted"},
    "verify/insufficient": {"action": "reject", "evidence_status": "contradicted"},
    "flag/invalid_value": {"action": "ground", "evidence_status": "supported"},
}
STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_FAILURE_MODE = "routing_defer_verify_boundary_confusion"
STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_CANDIDATE_POLICY = "insufficient_defer_vs_verify_boundary"
STAGE_A_ROUTING_DEFER_VERIFY_REJECTED_BY_CHOSEN_PAIR = {
    "defer/insufficient": {"action": "verify", "evidence_status": "insufficient"},
    "verify/insufficient": {"action": "defer", "evidence_status": "insufficient"},
}
STAGE_A_SAVED_OUTPUT_CALIBRATION_PROMPT_CONTRACT = "stage_a_v4_canonical_json"
STAGE_A_SAVED_OUTPUT_CALIBRATION_TARGET_PAIRS = (
    "defer/insufficient",
    "flag/invalid_value",
    "reject/contradicted",
    "verify/insufficient",
)
STAGE_A_SAVED_OUTPUT_CALIBRATION_REJECTED_PAIR = "ground/supported"
STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_PROMPT_CONTRACT = (
    "stage_a_saved_output_evidence_candidate_routing_v1"
)
STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_POLICY = (
    "all_valid_action_status_pairs_conditioned_on_visible_evidence"
)
STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_PAIRS = (
    "ground/supported",
    "reject/contradicted",
    "defer/insufficient",
    "verify/insufficient",
    "flag/invalid_value",
)
STAGE_A_STRICT_OUTPUT_KEYS = (
    "action",
    "evidence_status",
    "tool_calls",
    "cited_source_ids",
    "rationale",
)
STAGE_A_PROMPT_LEAK_TERMS = (
    "hidden_eval_metadata",
    "gold_evidence_status",
    "expected_terminal_action",
    "gold_source_ids",
    "source_task_id",
    "split_group",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for i, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{i} is not a JSON object")
            rows.append(obj)
    return rows


def validate(
    sft_rows: list[Mapping[str, Any]],
    preference_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    issues.extend(
        validate_sft_rows(
            sft_rows,
            manifest,
            label="sft",
            expected_dataset="negbiodb_ct_native_trajectory_v1",
        )
    )
    if len(preference_rows) != manifest.get("preference_pairs"):
        issues.append("preference_count_manifest_mismatch")

    modes = Counter(row.get("failure_mode") for row in preference_rows)
    if dict(modes) != manifest.get("preference_failure_modes"):
        issues.append("preference_failure_modes_manifest_mismatch")

    for row in preference_rows:
        if not row.get("chosen_score", {}).get("passed"):
            issues.append(f"{row.get('id')}:chosen_not_passing")
        if row.get("rejected_score", {}).get("passed"):
            issues.append(f"{row.get('id')}:rejected_is_passing")
        if len(row.get("prompt_messages", [])) != 2:
            issues.append(f"{row.get('id')}:prompt_should_be_system_user_only")
        if row.get("failure_mode") == "self_answering_without_tools":
            rejected_messages = row.get("rejected_messages", [])
            has_tool_call = any("tool_call" in message for message in rejected_messages)
            if has_tool_call:
                issues.append(f"{row.get('id')}:self_answer_rejected_trace_calls_tools")

    return issues


def validate_stage_a_exports(
    sft_rows: list[Mapping[str, Any]],
    preference_rows: list[Mapping[str, Any]],
    process_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if len(sft_rows) != manifest.get("sft_examples"):
        issues.append("stage_a_sft_count_manifest_mismatch")
    if len(preference_rows) != manifest.get("preference_pairs"):
        issues.append("stage_a_preference_count_manifest_mismatch")
    if len(process_rows) != manifest.get("process_examples"):
        issues.append("stage_a_process_count_manifest_mismatch")
    if manifest.get("sft_dataset") != STAGE_A_SFT_DATASET:
        issues.append("stage_a_sft_dataset_manifest_mismatch")
    if manifest.get("preference_dataset") != STAGE_A_PREFERENCE_DATASET:
        issues.append("stage_a_preference_dataset_manifest_mismatch")
    if manifest.get("process_dataset") != STAGE_A_PROCESS_DATASET:
        issues.append("stage_a_process_dataset_manifest_mismatch")
    if manifest.get("chosen_passed") != len(preference_rows):
        issues.append("stage_a_chosen_passed_manifest_mismatch")
    if manifest.get("rejected_passed") != 0:
        issues.append("stage_a_rejected_passed_manifest_mismatch")
    if manifest.get("split_group_overlap"):
        issues.append("stage_a_split_group_overlap")

    failure_modes = dict(sorted(Counter(row.get("failure_mode") for row in preference_rows).items()))
    if failure_modes != manifest.get("preference_failure_modes"):
        issues.append("stage_a_preference_failure_modes_manifest_mismatch")

    sft_case_ids = {str(row.get("source_manifest_case_id")) for row in sft_rows}
    preference_case_ids = {str(row.get("source_manifest_case_id")) for row in preference_rows}
    process_case_ids = {str(row.get("source_manifest_case_id")) for row in process_rows}
    if sft_case_ids != process_case_ids:
        issues.append("stage_a_sft_process_case_id_mismatch")
    if preference_case_ids != sft_case_ids:
        issues.append("stage_a_preference_case_id_mismatch")

    split_groups = [str(row.get("split_group")) for row in sft_rows if row.get("split_group")]
    split_group_overlap = sorted(group for group, count in Counter(split_groups).items() if count > 1)
    if split_group_overlap:
        issues.append("stage_a_sft_split_group_overlap")
    if sorted(split_groups) != manifest.get("split_groups"):
        issues.append("stage_a_split_groups_manifest_mismatch")

    for row in sft_rows:
        row_id = row.get("id")
        if row.get("dataset") != STAGE_A_SFT_DATASET:
            issues.append(f"{row_id}:stage_a_sft_unexpected_dataset")
        if row.get("oracle_target") is not True:
            issues.append(f"{row_id}:stage_a_sft_missing_oracle_target")
        if not row.get("score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_sft_not_passing")
        if row.get("score", {}).get("violations"):
            issues.append(f"{row_id}:stage_a_sft_has_violations")
        messages = row.get("messages", [])
        if len(messages) < 3:
            issues.append(f"{row_id}:stage_a_sft_missing_messages")
        elif messages[-1].get("tool_call", {}).get("name") != "submit_decision":
            issues.append(f"{row_id}:stage_a_sft_missing_submit_decision")
        issues.extend(stage_a_prompt_leak_issues(row_id, messages[:2], row))

    for row in preference_rows:
        row_id = row.get("id")
        if row.get("dataset") != STAGE_A_PREFERENCE_DATASET:
            issues.append(f"{row_id}:stage_a_preference_unexpected_dataset")
        if not row.get("failure_mode"):
            issues.append(f"{row_id}:stage_a_preference_missing_failure_mode")
        if not row.get("chosen_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_preference_chosen_not_passing")
        if row.get("rejected_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_preference_rejected_is_passing")
        prompt_messages = row.get("prompt_messages", [])
        if not prompt_messages or prompt_messages[-1].get("tool_call", {}).get("name") == "submit_decision":
            issues.append(f"{row_id}:stage_a_preference_prompt_contains_final_decision")
        if not ends_with_submit_decision(row.get("chosen_messages", [])):
            issues.append(f"{row_id}:stage_a_preference_chosen_missing_submit_decision")
        if not ends_with_submit_decision(row.get("rejected_messages", [])):
            issues.append(f"{row_id}:stage_a_preference_rejected_missing_submit_decision")
        issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))

    for row in process_rows:
        row_id = row.get("id")
        if row.get("dataset") != STAGE_A_PROCESS_DATASET:
            issues.append(f"{row_id}:stage_a_process_unexpected_dataset")
        if not row.get("score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_process_not_passing")
        prompt_messages = row.get("prompt_messages", [])
        if not prompt_messages or prompt_messages[-1].get("tool_call", {}).get("name") == "submit_decision":
            issues.append(f"{row_id}:stage_a_process_prompt_contains_final_decision")
        target_names = {step.get("name") for step in row.get("target_process", [])}
        for required_name in ("required_tools", "evidence_status", "terminal_action"):
            if required_name not in target_names:
                issues.append(f"{row_id}:stage_a_process_missing_{required_name}")
        issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))

    return issues


def validate_stage_a_strict_contract(
    sft_rows: list[Mapping[str, Any]],
    preference_rows: list[Mapping[str, Any]],
    process_rows: list[Mapping[str, Any]],
    train_sft_rows: list[Mapping[str, Any]],
    heldout_sft_rows: list[Mapping[str, Any]],
    train_preference_rows: list[Mapping[str, Any]],
    heldout_preference_rows: list[Mapping[str, Any]],
    train_process_rows: list[Mapping[str, Any]],
    heldout_process_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if manifest.get("dataset") != STAGE_A_STRICT_MANIFEST_DATASET:
        issues.append("stage_a_strict_manifest_dataset_mismatch")
    if manifest.get("prompt_contract") != STAGE_A_STRICT_PROMPT_CONTRACT:
        issues.append("stage_a_strict_prompt_contract_manifest_mismatch")
    if manifest.get("sft_dataset") != STAGE_A_STRICT_SFT_DATASET:
        issues.append("stage_a_strict_sft_dataset_manifest_mismatch")
    if manifest.get("preference_dataset") != STAGE_A_STRICT_PREFERENCE_DATASET:
        issues.append("stage_a_strict_preference_dataset_manifest_mismatch")
    if manifest.get("process_dataset") != STAGE_A_STRICT_PROCESS_DATASET:
        issues.append("stage_a_strict_process_dataset_manifest_mismatch")

    expected_counts = {
        "sft_examples": len(sft_rows),
        "preference_pairs": len(preference_rows),
        "process_examples": len(process_rows),
        "train_sft_examples": len(train_sft_rows),
        "heldout_sft_examples": len(heldout_sft_rows),
        "train_preference_pairs": len(train_preference_rows),
        "heldout_preference_pairs": len(heldout_preference_rows),
        "train_process_examples": len(train_process_rows),
        "heldout_process_examples": len(heldout_process_rows),
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_strict_{key}_manifest_mismatch")

    if manifest.get("chosen_passed") != len(preference_rows):
        issues.append("stage_a_strict_chosen_passed_manifest_mismatch")
    if manifest.get("rejected_passed") != 0:
        issues.append("stage_a_strict_rejected_passed_manifest_mismatch")

    failure_modes = dict(sorted(Counter(row.get("failure_mode") for row in preference_rows).items()))
    if failure_modes != manifest.get("preference_failure_modes"):
        issues.append("stage_a_strict_preference_failure_modes_manifest_mismatch")
    if set(failure_modes) != set(STAGE_A_STRICT_FAILURE_MODES):
        issues.append("stage_a_strict_unexpected_failure_modes")

    full_sft_case_ids = case_id_values(sft_rows)
    if case_id_values(process_rows) != full_sft_case_ids:
        issues.append("stage_a_strict_sft_process_case_id_mismatch")
    if case_id_values(preference_rows) != full_sft_case_ids:
        issues.append("stage_a_strict_preference_case_id_mismatch")

    train_case_ids = case_id_values(train_sft_rows)
    heldout_case_ids = case_id_values(heldout_sft_rows)
    if sorted(train_case_ids) != manifest.get("train_case_ids"):
        issues.append("stage_a_strict_train_case_ids_manifest_mismatch")
    if sorted(heldout_case_ids) != manifest.get("heldout_case_ids"):
        issues.append("stage_a_strict_heldout_case_ids_manifest_mismatch")
    overlap_case_ids = sorted(train_case_ids & heldout_case_ids)
    if overlap_case_ids:
        issues.append("stage_a_strict_case_overlap")
    if manifest.get("overlap_case_ids") != overlap_case_ids:
        issues.append("stage_a_strict_case_overlap_manifest_mismatch")

    train_split_groups = value_set(train_sft_rows, "split_group")
    heldout_split_groups = value_set(heldout_sft_rows, "split_group")
    overlap_split_groups = sorted(train_split_groups & heldout_split_groups)
    if overlap_split_groups:
        issues.append("stage_a_strict_split_group_overlap")
    if manifest.get("overlap_split_groups") != overlap_split_groups:
        issues.append("stage_a_strict_split_group_overlap_manifest_mismatch")

    train_source_task_ids = value_set(train_sft_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_sft_rows, "source_task_id")
    overlap_source_task_ids = sorted(train_source_task_ids & heldout_source_task_ids)
    if overlap_source_task_ids:
        issues.append("stage_a_strict_source_task_overlap")
    if manifest.get("overlap_source_task_ids") != overlap_source_task_ids:
        issues.append("stage_a_strict_source_task_overlap_manifest_mismatch")

    split_group_expectations = {
        "train_by_case_family": count_by_key(train_sft_rows, "case_family"),
        "heldout_by_case_family": count_by_key(heldout_sft_rows, "case_family"),
        "train_by_evidence_status": count_by_key(train_sft_rows, "gold_evidence_status"),
        "heldout_by_evidence_status": count_by_key(heldout_sft_rows, "gold_evidence_status"),
        "train_preference_failure_modes": count_by_key(train_preference_rows, "failure_mode"),
        "heldout_preference_failure_modes": count_by_key(heldout_preference_rows, "failure_mode"),
    }
    for key, value in split_group_expectations.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_strict_{key}_manifest_mismatch")

    issues.extend(
        validate_stage_a_strict_sft_rows(
            sft_rows,
            expected_split=None,
            label="sft",
        )
    )
    issues.extend(
        validate_stage_a_strict_preference_rows(
            preference_rows,
            expected_split=None,
            label="preference",
        )
    )
    issues.extend(
        validate_stage_a_strict_process_rows(
            process_rows,
            expected_split=None,
            label="process",
        )
    )
    issues.extend(
        validate_stage_a_strict_sft_rows(
            train_sft_rows,
            expected_split="train",
            label="train_sft",
        )
    )
    issues.extend(
        validate_stage_a_strict_sft_rows(
            heldout_sft_rows,
            expected_split="heldout",
            label="heldout_sft",
        )
    )
    issues.extend(
        validate_stage_a_strict_preference_rows(
            train_preference_rows,
            expected_split="train",
            label="train_preference",
        )
    )
    issues.extend(
        validate_stage_a_strict_preference_rows(
            heldout_preference_rows,
            expected_split="heldout",
            label="heldout_preference",
        )
    )
    issues.extend(
        validate_stage_a_strict_process_rows(
            train_process_rows,
            expected_split="train",
            label="train_process",
        )
    )
    issues.extend(
        validate_stage_a_strict_process_rows(
            heldout_process_rows,
            expected_split="heldout",
            label="heldout_process",
        )
    )

    return issues


def validate_stage_a_strict_sft_rows(
    rows: list[Mapping[str, Any]],
    *,
    expected_split: str | None,
    label: str,
) -> list[str]:
    issues: list[str] = []
    for row in rows:
        row_id = row.get("id")
        if row.get("dataset") != STAGE_A_STRICT_SFT_DATASET:
            issues.append(f"{row_id}:stage_a_strict_{label}_unexpected_dataset")
        if expected_split is not None and row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_strict_{label}_unexpected_split")
        if row.get("oracle_target") is not True:
            issues.append(f"{row_id}:stage_a_strict_{label}_missing_oracle_target")
        if row.get("prompt_contract") != STAGE_A_STRICT_PROMPT_CONTRACT:
            issues.append(f"{row_id}:stage_a_strict_{label}_wrong_prompt_contract")
        if not row.get("score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_strict_{label}_not_passing")
        if row.get("score", {}).get("violations"):
            issues.append(f"{row_id}:stage_a_strict_{label}_has_violations")
        messages = row.get("messages", [])
        if len(messages) < 3:
            issues.append(f"{row_id}:stage_a_strict_{label}_missing_messages")
        else:
            final_message = messages[-1]
            if final_message.get("role") != "assistant" or "content" not in final_message:
                issues.append(f"{row_id}:stage_a_strict_{label}_missing_json_assistant_target")
            prompt_messages = messages[:-1]
            issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))
        issues.extend(validate_stage_a_strict_output(row_id, row.get("target_output")))
    return issues


def validate_stage_a_strict_preference_rows(
    rows: list[Mapping[str, Any]],
    *,
    expected_split: str | None,
    label: str,
) -> list[str]:
    issues: list[str] = []
    for row in rows:
        row_id = row.get("id")
        if row.get("dataset") != STAGE_A_STRICT_PREFERENCE_DATASET:
            issues.append(f"{row_id}:stage_a_strict_{label}_unexpected_dataset")
        if expected_split is not None and row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_strict_{label}_unexpected_split")
        if row.get("prompt_contract") != STAGE_A_STRICT_PROMPT_CONTRACT:
            issues.append(f"{row_id}:stage_a_strict_{label}_wrong_prompt_contract")
        if row.get("failure_mode") not in STAGE_A_STRICT_FAILURE_MODES:
            issues.append(f"{row_id}:stage_a_strict_{label}_unexpected_failure_mode")
        if not row.get("chosen_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_strict_{label}_chosen_not_passing")
        if row.get("rejected_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_strict_{label}_rejected_is_passing")
        if not row.get("rejected_score", {}).get("violations"):
            issues.append(f"{row_id}:stage_a_strict_{label}_rejected_missing_violations")
        prompt_messages = row.get("prompt_messages", [])
        if not prompt_messages or prompt_messages[-1].get("tool_call", {}).get("name") == "submit_decision":
            issues.append(f"{row_id}:stage_a_strict_{label}_prompt_contains_final_decision")
        issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))
        if not strict_json_message_list(row.get("chosen_messages", [])):
            issues.append(f"{row_id}:stage_a_strict_{label}_chosen_missing_json_assistant_target")
        if not strict_json_message_list(row.get("rejected_messages", [])):
            issues.append(f"{row_id}:stage_a_strict_{label}_rejected_missing_json_assistant_target")
        issues.extend(validate_stage_a_strict_output(row_id, row.get("chosen_output"), prefix="chosen"))
        issues.extend(validate_stage_a_strict_output(row_id, row.get("rejected_output"), prefix="rejected"))
    return issues


def validate_stage_a_strict_process_rows(
    rows: list[Mapping[str, Any]],
    *,
    expected_split: str | None,
    label: str,
) -> list[str]:
    issues: list[str] = []
    for row in rows:
        row_id = row.get("id")
        if row.get("dataset") != STAGE_A_STRICT_PROCESS_DATASET:
            issues.append(f"{row_id}:stage_a_strict_{label}_unexpected_dataset")
        if expected_split is not None and row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_strict_{label}_unexpected_split")
        if row.get("prompt_contract") != STAGE_A_STRICT_PROMPT_CONTRACT:
            issues.append(f"{row_id}:stage_a_strict_{label}_wrong_prompt_contract")
        if not row.get("score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_strict_{label}_not_passing")
        prompt_messages = row.get("prompt_messages", [])
        if not prompt_messages or prompt_messages[-1].get("tool_call", {}).get("name") == "submit_decision":
            issues.append(f"{row_id}:stage_a_strict_{label}_prompt_contains_final_decision")
        issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))
        target_names = {step.get("name") for step in row.get("target_process", [])}
        for required_name in (
            "prompt_contract",
            "required_output_keys",
            "required_tool_sequence",
            "required_query_fields",
            "evidence_status",
            "terminal_action",
            "cited_source_ids",
        ):
            if required_name not in target_names:
                issues.append(f"{row_id}:stage_a_strict_{label}_missing_{required_name}")
        issues.extend(validate_stage_a_strict_output(row_id, row.get("target_output")))
    return issues


def strict_json_message_list(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) != 1:
        return False
    message = messages[0]
    if not isinstance(message, Mapping):
        return False
    if message.get("role") != "assistant" or "content" not in message:
        return False
    try:
        payload = json.loads(str(message["content"]))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict)


def validate_stage_a_strict_output(
    row_id: Any,
    output: Any,
    *,
    prefix: str = "target",
) -> list[str]:
    issues: list[str] = []
    if not isinstance(output, Mapping):
        return [f"{row_id}:stage_a_strict_{prefix}_output_not_object"]
    missing = [key for key in STAGE_A_STRICT_OUTPUT_KEYS if key not in output]
    if missing:
        issues.append(f"{row_id}:stage_a_strict_{prefix}_output_missing_keys")
    if set(output).difference(STAGE_A_STRICT_OUTPUT_KEYS):
        issues.append(f"{row_id}:stage_a_strict_{prefix}_output_extra_keys")
    if output.get("action") not in {"ground", "reject", "defer", "verify", "flag", "self_answer"}:
        issues.append(f"{row_id}:stage_a_strict_{prefix}_output_bad_action")
    if output.get("evidence_status") not in {"supported", "contradicted", "invalid_value", "insufficient", "unknown"}:
        issues.append(f"{row_id}:stage_a_strict_{prefix}_output_bad_evidence_status")
    tool_calls = output.get("tool_calls")
    if not isinstance(tool_calls, list):
        issues.append(f"{row_id}:stage_a_strict_{prefix}_output_tool_calls_not_list")
    else:
        for call in tool_calls:
            if not isinstance(call, Mapping):
                issues.append(f"{row_id}:stage_a_strict_{prefix}_output_tool_call_not_object")
                continue
            if not isinstance(call.get("name"), str) or not call.get("name"):
                issues.append(f"{row_id}:stage_a_strict_{prefix}_output_tool_call_missing_name")
            arguments = call.get("arguments")
            if not isinstance(arguments, Mapping):
                issues.append(f"{row_id}:stage_a_strict_{prefix}_output_tool_call_arguments_not_object")
            elif not {"drug_id", "condition_id"}.issubset(arguments):
                issues.append(f"{row_id}:stage_a_strict_{prefix}_output_tool_call_missing_required_argument")
    if not isinstance(output.get("cited_source_ids"), list):
        issues.append(f"{row_id}:stage_a_strict_{prefix}_output_citations_not_list")
    if not isinstance(output.get("rationale"), str) or not output.get("rationale"):
        issues.append(f"{row_id}:stage_a_strict_{prefix}_output_missing_rationale")
    return issues


def validate_stage_a_strict_component_targets(
    rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if manifest.get("dataset") != STAGE_A_STRICT_COMPONENT_MANIFEST_DATASET:
        issues.append("stage_a_strict_component_manifest_dataset_mismatch")
    if manifest.get("component_dataset") != STAGE_A_STRICT_COMPONENT_DATASET:
        issues.append("stage_a_strict_component_dataset_manifest_mismatch")
    if manifest.get("prompt_contract") != STAGE_A_STRICT_PROMPT_CONTRACT:
        issues.append("stage_a_strict_component_prompt_contract_manifest_mismatch")
    if tuple(manifest.get("components", ())) != STAGE_A_STRICT_COMPONENTS:
        issues.append("stage_a_strict_component_components_manifest_mismatch")
    if manifest.get("target_keys_by_component") != {
        key: list(value)
        for key, value in STAGE_A_STRICT_COMPONENT_TARGET_KEYS.items()
    }:
        issues.append("stage_a_strict_component_target_keys_manifest_mismatch")

    expected_counts = {
        "target_examples": len(rows),
        "train_target_examples": len(train_rows),
        "heldout_target_examples": len(heldout_rows),
        "source_cases": len(case_id_values(rows)),
        "train_cases": len(case_id_values(train_rows)),
        "heldout_cases": len(case_id_values(heldout_rows)),
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_strict_component_{key}_manifest_mismatch")

    if manifest.get("by_component") != count_by_key(rows, "component"):
        issues.append("stage_a_strict_component_by_component_manifest_mismatch")
    if manifest.get("train_by_component") != count_by_key(train_rows, "component"):
        issues.append("stage_a_strict_component_train_by_component_manifest_mismatch")
    if manifest.get("heldout_by_component") != count_by_key(heldout_rows, "component"):
        issues.append("stage_a_strict_component_heldout_by_component_manifest_mismatch")

    train_case_ids = case_id_values(train_rows)
    heldout_case_ids = case_id_values(heldout_rows)
    overlap_case_ids = sorted(train_case_ids & heldout_case_ids)
    if overlap_case_ids:
        issues.append("stage_a_strict_component_case_overlap")
    if manifest.get("overlap_case_ids") != overlap_case_ids:
        issues.append("stage_a_strict_component_case_overlap_manifest_mismatch")
    if manifest.get("train_case_ids") != sorted(train_case_ids):
        issues.append("stage_a_strict_component_train_case_ids_manifest_mismatch")
    if manifest.get("heldout_case_ids") != sorted(heldout_case_ids):
        issues.append("stage_a_strict_component_heldout_case_ids_manifest_mismatch")

    train_split_groups = value_set(train_rows, "split_group")
    heldout_split_groups = value_set(heldout_rows, "split_group")
    overlap_split_groups = sorted(train_split_groups & heldout_split_groups)
    if overlap_split_groups:
        issues.append("stage_a_strict_component_split_group_overlap")
    if manifest.get("overlap_split_groups") != overlap_split_groups:
        issues.append("stage_a_strict_component_split_group_overlap_manifest_mismatch")

    train_source_task_ids = value_set(train_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_rows, "source_task_id")
    overlap_source_task_ids = sorted(train_source_task_ids & heldout_source_task_ids)
    if overlap_source_task_ids:
        issues.append("stage_a_strict_component_source_task_overlap")
    if manifest.get("overlap_source_task_ids") != overlap_source_task_ids:
        issues.append("stage_a_strict_component_source_task_overlap_manifest_mismatch")

    issues.extend(validate_stage_a_strict_component_rows(rows, expected_split=None, label="targets"))
    issues.extend(validate_stage_a_strict_component_rows(train_rows, expected_split="train", label="train"))
    issues.extend(validate_stage_a_strict_component_rows(heldout_rows, expected_split="heldout", label="heldout"))

    components_by_case: dict[str, set[str]] = {}
    for row in rows:
        case_id = str(row.get("source_manifest_case_id"))
        components_by_case.setdefault(case_id, set()).add(str(row.get("component")))
    for case_id, components in sorted(components_by_case.items()):
        if components != set(STAGE_A_STRICT_COMPONENTS):
            issues.append(f"{case_id}:stage_a_strict_component_missing_component")

    return issues


def validate_stage_a_strict_component_rows(
    rows: list[Mapping[str, Any]],
    *,
    expected_split: str | None,
    label: str,
    expected_dataset: str = STAGE_A_STRICT_COMPONENT_DATASET,
    expected_prompt_contract: str = STAGE_A_STRICT_PROMPT_CONTRACT,
) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            issues.append(f"{row_id}:stage_a_strict_component_duplicate_id")
        seen_ids.add(str(row_id))
        component = str(row.get("component"))
        if row.get("dataset") != expected_dataset:
            issues.append(f"{row_id}:stage_a_strict_component_{label}_unexpected_dataset")
        if expected_split is not None and row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_strict_component_{label}_unexpected_split")
        if row.get("prompt_contract") != expected_prompt_contract:
            issues.append(f"{row_id}:stage_a_strict_component_{label}_wrong_prompt_contract")
        if row.get("oracle_target") is not True:
            issues.append(f"{row_id}:stage_a_strict_component_{label}_missing_oracle_target")
        if component not in STAGE_A_STRICT_COMPONENTS:
            issues.append(f"{row_id}:stage_a_strict_component_{label}_unknown_component")
            continue
        expected_keys = list(STAGE_A_STRICT_COMPONENT_TARGET_KEYS[component])
        if row.get("target_keys") != expected_keys:
            issues.append(f"{row_id}:stage_a_strict_component_{label}_target_keys_mismatch")
        prompt_messages = row.get("prompt_messages", [])
        if not isinstance(prompt_messages, list) or len(prompt_messages) != 2:
            issues.append(f"{row_id}:stage_a_strict_component_{label}_bad_prompt_message_count")
        else:
            if [message.get("role") for message in prompt_messages] != ["system", "user"]:
                issues.append(f"{row_id}:stage_a_strict_component_{label}_bad_prompt_roles")
            issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))
        issues.extend(validate_stage_a_strict_component_target(row_id, component, row.get("target_output")))
    return issues


def validate_stage_a_evidence_conditioned_component_targets(
    rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if manifest.get("dataset") != STAGE_A_EVIDENCE_COMPONENT_MANIFEST_DATASET:
        issues.append("stage_a_evidence_component_manifest_dataset_mismatch")
    if manifest.get("component_dataset") != STAGE_A_EVIDENCE_COMPONENT_DATASET:
        issues.append("stage_a_evidence_component_dataset_manifest_mismatch")
    if manifest.get("prompt_contract") != STAGE_A_EVIDENCE_COMPONENT_PROMPT_CONTRACT:
        issues.append("stage_a_evidence_component_prompt_contract_manifest_mismatch")
    if tuple(manifest.get("components", ())) != STAGE_A_STRICT_COMPONENTS:
        issues.append("stage_a_evidence_component_components_manifest_mismatch")
    if tuple(manifest.get("evidence_conditioned_components", ())) != STAGE_A_EVIDENCE_COMPONENTS:
        issues.append("stage_a_evidence_component_evidence_components_manifest_mismatch")
    if manifest.get("evidence_packet_policy") != "public_synthetic_tool_result_state_v1":
        issues.append("stage_a_evidence_component_policy_manifest_mismatch")
    if manifest.get("target_keys_by_component") != {
        key: list(value)
        for key, value in STAGE_A_STRICT_COMPONENT_TARGET_KEYS.items()
    }:
        issues.append("stage_a_evidence_component_target_keys_manifest_mismatch")

    expected_counts = {
        "target_examples": len(rows),
        "train_target_examples": len(train_rows),
        "heldout_target_examples": len(heldout_rows),
        "source_cases": len(case_id_values(rows)),
        "train_cases": len(case_id_values(train_rows)),
        "heldout_cases": len(case_id_values(heldout_rows)),
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_evidence_component_{key}_manifest_mismatch")

    evidence_rows = [row for row in rows if row.get("component") in STAGE_A_EVIDENCE_COMPONENTS]
    if manifest.get("evidence_conditioned_rows") != len(evidence_rows):
        issues.append("stage_a_evidence_component_conditioned_count_manifest_mismatch")
    if manifest.get("by_component") != count_by_key(rows, "component"):
        issues.append("stage_a_evidence_component_by_component_manifest_mismatch")
    if manifest.get("train_by_component") != count_by_key(train_rows, "component"):
        issues.append("stage_a_evidence_component_train_by_component_manifest_mismatch")
    if manifest.get("heldout_by_component") != count_by_key(heldout_rows, "component"):
        issues.append("stage_a_evidence_component_heldout_by_component_manifest_mismatch")

    train_case_ids = case_id_values(train_rows)
    heldout_case_ids = case_id_values(heldout_rows)
    overlap_case_ids = sorted(train_case_ids & heldout_case_ids)
    if overlap_case_ids:
        issues.append("stage_a_evidence_component_case_overlap")
    if manifest.get("overlap_case_ids") != overlap_case_ids:
        issues.append("stage_a_evidence_component_case_overlap_manifest_mismatch")
    if manifest.get("train_case_ids") != sorted(train_case_ids):
        issues.append("stage_a_evidence_component_train_case_ids_manifest_mismatch")
    if manifest.get("heldout_case_ids") != sorted(heldout_case_ids):
        issues.append("stage_a_evidence_component_heldout_case_ids_manifest_mismatch")

    train_split_groups = value_set(train_rows, "split_group")
    heldout_split_groups = value_set(heldout_rows, "split_group")
    overlap_split_groups = sorted(train_split_groups & heldout_split_groups)
    if overlap_split_groups:
        issues.append("stage_a_evidence_component_split_group_overlap")
    if manifest.get("overlap_split_groups") != overlap_split_groups:
        issues.append("stage_a_evidence_component_split_group_overlap_manifest_mismatch")

    train_source_task_ids = value_set(train_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_rows, "source_task_id")
    overlap_source_task_ids = sorted(train_source_task_ids & heldout_source_task_ids)
    if overlap_source_task_ids:
        issues.append("stage_a_evidence_component_source_task_overlap")
    if manifest.get("overlap_source_task_ids") != overlap_source_task_ids:
        issues.append("stage_a_evidence_component_source_task_overlap_manifest_mismatch")

    for label, split_rows, expected_split in (
        ("targets", rows, None),
        ("train", train_rows, "train"),
        ("heldout", heldout_rows, "heldout"),
    ):
        issues.extend(
            validate_stage_a_strict_component_rows(
                split_rows,
                expected_split=expected_split,
                label=label,
                expected_dataset=STAGE_A_EVIDENCE_COMPONENT_DATASET,
                expected_prompt_contract=STAGE_A_EVIDENCE_COMPONENT_PROMPT_CONTRACT,
            )
        )
        for row in split_rows:
            issues.extend(validate_stage_a_evidence_conditioning(row, label=label))

    return issues


def validate_stage_a_evidence_conditioning(row: Mapping[str, Any], *, label: str) -> list[str]:
    row_id = row.get("id")
    component = row.get("component")
    issues: list[str] = []
    try:
        prompt_messages = row.get("prompt_messages", [])
        user_payload = json.loads(str(prompt_messages[1]["content"]))
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return [f"{row_id}:stage_a_evidence_component_{label}_prompt_not_json"]
    if not isinstance(user_payload, Mapping):
        return [f"{row_id}:stage_a_evidence_component_{label}_prompt_not_object"]

    if component == "enum_action":
        packet = user_payload.get("evidence_packet")
        if not isinstance(packet, Mapping):
            issues.append(f"{row_id}:stage_a_evidence_component_{label}_missing_evidence_packet")
        elif not _tool_results_have_content(packet.get("tool_results")):
            issues.append(f"{row_id}:stage_a_evidence_component_{label}_evidence_packet_missing_content")
    elif component == "routing_after_loop":
        if not _tool_results_have_content(user_payload.get("observed_tool_loop")):
            issues.append(f"{row_id}:stage_a_evidence_component_{label}_observed_loop_missing_results")
    elif component == "tool_query":
        if "evidence_packet" in user_payload or "observed_tool_loop" in user_payload:
            issues.append(f"{row_id}:stage_a_evidence_component_{label}_tool_query_has_evidence_state")
    return issues


def _tool_results_have_content(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, Mapping) and isinstance(item.get("content"), Mapping) for item in value)


def validate_stage_a_strict_component_target(
    row_id: Any,
    component: str,
    target: Any,
) -> list[str]:
    issues: list[str] = []
    if not isinstance(target, Mapping):
        return [f"{row_id}:stage_a_strict_component_target_not_object"]
    expected_keys = set(STAGE_A_STRICT_COMPONENT_TARGET_KEYS[component])
    if set(target) != expected_keys:
        issues.append(f"{row_id}:stage_a_strict_component_target_key_mismatch")
    if "action" in target and target.get("action") not in {"ground", "reject", "defer", "verify", "flag", "self_answer"}:
        issues.append(f"{row_id}:stage_a_strict_component_bad_action")
    if "evidence_status" in target and target.get("evidence_status") not in {
        "supported",
        "contradicted",
        "invalid_value",
        "insufficient",
        "unknown",
    }:
        issues.append(f"{row_id}:stage_a_strict_component_bad_evidence_status")
    if component == "tool_query":
        tool_calls = target.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            issues.append(f"{row_id}:stage_a_strict_component_tool_calls_missing")
        elif not all(strict_component_tool_call_ok(call) for call in tool_calls):
            issues.append(f"{row_id}:stage_a_strict_component_tool_call_bad_shape")
    if component == "routing_after_loop" and not isinstance(target.get("cited_source_ids"), list):
        issues.append(f"{row_id}:stage_a_strict_component_citations_not_list")
    return issues


def validate_stage_a_enum_corrective_pairs(
    rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if manifest.get("dataset") != STAGE_A_ENUM_CORRECTIVE_MANIFEST_DATASET:
        issues.append("stage_a_enum_corrective_manifest_dataset_mismatch")
    if manifest.get("pair_dataset") != STAGE_A_ENUM_CORRECTIVE_PAIR_DATASET:
        issues.append("stage_a_enum_corrective_pair_dataset_manifest_mismatch")
    if manifest.get("source_component_dataset") != STAGE_A_STRICT_COMPONENT_DATASET:
        issues.append("stage_a_enum_corrective_source_dataset_manifest_mismatch")
    if manifest.get("prompt_contract") != STAGE_A_STRICT_PROMPT_CONTRACT:
        issues.append("stage_a_enum_corrective_prompt_contract_manifest_mismatch")
    if manifest.get("component") != "enum_action":
        issues.append("stage_a_enum_corrective_component_manifest_mismatch")
    if manifest.get("failure_mode") != STAGE_A_ENUM_CORRECTIVE_FAILURE_MODE:
        issues.append("stage_a_enum_corrective_failure_mode_manifest_mismatch")
    if manifest.get("rejected_output") != STAGE_A_ENUM_CORRECTIVE_REJECTED_OUTPUT:
        issues.append("stage_a_enum_corrective_rejected_output_manifest_mismatch")

    expected_counts = {
        "pair_examples": len(rows),
        "train_pairs": len(train_rows),
        "heldout_pairs": len(heldout_rows),
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_enum_corrective_{key}_manifest_mismatch")
    if manifest.get("by_chosen_pair") != count_by_key(rows, "chosen_pair"):
        issues.append("stage_a_enum_corrective_by_chosen_pair_manifest_mismatch")
    if manifest.get("train_by_chosen_pair") != count_by_key(train_rows, "chosen_pair"):
        issues.append("stage_a_enum_corrective_train_by_chosen_pair_manifest_mismatch")
    if manifest.get("heldout_by_chosen_pair") != count_by_key(heldout_rows, "chosen_pair"):
        issues.append("stage_a_enum_corrective_heldout_by_chosen_pair_manifest_mismatch")

    train_case_ids = case_id_values(train_rows)
    heldout_case_ids = case_id_values(heldout_rows)
    overlap_case_ids = sorted(train_case_ids & heldout_case_ids)
    if overlap_case_ids:
        issues.append("stage_a_enum_corrective_case_overlap")
    if manifest.get("overlap_case_ids") != overlap_case_ids:
        issues.append("stage_a_enum_corrective_case_overlap_manifest_mismatch")
    if manifest.get("train_case_ids") != sorted(train_case_ids):
        issues.append("stage_a_enum_corrective_train_case_ids_manifest_mismatch")
    if manifest.get("heldout_case_ids") != sorted(heldout_case_ids):
        issues.append("stage_a_enum_corrective_heldout_case_ids_manifest_mismatch")

    train_split_groups = value_set(train_rows, "split_group")
    heldout_split_groups = value_set(heldout_rows, "split_group")
    overlap_split_groups = sorted(train_split_groups & heldout_split_groups)
    if overlap_split_groups:
        issues.append("stage_a_enum_corrective_split_group_overlap")
    if manifest.get("overlap_split_groups") != overlap_split_groups:
        issues.append("stage_a_enum_corrective_split_group_overlap_manifest_mismatch")

    train_source_task_ids = value_set(train_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_rows, "source_task_id")
    overlap_source_task_ids = sorted(train_source_task_ids & heldout_source_task_ids)
    if overlap_source_task_ids:
        issues.append("stage_a_enum_corrective_source_task_overlap")
    if manifest.get("overlap_source_task_ids") != overlap_source_task_ids:
        issues.append("stage_a_enum_corrective_source_task_overlap_manifest_mismatch")

    issues.extend(validate_stage_a_enum_corrective_rows(rows, expected_split=None, label="pairs"))
    issues.extend(validate_stage_a_enum_corrective_rows(train_rows, expected_split="train", label="train"))
    issues.extend(validate_stage_a_enum_corrective_rows(heldout_rows, expected_split="heldout", label="heldout"))
    return issues


def validate_stage_a_enum_corrective_rows(
    rows: list[Mapping[str, Any]],
    *,
    expected_split: str | None,
    label: str,
) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            issues.append(f"{row_id}:stage_a_enum_corrective_duplicate_id")
        seen_ids.add(str(row_id))
        if row.get("dataset") != STAGE_A_ENUM_CORRECTIVE_PAIR_DATASET:
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_unexpected_dataset")
        if expected_split is not None and row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_unexpected_split")
        if row.get("component") != "enum_action":
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_wrong_component")
        if row.get("failure_mode") != STAGE_A_ENUM_CORRECTIVE_FAILURE_MODE:
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_wrong_failure_mode")
        if row.get("prompt_contract") != STAGE_A_STRICT_PROMPT_CONTRACT:
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_wrong_prompt_contract")
        if row.get("candidate_policy") != "train_observed_pairs":
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_wrong_candidate_policy")
        if row.get("source_component_dataset") != STAGE_A_STRICT_COMPONENT_DATASET:
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_wrong_source_dataset")
        if row.get("target_keys") != ["action", "evidence_status"]:
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_target_keys_mismatch")
        if row.get("oracle_target") is not True:
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_missing_oracle_target")
        prompt_messages = row.get("prompt_messages", [])
        if not isinstance(prompt_messages, list) or len(prompt_messages) != 2:
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_bad_prompt_message_count")
        else:
            if [message.get("role") for message in prompt_messages] != ["system", "user"]:
                issues.append(f"{row_id}:stage_a_enum_corrective_{label}_bad_prompt_roles")
            issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))
        chosen = row.get("chosen_output")
        rejected = row.get("rejected_output")
        issues.extend(validate_stage_a_strict_component_target(row_id, "enum_action", chosen))
        issues.extend(validate_stage_a_strict_component_target(row_id, "enum_action", rejected))
        if chosen == rejected:
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_chosen_equals_rejected")
        if rejected != STAGE_A_ENUM_CORRECTIVE_REJECTED_OUTPUT:
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_rejected_not_ground_supported")
        if row.get("chosen_pair") == row.get("rejected_pair"):
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_pair_labels_equal")
        if row.get("rejected_pair") != "ground/supported":
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_bad_rejected_pair")
        if not row.get("chosen_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_chosen_not_passing")
        if row.get("rejected_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_rejected_is_passing")
        if "target_mismatch" not in row.get("rejected_score", {}).get("violations", []):
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_rejected_missing_target_mismatch")
        if not strict_json_message_list(row.get("chosen_messages", [])):
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_chosen_missing_json_assistant_target")
        if not strict_json_message_list(row.get("rejected_messages", [])):
            issues.append(f"{row_id}:stage_a_enum_corrective_{label}_rejected_missing_json_assistant_target")
    return issues


def validate_stage_a_enum_action_contrast_pairs(
    rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if manifest.get("dataset") != STAGE_A_ENUM_ACTION_CONTRAST_MANIFEST_DATASET:
        issues.append("stage_a_enum_action_contrast_manifest_dataset_mismatch")
    if manifest.get("pair_dataset") != STAGE_A_ENUM_ACTION_CONTRAST_PAIR_DATASET:
        issues.append("stage_a_enum_action_contrast_pair_dataset_manifest_mismatch")
    if manifest.get("source_component_dataset") != STAGE_A_STRICT_COMPONENT_DATASET:
        issues.append("stage_a_enum_action_contrast_source_dataset_manifest_mismatch")
    if manifest.get("prompt_contract") != STAGE_A_STRICT_PROMPT_CONTRACT:
        issues.append("stage_a_enum_action_contrast_prompt_contract_manifest_mismatch")
    if manifest.get("component") != "enum_action":
        issues.append("stage_a_enum_action_contrast_component_manifest_mismatch")
    if manifest.get("failure_mode") != STAGE_A_ENUM_ACTION_CONTRAST_FAILURE_MODE:
        issues.append("stage_a_enum_action_contrast_failure_mode_manifest_mismatch")
    if manifest.get("contrast_axis") != "action":
        issues.append("stage_a_enum_action_contrast_axis_manifest_mismatch")
    if manifest.get("candidate_policy") != STAGE_A_ENUM_ACTION_CONTRAST_CANDIDATE_POLICY:
        issues.append("stage_a_enum_action_contrast_candidate_policy_manifest_mismatch")
    if manifest.get("rejected_action") != "ground":
        issues.append("stage_a_enum_action_contrast_rejected_action_manifest_mismatch")
    if manifest.get("rejected_evidence_status_policy") != "same_as_chosen":
        issues.append("stage_a_enum_action_contrast_status_policy_manifest_mismatch")

    expected_counts = {
        "pair_examples": len(rows),
        "train_pairs": len(train_rows),
        "heldout_pairs": len(heldout_rows),
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_enum_action_contrast_{key}_manifest_mismatch")
    if manifest.get("by_chosen_pair") != count_by_key(rows, "chosen_pair"):
        issues.append("stage_a_enum_action_contrast_by_chosen_pair_manifest_mismatch")
    if manifest.get("by_rejected_pair") != count_by_key(rows, "rejected_pair"):
        issues.append("stage_a_enum_action_contrast_by_rejected_pair_manifest_mismatch")
    if manifest.get("train_by_chosen_pair") != count_by_key(train_rows, "chosen_pair"):
        issues.append("stage_a_enum_action_contrast_train_by_chosen_pair_manifest_mismatch")
    if manifest.get("heldout_by_chosen_pair") != count_by_key(heldout_rows, "chosen_pair"):
        issues.append("stage_a_enum_action_contrast_heldout_by_chosen_pair_manifest_mismatch")

    train_case_ids = case_id_values(train_rows)
    heldout_case_ids = case_id_values(heldout_rows)
    overlap_case_ids = sorted(train_case_ids & heldout_case_ids)
    if overlap_case_ids:
        issues.append("stage_a_enum_action_contrast_case_overlap")
    if manifest.get("overlap_case_ids") != overlap_case_ids:
        issues.append("stage_a_enum_action_contrast_case_overlap_manifest_mismatch")
    if manifest.get("train_case_ids") != sorted(train_case_ids):
        issues.append("stage_a_enum_action_contrast_train_case_ids_manifest_mismatch")
    if manifest.get("heldout_case_ids") != sorted(heldout_case_ids):
        issues.append("stage_a_enum_action_contrast_heldout_case_ids_manifest_mismatch")

    train_split_groups = value_set(train_rows, "split_group")
    heldout_split_groups = value_set(heldout_rows, "split_group")
    overlap_split_groups = sorted(train_split_groups & heldout_split_groups)
    if overlap_split_groups:
        issues.append("stage_a_enum_action_contrast_split_group_overlap")
    if manifest.get("overlap_split_groups") != overlap_split_groups:
        issues.append("stage_a_enum_action_contrast_split_group_overlap_manifest_mismatch")

    train_source_task_ids = value_set(train_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_rows, "source_task_id")
    overlap_source_task_ids = sorted(train_source_task_ids & heldout_source_task_ids)
    if overlap_source_task_ids:
        issues.append("stage_a_enum_action_contrast_source_task_overlap")
    if manifest.get("overlap_source_task_ids") != overlap_source_task_ids:
        issues.append("stage_a_enum_action_contrast_source_task_overlap_manifest_mismatch")

    issues.extend(validate_stage_a_enum_action_contrast_rows(rows, expected_split=None, label="pairs"))
    issues.extend(validate_stage_a_enum_action_contrast_rows(train_rows, expected_split="train", label="train"))
    issues.extend(validate_stage_a_enum_action_contrast_rows(heldout_rows, expected_split="heldout", label="heldout"))
    return issues


def validate_stage_a_enum_action_contrast_rows(
    rows: list[Mapping[str, Any]],
    *,
    expected_split: str | None,
    label: str,
) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_duplicate_id")
        seen_ids.add(str(row_id))
        if row.get("dataset") != STAGE_A_ENUM_ACTION_CONTRAST_PAIR_DATASET:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_unexpected_dataset")
        if expected_split is not None and row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_unexpected_split")
        if row.get("component") != "enum_action":
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_wrong_component")
        if row.get("failure_mode") != STAGE_A_ENUM_ACTION_CONTRAST_FAILURE_MODE:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_wrong_failure_mode")
        if row.get("contrast_axis") != "action":
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_wrong_axis")
        if row.get("candidate_policy") != STAGE_A_ENUM_ACTION_CONTRAST_CANDIDATE_POLICY:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_wrong_candidate_policy")
        if row.get("prompt_contract") != STAGE_A_STRICT_PROMPT_CONTRACT:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_wrong_prompt_contract")
        if row.get("source_component_dataset") != STAGE_A_STRICT_COMPONENT_DATASET:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_wrong_source_dataset")
        if row.get("target_keys") != ["action", "evidence_status"]:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_target_keys_mismatch")
        if row.get("oracle_target") is not True:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_missing_oracle_target")

        prompt_messages = row.get("prompt_messages", [])
        if not isinstance(prompt_messages, list) or len(prompt_messages) != 2:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_bad_prompt_message_count")
        else:
            if [message.get("role") for message in prompt_messages] != ["system", "user"]:
                issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_bad_prompt_roles")
            issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))

        chosen = row.get("chosen_output")
        rejected = row.get("rejected_output")
        issues.extend(validate_stage_a_strict_component_target(row_id, "enum_action", chosen))
        issues.extend(validate_stage_a_strict_component_target(row_id, "enum_action", rejected))
        if not isinstance(chosen, Mapping) or not isinstance(rejected, Mapping):
            continue
        if chosen == rejected:
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_chosen_equals_rejected")
        if chosen.get("action") == "ground":
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_chosen_action_ground")
        if rejected.get("action") != "ground":
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_rejected_action_not_ground")
        if rejected.get("evidence_status") != chosen.get("evidence_status"):
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_status_not_preserved")
        if row.get("chosen_pair") == row.get("rejected_pair"):
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_pair_labels_equal")
        if not row.get("chosen_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_chosen_not_passing")
        if row.get("rejected_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_rejected_is_passing")
        if "target_mismatch" not in row.get("rejected_score", {}).get("violations", []):
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_rejected_missing_target_mismatch")
        if not strict_json_message_list(row.get("chosen_messages", [])):
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_chosen_missing_json_assistant_target")
        if not strict_json_message_list(row.get("rejected_messages", [])):
            issues.append(f"{row_id}:stage_a_enum_action_contrast_{label}_rejected_missing_json_assistant_target")
    return issues


def validate_stage_a_routing_action_status_contrast_pairs(
    rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if manifest.get("dataset") != STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_MANIFEST_DATASET:
        issues.append("stage_a_routing_action_status_contrast_manifest_dataset_mismatch")
    if manifest.get("pair_dataset") != STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_PAIR_DATASET:
        issues.append("stage_a_routing_action_status_contrast_pair_dataset_manifest_mismatch")
    if manifest.get("source_component_dataset") != STAGE_A_EVIDENCE_COMPONENT_DATASET:
        issues.append("stage_a_routing_action_status_contrast_source_dataset_manifest_mismatch")
    if manifest.get("prompt_contract") != STAGE_A_EVIDENCE_COMPONENT_PROMPT_CONTRACT:
        issues.append("stage_a_routing_action_status_contrast_prompt_contract_manifest_mismatch")
    if manifest.get("component") != "routing_after_loop":
        issues.append("stage_a_routing_action_status_contrast_component_manifest_mismatch")
    if manifest.get("failure_mode") != STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_FAILURE_MODE:
        issues.append("stage_a_routing_action_status_contrast_failure_mode_manifest_mismatch")
    if manifest.get("contrast_axis") != "action_status":
        issues.append("stage_a_routing_action_status_contrast_axis_manifest_mismatch")
    if manifest.get("candidate_policy") != STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_CANDIDATE_POLICY:
        issues.append("stage_a_routing_action_status_contrast_candidate_policy_manifest_mismatch")
    if manifest.get("rejected_by_chosen_pair") != STAGE_A_ROUTING_ACTION_STATUS_REJECTED_BY_CHOSEN_PAIR:
        issues.append("stage_a_routing_action_status_contrast_rejection_map_manifest_mismatch")

    expected_counts = {
        "pair_examples": len(rows),
        "train_pairs": len(train_rows),
        "heldout_pairs": len(heldout_rows),
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_routing_action_status_contrast_{key}_manifest_mismatch")
    if manifest.get("by_chosen_pair") != count_by_key(rows, "chosen_pair"):
        issues.append("stage_a_routing_action_status_contrast_by_chosen_pair_manifest_mismatch")
    if manifest.get("by_rejected_pair") != count_by_key(rows, "rejected_pair"):
        issues.append("stage_a_routing_action_status_contrast_by_rejected_pair_manifest_mismatch")
    if manifest.get("train_by_chosen_pair") != count_by_key(train_rows, "chosen_pair"):
        issues.append("stage_a_routing_action_status_contrast_train_by_chosen_pair_manifest_mismatch")
    if manifest.get("heldout_by_chosen_pair") != count_by_key(heldout_rows, "chosen_pair"):
        issues.append("stage_a_routing_action_status_contrast_heldout_by_chosen_pair_manifest_mismatch")

    train_case_ids = case_id_values(train_rows)
    heldout_case_ids = case_id_values(heldout_rows)
    overlap_case_ids = sorted(train_case_ids & heldout_case_ids)
    if overlap_case_ids:
        issues.append("stage_a_routing_action_status_contrast_case_overlap")
    if manifest.get("overlap_case_ids") != overlap_case_ids:
        issues.append("stage_a_routing_action_status_contrast_case_overlap_manifest_mismatch")
    if manifest.get("train_case_ids") != sorted(train_case_ids):
        issues.append("stage_a_routing_action_status_contrast_train_case_ids_manifest_mismatch")
    if manifest.get("heldout_case_ids") != sorted(heldout_case_ids):
        issues.append("stage_a_routing_action_status_contrast_heldout_case_ids_manifest_mismatch")

    train_split_groups = value_set(train_rows, "split_group")
    heldout_split_groups = value_set(heldout_rows, "split_group")
    overlap_split_groups = sorted(train_split_groups & heldout_split_groups)
    if overlap_split_groups:
        issues.append("stage_a_routing_action_status_contrast_split_group_overlap")
    if manifest.get("overlap_split_groups") != overlap_split_groups:
        issues.append("stage_a_routing_action_status_contrast_split_group_overlap_manifest_mismatch")

    train_source_task_ids = value_set(train_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_rows, "source_task_id")
    overlap_source_task_ids = sorted(train_source_task_ids & heldout_source_task_ids)
    if overlap_source_task_ids:
        issues.append("stage_a_routing_action_status_contrast_source_task_overlap")
    if manifest.get("overlap_source_task_ids") != overlap_source_task_ids:
        issues.append("stage_a_routing_action_status_contrast_source_task_overlap_manifest_mismatch")

    issues.extend(validate_stage_a_routing_action_status_contrast_rows(rows, expected_split=None, label="pairs"))
    issues.extend(
        validate_stage_a_routing_action_status_contrast_rows(train_rows, expected_split="train", label="train")
    )
    issues.extend(
        validate_stage_a_routing_action_status_contrast_rows(heldout_rows, expected_split="heldout", label="heldout")
    )
    return issues


def validate_stage_a_routing_action_status_contrast_rows(
    rows: list[Mapping[str, Any]],
    *,
    expected_split: str | None,
    label: str,
) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    allowed_chosen = set(STAGE_A_ROUTING_ACTION_STATUS_REJECTED_BY_CHOSEN_PAIR)
    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_duplicate_id")
        seen_ids.add(str(row_id))
        if row.get("dataset") != STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_PAIR_DATASET:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_unexpected_dataset")
        if expected_split is not None and row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_unexpected_split")
        if row.get("component") != "routing_after_loop":
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_wrong_component")
        if row.get("failure_mode") != STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_FAILURE_MODE:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_wrong_failure_mode")
        if row.get("contrast_axis") != "action_status":
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_wrong_axis")
        if row.get("candidate_policy") != STAGE_A_ROUTING_ACTION_STATUS_CONTRAST_CANDIDATE_POLICY:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_wrong_candidate_policy")
        if row.get("prompt_contract") != STAGE_A_EVIDENCE_COMPONENT_PROMPT_CONTRACT:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_wrong_prompt_contract")
        if row.get("source_component_dataset") != STAGE_A_EVIDENCE_COMPONENT_DATASET:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_wrong_source_dataset")
        if row.get("target_keys") != list(STAGE_A_STRICT_COMPONENT_TARGET_KEYS["routing_after_loop"]):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_target_keys_mismatch")
        if row.get("oracle_target") is not True:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_missing_oracle_target")

        prompt_messages = row.get("prompt_messages", [])
        if not isinstance(prompt_messages, list) or len(prompt_messages) != 2:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_bad_prompt_message_count")
        else:
            if [message.get("role") for message in prompt_messages] != ["system", "user"]:
                issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_bad_prompt_roles")
            issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))

        chosen = row.get("chosen_output")
        rejected = row.get("rejected_output")
        issues.extend(validate_stage_a_strict_component_target(row_id, "routing_after_loop", chosen))
        issues.extend(validate_stage_a_strict_component_target(row_id, "routing_after_loop", rejected))
        if not isinstance(chosen, Mapping) or not isinstance(rejected, Mapping):
            continue
        if chosen == rejected:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_chosen_equals_rejected")
        chosen_pair = row.get("chosen_pair")
        rejected_pair = row.get("rejected_pair")
        if chosen_pair not in allowed_chosen:
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_unexpected_chosen_pair")
        expected_rejected = STAGE_A_ROUTING_ACTION_STATUS_REJECTED_BY_CHOSEN_PAIR.get(str(chosen_pair))
        if expected_rejected is not None:
            for key, value in expected_rejected.items():
                if rejected.get(key) != value:
                    issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_bad_rejected_{key}")
        if row.get("chosen_pair") == row.get("rejected_pair"):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_pair_labels_equal")
        if chosen_pair == "flag/invalid_value" and not rejected.get("cited_source_ids"):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_missing_ground_citation")
        if chosen_pair != "flag/invalid_value" and rejected.get("cited_source_ids"):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_unexpected_rejected_citation")
        if rejected_pair != pair_label(rejected):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_rejected_pair_label_mismatch")
        if row.get("chosen_pair") != pair_label(chosen):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_chosen_pair_label_mismatch")
        if not row.get("chosen_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_chosen_not_passing")
        if row.get("rejected_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_rejected_is_passing")
        if "target_mismatch" not in row.get("rejected_score", {}).get("violations", []):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_rejected_missing_target_mismatch")
        if not strict_json_message_list(row.get("chosen_messages", [])):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_chosen_missing_json_assistant_target")
        if not strict_json_message_list(row.get("rejected_messages", [])):
            issues.append(f"{row_id}:stage_a_routing_action_status_contrast_{label}_rejected_missing_json_assistant_target")
    return issues


def pair_label(output: Mapping[str, Any]) -> str:
    return f"{output.get('action')}/{output.get('evidence_status')}"


def validate_stage_a_saved_output_calibration_probe(
    rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if manifest.get("dataset") != STAGE_A_SAVED_OUTPUT_CALIBRATION_PROBE_MANIFEST_DATASET:
        issues.append("stage_a_saved_output_calibration_probe_manifest_dataset_mismatch")
    if manifest.get("pair_dataset") != STAGE_A_SAVED_OUTPUT_CALIBRATION_PROBE_PAIR_DATASET:
        issues.append("stage_a_saved_output_calibration_probe_pair_dataset_manifest_mismatch")
    if manifest.get("prompt_contract") != STAGE_A_SAVED_OUTPUT_CALIBRATION_PROMPT_CONTRACT:
        issues.append("stage_a_saved_output_calibration_probe_prompt_contract_manifest_mismatch")
    if manifest.get("selected_next_step") != "targeted_action_status_calibration_probe":
        issues.append("stage_a_saved_output_calibration_probe_next_step_manifest_mismatch")
    if manifest.get("rejected_pair") != STAGE_A_SAVED_OUTPUT_CALIBRATION_REJECTED_PAIR:
        issues.append("stage_a_saved_output_calibration_probe_rejected_pair_manifest_mismatch")
    if manifest.get("selected_target_pairs") != list(STAGE_A_SAVED_OUTPUT_CALIBRATION_TARGET_PAIRS):
        issues.append("stage_a_saved_output_calibration_probe_target_pairs_manifest_mismatch")

    expected_counts = {
        "pair_examples": len(rows),
        "train_pairs": len(train_rows),
        "heldout_probe_pairs": len(heldout_rows),
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_saved_output_calibration_probe_{key}_manifest_mismatch")
    if manifest.get("by_chosen_pair") != count_by_key(rows, "chosen_pair"):
        issues.append("stage_a_saved_output_calibration_probe_by_chosen_pair_manifest_mismatch")
    if manifest.get("train_by_chosen_pair") != count_by_key(train_rows, "chosen_pair"):
        issues.append("stage_a_saved_output_calibration_probe_train_by_chosen_pair_manifest_mismatch")
    if manifest.get("heldout_by_chosen_pair") != count_by_key(heldout_rows, "chosen_pair"):
        issues.append("stage_a_saved_output_calibration_probe_heldout_by_chosen_pair_manifest_mismatch")
    if manifest.get("by_case_family") != count_by_key(rows, "case_family"):
        issues.append("stage_a_saved_output_calibration_probe_by_case_family_manifest_mismatch")

    train_case_ids = case_id_values(train_rows)
    heldout_case_ids = case_id_values(heldout_rows)
    overlap_case_ids = sorted(train_case_ids & heldout_case_ids)
    if overlap_case_ids:
        issues.append("stage_a_saved_output_calibration_probe_case_overlap")
    if manifest.get("overlap_case_ids") != overlap_case_ids:
        issues.append("stage_a_saved_output_calibration_probe_case_overlap_manifest_mismatch")
    if manifest.get("train_case_ids") != sorted(train_case_ids):
        issues.append("stage_a_saved_output_calibration_probe_train_case_ids_manifest_mismatch")
    if manifest.get("heldout_case_ids") != sorted(heldout_case_ids):
        issues.append("stage_a_saved_output_calibration_probe_heldout_case_ids_manifest_mismatch")

    train_split_groups = value_set(train_rows, "split_group")
    heldout_split_groups = value_set(heldout_rows, "split_group")
    overlap_split_groups = sorted(train_split_groups & heldout_split_groups)
    if overlap_split_groups:
        issues.append("stage_a_saved_output_calibration_probe_split_group_overlap")
    if manifest.get("overlap_split_groups") != overlap_split_groups:
        issues.append("stage_a_saved_output_calibration_probe_split_group_overlap_manifest_mismatch")

    train_source_task_ids = value_set(train_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_rows, "source_task_id")
    overlap_source_task_ids = sorted(train_source_task_ids & heldout_source_task_ids)
    if overlap_source_task_ids:
        issues.append("stage_a_saved_output_calibration_probe_source_task_overlap")
    if manifest.get("overlap_source_task_ids") != overlap_source_task_ids:
        issues.append("stage_a_saved_output_calibration_probe_source_task_overlap_manifest_mismatch")

    issues.extend(validate_stage_a_saved_output_calibration_probe_rows(rows, expected_split=None, label="pairs"))
    issues.extend(
        validate_stage_a_saved_output_calibration_probe_rows(train_rows, expected_split="train", label="train")
    )
    issues.extend(
        validate_stage_a_saved_output_calibration_probe_rows(
            heldout_rows,
            expected_split="heldout",
            label="heldout",
        )
    )
    return issues


def validate_stage_a_saved_output_calibration_probe_rows(
    rows: list[Mapping[str, Any]],
    *,
    expected_split: str | None,
    label: str,
) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    allowed_chosen = set(STAGE_A_SAVED_OUTPUT_CALIBRATION_TARGET_PAIRS)
    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_duplicate_id")
        seen_ids.add(str(row_id))
        if row.get("dataset") != STAGE_A_SAVED_OUTPUT_CALIBRATION_PROBE_PAIR_DATASET:
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_unexpected_dataset")
        if expected_split is not None and row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_unexpected_split")
        if row.get("prompt_contract") != STAGE_A_SAVED_OUTPUT_CALIBRATION_PROMPT_CONTRACT:
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_wrong_prompt_contract")
        if row.get("calibration_axis") != "target_pair_vs_ground_supported":
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_wrong_axis")
        if row.get("oracle_target") is not True:
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_missing_oracle_target")
        if row.get("chosen_pair") not in allowed_chosen:
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_unexpected_chosen_pair")
        if row.get("rejected_pair") != STAGE_A_SAVED_OUTPUT_CALIBRATION_REJECTED_PAIR:
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_unexpected_rejected_pair")
        if row.get("split") == "train":
            if row.get("training_allowed") is not True:
                issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_train_not_allowed")
            if row.get("evaluation_only") is not False:
                issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_train_marked_eval_only")
        else:
            if row.get("training_allowed") is not False:
                issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_heldout_training_allowed")
            if row.get("evaluation_only") is not True:
                issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_heldout_not_eval_only")

        prompt_messages = row.get("prompt_messages", [])
        if not isinstance(prompt_messages, list) or len(prompt_messages) != 2:
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_bad_prompt_message_count")
        else:
            if [message.get("role") for message in prompt_messages] != ["system", "user"]:
                issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_bad_prompt_roles")
            issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))

        chosen = row.get("chosen_output")
        rejected = row.get("rejected_output")
        issues.extend(validate_stage_a_strict_output(row_id, chosen, prefix=f"calibration_{label}_chosen"))
        issues.extend(validate_stage_a_strict_output(row_id, rejected, prefix=f"calibration_{label}_rejected"))
        if not isinstance(chosen, Mapping) or not isinstance(rejected, Mapping):
            continue
        if row.get("chosen_pair") != pair_label(chosen):
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_chosen_pair_label_mismatch")
        if row.get("rejected_pair") != pair_label(rejected):
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_rejected_pair_label_mismatch")
        if chosen == rejected:
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_chosen_equals_rejected")
        if not row.get("chosen_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_chosen_not_passing")
        if row.get("rejected_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_rejected_is_passing")
        if row.get("chosen_pair") == "flag/invalid_value" and not chosen.get("cited_source_ids"):
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_missing_invalid_value_citation")
        if not strict_json_message_list(row.get("chosen_messages", [])):
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_chosen_missing_json_message")
        if not strict_json_message_list(row.get("rejected_messages", [])):
            issues.append(f"{row_id}:stage_a_saved_output_calibration_probe_{label}_rejected_missing_json_message")
    return issues


def validate_stage_a_saved_output_evidence_candidate_routing(
    rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if manifest.get("dataset") != STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_MANIFEST_DATASET:
        issues.append("stage_a_saved_output_evidence_candidate_routing_manifest_dataset_mismatch")
    if manifest.get("row_dataset") != STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_DATASET:
        issues.append("stage_a_saved_output_evidence_candidate_routing_row_dataset_manifest_mismatch")
    if manifest.get("prompt_contract") != STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_PROMPT_CONTRACT:
        issues.append("stage_a_saved_output_evidence_candidate_routing_prompt_contract_manifest_mismatch")
    if manifest.get("candidate_policy") != STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_POLICY:
        issues.append("stage_a_saved_output_evidence_candidate_routing_policy_manifest_mismatch")
    if manifest.get("candidate_pairs") != list(STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_PAIRS):
        issues.append("stage_a_saved_output_evidence_candidate_routing_candidate_pairs_manifest_mismatch")

    expected_counts = {
        "row_count": len(rows),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "bridge_focus_rows": sum(1 for row in rows if row.get("bridge_focus_case") is True),
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_saved_output_evidence_candidate_routing_{key}_manifest_mismatch")
    if manifest.get("by_target_pair") != count_by_key(rows, "target_pair"):
        issues.append("stage_a_saved_output_evidence_candidate_routing_by_target_pair_manifest_mismatch")
    if manifest.get("train_by_target_pair") != count_by_key(train_rows, "target_pair"):
        issues.append("stage_a_saved_output_evidence_candidate_routing_train_by_target_pair_manifest_mismatch")
    if manifest.get("heldout_by_target_pair") != count_by_key(heldout_rows, "target_pair"):
        issues.append("stage_a_saved_output_evidence_candidate_routing_heldout_by_target_pair_manifest_mismatch")
    bridge_rows = [row for row in rows if row.get("bridge_focus_case") is True]
    if manifest.get("bridge_focus_by_target_pair") != count_by_key(bridge_rows, "target_pair"):
        issues.append("stage_a_saved_output_evidence_candidate_routing_bridge_by_target_pair_manifest_mismatch")

    train_case_ids = case_id_values(train_rows)
    heldout_case_ids = case_id_values(heldout_rows)
    overlap_case_ids = sorted(train_case_ids & heldout_case_ids)
    if overlap_case_ids:
        issues.append("stage_a_saved_output_evidence_candidate_routing_case_overlap")
    if manifest.get("overlap_case_ids") != overlap_case_ids:
        issues.append("stage_a_saved_output_evidence_candidate_routing_case_overlap_manifest_mismatch")
    if manifest.get("train_case_ids") != sorted(train_case_ids):
        issues.append("stage_a_saved_output_evidence_candidate_routing_train_case_ids_manifest_mismatch")
    if manifest.get("heldout_case_ids") != sorted(heldout_case_ids):
        issues.append("stage_a_saved_output_evidence_candidate_routing_heldout_case_ids_manifest_mismatch")

    train_split_groups = value_set(train_rows, "split_group")
    heldout_split_groups = value_set(heldout_rows, "split_group")
    overlap_split_groups = sorted(train_split_groups & heldout_split_groups)
    if overlap_split_groups:
        issues.append("stage_a_saved_output_evidence_candidate_routing_split_group_overlap")
    if manifest.get("overlap_split_groups") != overlap_split_groups:
        issues.append("stage_a_saved_output_evidence_candidate_routing_split_group_overlap_manifest_mismatch")

    train_source_task_ids = value_set(train_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_rows, "source_task_id")
    overlap_source_task_ids = sorted(train_source_task_ids & heldout_source_task_ids)
    if overlap_source_task_ids:
        issues.append("stage_a_saved_output_evidence_candidate_routing_source_task_overlap")
    if manifest.get("overlap_source_task_ids") != overlap_source_task_ids:
        issues.append("stage_a_saved_output_evidence_candidate_routing_source_task_overlap_manifest_mismatch")

    if any(row.get("bridge_focus_case") is True for row in train_rows):
        issues.append("stage_a_saved_output_evidence_candidate_routing_bridge_focus_in_train")
    if sorted(case_id_values(bridge_rows)) != manifest.get("bridge_focus_case_ids"):
        issues.append("stage_a_saved_output_evidence_candidate_routing_bridge_case_ids_manifest_mismatch")

    issues.extend(
        validate_stage_a_saved_output_evidence_candidate_routing_rows(
            rows, expected_split=None, label="rows"
        )
    )
    issues.extend(
        validate_stage_a_saved_output_evidence_candidate_routing_rows(
            train_rows, expected_split="train", label="train"
        )
    )
    issues.extend(
        validate_stage_a_saved_output_evidence_candidate_routing_rows(
            heldout_rows, expected_split="heldout", label="heldout"
        )
    )
    return issues


def validate_stage_a_saved_output_evidence_candidate_routing_rows(
    rows: list[Mapping[str, Any]],
    *,
    expected_split: str | None,
    label: str,
) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    expected_pairs = list(STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_PAIRS)
    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_duplicate_id")
        seen_ids.add(str(row_id))
        if row.get("dataset") != STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_DATASET:
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_unexpected_dataset")
        if row.get("component") != "routing_after_loop":
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_unexpected_component")
        if row.get("prompt_contract") != STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_PROMPT_CONTRACT:
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_wrong_prompt_contract")
        if row.get("candidate_policy") != STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_POLICY:
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_wrong_policy")
        if expected_split is not None and row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_unexpected_split")
        if row.get("split") == "train":
            if row.get("training_allowed") is not True:
                issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_train_not_allowed")
            if row.get("evaluation_only") is not False:
                issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_train_marked_eval_only")
        else:
            if row.get("training_allowed") is not False:
                issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_heldout_training_allowed")
            if row.get("evaluation_only") is not True:
                issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_heldout_not_eval_only")
        if row.get("oracle_target") is not True:
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_missing_oracle_target")

        target = row.get("target_output")
        if not isinstance(target, Mapping):
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_missing_target_output")
            continue
        if row.get("target_pair") != target.get("selected_pair"):
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_target_pair_mismatch")
        if row.get("target_pair") != pair_label(target):
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_target_action_status_mismatch")
        target_pair = row.get("target_pair")
        if target_pair not in expected_pairs:
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_unexpected_target_pair")
        elif row.get("target_index") != expected_pairs.index(target_pair):
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_target_index_mismatch")

        candidate_outputs = row.get("candidate_outputs")
        if not isinstance(candidate_outputs, list):
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_missing_candidate_outputs")
        else:
            candidate_pairs = [item.get("pair") for item in candidate_outputs if isinstance(item, Mapping)]
            if candidate_pairs != expected_pairs:
                issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_candidate_outputs_mismatch")

        task = row.get("model_visible_task")
        if not isinstance(task, Mapping):
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_missing_model_visible_task")
        else:
            if task.get("candidate_pairs") != expected_pairs:
                issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_task_candidate_pairs_mismatch")
            if task.get("component") != "saved_output_evidence_candidate_routing":
                issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_task_component_mismatch")
            if not isinstance(task.get("observed_tool_loop"), list) or not task.get("observed_tool_loop"):
                issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_missing_observed_tool_loop")
            if not isinstance(task.get("visible_evidence_features"), Mapping):
                issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_missing_visible_features")
            prompt_messages = [{"role": "user", "content": json.dumps(task, sort_keys=True)}]
            issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))
            prompt_text = json.dumps(task, sort_keys=True)
            if str(row.get("case_family")) in prompt_text:
                issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_task_leaks_case_family")

        if row.get("runtime_evidence_exact") is not True:
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_runtime_not_exact")
        if row.get("runtime_evidence_pair") != row.get("target_pair"):
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_runtime_pair_mismatch")
        if not strict_json_message_list(row.get("target_messages", [])):
            issues.append(f"{row_id}:stage_a_saved_output_evidence_candidate_routing_{label}_missing_json_target_message")
    return issues


def validate_stage_a_routing_defer_verify_contrast_pairs(
    rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if manifest.get("dataset") != STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_MANIFEST_DATASET:
        issues.append("stage_a_routing_defer_verify_contrast_manifest_dataset_mismatch")
    if manifest.get("pair_dataset") != STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_PAIR_DATASET:
        issues.append("stage_a_routing_defer_verify_contrast_pair_dataset_manifest_mismatch")
    if manifest.get("source_component_dataset") != STAGE_A_EVIDENCE_COMPONENT_DATASET:
        issues.append("stage_a_routing_defer_verify_contrast_source_dataset_manifest_mismatch")
    if manifest.get("prompt_contract") != STAGE_A_EVIDENCE_COMPONENT_PROMPT_CONTRACT:
        issues.append("stage_a_routing_defer_verify_contrast_prompt_contract_manifest_mismatch")
    if manifest.get("component") != "routing_after_loop":
        issues.append("stage_a_routing_defer_verify_contrast_component_manifest_mismatch")
    if manifest.get("failure_mode") != STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_FAILURE_MODE:
        issues.append("stage_a_routing_defer_verify_contrast_failure_mode_manifest_mismatch")
    if manifest.get("contrast_axis") != "defer_verify_boundary":
        issues.append("stage_a_routing_defer_verify_contrast_axis_manifest_mismatch")
    if manifest.get("candidate_policy") != STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_CANDIDATE_POLICY:
        issues.append("stage_a_routing_defer_verify_contrast_candidate_policy_manifest_mismatch")
    if manifest.get("rejected_by_chosen_pair") != STAGE_A_ROUTING_DEFER_VERIFY_REJECTED_BY_CHOSEN_PAIR:
        issues.append("stage_a_routing_defer_verify_contrast_rejection_map_manifest_mismatch")

    expected_counts = {
        "pair_examples": len(rows),
        "train_pairs": len(train_rows),
        "heldout_pairs": len(heldout_rows),
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_routing_defer_verify_contrast_{key}_manifest_mismatch")
    if manifest.get("by_chosen_pair") != count_by_key(rows, "chosen_pair"):
        issues.append("stage_a_routing_defer_verify_contrast_by_chosen_pair_manifest_mismatch")
    if manifest.get("by_rejected_pair") != count_by_key(rows, "rejected_pair"):
        issues.append("stage_a_routing_defer_verify_contrast_by_rejected_pair_manifest_mismatch")
    if manifest.get("train_by_chosen_pair") != count_by_key(train_rows, "chosen_pair"):
        issues.append("stage_a_routing_defer_verify_contrast_train_by_chosen_pair_manifest_mismatch")
    if manifest.get("heldout_by_chosen_pair") != count_by_key(heldout_rows, "chosen_pair"):
        issues.append("stage_a_routing_defer_verify_contrast_heldout_by_chosen_pair_manifest_mismatch")

    train_case_ids = case_id_values(train_rows)
    heldout_case_ids = case_id_values(heldout_rows)
    overlap_case_ids = sorted(train_case_ids & heldout_case_ids)
    if overlap_case_ids:
        issues.append("stage_a_routing_defer_verify_contrast_case_overlap")
    if manifest.get("overlap_case_ids") != overlap_case_ids:
        issues.append("stage_a_routing_defer_verify_contrast_case_overlap_manifest_mismatch")
    if manifest.get("train_case_ids") != sorted(train_case_ids):
        issues.append("stage_a_routing_defer_verify_contrast_train_case_ids_manifest_mismatch")
    if manifest.get("heldout_case_ids") != sorted(heldout_case_ids):
        issues.append("stage_a_routing_defer_verify_contrast_heldout_case_ids_manifest_mismatch")

    train_split_groups = value_set(train_rows, "split_group")
    heldout_split_groups = value_set(heldout_rows, "split_group")
    overlap_split_groups = sorted(train_split_groups & heldout_split_groups)
    if overlap_split_groups:
        issues.append("stage_a_routing_defer_verify_contrast_split_group_overlap")
    if manifest.get("overlap_split_groups") != overlap_split_groups:
        issues.append("stage_a_routing_defer_verify_contrast_split_group_overlap_manifest_mismatch")

    train_source_task_ids = value_set(train_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_rows, "source_task_id")
    overlap_source_task_ids = sorted(train_source_task_ids & heldout_source_task_ids)
    if overlap_source_task_ids:
        issues.append("stage_a_routing_defer_verify_contrast_source_task_overlap")
    if manifest.get("overlap_source_task_ids") != overlap_source_task_ids:
        issues.append("stage_a_routing_defer_verify_contrast_source_task_overlap_manifest_mismatch")

    issues.extend(validate_stage_a_routing_defer_verify_contrast_rows(rows, expected_split=None, label="pairs"))
    issues.extend(
        validate_stage_a_routing_defer_verify_contrast_rows(train_rows, expected_split="train", label="train")
    )
    issues.extend(
        validate_stage_a_routing_defer_verify_contrast_rows(heldout_rows, expected_split="heldout", label="heldout")
    )
    return issues


def validate_stage_a_routing_defer_verify_contrast_rows(
    rows: list[Mapping[str, Any]],
    *,
    expected_split: str | None,
    label: str,
) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    allowed_chosen = set(STAGE_A_ROUTING_DEFER_VERIFY_REJECTED_BY_CHOSEN_PAIR)
    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_duplicate_id")
        seen_ids.add(str(row_id))
        if row.get("dataset") != STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_PAIR_DATASET:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_unexpected_dataset")
        if expected_split is not None and row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_unexpected_split")
        if row.get("component") != "routing_after_loop":
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_wrong_component")
        if row.get("failure_mode") != STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_FAILURE_MODE:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_wrong_failure_mode")
        if row.get("contrast_axis") != "defer_verify_boundary":
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_wrong_axis")
        if row.get("candidate_policy") != STAGE_A_ROUTING_DEFER_VERIFY_CONTRAST_CANDIDATE_POLICY:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_wrong_candidate_policy")
        if row.get("prompt_contract") != STAGE_A_EVIDENCE_COMPONENT_PROMPT_CONTRACT:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_wrong_prompt_contract")
        if row.get("source_component_dataset") != STAGE_A_EVIDENCE_COMPONENT_DATASET:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_wrong_source_dataset")
        if row.get("target_keys") != list(STAGE_A_STRICT_COMPONENT_TARGET_KEYS["routing_after_loop"]):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_target_keys_mismatch")
        if row.get("oracle_target") is not True:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_missing_oracle_target")

        prompt_messages = row.get("prompt_messages", [])
        if not isinstance(prompt_messages, list) or len(prompt_messages) != 2:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_bad_prompt_message_count")
        else:
            if [message.get("role") for message in prompt_messages] != ["system", "user"]:
                issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_bad_prompt_roles")
            issues.extend(stage_a_prompt_leak_issues(row_id, prompt_messages, row))

        chosen = row.get("chosen_output")
        rejected = row.get("rejected_output")
        issues.extend(validate_stage_a_strict_component_target(row_id, "routing_after_loop", chosen))
        issues.extend(validate_stage_a_strict_component_target(row_id, "routing_after_loop", rejected))
        if not isinstance(chosen, Mapping) or not isinstance(rejected, Mapping):
            continue
        if chosen == rejected:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_chosen_equals_rejected")
        chosen_pair = row.get("chosen_pair")
        rejected_pair = row.get("rejected_pair")
        if chosen_pair not in allowed_chosen:
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_unexpected_chosen_pair")
        expected_rejected = STAGE_A_ROUTING_DEFER_VERIFY_REJECTED_BY_CHOSEN_PAIR.get(str(chosen_pair))
        if expected_rejected is not None:
            for key, value in expected_rejected.items():
                if rejected.get(key) != value:
                    issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_bad_rejected_{key}")
        if row.get("chosen_pair") == row.get("rejected_pair"):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_pair_labels_equal")
        if rejected.get("cited_source_ids"):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_unexpected_rejected_citation")
        if chosen.get("cited_source_ids"):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_unexpected_chosen_citation")
        if rejected_pair != pair_label(rejected):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_rejected_pair_label_mismatch")
        if row.get("chosen_pair") != pair_label(chosen):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_chosen_pair_label_mismatch")
        if not row.get("chosen_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_chosen_not_passing")
        if row.get("rejected_score", {}).get("passed"):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_rejected_is_passing")
        if "target_mismatch" not in row.get("rejected_score", {}).get("violations", []):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_rejected_missing_target_mismatch")
        if not strict_json_message_list(row.get("chosen_messages", [])):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_chosen_missing_json_assistant_target")
        if not strict_json_message_list(row.get("rejected_messages", [])):
            issues.append(f"{row_id}:stage_a_routing_defer_verify_contrast_{label}_rejected_missing_json_assistant_target")
    return issues


def strict_component_tool_call_ok(call: Any) -> bool:
    if not isinstance(call, Mapping):
        return False
    if not isinstance(call.get("name"), str) or not call.get("name"):
        return False
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        return False
    return {"drug_id", "condition_id"}.issubset(arguments)


def ends_with_submit_decision(messages: list[Any]) -> bool:
    return bool(messages) and messages[-1].get("tool_call", {}).get("name") == "submit_decision"


def stage_a_prompt_leak_issues(
    row_id: Any,
    prompt_messages: list[Any],
    row: Mapping[str, Any],
) -> list[str]:
    prompt_text = json.dumps(prompt_messages, sort_keys=True)
    issues: list[str] = []
    for term in STAGE_A_PROMPT_LEAK_TERMS:
        if term in prompt_text:
            issues.append(f"{row_id}:stage_a_prompt_leaks_{term}")
    source_task_id = row.get("source_task_id")
    if source_task_id and str(source_task_id) in prompt_text:
        issues.append(f"{row_id}:stage_a_prompt_leaks_source_task_id_value")
    split_group = row.get("split_group")
    if split_group and str(split_group) in prompt_text:
        issues.append(f"{row_id}:stage_a_prompt_leaks_split_group_value")
    return issues


def validate_stage_a_split(
    train_sft_rows: list[Mapping[str, Any]],
    heldout_sft_rows: list[Mapping[str, Any]],
    train_preference_rows: list[Mapping[str, Any]],
    heldout_preference_rows: list[Mapping[str, Any]],
    train_process_rows: list[Mapping[str, Any]],
    heldout_process_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    if manifest.get("dataset") != STAGE_A_SPLIT_DATASET:
        issues.append("stage_a_split_dataset_manifest_mismatch")
    if manifest.get("split_unit") != "source_manifest_case_id":
        issues.append("stage_a_split_unit_manifest_mismatch")

    expected_counts = {
        "train_sft_examples": len(train_sft_rows),
        "heldout_sft_examples": len(heldout_sft_rows),
        "train_preference_pairs": len(train_preference_rows),
        "heldout_preference_pairs": len(heldout_preference_rows),
        "train_process_examples": len(train_process_rows),
        "heldout_process_examples": len(heldout_process_rows),
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_split_{key}_manifest_mismatch")

    train_case_ids = case_id_values(train_sft_rows)
    heldout_case_ids = case_id_values(heldout_sft_rows)
    train_preference_case_ids = case_id_values(train_preference_rows)
    heldout_preference_case_ids = case_id_values(heldout_preference_rows)
    train_process_case_ids = case_id_values(train_process_rows)
    heldout_process_case_ids = case_id_values(heldout_process_rows)

    if train_preference_case_ids != train_case_ids:
        issues.append("stage_a_split_train_preference_case_mismatch")
    if heldout_preference_case_ids != heldout_case_ids:
        issues.append("stage_a_split_heldout_preference_case_mismatch")
    if train_process_case_ids != train_case_ids:
        issues.append("stage_a_split_train_process_case_mismatch")
    if heldout_process_case_ids != heldout_case_ids:
        issues.append("stage_a_split_heldout_process_case_mismatch")

    overlap_case_ids = sorted(train_case_ids & heldout_case_ids)
    if overlap_case_ids:
        issues.append("stage_a_split_case_overlap")
    if manifest.get("overlap_case_ids") != overlap_case_ids:
        issues.append("stage_a_split_case_overlap_manifest_mismatch")
    if manifest.get("train_case_ids") != sorted(train_case_ids):
        issues.append("stage_a_split_train_case_ids_manifest_mismatch")
    if manifest.get("heldout_case_ids") != sorted(heldout_case_ids):
        issues.append("stage_a_split_heldout_case_ids_manifest_mismatch")

    train_split_groups = value_set(train_sft_rows, "split_group")
    heldout_split_groups = value_set(heldout_sft_rows, "split_group")
    overlap_split_groups = sorted(train_split_groups & heldout_split_groups)
    if overlap_split_groups:
        issues.append("stage_a_split_group_overlap")
    if manifest.get("overlap_split_groups") != overlap_split_groups:
        issues.append("stage_a_split_group_overlap_manifest_mismatch")
    if manifest.get("train_split_groups") != sorted(train_split_groups):
        issues.append("stage_a_split_train_groups_manifest_mismatch")
    if manifest.get("heldout_split_groups") != sorted(heldout_split_groups):
        issues.append("stage_a_split_heldout_groups_manifest_mismatch")

    train_source_task_ids = value_set(train_sft_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_sft_rows, "source_task_id")
    overlap_source_task_ids = sorted(train_source_task_ids & heldout_source_task_ids)
    if overlap_source_task_ids:
        issues.append("stage_a_split_source_task_overlap")
    if manifest.get("overlap_source_task_ids") != overlap_source_task_ids:
        issues.append("stage_a_split_source_task_overlap_manifest_mismatch")

    count_expectations = {
        "train_by_case_family": count_by_key(train_sft_rows, "case_family"),
        "heldout_by_case_family": count_by_key(heldout_sft_rows, "case_family"),
        "train_by_evidence_status": count_by_key(train_sft_rows, "gold_evidence_status"),
        "heldout_by_evidence_status": count_by_key(heldout_sft_rows, "gold_evidence_status"),
        "train_preference_failure_modes": count_by_key(train_preference_rows, "failure_mode"),
        "heldout_preference_failure_modes": count_by_key(heldout_preference_rows, "failure_mode"),
    }
    for key, value in count_expectations.items():
        if manifest.get(key) != value:
            issues.append(f"stage_a_split_{key}_manifest_mismatch")

    issues.extend(validate_stage_a_split_rows(train_sft_rows, "train", STAGE_A_SFT_DATASET, "train_sft"))
    issues.extend(validate_stage_a_split_rows(heldout_sft_rows, "heldout", STAGE_A_SFT_DATASET, "heldout_sft"))
    issues.extend(
        validate_stage_a_split_rows(
            train_preference_rows,
            "train",
            STAGE_A_PREFERENCE_DATASET,
            "train_preference",
        )
    )
    issues.extend(
        validate_stage_a_split_rows(
            heldout_preference_rows,
            "heldout",
            STAGE_A_PREFERENCE_DATASET,
            "heldout_preference",
        )
    )
    issues.extend(validate_stage_a_split_rows(train_process_rows, "train", STAGE_A_PROCESS_DATASET, "train_process"))
    issues.extend(
        validate_stage_a_split_rows(heldout_process_rows, "heldout", STAGE_A_PROCESS_DATASET, "heldout_process")
    )

    return issues


def validate_stage_a_split_rows(
    rows: list[Mapping[str, Any]],
    expected_split: str,
    expected_dataset: str,
    label: str,
) -> list[str]:
    issues: list[str] = []
    for row in rows:
        row_id = row.get("id")
        if row.get("split") != expected_split:
            issues.append(f"{row_id}:stage_a_{label}_unexpected_split")
        if row.get("dataset") != expected_dataset:
            issues.append(f"{row_id}:stage_a_{label}_unexpected_dataset")
    return issues


def case_id_values(rows: list[Mapping[str, Any]]) -> set[str]:
    return value_set(rows, "source_manifest_case_id")


def value_set(rows: list[Mapping[str, Any]], key: str) -> set[str]:
    return {str(row.get(key)) for row in rows if row.get(key)}


def count_by_key(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def validate_boundary_preferences(
    preference_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    dataset = manifest.get("dataset")
    if len(preference_rows) != manifest.get("preference_pairs"):
        issues.append("boundary_preference_count_manifest_mismatch")

    failure_modes = dict(sorted(Counter(row.get("failure_mode") for row in preference_rows).items()))
    if failure_modes != manifest.get("pairs_by_failure_mode"):
        issues.append("boundary_preference_failure_modes_manifest_mismatch")

    chosen_actions = dict(sorted(Counter(row.get("evidence_derived_action") for row in preference_rows).items()))
    if chosen_actions != manifest.get("pairs_by_chosen_action"):
        issues.append("boundary_preference_chosen_actions_manifest_mismatch")

    rejected_actions = dict(sorted(Counter(row.get("rejected_action") for row in preference_rows).items()))
    if rejected_actions != manifest.get("pairs_by_rejected_action"):
        issues.append("boundary_preference_rejected_actions_manifest_mismatch")

    for row in preference_rows:
        row_id = row.get("id")
        if row.get("dataset") != dataset:
            issues.append(f"{row_id}:boundary_preference_unexpected_dataset")
        if row.get("tool_profile") != "native_ct":
            issues.append(f"{row_id}:boundary_preference_unexpected_tool_profile")
        if not row.get("chosen_score", {}).get("passed"):
            issues.append(f"{row_id}:boundary_preference_chosen_not_passing")
        if row.get("rejected_score", {}).get("passed"):
            issues.append(f"{row_id}:boundary_preference_rejected_is_passing")
        prompt_messages = row.get("prompt_messages", [])
        if not prompt_messages or prompt_messages[-1].get("tool_call", {}).get("name") == "submit_decision":
            issues.append(f"{row_id}:boundary_preference_prompt_contains_final_decision")
        if len(row.get("chosen_messages", [])) != 1:
            issues.append(f"{row_id}:boundary_preference_chosen_should_be_final_only")
        if len(row.get("rejected_messages", [])) != 1:
            issues.append(f"{row_id}:boundary_preference_rejected_should_be_final_only")
        chosen_final = row.get("chosen_messages", [{}])[0].get("tool_call", {})
        rejected_final = row.get("rejected_messages", [{}])[0].get("tool_call", {})
        if chosen_final.get("name") != "submit_decision":
            issues.append(f"{row_id}:boundary_preference_chosen_missing_submit_decision")
        if rejected_final.get("name") != "submit_decision":
            issues.append(f"{row_id}:boundary_preference_rejected_missing_submit_decision")
        prompt_text = json.dumps(prompt_messages, sort_keys=True)
        if "gold_action" in prompt_text or "scoring_key" in prompt_text:
            issues.append(f"{row_id}:boundary_preference_hidden_key_leaked_into_prompt")

    return issues


def split_boundary_manifest_part(manifest: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "dataset": manifest.get(f"{split}_dataset"),
        "preference_pairs": manifest.get(f"{split}_pairs"),
        "pairs_by_failure_mode": manifest.get(f"{split}_by_failure_mode"),
        "pairs_by_chosen_action": manifest.get(f"{split}_by_chosen_action"),
        "pairs_by_rejected_action": manifest.get(f"{split}_by_rejected_action"),
    }


def validate_boundary_preference_split(
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []

    issues.extend(validate_boundary_preferences(train_rows, split_boundary_manifest_part(manifest, "train")))
    issues.extend(validate_boundary_preferences(heldout_rows, split_boundary_manifest_part(manifest, "heldout")))

    train_source_ids = {str(row.get("source_hard_preference_id")) for row in train_rows}
    heldout_source_ids = {str(row.get("source_hard_preference_id")) for row in heldout_rows}
    overlap_source_ids = sorted(train_source_ids & heldout_source_ids)
    if overlap_source_ids:
        issues.append("hard_boundary_preference_split_source_overlap")
    if manifest.get("overlap_source_ids") != overlap_source_ids:
        issues.append("hard_boundary_preference_split_overlap_manifest_mismatch")
    if sorted(train_source_ids) != manifest.get("train_source_ids"):
        issues.append("hard_boundary_preference_train_source_ids_manifest_mismatch")
    if sorted(heldout_source_ids) != manifest.get("heldout_source_ids"):
        issues.append("hard_boundary_preference_heldout_source_ids_manifest_mismatch")

    for row in train_rows:
        row_id = row.get("id")
        if row.get("split") != "train":
            issues.append(f"{row_id}:hard_boundary_preference_unexpected_train_split")
        if not row.get("source_hard_preference_id"):
            issues.append(f"{row_id}:hard_boundary_preference_missing_source_id")
    for row in heldout_rows:
        row_id = row.get("id")
        if row.get("split") != "heldout":
            issues.append(f"{row_id}:hard_boundary_preference_unexpected_heldout_split")
        if not row.get("source_hard_preference_id"):
            issues.append(f"{row_id}:hard_boundary_preference_missing_source_id")

    return issues


def validate_sft_rows(
    rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    label: str,
    expected_dataset: str,
    require_oracle_target: bool = False,
) -> list[str]:
    issues: list[str] = []

    if len(rows) != manifest.get("sft_examples"):
        issues.append(f"{label}_count_manifest_mismatch")
    if manifest.get("dataset") != expected_dataset:
        issues.append(f"{label}_dataset_manifest_mismatch")

    expected_by_class = manifest.get("by_class")
    if expected_by_class is not None:
        by_class = dict(sorted(Counter(row.get("action_class") for row in rows).items()))
        if by_class != expected_by_class:
            issues.append(f"{label}_by_class_manifest_mismatch")

    for row in rows:
        row_id = row.get("id")
        if row.get("dataset") != expected_dataset:
            issues.append(f"{row_id}:unexpected_dataset")
        if row.get("tool_profile") != "native_ct":
            issues.append(f"{row_id}:unexpected_tool_profile")
        if require_oracle_target and row.get("oracle_target") is not True:
            issues.append(f"{row_id}:missing_oracle_target")
        messages = row.get("messages", [])
        if not messages or messages[-1].get("tool_call", {}).get("name") != "submit_decision":
            issues.append(f"{row_id}:missing_final_submit_decision")
        prompt_text = json.dumps(messages, sort_keys=True)
        if "gold_action" in prompt_text or "scoring_key" in prompt_text:
            issues.append(f"{row_id}:hidden_key_leaked_into_messages")
        if row.get("score", {}).get("violations"):
            issues.append(f"{row_id}:sft_example_has_violations")

    return issues


def validate_oracle_sft(
    rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues = validate_sft_rows(
        rows,
        manifest,
        label="oracle_sft",
        expected_dataset="negbiodb_ct_oracle_sft_v1",
        require_oracle_target=True,
    )
    if manifest.get("source_runner") != "deterministic_oracle_policy":
        issues.append("oracle_sft_source_runner_manifest_mismatch")
    if manifest.get("skipped"):
        issues.append("oracle_sft_skipped_rows_present")
    if "not live runner behavior" not in str(manifest.get("boundary", "")):
        issues.append("oracle_sft_boundary_missing")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--preferences", default="post_training/negbiodb_ct_native_preferences_v1.jsonl")
    parser.add_argument("--manifest", default="post_training/negbiodb_ct_native_manifest.json")
    parser.add_argument("--oracle-sft", default="post_training/negbiodb_ct_oracle_sft_v1.jsonl")
    parser.add_argument("--oracle-manifest", default="post_training/negbiodb_ct_oracle_sft_manifest.json")
    parser.add_argument(
        "--boundary-preferences",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_v1.jsonl",
    )
    parser.add_argument(
        "--boundary-preference-manifest",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_manifest.json",
    )
    parser.add_argument(
        "--hard-boundary-preferences",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_v1.jsonl",
    )
    parser.add_argument(
        "--hard-boundary-preference-manifest",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_manifest.json",
    )
    parser.add_argument(
        "--hard-boundary-preference-train",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl",
    )
    parser.add_argument(
        "--hard-boundary-preference-heldout",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--hard-boundary-preference-split-manifest",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_split_manifest.json",
    )
    parser.add_argument("--stage-a-sft", default="post_training/stage_a_sft_v1.jsonl")
    parser.add_argument("--stage-a-preferences", default="post_training/stage_a_preferences_v1.jsonl")
    parser.add_argument("--stage-a-process", default="post_training/stage_a_process_supervision_v1.jsonl")
    parser.add_argument("--stage-a-manifest", default="post_training/stage_a_export_manifest.json")
    parser.add_argument("--stage-a-sft-train", default="post_training/stage_a_sft_train_v1.jsonl")
    parser.add_argument("--stage-a-sft-heldout", default="post_training/stage_a_sft_heldout_v1.jsonl")
    parser.add_argument("--stage-a-preferences-train", default="post_training/stage_a_preferences_train_v1.jsonl")
    parser.add_argument("--stage-a-preferences-heldout", default="post_training/stage_a_preferences_heldout_v1.jsonl")
    parser.add_argument("--stage-a-process-train", default="post_training/stage_a_process_train_v1.jsonl")
    parser.add_argument("--stage-a-process-heldout", default="post_training/stage_a_process_heldout_v1.jsonl")
    parser.add_argument("--stage-a-split-manifest", default="post_training/stage_a_split_manifest.json")
    parser.add_argument("--stage-a-strict-sft", default="post_training/stage_a_strict_contract_sft_v1.jsonl")
    parser.add_argument(
        "--stage-a-strict-preferences",
        default="post_training/stage_a_strict_contract_preferences_v1.jsonl",
    )
    parser.add_argument("--stage-a-strict-process", default="post_training/stage_a_strict_contract_process_v1.jsonl")
    parser.add_argument(
        "--stage-a-strict-sft-train",
        default="post_training/stage_a_strict_contract_sft_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-strict-sft-heldout",
        default="post_training/stage_a_strict_contract_sft_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-strict-preferences-train",
        default="post_training/stage_a_strict_contract_preferences_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-strict-preferences-heldout",
        default="post_training/stage_a_strict_contract_preferences_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-strict-process-train",
        default="post_training/stage_a_strict_contract_process_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-strict-process-heldout",
        default="post_training/stage_a_strict_contract_process_heldout_v1.jsonl",
    )
    parser.add_argument("--stage-a-strict-manifest", default="post_training/stage_a_strict_contract_manifest.json")
    parser.add_argument(
        "--stage-a-strict-component-targets",
        default="post_training/stage_a_strict_component_targets_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-strict-component-targets-train",
        default="post_training/stage_a_strict_component_targets_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-strict-component-targets-heldout",
        default="post_training/stage_a_strict_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-strict-component-targets-manifest",
        default="post_training/stage_a_strict_component_targets_manifest.json",
    )
    parser.add_argument(
        "--stage-a-evidence-component-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-evidence-component-targets-train",
        default="post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-evidence-component-targets-heldout",
        default="post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-evidence-component-targets-manifest",
        default="post_training/stage_a_evidence_conditioned_component_targets_manifest.json",
    )
    parser.add_argument(
        "--stage-a-enum-corrective-pairs",
        default="post_training/stage_a_enum_corrective_pairs_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-enum-corrective-pairs-train",
        default="post_training/stage_a_enum_corrective_pairs_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-enum-corrective-pairs-heldout",
        default="post_training/stage_a_enum_corrective_pairs_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-enum-corrective-pairs-manifest",
        default="post_training/stage_a_enum_corrective_pairs_manifest.json",
    )
    parser.add_argument(
        "--stage-a-enum-action-contrast-pairs",
        default="post_training/stage_a_enum_action_contrast_pairs_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-enum-action-contrast-pairs-train",
        default="post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-enum-action-contrast-pairs-heldout",
        default="post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-enum-action-contrast-pairs-manifest",
        default="post_training/stage_a_enum_action_contrast_pairs_manifest.json",
    )
    parser.add_argument(
        "--stage-a-routing-action-status-contrast-pairs",
        default="post_training/stage_a_routing_action_status_contrast_pairs_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-routing-action-status-contrast-pairs-train",
        default="post_training/stage_a_routing_action_status_contrast_pairs_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-routing-action-status-contrast-pairs-heldout",
        default="post_training/stage_a_routing_action_status_contrast_pairs_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-routing-action-status-contrast-pairs-manifest",
        default="post_training/stage_a_routing_action_status_contrast_pairs_manifest.json",
    )
    parser.add_argument(
        "--stage-a-routing-defer-verify-contrast-pairs",
        default="post_training/stage_a_routing_defer_verify_contrast_pairs_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-routing-defer-verify-contrast-pairs-train",
        default="post_training/stage_a_routing_defer_verify_contrast_pairs_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-routing-defer-verify-contrast-pairs-heldout",
        default="post_training/stage_a_routing_defer_verify_contrast_pairs_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-routing-defer-verify-contrast-pairs-manifest",
        default="post_training/stage_a_routing_defer_verify_contrast_pairs_manifest.json",
    )
    parser.add_argument(
        "--stage-a-saved-output-calibration-probe-pairs",
        default="post_training/stage_a_saved_output_calibration_probe_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-saved-output-calibration-probe-pairs-train",
        default="post_training/stage_a_saved_output_calibration_probe_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-saved-output-calibration-probe-pairs-heldout",
        default="post_training/stage_a_saved_output_calibration_probe_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-saved-output-calibration-probe-pairs-manifest",
        default="post_training/stage_a_saved_output_calibration_probe_manifest.json",
    )
    parser.add_argument(
        "--stage-a-saved-output-evidence-candidate-routing-rows",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-saved-output-evidence-candidate-routing-train",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-saved-output-evidence-candidate-routing-heldout",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--stage-a-saved-output-evidence-candidate-routing-manifest",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_manifest.json",
    )
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-boundary-preferences", action="store_true")
    parser.add_argument("--skip-hard-boundary-preferences", action="store_true")
    parser.add_argument("--skip-hard-boundary-preference-split", action="store_true")
    parser.add_argument("--skip-stage-a", action="store_true")
    parser.add_argument("--skip-stage-a-split", action="store_true")
    parser.add_argument("--skip-stage-a-strict-contract", action="store_true")
    parser.add_argument("--skip-stage-a-strict-components", action="store_true")
    parser.add_argument("--skip-stage-a-evidence-components", action="store_true")
    parser.add_argument("--skip-stage-a-enum-corrective", action="store_true")
    parser.add_argument("--skip-stage-a-enum-action-contrast", action="store_true")
    parser.add_argument("--skip-stage-a-routing-action-status-contrast", action="store_true")
    parser.add_argument("--skip-stage-a-routing-defer-verify-contrast", action="store_true")
    parser.add_argument("--skip-stage-a-saved-output-calibration-probe", action="store_true")
    parser.add_argument("--skip-stage-a-saved-output-evidence-candidate-routing", action="store_true")
    args = parser.parse_args()

    sft_rows = load_jsonl(args.sft)
    preference_rows = load_jsonl(args.preferences)
    manifest = json.loads(Path(args.manifest).read_text())
    issues = validate(sft_rows, preference_rows, manifest)

    oracle_rows: list[dict[str, Any]] = []
    if not args.skip_oracle:
        oracle_rows = load_jsonl(args.oracle_sft)
        oracle_manifest = json.loads(Path(args.oracle_manifest).read_text())
        issues.extend(validate_oracle_sft(oracle_rows, oracle_manifest))

    boundary_preference_rows: list[dict[str, Any]] = []
    if not args.skip_boundary_preferences:
        boundary_preference_rows = load_jsonl(args.boundary_preferences)
        boundary_preference_manifest = json.loads(Path(args.boundary_preference_manifest).read_text())
        issues.extend(
            validate_boundary_preferences(
                boundary_preference_rows,
                boundary_preference_manifest,
            )
        )

    hard_boundary_preference_rows: list[dict[str, Any]] = []
    if not args.skip_hard_boundary_preferences:
        hard_boundary_preference_rows = load_jsonl(args.hard_boundary_preferences)
        hard_boundary_preference_manifest = json.loads(Path(args.hard_boundary_preference_manifest).read_text())
        issues.extend(
            validate_boundary_preferences(
                hard_boundary_preference_rows,
                hard_boundary_preference_manifest,
            )
        )

    hard_boundary_preference_train_rows: list[dict[str, Any]] = []
    hard_boundary_preference_heldout_rows: list[dict[str, Any]] = []
    if not args.skip_hard_boundary_preference_split:
        hard_boundary_preference_train_rows = load_jsonl(args.hard_boundary_preference_train)
        hard_boundary_preference_heldout_rows = load_jsonl(args.hard_boundary_preference_heldout)
        hard_boundary_preference_split_manifest = json.loads(
            Path(args.hard_boundary_preference_split_manifest).read_text()
        )
        issues.extend(
            validate_boundary_preference_split(
                hard_boundary_preference_train_rows,
                hard_boundary_preference_heldout_rows,
                hard_boundary_preference_split_manifest,
            )
        )

    stage_a_sft_rows: list[dict[str, Any]] = []
    stage_a_preference_rows: list[dict[str, Any]] = []
    stage_a_process_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a:
        stage_a_sft_rows = load_jsonl(args.stage_a_sft)
        stage_a_preference_rows = load_jsonl(args.stage_a_preferences)
        stage_a_process_rows = load_jsonl(args.stage_a_process)
        stage_a_manifest = json.loads(Path(args.stage_a_manifest).read_text())
        issues.extend(
            validate_stage_a_exports(
                stage_a_sft_rows,
                stage_a_preference_rows,
                stage_a_process_rows,
                stage_a_manifest,
            )
        )

    stage_a_train_sft_rows: list[dict[str, Any]] = []
    stage_a_heldout_sft_rows: list[dict[str, Any]] = []
    stage_a_train_preference_rows: list[dict[str, Any]] = []
    stage_a_heldout_preference_rows: list[dict[str, Any]] = []
    stage_a_train_process_rows: list[dict[str, Any]] = []
    stage_a_heldout_process_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a_split:
        stage_a_train_sft_rows = load_jsonl(args.stage_a_sft_train)
        stage_a_heldout_sft_rows = load_jsonl(args.stage_a_sft_heldout)
        stage_a_train_preference_rows = load_jsonl(args.stage_a_preferences_train)
        stage_a_heldout_preference_rows = load_jsonl(args.stage_a_preferences_heldout)
        stage_a_train_process_rows = load_jsonl(args.stage_a_process_train)
        stage_a_heldout_process_rows = load_jsonl(args.stage_a_process_heldout)
        stage_a_split_manifest = json.loads(Path(args.stage_a_split_manifest).read_text())
        issues.extend(
            validate_stage_a_split(
                stage_a_train_sft_rows,
                stage_a_heldout_sft_rows,
                stage_a_train_preference_rows,
                stage_a_heldout_preference_rows,
                stage_a_train_process_rows,
                stage_a_heldout_process_rows,
                stage_a_split_manifest,
            )
        )

    stage_a_strict_sft_rows: list[dict[str, Any]] = []
    stage_a_strict_preference_rows: list[dict[str, Any]] = []
    stage_a_strict_process_rows: list[dict[str, Any]] = []
    stage_a_strict_train_sft_rows: list[dict[str, Any]] = []
    stage_a_strict_heldout_sft_rows: list[dict[str, Any]] = []
    stage_a_strict_train_preference_rows: list[dict[str, Any]] = []
    stage_a_strict_heldout_preference_rows: list[dict[str, Any]] = []
    stage_a_strict_train_process_rows: list[dict[str, Any]] = []
    stage_a_strict_heldout_process_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a_strict_contract:
        stage_a_strict_sft_rows = load_jsonl(args.stage_a_strict_sft)
        stage_a_strict_preference_rows = load_jsonl(args.stage_a_strict_preferences)
        stage_a_strict_process_rows = load_jsonl(args.stage_a_strict_process)
        stage_a_strict_train_sft_rows = load_jsonl(args.stage_a_strict_sft_train)
        stage_a_strict_heldout_sft_rows = load_jsonl(args.stage_a_strict_sft_heldout)
        stage_a_strict_train_preference_rows = load_jsonl(args.stage_a_strict_preferences_train)
        stage_a_strict_heldout_preference_rows = load_jsonl(args.stage_a_strict_preferences_heldout)
        stage_a_strict_train_process_rows = load_jsonl(args.stage_a_strict_process_train)
        stage_a_strict_heldout_process_rows = load_jsonl(args.stage_a_strict_process_heldout)
        stage_a_strict_manifest = json.loads(Path(args.stage_a_strict_manifest).read_text())
        issues.extend(
            validate_stage_a_strict_contract(
                stage_a_strict_sft_rows,
                stage_a_strict_preference_rows,
                stage_a_strict_process_rows,
                stage_a_strict_train_sft_rows,
                stage_a_strict_heldout_sft_rows,
                stage_a_strict_train_preference_rows,
                stage_a_strict_heldout_preference_rows,
                stage_a_strict_train_process_rows,
                stage_a_strict_heldout_process_rows,
                stage_a_strict_manifest,
            )
        )

    stage_a_strict_component_rows: list[dict[str, Any]] = []
    stage_a_strict_component_train_rows: list[dict[str, Any]] = []
    stage_a_strict_component_heldout_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a_strict_components:
        stage_a_strict_component_rows = load_jsonl(args.stage_a_strict_component_targets)
        stage_a_strict_component_train_rows = load_jsonl(args.stage_a_strict_component_targets_train)
        stage_a_strict_component_heldout_rows = load_jsonl(args.stage_a_strict_component_targets_heldout)
        stage_a_strict_component_manifest = json.loads(
            Path(args.stage_a_strict_component_targets_manifest).read_text()
        )
        issues.extend(
            validate_stage_a_strict_component_targets(
                stage_a_strict_component_rows,
                stage_a_strict_component_train_rows,
                stage_a_strict_component_heldout_rows,
                stage_a_strict_component_manifest,
            )
        )

    stage_a_evidence_component_rows: list[dict[str, Any]] = []
    stage_a_evidence_component_train_rows: list[dict[str, Any]] = []
    stage_a_evidence_component_heldout_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a_evidence_components:
        stage_a_evidence_component_rows = load_jsonl(args.stage_a_evidence_component_targets)
        stage_a_evidence_component_train_rows = load_jsonl(args.stage_a_evidence_component_targets_train)
        stage_a_evidence_component_heldout_rows = load_jsonl(args.stage_a_evidence_component_targets_heldout)
        stage_a_evidence_component_manifest = json.loads(
            Path(args.stage_a_evidence_component_targets_manifest).read_text()
        )
        issues.extend(
            validate_stage_a_evidence_conditioned_component_targets(
                stage_a_evidence_component_rows,
                stage_a_evidence_component_train_rows,
                stage_a_evidence_component_heldout_rows,
                stage_a_evidence_component_manifest,
            )
        )

    stage_a_enum_corrective_rows: list[dict[str, Any]] = []
    stage_a_enum_corrective_train_rows: list[dict[str, Any]] = []
    stage_a_enum_corrective_heldout_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a_enum_corrective:
        stage_a_enum_corrective_rows = load_jsonl(args.stage_a_enum_corrective_pairs)
        stage_a_enum_corrective_train_rows = load_jsonl(args.stage_a_enum_corrective_pairs_train)
        stage_a_enum_corrective_heldout_rows = load_jsonl(args.stage_a_enum_corrective_pairs_heldout)
        stage_a_enum_corrective_manifest = json.loads(
            Path(args.stage_a_enum_corrective_pairs_manifest).read_text()
        )
        issues.extend(
            validate_stage_a_enum_corrective_pairs(
                stage_a_enum_corrective_rows,
                stage_a_enum_corrective_train_rows,
                stage_a_enum_corrective_heldout_rows,
                stage_a_enum_corrective_manifest,
            )
        )

    stage_a_enum_action_contrast_rows: list[dict[str, Any]] = []
    stage_a_enum_action_contrast_train_rows: list[dict[str, Any]] = []
    stage_a_enum_action_contrast_heldout_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a_enum_action_contrast:
        stage_a_enum_action_contrast_rows = load_jsonl(args.stage_a_enum_action_contrast_pairs)
        stage_a_enum_action_contrast_train_rows = load_jsonl(args.stage_a_enum_action_contrast_pairs_train)
        stage_a_enum_action_contrast_heldout_rows = load_jsonl(args.stage_a_enum_action_contrast_pairs_heldout)
        stage_a_enum_action_contrast_manifest = json.loads(
            Path(args.stage_a_enum_action_contrast_pairs_manifest).read_text()
        )
        issues.extend(
            validate_stage_a_enum_action_contrast_pairs(
                stage_a_enum_action_contrast_rows,
                stage_a_enum_action_contrast_train_rows,
                stage_a_enum_action_contrast_heldout_rows,
                stage_a_enum_action_contrast_manifest,
            )
        )

    stage_a_routing_action_status_contrast_rows: list[dict[str, Any]] = []
    stage_a_routing_action_status_contrast_train_rows: list[dict[str, Any]] = []
    stage_a_routing_action_status_contrast_heldout_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a_routing_action_status_contrast:
        stage_a_routing_action_status_contrast_rows = load_jsonl(
            args.stage_a_routing_action_status_contrast_pairs
        )
        stage_a_routing_action_status_contrast_train_rows = load_jsonl(
            args.stage_a_routing_action_status_contrast_pairs_train
        )
        stage_a_routing_action_status_contrast_heldout_rows = load_jsonl(
            args.stage_a_routing_action_status_contrast_pairs_heldout
        )
        stage_a_routing_action_status_contrast_manifest = json.loads(
            Path(args.stage_a_routing_action_status_contrast_pairs_manifest).read_text()
        )
        issues.extend(
            validate_stage_a_routing_action_status_contrast_pairs(
                stage_a_routing_action_status_contrast_rows,
                stage_a_routing_action_status_contrast_train_rows,
                stage_a_routing_action_status_contrast_heldout_rows,
                stage_a_routing_action_status_contrast_manifest,
            )
        )

    stage_a_routing_defer_verify_contrast_rows: list[dict[str, Any]] = []
    stage_a_routing_defer_verify_contrast_train_rows: list[dict[str, Any]] = []
    stage_a_routing_defer_verify_contrast_heldout_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a_routing_defer_verify_contrast:
        stage_a_routing_defer_verify_contrast_rows = load_jsonl(
            args.stage_a_routing_defer_verify_contrast_pairs
        )
        stage_a_routing_defer_verify_contrast_train_rows = load_jsonl(
            args.stage_a_routing_defer_verify_contrast_pairs_train
        )
        stage_a_routing_defer_verify_contrast_heldout_rows = load_jsonl(
            args.stage_a_routing_defer_verify_contrast_pairs_heldout
        )
        stage_a_routing_defer_verify_contrast_manifest = json.loads(
            Path(args.stage_a_routing_defer_verify_contrast_pairs_manifest).read_text()
        )
        issues.extend(
            validate_stage_a_routing_defer_verify_contrast_pairs(
                stage_a_routing_defer_verify_contrast_rows,
                stage_a_routing_defer_verify_contrast_train_rows,
                stage_a_routing_defer_verify_contrast_heldout_rows,
                stage_a_routing_defer_verify_contrast_manifest,
            )
        )

    stage_a_saved_output_calibration_probe_rows: list[dict[str, Any]] = []
    stage_a_saved_output_calibration_probe_train_rows: list[dict[str, Any]] = []
    stage_a_saved_output_calibration_probe_heldout_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a_saved_output_calibration_probe:
        stage_a_saved_output_calibration_probe_rows = load_jsonl(
            args.stage_a_saved_output_calibration_probe_pairs
        )
        stage_a_saved_output_calibration_probe_train_rows = load_jsonl(
            args.stage_a_saved_output_calibration_probe_pairs_train
        )
        stage_a_saved_output_calibration_probe_heldout_rows = load_jsonl(
            args.stage_a_saved_output_calibration_probe_pairs_heldout
        )
        stage_a_saved_output_calibration_probe_manifest = json.loads(
            Path(args.stage_a_saved_output_calibration_probe_pairs_manifest).read_text()
        )
        issues.extend(
            validate_stage_a_saved_output_calibration_probe(
                stage_a_saved_output_calibration_probe_rows,
                stage_a_saved_output_calibration_probe_train_rows,
                stage_a_saved_output_calibration_probe_heldout_rows,
                stage_a_saved_output_calibration_probe_manifest,
            )
        )

    stage_a_saved_output_evidence_candidate_routing_rows: list[dict[str, Any]] = []
    stage_a_saved_output_evidence_candidate_routing_train_rows: list[dict[str, Any]] = []
    stage_a_saved_output_evidence_candidate_routing_heldout_rows: list[dict[str, Any]] = []
    if not args.skip_stage_a_saved_output_evidence_candidate_routing:
        stage_a_saved_output_evidence_candidate_routing_rows = load_jsonl(
            args.stage_a_saved_output_evidence_candidate_routing_rows
        )
        stage_a_saved_output_evidence_candidate_routing_train_rows = load_jsonl(
            args.stage_a_saved_output_evidence_candidate_routing_train
        )
        stage_a_saved_output_evidence_candidate_routing_heldout_rows = load_jsonl(
            args.stage_a_saved_output_evidence_candidate_routing_heldout
        )
        stage_a_saved_output_evidence_candidate_routing_manifest = json.loads(
            Path(args.stage_a_saved_output_evidence_candidate_routing_manifest).read_text()
        )
        issues.extend(
            validate_stage_a_saved_output_evidence_candidate_routing(
                stage_a_saved_output_evidence_candidate_routing_rows,
                stage_a_saved_output_evidence_candidate_routing_train_rows,
                stage_a_saved_output_evidence_candidate_routing_heldout_rows,
                stage_a_saved_output_evidence_candidate_routing_manifest,
            )
        )

    summary = {
        "sft_examples": len(sft_rows),
        "preference_pairs": len(preference_rows),
        "preference_failure_modes": dict(Counter(row["failure_mode"] for row in preference_rows)),
        "issues": issues,
    }
    if not args.skip_oracle:
        summary["oracle_sft_examples"] = len(oracle_rows)
        summary["oracle_sft_by_class"] = dict(
            sorted(Counter(row["action_class"] for row in oracle_rows).items())
        )
    if not args.skip_boundary_preferences:
        summary["boundary_preference_pairs"] = len(boundary_preference_rows)
        summary["boundary_preference_failure_modes"] = dict(
            sorted(Counter(row["failure_mode"] for row in boundary_preference_rows).items())
        )
    if not args.skip_hard_boundary_preferences:
        summary["hard_boundary_preference_pairs"] = len(hard_boundary_preference_rows)
        summary["hard_boundary_preference_failure_modes"] = dict(
            sorted(Counter(row["failure_mode"] for row in hard_boundary_preference_rows).items())
        )
    if not args.skip_hard_boundary_preference_split:
        summary["hard_boundary_preference_train_pairs"] = len(hard_boundary_preference_train_rows)
        summary["hard_boundary_preference_train_failure_modes"] = dict(
            sorted(Counter(row["failure_mode"] for row in hard_boundary_preference_train_rows).items())
        )
        summary["hard_boundary_preference_heldout_pairs"] = len(hard_boundary_preference_heldout_rows)
        summary["hard_boundary_preference_heldout_failure_modes"] = dict(
            sorted(Counter(row["failure_mode"] for row in hard_boundary_preference_heldout_rows).items())
        )
    if not args.skip_stage_a:
        summary["stage_a_sft_examples"] = len(stage_a_sft_rows)
        summary["stage_a_preference_pairs"] = len(stage_a_preference_rows)
        summary["stage_a_process_examples"] = len(stage_a_process_rows)
        summary["stage_a_preference_failure_modes"] = dict(
            sorted(Counter(row["failure_mode"] for row in stage_a_preference_rows).items())
        )
    if not args.skip_stage_a_split:
        summary["stage_a_train_sft_examples"] = len(stage_a_train_sft_rows)
        summary["stage_a_heldout_sft_examples"] = len(stage_a_heldout_sft_rows)
        summary["stage_a_train_preference_pairs"] = len(stage_a_train_preference_rows)
        summary["stage_a_heldout_preference_pairs"] = len(stage_a_heldout_preference_rows)
        summary["stage_a_train_process_examples"] = len(stage_a_train_process_rows)
        summary["stage_a_heldout_process_examples"] = len(stage_a_heldout_process_rows)
    if not args.skip_stage_a_strict_contract:
        summary["stage_a_strict_sft_examples"] = len(stage_a_strict_sft_rows)
        summary["stage_a_strict_preference_pairs"] = len(stage_a_strict_preference_rows)
        summary["stage_a_strict_process_examples"] = len(stage_a_strict_process_rows)
        summary["stage_a_strict_preference_failure_modes"] = dict(
            sorted(Counter(row["failure_mode"] for row in stage_a_strict_preference_rows).items())
        )
        summary["stage_a_strict_train_sft_examples"] = len(stage_a_strict_train_sft_rows)
        summary["stage_a_strict_heldout_sft_examples"] = len(stage_a_strict_heldout_sft_rows)
        summary["stage_a_strict_train_preference_pairs"] = len(stage_a_strict_train_preference_rows)
        summary["stage_a_strict_heldout_preference_pairs"] = len(stage_a_strict_heldout_preference_rows)
        summary["stage_a_strict_train_process_examples"] = len(stage_a_strict_train_process_rows)
        summary["stage_a_strict_heldout_process_examples"] = len(stage_a_strict_heldout_process_rows)
    if not args.skip_stage_a_strict_components:
        summary["stage_a_strict_component_target_examples"] = len(stage_a_strict_component_rows)
        summary["stage_a_strict_component_train_examples"] = len(stage_a_strict_component_train_rows)
        summary["stage_a_strict_component_heldout_examples"] = len(stage_a_strict_component_heldout_rows)
        summary["stage_a_strict_component_by_component"] = dict(
            sorted(Counter(row["component"] for row in stage_a_strict_component_rows).items())
        )
    if not args.skip_stage_a_evidence_components:
        summary["stage_a_evidence_component_target_examples"] = len(stage_a_evidence_component_rows)
        summary["stage_a_evidence_component_train_examples"] = len(stage_a_evidence_component_train_rows)
        summary["stage_a_evidence_component_heldout_examples"] = len(stage_a_evidence_component_heldout_rows)
        summary["stage_a_evidence_component_by_component"] = dict(
            sorted(Counter(row["component"] for row in stage_a_evidence_component_rows).items())
        )
    if not args.skip_stage_a_enum_corrective:
        summary["stage_a_enum_corrective_pairs"] = len(stage_a_enum_corrective_rows)
        summary["stage_a_enum_corrective_train_pairs"] = len(stage_a_enum_corrective_train_rows)
        summary["stage_a_enum_corrective_heldout_pairs"] = len(stage_a_enum_corrective_heldout_rows)
        summary["stage_a_enum_corrective_by_chosen_pair"] = dict(
            sorted(Counter(row["chosen_pair"] for row in stage_a_enum_corrective_rows).items())
        )
    if not args.skip_stage_a_enum_action_contrast:
        summary["stage_a_enum_action_contrast_pairs"] = len(stage_a_enum_action_contrast_rows)
        summary["stage_a_enum_action_contrast_train_pairs"] = len(stage_a_enum_action_contrast_train_rows)
        summary["stage_a_enum_action_contrast_heldout_pairs"] = len(stage_a_enum_action_contrast_heldout_rows)
        summary["stage_a_enum_action_contrast_by_chosen_pair"] = dict(
            sorted(Counter(row["chosen_pair"] for row in stage_a_enum_action_contrast_rows).items())
        )
        summary["stage_a_enum_action_contrast_by_rejected_pair"] = dict(
            sorted(Counter(row["rejected_pair"] for row in stage_a_enum_action_contrast_rows).items())
        )
    if not args.skip_stage_a_routing_action_status_contrast:
        summary["stage_a_routing_action_status_contrast_pairs"] = len(
            stage_a_routing_action_status_contrast_rows
        )
        summary["stage_a_routing_action_status_contrast_train_pairs"] = len(
            stage_a_routing_action_status_contrast_train_rows
        )
        summary["stage_a_routing_action_status_contrast_heldout_pairs"] = len(
            stage_a_routing_action_status_contrast_heldout_rows
        )
        summary["stage_a_routing_action_status_contrast_by_chosen_pair"] = dict(
            sorted(Counter(row["chosen_pair"] for row in stage_a_routing_action_status_contrast_rows).items())
        )
        summary["stage_a_routing_action_status_contrast_by_rejected_pair"] = dict(
            sorted(Counter(row["rejected_pair"] for row in stage_a_routing_action_status_contrast_rows).items())
        )
    if not args.skip_stage_a_routing_defer_verify_contrast:
        summary["stage_a_routing_defer_verify_contrast_pairs"] = len(
            stage_a_routing_defer_verify_contrast_rows
        )
        summary["stage_a_routing_defer_verify_contrast_train_pairs"] = len(
            stage_a_routing_defer_verify_contrast_train_rows
        )
        summary["stage_a_routing_defer_verify_contrast_heldout_pairs"] = len(
            stage_a_routing_defer_verify_contrast_heldout_rows
        )
        summary["stage_a_routing_defer_verify_contrast_by_chosen_pair"] = dict(
            sorted(Counter(row["chosen_pair"] for row in stage_a_routing_defer_verify_contrast_rows).items())
        )
        summary["stage_a_routing_defer_verify_contrast_by_rejected_pair"] = dict(
            sorted(Counter(row["rejected_pair"] for row in stage_a_routing_defer_verify_contrast_rows).items())
        )
    if not args.skip_stage_a_saved_output_calibration_probe:
        summary["stage_a_saved_output_calibration_probe_pairs"] = len(
            stage_a_saved_output_calibration_probe_rows
        )
        summary["stage_a_saved_output_calibration_probe_train_pairs"] = len(
            stage_a_saved_output_calibration_probe_train_rows
        )
        summary["stage_a_saved_output_calibration_probe_heldout_pairs"] = len(
            stage_a_saved_output_calibration_probe_heldout_rows
        )
        summary["stage_a_saved_output_calibration_probe_by_chosen_pair"] = dict(
            sorted(Counter(row["chosen_pair"] for row in stage_a_saved_output_calibration_probe_rows).items())
        )
    if not args.skip_stage_a_saved_output_evidence_candidate_routing:
        summary["stage_a_saved_output_evidence_candidate_routing_rows"] = len(
            stage_a_saved_output_evidence_candidate_routing_rows
        )
        summary["stage_a_saved_output_evidence_candidate_routing_train_rows"] = len(
            stage_a_saved_output_evidence_candidate_routing_train_rows
        )
        summary["stage_a_saved_output_evidence_candidate_routing_heldout_rows"] = len(
            stage_a_saved_output_evidence_candidate_routing_heldout_rows
        )
        summary["stage_a_saved_output_evidence_candidate_routing_bridge_focus_rows"] = sum(
            1
            for row in stage_a_saved_output_evidence_candidate_routing_rows
            if row.get("bridge_focus_case") is True
        )
        summary["stage_a_saved_output_evidence_candidate_routing_by_target_pair"] = dict(
            sorted(
                Counter(
                    row["target_pair"]
                    for row in stage_a_saved_output_evidence_candidate_routing_rows
                ).items()
            )
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
