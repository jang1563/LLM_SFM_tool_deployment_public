import copy
import json

from llm_sfm_tool_deployment import Action

from c5_antibody_ood.manifest import (
    build_c5_policy_test_rows,
    c5_trust_schema_issues,
    ideal_trajectory_from_c5_row,
    load_c5_manifest,
    score_c5_trajectory,
    task_spec_from_c5_row,
    validate_c5_manifest,
    validate_c5_split_overlap,
)
from c5_antibody_ood.policies import c5_fail_closed_action


MANIFEST = "c5_antibody_ood/c5_policy_test_manifest_v1.jsonl"


def test_tracked_c5_manifest_is_balanced_and_valid():
    rows = load_c5_manifest(MANIFEST)

    assert len(rows) == 12
    assert validate_c5_manifest(rows) == []
    assert build_c5_policy_test_rows() == rows
    action_counts = {
        action: sum(
            row["hidden_eval_metadata"]["expected_terminal_action"] == action
            for row in rows
        )
        for action in (
            Action.TRUST_SPECIALIST_OUTPUT.value,
            Action.USE_CHEAP_BASELINE.value,
            Action.VERIFY_WITH_ASSAY_OR_DATABASE.value,
            Action.DEFER_OR_REQUEST_MORE_EVIDENCE.value,
        )
    }
    assert set(action_counts.values()) == {3}


def test_c5_hidden_interface_labels_do_not_leak_to_visible_tasks():
    for row in load_c5_manifest(MANIFEST):
        visible = json.dumps(row["model_visible_task"], sort_keys=True)
        hidden = row["hidden_eval_metadata"]

        assert "interface_label_status" not in visible
        assert "interface_label_source" not in visible
        assert hidden["interface_label_status"] not in visible
        assert hidden["interface_label_source"] not in visible


def test_c5_rows_project_to_canonical_task_and_oracle_pass():
    for row in load_c5_manifest(MANIFEST):
        task = task_spec_from_c5_row(row)
        trajectory = ideal_trajectory_from_c5_row(row)
        result = score_c5_trajectory(row, trajectory)

        assert task.input_id == row["case_id"]
        assert trajectory.evidence_packet.hidden_truth_pointer is None
        assert trajectory.evidence_packet.interface_label_source is None
        assert result.passed, (row["case_id"], result.violations)


def test_c5_trust_requires_metric_scope_dataset_threshold_and_regime_match():
    row = copy.deepcopy(load_c5_manifest(MANIFEST)[0])
    mutations = {
        "metric_scope": ("specialist_result", "metric_scope"),
        "calibration_dataset": ("calibration", "dataset_id"),
        "rcps_threshold": ("calibration", "threshold_id"),
    }
    for expected_reason, (section, key) in mutations.items():
        mutated = copy.deepcopy(row)
        mutated["model_visible_task"][section][key] = None
        issues = c5_trust_schema_issues(mutated["model_visible_task"])

        assert any(expected_reason in issue for issue in issues)
        assert c5_fail_closed_action(mutated) != Action.TRUST_SPECIALIST_OUTPUT

    row["model_visible_task"]["calibration"]["regime_match"] = False
    assert "calibration_regime_mismatch" in c5_trust_schema_issues(
        row["model_visible_task"]
    )
    assert c5_fail_closed_action(row) != Action.TRUST_SPECIALIST_OUTPUT


def test_c5_validator_flags_hidden_label_leakage():
    row = copy.deepcopy(build_c5_policy_test_rows()[0])
    row["model_visible_task"]["claim"] += " interface_success"

    issues = validate_c5_manifest([row], min_rows=1, min_action_count=0)

    assert any("hidden_value_leak:interface_label_status" in issue for issue in issues)


def test_c5_split_overlap_check_fails_closed():
    rows = build_c5_policy_test_rows()

    issues = validate_c5_split_overlap(rows[:2], rows[1:3])

    assert issues == [
        "train_eval_split_group_overlap:synthetic_complex::000002"
    ]


def test_c5_validator_reports_malformed_trust_row_without_raising():
    row = copy.deepcopy(build_c5_policy_test_rows()[0])
    row["model_visible_task"].pop("calibration")

    issues = validate_c5_manifest([row], min_rows=1, min_action_count=0)

    assert any("model_visible_missing:calibration" in issue for issue in issues)
    assert any("expected_trust_projection_failed" in issue for issue in issues)


def test_c5_validator_rejects_invalid_scientific_metadata_types():
    row = copy.deepcopy(build_c5_policy_test_rows()[0])
    row["model_visible_task"]["specialist_result"]["metric_value"] = 1.2
    row["model_visible_task"]["calibration"]["status"] = "claimed_calibrated"
    row["model_visible_task"]["calibration"]["regime_match"] = "yes"
    row["model_visible_task"]["baseline_result"]["available"] = False
    row["model_visible_task"]["baseline_result"]["dominates_specialist"] = True

    issues = validate_c5_manifest([row], min_rows=1, min_action_count=0)

    for expected in (
        "specialist_metric_value_invalid",
        "calibration_status_invalid",
        "calibration_regime_match_not_bool",
        "baseline_dominates_but_unavailable",
    ):
        assert any(expected in issue for issue in issues)
