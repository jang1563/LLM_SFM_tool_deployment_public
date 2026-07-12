from llm_sfm_tool_deployment import Action, EvidenceStatus
from negbiodb_ct.stage_a_manifest import (
    build_stage_a_manifest_rows,
    failure_trajectories_for_stage_a_row,
    ideal_trajectory_from_stage_a_row,
    load_stage_a_manifest,
    score_stage_a_trajectory,
    stage_a_row_from_task_record,
    task_spec_from_stage_a_row,
    validate_stage_a_manifest,
)


def record(action_class: str, *, gold_nct: str | None = None) -> dict:
    return {
        "packet_id": f"ct::{action_class}::1::2",
        "action_class": action_class,
        "available_actions": ["ground", "reject", "defer", "verify", "flag"],
        "observation": {
            "claim": "Has ToyDrug been tested and failed for ToyCondition? Use the available tools, then state your conclusion.",
            "drug_id": 1,
            "condition_id": 2,
        },
        "scoring_key": {
            "gold_action": action_class,
            "gold_nct": gold_nct,
            "gold_failure_category": "efficacy" if gold_nct else None,
            "inject_impossible_value": action_class == "flag",
            "note": None,
        },
    }


def balanced_records() -> list[dict]:
    rows: list[dict] = []
    for action_index, action in enumerate(("ground", "reject", "defer", "verify", "flag")):
        for idx in range(2):
            unique = action_index * 10 + idx
            nct = f"NCT{unique:08d}" if action in {"ground", "flag"} else None
            item = record(action, gold_nct=nct)
            item["packet_id"] = f"ct::{action}::{unique + 1}::{unique + 100}"
            item["observation"]["drug_id"] = unique + 1
            item["observation"]["condition_id"] = unique + 100
            rows.append(item)
    return rows


def test_stage_a_row_hides_eval_labels_from_model_visible_task() -> None:
    row = stage_a_row_from_task_record(record("ground", gold_nct="NCT12345678"), case_index=0)

    visible = str(row["model_visible_task"])
    assert "NCT12345678" not in visible
    assert "supported" not in visible.lower()
    assert "ground_with_attribution" not in visible
    assert row["model_visible_task"]["input_id"] == "stage_a::000000"


def test_stage_a_row_projects_to_task_spec_and_oracle_passes() -> None:
    row = stage_a_row_from_task_record(record("flag", gold_nct="NCT12345678"), case_index=1)
    task = task_spec_from_stage_a_row(row)
    trajectory = ideal_trajectory_from_stage_a_row(row)
    result = score_stage_a_trajectory(row, trajectory)

    assert task.gold_evidence_status == EvidenceStatus.INVALID_VALUE
    assert task.expected_terminal_action == Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM
    assert task.gold_source_ids == ("NCT12345678",)
    assert result.passed


def test_stage_a_validator_flags_hidden_label_leakage() -> None:
    row = stage_a_row_from_task_record(record("ground", gold_nct="NCT12345678"), case_index=0)
    row["model_visible_task"]["claim"] += " Source NCT12345678."

    issues = validate_stage_a_manifest([row], min_rows=1)

    assert any("hidden_source_id_leak" in issue for issue in issues)


def test_stage_a_validator_checks_balance_and_split_overlap() -> None:
    row = stage_a_row_from_task_record(record("ground", gold_nct="NCT12345678"), case_index=0)
    duplicate = stage_a_row_from_task_record(record("ground", gold_nct="NCT12345678"), case_index=1)

    issues = validate_stage_a_manifest([row, duplicate], min_rows=1)

    assert any("status_underrepresented:contradicted" in issue for issue in issues)
    assert any("split_group_overlap" in issue for issue in issues)


def test_build_stage_a_manifest_rows_balances_action_classes() -> None:
    rows = build_stage_a_manifest_rows(balanced_records(), per_action=2)
    issues = validate_stage_a_manifest(rows, min_rows=10, min_status_count=2)

    assert len(rows) == 10
    assert not issues


def test_failure_matrix_contains_required_bad_paths_and_scores_failures() -> None:
    row = stage_a_row_from_task_record(record("flag", gold_nct="NCT12345678"), case_index=0)
    variants = failure_trajectories_for_stage_a_row(row)

    for mode in (
        "self_answering_without_tools",
        "wrong_tool",
        "missing_tool",
        "partial_query",
        "missing_attribution",
        "invalid_value_missed",
        "unsupported_trust",
    ):
        assert mode in variants
        assert not score_stage_a_trajectory(row, variants[mode]).passed

    partial = score_stage_a_trajectory(row, variants["partial_query"])
    assert "query_filter_missing_required_field" in partial.violations


def test_insufficient_case_gets_negative_evidence_failure_variant() -> None:
    row = stage_a_row_from_task_record(record("defer"), case_index=0)
    variants = failure_trajectories_for_stage_a_row(row)
    result = score_stage_a_trajectory(row, variants["insufficient_as_negative"])

    assert "insufficient_as_negative" in variants
    assert "evidence_status_mismatch" in result.violations


def test_tracked_stage_a_mini_manifest_validates() -> None:
    rows = load_stage_a_manifest("negbiodb_ct/stage_a_mini_manifest.jsonl")
    issues = validate_stage_a_manifest(rows, min_rows=20)

    assert len(rows) == 25
    assert issues == []
