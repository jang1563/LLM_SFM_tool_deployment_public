import json
from pathlib import Path

from post_training.analyze_stage_a_enum_candidate_scores import (
    build_enum_candidate_rank_report,
)
from post_training.run_stage_a_strict_component_sft_smoke import (
    enum_action_candidate_outputs,
    enum_action_observed_pair_outputs,
    filter_component,
    load_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def heldout_enum_rows() -> list[dict]:
    rows = load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_heldout_v1.jsonl")
    return filter_component(rows, "enum_action")


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def candidate_scores_with_target_first(target: dict) -> list[dict]:
    scores = []
    for candidate in enum_action_candidate_outputs():
        score = 2.0 if candidate == target else 0.0
        scores.append({"candidate": candidate, "score": score})
    return sorted(scores, key=lambda item: (-item["score"], json.dumps(item["candidate"], sort_keys=True)))


def test_enum_candidate_rank_report_summarizes_full_candidate_scores(tmp_path: Path) -> None:
    expected = heldout_enum_rows()
    predictions = [
        {
            "id": f"oracle::{row['id']}",
            "source_component_target_id": row["id"],
            "run_id": "unit_full_candidate_scores",
            "prediction": row["target_output"],
            "candidate_scores": candidate_scores_with_target_first(row["target_output"]),
        }
        for row in expected
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions_path, predictions)

    report = build_enum_candidate_rank_report(
        expected_rows=expected,
        prediction_rows=predictions,
        predictions_path=predictions_path,
    )

    assert report["summary"]["exact_top1"] == 5
    assert report["summary"]["gold_in_retained_candidates"] == 5
    assert report["summary"]["all_candidates_retained_cases"] == 5
    assert report["summary"]["mean_gold_rank_observed"] == 1
    assert report["summary"]["mean_top_gold_margin_observed"] == 0
    assert report["summary"]["field_diagnostic"]["action_top1"] == 5
    assert report["summary"]["field_diagnostic"]["evidence_status_top1"] == 5
    assert report["summary"]["field_diagnostic"]["field_rank_patterns"] == {"pair_top1": 5}
    assert all(row["field_ranks"]["action"]["target_rank"] == 1 for row in report["rows"])
    assert all(row["field_ranks"]["evidence_status"]["target_rank"] == 1 for row in report["rows"])


def test_enum_candidate_rank_report_marks_truncated_missing_gold(tmp_path: Path) -> None:
    expected = heldout_enum_rows()
    wrong_candidate = {"action": "ground", "evidence_status": "supported"}
    predictions = [
        {
            "id": f"truncated::{row['id']}",
            "source_component_target_id": row["id"],
            "run_id": "unit_truncated_candidate_scores",
            "prediction": wrong_candidate,
            "candidate_scores": [{"candidate": wrong_candidate, "score": 1.0}],
        }
        for row in expected
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions_path, predictions)

    report = build_enum_candidate_rank_report(
        expected_rows=expected,
        prediction_rows=predictions,
        predictions_path=predictions_path,
    )

    assert report["summary"]["all_candidates_retained_cases"] == 0
    assert report["summary"]["retained_candidate_count_histogram"] == {1: 5}
    assert report["summary"]["gold_in_retained_candidates"] == 1
    assert sum(1 for row in report["rows"] if row["gold_rank"] is None) == 4
    assert report["summary"]["field_diagnostic"]["action_top1"] == 1
    assert report["summary"]["field_diagnostic"]["evidence_status_top1"] == 1
    assert report["summary"]["field_diagnostic"]["field_rank_patterns"] == {
        "field_not_retained": 4,
        "pair_top1": 1,
    }


def test_enum_candidate_rank_report_supports_train_observed_pair_policy(tmp_path: Path) -> None:
    expected = heldout_enum_rows()
    observed_candidates = enum_action_observed_pair_outputs(
        filter_component(
            load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_train_v1.jsonl"),
            "enum_action",
        )
    )
    cartesian_scores = [
        {"candidate": candidate, "score": 0.0}
        for candidate in enum_action_candidate_outputs()
    ]
    predictions = [
        {
            "id": f"policy::{row['id']}",
            "source_component_target_id": row["id"],
            "run_id": "unit_observed_pair_policy",
            "prediction": {"action": "ground", "evidence_status": "supported"},
            "candidate_scores": cartesian_scores,
        }
        for row in expected
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions_path, predictions)

    report = build_enum_candidate_rank_report(
        expected_rows=expected,
        prediction_rows=predictions,
        predictions_path=predictions_path,
        candidate_policy="train_observed_pairs",
        candidate_outputs=observed_candidates,
    )

    assert report["candidate_policy"] == "train_observed_pairs"
    assert report["candidate_space_size"] == 5
    assert report["summary"]["all_candidates_retained_cases"] == 5
    assert report["summary"]["retained_candidate_count_histogram"] == {5: 5}
    assert all(
        row["raw_retained_candidate_count"] == len(enum_action_candidate_outputs())
        for row in report["rows"]
    )
    assert report["summary"]["field_diagnostic"]["by_case_family"]["invalid_value_attribution_failure"][
        "cases"
    ] == 1


def test_enum_candidate_rank_report_separates_field_and_pair_rank(tmp_path: Path) -> None:
    expected = heldout_enum_rows()
    target_by_family = {row["case_family"]: row for row in expected}
    invalid_value = target_by_family["invalid_value_attribution_failure"]
    predictions = [
        {
            "id": f"field::{invalid_value['id']}",
            "source_component_target_id": invalid_value["id"],
            "run_id": "unit_field_pair_rank",
            "prediction": {"action": "ground", "evidence_status": "supported"},
            "candidate_scores": [
                {"candidate": {"action": "ground", "evidence_status": "supported"}, "score": 3.0},
                {"candidate": {"action": "flag", "evidence_status": "supported"}, "score": 2.0},
                {"candidate": {"action": "ground", "evidence_status": "invalid_value"}, "score": 1.5},
                {"candidate": {"action": "flag", "evidence_status": "invalid_value"}, "score": 1.0},
            ],
        }
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions_path, predictions)

    report = build_enum_candidate_rank_report(
        expected_rows=[invalid_value],
        prediction_rows=predictions,
        predictions_path=predictions_path,
    )

    row = report["rows"][0]
    assert row["gold_rank"] == 4
    assert row["field_ranks"]["action"]["target_rank"] == 2
    assert row["field_ranks"]["evidence_status"]["target_rank"] == 2
    assert row["field_rank_pattern"] == "joint_pair_representation_failure"
    assert report["summary"]["field_diagnostic"]["field_rank_patterns"] == {
        "joint_pair_representation_failure": 1
    }
