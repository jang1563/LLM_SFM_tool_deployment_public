import json
from pathlib import Path

from post_training.analyze_stage_a_routing_candidate_scores import (
    build_routing_candidate_rank_report,
)
from post_training.run_stage_a_strict_component_sft_smoke import (
    filter_component,
    load_jsonl,
    routing_candidates_for_row,
    routing_observed_pair_outputs,
)


ROOT = Path(__file__).resolve().parents[1]


def heldout_routing_rows() -> list[dict]:
    rows = load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl")
    return filter_component(rows, "routing_after_loop")


def train_routing_rows() -> list[dict]:
    rows = load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_train_v1.jsonl")
    return filter_component(rows, "routing_after_loop")


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def candidate_scores_with_target_first(row: dict, action_status_candidates: list[dict]) -> list[dict]:
    scores = []
    for candidate in routing_candidates_for_row(row, action_status_candidates):
        score = 2.0 if candidate == row["target_output"] else 0.0
        scores.append({"candidate": candidate, "score": score})
    return sorted(scores, key=lambda item: (-item["score"], json.dumps(item["candidate"], sort_keys=True)))


def test_routing_candidate_rank_report_summarizes_full_target_scores(tmp_path: Path) -> None:
    expected = heldout_routing_rows()
    train_rows = train_routing_rows()
    action_status_candidates = routing_observed_pair_outputs(train_rows)
    predictions = [
        {
            "id": f"oracle::{row['id']}",
            "source_component_target_id": row["id"],
            "run_id": "unit_routing_candidate_scores",
            "prediction": row["target_output"],
            "candidate_scores": candidate_scores_with_target_first(row, action_status_candidates),
        }
        for row in expected
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions_path, predictions)

    report = build_routing_candidate_rank_report(
        expected_rows=expected,
        prediction_rows=predictions,
        predictions_path=predictions_path,
        train_rows=train_rows,
    )

    assert report["component"] == "routing_after_loop"
    assert report["candidate_space_size"] == 5
    assert report["summary"]["exact_top1"] == 5
    assert report["summary"]["action_status_top1"] == 5
    assert report["summary"]["gold_in_retained_candidates"] == 5
    assert report["summary"]["all_candidates_retained_cases"] == 5
    assert report["summary"]["mean_gold_rank_observed"] == 1
    assert report["summary"]["field_diagnostic"]["field_rank_patterns"] == {"exact_top1": 5}


def test_routing_candidate_rank_report_separates_citation_failure(tmp_path: Path) -> None:
    expected = heldout_routing_rows()
    train_rows = train_routing_rows()
    supported = next(row for row in expected if row["case_family"] == "supported_negative_evidence")
    wrong_citation_same_pair = {
        "action": "ground",
        "evidence_status": "supported",
        "cited_source_ids": [],
    }
    predictions = [
        {
            "id": f"citation::{supported['id']}",
            "source_component_target_id": supported["id"],
            "run_id": "unit_routing_citation_failure",
            "prediction": wrong_citation_same_pair,
            "candidate_scores": [
                {"candidate": wrong_citation_same_pair, "score": 2.0},
                {"candidate": supported["target_output"], "score": 1.5},
                {
                    "candidate": {
                        "action": "reject",
                        "evidence_status": "contradicted",
                        "cited_source_ids": [],
                    },
                    "score": 1.0,
                },
            ],
        }
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions_path, predictions)

    report = build_routing_candidate_rank_report(
        expected_rows=[supported],
        prediction_rows=predictions,
        predictions_path=predictions_path,
        train_rows=train_rows,
    )

    row = report["rows"][0]
    assert row["exact_top1"] is False
    assert row["action_status_top1"] is True
    assert row["gold_rank"] == 2
    assert row["action_status_rank"] == 1
    assert row["field_rank_pattern"] == "citation_failure"
    assert report["summary"]["citation_required_cases"] == 1
    assert report["summary"]["citation_required_exact_top1"] == 0
