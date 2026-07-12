import json
from pathlib import Path

from post_training.analyze_stage_a_enum_candidate_gate import (
    build_candidate_gate_report,
    parse_thresholds,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def candidate_row(case_id: str, target: dict, scores: list[tuple[dict, float]]) -> dict:
    sorted_scores = sorted(
        [{"candidate": candidate, "score": score} for candidate, score in scores],
        key=lambda item: -item["score"],
    )
    top = sorted_scores[0]["candidate"]
    return {
        "id": f"candidate::{case_id}",
        "run_id": "unit_candidate_gate",
        "case_id": case_id,
        "case_family": f"family::{case_id}",
        "chosen_pair": f"{target['action']}/{target['evidence_status']}",
        "target_output": target,
        "prediction": top,
        "candidate_scores": sorted_scores,
        "enum_candidate_policy": "pair_observed_outputs",
        "score_label": "trained_candidate_heldout",
    }


def test_candidate_gate_report_tracks_unsafe_trust_and_zero_false_threshold(tmp_path: Path) -> None:
    correct = candidate_row(
        "correct_low_gap",
        {"action": "reject", "evidence_status": "contradicted"},
        [
            ({"action": "reject", "evidence_status": "contradicted"}, 1.00),
            ({"action": "verify", "evidence_status": "insufficient"}, 0.97),
            ({"action": "ground", "evidence_status": "supported"}, 0.5),
        ],
    )
    incorrect_high_gap = candidate_row(
        "incorrect_high_gap",
        {"action": "defer", "evidence_status": "insufficient"},
        [
            ({"action": "reject", "evidence_status": "contradicted"}, 1.00),
            ({"action": "defer", "evidence_status": "insufficient"}, 0.94),
            ({"action": "ground", "evidence_status": "supported"}, 0.5),
        ],
    )
    path = tmp_path / "candidates.jsonl"
    write_jsonl(path, [correct, incorrect_high_gap])

    report = build_candidate_gate_report(
        [correct, incorrect_high_gap],
        candidates_path=path,
        thresholds=[0.0, 0.04, 0.07],
    )

    assert report["summary"]["exact_top1"] == 1
    assert report["threshold_reports"][1]["trusted"] == 1
    assert report["threshold_reports"][1]["trusted_incorrect"] == 1
    assert report["threshold_reports"][1]["unsafe_trust_case_ids"] == ["incorrect_high_gap"]
    assert report["best_default_zero_false_report"]["threshold"] == 0.07
    assert report["best_default_zero_false_report"]["trusted"] == 0
    assert report["adaptive_zero_false_threshold"] == 0.060001
    assert report["adaptive_zero_false_report"]["trusted"] == 0


def test_candidate_gate_report_summarizes_field_rank_pattern(tmp_path: Path) -> None:
    row = candidate_row(
        "joint_pair_failure",
        {"action": "flag", "evidence_status": "invalid_value"},
        [
            ({"action": "ground", "evidence_status": "supported"}, 3.0),
            ({"action": "flag", "evidence_status": "supported"}, 2.0),
            ({"action": "ground", "evidence_status": "invalid_value"}, 1.9),
            ({"action": "flag", "evidence_status": "invalid_value"}, 1.0),
        ],
    )
    path = tmp_path / "candidates.jsonl"
    write_jsonl(path, [row])

    report = build_candidate_gate_report([row], candidates_path=path)

    compact = report["rows"][0]
    assert compact["field_ranks"]["action"]["target_rank"] == 2
    assert compact["field_ranks"]["evidence_status"]["target_rank"] == 2
    assert compact["field_rank_pattern"] == "both_field_failure"
    assert report["summary"]["field_rank_patterns"] == {"both_field_failure": 1}


def test_parse_thresholds_dedupes_and_rejects_negative() -> None:
    assert parse_thresholds("0.1, 0, 0.1") == [0.0, 0.1]

    try:
        parse_thresholds("-0.1")
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative threshold should fail")
