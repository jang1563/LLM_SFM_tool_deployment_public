import pytest

from post_training.analyze_stage_a_saved_output_candidate_calibration import (
    build_calibration_report,
)


def candidate(action: str, status: str) -> dict[str, str]:
    return {"action": action, "evidence_status": status}


def row(case_id: str, target: tuple[str, str], scores: dict[tuple[str, str], float], score_label: str) -> dict:
    return {
        "case_id": case_id,
        "run_id": "unit_saved_output_candidate_calibration",
        "model": "unit-model",
        "score_label": score_label,
        "candidate_policy": "train_observed_plus_rejected",
        "candidate_target_format": "full",
        "target_pair": candidate(*target),
        "candidate_scores": [
            {"candidate": candidate(*pair), "score": score}
            for pair, score in scores.items()
        ],
    }


def test_saved_output_candidate_calibration_uses_train_pair_means(tmp_path):
    train_path = tmp_path / "train_candidates.jsonl"
    heldout_path = tmp_path / "candidates.jsonl"
    train_path.write_text("{}\n")
    heldout_path.write_text("{}\n")

    train_rows = [
        row("train-a", ("verify", "insufficient"), {
            ("verify", "insufficient"): 1.1,
            ("flag", "invalid_value"): 1.2,
        }, "trained_train"),
        row("train-b", ("flag", "invalid_value"), {
            ("verify", "insufficient"): 1.0,
            ("flag", "invalid_value"): 1.3,
        }, "trained_train"),
    ]
    heldout_rows = [
        row("heldout-a", ("verify", "insufficient"), {
            ("verify", "insufficient"): 1.08,
            ("flag", "invalid_value"): 1.21,
        }, "trained_heldout"),
        row("heldout-b", ("flag", "invalid_value"), {
            ("verify", "insufficient"): 0.9,
            ("flag", "invalid_value"): 1.24,
        }, "trained_heldout"),
    ]

    report = build_calibration_report(
        train_rows,
        heldout_rows,
        train_candidates_path=train_path,
        heldout_candidates_path=heldout_path,
    )

    assert report["calibration_mode"] == "pair_mean_center"
    assert report["train_summary"]["top_pair_counts"] == {"flag/invalid_value": 2}
    assert report["calibrated_train_summary"]["exact_top1"] == 2
    assert report["raw_heldout_summary"]["exact_top1"] == 1
    assert report["calibrated_heldout_summary"]["exact_top1"] == 2
    assert report["train_selected_zero_unsafe_threshold"] == 0.0
    assert report["train_selected_gate_report"]["trusted"] == 2
    assert report["train_selected_gate_report"]["trusted_incorrect"] == 0
    assert report["train_selected_gate_report"]["strict_final_correct"] == 2
    assert report["rows"][0]["raw_top_pair_label"] == "flag/invalid_value"
    assert report["rows"][0]["calibrated_top_pair_label"] == "verify/insufficient"
    assert report["rows"][0]["calibrated_top_second_gap"] == pytest.approx(0.07)


def test_saved_output_candidate_calibration_train_selected_gate_can_fail_closed(tmp_path):
    train_path = tmp_path / "train_candidates.jsonl"
    heldout_path = tmp_path / "candidates.jsonl"
    train_path.write_text("{}\n")
    heldout_path.write_text("{}\n")

    train_rows = [
        row("train-a", ("verify", "insufficient"), {
            ("verify", "insufficient"): 1.0,
            ("flag", "invalid_value"): 1.4,
            ("defer", "insufficient"): 1.0,
        }, "trained_train"),
        row("train-b", ("flag", "invalid_value"), {
            ("verify", "insufficient"): 1.0,
            ("flag", "invalid_value"): 1.1,
            ("defer", "insufficient"): 1.0,
        }, "trained_train"),
    ]
    heldout_rows = [
        row("heldout-a", ("verify", "insufficient"), {
            ("verify", "insufficient"): 1.01,
            ("flag", "invalid_value"): 1.27,
            ("defer", "insufficient"): 1.0,
        }, "trained_heldout"),
        row("heldout-b", ("defer", "insufficient"), {
            ("verify", "insufficient"): 1.0,
            ("flag", "invalid_value"): 1.2,
            ("defer", "insufficient"): 1.0,
        }, "trained_heldout"),
    ]

    report = build_calibration_report(
        train_rows,
        heldout_rows,
        train_candidates_path=train_path,
        heldout_candidates_path=heldout_path,
    )

    assert report["train_selected_zero_unsafe_threshold"] > 0
    gate = report["train_selected_gate_report"]
    assert gate["trusted"] == 0
    assert gate["trusted_incorrect"] == 0
    assert gate["fail_closed"] == 2
    assert gate["fail_closed_exact_correct"] == 1
    assert gate["strict_final_correct"] == 1


def test_saved_output_candidate_calibration_fails_without_train_pair_coverage(tmp_path):
    train_path = tmp_path / "train_candidates.jsonl"
    heldout_path = tmp_path / "candidates.jsonl"
    train_path.write_text("{}\n")
    heldout_path.write_text("{}\n")

    train_rows = [
        row("train-a", ("verify", "insufficient"), {
            ("verify", "insufficient"): 1.1,
            ("flag", "invalid_value"): 1.2,
        }, "trained_train")
    ]
    heldout_rows = [
        row("heldout-a", ("reject", "contradicted"), {
            ("reject", "contradicted"): 1.0,
            ("flag", "invalid_value"): 1.2,
        }, "trained_heldout")
    ]

    with pytest.raises(ValueError, match="missing calibration pair: reject/contradicted"):
        build_calibration_report(
            train_rows,
            heldout_rows,
            train_candidates_path=train_path,
            heldout_candidates_path=heldout_path,
        )
