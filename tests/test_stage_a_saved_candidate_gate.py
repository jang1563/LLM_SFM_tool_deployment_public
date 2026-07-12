import json
from pathlib import Path

from post_training.analyze_stage_a_saved_candidate_gate import (
    build_saved_candidate_gate_report,
    parse_thresholds,
    write_markdown,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def candidate_rows(path: Path) -> list[dict]:
    rows = [
        {
            "case_id": "stage_a::000001",
            "case_family": "supported_negative_evidence",
            "run_id": "unit_saved_gate",
            "candidate_policy": "train_observed_pairs",
            "prompt_contract": "stage_a_v4_canonical_json",
            "model": "unit/model",
            "target_pair": {"action": "ground", "evidence_status": "supported"},
            "candidate_scores": [
                {
                    "candidate": {"action": "ground", "evidence_status": "supported"},
                    "score": -0.70,
                },
                {
                    "candidate": {"action": "verify", "evidence_status": "insufficient"},
                    "score": -0.74,
                },
            ],
        },
        {
            "case_id": "stage_a::000007",
            "case_family": "contradicted_or_mixed_endpoint_claim",
            "run_id": "unit_saved_gate",
            "candidate_policy": "train_observed_pairs",
            "prompt_contract": "stage_a_v4_canonical_json",
            "model": "unit/model",
            "target_pair": {"action": "reject", "evidence_status": "contradicted"},
            "candidate_scores": [
                {
                    "candidate": {"action": "ground", "evidence_status": "supported"},
                    "score": -0.72,
                },
                {
                    "candidate": {"action": "reject", "evidence_status": "contradicted"},
                    "score": -0.745,
                },
            ],
        },
        {
            "case_id": "stage_a::000012",
            "case_family": "insufficient_evidence",
            "run_id": "unit_saved_gate",
            "candidate_policy": "train_observed_pairs",
            "prompt_contract": "stage_a_v4_canonical_json",
            "model": "unit/model",
            "target_pair": {"action": "defer", "evidence_status": "insufficient"},
            "candidate_scores": [
                {
                    "candidate": {"action": "ground", "evidence_status": "supported"},
                    "score": -0.71,
                },
                {
                    "candidate": {"action": "defer", "evidence_status": "insufficient"},
                    "score": -0.735,
                },
            ],
        },
    ]
    write_jsonl(path, rows)
    return rows


def test_saved_candidate_gate_fails_closed_above_wrong_gap(tmp_path: Path) -> None:
    candidates = tmp_path / "saved_candidates.jsonl"
    rows = candidate_rows(candidates)

    report = build_saved_candidate_gate_report(
        rows,
        candidates_path=candidates,
        thresholds=[0.0, 0.026, 0.04],
    )

    assert report["summary"]["exact_top1"] == 1
    assert report["summary"]["top_pair_counts"] == {"ground/supported": 3}
    by_threshold = {row["threshold"]: row for row in report["threshold_reports"]}
    assert by_threshold[0.0]["trusted_incorrect"] == 2
    assert by_threshold[0.0]["strict_final_correct"] == 1
    assert by_threshold[0.026]["trusted_incorrect"] == 0
    assert by_threshold[0.026]["trusted_correct"] == 1
    assert by_threshold[0.026]["fail_closed_exact_correct"] == 1
    assert by_threshold[0.026]["strict_final_correct"] == 2
    assert report["best_default_zero_unsafe_report"]["threshold"] == 0.026
    assert report["adaptive_zero_unsafe_threshold"] == 0.025001


def test_saved_candidate_gate_markdown_is_public_safe(tmp_path: Path) -> None:
    candidates = tmp_path / "saved_candidates.jsonl"
    rows = candidate_rows(candidates)
    report = build_saved_candidate_gate_report(rows, candidates_path=candidates)
    out_md = tmp_path / "gate.md"

    write_markdown(report, out_md)

    text = out_md.read_text()
    assert "Stage A Saved-Candidate Gate Diagnostic" in text
    assert "candidate-score JSONL" in text
    assert "prompt_messages" not in text
    assert "raw_output" not in text
    assert "scheduler" in text


def test_parse_thresholds_rejects_negative_values() -> None:
    assert parse_thresholds("0.035,0.0,0.035") == [0.0, 0.035]
    try:
        parse_thresholds("-0.01")
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative threshold should fail")


def test_saved_candidate_gate_rejects_non_candidate_rows(tmp_path: Path) -> None:
    candidates = tmp_path / "not_candidate_scores.jsonl"
    rows = [{"case_id": "stage_a::bad", "prediction": {"action": "ground"}}]
    write_jsonl(candidates, rows)

    try:
        build_saved_candidate_gate_report(rows, candidates_path=candidates)
    except ValueError as exc:
        assert "candidate_scores" in str(exc)
        assert "target_pair" in str(exc)
    else:
        raise AssertionError("non-candidate rows should fail closed")
