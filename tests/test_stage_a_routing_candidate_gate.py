import json
from pathlib import Path

from post_training.analyze_stage_a_routing_candidate_gate import (
    build_routing_candidate_gate_report,
    parse_thresholds,
    write_markdown,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def boundary_rows(path: Path) -> list[dict]:
    rows = [
        {
            "id": "wrong_defer_case",
            "run_id": "unit_gate",
            "candidate_policy": "insufficient_defer_vs_verify_boundary",
            "score_label": "trained_candidate_heldout",
            "failure_mode": "routing_defer_verify_boundary_confusion",
            "case_id": "stage_a::000012",
            "case_family": "insufficient_evidence",
            "chosen_pair": "defer/insufficient",
            "target_output": {
                "action": "defer",
                "evidence_status": "insufficient",
                "cited_source_ids": [],
            },
            "candidate_scores": [
                {
                    "candidate": {
                        "action": "verify",
                        "evidence_status": "insufficient",
                        "cited_source_ids": [],
                    },
                    "score": -0.844626,
                },
                {
                    "candidate": {
                        "action": "defer",
                        "evidence_status": "insufficient",
                        "cited_source_ids": [],
                    },
                    "score": -0.868183,
                },
            ],
        },
        {
            "id": "correct_verify_case",
            "run_id": "unit_gate",
            "candidate_policy": "insufficient_defer_vs_verify_boundary",
            "score_label": "trained_candidate_heldout",
            "failure_mode": "routing_defer_verify_boundary_confusion",
            "case_id": "stage_a::000019",
            "case_family": "related_evidence_requires_verification",
            "chosen_pair": "verify/insufficient",
            "target_output": {
                "action": "verify",
                "evidence_status": "insufficient",
                "cited_source_ids": [],
            },
            "candidate_scores": [
                {
                    "candidate": {
                        "action": "verify",
                        "evidence_status": "insufficient",
                        "cited_source_ids": [],
                    },
                    "score": -0.871592,
                },
                {
                    "candidate": {
                        "action": "defer",
                        "evidence_status": "insufficient",
                        "cited_source_ids": [],
                    },
                    "score": -0.908076,
                },
            ],
        },
    ]
    write_jsonl(path, rows)
    return rows


def test_routing_candidate_gate_fails_closed_on_low_gap_wrong_case(tmp_path: Path) -> None:
    candidates = tmp_path / "routing_candidates.jsonl"
    rows = boundary_rows(candidates)

    report = build_routing_candidate_gate_report(
        rows,
        candidates_path=candidates,
        thresholds=[0.0, 0.025, 0.05],
    )

    assert report["component"] == "routing_after_loop"
    assert report["summary"]["exact_top1"] == 1
    assert report["summary"]["top_pair_counts"] == {"verify/insufficient": 2}
    by_threshold = {row["threshold"]: row for row in report["threshold_reports"]}
    assert by_threshold[0.0]["trusted_incorrect"] == 1
    assert by_threshold[0.0]["strict_final_correct"] == 1
    assert by_threshold[0.025]["trusted_incorrect"] == 0
    assert by_threshold[0.025]["trusted_correct"] == 1
    assert by_threshold[0.025]["fail_closed_exact_correct"] == 1
    assert by_threshold[0.025]["strict_final_correct"] == 2
    assert report["best_default_zero_unsafe_report"]["threshold"] == 0.025
    assert report["adaptive_zero_unsafe_threshold"] == 0.023558


def test_routing_candidate_gate_markdown_is_public_safe(tmp_path: Path) -> None:
    candidates = tmp_path / "routing_candidates.jsonl"
    rows = boundary_rows(candidates)
    report = build_routing_candidate_gate_report(rows, candidates_path=candidates)
    out_md = tmp_path / "gate.md"

    write_markdown(report, out_md)

    text = out_md.read_text()
    assert "Stage A Routing Candidate Gate Diagnostic" in text
    assert "prompt_messages" not in text
    assert "raw_output" not in text
    assert "trainable_state" not in text


def test_parse_thresholds_rejects_negative_values() -> None:
    assert parse_thresholds("0.05,0.0,0.05") == [0.0, 0.05]
    try:
        parse_thresholds("-0.1")
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative threshold should fail")
