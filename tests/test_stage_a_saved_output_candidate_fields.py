import json
from pathlib import Path

from post_training.analyze_stage_a_saved_output_candidate_fields import (
    build_saved_output_candidate_field_report,
    write_markdown,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def candidate_rows(path: Path) -> list[dict]:
    rows = [
        {
            "case_id": "stage_a::000007",
            "case_family": "contradicted_or_mixed_endpoint_claim",
            "run_id": "unit_saved_output_candidate_fields",
            "score_label": "trained_heldout",
            "candidate_policy": "train_observed_plus_rejected",
            "candidate_target_format": "full",
            "prompt_contract": "stage_a_v4_canonical_json",
            "model": "unit/model",
            "target_pair": {"action": "reject", "evidence_status": "contradicted"},
            "candidate_scores": [
                {"candidate": {"action": "flag", "evidence_status": "invalid_value"}, "score": -0.44},
                {"candidate": {"action": "verify", "evidence_status": "insufficient"}, "score": -0.48},
                {"candidate": {"action": "defer", "evidence_status": "insufficient"}, "score": -0.49},
                {"candidate": {"action": "reject", "evidence_status": "contradicted"}, "score": -0.50},
                {"candidate": {"action": "ground", "evidence_status": "supported"}, "score": -0.51},
            ],
        },
        {
            "case_id": "stage_a::000021",
            "case_family": "invalid_value_attribution_failure",
            "run_id": "unit_saved_output_candidate_fields",
            "score_label": "trained_heldout",
            "candidate_policy": "train_observed_plus_rejected",
            "candidate_target_format": "full",
            "prompt_contract": "stage_a_v4_canonical_json",
            "model": "unit/model",
            "target_pair": {"action": "flag", "evidence_status": "invalid_value"},
            "candidate_scores": [
                {"candidate": {"action": "flag", "evidence_status": "invalid_value"}, "score": -0.42},
                {"candidate": {"action": "ground", "evidence_status": "supported"}, "score": -0.44},
                {"candidate": {"action": "verify", "evidence_status": "insufficient"}, "score": -0.48},
            ],
        },
    ]
    write_jsonl(path, rows)
    return rows


def test_saved_output_candidate_field_report_summarizes_field_bias(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    rows = candidate_rows(candidates)

    report = build_saved_output_candidate_field_report(rows, candidates_path=candidates)

    assert report["dataset"] == "negbiodb_ct_stage_a_saved_output_candidate_field_diagnostic_v1"
    assert report["run_id"] == "unit_saved_output_candidate_fields"
    assert report["candidate_policy"] == "train_observed_plus_rejected"
    assert report["summary"]["exact_top1"] == 1
    assert report["summary"]["top_pair_counts"] == {"flag/invalid_value": 2}
    assert report["summary"]["field_diagnostic"]["action_top1"] == 1
    assert report["summary"]["field_diagnostic"]["evidence_status_top1"] == 1
    assert report["summary"]["field_diagnostic"]["field_rank_patterns"] == {
        "both_field_failure": 1,
        "pair_top1": 1,
    }
    failed = report["rows"][0]
    assert failed["target_rank"] == 4
    assert failed["field_ranks"]["action"]["target_rank"] == 4
    assert failed["field_ranks"]["evidence_status"]["target_rank"] == 3
    assert failed["field_rank_pattern"] == "both_field_failure"
    assert len(failed["top_candidates"]) == 3


def test_saved_output_candidate_field_markdown_is_public_safe(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    rows = candidate_rows(candidates)
    report = build_saved_output_candidate_field_report(rows, candidates_path=candidates)
    out_md = tmp_path / "field.md"

    write_markdown(report, out_md)

    text = out_md.read_text()
    assert "Stage A Saved-Output Candidate Field Diagnostic" in text
    assert "Action field top-1" in text
    assert "prompt_messages" not in text
    assert "raw_output" not in text
    assert "scheduler" not in text


def test_saved_output_candidate_field_rejects_non_candidate_rows(tmp_path: Path) -> None:
    candidates = tmp_path / "bad.jsonl"
    rows = [{"case_id": "stage_a::bad", "candidate_scores": []}]
    write_jsonl(candidates, rows)

    try:
        build_saved_output_candidate_field_report(rows, candidates_path=candidates)
    except ValueError as exc:
        assert "candidate_scores" in str(exc)
        assert "target_pair" in str(exc)
    else:
        raise AssertionError("non-candidate rows should fail closed")
