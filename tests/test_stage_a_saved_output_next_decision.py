import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_next_decision import (
    build_report,
    load_json,
    summarize_candidate_gates,
)


ROOT = Path(__file__).resolve().parents[1]


def readiness_path() -> Path:
    return ROOT / "post_training" / "stage_a_saved_prediction_readiness_2026-07-09.json"


def gate_paths() -> list[Path]:
    return [
        ROOT
        / "post_training"
        / "stage_a_saved_candidate_gate_train_observed_qwen05b_2026-07-09.json",
        ROOT / "post_training" / "stage_a_saved_candidate_gate_all_valid_qwen05b_2026-07-09.json",
    ]


def test_saved_output_next_decision_keeps_escalation_gated() -> None:
    report = build_report(readiness_path=readiness_path(), candidate_gate_paths=gate_paths())

    assert report["raw_model_outputs_used"] is False
    assert report["raw_run_folders_used"] is False
    assert report["bottleneck"]["active_bottleneck"] == (
        "runtime_evidence_arbitration_beats_saved_output_candidates"
    )
    assert report["bottleneck"]["best_real_saved_output_passed"] == 0
    assert report["bottleneck"]["best_candidate_gate_strict_final_correct"] == 2
    assert report["bottleneck"]["best_candidate_gate_trusted_incorrect"] == 0
    assert report["candidate_calibration_summary"]["raw_heldout_exact_top1"] == 1
    assert report["candidate_calibration_summary"]["calibrated_heldout_exact_top1"] == 2
    assert report["candidate_arbitration_summary"]["hidden_labels_used_by_arbitration"] is False
    assert report["candidate_arbitration_summary"]["evidence_gate_exact"] == 4
    assert report["candidate_arbitration_summary"]["hybrid_evidence_then_train_gate_exact"] == 4
    assert report["decision"]["selected_next_step"] == (
        "meet_or_beat_runtime_evidence_arbitration_baseline"
    )
    assert report["decision"]["minimum_success_criteria_for_next_cayuga_checkpoint"][
        "candidate_or_model_policy_exact_min"
    ] == 4
    assert "tool_query" in report["decision"]["keep_gated"]
    assert "DPO/RLVR" in report["decision"]["keep_gated"]


def test_saved_output_next_decision_without_arbitration_keeps_legacy_probe_decision() -> None:
    report = build_report(
        readiness_path=readiness_path(),
        candidate_gate_paths=gate_paths(),
        candidate_calibration_path=None,
        candidate_arbitration_path=None,
    )

    assert report["candidate_calibration_summary"] is None
    assert report["candidate_arbitration_summary"] is None
    assert report["bottleneck"]["active_bottleneck"] == "narrow_fail_closed_coverage_under_citationless"
    assert report["decision"]["selected_next_step"] == "targeted_action_status_calibration_probe"


def test_saved_output_next_decision_summarizes_failed_action_status_families() -> None:
    summaries = summarize_candidate_gates(gate_paths())
    all_valid = next(item for item in summaries if item["candidate_policy"] == "all_valid_pairs")

    assert all_valid["top_pair_counts"] == {"ground/supported": 5}
    assert all_valid["failure_top_pair_counts"] == {"ground/supported": 4}
    assert all_valid["failure_target_pair_counts"] == {
        "defer/insufficient": 1,
        "flag/invalid_value": 1,
        "reject/contradicted": 1,
        "verify/insufficient": 1,
    }
    assert len(all_valid["failure_rows"]) == 4


def test_saved_output_next_decision_cli_writes_public_safe_report(tmp_path: Path) -> None:
    out_json = tmp_path / "next_decision.json"
    out_md = tmp_path / "NEXT_DECISION.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_next_decision.py",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    stdout = json.loads(result.stdout)
    data = load_json(out_json)
    text = out_md.read_text()
    assert stdout["decision"]["selected_next_step"] == (
        "meet_or_beat_runtime_evidence_arbitration_baseline"
    )
    assert data["raw_run_folders_used"] is False
    assert data["candidate_arbitration_summary"]["hidden_labels_used_by_arbitration"] is False
    assert "# Stage A Saved-Output Next Decision" in text
    assert "Arbitration exact" in text
    assert "candidate-score JSONL" in text
    assert "raw_output" not in text
    assert "scheduler logs" in text
