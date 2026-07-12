import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_checkpoint_diagnosis import (
    build_report,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def test_saved_output_checkpoint_diagnosis_captures_current_bottleneck() -> None:
    report = build_report()

    teacher = report["diagnosis"]["teacher_forced_margin"]
    rank = report["diagnosis"]["finite_candidate_rank"]
    calibration = report["diagnosis"]["candidate_calibration"]
    arbitration = report["diagnosis"]["runtime_arbitration"]
    gate = report["diagnosis"]["meet_or_beat_gate"]

    assert teacher["trained_full_margin_wins"] == 4
    assert teacher["pairs"] == 4
    assert rank["trained_exact_top1"] == 1
    assert rank["trained_top_pair_counts"] == {"flag/invalid_value": 4}
    assert calibration["calibrated_heldout_exact_top1"] == 2
    assert gate["passes_gate"] is False
    assert gate["gate_violations"] == ["no_model_policy_meets_runtime_baseline"]
    assert report["decision"]["selected_next_step"] == "meet_or_beat_runtime_evidence_arbitration_baseline"
    assert set(report["decision"]["keep_gated"]) == {
        "tool_query",
        "DPO/RLVR",
        "Hugging Face publication",
        "release tagging",
        "broad retraining",
    }

    policies = {row["policy"]: row for row in arbitration["policies"]}
    assert policies["raw_candidate_top1"]["exact"] == 1
    assert policies["raw_candidate_top1"]["trusted_candidate_incorrect"] == 3
    assert policies["calibrated_candidate_top1"]["exact"] == 2
    assert policies["train_selected_score_gap_gate"]["trusted_candidate_incorrect"] == 0
    assert policies["evidence_gate_override"]["exact"] == 4
    assert policies["hybrid_evidence_then_train_gate"]["exact"] == 4


def test_saved_output_checkpoint_diagnosis_is_public_safe() -> None:
    report = build_report()
    text = render_markdown(report)
    serialized = json.dumps(report, sort_keys=True)

    assert report["public_safety_contract"] == {
        "raw_prediction_jsonl_read": False,
        "raw_candidate_score_jsonl_read": False,
        "scheduler_logs_read": False,
        "model_state_read": False,
        "ignored_run_folder_read": False,
        "raw_artifacts_committed": False,
    }
    assert "post_training/runs" not in serialized
    assert "post_training/runs" not in text
    assert "scheduler log" in text
    assert "raw_model_text" not in serialized


def test_saved_output_checkpoint_diagnosis_cli_writes_compact_files(tmp_path: Path) -> None:
    out_json = tmp_path / "diagnosis.json"
    out_md = tmp_path / "DIAGNOSIS.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_checkpoint_diagnosis.py",
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
    payload = json.loads(out_json.read_text())
    text = out_md.read_text()
    assert stdout["dataset"] == "negbiodb_ct_stage_a_saved_output_checkpoint_diagnosis_v1"
    assert payload["diagnosis"]["finite_candidate_rank"]["trained_exact_top1"] == 1
    assert "Stage A Saved-Output Checkpoint Diagnosis" in text
    assert "candidate-score JSONL" in text
    assert "post_training/runs" not in text
