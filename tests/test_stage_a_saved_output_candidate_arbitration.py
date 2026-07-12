import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_candidate_arbitration import (
    build_arbitration_report,
)
from post_training.run_stage_a_strict_component_sft_smoke import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_REPORT = (
    ROOT
    / "post_training"
    / "stage_a_saved_output_candidate_calibration_qwen05b_cayuga_summary_2026-07-10.json"
)
EVIDENCE_TARGETS = ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_v1.jsonl"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_saved_output_candidate_arbitration_prefers_runtime_evidence_gate() -> None:
    report = build_arbitration_report(
        calibration_report=load_json(CALIBRATION_REPORT),
        evidence_targets=load_jsonl(EVIDENCE_TARGETS),
        calibration_report_path=CALIBRATION_REPORT,
        evidence_targets_path=EVIDENCE_TARGETS,
    )

    by_policy = report["summary"]["by_policy"]
    assert report["hidden_labels_used_by_arbitration"] is False
    assert by_policy["raw_candidate_top1"]["exact"] == 1
    assert by_policy["calibrated_candidate_top1"]["exact"] == 2
    assert by_policy["train_selected_score_gap_gate"]["exact"] == 1
    assert by_policy["train_selected_score_gap_gate"]["trusted_candidate"] == 0
    assert by_policy["evidence_gate_override"]["exact"] == 4
    assert by_policy["hybrid_evidence_then_train_gate"]["exact"] == 4
    assert report["summary"]["best_policy_names"] == [
        "evidence_gate_override",
        "hybrid_evidence_then_train_gate",
    ]


def test_saved_output_candidate_arbitration_ignores_hidden_target_fields() -> None:
    targets = load_jsonl(EVIDENCE_TARGETS)
    mutated = deepcopy(targets)
    for row in mutated:
        row["case_family"] = "wrong_hidden_family"
        row["gold_evidence_status"] = "wrong_hidden_status"
        row["expected_terminal_action"] = "wrong_hidden_action"
        row["target_output"] = {
            "action": "wrong_hidden_action",
            "evidence_status": "wrong_hidden_status",
        }

    report = build_arbitration_report(
        calibration_report=load_json(CALIBRATION_REPORT),
        evidence_targets=mutated,
        calibration_report_path=CALIBRATION_REPORT,
        evidence_targets_path=EVIDENCE_TARGETS,
    )

    assert report["summary"]["by_policy"]["evidence_gate_override"]["exact"] == 4
    assert report["summary"]["by_policy"]["hybrid_evidence_then_train_gate"]["exact"] == 4


def test_saved_output_candidate_arbitration_cli_writes_public_safe_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "saved_output_candidate_arbitration.json"
    out_md = tmp_path / "saved_output_candidate_arbitration.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_candidate_arbitration.py",
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
    report = json.loads(out_json.read_text())
    assert report["summary"]["by_policy"]["hybrid_evidence_then_train_gate"]["exact"] == 4
    text = out_md.read_text()
    assert "Stage A Saved-Output Candidate Arbitration" in text
    assert "prompt_messages" not in text
    assert "raw_model_text" not in text
    assert "source_task_id" not in text
