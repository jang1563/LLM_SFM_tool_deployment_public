import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_nonflag_checkpoint_result import (
    build_report,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def _calibration_report() -> dict:
    return {
        "run_id": "stage_a_saved_output_nonflag_candidate_rank_qwen05b_cayuga_20260710",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "train_summary": {
            "exact_top1": 4,
            "cases": 16,
            "top_pair_counts": {"ground/supported": 4, "verify/insufficient": 12},
        },
        "raw_heldout_summary": {
            "exact_top1": 1,
            "cases": 4,
            "top_pair_counts": {"ground/supported": 1, "verify/insufficient": 3},
        },
        "calibrated_heldout_summary": {
            "exact_top1": 1,
            "cases": 4,
            "top_pair_counts": {
                "defer/insufficient": 2,
                "ground/supported": 1,
                "reject/contradicted": 1,
            },
        },
        "train_selected_gate_report": {
            "strict_final_correct": 1,
            "trusted": 0,
            "trusted_incorrect": 0,
            "threshold": 0.113555,
        },
    }


def _field_report() -> dict:
    return {
        "summary": {
            "exact_top1": 1,
            "cases": 4,
            "top_pair_counts": {"ground/supported": 1, "verify/insufficient": 3},
            "field_diagnostic": {
                "action_top1": 1,
                "evidence_status_top1": 2,
                "field_rank_patterns": {
                    "action_field_failure": 1,
                    "both_field_failure": 2,
                    "pair_top1": 1,
                },
            },
        },
    }


def _arbitration_report() -> dict:
    return {
        "summary": {
            "best_policy_names": [
                "evidence_gate_override",
                "hybrid_evidence_then_train_gate",
            ],
            "by_policy": {
                "raw_candidate_top1": {
                    "exact": 1,
                    "rows": 4,
                    "trusted_candidate": 4,
                    "trusted_candidate_incorrect": 3,
                    "error_case_ids": [
                        "stage_a::000007",
                        "stage_a::000012",
                        "stage_a::000021",
                    ],
                    "by_predicted_pair": {
                        "ground/supported": 1,
                        "verify/insufficient": 3,
                    },
                },
                "calibrated_candidate_top1": {
                    "exact": 1,
                    "rows": 4,
                    "trusted_candidate": 4,
                    "trusted_candidate_incorrect": 3,
                    "error_case_ids": [
                        "stage_a::000007",
                        "stage_a::000019",
                        "stage_a::000021",
                    ],
                    "by_predicted_pair": {
                        "defer/insufficient": 2,
                        "ground/supported": 1,
                        "reject/contradicted": 1,
                    },
                },
                "train_selected_score_gap_gate": {
                    "exact": 1,
                    "rows": 4,
                    "trusted_candidate": 0,
                    "trusted_candidate_incorrect": 0,
                    "error_case_ids": [
                        "stage_a::000007",
                        "stage_a::000019",
                        "stage_a::000021",
                    ],
                    "by_predicted_pair": {"defer/insufficient": 4},
                },
                "evidence_gate_override": {
                    "exact": 4,
                    "rows": 4,
                    "trusted_candidate": 0,
                    "trusted_candidate_incorrect": 0,
                    "error_case_ids": [],
                    "by_predicted_pair": {
                        "defer/insufficient": 1,
                        "flag/invalid_value": 1,
                        "reject/contradicted": 1,
                        "verify/insufficient": 1,
                    },
                },
                "hybrid_evidence_then_train_gate": {
                    "exact": 4,
                    "rows": 4,
                    "trusted_candidate": 0,
                    "trusted_candidate_incorrect": 0,
                    "error_case_ids": [],
                    "by_predicted_pair": {
                        "defer/insufficient": 1,
                        "flag/invalid_value": 1,
                        "reject/contradicted": 1,
                        "verify/insufficient": 1,
                    },
                },
            },
        },
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    calibration = tmp_path / "calibration.json"
    field = tmp_path / "field.json"
    arbitration = tmp_path / "arbitration.json"
    calibration.write_text(json.dumps(_calibration_report()))
    field.write_text(json.dumps(_field_report()))
    arbitration.write_text(json.dumps(_arbitration_report()))
    return calibration, field, arbitration


def test_nonflag_checkpoint_result_keeps_runtime_baseline_gate_closed(tmp_path: Path) -> None:
    calibration, field, arbitration = _write_inputs(tmp_path)
    report = build_report(
        calibration_report=_calibration_report(),
        field_report=_field_report(),
        arbitration_report=_arbitration_report(),
        calibration_report_path=calibration,
        field_report_path=field,
        arbitration_report_path=arbitration,
    )

    result = report["result"]
    assert result["train_candidate_top1"] == {
        "exact": 4,
        "rows": 16,
        "top_pair_counts": {"ground/supported": 4, "verify/insufficient": 12},
    }
    assert result["raw_heldout_candidate_top1"]["exact"] == 1
    assert result["calibrated_heldout_candidate_top1"]["exact"] == 1
    assert result["field_diagnostic"]["action_top1"] == 1
    assert result["field_diagnostic"]["evidence_status_top1"] == 2

    policies = {row["policy"]: row for row in result["runtime_arbitration"]["policies"]}
    assert policies["raw_candidate_top1"]["trusted_candidate_incorrect"] == 3
    assert policies["calibrated_candidate_top1"]["trusted_candidate_incorrect"] == 3
    assert policies["evidence_gate_override"]["exact"] == 4
    assert policies["hybrid_evidence_then_train_gate"]["exact"] == 4
    assert report["decision"]["passes_meet_or_beat_gate"] is False
    assert report["decision"]["selected_next_step"] == "keep_runtime_evidence_arbitration_baseline"


def test_nonflag_checkpoint_result_is_public_safe(tmp_path: Path) -> None:
    calibration, field, arbitration = _write_inputs(tmp_path)
    report = build_report(
        calibration_report=_calibration_report(),
        field_report=_field_report(),
        arbitration_report=_arbitration_report(),
        calibration_report_path=calibration,
        field_report_path=field,
        arbitration_report_path=arbitration,
    )
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
    assert "private_compact_analyzer_output_not_public_source" in serialized
    assert "post_training/runs" not in serialized
    assert "post_training/runs" not in text
    assert "/tmp/" not in serialized
    assert "/tmp/" not in text
    assert "candidate-score JSONL" in text


def test_nonflag_checkpoint_result_cli_writes_public_safe_outputs(tmp_path: Path) -> None:
    calibration, field, arbitration = _write_inputs(tmp_path)
    out_json = tmp_path / "nonflag_result.json"
    out_md = tmp_path / "NONFLAG_RESULT.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_nonflag_checkpoint_result.py",
            "--calibration-report",
            str(calibration),
            "--field-report",
            str(field),
            "--arbitration-report",
            str(arbitration),
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
    assert stdout["dataset"] == "negbiodb_ct_stage_a_saved_output_nonflag_checkpoint_result_v1"
    assert payload["result"]["raw_heldout_candidate_top1"]["exact"] == 1
    assert payload["result"]["runtime_arbitration"]["best_policy_names"] == [
        "evidence_gate_override",
        "hybrid_evidence_then_train_gate",
    ]
    assert "Stage A Saved-Output Non-Flag Checkpoint Result" in text
    assert "raw saved predictions" in text
