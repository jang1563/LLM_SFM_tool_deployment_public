import json
import subprocess
import sys
from pathlib import Path

import pytest

from post_training.build_stage_a_saved_output_policy_summary import build_policy_summary
from post_training.evaluate_stage_a_saved_output_meet_or_beat_gate import build_report


ROOT = Path(__file__).resolve().parents[1]
NEXT_DECISION = ROOT / "post_training" / "stage_a_saved_output_next_decision_2026-07-10.json"
ARBITRATION = ROOT / "post_training" / "stage_a_saved_output_candidate_arbitration_2026-07-10.json"
PREDICTION_SUMMARY = ROOT / "post_training" / "stage_a_v4_canonical_json_qwen05b_cayuga_summary_2026-07-09.json"
CANDIDATE_GATE = ROOT / "post_training" / "stage_a_saved_candidate_gate_train_observed_qwen05b_2026-07-09.json"


def test_policy_summary_from_prediction_summary_is_public_safe() -> None:
    summary = build_policy_summary(
        source_path=PREDICTION_SUMMARY,
        source_kind="prediction-summary",
        policy_name="v4_canonical_json_direct",
    )

    assert summary["dataset"] == "negbiodb_ct_stage_a_saved_output_policy_summary_v1"
    assert summary["policy"] == "v4_canonical_json_direct"
    assert summary["exact"] == 0
    assert summary["rows"] == 5
    assert summary["trusted_candidate"] == 0
    assert summary["trusted_candidate_incorrect"] == 0
    assert summary["source_report"] == PREDICTION_SUMMARY.relative_to(ROOT).as_posix()
    assert summary["raw_predictions_committed"] is False
    assert summary["raw_candidate_scores_committed"] is False
    assert summary["raw_eval_report_committed"] is False
    assert summary["raw_scheduler_logs_committed"] is False
    assert summary["model_state_committed"] is False
    assert summary["public_safety_contract"] == {
        "raw_prediction_jsonl_read": False,
        "raw_candidate_score_jsonl_read": False,
        "scheduler_logs_read": False,
        "model_state_read": False,
        "ignored_run_folder_read": False,
    }


def test_policy_summary_from_candidate_gate_uses_strict_final_and_unsafe_trust() -> None:
    summary = build_policy_summary(
        source_path=CANDIDATE_GATE,
        source_kind="candidate-gate-summary",
        policy_name="train_observed_best_zero_unsafe",
    )

    assert summary["policy"] == "train_observed_best_zero_unsafe"
    assert summary["exact"] == 2
    assert summary["rows"] == 5
    assert summary["trusted_candidate"] == 1
    assert summary["trusted_candidate_incorrect"] == 0
    assert summary["error_case_ids"] == []
    assert summary["source_report"] == CANDIDATE_GATE.relative_to(ROOT).as_posix()
    assert summary["raw_candidate_scores_committed"] is False


def test_policy_summary_from_arbitration_can_feed_meet_or_beat_gate(tmp_path: Path) -> None:
    summary = build_policy_summary(
        source_path=ARBITRATION,
        source_kind="candidate-arbitration-policy",
        policy_name="train_selected_score_gap_gate",
    )
    summary_path = tmp_path / "policy_summary.json"
    summary_path.write_text(json.dumps(summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(summary_path,),
        model_policy_names=(),
    )

    assert report["passes_gate"] is False
    assert report["gate_violations"] == ["no_model_policy_meets_runtime_baseline"]
    assert report["model_policies_under_test"][0]["policy"] == "train_selected_score_gap_gate"
    assert report["model_policies_under_test"][0]["violations"] == [
        "below_runtime_arbitration_exact_min"
    ]


def test_policy_summary_propagates_raw_commit_flags_into_gate(tmp_path: Path) -> None:
    source = tmp_path / "unsafe_prediction_summary.json"
    source.write_text(
        json.dumps(
            {
                "artifact_policy": {
                    "raw_candidate_scores_committed": True,
                    "raw_eval_report_committed": False,
                    "raw_predictions_committed": False,
                    "raw_scheduler_logs_committed": False,
                    "summary_only": False,
                },
                "result": {
                    "cases": 4,
                    "passed": 4,
                },
                "run_id": "unsafe_candidate_score_summary",
            }
        )
    )
    summary = build_policy_summary(
        source_path=source,
        source_kind="prediction-summary",
    )
    summary_path = tmp_path / "policy_summary.json"
    summary_path.write_text(json.dumps(summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(summary_path,),
        model_policy_names=(),
    )

    assert summary["raw_candidate_scores_committed"] is True
    assert report["passes_gate"] is False
    assert "unsafe_candidate_score_summary:raw_candidate_scores_committed" in report["gate_violations"]
    assert "no_model_policy_meets_runtime_baseline" in report["gate_violations"]
    assert "policy_summary_source_report_not_repo_relative" in report["model_policies_under_test"][0]["violations"]


def test_policy_summary_adapter_rejects_coerced_source_counts(tmp_path: Path) -> None:
    source = tmp_path / "stringy_prediction_summary.json"
    source.write_text(
        json.dumps(
            {
                "artifact_policy": {
                    "raw_candidate_scores_committed": False,
                    "raw_eval_report_committed": False,
                    "raw_predictions_committed": False,
                    "raw_scheduler_logs_committed": False,
                    "summary_only": True,
                },
                "result": {
                    "cases": "4",
                    "passed": 4.0,
                },
                "run_id": "stringy_prediction_summary",
            }
        )
    )

    with pytest.raises(ValueError, match="must be a JSON integer"):
        build_policy_summary(
            source_path=source,
            source_kind="prediction-summary",
        )


def test_policy_summary_cli_writes_gate_compatible_json(tmp_path: Path) -> None:
    out = tmp_path / "candidate_policy_summary.json"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/build_stage_a_saved_output_policy_summary.py",
            "--source",
            str(ARBITRATION),
            "--source-kind",
            "candidate-arbitration-policy",
            "--policy",
            "calibrated_candidate_top1",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(out.read_text())
    assert payload["policy"] == "calibrated_candidate_top1"
    assert payload["exact"] == 2
    assert payload["rows"] == 4
    assert payload["trusted_candidate_incorrect"] == 2
    assert payload["source_report"] == ARBITRATION.relative_to(ROOT).as_posix()
    assert "raw_model_text" not in out.read_text()
