import hashlib
import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_meet_or_beat_gate import build_report


ROOT = Path(__file__).resolve().parents[1]
NEXT_DECISION = ROOT / "post_training" / "stage_a_saved_output_next_decision_2026-07-10.json"
ARBITRATION = ROOT / "post_training" / "stage_a_saved_output_candidate_arbitration_2026-07-10.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def future_policy_summary(
    tmp_path: Path,
    *,
    policy: str = "future_model_policy",
    exact: int = 4,
    rows: int = 4,
    trusted_candidate: int = 4,
    trusted_candidate_incorrect: int = 0,
) -> dict:
    return {
        "dataset": "negbiodb_ct_stage_a_saved_output_policy_summary_v1",
        "policy": policy,
        "source_kind": "candidate-arbitration-policy",
        "source_report": ARBITRATION.relative_to(ROOT).as_posix(),
        "source_report_sha256": sha256_file(ARBITRATION),
        "exact": exact,
        "rows": rows,
        "trusted_candidate": trusted_candidate,
        "trusted_candidate_incorrect": trusted_candidate_incorrect,
        "error_case_ids": [],
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
        },
        "raw_model_outputs_used": False,
        "raw_run_folders_used": False,
        "raw_predictions_committed": False,
        "raw_candidate_scores_committed": False,
        "raw_eval_report_committed": False,
        "raw_scheduler_logs_committed": False,
        "model_state_committed": False,
    }


def test_saved_output_meet_or_beat_gate_fails_current_candidate_policies() -> None:
    report = build_report(next_decision_path=NEXT_DECISION, arbitration_path=ARBITRATION)

    assert report["raw_model_outputs_used"] is False
    assert report["raw_run_folders_used"] is False
    assert report["raw_predictions_committed"] is False
    assert report["hidden_labels_used_by_arbitration"] is False
    assert report["requirements"]["selected_next_step"] == (
        "meet_or_beat_runtime_evidence_arbitration_baseline"
    )
    assert report["requirements"]["candidate_or_model_policy_exact_min"] == 4
    assert report["requirements"]["candidate_or_model_policy_rows_required"] == 4
    assert report["runtime_baseline"]["exact_min_to_meet_or_beat"] == 4
    assert report["runtime_baseline"]["rows_required_to_compare"] == 4
    assert report["passes_gate"] is False
    assert report["gate_violations"] == ["no_model_policy_meets_runtime_baseline"]

    by_policy = {row["policy"]: row for row in report["model_policies_under_test"]}
    assert by_policy["raw_candidate_top1"]["violations"] == [
        "below_runtime_arbitration_exact_min",
        "unsafe_candidate_trust",
    ]
    assert by_policy["calibrated_candidate_top1"]["violations"] == [
        "below_runtime_arbitration_exact_min",
        "unsafe_candidate_trust",
    ]
    assert by_policy["train_selected_score_gap_gate"]["violations"] == [
        "below_runtime_arbitration_exact_min"
    ]


def test_saved_output_meet_or_beat_gate_can_pass_a_future_model_policy(tmp_path: Path) -> None:
    future_summary = future_policy_summary(tmp_path)
    future_path = tmp_path / "future_policy_summary.json"
    future_path.write_text(json.dumps(future_summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(future_path,),
        model_policy_names=(),
    )

    assert report["passes_gate"] is True
    assert report["gate_violations"] == []
    assert report["model_policies_under_test"][0]["passes_gate"] is True
    assert report["external_policy_summary_artifacts"][0]["path"] == str(future_path)


def test_saved_output_meet_or_beat_gate_rejects_unprovenanced_future_policy(tmp_path: Path) -> None:
    future_summary = {
        "policy": "future_unprovenanced_policy",
        "exact": 4,
        "rows": 4,
        "trusted_candidate": 4,
        "trusted_candidate_incorrect": 0,
        "error_case_ids": [],
        "raw_model_outputs_used": False,
        "raw_run_folders_used": False,
        "raw_predictions_committed": False,
    }
    future_path = tmp_path / "future_unprovenanced_policy_summary.json"
    future_path.write_text(json.dumps(future_summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(future_path,),
        model_policy_names=(),
    )

    policy = report["model_policies_under_test"][0]
    assert report["passes_gate"] is False
    assert report["gate_violations"] == ["no_model_policy_meets_runtime_baseline"]
    assert policy["violations"] == [
        "policy_summary_dataset_mismatch",
        "policy_summary_source_kind_invalid",
        "policy_summary_source_report_missing",
        "policy_summary_public_safety_contract_missing",
        "policy_summary_raw_candidate_scores_committed_missing",
        "policy_summary_raw_eval_report_committed_missing",
        "policy_summary_raw_scheduler_logs_committed_missing",
        "policy_summary_model_state_committed_missing",
    ]


def test_saved_output_meet_or_beat_gate_rejects_source_sha_mismatch(tmp_path: Path) -> None:
    future_summary = future_policy_summary(tmp_path, policy="future_stale_source_policy")
    future_summary["source_report_sha256"] = "0" * 64
    future_path = tmp_path / "future_stale_source_policy_summary.json"
    future_path.write_text(json.dumps(future_summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(future_path,),
        model_policy_names=(),
    )

    policy = report["model_policies_under_test"][0]
    assert report["passes_gate"] is False
    assert policy["violations"] == ["policy_summary_source_report_sha256_mismatch"]


def test_saved_output_meet_or_beat_gate_rejects_unmanifested_source_report(tmp_path: Path) -> None:
    source_report = tmp_path / "compact_source_report.json"
    source_report.write_text(json.dumps({"source": "not a public manifest artifact"}))
    future_summary = future_policy_summary(tmp_path, policy="future_tmp_source_policy")
    future_summary["source_report"] = str(source_report)
    future_summary["source_report_sha256"] = sha256_file(source_report)
    future_path = tmp_path / "future_tmp_source_policy_summary.json"
    future_path.write_text(json.dumps(future_summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(future_path,),
        model_policy_names=(),
    )

    policy = report["model_policies_under_test"][0]
    assert report["passes_gate"] is False
    assert policy["violations"] == ["policy_summary_source_report_not_repo_relative"]


def test_saved_output_meet_or_beat_gate_rejects_raw_future_policy_artifact(tmp_path: Path) -> None:
    future_summary = future_policy_summary(tmp_path, policy="future_raw_policy")
    future_summary["raw_model_outputs_used"] = True
    future_path = tmp_path / "future_raw_policy_summary.json"
    future_path.write_text(json.dumps(future_summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(future_path,),
        model_policy_names=(),
    )

    assert report["passes_gate"] is False
    assert report["gate_violations"] == ["future_raw_policy:raw_artifact_access"]


def test_saved_output_meet_or_beat_gate_rejects_committed_raw_score_artifact(tmp_path: Path) -> None:
    future_summary = future_policy_summary(tmp_path, policy="future_raw_score_policy")
    future_summary["raw_candidate_scores_committed"] = True
    future_path = tmp_path / "future_raw_score_policy_summary.json"
    future_path.write_text(json.dumps(future_summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(future_path,),
        model_policy_names=(),
    )

    assert report["passes_gate"] is False
    assert report["gate_violations"] == [
        "future_raw_score_policy:raw_candidate_scores_committed"
    ]


def test_saved_output_meet_or_beat_gate_rejects_impossible_policy_counts(tmp_path: Path) -> None:
    future_summary = future_policy_summary(
        tmp_path,
        policy="future_impossible_policy",
        exact=4,
        rows=2,
        trusted_candidate=1,
        trusted_candidate_incorrect=0,
    )
    future_path = tmp_path / "future_impossible_policy_summary.json"
    future_path.write_text(json.dumps(future_summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(future_path,),
        model_policy_names=(),
    )

    policy = report["model_policies_under_test"][0]
    assert report["passes_gate"] is False
    assert report["gate_violations"] == ["no_model_policy_meets_runtime_baseline"]
    assert policy["passes_gate"] is False
    assert policy["violations"] == [
        "exact_exceeds_rows",
        "rows_mismatch_runtime_baseline",
    ]


def test_saved_output_meet_or_beat_gate_rejects_noncomparable_row_count(tmp_path: Path) -> None:
    future_summary = future_policy_summary(
        tmp_path,
        policy="future_noncomparable_policy",
        exact=4,
        rows=5,
        trusted_candidate=0,
        trusted_candidate_incorrect=0,
    )
    future_path = tmp_path / "future_noncomparable_policy_summary.json"
    future_path.write_text(json.dumps(future_summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(future_path,),
        model_policy_names=(),
    )

    policy = report["model_policies_under_test"][0]
    assert report["passes_gate"] is False
    assert policy["violations"] == ["rows_mismatch_runtime_baseline"]


def test_saved_output_meet_or_beat_gate_rejects_negative_policy_counts(tmp_path: Path) -> None:
    future_summary = future_policy_summary(
        tmp_path,
        policy="future_negative_count_policy",
        exact=4,
        rows=4,
        trusted_candidate=0,
        trusted_candidate_incorrect=-1,
    )
    future_path = tmp_path / "future_negative_count_policy_summary.json"
    future_path.write_text(json.dumps(future_summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(future_path,),
        model_policy_names=(),
    )

    policy = report["model_policies_under_test"][0]
    assert report["passes_gate"] is False
    assert policy["violations"] == ["negative_policy_count"]


def test_saved_output_meet_or_beat_gate_rejects_coerced_policy_counts(tmp_path: Path) -> None:
    future_summary = future_policy_summary(tmp_path, policy="future_stringy_count_policy")
    future_summary["exact"] = "4"
    future_summary["rows"] = "4"
    future_summary["trusted_candidate"] = 4.0
    future_summary["trusted_candidate_incorrect"] = False
    future_path = tmp_path / "future_stringy_count_policy_summary.json"
    future_path.write_text(json.dumps(future_summary))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        arbitration_path=ARBITRATION,
        external_policy_summary_paths=(future_path,),
        model_policy_names=(),
    )

    policy = report["model_policies_under_test"][0]
    assert report["passes_gate"] is False
    assert report["gate_violations"] == ["no_model_policy_meets_runtime_baseline"]
    assert policy["passes_gate"] is False
    assert policy["violations"] == ["non_integer_policy_count"]
    assert policy["non_integer_count_fields"] == [
        "exact",
        "rows",
        "trusted_candidate",
        "trusted_candidate_incorrect",
    ]


def test_saved_output_meet_or_beat_gate_cli_writes_public_safe_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "meet_or_beat_gate.json"
    out_md = tmp_path / "MEET_OR_BEAT_GATE.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_meet_or_beat_gate.py",
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
    assert report["passes_gate"] is False
    assert report["future_policy_input_contract"]["required_fields"] == [
        "dataset",
        "policy",
        "source_kind",
        "source_report",
        "source_report_sha256",
        "exact",
        "rows",
        "trusted_candidate_incorrect",
        "public_safety_contract",
    ]
    assert report["future_policy_input_contract"]["required_dataset"] == (
        "negbiodb_ct_stage_a_saved_output_policy_summary_v1"
    )
    assert "candidate-arbitration-policy" in report["future_policy_input_contract"]["allowed_source_kinds"]
    assert "source_report must be a repo-relative public manifest path" in report["future_policy_input_contract"]["source_provenance_rules"]
    assert "source_report_sha256 must match the source_report contents" in report["future_policy_input_contract"]["source_provenance_rules"]
    assert "release/public_release_manifest.json must mark source_report safe_to_publish with the same SHA-256" in report["future_policy_input_contract"]["source_provenance_rules"]
    assert "raw_candidate_scores_committed=false" in report["future_policy_input_contract"]["public_safe_flags"]
    assert "raw_scheduler_logs_committed=false" in report["future_policy_input_contract"]["public_safe_flags"]
    assert "exact <= rows" in report["future_policy_input_contract"]["numeric_validity_rules"]
    text = out_md.read_text()
    assert "Stage A Saved-Output Meet-Or-Beat Gate" in text
    assert "Future Policy Input Contract" in text
    assert "Source provenance rules" in text
    assert "candidate-score JSONL" in text
    assert "raw_model_text" not in text
    assert "prompt_messages" not in text
