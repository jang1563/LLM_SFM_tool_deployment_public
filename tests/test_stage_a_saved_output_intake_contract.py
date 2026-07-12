import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_intake_contract import build_report


ROOT = Path(__file__).resolve().parents[1]
NEXT_DECISION = ROOT / "post_training" / "stage_a_saved_output_next_decision_2026-07-10.json"
MEET_OR_BEAT = ROOT / "post_training" / "stage_a_saved_output_meet_or_beat_gate_2026-07-10.json"


def test_saved_output_intake_contract_passes_current_compact_bundle() -> None:
    report = build_report(
        next_decision_path=NEXT_DECISION,
        meet_or_beat_gate_path=MEET_OR_BEAT,
    )

    assert report["raw_prediction_jsonl_read"] is False
    assert report["raw_candidate_score_jsonl_read"] is False
    assert report["scheduler_logs_read"] is False
    assert report["model_state_read"] is False
    assert report["ignored_run_folder_read"] is False
    assert report["passes_contract"] is True
    assert report["violations"] == []
    assert report["criteria_match"] is True
    assert report["next_artifacts_required"] == [
        "compact saved-output summary",
        "compact candidate calibration summary",
        "compact candidate arbitration summary",
        "updated saved-output next-decision report",
    ]
    assert report["next_decision_input_artifacts"]["candidate_calibration"]["sha256_matches"] is True
    assert report["next_decision_input_artifacts"]["candidate_arbitration"]["sha256_matches"] is True
    assert report["meet_or_beat_input_artifacts"]["next_decision"]["sha256_matches"] is True
    assert report["meet_or_beat_input_artifacts"]["candidate_arbitration"]["sha256_matches"] is True
    assert report["future_policy_public_safe_contract"]["complete"] is True
    assert report["future_policy_adapter_contract"]["complete"] is True
    assert report["future_policy_adapter_contract"]["required_dataset_matches"] is True
    assert report["future_policy_adapter_contract"]["missing_required_fields"] == []
    assert report["future_policy_adapter_contract"]["missing_source_provenance_rules"] == []


def test_saved_output_intake_contract_fails_on_stale_meet_or_beat_hash(tmp_path: Path) -> None:
    stale_gate = json.loads(MEET_OR_BEAT.read_text())
    stale_gate["input_artifacts"]["next_decision"]["sha256"] = "0" * 64
    stale_path = tmp_path / "stale_meet_or_beat.json"
    stale_path.write_text(json.dumps(stale_gate))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        meet_or_beat_gate_path=stale_path,
    )

    assert report["passes_contract"] is False
    assert "meet_or_beat_next_decision_sha256_mismatch" in report["violations"]


def test_saved_output_intake_contract_fails_on_missing_public_safe_flag(tmp_path: Path) -> None:
    gate = json.loads(MEET_OR_BEAT.read_text())
    gate["future_policy_input_contract"]["public_safe_flags"].remove(
        "raw_candidate_scores_committed=false"
    )
    gate_path = tmp_path / "missing_flag_meet_or_beat.json"
    gate_path.write_text(json.dumps(gate))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        meet_or_beat_gate_path=gate_path,
    )

    assert report["passes_contract"] is False
    assert "future_policy_public_safe_flags_incomplete" in report["violations"]
    assert report["future_policy_public_safe_contract"]["missing_required_flags"] == [
        "raw_candidate_scores_committed=false"
    ]


def test_saved_output_intake_contract_fails_on_missing_adapter_provenance(tmp_path: Path) -> None:
    gate = json.loads(MEET_OR_BEAT.read_text())
    gate["future_policy_input_contract"]["required_fields"].remove("source_report_sha256")
    gate["future_policy_input_contract"]["source_provenance_rules"].remove(
        "release/public_release_manifest.json must mark source_report safe_to_publish with the same SHA-256"
    )
    gate_path = tmp_path / "missing_adapter_provenance_meet_or_beat.json"
    gate_path.write_text(json.dumps(gate))

    report = build_report(
        next_decision_path=NEXT_DECISION,
        meet_or_beat_gate_path=gate_path,
    )

    assert report["passes_contract"] is False
    assert "future_policy_adapter_contract_incomplete" in report["violations"]
    assert report["future_policy_adapter_contract"]["missing_required_fields"] == [
        "source_report_sha256"
    ]
    assert report["future_policy_adapter_contract"]["missing_source_provenance_rules"] == [
        "release/public_release_manifest.json must mark source_report safe_to_publish with the same SHA-256"
    ]


def test_saved_output_intake_contract_cli_writes_public_safe_report(tmp_path: Path) -> None:
    out_json = tmp_path / "intake_contract.json"
    out_md = tmp_path / "INTAKE_CONTRACT.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_intake_contract.py",
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
    report = json.loads(out_json.read_text())
    text = out_md.read_text()
    assert stdout["passes_contract"] is True
    assert report["passes_contract"] is True
    assert "Stage A Saved-Output Intake Contract" in text
    assert "Future Policy Adapter Contract" in text
    assert "candidate-score JSONL" in text
    assert "raw_model_text" not in text
    assert "prompt_messages" not in text
