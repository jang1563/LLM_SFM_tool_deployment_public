import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_candidate_ce_checkpoint_spec import (
    build_report,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_ce_checkpoint_spec_targets_nonflag_failure() -> None:
    report = build_report()
    failure = report["observed_failure"]
    checkpoint = report["next_checkpoint"]
    env = checkpoint["env"]

    assert failure["bottleneck"] == "candidate_selection_not_repaired_by_nonflag_oversampling"
    assert failure["raw_heldout_candidate_top1"]["exact"] == 1
    assert failure["calibrated_heldout_candidate_top1"]["exact"] == 1
    assert failure["field_diagnostic"]["action_top1"] == 1
    assert failure["field_diagnostic"]["evidence_status_top1"] == 2
    assert failure["runtime_evidence_baseline"] == {
        "policy": "evidence_gate_override",
        "exact": 4,
        "rows": 4,
        "trusted_candidate": 0,
        "trusted_candidate_incorrect": 0,
    }

    assert checkpoint["name"] == "candidate_ce_action_status_pair_field_readout"
    assert "listwise candidate CE objective" in checkpoint["question"]
    assert env["CANDIDATE_CE_WEIGHT"] == "1"
    assert env["CANDIDATE_CE_MODE"] == "pair_plus_field"
    assert env["CANDIDATE_TARGET_FORMAT"] == "action_status_only"
    assert env["FOCUS_CHOSEN_PAIRS"] == ""
    assert env["FOCUS_REPEAT"] == "1"
    assert "--candidate-ce-weight 1" in checkpoint["dry_run_command"]


def test_candidate_ce_checkpoint_spec_keeps_release_gates_closed() -> None:
    report = build_report()
    text = render_markdown(report)
    serialized = json.dumps(report, sort_keys=True)

    assert set(report["keep_gated"]) == {
        "tool_query",
        "DPO/RLVR",
        "Hugging Face publication",
        "release tagging",
        "broad retraining",
    }
    assert report["acceptance_gate"]["candidate_or_model_policy_exact_min"] == 4
    assert report["acceptance_gate"]["trusted_candidate_incorrect_max"] == 0
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
    assert "candidate-score JSONL" in text


def test_candidate_ce_checkpoint_spec_cli_writes_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "candidate_ce_spec.json"
    out_md = tmp_path / "CANDIDATE_CE_SPEC.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_candidate_ce_checkpoint_spec.py",
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
    assert stdout["dataset"] == "negbiodb_ct_stage_a_saved_output_candidate_ce_checkpoint_spec_v1"
    assert payload["next_checkpoint"]["env"]["CANDIDATE_CE_MODE"] == "pair_plus_field"
    assert "Stage A Saved-Output Candidate-CE Checkpoint Spec" in text
    assert "runtime evidence arbitration" in text
