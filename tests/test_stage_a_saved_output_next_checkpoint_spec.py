import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_next_checkpoint_spec import (
    build_report,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def test_saved_output_next_checkpoint_spec_targets_flag_overselection() -> None:
    report = build_report()
    failure = report["observed_failure"]
    checkpoint = report["next_checkpoint"]
    env = checkpoint["env"]

    assert failure["bottleneck"] == "candidate_selection_bias_flag_invalid_value_overselection"
    assert failure["teacher_forced_margin_repaired"] is True
    assert failure["heldout_candidate_top1"] == {
        "exact": 1,
        "rows": 4,
        "top_pair_counts": {"flag/invalid_value": 4},
    }
    assert failure["heldout_field_diagnostic"]["field_rank_patterns"] == {
        "both_field_failure": 3,
        "pair_top1": 1,
    }
    assert failure["train_candidate_bias"] == {
        "exact_top1": 4,
        "rows": 16,
        "top_pair_counts": {"flag/invalid_value": 16},
    }

    assert checkpoint["name"] == "balanced_nonflag_candidate_rank_readout"
    assert env["FOCUS_CHOSEN_PAIRS"] == "defer/insufficient,reject/contradicted,verify/insufficient"
    assert env["FOCUS_ONLY"] == "0"
    assert env["SCORE_TRAINED_CANDIDATES"] == "1"
    assert env["SCORE_TRAIN_CANDIDATES"] == "1"
    assert env["SCORE_BASE_CANDIDATES"] == "1"
    assert env["CANDIDATE_POLICY"] == "train_observed_plus_rejected"


def test_saved_output_next_checkpoint_spec_keeps_release_gates_closed() -> None:
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
    assert report["acceptance_gate"] == {
        "candidate_or_model_policy_exact_min": 4,
        "trusted_candidate_incorrect_max": 0,
        "hidden_labels_used_by_arbitration": False,
        "raw_predictions_remain_uncommitted": True,
        "interpretation_rule": (
            "Only compact policy summaries that meet or beat runtime evidence "
            "arbitration reopen the downstream DPO/RLVR or release decision."
        ),
    }
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
    assert "release/public_release_manifest.json" in text


def test_saved_output_next_checkpoint_spec_cli_writes_public_safe_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "next_checkpoint_spec.json"
    out_md = tmp_path / "NEXT_CHECKPOINT_SPEC.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_next_checkpoint_spec.py",
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
    assert stdout["dataset"] == "negbiodb_ct_stage_a_saved_output_next_checkpoint_spec_v1"
    assert payload["next_checkpoint"]["name"] == "balanced_nonflag_candidate_rank_readout"
    assert "Stage A Saved-Output Next Checkpoint Spec" in text
    assert "balanced_nonflag_candidate_rank_readout" in text
    assert "raw_model_text" not in json.dumps(payload, sort_keys=True)
