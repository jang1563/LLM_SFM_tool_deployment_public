from pathlib import Path
import json
import subprocess
import sys

from post_training.run_stage_a_sft_smoke_eval import (
    build_report,
    load_jsonl,
    load_manifest_rows,
    trajectory_from_payload,
)
from negbiodb_ct.stage_a_manifest import score_stage_a_trajectory


ROOT = Path(__file__).resolve().parents[1]


def test_stage_a_sft_smoke_eval_script_reports_oracle_and_shortcut_baselines() -> None:
    result = subprocess.run(
        [sys.executable, "post_training/run_stage_a_sft_smoke_eval.py", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["heldout"]["oracle_replay"]["passed"] == 5
    assert summary["heldout"]["self_answer"]["passed"] == 0
    assert summary["heldout"]["train_majority_replay"]["cases"] == 5
    assert "terminal_action" in summary["heldout"]["train_majority_replay"]["gate_accuracy"]


def test_stage_a_sft_smoke_eval_oracle_trajectory_roundtrip_passes() -> None:
    manifest_rows = load_manifest_rows(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl")
    heldout_rows = load_jsonl(ROOT / "post_training" / "stage_a_sft_heldout_v1.jsonl")
    manifest_by_case = {row["case_id"]: row for row in manifest_rows}
    row = heldout_rows[0]
    trajectory = trajectory_from_payload(
        row["target_trajectory"],
        target_input_id=row["source_manifest_case_id"],
    )

    result = score_stage_a_trajectory(manifest_by_case[row["source_manifest_case_id"]], trajectory)

    assert result.passed


def test_stage_a_sft_smoke_eval_nearest_policy_uses_train_source_only() -> None:
    report = build_report(
        manifest_rows=load_manifest_rows(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl"),
        train_rows=load_jsonl(ROOT / "post_training" / "stage_a_sft_train_v1.jsonl"),
        heldout_rows=load_jsonl(ROOT / "post_training" / "stage_a_sft_heldout_v1.jsonl"),
        policies=("nearest_train_replay",),
    )
    train_case_ids = {
        row["source_manifest_case_id"]
        for row in load_jsonl(ROOT / "post_training" / "stage_a_sft_train_v1.jsonl")
    }
    heldout_nearest_rows = report["splits"]["heldout"]["nearest_train_replay"]["rows"]

    assert heldout_nearest_rows
    assert all(row["source_train_case_id"] in train_case_ids for row in heldout_nearest_rows)
