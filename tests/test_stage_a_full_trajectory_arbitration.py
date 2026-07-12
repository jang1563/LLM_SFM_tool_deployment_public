import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_full_trajectory_arbitration import build_report
from post_training.run_stage_a_sft_smoke_eval import load_jsonl, load_manifest_rows


ROOT = Path(__file__).resolve().parents[1]


def report() -> dict:
    return build_report(
        manifest_rows=load_manifest_rows(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl"),
        train_rows=load_jsonl(ROOT / "post_training" / "stage_a_sft_train_v1.jsonl"),
        heldout_rows=load_jsonl(ROOT / "post_training" / "stage_a_sft_heldout_v1.jsonl"),
        component_rows=load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_v1.jsonl"),
    )


def test_runtime_gate_and_hybrid_pass_full_trajectory_evaluator() -> None:
    data = report()

    assert data["canonical_evaluator"] == "negbiodb_ct.stage_a_manifest.score_stage_a_trajectory"
    assert data["raw_model_outputs_used"] is False
    assert data["summary"]["all"]["runtime_gate_full"]["passed"] == 25
    assert data["summary"]["heldout"]["runtime_gate_full"]["passed"] == 5
    assert data["summary"]["all"]["hybrid_runtime_over_collapse"]["passed"] == 25
    assert data["summary"]["heldout"]["hybrid_runtime_over_collapse"]["passed"] == 5


def test_collapse_and_citationless_fail_different_full_trajectory_gates() -> None:
    data = report()
    collapse = data["summary"]["all"]["ground_supported_collapse"]
    citationless = data["summary"]["all"]["citationless_runtime_action"]

    assert collapse["passed"] == 5
    assert collapse["unsafe_ground_supported_overrides"] == 20
    assert collapse["violations"]["terminal_action_mismatch"] == 20
    assert citationless["passed"] == 15
    assert citationless["unsafe_ground_supported_overrides"] == 0
    assert citationless["violations"] == {"missing_required_attribution": 10}


def test_full_trajectory_arbitration_cli_writes_public_safe_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "arbitration.json"
    out_md = tmp_path / "arbitration.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_full_trajectory_arbitration.py",
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
    data = json.loads(out_json.read_text())
    assert data["summary"]["heldout"]["runtime_gate_full"]["passed"] == 5
    text = out_md.read_text()
    assert "Stage A Full-Trajectory Arbitration" in text
    assert "hidden_eval_metadata" not in text
    assert "source_task_id" not in text
