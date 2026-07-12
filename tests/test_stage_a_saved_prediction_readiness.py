import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_prediction_readiness import (
    build_report,
    candidate_gate_records,
    compact_summary_records,
    load_json,
)
from post_training.run_stage_a_sft_smoke_eval import load_jsonl, load_manifest_rows


ROOT = Path(__file__).resolve().parents[1]


def manifest_rows() -> list[dict]:
    return load_manifest_rows(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl")


def heldout_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_sft_heldout_v1.jsonl")


def full_arbitration() -> dict:
    return load_json(ROOT / "post_training" / "stage_a_full_trajectory_arbitration_2026-07-09.json")


def compact_paths() -> list[Path]:
    return [
        ROOT / "post_training" / "stage_a_cayuga_hf_chat_baseline_summary_2026-07-04.json",
        ROOT / "post_training" / "stage_a_cayuga_strict_contract_summary_2026-07-04.json",
        ROOT / "post_training" / "stage_a_strict_sft_cayuga_smoke_summary_2026-07-04.json",
    ]


def candidate_gate_paths() -> list[Path]:
    return [
        ROOT
        / "post_training"
        / "stage_a_saved_candidate_gate_train_observed_qwen05b_2026-07-09.json",
        ROOT / "post_training" / "stage_a_saved_candidate_gate_all_valid_qwen05b_2026-07-09.json",
    ]


def test_saved_prediction_readiness_keeps_real_outputs_gated() -> None:
    report = build_report(
        full_arbitration=full_arbitration(),
        manifest_rows=manifest_rows(),
        heldout_rows=heldout_rows(),
        compact_summary_paths=compact_paths(),
        candidate_gate_paths=candidate_gate_paths(),
    )

    assert report["raw_model_outputs_used"] is False
    assert report["raw_model_outputs_committed"] is False
    assert report["heldout_scorecard"]["ground_supported_collapse"]["passed"] == 1
    assert report["heldout_scorecard"]["citationless_runtime_action"]["passed"] == 3
    assert report["heldout_scorecard"]["runtime_gate_full"]["passed"] == 5
    assert report["decision"]["ready_for_tool_query"] is False
    assert report["decision"]["ready_for_dpo_rlvr"] is False
    assert "does not beat the collapse baseline" in " ".join(report["decision"]["blockers"])
    assert report["decision"]["best_real_saved_output"]["passed"] == 0
    assert report["decision"]["best_saved_candidate_gate"]["strict_final_correct"] == 2
    assert report["decision"]["best_saved_candidate_gate"]["trusted_incorrect"] == 0
    assert "saved-candidate gate remains below citationless" in " ".join(
        report["decision"]["blockers"]
    )


def test_candidate_gate_records_capture_zero_unsafe_fail_closed_result() -> None:
    records = candidate_gate_records(candidate_gate_paths())

    assert {record["candidate_policy"] for record in records} == {
        "train_observed_pairs",
        "all_valid_pairs",
    }
    assert all(record["trusted"] == 1 for record in records)
    assert all(record["trusted_incorrect"] == 0 for record in records)
    assert all(record["strict_final_correct"] == 2 for record in records)
    assert all(record["raw_predictions_committed"] is False for record in records)


def test_deterministic_oracle_validates_adapter_but_not_model_candidate() -> None:
    report = build_report(
        full_arbitration=full_arbitration(),
        manifest_rows=manifest_rows(),
        heldout_rows=heldout_rows(),
        compact_summary_paths=compact_paths()[:1],
    )
    records = {record["name"]: record for record in report["records"]}

    oracle = records["deterministic_saved_oracle"]
    assert oracle["passed"] == 5
    assert oracle["model_candidate"] is False
    assert oracle["comparison"]["matches_runtime_pass_count"] is True

    compact_tool = records["deterministic_compact_tool_names_oracle"]
    assert compact_tool["passed"] == 0
    assert compact_tool["violations"]["query_filter_missing_required_field"] == 5


def test_compact_summary_records_flattens_strict_sft_runs() -> None:
    records = compact_summary_records([compact_paths()[2]])

    assert {record["name"] for record in records} == {
        "stage_a_strict_sft_cayuga_qwen05b_2026_07_04",
        "stage_a_strict_sft_cayuga_qwen15b_2026_07_04",
    }
    assert all(record["source_type"] == "real_saved_sft_compact_summary" for record in records)
    assert all(record["model_candidate"] for record in records)


def test_saved_prediction_readiness_cli_writes_public_safe_report(tmp_path: Path) -> None:
    out_json = tmp_path / "readiness.json"
    out_md = tmp_path / "readiness.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_prediction_readiness.py",
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
    assert stdout["raw_model_outputs_used"] is False
    data = json.loads(out_json.read_text())
    text = out_md.read_text()
    assert data["decision"]["runtime_enforcement_required"] is True
    assert data["decision"]["best_saved_candidate_gate"]["strict_final_correct"] == 2
    assert "# Stage A Saved-Prediction Readiness" in text
    assert "Saved-Candidate Gates" in text
    assert "hidden_eval_metadata" not in text
    assert "raw_predictions_location" not in text
