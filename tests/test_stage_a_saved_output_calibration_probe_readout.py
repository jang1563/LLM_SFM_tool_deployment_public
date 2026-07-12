import json
import subprocess
import sys
from pathlib import Path

from post_training.run_stage_a_saved_output_calibration_probe_readout import (
    build_gate_report,
    build_readout_rows,
    evaluate_threshold,
    parse_thresholds,
    write_markdown,
)
from post_training.run_stage_a_sft_smoke_eval import load_jsonl
from post_training.run_stage_a_strict_component_sft_smoke import write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def probe_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_saved_output_calibration_probe_v1.jsonl")


def probe_manifest() -> dict:
    return json.loads((ROOT / "post_training" / "stage_a_saved_output_calibration_probe_manifest.json").read_text())


def test_saved_output_calibration_probe_readout_dry_run_is_split_safe(tmp_path: Path) -> None:
    readout_path = tmp_path / "readout.jsonl"
    rows = build_readout_rows(
        probe_rows(),
        run_id="unit_probe_readout",
        model=None,
        dry_run=True,
        client=None,
        max_length=128,
    )
    write_jsonl(readout_path, rows)

    report = build_gate_report(
        rows,
        probe_manifest=probe_manifest(),
        probe_pairs_path=ROOT / "post_training" / "stage_a_saved_output_calibration_probe_v1.jsonl",
        train_pairs_path=ROOT / "post_training" / "stage_a_saved_output_calibration_probe_train_v1.jsonl",
        heldout_pairs_path=ROOT / "post_training" / "stage_a_saved_output_calibration_probe_heldout_v1.jsonl",
        readout_path=readout_path,
        thresholds=[0.0, 0.05, 0.1],
    )

    assert report["rows"] == 20
    assert report["train_rows"] == 16
    assert report["heldout_rows"] == 4
    assert report["summary"]["train"]["exact_top1"] == 16
    assert report["summary"]["heldout"]["exact_top1"] == 4
    assert report["train_selected_zero_unsafe_report"]["trusted_incorrect"] == 0
    assert report["heldout_at_train_selected_threshold"]["strict_final_correct"] == 4
    assert report["probe_gate_target"]["heldout_probe_strict_final_correct_min"] == 4
    assert report["artifact_policy"]["candidate_score_jsonl_committed"] is False


def test_saved_output_calibration_probe_readout_gate_catches_high_gap_collapse() -> None:
    rows = [
        {
            "case_id": "stage_a::good",
            "target_pair": {"action": "flag", "evidence_status": "invalid_value"},
            "top_pair": {"action": "flag", "evidence_status": "invalid_value"},
            "exact_top1": True,
            "top_second_gap": 0.04,
        },
        {
            "case_id": "stage_a::bad",
            "target_pair": {"action": "reject", "evidence_status": "contradicted"},
            "top_pair": {"action": "ground", "evidence_status": "supported"},
            "exact_top1": False,
            "top_second_gap": 0.03,
        },
        {
            "case_id": "stage_a::defer",
            "target_pair": {"action": "defer", "evidence_status": "insufficient"},
            "top_pair": {"action": "ground", "evidence_status": "supported"},
            "exact_top1": False,
            "top_second_gap": 0.01,
        },
    ]

    loose = evaluate_threshold(rows, threshold=0.0)
    tight = evaluate_threshold(rows, threshold=0.031)

    assert loose["trusted_incorrect"] == 2
    assert loose["strict_final_correct"] == 1
    assert tight["trusted_incorrect"] == 0
    assert tight["trusted_correct"] == 1
    assert tight["fail_closed_exact_correct"] == 1
    assert tight["strict_final_correct"] == 2


def test_saved_output_calibration_probe_readout_cli_writes_public_safe_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "readout"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_output_calibration_probe_readout.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_probe_readout_cli",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert len(load_jsonl(out_dir / "readout.jsonl")) == 20
    assert report["heldout_at_train_selected_threshold"]["trusted_incorrect"] == 0
    assert "prompt_messages" not in (out_dir / "REPORT.md").read_text()
    assert "raw_output" not in (out_dir / "REPORT.md").read_text()
    assert "scheduler logs" in (out_dir / "REPORT.md").read_text()


def test_saved_output_calibration_probe_readout_requires_explicit_model_load() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_output_calibration_probe_readout.py",
            "--out-dir",
            "/tmp/stage_a_probe_readout_should_not_run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--allow-model-load" in result.stderr


def test_saved_output_calibration_probe_parse_thresholds_rejects_negative_values() -> None:
    assert parse_thresholds("0.1,0,0.1") == [0.0, 0.1]
    try:
        parse_thresholds("-0.1")
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative threshold should fail")


def test_saved_output_calibration_probe_markdown_is_public_safe(tmp_path: Path) -> None:
    readout_path = tmp_path / "readout.jsonl"
    rows = build_readout_rows(
        probe_rows()[:2],
        run_id="unit_probe_readout_md",
        model=None,
        dry_run=True,
        client=None,
        max_length=128,
    )
    write_jsonl(readout_path, rows)
    report = build_gate_report(
        rows,
        probe_manifest=probe_manifest(),
        probe_pairs_path=ROOT / "post_training" / "stage_a_saved_output_calibration_probe_v1.jsonl",
        train_pairs_path=ROOT / "post_training" / "stage_a_saved_output_calibration_probe_train_v1.jsonl",
        heldout_pairs_path=ROOT / "post_training" / "stage_a_saved_output_calibration_probe_heldout_v1.jsonl",
        readout_path=readout_path,
        thresholds=[0.0],
    )
    out_md = tmp_path / "REPORT.md"

    write_markdown(report, out_md)

    text = out_md.read_text()
    assert "Stage A Saved-Output Calibration Probe Readout" in text
    assert "raw prompts" in text
    assert "prompt_messages" not in text
    assert "raw_output" not in text


def test_saved_output_calibration_probe_cluster_templates_keep_outputs_ignored() -> None:
    for name in (
        "run_stage_a_saved_output_calibration_probe_readout_cayuga.sbatch",
        "run_stage_a_saved_output_calibration_probe_readout_expanse.sbatch",
    ):
        text = (ROOT / "post_training" / name).read_text()
        assert "--allow-model-load" in text
        assert "post_training/runs/${RUN_ID}" in text
        assert "<allocation>" in text
        assert "scheduler logs" in text
        assert "should not be committed" in text
