import json
import subprocess
import sys
from pathlib import Path

import pytest

from post_training.run_stage_a_saved_prediction_candidate_readout import (
    build_prediction_rows,
    candidate_pairs,
    load_jsonl,
    summarize_candidate_readout,
)
from post_training.run_stage_a_sft_smoke_eval import load_manifest_rows
from post_training.evaluate_stage_a_predictions import build_report, expected_case_ids_from_rows


ROOT = Path(__file__).resolve().parents[1]


def train_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_sft_train_v1.jsonl")


def heldout_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_sft_heldout_v1.jsonl")


def manifest_rows() -> list[dict]:
    return load_manifest_rows(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl")


def test_train_observed_candidate_pairs_cover_stage_a_actions() -> None:
    pairs = candidate_pairs(policy="train_observed_pairs", train_rows=train_rows())

    assert pairs == [
        {"action": "defer", "evidence_status": "insufficient"},
        {"action": "flag", "evidence_status": "invalid_value"},
        {"action": "ground", "evidence_status": "supported"},
        {"action": "reject", "evidence_status": "contradicted"},
        {"action": "verify", "evidence_status": "insufficient"},
    ]


def test_dry_run_candidate_readout_is_citation_limited_not_schema_limited() -> None:
    heldout = heldout_rows()
    predictions = build_prediction_rows(
        heldout_rows=heldout,
        train_rows=train_rows(),
        run_id="unit_candidate_dry",
        prompt_contract="stage_a_v4_canonical_json",
        candidate_policy="train_observed_pairs",
        model=None,
        client=None,
        max_length=2048,
    )
    report = build_report(
        manifest_rows=manifest_rows(),
        prediction_rows=predictions,
        expected_case_ids=expected_case_ids_from_rows(heldout),
        run_id="unit_candidate_dry",
    )

    assert report["summary"]["passed"] == 3
    assert report["summary"]["mean_score"] == pytest.approx(0.943)
    assert report["summary"]["gate_accuracy"]["required_tool_sequence"] == 1.0
    assert report["summary"]["gate_accuracy"]["query_filter_completeness"] == 1.0
    assert report["summary"]["gate_accuracy"]["terminal_action"] == 1.0
    assert report["summary"]["violations"] == {"missing_required_attribution": 2}
    assert summarize_candidate_readout(predictions)["exact_pair_top1"] == 5
    assert "NCT00588770" not in json.dumps(predictions, sort_keys=True)
    assert "NCT00828178" not in json.dumps(predictions, sort_keys=True)


def test_candidate_readout_cli_dry_run_writes_public_safe_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "candidate_readout"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_prediction_candidate_readout.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_candidate_readout_cli",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    predictions = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    eval_report = json.loads((out_dir / "eval_report.json").read_text())
    assert report["dry_run"] is True
    assert report["candidate_space_size"] == 5
    assert report["eval_summary"]["passed"] == 3
    assert len(predictions) == 5
    assert eval_report["summary"]["violations"] == {"missing_required_attribution": 2}


def test_candidate_readout_requires_explicit_model_load_without_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_prediction_candidate_readout.py",
            "--out-dir",
            "/tmp/stage_a_candidate_readout_should_not_write",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--allow-model-load" in result.stderr
