import json
import subprocess
import sys
from pathlib import Path

from post_training.generate_stage_a_component_predictions import (
    build_saved_prediction_report,
    prediction_rows_for_component,
)
from post_training.run_stage_a_strict_component_sft_smoke import filter_component, load_jsonl


ROOT = Path(__file__).resolve().parents[1]


def evidence_heldout_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl")


def test_component_saved_predictions_oracle_routing_passes(tmp_path: Path) -> None:
    expected = filter_component(evidence_heldout_rows(), "routing_after_loop")
    predictions = prediction_rows_for_component(
        expected,
        component="routing_after_loop",
        mode="oracle",
        run_id="unit_component_oracle",
    )
    report = build_saved_prediction_report(
        expected_rows=expected,
        prediction_rows=predictions,
        component="routing_after_loop",
        mode="oracle",
        run_id="unit_component_oracle",
        targets_path="unit_targets.jsonl",
        predictions_path=tmp_path / "predictions.jsonl",
        eval_report_path=tmp_path / "eval_report.json",
    )

    assert report["eval_summary"]["passed"] == 5
    assert report["eval_summary"]["mean_score"] == 1.0
    assert report["eval_summary"]["violations"] == {}


def test_component_saved_predictions_ground_supported_collapse_is_partial_not_pass(tmp_path: Path) -> None:
    expected = filter_component(evidence_heldout_rows(), "routing_after_loop")
    predictions = prediction_rows_for_component(
        expected,
        component="routing_after_loop",
        mode="majority_ground_supported",
        run_id="unit_component_ground_supported",
    )
    report = build_saved_prediction_report(
        expected_rows=expected,
        prediction_rows=predictions,
        component="routing_after_loop",
        mode="majority_ground_supported",
        run_id="unit_component_ground_supported",
        targets_path="unit_targets.jsonl",
        predictions_path=tmp_path / "predictions.jsonl",
        eval_report_path=tmp_path / "eval_report.json",
    )

    assert report["eval_summary"]["passed"] == 1
    assert report["eval_summary"]["mean_score"] == 0.8
    assert report["eval_summary"]["violations"] == {"target_mismatch": 4}


def test_component_saved_predictions_routing_no_citations_separates_citation_gate(tmp_path: Path) -> None:
    expected = filter_component(evidence_heldout_rows(), "routing_after_loop")
    predictions = prediction_rows_for_component(
        expected,
        component="routing_after_loop",
        mode="routing_no_citations",
        run_id="unit_component_routing_no_citations",
    )
    report = build_saved_prediction_report(
        expected_rows=expected,
        prediction_rows=predictions,
        component="routing_after_loop",
        mode="routing_no_citations",
        run_id="unit_component_routing_no_citations",
        targets_path="unit_targets.jsonl",
        predictions_path=tmp_path / "predictions.jsonl",
        eval_report_path=tmp_path / "eval_report.json",
    )

    assert report["eval_summary"]["passed"] == 3
    assert report["eval_summary"]["mean_score"] == 0.9
    assert report["eval_summary"]["violations"] == {"target_mismatch": 2}


def test_component_saved_predictions_tool_names_only_fails_tool_shape(tmp_path: Path) -> None:
    expected = filter_component(evidence_heldout_rows(), "tool_query")
    predictions = prediction_rows_for_component(
        expected,
        component="tool_query",
        mode="tool_names_only",
        run_id="unit_component_tool_names_only",
    )
    report = build_saved_prediction_report(
        expected_rows=expected,
        prediction_rows=predictions,
        component="tool_query",
        mode="tool_names_only",
        run_id="unit_component_tool_names_only",
        targets_path="unit_targets.jsonl",
        predictions_path=tmp_path / "predictions.jsonl",
        eval_report_path=tmp_path / "eval_report.json",
    )

    assert report["eval_summary"]["passed"] == 0
    assert report["eval_summary"]["violations"]["tool_query_shape_invalid"] == 5
    assert report["eval_summary"]["violations"]["target_mismatch"] == 5


def test_component_saved_prediction_cli_writes_prediction_and_eval_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "component_saved_predictions"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/generate_stage_a_component_predictions.py",
            "--component",
            "routing_after_loop",
            "--mode",
            "routing_no_citations",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_component_saved_prediction_cli",
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
    assert report["mode"] == "routing_no_citations"
    assert report["examples"] == 5
    assert len(predictions) == 5
    assert eval_report["summary"]["passed"] == 3
