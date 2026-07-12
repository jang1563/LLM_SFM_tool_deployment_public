import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_predictions import (
    build_report,
    expected_case_ids_from_rows,
    load_jsonl,
)
from post_training.run_stage_a_sft_smoke_eval import load_manifest_rows


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def manifest_rows() -> list[dict]:
    return load_manifest_rows(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl")


def heldout_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_sft_heldout_v1.jsonl")


def test_stage_a_prediction_eval_scores_full_trajectory_prediction() -> None:
    heldout = heldout_rows()
    prediction = {
        "case_id": heldout[0]["source_manifest_case_id"],
        "source": "oracle_debug",
        "trajectory": heldout[0]["target_trajectory"],
    }

    report = build_report(
        manifest_rows=manifest_rows(),
        prediction_rows=[prediction],
        expected_case_ids=[heldout[0]["source_manifest_case_id"]],
        run_id="unit",
    )

    assert report["summary"]["passed"] == 1
    assert report["rows"][0]["passed"]
    assert report["rows"][0]["prediction_source"] == "oracle_debug"


def test_stage_a_prediction_eval_does_not_autofill_missing_tool_arguments() -> None:
    heldout = heldout_rows()
    trajectory = heldout[0]["target_trajectory"]
    prediction = {
        "case_id": heldout[0]["source_manifest_case_id"],
        "source": "compact_model_output",
        "prediction": {
            "action": trajectory["terminal_action"],
            "evidence_status": trajectory["predicted_evidence_status"],
            "tool_calls": [step["name"] for step in trajectory["steps"]],
            "cited_source_ids": trajectory["cited_source_ids"],
        },
    }

    report = build_report(
        manifest_rows=manifest_rows(),
        prediction_rows=[prediction],
        expected_case_ids=[heldout[0]["source_manifest_case_id"]],
        run_id="unit",
    )

    assert report["summary"]["passed"] == 0
    assert "query_filter_missing_required_field" in report["rows"][0]["violations"]


def test_stage_a_prediction_eval_fails_closed_for_missing_and_unexpected_cases() -> None:
    heldout = heldout_rows()
    prediction = {
        "case_id": heldout[0]["source_manifest_case_id"],
        "trajectory": heldout[0]["target_trajectory"],
    }

    report = build_report(
        manifest_rows=manifest_rows(),
        prediction_rows=[prediction],
        expected_case_ids=[
            heldout[0]["source_manifest_case_id"],
            heldout[1]["source_manifest_case_id"],
        ],
        run_id="unit",
    )

    assert report["summary"]["passed"] == 1
    assert report["summary"]["violations"]["missing_prediction"] == 1

    extra = dict(prediction, case_id=heldout[2]["source_manifest_case_id"])
    report_with_extra = build_report(
        manifest_rows=manifest_rows(),
        prediction_rows=[prediction, extra],
        expected_case_ids=[heldout[0]["source_manifest_case_id"]],
        run_id="unit",
    )

    assert report_with_extra["summary"]["violations"]["unexpected_prediction_case_id"] == 1


def test_stage_a_prediction_eval_cli_scores_expected_sft_file(tmp_path: Path) -> None:
    heldout = heldout_rows()
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "case_id": heldout[0]["source_manifest_case_id"],
                "trajectory": heldout[0]["target_trajectory"],
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_predictions.py",
            "--predictions",
            str(predictions),
            "--expected-sft",
            "post_training/stage_a_sft_heldout_v1.jsonl",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["cases"] == 5
    assert summary["passed"] == 1
    assert summary["violations"]["missing_prediction"] == 4


def test_expected_case_ids_from_sft_rows_preserves_order() -> None:
    heldout = heldout_rows()

    assert expected_case_ids_from_rows(heldout) == [
        row["source_manifest_case_id"] for row in heldout
    ]
