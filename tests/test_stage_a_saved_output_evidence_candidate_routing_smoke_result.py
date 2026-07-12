import json
import os
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_evidence_candidate_routing_smoke_result import (
    DATASET,
    build_report,
    render_markdown,
)
from post_training.run_stage_a_saved_output_evidence_candidate_routing_smoke import (
    DATASET as RUNNER_DATASET,
    build_eval_report,
    load_jsonl,
    target_output_from_row,
)


ROOT = Path(__file__).resolve().parents[1]


def tracked_rows() -> tuple[list[dict], list[dict]]:
    train_rows = load_jsonl(
        ROOT / "post_training" / "stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl"
    )
    heldout_rows = load_jsonl(
        ROOT / "post_training" / "stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl"
    )
    return train_rows, heldout_rows


def eval_payload(*, oracle: bool = True) -> dict:
    train_rows, heldout_rows = tracked_rows()

    def prediction(row: dict) -> dict:
        if oracle:
            return target_output_from_row(row)
        return {
            "action": "ground",
            "evidence_status": "supported",
            "selected_pair": "ground/supported",
        }

    train_predictions = [
        {
            "id": f"synthetic::{row['id']}",
            "source_candidate_routing_id": row["id"],
            "split": "train",
            "prediction": prediction(row),
        }
        for row in train_rows
    ]
    heldout_predictions = [
        {
            "id": f"synthetic::{row['id']}",
            "source_candidate_routing_id": row["id"],
            "split": "heldout",
            "prediction": prediction(row),
        }
        for row in heldout_rows
    ]
    return {
        "dataset": RUNNER_DATASET,
        "run_id": "synthetic_oracle" if oracle else "synthetic_static_ground",
        "train": build_eval_report(
            expected_rows=train_rows,
            prediction_rows=train_predictions,
            split="train",
            run_id="synthetic_train",
        ),
        "heldout": build_eval_report(
            expected_rows=heldout_rows,
            prediction_rows=heldout_predictions,
            split="heldout",
            run_id="synthetic_heldout",
        ),
    }


def write_eval(tmp_path: Path, payload: dict, name: str = "eval_report.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def test_evidence_candidate_routing_smoke_result_oracle_passes(tmp_path: Path) -> None:
    report = build_report(eval_report_path=write_eval(tmp_path, eval_payload(oracle=True)))

    assert report["dataset"] == DATASET
    assert report["passes_gate"] is True
    assert report["gate_violations"] == []
    assert report["policy_summary"]["exact"] == 5
    assert report["policy_summary"]["rows"] == 5
    assert report["policy_summary"]["bridge_focus_exact"] == 4
    assert report["decision"]["ready_for_escalation_review"] is True
    assert report["decision"]["ready_for_dpo_rlvr"] is False
    assert report["decision"]["selected_next_step"] == (
        "review_next_stage_a_model_heavy_step"
    )


def test_evidence_candidate_routing_smoke_result_static_prior_fails(tmp_path: Path) -> None:
    report = build_report(eval_report_path=write_eval(tmp_path, eval_payload(oracle=False)))

    assert report["passes_gate"] is False
    assert report["heldout_summary"]["exact"] == 1
    assert report["heldout_summary"]["bridge_focus_exact"] == 0
    assert report["gate_violations"] == [
        "below_heldout_exact_min",
        "below_bridge_focus_exact_min",
        "does_not_beat_static_prior",
    ]
    assert report["decision"]["ready_for_escalation_review"] is False
    assert report["decision"]["selected_next_step"] == (
        "keep_runtime_gate_and_review_candidate_collapse"
    )
    assert "failed" in report["decision"]["interpretation"]


def test_evidence_candidate_routing_smoke_result_fails_closed_on_raw_fields(
    tmp_path: Path,
) -> None:
    payload = eval_payload(oracle=True)
    payload["heldout"]["rows"][0]["candidate_scores"] = [
        {"score": 0.0, "candidate": {"selected_pair": "ground/supported"}}
    ]
    report = build_report(eval_report_path=write_eval(tmp_path, payload))

    assert report["passes_gate"] is False
    assert report["gate_violations"] == ["raw_fields_present_in_eval_report"]
    assert report["public_safety_contract"]["raw_fields_in_eval_report"] is True
    assert report["raw_field_paths"][0].endswith(".candidate_scores")


def test_evidence_candidate_routing_smoke_result_redacts_external_input_paths(
    tmp_path: Path,
) -> None:
    eval_path = write_eval(tmp_path, eval_payload(oracle=True))
    report = build_report(eval_report_path=eval_path)

    assert report["input_artifacts"]["eval_report"]["path"] == (
        "external_compact_input::eval_report.json"
    )
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)


def test_evidence_candidate_routing_smoke_result_redacts_relative_external_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_path = write_eval(tmp_path, eval_payload(oracle=True))
    monkeypatch.chdir(ROOT)
    relative_eval_path = Path(os.path.relpath(eval_path, ROOT))
    report = build_report(eval_report_path=relative_eval_path)

    assert report["input_artifacts"]["eval_report"]["path"] == (
        "external_compact_input::eval_report.json"
    )
    assert ".." not in report["input_artifacts"]["eval_report"]["path"]


def test_evidence_candidate_routing_smoke_result_markdown(tmp_path: Path) -> None:
    report = build_report(eval_report_path=write_eval(tmp_path, eval_payload(oracle=True)))
    markdown = render_markdown(report)

    assert "Held-out exact: 5/5" in markdown
    assert "Bridge-focus exact: 4/4" in markdown
    assert "Passes gate: `True`" in markdown
    assert "raw candidate-score JSONL" in markdown


def test_evidence_candidate_routing_smoke_result_cli_writes_outputs(tmp_path: Path) -> None:
    eval_path = write_eval(tmp_path, eval_payload(oracle=False))
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "SUMMARY.md"
    completed = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_smoke_result.py",
            "--eval-report",
            str(eval_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--policy",
            "synthetic_static_ground",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(completed.stdout)
    report = json.loads(out_json.read_text())
    assert stdout == report
    assert report["policy"] == "synthetic_static_ground"
    assert report["passes_gate"] is False
    assert "Passes gate: `False`" in out_md.read_text()
