import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from post_training.evaluate_stage_a_routing_evidence_boundary_gate import (
    build_report,
    gate_output,
)
from post_training.run_stage_a_strict_component_sft_smoke import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


def tracked_rows() -> tuple[list[dict], list[dict], list[dict]]:
    return (
        load_jsonl(ROOT / "post_training" / "stage_a_routing_defer_verify_contrast_pairs_v1.jsonl"),
        load_jsonl(ROOT / "post_training" / "stage_a_routing_defer_verify_contrast_pairs_train_v1.jsonl"),
        load_jsonl(ROOT / "post_training" / "stage_a_routing_defer_verify_contrast_pairs_heldout_v1.jsonl"),
    )


def test_evidence_boundary_gate_solves_tracked_defer_verify_slice() -> None:
    rows, train_rows, heldout_rows = tracked_rows()
    report = build_report(
        pairs=rows,
        train_pairs=train_rows,
        heldout_pairs=heldout_rows,
        pairs_path=ROOT / "post_training" / "stage_a_routing_defer_verify_contrast_pairs_v1.jsonl",
    )

    assert report["hidden_labels_used_by_gate"] is False
    assert report["summary"]["all"]["exact"] == 10
    assert report["summary"]["train"]["exact"] == 8
    assert report["summary"]["heldout"]["exact"] == 2
    assert report["summary"]["all"]["by_predicted_pair"] == {
        "defer/insufficient": 5,
        "verify/insufficient": 5,
    }
    assert report["summary"]["all"]["by_reason"] == {
        "no_same_indication_or_related_failure_record": 5,
        "related_evidence_without_same_indication_record": 5,
    }


def test_evidence_boundary_gate_fails_closed_when_related_count_missing() -> None:
    output, reason = gate_output(
        {
            "same_indication_record_count": 0,
            "records_considered": 0,
            "citation_candidate_count": 0,
            "related_negative_evidence_count": None,
            "completeness_signal": None,
        }
    )

    assert output == {"action": "defer", "evidence_status": "insufficient", "cited_source_ids": []}
    assert reason == "missing_related_count_fail_closed"


def test_evidence_boundary_gate_ignores_hidden_label_fields() -> None:
    rows, train_rows, heldout_rows = tracked_rows()
    mutated = deepcopy(rows)
    for row in mutated:
        row["case_family"] = "wrong_hidden_family"
        row["gold_evidence_status"] = "wrong_hidden_status"
        row["expected_terminal_action"] = "wrong_hidden_action"

    report = build_report(
        pairs=mutated,
        train_pairs=train_rows,
        heldout_pairs=heldout_rows,
        pairs_path=ROOT / "post_training" / "stage_a_routing_defer_verify_contrast_pairs_v1.jsonl",
    )

    assert report["summary"]["all"]["exact"] == 10


def test_evidence_boundary_gate_cli_writes_public_safe_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "gate.json"
    out_md = tmp_path / "gate.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_routing_evidence_boundary_gate.py",
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
    report = json.loads(out_json.read_text())
    assert report["summary"]["heldout"]["exact"] == 2
    text = out_md.read_text()
    assert "hidden_eval_metadata" not in text
    assert "source_task_id" not in text
    assert "Stage A Routing Evidence Boundary Gate" in text
