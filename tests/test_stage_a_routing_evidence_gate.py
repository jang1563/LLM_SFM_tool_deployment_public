import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from post_training.evaluate_stage_a_routing_evidence_gate import (
    build_report,
    gate_output,
)
from post_training.run_stage_a_strict_component_sft_smoke import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


def tracked_targets() -> tuple[list[dict], list[dict], list[dict]]:
    return (
        load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_v1.jsonl"),
        load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_train_v1.jsonl"),
        load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl"),
    )


def test_routing_evidence_gate_solves_all_stage_a_routing_targets() -> None:
    targets, train_targets, heldout_targets = tracked_targets()
    report = build_report(
        targets=targets,
        train_targets=train_targets,
        heldout_targets=heldout_targets,
        targets_path=ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_v1.jsonl",
    )

    assert report["hidden_labels_used_by_gate"] is False
    assert report["summary"]["all"]["exact"] == 25
    assert report["summary"]["train"]["exact"] == 20
    assert report["summary"]["heldout"]["exact"] == 5
    assert report["summary"]["all"]["by_predicted_pair"] == {
        "defer/insufficient": 5,
        "flag/invalid_value": 5,
        "ground/supported": 5,
        "reject/contradicted": 5,
        "verify/insufficient": 5,
    }


def test_routing_evidence_gate_handles_invalid_value_with_visible_source_id() -> None:
    output, reason = gate_output(
        {
            "value_validity_findings": [
                {
                    "field": "p_value",
                    "source_id": "NCT00000001",
                    "valid_range": [0.0, 1.0],
                    "value": 14.73,
                }
            ],
            "completeness_signal": "same_indication_record_has_invalid_numeric_value",
        }
    )

    assert output == {
        "action": "flag",
        "evidence_status": "invalid_value",
        "cited_source_ids": ["NCT00000001"],
    }
    assert reason == "invalid_numeric_value_in_same_indication_record"


def test_routing_evidence_gate_fails_closed_on_unknown_state() -> None:
    output, reason = gate_output(
        {
            "value_validity_findings": [],
            "completeness_signal": None,
            "same_indication_record_count": None,
            "records_considered": None,
            "related_negative_evidence_count": None,
        }
    )

    assert output == {"action": "defer", "evidence_status": "insufficient", "cited_source_ids": []}
    assert reason == "unknown_evidence_state_fail_closed"


def test_routing_evidence_gate_ignores_hidden_label_fields() -> None:
    targets, train_targets, heldout_targets = tracked_targets()
    mutated = deepcopy(targets)
    for row in mutated:
        row["case_family"] = "wrong_hidden_family"
        row["gold_evidence_status"] = "wrong_hidden_status"
        row["expected_terminal_action"] = "wrong_hidden_action"

    report = build_report(
        targets=mutated,
        train_targets=train_targets,
        heldout_targets=heldout_targets,
        targets_path=ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_v1.jsonl",
    )

    assert report["summary"]["all"]["exact"] == 25


def test_routing_evidence_gate_cli_writes_public_safe_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "routing_gate.json"
    out_md = tmp_path / "routing_gate.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_routing_evidence_gate.py",
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
    assert report["summary"]["heldout"]["exact"] == 5
    text = out_md.read_text()
    assert "Stage A Routing Evidence Gate" in text
    assert "hidden_eval_metadata" not in text
    assert "source_task_id" not in text
