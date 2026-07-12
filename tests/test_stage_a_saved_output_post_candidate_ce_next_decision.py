import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_post_candidate_ce_next_decision import (
    DATASET,
    build_report,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def test_post_candidate_ce_next_decision_selects_evidence_bridge() -> None:
    report = build_report()

    assert report["dataset"] == DATASET
    decision = report["decision"]
    assert decision["selected_next_step"] == "build_evidence_conditioned_saved_output_bridge"
    assert decision["passes_meet_or_beat_gate"] is False
    assert decision["candidate_best_exact"] == 1
    assert decision["candidate_rows"] == 4
    assert decision["runtime_arbitration_best_exact"] == 4
    assert decision["routing_evidence_gate_heldout_exact"] == 5
    assert decision["routing_evidence_gate_heldout_rows"] == 5
    assert "more_standalone_candidate_sft" in decision["rejected_next_steps"]
    assert "DPO_or_preference_optimization" in decision["rejected_next_steps"]
    assert "audited_RLVR" in decision["rejected_next_steps"]


def test_post_candidate_ce_next_decision_keeps_public_safe_inputs() -> None:
    report = build_report()
    serialized = json.dumps(report, sort_keys=True)
    markdown = render_markdown(report)

    assert report["public_safety_contract"] == {
        "raw_prediction_jsonl_read": False,
        "raw_candidate_score_jsonl_read": False,
        "scheduler_logs_read": False,
        "model_state_read": False,
        "ignored_run_folder_read": False,
        "raw_artifacts_committed": False,
        "hidden_labels_used_for_decision": False,
    }
    assert "post_training/runs" not in serialized
    assert "scheduler" not in markdown.lower()
    assert "allocation" not in serialized.lower()
    assert "partition" not in serialized.lower()


def test_post_candidate_ce_next_decision_summarizes_required_references() -> None:
    report = build_report()
    evidence = report["current_evidence"]

    checkpoints = {
        row["label"]: row
        for row in evidence["standalone_candidate_checkpoints"]
    }
    assert checkpoints["balanced_nonflag_candidate_rank_readout"]["best_candidate_exact"] == 1
    assert (
        checkpoints["candidate_ce_action_status_pair_field_readout"]["best_candidate_exact"]
        == 1
    )
    assert evidence["routing_evidence_gate"]["hidden_labels_used_by_gate"] is False
    assert evidence["routing_evidence_gate"]["model_visible_fields_only"] is True
    assert evidence["routing_evidence_gate"]["all"]["exact"] == 25
    hybrid = evidence["full_trajectory_arbitration"]["policies"][
        "hybrid_runtime_over_collapse"
    ]
    assert hybrid["passed"] == 25
    assert hybrid["cases"] == 25
    assert hybrid["unsafe_ground_supported_overrides"] == 0


def test_post_candidate_ce_next_decision_cli_writes_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "decision.json"
    out_md = tmp_path / "decision.md"
    completed = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_post_candidate_ce_next_decision.py",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    written = json.loads(out_json.read_text())
    assert stdout["dataset"] == DATASET
    assert written == stdout
    assert out_md.read_text().startswith(
        "# Stage A Saved-Output Post-Candidate-CE Next Decision"
    )
