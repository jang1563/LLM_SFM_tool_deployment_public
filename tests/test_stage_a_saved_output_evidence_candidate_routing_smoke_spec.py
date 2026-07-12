import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_evidence_candidate_routing_smoke_spec import (
    DATASET,
    build_report,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_candidate_routing_smoke_spec_preconditions_pass() -> None:
    report = build_report()
    preconditions = report["preconditions"]

    assert report["dataset"] == DATASET
    assert preconditions["all_pass"] is True
    assert preconditions["row_counts"] == {
        "row_count": 25,
        "train_rows": 20,
        "heldout_rows": 5,
        "bridge_focus_rows": 4,
    }
    assert all(preconditions["checks"].values())
    assert preconditions["runtime_evidence_gate"] == {
        "policy": "runtime_evidence_gate",
        "exact": 5,
        "rows": 5,
        "bridge_focus_exact": 4,
        "bridge_focus_rows": 4,
    }
    assert preconditions["best_static_prior_heldout_exact"] == 1
    assert preconditions["checks"]["runner_exists"] is True
    assert preconditions["checks"]["cayuga_sbatch_exists"] is True


def test_evidence_candidate_routing_smoke_spec_acceptance_gate_is_stricter_than_static_prior() -> None:
    report = build_report()
    gate = report["acceptance_gate"]
    spec = report["smoke_spec"]

    assert gate["candidate_model_heldout_exact_min"] == 5
    assert gate["candidate_model_heldout_rows"] == 5
    assert gate["candidate_model_bridge_focus_exact_min"] == 4
    assert gate["candidate_model_bridge_focus_rows"] == 4
    assert gate["best_static_prior_heldout_exact_to_beat"] == 1
    assert gate["runtime_evidence_gate_to_match"]["exact"] == 5
    assert "violation_counts" in gate["required_compact_outputs"]
    assert spec["candidate_pairs"] == [
        "ground/supported",
        "reject/contradicted",
        "defer/insufficient",
        "verify/insufficient",
        "flag/invalid_value",
    ]
    assert spec["train_rows"]["rows"] == 20
    assert spec["heldout_rows"]["rows"] == 5
    assert spec["heldout_rows"]["bridge_focus_rows"] == 4
    assert spec["runner"] == (
        "post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke.py"
    )
    assert spec["cayuga_sbatch"] == (
        "post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke_cayuga.sbatch"
    )


def test_evidence_candidate_routing_smoke_spec_keeps_escalation_closed() -> None:
    report = build_report()
    decision = report["decision"]
    serialized = json.dumps(report, sort_keys=True)

    assert decision["runner_implemented"] is True
    assert decision["ready_for_cayuga_dry_run"] is True
    assert decision["ready_for_cayuga_submission"] is False
    assert decision["ready_for_tool_query"] is False
    assert decision["ready_for_dpo_rlvr"] is False
    assert decision["ready_for_hugging_face_publication"] is False
    assert decision["ready_for_release_tagging"] is False
    assert report["public_safety_contract"] == {
        "raw_prediction_jsonl_read": False,
        "raw_candidate_score_jsonl_read": False,
        "scheduler_logs_read": False,
        "model_state_read": False,
        "ignored_run_folder_read": False,
        "hidden_labels_used_for_spec": False,
        "raw_artifacts_committed": False,
    }
    assert "post_training/runs" not in serialized


def test_evidence_candidate_routing_smoke_spec_markdown() -> None:
    report = build_report()
    markdown = render_markdown(report)

    assert "Stage A Saved-Output Evidence Candidate-Routing Smoke Spec" in markdown
    assert "Runtime evidence gate: 5/5 held-out, 4/4 bridge-focus" in markdown
    assert "Runner implemented: `True`" in markdown
    assert "Ready for Cayuga dry-run: `True`" in markdown
    assert "Candidate model held-out exact minimum: 5/5" in markdown
    assert "Ready for Cayuga submission: `False`" in markdown


def test_evidence_candidate_routing_smoke_spec_cli_writes_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "smoke_spec.json"
    out_md = tmp_path / "SMOKE_SPEC.md"
    completed = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_smoke_spec.py",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(completed.stdout)
    report = json.loads(out_json.read_text())
    markdown = out_md.read_text()
    assert stdout == report
    assert report["decision"]["runner_implemented"] is True
    assert report["decision"]["ready_for_cayuga_dry_run"] is True
    assert "Static prior held-out exact to beat: 1/5" in markdown
