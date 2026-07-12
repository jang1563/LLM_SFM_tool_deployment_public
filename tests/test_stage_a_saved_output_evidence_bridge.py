import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_evidence_bridge import (
    DATASET,
    build_report,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def test_saved_output_evidence_bridge_maps_failures_to_visible_reasons() -> None:
    report = build_report()
    summary = report["bridge_summary"]

    assert report["dataset"] == DATASET
    assert summary["missing_evidence_rows"] == 0
    assert summary["all_joined_rows_runtime_exact"] is True
    assert summary["unique_failure_case_ids"] == [
        "stage_a::000007",
        "stage_a::000012",
        "stage_a::000019",
        "stage_a::000021",
    ]
    assert summary["runtime_reasons_by_case"] == {
        "stage_a::000007": "mixed_endpoint_records_for_same_claim",
        "stage_a::000012": "no_same_indication_or_related_failure_record",
        "stage_a::000019": "related_evidence_without_same_indication_record",
        "stage_a::000021": "invalid_numeric_value_in_same_indication_record",
    }
    assert summary["target_pairs_by_case"] == {
        "stage_a::000007": "reject/contradicted",
        "stage_a::000012": "defer/insufficient",
        "stage_a::000019": "verify/insufficient",
        "stage_a::000021": "flag/invalid_value",
    }
    assert summary["candidate_prediction_granularity"] == (
        "policy_level_predicted_pair_counts_only"
    )


def test_saved_output_evidence_bridge_keeps_escalation_closed() -> None:
    report = build_report()
    decision = report["decision"]

    assert decision["source_next_decision"] == (
        "build_evidence_conditioned_saved_output_bridge"
    )
    assert decision["selected_next_step"] == (
        "build_evidence_conditioned_candidate_routing_slice"
    )
    assert decision["do_not_run_more_standalone_sft_yet"] is True
    assert decision["ready_for_tool_query"] is False
    assert decision["ready_for_dpo_rlvr"] is False
    assert decision["ready_for_hugging_face_publication"] is False
    assert decision["ready_for_release_tagging"] is False
    assert report["next_data_contract"]["acceptance_gate"] == {
        "no_missing_evidence_gate_rows": True,
        "runtime_evidence_rows_exact": True,
        "candidate_granularity_disclosed": True,
        "raw_outputs_uncommitted": True,
    }


def test_saved_output_evidence_bridge_is_public_safe() -> None:
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
        "hidden_labels_used_for_bridge": False,
    }
    assert "post_training/runs" not in serialized
    assert "raw_candidate_score_table" in serialized
    assert "raw_candidate_score_table" not in markdown
    assert "allocation" not in serialized.lower()
    assert "partition" not in serialized.lower()


def test_saved_output_evidence_bridge_cli_writes_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "bridge.json"
    out_md = tmp_path / "bridge.md"
    completed = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_evidence_bridge.py",
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
        "# Stage A Saved-Output Evidence-Conditioned Bridge"
    )
