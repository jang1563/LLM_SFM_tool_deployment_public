import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_evidence_candidate_routing_readout import (
    DATASET,
    build_report,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def test_saved_output_evidence_candidate_routing_readout_scores_runtime_gate() -> None:
    report = build_report()
    decision = report["decision"]
    runtime = report["policies"]["runtime_evidence_gate"]["heldout"]

    assert report["dataset"] == DATASET
    assert decision["selected_next_step"] == (
        "prepare_evidence_conditioned_candidate_routing_smoke_spec"
    )
    assert decision["passes_no_model_readout"] is True
    assert decision["ready_for_model_heavy_candidate_smoke_spec"] is True
    assert decision["ready_for_tool_query"] is False
    assert decision["ready_for_dpo_rlvr"] is False
    assert decision["ready_for_hugging_face_publication"] is False
    assert decision["ready_for_release_tagging"] is False
    assert runtime["exact"] == 5
    assert runtime["rows"] == 5
    assert runtime["bridge_focus_exact"] == 4
    assert runtime["bridge_focus_rows"] == 4
    assert runtime["error_case_ids"] == []
    assert runtime["bridge_focus_error_case_ids"] == []


def test_saved_output_evidence_candidate_routing_readout_static_priors_stay_bounded() -> None:
    report = build_report()

    expected_bridge_exact = {
        "static_ground_supported": 0,
        "static_reject_contradicted": 1,
        "static_defer_insufficient": 1,
        "static_verify_insufficient": 1,
        "static_flag_invalid_value": 1,
    }
    for policy, bridge_exact in expected_bridge_exact.items():
        heldout = report["policies"][policy]["heldout"]
        assert heldout["exact"] == 1
        assert heldout["rows"] == 5
        assert heldout["bridge_focus_exact"] == bridge_exact
        assert heldout["bridge_focus_rows"] == 4

    assert report["decision"]["best_static_prior_heldout_exact"] == 1


def test_saved_output_evidence_candidate_routing_readout_is_public_safe() -> None:
    report = build_report()
    summary = report["row_manifest_summary"]

    assert summary["overlap_case_ids"] == []
    assert summary["overlap_split_groups"] == []
    assert summary["overlap_source_task_ids"] == []
    assert all(value is False for value in report["public_safety_contract"].values())
    assert "post_training/runs" not in json.dumps(report, sort_keys=True)


def test_saved_output_evidence_candidate_routing_readout_markdown() -> None:
    report = build_report()
    markdown = render_markdown(report)

    assert "Runtime evidence gate held-out exact: 5/5" in markdown
    assert "Runtime evidence gate bridge-focus exact: 4/4" in markdown
    assert "Best static prior held-out exact: 1/5" in markdown
    assert "Ready for DPO/RLVR: `False`" in markdown


def test_saved_output_evidence_candidate_routing_readout_cli_writes_outputs(
    tmp_path: Path,
) -> None:
    out_json = tmp_path / "readout.json"
    out_md = tmp_path / "READOUT.md"
    completed = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_readout.py",
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
    assert report["decision"]["passes_no_model_readout"] is True
    assert "Runtime evidence gate held-out exact: 5/5" in markdown
    assert "| `runtime_evidence_gate` | 5/5 | 4/4 |" in markdown
    assert "Ready for DPO/RLVR: `False`" in markdown
