import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_routing_model_readiness import build_report, load_json


ROOT = Path(__file__).resolve().parents[1]


def report() -> dict:
    return build_report(
        baseline=load_json(ROOT / "post_training" / "stage_a_routing_gate_baseline_comparison_2026-07-09.json"),
        freeform=load_json(ROOT / "post_training" / "stage_a_evidence_routing_after_loop_cayuga_summary_2026-07-06.json"),
        constrained=load_json(
            ROOT / "post_training" / "stage_a_evidence_routing_observed_pair_cayuga_summary_2026-07-08.json"
        ),
        contrast_candidate=load_json(
            ROOT / "post_training" / "stage_a_routing_contrast_candidate_cayuga_summary_2026-07-08.json"
        ),
        defer_verify=load_json(ROOT / "post_training" / "stage_a_routing_defer_verify_cayuga_summary_2026-07-08.json"),
        fail_closed_gate=load_json(
            ROOT / "post_training" / "stage_a_routing_defer_verify_gate_trained_2026-07-08.json"
        ),
    )


def test_model_readiness_keeps_escalation_gated() -> None:
    data = report()
    decision = data["decision"]

    assert decision["ready_for_tool_query"] is False
    assert decision["ready_for_dpo_rlvr"] is False
    assert decision["ready_for_hugging_face_publication"] is False
    assert decision["ready_for_release_tagging"] is False
    assert decision["runtime_enforcement_required"] is True
    assert "best all-family model readout is below runtime evidence gate at 5/5" in decision["blockers"]


def test_constrained_routing_beats_collapse_but_not_citationless_or_runtime_gate() -> None:
    data = report()
    constrained = data["all_family_model_results"]["constrained_routing_observed_pair"]
    readiness = constrained["readiness"]

    assert constrained["heldout_summary"]["passed"] == 2
    assert readiness["beats_ground_supported_collapse"] is True
    assert readiness["beats_citationless_routing"] is False
    assert readiness["competitive_with_runtime_gate"] is False
    assert readiness["ready_for_escalation"] is False


def test_targeted_fail_closed_gate_is_not_all_family_readiness_evidence() -> None:
    data = report()
    targeted = data["targeted_model_results"]
    fail_closed = targeted["defer_verify_fail_closed_gate_subset"]

    assert targeted["routing_contrast_candidate_subset"]["trained_exact_top1"] == 2
    assert targeted["defer_verify_candidate_subset"]["trained_exact_top1"] == 1
    assert fail_closed["strict_final_correct"] == 2
    assert fail_closed["all_family_readiness_evidence"] is False


def test_model_readiness_cli_writes_public_safe_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "model_readiness.json"
    out_md = tmp_path / "model_readiness.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_routing_model_readiness.py",
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
    data = json.loads(out_json.read_text())
    assert data["decision"]["ready_for_tool_query"] is False
    text = out_md.read_text()
    assert "Stage A Routing Model Readiness" in text
    assert "hidden_eval_metadata" not in text
    assert "post_training/runs/" not in text
