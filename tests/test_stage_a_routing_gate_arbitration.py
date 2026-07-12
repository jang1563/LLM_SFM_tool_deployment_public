import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_routing_gate_arbitration import (
    build_arbitration_report,
    load_json,
)


ROOT = Path(__file__).resolve().parents[1]


def test_routing_gate_arbitration_compares_runtime_policies() -> None:
    report = build_arbitration_report(
        candidate_report=load_json(ROOT / "post_training" / "stage_a_routing_defer_verify_gate_trained_2026-07-08.json"),
        evidence_report=load_json(ROOT / "post_training" / "stage_a_routing_evidence_boundary_gate_2026-07-08.json"),
        candidate_report_path="candidate.json",
        evidence_report_path="evidence.json",
    )

    by_policy = report["summary"]["by_policy"]
    assert report["score_gap_threshold"] == 0.025
    assert by_policy["raw_candidate_top1"]["exact"] == 1
    assert by_policy["score_gap_fail_closed"]["exact"] == 2
    assert by_policy["evidence_boundary_override"]["exact"] == 2
    assert by_policy["hybrid_evidence_then_score_gap"]["exact"] == 2
    assert by_policy["raw_candidate_top1"]["error_case_ids"] == ["stage_a::000012"]


def test_routing_gate_arbitration_threshold_zero_matches_raw_top1_for_score_gap() -> None:
    report = build_arbitration_report(
        candidate_report=load_json(ROOT / "post_training" / "stage_a_routing_defer_verify_gate_trained_2026-07-08.json"),
        evidence_report=load_json(ROOT / "post_training" / "stage_a_routing_evidence_boundary_gate_2026-07-08.json"),
        candidate_report_path="candidate.json",
        evidence_report_path="evidence.json",
        score_gap_threshold=0.0,
    )

    by_policy = report["summary"]["by_policy"]
    assert by_policy["score_gap_fail_closed"]["exact"] == 1
    assert by_policy["score_gap_fail_closed"]["error_case_ids"] == ["stage_a::000012"]
    assert by_policy["evidence_boundary_override"]["exact"] == 2


def test_routing_gate_arbitration_cli_writes_public_safe_report(tmp_path: Path) -> None:
    out_json = tmp_path / "arbitration.json"
    out_md = tmp_path / "arbitration.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_routing_gate_arbitration.py",
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
    assert report["summary"]["by_policy"]["hybrid_evidence_then_score_gap"]["exact"] == 2
    text = out_md.read_text()
    assert "Stage A Routing Gate Arbitration" in text
    assert "prompt_messages" not in text
    assert "raw_output" not in text
