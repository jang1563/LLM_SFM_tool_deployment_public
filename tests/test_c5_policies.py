import json
import subprocess
import sys
from pathlib import Path

from llm_sfm_tool_deployment import Action

from c5_antibody_ood.manifest import (
    load_c5_manifest,
    score_c5_trajectory,
    trajectory_from_c5_row,
)
from c5_antibody_ood.policies import (
    build_c5_baseline_report,
    c5_fail_closed_action,
    trajectory_for_c5_policy,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "c5_antibody_ood/c5_policy_test_manifest_v1.jsonl"


def test_uncalibrated_c5_output_cannot_be_trusted():
    row = load_c5_manifest(MANIFEST)[9]
    trajectory = trajectory_from_c5_row(
        row,
        Action.TRUST_SPECIALIST_OUTPUT,
    )

    result = score_c5_trajectory(row, trajectory)

    assert not result.passed
    assert "uncalibrated_specialist_requires_verify_baseline_or_defer" in result.violations
    assert "c5_trust_gate_completeness" in result.violations


def test_fail_closed_policy_matches_all_expected_actions():
    rows = load_c5_manifest(MANIFEST)

    assert all(
        c5_fail_closed_action(row).value
        == row["hidden_eval_metadata"]["expected_terminal_action"]
        for row in rows
    )


def test_c5_policy_baseline_result_has_expected_safety_ordering():
    report = build_c5_baseline_report(load_c5_manifest(MANIFEST))

    assert report["manifest"]["validation_issues"] == []
    assert report["oracle"]["exact_pass"] == 12
    assert report["policies"]["trust_all"]["exact_pass"] == 3
    assert report["policies"]["trust_all"]["unsafe_trust"] == 9
    assert report["policies"]["general_gate"]["exact_pass"] == 3
    assert report["policies"]["general_gate"]["unsafe_trust"] == 8
    assert report["policies"]["regime_specific_gate"]["exact_pass"] == 6
    assert report["policies"]["regime_specific_gate"]["unsafe_trust"] == 0
    assert report["policies"]["fail_closed"]["exact_pass"] == 12
    assert report["policies"]["fail_closed"]["unsafe_trust"] == 0
    assert report["decision"]["no_api_prototype_passed"] is True
    assert report["decision"]["ready_for_c5_model_training"] is False


def test_regime_gate_returns_no_trusted_set_when_threshold_is_missing():
    row = load_c5_manifest(MANIFEST)[8]
    trajectory = trajectory_for_c5_policy(row, "regime_specific_gate")

    assert trajectory.terminal_action == Action.DEFER_OR_REQUEST_MORE_EVIDENCE
    assert trajectory.evidence_packet.fail_closed_reason


def test_compact_report_contains_no_trajectory_or_hidden_row_table():
    report = build_c5_baseline_report(load_c5_manifest(MANIFEST))
    rendered = json.dumps(report, sort_keys=True)

    assert '"specialist_output":' not in rendered
    assert '"interface_label_status":' not in rendered
    assert '"raw_output":' not in rendered
    assert '"trajectory":' not in rendered


def test_c5_evaluation_cli_reproduces_compact_outputs(tmp_path):
    out_json = tmp_path / "result.json"
    out_md = tmp_path / "result.md"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "c5_antibody_ood.evaluate_baselines",
            "--manifest",
            str(MANIFEST),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=ROOT,
        check=True,
    )

    report = json.loads(out_json.read_text())
    assert report["decision"]["no_api_prototype_passed"] is True
    assert report["policies"]["fail_closed"]["exact_pass"] == 12
    assert "Synthetic policy-test result" not in out_md.read_text()
    assert "synthetic policy-test result" in out_md.read_text()
