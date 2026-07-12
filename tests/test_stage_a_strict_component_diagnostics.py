import json
import subprocess
import sys
from pathlib import Path

from post_training.run_stage_a_sft_smoke_eval import load_manifest_rows
from post_training.run_stage_a_strict_component_diagnostics import build_component_report
from post_training.run_stage_a_strict_contract_sft_smoke import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


def component_report() -> dict:
    return build_component_report(
        manifest_rows=load_manifest_rows(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl"),
        heldout_rows=load_jsonl(ROOT / "post_training" / "stage_a_strict_contract_sft_heldout_v1.jsonl"),
        include_rows=True,
    )


def test_stage_a_strict_component_diagnostics_isolates_enum_failures() -> None:
    report = component_report()

    oracle = report["variants"]["oracle_full"]["summary"]
    invalid_enum = report["variants"]["invalid_enum_verified"]
    constrained = report["variants"]["enum_constrained_from_action"]["summary"]

    assert oracle["passed"] == 5
    assert oracle["mean_score"] == 1.0
    assert constrained["passed"] == 5
    assert constrained["mean_score"] == 1.0
    assert invalid_enum["summary"]["passed"] == 0
    assert invalid_enum["summary"]["violations"] == {"prediction_parse_error": 5}
    assert invalid_enum["parse_errors"] == {"Unknown evidence_status: 'verified'": 5}


def test_stage_a_strict_component_diagnostics_separates_tool_loop_from_routing() -> None:
    report = component_report()
    route_only = report["variants"]["route_only_correct_no_tools"]["summary"]
    names_only = report["variants"]["ordered_tool_names_only"]["summary"]
    tool_loop_wrong_route = report["variants"]["tool_loop_with_ground_route"]["summary"]

    assert route_only["gate_accuracy"]["evidence_status"] == 1.0
    assert route_only["gate_accuracy"]["terminal_action"] == 1.0
    assert route_only["gate_accuracy"]["required_tool_sequence"] == 0.0
    assert route_only["gate_accuracy"]["query_filter_completeness"] == 0.0
    assert route_only["violations"]["missing_required_tool_sequence"] == 5

    assert names_only["gate_accuracy"]["required_tool_sequence"] == 1.0
    assert names_only["gate_accuracy"]["query_filter_completeness"] == 0.0
    assert names_only["violations"] == {"query_filter_missing_required_field": 5}

    assert tool_loop_wrong_route["gate_accuracy"]["required_tool_sequence"] == 1.0
    assert tool_loop_wrong_route["gate_accuracy"]["query_filter_completeness"] == 1.0
    assert tool_loop_wrong_route["gate_accuracy"]["evidence_status"] < 1.0
    assert tool_loop_wrong_route["gate_accuracy"]["terminal_action"] < 1.0
    assert tool_loop_wrong_route["passed"] < 5


def test_stage_a_strict_component_diagnostics_cli_writes_compact_report(tmp_path: Path) -> None:
    out = tmp_path / "component_diagnostics.json"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_strict_component_diagnostics.py",
            "--compact",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text())
    assert report["dataset"] == "negbiodb_ct_stage_a_strict_component_diagnostics_v1"
    assert report["cases_expected"] == 5
    assert "rows" not in report["variants"]["oracle_full"]
    assert "Do not escalate to DPO or RLVR" in report["next_decision"]
