import json
from pathlib import Path

from post_training.analyze_stage_a_component_visibility import (
    build_visibility_report,
    load_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def test_component_visibility_audit_marks_current_enum_and_routing_underconditioned() -> None:
    targets_path = ROOT / "post_training" / "stage_a_strict_component_targets_v1.jsonl"
    report = build_visibility_report(
        load_jsonl(targets_path),
        targets_path=targets_path,
        run_id="unit_visibility_audit",
    )

    assert report["summary"]["hidden_label_leak_rows"] == 0
    assert report["by_component"]["enum_action"]["underdetermined_evidence_routing"] == 25
    assert report["by_component"]["routing_after_loop"]["underdetermined_evidence_routing"] == 25
    assert report["by_component"]["tool_query"]["underdetermined_evidence_routing"] == 0
    assert report["by_component"]["routing_after_loop"]["has_observed_tool_loop"] == 25
    assert report["by_component"]["routing_after_loop"]["observed_tool_loop_has_tool_results"] == 0
    assert report["summary"]["components_with_underdetermined_routing"] == [
        "enum_action",
        "routing_after_loop",
    ]


def test_component_visibility_audit_detects_tool_result_conditioning(tmp_path: Path) -> None:
    row = {
        "id": "unit::routing",
        "component": "routing_after_loop",
        "case_family": "unit",
        "split": "heldout",
        "target_keys": ["action", "evidence_status", "cited_source_ids"],
        "target_output": {"action": "ground", "evidence_status": "supported"},
        "prompt_messages": [
            {"role": "system", "content": "Return JSON."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "component": "routing_after_loop",
                        "claim": "Synthetic claim",
                        "observed_tool_loop": [
                            {
                                "name": "toy_tool",
                                "arguments": {},
                                "content": [{"endpoint_met": 0, "p_value": 0.1}],
                            }
                        ],
                    },
                    sort_keys=True,
                ),
            },
        ],
    }
    path = tmp_path / "targets.jsonl"
    path.write_text(json.dumps(row, sort_keys=True) + "\n")

    report = build_visibility_report([row], targets_path=path, run_id="unit_visible_results")

    assert report["by_component"]["routing_after_loop"]["has_evidence_for_routing"] == 1
    assert report["by_component"]["routing_after_loop"]["underdetermined_evidence_routing"] == 0
