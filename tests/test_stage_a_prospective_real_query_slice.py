import json
from collections import Counter
from pathlib import Path

from negbiodb_ct.adapter import load_task_records
from negbiodb_ct.stage_a_manifest import (
    ideal_trajectory_from_stage_a_row,
    load_stage_a_manifest,
    stage_a_row_from_task_record,
)
from post_training.build_stage_a_prospective_real_query_slice import (
    PERTURBATIONS,
    build_base_rows,
    build_routing_rows,
    build_tool_query_rows,
    load_json,
    select_records,
    validate_rows,
    validate_sealed_separation_commitment,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "negbiodb_ct/tasks_pilot.jsonl"
EXISTING_MANIFEST = ROOT / "negbiodb_ct/stage_a_mini_manifest.jsonl"
SEALED_COMMITMENT = (
    ROOT / "post_training/stage_a_sealed_extension_commitment_2026-07-10.json"
)


def build_rows():
    tasks = load_task_records(TASKS)
    excluded = load_stage_a_manifest(EXISTING_MANIFEST)
    selected = select_records(
        tasks,
        excluded_rows=excluded,
        per_action=5,
        seed=20260723,
    )
    base = build_base_rows(selected, per_action=5, seed=20260723)
    tool_query = build_tool_query_rows(base)
    routing = build_routing_rows(base)
    return excluded, base, tool_query, routing


def test_canonical_stage_a_row_uses_visible_query_values_in_ideal_trajectory():
    record = load_task_records(TASKS, limit=1)[0]
    row = stage_a_row_from_task_record(
        record,
        case_index=0,
        include_query_values=True,
    )
    trajectory = ideal_trajectory_from_stage_a_row(row)

    query = row["model_visible_task"]["query"]
    assert query["drug_id"]["namespace"] == "negbiodb_ct.intervention_id"
    assert query["condition_id"]["namespace"] == "negbiodb_ct.condition_id"
    assert all(
        step.arguments
        == {
            "drug_id": query["drug_id"]["value"],
            "condition_id": query["condition_id"]["value"],
        }
        for step in trajectory.steps
    )


def test_prospective_selection_and_targets_are_case_specific_and_disjoint():
    excluded, base, tool_query, routing = build_rows()
    issues = validate_rows(
        base_rows=base,
        tool_query_rows=tool_query,
        routing_rows=routing,
        excluded_rows=excluded,
        per_action=5,
    )

    assert issues == []
    assert len(base) == 25
    assert len(tool_query) == 25
    assert len(routing) == 180
    assert len(
        {
            json.dumps(row["target_output"], sort_keys=True)
            for row in tool_query
        }
    ) == 25
    assert all(
        "<drug_id>" not in json.dumps(row)
        and "<condition_id>" not in json.dumps(row)
        for row in tool_query
    )


def test_routing_mutations_are_hidden_and_have_expected_shape():
    _, _, _, routing = build_rows()
    counts = Counter(
        row["hidden_eval_metadata"]["perturbation"] for row in routing
    )
    assert set(counts) == set(PERTURBATIONS)
    assert counts == {
        "clean": 25,
        "missing_attribution": 15,
        "stale_source": 15,
        "contradiction": 25,
        "invalid_numeric_value": 25,
        "partial_query": 25,
        "wrong_tool": 25,
        "unavailable_tool": 25,
    }

    by_perturbation = {}
    for row in routing:
        by_perturbation.setdefault(
            row["hidden_eval_metadata"]["perturbation"], row
        )
        visible = json.dumps(row["model_visible_task"], sort_keys=True)
        assert "hidden_eval_metadata" not in visible
        assert '"perturbation"' not in visible
        assert '"target_pair"' not in visible

    missing = by_perturbation["missing_attribution"]
    missing_records = missing["model_visible_task"]["observed_tool_loop"][0][
        "content"
    ]["same_indication_records"]
    assert missing_records
    assert all("source_id" not in record for record in missing_records)
    assert missing["target_pair"] == "verify/insufficient"

    partial = by_perturbation["partial_query"]
    assert all(
        "condition_id" not in result["arguments"]
        for result in partial["model_visible_task"]["observed_tool_loop"]
    )
    assert partial["target_pair"] == "defer/insufficient"

    wrong = by_perturbation["wrong_tool"]
    assert (
        wrong["model_visible_task"]["observed_tool_loop"][0]["name"]
        == "nullatlas_unapproved_lookup"
    )

    unavailable = by_perturbation["unavailable_tool"]
    verifier = unavailable["model_visible_task"]["observed_tool_loop"][1]
    assert verifier["content"]["error"] == "tool_unavailable"


def test_public_source_is_hash_matched_sealed_exclusion_without_sealed_rows():
    issues = validate_sealed_separation_commitment(
        load_json(SEALED_COMMITMENT),
        source_tasks_path=TASKS,
    )
    assert issues == []
