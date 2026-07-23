import copy
import json
from pathlib import Path

from post_training.build_stage_a_prospective_tool_query_transfer_freeze import (
    build_freeze,
)
from post_training.build_stage_a_prospective_tool_query_prompt_repair_freeze import (
    build_freeze as build_prompt_repair_freeze,
)
from post_training.evaluate_stage_a_prospective_runtime_hybrid import (
    load_json,
    sha256_file,
)
from post_training.run_stage_a_prospective_tool_query_prompt_repair import (
    EXPLICIT_SYSTEM_PROMPT,
    render_markdown as render_prompt_repair_markdown,
    validate_inputs as validate_prompt_repair_inputs,
)
from post_training.run_stage_a_prospective_tool_query_transfer import (
    render_markdown,
    score_prediction,
    summarize_policy,
    validate_inputs,
)
from post_training.run_stage_a_strict_contract_sft_smoke import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
ROWS = (
    ROOT / "post_training/stage_a_prospective_real_query_tool_query_v1.jsonl"
)
MANIFEST = (
    ROOT
    / "post_training/stage_a_prospective_real_query_experiment_manifest.json"
)
FREEZE = (
    ROOT
    / "post_training/stage_a_prospective_tool_query_transfer_freeze_2026-07-23.json"
)
PROMPT_REPAIR_FREEZE = (
    ROOT
    / "post_training/"
    "stage_a_prospective_tool_query_prompt_repair_freeze_2026-07-23.json"
)
SOURCE_RESULT = (
    ROOT
    / "post_training/stage_a_tool_query_sft_smoke_result_qwen05b_cayuga_2026-07-23.json"
)


def test_transfer_freeze_is_deterministic_and_prohibits_retraining():
    state_sha256 = "5" * 64
    kwargs = {
        "state_sha256": state_sha256,
        "model_revision": "test-revision",
        "source_result_path": SOURCE_RESULT,
        "query_rows_path": ROWS,
        "experiment_manifest_path": MANIFEST,
    }

    first = build_freeze(**kwargs)
    second = build_freeze(**kwargs)

    assert first == second
    assert first["authorization"] == {
        "ready_for_base_vs_frozen_transfer": True,
        "training_on_prospective_rows_allowed": False,
        "retraining_before_transfer_allowed": False,
        "dpo_rlvr_allowed": False,
    }
    assert first["scientific_boundary"]["state_trained_before_prospective_rows"]
    assert first["frozen_artifacts"]["trainable_state"]["sha256"] == state_sha256


def test_transfer_inputs_validate_with_hash_matched_external_state(tmp_path):
    rows = load_jsonl(ROWS)
    manifest = load_json(MANIFEST)
    freeze = copy.deepcopy(load_json(FREEZE))
    state = tmp_path / "trainable_state.pt"
    state.write_bytes(b"test pre-prospective tool-query state")
    freeze["frozen_artifacts"]["trainable_state"]["sha256"] = sha256_file(state)

    issues = validate_inputs(
        rows=rows,
        rows_path=ROWS,
        manifest=manifest,
        freeze=freeze,
        trainable_state_path=state,
    )

    assert issues == []


def test_exact_case_specific_tool_query_scores_all_contract_dimensions():
    row = load_jsonl(ROWS)[0]
    prediction = {
        "source_row_id": row["id"],
        "policy": "base",
        "raw_output": json.dumps(row["target_output"], sort_keys=True),
    }

    score = score_prediction(row, prediction)

    assert score == {
        "parseable_json": True,
        "target_keys": True,
        "tool_query_shape": True,
        "tool_sequence": True,
        "query_fields": True,
        "query_values": True,
        "exact": True,
        "violations": [],
    }


def test_wrong_tool_and_partial_query_fail_for_intended_reasons():
    row = load_jsonl(ROWS)[0]
    output = copy.deepcopy(row["target_output"])
    output["tool_calls"][0]["name"] = "wrong_tool"
    output["tool_calls"][1]["arguments"].pop("condition_id")
    prediction = {
        "source_row_id": row["id"],
        "policy": "base",
        "raw_output": json.dumps(output, sort_keys=True),
    }

    score = score_prediction(row, prediction)

    assert score["parseable_json"]
    assert score["tool_query_shape"]
    assert not score["tool_sequence"]
    assert not score["query_fields"]
    assert not score["query_values"]
    assert not score["exact"]
    assert set(score["violations"]) == {
        "tool_sequence_mismatch",
        "query_fields_mismatch",
        "query_values_mismatch",
        "target_mismatch",
    }


def test_policy_summary_contains_only_aggregate_counts():
    rows = load_jsonl(ROWS)[:2]
    predictions = [
        {
            "source_row_id": row["id"],
            "policy": "base",
            "raw_output": json.dumps(row["target_output"], sort_keys=True),
        }
        for row in rows
    ]

    summary = summarize_policy(rows, predictions, policy="base")

    assert summary["rows"] == 2
    assert summary["exact"] == 2
    assert summary["query_values"] == 2
    assert set(summary) == {
        "rows",
        "parseable_json",
        "target_keys",
        "tool_query_shape",
        "tool_sequence",
        "query_fields",
        "query_values",
        "exact",
        "accuracy",
        "violations",
        "unique_raw_outputs",
    }
    assert rows[0]["id"] not in json.dumps(summary)


def test_public_markdown_exposes_only_aggregate_policy_results():
    policy = {
        "rows": 25,
        "parseable_json": 10,
        "tool_sequence": 5,
        "query_fields": 4,
        "query_values": 3,
        "exact": 2,
    }
    report = {
        "policies": {
            "base": policy,
            "frozen_tool_query_sft": policy,
        },
        "decision": {"frozen_sft_improves_exact": False},
    }

    rendered = render_markdown(report)

    assert "`base`" in rendered
    assert "2/25" in rendered
    assert "Raw generations" in rendered
    assert "raw_output" not in rendered


def test_prompt_repair_freeze_is_adaptive_and_prohibits_training():
    freeze = build_prompt_repair_freeze(
        model_id="test/model",
        model_revision="test-revision",
        max_new_tokens=64,
        query_rows_path=ROWS,
        experiment_manifest_path=MANIFEST,
        observed_transfer_result_path=(
            ROOT
            / "post_training/"
            "stage_a_prospective_tool_query_transfer_result_qwen05b_cayuga_2026-07-23.json"
        ),
    )

    assert freeze["policy"]["system_prompt"] == EXPLICIT_SYSTEM_PROMPT
    assert freeze["policy"]["training_allowed"] is False
    assert freeze["scientific_boundary"][
        "adaptive_after_observing_input_echo_failure"
    ]


def test_prompt_repair_inputs_validate_against_frozen_artifacts():
    rows = load_jsonl(ROWS)
    issues = validate_prompt_repair_inputs(
        rows=rows,
        rows_path=ROWS,
        manifest=load_json(MANIFEST),
        freeze=load_json(PROMPT_REPAIR_FREEZE),
    )

    assert issues == []


def test_prompt_repair_markdown_marks_adaptive_boundary():
    policy = {
        "rows": 25,
        "parseable_json": 25,
        "tool_sequence": 20,
        "query_fields": 20,
        "query_values": 18,
        "exact": 18,
    }
    report = {
        "policy": policy,
        "decision": {"prompt_contract_resolves_transfer": False},
    }

    rendered = render_prompt_repair_markdown(report)

    assert "adaptive public-development diagnostic" in rendered
    assert "18/25" in rendered
    assert "not a sealed-test estimate" in rendered
