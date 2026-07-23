import json
from pathlib import Path

from post_training.evaluate_stage_a_tool_query_sft_smoke_result import (
    build_report,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "post_training/stage_a_strict_component_targets_train_v1.jsonl"
HELDOUT = ROOT / "post_training/stage_a_strict_component_targets_heldout_v1.jsonl"
RUNNER_DATASET = "negbiodb_ct_stage_a_strict_component_sft_smoke_v1"


def tool_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("component") == "tool_query"
    ]


def write_inputs(tmp_path: Path, *, echo_prompt: bool) -> tuple[Path, Path, Path]:
    rows = tool_rows(HELDOUT)
    run_id = "tool_query_test"
    predictions = []
    eval_rows = []
    for row in rows:
        if echo_prompt:
            raw_output = row["prompt_messages"][1]["content"]
            reward = {
                "target_keys": 0.0,
                "enum_validity": 1.0,
                "tool_query_shape": 0.0,
                "exact_match": 0.0,
            }
            violations = [
                "target_key_mismatch",
                "tool_query_shape_invalid",
                "target_mismatch",
            ]
        else:
            raw_output = json.dumps(row["target_output"], sort_keys=True)
            reward = {
                "target_keys": 1.0,
                "enum_validity": 1.0,
                "tool_query_shape": 1.0,
                "exact_match": 1.0,
            }
            violations = []
        predictions.append(
            {
                "id": f"{run_id}::{row['id']}",
                "source_component_target_id": row["id"],
                "raw_output": raw_output,
            }
        )
        eval_rows.append(
            {
                "passed": not violations,
                "score": sum(reward.values()) / len(reward),
                "reward_breakdown": reward,
                "violations": violations,
            }
        )

    counts = {}
    for row in eval_rows:
        for violation in row["violations"]:
            counts[violation] = counts.get(violation, 0) + 1
    gate_accuracy = {
        key: sum(row["reward_breakdown"][key] for row in eval_rows) / len(eval_rows)
        for key in eval_rows[0]["reward_breakdown"]
    }
    summary = {
        "cases": len(eval_rows),
        "passed": sum(row["passed"] for row in eval_rows),
        "mean_score": sum(row["score"] for row in eval_rows) / len(eval_rows),
        "gate_accuracy": gate_accuracy,
        "violations": counts,
    }
    run_report = {
        "dataset": RUNNER_DATASET,
        "run_id": run_id,
        "component": "tool_query",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "device": "cuda",
        "train_examples": 20,
        "heldout_examples": 5,
        "max_steps": 20,
        "batch_size": 1,
        "max_length": 1024,
        "max_new_tokens": 256,
        "decode_mode": "freeform",
        "train_last_layers": 1,
        "trainable_params": 1,
        "losses": [0.4, 0.2],
        "loss_delta": -0.2,
    }
    eval_report = {
        "dataset": RUNNER_DATASET,
        "run_id": run_id,
        "component": "tool_query",
        "summary": summary,
    }
    run_path = tmp_path / "report.json"
    eval_path = tmp_path / "eval_report.json"
    predictions_path = tmp_path / "predictions.jsonl"
    run_path.write_text(json.dumps(run_report))
    eval_path.write_text(json.dumps(eval_report))
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )
    return run_path, eval_path, predictions_path


def test_echo_failure_is_compact_and_scope_limited(tmp_path):
    run_path, eval_path, predictions_path = write_inputs(tmp_path, echo_prompt=True)
    report = build_report(
        run_report_path=run_path,
        eval_report_path=eval_path,
        predictions_path=predictions_path,
        train_targets_path=TRAIN,
        heldout_targets_path=HELDOUT,
    )

    assert report["heldout_result"]["passed"] == 0
    assert report["output_behavior"]["prompt_payload_echoes"] == 5
    assert report["output_behavior"]["prompt_field_copies"] == 5
    assert report["output_behavior"]["prompt_schema_outputs"] == 5
    assert report["acceptance_gate"]["passes"] is False
    assert report["experiment_scope"]["unique_train_targets"] == 1
    assert report["experiment_scope"]["unique_heldout_targets"] == 1
    assert report["experiment_scope"]["actual_identifier_resolution_evaluated"] is False
    assert report["public_safety_contract"]["raw_model_text_emitted"] is False


def test_exact_outputs_pass_schema_gate(tmp_path):
    run_path, eval_path, predictions_path = write_inputs(tmp_path, echo_prompt=False)
    report = build_report(
        run_report_path=run_path,
        eval_report_path=eval_path,
        predictions_path=predictions_path,
        train_targets_path=TRAIN,
        heldout_targets_path=HELDOUT,
    )

    assert report["heldout_result"]["passed"] == 5
    assert report["output_behavior"]["tool_calls_key_present"] == 5
    assert report["acceptance_gate"]["passes"] is True


def test_rendered_artifacts_do_not_copy_raw_predictions(tmp_path):
    run_path, eval_path, predictions_path = write_inputs(tmp_path, echo_prompt=True)
    report = build_report(
        run_report_path=run_path,
        eval_report_path=eval_path,
        predictions_path=predictions_path,
        train_targets_path=TRAIN,
        heldout_targets_path=HELDOUT,
    )
    rendered = json.dumps(report, sort_keys=True) + render_markdown(report)

    first_raw = json.loads(predictions_path.read_text().splitlines()[0])["raw_output"]
    assert first_raw not in rendered
    assert "stage_a::000001" not in rendered
    assert str(tmp_path) not in rendered
