#!/usr/bin/env python3
"""Create a compact public-safe Stage A tool-query smoke result.

The adapter may read private/ignored prediction JSONL on the cluster, but it
emits aggregate behavior only. Raw model text, prompts, row identifiers,
scheduler logs, model state, and private paths are never copied into the
result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATASET = "negbiodb_ct_stage_a_tool_query_sft_smoke_result_v1"
RUNNER_DATASET = "negbiodb_ct_stage_a_strict_component_sft_smoke_v1"
COMPONENT = "tool_query"
REQUIRED_FIELDS = ("drug_id", "condition_id")
EXPECTED_TOOL_SEQUENCE = (
    "nullatlas_survey_prior_failures",
    "nullatlas_verify_trial_claims",
    "nullatlas_check_value_validity",
    "nullatlas_negative_evidence_completeness",
)


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(payload)
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str | Path, *, role: str, private: bool) -> dict[str, str]:
    resolved = Path(path).resolve()
    if private:
        display = f"private_run_input::{resolved.name}"
    else:
        try:
            display = resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            display = f"external_public_input::{resolved.name}"
    return {"path": display, "role": role, "sha256": sha256_file(resolved)}


def component_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("component") == COMPONENT]


def target_output(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("target_output")
    if not isinstance(value, Mapping):
        raise ValueError(f"tool-query row lacks target_output: {row.get('id')!r}")
    return dict(value)


def target_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(target_output(row), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def placeholder_target_ok(row: Mapping[str, Any]) -> bool:
    calls = target_output(row).get("tool_calls")
    if not isinstance(calls, list) or len(calls) != len(EXPECTED_TOOL_SEQUENCE):
        return False
    names: list[str] = []
    for call in calls:
        if not isinstance(call, Mapping):
            return False
        names.append(str(call.get("name")))
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            return False
        if set(arguments) != set(REQUIRED_FIELDS):
            return False
        if any(arguments[field] != f"<{field}>" for field in REQUIRED_FIELDS):
            return False
    return tuple(names) == EXPECTED_TOOL_SEQUENCE


def prompt_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    messages = row.get("prompt_messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user = messages[1]
    if not isinstance(user, Mapping):
        return None
    try:
        payload = json.loads(str(user.get("content", "")))
    except json.JSONDecodeError:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def parse_raw_output(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, str):
        return None, "not_string"
    text = value.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, "json_parse_error"
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    return parsed, None


def output_behavior(
    heldout_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_by_id = {str(row["id"]): row for row in heldout_rows}
    parse_errors: Counter[str] = Counter()
    keysets: Counter[str] = Counter()
    parsed_objects = 0
    prompt_echoes = 0
    prompt_field_copies = 0
    prompt_schema_outputs = 0
    tool_calls_key_present = 0
    unique_output_hashes: set[str] = set()

    for prediction in prediction_rows:
        source_id = str(
            prediction.get("source_component_target_id") or prediction.get("id")
        )
        raw = prediction.get("raw_output")
        if isinstance(raw, str):
            unique_output_hashes.add(hashlib.sha256(raw.encode()).hexdigest())
        parsed, error = parse_raw_output(raw)
        if error:
            parse_errors[error] += 1
            continue
        assert parsed is not None
        parsed_objects += 1
        keysets["|".join(sorted(str(key) for key in parsed))] += 1
        if "tool_calls" in parsed:
            tool_calls_key_present += 1
        expected = expected_by_id.get(source_id)
        if expected is not None:
            visible_prompt = prompt_payload(expected)
            if parsed == visible_prompt:
                prompt_echoes += 1
            if (
                visible_prompt is not None
                and set(parsed).issubset(visible_prompt)
                and {"component", "input_id", "claim", "allowed_tools"}.issubset(parsed)
            ):
                prompt_schema_outputs += 1
            if (
                visible_prompt is not None
                and set(parsed).issubset(visible_prompt)
                and {"component", "input_id", "claim", "allowed_tools"}.issubset(parsed)
                and all(parsed[key] == visible_prompt[key] for key in parsed)
            ):
                prompt_field_copies += 1

    return {
        "rows": len(prediction_rows),
        "parseable_json_objects": parsed_objects,
        "parse_errors": dict(sorted(parse_errors.items())),
        "prompt_payload_echoes": prompt_echoes,
        "prompt_field_copies": prompt_field_copies,
        "prompt_schema_outputs": prompt_schema_outputs,
        "tool_calls_key_present": tool_calls_key_present,
        "unique_raw_outputs": len(unique_output_hashes),
        "top_level_keyset_counts": dict(sorted(keysets.items())),
    }


def as_summary(eval_report: Mapping[str, Any]) -> dict[str, Any]:
    summary = eval_report.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("eval report lacks summary")
    return {
        "cases": summary.get("cases"),
        "passed": summary.get("passed"),
        "mean_score": summary.get("mean_score"),
        "gate_accuracy": dict(summary.get("gate_accuracy", {})),
        "violations": dict(summary.get("violations", {})),
    }


def training_summary(run_report: Mapping[str, Any]) -> dict[str, Any]:
    losses = run_report.get("losses")
    if not isinstance(losses, list) or not losses:
        raise ValueError("run report lacks non-empty losses")
    return {
        "model": run_report.get("model"),
        "device": run_report.get("device"),
        "train_examples": run_report.get("train_examples"),
        "heldout_examples": run_report.get("heldout_examples"),
        "max_steps": run_report.get("max_steps"),
        "batch_size": run_report.get("batch_size"),
        "max_length": run_report.get("max_length"),
        "max_new_tokens": run_report.get("max_new_tokens"),
        "decode_mode": run_report.get("decode_mode"),
        "train_last_layers": run_report.get("train_last_layers"),
        "trainable_params": run_report.get("trainable_params"),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_delta": run_report.get("loss_delta"),
    }


def gate_violations(result: Mapping[str, Any], expected_cases: int) -> list[str]:
    violations: list[str] = []
    gate_accuracy = result.get("gate_accuracy", {})
    if result.get("cases") != expected_cases:
        violations.append("heldout_case_count_mismatch")
    if result.get("passed") != expected_cases:
        violations.append("below_exact_pass_requirement")
    for gate in ("target_keys", "tool_query_shape", "exact_match"):
        if gate_accuracy.get(gate) != 1.0:
            violations.append(f"{gate}_below_requirement")
    return violations


def build_report(
    *,
    run_report_path: str | Path,
    eval_report_path: str | Path,
    predictions_path: str | Path,
    train_targets_path: str | Path,
    heldout_targets_path: str | Path,
) -> dict[str, Any]:
    run_report = load_json(run_report_path)
    eval_report = load_json(eval_report_path)
    predictions = load_jsonl(predictions_path)
    train_rows = component_rows(load_jsonl(train_targets_path))
    heldout_rows = component_rows(load_jsonl(heldout_targets_path))

    if run_report.get("dataset") != RUNNER_DATASET:
        raise ValueError("run report dataset mismatch")
    if eval_report.get("dataset") != RUNNER_DATASET:
        raise ValueError("eval report dataset mismatch")
    if run_report.get("component") != COMPONENT or eval_report.get("component") != COMPONENT:
        raise ValueError("tool-query component mismatch")
    if run_report.get("run_id") != eval_report.get("run_id"):
        raise ValueError("run_id mismatch")

    train_hashes = {target_hash(row) for row in train_rows}
    heldout_hashes = {target_hash(row) for row in heldout_rows}
    result = as_summary(eval_report)
    violations = gate_violations(result, len(heldout_rows))
    behavior = output_behavior(heldout_rows, predictions)

    return {
        "dataset": DATASET,
        "run_id": run_report.get("run_id"),
        "component": COMPONENT,
        "experiment_scope": {
            "diagnostic": "ordered_tool_call_and_placeholder_schema_compliance",
            "actual_identifier_resolution_evaluated": False,
            "actual_tool_execution_evaluated": False,
            "unique_train_targets": len(train_hashes),
            "unique_heldout_targets": len(heldout_hashes),
            "shared_target_hashes": len(train_hashes & heldout_hashes),
            "all_targets_use_expected_placeholder_sequence": all(
                placeholder_target_ok(row) for row in train_rows + heldout_rows
            ),
        },
        "training": training_summary(run_report),
        "heldout_result": result,
        "output_behavior": behavior,
        "acceptance_gate": {
            "required_cases": len(heldout_rows),
            "required_passed": len(heldout_rows),
            "required_gate_accuracy": {
                "target_keys": 1.0,
                "tool_query_shape": 1.0,
                "exact_match": 1.0,
            },
            "passes": not violations,
            "violations": violations,
        },
        "decision": {
            "diagnostic_complete": True,
            "tool_query_schema_gate_passed": not violations,
            "ready_for_dpo_rlvr": False,
            "ready_for_hugging_face_publication": False,
            "selected_next_step": (
                "freeze_candidate_routing_policy_then_run_one_time_sealed_routing_evaluation"
            ),
            "interpretation": (
                "The tool-query slice measures one shared placeholder output "
                "shape. It is complete as a schema diagnostic but cannot support "
                "a claim about resolving real query identifiers."
            ),
        },
        "input_artifacts": {
            "run_report": artifact(
                run_report_path, role="private compact training report", private=True
            ),
            "eval_report": artifact(
                eval_report_path, role="private held-out eval report", private=True
            ),
            "predictions": artifact(
                predictions_path, role="private raw prediction JSONL", private=True
            ),
            "train_targets": artifact(
                train_targets_path, role="tracked train component targets", private=False
            ),
            "heldout_targets": artifact(
                heldout_targets_path,
                role="tracked held-out component targets",
                private=False,
            ),
        },
        "public_safety_contract": {
            "private_raw_predictions_read": True,
            "raw_model_text_emitted": False,
            "prompt_text_emitted": False,
            "row_identifiers_emitted": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "private_paths_redacted": True,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    result = report["heldout_result"]
    gate = report["acceptance_gate"]
    scope = report["experiment_scope"]
    behavior = report["output_behavior"]
    training = report["training"]
    lines = [
        "# Stage A Tool-Query Component SFT Result",
        "",
        "Purpose: close the missing Stage A tool-query component checkpoint with",
        "a compact public-safe Cayuga result.",
        "",
        "## Result",
        "",
        f"- Model: `{training['model']}`",
        f"- Held-out pass: {result['passed']}/{result['cases']}",
        f"- Mean score: {result['mean_score']}",
        f"- Prompt-schema outputs: {behavior['prompt_schema_outputs']}/{behavior['rows']}",
        f"- Tool-call key present: {behavior['tool_calls_key_present']}/{behavior['rows']}",
        f"- Passes schema gate: `{gate['passes']}`",
        f"- Gate violations: `{json.dumps(gate['violations'], sort_keys=True)}`",
        "",
        "## Scope",
        "",
        f"- Unique train targets: {scope['unique_train_targets']}",
        f"- Unique held-out targets: {scope['unique_heldout_targets']}",
        f"- Shared target hashes: {scope['shared_target_hashes']}",
        "- Actual identifier resolution evaluated: `False`",
        "- Actual tool execution evaluated: `False`",
        "",
        "This is an ordered tool-call and placeholder-schema diagnostic. It does",
        "not measure drug/condition identifier resolution or live tool execution.",
        "Raw predictions, prompts, model state, scheduler logs, and private paths",
        "are not included in this artifact.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", required=True)
    parser.add_argument("--eval-report", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--train-targets",
        default="post_training/stage_a_strict_component_targets_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-targets",
        default="post_training/stage_a_strict_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = build_report(
        run_report_path=args.run_report,
        eval_report_path=args.eval_report,
        predictions_path=args.predictions,
        train_targets_path=args.train_targets,
        heldout_targets_path=args.heldout_targets,
    )
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
