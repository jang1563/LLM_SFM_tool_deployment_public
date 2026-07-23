#!/usr/bin/env python3
"""Compare base and frozen tool-query SFT policies on real query identifiers.

The run performs no training. It scores the base model first, applies the
pre-prospective hashed tool-query trainable state, then scores the same 25
public development prompts. Raw generations stay private; only aggregate
schema and query-value metrics are suitable for public release.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.build_stage_a_prospective_real_query_slice import (  # noqa: E402
    MANIFEST_DATASET,
    REQUIRED_QUERY_FIELDS,
    TOOL_QUERY_DATASET,
)
from post_training.build_stage_a_sealed_extension import (  # noqa: E402
    require_external_private_path,
)
from post_training.evaluate_stage_a_prospective_runtime_hybrid import (  # noqa: E402
    load_json,
    sha256_file,
)
from post_training.generate_stage_a_predictions import (  # noqa: E402
    disable_transformers_torchvision_probe,
)
from post_training.run_stage_a_prospective_frozen_policy import (  # noqa: E402
    private_output_path,
)
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    parse_component_output,
)
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    choose_device,
    load_jsonl,
    prompt_text_for_tokenizer,
    set_trainable_last_layers,
)


DATASET = "negbiodb_ct_stage_a_prospective_tool_query_transfer_run_v1"
PREDICTION_DATASET = (
    "negbiodb_ct_stage_a_prospective_tool_query_transfer_private_predictions_v1"
)
FREEZE_DATASET = "negbiodb_ct_stage_a_prospective_tool_query_transfer_freeze_v1"
POLICIES = ("base", "frozen_tool_query_sft")
SYSTEM_PROMPT = (
    "You are a Stage A component agent. Return exactly one JSON "
    "object for the requested component. Do not include prose, "
    "markdown, evaluator metadata, or hidden labels."
)


def prompt_messages(
    row: Mapping[str, Any],
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    task = row.get("model_visible_task")
    if not isinstance(task, Mapping):
        raise ValueError(f"{row.get('id')} lacks model_visible_task")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(dict(task), sort_keys=True)},
    ]


def expected_arguments(row: Mapping[str, Any]) -> dict[str, Any]:
    task = row["model_visible_task"]
    query = task.get("query")
    if not isinstance(query, Mapping):
        raise ValueError(f"{row.get('id')} lacks visible query")
    out: dict[str, Any] = {}
    for field in REQUIRED_QUERY_FIELDS:
        payload = query.get(field)
        if not isinstance(payload, Mapping) or "value" not in payload:
            raise ValueError(f"{row.get('id')} lacks visible {field}")
        out[field] = payload["value"]
    return out


def target_tool_calls(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    target = row.get("target_output")
    calls = target.get("tool_calls") if isinstance(target, Mapping) else None
    if not isinstance(calls, list):
        raise ValueError(f"{row.get('id')} lacks target tool_calls")
    return [dict(call) for call in calls if isinstance(call, Mapping)]


def validate_inputs(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_path: str | Path,
    manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    trainable_state_path: str | Path,
) -> list[str]:
    issues: list[str] = []
    require_external_private_path(
        trainable_state_path,
        role="frozen tool-query trainable state",
    )
    if manifest.get("dataset") != MANIFEST_DATASET:
        issues.append("unexpected_experiment_manifest_dataset")
    expected_rows = manifest.get("artifacts", {}).get("tool_query_rows", {})
    if expected_rows.get("sha256") != sha256_file(rows_path):
        issues.append("tool_query_rows_sha256_mismatch")
    if expected_rows.get("records") != len(rows):
        issues.append("tool_query_rows_record_count_mismatch")
    if freeze.get("dataset") != FREEZE_DATASET:
        issues.append("unexpected_transfer_freeze_dataset")
    frozen_rows = freeze.get("frozen_artifacts", {}).get("tool_query_rows", {})
    if frozen_rows.get("sha256") != sha256_file(rows_path):
        issues.append("transfer_freeze_rows_sha256_mismatch")
    state = freeze.get("frozen_artifacts", {}).get("trainable_state", {})
    if state.get("sha256") != sha256_file(trainable_state_path):
        issues.append("trainable_state_sha256_mismatch")
    policy = freeze.get("policy", {})
    if policy.get("retraining_allowed") is not False:
        issues.append("transfer_freeze_does_not_prohibit_retraining")
    if policy.get("decode_mode") != "freeform":
        issues.append("unexpected_transfer_decode_mode")

    seen_ids: set[str] = set()
    target_hashes: set[str] = set()
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in seen_ids:
            issues.append(f"{row_id}:duplicate_row_id")
        seen_ids.add(row_id)
        if row.get("dataset") != TOOL_QUERY_DATASET:
            issues.append(f"{row_id}:unexpected_dataset")
        visible = json.dumps(prompt_messages(row), sort_keys=True)
        for forbidden in (
            "hidden_eval_metadata",
            "source_task_id",
            "split_group",
            '"target_output"',
        ):
            if forbidden in visible:
                issues.append(f"{row_id}:prompt_leak:{forbidden}")
        arguments = expected_arguments(row)
        calls = target_tool_calls(row)
        if len(calls) != 4:
            issues.append(f"{row_id}:target_tool_call_count")
        if any(call.get("arguments") != arguments for call in calls):
            issues.append(f"{row_id}:target_query_value_mismatch")
        target_hashes.add(
            json.dumps(row.get("target_output"), separators=(",", ":"), sort_keys=True)
        )
    if len(target_hashes) != len(rows):
        issues.append("tool_query_targets_not_case_specific")
    return sorted(set(issues))


def load_base_model(
    *,
    model_id: str,
    model_revision: str,
    device: str,
    allow_download: bool,
) -> tuple[Any, Any, str]:
    disable_transformers_torchvision_probe()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected_device = choose_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        local_files_only=not allow_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        local_files_only=not allow_download,
        torch_dtype="auto",
    )
    model.to(selected_device)
    model.eval()
    return model, tokenizer, selected_device


def apply_trainable_state(
    model: Any,
    *,
    trainable_state_path: str | Path,
    train_last_layers: int,
) -> int:
    import torch

    trainable_params = set_trainable_last_layers(model, train_last_layers)
    state = torch.load(trainable_state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("frozen tool-query state is not a mapping")
    expected_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected_names:
        raise ValueError("frozen tool-query state parameter names do not match model")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))
    model.eval()
    return trainable_params


def generate_policy_predictions(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    device: str,
    max_new_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    import torch

    predictions: list[dict[str, Any]] = []
    model.eval()
    for index, row in enumerate(rows):
        prompt = prompt_text_for_tokenizer(
            tokenizer,
            prompt_messages(row, system_prompt=system_prompt),
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append(
            {
                "id": f"prospective_tool_query::{policy}::{index:06d}",
                "dataset": PREDICTION_DATASET,
                "source_row_id": row["id"],
                "policy": policy,
                "raw_output": raw_output,
            }
        )
        print(f"[{policy} {index + 1}/{len(rows)}] generated", flush=True)
    return predictions


def score_prediction(
    row: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    parsed, parse_error = parse_component_output(prediction.get("raw_output"))
    expected_calls = target_tool_calls(row)
    expected_names = [str(call.get("name")) for call in expected_calls]
    expected_args = expected_arguments(row)
    violations: list[str] = []
    target_keys = False
    shape = False
    tool_sequence = False
    query_fields = False
    query_values = False
    exact = False
    if parse_error:
        violations.append("prediction_parse_error")
    else:
        assert parsed is not None
        target_keys = set(parsed) == {"tool_calls"}
        if not target_keys:
            violations.append("target_key_mismatch")
        calls = parsed.get("tool_calls")
        if isinstance(calls, list) and len(calls) == len(expected_calls):
            shape = all(
                isinstance(call, Mapping)
                and isinstance(call.get("name"), str)
                and isinstance(call.get("arguments"), Mapping)
                for call in calls
            )
        if not shape:
            violations.append("tool_query_shape_invalid")
            calls = []
        tool_sequence = shape and [
            str(call.get("name")) for call in calls
        ] == expected_names
        if not tool_sequence:
            violations.append("tool_sequence_mismatch")
        query_fields = shape and all(
            set(call["arguments"]) == set(REQUIRED_QUERY_FIELDS) for call in calls
        )
        if not query_fields:
            violations.append("query_fields_mismatch")
        query_values = shape and all(
            dict(call["arguments"]) == expected_args for call in calls
        )
        if not query_values:
            violations.append("query_values_mismatch")
        exact = parsed == row.get("target_output")
        if not exact:
            violations.append("target_mismatch")
    return {
        "parseable_json": parse_error is None,
        "target_keys": target_keys,
        "tool_query_shape": shape,
        "tool_sequence": tool_sequence,
        "query_fields": query_fields,
        "query_values": query_values,
        "exact": exact,
        "violations": violations,
    }


def summarize_policy(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    by_source = {
        str(prediction.get("source_row_id")): prediction
        for prediction in predictions
        if prediction.get("policy") == policy
    }
    expected_ids = [str(row["id"]) for row in rows]
    if set(by_source) != set(expected_ids):
        raise ValueError(f"{policy} prediction alignment mismatch")
    scores = [score_prediction(row, by_source[str(row["id"])]) for row in rows]
    metrics = (
        "parseable_json",
        "target_keys",
        "tool_query_shape",
        "tool_sequence",
        "query_fields",
        "query_values",
        "exact",
    )
    violation_counts: Counter[str] = Counter()
    for score in scores:
        violation_counts.update(score["violations"])
    return {
        "rows": len(rows),
        **{
            metric: sum(bool(score[metric]) for score in scores)
            for metric in metrics
        },
        "accuracy": {
            metric: round(
                sum(bool(score[metric]) for score in scores) / len(rows),
                6,
            )
            if rows
            else 0.0
            for metric in metrics
        },
        "violations": dict(sorted(violation_counts.items())),
        "unique_raw_outputs": len(
            {
                str(prediction.get("raw_output"))
                for prediction in predictions
                if prediction.get("policy") == policy
            }
        ),
    }


def write_private_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    output = private_output_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    output.chmod(0o600)


def dry_run_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_path: str | Path,
    manifest_path: str | Path,
    freeze: Mapping[str, Any],
    trainable_state_path: str | Path,
    issues: Sequence[str],
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "dry_run": True,
        "rows": len(rows),
        "unique_targets": len(
            {
                json.dumps(row.get("target_output"), sort_keys=True)
                for row in rows
            }
        ),
        "tool_query_rows_sha256": sha256_file(rows_path),
        "experiment_manifest_sha256": sha256_file(manifest_path),
        "freeze_id": freeze.get("freeze_id"),
        "trainable_state_sha256": sha256_file(trainable_state_path),
        "issues": list(issues),
        "ready_for_full_mode": not issues,
        "training_performed": False,
        "completed_sealed_rows_used": False,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    policies = report["policies"]
    lines = [
        "# Stage A Prospective Real-Query Tool-Query Transfer",
        "",
        "Scope: base versus pre-prospective frozen tool-query SFT on 25 public",
        "development prompts with case-specific drug and condition identifiers.",
        "No prospective-row training or live-tool execution was performed.",
        "",
        "## Policy Summary",
        "",
        "| Policy | Parseable | Tool sequence | Query fields | Query values | Exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in POLICIES:
        summary = policies[name]
        rows = summary["rows"]
        lines.append(
            "| `{name}` | {parseable}/{rows} | {sequence}/{rows} | "
            "{fields}/{rows} | {values}/{rows} | {exact}/{rows} |".format(
                name=name,
                parseable=summary["parseable_json"],
                rows=rows,
                sequence=summary["tool_sequence"],
                fields=summary["query_fields"],
                values=summary["query_values"],
                exact=summary["exact"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "- Frozen SFT improves exact transfer: "
                f"`{str(report['decision']['frozen_sft_improves_exact']).lower()}`."
            ),
            "- New training, DPO/RLVR, and Hugging Face publication remain closed.",
            "- Raw generations and the trainable state remain private and uncommitted.",
            "- This development result is not an independent sealed-test estimate.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_full(
    *,
    rows: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    trainable_state_path: str | Path,
    predictions_out: str | Path,
    device: str,
    allow_download: bool,
) -> dict[str, Any]:
    policy = freeze["policy"]
    model, tokenizer, selected_device = load_base_model(
        model_id=str(policy["model_id"]),
        model_revision=str(policy["model_revision"]),
        device=device,
        allow_download=allow_download,
    )
    base_predictions = generate_policy_predictions(
        model,
        tokenizer,
        rows,
        policy="base",
        device=selected_device,
        max_new_tokens=int(policy["max_new_tokens"]),
    )
    trainable_params = apply_trainable_state(
        model,
        trainable_state_path=trainable_state_path,
        train_last_layers=int(policy["train_last_layers"]),
    )
    frozen_predictions = generate_policy_predictions(
        model,
        tokenizer,
        rows,
        policy="frozen_tool_query_sft",
        device=selected_device,
        max_new_tokens=int(policy["max_new_tokens"]),
    )
    predictions = base_predictions + frozen_predictions
    write_private_jsonl(predictions_out, predictions)
    summaries = {
        policy_name: summarize_policy(
            rows,
            predictions,
            policy=policy_name,
        )
        for policy_name in POLICIES
    }
    return {
        "dataset": DATASET,
        "dry_run": False,
        "evaluation_scope": "public_development_case_specific_real_query_transfer",
        "rows_per_policy": len(rows),
        "freeze_id": freeze.get("freeze_id"),
        "model_id": policy["model_id"],
        "model_revision": policy["model_revision"],
        "device_class": (
            "cuda" if selected_device.startswith("cuda") else selected_device
        ),
        "trainable_params_loaded": trainable_params,
        "training_performed": False,
        "completed_sealed_rows_used": False,
        "private_predictions": {
            "path": f"private_output::{Path(predictions_out).name}",
            "sha256": sha256_file(predictions_out),
            "committed": False,
        },
        "policies": summaries,
        "decision": {
            "base_exact": summaries["base"]["exact"],
            "frozen_sft_exact": summaries["frozen_tool_query_sft"]["exact"],
            "frozen_sft_improves_exact": (
                summaries["frozen_tool_query_sft"]["exact"]
                > summaries["base"]["exact"]
            ),
            "frozen_sft_query_values_exact": summaries[
                "frozen_tool_query_sft"
            ]["query_values"],
            "ready_for_new_training": False,
            "ready_for_dpo_rlvr": False,
            "ready_for_hugging_face_publication": False,
        },
        "scientific_boundary": {
            "development_only": True,
            "independent_test_claimed": False,
            "actual_query_identifier_values_visible": True,
            "live_tool_execution_evaluated": False,
            "state_trained_before_prospective_rows": True,
            "retraining_performed": False,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        default="post_training/stage_a_prospective_real_query_tool_query_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default=(
            "post_training/stage_a_prospective_real_query_experiment_manifest.json"
        ),
    )
    parser.add_argument(
        "--transfer-freeze",
        default=(
            "post_training/"
            "stage_a_prospective_tool_query_transfer_freeze_2026-07-23.json"
        ),
    )
    parser.add_argument("--trainable-state", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = load_jsonl(args.rows)
    manifest = load_json(args.manifest)
    freeze = load_json(args.transfer_freeze)
    issues = validate_inputs(
        rows=rows,
        rows_path=args.rows,
        manifest=manifest,
        freeze=freeze,
        trainable_state_path=args.trainable_state,
    )
    if issues:
        raise SystemExit("Tool-query transfer validation failed:\n- " + "\n- ".join(issues))
    if args.dry_run:
        report = dry_run_report(
            rows=rows,
            rows_path=args.rows,
            manifest_path=args.manifest,
            freeze=freeze,
            trainable_state_path=args.trainable_state,
            issues=issues,
        )
    else:
        if not args.allow_model_load:
            raise SystemExit(
                "Full mode requires --allow-model-load; run --dry-run first."
            )
        report = run_full(
            rows=rows,
            freeze=freeze,
            trainable_state_path=args.trainable_state,
            predictions_out=args.predictions_out,
            device=args.device,
            allow_download=args.allow_download,
        )
    if args.report_out:
        report_path = private_output_path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report_path.chmod(0o600)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
