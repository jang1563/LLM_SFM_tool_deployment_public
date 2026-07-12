#!/usr/bin/env python3
"""Run strict Stage A component-slice SFT smoke experiments.

This is the cluster-oriented follow-up to the strict component target export.
Each run trains and evaluates exactly one component slice:

- enum_action;
- tool_query;
- routing_after_loop.

The `--dry-run` path validates artifacts and split boundaries without loading
model weights. Full mode requires `--allow-model-load` and is intended for
Cayuga/Expanse or another GPU environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.export_stage_a_strict_component_targets import (  # noqa: E402
    ALLOWED_ACTIONS,
    ALLOWED_EVIDENCE_STATUSES,
    COMPONENTS,
    DATASET as COMPONENT_TARGET_DATASET,
    PROMPT_CONTRACT,
    TARGET_KEYS_BY_COMPONENT,
)
from post_training.generate_stage_a_predictions import disable_transformers_torchvision_probe  # noqa: E402
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    choose_device,
    collate,
    load_jsonl,
    prompt_text_for_tokenizer,
    save_trainable_state,
    set_trainable_last_layers,
    write_json,
    write_jsonl,
)


DATASET = "negbiodb_ct_stage_a_strict_component_sft_smoke_v1"
PREDICTION_DATASET = "negbiodb_ct_stage_a_strict_component_sft_predictions_v1"
EVIDENCE_COMPONENT_TARGET_DATASET = "negbiodb_ct_stage_a_evidence_conditioned_component_targets_v1"
EVIDENCE_COMPONENT_PROMPT_CONTRACT = "stage_a_v2_evidence_conditioned_component"
ALLOWED_COMPONENT_TARGET_DATASETS = {
    COMPONENT_TARGET_DATASET,
    EVIDENCE_COMPONENT_TARGET_DATASET,
}
ALLOWED_COMPONENT_PROMPT_CONTRACTS = {
    PROMPT_CONTRACT,
    EVIDENCE_COMPONENT_PROMPT_CONTRACT,
}
DECODE_MODES = (
    "freeform",
    "enum_candidate_score",
    "enum_observed_pair_score",
    "routing_observed_pair_score",
)
ENUM_CANDIDATE_DECODE_MODES = ("enum_candidate_score", "enum_observed_pair_score")
ROUTING_CANDIDATE_DECODE_MODES = ("routing_observed_pair_score",)


def component_case_id(row: Mapping[str, Any]) -> str:
    value = row.get("source_manifest_case_id") or row.get("case_id") or row.get("task_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Component row is missing case id: {row.get('id')!r}")
    return value


def row_value_counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def filter_component(rows: Sequence[Mapping[str, Any]], component: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("component") == component]


def prompt_messages_from_row(row: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = row.get("prompt_messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{row.get('id')} has invalid component prompt messages")
    out: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError(f"{row.get('id')} has malformed prompt message")
        role = message.get("role")
        if role not in {"system", "user"}:
            raise ValueError(f"{row.get('id')} has unexpected prompt role: {role!r}")
        out.append({"role": str(role), "content": str(message.get("content", ""))})
    return out


def target_output_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    target = row.get("target_output")
    if not isinstance(target, Mapping):
        raise ValueError(f"{row.get('id')} has no target_output object")
    return dict(target)


def target_text_from_row(row: Mapping[str, Any]) -> str:
    return json.dumps(target_output_from_row(row), sort_keys=True)


def enum_action_candidate_outputs() -> list[dict[str, str]]:
    """Return the finite valid output space for the enum/action component."""

    return [
        {"action": action, "evidence_status": evidence_status}
        for action in ALLOWED_ACTIONS
        for evidence_status in ALLOWED_EVIDENCE_STATUSES
    ]


def enum_action_observed_pair_outputs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return unique enum/action target pairs observed in the supplied rows."""

    outputs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        target = target_output_from_row(row)
        action = target.get("action")
        evidence_status = target.get("evidence_status")
        if action not in ALLOWED_ACTIONS or evidence_status not in ALLOWED_EVIDENCE_STATUSES:
            continue
        key = (str(action), str(evidence_status))
        if key not in seen:
            seen.add(key)
            outputs.append({"action": key[0], "evidence_status": key[1]})
    return outputs


def routing_observed_pair_outputs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return unique action/status pairs observed in routing component targets."""

    outputs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        target = target_output_from_row(row)
        action = target.get("action")
        evidence_status = target.get("evidence_status")
        if action not in ALLOWED_ACTIONS or evidence_status not in ALLOWED_EVIDENCE_STATUSES:
            continue
        key = (str(action), str(evidence_status))
        if key not in seen:
            seen.add(key)
            outputs.append({"action": key[0], "evidence_status": key[1]})
    return outputs


def enum_candidate_outputs_for_decode(
    decode_mode: str,
    train_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]] | None:
    if decode_mode == "enum_candidate_score":
        return enum_action_candidate_outputs()
    if decode_mode == "enum_observed_pair_score":
        outputs = enum_action_observed_pair_outputs(train_rows)
        if not outputs:
            raise ValueError("enum_observed_pair_score needs non-empty train-observed enum/action pairs")
        return outputs
    return None


def candidate_outputs_for_decode(
    decode_mode: str,
    train_rows: Sequence[Mapping[str, Any]],
    *,
    component: str,
) -> list[dict[str, Any]] | None:
    if decode_mode in ENUM_CANDIDATE_DECODE_MODES:
        if component != "enum_action":
            raise ValueError("enum candidate decode modes require component=enum_action")
        return enum_candidate_outputs_for_decode(decode_mode, train_rows)
    if decode_mode == "routing_observed_pair_score":
        if component != "routing_after_loop":
            raise ValueError("routing_observed_pair_score requires component=routing_after_loop")
        outputs = routing_observed_pair_outputs(train_rows)
        if not outputs:
            raise ValueError("routing_observed_pair_score needs non-empty train-observed routing pairs")
        return outputs
    return None


def visible_citation_ids(row: Mapping[str, Any]) -> list[str]:
    """Extract public-safe citation IDs from model-visible tool-result state."""

    try:
        payload = json.loads(prompt_messages_from_row(row)[1]["content"])
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, Mapping):
        return []
    tool_results = payload.get("observed_tool_loop")
    evidence_packet = payload.get("evidence_packet")
    if not isinstance(tool_results, list) and isinstance(evidence_packet, Mapping):
        tool_results = evidence_packet.get("tool_results")
    if not isinstance(tool_results, list):
        return []
    out: list[str] = []
    for item in tool_results:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if not isinstance(content, Mapping):
            continue
        for record in content.get("same_indication_records", ()) or ():
            if isinstance(record, Mapping) and isinstance(record.get("source_id"), str):
                out.append(record["source_id"])
        for source_id in content.get("citation_candidates", ()) or ():
            if isinstance(source_id, str):
                out.append(source_id)
    clean: list[str] = []
    seen: set[str] = set()
    for source_id in out:
        if source_id.startswith("visible_evidence::"):
            continue
        if source_id not in seen:
            seen.add(source_id)
            clean.append(source_id)
    return clean


def cited_source_ids_for_candidate(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    action = candidate.get("action")
    evidence_status = candidate.get("evidence_status")
    if (action, evidence_status) in {
        ("ground", "supported"),
        ("flag", "invalid_value"),
    }:
        return visible_citation_ids(row)[:1]
    return []


def routing_candidates_for_row(
    row: Mapping[str, Any],
    action_status_candidates: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [
        {
            "action": str(candidate["action"]),
            "evidence_status": str(candidate["evidence_status"]),
            "cited_source_ids": cited_source_ids_for_candidate(row, candidate),
        }
        for candidate in (action_status_candidates or routing_observed_pair_outputs([row]))
    ]


def validate_component_rows(
    all_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    component: str,
) -> list[str]:
    issues: list[str] = []
    if component not in COMPONENTS:
        issues.append(f"unknown_component:{component}")
        return issues

    if not train_rows:
        issues.append(f"{component}:empty_train_rows")
    if not heldout_rows:
        issues.append(f"{component}:empty_heldout_rows")

    all_counts = Counter(str(row.get("component")) for row in all_rows)
    for required in COMPONENTS:
        if all_counts[required] == 0:
            issues.append(f"missing_component:{required}")

    train_case_ids = [component_case_id(row) for row in train_rows]
    heldout_case_ids = [component_case_id(row) for row in heldout_rows]
    if len(set(train_case_ids)) != len(train_case_ids):
        issues.append(f"{component}:train_duplicate_case_id")
    if len(set(heldout_case_ids)) != len(heldout_case_ids):
        issues.append(f"{component}:heldout_duplicate_case_id")
    if set(train_case_ids) & set(heldout_case_ids):
        issues.append(f"{component}:train_heldout_case_overlap")

    for split, rows in (("train", train_rows), ("heldout", heldout_rows)):
        for row in rows:
            row_id = row.get("id")
            if row.get("dataset") not in ALLOWED_COMPONENT_TARGET_DATASETS:
                issues.append(f"{row_id}:{split}_unexpected_dataset")
            if row.get("component") != component:
                issues.append(f"{row_id}:{split}_wrong_component")
            if row.get("prompt_contract") not in ALLOWED_COMPONENT_PROMPT_CONTRACTS:
                issues.append(f"{row_id}:{split}_wrong_prompt_contract")
            if row.get("oracle_target") is not True:
                issues.append(f"{row_id}:{split}_missing_oracle_target")
            if row.get("split") != split:
                issues.append(f"{row_id}:{split}_wrong_split")
            if row.get("target_keys") != list(TARGET_KEYS_BY_COMPONENT[component]):
                issues.append(f"{row_id}:{split}_target_keys_mismatch")
            try:
                prompt_messages = prompt_messages_from_row(row)
            except ValueError as exc:
                issues.append(f"{row_id}:{split}_bad_prompt:{exc}")
                prompt_messages = []
            prompt_text = json.dumps(prompt_messages, sort_keys=True)
            for leak in (
                "hidden_eval_metadata",
                "gold_evidence_status",
                "expected_terminal_action",
                "source_task_id",
                "split_group",
            ):
                if leak in prompt_text:
                    issues.append(f"{row_id}:{split}_prompt_leaks_{leak}")
            target = target_output_from_row(row)
            issues.extend(component_target_issues(row_id, component, target, split=split))
    return issues


def component_target_issues(
    row_id: Any,
    component: str,
    target: Mapping[str, Any],
    *,
    split: str,
) -> list[str]:
    issues: list[str] = []
    if set(target) != set(TARGET_KEYS_BY_COMPONENT[component]):
        issues.append(f"{row_id}:{split}_target_key_mismatch")
    if "action" in target and target.get("action") not in ALLOWED_ACTIONS:
        issues.append(f"{row_id}:{split}_bad_action")
    if "evidence_status" in target and target.get("evidence_status") not in ALLOWED_EVIDENCE_STATUSES:
        issues.append(f"{row_id}:{split}_bad_evidence_status")
    if component == "tool_query":
        tool_calls = target.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            issues.append(f"{row_id}:{split}_missing_tool_calls")
        else:
            for call in tool_calls:
                if not tool_call_ok(call):
                    issues.append(f"{row_id}:{split}_bad_tool_call")
                    break
    if component == "routing_after_loop" and not isinstance(target.get("cited_source_ids"), list):
        issues.append(f"{row_id}:{split}_bad_citations")
    return issues


def tool_call_ok(call: Any) -> bool:
    if not isinstance(call, Mapping):
        return False
    if not isinstance(call.get("name"), str) or not call.get("name"):
        return False
    args = call.get("arguments")
    if not isinstance(args, Mapping):
        return False
    return {"drug_id", "condition_id"}.issubset(args)


def encode_example(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_length: int,
) -> dict[str, Any]:
    import torch

    prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_row(row))
    target = target_text_from_row(row) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        prompt_trim = min(overflow, max(0, len(prompt_ids) - 1))
        input_ids = input_ids[prompt_trim:]
        labels = labels[prompt_trim:]
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def parse_component_output(raw_output: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw_output, Mapping):
        return dict(raw_output), None
    if not isinstance(raw_output, str):
        return None, "prediction_output_not_string_or_object"
    text = raw_output.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"json_parse_error:{exc.msg}"
    if not isinstance(parsed, dict):
        return None, "prediction_json_not_object"
    return parsed, None


def score_component_prediction(
    expected_row: Mapping[str, Any],
    prediction_row: Mapping[str, Any],
) -> dict[str, Any]:
    component = str(expected_row["component"])
    target = target_output_from_row(expected_row)
    payload = prediction_row.get("prediction", prediction_row.get("raw_output"))
    parsed, parse_error = parse_component_output(payload)
    if parse_error:
        return component_failed_row(expected_row, prediction_row, ("prediction_parse_error",), parse_error=parse_error)
    assert parsed is not None
    violations: list[str] = []
    reward: dict[str, float] = {}

    keys_ok = set(parsed) == set(TARGET_KEYS_BY_COMPONENT[component])
    reward["target_keys"] = float(keys_ok)
    if not keys_ok:
        violations.append("target_key_mismatch")

    enum_ok = enum_fields_ok(component, parsed)
    reward["enum_validity"] = float(enum_ok)
    if not enum_ok:
        violations.append("enum_value_invalid")

    tool_shape_ok = True if component != "tool_query" else tool_query_shape_ok(parsed)
    reward["tool_query_shape"] = float(tool_shape_ok)
    if not tool_shape_ok:
        violations.append("tool_query_shape_invalid")

    exact_ok = parsed == target
    reward["exact_match"] = float(exact_ok)
    if not exact_ok:
        violations.append("target_mismatch")

    return {
        "id": prediction_row.get("id"),
        "case_id": component_case_id(expected_row),
        "component": component,
        "split": prediction_row.get("split"),
        "case_family": expected_row.get("case_family"),
        "score": round(sum(reward.values()) / len(reward), 3),
        "passed": not violations and sum(reward.values()) == len(reward),
        "reward_breakdown": reward,
        "violations": violations,
        "parse_error": None,
    }


def enum_fields_ok(component: str, output: Mapping[str, Any]) -> bool:
    if component in {"enum_action", "routing_after_loop"}:
        if output.get("action") not in ALLOWED_ACTIONS:
            return False
        if output.get("evidence_status") not in ALLOWED_EVIDENCE_STATUSES:
            return False
    return True


def tool_query_shape_ok(output: Mapping[str, Any]) -> bool:
    tool_calls = output.get("tool_calls")
    return isinstance(tool_calls, list) and bool(tool_calls) and all(tool_call_ok(call) for call in tool_calls)


def component_failed_row(
    expected_row: Mapping[str, Any],
    prediction_row: Mapping[str, Any] | None,
    violations: Sequence[str],
    *,
    parse_error: str | None = None,
) -> dict[str, Any]:
    return {
        "id": prediction_row.get("id") if prediction_row else None,
        "case_id": component_case_id(expected_row),
        "component": expected_row.get("component"),
        "split": prediction_row.get("split") if prediction_row else expected_row.get("split"),
        "case_family": expected_row.get("case_family"),
        "score": 0.0,
        "passed": False,
        "reward_breakdown": {},
        "violations": list(violations),
        "parse_error": parse_error,
    }


def summarize_component_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "passed": 0,
            "mean_score": 0.0,
            "gate_accuracy": {},
            "violations": {},
            "by_component": {},
        }
    reward_keys = sorted({key for row in rows for key in row.get("reward_breakdown", {})})
    violations = Counter(violation for row in rows for violation in row.get("violations", ()))
    by_component: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_component[str(row.get("component"))].append(row)
    return {
        "cases": len(rows),
        "passed": sum(1 for row in rows if row.get("passed")),
        "mean_score": round(sum(float(row.get("score", 0.0)) for row in rows) / len(rows), 3),
        "gate_accuracy": {
            key: round(
                sum(float(row.get("reward_breakdown", {}).get(key, 0.0)) for row in rows) / len(rows),
                3,
            )
            for key in reward_keys
        },
        "violations": dict(sorted(violations.items())),
        "by_component": {
            component: {
                "cases": len(items),
                "passed": sum(1 for item in items if item.get("passed")),
                "mean_score": round(sum(float(item.get("score", 0.0)) for item in items) / len(items), 3),
            }
            for component, items in sorted(by_component.items())
        },
    }


def build_component_eval_report(
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    component: str,
    run_id: str,
) -> dict[str, Any]:
    expected_by_id = {str(row["id"]): row for row in expected_rows}
    predictions_by_id = {
        str(row.get("source_component_target_id") or row.get("id")): row
        for row in prediction_rows
    }
    row_reports = []
    for row_id, expected_row in expected_by_id.items():
        prediction_row = predictions_by_id.get(row_id)
        if prediction_row is None:
            row_reports.append(component_failed_row(expected_row, None, ("missing_prediction",)))
        else:
            row_reports.append(score_component_prediction(expected_row, prediction_row))
    unexpected = sorted(set(predictions_by_id) - set(expected_by_id))
    for row_id in unexpected:
        row_reports.append(
            {
                "id": row_id,
                "case_id": None,
                "component": component,
                "split": predictions_by_id[row_id].get("split"),
                "case_family": None,
                "score": 0.0,
                "passed": False,
                "reward_breakdown": {},
                "violations": ["unexpected_prediction_id"],
                "parse_error": None,
            }
        )
    return {
        "dataset": DATASET,
        "run_id": run_id,
        "component": component,
        "boundary": "Offline component-slice scorer; no live API calls and no model weights are loaded.",
        "cases_expected": len(expected_rows),
        "predictions_received": len(prediction_rows),
        "summary": summarize_component_rows(row_reports),
        "rows": row_reports,
    }


def dry_run_report(
    *,
    model: str,
    component: str,
    decode_mode: str,
    train_targets: str | Path,
    heldout_targets: str | Path,
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    issues: Sequence[str],
) -> dict[str, Any]:
    candidate_outputs = candidate_outputs_for_decode(decode_mode, train_rows, component=component)
    component_rows = list(train_rows) + list(heldout_rows)
    return {
        "dataset": DATASET,
        "dry_run": True,
        "model": model,
        "component": component,
        "decode_mode": decode_mode,
        "candidate_space_size": len(candidate_outputs) if candidate_outputs is not None else None,
        "candidate_outputs": candidate_outputs,
        "source_target_datasets": row_value_counts(component_rows, "dataset"),
        "source_prompt_contracts": row_value_counts(component_rows, "prompt_contract"),
        "train_targets": str(train_targets),
        "heldout_targets": str(heldout_targets),
        "train_examples": len(train_rows),
        "heldout_examples": len(heldout_rows),
        "train_case_ids": [component_case_id(row) for row in train_rows],
        "heldout_case_ids": [component_case_id(row) for row in heldout_rows],
        "issues": list(issues),
        "boundary": (
            "Dry run validates component-slice artifacts and split boundaries "
            "without loading model weights or running local heavy compute."
        ),
    }


def generate_prediction_rows(
    model: Any,
    tokenizer: Any,
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    model_id: str,
    component: str,
    device: str,
    max_length: int,
    max_new_tokens: int,
    decode_mode: str,
    candidate_outputs: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    import torch

    predictions: list[dict[str, Any]] = []
    model.eval()
    for index, row in enumerate(heldout_rows):
        prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_row(row))
        if decode_mode == "enum_candidate_score":
            candidate_result = score_enum_action_candidates(
                model,
                tokenizer,
                prompt,
                device=device,
                max_length=max_length,
                candidate_outputs=candidate_outputs,
            )
            raw_output = candidate_result["raw_output"]
            prediction = candidate_result["prediction"]
            candidate_scores = candidate_result["candidate_scores"]
        elif decode_mode == "enum_observed_pair_score":
            candidate_result = score_enum_action_candidates(
                model,
                tokenizer,
                prompt,
                device=device,
                max_length=max_length,
                candidate_outputs=candidate_outputs,
            )
            raw_output = candidate_result["raw_output"]
            prediction = candidate_result["prediction"]
            candidate_scores = candidate_result["candidate_scores"]
        elif decode_mode == "routing_observed_pair_score":
            row_candidates = routing_candidates_for_row(row, candidate_outputs)
            candidate_result = score_component_candidates(
                model,
                tokenizer,
                prompt,
                device=device,
                max_length=max_length,
                candidate_outputs=row_candidates,
            )
            raw_output = candidate_result["raw_output"]
            prediction = candidate_result["prediction"]
            candidate_scores = candidate_result["candidate_scores"]
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output_ids[0][inputs["input_ids"].shape[-1]:]
            raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
            prediction = None
            candidate_scores = None
        predictions.append(
            {
                "id": f"{run_id}::{row['id']}",
                "source_component_target_id": row["id"],
                "case_id": component_case_id(row),
                "dataset": PREDICTION_DATASET,
                "source": "stage_a_strict_component_sft_smoke",
                "run_id": run_id,
                "model": model_id,
                "component": component,
                "prompt_contract": row.get("prompt_contract", PROMPT_CONTRACT),
                "split": row.get("split"),
                "generation_prompt_hash": row.get("generation_prompt_hash"),
                "decode_mode": decode_mode,
                "candidate_space_size": len(candidate_outputs) if candidate_outputs is not None else None,
                "raw_output": raw_output,
            }
        )
        if prediction is not None:
            predictions[-1]["prediction"] = prediction
        if candidate_scores is not None:
            predictions[-1]["candidate_scores"] = candidate_scores
        print(f"[{index + 1}/{len(heldout_rows)}] generated {row['id']}", flush=True)
    return predictions


def score_enum_action_candidates(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
    max_length: int,
    candidate_outputs: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    return score_component_candidates(
        model,
        tokenizer,
        prompt,
        device=device,
        max_length=max_length,
        candidate_outputs=candidate_outputs or enum_action_candidate_outputs(),
    )


def score_component_candidates(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
    max_length: int,
    candidate_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    candidates = [dict(candidate) for candidate in candidate_outputs]
    for candidate in candidates:
        score = score_candidate_target(
            model,
            tokenizer,
            prompt,
            json.dumps(candidate, sort_keys=True),
            device=device,
            max_length=max_length,
        )
        scored.append({"score": score, "candidate": candidate})
    scored.sort(key=lambda item: item["score"], reverse=True)
    winner = scored[0]["candidate"]
    return {
        "prediction": winner,
        "raw_output": json.dumps(winner, sort_keys=True),
        "candidate_scores": scored,
    }


def score_candidate_target(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_text: str,
    *,
    device: str,
    max_length: int,
) -> float:
    import torch

    target = target_text + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        prompt_trim = min(overflow, max(0, len(prompt_ids) - 1))
        input_ids = input_ids[prompt_trim:]
        labels = labels[prompt_trim:]
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }
    with torch.no_grad():
        loss = model(**batch).loss
    return round(-float(loss.detach().cpu()), 6)


def run_training_and_eval(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    disable_transformers_torchvision_probe()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    candidate_outputs = candidate_outputs_for_decode(args.decode_mode, train_rows, component=args.component)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_example(tokenizer, row, max_length=args.max_length) for row in train_rows]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        torch_dtype="auto",
    )
    model.to(device)
    model.train()
    trainable_params = set_trainable_last_layers(model, args.train_last_layers)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
    )

    losses: list[float] = []
    cursor = 0
    for step in range(args.max_steps):
        batch_features = []
        for _ in range(args.batch_size):
            batch_features.append(encoded[cursor % len(encoded)])
            cursor += 1
        batch = collate(batch_features, tokenizer.pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"step={step + 1} loss={losses[-1]:.4f}", flush=True)

    state_path = None
    if not args.no_save_trainable_state:
        state_path = save_trainable_state(model, out_dir)

    predictions = generate_prediction_rows(
        model,
        tokenizer,
        heldout_rows,
        run_id=args.run_id,
        model_id=args.model,
        component=args.component,
        device=device,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        decode_mode=args.decode_mode,
        candidate_outputs=candidate_outputs,
    )
    predictions_path = Path(args.predictions_out)
    eval_report_path = Path(args.eval_out)
    write_jsonl(predictions_path, predictions)

    eval_report = build_component_eval_report(
        expected_rows=heldout_rows,
        prediction_rows=predictions,
        component=args.component,
        run_id=args.run_id,
    )
    write_json(eval_report_path, eval_report)

    return {
        "dataset": DATASET,
        "dry_run": False,
        "run_id": args.run_id,
        "model": args.model,
        "component": args.component,
        "device": device,
        "source_target_datasets": row_value_counts(list(train_rows) + list(heldout_rows), "dataset"),
        "source_prompt_contracts": row_value_counts(list(train_rows) + list(heldout_rows), "prompt_contract"),
        "train_targets": args.train_targets,
        "heldout_targets": args.heldout_targets,
        "train_examples": len(train_rows),
        "heldout_examples": len(heldout_rows),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "decode_mode": args.decode_mode,
        "candidate_space_size": len(candidate_outputs) if candidate_outputs is not None else None,
        "candidate_outputs": candidate_outputs,
        "train_last_layers": args.train_last_layers,
        "trainable_params": trainable_params,
        "losses": losses,
        "loss_delta": losses[-1] - losses[0] if len(losses) > 1 else 0.0,
        "trainable_state": str(state_path) if state_path else None,
        "predictions": str(predictions_path),
        "eval_report": str(eval_report_path),
        "eval_summary": eval_report["summary"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--component", choices=COMPONENTS, default="enum_action")
    parser.add_argument("--targets", default="post_training/stage_a_strict_component_targets_v1.jsonl")
    parser.add_argument("--train-targets", default="post_training/stage_a_strict_component_targets_train_v1.jsonl")
    parser.add_argument("--heldout-targets", default="post_training/stage_a_strict_component_targets_heldout_v1.jsonl")
    parser.add_argument("--out-dir", default="post_training/runs/stage_a_strict_component_sft_smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-heldout", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--decode-mode", choices=DECODE_MODES, default="freeform")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-trainable-state", action="store_true")
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument("--eval-out", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.run_id is None:
        args.run_id = f"stage_a_strict_component_{args.component}_sft_smoke"
    if args.decode_mode in ENUM_CANDIDATE_DECODE_MODES and args.component != "enum_action":
        raise SystemExit(
            "--decode-mode enum_candidate_score/enum_observed_pair_score "
            "is only valid with --component enum_action."
        )
    if args.decode_mode in ROUTING_CANDIDATE_DECODE_MODES and args.component != "routing_after_loop":
        raise SystemExit(
            "--decode-mode routing_observed_pair_score is only valid with "
            "--component routing_after_loop."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report_out is None:
        args.report_out = str(out_dir / "report.json")
    if args.predictions_out is None:
        args.predictions_out = str(out_dir / "predictions.jsonl")
    if args.eval_out is None:
        args.eval_out = str(out_dir / "eval_report.json")

    all_rows = load_jsonl(args.targets)
    train_rows = filter_component(load_jsonl(args.train_targets), args.component)
    heldout_rows = filter_component(load_jsonl(args.heldout_targets), args.component)
    if args.limit_train is not None:
        train_rows = train_rows[: args.limit_train]
    if args.limit_heldout is not None:
        heldout_rows = heldout_rows[: args.limit_heldout]

    issues = validate_component_rows(
        all_rows,
        train_rows,
        heldout_rows,
        component=args.component,
    )
    if issues:
        write_json(
            args.report_out,
            dry_run_report(
                model=args.model,
                component=args.component,
                decode_mode=args.decode_mode,
                train_targets=args.train_targets,
                heldout_targets=args.heldout_targets,
                train_rows=train_rows,
                heldout_rows=heldout_rows,
                issues=issues,
            ),
        )
        raise SystemExit("Component SFT smoke validation failed:\n- " + "\n- ".join(issues))

    if args.dry_run:
        report = dry_run_report(
            model=args.model,
            component=args.component,
            decode_mode=args.decode_mode,
            train_targets=args.train_targets,
            heldout_targets=args.heldout_targets,
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            issues=issues,
        )
    else:
        if not args.allow_model_load:
            raise SystemExit("Full component SFT smoke requires --allow-model-load. Use --dry-run for local validation.")
        report = run_training_and_eval(args, train_rows, heldout_rows)
    write_json(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
