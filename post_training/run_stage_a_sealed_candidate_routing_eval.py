#!/usr/bin/env python3
"""Run a one-time private Stage A sealed candidate-routing evaluation.

This evaluates a frozen evidence-conditioned routing policy. Hidden sealed
labels are used only to construct a synthetic oracle tool-result state and to
score predictions; they are never included in model-visible prompts. The run
therefore measures routing after a completed oracle tool loop, not end-to-end
tool querying or live database execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from negbiodb_ct.stage_a_manifest import (  # noqa: E402
    load_stage_a_manifest,
    validate_stage_a_manifest,
)
from post_training.build_stage_a_sealed_extension import (  # noqa: E402
    require_external_private_path,
)
from post_training.evaluate_stage_a_routing_evidence_gate import (  # noqa: E402
    evidence_features,
    gate_output,
    pair_label,
)
from post_training.export_stage_a_evidence_conditioned_component_targets import (  # noqa: E402
    evidence_packet_for_row,
)
from post_training.export_stage_a_saved_output_evidence_candidate_routing_rows import (  # noqa: E402
    CANDIDATE_PAIRS,
    candidate_outputs,
    candidate_task,
)
from post_training.generate_stage_a_predictions import (  # noqa: E402
    disable_transformers_torchvision_probe,
)
from post_training.run_stage_a_saved_output_evidence_candidate_routing_smoke import (  # noqa: E402
    SYSTEM_PROMPT,
    score_candidate_outputs,
)
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    score_component_prediction,
)
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    choose_device,
    prompt_text_for_tokenizer,
    set_trainable_last_layers,
)


DATASET = "negbiodb_ct_stage_a_sealed_candidate_routing_eval_v1"
FREEZE_DATASET = "negbiodb_ct_stage_a_candidate_routing_policy_freeze_v1"
COMMITMENT_DATASET = "negbiodb_ct_stage_a_sealed_extension_commitment_v1"
PRIVATE_PREDICTION_DATASET = (
    "negbiodb_ct_stage_a_sealed_candidate_routing_private_predictions_v1"
)
EVAL_SCOPE = "routing_after_synthetic_oracle_tool_loop"
PAIR_FROM_TERMINAL_STATUS = {
    ("ground_with_attribution", "supported"): "ground/supported",
    ("reject_or_flag_unsupported_claim", "contradicted"): "reject/contradicted",
    ("defer_or_request_more_evidence", "insufficient"): "defer/insufficient",
    ("verify_with_assay_or_database", "insufficient"): "verify/insufficient",
    ("reject_or_flag_unsupported_claim", "invalid_value"): "flag/invalid_value",
}
RAW_PUBLIC_FORBIDDEN_KEYS = {
    "predictions",
    "candidate_scores",
    "raw_output",
    "case_ids",
    "source_task_ids",
    "split_groups",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def compact_action_status(row: Mapping[str, Any]) -> tuple[str, str, str]:
    hidden = row.get("hidden_eval_metadata")
    if not isinstance(hidden, Mapping):
        raise ValueError("sealed row lacks hidden_eval_metadata")
    terminal = str(hidden.get("expected_terminal_action"))
    status = str(hidden.get("gold_evidence_status"))
    pair = PAIR_FROM_TERMINAL_STATUS.get((terminal, status))
    if pair is None:
        raise ValueError(f"unsupported sealed terminal/status pair: {terminal}/{status}")
    action, _ = pair.split("/", maxsplit=1)
    return action, status, pair


def _strict_like_row(row: Mapping[str, Any]) -> dict[str, Any]:
    visible = row.get("model_visible_task")
    hidden = row.get("hidden_eval_metadata")
    if not isinstance(visible, Mapping) or not isinstance(hidden, Mapping):
        raise ValueError("sealed row lacks visible or hidden mapping")
    action, status, _ = compact_action_status(row)
    source_ids = [str(value) for value in hidden.get("gold_source_ids", ())]
    tool_calls = [
        {
            "name": str(tool),
            "arguments": {
                str(field): f"<{field}>"
                for field in hidden.get("required_query_fields", ("drug_id", "condition_id"))
            },
        }
        for tool in hidden.get("required_tools", ())
    ]
    user_payload = {
        "input_id": visible.get("input_id"),
        "claim": visible.get("claim"),
        "allowed_tools": list(visible.get("allowed_tools", ())),
    }
    return {
        "source_manifest_case_id": row.get("case_id"),
        "messages": [
            {"role": "system", "content": "private sealed routing adapter"},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ],
        "target_output": {
            "action": action,
            "evidence_status": status,
            "cited_source_ids": source_ids,
            "tool_calls": tool_calls,
        },
    }


def sealed_candidate_row(row: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    visible = row["model_visible_task"]
    hidden = row["hidden_eval_metadata"]
    action, status, target_pair = compact_action_status(row)
    strict_like = _strict_like_row(row)
    packet = evidence_packet_for_row(strict_like)
    routing_prompt = {
        "prompt_messages": [
            {"role": "system", "content": "private sealed evidence adapter"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "input_id": visible.get("input_id"),
                        "claim": visible.get("claim"),
                        "observed_tool_loop": packet["tool_results"],
                    },
                    sort_keys=True,
                ),
            },
        ],
        "target_output": {
            "action": action,
            "evidence_status": status,
            "cited_source_ids": [
                str(value) for value in hidden.get("gold_source_ids", ())
            ],
        },
    }
    features = evidence_features(routing_prompt)
    runtime_output, runtime_reason = gate_output(features)
    runtime_pair = pair_label(runtime_output)
    if runtime_pair != target_pair:
        raise ValueError(
            f"sealed row {ordinal} oracle runtime pair mismatch: "
            f"{runtime_pair} != {target_pair}"
        )
    model_visible_task = candidate_task(routing_prompt, features=features)
    visible_text = json.dumps(model_visible_task, sort_keys=True)
    for forbidden in (
        "hidden_eval_metadata",
        "source_task_id",
        "split_group",
        "gold_evidence_status",
        "expected_terminal_action",
    ):
        if forbidden in visible_text:
            raise ValueError(f"sealed row {ordinal} model-visible leak: {forbidden}")
    target_payload = {
        "selected_pair": target_pair,
        "action": action,
        "evidence_status": status,
    }
    return {
        "id": f"sealed_candidate_routing::{ordinal:06d}",
        "source_manifest_case_id": str(row.get("case_id")),
        "model_visible_task": model_visible_task,
        "candidate_outputs": candidate_outputs(),
        "target_output": target_payload,
        "target_pair": target_pair,
        "runtime_pair": runtime_pair,
        "runtime_reason": runtime_reason,
    }


def build_sealed_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [sealed_candidate_row(row, ordinal=index) for index, row in enumerate(rows)]


def validate_inputs(
    *,
    sealed_manifest_path: str | Path,
    commitment_path: str | Path,
    freeze_path: str | Path,
    trainable_state_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str]]:
    require_external_private_path(sealed_manifest_path, role="private sealed manifest")
    require_external_private_path(trainable_state_path, role="frozen trainable state")
    commitment = load_json(commitment_path)
    freeze = load_json(freeze_path)
    rows = load_stage_a_manifest(sealed_manifest_path)
    issues = validate_stage_a_manifest(
        rows,
        min_rows=25,
        min_status_count=5,
        require_unique_split_groups=True,
    )
    if commitment.get("dataset") != COMMITMENT_DATASET:
        issues.append("sealed_commitment_dataset_mismatch")
    if freeze.get("dataset") != FREEZE_DATASET:
        issues.append("policy_freeze_dataset_mismatch")
    expected_manifest = (
        commitment.get("input_artifacts", {})
        .get("private_sealed_manifest", {})
    )
    if expected_manifest.get("sha256") != sha256_file(sealed_manifest_path):
        issues.append("sealed_manifest_sha256_mismatch")
    if expected_manifest.get("records") != len(rows):
        issues.append("sealed_manifest_record_count_mismatch")
    frozen_state = freeze.get("frozen_artifacts", {}).get("trainable_state", {})
    if frozen_state.get("sha256") != sha256_file(trainable_state_path):
        issues.append("trainable_state_sha256_mismatch")
    evaluator = freeze.get("frozen_artifacts", {}).get("sealed_evaluator", {})
    if evaluator.get("sha256") != sha256_file(__file__):
        issues.append("sealed_evaluator_sha256_mismatch")
    return rows, commitment, freeze, sorted(set(issues))


def private_reference(path: str | Path, *, role: str) -> dict[str, str]:
    value = Path(path)
    return {
        "path": f"external_private_input::{value.name}",
        "role": role,
        "sha256": sha256_file(value),
    }


def dry_run_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    commitment: Mapping[str, Any],
    freeze: Mapping[str, Any],
    sealed_manifest_path: str | Path,
    trainable_state_path: str | Path,
    issues: Sequence[str],
) -> dict[str, Any]:
    candidate_rows = build_sealed_candidate_rows(rows) if not issues else []
    target_counts = Counter(row["target_pair"] for row in candidate_rows)
    runtime_counts = Counter(row["runtime_pair"] for row in candidate_rows)
    return {
        "dataset": DATASET,
        "dry_run": True,
        "evaluation_scope": EVAL_SCOPE,
        "sealed_rows": len(rows),
        "candidate_space_size": len(CANDIDATE_PAIRS),
        "target_pair_counts": dict(sorted(target_counts.items())),
        "runtime_pair_counts": dict(sorted(runtime_counts.items())),
        "runtime_oracle_exact": sum(
            row["target_pair"] == row["runtime_pair"] for row in candidate_rows
        ),
        "freeze_id": freeze.get("freeze_id"),
        "sealed_commitment_sha256": sha256_file(
            ROOT / "post_training/stage_a_sealed_extension_commitment_2026-07-10.json"
        )
        if Path(
            ROOT / "post_training/stage_a_sealed_extension_commitment_2026-07-10.json"
        ).exists()
        else None,
        "input_artifacts": {
            "sealed_manifest": private_reference(
                sealed_manifest_path, role="private sealed Stage A manifest"
            ),
            "trainable_state": private_reference(
                trainable_state_path, role="private frozen trainable state"
            ),
        },
        "issues": list(issues),
        "ready_for_one_time_full_mode": not issues,
        "privacy_boundary": {
            "hidden_labels_used_to_construct_oracle_tool_state": True,
            "hidden_labels_visible_to_model": False,
            "row_level_labels_emitted": False,
            "actual_tool_query_evaluated": False,
            "live_tool_execution_evaluated": False,
        },
        "commitment_ready": bool(
            commitment.get("decision", {}).get("ready_for_one_time_sealed_evaluation")
        ),
    }


def create_one_time_lock(
    path: str | Path,
    *,
    sealed_sha256: str,
    freeze_id: str,
) -> None:
    require_external_private_path(path, role="one-time evaluation lock")
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "negbiodb_ct_stage_a_sealed_evaluation_lock_v1",
        "status": "started",
        "sealed_manifest_sha256": sealed_sha256,
        "freeze_id": freeze_id,
    }
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("one-time sealed evaluation lock already exists") from exc
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def complete_one_time_lock(path: str | Path, *, compact_sha256: str) -> None:
    lock_path = Path(path)
    payload = load_json(lock_path)
    payload["status"] = "complete"
    payload["compact_result_sha256"] = compact_sha256
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lock_path.chmod(0o600)


def load_frozen_model(
    *,
    model_id: str,
    trainable_state_path: str | Path,
    train_last_layers: int,
    device: str,
    allow_download: bool,
) -> tuple[Any, Any, str, int]:
    import torch

    disable_transformers_torchvision_probe()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected_device = choose_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=not allow_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=not allow_download,
        torch_dtype="auto",
    )
    model.to(selected_device)
    trainable_params = set_trainable_last_layers(model, train_last_layers)
    state = torch.load(trainable_state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("frozen trainable state is not a mapping")
    expected_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected_names:
        missing = sorted(expected_names - set(state))
        unexpected = sorted(set(state) - expected_names)
        raise ValueError(
            f"frozen state parameter mismatch: missing={missing[:3]} "
            f"unexpected={unexpected[:3]}"
        )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))
    model.eval()
    return model, tokenizer, selected_device, trainable_params


def private_prediction_row(
    *,
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    return {
        "id": f"sealed_prediction::{ordinal:06d}",
        "dataset": PRIVATE_PREDICTION_DATASET,
        "source_candidate_routing_id": row["id"],
        "case_id": row["source_manifest_case_id"],
        "target_pair": row["target_pair"],
        "prediction": dict(result["prediction"]),
        "candidate_scores": list(result["candidate_scores"]),
    }


def aggregate_private_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError(
            f"private prediction count mismatch: {len(predictions)} != {len(rows)}"
        )
    by_target: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "exact": 0}
    )
    predicted_pairs: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    scores: list[float] = []
    exact = 0
    unsafe_ground = 0
    for row, prediction_row in zip(rows, predictions, strict=True):
        expected_prediction_id = str(row["id"]).replace(
            "sealed_candidate_routing::", "sealed_prediction::", 1
        )
        if prediction_row.get("id") != expected_prediction_id:
            raise ValueError(
                "private prediction id mismatch: "
                f"{prediction_row.get('id')} != {expected_prediction_id}"
            )
        if prediction_row.get("source_candidate_routing_id") != row["id"]:
            raise ValueError("private prediction source routing id mismatch")
        if prediction_row.get("case_id") != row["source_manifest_case_id"]:
            raise ValueError("private prediction case id mismatch")
        if prediction_row.get("target_pair") != row["target_pair"]:
            raise ValueError("private prediction target pair mismatch")
        prediction = prediction_row.get("prediction")
        if not isinstance(prediction, Mapping):
            raise ValueError("private prediction payload is not a mapping")
        selected_pair = str(prediction.get("selected_pair"))
        predicted_action = str(prediction.get("action"))
        predicted_status = str(prediction.get("evidence_status"))
        if selected_pair not in CANDIDATE_PAIRS:
            raise ValueError(f"private prediction selected invalid pair: {selected_pair}")
        if selected_pair != f"{predicted_action}/{predicted_status}":
            raise ValueError("private prediction selected pair disagrees with fields")
        scored = score_component_prediction(
            {
                "component": "enum_action",
                "target_output": {
                    "action": row["target_output"]["action"],
                    "evidence_status": row["target_output"]["evidence_status"],
                },
                "source_manifest_case_id": row["source_manifest_case_id"],
                "case_family": None,
            },
            {
                "id": prediction_row["id"],
                "split": "sealed",
                "prediction": {
                    "action": predicted_action,
                    "evidence_status": predicted_status,
                },
            },
        )
        target_pair = row["target_pair"]
        predicted_pair = selected_pair
        is_exact = predicted_pair == target_pair
        exact += int(is_exact)
        scores.append(float(scored["score"]))
        by_target[target_pair]["rows"] += 1
        by_target[target_pair]["exact"] += int(is_exact)
        predicted_pairs[predicted_pair] += 1
        for violation in scored["violations"]:
            violations[violation] += 1
        if predicted_pair == "ground/supported" and target_pair != predicted_pair:
            unsafe_ground += 1
    return {
        "rows": len(rows),
        "exact": exact,
        "accuracy": round(exact / len(rows), 6) if rows else 0.0,
        "mean_component_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "by_target_pair": dict(sorted(by_target.items())),
        "by_predicted_pair": dict(sorted(predicted_pairs.items())),
        "violations": dict(sorted(violations.items())),
        "unsafe_ground_supported": unsafe_ground,
    }


def write_private_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require_external_private_path(path, role="private sealed predictions")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    output.chmod(0o600)


def load_private_jsonl(path: str | Path) -> list[dict[str, Any]]:
    require_external_private_path(path, role="private sealed predictions")
    output: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"private predictions row {line_no} is not an object")
            output.append(value)
    return output


def assert_compact_public_safe(report: Mapping[str, Any]) -> None:
    stack: list[tuple[str, Any]] = [("$", report)]
    while stack:
        prefix, value = stack.pop()
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in RAW_PUBLIC_FORBIDDEN_KEYS:
                    raise ValueError(f"compact result contains forbidden key: {prefix}.{key}")
                stack.append((f"{prefix}.{key}", item))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                stack.append((f"{prefix}[{index}]", item))


def compact_result(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    sealed_manifest_path: str | Path,
    trainable_state_path: str | Path,
    device_class: str,
    trainable_params: int,
    postprocessing_recovery: bool,
) -> dict[str, Any]:
    policy = freeze.get("policy", {})
    summary = aggregate_private_predictions(candidate_rows, predictions)
    runtime_exact = sum(
        row["runtime_pair"] == row["target_pair"] for row in candidate_rows
    )
    report = {
        "dataset": DATASET,
        "dry_run": False,
        "evaluation_scope": EVAL_SCOPE,
        "freeze_id": freeze.get("freeze_id"),
        "model": {
            "model_id": policy.get("model_id"),
            "revision": policy.get("model_revision"),
            "device_class": device_class,
            "trainable_params_loaded": trainable_params,
        },
        "sealed_summary": summary,
        "baselines": {
            "runtime_oracle_exact": runtime_exact,
            "runtime_oracle_rows": len(candidate_rows),
            "best_static_single_pair_exact": max(
                Counter(row["target_pair"] for row in candidate_rows).values()
            ),
        },
        "decision": {
            "candidate_policy_beats_static_prior": (
                summary["exact"]
                > max(Counter(row["target_pair"] for row in candidate_rows).values())
            ),
            "candidate_policy_matches_runtime_oracle": summary["exact"] == runtime_exact,
            "unsafe_ground_supported_zero": summary["unsafe_ground_supported"] == 0,
            "ready_for_dpo_rlvr": False,
            "ready_for_hugging_face_publication": False,
        },
        "input_commitments": {
            "sealed_manifest_sha256": sha256_file(sealed_manifest_path),
            "trainable_state_sha256": sha256_file(trainable_state_path),
            "policy_freeze_sha256": sha256_file(
                ROOT / "post_training/stage_a_candidate_routing_policy_freeze_2026-07-23.json"
            )
            if (
                ROOT
                / "post_training/stage_a_candidate_routing_policy_freeze_2026-07-23.json"
            ).exists()
            else None,
            "executed_scoring_evaluator_sha256": (
                freeze.get("frozen_artifacts", {})
                .get("sealed_evaluator", {})
                .get("sha256")
            ),
        },
        "postprocessing": {
            "recovered_from_completed_private_predictions": postprocessing_recovery,
            "model_rescored_during_recovery": False,
            "finalizer_sha256": sha256_file(__file__),
        },
        "privacy_boundary": {
            "hidden_labels_used_to_construct_oracle_tool_state": True,
            "hidden_labels_visible_to_model": False,
            "row_level_labels_emitted": False,
            "row_level_predictions_emitted": False,
            "raw_candidate_scores_emitted": False,
            "private_predictions_committed": False,
            "actual_tool_query_evaluated": False,
            "live_tool_execution_evaluated": False,
        },
    }
    assert_compact_public_safe(report)
    return report


def run_full(
    *,
    rows: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    sealed_manifest_path: str | Path,
    trainable_state_path: str | Path,
    private_predictions_path: str | Path,
    compact_output_path: str | Path,
    lock_path: str | Path,
    device: str,
    allow_download: bool,
) -> dict[str, Any]:
    policy = freeze.get("policy", {})
    model_id = str(policy.get("model_id"))
    train_last_layers = int(policy.get("train_last_layers"))
    max_length = int(policy.get("max_length"))
    candidate_rows = build_sealed_candidate_rows(rows)
    model, tokenizer, selected_device, trainable_params = load_frozen_model(
        model_id=model_id,
        trainable_state_path=trainable_state_path,
        train_last_layers=train_last_layers,
        device=device,
        allow_download=allow_download,
    )
    create_one_time_lock(
        lock_path,
        sealed_sha256=sha256_file(sealed_manifest_path),
        freeze_id=str(freeze.get("freeze_id")),
    )

    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(candidate_rows):
        result = score_candidate_outputs(
            model,
            tokenizer,
            row,
            device=selected_device,
            max_length=max_length,
        )
        predictions.append(
            private_prediction_row(row=row, result=result, ordinal=index)
        )
        print(f"[sealed {index + 1}/{len(candidate_rows)}] scored", flush=True)
    write_private_jsonl(private_predictions_path, predictions)
    report = compact_result(
        candidate_rows=candidate_rows,
        predictions=predictions,
        freeze=freeze,
        sealed_manifest_path=sealed_manifest_path,
        trainable_state_path=trainable_state_path,
        device_class="cuda" if selected_device.startswith("cuda") else selected_device,
        trainable_params=trainable_params,
        postprocessing_recovery=False,
    )
    compact_path = Path(compact_output_path)
    compact_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    compact_path.chmod(0o600)
    complete_one_time_lock(lock_path, compact_sha256=sha256_file(compact_path))
    return report


def finalize_existing_predictions(
    *,
    rows: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    sealed_manifest_path: str | Path,
    trainable_state_path: str | Path,
    private_predictions_path: str | Path,
    compact_output_path: str | Path,
    lock_path: str | Path,
) -> dict[str, Any]:
    lock = load_json(lock_path)
    if lock.get("status") != "started":
        raise ValueError("recovery requires an existing started one-time lock")
    if lock.get("freeze_id") != freeze.get("freeze_id"):
        raise ValueError("one-time lock freeze_id mismatch")
    if lock.get("sealed_manifest_sha256") != sha256_file(sealed_manifest_path):
        raise ValueError("one-time lock sealed manifest hash mismatch")
    candidate_rows = build_sealed_candidate_rows(rows)
    predictions = load_private_jsonl(private_predictions_path)
    report = compact_result(
        candidate_rows=candidate_rows,
        predictions=predictions,
        freeze=freeze,
        sealed_manifest_path=sealed_manifest_path,
        trainable_state_path=trainable_state_path,
        device_class="cuda",
        trainable_params=int(freeze.get("policy", {}).get("trainable_params", 0)),
        postprocessing_recovery=True,
    )
    compact_path = Path(compact_output_path)
    compact_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    compact_path.chmod(0o600)
    complete_one_time_lock(lock_path, compact_sha256=sha256_file(compact_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-manifest", required=True)
    parser.add_argument(
        "--sealed-commitment",
        default="post_training/stage_a_sealed_extension_commitment_2026-07-10.json",
    )
    parser.add_argument(
        "--policy-freeze",
        default="post_training/stage_a_candidate_routing_policy_freeze_2026-07-23.json",
    )
    parser.add_argument("--trainable-state", required=True)
    parser.add_argument("--private-predictions-out", required=True)
    parser.add_argument("--compact-out", required=True)
    parser.add_argument("--evaluation-lock", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-existing-private-predictions", action="store_true")
    args = parser.parse_args()

    rows, commitment, freeze, issues = validate_inputs(
        sealed_manifest_path=args.sealed_manifest,
        commitment_path=args.sealed_commitment,
        freeze_path=args.policy_freeze,
        trainable_state_path=args.trainable_state,
    )
    if issues and not args.finalize_existing_private_predictions:
        raise SystemExit("Sealed evaluation validation failed:\n- " + "\n- ".join(issues))
    if args.finalize_existing_private_predictions:
        non_evaluator_issues = [
            issue for issue in issues if issue != "sealed_evaluator_sha256_mismatch"
        ]
        if non_evaluator_issues:
            raise SystemExit(
                "Sealed recovery validation failed:\n- "
                + "\n- ".join(non_evaluator_issues)
            )
        report = finalize_existing_predictions(
            rows=rows,
            freeze=freeze,
            sealed_manifest_path=args.sealed_manifest,
            trainable_state_path=args.trainable_state,
            private_predictions_path=args.private_predictions_out,
            compact_output_path=args.compact_out,
            lock_path=args.evaluation_lock,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        report = dry_run_report(
            rows=rows,
            commitment=commitment,
            freeze=freeze,
            sealed_manifest_path=args.sealed_manifest,
            trainable_state_path=args.trainable_state,
            issues=issues,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if not args.allow_model_load:
        raise SystemExit(
            "Full sealed evaluation requires --allow-model-load; use --dry-run first."
        )
    report = run_full(
        rows=rows,
        freeze=freeze,
        sealed_manifest_path=args.sealed_manifest,
        trainable_state_path=args.trainable_state,
        private_predictions_path=args.private_predictions_out,
        compact_output_path=args.compact_out,
        lock_path=args.evaluation_lock,
        device=args.device,
        allow_download=args.allow_download,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
