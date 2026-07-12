#!/usr/bin/env python3
"""Run Stage A enum corrective SFT/margin smoke experiments.

This is the cluster-oriented follow-up to the enum corrective contrast-pair
export. It trains only on chosen enum/action corrective pairs and evaluates
held-out pairs by teacher-forced chosen-vs-rejected likelihood margin.

The `--dry-run` path validates artifacts and split boundaries without loading
model weights. Full mode requires `--allow-model-load` and is intended for
Cayuga/Expanse or another GPU environment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.export_stage_a_enum_corrective_pairs import (  # noqa: E402
    COLLAPSE_OUTPUT,
    COMPONENT,
    DATASET as PAIR_DATASET,
    FAILURE_MODE,
    PROMPT_CONTRACT,
)
from post_training.export_stage_a_enum_action_contrast_pairs import (  # noqa: E402
    CANDIDATE_POLICY as ACTION_CONTRAST_CANDIDATE_POLICY,
    DATASET as ACTION_CONTRAST_PAIR_DATASET,
    FAILURE_MODE as ACTION_CONTRAST_FAILURE_MODE,
)
from post_training.generate_stage_a_predictions import disable_transformers_torchvision_probe  # noqa: E402
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    enum_action_candidate_outputs,
    prompt_messages_from_row,
    score_candidate_target,
    score_enum_action_candidates,
)
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
from post_training.validate_post_training_data import (  # noqa: E402
    validate_stage_a_enum_action_contrast_pairs,
    validate_stage_a_enum_corrective_pairs,
)


DATASET = "negbiodb_ct_stage_a_enum_corrective_sft_smoke_v1"
MARGIN_DATASET = "negbiodb_ct_stage_a_enum_corrective_margin_rows_v1"
MARGIN_DELTA_DATASET = "negbiodb_ct_stage_a_enum_corrective_margin_delta_v1"
CANDIDATE_SELECTION_DATASET = "negbiodb_ct_stage_a_enum_corrective_candidate_selection_v1"
CANDIDATE_EVAL_DATASET = "negbiodb_ct_stage_a_enum_corrective_candidate_eval_v1"
TARGET_FORMATS = ("full", "action_only")
ENUM_CANDIDATE_POLICIES = ("all_valid_pairs", "pair_observed_outputs")


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def pair_case_id(row: Mapping[str, Any]) -> str:
    value = row.get("source_manifest_case_id") or row.get("source_component_target_id") or row.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Corrective pair is missing case id: {row.get('id')!r}")
    return value


def pair_output_text(output: Mapping[str, Any]) -> str:
    return json.dumps(dict(output), sort_keys=True)


def chosen_output_from_pair(row: Mapping[str, Any]) -> dict[str, Any]:
    output = row.get("chosen_output")
    if not isinstance(output, Mapping):
        raise ValueError(f"{row.get('id')} has no chosen_output object")
    return dict(output)


def rejected_output_from_pair(row: Mapping[str, Any]) -> dict[str, Any]:
    output = row.get("rejected_output")
    if not isinstance(output, Mapping):
        raise ValueError(f"{row.get('id')} has no rejected_output object")
    return dict(output)


def project_output_for_target_format(output: Mapping[str, Any], target_format: str) -> dict[str, Any]:
    if target_format == "full":
        return dict(output)
    if target_format == "action_only":
        action = output.get("action")
        if not isinstance(action, str) or not action:
            raise ValueError(f"Cannot project output without action field: {output!r}")
        return {"action": action}
    raise ValueError(f"Unknown target_format: {target_format}")


def chosen_output_for_target_format(row: Mapping[str, Any], target_format: str) -> dict[str, Any]:
    return project_output_for_target_format(chosen_output_from_pair(row), target_format)


def rejected_output_for_target_format(row: Mapping[str, Any], target_format: str) -> dict[str, Any]:
    return project_output_for_target_format(rejected_output_from_pair(row), target_format)


def manifest_pair_dataset(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("pair_dataset") or PAIR_DATASET)


def manifest_failure_mode(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("failure_mode") or FAILURE_MODE)


def manifest_candidate_policy(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("candidate_policy") or "train_observed_pairs")


def row_value_summary(rows: Sequence[Mapping[str, Any]], key: str, default: str) -> str:
    values = sorted({str(row.get(key)) for row in rows if row.get(key) is not None})
    if len(values) == 1:
        return values[0]
    if not values:
        return default
    return "mixed"


def encode_chosen_pair(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_length: int,
    target_format: str = "full",
) -> dict[str, Any]:
    return encode_pair_output(
        tokenizer,
        row,
        chosen_output_for_target_format(row, target_format),
        max_length=max_length,
    )


def encode_pair_output(
    tokenizer: Any,
    row: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    max_length: int,
) -> dict[str, Any]:
    import torch

    prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_row(row))
    target = pair_output_text(output) + tokenizer.eos_token
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


def encode_pairwise_margin_pair(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_length: int,
    target_format: str,
) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "chosen_pair": row.get("chosen_pair"),
        "rejected_pair": row.get("rejected_pair"),
        "chosen": encode_pair_output(
            tokenizer,
            row,
            chosen_output_for_target_format(row, target_format),
            max_length=max_length,
        ),
        "rejected": encode_pair_output(
            tokenizer,
            row,
            rejected_output_for_target_format(row, target_format),
            max_length=max_length,
        ),
    }


def candidate_index_for_row(
    row: Mapping[str, Any],
    candidate_outputs: Sequence[Mapping[str, Any]],
) -> int:
    target_key = pair_output_key(chosen_output_for_target_format(row, "full"))
    for index, candidate in enumerate(candidate_outputs):
        if pair_output_key(candidate) == target_key:
            return index
    raise ValueError(f"target enum pair not in candidate policy for {row.get('id')}: {target_key!r}")


def encode_candidate_ce_set(
    tokenizer: Any,
    row: Mapping[str, Any],
    candidate_outputs: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "chosen_pair": row.get("chosen_pair"),
        "expected_output": chosen_output_for_target_format(row, "full"),
        "expected_index": candidate_index_for_row(row, candidate_outputs),
        "candidate_outputs": [dict(candidate) for candidate in candidate_outputs],
        "candidates": [
            encode_pair_output(
                tokenizer,
                row,
                candidate,
                max_length=max_length,
            )
            for candidate in candidate_outputs
        ],
    }


def pairwise_margin_loss_from_logps(chosen_logps: Any, rejected_logps: Any, *, margin: float) -> Any:
    if margin < 0:
        raise ValueError("pairwise margin must be non-negative")
    import torch.nn.functional as F

    desired = chosen_logps.new_tensor(float(margin))
    return F.relu(desired - (chosen_logps - rejected_logps)).mean()


def candidate_ce_loss_from_logps(logps: Any, expected_index: int) -> Any:
    import torch
    import torch.nn.functional as F

    if expected_index < 0 or expected_index >= int(logps.shape[-1]):
        raise ValueError(f"expected_index out of range: {expected_index}")
    target = torch.tensor([expected_index], dtype=torch.long, device=logps.device)
    return F.cross_entropy(logps.unsqueeze(0), target)


def candidate_field_values(
    candidate_outputs: Sequence[Mapping[str, Any]],
    field_name: str,
) -> list[str]:
    values: list[str] = []
    for candidate in candidate_outputs:
        value = candidate.get(field_name)
        if value is None:
            continue
        key = str(value)
        if key not in values:
            values.append(key)
    return values


def candidate_field_ce_loss_from_logps(
    logps: Any,
    candidate_outputs: Sequence[Mapping[str, Any]],
    expected_output: Mapping[str, Any],
) -> Any:
    import torch
    import torch.nn.functional as F

    losses = []
    for field_name in ("action", "evidence_status"):
        values = candidate_field_values(candidate_outputs, field_name)
        expected_value = expected_output.get(field_name)
        if expected_value is None:
            raise ValueError(f"expected output missing {field_name}")
        expected_key = str(expected_value)
        if expected_key not in values:
            raise ValueError(f"expected {field_name} not in candidate policy: {expected_key!r}")
        field_logits = []
        for value in values:
            indices = [
                index
                for index, candidate in enumerate(candidate_outputs)
                if str(candidate.get(field_name)) == value
            ]
            if not indices:
                continue
            field_logits.append(torch.logsumexp(logps[indices], dim=0))
        target = torch.tensor([values.index(expected_key)], dtype=torch.long, device=logps.device)
        losses.append(F.cross_entropy(torch.stack(field_logits).unsqueeze(0), target))
    return torch.stack(losses).mean()


def validate_corrective_artifacts(
    all_rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    pair_dataset = manifest_pair_dataset(manifest)
    if pair_dataset == ACTION_CONTRAST_PAIR_DATASET:
        issues = validate_stage_a_enum_action_contrast_pairs(all_rows, train_rows, heldout_rows, manifest)
    else:
        issues = validate_stage_a_enum_corrective_pairs(all_rows, train_rows, heldout_rows, manifest)
    if not train_rows:
        issues.append("empty_train_pairs")
    if not heldout_rows:
        issues.append("empty_heldout_pairs")
    expected_failure_mode = manifest_failure_mode(manifest)
    expected_candidate_policy = manifest_candidate_policy(manifest)
    for split, rows in (("train", train_rows), ("heldout", heldout_rows)):
        for row in rows:
            row_id = row.get("id")
            if row.get("component") != COMPONENT:
                issues.append(f"{row_id}:{split}_wrong_component")
            if row.get("dataset") != pair_dataset:
                issues.append(f"{row_id}:{split}_unexpected_dataset")
            if row.get("failure_mode") != expected_failure_mode:
                issues.append(f"{row_id}:{split}_wrong_failure_mode")
            if row.get("prompt_contract") != PROMPT_CONTRACT:
                issues.append(f"{row_id}:{split}_wrong_prompt_contract")
            if row.get("candidate_policy") != expected_candidate_policy:
                issues.append(f"{row_id}:{split}_wrong_candidate_policy")
            if pair_dataset == PAIR_DATASET and rejected_output_from_pair(row) != COLLAPSE_OUTPUT:
                issues.append(f"{row_id}:{split}_rejected_not_ground_supported")
    return issues


def count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def pair_output_key(output: Mapping[str, Any]) -> tuple[str, str]:
    action = output.get("action")
    evidence_status = output.get("evidence_status")
    if not isinstance(action, str) or not isinstance(evidence_status, str):
        raise ValueError(f"Expected full enum action/status output, got: {output!r}")
    return action, evidence_status


def enum_candidate_outputs_for_policy(
    policy: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if policy == "all_valid_pairs":
        return enum_action_candidate_outputs()
    if policy != "pair_observed_outputs":
        raise ValueError(f"unknown enum candidate policy: {policy}")

    outputs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        for output in (chosen_output_from_pair(row), rejected_output_from_pair(row)):
            key = pair_output_key(output)
            if key not in seen:
                seen.add(key)
                outputs.append({"action": key[0], "evidence_status": key[1]})
    if not outputs:
        raise ValueError("pair_observed_outputs selected no enum candidates")
    return outputs


def parse_focus_chosen_pairs(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    pairs = tuple(item.strip() for item in re.split(r"[,:]", value) if item.strip())
    return tuple(dict.fromkeys(pairs))


def expand_training_rows(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    focus_chosen_pairs: Sequence[str],
    focus_repeat: int,
    focus_only: bool,
) -> list[dict[str, Any]]:
    if focus_repeat < 1:
        raise ValueError("focus_repeat must be >= 1")
    focus = set(focus_chosen_pairs)
    if not focus:
        return [dict(row) for row in train_rows]

    available = {str(row.get("chosen_pair")) for row in train_rows}
    missing = sorted(focus - available)
    if missing:
        raise ValueError("unknown focus_chosen_pairs: " + ", ".join(missing))

    focused_rows = [dict(row) for row in train_rows if str(row.get("chosen_pair")) in focus]
    if not focused_rows:
        raise ValueError("focus_chosen_pairs selected no train rows")

    training_rows = focused_rows[:] if focus_only else [dict(row) for row in train_rows]
    for _ in range(focus_repeat - 1):
        training_rows.extend(dict(row) for row in focused_rows)
    return training_rows


def margin_row_for_pair(
    row: Mapping[str, Any],
    *,
    run_id: str,
    model_id: str,
    chosen_score: float,
    rejected_score: float,
    score_label: str = "trained_heldout",
    target_format: str = "full",
) -> dict[str, Any]:
    margin = round(float(chosen_score) - float(rejected_score), 6)
    passed = margin > 0.0
    return {
        "id": f"{run_id}::{score_label}::{row['id']}",
        "dataset": MARGIN_DATASET,
        "source": "stage_a_enum_corrective_sft_smoke",
        "run_id": run_id,
        "model": model_id,
        "score_label": score_label,
        "source_enum_corrective_pair_id": row["id"],
        "source_component_target_id": row.get("source_component_target_id"),
        "case_id": pair_case_id(row),
        "component": COMPONENT,
        "failure_mode": row.get("failure_mode", FAILURE_MODE),
        "contrast_axis": row.get("contrast_axis"),
        "candidate_policy": row.get("candidate_policy"),
        "target_format": target_format,
        "prompt_contract": PROMPT_CONTRACT,
        "split": row.get("split"),
        "case_family": row.get("case_family"),
        "chosen_pair": row.get("chosen_pair"),
        "rejected_pair": row.get("rejected_pair"),
        "chosen_output": chosen_output_for_target_format(row, target_format),
        "rejected_output": rejected_output_for_target_format(row, target_format),
        "source_chosen_output": chosen_output_from_pair(row),
        "source_rejected_output": rejected_output_from_pair(row),
        "chosen_score": round(float(chosen_score), 6),
        "rejected_score": round(float(rejected_score), 6),
        "margin": margin,
        "passed": passed,
        "violations": [] if passed else ["chosen_not_above_rejected"],
    }


def summarize_margin_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "heldout_pairs": 0,
            "margin_wins": 0,
            "margin_accuracy": 0.0,
            "mean_chosen_score": 0.0,
            "mean_rejected_score": 0.0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "violations": {},
            "by_chosen_pair": {},
        }

    violations = Counter(violation for row in rows for violation in row.get("violations", ()))
    by_chosen_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chosen_pair[str(row.get("chosen_pair"))].append(row)

    def mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    margins = [float(row.get("margin", 0.0)) for row in rows]
    return {
        "heldout_pairs": len(rows),
        "margin_wins": sum(1 for row in rows if row.get("passed")),
        "margin_accuracy": round(sum(1 for row in rows if row.get("passed")) / len(rows), 3),
        "mean_chosen_score": mean([float(row.get("chosen_score", 0.0)) for row in rows]),
        "mean_rejected_score": mean([float(row.get("rejected_score", 0.0)) for row in rows]),
        "mean_margin": mean(margins),
        "min_margin": round(min(margins), 6),
        "violations": dict(sorted(violations.items())),
        "by_chosen_pair": {
            pair: {
                "pairs": len(items),
                "margin_wins": sum(1 for item in items if item.get("passed")),
                "mean_margin": mean([float(item.get("margin", 0.0)) for item in items]),
                "min_margin": round(min(float(item.get("margin", 0.0)) for item in items), 6),
            }
            for pair, items in sorted(by_chosen_pair.items())
        },
    }


def margin_delta_row(
    base_row: Mapping[str, Any],
    trained_row: Mapping[str, Any],
) -> dict[str, Any]:
    pair_id = str(base_row.get("source_enum_corrective_pair_id"))
    if pair_id != str(trained_row.get("source_enum_corrective_pair_id")):
        raise ValueError(f"Cannot compare different corrective pairs: {pair_id} vs {trained_row.get('source_enum_corrective_pair_id')}")
    base_margin = float(base_row.get("margin", 0.0))
    trained_margin = float(trained_row.get("margin", 0.0))
    base_passed = bool(base_row.get("passed"))
    trained_passed = bool(trained_row.get("passed"))
    if not base_passed and trained_passed:
        outcome = "newly_won"
    elif base_passed and not trained_passed:
        outcome = "newly_lost"
    elif trained_passed:
        outcome = "remained_won"
    else:
        outcome = "remained_lost"
    return {
        "id": f"{trained_row.get('run_id')}::delta::{pair_id}",
        "dataset": MARGIN_DELTA_DATASET,
        "source": "stage_a_enum_corrective_sft_smoke",
        "run_id": trained_row.get("run_id"),
        "model": trained_row.get("model"),
        "source_enum_corrective_pair_id": pair_id,
        "case_id": trained_row.get("case_id"),
        "component": COMPONENT,
        "failure_mode": trained_row.get("failure_mode", FAILURE_MODE),
        "contrast_axis": trained_row.get("contrast_axis"),
        "candidate_policy": trained_row.get("candidate_policy"),
        "target_format": trained_row.get("target_format", "full"),
        "prompt_contract": PROMPT_CONTRACT,
        "split": trained_row.get("split"),
        "case_family": trained_row.get("case_family"),
        "chosen_pair": trained_row.get("chosen_pair"),
        "rejected_pair": trained_row.get("rejected_pair"),
        "base_margin": round(base_margin, 6),
        "trained_margin": round(trained_margin, 6),
        "margin_delta": round(trained_margin - base_margin, 6),
        "base_passed": base_passed,
        "trained_passed": trained_passed,
        "outcome": outcome,
    }


def build_margin_delta_report(
    *,
    run_id: str,
    model_id: str,
    base_rows: Sequence[Mapping[str, Any]],
    trained_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_by_pair = {str(row.get("source_enum_corrective_pair_id")): row for row in base_rows}
    trained_by_pair = {str(row.get("source_enum_corrective_pair_id")): row for row in trained_rows}
    row_reports = []
    missing_base = sorted(set(trained_by_pair) - set(base_by_pair))
    missing_trained = sorted(set(base_by_pair) - set(trained_by_pair))
    for pair_id in sorted(set(base_by_pair) & set(trained_by_pair)):
        row_reports.append(margin_delta_row(base_by_pair[pair_id], trained_by_pair[pair_id]))
    return {
        "dataset": MARGIN_DELTA_DATASET,
        "run_id": run_id,
        "model": model_id,
        "component": COMPONENT,
        "failure_mode": row_value_summary(trained_rows, "failure_mode", FAILURE_MODE),
        "failure_modes": count_by(trained_rows, "failure_mode"),
        "contrast_axes": count_by(trained_rows, "contrast_axis"),
        "target_format": row_value_summary(trained_rows, "target_format", "full"),
        "prompt_contract": PROMPT_CONTRACT,
        "summary": summarize_margin_delta_rows(row_reports),
        "missing_base_pair_ids": missing_base,
        "missing_trained_pair_ids": missing_trained,
        "rows": row_reports,
        "boundary": (
            "Margin delta compares base-model held-out contrast margins with "
            "post-SFT held-out contrast margins. This is a diagnostic for "
            "corrective SFT movement, not DPO/RLVR."
        ),
    }


def summarize_margin_delta_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "pairs": 0,
            "base_margin_wins": 0,
            "trained_margin_wins": 0,
            "newly_won": 0,
            "newly_lost": 0,
            "mean_base_margin": 0.0,
            "mean_trained_margin": 0.0,
            "mean_margin_delta": 0.0,
            "min_margin_delta": 0.0,
            "outcomes": {},
            "by_chosen_pair": {},
        }

    def mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    outcomes = Counter(str(row.get("outcome")) for row in rows)
    by_chosen_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chosen_pair[str(row.get("chosen_pair"))].append(row)
    deltas = [float(row.get("margin_delta", 0.0)) for row in rows]
    return {
        "pairs": len(rows),
        "base_margin_wins": sum(1 for row in rows if row.get("base_passed")),
        "trained_margin_wins": sum(1 for row in rows if row.get("trained_passed")),
        "newly_won": outcomes.get("newly_won", 0),
        "newly_lost": outcomes.get("newly_lost", 0),
        "mean_base_margin": mean([float(row.get("base_margin", 0.0)) for row in rows]),
        "mean_trained_margin": mean([float(row.get("trained_margin", 0.0)) for row in rows]),
        "mean_margin_delta": mean(deltas),
        "min_margin_delta": round(min(deltas), 6),
        "outcomes": dict(sorted(outcomes.items())),
        "by_chosen_pair": {
            pair: {
                "pairs": len(items),
                "base_margin_wins": sum(1 for item in items if item.get("base_passed")),
                "trained_margin_wins": sum(1 for item in items if item.get("trained_passed")),
                "mean_margin_delta": mean([float(item.get("margin_delta", 0.0)) for item in items]),
                "min_margin_delta": round(min(float(item.get("margin_delta", 0.0)) for item in items), 6),
            }
            for pair, items in sorted(by_chosen_pair.items())
        },
    }


def build_margin_report(
    *,
    run_id: str,
    model_id: str,
    rows: Sequence[Mapping[str, Any]],
    score_label: str = "trained_heldout",
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "dry_run": False,
        "run_id": run_id,
        "model": model_id,
        "score_label": score_label,
        "component": COMPONENT,
        "failure_mode": row_value_summary(rows, "failure_mode", FAILURE_MODE),
        "failure_modes": count_by(rows, "failure_mode"),
        "contrast_axes": count_by(rows, "contrast_axis"),
        "target_format": row_value_summary(rows, "target_format", "full"),
        "prompt_contract": PROMPT_CONTRACT,
        "heldout_pairs": len(rows),
        "summary": summarize_margin_rows(rows),
        "rows": list(rows),
        "boundary": (
            "Enum corrective SFT smoke scores held-out chosen-vs-rejected "
            "teacher-forced margins. This is not DPO/RLVR and does not score "
            "free-form explanation quality."
        ),
    }


def dry_run_report(
    *,
    model: str,
    pairs: str | Path,
    train_pairs: str | Path,
    heldout_pairs: str | Path,
    manifest: str | Path,
    train_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    issues: Sequence[str],
    focus_chosen_pairs: Sequence[str],
    focus_repeat: int,
    focus_only: bool,
    target_format: str,
    pairwise_margin_weight: float,
    pairwise_margin: float,
    margin_logprob_mode: str,
    candidate_ce_weight: float,
    candidate_ce_mode: str,
    candidate_ce_logprob_mode: str,
    score_base_enum_candidates: bool,
    score_enum_candidates: bool,
    enum_candidate_policy: str,
) -> dict[str, Any]:
    selected_rows = [dict(row) for row in train_rows] + [dict(row) for row in heldout_rows]
    enum_candidate_outputs = enum_candidate_outputs_for_policy(enum_candidate_policy, train_rows)
    return {
        "dataset": DATASET,
        "dry_run": True,
        "model": model,
        "component": COMPONENT,
        "failure_mode": row_value_summary(selected_rows, "failure_mode", FAILURE_MODE),
        "failure_modes": count_by(selected_rows, "failure_mode"),
        "candidate_policies": count_by(selected_rows, "candidate_policy"),
        "contrast_axes": count_by(selected_rows, "contrast_axis"),
        "target_format": target_format,
        "pairwise_margin_weight": pairwise_margin_weight,
        "pairwise_margin": pairwise_margin,
        "margin_logprob_mode": margin_logprob_mode,
        "candidate_ce_weight": candidate_ce_weight,
        "candidate_ce_mode": candidate_ce_mode,
        "candidate_ce_logprob_mode": candidate_ce_logprob_mode,
        "score_base_enum_candidates": score_base_enum_candidates,
        "score_enum_candidates": score_enum_candidates,
        "enum_candidate_policy": enum_candidate_policy,
        "enum_candidate_space_size": len(enum_candidate_outputs),
        "enum_candidate_outputs": enum_candidate_outputs,
        "prompt_contract": PROMPT_CONTRACT,
        "pairs": str(pairs),
        "train_pairs": str(train_pairs),
        "heldout_pairs": str(heldout_pairs),
        "manifest": str(manifest),
        "train_examples": len(train_rows),
        "training_examples": len(training_rows),
        "heldout_examples": len(heldout_rows),
        "focus_chosen_pairs": list(focus_chosen_pairs),
        "focus_repeat": focus_repeat,
        "focus_only": focus_only,
        "train_by_chosen_pair": count_by(train_rows, "chosen_pair"),
        "training_by_chosen_pair": count_by(training_rows, "chosen_pair"),
        "heldout_by_chosen_pair": count_by(heldout_rows, "chosen_pair"),
        "train_case_ids": [pair_case_id(row) for row in train_rows],
        "heldout_case_ids": [pair_case_id(row) for row in heldout_rows],
        "issues": list(issues),
        "boundary": (
            "Dry run validates enum corrective pair artifacts and split "
            "boundaries without loading model weights or running local heavy compute."
        ),
    }


def compact_candidate_scores(scores: Sequence[Mapping[str, Any]], *, top_k: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "candidate": dict(score.get("candidate", {})),
            "score": round(float(score.get("score", 0.0)), 6),
        }
        for score in scores[:top_k]
    ]


def find_candidate_score(
    scores: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> tuple[int | None, float | None]:
    target_key = pair_output_key(target)
    for index, score in enumerate(scores, start=1):
        candidate = score.get("candidate")
        if isinstance(candidate, Mapping) and pair_output_key(candidate) == target_key:
            return index, float(score.get("score", 0.0))
    return None, None


def candidate_selection_row_for_pair(
    row: Mapping[str, Any],
    *,
    run_id: str,
    model_id: str,
    score_label: str,
    enum_candidate_policy: str,
    candidate_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target = chosen_output_for_target_format(row, "full")
    winner = dict(candidate_scores[0]["candidate"]) if candidate_scores else None
    winner_score = float(candidate_scores[0]["score"]) if candidate_scores else None
    gold_rank, gold_score = find_candidate_score(candidate_scores, target)
    exact_top1 = winner is not None and pair_output_key(winner) == pair_output_key(target)
    margin = None
    if winner_score is not None and gold_score is not None:
        margin = round(winner_score - gold_score, 6)
    return {
        "id": f"{run_id}::{score_label}::{row['id']}",
        "dataset": CANDIDATE_SELECTION_DATASET,
        "source": "stage_a_enum_corrective_sft_smoke",
        "run_id": run_id,
        "model": model_id,
        "score_label": score_label,
        "source_enum_corrective_pair_id": row["id"],
        "source_component_target_id": row.get("source_component_target_id"),
        "case_id": pair_case_id(row),
        "component": COMPONENT,
        "failure_mode": row.get("failure_mode", FAILURE_MODE),
        "contrast_axis": row.get("contrast_axis"),
        "candidate_policy": row.get("candidate_policy"),
        "enum_candidate_policy": enum_candidate_policy,
        "target_format": "full",
        "prompt_contract": PROMPT_CONTRACT,
        "split": row.get("split"),
        "case_family": row.get("case_family"),
        "chosen_pair": row.get("chosen_pair"),
        "rejected_pair": row.get("rejected_pair"),
        "target_output": target,
        "prediction": winner,
        "raw_output": json.dumps(winner, sort_keys=True) if winner is not None else None,
        "candidate_scores": [dict(score) for score in candidate_scores],
        "gold_rank": gold_rank,
        "gold_score": round(gold_score, 6) if gold_score is not None else None,
        "top_score": round(winner_score, 6) if winner_score is not None else None,
        "top_gold_margin": margin,
        "exact_top1": exact_top1,
        "passed": exact_top1,
        "violations": [] if exact_top1 else ["gold_not_top_candidate"],
    }


def summarize_candidate_selection_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "heldout_pairs": 0,
            "exact_top1": 0,
            "candidate_accuracy": 0.0,
            "mean_gold_rank": None,
            "mean_top_gold_margin": None,
            "violations": {},
            "top_pair_counts": {},
            "by_chosen_pair": {},
        }

    violations = Counter(violation for row in rows for violation in row.get("violations", ()))
    gold_ranks = [int(row["gold_rank"]) for row in rows if isinstance(row.get("gold_rank"), int)]
    margins = [float(row["top_gold_margin"]) for row in rows if row.get("top_gold_margin") is not None]
    top_pair_counts = Counter(
        f"{prediction.get('action')}/{prediction.get('evidence_status')}"
        for row in rows
        if isinstance((prediction := row.get("prediction")), Mapping)
    )
    by_chosen_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chosen_pair[str(row.get("chosen_pair"))].append(row)

    def mean(values: Sequence[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    return {
        "heldout_pairs": len(rows),
        "exact_top1": sum(1 for row in rows if row.get("exact_top1")),
        "candidate_accuracy": round(sum(1 for row in rows if row.get("exact_top1")) / len(rows), 3),
        "gold_in_candidates": sum(1 for row in rows if isinstance(row.get("gold_rank"), int)),
        "mean_gold_rank": mean([float(rank) for rank in gold_ranks]),
        "mean_top_gold_margin": mean(margins),
        "violations": dict(sorted(violations.items())),
        "top_pair_counts": dict(sorted(top_pair_counts.items())),
        "by_chosen_pair": {
            pair: {
                "pairs": len(items),
                "exact_top1": sum(1 for item in items if item.get("exact_top1")),
                "mean_gold_rank": mean(
                    [float(item["gold_rank"]) for item in items if isinstance(item.get("gold_rank"), int)]
                ),
                "mean_top_gold_margin": mean(
                    [
                        float(item["top_gold_margin"])
                        for item in items
                        if item.get("top_gold_margin") is not None
                    ]
                ),
            }
            for pair, items in sorted(by_chosen_pair.items())
        },
    }


def build_candidate_selection_report(
    *,
    run_id: str,
    model_id: str,
    rows: Sequence[Mapping[str, Any]],
    enum_candidate_policy: str,
    enum_candidate_outputs: Sequence[Mapping[str, Any]],
    score_label: str,
) -> dict[str, Any]:
    compact_rows = []
    for row in rows:
        compact_rows.append(
            {
                "source_enum_corrective_pair_id": row.get("source_enum_corrective_pair_id"),
                "source_component_target_id": row.get("source_component_target_id"),
                "case_id": row.get("case_id"),
                "case_family": row.get("case_family"),
                "chosen_pair": row.get("chosen_pair"),
                "rejected_pair": row.get("rejected_pair"),
                "target_output": row.get("target_output"),
                "prediction": row.get("prediction"),
                "gold_rank": row.get("gold_rank"),
                "gold_score": row.get("gold_score"),
                "top_score": row.get("top_score"),
                "top_gold_margin": row.get("top_gold_margin"),
                "exact_top1": row.get("exact_top1"),
                "top_candidates": compact_candidate_scores(row.get("candidate_scores", ())),
                "violations": row.get("violations", []),
            }
        )
    return {
        "dataset": CANDIDATE_EVAL_DATASET,
        "run_id": run_id,
        "model": model_id,
        "score_label": score_label,
        "component": COMPONENT,
        "failure_mode": row_value_summary(rows, "failure_mode", FAILURE_MODE),
        "failure_modes": count_by(rows, "failure_mode"),
        "contrast_axes": count_by(rows, "contrast_axis"),
        "target_format": "full",
        "enum_candidate_policy": enum_candidate_policy,
        "enum_candidate_space_size": len(enum_candidate_outputs),
        "enum_candidate_outputs": [dict(output) for output in enum_candidate_outputs],
        "prompt_contract": PROMPT_CONTRACT,
        "heldout_pairs": len(rows),
        "summary": summarize_candidate_selection_rows(rows),
        "rows": compact_rows,
        "boundary": (
            "Finite-candidate enum selection scores full action+evidence_status "
            "targets after corrective SFT. This is still component-level "
            "candidate scoring, not free-generation, full tool-query routing, "
            "DPO, or RLVR."
        ),
    }


def score_enum_candidate_selection(
    model: Any,
    tokenizer: Any,
    rows_to_score: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    model_id: str,
    device: str,
    max_length: int,
    enum_candidate_policy: str,
    enum_candidate_outputs: Sequence[Mapping[str, str]],
    score_label: str,
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for index, row in enumerate(rows_to_score):
        prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_row(row))
        candidate_result = score_enum_action_candidates(
            model,
            tokenizer,
            prompt,
            device=device,
            max_length=max_length,
            candidate_outputs=enum_candidate_outputs,
        )
        rows.append(
            candidate_selection_row_for_pair(
                row,
                run_id=run_id,
                model_id=model_id,
                score_label=score_label,
                enum_candidate_policy=enum_candidate_policy,
                candidate_scores=candidate_result["candidate_scores"],
            )
        )
        print(
            f"[{index + 1}/{len(rows_to_score)}] scored {score_label} {row['id']} "
            f"top1={rows[-1]['exact_top1']} rank={rows[-1]['gold_rank']}",
            flush=True,
        )
    return rows


def score_heldout_margins(
    model: Any,
    tokenizer: Any,
    rows_to_score: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    model_id: str,
    device: str,
    max_length: int,
    score_label: str = "trained_heldout",
    target_format: str = "full",
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for index, row in enumerate(rows_to_score):
        prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_row(row))
        chosen_score = score_candidate_target(
            model,
            tokenizer,
            prompt,
            pair_output_text(chosen_output_for_target_format(row, target_format)),
            device=device,
            max_length=max_length,
        )
        rejected_score = score_candidate_target(
            model,
            tokenizer,
            prompt,
            pair_output_text(rejected_output_for_target_format(row, target_format)),
            device=device,
            max_length=max_length,
        )
        rows.append(
            margin_row_for_pair(
                row,
                run_id=run_id,
                model_id=model_id,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                score_label=score_label,
                target_format=target_format,
            )
        )
        print(
            f"[{index + 1}/{len(rows_to_score)}] scored {score_label} {row['id']} "
            f"margin={rows[-1]['margin']:.6f}",
            flush=True,
        )
    return rows


def run_training_and_eval(
    args: argparse.Namespace,
    training_rows: list[dict[str, Any]],
    score_train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch
    from post_training.run_boundary_preference_dpo_smoke import sequence_logps

    disable_transformers_torchvision_probe()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enum_candidate_outputs = enum_candidate_outputs_for_policy(args.enum_candidate_policy, score_train_rows)
    encoded = [
        encode_chosen_pair(tokenizer, row, max_length=args.max_length, target_format=args.target_format)
        for row in training_rows
    ]
    encoded_margin_pairs = [
        encode_pairwise_margin_pair(
            tokenizer,
            row,
            max_length=args.max_length,
            target_format=args.target_format,
        )
        for row in training_rows
    ] if args.pairwise_margin_weight > 0 else []
    encoded_candidate_sets = [
        encode_candidate_ce_set(
            tokenizer,
            row,
            enum_candidate_outputs,
            max_length=args.max_length,
        )
        for row in training_rows
    ] if args.candidate_ce_weight > 0 else []

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        torch_dtype="auto",
    )
    model.to(device)

    base_margin_rows: list[dict[str, Any]] = []
    if args.score_base_margins:
        base_margin_rows = score_heldout_margins(
            model,
            tokenizer,
            heldout_rows,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            score_label="base_heldout",
            target_format=args.target_format,
        )
        write_jsonl(args.base_margins_out, base_margin_rows)
        write_json(
            args.base_eval_out,
            build_margin_report(
                run_id=args.run_id,
                model_id=args.model,
                rows=base_margin_rows,
                score_label="base_heldout",
            ),
        )

    base_candidate_rows: list[dict[str, Any]] = []
    if args.score_base_enum_candidates:
        base_candidate_rows = score_enum_candidate_selection(
            model,
            tokenizer,
            heldout_rows,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            enum_candidate_policy=args.enum_candidate_policy,
            enum_candidate_outputs=enum_candidate_outputs,
            score_label="base_candidate_heldout",
        )
        write_jsonl(args.base_enum_candidates_out, base_candidate_rows)
        write_json(
            args.base_enum_eval_out,
            build_candidate_selection_report(
                run_id=args.run_id,
                model_id=args.model,
                rows=base_candidate_rows,
                enum_candidate_policy=args.enum_candidate_policy,
                enum_candidate_outputs=enum_candidate_outputs,
                score_label="base_candidate_heldout",
            ),
        )

    model.train()
    trainable_params = set_trainable_last_layers(model, args.train_last_layers)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
    )

    losses: list[float] = []
    ce_losses: list[float] = []
    pairwise_margin_losses: list[float] = []
    pairwise_train_margins: list[float] = []
    candidate_ce_losses: list[float] = []
    candidate_pair_ce_losses: list[float] = []
    candidate_field_ce_losses: list[float] = []
    candidate_ce_train_ranks: list[float] = []
    candidate_ce_train_top1: list[float] = []
    cursor = 0
    for step in range(args.max_steps):
        batch_features = []
        batch_margin_pairs = []
        batch_candidate_sets = []
        for _ in range(args.batch_size):
            index = cursor % len(encoded)
            batch_features.append(encoded[index])
            if encoded_margin_pairs:
                batch_margin_pairs.append(encoded_margin_pairs[index])
            if encoded_candidate_sets:
                batch_candidate_sets.append(encoded_candidate_sets[index])
            cursor += 1
        batch = collate(batch_features, tokenizer.pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        ce_loss = model(**batch).loss
        if batch_margin_pairs:
            chosen_batch = collate([pair["chosen"] for pair in batch_margin_pairs], tokenizer.pad_token_id)
            rejected_batch = collate([pair["rejected"] for pair in batch_margin_pairs], tokenizer.pad_token_id)
            chosen_batch = {key: value.to(device) for key, value in chosen_batch.items()}
            rejected_batch = {key: value.to(device) for key, value in rejected_batch.items()}
            chosen_logps = sequence_logps(model, chosen_batch, logprob_mode=args.margin_logprob_mode)
            rejected_logps = sequence_logps(model, rejected_batch, logprob_mode=args.margin_logprob_mode)
            pairwise_margin_loss = pairwise_margin_loss_from_logps(
                chosen_logps,
                rejected_logps,
                margin=args.pairwise_margin,
            )
            step_pairwise_train_margin = float((chosen_logps - rejected_logps).mean().detach().cpu())
        else:
            pairwise_margin_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
            step_pairwise_train_margin = 0.0

        if batch_candidate_sets:
            candidate_losses = []
            pair_losses = []
            field_losses = []
            ranks = []
            top1_values = []
            for candidate_set in batch_candidate_sets:
                candidate_batch = collate(candidate_set["candidates"], tokenizer.pad_token_id)
                candidate_batch = {key: value.to(device) for key, value in candidate_batch.items()}
                candidate_logps = sequence_logps(
                    model,
                    candidate_batch,
                    logprob_mode=args.candidate_ce_logprob_mode,
                )
                expected_index = int(candidate_set["expected_index"])
                row_losses = []
                if args.candidate_ce_mode in {"pair", "pair_plus_field"}:
                    pair_loss = candidate_ce_loss_from_logps(candidate_logps, expected_index)
                    pair_losses.append(pair_loss)
                    row_losses.append(pair_loss)
                if args.candidate_ce_mode in {"field", "pair_plus_field"}:
                    field_loss = candidate_field_ce_loss_from_logps(
                        candidate_logps,
                        candidate_set["candidate_outputs"],
                        candidate_set["expected_output"],
                    )
                    field_losses.append(field_loss)
                    row_losses.append(field_loss)
                candidate_losses.append(torch.stack(row_losses).mean())
                target_logp = candidate_logps[expected_index]
                rank = int((candidate_logps > target_logp).sum().detach().cpu()) + 1
                ranks.append(float(rank))
                top1_values.append(float(rank == 1))
            candidate_ce_loss = torch.stack(candidate_losses).mean()
            candidate_pair_ce_loss = (
                torch.stack(pair_losses).mean()
                if pair_losses
                else torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
            )
            candidate_field_ce_loss = (
                torch.stack(field_losses).mean()
                if field_losses
                else torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
            )
            step_candidate_ce_train_rank = sum(ranks) / len(ranks)
            step_candidate_ce_train_top1 = sum(top1_values) / len(top1_values)
        else:
            candidate_ce_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
            candidate_pair_ce_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
            candidate_field_ce_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
            step_candidate_ce_train_rank = 0.0
            step_candidate_ce_train_top1 = 0.0

        loss = (
            ce_loss
            + (args.pairwise_margin_weight * pairwise_margin_loss)
            + (args.candidate_ce_weight * candidate_ce_loss)
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ce_losses.append(float(ce_loss.detach().cpu()))
        pairwise_margin_losses.append(float(pairwise_margin_loss.detach().cpu()))
        pairwise_train_margins.append(step_pairwise_train_margin)
        candidate_ce_losses.append(float(candidate_ce_loss.detach().cpu()))
        candidate_pair_ce_losses.append(float(candidate_pair_ce_loss.detach().cpu()))
        candidate_field_ce_losses.append(float(candidate_field_ce_loss.detach().cpu()))
        candidate_ce_train_ranks.append(step_candidate_ce_train_rank)
        candidate_ce_train_top1.append(step_candidate_ce_train_top1)
        print(
            f"step={step + 1} loss={losses[-1]:.4f} "
            f"ce={ce_losses[-1]:.4f} pairwise={pairwise_margin_losses[-1]:.4f} "
            f"candidate_ce={candidate_ce_losses[-1]:.4f} "
            f"candidate_field_ce={candidate_field_ce_losses[-1]:.4f} "
            f"train_margin={pairwise_train_margins[-1]:.4f} "
            f"candidate_rank={candidate_ce_train_ranks[-1]:.2f}",
            flush=True,
        )

    state_path = None
    if not args.no_save_trainable_state:
        state_path = save_trainable_state(model, out_dir)

    train_margin_rows: list[dict[str, Any]] = []
    if args.score_train_margins:
        train_margin_rows = score_heldout_margins(
            model,
            tokenizer,
            score_train_rows,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            score_label="trained_train",
            target_format=args.target_format,
        )
        write_jsonl(args.train_margins_out, train_margin_rows)
        write_json(
            args.train_eval_out,
            build_margin_report(
                run_id=args.run_id,
                model_id=args.model,
                rows=train_margin_rows,
                score_label="trained_train",
            ),
        )
        model.train()

    margin_rows = score_heldout_margins(
        model,
        tokenizer,
        heldout_rows,
        run_id=args.run_id,
        model_id=args.model,
        device=device,
        max_length=args.max_length,
        score_label="trained_heldout",
        target_format=args.target_format,
    )
    write_jsonl(args.margins_out, margin_rows)
    margin_report = build_margin_report(
        run_id=args.run_id,
        model_id=args.model,
        rows=margin_rows,
        score_label="trained_heldout",
    )
    write_json(args.eval_out, margin_report)

    candidate_rows: list[dict[str, Any]] = []
    candidate_report = None
    if args.score_enum_candidates:
        candidate_rows = score_enum_candidate_selection(
            model,
            tokenizer,
            heldout_rows,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            enum_candidate_policy=args.enum_candidate_policy,
            enum_candidate_outputs=enum_candidate_outputs,
            score_label="trained_candidate_heldout",
        )
        write_jsonl(args.enum_candidates_out, candidate_rows)
        candidate_report = build_candidate_selection_report(
            run_id=args.run_id,
            model_id=args.model,
            rows=candidate_rows,
            enum_candidate_policy=args.enum_candidate_policy,
            enum_candidate_outputs=enum_candidate_outputs,
            score_label="trained_candidate_heldout",
        )
        write_json(args.enum_eval_out, candidate_report)

    delta_report = None
    if base_margin_rows:
        delta_report = build_margin_delta_report(
            run_id=args.run_id,
            model_id=args.model,
            base_rows=base_margin_rows,
            trained_rows=margin_rows,
        )
        write_json(args.delta_eval_out, delta_report)

    return {
        "dataset": DATASET,
        "dry_run": False,
        "run_id": args.run_id,
        "model": args.model,
        "component": COMPONENT,
        "failure_mode": row_value_summary(score_train_rows + heldout_rows, "failure_mode", FAILURE_MODE),
        "failure_modes": count_by(score_train_rows + heldout_rows, "failure_mode"),
        "candidate_policies": count_by(score_train_rows + heldout_rows, "candidate_policy"),
        "contrast_axes": count_by(score_train_rows + heldout_rows, "contrast_axis"),
        "target_format": args.target_format,
        "device": device,
        "prompt_contract": PROMPT_CONTRACT,
        "pairs": args.pairs,
        "train_pairs": args.train_pairs,
        "heldout_pairs": args.heldout_pairs,
        "train_examples": len(score_train_rows),
        "training_examples": len(training_rows),
        "heldout_examples": len(heldout_rows),
        "focus_chosen_pairs": list(args.focus_chosen_pairs_parsed),
        "focus_repeat": args.focus_repeat,
        "focus_only": args.focus_only,
        "train_by_chosen_pair": count_by(score_train_rows, "chosen_pair"),
        "training_by_chosen_pair": count_by(training_rows, "chosen_pair"),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_last_layers": args.train_last_layers,
        "pairwise_margin_weight": args.pairwise_margin_weight,
        "pairwise_margin": args.pairwise_margin,
        "margin_logprob_mode": args.margin_logprob_mode,
        "candidate_ce_weight": args.candidate_ce_weight,
        "candidate_ce_mode": args.candidate_ce_mode,
        "candidate_ce_logprob_mode": args.candidate_ce_logprob_mode,
        "score_base_enum_candidates": args.score_base_enum_candidates,
        "score_enum_candidates": args.score_enum_candidates,
        "enum_candidate_policy": args.enum_candidate_policy,
        "enum_candidate_space_size": len(enum_candidate_outputs),
        "trainable_params": trainable_params,
        "losses": losses,
        "ce_losses": ce_losses,
        "pairwise_margin_losses": pairwise_margin_losses,
        "pairwise_train_margins": pairwise_train_margins,
        "candidate_ce_losses": candidate_ce_losses,
        "candidate_pair_ce_losses": candidate_pair_ce_losses,
        "candidate_field_ce_losses": candidate_field_ce_losses,
        "candidate_ce_train_ranks": candidate_ce_train_ranks,
        "candidate_ce_train_top1": candidate_ce_train_top1,
        "loss_delta": round(losses[-1] - losses[0], 6) if len(losses) > 1 else 0.0,
        "trainable_state": str(state_path) if state_path else None,
        "base_margins": str(args.base_margins_out) if base_margin_rows else None,
        "base_eval_report": str(args.base_eval_out) if base_margin_rows else None,
        "train_margins": str(args.train_margins_out) if train_margin_rows else None,
        "train_eval_report": str(args.train_eval_out) if train_margin_rows else None,
        "margins": str(args.margins_out),
        "eval_report": str(args.eval_out),
        "eval_summary": margin_report["summary"],
        "base_enum_candidates": str(args.base_enum_candidates_out) if base_candidate_rows else None,
        "base_enum_eval_report": str(args.base_enum_eval_out) if base_candidate_rows else None,
        "base_enum_eval_summary": (
            build_candidate_selection_report(
                run_id=args.run_id,
                model_id=args.model,
                rows=base_candidate_rows,
                enum_candidate_policy=args.enum_candidate_policy,
                enum_candidate_outputs=enum_candidate_outputs,
                score_label="base_candidate_heldout",
            )["summary"]
            if base_candidate_rows
            else None
        ),
        "enum_candidates": str(args.enum_candidates_out) if candidate_rows else None,
        "enum_eval_report": str(args.enum_eval_out) if candidate_report else None,
        "enum_eval_summary": candidate_report["summary"] if candidate_report else None,
        "delta_eval_report": str(args.delta_eval_out) if delta_report else None,
        "delta_summary": delta_report["summary"] if delta_report else None,
        "boundary": (
            "Full mode trains on chosen enum corrective targets, then scores "
            "held-out chosen/rejected margins. Raw run artifacts belong under "
            "post_training/runs/."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--pairs", default="post_training/stage_a_enum_corrective_pairs_v1.jsonl")
    parser.add_argument("--train-pairs", default="post_training/stage_a_enum_corrective_pairs_train_v1.jsonl")
    parser.add_argument("--heldout-pairs", default="post_training/stage_a_enum_corrective_pairs_heldout_v1.jsonl")
    parser.add_argument("--manifest", default="post_training/stage_a_enum_corrective_pairs_manifest.json")
    parser.add_argument("--out-dir", default="post_training/runs/stage_a_enum_corrective_sft_smoke")
    parser.add_argument("--run-id", default="stage_a_enum_corrective_sft_smoke")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-heldout", type=int, default=None)
    parser.add_argument(
        "--focus-chosen-pairs",
        default="",
        help=(
            "Comma- or colon-separated chosen_pair labels to oversample during "
            "training, e.g. flag/invalid_value,defer/insufficient. Use colon "
            "when passing through Slurm --export."
        ),
    )
    parser.add_argument("--focus-repeat", type=int, default=1)
    parser.add_argument("--focus-only", action="store_true")
    parser.add_argument(
        "--target-format",
        choices=TARGET_FORMATS,
        default="full",
        help=(
            "Projection used for SFT targets and teacher-forced margins. "
            "'full' scores the action+evidence_status JSON; 'action_only' "
            "scores only {'action': ...} to diagnose target-format coupling."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument(
        "--pairwise-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Optional supervised hinge-loss weight. When positive, training "
            "adds max(0, margin - (logp(chosen)-logp(rejected))) over the "
            "same corrective pairs. This is not DPO/RLVR."
        ),
    )
    parser.add_argument(
        "--pairwise-margin",
        type=float,
        default=0.0,
        help="Required chosen-vs-rejected log-probability margin for pairwise hinge loss.",
    )
    parser.add_argument(
        "--margin-logprob-mode",
        choices=("mean", "sum"),
        default="mean",
        help="Log-probability reduction used by the optional pairwise margin objective.",
    )
    parser.add_argument(
        "--candidate-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional supervised finite-candidate selection loss weight. "
            "When positive, training adds cross-entropy over the enum "
            "candidate policy. This is not DPO/RLVR."
        ),
    )
    parser.add_argument(
        "--candidate-ce-mode",
        choices=("pair", "field", "pair_plus_field"),
        default="pair",
        help=(
            "Finite-candidate CE target. pair trains the exact action/status "
            "candidate, field trains action and evidence_status marginals, "
            "and pair_plus_field averages both supervised losses."
        ),
    )
    parser.add_argument(
        "--candidate-ce-logprob-mode",
        choices=("mean", "sum"),
        default="mean",
        help="Log-probability reduction used by the optional enum candidate CE objective.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--score-base-margins", action="store_true")
    parser.add_argument("--score-train-margins", action="store_true")
    parser.add_argument(
        "--score-base-enum-candidates",
        action="store_true",
        help="Score finite enum candidates before training; raw scores are written under out-dir.",
    )
    parser.add_argument(
        "--score-enum-candidates",
        action="store_true",
        help="Score finite enum candidates after training; this is a component readout, not free generation.",
    )
    parser.add_argument(
        "--enum-candidate-policy",
        choices=ENUM_CANDIDATE_POLICIES,
        default="all_valid_pairs",
        help=(
            "Finite enum candidate space for --score-*-enum-candidates. "
            "all_valid_pairs scores the full action/status cross-product; "
            "pair_observed_outputs scores unique chosen/rejected pair outputs."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-trainable-state", action="store_true")
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--base-margins-out", default=None)
    parser.add_argument("--base-eval-out", default=None)
    parser.add_argument("--train-margins-out", default=None)
    parser.add_argument("--train-eval-out", default=None)
    parser.add_argument("--margins-out", default=None)
    parser.add_argument("--eval-out", default=None)
    parser.add_argument("--delta-eval-out", default=None)
    parser.add_argument("--base-enum-candidates-out", default=None)
    parser.add_argument("--base-enum-eval-out", default=None)
    parser.add_argument("--enum-candidates-out", default=None)
    parser.add_argument("--enum-eval-out", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report_out is None:
        args.report_out = str(out_dir / "report.json")
    if args.base_margins_out is None:
        args.base_margins_out = str(out_dir / "base_margins.jsonl")
    if args.base_eval_out is None:
        args.base_eval_out = str(out_dir / "base_margin_report.json")
    if args.train_margins_out is None:
        args.train_margins_out = str(out_dir / "train_margins.jsonl")
    if args.train_eval_out is None:
        args.train_eval_out = str(out_dir / "train_margin_report.json")
    if args.margins_out is None:
        args.margins_out = str(out_dir / "margins.jsonl")
    if args.eval_out is None:
        args.eval_out = str(out_dir / "margin_report.json")
    if args.delta_eval_out is None:
        args.delta_eval_out = str(out_dir / "margin_delta_report.json")
    if args.base_enum_candidates_out is None:
        args.base_enum_candidates_out = str(out_dir / "base_enum_candidates.jsonl")
    if args.base_enum_eval_out is None:
        args.base_enum_eval_out = str(out_dir / "base_enum_candidate_report.json")
    if args.enum_candidates_out is None:
        args.enum_candidates_out = str(out_dir / "enum_candidates.jsonl")
    if args.enum_eval_out is None:
        args.enum_eval_out = str(out_dir / "enum_candidate_report.json")

    all_rows = load_jsonl(args.pairs)
    all_train_rows = load_jsonl(args.train_pairs)
    all_heldout_rows = load_jsonl(args.heldout_pairs)
    manifest = load_manifest(args.manifest)
    issues = validate_corrective_artifacts(all_rows, all_train_rows, all_heldout_rows, manifest)
    if args.pairwise_margin_weight < 0:
        issues.append("pairwise_margin_weight_negative")
    if args.pairwise_margin < 0:
        issues.append("pairwise_margin_negative")
    if args.candidate_ce_weight < 0:
        issues.append("candidate_ce_weight_negative")
    if (
        args.score_base_enum_candidates
        or args.score_enum_candidates
        or args.candidate_ce_weight > 0
    ) and args.target_format != "full":
        issues.append("enum_candidate_scoring_requires_full_target_format")

    train_rows = all_train_rows[: args.limit_train] if args.limit_train is not None else all_train_rows
    heldout_rows = all_heldout_rows[: args.limit_heldout] if args.limit_heldout is not None else all_heldout_rows
    args.focus_chosen_pairs_parsed = parse_focus_chosen_pairs(args.focus_chosen_pairs)
    training_rows: list[dict[str, Any]] = []
    try:
        training_rows = expand_training_rows(
            train_rows,
            focus_chosen_pairs=args.focus_chosen_pairs_parsed,
            focus_repeat=args.focus_repeat,
            focus_only=args.focus_only,
        )
    except ValueError as exc:
        issues.append(f"training_focus_invalid:{exc}")
    if not train_rows:
        issues.append("selected_train_pairs_empty")
    if not training_rows:
        issues.append("selected_training_pairs_empty")
    if not heldout_rows:
        issues.append("selected_heldout_pairs_empty")

    if issues:
        write_json(
            args.report_out,
            dry_run_report(
                model=args.model,
                pairs=args.pairs,
                train_pairs=args.train_pairs,
                heldout_pairs=args.heldout_pairs,
                manifest=args.manifest,
                train_rows=train_rows,
                training_rows=training_rows,
                heldout_rows=heldout_rows,
                issues=issues,
                focus_chosen_pairs=args.focus_chosen_pairs_parsed,
                focus_repeat=args.focus_repeat,
                focus_only=args.focus_only,
                target_format=args.target_format,
                pairwise_margin_weight=args.pairwise_margin_weight,
                pairwise_margin=args.pairwise_margin,
                margin_logprob_mode=args.margin_logprob_mode,
                candidate_ce_weight=args.candidate_ce_weight,
                candidate_ce_mode=args.candidate_ce_mode,
                candidate_ce_logprob_mode=args.candidate_ce_logprob_mode,
                score_base_enum_candidates=args.score_base_enum_candidates,
                score_enum_candidates=args.score_enum_candidates,
                enum_candidate_policy=args.enum_candidate_policy,
            ),
        )
        raise SystemExit("Enum corrective SFT smoke validation failed:\n- " + "\n- ".join(issues))

    if args.dry_run:
        report = dry_run_report(
            model=args.model,
            pairs=args.pairs,
            train_pairs=args.train_pairs,
            heldout_pairs=args.heldout_pairs,
            manifest=args.manifest,
            train_rows=train_rows,
            training_rows=training_rows,
            heldout_rows=heldout_rows,
            issues=issues,
            focus_chosen_pairs=args.focus_chosen_pairs_parsed,
            focus_repeat=args.focus_repeat,
            focus_only=args.focus_only,
            target_format=args.target_format,
            pairwise_margin_weight=args.pairwise_margin_weight,
            pairwise_margin=args.pairwise_margin,
            margin_logprob_mode=args.margin_logprob_mode,
            candidate_ce_weight=args.candidate_ce_weight,
            candidate_ce_mode=args.candidate_ce_mode,
            candidate_ce_logprob_mode=args.candidate_ce_logprob_mode,
            score_base_enum_candidates=args.score_base_enum_candidates,
            score_enum_candidates=args.score_enum_candidates,
            enum_candidate_policy=args.enum_candidate_policy,
        )
    else:
        if not args.allow_model_load:
            raise SystemExit("Full enum corrective SFT smoke requires --allow-model-load. Use --dry-run locally.")
        report = run_training_and_eval(args, training_rows, train_rows, heldout_rows)
    write_json(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
