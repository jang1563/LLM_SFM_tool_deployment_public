#!/usr/bin/env python3
"""Run Stage A saved-output calibration margin SFT smoke experiments.

This is the cluster-side follow-up to the saved-output calibration probe
readout. It trains only on train-allowed target-vs-ground/supported probe pairs
and evaluates held-out pairs by teacher-forced chosen-vs-rejected margin.

Dry-run mode validates split boundaries without loading model weights. Full
mode requires --allow-model-load and is intended for Cayuga/Expanse.
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

from post_training.generate_stage_a_predictions import disable_transformers_torchvision_probe  # noqa: E402
from post_training.run_stage_a_sft_smoke_eval import load_jsonl  # noqa: E402
from post_training.run_stage_a_strict_component_sft_smoke import score_candidate_target  # noqa: E402
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    choose_device,
    collate,
    prompt_text_for_tokenizer,
    save_trainable_state,
    set_trainable_last_layers,
    write_json,
    write_jsonl,
)
from post_training.validate_post_training_data import (  # noqa: E402
    validate_stage_a_saved_output_calibration_probe,
)


DATASET = "negbiodb_ct_stage_a_saved_output_calibration_margin_sft_v1"
MARGIN_DATASET = "negbiodb_ct_stage_a_saved_output_calibration_margin_rows_v1"
MARGIN_DELTA_DATASET = "negbiodb_ct_stage_a_saved_output_calibration_margin_delta_v1"
CANDIDATE_DATASET = "negbiodb_ct_stage_a_saved_output_calibration_candidate_rows_v1"
CANDIDATE_REPORT_DATASET = "negbiodb_ct_stage_a_saved_output_calibration_candidate_report_v1"
PROMPT_CONTRACT = "stage_a_v4_canonical_json"
CALIBRATION_AXIS = "target_pair_vs_ground_supported"
FAILURE_MODE = "saved_output_ground_supported_collapse"
TARGET_FORMATS = ("full", "action_status_only", "action_only", "status_only")
CANDIDATE_POLICIES = ("train_observed_plus_rejected", "train_observed_pairs", "all_valid_pairs")
CANDIDATE_CE_MODES = ("pair", "field", "pair_plus_field")
FIELD_NAMES = ("action", "evidence_status")
EXTERNAL_ACTIONS = ("ground", "reject", "defer", "verify", "flag")
EXTERNAL_EVIDENCE_STATUSES = ("supported", "contradicted", "invalid_value", "insufficient")


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def project_output(output: Mapping[str, Any], *, target_format: str) -> dict[str, Any]:
    if target_format not in TARGET_FORMATS:
        raise ValueError(f"Unknown target_format: {target_format}")
    full_output = dict(output)
    if target_format == "full":
        return full_output
    if target_format == "action_status_only":
        return {
            "action": full_output.get("action"),
            "evidence_status": full_output.get("evidence_status"),
        }
    if target_format == "action_only":
        return {"action": full_output.get("action")}
    return {"evidence_status": full_output.get("evidence_status")}


def output_text(output: Mapping[str, Any], *, target_format: str = "full") -> str:
    return json.dumps(project_output(output, target_format=target_format), sort_keys=True)


def pair_label(output: Mapping[str, Any] | None) -> str:
    output = output or {}
    return f"{output.get('action')}/{output.get('evidence_status')}"


def pair_key(output: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    output = output or {}
    action = output.get("action")
    status = output.get("evidence_status")
    return (
        str(action) if action is not None else None,
        str(status) if status is not None else None,
    )


def pair_from_label(label: str) -> dict[str, str]:
    if "/" not in label:
        raise ValueError(f"Invalid pair label: {label!r}")
    action, evidence_status = label.split("/", 1)
    if action not in EXTERNAL_ACTIONS:
        raise ValueError(f"Unknown action in pair label: {label!r}")
    if evidence_status not in EXTERNAL_EVIDENCE_STATUSES:
        raise ValueError(f"Unknown evidence_status in pair label: {label!r}")
    return {"action": action, "evidence_status": evidence_status}


def compact_pair(output: Mapping[str, Any] | None) -> dict[str, str | None]:
    action, evidence_status = pair_key(output)
    return {"action": action, "evidence_status": evidence_status}


def all_valid_candidate_pairs() -> list[dict[str, str]]:
    return [
        {"action": action, "evidence_status": evidence_status}
        for action in EXTERNAL_ACTIONS
        for evidence_status in EXTERNAL_EVIDENCE_STATUSES
    ]


def candidate_pairs_for_policy(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
) -> list[dict[str, str]]:
    if policy == "all_valid_pairs":
        return all_valid_candidate_pairs()
    if policy not in CANDIDATE_POLICIES:
        raise ValueError(f"Unknown candidate_policy: {policy}")

    labels: set[str] = set()
    for row in train_rows:
        if isinstance(row.get("chosen_pair"), str):
            labels.add(str(row["chosen_pair"]))
        if policy == "train_observed_plus_rejected" and isinstance(row.get("rejected_pair"), str):
            labels.add(str(row["rejected_pair"]))
    pairs = [pair_from_label(label) for label in sorted(labels)]
    if not pairs:
        raise ValueError(f"{policy} produced no candidate pairs")
    return sorted(pairs, key=lambda item: (item["action"], item["evidence_status"]))


def candidate_output_for_pair(row: Mapping[str, Any], pair: Mapping[str, str]) -> dict[str, Any]:
    target_key = (str(pair["action"]), str(pair["evidence_status"]))
    chosen = chosen_output_from_pair(row)
    rejected = rejected_output_from_pair(row)
    if target_key == pair_key(chosen):
        base = dict(chosen)
    elif target_key == pair_key(rejected):
        base = dict(rejected)
    else:
        base = dict(chosen)
        base["action"] = target_key[0]
        base["evidence_status"] = target_key[1]
    base["action"] = target_key[0]
    base["evidence_status"] = target_key[1]
    return base


def pair_case_id(row: Mapping[str, Any]) -> str:
    value = row.get("case_id") or row.get("source_manifest_case_id") or row.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Saved-output calibration pair is missing case id: {row.get('id')!r}")
    return value


def prompt_messages_from_pair(row: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = row.get("prompt_messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{row.get('id')} has invalid prompt_messages")
    out: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError(f"{row.get('id')} has malformed prompt message")
        role = message.get("role")
        if role not in {"system", "user"}:
            raise ValueError(f"{row.get('id')} has unexpected prompt role: {role!r}")
        out.append({"role": str(role), "content": str(message.get("content", ""))})
    return out


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


def count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def row_value_summary(rows: Sequence[Mapping[str, Any]], key: str, default: str) -> str:
    values = sorted({str(row.get(key)) for row in rows if row.get(key) is not None})
    if len(values) == 1:
        return values[0]
    if not values:
        return default
    return "mixed"


def parse_focus_chosen_pairs(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    pairs = tuple(item.strip() for item in re.split(r"[,:]", value) if item.strip())
    return tuple(dict.fromkeys(pairs))


def parse_score_target_formats(value: str | None, *, training_target_format: str) -> tuple[str, ...]:
    raw_items = re.split(r"[,:]", value or "")
    formats = [item.strip() for item in raw_items if item.strip()]
    if not formats:
        formats = [training_target_format]
    formats = list(dict.fromkeys(formats))
    if training_target_format not in formats:
        formats.insert(0, training_target_format)
    unknown = [item for item in formats if item not in TARGET_FORMATS]
    if unknown:
        raise ValueError("unknown score_target_formats: " + ", ".join(unknown))
    return tuple(formats)


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


def validate_saved_output_calibration_margin_artifacts(
    all_rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues = validate_stage_a_saved_output_calibration_probe(
        all_rows,
        train_rows,
        heldout_rows,
        manifest,
    )
    if not train_rows:
        issues.append("empty_train_pairs")
    if not heldout_rows:
        issues.append("empty_heldout_pairs")
    if set(pair_case_id(row) for row in train_rows) & set(pair_case_id(row) for row in heldout_rows):
        issues.append("train_heldout_case_overlap")
    for split, rows in (("train", train_rows), ("heldout", heldout_rows)):
        for row in rows:
            row_id = row.get("id")
            if row.get("prompt_contract") != PROMPT_CONTRACT:
                issues.append(f"{row_id}:{split}_wrong_prompt_contract")
            if row.get("calibration_axis") != CALIBRATION_AXIS:
                issues.append(f"{row_id}:{split}_wrong_calibration_axis")
            if row.get("chosen_pair") != pair_label(row.get("chosen_output")):
                issues.append(f"{row_id}:{split}_chosen_pair_label_mismatch")
            if row.get("rejected_pair") != pair_label(row.get("rejected_output")):
                issues.append(f"{row_id}:{split}_rejected_pair_label_mismatch")
            if split == "train" and row.get("training_allowed") is not True:
                issues.append(f"{row_id}:train_pair_not_training_allowed")
            if split == "heldout" and row.get("evaluation_only") is not True:
                issues.append(f"{row_id}:heldout_pair_not_evaluation_only")
    return issues


def encode_pair_output(
    tokenizer: Any,
    row: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    max_length: int,
    target_format: str,
) -> dict[str, Any]:
    import torch

    prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_pair(row))
    eos = tokenizer.eos_token or ""
    target = output_text(output, target_format=target_format) + eos
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


def encode_chosen_pair(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_length: int,
    target_format: str,
) -> dict[str, Any]:
    return encode_pair_output(
        tokenizer,
        row,
        chosen_output_from_pair(row),
        max_length=max_length,
        target_format=target_format,
    )


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
            chosen_output_from_pair(row),
            max_length=max_length,
            target_format=target_format,
        ),
        "rejected": encode_pair_output(
            tokenizer,
            row,
            rejected_output_from_pair(row),
            max_length=max_length,
            target_format=target_format,
        ),
    }


def pairwise_margin_loss_from_logps(chosen_logps: Any, rejected_logps: Any, *, margin: float) -> Any:
    if margin < 0:
        raise ValueError("pairwise margin must be non-negative")
    import torch.nn.functional as F

    desired = chosen_logps.new_tensor(float(margin))
    return F.relu(desired - (chosen_logps - rejected_logps)).mean()


def candidate_index_for_row(
    row: Mapping[str, Any],
    candidate_pairs: Sequence[Mapping[str, str]],
) -> int:
    target_key = pair_key(chosen_output_from_pair(row))
    for index, pair in enumerate(candidate_pairs):
        if pair_key(pair) == target_key:
            return index
    raise ValueError(f"target pair not in candidate policy for {row.get('id')}: {target_key!r}")


def encode_candidate_ce_set(
    tokenizer: Any,
    row: Mapping[str, Any],
    candidate_pairs: Sequence[Mapping[str, str]],
    *,
    max_length: int,
    target_format: str,
) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "chosen_pair": row.get("chosen_pair"),
        "expected_output": compact_pair(chosen_output_from_pair(row)),
        "expected_index": candidate_index_for_row(row, candidate_pairs),
        "candidate_outputs": [compact_pair(candidate_output_for_pair(row, pair)) for pair in candidate_pairs],
        "candidates": [
            encode_pair_output(
                tokenizer,
                row,
                candidate_output_for_pair(row, pair),
                max_length=max_length,
                target_format=target_format,
            )
            for pair in candidate_pairs
        ],
    }


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
    for field_name in FIELD_NAMES:
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
            if indices:
                field_logits.append(torch.logsumexp(logps[indices], dim=0))
        target = torch.tensor([values.index(expected_key)], dtype=torch.long, device=logps.device)
        losses.append(F.cross_entropy(torch.stack(field_logits).unsqueeze(0), target))
    return torch.stack(losses).mean()


def margin_row_for_pair(
    row: Mapping[str, Any],
    *,
    run_id: str,
    model_id: str,
    chosen_score: float,
    rejected_score: float,
    score_label: str,
    target_format: str = "full",
) -> dict[str, Any]:
    margin = round(float(chosen_score) - float(rejected_score), 6)
    passed = margin > 0.0
    return {
        "id": f"{run_id}::{score_label}::{row['id']}",
        "dataset": MARGIN_DATASET,
        "source": "stage_a_saved_output_calibration_margin_sft",
        "run_id": run_id,
        "model": model_id,
        "score_label": score_label,
        "source_probe_pair_id": row["id"],
        "source_sft_id": row.get("source_sft_id"),
        "source_task_id": row.get("source_task_id"),
        "case_id": pair_case_id(row),
        "case_family": row.get("case_family"),
        "split": row.get("split"),
        "split_group": row.get("split_group"),
        "prompt_contract": row.get("prompt_contract"),
        "calibration_axis": row.get("calibration_axis"),
        "target_format": target_format,
        "chosen_pair": row.get("chosen_pair"),
        "rejected_pair": row.get("rejected_pair"),
        "chosen_score": round(float(chosen_score), 6),
        "rejected_score": round(float(rejected_score), 6),
        "margin": margin,
        "passed": passed,
        "violations": [] if passed else ["chosen_not_above_rejected"],
    }


def summarize_margin_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "pairs": 0,
            "margin_wins": 0,
            "margin_accuracy": 0.0,
            "mean_chosen_score": 0.0,
            "mean_rejected_score": 0.0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "violations": {},
            "by_chosen_pair": {},
        }

    def mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    violations = Counter(violation for row in rows for violation in row.get("violations", ()))
    by_chosen_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chosen_pair[str(row.get("chosen_pair"))].append(row)
    margins = [float(row.get("margin", 0.0)) for row in rows]
    return {
        "pairs": len(rows),
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


def build_margin_report(
    *,
    run_id: str,
    model_id: str,
    rows: Sequence[Mapping[str, Any]],
    score_label: str,
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "dry_run": False,
        "run_id": run_id,
        "model": model_id,
        "score_label": score_label,
        "failure_mode": FAILURE_MODE,
        "calibration_axis": row_value_summary(rows, "calibration_axis", CALIBRATION_AXIS),
        "prompt_contract": row_value_summary(rows, "prompt_contract", PROMPT_CONTRACT),
        "target_format": row_value_summary(rows, "target_format", "full"),
        "pairs": len(rows),
        "summary": summarize_margin_rows(rows),
        "rows": list(rows),
        "boundary": (
            "Saved-output calibration margin SFT scores teacher-forced "
            "target-vs-ground/supported margins. This is not DPO/RLVR and does "
            "not score free-form explanation quality."
        ),
    }


def margin_delta_row(base_row: Mapping[str, Any], trained_row: Mapping[str, Any]) -> dict[str, Any]:
    pair_id = str(base_row.get("source_probe_pair_id"))
    if pair_id != str(trained_row.get("source_probe_pair_id")):
        raise ValueError(f"Cannot compare different probe pairs: {pair_id} vs {trained_row.get('source_probe_pair_id')}")
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
        "source": "stage_a_saved_output_calibration_margin_sft",
        "run_id": trained_row.get("run_id"),
        "model": trained_row.get("model"),
        "source_probe_pair_id": pair_id,
        "case_id": trained_row.get("case_id"),
        "case_family": trained_row.get("case_family"),
        "split": trained_row.get("split"),
        "prompt_contract": trained_row.get("prompt_contract"),
        "calibration_axis": trained_row.get("calibration_axis"),
        "target_format": trained_row.get("target_format"),
        "chosen_pair": trained_row.get("chosen_pair"),
        "rejected_pair": trained_row.get("rejected_pair"),
        "base_margin": round(base_margin, 6),
        "trained_margin": round(trained_margin, 6),
        "margin_delta": round(trained_margin - base_margin, 6),
        "base_passed": base_passed,
        "trained_passed": trained_passed,
        "outcome": outcome,
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


def candidate_row_for_probe_pair(
    row: Mapping[str, Any],
    *,
    run_id: str,
    model_id: str,
    score_label: str,
    candidate_policy: str,
    target_format: str,
    candidate_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sorted_scores = sorted(
        candidate_scores,
        key=lambda item: (-float(item["score"]), json.dumps(item["candidate"], sort_keys=True)),
    )
    target_pair = compact_pair(chosen_output_from_pair(row))
    rejected_pair = compact_pair(rejected_output_from_pair(row))
    top = sorted_scores[0]
    top_pair = compact_pair(top.get("candidate") if isinstance(top, Mapping) else {})
    target_rank = None
    target_score = None
    for index, item in enumerate(sorted_scores, start=1):
        candidate = item.get("candidate")
        if isinstance(candidate, Mapping) and pair_key(candidate) == pair_key(target_pair):
            target_rank = index
            target_score = round(float(item["score"]), 6)
            break
    top_score = round(float(top["score"]), 6)
    top_target_margin = None
    if target_score is not None:
        top_target_margin = round(top_score - target_score, 6)
    exact_top1 = pair_key(top_pair) == pair_key(target_pair)
    return {
        "id": f"{run_id}::{score_label}::candidate::{row['id']}",
        "dataset": CANDIDATE_DATASET,
        "source": "stage_a_saved_output_calibration_margin_sft",
        "run_id": run_id,
        "model": model_id,
        "score_label": score_label,
        "source_probe_pair_id": row["id"],
        "source_sft_id": row.get("source_sft_id"),
        "source_task_id": row.get("source_task_id"),
        "case_id": pair_case_id(row),
        "case_family": row.get("case_family"),
        "split": row.get("split"),
        "split_group": row.get("split_group"),
        "prompt_contract": row.get("prompt_contract"),
        "calibration_axis": row.get("calibration_axis"),
        "candidate_policy": candidate_policy,
        "candidate_target_format": target_format,
        "candidate_space_size": len(sorted_scores),
        "target_pair": target_pair,
        "target_pair_label": pair_label(target_pair),
        "rejected_pair": rejected_pair,
        "rejected_pair_label": pair_label(rejected_pair),
        "top_pair": top_pair,
        "top_pair_label": pair_label(top_pair),
        "exact_top1": exact_top1,
        "target_rank": target_rank,
        "target_score": target_score,
        "top_score": top_score,
        "top_target_margin": top_target_margin,
        "violations": [] if exact_top1 else ["target_not_top1"],
        "candidate_scores": [
            {
                "score": round(float(item["score"]), 6),
                "candidate": compact_pair(item.get("candidate") if isinstance(item, Mapping) else {}),
            }
            for item in sorted_scores
        ],
    }


def summarize_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "pairs": 0,
            "exact_top1": 0,
            "exact_top1_accuracy": 0.0,
            "mean_target_rank": None,
            "mean_top_target_margin": None,
            "top_pair_counts": {},
            "target_pair_counts": {},
            "candidate_count_histogram": {},
            "violations": {},
        }

    def mean(values: Sequence[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    ranks = [float(row["target_rank"]) for row in rows if isinstance(row.get("target_rank"), int)]
    margins = [
        float(row["top_target_margin"])
        for row in rows
        if row.get("top_target_margin") is not None
    ]
    violations = Counter(violation for row in rows for violation in row.get("violations", ()))
    top_pair_counts = Counter(str(row.get("top_pair_label")) for row in rows)
    target_pair_counts = Counter(str(row.get("target_pair_label")) for row in rows)
    candidate_counts = Counter(int(row.get("candidate_space_size", 0)) for row in rows)
    exact = sum(1 for row in rows if row.get("exact_top1"))
    return {
        "pairs": len(rows),
        "exact_top1": exact,
        "exact_top1_accuracy": round(exact / len(rows), 3),
        "mean_target_rank": mean(ranks),
        "mean_top_target_margin": mean(margins),
        "top_pair_counts": dict(sorted(top_pair_counts.items())),
        "target_pair_counts": dict(sorted(target_pair_counts.items())),
        "candidate_count_histogram": dict(sorted(candidate_counts.items())),
        "violations": dict(sorted(violations.items())),
    }


def build_candidate_report(
    *,
    run_id: str,
    model_id: str,
    rows: Sequence[Mapping[str, Any]],
    score_label: str,
    candidate_policy: str,
    target_format: str,
) -> dict[str, Any]:
    return {
        "dataset": CANDIDATE_REPORT_DATASET,
        "dry_run": False,
        "run_id": run_id,
        "model": model_id,
        "score_label": score_label,
        "failure_mode": FAILURE_MODE,
        "calibration_axis": row_value_summary(rows, "calibration_axis", CALIBRATION_AXIS),
        "prompt_contract": row_value_summary(rows, "prompt_contract", PROMPT_CONTRACT),
        "candidate_policy": candidate_policy,
        "candidate_target_format": target_format,
        "pairs": len(rows),
        "summary": summarize_candidate_rows(rows),
        "rows": list(rows),
        "boundary": (
            "Saved-output calibration candidate scoring ranks finite action/status "
            "outputs after margin SFT. It is a candidate-readout diagnostic, not "
            "free-form generation, DPO/RLVR, or full trajectory readiness."
        ),
    }


def build_margin_delta_report(
    *,
    run_id: str,
    model_id: str,
    base_rows: Sequence[Mapping[str, Any]],
    trained_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_by_pair = {str(row.get("source_probe_pair_id")): row for row in base_rows}
    trained_by_pair = {str(row.get("source_probe_pair_id")): row for row in trained_rows}
    row_reports = []
    missing_base = sorted(set(trained_by_pair) - set(base_by_pair))
    missing_trained = sorted(set(base_by_pair) - set(trained_by_pair))
    for pair_id in sorted(set(base_by_pair) & set(trained_by_pair)):
        row_reports.append(margin_delta_row(base_by_pair[pair_id], trained_by_pair[pair_id]))
    return {
        "dataset": MARGIN_DELTA_DATASET,
        "run_id": run_id,
        "model": model_id,
        "failure_mode": FAILURE_MODE,
        "calibration_axis": row_value_summary(trained_rows, "calibration_axis", CALIBRATION_AXIS),
        "prompt_contract": row_value_summary(trained_rows, "prompt_contract", PROMPT_CONTRACT),
        "target_format": row_value_summary(trained_rows, "target_format", "full"),
        "summary": summarize_margin_delta_rows(row_reports),
        "missing_base_pair_ids": missing_base,
        "missing_trained_pair_ids": missing_trained,
        "rows": row_reports,
        "boundary": (
            "Margin delta compares base-model held-out calibration margins with "
            "post-SFT held-out calibration margins. This is diagnostic SFT "
            "movement, not DPO/RLVR."
        ),
    }


def target_format_output_path(path: str | Path, target_format: str, primary_target_format: str) -> Path:
    out = Path(path)
    if target_format == primary_target_format:
        return out
    return out.with_name(f"{out.stem}_{target_format}{out.suffix}")


def score_margins(
    model: Any,
    tokenizer: Any,
    rows_to_score: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    model_id: str,
    device: str,
    max_length: int,
    score_label: str,
    target_format: str,
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for index, row in enumerate(rows_to_score):
        prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_pair(row))
        chosen_score = score_candidate_target(
            model,
            tokenizer,
            prompt,
            output_text(chosen_output_from_pair(row), target_format=target_format),
            device=device,
            max_length=max_length,
        )
        rejected_score = score_candidate_target(
            model,
            tokenizer,
            prompt,
            output_text(rejected_output_from_pair(row), target_format=target_format),
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


def score_candidates(
    model: Any,
    tokenizer: Any,
    rows_to_score: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    model_id: str,
    device: str,
    max_length: int,
    score_label: str,
    candidate_pairs: Sequence[Mapping[str, str]],
    candidate_policy: str,
    target_format: str,
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for index, row in enumerate(rows_to_score):
        prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_pair(row))
        scored = []
        for pair in candidate_pairs:
            candidate = candidate_output_for_pair(row, pair)
            score = score_candidate_target(
                model,
                tokenizer,
                prompt,
                output_text(candidate, target_format=target_format),
                device=device,
                max_length=max_length,
            )
            scored.append({"score": score, "candidate": candidate})
        rows.append(
            candidate_row_for_probe_pair(
                row,
                run_id=run_id,
                model_id=model_id,
                score_label=score_label,
                candidate_policy=candidate_policy,
                target_format=target_format,
                candidate_scores=scored,
            )
        )
        print(
            f"[{index + 1}/{len(rows_to_score)}] scored {score_label} candidates {row['id']} "
            f"top={rows[-1]['top_pair_label']} target_rank={rows[-1]['target_rank']}",
            flush=True,
        )
    return rows


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
    pairwise_margin_weight: float,
    pairwise_margin: float,
    margin_logprob_mode: str,
    candidate_ce_weight: float,
    candidate_ce_mode: str,
    candidate_ce_logprob_mode: str,
    target_format: str,
    score_target_formats: Sequence[str],
    score_base_margins: bool,
    score_train_margins: bool,
    score_base_candidates: bool,
    score_train_candidates: bool,
    score_trained_candidates: bool,
    candidate_policy: str,
    candidate_target_format: str,
) -> dict[str, Any]:
    selected_rows = [dict(row) for row in train_rows] + [dict(row) for row in heldout_rows]
    try:
        candidate_space_size = len(candidate_pairs_for_policy(train_rows, policy=candidate_policy))
    except ValueError:
        candidate_space_size = 0
    return {
        "dataset": DATASET,
        "dry_run": True,
        "model": model,
        "failure_mode": FAILURE_MODE,
        "calibration_axis": row_value_summary(selected_rows, "calibration_axis", CALIBRATION_AXIS),
        "prompt_contract": row_value_summary(selected_rows, "prompt_contract", PROMPT_CONTRACT),
        "pairwise_margin_weight": pairwise_margin_weight,
        "pairwise_margin": pairwise_margin,
        "margin_logprob_mode": margin_logprob_mode,
        "candidate_ce_weight": candidate_ce_weight,
        "candidate_ce_mode": candidate_ce_mode,
        "candidate_ce_logprob_mode": candidate_ce_logprob_mode,
        "target_format": target_format,
        "score_target_formats": list(score_target_formats),
        "score_base_margins": score_base_margins,
        "score_train_margins": score_train_margins,
        "score_base_candidates": score_base_candidates,
        "score_train_candidates": score_train_candidates,
        "score_trained_candidates": score_trained_candidates,
        "candidate_policy": candidate_policy,
        "candidate_target_format": candidate_target_format,
        "candidate_space_size": candidate_space_size,
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
        "heldout_used_for_training": False,
        "issues": list(issues),
        "artifact_policy": {
            "raw_prompt_messages_committed": False,
            "raw_model_text_committed": False,
            "scheduler_logs_committed": False,
            "model_state_committed": False,
            "margin_jsonl_committed": False,
            "candidate_jsonl_committed": False,
        },
        "boundary": (
            "Dry run validates saved-output calibration margin SFT artifacts "
            "and split boundaries without loading model weights or running "
            "local heavy compute."
        ),
    }


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
    candidate_pairs = candidate_pairs_for_policy(score_train_rows, policy=args.candidate_policy)
    encoded = [
        encode_chosen_pair(
            tokenizer,
            row,
            max_length=args.max_length,
            target_format=args.target_format,
        )
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
            candidate_pairs,
            max_length=args.max_length,
            target_format=args.candidate_target_format,
        )
        for row in training_rows
    ] if args.candidate_ce_weight > 0 else []

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        torch_dtype="auto",
    )
    model.to(device)

    base_margin_rows_by_format: dict[str, list[dict[str, Any]]] = {}
    base_report_paths_by_format: dict[str, str] = {}
    base_candidate_rows: list[dict[str, Any]] = []
    base_candidate_report: dict[str, Any] | None = None
    if args.score_base_margins:
        for score_target_format in args.score_target_formats_parsed:
            base_margin_rows = score_margins(
                model,
                tokenizer,
                heldout_rows,
                run_id=args.run_id,
                model_id=args.model,
                device=device,
                max_length=args.max_length,
                score_label="base_heldout",
                target_format=score_target_format,
            )
            base_margin_rows_by_format[score_target_format] = base_margin_rows
            base_margins_path = target_format_output_path(
                args.base_margins_out,
                score_target_format,
                args.target_format,
            )
            base_eval_path = target_format_output_path(
                args.base_eval_out,
                score_target_format,
                args.target_format,
            )
            write_jsonl(base_margins_path, base_margin_rows)
            write_json(
                base_eval_path,
                build_margin_report(
                    run_id=args.run_id,
                    model_id=args.model,
                    rows=base_margin_rows,
                    score_label="base_heldout",
                ),
            )
            base_report_paths_by_format[score_target_format] = str(base_eval_path)

    if args.score_base_candidates:
        base_candidate_rows = score_candidates(
            model,
            tokenizer,
            heldout_rows,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            score_label="base_heldout",
            candidate_pairs=candidate_pairs,
            candidate_policy=args.candidate_policy,
            target_format=args.candidate_target_format,
        )
        write_jsonl(args.base_candidates_out, base_candidate_rows)
        base_candidate_report = build_candidate_report(
            run_id=args.run_id,
            model_id=args.model,
            rows=base_candidate_rows,
            score_label="base_heldout",
            candidate_policy=args.candidate_policy,
            target_format=args.candidate_target_format,
        )
        write_json(args.base_candidate_eval_out, base_candidate_report)

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
            f"step={step + 1} loss={losses[-1]:.4f} ce={ce_losses[-1]:.4f} "
            f"pairwise={pairwise_margin_losses[-1]:.4f} "
            f"candidate_ce={candidate_ce_losses[-1]:.4f} "
            f"candidate_field_ce={candidate_field_ce_losses[-1]:.4f} "
            f"train_margin={pairwise_train_margins[-1]:.4f} "
            f"candidate_rank={candidate_ce_train_ranks[-1]:.2f}",
            flush=True,
        )

    state_path = None
    if not args.no_save_trainable_state:
        state_path = save_trainable_state(model, out_dir)

    train_margin_rows_by_format: dict[str, list[dict[str, Any]]] = {}
    train_report_paths_by_format: dict[str, str] = {}
    train_candidate_rows: list[dict[str, Any]] = []
    train_candidate_report: dict[str, Any] | None = None
    if args.score_train_margins:
        for score_target_format in args.score_target_formats_parsed:
            train_margin_rows = score_margins(
                model,
                tokenizer,
                score_train_rows,
                run_id=args.run_id,
                model_id=args.model,
                device=device,
                max_length=args.max_length,
                score_label="trained_train",
                target_format=score_target_format,
            )
            train_margin_rows_by_format[score_target_format] = train_margin_rows
            train_margins_path = target_format_output_path(
                args.train_margins_out,
                score_target_format,
                args.target_format,
            )
            train_eval_path = target_format_output_path(
                args.train_eval_out,
                score_target_format,
                args.target_format,
            )
            write_jsonl(train_margins_path, train_margin_rows)
            write_json(
                train_eval_path,
                build_margin_report(
                    run_id=args.run_id,
                    model_id=args.model,
                    rows=train_margin_rows,
                    score_label="trained_train",
                ),
            )
            train_report_paths_by_format[score_target_format] = str(train_eval_path)
            model.train()

    if args.score_train_candidates:
        train_candidate_rows = score_candidates(
            model,
            tokenizer,
            score_train_rows,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            score_label="trained_train",
            candidate_pairs=candidate_pairs,
            candidate_policy=args.candidate_policy,
            target_format=args.candidate_target_format,
        )
        write_jsonl(args.train_candidates_out, train_candidate_rows)
        train_candidate_report = build_candidate_report(
            run_id=args.run_id,
            model_id=args.model,
            rows=train_candidate_rows,
            score_label="trained_train",
            candidate_policy=args.candidate_policy,
            target_format=args.candidate_target_format,
        )
        write_json(args.train_candidate_eval_out, train_candidate_report)
        model.train()

    margin_rows_by_format: dict[str, list[dict[str, Any]]] = {}
    margin_reports_by_format: dict[str, dict[str, Any]] = {}
    eval_report_paths_by_format: dict[str, str] = {}
    delta_reports_by_format: dict[str, dict[str, Any]] = {}
    delta_report_paths_by_format: dict[str, str] = {}
    for score_target_format in args.score_target_formats_parsed:
        margin_rows = score_margins(
            model,
            tokenizer,
            heldout_rows,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            score_label="trained_heldout",
            target_format=score_target_format,
        )
        margin_rows_by_format[score_target_format] = margin_rows
        margins_path = target_format_output_path(args.margins_out, score_target_format, args.target_format)
        eval_path = target_format_output_path(args.eval_out, score_target_format, args.target_format)
        write_jsonl(margins_path, margin_rows)
        margin_report_for_format = build_margin_report(
            run_id=args.run_id,
            model_id=args.model,
            rows=margin_rows,
            score_label="trained_heldout",
        )
        margin_reports_by_format[score_target_format] = margin_report_for_format
        write_json(eval_path, margin_report_for_format)
        eval_report_paths_by_format[score_target_format] = str(eval_path)

        base_margin_rows = base_margin_rows_by_format.get(score_target_format, [])
        if base_margin_rows:
            delta_report_for_format = build_margin_delta_report(
                run_id=args.run_id,
                model_id=args.model,
                base_rows=base_margin_rows,
                trained_rows=margin_rows,
            )
            delta_reports_by_format[score_target_format] = delta_report_for_format
            delta_eval_path = target_format_output_path(
                args.delta_eval_out,
                score_target_format,
                args.target_format,
            )
            write_json(delta_eval_path, delta_report_for_format)
            delta_report_paths_by_format[score_target_format] = str(delta_eval_path)

    primary_margin_report = margin_reports_by_format[args.target_format]
    primary_delta_report = delta_reports_by_format.get(args.target_format)

    trained_candidate_rows: list[dict[str, Any]] = []
    trained_candidate_report: dict[str, Any] | None = None
    if args.score_trained_candidates:
        trained_candidate_rows = score_candidates(
            model,
            tokenizer,
            heldout_rows,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            score_label="trained_heldout",
            candidate_pairs=candidate_pairs,
            candidate_policy=args.candidate_policy,
            target_format=args.candidate_target_format,
        )
        write_jsonl(args.candidates_out, trained_candidate_rows)
        trained_candidate_report = build_candidate_report(
            run_id=args.run_id,
            model_id=args.model,
            rows=trained_candidate_rows,
            score_label="trained_heldout",
            candidate_policy=args.candidate_policy,
            target_format=args.candidate_target_format,
        )
        write_json(args.candidate_eval_out, trained_candidate_report)

    return {
        "dataset": DATASET,
        "dry_run": False,
        "run_id": args.run_id,
        "model": args.model,
        "failure_mode": FAILURE_MODE,
        "calibration_axis": row_value_summary(score_train_rows + heldout_rows, "calibration_axis", CALIBRATION_AXIS),
        "device": device,
        "prompt_contract": PROMPT_CONTRACT,
        "target_format": args.target_format,
        "pairs": args.pairs,
        "train_pairs": args.train_pairs,
        "heldout_pairs": args.heldout_pairs,
        "train_examples": len(score_train_rows),
        "training_examples": len(training_rows),
        "heldout_examples": len(heldout_rows),
        "heldout_used_for_training": False,
        "focus_chosen_pairs": list(args.focus_chosen_pairs_parsed),
        "focus_repeat": args.focus_repeat,
        "focus_only": args.focus_only,
        "train_by_chosen_pair": count_by(score_train_rows, "chosen_pair"),
        "training_by_chosen_pair": count_by(training_rows, "chosen_pair"),
        "heldout_by_chosen_pair": count_by(heldout_rows, "chosen_pair"),
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
        "target_format": args.target_format,
        "score_target_formats": list(args.score_target_formats_parsed),
        "score_base_candidates": args.score_base_candidates,
        "score_train_candidates": args.score_train_candidates,
        "score_trained_candidates": args.score_trained_candidates,
        "candidate_policy": args.candidate_policy,
        "candidate_target_format": args.candidate_target_format,
        "candidate_space_size": len(candidate_pairs),
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
        "base_margins": str(args.base_margins_out) if args.target_format in base_margin_rows_by_format else None,
        "base_eval_report": str(args.base_eval_out) if args.target_format in base_margin_rows_by_format else None,
        "train_margins": str(args.train_margins_out) if args.target_format in train_margin_rows_by_format else None,
        "train_eval_report": str(args.train_eval_out) if args.target_format in train_margin_rows_by_format else None,
        "margins": str(args.margins_out),
        "eval_report": str(args.eval_out),
        "eval_summary": primary_margin_report["summary"],
        "delta_eval_report": str(args.delta_eval_out) if primary_delta_report else None,
        "delta_summary": primary_delta_report["summary"] if primary_delta_report else None,
        "base_candidates": str(args.base_candidates_out) if base_candidate_rows else None,
        "base_candidate_eval_report": str(args.base_candidate_eval_out) if base_candidate_report else None,
        "base_candidate_summary": base_candidate_report["summary"] if base_candidate_report else None,
        "train_candidates": str(args.train_candidates_out) if train_candidate_rows else None,
        "train_candidate_eval_report": (
            str(args.train_candidate_eval_out) if train_candidate_report else None
        ),
        "train_candidate_summary": train_candidate_report["summary"] if train_candidate_report else None,
        "candidates": str(args.candidates_out) if trained_candidate_rows else None,
        "candidate_eval_report": str(args.candidate_eval_out) if trained_candidate_report else None,
        "candidate_summary": trained_candidate_report["summary"] if trained_candidate_report else None,
        "score_target_format_reports": {
            target_format: {
                "base_eval_report": base_report_paths_by_format.get(target_format),
                "train_eval_report": train_report_paths_by_format.get(target_format),
                "eval_report": eval_report_paths_by_format.get(target_format),
                "delta_eval_report": delta_report_paths_by_format.get(target_format),
                "eval_summary": margin_reports_by_format[target_format]["summary"],
                "delta_summary": (
                    delta_reports_by_format[target_format]["summary"]
                    if target_format in delta_reports_by_format
                    else None
                ),
            }
            for target_format in args.score_target_formats_parsed
        },
        "artifact_policy": {
            "raw_prompt_messages_committed": False,
            "raw_model_text_committed": False,
            "scheduler_logs_committed": False,
            "model_state_committed": False,
            "margin_jsonl_committed": False,
            "candidate_jsonl_committed": False,
        },
        "boundary": (
            "Full mode trains on train-allowed saved-output calibration targets, "
            "then scores held-out target-vs-ground/supported margins. Raw run "
            "artifacts belong under post_training/runs/. This is not DPO/RLVR."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--pairs", default="post_training/stage_a_saved_output_calibration_probe_v1.jsonl")
    parser.add_argument(
        "--train-pairs",
        default="post_training/stage_a_saved_output_calibration_probe_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-pairs",
        default="post_training/stage_a_saved_output_calibration_probe_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="post_training/stage_a_saved_output_calibration_probe_manifest.json",
    )
    parser.add_argument("--out-dir", default="post_training/runs/stage_a_saved_output_calibration_margin_sft")
    parser.add_argument("--run-id", default="stage_a_saved_output_calibration_margin_sft")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-heldout", type=int, default=None)
    parser.add_argument(
        "--focus-chosen-pairs",
        default="",
        help=(
            "Comma- or colon-separated chosen_pair labels to oversample during "
            "training, e.g. flag/invalid_value,defer/insufficient."
        ),
    )
    parser.add_argument("--focus-repeat", type=int, default=1)
    parser.add_argument("--focus-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument(
        "--pairwise-margin-weight",
        type=float,
        default=1.0,
        help=(
            "Supervised hinge-loss weight over chosen target vs rejected "
            "ground/supported collapse outputs. This is not DPO/RLVR."
        ),
    )
    parser.add_argument("--pairwise-margin", type=float, default=0.05)
    parser.add_argument(
        "--margin-logprob-mode",
        choices=("mean", "sum"),
        default="mean",
        help="Log-probability reduction used by the pairwise margin objective.",
    )
    parser.add_argument(
        "--candidate-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Optional listwise finite-candidate CE objective over action/status "
            "candidates. This targets candidate routing directly and is not "
            "DPO/RLVR."
        ),
    )
    parser.add_argument(
        "--candidate-ce-mode",
        choices=CANDIDATE_CE_MODES,
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
        help="Log-probability reduction used by the optional candidate CE objective.",
    )
    parser.add_argument(
        "--target-format",
        choices=TARGET_FORMATS,
        default="full",
        help=(
            "Projection used for chosen/rejected teacher-forced targets. "
            "Use action_only or action_status_only to isolate action/status "
            "learning from full JSON/citation/tool formatting."
        ),
    )
    parser.add_argument(
        "--score-target-formats",
        default="",
        help=(
            "Optional comma- or colon-separated target projections to score "
            "on the same base/trained model. The training --target-format is "
            "always included."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--score-base-margins", action="store_true")
    parser.add_argument("--score-train-margins", action="store_true")
    parser.add_argument("--score-base-candidates", action="store_true")
    parser.add_argument(
        "--score-train-candidates",
        action="store_true",
        help=(
            "After training, score finite candidates on train rows. Use this "
            "for train-side calibration diagnostics; do not tune on held-out "
            "candidate scores."
        ),
    )
    parser.add_argument("--score-trained-candidates", action="store_true")
    parser.add_argument(
        "--candidate-policy",
        choices=CANDIDATE_POLICIES,
        default="train_observed_plus_rejected",
        help=(
            "Finite candidate space for post-SFT ranking. "
            "train_observed_plus_rejected includes train target pairs plus "
            "the observed ground/supported collapse."
        ),
    )
    parser.add_argument(
        "--candidate-target-format",
        choices=TARGET_FORMATS,
        default="full",
        help="Output projection used for finite-candidate scoring.",
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
    parser.add_argument("--base-candidates-out", default=None)
    parser.add_argument("--base-candidate-eval-out", default=None)
    parser.add_argument("--train-candidates-out", default=None)
    parser.add_argument("--train-candidate-eval-out", default=None)
    parser.add_argument("--candidates-out", default=None)
    parser.add_argument("--candidate-eval-out", default=None)
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
    if args.base_candidates_out is None:
        args.base_candidates_out = str(out_dir / "base_candidates.jsonl")
    if args.base_candidate_eval_out is None:
        args.base_candidate_eval_out = str(out_dir / "base_candidate_report.json")
    if args.train_candidates_out is None:
        args.train_candidates_out = str(out_dir / "train_candidates.jsonl")
    if args.train_candidate_eval_out is None:
        args.train_candidate_eval_out = str(out_dir / "train_candidate_report.json")
    if args.candidates_out is None:
        args.candidates_out = str(out_dir / "candidates.jsonl")
    if args.candidate_eval_out is None:
        args.candidate_eval_out = str(out_dir / "candidate_report.json")

    all_rows = load_jsonl(args.pairs)
    all_train_rows = load_jsonl(args.train_pairs)
    all_heldout_rows = load_jsonl(args.heldout_pairs)
    manifest = load_manifest(args.manifest)
    issues = validate_saved_output_calibration_margin_artifacts(
        all_rows,
        all_train_rows,
        all_heldout_rows,
        manifest,
    )
    if args.pairwise_margin_weight < 0:
        issues.append("pairwise_margin_weight_negative")
    if args.pairwise_margin < 0:
        issues.append("pairwise_margin_negative")
    if args.candidate_ce_weight < 0:
        issues.append("candidate_ce_weight_negative")

    train_rows = all_train_rows[: args.limit_train] if args.limit_train is not None else all_train_rows
    heldout_rows = all_heldout_rows[: args.limit_heldout] if args.limit_heldout is not None else all_heldout_rows
    args.focus_chosen_pairs_parsed = parse_focus_chosen_pairs(args.focus_chosen_pairs)
    try:
        args.score_target_formats_parsed = parse_score_target_formats(
            args.score_target_formats,
            training_target_format=args.target_format,
        )
    except ValueError as exc:
        issues.append(f"score_target_formats_invalid:{exc}")
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
    try:
        candidate_pairs_for_policy(train_rows, policy=args.candidate_policy)
    except ValueError as exc:
        issues.append(f"candidate_policy_invalid:{exc}")

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
                pairwise_margin_weight=args.pairwise_margin_weight,
                pairwise_margin=args.pairwise_margin,
                margin_logprob_mode=args.margin_logprob_mode,
                candidate_ce_weight=args.candidate_ce_weight,
                candidate_ce_mode=args.candidate_ce_mode,
                candidate_ce_logprob_mode=args.candidate_ce_logprob_mode,
                target_format=args.target_format,
                score_target_formats=getattr(args, "score_target_formats_parsed", (args.target_format,)),
                score_base_margins=args.score_base_margins,
                score_train_margins=args.score_train_margins,
                score_base_candidates=args.score_base_candidates,
                score_train_candidates=args.score_train_candidates,
                score_trained_candidates=args.score_trained_candidates,
                candidate_policy=args.candidate_policy,
                candidate_target_format=args.candidate_target_format,
            ),
        )
        raise SystemExit("Saved-output calibration margin SFT validation failed:\n- " + "\n- ".join(issues))

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
            pairwise_margin_weight=args.pairwise_margin_weight,
            pairwise_margin=args.pairwise_margin,
            margin_logprob_mode=args.margin_logprob_mode,
            candidate_ce_weight=args.candidate_ce_weight,
            candidate_ce_mode=args.candidate_ce_mode,
            candidate_ce_logprob_mode=args.candidate_ce_logprob_mode,
            target_format=args.target_format,
            score_target_formats=args.score_target_formats_parsed,
            score_base_margins=args.score_base_margins,
            score_train_margins=args.score_train_margins,
            score_base_candidates=args.score_base_candidates,
            score_train_candidates=args.score_train_candidates,
            score_trained_candidates=args.score_trained_candidates,
            candidate_policy=args.candidate_policy,
            candidate_target_format=args.candidate_target_format,
        )
    else:
        if not args.allow_model_load:
            raise SystemExit(
                "Full saved-output calibration margin SFT requires --allow-model-load. Use --dry-run locally."
            )
        report = run_training_and_eval(args, training_rows, train_rows, heldout_rows)
    write_json(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
