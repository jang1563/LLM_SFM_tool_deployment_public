#!/usr/bin/env python3
"""Run Stage A routing action/status contrast SFT/margin smoke experiments.

This is the cluster-oriented follow-up to the routing action/status contrast
export. It trains on chosen `routing_after_loop` targets and evaluates held-out
pairs by teacher-forced chosen-vs-rejected likelihood margin.

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

from post_training.export_stage_a_evidence_conditioned_component_targets import (  # noqa: E402
    PROMPT_CONTRACT,
)
from post_training.export_stage_a_routing_action_status_contrast_pairs import (  # noqa: E402
    CANDIDATE_POLICY,
    COMPONENT,
    DATASET as PAIR_DATASET,
    FAILURE_MODE,
)
from post_training.export_stage_a_routing_defer_verify_contrast_pairs import (  # noqa: E402
    CANDIDATE_POLICY as DEFER_VERIFY_CANDIDATE_POLICY,
    DATASET as DEFER_VERIFY_PAIR_DATASET,
    FAILURE_MODE as DEFER_VERIFY_FAILURE_MODE,
)
from post_training.generate_stage_a_predictions import disable_transformers_torchvision_probe  # noqa: E402
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    prompt_messages_from_row,
    routing_candidates_for_row,
    score_candidate_target,
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
    validate_stage_a_routing_defer_verify_contrast_pairs,
    validate_stage_a_routing_action_status_contrast_pairs,
)


DATASET = "negbiodb_ct_stage_a_routing_contrast_sft_smoke_v1"
MARGIN_DATASET = "negbiodb_ct_stage_a_routing_contrast_margin_rows_v1"
MARGIN_DELTA_DATASET = "negbiodb_ct_stage_a_routing_contrast_margin_delta_v1"
CANDIDATE_PREDICTION_DATASET = "negbiodb_ct_stage_a_routing_contrast_candidate_predictions_v1"
CANDIDATE_RANK_DATASET = "negbiodb_ct_stage_a_routing_contrast_candidate_rank_v1"


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def pair_case_id(row: Mapping[str, Any]) -> str:
    value = row.get("source_manifest_case_id") or row.get("source_component_target_id") or row.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Routing contrast pair is missing case id: {row.get('id')!r}")
    return value


def output_text(output: Mapping[str, Any]) -> str:
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


def action_status_key(output: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    if not isinstance(output, Mapping):
        return None, None
    action = output.get("action")
    status = output.get("evidence_status")
    return (
        str(action) if action is not None else None,
        str(status) if status is not None else None,
    )


def candidate_key(output: Mapping[str, Any] | None) -> tuple[str | None, str | None, tuple[str, ...]]:
    action, status = action_status_key(output)
    citations = output.get("cited_source_ids") if isinstance(output, Mapping) else None
    return (
        action,
        status,
        tuple(str(item) for item in citations) if isinstance(citations, list) else (),
    )


def pair_label(output: Mapping[str, Any] | None) -> str:
    action, status = action_status_key(output)
    return f"{action}/{status}"


def routing_action_status_candidates_from_pairs(
    train_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in train_rows:
        for output in (chosen_output_from_pair(row), rejected_output_from_pair(row)):
            action = output.get("action")
            evidence_status = output.get("evidence_status")
            if not isinstance(action, str) or not isinstance(evidence_status, str):
                continue
            key = (action, evidence_status)
            if key in seen:
                continue
            seen.add(key)
            outputs.append({"action": action, "evidence_status": evidence_status})
    return outputs


def sorted_candidate_scores(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    scores = row.get("candidate_scores")
    if not isinstance(scores, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in scores:
        if not isinstance(item, Mapping):
            continue
        candidate = item.get("candidate")
        if not isinstance(candidate, Mapping):
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        cleaned.append({"candidate": dict(candidate), "score": score})
    return sorted(cleaned, key=lambda item: (-item["score"], json.dumps(item["candidate"], sort_keys=True)))


def find_candidate_rank(
    scores: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> tuple[int | None, float | None]:
    target_key = candidate_key(target)
    for index, score_row in enumerate(scores, start=1):
        candidate = score_row.get("candidate")
        if isinstance(candidate, Mapping) and candidate_key(candidate) == target_key:
            return index, float(score_row["score"])
    return None, None


def find_action_status_rank(
    scores: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> tuple[int | None, float | None]:
    target_key = action_status_key(target)
    for index, score_row in enumerate(scores, start=1):
        candidate = score_row.get("candidate")
        if isinstance(candidate, Mapping) and action_status_key(candidate) == target_key:
            return index, float(score_row["score"])
    return None, None


def encode_pair_output(
    tokenizer: Any,
    row: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    max_length: int,
) -> dict[str, Any]:
    import torch

    prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_row(row))
    target = output_text(output) + tokenizer.eos_token
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


def encode_chosen_pair(tokenizer: Any, row: Mapping[str, Any], *, max_length: int) -> dict[str, Any]:
    return encode_pair_output(tokenizer, row, chosen_output_from_pair(row), max_length=max_length)


def encode_pairwise_margin_pair(tokenizer: Any, row: Mapping[str, Any], *, max_length: int) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "chosen_pair": row.get("chosen_pair"),
        "rejected_pair": row.get("rejected_pair"),
        "chosen": encode_pair_output(tokenizer, row, chosen_output_from_pair(row), max_length=max_length),
        "rejected": encode_pair_output(tokenizer, row, rejected_output_from_pair(row), max_length=max_length),
    }


def pairwise_margin_loss_from_logps(chosen_logps: Any, rejected_logps: Any, *, margin: float) -> Any:
    if margin < 0:
        raise ValueError("pairwise margin must be non-negative")
    import torch.nn.functional as F

    desired = chosen_logps.new_tensor(float(margin))
    return F.relu(desired - (chosen_logps - rejected_logps)).mean()


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


def validate_routing_contrast_artifacts(
    all_rows: list[Mapping[str, Any]],
    train_rows: list[Mapping[str, Any]],
    heldout_rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    pair_dataset = manifest.get("pair_dataset")
    if pair_dataset == PAIR_DATASET:
        issues = validate_stage_a_routing_action_status_contrast_pairs(
            all_rows,
            train_rows,
            heldout_rows,
            manifest,
        )
    elif pair_dataset == DEFER_VERIFY_PAIR_DATASET:
        issues = validate_stage_a_routing_defer_verify_contrast_pairs(
            all_rows,
            train_rows,
            heldout_rows,
            manifest,
        )
    else:
        issues = [f"unsupported_routing_contrast_pair_dataset:{pair_dataset}"]
    expected_failure_mode = str(manifest.get("failure_mode", FAILURE_MODE))
    expected_candidate_policy = str(manifest.get("candidate_policy", CANDIDATE_POLICY))
    if not train_rows:
        issues.append("empty_train_pairs")
    if not heldout_rows:
        issues.append("empty_heldout_pairs")
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
    return issues


def margin_row_for_pair(
    row: Mapping[str, Any],
    *,
    run_id: str,
    model_id: str,
    chosen_score: float,
    rejected_score: float,
    score_label: str,
) -> dict[str, Any]:
    margin = round(float(chosen_score) - float(rejected_score), 6)
    passed = margin > 0.0
    return {
        "id": f"{run_id}::{score_label}::{row['id']}",
        "dataset": MARGIN_DATASET,
        "source": "stage_a_routing_contrast_sft_smoke",
        "run_id": run_id,
        "model": model_id,
        "score_label": score_label,
        "source_routing_contrast_pair_id": row["id"],
        "source_component_target_id": row.get("source_component_target_id"),
        "case_id": pair_case_id(row),
        "component": COMPONENT,
        "failure_mode": row.get("failure_mode", FAILURE_MODE),
        "contrast_axis": row.get("contrast_axis"),
        "candidate_policy": row.get("candidate_policy"),
        "prompt_contract": PROMPT_CONTRACT,
        "split": row.get("split"),
        "case_family": row.get("case_family"),
        "chosen_pair": row.get("chosen_pair"),
        "rejected_pair": row.get("rejected_pair"),
        "chosen_output": chosen_output_from_pair(row),
        "rejected_output": rejected_output_from_pair(row),
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
        "component": COMPONENT,
        "failure_mode": row_value_summary(rows, "failure_mode", FAILURE_MODE),
        "failure_modes": count_by(rows, "failure_mode"),
        "contrast_axes": count_by(rows, "contrast_axis"),
        "candidate_policies": count_by(rows, "candidate_policy"),
        "prompt_contract": PROMPT_CONTRACT,
        "pairs": len(rows),
        "summary": summarize_margin_rows(rows),
        "rows": list(rows),
        "boundary": (
            "Routing contrast SFT smoke scores teacher-forced chosen-vs-rejected "
            "margins. This is not DPO/RLVR and does not score free-form "
            "explanation quality."
        ),
    }


def margin_delta_row(base_row: Mapping[str, Any], trained_row: Mapping[str, Any]) -> dict[str, Any]:
    pair_id = str(base_row.get("source_routing_contrast_pair_id"))
    if pair_id != str(trained_row.get("source_routing_contrast_pair_id")):
        raise ValueError(f"Cannot compare different routing pairs: {pair_id} vs {trained_row.get('source_routing_contrast_pair_id')}")
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
        "source": "stage_a_routing_contrast_sft_smoke",
        "run_id": trained_row.get("run_id"),
        "model": trained_row.get("model"),
        "source_routing_contrast_pair_id": pair_id,
        "case_id": trained_row.get("case_id"),
        "component": COMPONENT,
        "failure_mode": trained_row.get("failure_mode", FAILURE_MODE),
        "contrast_axis": trained_row.get("contrast_axis"),
        "candidate_policy": trained_row.get("candidate_policy"),
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


def build_margin_delta_report(
    *,
    run_id: str,
    model_id: str,
    base_rows: Sequence[Mapping[str, Any]],
    trained_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_by_pair = {str(row.get("source_routing_contrast_pair_id")): row for row in base_rows}
    trained_by_pair = {str(row.get("source_routing_contrast_pair_id")): row for row in trained_rows}
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
        "prompt_contract": PROMPT_CONTRACT,
        "summary": summarize_margin_delta_rows(row_reports),
        "missing_base_pair_ids": missing_base,
        "missing_trained_pair_ids": missing_trained,
        "rows": row_reports,
        "boundary": (
            "Margin delta compares base-model held-out routing contrast margins "
            "with post-SFT held-out routing contrast margins. This is diagnostic "
            "SFT movement, not DPO/RLVR."
        ),
    }


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
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for index, row in enumerate(rows_to_score):
        prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_row(row))
        chosen_score = score_candidate_target(
            model,
            tokenizer,
            prompt,
            output_text(chosen_output_from_pair(row)),
            device=device,
            max_length=max_length,
        )
        rejected_score = score_candidate_target(
            model,
            tokenizer,
            prompt,
            output_text(rejected_output_from_pair(row)),
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
            )
        )
        print(
            f"[{index + 1}/{len(rows_to_score)}] scored {score_label} {row['id']} "
            f"margin={rows[-1]['margin']:.6f}",
            flush=True,
        )
    return rows


def score_routing_candidates(
    model: Any,
    tokenizer: Any,
    rows_to_score: Sequence[Mapping[str, Any]],
    *,
    action_status_candidates: Sequence[Mapping[str, Any]],
    run_id: str,
    model_id: str,
    device: str,
    max_length: int,
    score_label: str,
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for index, row in enumerate(rows_to_score):
        prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_row(row))
        scored: list[dict[str, Any]] = []
        row_candidates = routing_candidates_for_row(row, action_status_candidates)
        for candidate in row_candidates:
            score = score_candidate_target(
                model,
                tokenizer,
                prompt,
                output_text(candidate),
                device=device,
                max_length=max_length,
            )
            scored.append({"score": score, "candidate": candidate})
        scored.sort(key=lambda item: item["score"], reverse=True)
        winner = scored[0]["candidate"]
        rows.append(
            {
                "id": f"{run_id}::{score_label}::{row['id']}",
                "dataset": CANDIDATE_PREDICTION_DATASET,
                "source": "stage_a_routing_contrast_sft_smoke",
                "run_id": run_id,
                "model": model_id,
                "score_label": score_label,
                "source_routing_contrast_pair_id": row["id"],
                "source_component_target_id": row.get("source_component_target_id"),
                "case_id": pair_case_id(row),
                "component": COMPONENT,
                "failure_mode": row.get("failure_mode", FAILURE_MODE),
                "contrast_axis": row.get("contrast_axis"),
                "candidate_policy": row.get("candidate_policy"),
                "prompt_contract": PROMPT_CONTRACT,
                "split": row.get("split"),
                "case_family": row.get("case_family"),
                "chosen_pair": row.get("chosen_pair"),
                "rejected_pair": row.get("rejected_pair"),
                "target_output": chosen_output_from_pair(row),
                "prediction": winner,
                "raw_output": output_text(winner),
                "candidate_space_size": len(action_status_candidates),
                "candidate_scores": scored,
            }
        )
        print(
            f"[{index + 1}/{len(rows_to_score)}] scored {score_label} candidates "
            f"{row['id']} top={pair_label(winner)}",
            flush=True,
        )
    return rows


def candidate_rank_row(row: Mapping[str, Any]) -> dict[str, Any]:
    raw_target = row.get("target_output")
    target = dict(raw_target) if isinstance(raw_target, Mapping) else chosen_output_from_pair(row)
    scores = sorted_candidate_scores(row)
    top = scores[0] if scores else None
    top_candidate = top["candidate"] if top else None
    top_score = float(top["score"]) if top else None
    gold_rank, gold_score = find_candidate_rank(scores, target)
    pair_rank, _ = find_action_status_rank(scores, target)
    top_gold_margin = None
    if top_score is not None and gold_score is not None:
        top_gold_margin = round(top_score - gold_score, 6)
    exact_top1 = top_candidate is not None and candidate_key(top_candidate) == candidate_key(target)
    action_status_top1 = top_candidate is not None and action_status_key(top_candidate) == action_status_key(target)
    return {
        "source_routing_contrast_pair_id": row.get("source_routing_contrast_pair_id"),
        "case_id": row.get("case_id"),
        "case_family": row.get("case_family"),
        "chosen_pair": row.get("chosen_pair"),
        "rejected_pair": row.get("rejected_pair"),
        "expected": target,
        "top_candidate": top_candidate,
        "exact_top1": exact_top1,
        "action_status_top1": action_status_top1,
        "candidate_scores_present": bool(scores),
        "retained_candidate_count": len(scores),
        "candidate_space_size": row.get("candidate_space_size"),
        "all_candidates_retained": len(scores) == row.get("candidate_space_size"),
        "gold_in_retained_candidates": gold_rank is not None,
        "gold_rank": gold_rank,
        "action_status_rank": pair_rank,
        "gold_score": gold_score,
        "top_score": top_score,
        "top_gold_margin": top_gold_margin,
        "top_candidates": scores[:3],
    }


def summarize_candidate_rank_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "exact_top1": 0,
            "action_status_top1": 0,
            "gold_in_retained_candidates": 0,
            "all_candidates_retained_cases": 0,
            "mean_gold_rank_observed": None,
            "mean_action_status_rank_observed": None,
            "mean_top_gold_margin_observed": None,
            "top_pair_counts": {},
            "target_pair_counts": {},
            "by_chosen_pair": {},
        }

    def mean(values: Sequence[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    gold_ranks = [int(row["gold_rank"]) for row in rows if isinstance(row.get("gold_rank"), int)]
    pair_ranks = [
        int(row["action_status_rank"])
        for row in rows
        if isinstance(row.get("action_status_rank"), int)
    ]
    margins = [float(row["top_gold_margin"]) for row in rows if row.get("top_gold_margin") is not None]
    by_chosen_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chosen_pair[str(row.get("chosen_pair"))].append(row)
    return {
        "cases": len(rows),
        "exact_top1": sum(1 for row in rows if row.get("exact_top1")),
        "action_status_top1": sum(1 for row in rows if row.get("action_status_top1")),
        "gold_in_retained_candidates": sum(1 for row in rows if row.get("gold_in_retained_candidates")),
        "all_candidates_retained_cases": sum(1 for row in rows if row.get("all_candidates_retained")),
        "missing_candidate_scores": sum(1 for row in rows if not row.get("candidate_scores_present")),
        "gold_rank_histogram": dict(sorted(Counter(gold_ranks).items())),
        "action_status_rank_histogram": dict(sorted(Counter(pair_ranks).items())),
        "mean_gold_rank_observed": mean([float(rank) for rank in gold_ranks]),
        "mean_action_status_rank_observed": mean([float(rank) for rank in pair_ranks]),
        "mean_top_gold_margin_observed": mean(margins),
        "top_pair_counts": dict(
            sorted(
                Counter(
                    pair_label(row.get("top_candidate"))
                    for row in rows
                    if isinstance(row.get("top_candidate"), Mapping)
                ).items()
            )
        ),
        "target_pair_counts": dict(
            sorted(
                Counter(
                    pair_label(row.get("expected"))
                    for row in rows
                    if isinstance(row.get("expected"), Mapping)
                ).items()
            )
        ),
        "by_chosen_pair": {
            pair: {
                "cases": len(items),
                "exact_top1": sum(1 for item in items if item.get("exact_top1")),
                "action_status_top1": sum(1 for item in items if item.get("action_status_top1")),
                "mean_gold_rank_observed": mean(
                    [float(item["gold_rank"]) for item in items if isinstance(item.get("gold_rank"), int)]
                ),
                "mean_top_gold_margin_observed": mean(
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


def build_candidate_rank_report(
    *,
    run_id: str,
    model_id: str,
    rows: Sequence[Mapping[str, Any]],
    score_label: str,
    action_status_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rank_rows = [candidate_rank_row(row) for row in rows]
    return {
        "dataset": CANDIDATE_RANK_DATASET,
        "run_id": run_id,
        "model": model_id,
        "score_label": score_label,
        "component": COMPONENT,
        "failure_mode": row_value_summary(rows, "failure_mode", FAILURE_MODE),
        "failure_modes": count_by(rows, "failure_mode"),
        "contrast_axes": count_by(rows, "contrast_axis"),
        "candidate_policy": "train_observed_routing_contrast_pairs_with_visible_citations",
        "candidate_outputs": [dict(candidate) for candidate in action_status_candidates],
        "candidate_space_size": len(action_status_candidates),
        "prompt_contract": PROMPT_CONTRACT,
        "summary": summarize_candidate_rank_rows(rank_rows),
        "rows": rank_rows,
        "boundary": (
            "Routing contrast candidate rank scoring evaluates finite-candidate "
            "action/status selection after the margin repair. It is not "
            "free-form generation, DPO, RLVR, or trajectory-level scoring."
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
    pairwise_margin_weight: float,
    pairwise_margin: float,
    margin_logprob_mode: str,
    score_base_margins: bool,
    score_train_margins: bool,
    score_base_routing_candidates: bool,
    score_trained_routing_candidates: bool,
) -> dict[str, Any]:
    selected_rows = [dict(row) for row in train_rows] + [dict(row) for row in heldout_rows]
    action_status_candidates = routing_action_status_candidates_from_pairs(train_rows)
    return {
        "dataset": DATASET,
        "dry_run": True,
        "model": model,
        "component": COMPONENT,
        "failure_mode": row_value_summary(selected_rows, "failure_mode", FAILURE_MODE),
        "failure_modes": count_by(selected_rows, "failure_mode"),
        "candidate_policies": count_by(selected_rows, "candidate_policy"),
        "contrast_axes": count_by(selected_rows, "contrast_axis"),
        "pairwise_margin_weight": pairwise_margin_weight,
        "pairwise_margin": pairwise_margin,
        "margin_logprob_mode": margin_logprob_mode,
        "score_base_margins": score_base_margins,
        "score_train_margins": score_train_margins,
        "score_base_routing_candidates": score_base_routing_candidates,
        "score_trained_routing_candidates": score_trained_routing_candidates,
        "routing_candidate_policy": "train_observed_routing_contrast_pairs_with_visible_citations",
        "routing_candidate_space_size": len(action_status_candidates),
        "routing_candidate_outputs": action_status_candidates,
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
            "Dry run validates routing contrast pair artifacts and split "
            "boundaries without loading model weights or running local heavy compute."
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
    encoded = [encode_chosen_pair(tokenizer, row, max_length=args.max_length) for row in training_rows]
    encoded_margin_pairs = [
        encode_pairwise_margin_pair(tokenizer, row, max_length=args.max_length)
        for row in training_rows
    ] if args.pairwise_margin_weight > 0 else []

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        torch_dtype="auto",
    )
    model.to(device)
    action_status_candidates = routing_action_status_candidates_from_pairs(score_train_rows)

    base_margin_rows: list[dict[str, Any]] = []
    if args.score_base_margins:
        base_margin_rows = score_margins(
            model,
            tokenizer,
            heldout_rows,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            score_label="base_heldout",
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
    base_candidate_report = None
    if args.score_base_routing_candidates:
        base_candidate_rows = score_routing_candidates(
            model,
            tokenizer,
            heldout_rows,
            action_status_candidates=action_status_candidates,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            score_label="base_candidate_heldout",
        )
        write_jsonl(args.base_routing_candidates_out, base_candidate_rows)
        base_candidate_report = build_candidate_rank_report(
            run_id=args.run_id,
            model_id=args.model,
            rows=base_candidate_rows,
            score_label="base_candidate_heldout",
            action_status_candidates=action_status_candidates,
        )
        write_json(args.base_routing_candidate_report_out, base_candidate_report)

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
    cursor = 0
    for step in range(args.max_steps):
        batch_features = []
        batch_margin_pairs = []
        for _ in range(args.batch_size):
            index = cursor % len(encoded)
            batch_features.append(encoded[index])
            if encoded_margin_pairs:
                batch_margin_pairs.append(encoded_margin_pairs[index])
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

        loss = ce_loss + (args.pairwise_margin_weight * pairwise_margin_loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ce_losses.append(float(ce_loss.detach().cpu()))
        pairwise_margin_losses.append(float(pairwise_margin_loss.detach().cpu()))
        pairwise_train_margins.append(step_pairwise_train_margin)
        print(
            f"step={step + 1} loss={losses[-1]:.4f} ce={ce_losses[-1]:.4f} "
            f"pairwise={pairwise_margin_losses[-1]:.4f} "
            f"train_margin={pairwise_train_margins[-1]:.4f}",
            flush=True,
        )

    state_path = None
    if not args.no_save_trainable_state:
        state_path = save_trainable_state(model, out_dir)

    train_margin_rows: list[dict[str, Any]] = []
    if args.score_train_margins:
        train_margin_rows = score_margins(
            model,
            tokenizer,
            score_train_rows,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            score_label="trained_train",
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

    margin_rows = score_margins(
        model,
        tokenizer,
        heldout_rows,
        run_id=args.run_id,
        model_id=args.model,
        device=device,
        max_length=args.max_length,
        score_label="trained_heldout",
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
    if args.score_trained_routing_candidates:
        candidate_rows = score_routing_candidates(
            model,
            tokenizer,
            heldout_rows,
            action_status_candidates=action_status_candidates,
            run_id=args.run_id,
            model_id=args.model,
            device=device,
            max_length=args.max_length,
            score_label="trained_candidate_heldout",
        )
        write_jsonl(args.routing_candidates_out, candidate_rows)
        candidate_report = build_candidate_rank_report(
            run_id=args.run_id,
            model_id=args.model,
            rows=candidate_rows,
            score_label="trained_candidate_heldout",
            action_status_candidates=action_status_candidates,
        )
        write_json(args.routing_candidate_report_out, candidate_report)

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
        "heldout_by_chosen_pair": count_by(heldout_rows, "chosen_pair"),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_last_layers": args.train_last_layers,
        "pairwise_margin_weight": args.pairwise_margin_weight,
        "pairwise_margin": args.pairwise_margin,
        "margin_logprob_mode": args.margin_logprob_mode,
        "routing_candidate_policy": "train_observed_routing_contrast_pairs_with_visible_citations",
        "routing_candidate_space_size": len(action_status_candidates),
        "routing_candidate_outputs": action_status_candidates,
        "trainable_params": trainable_params,
        "losses": losses,
        "ce_losses": ce_losses,
        "pairwise_margin_losses": pairwise_margin_losses,
        "pairwise_train_margins": pairwise_train_margins,
        "loss_delta": round(losses[-1] - losses[0], 6) if len(losses) > 1 else 0.0,
        "trainable_state": str(state_path) if state_path else None,
        "base_margins": str(args.base_margins_out) if base_margin_rows else None,
        "base_eval_report": str(args.base_eval_out) if base_margin_rows else None,
        "base_routing_candidates": str(args.base_routing_candidates_out) if base_candidate_rows else None,
        "base_routing_candidate_report": (
            str(args.base_routing_candidate_report_out) if base_candidate_report else None
        ),
        "base_routing_candidate_summary": (
            base_candidate_report["summary"] if base_candidate_report else None
        ),
        "train_margins": str(args.train_margins_out) if train_margin_rows else None,
        "train_eval_report": str(args.train_eval_out) if train_margin_rows else None,
        "margins": str(args.margins_out),
        "eval_report": str(args.eval_out),
        "eval_summary": margin_report["summary"],
        "routing_candidates": str(args.routing_candidates_out) if candidate_rows else None,
        "routing_candidate_report": str(args.routing_candidate_report_out) if candidate_report else None,
        "routing_candidate_summary": candidate_report["summary"] if candidate_report else None,
        "delta_eval_report": str(args.delta_eval_out) if delta_report else None,
        "delta_summary": delta_report["summary"] if delta_report else None,
        "boundary": (
            "Full mode trains on chosen routing contrast targets, then scores "
            "held-out chosen/rejected margins. Raw run artifacts belong under "
            "post_training/runs/. This is not DPO/RLVR."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--pairs", default="post_training/stage_a_routing_action_status_contrast_pairs_v1.jsonl")
    parser.add_argument(
        "--train-pairs",
        default="post_training/stage_a_routing_action_status_contrast_pairs_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-pairs",
        default="post_training/stage_a_routing_action_status_contrast_pairs_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="post_training/stage_a_routing_action_status_contrast_pairs_manifest.json",
    )
    parser.add_argument("--out-dir", default="post_training/runs/stage_a_routing_contrast_sft_smoke")
    parser.add_argument("--run-id", default="stage_a_routing_contrast_sft_smoke")
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
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument(
        "--pairwise-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Optional supervised hinge-loss weight. When positive, training "
            "adds max(0, margin - (logp(chosen)-logp(rejected))) over the same "
            "routing contrast pairs. This is not DPO/RLVR."
        ),
    )
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    parser.add_argument(
        "--margin-logprob-mode",
        choices=("mean", "sum"),
        default="mean",
        help="Log-probability reduction used by the optional pairwise margin objective.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--score-base-margins", action="store_true")
    parser.add_argument("--score-train-margins", action="store_true")
    parser.add_argument(
        "--score-base-routing-candidates",
        action="store_true",
        help=(
            "Before training, score held-out finite routing candidates built from "
            "train-observed contrast action/status pairs. Writes ignored raw "
            "candidate JSONL plus a compact rank report."
        ),
    )
    parser.add_argument(
        "--score-trained-routing-candidates",
        action="store_true",
        help=(
            "After training, score held-out finite routing candidates built from "
            "train-observed contrast action/status pairs. This is a candidate "
            "rank diagnostic, not DPO/RLVR."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-trainable-state", action="store_true")
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--base-margins-out", default=None)
    parser.add_argument("--base-eval-out", default=None)
    parser.add_argument("--base-routing-candidates-out", default=None)
    parser.add_argument("--base-routing-candidate-report-out", default=None)
    parser.add_argument("--train-margins-out", default=None)
    parser.add_argument("--train-eval-out", default=None)
    parser.add_argument("--margins-out", default=None)
    parser.add_argument("--eval-out", default=None)
    parser.add_argument("--routing-candidates-out", default=None)
    parser.add_argument("--routing-candidate-report-out", default=None)
    parser.add_argument("--delta-eval-out", default=None)
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
    if args.base_routing_candidates_out is None:
        args.base_routing_candidates_out = str(out_dir / "base_routing_candidates.jsonl")
    if args.base_routing_candidate_report_out is None:
        args.base_routing_candidate_report_out = str(out_dir / "base_routing_candidate_report.json")
    if args.train_margins_out is None:
        args.train_margins_out = str(out_dir / "train_margins.jsonl")
    if args.train_eval_out is None:
        args.train_eval_out = str(out_dir / "train_margin_report.json")
    if args.margins_out is None:
        args.margins_out = str(out_dir / "margins.jsonl")
    if args.eval_out is None:
        args.eval_out = str(out_dir / "margin_report.json")
    if args.routing_candidates_out is None:
        args.routing_candidates_out = str(out_dir / "routing_candidates.jsonl")
    if args.routing_candidate_report_out is None:
        args.routing_candidate_report_out = str(out_dir / "routing_candidate_report.json")
    if args.delta_eval_out is None:
        args.delta_eval_out = str(out_dir / "margin_delta_report.json")

    all_rows = load_jsonl(args.pairs)
    all_train_rows = load_jsonl(args.train_pairs)
    all_heldout_rows = load_jsonl(args.heldout_pairs)
    manifest = load_manifest(args.manifest)
    issues = validate_routing_contrast_artifacts(all_rows, all_train_rows, all_heldout_rows, manifest)
    if args.pairwise_margin_weight < 0:
        issues.append("pairwise_margin_weight_negative")
    if args.pairwise_margin < 0:
        issues.append("pairwise_margin_negative")

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
                pairwise_margin_weight=args.pairwise_margin_weight,
                pairwise_margin=args.pairwise_margin,
                margin_logprob_mode=args.margin_logprob_mode,
                score_base_margins=args.score_base_margins,
                score_train_margins=args.score_train_margins,
                score_base_routing_candidates=args.score_base_routing_candidates,
                score_trained_routing_candidates=args.score_trained_routing_candidates,
            ),
        )
        raise SystemExit("Routing contrast SFT smoke validation failed:\n- " + "\n- ".join(issues))

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
            score_base_margins=args.score_base_margins,
            score_train_margins=args.score_train_margins,
            score_base_routing_candidates=args.score_base_routing_candidates,
            score_trained_routing_candidates=args.score_trained_routing_candidates,
        )
    else:
        if not args.allow_model_load:
            raise SystemExit("Full routing contrast SFT smoke requires --allow-model-load. Use --dry-run locally.")
        report = run_training_and_eval(args, training_rows, train_rows, heldout_rows)
    write_json(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
