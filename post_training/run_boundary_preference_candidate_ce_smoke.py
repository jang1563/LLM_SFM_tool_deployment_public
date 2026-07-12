#!/usr/bin/env python3
"""All-candidate CE smoke loop for boundary preference prompts."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_boundary_preference_candidate_eval import (  # noqa: E402
    candidate_actions_match,
    candidate_key,
    candidates_for_pair,
    candidates_match,
)
from post_training.run_boundary_preference_dpo_smoke import (  # noqa: E402
    apply_limit,
    encode_candidate_target,
    filter_pairs,
    parse_failure_modes,
    sequence_logps,
)
from post_training.run_boundary_preference_margin import (  # noqa: E402
    final_candidate,
    margin_stats,
    prompt_from_pair,
)
from post_training.run_sft_smoke import (  # noqa: E402
    choose_device,
    collate,
    load_jsonl,
    save_trainable_state,
    set_trainable_last_layers,
)


BOUNDARY_CURRICULUM_PHASES = {
    "defer": ("boundary_defer_over_verify",),
    "flag": ("boundary_flag_over_ground",),
    "reject": ("boundary_reject_over_ground", "boundary_reject_over_flag"),
}


def parse_phase_order(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ("defer", "flag", "reject")
    phases = tuple(part.strip() for part in value.split(",") if part.strip())
    if not phases:
        raise ValueError("Phase order cannot be empty.")
    unknown = sorted(set(phases) - set(BOUNDARY_CURRICULUM_PHASES))
    if unknown:
        raise ValueError(f"Unknown curriculum phase(s): {', '.join(unknown)}")
    return phases


def parse_action_loss_weights(value: str | None) -> dict[str, float]:
    if value is None or not value.strip():
        return {}
    weights: dict[str, float] = {}
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Action loss weight must use action=value format: {item}")
        action, raw_weight = (piece.strip() for piece in item.split("=", 1))
        if not action:
            raise ValueError(f"Action loss weight has empty action name: {item}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid action loss weight for {action}: {raw_weight}") from exc
        if weight <= 0:
            raise ValueError(f"Action loss weight must be positive for {action}: {weight}")
        weights[action] = weight
    return weights


def parse_action_floors(value: str | None) -> dict[str, float]:
    if value is None or not value.strip():
        return {}
    floors: dict[str, float] = {}
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Action floor must use action=value format: {item}")
        action, raw_floor = (piece.strip() for piece in item.split("=", 1))
        if not action:
            raise ValueError(f"Action floor has empty action name: {item}")
        try:
            floor = float(raw_floor)
        except ValueError as exc:
            raise ValueError(f"Invalid action floor for {action}: {raw_floor}") from exc
        if floor < 0 or floor > 1:
            raise ValueError(f"Action floor must be between 0 and 1 for {action}: {floor}")
        floors[action] = floor
    return floors


def parse_action_margin_penalties(value: str | None) -> dict[str, float]:
    if value is None or not value.strip():
        return {}
    penalties: dict[str, float] = {}
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Action margin penalty must use target>competitor=value format: {item}")
        raw_key, raw_margin = (piece.strip() for piece in item.split("=", 1))
        if ">" not in raw_key:
            raise ValueError(f"Action margin penalty must use target>competitor key: {item}")
        target, competitor = (piece.strip() for piece in raw_key.split(">", 1))
        if not target or not competitor:
            raise ValueError(f"Action margin penalty has empty target or competitor: {item}")
        if target == competitor:
            raise ValueError(f"Action margin penalty target and competitor must differ: {item}")
        try:
            margin = float(raw_margin)
        except ValueError as exc:
            raise ValueError(f"Invalid action margin for {raw_key}: {raw_margin}") from exc
        if margin < 0:
            raise ValueError(f"Action margin must be non-negative for {raw_key}: {margin}")
        penalties[f"{target}>{competitor}"] = margin
    return penalties


def split_action_margin_key(key: str) -> tuple[str, str]:
    if ">" not in key:
        raise ValueError(f"Action margin penalty key must use target>competitor format: {key}")
    target, competitor = (piece.strip() for piece in key.split(">", 1))
    if not target or not competitor:
        raise ValueError(f"Action margin penalty key has empty target or competitor: {key}")
    return target, competitor


def maybe_phase_for_failure_mode(failure_mode: str) -> str | None:
    for phase, modes in BOUNDARY_CURRICULUM_PHASES.items():
        if failure_mode in modes:
            return phase
    return None


def phase_for_failure_mode(failure_mode: str) -> str:
    phase = maybe_phase_for_failure_mode(failure_mode)
    if phase is not None:
        return phase
    raise ValueError(f"Unsupported boundary curriculum failure mode: {failure_mode}")


def unique_candidate_sets(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    out: list[Mapping[str, Any]] = []
    for pair in pairs:
        expected = final_candidate(pair["chosen_messages"])
        key = (str(pair["task_id"]), candidate_key(expected))
        if key in seen:
            continue
        seen.add(key)
        out.append(pair)
    return out


def shuffle_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    seed: int | None,
) -> list[Mapping[str, Any]]:
    out = list(pairs)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(out)
    return out


def limit_by_expected_action(
    pairs: Sequence[Mapping[str, Any]],
    *,
    limit_per_action: int | None,
) -> list[Mapping[str, Any]]:
    if limit_per_action is None or limit_per_action <= 0:
        return list(pairs)
    counts: Counter[str] = Counter()
    out: list[Mapping[str, Any]] = []
    for pair in pairs:
        action = str(pair.get("evidence_derived_action"))
        if counts[action] >= limit_per_action:
            continue
        counts[action] += 1
        out.append(pair)
    return out


def dedupe_candidates(candidates: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    out: list[dict[str, str]] = []
    for candidate in candidates:
        key = candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(candidate))
    return out


def expected_index(candidates: Sequence[Mapping[str, str]], expected: Mapping[str, str]) -> int:
    for index, candidate in enumerate(candidates):
        if candidates_match(candidate, expected):
            return index
    raise ValueError(f"Expected candidate is missing from candidate list: {expected}")


def encode_candidate_set(
    tokenizer: Any,
    pair: Mapping[str, Any],
    *,
    max_length: int,
) -> dict[str, Any]:
    prompt = prompt_from_pair(pair)
    expected = final_candidate(pair["chosen_messages"])
    candidates, expected_in_candidates = candidates_for_pair(pair)
    candidates = dedupe_candidates(candidates)
    expected_idx = expected_index(candidates, expected)
    encoded_candidates = [
        encode_candidate_target(tokenizer, prompt, candidate, max_length=max_length)
        for candidate in candidates
    ]
    lengths = [int(item["input_ids"].shape[0]) for item in encoded_candidates]
    return {
        "id": pair["id"],
        "task_id": pair["task_id"],
        "failure_mode": pair["failure_mode"],
        "expected_action": pair["evidence_derived_action"],
        "expected": expected,
        "rejected_action": pair["rejected_action"],
        "candidates": candidates,
        "encoded_candidates": encoded_candidates,
        "expected_index": expected_idx,
        "expected_in_candidates": expected_in_candidates,
        "candidate_count": len(candidates),
        "candidate_lengths": lengths,
    }


def candidate_ce_loss(logps: torch.Tensor, expected_idx: int, *, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    expected = torch.tensor([expected_idx], dtype=torch.long, device=logps.device)
    return F.cross_entropy((logps / temperature).unsqueeze(0), expected)


def action_scores_by_action(
    logps: torch.Tensor,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, torch.Tensor]]:
    action_order: list[str] = []
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        action = str(candidate.get("action"))
        if action not in grouped:
            action_order.append(action)
        grouped[action].append(logps[idx])
    return action_order, {
        action: torch.stack(grouped[action]).max()
        for action in action_order
    }


def action_ce_loss(
    logps: torch.Tensor,
    candidates: Sequence[Mapping[str, Any]],
    expected_action: str,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    action_order, action_scores_by_name = action_scores_by_action(logps, candidates)
    if expected_action not in action_scores_by_name:
        raise ValueError(f"Expected action is not present in candidates: {expected_action}")
    action_scores = torch.stack([
        action_scores_by_name[action]
        for action in action_order
    ])
    expected_idx = action_order.index(expected_action)
    expected = torch.tensor([expected_idx], dtype=torch.long, device=logps.device)
    return F.cross_entropy((action_scores / temperature).unsqueeze(0), expected)


def action_margin_penalty_loss(
    logps: torch.Tensor,
    candidates: Sequence[Mapping[str, Any]],
    expected_action: str,
    penalties: Mapping[str, float],
) -> torch.Tensor:
    _, action_scores = action_scores_by_action(logps, candidates)
    margin_losses = []
    for key, margin in sorted(penalties.items()):
        target, competitor = split_action_margin_key(key)
        if expected_action != target:
            continue
        if target not in action_scores or competitor not in action_scores:
            continue
        desired = torch.tensor(float(margin), dtype=logps.dtype, device=logps.device)
        margin_losses.append(F.relu(desired - (action_scores[target] - action_scores[competitor])))
    if not margin_losses:
        return torch.zeros((), dtype=logps.dtype, device=logps.device)
    return torch.stack(margin_losses).mean()


def action_loss_weight(action: str, weights: Mapping[str, float]) -> float:
    return float(weights.get(action, 1.0))


def rank_from_logps(logps: torch.Tensor, expected_idx: int) -> tuple[int, float]:
    expected_logp = float(logps[expected_idx].detach().cpu())
    winner_logp = float(logps.max().detach().cpu())
    rank = int((logps > logps[expected_idx]).sum().detach().cpu()) + 1
    return rank, winner_logp - expected_logp


def summarize_group(rows: list[Mapping[str, Any]], *, key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    summary: dict[str, Any] = {}
    for name, group_rows in sorted(groups.items()):
        ranks = [int(row["expected_rank"]) for row in group_rows]
        margins = [float(row["expected_margin_from_winner"]) for row in group_rows]
        pred_actions = Counter(str(row["pred"]["action"]) for row in group_rows)
        summary[name] = {
            "n": len(group_rows),
            "action_accuracy": round(sum(bool(row["action_correct"]) for row in group_rows) / len(group_rows), 3),
            "exact_candidate_accuracy": round(
                sum(bool(row["exact_candidate_correct"]) for row in group_rows) / len(group_rows), 3
            ),
            "expected_rank_counts": dict(sorted(Counter(ranks).items())),
            "expected_margin_from_winner": margin_stats(margins),
            "pred_actions": dict(sorted(pred_actions.items())),
        }
    return summary


def evaluate_candidate_sets(
    model: Any,
    encoded_sets: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    device: str,
    logprob_mode: str,
    candidate_batch_size: int,
) -> dict[str, Any]:
    model.eval()
    rows = []
    with torch.no_grad():
        for item in encoded_sets:
            logps = candidate_logps(
                model,
                list(item["encoded_candidates"]),
                pad_token_id=pad_token_id,
                device=device,
                logprob_mode=logprob_mode,
                candidate_batch_size=candidate_batch_size,
            )
            winner_idx = int(torch.argmax(logps).detach().cpu())
            rank, margin = rank_from_logps(logps, int(item["expected_index"]))
            pred = dict(item["candidates"][winner_idx])
            expected = dict(item["expected"])
            rows.append({
                "id": item["id"],
                "task_id": item["task_id"],
                "failure_mode": item["failure_mode"],
                "expected_action": item["expected_action"],
                "expected": expected,
                "pred": pred,
                "action_correct": candidate_actions_match(pred, expected),
                "exact_candidate_correct": candidates_match(pred, expected),
                "expected_rank": rank,
                "expected_margin_from_winner": margin,
                "candidate_count": item["candidate_count"],
                "winner_logp": float(logps[winner_idx].detach().cpu()),
                "expected_logp": float(logps[int(item["expected_index"])].detach().cpu()),
            })

    action_correct = sum(bool(row["action_correct"]) for row in rows)
    exact_correct = sum(bool(row["exact_candidate_correct"]) for row in rows)
    ranks = [int(row["expected_rank"]) for row in rows]
    margins = [float(row["expected_margin_from_winner"]) for row in rows]
    candidate_counts = [int(row["candidate_count"]) for row in rows]
    return {
        "n": len(rows),
        "action_accuracy": round(action_correct / len(rows), 3) if rows else None,
        "exact_candidate_accuracy": round(exact_correct / len(rows), 3) if rows else None,
        "expected_rank_counts": dict(sorted(Counter(ranks).items())),
        "expected_margin_from_winner": margin_stats(margins),
        "candidate_count": {
            "mean": round(sum(candidate_counts) / len(candidate_counts), 4) if candidate_counts else None,
            "median": round(statistics.median(candidate_counts), 4) if candidate_counts else None,
            "min": min(candidate_counts) if candidate_counts else None,
            "max": max(candidate_counts) if candidate_counts else None,
        },
        "by_failure_mode": summarize_group(rows, key="failure_mode"),
        "by_expected_action": summarize_group(rows, key="expected_action"),
        "rows": rows,
    }


def compact_eval_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "rows"}


def compact_optional_eval_summary(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return compact_eval_summary(summary) if summary is not None else None


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def min_expected_action_accuracy(summary: Mapping[str, Any]) -> float:
    by_action = summary.get("by_expected_action")
    if not isinstance(by_action, Mapping) or not by_action:
        return -1.0
    accuracies = []
    for row in by_action.values():
        if not isinstance(row, Mapping):
            continue
        value = row.get("action_accuracy")
        if value is not None:
            accuracies.append(float(value))
    return min(accuracies) if accuracies else -1.0


def expected_action_accuracy(summary: Mapping[str, Any], action: str) -> float:
    by_action = summary.get("by_expected_action")
    if not isinstance(by_action, Mapping):
        return -1.0
    row = by_action.get(action)
    if not isinstance(row, Mapping):
        return -1.0
    value = row.get("action_accuracy")
    return float(value) if value is not None else -1.0


def metric_or_default(summary: Mapping[str, Any], key: str, default: float = -1.0) -> float:
    value = summary.get(key)
    return float(value) if value is not None else default


def mean_margin_score(summary: Mapping[str, Any]) -> float:
    margins = summary.get("expected_margin_from_winner")
    if not isinstance(margins, Mapping) or margins.get("mean") is None:
        return -1_000_000_000.0
    return -float(margins["mean"])


def action_floor_components(
    summary: Mapping[str, Any],
    action_floors: Mapping[str, float],
) -> dict[str, Any]:
    accuracies = {
        action: expected_action_accuracy(summary, action)
        for action in sorted(action_floors)
    }
    deficits = {
        action: max(0.0, floor - max(0.0, accuracies[action]))
        for action, floor in sorted(action_floors.items())
    }
    floor_satisfied = bool(action_floors) and all(deficit == 0.0 for deficit in deficits.values())
    return {
        "action_floors": dict(sorted(action_floors.items())),
        "action_accuracies": accuracies,
        "deficits": deficits,
        "total_deficit": round(sum(deficits.values()), 6),
        "min_required_action_accuracy": min(accuracies.values()) if accuracies else -1.0,
        "floor_satisfied": floor_satisfied,
    }


def checkpoint_selection_score(
    summary: Mapping[str, Any],
    mode: str,
    *,
    action_floors: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    action_accuracy = metric_or_default(summary, "action_accuracy")
    exact_accuracy = metric_or_default(summary, "exact_candidate_accuracy")
    min_action_accuracy = min_expected_action_accuracy(summary)
    margin_score = mean_margin_score(summary)
    if mode == "action_accuracy":
        return (action_accuracy, min_action_accuracy, exact_accuracy, margin_score)
    if mode == "min_action_accuracy":
        return (min_action_accuracy, action_accuracy, exact_accuracy, margin_score)
    if mode == "action_floor":
        floors = dict(action_floors or {})
        if not floors:
            raise ValueError("action_floor checkpoint selection requires --checkpoint-action-floors.")
        floor = action_floor_components(summary, floors)
        return (
            1.0 if floor["floor_satisfied"] else 0.0,
            -float(floor["total_deficit"]),
            float(floor["min_required_action_accuracy"]),
            min_action_accuracy,
            action_accuracy,
            exact_accuracy,
            margin_score,
        )
    raise ValueError(f"Unsupported checkpoint selection mode: {mode}")


def snapshot_trainable_state(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def restore_trainable_state(model: Any, state: Mapping[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    missing = sorted(set(state) - set(params))
    if missing:
        raise ValueError(f"Selected checkpoint has unknown trainable tensor(s): {missing}")
    for name, tensor in state.items():
        target = params[name]
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))


def candidate_logps(
    model: Any,
    encoded_candidates: Sequence[Mapping[str, torch.Tensor]],
    *,
    pad_token_id: int,
    device: str,
    logprob_mode: str,
    candidate_batch_size: int,
) -> torch.Tensor:
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive.")
    chunks = []
    for start in range(0, len(encoded_candidates), candidate_batch_size):
        batch = collate(list(encoded_candidates[start:start + candidate_batch_size]), pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        chunks.append(sequence_logps(model, batch, logprob_mode=logprob_mode))
    return torch.cat(chunks, dim=0)


def set_batch(
    encoded_sets: Sequence[Mapping[str, Any]],
    cursor: int,
    batch_size: int,
) -> list[Mapping[str, Any]]:
    return [encoded_sets[(cursor + offset) % len(encoded_sets)] for offset in range(batch_size)]


def group_by_curriculum_phase(
    encoded_sets: Sequence[Mapping[str, Any]],
    phase_order: Sequence[str],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {phase: [] for phase in phase_order}
    for item in encoded_sets:
        phase = phase_for_failure_mode(str(item["failure_mode"]))
        if phase in grouped:
            grouped[phase].append(item)
    missing = [phase for phase, rows in grouped.items() if not rows]
    if missing:
        raise ValueError(f"No training examples for curriculum phase(s): {', '.join(missing)}")
    return grouped


def curriculum_step_batch(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    phase: str,
    cursors: Counter[str],
    batch_size: int,
) -> list[Mapping[str, Any]]:
    rows = grouped[phase]
    cursor = cursors[phase]
    cursors[phase] += batch_size
    return [rows[(cursor + offset) % len(rows)] for offset in range(batch_size)]


def curriculum_phase_batch(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    phase_order: Sequence[str],
    cursors: Counter[str],
    batch_size: int,
) -> list[Mapping[str, Any]]:
    batch = []
    for phase in phase_order:
        batch.extend(curriculum_step_batch(grouped, phase, cursors, batch_size))
    return batch


def count_by_curriculum_phase(encoded_sets: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(
        maybe_phase_for_failure_mode(str(row["failure_mode"])) or "other"
        for row in encoded_sets
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--preferences",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl",
    )
    parser.add_argument(
        "--eval-preferences",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl",
    )
    parser.add_argument("--out-dir", default="post_training/runs/qwen_boundary_preference_candidate_ce_smoke")
    parser.add_argument("--failure-modes", default=None)
    parser.add_argument("--eval-failure-modes", default=None)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=16)
    parser.add_argument("--limit-per-action", type=int, default=None)
    parser.add_argument("--eval-limit-per-action", type=int, default=None)
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=None,
        help="Shuffle candidate sets with this seed before applying per-action/global limits.",
    )
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument(
        "--training-schedule",
        choices=("source_order", "boundary_round_robin", "boundary_phase_batch"),
        default="source_order",
    )
    parser.add_argument("--phase-order", default="defer,flag,reject")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--action-loss-weights",
        default=None,
        help="Optional comma-separated action weights for CE loss, e.g. flag=2.0,reject=1.5.",
    )
    parser.add_argument(
        "--action-margin-penalties",
        default=None,
        help="Optional comma-separated asymmetric action margins, e.g. flag>reject=0.25.",
    )
    parser.add_argument(
        "--action-margin-weight",
        type=float,
        default=1.0,
        help="Multiplier for action margin penalties when --action-margin-penalties is set.",
    )
    parser.add_argument(
        "--loss-target",
        choices=("candidate", "action"),
        default="candidate",
        help="Train against the exact candidate or the expected action aggregated across candidates.",
    )
    parser.add_argument("--logprob-mode", choices=("mean", "sum"), default="mean")
    parser.add_argument("--eval-checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("action_accuracy", "min_action_accuracy", "action_floor"),
        default="min_action_accuracy",
    )
    parser.add_argument(
        "--checkpoint-action-floors",
        default=None,
        help="Optional comma-separated action floors for action_floor selection, e.g. flag=0.25,reject=0.25.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-trainable-state", action="store_true")
    parser.add_argument(
        "--skip-train-eval",
        action="store_true",
        help="Skip pre/post evaluation on the training candidate sets for faster sweep probes.",
    )
    parser.add_argument(
        "--skip-pre-eval",
        action="store_true",
        help="Skip the initial held-out evaluation; checkpoint and final held-out eval still run.",
    )
    args = parser.parse_args()

    if args.eval_checkpoint_every < 0:
        raise ValueError("--eval-checkpoint-every must be non-negative.")
    if args.action_margin_weight < 0:
        raise ValueError("--action-margin-weight must be non-negative.")
    action_loss_weights = parse_action_loss_weights(args.action_loss_weights)
    action_margin_penalties = parse_action_margin_penalties(args.action_margin_penalties)
    checkpoint_action_floors = parse_action_floors(args.checkpoint_action_floors)
    if args.checkpoint_selection == "action_floor" and not checkpoint_action_floors:
        raise ValueError("--checkpoint-selection action_floor requires --checkpoint-action-floors.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_modes = parse_failure_modes(args.failure_modes)
    train_pairs = filter_pairs(load_jsonl(args.preferences), failure_modes=failure_modes)
    source_train_pairs = len(train_pairs)
    if not args.no_dedupe:
        train_pairs = unique_candidate_sets(train_pairs)
    train_pairs = shuffle_pairs(train_pairs, seed=args.selection_seed)
    train_pairs = limit_by_expected_action(train_pairs, limit_per_action=args.limit_per_action)
    train_pairs = apply_limit(train_pairs, args.limit)
    if not train_pairs:
        raise ValueError("No train preference prompts selected.")

    eval_failure_modes = parse_failure_modes(args.eval_failure_modes)
    eval_pairs = filter_pairs(load_jsonl(args.eval_preferences), failure_modes=eval_failure_modes)
    source_eval_pairs = len(eval_pairs)
    if not args.no_dedupe:
        eval_pairs = unique_candidate_sets(eval_pairs)
    eval_pairs = shuffle_pairs(eval_pairs, seed=args.selection_seed)
    eval_pairs = limit_by_expected_action(eval_pairs, limit_per_action=args.eval_limit_per_action)
    eval_pairs = apply_limit(eval_pairs, args.eval_limit)
    if not eval_pairs:
        raise ValueError("No eval preference prompts selected.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_train = [encode_candidate_set(tokenizer, pair, max_length=args.max_length) for pair in train_pairs]
    encoded_eval = [encode_candidate_set(tokenizer, pair, max_length=args.max_length) for pair in eval_pairs]
    phase_order = parse_phase_order(args.phase_order)
    train_by_curriculum_phase = count_by_curriculum_phase(encoded_train)
    eval_by_curriculum_phase = count_by_curriculum_phase(encoded_eval)
    grouped_train = None
    if args.training_schedule in {"boundary_round_robin", "boundary_phase_batch"}:
        grouped_train = group_by_curriculum_phase(encoded_train, phase_order)

    if args.dry_run:
        report = {
            "model": args.model,
            "preferences": args.preferences,
            "eval_preferences": args.eval_preferences,
            "source_train_pairs": source_train_pairs,
            "selected_train_sets": len(encoded_train),
            "source_eval_pairs": source_eval_pairs,
            "selected_eval_sets": len(encoded_eval),
            "dedupe": not args.no_dedupe,
            "selection_seed": args.selection_seed,
            "limit_per_action": args.limit_per_action,
            "eval_limit_per_action": args.eval_limit_per_action,
            "training_schedule": args.training_schedule,
            "phase_order": list(phase_order),
            "eval_checkpoint_every": args.eval_checkpoint_every,
            "checkpoint_selection": args.checkpoint_selection,
            "checkpoint_action_floors": checkpoint_action_floors,
            "action_loss_weights": action_loss_weights,
            "action_margin_penalties": action_margin_penalties,
            "action_margin_weight": args.action_margin_weight,
            "loss_target": args.loss_target,
            "skip_train_eval": args.skip_train_eval,
            "skip_pre_eval": args.skip_pre_eval,
            "max_length": args.max_length,
            "train_by_expected_action": dict(sorted(Counter(row["expected_action"] for row in encoded_train).items())),
            "eval_by_expected_action": dict(sorted(Counter(row["expected_action"] for row in encoded_eval).items())),
            "train_by_curriculum_phase": dict(sorted(train_by_curriculum_phase.items())),
            "eval_by_curriculum_phase": dict(sorted(eval_by_curriculum_phase.items())),
            "train_set_ids": [str(item["id"]) for item in encoded_train],
            "eval_set_ids": [str(item["id"]) for item in encoded_eval],
            "train_candidate_counts": [item["candidate_count"] for item in encoded_train],
            "eval_candidate_counts": [item["candidate_count"] for item in encoded_eval],
            "train_missing_expected_candidates": sum(not item["expected_in_candidates"] for item in encoded_train),
            "eval_missing_expected_candidates": sum(not item["expected_in_candidates"] for item in encoded_eval),
            "train_max_lengths": [max(item["candidate_lengths"]) for item in encoded_train],
            "eval_max_lengths": [max(item["candidate_lengths"]) for item in encoded_eval],
            "dry_run": True,
        }
        write_report(out_dir / "report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    device = choose_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        torch_dtype="auto",
    )
    model.to(device)
    trainable_params = set_trainable_last_layers(model, args.train_last_layers)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
    )

    pre_train = None
    if not args.skip_train_eval:
        pre_train = evaluate_candidate_sets(
            model,
            encoded_train,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            logprob_mode=args.logprob_mode,
            candidate_batch_size=args.candidate_batch_size,
        )
    pre_eval = None
    if not args.skip_pre_eval:
        pre_eval = evaluate_candidate_sets(
            model,
            encoded_eval,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            logprob_mode=args.logprob_mode,
            candidate_batch_size=args.candidate_batch_size,
        )

    losses = []
    loss_trace = []
    eval_checkpoints = []
    selected_checkpoint = None
    selected_checkpoint_score = None
    selected_trainable_state = None
    cursor = 0
    phase_cursors: Counter[str] = Counter()
    model.train()
    for step in range(args.max_steps):
        phase = "source_order"
        if args.training_schedule == "boundary_round_robin":
            assert grouped_train is not None
            phase = phase_order[step % len(phase_order)]
            batch_sets = curriculum_step_batch(grouped_train, phase, phase_cursors, args.batch_size)
        elif args.training_schedule == "boundary_phase_batch":
            assert grouped_train is not None
            phase = "phase_batch:" + "+".join(phase_order)
            batch_sets = curriculum_phase_batch(grouped_train, phase_order, phase_cursors, args.batch_size)
        else:
            batch_sets = set_batch(encoded_train, cursor, args.batch_size)
            cursor += args.batch_size
        batch_losses = []
        batch_weights = []
        batch_margin_penalties = []
        optimizer.zero_grad(set_to_none=True)
        for item in batch_sets:
            logps = candidate_logps(
                model,
                list(item["encoded_candidates"]),
                pad_token_id=tokenizer.pad_token_id,
                device=device,
                logprob_mode=args.logprob_mode,
                candidate_batch_size=args.candidate_batch_size,
            )
            if args.loss_target == "candidate":
                batch_losses.append(candidate_ce_loss(logps, int(item["expected_index"]), temperature=args.temperature))
            else:
                batch_losses.append(
                    action_ce_loss(
                        logps,
                        list(item["candidates"]),
                        str(item["expected_action"]),
                        temperature=args.temperature,
                    )
                )
            batch_weights.append(action_loss_weight(str(item["expected_action"]), action_loss_weights))
            if action_margin_penalties:
                batch_margin_penalties.append(
                    action_margin_penalty_loss(
                        logps,
                        list(item["candidates"]),
                        str(item["expected_action"]),
                        action_margin_penalties,
                    )
                )
        loss_tensor = torch.stack(batch_losses)
        weights_tensor = torch.tensor(batch_weights, dtype=loss_tensor.dtype, device=loss_tensor.device)
        ce_loss = (loss_tensor * weights_tensor).sum() / weights_tensor.sum()
        if batch_margin_penalties and args.action_margin_weight > 0:
            margin_penalty = torch.stack(batch_margin_penalties).mean()
        else:
            margin_penalty = torch.zeros((), dtype=loss_tensor.dtype, device=loss_tensor.device)
        loss = ce_loss + (args.action_margin_weight * margin_penalty)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ce_loss_value = float(ce_loss.detach().cpu())
        margin_penalty_value = float(margin_penalty.detach().cpu())
        batch_phase_counts = dict(sorted(count_by_curriculum_phase(batch_sets).items()))
        batch_action_loss_weights = {
            action: action_loss_weight(action, action_loss_weights)
            for action in sorted({str(item["expected_action"]) for item in batch_sets})
        }
        loss_trace.append({
            "step": step + 1,
            "phase": phase,
            "batch_phase_counts": batch_phase_counts,
            "batch_action_loss_weights": batch_action_loss_weights,
            "action_margin_penalties": action_margin_penalties,
            "action_margin_weight": args.action_margin_weight,
            "ce_loss": ce_loss_value,
            "action_margin_penalty": margin_penalty_value,
            "loss": losses[-1],
        })
        print(f"step={step + 1} phase={phase} loss={losses[-1]:.4f}", flush=True)
        if args.eval_checkpoint_every > 0 and (
            (step + 1) % args.eval_checkpoint_every == 0 or step + 1 == args.max_steps
        ):
            checkpoint_eval = evaluate_candidate_sets(
                model,
                encoded_eval,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
                logprob_mode=args.logprob_mode,
                candidate_batch_size=args.candidate_batch_size,
            )
            checkpoint_summary = compact_eval_summary(checkpoint_eval)
            score = checkpoint_selection_score(
                checkpoint_summary,
                args.checkpoint_selection,
                action_floors=checkpoint_action_floors,
            )
            floor_summary = (
                action_floor_components(checkpoint_summary, checkpoint_action_floors)
                if checkpoint_action_floors else None
            )
            checkpoint = {
                "step": step + 1,
                "phase": phase,
                "selection": args.checkpoint_selection,
                "selection_action_floor": floor_summary,
                "selection_score": list(score),
                "eval": checkpoint_summary,
            }
            eval_checkpoints.append(checkpoint)
            if selected_checkpoint_score is None or score > selected_checkpoint_score:
                selected_checkpoint_score = score
                selected_checkpoint = {
                    key: value for key, value in checkpoint.items() if key != "eval"
                } | {"eval": checkpoint_summary}
                selected_trainable_state = snapshot_trainable_state(model)
            write_report(
                out_dir / "checkpoint_report.json",
                {
                    "condition": "boundary_preference_all_candidate_ce_smoke",
                    "partial": True,
                    "model": args.model,
                    "preferences": args.preferences,
                    "eval_preferences": args.eval_preferences,
                    "selection_seed": args.selection_seed,
                    "selected_train_sets": len(encoded_train),
                    "selected_eval_sets": len(encoded_eval),
                    "training_schedule": args.training_schedule,
                    "phase_order": list(phase_order),
                    "action_loss_weights": action_loss_weights,
                    "action_margin_penalties": action_margin_penalties,
                    "action_margin_weight": args.action_margin_weight,
                    "loss_target": args.loss_target,
                    "skip_train_eval": args.skip_train_eval,
                    "skip_pre_eval": args.skip_pre_eval,
                    "eval_checkpoint_every": args.eval_checkpoint_every,
                    "checkpoint_selection": args.checkpoint_selection,
                    "checkpoint_action_floors": checkpoint_action_floors,
                    "losses": losses,
                    "loss_trace": loss_trace,
                    "eval_checkpoints": eval_checkpoints,
                    "selected_checkpoint": selected_checkpoint,
                },
            )
            model.train()

    post_evaluation_state = "final"
    if selected_trainable_state is not None:
        restore_trainable_state(model, selected_trainable_state)
        post_evaluation_state = "selected_checkpoint"

    post_train = None
    if not args.skip_train_eval:
        post_train = evaluate_candidate_sets(
            model,
            encoded_train,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            logprob_mode=args.logprob_mode,
            candidate_batch_size=args.candidate_batch_size,
        )
    post_eval = evaluate_candidate_sets(
        model,
        encoded_eval,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        logprob_mode=args.logprob_mode,
        candidate_batch_size=args.candidate_batch_size,
    )

    state_path = None
    if not args.no_save_trainable_state:
        state_path = save_trainable_state(model, out_dir)

    report = {
        "model": args.model,
        "condition": "boundary_preference_all_candidate_ce_smoke",
        "preferences": args.preferences,
        "eval_preferences": args.eval_preferences,
        "source_train_pairs": source_train_pairs,
        "selected_train_sets": len(encoded_train),
        "source_eval_pairs": source_eval_pairs,
        "selected_eval_sets": len(encoded_eval),
        "dedupe": not args.no_dedupe,
        "selection_seed": args.selection_seed,
        "limit_per_action": args.limit_per_action,
        "eval_limit_per_action": args.eval_limit_per_action,
        "training_schedule": args.training_schedule,
        "phase_order": list(phase_order),
        "train_by_expected_action": dict(sorted(Counter(row["expected_action"] for row in encoded_train).items())),
        "eval_by_expected_action": dict(sorted(Counter(row["expected_action"] for row in encoded_eval).items())),
        "train_by_curriculum_phase": dict(sorted(train_by_curriculum_phase.items())),
        "eval_by_curriculum_phase": dict(sorted(eval_by_curriculum_phase.items())),
        "train_set_ids": [str(item["id"]) for item in encoded_train],
        "eval_set_ids": [str(item["id"]) for item in encoded_eval],
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "candidate_batch_size": args.candidate_batch_size,
        "max_length": args.max_length,
        "device": device,
        "train_last_layers": args.train_last_layers,
        "trainable_params": trainable_params,
        "lr": args.lr,
        "temperature": args.temperature,
        "logprob_mode": args.logprob_mode,
        "eval_checkpoint_every": args.eval_checkpoint_every,
        "checkpoint_selection": args.checkpoint_selection,
        "checkpoint_action_floors": checkpoint_action_floors,
        "action_loss_weights": action_loss_weights,
        "action_margin_penalties": action_margin_penalties,
        "action_margin_weight": args.action_margin_weight,
        "loss_target": args.loss_target,
        "skip_train_eval": args.skip_train_eval,
        "skip_pre_eval": args.skip_pre_eval,
        "eval_checkpoints": eval_checkpoints,
        "selected_checkpoint": selected_checkpoint,
        "post_evaluation_state": post_evaluation_state,
        "losses": losses,
        "loss_trace": loss_trace,
        "loss_delta": losses[-1] - losses[0] if len(losses) > 1 else 0.0,
        "pre_train": compact_optional_eval_summary(pre_train),
        "post_train": compact_optional_eval_summary(post_train),
        "pre_eval": compact_optional_eval_summary(pre_eval),
        "post_eval": compact_eval_summary(post_eval),
        "trainable_state": str(state_path) if state_path else None,
    }
    write_report(out_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
