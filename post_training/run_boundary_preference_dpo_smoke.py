#!/usr/bin/env python3
"""Reference-free DPO-style smoke loop for boundary preference pairs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_boundary_preference_margin import final_candidate, prompt_from_pair  # noqa: E402
from post_training.run_sft_smoke import (  # noqa: E402
    choose_device,
    collate,
    load_jsonl,
    save_trainable_state,
    set_trainable_last_layers,
)


def parse_failure_modes(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modes:
        raise ValueError("Failure-mode filter cannot be empty.")
    return modes


def filter_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    failure_modes: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    if failure_modes is None:
        return list(pairs)
    mode_set = set(failure_modes)
    return [pair for pair in pairs if str(pair.get("failure_mode")) in mode_set]


def apply_limit(pairs: Sequence[Mapping[str, Any]], limit: int | None) -> list[Mapping[str, Any]]:
    if limit is None or limit <= 0:
        return list(pairs)
    return list(pairs[:limit])


def target_text(candidate: Mapping[str, str], eos_token: str) -> str:
    return json.dumps(dict(candidate), sort_keys=True) + eos_token


def encode_candidate_target(
    tokenizer: Any,
    prompt: str,
    candidate: Mapping[str, str],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text(candidate, tokenizer.eos_token), add_special_tokens=False)["input_ids"]
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


def encode_pair(
    tokenizer: Any,
    pair: Mapping[str, Any],
    *,
    max_length: int,
) -> dict[str, Any]:
    prompt = prompt_from_pair(pair)
    chosen = final_candidate(pair["chosen_messages"])
    rejected = final_candidate(pair["rejected_messages"])
    chosen_encoded = encode_candidate_target(tokenizer, prompt, chosen, max_length=max_length)
    rejected_encoded = encode_candidate_target(tokenizer, prompt, rejected, max_length=max_length)
    return {
        "id": pair["id"],
        "task_id": pair["task_id"],
        "failure_mode": pair["failure_mode"],
        "chosen_action": pair["evidence_derived_action"],
        "rejected_action": pair["rejected_action"],
        "chosen": chosen_encoded,
        "rejected": rejected_encoded,
        "chosen_length": int(chosen_encoded["input_ids"].shape[0]),
        "rejected_length": int(rejected_encoded["input_ids"].shape[0]),
    }


def sequence_logps(model: Any, batch: Mapping[str, torch.Tensor], *, logprob_mode: str) -> torch.Tensor:
    if logprob_mode not in {"mean", "sum"}:
        raise ValueError(f"Unsupported logprob_mode: {logprob_mode}")
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    labels = batch["labels"]
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    sum_logps = (token_logps * mask).sum(dim=-1)
    if logprob_mode == "sum":
        return sum_logps
    target_tokens = mask.sum(dim=-1).clamp(min=1)
    return sum_logps / target_tokens


def dpo_loss_from_logps(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    return -F.logsigmoid(beta * (chosen_logps - rejected_logps)).mean()


def margin_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": round(sum(values) / len(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def summarize_margins(margins: Sequence[float]) -> dict[str, Any]:
    wins = sum(margin > 0 for margin in margins)
    return {
        "n": len(margins),
        "win_rate": round(wins / len(margins), 3) if margins else None,
        "margin": margin_stats(margins),
    }


def evaluate_margins(
    model: Any,
    encoded_pairs: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    batch_size: int,
    device: str,
    logprob_mode: str,
) -> dict[str, Any]:
    model.eval()
    margins: list[float] = []
    margins_by_mode: dict[str, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(encoded_pairs), batch_size):
            batch_pairs = encoded_pairs[start:start + batch_size]
            chosen_batch = collate([pair["chosen"] for pair in batch_pairs], pad_token_id)
            rejected_batch = collate([pair["rejected"] for pair in batch_pairs], pad_token_id)
            chosen_batch = {key: value.to(device) for key, value in chosen_batch.items()}
            rejected_batch = {key: value.to(device) for key, value in rejected_batch.items()}
            chosen_logps = sequence_logps(model, chosen_batch, logprob_mode=logprob_mode)
            rejected_logps = sequence_logps(model, rejected_batch, logprob_mode=logprob_mode)
            batch_margins = [float(value.detach().cpu()) for value in chosen_logps - rejected_logps]
            margins.extend(batch_margins)
            for pair, margin in zip(batch_pairs, batch_margins):
                margins_by_mode.setdefault(str(pair["failure_mode"]), []).append(margin)
    summary = summarize_margins(margins)
    summary["by_failure_mode"] = {
        mode: summarize_margins(mode_margins)
        for mode, mode_margins in sorted(margins_by_mode.items())
    }
    return summary


def pair_batch(encoded_pairs: Sequence[Mapping[str, Any]], cursor: int, batch_size: int) -> list[Mapping[str, Any]]:
    return [encoded_pairs[(cursor + offset) % len(encoded_pairs)] for offset in range(batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--preferences",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_v1.jsonl",
    )
    parser.add_argument("--out-dir", default="post_training/runs/qwen_boundary_preference_dpo_hard_smoke")
    parser.add_argument("--failure-modes", default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--train-eval-limit", type=int, default=None)
    parser.add_argument("--eval-preferences", default=None)
    parser.add_argument("--eval-failure-modes", default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logprob-mode", choices=("mean", "sum"), default="mean")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-trainable-state", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_modes = parse_failure_modes(args.failure_modes)
    pairs = apply_limit(
        filter_pairs(load_jsonl(args.preferences), failure_modes=failure_modes),
        args.limit,
    )
    if not pairs:
        raise ValueError("No preference pairs selected.")
    eval_pairs = []
    if args.eval_preferences:
        eval_failure_modes = parse_failure_modes(args.eval_failure_modes)
        eval_pairs = apply_limit(
            filter_pairs(load_jsonl(args.eval_preferences), failure_modes=eval_failure_modes),
            args.eval_limit,
        )
        if not eval_pairs:
            raise ValueError("No eval preference pairs selected.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded_pairs = [encode_pair(tokenizer, pair, max_length=args.max_length) for pair in pairs]
    encoded_eval_pairs = [encode_pair(tokenizer, pair, max_length=args.max_length) for pair in eval_pairs]

    if args.dry_run:
        report = {
            "model": args.model,
            "preferences": args.preferences,
            "selected_pairs": len(encoded_pairs),
            "failure_modes": sorted({str(pair["failure_mode"]) for pair in encoded_pairs}),
            "max_length": args.max_length,
            "chosen_lengths": [int(pair["chosen_length"]) for pair in encoded_pairs],
            "rejected_lengths": [int(pair["rejected_length"]) for pair in encoded_pairs],
            "eval_preferences": args.eval_preferences,
            "selected_eval_pairs": len(encoded_eval_pairs),
            "eval_failure_modes": sorted({str(pair["failure_mode"]) for pair in encoded_eval_pairs}),
            "eval_chosen_lengths": [int(pair["chosen_length"]) for pair in encoded_eval_pairs],
            "eval_rejected_lengths": [int(pair["rejected_length"]) for pair in encoded_eval_pairs],
            "dry_run": True,
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
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

    train_eval_pairs = apply_limit(encoded_pairs, args.train_eval_limit)

    pre_train = evaluate_margins(
        model,
        train_eval_pairs,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
        device=device,
        logprob_mode=args.logprob_mode,
    )
    pre_eval = None
    if encoded_eval_pairs:
        pre_eval = evaluate_margins(
            model,
            encoded_eval_pairs,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=args.batch_size,
            device=device,
            logprob_mode=args.logprob_mode,
        )

    losses = []
    margins = []
    cursor = 0
    model.train()
    for step in range(args.max_steps):
        batch_pairs = pair_batch(encoded_pairs, cursor, args.batch_size)
        cursor += args.batch_size
        chosen_batch = collate([pair["chosen"] for pair in batch_pairs], tokenizer.pad_token_id)
        rejected_batch = collate([pair["rejected"] for pair in batch_pairs], tokenizer.pad_token_id)
        chosen_batch = {key: value.to(device) for key, value in chosen_batch.items()}
        rejected_batch = {key: value.to(device) for key, value in rejected_batch.items()}

        optimizer.zero_grad(set_to_none=True)
        chosen_logps = sequence_logps(model, chosen_batch, logprob_mode=args.logprob_mode)
        rejected_logps = sequence_logps(model, rejected_batch, logprob_mode=args.logprob_mode)
        loss = dpo_loss_from_logps(chosen_logps, rejected_logps, beta=args.beta)
        loss.backward()
        optimizer.step()

        step_margin = float((chosen_logps - rejected_logps).mean().detach().cpu())
        losses.append(float(loss.detach().cpu()))
        margins.append(step_margin)
        print(
            f"step={step + 1} loss={losses[-1]:.4f} "
            f"margin={step_margin:+.4f}",
            flush=True,
        )

    post_train = evaluate_margins(
        model,
        train_eval_pairs,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
        device=device,
        logprob_mode=args.logprob_mode,
    )
    post_eval = None
    if encoded_eval_pairs:
        post_eval = evaluate_margins(
            model,
            encoded_eval_pairs,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=args.batch_size,
            device=device,
            logprob_mode=args.logprob_mode,
        )

    state_path = None
    if not args.no_save_trainable_state:
        state_path = save_trainable_state(model, out_dir)

    report = {
        "model": args.model,
        "condition": "boundary_preference_reference_free_dpo_smoke",
        "preferences": args.preferences,
        "selected_pairs": len(encoded_pairs),
        "train_eval_pairs": len(train_eval_pairs),
        "eval_preferences": args.eval_preferences,
        "selected_eval_pairs": len(encoded_eval_pairs),
        "failure_modes": sorted({str(pair["failure_mode"]) for pair in encoded_pairs}),
        "eval_failure_modes": sorted({str(pair["failure_mode"]) for pair in encoded_eval_pairs}),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": device,
        "train_last_layers": args.train_last_layers,
        "trainable_params": trainable_params,
        "lr": args.lr,
        "beta": args.beta,
        "logprob_mode": args.logprob_mode,
        "losses": losses,
        "loss_delta": losses[-1] - losses[0] if len(losses) > 1 else 0.0,
        "step_margins": margins,
        "step_margin_delta": margins[-1] - margins[0] if len(margins) > 1 else 0.0,
        "pre_train": pre_train,
        "post_train": post_train,
        "pre_eval": pre_eval,
        "post_eval": post_eval,
        "trainable_state": str(state_path) if state_path else None,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
