#!/usr/bin/env python3
"""Teacher-forced loss diagnostics for tracked NegBioDB-CT SFT examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_sft_smoke import (
    choose_device,
    collate,
    encode_example,
    load_jsonl,
    load_trainable_state,
)


def batched(items: list[dict[str, torch.Tensor]], batch_size: int) -> list[list[dict[str, torch.Tensor]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def evaluate_loss(
    model: Any,
    encoded: list[dict[str, torch.Tensor]],
    *,
    pad_token_id: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_losses = []
    with torch.no_grad():
        for features in batched(encoded, batch_size):
            batch = collate(features, pad_token_id)
            token_count = int((batch["labels"] != -100).sum().item())
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss_value = float(loss.detach().cpu())
            batch_losses.append(loss_value)
            total_loss += loss_value * token_count
            total_tokens += token_count

    return {
        "loss": total_loss / total_tokens if total_tokens else None,
        "batch_losses": batch_losses,
        "target_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--state", default="post_training/runs/qwen_sft_smoke/trainable_state.pt")
    parser.add_argument("--sft", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--out", default="post_training/runs/qwen_sft_loss_compare.json")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(args.device)
    examples = load_jsonl(args.sft, limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_example(tokenizer, example, max_length=args.max_length) for example in examples]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        torch_dtype="auto",
    )
    model.to(device)

    base = evaluate_loss(
        model,
        encoded,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
        device=device,
    )
    loaded_state = load_trainable_state(model, args.state)
    loaded = evaluate_loss(
        model,
        encoded,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
        device=device,
    )

    base_loss = base["loss"]
    loaded_loss = loaded["loss"]
    report = {
        "model": args.model,
        "sft": args.sft,
        "state": args.state,
        "loaded_state": loaded_state,
        "examples": len(examples),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": device,
        "base": base,
        "loaded": loaded,
        "loss_delta": loaded_loss - base_loss if base_loss is not None and loaded_loss is not None else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
