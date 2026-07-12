#!/usr/bin/env python3
"""Minimal Qwen SFT smoke loop over tracked NegBioDB-CT SFT examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for SFT smoke tensor/model operations") from exc
    return torch


def final_decision_from_example(example: Mapping[str, Any]) -> dict[str, Any]:
    final = example["messages"][-1]["tool_call"]
    if final["name"] != "submit_decision":
        raise ValueError(f"Final message is not submit_decision: {example['id']}")
    return dict(final["arguments"])


def format_prompt(example: Mapping[str, Any]) -> str:
    messages = example["messages"][:-1]
    prompt_header = example.get("sft_prompt_header")
    if isinstance(prompt_header, str) and prompt_header.strip():
        header_lines = [line for line in prompt_header.strip().splitlines()]
    else:
        header_lines = [
            "You are learning the final decision step for a clinical-trial tool-use trajectory.",
            "Given the claim and already-returned tool observations, output only the final submit_decision JSON.",
        ]
    lines = [
        *header_lines,
        "",
    ]
    for message in messages:
        role = message["role"]
        if role in {"system", "user"}:
            lines.append(f"{role.upper()}: {message['content']}")
        elif role == "assistant" and "tool_call" in message:
            lines.append(f"ASSISTANT_TOOL_CALL: {json.dumps(message['tool_call'], sort_keys=True)}")
        elif role == "tool":
            content = json.dumps(message["content"], sort_keys=True)
            lines.append(f"TOOL_RESULT {message['name']}: {content}")
        else:
            raise ValueError(f"Unsupported message shape in {example['id']}: {message}")
    prompt_suffix = example.get("sft_prompt_suffix")
    lines.append(str(prompt_suffix) if isinstance(prompt_suffix, str) and prompt_suffix.strip() else "FINAL_SUBMIT_DECISION_JSON:")
    return "\n".join(lines)


def format_target(example: Mapping[str, Any]) -> str:
    target_text = example.get("sft_target_text")
    if isinstance(target_text, str) and target_text.strip():
        return target_text
    return json.dumps(final_decision_from_example(example), sort_keys=True)


def encode_example(
    tokenizer: Any,
    example: Mapping[str, Any],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    torch = import_torch()
    prompt_ids = tokenizer(format_prompt(example), add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(format_target(example) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        prompt_trim = min(overflow, max(0, len(prompt_ids) - 1))
        prompt_ids = prompt_ids[prompt_trim:]
        labels = labels[prompt_trim:]
        input_ids = input_ids[prompt_trim:]
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate(features: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    torch = import_torch()
    max_len = max(feature["input_ids"].shape[0] for feature in features)
    input_rows = []
    label_rows = []
    attention_rows = []
    for feature in features:
        pad = max_len - feature["input_ids"].shape[0]
        input_rows.append(torch.cat([feature["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        label_rows.append(torch.cat([feature["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        attention_rows.append(torch.cat([torch.ones(feature["input_ids"].shape[0], dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_rows),
        "labels": torch.stack(label_rows),
        "attention_mask": torch.stack(attention_rows),
    }


def set_trainable_last_layers(model: Any, layers: int) -> int:
    for param in model.parameters():
        param.requires_grad = False
    if layers <= 0:
        return 0
    blocks = model.model.layers[-layers:]
    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True
    if hasattr(model.model, "norm"):
        for param in model.model.norm.parameters():
            param.requires_grad = True
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def trainable_state_dict(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def save_trainable_state(model: Any, out_dir: str | Path) -> Path:
    torch = import_torch()
    path = Path(out_dir) / "trainable_state.pt"
    torch.save(trainable_state_dict(model), path)
    return path


def load_trainable_state(model: Any, state_path: str | Path) -> dict[str, Any]:
    torch = import_torch()
    state = torch.load(Path(state_path), map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Trainable state must be a dict: {state_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "state_path": str(state_path),
        "loaded_tensors": len(state),
        "missing_tensors": len(missing),
        "unexpected_tensors": list(unexpected),
    }


def choose_device(device: str) -> str:
    torch = import_torch()
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--out-dir", default="post_training/runs/qwen_sft_smoke")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-trainable-state", action="store_true")
    args = parser.parse_args()

    torch = import_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_jsonl(args.sft, limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_example(tokenizer, example, max_length=args.max_length) for example in examples]

    if args.dry_run:
        report = {
            "model": args.model,
            "sft": args.sft,
            "examples": len(examples),
            "max_length": args.max_length,
            "encoded_lengths": [int(item["input_ids"].shape[0]) for item in encoded],
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
    model.train()
    trainable_params = set_trainable_last_layers(model, args.train_last_layers)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
    )

    losses = []
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

    report = {
        "model": args.model,
        "sft": args.sft,
        "examples": len(examples),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": device,
        "train_last_layers": args.train_last_layers,
        "trainable_params": trainable_params,
        "losses": losses,
        "loss_delta": losses[-1] - losses[0] if len(losses) > 1 else 0.0,
        "trainable_state": str(state_path) if state_path else None,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
