#!/usr/bin/env python3
"""Run a strict-contract Stage A SFT smoke experiment.

This is the cluster-side follow-up to the strict prompt-contract baseline. It
trains on compact JSON targets from `stage_a_strict_contract_sft_train_v1.jsonl`,
generates held-out saved predictions, and scores them through the same offline
Stage A evaluator used for API/HPC baselines.

The `--dry-run` path validates artifacts and writes a plan without loading
model weights. Full training requires `--allow-model-load` and is intended for
Cayuga/Expanse or another GPU environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.evaluate_stage_a_predictions import build_report
from post_training.generate_stage_a_predictions import disable_transformers_torchvision_probe
from post_training.run_stage_a_sft_smoke_eval import load_manifest_rows


DATASET = "negbiodb_ct_stage_a_strict_contract_sft_smoke_v1"
PREDICTION_DATASET = "negbiodb_ct_stage_a_strict_contract_sft_smoke_predictions_v1"
PROMPT_CONTRACT = "stage_a_v2_strict"
STRICT_OUTPUT_KEYS = (
    "action",
    "evidence_status",
    "tool_calls",
    "cited_source_ids",
    "rationale",
)
PROMPT_LEAK_TERMS = (
    "hidden_eval_metadata",
    "gold_evidence_status",
    "expected_terminal_action",
    "gold_source_ids",
    "source_task_id",
    "split_group",
)


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def source_case_id(row: Mapping[str, Any]) -> str:
    value = row.get("source_manifest_case_id") or row.get("task_id") or row.get("case_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Strict SFT row is missing a case id: {row.get('id')!r}")
    return value


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True)


def prompt_messages_from_row(row: Mapping[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in row.get("messages", ()):
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "assistant":
            break
        if role in {"system", "user"}:
            messages.append({"role": str(role), "content": content_text(message.get("content", ""))})
    if not messages:
        raise ValueError(f"{source_case_id(row)} has no model-visible prompt messages")
    return messages


def target_output_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = row.get("target_output")
    if isinstance(output, Mapping):
        return dict(output)
    messages = row.get("messages", ())
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, Mapping) and last.get("role") == "assistant":
            content = last.get("content")
            if isinstance(content, str):
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
    raise ValueError(f"{source_case_id(row)} has no strict target output")


def target_text_from_row(row: Mapping[str, Any]) -> str:
    return json.dumps(target_output_from_row(row), sort_keys=True)


def prompt_text_for_tokenizer(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def validate_strict_rows(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    issues: list[str] = []
    train_case_ids = [source_case_id(row) for row in train_rows]
    heldout_case_ids = [source_case_id(row) for row in heldout_rows]

    if len(set(train_case_ids)) != len(train_case_ids):
        issues.append("train_duplicate_case_id")
    if len(set(heldout_case_ids)) != len(heldout_case_ids):
        issues.append("heldout_duplicate_case_id")
    if set(train_case_ids) & set(heldout_case_ids):
        issues.append("train_heldout_case_overlap")

    for split, rows in (("train", train_rows), ("heldout", heldout_rows)):
        for row in rows:
            row_id = row.get("id")
            if row.get("prompt_contract") != PROMPT_CONTRACT:
                issues.append(f"{row_id}:{split}_wrong_prompt_contract")
            prompt_messages = prompt_messages_from_row(row)
            prompt_text = json.dumps(prompt_messages, sort_keys=True)
            if "Strict Stage A output contract" not in prompt_text:
                issues.append(f"{row_id}:{split}_missing_strict_prompt_contract")
            for term in PROMPT_LEAK_TERMS:
                if term in prompt_text:
                    issues.append(f"{row_id}:{split}_prompt_leaks_{term}")
            source_task_id = row.get("source_task_id")
            if source_task_id and str(source_task_id) in prompt_text:
                issues.append(f"{row_id}:{split}_prompt_leaks_source_task_id_value")
            output = target_output_from_row(row)
            missing = [key for key in STRICT_OUTPUT_KEYS if key not in output]
            if missing:
                issues.append(f"{row_id}:{split}_target_missing_keys")
            if set(output).difference(STRICT_OUTPUT_KEYS):
                issues.append(f"{row_id}:{split}_target_extra_keys")
            if not isinstance(output.get("tool_calls"), list):
                issues.append(f"{row_id}:{split}_target_tool_calls_not_list")
    return issues


def dry_run_report(
    *,
    model: str,
    train_sft: str | Path,
    heldout_sft: str | Path,
    manifest: str | Path,
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    issues: Sequence[str],
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "dry_run": True,
        "model": model,
        "prompt_contract": PROMPT_CONTRACT,
        "train_sft": str(train_sft),
        "heldout_sft": str(heldout_sft),
        "manifest": str(manifest),
        "train_examples": len(train_rows),
        "heldout_examples": len(heldout_rows),
        "train_case_ids": [source_case_id(row) for row in train_rows],
        "heldout_case_ids": [source_case_id(row) for row in heldout_rows],
        "issues": list(issues),
        "boundary": (
            "Dry run validates strict-contract artifacts and split boundaries "
            "without loading model weights or running local heavy compute."
        ),
    }


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def collate(features: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_len = max(feature["input_ids"].shape[0] for feature in features)
    input_rows = []
    label_rows = []
    attention_rows = []
    for feature in features:
        pad = max_len - feature["input_ids"].shape[0]
        input_rows.append(torch.cat([feature["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        label_rows.append(torch.cat([feature["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        attention_rows.append(
            torch.cat([torch.ones(feature["input_ids"].shape[0], dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
        )
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


def save_trainable_state(model: Any, out_dir: str | Path) -> Path:
    import torch

    path = Path(out_dir) / "trainable_state.pt"
    torch.save(
        {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        },
        path,
    )
    return path


def generate_prediction_rows(
    model: Any,
    tokenizer: Any,
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    model_id: str,
    device: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    predictions: list[dict[str, Any]] = []
    model.eval()
    for index, row in enumerate(heldout_rows):
        prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_row(row))
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
        predictions.append(
            {
                "case_id": source_case_id(row),
                "dataset": PREDICTION_DATASET,
                "source": "stage_a_strict_contract_sft_smoke",
                "run_id": run_id,
                "model": model_id,
                "prompt_contract": PROMPT_CONTRACT,
                "split": row.get("split"),
                "generation_prompt_hash": row.get("generation_prompt_hash"),
                "raw_output": raw_output,
            }
        )
        print(f"[{index + 1}/{len(heldout_rows)}] generated {source_case_id(row)}", flush=True)
    return predictions


def run_training_and_eval(args: argparse.Namespace, train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    disable_transformers_torchvision_probe()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

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
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    predictions_path = Path(args.predictions_out)
    eval_report_path = Path(args.eval_out)
    write_jsonl(predictions_path, predictions)

    eval_report = build_report(
        manifest_rows=load_manifest_rows(args.manifest),
        prediction_rows=predictions,
        expected_case_ids=[source_case_id(row) for row in heldout_rows],
        run_id=args.run_id,
    )
    write_json(eval_report_path, eval_report)

    return {
        "dataset": DATASET,
        "dry_run": False,
        "run_id": args.run_id,
        "model": args.model,
        "device": device,
        "prompt_contract": PROMPT_CONTRACT,
        "train_sft": args.train_sft,
        "heldout_sft": args.heldout_sft,
        "manifest": args.manifest,
        "train_examples": len(train_rows),
        "heldout_examples": len(heldout_rows),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "train_last_layers": args.train_last_layers,
        "trainable_params": trainable_params,
        "losses": losses,
        "loss_delta": losses[-1] - losses[0] if len(losses) > 1 else 0.0,
        "trainable_state": str(state_path) if state_path else None,
        "predictions": str(predictions_path),
        "eval_report": str(eval_report_path),
        "eval_summary": eval_report["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-sft", default="post_training/stage_a_strict_contract_sft_train_v1.jsonl")
    parser.add_argument("--heldout-sft", default="post_training/stage_a_strict_contract_sft_heldout_v1.jsonl")
    parser.add_argument("--manifest", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument("--out-dir", default="post_training/runs/stage_a_strict_sft_smoke")
    parser.add_argument("--run-id", default="stage_a_strict_contract_sft_smoke")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-heldout", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-trainable-state", action="store_true")
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument("--eval-out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report_out is None:
        args.report_out = str(out_dir / "report.json")
    if args.predictions_out is None:
        args.predictions_out = str(out_dir / "predictions.jsonl")
    if args.eval_out is None:
        args.eval_out = str(out_dir / "eval_report.json")

    train_rows = load_jsonl(args.train_sft, limit=args.limit_train)
    heldout_rows = load_jsonl(args.heldout_sft, limit=args.limit_heldout)
    issues = validate_strict_rows(train_rows, heldout_rows)
    if issues:
        write_json(args.report_out, dry_run_report(
            model=args.model,
            train_sft=args.train_sft,
            heldout_sft=args.heldout_sft,
            manifest=args.manifest,
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            issues=issues,
        ))
        raise SystemExit("Strict-contract SFT smoke validation failed:\n- " + "\n- ".join(issues))

    if args.dry_run:
        report = dry_run_report(
            model=args.model,
            train_sft=args.train_sft,
            heldout_sft=args.heldout_sft,
            manifest=args.manifest,
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            issues=issues,
        )
    else:
        if not args.allow_model_load:
            raise SystemExit("Full strict SFT smoke requires --allow-model-load. Use --dry-run for local validation.")
        report = run_training_and_eval(args, train_rows, heldout_rows)
    write_json(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
