#!/usr/bin/env python3
"""Constrained final-decision scoring for tracked NegBioDB-CT SFT examples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from negbiodb_ct import load_task_records
from negbiodb_ct.baselines import score_decision
from negbiodb_ct.run_agent import generic_score
from post_training.run_sft_decision_eval import tool_calls_from_example
from post_training.run_sft_smoke import (
    choose_device,
    format_prompt,
    load_jsonl,
    load_trainable_state,
)


BASE_ACTIONS = ("defer", "verify", "reject")


def import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for constrained SFT candidate scoring") from exc
    return torch


def returned_ncts(example: Mapping[str, Any]) -> list[str]:
    ncts = []
    for message in example["messages"]:
        if message.get("role") != "tool" or message.get("name") != "search_failures":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for row in content:
            if isinstance(row, Mapping) and row.get("nct"):
                nct = str(row["nct"])
                if nct not in ncts:
                    ncts.append(nct)
    return ncts


def candidate_decisions(example: Mapping[str, Any]) -> list[dict[str, str]]:
    candidates = [{"action": action} for action in BASE_ACTIONS]
    for nct in returned_ncts(example):
        candidates.append({"action": "ground", "nct": nct})
        candidates.append({"action": "flag", "nct": nct})
    return candidates


def candidate_to_decision(candidate: Mapping[str, str]) -> dict[str, str | None]:
    action = candidate["action"]
    return {
        "action": action,
        "cited_nct": candidate.get("nct") if action in {"ground", "flag"} else None,
    }


def target_json(candidate: Mapping[str, str], eos_token: str) -> str:
    return json.dumps(dict(candidate), sort_keys=True) + eos_token


def encode_candidate(
    tokenizer: Any,
    prompt: str,
    candidate: Mapping[str, str],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    torch = import_torch()
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_json(candidate, tokenizer.eos_token), add_special_tokens=False)["input_ids"]
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
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
    }


def score_candidate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate: Mapping[str, str],
    *,
    max_length: int,
    device: str,
) -> dict[str, Any]:
    torch = import_torch()
    batch = encode_candidate(tokenizer, prompt, candidate, max_length=max_length)
    target_tokens = int((batch["labels"] != -100).sum().item())
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        loss = model(**batch).loss
    mean_nll = float(loss.detach().cpu())
    return {
        "candidate": dict(candidate),
        "mean_nll": mean_nll,
        "sum_nll": mean_nll * target_tokens,
        "target_tokens": target_tokens,
    }


def choose_scored_candidate(
    scores: list[dict[str, Any]],
    *,
    score_mode: str,
) -> dict[str, Any]:
    if score_mode not in {"mean", "sum"}:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    key = "mean_nll" if score_mode == "mean" else "sum_nll"
    return min(scores, key=lambda item: item[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--state", default=None)
    parser.add_argument("--sft", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument("--out", default="post_training/runs/qwen_sft_constrained_eval_n40.json")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-mode", choices=("mean", "sum"), default="mean")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(args.device)
    examples = load_jsonl(args.sft, limit=args.limit)
    records = {record["packet_id"]: record for record in load_task_records(args.tasks)}

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        torch_dtype="auto",
    )
    loaded_state = load_trainable_state(model, args.state) if args.state else None
    model.to(device)
    model.eval()

    rows = []
    by_class = defaultdict(lambda: [0, 0])
    rewards = []
    generic_scores = []
    toolrate = 0
    for i, example in enumerate(examples):
        task_id = example["task_id"]
        record = records[task_id]
        prompt = format_prompt(example)
        candidates = candidate_decisions(example)
        scores = [
            score_candidate(
                model,
                tokenizer,
                prompt,
                candidate,
                max_length=args.max_length,
                device=device,
            )
            for candidate in candidates
        ]
        winner = choose_scored_candidate(scores, score_mode=args.score_mode)
        decision = candidate_to_decision(winner["candidate"])
        called = tool_calls_from_example(example)
        score = score_decision(decision, record["scoring_key"])
        model_output, generic_result = generic_score(record, decision, called)
        by_class[record["action_class"]][0] += score["correct"]
        by_class[record["action_class"]][1] += 1
        rewards.append(score["reward"])
        generic_scores.append(generic_result.score)
        toolrate += bool(called)
        rows.append({
            "packet_id": task_id,
            "class": record["action_class"],
            "gold": record["scoring_key"]["gold_action"],
            "pred": decision,
            "called": called,
            "winner": winner,
            "candidate_scores": sorted(scores, key=lambda item: item["mean_nll"]),
            "model_output": model_output,
            "correct": score["correct"],
            "reward": score["reward"],
            "generic_score": round(generic_result.score, 3),
            "generic_violations": list(generic_result.violations),
        })
        print(
            f"  [{i:2d}] {record['action_class']:7s} pred={decision['action']:7s} "
            f"gold={record['scoring_key']['gold_action']:7s} {'OK' if score['correct'] else 'X '} "
            f"nll={winner['mean_nll']:.3f}",
            flush=True,
        )

    summary = {
        "model": args.model,
        "condition": "sft_constrained_final_decision",
        "state": args.state,
        "loaded_state": loaded_state,
        "n": len(examples),
        "sft": args.sft,
        "score_mode": args.score_mode,
        "max_length": args.max_length,
        "action_accuracy": round(sum(correct for correct, _ in by_class.values()) / len(examples), 3),
        "mean_reward": round(sum(rewards) / len(rewards), 3),
        "generic_mean_score": round(sum(generic_scores) / len(generic_scores), 3),
        "tool_call_rate": round(toolrate / len(examples), 3),
        "parse_failures": 0,
        "by_class": {task_class: f"{values[0]}/{values[1]}" for task_class, values in sorted(by_class.items())},
    }
    print("\n=== SFT constrained decision eval ===\n" + json.dumps(summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
