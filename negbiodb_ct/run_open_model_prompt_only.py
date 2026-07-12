#!/usr/bin/env python3
"""Open-model prompt-only baseline for NegBioDB-CT packet sets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
NEGBIODB = Path(os.environ.get("NEGBIODB_ROOT", ROOT.parent / "Negative_result_DB")).expanduser()
sys.path.insert(0, str(NEGBIODB / "src"))
sys.path.insert(0, str(ROOT))

from negbiodb_ct import load_task_records
from negbiodb_ct.baselines import score_decision
from negbiodb_ct.run_agent import generic_score
from negbiodb_ct.run_prompt_only import (
    SYS_PROMPT_ONLY,
    load_packet_ids,
    parse_prompt_only_decision,
    select_tasks,
)


class TransformersPromptOnlyClient:
    """Tiny wrapper around local Hugging Face causal LMs."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        max_new_tokens: int = 96,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            torch_dtype="auto",
        )
        self.model.to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, system: str, user: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"System: {system}\nUser: {user}\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_open_model_task(
    client: TransformersPromptOnlyClient,
    task: Mapping[str, Any],
) -> tuple[dict[str, str | None], str]:
    raw_text = client.generate(SYS_PROMPT_ONLY, str(task["observation"]["claim"]))
    return parse_prompt_only_decision(raw_text), raw_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--tasks", default=str(Path(__file__).parent / "tasks_pilot.jsonl"))
    parser.add_argument("--task-ids-from", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--out", default=str(Path(__file__).parent / "agent_open_model_qwen05_n40.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    tasks = load_task_records(args.tasks)
    packet_ids = load_packet_ids(args.task_ids_from) if args.task_ids_from else None
    sample = select_tasks(tasks, n=args.n, packet_ids=packet_ids)

    client = None
    if not args.dry_run:
        client = TransformersPromptOnlyClient(
            args.model,
            device=args.device,
            local_files_only=not args.allow_download,
            max_new_tokens=args.max_new_tokens,
        )

    rows = []
    by_class = defaultdict(lambda: [0, 0])
    rewards = []
    generic_scores = []
    for i, task in enumerate(sample):
        if args.dry_run:
            decision, raw_text = {"action": None, "cited_nct": None}, ""
        else:
            decision, raw_text = run_open_model_task(client, task)

        score = score_decision(decision, task["scoring_key"])
        model_output, generic_result = generic_score(task, decision, [])
        by_class[task["action_class"]][0] += score["correct"]
        by_class[task["action_class"]][1] += 1
        rewards.append(score["reward"])
        generic_scores.append(generic_result.score)
        rows.append({
            "packet_id": task["packet_id"],
            "class": task["action_class"],
            "gold": task["scoring_key"]["gold_action"],
            "pred": decision,
            "called": [],
            "raw_text": raw_text,
            "model_output": model_output,
            "correct": score["correct"],
            "reward": score["reward"],
            "generic_score": round(generic_result.score, 3),
            "generic_violations": list(generic_result.violations),
        })
        print(
            f"  [{i:2d}] {task['action_class']:7s} pred={str(decision['action']):7s} "
            f"gold={task['scoring_key']['gold_action']:7s} {'OK' if score['correct'] else 'X '} "
            f"generic={generic_result.score:.2f}",
            flush=True,
        )

    summary = {
        "model": args.model,
        "condition": "open_model_prompt_only_no_tools",
        "n": len(sample),
        "task_ids_from": args.task_ids_from,
        "action_accuracy": round(sum(correct for correct, _ in by_class.values()) / len(sample), 3),
        "mean_reward": round(sum(rewards) / len(rewards), 3),
        "generic_mean_score": round(sum(generic_scores) / len(generic_scores), 3),
        "tool_call_rate": 0.0,
        "by_class": {task_class: f"{values[0]}/{values[1]}" for task_class, values in sorted(by_class.items())},
    }
    print("\n=== Open-model prompt-only baseline ===\n" + json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
