#!/usr/bin/env python3
"""Evaluate an SFT smoke checkpoint on final decisions from native tool context."""

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
from negbiodb_ct.baselines import extract_nct, score_decision
from negbiodb_ct.model_output import ModelOutputParseError, parse_model_output_json
from negbiodb_ct.run_agent import generic_score
from post_training.run_sft_smoke import (
    choose_device,
    format_prompt,
    load_jsonl,
    load_trainable_state,
)


VALID_ACTIONS = {"ground", "reject", "defer", "verify", "flag"}


def tool_calls_from_example(example: Mapping[str, Any]) -> list[str]:
    calls = []
    for message in example["messages"]:
        tool_call = message.get("tool_call") if isinstance(message, Mapping) else None
        if not isinstance(tool_call, Mapping):
            continue
        name = tool_call.get("name")
        if name and name != "submit_decision":
            calls.append(str(name))
    return calls


def parse_final_decision(raw: str) -> dict[str, str | None]:
    try:
        obj = parse_model_output_json(raw)
    except ModelOutputParseError:
        return {"action": None, "cited_nct": None}

    action = obj.get("action")
    if not isinstance(action, str):
        return {"action": None, "cited_nct": None}
    action = action.strip().lower()
    if action not in VALID_ACTIONS:
        return {"action": None, "cited_nct": None}

    cited = extract_nct(str(obj.get("nct") or obj.get("cited_nct") or ""))
    return {
        "action": action,
        "cited_nct": cited if action in {"ground", "flag"} else None,
    }


class TransformersSFTDecisionClient:
    """Local causal-LM generator for the SFT final-decision prompt."""

    def __init__(
        self,
        model_id: str,
        *,
        state_path: str | None,
        device: str = "auto",
        local_files_only: bool = True,
        max_new_tokens: int = 64,
        json_prefill: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = choose_device(device)
        self.max_new_tokens = max_new_tokens
        self.json_prefill = json_prefill
        self.prefill_text = '\n{"action": "' if json_prefill else ""
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            torch_dtype="auto",
        )
        self.loaded_state: dict[str, Any] | None = None
        if state_path:
            self.loaded_state = load_trainable_state(self.model, state_path)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt + self.prefill_text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return (self.prefill_text + self.tokenizer.decode(generated, skip_special_tokens=True)).strip()


def run_example(
    client: TransformersSFTDecisionClient | None,
    example: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    dry_run: bool,
) -> tuple[dict[str, str | None], str, list[str], dict[str, Any], Any]:
    raw_text = "" if dry_run else client.generate(format_prompt(example))  # type: ignore[union-attr]
    decision = {"action": None, "cited_nct": None} if dry_run else parse_final_decision(raw_text)
    called = tool_calls_from_example(example)
    model_output, generic_result = generic_score(record, decision, called)
    return decision, raw_text, called, model_output, generic_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--state", default="post_training/runs/qwen_sft_smoke/trainable_state.pt")
    parser.add_argument("--sft", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument("--out", default="post_training/runs/qwen_sft_decision_eval_n40.json")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-prefill", action="store_true")
    args = parser.parse_args()

    examples = load_jsonl(args.sft, limit=args.limit)
    records = {record["packet_id"]: record for record in load_task_records(args.tasks)}

    client = None
    loaded_state = None
    if not args.dry_run:
        client = TransformersSFTDecisionClient(
            args.model,
            state_path=args.state,
            device=args.device,
            local_files_only=not args.allow_download,
            max_new_tokens=args.max_new_tokens,
            json_prefill=args.json_prefill,
        )
        loaded_state = client.loaded_state

    rows = []
    by_class = defaultdict(lambda: [0, 0])
    rewards = []
    generic_scores = []
    parse_failures = 0
    toolrate = 0
    for i, example in enumerate(examples):
        task_id = example["task_id"]
        record = records[task_id]
        decision, raw_text, called, model_output, generic_result = run_example(
            client,
            example,
            record,
            dry_run=args.dry_run,
        )
        if decision["action"] is None:
            parse_failures += 1
        score = score_decision(decision, record["scoring_key"])
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
            "raw_text": raw_text,
            "model_output": model_output,
            "correct": score["correct"],
            "reward": score["reward"],
            "generic_score": round(generic_result.score, 3),
            "generic_violations": list(generic_result.violations),
        })
        print(
            f"  [{i:2d}] {record['action_class']:7s} pred={str(decision['action']):7s} "
            f"gold={record['scoring_key']['gold_action']:7s} {'OK' if score['correct'] else 'X '} "
            f"generic={generic_result.score:.2f}",
            flush=True,
        )

    summary = {
        "model": args.model,
        "condition": (
            "sft_decision_json_prefill_from_native_tool_context"
            if args.json_prefill
            else "sft_decision_from_native_tool_context"
        ),
        "state": args.state,
        "loaded_state": loaded_state,
        "json_prefill": args.json_prefill,
        "n": len(examples),
        "sft": args.sft,
        "action_accuracy": round(sum(correct for correct, _ in by_class.values()) / len(examples), 3),
        "mean_reward": round(sum(rewards) / len(rewards), 3),
        "generic_mean_score": round(sum(generic_scores) / len(generic_scores), 3),
        "tool_call_rate": round(toolrate / len(examples), 3),
        "parse_failures": parse_failures,
        "by_class": {task_class: f"{values[0]}/{values[1]}" for task_class, values in sorted(by_class.items())},
    }
    print("\n=== SFT decision eval ===\n" + json.dumps(summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
