#!/usr/bin/env python3
"""Prompt-only NegBioDB-CT baseline over the same packet set as post-training data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from negbiodb_ct.paths import NEGBIODB_ROOT
NEGBIODB = NEGBIODB_ROOT
sys.path.insert(0, str(NEGBIODB / "src"))

from negbiodb_ct import load_task_records
from negbiodb_ct.model_output import ModelOutputParseError, parse_model_output_json
from negbiodb_ct.run_agent import generic_score
from negbiodb_ct.baselines import extract_nct, score_decision


SYS_PROMPT_ONLY = (
    "You are a prompt-only baseline for a clinical-trial claim task. "
    "No database or tool results are available. Decide whether a drug has been "
    "tested and FAILED for an indication using only the claim text and your prior "
    "knowledge. Output JSON only, exactly one object with action in "
    "ground|reject|flag|verify|defer. Use ground or flag only if you can cite a "
    "specific NCT id in an 'nct' field; otherwise prefer verify or defer."
)


def parse_prompt_only_decision(raw: str) -> dict[str, str | None]:
    try:
        obj = parse_model_output_json(raw)
    except ModelOutputParseError:
        return {"action": None, "cited_nct": None}

    action = obj.get("action")
    if not isinstance(action, str):
        return {"action": None, "cited_nct": None}
    action = action.strip().lower()
    if action not in {"ground", "reject", "flag", "verify", "defer"}:
        return {"action": None, "cited_nct": None}
    cited_nct = extract_nct(str(obj.get("nct") or obj.get("cited_nct") or ""))
    return {
        "action": action,
        "cited_nct": cited_nct if action in {"ground", "flag"} else None,
    }


def load_packet_ids(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    ids = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            task_id = obj.get("task_id") or obj.get("packet_id")
            if not isinstance(task_id, str):
                raise ValueError(f"Missing task_id/packet_id in {path}")
            ids.append(task_id)
    return ids


def select_tasks(
    tasks: list[dict[str, Any]],
    *,
    n: int,
    packet_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if packet_ids is not None:
        by_id = {task["packet_id"]: task for task in tasks}
        missing = [packet_id for packet_id in packet_ids if packet_id not in by_id]
        if missing:
            raise ValueError(f"packet_id(s) not found: {missing[:5]}")
        return [by_id[packet_id] for packet_id in packet_ids]

    rng = random.Random(42)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_class[task["action_class"]].append(task)
    per = max(1, n // len(by_class))
    sample = []
    for task_class, class_tasks in by_class.items():
        sample += rng.sample(class_tasks, min(per, len(class_tasks)))
    rng.shuffle(sample)
    return sample


def run_prompt_only_task(client: Any, model: str, task: Mapping[str, Any]) -> tuple[dict[str, str | None], str]:
    response = client.messages.create(
        model=model,
        system=SYS_PROMPT_ONLY,
        max_tokens=256,
        temperature=0.0,
        messages=[{"role": "user", "content": str(task["observation"]["claim"])}],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return parse_prompt_only_decision(text), text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--tasks", default=str(Path(__file__).parent / "tasks_pilot.jsonl"))
    parser.add_argument("--task-ids-from", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--out", default=str(Path(__file__).parent / "agent_prompt_only_sonnet_n40.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = load_task_records(args.tasks)
    packet_ids = load_packet_ids(args.task_ids_from) if args.task_ids_from else None
    sample = select_tasks(tasks, n=args.n, packet_ids=packet_ids)

    client = None
    if not args.dry_run:
        import anthropic
        from negbiodb.llm_client import LLMClient

        client = anthropic.Anthropic(api_key=LLMClient(provider="anthropic", model=args.model).api_key)

    rows = []
    by_class = defaultdict(lambda: [0, 0])
    rewards = []
    generic_scores = []
    for i, task in enumerate(sample):
        if args.dry_run:
            decision, raw_text = {"action": None, "cited_nct": None}, ""
        else:
            decision, raw_text = run_prompt_only_task(client, args.model, task)

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
        "condition": "prompt_only_no_tools",
        "n": len(sample),
        "task_ids_from": args.task_ids_from,
        "action_accuracy": round(sum(correct for correct, _ in by_class.values()) / len(sample), 3),
        "mean_reward": round(sum(rewards) / len(rewards), 3),
        "generic_mean_score": round(sum(generic_scores) / len(generic_scores), 3),
        "tool_call_rate": 0.0,
        "by_class": {task_class: f"{values[0]}/{values[1]}" for task_class, values in sorted(by_class.items())},
    }
    print("\n=== Prompt-only baseline ===\n" + json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
