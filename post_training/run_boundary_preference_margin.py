#!/usr/bin/env python3
"""Score evidence-boundary preference pairs by model likelihood margin."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_sft_constrained_eval import score_candidate  # noqa: E402
from post_training.run_sft_smoke import (  # noqa: E402
    choose_device,
    format_prompt,
    load_jsonl,
    load_trainable_state,
)


def final_candidate(messages: list[Mapping[str, Any]]) -> dict[str, str]:
    if len(messages) != 1:
        raise ValueError("Preference branch should contain exactly one final message.")
    tool_call = messages[0].get("tool_call", {})
    if tool_call.get("name") != "submit_decision":
        raise ValueError("Preference branch is missing submit_decision.")
    args = tool_call.get("arguments")
    if not isinstance(args, Mapping) or not isinstance(args.get("action"), str):
        raise ValueError("submit_decision arguments must include action.")
    candidate = {"action": str(args["action"])}
    if args.get("nct"):
        candidate["nct"] = str(args["nct"])
    return candidate


def prompt_from_pair(pair: Mapping[str, Any]) -> str:
    chosen_messages = pair.get("chosen_messages")
    if not isinstance(chosen_messages, list):
        raise ValueError(f"Missing chosen_messages for {pair.get('id')}")
    example = {
        "id": pair.get("id"),
        "messages": list(pair["prompt_messages"]) + list(chosen_messages),
    }
    return format_prompt(example)


def margin_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": round(sum(values) / len(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def summarize_group(rows: list[Mapping[str, Any]], *, key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    out = {}
    for name, group_rows in sorted(groups.items()):
        margins = [float(row["mean_margin"]) for row in group_rows]
        wins = sum(bool(row["mean_chosen_wins"]) for row in group_rows)
        out[name] = {
            "n": len(group_rows),
            "mean_win_rate": round(wins / len(group_rows), 3),
            "mean_margin": margin_stats(margins),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--state", default=None)
    parser.add_argument("--preferences", default="post_training/negbiodb_ct_oracle_boundary_preferences_v1.jsonl")
    parser.add_argument("--out", default="post_training/runs/qwen_boundary_preference_margin/base.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(args.device)
    pairs = load_jsonl(args.preferences, limit=args.limit)
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
    for i, pair in enumerate(pairs):
        prompt = prompt_from_pair(pair)
        chosen = final_candidate(pair["chosen_messages"])
        rejected = final_candidate(pair["rejected_messages"])
        chosen_score = score_candidate(
            model,
            tokenizer,
            prompt,
            chosen,
            max_length=args.max_length,
            device=device,
        )
        rejected_score = score_candidate(
            model,
            tokenizer,
            prompt,
            rejected,
            max_length=args.max_length,
            device=device,
        )
        mean_margin = rejected_score["mean_nll"] - chosen_score["mean_nll"]
        sum_margin = rejected_score["sum_nll"] - chosen_score["sum_nll"]
        row = {
            "id": pair["id"],
            "task_id": pair["task_id"],
            "failure_mode": pair["failure_mode"],
            "evidence_derived_action": pair["evidence_derived_action"],
            "rejected_action": pair["rejected_action"],
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "mean_margin": mean_margin,
            "sum_margin": sum_margin,
            "mean_chosen_wins": mean_margin > 0,
            "sum_chosen_wins": sum_margin > 0,
        }
        rows.append(row)
        print(
            f"  [{i:3d}] {pair['failure_mode']:28s} "
            f"mean_margin={mean_margin:+.4f} "
            f"{'WIN' if row['mean_chosen_wins'] else 'loss'}",
            flush=True,
        )

    mean_margins = [float(row["mean_margin"]) for row in rows]
    sum_margins = [float(row["sum_margin"]) for row in rows]
    mean_wins = sum(bool(row["mean_chosen_wins"]) for row in rows)
    sum_wins = sum(bool(row["sum_chosen_wins"]) for row in rows)
    summary = {
        "model": args.model,
        "condition": "boundary_preference_margin",
        "state": args.state,
        "loaded_state": loaded_state,
        "preferences": args.preferences,
        "n": len(rows),
        "max_length": args.max_length,
        "device": device,
        "mean_win_rate": round(mean_wins / len(rows), 3) if rows else None,
        "sum_win_rate": round(sum_wins / len(rows), 3) if rows else None,
        "mean_margin": margin_stats(mean_margins),
        "sum_margin": margin_stats(sum_margins),
        "by_failure_mode": summarize_group(rows, key="failure_mode"),
        "by_chosen_action": summarize_group(rows, key="evidence_derived_action"),
        "by_rejected_action": summarize_group(rows, key="rejected_action"),
    }
    print("\n=== Boundary preference margin ===\n" + json.dumps(summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
