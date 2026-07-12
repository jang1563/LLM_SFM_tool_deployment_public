#!/usr/bin/env python3
"""Constrained final-decision candidate eval for boundary preference pairs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_boundary_preference_margin import (  # noqa: E402
    final_candidate,
    margin_stats,
    prompt_from_pair,
)
from post_training.run_sft_constrained_eval import (  # noqa: E402
    candidate_decisions,
    choose_scored_candidate,
    score_candidate,
)
from post_training.run_sft_smoke import (  # noqa: E402
    choose_device,
    load_jsonl,
    load_trainable_state,
)


def load_preference_rows(path: str | Path, limit: int | None) -> list[dict[str, Any]]:
    return load_jsonl(path, limit=None if limit is not None and limit <= 0 else limit)


def candidate_key(candidate: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in candidate.items()))


def candidates_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return candidate_key(left) == candidate_key(right)


def candidate_actions_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("action") == right.get("action")


def candidates_for_pair(pair: Mapping[str, Any]) -> tuple[list[dict[str, str]], bool]:
    expected = final_candidate(pair["chosen_messages"])
    example = {"messages": list(pair["prompt_messages"])}
    candidates = candidate_decisions(example)
    expected_in_candidates = any(candidates_match(candidate, expected) for candidate in candidates)
    if not expected_in_candidates:
        candidates.append(expected)
    return candidates, expected_in_candidates


def score_key(score: Mapping[str, Any], score_mode: str) -> float:
    if score_mode == "mean":
        return float(score["mean_nll"])
    if score_mode == "sum":
        return float(score["sum_nll"])
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def expected_rank(
    scores: list[Mapping[str, Any]],
    expected: Mapping[str, Any],
    *,
    score_mode: str,
) -> dict[str, Any]:
    sorted_scores = sorted(scores, key=lambda item: score_key(item, score_mode))
    winner_score = score_key(sorted_scores[0], score_mode)
    for index, score in enumerate(sorted_scores, start=1):
        if candidates_match(score["candidate"], expected):
            expected_score = score_key(score, score_mode)
            return {
                "rank": index,
                "score": expected_score,
                "margin_from_winner": expected_score - winner_score,
            }
    raise ValueError(f"Expected candidate was not scored: {expected}")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--state", default=None)
    parser.add_argument(
        "--preferences",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--out",
        default="post_training/runs/qwen_boundary_preference_candidate_eval/hard_heldout.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-mode", choices=("mean", "sum"), default="mean")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(args.device)
    pairs = load_preference_rows(args.preferences, args.limit)
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
    for index, pair in enumerate(pairs):
        prompt = prompt_from_pair(pair)
        expected = final_candidate(pair["chosen_messages"])
        candidates, expected_in_candidates = candidates_for_pair(pair)
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
        rank = expected_rank(scores, expected, score_mode=args.score_mode)
        pred = dict(winner["candidate"])
        row = {
            "id": pair["id"],
            "task_id": pair["task_id"],
            "failure_mode": pair["failure_mode"],
            "expected_action": pair["evidence_derived_action"],
            "rejected_action": pair["rejected_action"],
            "expected": expected,
            "pred": pred,
            "action_correct": candidate_actions_match(pred, expected),
            "exact_candidate_correct": candidates_match(pred, expected),
            "expected_rank": rank["rank"],
            "expected_margin_from_winner": rank["margin_from_winner"],
            "expected_in_candidates": expected_in_candidates,
            "winner": winner,
            "candidate_scores": sorted(scores, key=lambda item: score_key(item, args.score_mode)),
        }
        rows.append(row)
        print(
            f"  [{index:3d}] {pair['failure_mode']:28s} "
            f"pred={pred.get('action'):7s} expected={expected.get('action'):7s} "
            f"rank={rank['rank']} {'OK' if row['exact_candidate_correct'] else 'X '}",
            flush=True,
        )

    action_correct = sum(bool(row["action_correct"]) for row in rows)
    exact_correct = sum(bool(row["exact_candidate_correct"]) for row in rows)
    margins = [float(row["expected_margin_from_winner"]) for row in rows]
    ranks = [int(row["expected_rank"]) for row in rows]
    summary = {
        "model": args.model,
        "condition": "boundary_preference_constrained_candidate_eval",
        "state": args.state,
        "loaded_state": loaded_state,
        "preferences": args.preferences,
        "n": len(rows),
        "score_mode": args.score_mode,
        "max_length": args.max_length,
        "device": device,
        "action_accuracy": round(action_correct / len(rows), 3) if rows else None,
        "exact_candidate_accuracy": round(exact_correct / len(rows), 3) if rows else None,
        "expected_rank_counts": dict(sorted(Counter(ranks).items())),
        "expected_margin_from_winner": margin_stats(margins),
        "missing_expected_candidates": sum(not bool(row["expected_in_candidates"]) for row in rows),
        "by_failure_mode": summarize_group(rows, key="failure_mode"),
        "by_expected_action": summarize_group(rows, key="expected_action"),
        "by_rejected_action": summarize_group(rows, key="rejected_action"),
    }
    print("\n=== Boundary preference constrained candidate eval ===\n" + json.dumps(summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
