#!/usr/bin/env python3
"""Run the Stage A evidence-conditioned candidate-routing smoke.

The dry-run path validates the public-safe finite-candidate rows without
loading model weights. Full mode requires `--allow-model-load` and is intended
for Cayuga/Expanse. Raw predictions, candidate scores, trainable state, and
logs belong under ignored run directories; only compact summaries should be
curated back into the public repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.export_stage_a_saved_output_evidence_candidate_routing_rows import (  # noqa: E402
    CANDIDATE_PAIRS,
    DATASET as ROW_DATASET,
    PROMPT_CONTRACT,
)
from post_training.generate_stage_a_predictions import disable_transformers_torchvision_probe  # noqa: E402
from post_training.run_stage_a_strict_component_sft_smoke import score_candidate_target  # noqa: E402
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    choose_device,
    collate,
    load_jsonl,
    prompt_text_for_tokenizer,
    save_trainable_state,
    set_trainable_last_layers,
    write_json,
    write_jsonl,
)
from post_training.validate_post_training_data import (  # noqa: E402
    validate_stage_a_saved_output_evidence_candidate_routing,
)


DATASET = "negbiodb_ct_stage_a_saved_output_evidence_candidate_routing_smoke_v1"
PREDICTION_DATASET = (
    "negbiodb_ct_stage_a_saved_output_evidence_candidate_routing_predictions_v1"
)
DEFAULT_ROWS = "post_training/stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl"
DEFAULT_TRAIN_ROWS = (
    "post_training/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl"
)
DEFAULT_HELDOUT_ROWS = (
    "post_training/stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl"
)
DEFAULT_MANIFEST = "post_training/stage_a_saved_output_evidence_candidate_routing_manifest.json"
TARGET_KEYS = ("action", "evidence_status", "selected_pair")
PROMPT_LEAK_TERMS = (
    "hidden_eval_metadata",
    "target_output",
    "target_pair",
    "target_messages",
    "source_task_id",
    "split_group",
    "case_family",
    "oracle_target",
)
SYSTEM_PROMPT = (
    "You are scoring a Stage A biology tool-use candidate-routing component. "
    "Select exactly one candidate pair from the visible evidence state and "
    "return only compact JSON with selected_pair, action, and evidence_status."
)


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def source_case_id(row: Mapping[str, Any]) -> str:
    value = row.get("source_manifest_case_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Candidate-routing row is missing case id: {row.get('id')!r}")
    return value


def pair_for_output(output: Mapping[str, Any]) -> str:
    selected = output.get("selected_pair")
    if isinstance(selected, str) and selected:
        return selected
    return f"{output.get('action')}/{output.get('evidence_status')}"


def row_value_counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def prompt_messages_from_row(row: Mapping[str, Any]) -> list[dict[str, str]]:
    task = row.get("model_visible_task")
    if not isinstance(task, Mapping):
        raise ValueError(f"{row.get('id')} has no model_visible_task")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(dict(task), sort_keys=True)},
    ]


def target_output_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    target = row.get("target_output")
    if not isinstance(target, Mapping):
        raise ValueError(f"{row.get('id')} has no target_output")
    return {
        "action": str(target.get("action")),
        "evidence_status": str(target.get("evidence_status")),
        "selected_pair": str(target.get("selected_pair")),
    }


def target_text_from_row(row: Mapping[str, Any]) -> str:
    return json.dumps(target_output_from_row(row), sort_keys=True)


def candidate_outputs_from_row(row: Mapping[str, Any]) -> list[dict[str, str]]:
    candidates = row.get("candidate_outputs")
    if not isinstance(candidates, list):
        raise ValueError(f"{row.get('id')} has no candidate_outputs list")
    out: list[dict[str, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{row.get('id')} has malformed candidate output")
        pair = str(candidate.get("pair"))
        action = str(candidate.get("action"))
        evidence_status = str(candidate.get("evidence_status"))
        out.append(
            {
                "action": action,
                "evidence_status": evidence_status,
                "selected_pair": pair,
            }
        )
    return out


def encode_example(tokenizer: Any, row: Mapping[str, Any], *, max_length: int) -> dict[str, Any]:
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


def validate_candidate_routing_rows(
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    issues = validate_stage_a_saved_output_evidence_candidate_routing(
        [dict(row) for row in rows],
        [dict(row) for row in train_rows],
        [dict(row) for row in heldout_rows],
        dict(manifest),
    )
    all_ids = {str(row.get("id")) for row in rows}
    split_ids = {str(row.get("id")) for row in list(train_rows) + list(heldout_rows)}
    if all_ids != split_ids:
        issues.append("rows_do_not_equal_train_plus_heldout_ids")

    for split, split_rows in (("train", train_rows), ("heldout", heldout_rows)):
        for row in split_rows:
            row_id = row.get("id")
            if row.get("dataset") != ROW_DATASET:
                issues.append(f"{row_id}:{split}_unexpected_dataset")
            if row.get("prompt_contract") != PROMPT_CONTRACT:
                issues.append(f"{row_id}:{split}_unexpected_prompt_contract")
            if row.get("component") != "routing_after_loop":
                issues.append(f"{row_id}:{split}_unexpected_component")
            target = target_output_from_row(row)
            if tuple(sorted(target)) != tuple(sorted(TARGET_KEYS)):
                issues.append(f"{row_id}:{split}_target_key_mismatch")
            if target["selected_pair"] not in CANDIDATE_PAIRS:
                issues.append(f"{row_id}:{split}_target_pair_not_candidate")
            if target["selected_pair"] != f"{target['action']}/{target['evidence_status']}":
                issues.append(f"{row_id}:{split}_target_pair_field_mismatch")
            if row.get("target_pair") != target["selected_pair"]:
                issues.append(f"{row_id}:{split}_target_pair_mismatch")
            if row.get("runtime_evidence_pair") != target["selected_pair"]:
                issues.append(f"{row_id}:{split}_runtime_pair_mismatch")
            if row.get("runtime_evidence_exact") is not True:
                issues.append(f"{row_id}:{split}_runtime_not_exact")
            if split == "train" and row.get("training_allowed") is not True:
                issues.append(f"{row_id}:{split}_training_not_allowed")
            if split == "heldout" and row.get("evaluation_only") is not True:
                issues.append(f"{row_id}:{split}_heldout_not_evaluation_only")
            candidate_pairs = [candidate["selected_pair"] for candidate in candidate_outputs_from_row(row)]
            if candidate_pairs != list(CANDIDATE_PAIRS):
                issues.append(f"{row_id}:{split}_candidate_pairs_mismatch")

            prompt_text = json.dumps(prompt_messages_from_row(row), sort_keys=True)
            for leak in PROMPT_LEAK_TERMS:
                if leak in prompt_text:
                    issues.append(f"{row_id}:{split}_prompt_leaks_{leak}")
            for source_key in ("source_task_id", "split_group", "case_family"):
                value = row.get(source_key)
                if isinstance(value, str) and value and value in prompt_text:
                    issues.append(f"{row_id}:{split}_prompt_leaks_{source_key}_value")
    return sorted(set(issues))


def dry_run_report(
    *,
    model: str,
    rows_path: str | Path,
    train_rows_path: str | Path,
    heldout_rows_path: str | Path,
    manifest_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    issues: Sequence[str],
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "dry_run": True,
        "model": model,
        "rows": str(rows_path),
        "train_rows": str(train_rows_path),
        "heldout_rows": str(heldout_rows_path),
        "manifest": str(manifest_path),
        "row_dataset": ROW_DATASET,
        "prompt_contract": PROMPT_CONTRACT,
        "candidate_pairs": list(CANDIDATE_PAIRS),
        "candidate_space_size": len(CANDIDATE_PAIRS),
        "train_examples": len(train_rows),
        "heldout_examples": len(heldout_rows),
        "bridge_focus_heldout_examples": sum(
            1 for row in heldout_rows if row.get("bridge_focus_case")
        ),
        "train_by_target_pair": row_value_counts(train_rows, "target_pair"),
        "heldout_by_target_pair": row_value_counts(heldout_rows, "target_pair"),
        "train_case_ids": [source_case_id(row) for row in train_rows],
        "heldout_case_ids": [source_case_id(row) for row in heldout_rows],
        "issues": list(issues),
        "ready_for_full_mode": not issues,
        "full_mode_requires_allow_model_load": True,
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
            "hidden_labels_used_for_training": False,
        },
        "boundary": (
            "Dry run validates candidate-routing artifacts and split boundaries "
            "without loading model weights or running local heavy compute."
        ),
    }


def score_candidate_outputs(
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    device: str,
    max_length: int,
) -> dict[str, Any]:
    prompt = prompt_text_for_tokenizer(tokenizer, prompt_messages_from_row(row))
    scored: list[dict[str, Any]] = []
    for candidate in candidate_outputs_from_row(row):
        score = score_candidate_target(
            model,
            tokenizer,
            prompt,
            json.dumps(candidate, sort_keys=True),
            device=device,
            max_length=max_length,
        )
        scored.append({"score": score, "candidate": candidate})
    scored.sort(key=lambda item: item["score"], reverse=True)
    winner = scored[0]["candidate"]
    return {
        "prediction": winner,
        "raw_output": json.dumps(winner, sort_keys=True),
        "candidate_scores": scored,
    }


def prediction_row(
    *,
    row: Mapping[str, Any],
    run_id: str,
    model_id: str,
    split: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"{run_id}::{row['id']}",
        "dataset": PREDICTION_DATASET,
        "source": "stage_a_saved_output_evidence_candidate_routing_smoke",
        "run_id": run_id,
        "model": model_id,
        "source_candidate_routing_id": row["id"],
        "case_id": source_case_id(row),
        "split": split,
        "bridge_focus_case": bool(row.get("bridge_focus_case", False)),
        "target_pair": row.get("target_pair"),
        "prediction": dict(result["prediction"]),
        "raw_output": result["raw_output"],
        "candidate_scores": list(result["candidate_scores"]),
    }


def build_candidate_prediction_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    model_id: str,
    split: str,
    device: str,
    max_length: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        result = score_candidate_outputs(
            model,
            tokenizer,
            row,
            device=device,
            max_length=max_length,
        )
        predictions.append(
            prediction_row(
                row=row,
                run_id=run_id,
                model_id=model_id,
                split=split,
                result=result,
            )
        )
        print(f"[{split} {index + 1}/{len(rows)}] scored {row['id']}", flush=True)
    return predictions


def score_prediction_row(expected: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    pred = prediction.get("prediction")
    if not isinstance(pred, Mapping):
        return failed_eval_row(expected, prediction, ["prediction_not_object"])
    target = target_output_from_row(expected)
    violations: list[str] = []
    reward = {
        "selected_pair": float(pred.get("selected_pair") == target["selected_pair"]),
        "action": float(pred.get("action") == target["action"]),
        "evidence_status": float(pred.get("evidence_status") == target["evidence_status"]),
    }
    for key, ok in reward.items():
        if not ok:
            violations.append(f"{key}_mismatch")
    exact = not violations
    return {
        "id": prediction.get("id"),
        "case_id": source_case_id(expected),
        "split": expected.get("split"),
        "bridge_focus_case": bool(expected.get("bridge_focus_case", False)),
        "target_pair": target["selected_pair"],
        "predicted_pair": pair_for_output(pred),
        "score": round(sum(reward.values()) / len(reward), 3),
        "passed": exact,
        "reward_breakdown": reward,
        "violations": violations,
    }


def failed_eval_row(
    expected: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    violations: Sequence[str],
) -> dict[str, Any]:
    return {
        "id": prediction.get("id") if prediction else None,
        "case_id": source_case_id(expected),
        "split": expected.get("split"),
        "bridge_focus_case": bool(expected.get("bridge_focus_case", False)),
        "target_pair": expected.get("target_pair"),
        "predicted_pair": None,
        "score": 0.0,
        "passed": False,
        "reward_breakdown": {},
        "violations": list(violations),
    }


def summarize_eval_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "exact": 0,
            "accuracy": 0.0,
            "bridge_focus_rows": 0,
            "bridge_focus_exact": 0,
            "bridge_focus_accuracy": None,
            "violations": {},
        }
    exact = sum(1 for row in rows if row.get("passed"))
    bridge_rows = [row for row in rows if row.get("bridge_focus_case")]
    bridge_exact = sum(1 for row in bridge_rows if row.get("passed"))
    violations = Counter(violation for row in rows for violation in row.get("violations", ()))
    return {
        "rows": len(rows),
        "exact": exact,
        "accuracy": round(exact / len(rows), 6),
        "bridge_focus_rows": len(bridge_rows),
        "bridge_focus_exact": bridge_exact,
        "bridge_focus_accuracy": round(bridge_exact / len(bridge_rows), 6)
        if bridge_rows
        else None,
        "mean_score": round(sum(float(row.get("score", 0.0)) for row in rows) / len(rows), 6),
        "by_target_pair": dict(sorted(Counter(str(row.get("target_pair")) for row in rows).items())),
        "by_predicted_pair": dict(
            sorted(Counter(str(row.get("predicted_pair")) for row in rows).items())
        ),
        "violations": dict(sorted(violations.items())),
        "error_case_ids": [str(row.get("case_id")) for row in rows if not row.get("passed")],
        "bridge_focus_error_case_ids": [
            str(row.get("case_id"))
            for row in bridge_rows
            if not row.get("passed")
        ],
    }


def build_eval_report(
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    split: str,
    run_id: str,
) -> dict[str, Any]:
    expected_by_id = {str(row["id"]): row for row in expected_rows}
    predictions_by_id = {
        str(row.get("source_candidate_routing_id") or row.get("id")): row
        for row in prediction_rows
    }
    row_reports: list[dict[str, Any]] = []
    for row_id, expected in expected_by_id.items():
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            row_reports.append(failed_eval_row(expected, None, ["missing_prediction"]))
        else:
            row_reports.append(score_prediction_row(expected, prediction))
    for row_id in sorted(set(predictions_by_id) - set(expected_by_id)):
        row_reports.append(
            {
                "id": row_id,
                "case_id": None,
                "split": split,
                "bridge_focus_case": False,
                "target_pair": None,
                "predicted_pair": None,
                "score": 0.0,
                "passed": False,
                "reward_breakdown": {},
                "violations": ["unexpected_prediction_id"],
            }
        )
    return {
        "dataset": DATASET,
        "run_id": run_id,
        "split": split,
        "boundary": "Offline finite-candidate scorer; no live API calls.",
        "rows_expected": len(expected_rows),
        "predictions_received": len(prediction_rows),
        "summary": summarize_eval_rows(row_reports),
        "rows": row_reports,
    }


def run_training_and_eval(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    disable_transformers_torchvision_probe()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
    )
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

    train_predictions = build_candidate_prediction_rows(
        model,
        tokenizer,
        train_rows,
        run_id=args.run_id,
        model_id=args.model,
        split="train",
        device=device,
        max_length=args.max_length,
    )
    heldout_predictions = build_candidate_prediction_rows(
        model,
        tokenizer,
        heldout_rows,
        run_id=args.run_id,
        model_id=args.model,
        split="heldout",
        device=device,
        max_length=args.max_length,
    )
    write_jsonl(args.train_candidates_out, train_predictions)
    write_jsonl(args.heldout_candidates_out, heldout_predictions)

    train_eval = build_eval_report(
        expected_rows=train_rows,
        prediction_rows=train_predictions,
        split="train",
        run_id=args.run_id,
    )
    heldout_eval = build_eval_report(
        expected_rows=heldout_rows,
        prediction_rows=heldout_predictions,
        split="heldout",
        run_id=args.run_id,
    )
    eval_report = {
        "dataset": DATASET,
        "run_id": args.run_id,
        "train": train_eval,
        "heldout": heldout_eval,
        "acceptance_gate": {
            "heldout_exact_min": 5,
            "bridge_focus_exact_min": 4,
            "passes": heldout_eval["summary"]["exact"] == 5
            and heldout_eval["summary"]["bridge_focus_exact"] == 4,
        },
    }
    write_json(args.eval_out, eval_report)

    return {
        "dataset": DATASET,
        "dry_run": False,
        "run_id": args.run_id,
        "model": args.model,
        "device": device,
        "row_dataset": ROW_DATASET,
        "prompt_contract": PROMPT_CONTRACT,
        "candidate_pairs": list(CANDIDATE_PAIRS),
        "candidate_space_size": len(CANDIDATE_PAIRS),
        "train_examples": len(train_rows),
        "heldout_examples": len(heldout_rows),
        "bridge_focus_heldout_examples": sum(
            1 for row in heldout_rows if row.get("bridge_focus_case")
        ),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_last_layers": args.train_last_layers,
        "trainable_params": trainable_params,
        "losses": losses,
        "loss_delta": losses[-1] - losses[0] if len(losses) > 1 else 0.0,
        "trainable_state": str(state_path) if state_path else None,
        "train_candidates": str(args.train_candidates_out),
        "heldout_candidates": str(args.heldout_candidates_out),
        "eval_report": str(args.eval_out),
        "train_summary": train_eval["summary"],
        "heldout_summary": heldout_eval["summary"],
        "acceptance_gate": eval_report["acceptance_gate"],
        "boundary": (
            "Full mode writes raw candidate scores under the configured run "
            "directory. Curate only compact summaries into public artifacts."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--train-rows", default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--heldout-rows", default=DEFAULT_HELDOUT_ROWS)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out-dir",
        default="post_training/runs/stage_a_saved_output_evidence_candidate_routing_smoke",
    )
    parser.add_argument("--run-id", default="stage_a_saved_output_evidence_candidate_routing_smoke")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-trainable-state", action="store_true")
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--train-candidates-out", default=None)
    parser.add_argument("--heldout-candidates-out", default=None)
    parser.add_argument("--eval-out", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report_out is None:
        args.report_out = str(out_dir / "report.json")
    if args.train_candidates_out is None:
        args.train_candidates_out = str(out_dir / "train_candidates.jsonl")
    if args.heldout_candidates_out is None:
        args.heldout_candidates_out = str(out_dir / "heldout_candidates.jsonl")
    if args.eval_out is None:
        args.eval_out = str(out_dir / "eval_report.json")

    rows = load_jsonl(args.rows)
    train_rows = load_jsonl(args.train_rows)
    heldout_rows = load_jsonl(args.heldout_rows)
    manifest = load_json(args.manifest)
    issues = validate_candidate_routing_rows(rows, train_rows, heldout_rows, manifest)
    if issues:
        write_json(
            args.report_out,
            dry_run_report(
                model=args.model,
                rows_path=args.rows,
                train_rows_path=args.train_rows,
                heldout_rows_path=args.heldout_rows,
                manifest_path=args.manifest,
                rows=rows,
                train_rows=train_rows,
                heldout_rows=heldout_rows,
                issues=issues,
            ),
        )
        raise SystemExit(
            "Evidence candidate-routing smoke validation failed:\n- "
            + "\n- ".join(issues)
        )

    if args.dry_run:
        report = dry_run_report(
            model=args.model,
            rows_path=args.rows,
            train_rows_path=args.train_rows,
            heldout_rows_path=args.heldout_rows,
            manifest_path=args.manifest,
            rows=rows,
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            issues=issues,
        )
    else:
        if not args.allow_model_load:
            raise SystemExit(
                "Full evidence candidate-routing smoke requires --allow-model-load. "
                "Use --dry-run for local validation."
            )
        report = run_training_and_eval(args, train_rows, heldout_rows)
    write_json(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
