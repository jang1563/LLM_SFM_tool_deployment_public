#!/usr/bin/env python3
"""Run constrained finite-candidate readouts for Stage A saved predictions.

This is the follow-up to the v3/v4 free-form saved-output failures. Instead of
letting a model generate an arbitrary JSON envelope, this script scores a finite
set of canonical compact prediction candidates and writes the selected saved
predictions for offline trajectory evaluation.

The dry-run path uses oracle action/status pairs with the same visible-citation
policy. It loads no model weights and is intended for public CI. Full mode
requires --allow-model-load and should run on Cayuga/Expanse.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.evaluate_stage_a_predictions import (  # noqa: E402
    build_report as build_prediction_eval_report,
    expected_case_ids_from_rows,
)
from post_training.generate_stage_a_predictions import (  # noqa: E402
    DEFAULT_HF_MODEL,
    PROMPT_CONTRACTS,
    api_prompt_messages,
    generation_prompt_hash,
    get_hf_chat_client,
    prompt_hash,
    source_case_id,
)
from post_training.run_stage_a_sft_smoke_eval import (  # noqa: E402
    load_jsonl,
    load_manifest_rows,
    write_json,
)
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    prompt_text_for_tokenizer,
    score_candidate_target,
    write_jsonl,
)


DATASET = "negbiodb_ct_stage_a_saved_prediction_candidate_readout_v1"
PREDICTION_DATASET = "negbiodb_ct_stage_a_saved_predictions_v1"
EXTERNAL_ACTIONS = ("ground", "reject", "defer", "verify", "flag")
EXTERNAL_EVIDENCE_STATUSES = (
    "supported",
    "contradicted",
    "invalid_value",
    "insufficient",
)
REQUIRED_TOOL_TRACE = (
    "nullatlas_survey_prior_failures",
    "nullatlas_verify_trial_claims",
    "nullatlas_check_value_validity",
    "nullatlas_negative_evidence_completeness",
)
TERMINAL_TO_ACTION = {
    "ground_with_attribution": "ground",
    "defer_or_request_more_evidence": "defer",
    "verify_with_assay_or_database": "verify",
}


def target_pair_from_sft_row(row: Mapping[str, Any]) -> dict[str, str]:
    trajectory = row.get("target_trajectory")
    if not isinstance(trajectory, Mapping):
        raise ValueError(f"{source_case_id(row)} is missing target_trajectory")
    status = str(trajectory.get("predicted_evidence_status"))
    terminal = str(trajectory.get("terminal_action"))
    if terminal == "reject_or_flag_unsupported_claim":
        action = "flag" if status == "invalid_value" else "reject"
    else:
        try:
            action = TERMINAL_TO_ACTION[terminal]
        except KeyError as exc:
            raise ValueError(f"{source_case_id(row)} has unsupported terminal action: {terminal}") from exc
    return {"action": action, "evidence_status": status}


def all_valid_pairs() -> list[dict[str, str]]:
    return [
        {"action": action, "evidence_status": evidence_status}
        for action in EXTERNAL_ACTIONS
        for evidence_status in EXTERNAL_EVIDENCE_STATUSES
    ]


def train_observed_pairs(train_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in train_rows:
        pair = target_pair_from_sft_row(row)
        key = (pair["action"], pair["evidence_status"])
        if key not in seen:
            seen.add(key)
            out.append(pair)
    return sorted(out, key=lambda item: (item["action"], item["evidence_status"]))


def candidate_pairs(
    *,
    policy: str,
    train_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if policy == "all_valid_pairs":
        return all_valid_pairs()
    if policy == "train_observed_pairs":
        pairs = train_observed_pairs(train_rows)
        if not pairs:
            raise ValueError("train_observed_pairs policy produced no candidates")
        return pairs
    raise ValueError(f"Unknown candidate policy: {policy}")


def visible_source_ids(row: Mapping[str, Any], *, prompt_contract: str) -> list[str]:
    text = json.dumps(api_prompt_messages(row, prompt_contract=prompt_contract), sort_keys=True)
    out: list[str] = []
    seen: set[str] = set()
    for source_id in re.findall(r"NCT\d+", text):
        if source_id not in seen:
            seen.add(source_id)
            out.append(source_id)
    return out


def full_tool_trace() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "arguments": {
                "drug_id": "<drug_id>",
                "condition_id": "<condition_id>",
            },
        }
        for name in REQUIRED_TOOL_TRACE
    ]


def candidate_prediction_for_row(
    row: Mapping[str, Any],
    pair: Mapping[str, str],
    *,
    prompt_contract: str,
) -> dict[str, Any]:
    action = str(pair["action"])
    evidence_status = str(pair["evidence_status"])
    citations: list[str] = []
    if (action, evidence_status) in {
        ("ground", "supported"),
        ("flag", "invalid_value"),
    }:
        citations = visible_source_ids(row, prompt_contract=prompt_contract)[:1]
    return {
        "action": action,
        "evidence_status": evidence_status,
        "tool_calls": full_tool_trace(),
        "cited_source_ids": citations,
        "rationale": "Finite-candidate constrained Stage A readout.",
    }


def candidate_key(candidate: Mapping[str, Any]) -> tuple[str | None, str | None]:
    return (
        str(candidate.get("action")) if candidate.get("action") is not None else None,
        str(candidate.get("evidence_status")) if candidate.get("evidence_status") is not None else None,
    )


def candidate_score_rank(
    candidate_scores: Sequence[Mapping[str, Any]],
    target_pair: Mapping[str, str],
) -> tuple[int | None, float | None]:
    target_key = candidate_key(target_pair)
    for index, item in enumerate(candidate_scores, start=1):
        candidate = item.get("candidate")
        if isinstance(candidate, Mapping) and candidate_key(candidate) == target_key:
            return index, float(item["score"])
    return None, None


def score_candidates_for_row(
    *,
    client: Any,
    row: Mapping[str, Any],
    pairs: Sequence[Mapping[str, str]],
    prompt_contract: str,
    max_length: int,
) -> dict[str, Any]:
    prompt = prompt_text_for_tokenizer(
        client.tokenizer,
        api_prompt_messages(row, prompt_contract=prompt_contract),
    )
    scored: list[dict[str, Any]] = []
    for pair in pairs:
        candidate = candidate_prediction_for_row(row, pair, prompt_contract=prompt_contract)
        score = score_candidate_target(
            client.model,
            client.tokenizer,
            prompt,
            json.dumps(candidate, sort_keys=True),
            device=client.device,
            max_length=max_length,
        )
        scored.append({"score": score, "candidate": candidate})
    scored.sort(key=lambda item: (item["score"], json.dumps(item["candidate"], sort_keys=True)), reverse=True)
    return {
        "prediction": dict(scored[0]["candidate"]),
        "candidate_scores": scored,
    }


def base_prediction_row(
    row: Mapping[str, Any],
    *,
    run_id: str,
    source: str,
    prompt_contract: str,
    model: str | None,
    candidate_policy: str,
    candidate_space_size: int,
) -> dict[str, Any]:
    return {
        "case_id": source_case_id(row),
        "dataset": PREDICTION_DATASET,
        "source": source,
        "run_id": run_id,
        "split": row.get("split"),
        "case_family": row.get("case_family"),
        "prompt_hash": prompt_hash(row),
        "generation_prompt_hash": generation_prompt_hash(row, prompt_contract=prompt_contract),
        "prompt_contract": prompt_contract,
        "model": model,
        "candidate_policy": candidate_policy,
        "candidate_space_size": candidate_space_size,
    }


def build_prediction_rows(
    *,
    heldout_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    run_id: str,
    prompt_contract: str,
    candidate_policy: str,
    model: str | None,
    client: Any | None,
    max_length: int,
) -> list[dict[str, Any]]:
    pairs = candidate_pairs(policy=candidate_policy, train_rows=train_rows)
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(heldout_rows):
        target_pair = target_pair_from_sft_row(row)
        if client is None:
            prediction = candidate_prediction_for_row(row, target_pair, prompt_contract=prompt_contract)
            candidate_scores: list[dict[str, Any]] = []
            source = "stage_a_saved_prediction_candidate_readout::oracle_pair_dry_run"
        else:
            result = score_candidates_for_row(
                client=client,
                row=row,
                pairs=pairs,
                prompt_contract=prompt_contract,
                max_length=max_length,
            )
            prediction = result["prediction"]
            candidate_scores = result["candidate_scores"]
            source = "stage_a_saved_prediction_candidate_readout::hf_candidate_score"
        rank, target_score = candidate_score_rank(candidate_scores, target_pair)
        out = base_prediction_row(
            row,
            run_id=run_id,
            source=source,
            prompt_contract=prompt_contract,
            model=model,
            candidate_policy=candidate_policy,
            candidate_space_size=len(pairs),
        )
        out.update(
            {
                "prediction": prediction,
                "raw_output": json.dumps(prediction, sort_keys=True),
                "target_pair": target_pair,
                "top_pair": {
                    "action": prediction.get("action"),
                    "evidence_status": prediction.get("evidence_status"),
                },
                "target_pair_rank": rank,
                "target_pair_score": target_score,
            }
        )
        if candidate_scores:
            out["candidate_scores"] = candidate_scores
        predictions.append(out)
        print(f"[{index + 1}/{len(heldout_rows)}] scored {source_case_id(row)}", flush=True)
    return predictions


def summarize_candidate_readout(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(predictions)
    exact = 0
    target_ranks: list[int] = []
    top_pairs: dict[str, int] = {}
    for row in rows:
        target = row.get("target_pair")
        top = row.get("top_pair")
        if isinstance(target, Mapping) and isinstance(top, Mapping) and candidate_key(target) == candidate_key(top):
            exact += 1
        rank = row.get("target_pair_rank")
        if isinstance(rank, int):
            target_ranks.append(rank)
        if isinstance(top, Mapping):
            label = f"{top.get('action')}/{top.get('evidence_status')}"
            top_pairs[label] = top_pairs.get(label, 0) + 1
    return {
        "cases": len(rows),
        "exact_pair_top1": exact,
        "mean_target_pair_rank": round(sum(target_ranks) / len(target_ranks), 3) if target_ranks else None,
        "top_pairs": dict(sorted(top_pairs.items())),
        "candidate_scores_present": sum(1 for row in rows if row.get("candidate_scores")),
    }


def build_report(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    run_id: str,
    prompt_contract: str,
    candidate_policy: str,
    model: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    eval_report = build_prediction_eval_report(
        manifest_rows=manifest_rows,
        prediction_rows=prediction_rows,
        expected_case_ids=expected_case_ids_from_rows(heldout_rows),
        run_id=run_id,
    )
    return {
        "dataset": DATASET,
        "run_id": run_id,
        "model": model,
        "dry_run": dry_run,
        "prompt_contract": prompt_contract,
        "candidate_policy": candidate_policy,
        "candidate_outputs": candidate_pairs(policy=candidate_policy, train_rows=train_rows),
        "candidate_space_size": len(candidate_pairs(policy=candidate_policy, train_rows=train_rows)),
        "heldout_rows": len(heldout_rows),
        "raw_predictions_committed": False,
        "eval_summary": eval_report["summary"],
        "parse_errors": eval_report["parse_errors"],
        "candidate_readout_summary": summarize_candidate_readout(prediction_rows),
        "boundary": (
            "Finite-candidate saved-prediction readout. Full mode scores "
            "canonical compact candidates with a model; dry-run mode validates "
            "candidate construction and visible-citation evaluation without "
            "loading model weights."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sft", default="post_training/stage_a_sft_train_v1.jsonl")
    parser.add_argument("--heldout-sft", default="post_training/stage_a_sft_heldout_v1.jsonl")
    parser.add_argument("--manifest", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument("--out-dir", default="post_training/runs/stage_a_saved_prediction_candidate_readout")
    parser.add_argument("--run-id", default="stage_a_saved_prediction_candidate_readout")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--prompt-contract", choices=PROMPT_CONTRACTS, default="stage_a_v4_canonical_json")
    parser.add_argument(
        "--candidate-policy",
        choices=("train_observed_pairs", "all_valid_pairs"),
        default="train_observed_pairs",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument("--eval-out", default=None)
    parser.add_argument("--report-out", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.dry_run and not args.allow_model_load:
        raise RuntimeError("Full candidate readout requires --allow-model-load; use --dry-run for no-model CI.")

    train_rows = load_jsonl(args.train_sft)
    heldout_rows = load_jsonl(args.heldout_sft)
    manifest_rows = load_manifest_rows(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(args.predictions_out) if args.predictions_out else out_dir / "predictions.jsonl"
    eval_path = Path(args.eval_out) if args.eval_out else out_dir / "eval_report.json"
    report_path = Path(args.report_out) if args.report_out else out_dir / "report.json"

    client = None
    model_id: str | None = None
    if not args.dry_run:
        client = get_hf_chat_client(
            model=args.model,
            allow_model_load=args.allow_model_load,
            device=args.device,
            max_new_tokens=1,
            local_files_only=not args.allow_download,
        )
        model_id = client.model_id

    predictions = build_prediction_rows(
        heldout_rows=heldout_rows,
        train_rows=train_rows,
        run_id=args.run_id,
        prompt_contract=args.prompt_contract,
        candidate_policy=args.candidate_policy,
        model=model_id,
        client=client,
        max_length=args.max_length,
    )
    write_jsonl(predictions_path, predictions)
    eval_report = build_prediction_eval_report(
        manifest_rows=manifest_rows,
        prediction_rows=predictions,
        expected_case_ids=expected_case_ids_from_rows(heldout_rows),
        run_id=args.run_id,
    )
    write_json(eval_path, eval_report)
    report = build_report(
        manifest_rows=manifest_rows,
        heldout_rows=heldout_rows,
        train_rows=train_rows,
        prediction_rows=predictions,
        run_id=args.run_id,
        prompt_contract=args.prompt_contract,
        candidate_policy=args.candidate_policy,
        model=model_id,
        dry_run=args.dry_run,
    )
    report["predictions"] = str(predictions_path)
    report["eval_report"] = str(eval_path)
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
