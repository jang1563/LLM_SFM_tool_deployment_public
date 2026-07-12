#!/usr/bin/env python3
"""Export strict-contract Stage A post-training targets.

These artifacts are built for the `stage_a_v2_strict` saved-prediction prompt
contract. They keep the model-visible prompt identical to the cluster/API
producer prompt and target one compact JSON object containing the intended tool
loop, evidence status, citations, and terminal action.
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

from post_training.evaluate_stage_a_predictions import build_report
from post_training.generate_stage_a_predictions import (
    api_prompt_messages,
    generation_prompt_hash,
)
from post_training.run_stage_a_sft_smoke_eval import load_manifest_rows


STRICT_PROMPT_CONTRACT = "stage_a_v2_strict"
STRICT_SFT_DATASET = "negbiodb_ct_stage_a_strict_contract_sft_v1"
STRICT_PREFERENCE_DATASET = "negbiodb_ct_stage_a_strict_contract_preferences_v1"
STRICT_PROCESS_DATASET = "negbiodb_ct_stage_a_strict_contract_process_v1"
STRICT_MANIFEST_DATASET = "negbiodb_ct_stage_a_strict_contract_export_v1"
STRICT_EXPORT_STRATEGY = "strict_prompt_contract_targets_v1"
STRICT_FAILURE_MODES = (
    "observed_single_tool_verify_supported",
    "full_loop_verify_supported",
)
ACTION_TO_STRICT = {
    "ground_with_attribution": "ground",
    "verify_with_assay_or_database": "verify",
    "defer_or_request_more_evidence": "defer",
    "answer_self": "self_answer",
    "use_cheap_baseline": "verify",
    "trust_specialist_output": "verify",
}
REQUIRED_OUTPUT_KEYS = (
    "action",
    "evidence_status",
    "tool_calls",
    "cited_source_ids",
    "rationale",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def source_case_id(row: Mapping[str, Any]) -> str:
    value = row.get("source_manifest_case_id") or row.get("task_id") or row.get("case_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing Stage A case id in row {row.get('id')!r}")
    return value


def strict_action_from_trajectory(trajectory: Mapping[str, Any]) -> str:
    terminal = str(trajectory.get("terminal_action", ""))
    status = str(trajectory.get("predicted_evidence_status", ""))
    if terminal == "reject_or_flag_unsupported_claim":
        return "flag" if status == "invalid_value" else "reject"
    try:
        return ACTION_TO_STRICT[terminal]
    except KeyError as exc:
        raise ValueError(f"Unsupported terminal action for strict contract: {terminal!r}") from exc


def compact_output_from_target(row: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = row.get("target_trajectory")
    if not isinstance(trajectory, Mapping):
        raise ValueError(f"{source_case_id(row)} is missing target_trajectory")
    return {
        "action": strict_action_from_trajectory(trajectory),
        "evidence_status": str(trajectory.get("predicted_evidence_status")),
        "tool_calls": [
            {
                "name": str(step["name"]),
                "arguments": dict(step.get("arguments", {})),
            }
            for step in trajectory.get("steps", ())
            if isinstance(step, Mapping)
        ],
        "cited_source_ids": list(trajectory.get("cited_source_ids", ())),
        "rationale": rationale_for_row(row),
    }


def rationale_for_row(row: Mapping[str, Any]) -> str:
    status = str(row.get("gold_evidence_status"))
    action = str(row.get("expected_terminal_action"))
    return (
        "The required Stage A tool loop was completed, the evidence packet was "
        f"classified as {status}, and the terminal action is {action}."
    )


def observed_single_tool_verify_supported(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action": "verify",
        "evidence_status": "supported",
        "tool_calls": [
            {
                "name": "nullatlas_verify_trial_claims",
                "arguments": {
                    "drug_id": "<drug_id>",
                    "condition_id": "<condition_id>",
                },
            }
        ],
        "cited_source_ids": [],
        "rationale": (
            "Observed strict-prompt collapse: one verification-like tool call, "
            "supported evidence, and verify action."
        ),
    }


def full_loop_verify_supported(row: Mapping[str, Any]) -> dict[str, Any]:
    output = compact_output_from_target(row)
    output.update(
        {
            "action": "verify",
            "evidence_status": "supported",
            "cited_source_ids": [],
            "rationale": (
                "Observed routing collapse isolated from tool-loop validity: "
                "full tool sequence but supported/verify final routing."
            ),
        }
    )
    return output


def rejected_output(row: Mapping[str, Any], failure_mode: str) -> dict[str, Any]:
    if failure_mode == "observed_single_tool_verify_supported":
        return observed_single_tool_verify_supported(row)
    if failure_mode == "full_loop_verify_supported":
        return full_loop_verify_supported(row)
    raise ValueError(f"Unknown strict-contract failure mode: {failure_mode}")


def assistant_json_message(output: Mapping[str, Any]) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": json.dumps(output, sort_keys=True),
    }


def prediction_row(case_id: str, output: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "dataset": "negbiodb_ct_stage_a_strict_contract_prediction_target_v1",
        "source": source,
        "prediction": dict(output),
    }


def score_output(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    output: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    report = build_report(
        manifest_rows=manifest_rows,
        prediction_rows=[prediction_row(case_id, output, source=source)],
        expected_case_ids=[case_id],
        run_id=source,
    )
    row = report["rows"][0]
    return {
        "passed": bool(row["passed"]),
        "score": row["score"],
        "reward_breakdown": dict(row.get("reward_breakdown", {})),
        "violations": list(row.get("violations", ())),
        "parse_error": row.get("parse_error"),
    }


def row_common(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_manifest_case_id": source_case_id(row),
        "source_task_id": row.get("source_task_id"),
        "task_id": row.get("task_id"),
        "tool_profile": row.get("tool_profile"),
        "case_family": row.get("case_family"),
        "gold_evidence_status": row.get("gold_evidence_status"),
        "expected_terminal_action": row.get("expected_terminal_action"),
        "split_group": row.get("split_group"),
        "prompt_contract": STRICT_PROMPT_CONTRACT,
        "generation_prompt_hash": generation_prompt_hash(
            row,
            prompt_contract=STRICT_PROMPT_CONTRACT,
        ),
    }


def sft_row(
    row: Mapping[str, Any],
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    case_id = source_case_id(row)
    output = compact_output_from_target(row)
    score = score_output(
        manifest_rows,
        case_id=case_id,
        output=output,
        source="strict_contract_chosen_target",
    )
    if not score["passed"]:
        raise ValueError(f"Strict-contract chosen output does not pass: {case_id}")
    return {
        "id": f"stage_a_strict_sft::{case_id}",
        "dataset": STRICT_SFT_DATASET,
        "oracle_target": True,
        **row_common(row),
        "messages": api_prompt_messages(row, prompt_contract=STRICT_PROMPT_CONTRACT)
        + [assistant_json_message(output)],
        "target_output": output,
        "target_trajectory": row.get("target_trajectory"),
        "score": score,
    }


def preference_rows(
    row: Mapping[str, Any],
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    case_id = source_case_id(row)
    chosen = compact_output_from_target(row)
    chosen_score = score_output(
        manifest_rows,
        case_id=case_id,
        output=chosen,
        source="strict_contract_chosen_target",
    )
    if not chosen_score["passed"]:
        raise ValueError(f"Strict-contract chosen output does not pass: {case_id}")

    out: list[dict[str, Any]] = []
    for failure_mode in STRICT_FAILURE_MODES:
        rejected = rejected_output(row, failure_mode)
        rejected_score = score_output(
            manifest_rows,
            case_id=case_id,
            output=rejected,
            source=f"strict_contract_rejected::{failure_mode}",
        )
        if rejected_score["passed"]:
            raise ValueError(f"Strict-contract rejected output unexpectedly passes: {case_id} {failure_mode}")
        out.append(
            {
                "id": f"stage_a_strict_pref::{case_id}::{failure_mode}",
                "dataset": STRICT_PREFERENCE_DATASET,
                **row_common(row),
                "strategy": STRICT_EXPORT_STRATEGY,
                "failure_mode": failure_mode,
                "prompt_messages": api_prompt_messages(row, prompt_contract=STRICT_PROMPT_CONTRACT),
                "chosen": json.dumps(chosen, sort_keys=True),
                "rejected": json.dumps(rejected, sort_keys=True),
                "chosen_messages": [assistant_json_message(chosen)],
                "rejected_messages": [assistant_json_message(rejected)],
                "chosen_output": chosen,
                "rejected_output": rejected,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
            }
        )
    return out


def process_row(
    row: Mapping[str, Any],
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    case_id = source_case_id(row)
    output = compact_output_from_target(row)
    score = score_output(
        manifest_rows,
        case_id=case_id,
        output=output,
        source="strict_contract_process_target",
    )
    if not score["passed"]:
        raise ValueError(f"Strict-contract process output does not pass: {case_id}")
    return {
        "id": f"stage_a_strict_process::{case_id}",
        "dataset": STRICT_PROCESS_DATASET,
        **row_common(row),
        "prompt_messages": api_prompt_messages(row, prompt_contract=STRICT_PROMPT_CONTRACT),
        "target_process": [
            {"name": "prompt_contract", "value": STRICT_PROMPT_CONTRACT},
            {"name": "required_output_keys", "value": list(REQUIRED_OUTPUT_KEYS)},
            {
                "name": "required_tool_sequence",
                "value": [call["name"] for call in output["tool_calls"]],
            },
            {"name": "required_query_fields", "value": ["drug_id", "condition_id"]},
            {"name": "evidence_status", "value": output["evidence_status"]},
            {"name": "terminal_action", "value": output["action"]},
            {"name": "cited_source_ids", "value": list(output["cited_source_ids"])},
        ],
        "target_output": output,
        "target_trajectory": row.get("target_trajectory"),
        "score": score,
    }


def build_strict_contract_exports(
    sft_source_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sft = [sft_row(row, manifest_rows=manifest_rows) for row in sft_source_rows]
    preferences: list[dict[str, Any]] = []
    process = [process_row(row, manifest_rows=manifest_rows) for row in sft_source_rows]
    for row in sft_source_rows:
        preferences.extend(preference_rows(row, manifest_rows=manifest_rows))
    return sft, preferences, process


def split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_case_ids: set[str],
    heldout_case_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row["source_manifest_case_id"])
        copied = dict(row)
        if case_id in train_case_ids:
            copied["split"] = "train"
            train.append(copied)
        elif case_id in heldout_case_ids:
            copied["split"] = "heldout"
            heldout.append(copied)
        else:
            raise ValueError(f"Case id {case_id!r} is absent from split manifest")
    return train, heldout


def apply_split_manifest(
    sft: Sequence[Mapping[str, Any]],
    preferences: Sequence[Mapping[str, Any]],
    process: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    train_case_ids = set(str(value) for value in split_manifest["train_case_ids"])
    heldout_case_ids = set(str(value) for value in split_manifest["heldout_case_ids"])
    if train_case_ids & heldout_case_ids:
        raise ValueError("Stage A strict split manifest has overlapping case ids")
    train_sft, heldout_sft = split_rows(sft, train_case_ids=train_case_ids, heldout_case_ids=heldout_case_ids)
    train_preferences, heldout_preferences = split_rows(
        preferences,
        train_case_ids=train_case_ids,
        heldout_case_ids=heldout_case_ids,
    )
    train_process, heldout_process = split_rows(
        process,
        train_case_ids=train_case_ids,
        heldout_case_ids=heldout_case_ids,
    )
    return {
        "train_sft": train_sft,
        "heldout_sft": heldout_sft,
        "train_preferences": train_preferences,
        "heldout_preferences": heldout_preferences,
        "train_process": train_process,
        "heldout_process": heldout_process,
    }


def manifest_for_exports(
    *,
    source_sft: str | Path,
    source_manifest: str | Path,
    source_split_manifest: str | Path,
    sft_out: str | Path,
    preference_out: str | Path,
    process_out: str | Path,
    split_paths: Mapping[str, str | Path],
    sft_rows: Sequence[Mapping[str, Any]],
    preference_rows_out: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    train_sft = list(splits["train_sft"])
    heldout_sft = list(splits["heldout_sft"])
    train_preferences = list(splits["train_preferences"])
    heldout_preferences = list(splits["heldout_preferences"])
    train_process = list(splits["train_process"])
    heldout_process = list(splits["heldout_process"])
    train_case_ids = sorted(case_ids(train_sft))
    heldout_case_ids = sorted(case_ids(heldout_sft))
    train_split_groups = sorted(value_set(train_sft, "split_group"))
    heldout_split_groups = sorted(value_set(heldout_sft, "split_group"))
    train_source_task_ids = sorted(value_set(train_sft, "source_task_id"))
    heldout_source_task_ids = sorted(value_set(heldout_sft, "source_task_id"))

    return {
        "dataset": STRICT_MANIFEST_DATASET,
        "strategy": STRICT_EXPORT_STRATEGY,
        "prompt_contract": STRICT_PROMPT_CONTRACT,
        "source_sft": display_path(source_sft),
        "source_manifest": display_path(source_manifest),
        "source_split_manifest": display_path(source_split_manifest),
        "sft": display_path(sft_out),
        "preferences": display_path(preference_out),
        "process": display_path(process_out),
        "sft_dataset": STRICT_SFT_DATASET,
        "preference_dataset": STRICT_PREFERENCE_DATASET,
        "process_dataset": STRICT_PROCESS_DATASET,
        "sft_examples": len(sft_rows),
        "preference_pairs": len(preference_rows_out),
        "process_examples": len(process_rows),
        "preference_failure_modes": count_by(preference_rows_out, "failure_mode"),
        "chosen_passed": sum(bool(row.get("chosen_score", {}).get("passed")) for row in preference_rows_out),
        "rejected_passed": sum(bool(row.get("rejected_score", {}).get("passed")) for row in preference_rows_out),
        "by_case_family": count_by(sft_rows, "case_family"),
        "by_evidence_status": count_by(sft_rows, "gold_evidence_status"),
        "train_sft": display_path(split_paths["train_sft"]),
        "heldout_sft": display_path(split_paths["heldout_sft"]),
        "train_preferences": display_path(split_paths["train_preferences"]),
        "heldout_preferences": display_path(split_paths["heldout_preferences"]),
        "train_process": display_path(split_paths["train_process"]),
        "heldout_process": display_path(split_paths["heldout_process"]),
        "train_sft_examples": len(train_sft),
        "heldout_sft_examples": len(heldout_sft),
        "train_preference_pairs": len(train_preferences),
        "heldout_preference_pairs": len(heldout_preferences),
        "train_process_examples": len(train_process),
        "heldout_process_examples": len(heldout_process),
        "train_case_ids": train_case_ids,
        "heldout_case_ids": heldout_case_ids,
        "overlap_case_ids": sorted(set(train_case_ids) & set(heldout_case_ids)),
        "train_split_groups": train_split_groups,
        "heldout_split_groups": heldout_split_groups,
        "overlap_split_groups": sorted(set(train_split_groups) & set(heldout_split_groups)),
        "train_source_task_ids": train_source_task_ids,
        "heldout_source_task_ids": heldout_source_task_ids,
        "overlap_source_task_ids": sorted(set(train_source_task_ids) & set(heldout_source_task_ids)),
        "train_by_case_family": count_by(train_sft, "case_family"),
        "heldout_by_case_family": count_by(heldout_sft, "case_family"),
        "train_by_evidence_status": count_by(train_sft, "gold_evidence_status"),
        "heldout_by_evidence_status": count_by(heldout_sft, "gold_evidence_status"),
        "train_preference_failure_modes": count_by(train_preferences, "failure_mode"),
        "heldout_preference_failure_modes": count_by(heldout_preferences, "failure_mode"),
        "boundary": (
            "Stage A strict-contract targets reuse the saved-prediction prompt "
            "contract and the shared trajectory evaluator; no hidden label is "
            "introduced into model-visible prompt messages."
        ),
    }


def case_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("source_manifest_case_id")) for row in rows if row.get("source_manifest_case_id")}


def value_set(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    return {str(row.get(key)) for row in rows if row.get(key)}


def count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sft", default="post_training/stage_a_sft_v1.jsonl")
    parser.add_argument("--source-manifest", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument("--source-split-manifest", default="post_training/stage_a_split_manifest.json")
    parser.add_argument("--sft-out", default="post_training/stage_a_strict_contract_sft_v1.jsonl")
    parser.add_argument("--preference-out", default="post_training/stage_a_strict_contract_preferences_v1.jsonl")
    parser.add_argument("--process-out", default="post_training/stage_a_strict_contract_process_v1.jsonl")
    parser.add_argument("--train-sft-out", default="post_training/stage_a_strict_contract_sft_train_v1.jsonl")
    parser.add_argument("--heldout-sft-out", default="post_training/stage_a_strict_contract_sft_heldout_v1.jsonl")
    parser.add_argument(
        "--train-preferences-out",
        default="post_training/stage_a_strict_contract_preferences_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-preferences-out",
        default="post_training/stage_a_strict_contract_preferences_heldout_v1.jsonl",
    )
    parser.add_argument("--train-process-out", default="post_training/stage_a_strict_contract_process_train_v1.jsonl")
    parser.add_argument("--heldout-process-out", default="post_training/stage_a_strict_contract_process_heldout_v1.jsonl")
    parser.add_argument("--manifest-out", default="post_training/stage_a_strict_contract_manifest.json")
    args = parser.parse_args()

    manifest_rows = load_manifest_rows(args.source_manifest)
    sft_source_rows = load_jsonl(args.source_sft)
    split_manifest = json.loads(Path(args.source_split_manifest).read_text())

    sft, preferences, process = build_strict_contract_exports(sft_source_rows, manifest_rows)
    splits = apply_split_manifest(sft, preferences, process, split_manifest)

    split_paths = {
        "train_sft": args.train_sft_out,
        "heldout_sft": args.heldout_sft_out,
        "train_preferences": args.train_preferences_out,
        "heldout_preferences": args.heldout_preferences_out,
        "train_process": args.train_process_out,
        "heldout_process": args.heldout_process_out,
    }

    write_jsonl(args.sft_out, sft)
    write_jsonl(args.preference_out, preferences)
    write_jsonl(args.process_out, process)
    write_jsonl(args.train_sft_out, splits["train_sft"])
    write_jsonl(args.heldout_sft_out, splits["heldout_sft"])
    write_jsonl(args.train_preferences_out, splits["train_preferences"])
    write_jsonl(args.heldout_preferences_out, splits["heldout_preferences"])
    write_jsonl(args.train_process_out, splits["train_process"])
    write_jsonl(args.heldout_process_out, splits["heldout_process"])

    manifest = manifest_for_exports(
        source_sft=args.source_sft,
        source_manifest=args.source_manifest,
        source_split_manifest=args.source_split_manifest,
        sft_out=args.sft_out,
        preference_out=args.preference_out,
        process_out=args.process_out,
        split_paths=split_paths,
        sft_rows=sft,
        preference_rows_out=preferences,
        process_rows=process,
        splits=splits,
    )
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if (
        manifest["rejected_passed"]
        or manifest["overlap_case_ids"]
        or manifest["overlap_split_groups"]
        or manifest["overlap_source_task_ids"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
