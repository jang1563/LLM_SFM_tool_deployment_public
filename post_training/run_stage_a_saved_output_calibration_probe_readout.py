#!/usr/bin/env python3
"""Score the Stage A saved-output calibration probe with finite candidates.

The probe contains target-vs-ground/supported pairs derived from compact
saved-output readiness evidence. This runner scores only the chosen target
output and the rejected collapse output for each probe row, then evaluates a
train-selected fail-closed score-gap gate on held-out probe rows.

Dry-run mode loads no model weights and is intended for public CI. Full mode
requires --allow-model-load and should run on Cayuga/Expanse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.generate_stage_a_predictions import (  # noqa: E402
    DEFAULT_HF_MODEL,
    PROMPT_CONTRACTS,
    get_hf_chat_client,
)
from post_training.run_stage_a_sft_smoke_eval import load_jsonl, write_json  # noqa: E402
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    prompt_text_for_tokenizer,
    score_candidate_target,
    write_jsonl,
)


DATASET = "negbiodb_ct_stage_a_saved_output_calibration_probe_readout_v1"
DEFAULT_THRESHOLDS = (0.0, 0.005, 0.01, 0.025, 0.035, 0.05, 0.075, 0.1)
DEFAULT_FAIL_CLOSED_PAIR = {"action": "defer", "evidence_status": "insufficient"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_pair(obj: Mapping[str, Any] | None) -> dict[str, str | None]:
    obj = obj or {}
    return {
        "action": str(obj.get("action")) if obj.get("action") is not None else None,
        "evidence_status": str(obj.get("evidence_status")) if obj.get("evidence_status") is not None else None,
    }


def pair_label(obj: Mapping[str, Any] | None) -> str:
    pair = compact_pair(obj)
    return f"{pair['action']}/{pair['evidence_status']}"


def pair_key(obj: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    pair = compact_pair(obj)
    return pair["action"], pair["evidence_status"]


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if any(threshold < 0 for threshold in thresholds):
        raise ValueError("thresholds must be non-negative")
    return thresholds


def score_outputs_for_probe_row(
    row: Mapping[str, Any],
    *,
    client: Any | None,
    max_length: int,
) -> tuple[float, float]:
    if client is None:
        return 0.0, -0.1
    prompt = prompt_text_for_tokenizer(client.tokenizer, row["prompt_messages"])
    chosen_score = score_candidate_target(
        client.model,
        client.tokenizer,
        prompt,
        json.dumps(row["chosen_output"], sort_keys=True),
        device=client.device,
        max_length=max_length,
    )
    rejected_score = score_candidate_target(
        client.model,
        client.tokenizer,
        prompt,
        json.dumps(row["rejected_output"], sort_keys=True),
        device=client.device,
        max_length=max_length,
    )
    return float(chosen_score), float(rejected_score)


def readout_row(
    row: Mapping[str, Any],
    *,
    run_id: str,
    model: str | None,
    dry_run: bool,
    client: Any | None,
    max_length: int,
) -> dict[str, Any]:
    chosen_score, rejected_score = score_outputs_for_probe_row(
        row,
        client=client,
        max_length=max_length,
    )
    chosen_pair = compact_pair(row.get("chosen_output") if isinstance(row.get("chosen_output"), Mapping) else {})
    rejected_pair = compact_pair(row.get("rejected_output") if isinstance(row.get("rejected_output"), Mapping) else {})
    margin = round(chosen_score - rejected_score, 6)
    top_role = "chosen" if margin > 0 else "rejected"
    top_pair = chosen_pair if top_role == "chosen" else rejected_pair
    top_second_gap = round(abs(margin), 6)
    return {
        "id": f"{run_id}::{row.get('case_id')}::saved_output_calibration_probe_readout",
        "dataset": DATASET,
        "run_id": run_id,
        "model": model,
        "dry_run": dry_run,
        "source_probe_id": row.get("id"),
        "source_next_decision": row.get("source_next_decision"),
        "case_id": row.get("case_id"),
        "case_family": row.get("case_family"),
        "split": row.get("split"),
        "split_group": row.get("split_group"),
        "source_task_id": row.get("source_task_id"),
        "prompt_contract": row.get("prompt_contract"),
        "calibration_axis": row.get("calibration_axis"),
        "training_allowed": row.get("training_allowed") is True,
        "evaluation_only": row.get("evaluation_only") is True,
        "target_pair": chosen_pair,
        "target_pair_label": pair_label(chosen_pair),
        "rejected_pair": rejected_pair,
        "rejected_pair_label": pair_label(rejected_pair),
        "candidate_scores": [
            {"role": "chosen", "candidate": chosen_pair, "score": chosen_score},
            {"role": "rejected", "candidate": rejected_pair, "score": rejected_score},
        ],
        "top_role": top_role,
        "top_pair": top_pair,
        "top_pair_label": pair_label(top_pair),
        "exact_top1": top_role == "chosen",
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "chosen_minus_rejected_margin": margin,
        "top_second_gap": top_second_gap,
        "candidate_count": 2,
    }


def build_readout_rows(
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    model: str | None,
    dry_run: bool,
    client: Any | None,
    max_length: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(probe_rows, start=1):
        rows.append(
            readout_row(
                row,
                run_id=run_id,
                model=model,
                dry_run=dry_run,
                client=client,
                max_length=max_length,
            )
        )
        print(f"[{index}/{len(probe_rows)}] scored {row.get('case_id')}", flush=True)
    return rows


def evaluate_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    fail_closed_pair: Mapping[str, Any] = DEFAULT_FAIL_CLOSED_PAIR,
) -> dict[str, Any]:
    trusted = [
        row
        for row in rows
        if row.get("top_second_gap") is not None and float(row["top_second_gap"]) >= threshold
    ]
    fail_closed = [row for row in rows if row not in trusted]
    trusted_correct = [row for row in trusted if row.get("exact_top1")]
    trusted_incorrect = [row for row in trusted if not row.get("exact_top1")]
    fail_closed_exact = [
        row for row in fail_closed if pair_key(row.get("target_pair")) == pair_key(fail_closed_pair)
    ]
    strict_final_correct = len(trusted_correct) + len(fail_closed_exact)
    return {
        "threshold": threshold,
        "rows": len(rows),
        "trusted": len(trusted),
        "trusted_correct": len(trusted_correct),
        "trusted_incorrect": len(trusted_incorrect),
        "fail_closed": len(fail_closed),
        "fail_closed_exact_correct": len(fail_closed_exact),
        "strict_final_correct": strict_final_correct,
        "coverage": round(len(trusted) / len(rows), 6) if rows else 0.0,
        "trusted_precision": round(len(trusted_correct) / len(trusted), 6) if trusted else None,
        "strict_final_accuracy": round(strict_final_correct / len(rows), 6) if rows else 0.0,
        "trusted_case_ids": [str(row.get("case_id")) for row in trusted],
        "unsafe_trust_case_ids": [str(row.get("case_id")) for row in trusted_incorrect],
        "fail_closed_case_ids": [str(row.get("case_id")) for row in fail_closed],
    }


def best_zero_unsafe_report(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    zero_unsafe = [report for report in reports if report["trusted_incorrect"] == 0]
    if not zero_unsafe:
        return None
    return sorted(
        zero_unsafe,
        key=lambda report: (
            -int(report["strict_final_correct"]),
            -int(report["trusted_correct"]),
            float(report["threshold"]),
        ),
    )[0]


def adaptive_zero_unsafe_threshold(rows: Sequence[Mapping[str, Any]]) -> float | None:
    incorrect_gaps = [
        float(row["top_second_gap"])
        for row in rows
        if not row.get("exact_top1") and row.get("top_second_gap") is not None
    ]
    if not incorrect_gaps:
        return 0.0
    return round(max(incorrect_gaps) + 0.000001, 6)


def split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    margins = [
        float(row["chosen_minus_rejected_margin"])
        for row in rows
        if row.get("chosen_minus_rejected_margin") is not None
    ]
    gaps = [float(row["top_second_gap"]) for row in rows if row.get("top_second_gap") is not None]
    return {
        "rows": len(rows),
        "exact_top1": sum(1 for row in rows if row.get("exact_top1")),
        "mean_margin": round(mean(margins), 6) if margins else None,
        "min_margin": round(min(margins), 6) if margins else None,
        "mean_top_second_gap": round(mean(gaps), 6) if gaps else None,
        "top_pair_counts": dict(sorted(Counter(str(row.get("top_pair_label")) for row in rows).items())),
        "target_pair_counts": dict(sorted(Counter(str(row.get("target_pair_label")) for row in rows).items())),
    }


def build_gate_report(
    readout_rows: Sequence[Mapping[str, Any]],
    *,
    probe_manifest: Mapping[str, Any],
    probe_pairs_path: str | Path,
    train_pairs_path: str | Path,
    heldout_pairs_path: str | Path,
    readout_path: str | Path,
    thresholds: Sequence[float],
    fail_closed_pair: Mapping[str, Any] = DEFAULT_FAIL_CLOSED_PAIR,
) -> dict[str, Any]:
    rows = list(readout_rows)
    train_rows = [row for row in rows if row.get("split") == "train"]
    heldout_rows = [row for row in rows if row.get("split") == "heldout"]
    thresholds = sorted(set(float(threshold) for threshold in thresholds))
    train_threshold_reports = [
        evaluate_threshold(train_rows, threshold=threshold, fail_closed_pair=fail_closed_pair)
        for threshold in thresholds
    ]
    heldout_threshold_reports = [
        evaluate_threshold(heldout_rows, threshold=threshold, fail_closed_pair=fail_closed_pair)
        for threshold in thresholds
    ]
    train_selected = best_zero_unsafe_report(train_threshold_reports)
    train_selected_threshold = train_selected.get("threshold") if train_selected else None
    heldout_at_train_selected = (
        evaluate_threshold(
            heldout_rows,
            threshold=float(train_selected_threshold),
            fail_closed_pair=fail_closed_pair,
        )
        if train_selected_threshold is not None
        else None
    )
    adaptive_train_threshold = adaptive_zero_unsafe_threshold(train_rows)
    adaptive_heldout = (
        evaluate_threshold(
            heldout_rows,
            threshold=adaptive_train_threshold,
            fail_closed_pair=fail_closed_pair,
        )
        if adaptive_train_threshold is not None
        else None
    )
    return {
        "dataset": DATASET,
        "run_id": rows[0].get("run_id") if rows else None,
        "model": rows[0].get("model") if rows else None,
        "dry_run": rows[0].get("dry_run") if rows else None,
        "input_probe_pairs_sha256": sha256_file(probe_pairs_path),
        "input_probe_train_sha256": sha256_file(train_pairs_path),
        "input_probe_heldout_sha256": sha256_file(heldout_pairs_path),
        "input_readout_sha256": sha256_file(readout_path),
        "source_probe_dataset": probe_manifest.get("pair_dataset"),
        "source_probe_next_step": probe_manifest.get("selected_next_step"),
        "prompt_contract": rows[0].get("prompt_contract") if rows else None,
        "fail_closed_pair": compact_pair(fail_closed_pair),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "summary": {
            "all": split_summary(rows),
            "train": split_summary(train_rows),
            "heldout": split_summary(heldout_rows),
        },
        "train_threshold_reports": train_threshold_reports,
        "heldout_threshold_reports": heldout_threshold_reports,
        "train_selected_zero_unsafe_report": train_selected,
        "heldout_at_train_selected_threshold": heldout_at_train_selected,
        "adaptive_train_zero_unsafe_threshold": adaptive_train_threshold,
        "heldout_at_adaptive_train_threshold": adaptive_heldout,
        "probe_gate_target": {
            "heldout_probe_strict_final_correct_min": len(heldout_rows),
            "heldout_probe_trusted_incorrect": 0,
            "full_saved_output_next_gate_remains": probe_manifest.get("minimum_next_gate"),
        },
        "artifact_policy": {
            "raw_prompt_messages_committed": False,
            "raw_model_text_committed": False,
            "scheduler_logs_committed": False,
            "model_state_committed": False,
            "candidate_score_jsonl_committed": False,
        },
        "scientific_readout": {
            "diagnostic_question": (
                "Can model score gaps distinguish the target action/status output "
                "from the observed ground/supported collapse on the calibration probe?"
            ),
            "split_rule": (
                "Threshold selection uses train rows only. Held-out probe rows are "
                "evaluation-only and must not be used for calibration."
            ),
            "decision_boundary": (
                "This is a candidate-calibration diagnostic, not DPO/RLVR data, "
                "not broad retraining, and not a release-readiness claim."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: str | Path) -> None:
    summary = report["summary"]
    selected = report.get("train_selected_zero_unsafe_report") or {}
    heldout = report.get("heldout_at_train_selected_threshold") or {}
    lines = [
        "# Stage A Saved-Output Calibration Probe Readout",
        "",
        "Purpose: score target-vs-ground/supported probe pairs and evaluate a",
        "train-selected fail-closed gate on held-out probe rows without publishing",
        "raw prompts, raw model text, scheduler logs, or model state.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Model: `{report.get('model')}`",
        f"- Dry run: `{report.get('dry_run')}`",
        f"- Rows: {report['rows']} total, {report['train_rows']} train, {report['heldout_rows']} held-out",
        f"- Train exact top-1: {summary['train']['exact_top1']}/{summary['train']['rows']}",
        f"- Held-out exact top-1: {summary['heldout']['exact_top1']}/{summary['heldout']['rows']}",
        f"- Train top pairs: `{json.dumps(summary['train']['top_pair_counts'], sort_keys=True)}`",
        f"- Held-out top pairs: `{json.dumps(summary['heldout']['top_pair_counts'], sort_keys=True)}`",
        "",
        "## Gate",
        "",
        f"- Train-selected zero-unsafe threshold: `{selected.get('threshold')}`",
        f"- Train trusted incorrect: {selected.get('trusted_incorrect')}",
        f"- Train strict final correct: {selected.get('strict_final_correct')}/{selected.get('rows')}",
        f"- Held-out trusted incorrect at train threshold: {heldout.get('trusted_incorrect')}",
        f"- Held-out strict final correct at train threshold: {heldout.get('strict_final_correct')}/{heldout.get('rows')}",
        "",
        "## Decision",
        "",
        "This readout is a calibration diagnostic only. Keep `tool_query`,",
        "DPO/RLVR, Hugging Face publication, release tagging, and broad retraining",
        "gated until a compact saved-output summary and fail-closed gate meet the",
        "next threshold.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="post_training/stage_a_saved_output_calibration_probe_v1.jsonl")
    parser.add_argument(
        "--train-pairs",
        default="post_training/stage_a_saved_output_calibration_probe_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-pairs",
        default="post_training/stage_a_saved_output_calibration_probe_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="post_training/stage_a_saved_output_calibration_probe_manifest.json",
    )
    parser.add_argument("--out-dir", default="post_training/runs/stage_a_saved_output_calibration_probe_readout")
    parser.add_argument("--run-id", default="stage_a_saved_output_calibration_probe_readout")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--prompt-contract", choices=PROMPT_CONTRACTS, default="stage_a_v4_canonical_json")
    parser.add_argument("--thresholds", default=",".join(str(item) for item in DEFAULT_THRESHOLDS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--readout-out", default=None)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--md-out", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.dry_run and not args.allow_model_load:
        raise RuntimeError("Full calibration readout requires --allow-model-load; use --dry-run for no-model CI.")

    probe_rows = load_jsonl(args.pairs)
    probe_manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readout_path = Path(args.readout_out) if args.readout_out else out_dir / "readout.jsonl"
    report_path = Path(args.report_out) if args.report_out else out_dir / "report.json"
    md_path = Path(args.md_out) if args.md_out else out_dir / "REPORT.md"

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

    rows = build_readout_rows(
        probe_rows,
        run_id=args.run_id,
        model=model_id,
        dry_run=args.dry_run,
        client=client,
        max_length=args.max_length,
    )
    write_jsonl(readout_path, rows)
    report = build_gate_report(
        rows,
        probe_manifest=probe_manifest,
        probe_pairs_path=args.pairs,
        train_pairs_path=args.train_pairs,
        heldout_pairs_path=args.heldout_pairs,
        readout_path=readout_path,
        thresholds=parse_thresholds(args.thresholds),
    )
    report["readout"] = str(readout_path)
    write_json(report_path, report)
    write_markdown(report, md_path)
    stdout_report = {
        "dataset": report["dataset"],
        "run_id": report["run_id"],
        "dry_run": report["dry_run"],
        "rows": report["rows"],
        "train_rows": report["train_rows"],
        "heldout_rows": report["heldout_rows"],
        "summary": report["summary"],
        "train_selected_zero_unsafe_report": report["train_selected_zero_unsafe_report"],
        "heldout_at_train_selected_threshold": report["heldout_at_train_selected_threshold"],
    }
    print(json.dumps(stdout_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
