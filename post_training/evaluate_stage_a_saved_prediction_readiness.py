#!/usr/bin/env python3
"""Compare saved Stage A prediction reports against full-trajectory gates.

This report is intentionally compact and public-safe. It reads already-curated
saved-prediction summaries and deterministic no-model smoke outputs, then
compares them with the full-trajectory arbitration scorecard. It never reads
raw model prediction JSONL, Slurm logs, model states, or ignored run folders.
"""

from __future__ import annotations

import argparse
import json
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
from post_training.generate_stage_a_predictions import generate_rows  # noqa: E402
from post_training.run_stage_a_sft_smoke_eval import (  # noqa: E402
    load_jsonl,
    load_manifest_rows,
    write_json,
)

DATASET = "negbiodb_ct_stage_a_saved_prediction_readiness_v1"
DEFAULT_COMPACT_SUMMARIES = (
    "post_training/stage_a_cayuga_hf_chat_baseline_summary_2026-07-04.json",
    "post_training/stage_a_cayuga_strict_contract_summary_2026-07-04.json",
    "post_training/stage_a_strict_sft_cayuga_smoke_summary_2026-07-04.json",
    "post_training/stage_a_v3_tool_trace_qwen05b_cayuga_summary_2026-07-09.json",
    "post_training/stage_a_v4_canonical_json_qwen05b_cayuga_summary_2026-07-09.json",
    "post_training/stage_a_saved_candidate_readout_qwen05b_train_observed_summary_2026-07-09.json",
    "post_training/stage_a_saved_candidate_readout_qwen05b_all_valid_summary_2026-07-09.json",
)
DEFAULT_CANDIDATE_GATE_SUMMARIES = (
    "post_training/stage_a_saved_candidate_gate_train_observed_qwen05b_2026-07-09.json",
    "post_training/stage_a_saved_candidate_gate_all_valid_qwen05b_2026-07-09.json",
)
DETERMINISTIC_MODES = (
    ("deterministic_saved_oracle", "oracle"),
    ("deterministic_saved_self_answer", "self_answer"),
    ("deterministic_compact_tool_names_oracle", "compact_tool_names_oracle"),
)


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def deterministic_records(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = expected_case_ids_from_rows(heldout_rows)
    records: list[dict[str, Any]] = []
    for name, mode in DETERMINISTIC_MODES:
        predictions = generate_rows(heldout_rows, mode=mode, run_id=name)
        report = build_prediction_eval_report(
            manifest_rows=manifest_rows,
            prediction_rows=predictions,
            expected_case_ids=expected,
            run_id=name,
        )
        records.append(
            normalize_record(
                name=name,
                source_type="deterministic_public_smoke",
                summary=report["summary"],
                run_id=name,
                model_candidate=False,
                raw_predictions_committed=False,
                compact_report_path=None,
                notes="Generated in memory from public deterministic prediction modes.",
            )
        )
    return records


def compact_summary_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        if isinstance(payload.get("runs"), list):
            records.extend(records_from_strict_sft_summary(payload, compact_report_path=str(path)))
            continue
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise ValueError(f"{path} has no result object or runs list")
        name = str(payload.get("run_id") or Path(path).stem)
        records.append(
            normalize_record(
                name=name,
                source_type="real_saved_model_compact_summary",
                summary=result,
                run_id=payload.get("run_id"),
                model_candidate=True,
                raw_predictions_committed=raw_predictions_committed(payload),
                compact_report_path=str(path),
                model=payload.get("model"),
                generation_mode=payload.get("generation_mode"),
                prompt_contract=payload.get("prompt_contract"),
                raw_predictions_sha256_present=isinstance(payload.get("predictions_sha256"), str),
                notes="Compact public summary; raw saved predictions remain uncommitted.",
            )
        )
    return records


def candidate_gate_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        best = payload.get("best_default_zero_unsafe_report")
        if not isinstance(best, Mapping):
            raise ValueError(f"{path} has no best_default_zero_unsafe_report object")
        summary = payload.get("summary", {})
        if not isinstance(summary, Mapping):
            summary = {}
        records.append(
            {
                "name": str(payload.get("run_id") or Path(path).stem),
                "source_type": "real_saved_candidate_gate_compact_summary",
                "run_id": payload.get("run_id"),
                "model": payload.get("model"),
                "prompt_contract": payload.get("prompt_contract"),
                "candidate_policy": payload.get("candidate_policy"),
                "compact_report_path": str(path),
                "model_candidate": True,
                "raw_predictions_committed": False,
                "cases": int(payload.get("cases", 0)),
                "exact_top1": int(summary.get("exact_top1", 0)),
                "mean_target_rank": summary.get("mean_target_rank"),
                "top_pair_counts": dict(summary.get("top_pair_counts", {})),
                "threshold": float(best.get("threshold", 0.0)),
                "trusted": int(best.get("trusted", 0)),
                "trusted_correct": int(best.get("trusted_correct", 0)),
                "trusted_incorrect": int(best.get("trusted_incorrect", 0)),
                "fail_closed": int(best.get("fail_closed", 0)),
                "fail_closed_exact_correct": int(best.get("fail_closed_exact_correct", 0)),
                "strict_final_correct": int(best.get("strict_final_correct", 0)),
                "strict_final_accuracy": round(float(best.get("strict_final_accuracy", 0.0)), 3),
                "trusted_precision": best.get("trusted_precision"),
                "unsafe_trust_case_ids": list(best.get("unsafe_trust_case_ids", ())),
                "trusted_case_ids": list(best.get("trusted_case_ids", ())),
                "fail_closed_case_ids": list(best.get("fail_closed_case_ids", ())),
                "notes": (
                    "Compact public fail-closed gate summary over ignored candidate-score JSONL; "
                    "raw candidate scores and raw saved predictions remain uncommitted."
                ),
            }
        )
    return records


def records_from_strict_sft_summary(
    payload: Mapping[str, Any],
    *,
    compact_report_path: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in payload.get("runs", ()):
        if not isinstance(item, Mapping) or not isinstance(item.get("result"), Mapping):
            raise ValueError(f"{compact_report_path} contains a malformed run item")
        records.append(
            normalize_record(
                name=str(item.get("run_id") or item.get("model") or "strict_sft_run"),
                source_type="real_saved_sft_compact_summary",
                summary=item["result"],
                run_id=item.get("run_id"),
                model_candidate=True,
                raw_predictions_committed=raw_predictions_committed(payload),
                compact_report_path=compact_report_path,
                model=item.get("model"),
                generation_mode="strict_contract_sft_smoke",
                prompt_contract=payload.get("prompt_contract"),
                raw_predictions_sha256_present=isinstance(item.get("predictions_sha256"), str),
                notes="Compact public SFT smoke summary; raw saved predictions remain uncommitted.",
            )
        )
    return records


def raw_predictions_committed(payload: Mapping[str, Any]) -> bool:
    policy = payload.get("artifact_policy", {})
    if not isinstance(policy, Mapping):
        return False
    return bool(policy.get("raw_predictions_committed", False))


def normalize_record(
    *,
    name: str,
    source_type: str,
    summary: Mapping[str, Any],
    run_id: Any,
    model_candidate: bool,
    raw_predictions_committed: bool,
    compact_report_path: str | None,
    model: Any = None,
    generation_mode: Any = None,
    prompt_contract: Any = None,
    raw_predictions_sha256_present: bool = False,
    notes: str,
) -> dict[str, Any]:
    parse_errors = summary.get("parse_errors", {})
    if not isinstance(parse_errors, Mapping):
        parse_errors = {}
    violations = summary.get("violations", {})
    if not isinstance(violations, Mapping):
        violations = {}
    gate_accuracy = summary.get("gate_accuracy", {})
    if not isinstance(gate_accuracy, Mapping):
        gate_accuracy = {}
    return {
        "name": name,
        "source_type": source_type,
        "run_id": run_id,
        "model": model,
        "generation_mode": generation_mode,
        "prompt_contract": prompt_contract,
        "compact_report_path": compact_report_path,
        "model_candidate": model_candidate,
        "raw_predictions_committed": raw_predictions_committed,
        "raw_predictions_sha256_present": raw_predictions_sha256_present,
        "cases": int(summary.get("cases", 0)),
        "passed": int(summary.get("passed", 0)),
        "mean_score": round(float(summary.get("mean_score", 0.0)), 3),
        "parse_error_count": sum(int(value) for value in parse_errors.values()),
        "parse_errors": dict(sorted((str(key), int(value)) for key, value in parse_errors.items())),
        "violations": dict(sorted((str(key), int(value)) for key, value in violations.items())),
        "gate_accuracy": {
            str(key): round(float(value), 3) for key, value in sorted(gate_accuracy.items())
        },
        "notes": notes,
    }


def heldout_scorecard(full_arbitration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    summary = full_arbitration.get("summary", {}).get("heldout", {})
    if not isinstance(summary, Mapping):
        raise ValueError("Full-trajectory arbitration report is missing summary.heldout")
    required = (
        "self_answer",
        "ground_supported_collapse",
        "citationless_runtime_action",
        "runtime_gate_full",
        "hybrid_runtime_over_collapse",
    )
    out: dict[str, dict[str, Any]] = {}
    for policy in required:
        payload = summary.get(policy)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Full-trajectory arbitration report is missing {policy}")
        out[policy] = {
            "cases": int(payload.get("cases", 0)),
            "passed": int(payload.get("passed", 0)),
            "mean_score": round(float(payload.get("mean_score", 0.0)), 3),
            "violations": dict(payload.get("violations", {})),
        }
    return out


def attach_comparisons(
    records: Sequence[Mapping[str, Any]],
    *,
    scorecard: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    collapse = scorecard["ground_supported_collapse"]
    citationless = scorecard["citationless_runtime_action"]
    runtime = scorecard["runtime_gate_full"]
    out = []
    for record in records:
        item = dict(record)
        item["comparison"] = {
            "beats_collapse_pass_count": int(item["passed"]) > int(collapse["passed"]),
            "beats_citationless_pass_count": int(item["passed"]) > int(citationless["passed"]),
            "matches_runtime_pass_count": int(item["passed"]) >= int(runtime["passed"]),
            "mean_score_above_collapse": float(item["mean_score"]) > float(collapse["mean_score"]),
            "mean_score_above_citationless": float(item["mean_score"]) > float(citationless["mean_score"]),
            "mean_score_matches_runtime": float(item["mean_score"]) >= float(runtime["mean_score"]),
        }
        out.append(item)
    return out


def attach_gate_comparisons(
    gate_records: Sequence[Mapping[str, Any]],
    *,
    scorecard: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    collapse = scorecard["ground_supported_collapse"]
    citationless = scorecard["citationless_runtime_action"]
    runtime = scorecard["runtime_gate_full"]
    out = []
    for record in gate_records:
        item = dict(record)
        strict_final = int(item.get("strict_final_correct", 0))
        item["comparison"] = {
            "zero_unsafe_trust": int(item.get("trusted_incorrect", 0)) == 0,
            "beats_collapse_strict_final": strict_final > int(collapse["passed"]),
            "beats_citationless_strict_final": strict_final > int(citationless["passed"]),
            "matches_runtime_strict_final": strict_final >= int(runtime["passed"]),
        }
        out.append(item)
    return out


def best_model_candidate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = [dict(record) for record in records if record.get("model_candidate")]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            int(item.get("passed", 0)),
            float(item.get("mean_score", 0.0)),
            -int(item.get("parse_error_count", 0)),
            str(item.get("name")),
        ),
        reverse=True,
    )[0]


def best_candidate_gate(gate_records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = [dict(record) for record in gate_records if record.get("model_candidate")]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            -int(item.get("trusted_incorrect", 0)),
            int(item.get("strict_final_correct", 0)),
            int(item.get("trusted_correct", 0)),
            -float(item.get("threshold", 0.0)),
            str(item.get("name")),
        ),
        reverse=True,
    )[0]


def build_decision(
    *,
    records: Sequence[Mapping[str, Any]],
    candidate_gates: Sequence[Mapping[str, Any]],
    scorecard: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    best = best_model_candidate(records)
    best_gate = best_candidate_gate(candidate_gates)
    blockers: list[str] = []
    if best is None:
        blockers.append("No real saved model compact summary was provided.")
    else:
        if int(best["passed"]) <= int(scorecard["ground_supported_collapse"]["passed"]):
            blockers.append("Best real saved output does not beat the collapse baseline.")
        if int(best["passed"]) <= int(scorecard["citationless_runtime_action"]["passed"]):
            blockers.append("Best real saved output does not beat the citationless routing baseline.")
        if int(best["passed"]) < int(scorecard["runtime_gate_full"]["passed"]):
            blockers.append("Best real saved output remains below runtime-gate full-trajectory performance.")
        if int(best["parse_error_count"]) > 0:
            blockers.append("Best real saved output still has parse errors.")
        violations = best.get("violations", {})
        if isinstance(violations, Mapping):
            if int(violations.get("missing_required_tool_sequence", 0)) > 0:
                blockers.append("Best real saved output still misses required tool sequence.")
            if int(violations.get("query_filter_missing_required_field", 0)) > 0:
                blockers.append("Best real saved output still misses required query fields.")
    if best_gate is None:
        blockers.append("No saved-candidate fail-closed gate summary was provided.")
    else:
        if int(best_gate.get("trusted_incorrect", 0)) > 0:
            blockers.append("Best saved-candidate gate still has unsafe trusted rows.")
        if int(best_gate.get("strict_final_correct", 0)) <= int(scorecard["citationless_runtime_action"]["passed"]):
            blockers.append("Best saved-candidate gate remains below citationless routing.")
        if int(best_gate.get("strict_final_correct", 0)) < int(scorecard["runtime_gate_full"]["passed"]):
            blockers.append("Best saved-candidate gate remains below runtime-gate full-trajectory performance.")
    return {
        "best_real_saved_output": best,
        "best_saved_candidate_gate": best_gate,
        "ready_for_tool_query": False,
        "ready_for_dpo_rlvr": False,
        "ready_for_hugging_face_publication": False,
        "ready_for_release_tagging": False,
        "runtime_enforcement_required": True,
        "blockers": blockers,
        "next_decision": (
            "Use the saved-prediction scorer and saved-candidate gate analyzer "
            "for the next real Cayuga output, then compare compact reports here. "
            "Do not reopen tool_query, DPO/RLVR, Hugging Face publication, or "
            "release tagging until a real saved model output and its fail-closed "
            "gate beat collapse/citationless baselines and approach the runtime "
            "full-trajectory gate."
        ),
    }


def build_report(
    *,
    full_arbitration: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    compact_summary_paths: Sequence[str | Path],
    candidate_gate_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    scorecard = heldout_scorecard(full_arbitration)
    records = deterministic_records(manifest_rows=manifest_rows, heldout_rows=heldout_rows)
    records.extend(compact_summary_records(compact_summary_paths))
    records = attach_comparisons(records, scorecard=scorecard)
    candidate_gates = attach_gate_comparisons(
        candidate_gate_records(candidate_gate_paths),
        scorecard=scorecard,
    )
    decision = build_decision(records=records, candidate_gates=candidate_gates, scorecard=scorecard)
    return {
        "dataset": DATASET,
        "canonical_evaluator": "negbiodb_ct.stage_a_manifest.score_stage_a_trajectory",
        "full_trajectory_scorecard_source": "stage_a_full_trajectory_arbitration_2026-07-09",
        "raw_model_outputs_used": False,
        "raw_model_outputs_committed": any(bool(record["raw_predictions_committed"]) for record in records),
        "heldout_scorecard": scorecard,
        "records": records,
        "candidate_gates": candidate_gates,
        "decision": decision,
        "scientific_readout": {
            "diagnostic_question": (
                "Do existing saved Stage A model outputs beat deterministic "
                "full-trajectory collapse/citationless/runtime baselines, and "
                "do score-gap gates provide zero-unsafe fail-closed coverage?"
            ),
            "interpretation_rule": (
                "Only real saved model outputs count for escalation. Deterministic "
                "oracle rows validate the adapter but are not model competence."
            ),
            "next_decision": decision["next_decision"],
        },
    }


def format_record_cell(record: Mapping[str, Any]) -> str:
    return f"{record['passed']}/{record['cases']} pass; mean {float(record['mean_score']):.3f}"


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Stage A Saved-Prediction Readiness",
        "",
        "Purpose: compare public deterministic saved-output smokes and compact",
        "real Cayuga saved-output summaries against the full-trajectory",
        "arbitration scorecard.",
        "",
        "## Held-out Scorecard",
        "",
        "| Baseline | Result |",
        "| --- | --- |",
    ]
    for key, label in (
        ("ground_supported_collapse", "collapse"),
        ("citationless_runtime_action", "citationless routing"),
        ("runtime_gate_full", "runtime gate"),
    ):
        item = report["heldout_scorecard"][key]
        lines.append(f"| `{label}` | {item['passed']}/{item['cases']} pass; mean {item['mean_score']:.3f} |")
    lines.extend(
        [
            "",
            "## Saved Output Reports",
            "",
            "| Report | Source | Candidate? | Result | Parse errors | Beats citationless? |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for record in report["records"]:
        comparison = record["comparison"]
        lines.append(
            "| `{name}` | `{source}` | {candidate} | {result} | {parse_errors} | {beats} |".format(
                name=record["name"],
                source=record["source_type"],
                candidate="yes" if record["model_candidate"] else "no",
                result=format_record_cell(record),
                parse_errors=record["parse_error_count"],
                beats="yes" if comparison["beats_citationless_pass_count"] else "no",
            )
        )
    if report.get("candidate_gates"):
        lines.extend(
            [
                "",
                "## Saved-Candidate Gates",
                "",
                "| Gate | Policy | Trusted | Unsafe trust | Strict final | Beats citationless? |",
                "| --- | --- | ---: | ---: | --- | ---: |",
            ]
        )
        for gate in report["candidate_gates"]:
            comparison = gate["comparison"]
            lines.append(
                "| `{name}` | `{policy}` | {trusted}/{cases} | {unsafe} | {strict}/{cases} | {beats} |".format(
                    name=gate["name"],
                    policy=gate.get("candidate_policy"),
                    trusted=gate["trusted"],
                    cases=gate["cases"],
                    unsafe=gate["trusted_incorrect"],
                    strict=gate["strict_final_correct"],
                    beats="yes" if comparison["beats_citationless_strict_final"] else "no",
                )
            )
    best_gate = report["decision"].get("best_saved_candidate_gate")
    if isinstance(best_gate, Mapping):
        lines.extend(
            [
                "",
                "Best fail-closed candidate gate:",
                (
                    f"- `{best_gate['name']}` trusts {best_gate['trusted']}/{best_gate['cases']} "
                    f"rows with {best_gate['trusted_incorrect']} unsafe trust and "
                    f"{best_gate['strict_final_correct']}/{best_gate['cases']} strict final correct."
                ),
            ]
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Ready for `tool_query`: `{decision['ready_for_tool_query']}`",
            f"- Ready for DPO/RLVR: `{decision['ready_for_dpo_rlvr']}`",
            f"- Runtime enforcement required: `{decision['runtime_enforcement_required']}`",
            "",
            "Blockers:",
        ]
    )
    for blocker in decision["blockers"]:
        lines.append(f"- {blocker}")
    lines.extend(["", str(decision["next_decision"])])
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-arbitration",
        default="post_training/stage_a_full_trajectory_arbitration_2026-07-09.json",
    )
    parser.add_argument("--manifest", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument("--heldout-sft", default="post_training/stage_a_sft_heldout_v1.jsonl")
    parser.add_argument(
        "--compact-summary",
        action="append",
        default=None,
        help="Compact public saved-output summary JSON. Repeatable.",
    )
    parser.add_argument(
        "--candidate-gate-summary",
        action="append",
        default=None,
        help="Compact public saved-candidate fail-closed gate JSON. Repeatable.",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    compact_summary_paths = args.compact_summary or list(DEFAULT_COMPACT_SUMMARIES)
    candidate_gate_paths = args.candidate_gate_summary or list(DEFAULT_CANDIDATE_GATE_SUMMARIES)
    report = build_report(
        full_arbitration=load_json(args.full_arbitration),
        manifest_rows=load_manifest_rows(args.manifest),
        heldout_rows=load_jsonl(args.heldout_sft),
        compact_summary_paths=compact_summary_paths,
        candidate_gate_paths=candidate_gate_paths,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    stdout_report = {
        "dataset": report["dataset"],
        "raw_model_outputs_used": report["raw_model_outputs_used"],
        "heldout_scorecard": report["heldout_scorecard"],
        "decision": report["decision"],
    }
    print(json.dumps(stdout_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
