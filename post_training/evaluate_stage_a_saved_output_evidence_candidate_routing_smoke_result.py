#!/usr/bin/env python3
"""Summarize evidence-conditioned candidate-routing smoke results.

This adapter reads only the compact eval report emitted by
`run_stage_a_saved_output_evidence_candidate_routing_smoke.py`. It does not read
raw candidate-score JSONL, raw model text, scheduler logs, model state, or
ignored run folders. Use it after a Cayuga smoke run to create the small
public-safe result summary that can be curated into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.evaluate_stage_a_saved_output_evidence_candidate_routing_smoke_spec import (  # noqa: E402
    DEFAULT_READOUT,
    DATASET as SPEC_DATASET,
)
from post_training.run_stage_a_sft_smoke_eval import write_json  # noqa: E402
from post_training.run_stage_a_saved_output_evidence_candidate_routing_smoke import (  # noqa: E402
    DATASET as RUNNER_DATASET,
)


DATASET = "negbiodb_ct_stage_a_saved_output_evidence_candidate_routing_smoke_result_v1"
DEFAULT_SPEC = (
    "post_training/stage_a_saved_output_evidence_candidate_routing_smoke_spec_2026-07-10.json"
)
RAW_EVAL_KEYS = (
    "candidate_scores",
    "raw_output",
    "raw_model_text",
    "prompt",
    "completion",
    "logits",
    "token_logprobs",
)


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"external_compact_input::{path.name}"


def input_artifact(path: str | Path, *, role: str) -> dict[str, str]:
    return {
        "path": public_path(path),
        "role": role,
        "sha256": sha256_file(path),
    }


def raw_key_paths(value: Any, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child = f"{prefix}.{key_text}"
            if key_text in RAW_EVAL_KEYS:
                paths.append(child)
            paths.extend(raw_key_paths(item, prefix=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(raw_key_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def as_int(payload: Mapping[str, Any], key: str, *, context: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} missing integer field: {key}")
    return value


def summary_from_eval(eval_report: Mapping[str, Any], split: str) -> dict[str, Any]:
    section = eval_report.get(split)
    if not isinstance(section, Mapping):
        raise ValueError(f"eval report missing {split} object")
    summary = section.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"eval report missing {split}.summary object")
    return {
        "split": split,
        "rows": as_int(summary, "rows", context=f"{split}.summary"),
        "exact": as_int(summary, "exact", context=f"{split}.summary"),
        "accuracy": summary.get("accuracy"),
        "mean_score": summary.get("mean_score"),
        "bridge_focus_rows": as_int(
            summary,
            "bridge_focus_rows",
            context=f"{split}.summary",
        ),
        "bridge_focus_exact": as_int(
            summary,
            "bridge_focus_exact",
            context=f"{split}.summary",
        ),
        "bridge_focus_accuracy": summary.get("bridge_focus_accuracy"),
        "by_target_pair": dict(summary.get("by_target_pair", {})),
        "by_predicted_pair": dict(summary.get("by_predicted_pair", {})),
        "violations": dict(summary.get("violations", {})),
        "error_case_ids": list(summary.get("error_case_ids", ())),
        "bridge_focus_error_case_ids": list(summary.get("bridge_focus_error_case_ids", ())),
    }


def gate_requirements(spec: Mapping[str, Any]) -> dict[str, int]:
    gate = spec.get("acceptance_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("smoke spec missing acceptance_gate object")
    return {
        "heldout_exact_min": as_int(
            gate,
            "candidate_model_heldout_exact_min",
            context="acceptance_gate",
        ),
        "heldout_rows": as_int(
            gate,
            "candidate_model_heldout_rows",
            context="acceptance_gate",
        ),
        "bridge_focus_exact_min": as_int(
            gate,
            "candidate_model_bridge_focus_exact_min",
            context="acceptance_gate",
        ),
        "bridge_focus_rows": as_int(
            gate,
            "candidate_model_bridge_focus_rows",
            context="acceptance_gate",
        ),
        "best_static_prior_heldout_exact_to_beat": as_int(
            gate,
            "best_static_prior_heldout_exact_to_beat",
            context="acceptance_gate",
        ),
    }


def gate_violations(
    *,
    train: Mapping[str, Any],
    heldout: Mapping[str, Any],
    requirements: Mapping[str, int],
    raw_paths: list[str],
    eval_report: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    if eval_report.get("dataset") != RUNNER_DATASET:
        violations.append("eval_report_dataset_mismatch")
    if spec.get("dataset") != SPEC_DATASET:
        violations.append("smoke_spec_dataset_mismatch")
    if raw_paths:
        violations.append("raw_fields_present_in_eval_report")
    if heldout["rows"] != requirements["heldout_rows"]:
        violations.append("heldout_rows_mismatch")
    if heldout["exact"] < requirements["heldout_exact_min"]:
        violations.append("below_heldout_exact_min")
    if heldout["bridge_focus_rows"] != requirements["bridge_focus_rows"]:
        violations.append("bridge_focus_rows_mismatch")
    if heldout["bridge_focus_exact"] < requirements["bridge_focus_exact_min"]:
        violations.append("below_bridge_focus_exact_min")
    if heldout["exact"] <= requirements["best_static_prior_heldout_exact_to_beat"]:
        violations.append("does_not_beat_static_prior")
    if train["rows"] <= 0:
        violations.append("train_summary_missing")
    return violations


def build_report(
    *,
    eval_report_path: str | Path,
    smoke_spec_path: str | Path = DEFAULT_SPEC,
    readout_path: str | Path = DEFAULT_READOUT,
    policy_name: str = "evidence_candidate_routing_model_policy",
) -> dict[str, Any]:
    eval_report = load_json(eval_report_path)
    smoke_spec = load_json(smoke_spec_path)
    readout = load_json(readout_path)
    train = summary_from_eval(eval_report, "train")
    heldout = summary_from_eval(eval_report, "heldout")
    requirements = gate_requirements(smoke_spec)
    raw_paths = raw_key_paths(eval_report)
    violations = gate_violations(
        train=train,
        heldout=heldout,
        requirements=requirements,
        raw_paths=raw_paths,
        eval_report=eval_report,
        spec=smoke_spec,
    )
    passes_gate = not violations
    if passes_gate:
        selected_next_step = "review_next_stage_a_model_heavy_step"
        interpretation = (
            "The compact policy met the smoke gate and may reopen review of the "
            "next Stage A model-heavy step, but it does not by itself open "
            "DPO/RLVR, tool_query, HF publication, or release tagging."
        )
    else:
        selected_next_step = "keep_runtime_gate_and_review_candidate_collapse"
        interpretation = (
            "The compact policy failed the held-out candidate-routing gate. Keep "
            "runtime evidence arbitration as the baseline and keep DPO/RLVR, "
            "tool_query, HF publication, and release tagging closed."
        )
    return {
        "dataset": DATASET,
        "run_id": eval_report.get("run_id"),
        "policy": policy_name,
        "input_artifacts": {
            "eval_report": input_artifact(
                eval_report_path,
                role="compact evidence candidate-routing smoke eval report",
            ),
            "smoke_spec": input_artifact(
                smoke_spec_path,
                role="tracked smoke acceptance spec",
            ),
            "readout": input_artifact(
                readout_path,
                role="tracked no-model readout baseline",
            ),
        },
        "train_summary": train,
        "heldout_summary": heldout,
        "requirements": dict(requirements),
        "policy_summary": {
            "policy": policy_name,
            "exact": heldout["exact"],
            "rows": heldout["rows"],
            "bridge_focus_exact": heldout["bridge_focus_exact"],
            "bridge_focus_rows": heldout["bridge_focus_rows"],
            "mean_score": heldout.get("mean_score"),
            "error_case_ids": heldout["error_case_ids"],
            "bridge_focus_error_case_ids": heldout["bridge_focus_error_case_ids"],
        },
        "baseline": {
            "runtime_evidence_gate": smoke_spec.get("acceptance_gate", {}).get(
                "runtime_evidence_gate_to_match"
            ),
            "best_static_prior_heldout_exact": readout.get("decision", {}).get(
                "best_static_prior_heldout_exact"
            ),
        },
        "passes_gate": passes_gate,
        "gate_violations": violations,
        "raw_field_paths": raw_paths[:20],
        "decision": {
            "ready_for_escalation_review": passes_gate,
            "ready_for_dpo_rlvr": False,
            "ready_for_tool_query": False,
            "ready_for_hugging_face_publication": False,
            "ready_for_release_tagging": False,
            "selected_next_step": selected_next_step,
            "interpretation": interpretation,
        },
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
            "raw_fields_in_eval_report": bool(raw_paths),
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    heldout = report["heldout_summary"]
    train = report["train_summary"]
    req = report["requirements"]
    lines = [
        "# Stage A Evidence Candidate-Routing Smoke Result",
        "",
        "Purpose: summarize a compact smoke eval report without publishing raw candidate scores.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Train exact: {train['exact']}/{train['rows']}",
        f"- Held-out exact: {heldout['exact']}/{heldout['rows']}",
        (
            f"- Bridge-focus exact: {heldout['bridge_focus_exact']}/"
            f"{heldout['bridge_focus_rows']}"
        ),
        f"- Passes gate: `{report['passes_gate']}`",
        f"- Gate violations: `{json.dumps(report['gate_violations'], sort_keys=True)}`",
        "",
        "## Gate",
        "",
        f"- Held-out exact minimum: {req['heldout_exact_min']}/{req['heldout_rows']}",
        (
            f"- Bridge-focus exact minimum: {req['bridge_focus_exact_min']}/"
            f"{req['bridge_focus_rows']}"
        ),
        (
            f"- Static prior held-out exact to beat: "
            f"{req['best_static_prior_heldout_exact_to_beat']}/{req['heldout_rows']}"
        ),
        "",
        "Public-safety contract: this adapter reads compact eval reports only;",
        "raw candidate-score JSONL, raw model text, scheduler logs, model state,",
        "and ignored run folders are not public artifacts.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-report", required=True)
    parser.add_argument("--smoke-spec", default=DEFAULT_SPEC)
    parser.add_argument("--readout", default=DEFAULT_READOUT)
    parser.add_argument("--policy", default="evidence_candidate_routing_model_policy")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = build_report(
        eval_report_path=args.eval_report,
        smoke_spec_path=args.smoke_spec,
        readout_path=args.readout,
        policy_name=args.policy,
    )
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
