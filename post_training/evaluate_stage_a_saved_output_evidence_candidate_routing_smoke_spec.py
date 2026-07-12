#!/usr/bin/env python3
"""Define the evidence-conditioned candidate-routing Cayuga smoke spec.

This spec is the public-safe follow-up to the no-model candidate-routing
readout. It reads only tracked compact artifacts and emits a reproducible
experiment contract; raw saved predictions, candidate-score JSONL, scheduler
logs, model state, and ignored run folders are not read.
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

from post_training.run_stage_a_sft_smoke_eval import write_json  # noqa: E402


DATASET = "negbiodb_ct_stage_a_saved_output_evidence_candidate_routing_smoke_spec_v1"
DEFAULT_MANIFEST = "post_training/stage_a_saved_output_evidence_candidate_routing_manifest.json"
DEFAULT_READOUT = (
    "post_training/stage_a_saved_output_evidence_candidate_routing_readout_2026-07-10.json"
)
RUNNER = "post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke.py"
CAYUGA_SBATCH = (
    "post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke_cayuga.sbatch"
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
        return path.as_posix()


def input_artifact(path: str | Path, *, role: str) -> dict[str, str]:
    return {
        "path": public_path(path),
        "role": role,
        "sha256": sha256_file(path),
    }


def policy_exact(readout: Mapping[str, Any], policy: str) -> dict[str, Any]:
    heldout = readout.get("policies", {}).get(policy, {}).get("heldout", {})
    if not isinstance(heldout, Mapping):
        heldout = {}
    return {
        "policy": policy,
        "exact": int(heldout.get("exact", 0)),
        "rows": int(heldout.get("rows", 0)),
        "bridge_focus_exact": int(heldout.get("bridge_focus_exact", 0)),
        "bridge_focus_rows": int(heldout.get("bridge_focus_rows", 0)),
    }


def static_prior_max(readout: Mapping[str, Any]) -> int:
    exacts: list[int] = []
    for name, summary in readout.get("policies", {}).items():
        if not str(name).startswith("static_") or not isinstance(summary, Mapping):
            continue
        heldout = summary.get("heldout", {})
        if isinstance(heldout, Mapping):
            exacts.append(int(heldout.get("exact", 0)))
    return max(exacts) if exacts else 0


def expected_pair_counts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "all": dict(manifest.get("by_target_pair", {})),
        "train": dict(manifest.get("train_by_target_pair", {})),
        "heldout": dict(manifest.get("heldout_by_target_pair", {})),
        "bridge_focus": dict(manifest.get("bridge_focus_by_target_pair", {})),
    }


def preconditions(manifest: Mapping[str, Any], readout: Mapping[str, Any]) -> dict[str, Any]:
    decision = readout.get("decision", {})
    row_counts = {
        "row_count": int(manifest.get("row_count", 0)),
        "train_rows": int(manifest.get("train_rows", 0)),
        "heldout_rows": int(manifest.get("heldout_rows", 0)),
        "bridge_focus_rows": int(manifest.get("bridge_focus_rows", 0)),
    }
    runtime = policy_exact(readout, "runtime_evidence_gate")
    prior_max = static_prior_max(readout)
    checks = {
        "row_count_is_25": row_counts["row_count"] == 25,
        "train_rows_are_20": row_counts["train_rows"] == 20,
        "heldout_rows_are_5": row_counts["heldout_rows"] == 5,
        "bridge_focus_rows_are_4": row_counts["bridge_focus_rows"] == 4,
        "no_train_heldout_case_overlap": not manifest.get("overlap_case_ids"),
        "no_train_heldout_source_overlap": not manifest.get("overlap_source_task_ids"),
        "no_train_heldout_split_group_overlap": not manifest.get("overlap_split_groups"),
        "no_model_readout_passed": bool(decision.get("passes_no_model_readout")),
        "runtime_gate_heldout_is_5_of_5": runtime["exact"] == 5
        and runtime["rows"] == 5,
        "runtime_gate_bridge_focus_is_4_of_4": runtime["bridge_focus_exact"] == 4
        and runtime["bridge_focus_rows"] == 4,
        "static_prior_max_is_1_of_5": prior_max == 1,
        "runner_exists": (ROOT / RUNNER).exists(),
        "cayuga_sbatch_exists": (ROOT / CAYUGA_SBATCH).exists(),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "row_counts": row_counts,
        "runtime_evidence_gate": runtime,
        "best_static_prior_heldout_exact": prior_max,
        "pair_counts": expected_pair_counts(manifest),
    }


def preflight_commands() -> list[str]:
    return [
        (
            "python post_training/export_stage_a_saved_output_evidence_candidate_routing_rows.py "
            "--rows-out /tmp/stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl "
            "--train-out /tmp/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl "
            "--heldout-out /tmp/stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl "
            "--manifest-out /tmp/stage_a_saved_output_evidence_candidate_routing_manifest.json"
        ),
        (
            "python post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_readout.py "
            "--out-json /tmp/stage_a_saved_output_evidence_candidate_routing_readout.json "
            "--out-md /tmp/STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_READOUT.md"
        ),
        "python post_training/validate_post_training_data.py",
    ]


def smoke_env() -> dict[str, str]:
    return {
        "RUN_ID": "stage_a_evidence_candidate_routing_qwen05b_cayuga_${JOB_ID}",
        "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
        "MAX_STEPS": "40",
        "BATCH_SIZE": "1",
        "LR": "1e-5",
        "TRAIN_LAST_LAYERS": "1",
        "MAX_LENGTH": "1536",
        "TARGET_FORMAT": "selected_pair_action_status_json",
        "CANDIDATE_SCORE_MODE": "finite_pair_mean_logprob",
        "ALLOW_DOWNLOAD": "0",
    }


def build_report(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    readout_path: str | Path = DEFAULT_READOUT,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    readout = load_json(readout_path)
    gate = preconditions(manifest, readout)
    return {
        "dataset": DATASET,
        "input_artifacts": {
            "candidate_routing_manifest": input_artifact(
                manifest_path,
                role="evidence-conditioned candidate-routing manifest",
            ),
            "candidate_routing_readout": input_artifact(
                readout_path,
                role="no-model evidence-conditioned candidate-routing readout",
            ),
        },
        "preconditions": gate,
        "smoke_spec": {
            "name": "evidence_conditioned_candidate_routing_sft_smoke",
            "runner": RUNNER,
            "cayuga_sbatch": CAYUGA_SBATCH,
            "research_question": (
                "Can a small supervised model select the correct action/status "
                "candidate from prompt-visible evidence features, beating every "
                "static prior and matching runtime evidence routing on held-out "
                "and bridge-focus rows?"
            ),
            "component": "saved_output_evidence_candidate_routing",
            "prompt_contract": manifest.get("prompt_contract"),
            "candidate_pairs": list(manifest.get("candidate_pairs", ())),
            "train_rows": {
                "path": manifest.get("train_rows_path"),
                "rows": manifest.get("train_rows"),
                "allowed_for_training": True,
            },
            "heldout_rows": {
                "path": manifest.get("heldout_rows_path"),
                "rows": manifest.get("heldout_rows"),
                "bridge_focus_rows": manifest.get("bridge_focus_rows"),
                "evaluation_only": True,
            },
            "target_output_schema": {
                "selected_pair": "one of candidate_pairs",
                "action": "ground | reject | defer | verify | flag",
                "evidence_status": "supported | contradicted | insufficient | invalid_value",
            },
            "training_contract": (
                "SFT only on train rows' assistant target JSON. Held-out bridge "
                "focus rows stay evaluation-only. The model-visible task may "
                "include observed tool-loop state and visible evidence features, "
                "but not hidden labels, source task IDs, split groups, or target "
                "outputs."
            ),
            "scoring_contract": (
                "Score the five finite candidate JSON outputs by conditional "
                "log probability and report train, held-out, and bridge-focus "
                "exact accuracy plus violation counts."
            ),
            "environment": smoke_env(),
            "dry_run_preflight_commands": preflight_commands(),
            "submit_template": (
                "Submit only after dry-run validation and explicit full-run approval: "
                "sbatch --account=<allocation> --partition=<gpu-partition> "
                "--gres=gpu:1 --export=ALL,WORK=$PWD,<env-overrides> "
                f"{CAYUGA_SBATCH}"
            ),
        },
        "acceptance_gate": {
            "candidate_model_heldout_exact_min": 5,
            "candidate_model_heldout_rows": 5,
            "candidate_model_bridge_focus_exact_min": 4,
            "candidate_model_bridge_focus_rows": 4,
            "best_static_prior_heldout_exact_to_beat": gate[
                "best_static_prior_heldout_exact"
            ],
            "runtime_evidence_gate_to_match": gate["runtime_evidence_gate"],
            "hidden_labels_used_for_scoring": False,
            "raw_predictions_remain_uncommitted": True,
            "required_compact_outputs": [
                "train_candidate_summary",
                "heldout_candidate_summary",
                "bridge_focus_summary",
                "by_target_pair_summary",
                "violation_counts",
                "meet_or_beat_runtime_gate",
            ],
        },
        "decision": {
            "selected_next_step": "run_evidence_conditioned_candidate_routing_smoke_dry_run_on_cayuga",
            "runner_implemented": (ROOT / RUNNER).exists()
            and (ROOT / CAYUGA_SBATCH).exists(),
            "ready_for_cayuga_dry_run": gate["all_pass"]
            and (ROOT / RUNNER).exists()
            and (ROOT / CAYUGA_SBATCH).exists(),
            "ready_for_cayuga_submission": False,
            "ready_for_tool_query": False,
            "ready_for_dpo_rlvr": False,
            "ready_for_hugging_face_publication": False,
            "ready_for_release_tagging": False,
            "interpretation": (
                "The evidence-conditioned slice is ready for a small Cayuga "
                "dry-run using the implemented runner, but not for unapproved "
                "full submission, DPO/RLVR, tool_query, HF publication, or "
                "release tagging. A model policy must reach 5/5 held-out and "
                "4/4 bridge-focus exact before escalation."
            ),
        },
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
            "hidden_labels_used_for_spec": False,
            "raw_artifacts_committed": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    spec = report["smoke_spec"]
    pre = report["preconditions"]
    gate = report["acceptance_gate"]
    decision = report["decision"]
    lines = [
        "# Stage A Saved-Output Evidence Candidate-Routing Smoke Spec",
        "",
        "Purpose: define the next small Cayuga smoke contract from public-safe candidate-routing artifacts.",
        "",
        "## Preconditions",
        "",
        f"- Rows: {pre['row_counts']['row_count']} total, {pre['row_counts']['train_rows']} train, {pre['row_counts']['heldout_rows']} held-out",
        (
            f"- Runtime evidence gate: {pre['runtime_evidence_gate']['exact']}/"
            f"{pre['runtime_evidence_gate']['rows']} held-out, "
            f"{pre['runtime_evidence_gate']['bridge_focus_exact']}/"
            f"{pre['runtime_evidence_gate']['bridge_focus_rows']} bridge-focus"
        ),
        f"- Best static prior: {pre['best_static_prior_heldout_exact']}/5 held-out",
        f"- All preconditions pass: `{pre['all_pass']}`",
        "",
        "## Smoke Contract",
        "",
        f"- Name: `{spec['name']}`",
        f"- Runner: `{spec['runner']}`",
        f"- Cayuga wrapper: `{spec['cayuga_sbatch']}`",
        f"- Component: `{spec['component']}`",
        f"- Prompt contract: `{spec['prompt_contract']}`",
        f"- Question: {spec['research_question']}",
        f"- Training rows: `{spec['train_rows']['path']}`",
        f"- Held-out rows: `{spec['heldout_rows']['path']}`",
        "",
        "Environment overrides:",
        "",
        "```bash",
    ]
    for key, value in spec["environment"].items():
        lines.append(f"export {key}='{value}'")
    lines.extend(
        [
            "```",
            "",
            "Dry-run preflight:",
            "",
        ]
    )
    for command in spec["dry_run_preflight_commands"]:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "## Acceptance Gate",
            "",
            (
                f"- Candidate model held-out exact minimum: "
                f"{gate['candidate_model_heldout_exact_min']}/"
                f"{gate['candidate_model_heldout_rows']}"
            ),
            (
                f"- Candidate model bridge-focus exact minimum: "
                f"{gate['candidate_model_bridge_focus_exact_min']}/"
                f"{gate['candidate_model_bridge_focus_rows']}"
            ),
            (
                f"- Static prior held-out exact to beat: "
                f"{gate['best_static_prior_heldout_exact_to_beat']}/5"
            ),
            f"- Raw predictions remain uncommitted: `{gate['raw_predictions_remain_uncommitted']}`",
            "",
            "## Decision",
            "",
            f"- Selected next step: `{decision['selected_next_step']}`",
            f"- Runner implemented: `{decision['runner_implemented']}`",
            f"- Ready for Cayuga dry-run: `{decision['ready_for_cayuga_dry_run']}`",
            f"- Ready for Cayuga submission: `{decision['ready_for_cayuga_submission']}`",
            f"- Ready for DPO/RLVR: `{decision['ready_for_dpo_rlvr']}`",
            "",
            decision["interpretation"],
            "",
            "Public-safety contract: raw predictions, candidate-score JSONL,",
            "scheduler logs, model state, and ignored run folders are private run",
            "outputs and are not public artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--readout", default=DEFAULT_READOUT)
    parser.add_argument(
        "--out-json",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_smoke_spec_2026-07-10.json",
    )
    parser.add_argument(
        "--out-md",
        default="post_training/STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_SMOKE_SPEC_2026-07-10.md",
    )
    args = parser.parse_args()

    report = build_report(manifest_path=args.manifest, readout_path=args.readout)
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
