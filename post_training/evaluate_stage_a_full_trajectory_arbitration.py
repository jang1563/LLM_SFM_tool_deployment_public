#!/usr/bin/env python3
"""Evaluate public-safe Stage A full-trajectory arbitration policies.

This no-model report projects the routing evidence gate back into the canonical
TaskSpec / Trajectory / EvidencePacket evaluator. It asks whether runtime
arbitration can rescue full trajectory failures before any new tool_query,
DPO/RLVR, Hugging Face publication, or release tagging is justified.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import Action, EvidencePacket, Trajectory  # noqa: E402
from negbiodb_ct.stage_a_manifest import score_stage_a_trajectory  # noqa: E402
from post_training.evaluate_stage_a_routing_evidence_gate import (  # noqa: E402
    evidence_features,
    gate_output,
)
from post_training.run_stage_a_sft_smoke_eval import (  # noqa: E402
    load_jsonl,
    load_manifest_rows,
    summarize_rows,
    trajectory_from_payload,
    write_json,
)
from post_training.run_stage_a_strict_component_sft_smoke import filter_component  # noqa: E402

DATASET = "negbiodb_ct_stage_a_full_trajectory_arbitration_v1"
COMPONENT = "routing_after_loop"
POLICIES = (
    "oracle_full",
    "self_answer",
    "ground_supported_collapse",
    "citationless_runtime_action",
    "runtime_gate_full",
    "hybrid_runtime_over_collapse",
)

ACTION_TO_TERMINAL = {
    "ground": Action.GROUND_WITH_ATTRIBUTION.value,
    "reject": Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM.value,
    "flag": Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM.value,
    "defer": Action.DEFER_OR_REQUEST_MORE_EVIDENCE.value,
    "verify": Action.VERIFY_WITH_ASSAY_OR_DATABASE.value,
    "self_answer": Action.ANSWER_SELF.value,
}


def case_id(row: Mapping[str, Any]) -> str:
    return str(row.get("source_manifest_case_id") or row.get("task_id") or row.get("case_id"))


def hidden_status(manifest_row: Mapping[str, Any]) -> str:
    return str(manifest_row["hidden_eval_metadata"]["gold_evidence_status"])


def hidden_terminal(manifest_row: Mapping[str, Any]) -> str:
    return str(manifest_row["hidden_eval_metadata"]["expected_terminal_action"])


def hidden_sources(manifest_row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(source) for source in manifest_row["hidden_eval_metadata"].get("gold_source_ids", ()))


def target_trajectory(row: Mapping[str, Any]) -> Trajectory:
    return trajectory_from_payload(row["target_trajectory"], target_input_id=case_id(row))


def gate_outputs_by_case(component_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in filter_component(list(component_rows), COMPONENT):
        prediction, reason = gate_output(evidence_features(row))
        out[str(row["source_manifest_case_id"])] = {
            "prediction": prediction,
            "reason": reason,
        }
    return out


def trajectory_with_decision(
    base: Trajectory,
    *,
    action: str,
    evidence_status: str,
    cited_source_ids: Sequence[str],
    fail_closed_reason: str | None = None,
) -> Trajectory:
    return replace(
        base,
        evidence_packet=replace(
            base.evidence_packet,
            negative_evidence_status=evidence_status,
            fail_closed_reason=fail_closed_reason,
        ),
        terminal_action=ACTION_TO_TERMINAL[action],
        cited_source_ids=tuple(str(source) for source in cited_source_ids),
        predicted_evidence_status=evidence_status,
    )


def self_answer_trajectory(base: Trajectory) -> Trajectory:
    return Trajectory(
        input_id=base.input_id,
        steps=(),
        evidence_packet=EvidencePacket(
            input_id=base.input_id,
            representation_type="drug_indication_claim",
            negative_evidence_status="unknown",
            claim_guard_status="unchecked",
        ),
        terminal_action=Action.ANSWER_SELF.value,
        cited_source_ids=(),
        predicted_evidence_status="unknown",
    )


def policy_trajectory(
    *,
    policy: str,
    sft_row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    gate_by_case: Mapping[str, Mapping[str, Any]],
) -> tuple[Trajectory, str]:
    base = target_trajectory(sft_row)
    current_case = case_id(sft_row)
    gate_payload = gate_by_case[current_case]
    gate_prediction = gate_payload["prediction"]
    gate_reason = str(gate_payload["reason"])

    if policy == "oracle_full":
        return base, "oracle_target_trajectory"
    if policy == "self_answer":
        return self_answer_trajectory(base), "no_external_tool_or_evidence_packet"
    if policy == "ground_supported_collapse":
        citations = hidden_sources(manifest_row) if hidden_sources(manifest_row) else ()
        return trajectory_with_decision(
            base,
            action="ground",
            evidence_status="supported",
            cited_source_ids=citations,
        ), "model_like_ground_supported_collapse"
    if policy == "citationless_runtime_action":
        return trajectory_with_decision(
            base,
            action=str(gate_prediction["action"]),
            evidence_status=str(gate_prediction["evidence_status"]),
            cited_source_ids=(),
        ), "runtime_action_status_without_evidence_packet_citations"
    if policy == "runtime_gate_full":
        return trajectory_with_decision(
            base,
            action=str(gate_prediction["action"]),
            evidence_status=str(gate_prediction["evidence_status"]),
            cited_source_ids=gate_prediction.get("cited_source_ids", ()),
            fail_closed_reason=gate_reason,
        ), gate_reason
    if policy == "hybrid_runtime_over_collapse":
        # The model-like proposal collapses to ground/supported, then runtime
        # arbitration overrides the final action/status/citations from visible
        # evidence before submission.
        return trajectory_with_decision(
            base,
            action=str(gate_prediction["action"]),
            evidence_status=str(gate_prediction["evidence_status"]),
            cited_source_ids=gate_prediction.get("cited_source_ids", ()),
            fail_closed_reason=f"runtime_override_after_collapse:{gate_reason}",
        ), f"runtime_override_after_collapse:{gate_reason}"
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_policy_rows(
    *,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    manifest_by_case: Mapping[str, Mapping[str, Any]],
    gate_by_case: Mapping[str, Mapping[str, Any]],
    policy: str,
) -> dict[str, Any]:
    row_reports = []
    unsafe_ground_supported = 0
    for row in rows:
        current_case = case_id(row)
        manifest_row = manifest_by_case[current_case]
        trajectory, arbitration_reason = policy_trajectory(
            policy=policy,
            sft_row=row,
            manifest_row=manifest_row,
            gate_by_case=gate_by_case,
        )
        result = score_stage_a_trajectory(manifest_row, trajectory)
        predicted_pair = f"{trajectory.predicted_evidence_status}/{trajectory.terminal_action}"
        if (
            trajectory.terminal_action == Action.GROUND_WITH_ATTRIBUTION.value
            and hidden_terminal(manifest_row) != Action.GROUND_WITH_ATTRIBUTION.value
        ):
            unsafe_ground_supported += 1
        row_reports.append(
            {
                "case_id": current_case,
                "split": split,
                "case_family": str(row.get("case_family")),
                "gold_evidence_status": hidden_status(manifest_row),
                "expected_terminal_action": hidden_terminal(manifest_row),
                "predicted_evidence_status": str(trajectory.predicted_evidence_status),
                "predicted_terminal_action": str(trajectory.terminal_action),
                "predicted_pair": predicted_pair,
                "arbitration_reason": arbitration_reason,
                "score": round(result.score, 3),
                "passed": result.passed,
                "reward_breakdown": dict(result.reward_breakdown),
                "violations": list(result.violations),
            }
        )
    summary = summarize_rows(row_reports)
    summary["unsafe_ground_supported_overrides"] = unsafe_ground_supported
    return {
        "summary": summary,
        "rows": row_reports,
    }


def evaluate_split(
    *,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    manifest_by_case: Mapping[str, Mapping[str, Any]],
    gate_by_case: Mapping[str, Mapping[str, Any]],
    policies: Sequence[str],
) -> dict[str, Any]:
    return {
        policy: evaluate_policy_rows(
            split=split,
            rows=rows,
            manifest_by_case=manifest_by_case,
            gate_by_case=gate_by_case,
            policy=policy,
        )
        for policy in policies
    }


def build_report(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    policies: Sequence[str] = POLICIES,
) -> dict[str, Any]:
    manifest_by_case = {str(row["case_id"]): row for row in manifest_rows}
    gate_by_case = gate_outputs_by_case(component_rows)
    all_rows = list(train_rows) + list(heldout_rows)
    splits = {
        "all": evaluate_split(
            split="all",
            rows=all_rows,
            manifest_by_case=manifest_by_case,
            gate_by_case=gate_by_case,
            policies=policies,
        ),
        "train": evaluate_split(
            split="train",
            rows=train_rows,
            manifest_by_case=manifest_by_case,
            gate_by_case=gate_by_case,
            policies=policies,
        ),
        "heldout": evaluate_split(
            split="heldout",
            rows=heldout_rows,
            manifest_by_case=manifest_by_case,
            gate_by_case=gate_by_case,
            policies=policies,
        ),
    }
    return {
        "dataset": DATASET,
        "component_gate_source": "stage_a_evidence_conditioned_component_targets_v1",
        "canonical_evaluator": "negbiodb_ct.stage_a_manifest.score_stage_a_trajectory",
        "raw_model_outputs_used": False,
        "policies": list(policies),
        "cases": {
            "all": len(all_rows),
            "train": len(train_rows),
            "heldout": len(heldout_rows),
        },
        "splits": splits,
        "summary": {
            split: {policy: payload["summary"] for policy, payload in split_payload.items()}
            for split, split_payload in splits.items()
        },
        "scientific_readout": {
            "diagnostic_question": (
                "Can the routing runtime gate be lifted into full Stage A "
                "trajectory arbitration without forking the evaluator schema?"
            ),
            "interpretation_rule": (
                "Runtime arbitration may rescue model-like collapse only when "
                "the final action, evidence status, citation packet, and tool "
                "query trajectory all pass the canonical evaluator."
            ),
            "next_decision": (
                "Use this scaffold for saved model outputs next. Keep tool_query, "
                "DPO/RLVR, Hugging Face publication, and release tagging gated "
                "until real saved trajectories beat citationless/collapse "
                "baselines and approach runtime-gate full-trajectory performance."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Stage A Full-Trajectory Arbitration",
        "",
        "Purpose: project the routing runtime gate into the canonical Stage A",
        "trajectory evaluator and compare oracle, collapse, citationless,",
        "runtime-gate, and hybrid runtime-over-model policies.",
        "",
        "## Summary",
        "",
        "| Policy | All | Train | Held-out | Unsafe ground/supported overrides (all) |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for policy in report["policies"]:
        all_summary = report["summary"]["all"][policy]
        cells = []
        for split in ("all", "train", "heldout"):
            item = report["summary"][split][policy]
            cells.append(f"{item['passed']}/{item['cases']} pass; mean {item['mean_score']:.3f}")
        lines.append(
            "| `{policy}` | {all_cell} | {train_cell} | {heldout_cell} | {unsafe} |".format(
                policy=policy,
                all_cell=cells[0],
                train_cell=cells[1],
                heldout_cell=cells[2],
                unsafe=all_summary["unsafe_ground_supported_overrides"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(report["scientific_readout"]["interpretation_rule"]),
            "",
            str(report["scientific_readout"]["next_decision"]),
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument("--train-sft", default="post_training/stage_a_sft_train_v1.jsonl")
    parser.add_argument("--heldout-sft", default="post_training/stage_a_sft_heldout_v1.jsonl")
    parser.add_argument(
        "--component-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(
        manifest_rows=load_manifest_rows(args.manifest),
        train_rows=load_jsonl(args.train_sft),
        heldout_rows=load_jsonl(args.heldout_sft),
        component_rows=load_jsonl(args.component_targets),
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    stdout_report = {
        "dataset": report["dataset"],
        "canonical_evaluator": report["canonical_evaluator"],
        "raw_model_outputs_used": report["raw_model_outputs_used"],
        "cases": report["cases"],
        "summary": report["summary"],
        "next_decision": report["scientific_readout"]["next_decision"],
    }
    print(json.dumps(stdout_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
