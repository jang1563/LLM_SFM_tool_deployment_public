#!/usr/bin/env python3
"""Evaluate Stage A routing model-readiness against public baselines.

This script reads compact public summaries only. It does not inspect raw
candidate JSONL files, model states, cluster logs, or ignored run folders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_stage_a_strict_component_sft_smoke import write_json  # noqa: E402

DATASET = "negbiodb_ct_stage_a_routing_model_readiness_v1"
COMPONENT = "routing_after_loop"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def policy_summary(baseline: Mapping[str, Any], policy: str, split: str = "heldout") -> Mapping[str, Any]:
    return baseline["policy_reports"][policy][split]["summary"]


def readiness_thresholds(baseline: Mapping[str, Any]) -> dict[str, Any]:
    runtime_heldout = policy_summary(baseline, "runtime_evidence_gate", "heldout")
    collapse_heldout = policy_summary(baseline, "majority_ground_supported", "heldout")
    citationless_heldout = policy_summary(baseline, "routing_no_citations", "heldout")
    runtime_all = policy_summary(baseline, "runtime_evidence_gate", "all")
    return {
        "runtime_gate_all_exact": runtime_all["target_exact"],
        "runtime_gate_all_cases": runtime_all["cases"],
        "runtime_gate_heldout_exact": runtime_heldout["target_exact"],
        "runtime_gate_heldout_cases": runtime_heldout["cases"],
        "collapse_heldout_exact": collapse_heldout["target_exact"],
        "citationless_heldout_exact": citationless_heldout["target_exact"],
        "citationless_heldout_citation_mismatches": citationless_heldout["citation_mismatches_vs_target"],
        "minimum_all_family_model_exact_for_readiness": runtime_heldout["target_exact"],
        "minimum_all_family_model_cases_for_readiness": runtime_heldout["cases"],
    }


def classify_all_family_result(*, exact: int, cases: int, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    collapse = int(thresholds["collapse_heldout_exact"])
    citationless = int(thresholds["citationless_heldout_exact"])
    runtime = int(thresholds["runtime_gate_heldout_exact"])
    return {
        "heldout_exact": exact,
        "heldout_cases": cases,
        "beats_ground_supported_collapse": exact > collapse,
        "beats_citationless_routing": exact > citationless,
        "competitive_with_runtime_gate": exact >= runtime and cases >= int(thresholds["runtime_gate_heldout_cases"]),
        "ready_for_escalation": exact >= runtime and cases >= int(thresholds["runtime_gate_heldout_cases"]),
    }


def all_family_model_results(
    *,
    freeform: Mapping[str, Any],
    constrained: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    freeform_summary = freeform["heldout_summary"]
    constrained_summary = constrained["heldout_summary"]
    constrained_rank = constrained["candidate_rank_summary"]
    freeform_gate = classify_all_family_result(
        exact=int(freeform_summary["passed"]),
        cases=int(freeform_summary["cases"]),
        thresholds=thresholds,
    )
    constrained_gate = classify_all_family_result(
        exact=int(constrained_summary["passed"]),
        cases=int(constrained_summary["cases"]),
        thresholds=thresholds,
    )
    return {
        "freeform_routing_after_loop": {
            "source": "stage_a_evidence_routing_after_loop_cayuga_summary_2026-07-06.json",
            "decode_mode": freeform.get("decode_mode"),
            "model": freeform.get("model"),
            "heldout_summary": freeform_summary,
            "readiness": freeform_gate,
        },
        "constrained_routing_observed_pair": {
            "source": "stage_a_evidence_routing_observed_pair_cayuga_summary_2026-07-08.json",
            "decode_mode": constrained.get("decode_mode"),
            "model": constrained.get("model"),
            "candidate_policy": constrained.get("candidate_policy"),
            "heldout_summary": constrained_summary,
            "candidate_rank_summary": {
                "exact_top1": constrained_rank["exact_top1"],
                "action_status_top1": constrained_rank["action_status_top1"],
                "citation_required_cases": constrained_rank["citation_required_cases"],
                "citation_required_exact_top1": constrained_rank["citation_required_exact_top1"],
                "mean_gold_rank_observed": constrained_rank["mean_gold_rank_observed"],
                "top_pair_counts": constrained_rank["top_pair_counts"],
            },
            "readiness": constrained_gate,
        },
    }


def targeted_model_results(
    *,
    contrast_candidate: Mapping[str, Any],
    defer_verify: Mapping[str, Any],
    fail_closed_gate: Mapping[str, Any],
) -> dict[str, Any]:
    trained_candidate = contrast_candidate["trained_candidate_summary"]
    base_candidate = contrast_candidate["base_candidate_summary"]
    trained_defer_verify = defer_verify["trained_routing_candidates"]
    base_defer_verify = defer_verify["base_routing_candidates"]
    fail_closed = fail_closed_gate["best_default_zero_unsafe_report"]
    return {
        "routing_contrast_candidate_subset": {
            "source": "stage_a_routing_contrast_candidate_cayuga_summary_2026-07-08.json",
            "scope": "three unresolved routing contrast held-out families",
            "base_exact_top1": base_candidate["exact_top1"],
            "trained_exact_top1": trained_candidate["exact_top1"],
            "cases": trained_candidate["cases"],
            "mean_gold_rank_observed": trained_candidate["mean_gold_rank_observed"],
            "top_pair_counts": trained_candidate["top_pair_counts"],
            "all_family_readiness_evidence": False,
        },
        "defer_verify_candidate_subset": {
            "source": "stage_a_routing_defer_verify_cayuga_summary_2026-07-08.json",
            "scope": "two-case defer-vs-verify boundary",
            "base_exact_top1": base_defer_verify["exact_top1"],
            "trained_exact_top1": trained_defer_verify["exact_top1"],
            "cases": trained_defer_verify["cases"],
            "top_pair_counts": trained_defer_verify["top_pair_counts"],
            "all_family_readiness_evidence": False,
        },
        "defer_verify_fail_closed_gate_subset": {
            "source": "stage_a_routing_defer_verify_gate_trained_2026-07-08.json",
            "scope": "two-case defer-vs-verify boundary with score-gap fail-closed routing",
            "threshold": fail_closed["threshold"],
            "trusted": fail_closed["trusted"],
            "fail_closed": fail_closed["fail_closed"],
            "trusted_incorrect": fail_closed["trusted_incorrect"],
            "strict_final_correct": fail_closed["strict_final_correct"],
            "cases": fail_closed_gate["cases"],
            "strict_final_accuracy": fail_closed["strict_final_accuracy"],
            "all_family_readiness_evidence": False,
        },
    }


def build_decision(
    *,
    all_family_results: Mapping[str, Any],
    targeted_results: Mapping[str, Any],
) -> dict[str, Any]:
    constrained = all_family_results["constrained_routing_observed_pair"]["readiness"]
    freeform = all_family_results["freeform_routing_after_loop"]["readiness"]
    fail_closed = targeted_results["defer_verify_fail_closed_gate_subset"]
    blockers = []
    if not constrained["beats_citationless_routing"]:
        blockers.append("best all-family model readout is 2/5, below citationless routing at 3/5")
    if not constrained["competitive_with_runtime_gate"]:
        blockers.append("best all-family model readout is below runtime evidence gate at 5/5")
    if not freeform["beats_ground_supported_collapse"]:
        blockers.append("free-form routing is 0/5 and does not beat collapse baseline")
    if fail_closed["all_family_readiness_evidence"] is False:
        blockers.append("score-gap fail-closed result is only a two-case boundary diagnostic")
    return {
        "ready_for_tool_query": False,
        "ready_for_dpo_rlvr": False,
        "ready_for_hugging_face_publication": False,
        "ready_for_release_tagging": False,
        "runtime_enforcement_required": True,
        "blockers": blockers,
        "next_decision": (
            "Keep tool_query, DPO/RLVR, Hugging Face publication, and release "
            "tagging gated. Next compare saved model/component outputs against "
            "the all-family runtime gate or wrap the gate into full trajectory "
            "arbitration before adding optimization objectives."
        ),
    }


def build_report(
    *,
    baseline: Mapping[str, Any],
    freeform: Mapping[str, Any],
    constrained: Mapping[str, Any],
    contrast_candidate: Mapping[str, Any],
    defer_verify: Mapping[str, Any],
    fail_closed_gate: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = readiness_thresholds(baseline)
    all_family_results = all_family_model_results(
        freeform=freeform,
        constrained=constrained,
        thresholds=thresholds,
    )
    targeted_results = targeted_model_results(
        contrast_candidate=contrast_candidate,
        defer_verify=defer_verify,
        fail_closed_gate=fail_closed_gate,
    )
    return {
        "dataset": DATASET,
        "component": COMPONENT,
        "inputs": {
            "baseline_comparison": "post_training/stage_a_routing_gate_baseline_comparison_2026-07-09.json",
            "freeform_routing": "post_training/stage_a_evidence_routing_after_loop_cayuga_summary_2026-07-06.json",
            "constrained_routing": "post_training/stage_a_evidence_routing_observed_pair_cayuga_summary_2026-07-08.json",
            "routing_contrast_candidate": "post_training/stage_a_routing_contrast_candidate_cayuga_summary_2026-07-08.json",
            "defer_verify": "post_training/stage_a_routing_defer_verify_cayuga_summary_2026-07-08.json",
            "defer_verify_fail_closed": "post_training/stage_a_routing_defer_verify_gate_trained_2026-07-08.json",
        },
        "raw_outputs_used": False,
        "readiness_thresholds": thresholds,
        "all_family_model_results": all_family_results,
        "targeted_model_results": targeted_results,
        "decision": build_decision(all_family_results=all_family_results, targeted_results=targeted_results),
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    thresholds = report["readiness_thresholds"]
    all_family = report["all_family_model_results"]
    targeted = report["targeted_model_results"]
    lines = [
        "# Stage A Routing Model Readiness",
        "",
        "Purpose: compare compact Cayuga routing summaries against the public",
        "runtime/baseline scorecard before reopening `tool_query`, DPO/RLVR,",
        "Hugging Face publication, or release tagging.",
        "",
        "## Baselines",
        "",
        f"- Runtime gate held-out exact: {thresholds['runtime_gate_heldout_exact']}/{thresholds['runtime_gate_heldout_cases']}",
        f"- Collapse held-out exact: {thresholds['collapse_heldout_exact']}/{thresholds['runtime_gate_heldout_cases']}",
        f"- Citationless held-out exact: {thresholds['citationless_heldout_exact']}/{thresholds['runtime_gate_heldout_cases']}",
        "",
        "## All-Family Model Readouts",
        "",
        "| Readout | Exact | Mean score | Beats collapse | Beats citationless | Competitive with runtime gate |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for label, result in all_family.items():
        summary = result["heldout_summary"]
        readiness = result["readiness"]
        lines.append(
            "| `{label}` | {passed}/{cases} | {mean:.3f} | {collapse} | {citationless} | {runtime} |".format(
                label=label,
                passed=summary["passed"],
                cases=summary["cases"],
                mean=float(summary["mean_score"]),
                collapse=str(readiness["beats_ground_supported_collapse"]),
                citationless=str(readiness["beats_citationless_routing"]),
                runtime=str(readiness["competitive_with_runtime_gate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Targeted Diagnostics",
            "",
            "| Diagnostic | Result | Scope note |",
            "| --- | --- | --- |",
            (
                "| `routing_contrast_candidate_subset` | "
                f"{targeted['routing_contrast_candidate_subset']['trained_exact_top1']}/"
                f"{targeted['routing_contrast_candidate_subset']['cases']} exact top-1 | "
                "Targeted 3-case subset; not all-family readiness evidence |"
            ),
            (
                "| `defer_verify_candidate_subset` | "
                f"{targeted['defer_verify_candidate_subset']['trained_exact_top1']}/"
                f"{targeted['defer_verify_candidate_subset']['cases']} exact top-1 | "
                "Targeted 2-case boundary; not all-family readiness evidence |"
            ),
            (
                "| `defer_verify_fail_closed_gate_subset` | "
                f"{targeted['defer_verify_fail_closed_gate_subset']['strict_final_correct']}/"
                f"{targeted['defer_verify_fail_closed_gate_subset']['cases']} strict final correct | "
                "Useful boundary gate diagnostic; not all-family model readiness evidence |"
            ),
            "",
            "## Decision",
            "",
            f"- Ready for `tool_query`: `{report['decision']['ready_for_tool_query']}`",
            f"- Ready for DPO/RLVR: `{report['decision']['ready_for_dpo_rlvr']}`",
            f"- Runtime enforcement required: `{report['decision']['runtime_enforcement_required']}`",
            "",
            "Blockers:",
        ]
    )
    for blocker in report["decision"]["blockers"]:
        lines.append(f"- {blocker}")
    lines.extend(["", report["decision"]["next_decision"]])
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="post_training/stage_a_routing_gate_baseline_comparison_2026-07-09.json",
    )
    parser.add_argument(
        "--freeform",
        default="post_training/stage_a_evidence_routing_after_loop_cayuga_summary_2026-07-06.json",
    )
    parser.add_argument(
        "--constrained",
        default="post_training/stage_a_evidence_routing_observed_pair_cayuga_summary_2026-07-08.json",
    )
    parser.add_argument(
        "--contrast-candidate",
        default="post_training/stage_a_routing_contrast_candidate_cayuga_summary_2026-07-08.json",
    )
    parser.add_argument(
        "--defer-verify",
        default="post_training/stage_a_routing_defer_verify_cayuga_summary_2026-07-08.json",
    )
    parser.add_argument(
        "--fail-closed-gate",
        default="post_training/stage_a_routing_defer_verify_gate_trained_2026-07-08.json",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(
        baseline=load_json(args.baseline),
        freeform=load_json(args.freeform),
        constrained=load_json(args.constrained),
        contrast_candidate=load_json(args.contrast_candidate),
        defer_verify=load_json(args.defer_verify),
        fail_closed_gate=load_json(args.fail_closed_gate),
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
