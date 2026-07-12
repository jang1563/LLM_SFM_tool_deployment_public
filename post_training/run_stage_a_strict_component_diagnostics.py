#!/usr/bin/env python3
"""Decompose strict Stage A held-out failures into scoreable components.

This harness uses the same saved-prediction scorer as API, prompt-only, and
cluster SFT artifacts. It creates deterministic counterfactual prediction
variants from the strict-contract held-out targets to separate:

- enum/action contract failures;
- ordered tool-loop and query-argument failures;
- evidence-status and terminal-action routing failures.

The script performs no live API calls and loads no model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.evaluate_stage_a_predictions import (  # noqa: E402
    ACTION_CLASS_DEFAULT_STATUS,
    build_report as build_prediction_report,
    expected_case_ids_from_rows,
)
from post_training.run_stage_a_sft_smoke_eval import load_manifest_rows  # noqa: E402
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    load_jsonl,
    source_case_id,
    target_output_from_row,
    write_json,
)


DATASET = "negbiodb_ct_stage_a_strict_component_diagnostics_v1"
PROMPT_CONTRACT = "stage_a_v2_strict"


def _clone_target(row: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(target_output_from_row(row))


def _tool_names(row: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for call in _clone_target(row).get("tool_calls", ()):
        if isinstance(call, Mapping):
            names.append(str(call["name"]))
        else:
            names.append(str(call))
    return names


def oracle_full(row: Mapping[str, Any]) -> dict[str, Any]:
    """Positive control: all strict-contract components are correct."""

    return _clone_target(row)


def invalid_enum_verified(row: Mapping[str, Any]) -> dict[str, Any]:
    """Only break enum validity, mirroring the observed `verified` error."""

    output = _clone_target(row)
    output["evidence_status"] = "verified"
    return output


def enum_constrained_from_action(row: Mapping[str, Any]) -> dict[str, Any]:
    """Repair enum status by mapping the action to the allowed status set."""

    output = _clone_target(row)
    action = str(output["action"])
    output["evidence_status"] = ACTION_CLASS_DEFAULT_STATUS[action]
    return output


def route_only_correct_no_tools(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep routing correct while removing the tool loop entirely."""

    output = _clone_target(row)
    output["tool_calls"] = []
    return output


def ordered_tool_names_only(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep ordered tool names but remove structured query arguments."""

    output = _clone_target(row)
    output["tool_calls"] = _tool_names(row)
    return output


def tool_loop_with_ground_route(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the full tool loop but collapse routing to ground/supported."""

    output = _clone_target(row)
    output["action"] = "ground"
    output["evidence_status"] = "supported"
    return output


VariantFn = Callable[[Mapping[str, Any]], dict[str, Any]]


VARIANTS: tuple[tuple[str, str, VariantFn], ...] = (
    (
        "oracle_full",
        "Positive control with correct enum, full structured tool loop, citations, and routing.",
        oracle_full,
    ),
    (
        "invalid_enum_verified",
        "Only the evidence_status enum is invalid; this should fail at parse time.",
        invalid_enum_verified,
    ),
    (
        "enum_constrained_from_action",
        "Allowed evidence_status is derived from the action while other target fields stay fixed.",
        enum_constrained_from_action,
    ),
    (
        "route_only_correct_no_tools",
        "Correct evidence/action/citations with no external tool loop.",
        route_only_correct_no_tools,
    ),
    (
        "ordered_tool_names_only",
        "Correct ordered tool names without required drug_id/condition_id arguments.",
        ordered_tool_names_only,
    ),
    (
        "tool_loop_with_ground_route",
        "Full structured tool loop with a collapsed ground/supported route.",
        tool_loop_with_ground_route,
    ),
)


def prediction_rows_for_variant(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant_name: str,
    variant_fn: VariantFn,
) -> list[dict[str, Any]]:
    """Build saved-prediction rows in the existing scorer input format."""

    predictions: list[dict[str, Any]] = []
    for row in rows:
        predictions.append(
            {
                "case_id": source_case_id(row),
                "dataset": DATASET,
                "prompt_contract": PROMPT_CONTRACT,
                "source": f"component_diagnostic::{variant_name}",
                "split": row.get("split", "heldout"),
                "prediction": variant_fn(row),
            }
        )
    return predictions


def compact_variant_report(report: Mapping[str, Any], *, include_rows: bool) -> dict[str, Any]:
    """Keep the nested scorer output compact and public-release friendly."""

    payload: dict[str, Any] = {
        "summary": report["summary"],
        "parse_errors": report["parse_errors"],
    }
    if include_rows:
        payload["rows"] = report["rows"]
    return payload


def build_component_report(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    include_rows: bool = True,
) -> dict[str, Any]:
    """Evaluate all strict Stage A component diagnostics."""

    expected_case_ids = expected_case_ids_from_rows(heldout_rows)
    variants: dict[str, dict[str, Any]] = {}
    for variant_name, description, variant_fn in VARIANTS:
        scorer_report = build_prediction_report(
            manifest_rows=manifest_rows,
            prediction_rows=prediction_rows_for_variant(
                heldout_rows,
                variant_name=variant_name,
                variant_fn=variant_fn,
            ),
            expected_case_ids=expected_case_ids,
            run_id=f"stage_a_strict_component::{variant_name}",
        )
        variants[variant_name] = {
            "description": description,
            **compact_variant_report(scorer_report, include_rows=include_rows),
        }

    return {
        "dataset": DATASET,
        "prompt_contract": PROMPT_CONTRACT,
        "boundary": (
            "No-API, no-model diagnostic harness. Counterfactual rows are "
            "derived from strict held-out targets and scored by the same "
            "offline trajectory evaluator used for model artifacts."
        ),
        "cases_expected": len(expected_case_ids),
        "variants": variants,
        "diagnostic_readout": {
            "enum_contract": (
                "`invalid_enum_verified` isolates parse-time enum failure; "
                "`enum_constrained_from_action` is the constrained-decoder upper bound."
            ),
            "tool_loop": (
                "`route_only_correct_no_tools` shows correct routing is not enough; "
                "`ordered_tool_names_only` separates ordered tool selection from "
                "required query arguments."
            ),
            "routing": (
                "`tool_loop_with_ground_route` shows a full structured tool loop is "
                "not enough when evidence/action routing collapses."
            ),
        },
        "next_decision": (
            "Do not escalate to DPO or RLVR until enum/action decoding, ordered "
            "tool calls with required query fields, and evidence/action routing "
            "are each measurable on held-out Stage A cases."
        ),
    }


def print_summary(report: Mapping[str, Any]) -> None:
    print("variant                       cases  passed  mean_score")
    print("----------------------------  -----  ------  ----------")
    for variant_name, _, _ in VARIANTS:
        summary = report["variants"][variant_name]["summary"]
        print(
            f"{variant_name:<28}  "
            f"{summary['cases']:<5}  "
            f"{summary['passed']:<6}  "
            f"{summary['mean_score']:<10.3f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument(
        "--heldout-sft",
        default="post_training/stage_a_strict_contract_sft_heldout_v1.jsonl",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit per-case rows from the written or printed JSON report.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_component_report(
        manifest_rows=load_manifest_rows(args.manifest),
        heldout_rows=load_jsonl(args.heldout_sft),
        include_rows=not args.compact,
    )
    if args.out:
        write_json(args.out, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_summary(report)


if __name__ == "__main__":
    main()
