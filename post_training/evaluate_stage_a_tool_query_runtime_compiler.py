#!/usr/bin/env python3
"""Evaluate fail-closed runtime compilation for Stage A tool queries."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from negbiodb_ct.tool_query_runtime import (  # noqa: E402
    ToolQueryContractError,
    compile_tool_query,
)
from post_training.evaluate_stage_a_prospective_runtime_hybrid import (  # noqa: E402
    load_json,
    sha256_file,
)
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    load_jsonl,
)


DATASET = "negbiodb_ct_stage_a_tool_query_runtime_compiler_eval_v1"
Mutation = Callable[[dict[str, Any]], None]


def remove_condition(task: dict[str, Any]) -> None:
    task["query"].pop("condition_id")


def wrong_drug_namespace(task: dict[str, Any]) -> None:
    task["query"]["drug_id"]["namespace"] = "negbiodb_ct.condition_id"


def stringify_drug_id(task: dict[str, Any]) -> None:
    value = task["query"]["drug_id"]["value"]
    task["query"]["drug_id"]["value"] = str(value)


def reorder_tools(task: dict[str, Any]) -> None:
    task["allowed_tools"] = list(reversed(task["allowed_tools"]))


def insert_unapproved_tool(task: dict[str, Any]) -> None:
    task["allowed_tools"][-1] = "nullatlas_unapproved_lookup"


def add_query_field(task: dict[str, Any]) -> None:
    task["query"]["endpoint_id"] = {
        "namespace": "negbiodb_ct.endpoint_id",
        "value": 1,
    }


MUTATIONS: dict[str, tuple[Mutation, str]] = {
    "missing_condition": (remove_condition, "query_field_set_mismatch"),
    "wrong_drug_namespace": (
        wrong_drug_namespace,
        "drug_id_namespace_mismatch",
    ),
    "stringified_drug_id": (
        stringify_drug_id,
        "drug_id_value_not_positive_integer",
    ),
    "reordered_tools": (
        reorder_tools,
        "tool_sequence_contract_mismatch",
    ),
    "unapproved_tool": (
        insert_unapproved_tool,
        "tool_sequence_contract_mismatch",
    ),
    "extra_query_field": (add_query_field, "query_field_set_mismatch"),
}


def evaluate_compiler(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clean_exact = 0
    clean_errors: Counter[str] = Counter()
    mutation_counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = row.get("model_visible_task")
        if not isinstance(task, Mapping):
            clean_errors["task_not_object"] += 1
            continue
        try:
            output = compile_tool_query(task)
        except ToolQueryContractError as exc:
            clean_errors[exc.code] += 1
        else:
            clean_exact += output == row.get("target_output")

    for name, (mutate, expected_code) in MUTATIONS.items():
        rejected = 0
        intended_reason = 0
        accepted = 0
        reasons: Counter[str] = Counter()
        for row in rows:
            task = copy.deepcopy(row["model_visible_task"])
            mutate(task)
            try:
                compile_tool_query(task)
            except ToolQueryContractError as exc:
                rejected += 1
                intended_reason += exc.code == expected_code
                reasons[exc.code] += 1
            else:
                accepted += 1
        mutation_counts[name] = {
            "rows": len(rows),
            "rejected": rejected,
            "accepted": accepted,
            "intended_reason": intended_reason,
            "expected_reason": expected_code,
            "observed_reasons": dict(sorted(reasons.items())),
        }

    total_mutations = len(rows) * len(MUTATIONS)
    total_rejected = sum(
        value["rejected"] for value in mutation_counts.values()
    )
    intended_reasons = sum(
        value["intended_reason"] for value in mutation_counts.values()
    )
    return {
        "clean": {
            "rows": len(rows),
            "exact": clean_exact,
            "accuracy": round(clean_exact / len(rows), 6) if rows else 0.0,
            "errors": dict(sorted(clean_errors.items())),
        },
        "malformed": {
            "rows": total_mutations,
            "rejected": total_rejected,
            "fail_closed_rate": (
                round(total_rejected / total_mutations, 6)
                if total_mutations
                else 0.0
            ),
            "intended_reason": intended_reasons,
            "intended_reason_rate": (
                round(intended_reasons / total_mutations, 6)
                if total_mutations
                else 0.0
            ),
            "by_mutation": mutation_counts,
        },
    }


def compact_model_results(
    transfer: Mapping[str, Any],
    prompt_repair: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = (
        "parseable_json",
        "target_keys",
        "tool_query_shape",
        "tool_sequence",
        "query_fields",
        "query_values",
        "exact",
    )
    policies = transfer["policies"]
    return {
        "base_minimal_contract": {
            metric: policies["base"][metric] for metric in metrics
        },
        "frozen_placeholder_sft": {
            metric: policies["frozen_tool_query_sft"][metric]
            for metric in metrics
        },
        "base_explicit_contract": {
            metric: prompt_repair["policy"][metric] for metric in metrics
        },
    }


def build_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_path: str | Path,
    transfer_result: Mapping[str, Any],
    transfer_result_path: str | Path,
    prompt_repair_result: Mapping[str, Any],
    prompt_repair_result_path: str | Path,
) -> dict[str, Any]:
    compiler = evaluate_compiler(rows)
    model_results = compact_model_results(
        transfer_result,
        prompt_repair_result,
    )
    clean_pass = compiler["clean"]["exact"] == len(rows)
    malformed_pass = (
        compiler["malformed"]["rejected"] == compiler["malformed"]["rows"]
        and compiler["malformed"]["intended_reason"]
        == compiler["malformed"]["rows"]
    )
    return {
        "dataset": DATASET,
        "evaluation_scope": (
            "public_development_deterministic_tool_query_compilation"
        ),
        "input_artifacts": {
            "tool_query_rows": {
                "path": str(Path(rows_path)),
                "records": len(rows),
                "sha256": sha256_file(rows_path),
            },
            "model_transfer_result": {
                "path": str(Path(transfer_result_path)),
                "sha256": sha256_file(transfer_result_path),
            },
            "prompt_repair_result": {
                "path": str(Path(prompt_repair_result_path)),
                "sha256": sha256_file(prompt_repair_result_path),
            },
        },
        "model_results": model_results,
        "runtime_compiler": compiler,
        "decision": {
            "runtime_compiler_clean_exact": clean_pass,
            "runtime_compiler_fail_closed": malformed_pass,
            "runtime_compiler_approved_for_stage_a": clean_pass and malformed_pass,
            "corrective_sft_selected": False,
            "reason": (
                "The current fixed-order copy-only query operation is a "
                "deterministic runtime contract, not a learned decision."
            ),
            "ready_for_dpo_rlvr": False,
            "ready_for_hugging_face_publication": False,
        },
        "scientific_boundary": {
            "development_only": True,
            "independent_test_claimed": False,
            "deterministic_mutations_define_expected_rejections": True,
            "live_tool_execution_evaluated": False,
            "completed_sealed_rows_used": False,
            "llm_judge_used": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    model = report["model_results"]
    compiler = report["runtime_compiler"]
    lines = [
        "# Stage A Tool-Query Runtime Compiler Evaluation",
        "",
        "The current Stage A query step uses a fixed four-tool sequence and copies",
        "two typed visible identifiers. This report treats it as a fail-closed",
        "runtime contract rather than a learned policy.",
        "",
        "## Model Diagnostics",
        "",
        "| Policy | Parseable | Target keys | Shape | Query values | Exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in model.items():
        lines.append(
            f"| `{name}` | {summary['parseable_json']}/25 | "
            f"{summary['target_keys']}/25 | {summary['tool_query_shape']}/25 | "
            f"{summary['query_values']}/25 | {summary['exact']}/25 |"
        )
    lines.extend(
        [
            "",
            "## Runtime Result",
            "",
            (
                f"- Clean exact compilation: `{compiler['clean']['exact']}/"
                f"{compiler['clean']['rows']}`."
            ),
            (
                f"- Malformed inputs rejected: `{compiler['malformed']['rejected']}/"
                f"{compiler['malformed']['rows']}`."
            ),
            (
                "- Rejections used the intended contract reason: "
                f"`{compiler['malformed']['intended_reason']}/"
                f"{compiler['malformed']['rows']}`."
            ),
            "- Corrective SFT is not selected for this deterministic operation.",
            "- DPO/RLVR and Hugging Face publication remain closed.",
            "",
            "This is a public-development systems result, not a sealed-test estimate",
            "or a live-tool execution result.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        default="post_training/stage_a_prospective_real_query_tool_query_v1.jsonl",
    )
    parser.add_argument(
        "--transfer-result",
        default=(
            "post_training/"
            "stage_a_prospective_tool_query_transfer_result_qwen05b_cayuga_2026-07-23.json"
        ),
    )
    parser.add_argument(
        "--prompt-repair-result",
        default=(
            "post_training/"
            "stage_a_prospective_tool_query_prompt_repair_result_qwen05b_cayuga_2026-07-23.json"
        ),
    )
    parser.add_argument(
        "--out-json",
        default=(
            "post_training/"
            "stage_a_tool_query_runtime_compiler_result_2026-07-23.json"
        ),
    )
    parser.add_argument(
        "--out-md",
        default=(
            "post_training/"
            "STAGE_A_TOOL_QUERY_RUNTIME_COMPILER_RESULT_2026-07-23.md"
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = load_jsonl(args.rows)
    report = build_report(
        rows=rows,
        rows_path=args.rows,
        transfer_result=load_json(args.transfer_result),
        transfer_result_path=args.transfer_result,
        prompt_repair_result=load_json(args.prompt_repair_result),
        prompt_repair_result_path=args.prompt_repair_result,
    )
    write_json(args.out_json, report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
