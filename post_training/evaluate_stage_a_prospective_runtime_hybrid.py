#!/usr/bin/env python3
"""Evaluate Stage A real-query perturbations and runtime arbitration.

The deterministic gate parses raw model-visible tool state. Optional frozen
model predictions are scored offline, then combined with the gate through a
fail-closed hybrid: model choices are accepted only when they agree with the
gate; disagreements on decisive outputs become verification requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.build_stage_a_prospective_real_query_slice import (  # noqa: E402
    CANDIDATE_PAIRS,
    MANIFEST_DATASET,
    REQUIRED_QUERY_FIELDS,
    ROUTING_DATASET,
)
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    load_jsonl,
    write_json,
)


DATASET = "negbiodb_ct_stage_a_prospective_runtime_hybrid_eval_v1"
EXPECTED_TOOLS = (
    "nullatlas_survey_prior_failures",
    "nullatlas_verify_trial_claims",
    "nullatlas_check_value_validity",
    "nullatlas_negative_evidence_completeness",
)
FAIL_CLOSED_PAIRS = {"verify/insufficient", "defer/insufficient"}
DECISIVE_PAIRS = {
    "ground/supported",
    "reject/contradicted",
    "flag/invalid_value",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"external_input::{resolved.name}"


def query_values(task: Mapping[str, Any]) -> dict[str, Any]:
    query = task.get("query")
    if not isinstance(query, Mapping):
        return {}
    out: dict[str, Any] = {}
    for field in REQUIRED_QUERY_FIELDS:
        payload = query.get(field)
        if isinstance(payload, Mapping) and "value" in payload:
            out[field] = payload["value"]
    return out


def tool_loop(task: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    loop = task.get("observed_tool_loop")
    if not isinstance(loop, list):
        return []
    return [item for item in loop if isinstance(item, Mapping)]


def tool_content(
    loop: Sequence[Mapping[str, Any]],
    name: str,
) -> Mapping[str, Any]:
    for item in loop:
        if item.get("name") == name and isinstance(item.get("content"), Mapping):
            return item["content"]
    return {}


def prospective_features(row: Mapping[str, Any]) -> dict[str, Any]:
    task = row.get("model_visible_task")
    if not isinstance(task, Mapping):
        return {
            "model_visible_task_present": False,
            "observed_tool_loop_present": False,
        }
    expected_query = query_values(task)
    loop = tool_loop(task)
    tool_names = [str(item.get("name")) for item in loop]
    missing_tools = sorted(set(EXPECTED_TOOLS) - set(tool_names))
    wrong_tools = sorted(set(tool_names) - set(EXPECTED_TOOLS))
    query_mismatch_tools: list[str] = []
    tool_errors: list[str] = []
    for item in loop:
        name = str(item.get("name"))
        arguments = item.get("arguments")
        if not isinstance(arguments, Mapping) or dict(arguments) != expected_query:
            query_mismatch_tools.append(name)
        content = item.get("content")
        if isinstance(content, Mapping) and content.get("error"):
            tool_errors.append(name)

    survey = tool_content(loop, "nullatlas_survey_prior_failures")
    verifier = tool_content(loop, "nullatlas_verify_trial_claims")
    validity = tool_content(loop, "nullatlas_check_value_validity")
    completeness = tool_content(loop, "nullatlas_negative_evidence_completeness")
    raw_records = survey.get("same_indication_records")
    records = (
        [dict(item) for item in raw_records if isinstance(item, Mapping)]
        if isinstance(raw_records, list)
        else []
    )
    raw_findings = validity.get("value_validity_findings")
    findings = (
        [dict(item) for item in raw_findings if isinstance(item, Mapping)]
        if isinstance(raw_findings, list)
        else []
    )
    raw_citations = completeness.get("citation_candidates")
    citations = (
        [str(item) for item in raw_citations]
        if isinstance(raw_citations, list)
        else []
    )
    missing_attribution = [
        index
        for index, record in enumerate(records)
        if not record.get("source_id")
    ]
    stale_sources = [
        str(record.get("source_id") or f"record::{index}")
        for index, record in enumerate(records)
        if record.get("source_freshness") not in {None, "current"}
    ]
    return {
        "model_visible_task_present": True,
        "observed_tool_loop_present": bool(loop),
        "expected_query_complete": set(expected_query) == set(REQUIRED_QUERY_FIELDS),
        "tool_names": tool_names,
        "missing_tools": missing_tools,
        "wrong_tools": wrong_tools,
        "query_mismatch_tools": query_mismatch_tools,
        "tool_errors": tool_errors,
        "same_indication_records": records,
        "same_indication_record_count": len(records),
        "missing_attribution_record_indexes": missing_attribution,
        "stale_source_ids": stale_sources,
        "related_negative_evidence_count": survey.get(
            "related_negative_evidence_count"
        ),
        "records_considered": verifier.get("records_considered"),
        "value_validity_findings": findings,
        "completeness_signal": completeness.get("completeness_signal"),
        "citation_candidates": citations,
    }


def first_source_id(items: Sequence[Mapping[str, Any]]) -> str | None:
    for item in items:
        if item.get("source_id"):
            return str(item["source_id"])
    return None


def pair_output(pair: str, *, citations: Sequence[str] = ()) -> dict[str, Any]:
    action, status = pair.split("/", maxsplit=1)
    return {
        "selected_pair": pair,
        "action": action,
        "evidence_status": status,
        "cited_source_ids": list(citations),
    }


def deterministic_gate(
    features: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if not features.get("model_visible_task_present"):
        return pair_output("defer/insufficient"), "missing_model_visible_task"
    if not features.get("observed_tool_loop_present"):
        return pair_output("defer/insufficient"), "missing_tool_loop"
    if features.get("tool_errors"):
        return pair_output("defer/insufficient"), "tool_unavailable"
    if features.get("missing_tools") or features.get("wrong_tools"):
        return pair_output("defer/insufficient"), "tool_contract_failure"
    if (
        not features.get("expected_query_complete")
        or features.get("query_mismatch_tools")
    ):
        return pair_output("defer/insufficient"), "query_contract_failure"
    if features.get("missing_attribution_record_indexes"):
        return pair_output("verify/insufficient"), "missing_source_attribution"
    if features.get("stale_source_ids"):
        return pair_output("verify/insufficient"), "stale_source_requires_verification"

    findings = features.get("value_validity_findings")
    if isinstance(findings, list) and findings:
        source_id = first_source_id(
            [item for item in findings if isinstance(item, Mapping)]
        )
        return pair_output(
            "flag/invalid_value",
            citations=[source_id] if source_id else [],
        ), "invalid_numeric_value"

    completeness = features.get("completeness_signal")
    citations = features.get("citation_candidates")
    citation_ids = (
        [str(item) for item in citations] if isinstance(citations, list) else []
    )
    if completeness == "mixed_endpoint_records_for_same_claim":
        return pair_output("reject/contradicted"), "mixed_endpoint_contradiction"
    if completeness == "same_indication_failure_record_found":
        return pair_output(
            "ground/supported",
            citations=citation_ids,
        ), "same_indication_failure_supported"
    if completeness == "related_evidence_exists_but_same_indication_record_absent":
        return pair_output(
            "verify/insufficient"
        ), "related_evidence_requires_verification"
    if completeness == "no_same_indication_or_related_failure_record":
        return pair_output("defer/insufficient"), "no_relevant_evidence"
    return pair_output("defer/insufficient"), "unknown_evidence_state_fail_closed"


def target_pair(row: Mapping[str, Any]) -> str:
    value = row.get("target_pair")
    if value not in CANDIDATE_PAIRS:
        raise ValueError(f"{row.get('id')} has invalid target pair: {value}")
    return str(value)


def perturbation(row: Mapping[str, Any]) -> str:
    hidden = row.get("hidden_eval_metadata")
    if not isinstance(hidden, Mapping) or not hidden.get("perturbation"):
        raise ValueError(f"{row.get('id')} lacks perturbation metadata")
    return str(hidden["perturbation"])


def summarize_strategy(
    rows: Sequence[Mapping[str, Any]],
    predicted_pairs: Sequence[str],
    *,
    reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    if len(rows) != len(predicted_pairs):
        raise ValueError("strategy prediction count mismatch")
    expected_pairs = [target_pair(row) for row in rows]
    exact_flags = [
        predicted == expected
        for predicted, expected in zip(predicted_pairs, expected_pairs, strict=True)
    ]
    by_target: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "exact": 0}
    )
    by_perturbation: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "exact": 0, "unsafe_ground_supported": 0}
    )
    unsafe_ground = 0
    decisive = 0
    decisive_incorrect = 0
    fail_closed = 0
    for row, predicted, expected, exact in zip(
        rows,
        predicted_pairs,
        expected_pairs,
        exact_flags,
        strict=True,
    ):
        name = perturbation(row)
        by_target[expected]["rows"] += 1
        by_target[expected]["exact"] += int(exact)
        by_perturbation[name]["rows"] += 1
        by_perturbation[name]["exact"] += int(exact)
        is_unsafe_ground = predicted == "ground/supported" and not exact
        unsafe_ground += int(is_unsafe_ground)
        by_perturbation[name]["unsafe_ground_supported"] += int(is_unsafe_ground)
        if predicted in DECISIVE_PAIRS:
            decisive += 1
            decisive_incorrect += int(not exact)
        if predicted in FAIL_CLOSED_PAIRS:
            fail_closed += 1
    recalls = [
        values["exact"] / values["rows"]
        for values in by_target.values()
        if values["rows"]
    ]
    summary = {
        "rows": len(rows),
        "exact": sum(exact_flags),
        "accuracy": round(sum(exact_flags) / len(rows), 6) if rows else 0.0,
        "macro_pair_accuracy": (
            round(sum(recalls) / len(recalls), 6) if recalls else 0.0
        ),
        "by_target_pair": dict(sorted(by_target.items())),
        "by_predicted_pair": dict(sorted(Counter(predicted_pairs).items())),
        "by_perturbation": dict(sorted(by_perturbation.items())),
        "unsafe_ground_supported": unsafe_ground,
        "decisive_coverage": round(decisive / len(rows), 6) if rows else 0.0,
        "decisive_predictions": decisive,
        "selective_risk": (
            round(decisive_incorrect / decisive, 6) if decisive else 0.0
        ),
        "fail_closed_rate": round(fail_closed / len(rows), 6) if rows else 0.0,
    }
    if reasons is not None:
        if len(reasons) != len(rows):
            raise ValueError("strategy reason count mismatch")
        summary["by_reason"] = dict(sorted(Counter(reasons).items()))
    return summary


def load_model_predictions(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    predictions = load_jsonl(path)
    by_source: dict[str, str] = {}
    for prediction_row in predictions:
        source_id = prediction_row.get("source_row_id")
        prediction = prediction_row.get("prediction")
        if not isinstance(source_id, str) or not isinstance(prediction, Mapping):
            raise ValueError("malformed frozen-model prediction row")
        pair = prediction.get("selected_pair")
        if pair not in CANDIDATE_PAIRS:
            raise ValueError(f"{source_id} has invalid predicted pair: {pair}")
        if source_id in by_source:
            raise ValueError(f"duplicate prediction for {source_id}")
        if pair != f"{prediction.get('action')}/{prediction.get('evidence_status')}":
            raise ValueError(f"{source_id} prediction pair disagrees with fields")
        by_source[source_id] = str(pair)
    expected_ids = [str(row.get("id")) for row in rows]
    missing = sorted(set(expected_ids) - set(by_source))
    extra = sorted(set(by_source) - set(expected_ids))
    if missing or extra:
        raise ValueError(
            f"prediction alignment mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    return [by_source[row_id] for row_id in expected_ids]


def hybrid_pair(model_pair: str, gate_pair: str) -> tuple[str, str]:
    if model_pair == gate_pair:
        return model_pair, "model_gate_agreement"
    if gate_pair in FAIL_CLOSED_PAIRS:
        return gate_pair, "runtime_fail_closed_override"
    return "verify/insufficient", "decisive_disagreement_requires_verification"


def validate_inputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    rows_path: str | Path,
    manifest: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    if manifest.get("dataset") != MANIFEST_DATASET:
        issues.append("unexpected_experiment_manifest_dataset")
    if manifest.get("routing_dataset") != ROUTING_DATASET:
        issues.append("unexpected_routing_dataset")
    expected_artifact = manifest.get("artifacts", {}).get("routing_rows", {})
    if expected_artifact.get("sha256") != sha256_file(rows_path):
        issues.append("routing_rows_sha256_mismatch")
    if expected_artifact.get("records") != len(rows):
        issues.append("routing_rows_record_count_mismatch")
    if manifest.get("ready_for_no_model_baselines") is not True:
        issues.append("experiment_manifest_not_ready")
    for row in rows:
        if row.get("dataset") != ROUTING_DATASET:
            issues.append(f"{row.get('id')}:unexpected_dataset")
        task = row.get("model_visible_task")
        if not isinstance(task, Mapping):
            issues.append(f"{row.get('id')}:missing_model_visible_task")
            continue
        visible = json.dumps(task, sort_keys=True)
        for forbidden in (
            "hidden_eval_metadata",
            '"perturbation"',
            '"target_pair"',
            '"target_output"',
        ):
            if forbidden in visible:
                issues.append(f"{row.get('id')}:prompt_leak:{forbidden}")
    return sorted(set(issues))


def build_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_path: str | Path,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    issues = validate_inputs(rows, rows_path=rows_path, manifest=manifest)
    if issues:
        raise ValueError("prospective evaluation validation failed: " + "; ".join(issues))
    gate_outputs: list[dict[str, Any]] = []
    gate_reasons: list[str] = []
    for row in rows:
        output, reason = deterministic_gate(prospective_features(row))
        gate_outputs.append(output)
        gate_reasons.append(reason)
    gate_pairs = [str(output["selected_pair"]) for output in gate_outputs]
    target_counts = Counter(target_pair(row) for row in rows)
    best_static_pair, best_static_exact = sorted(
        target_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]
    strategies: dict[str, Any] = {
        "trust_all": summarize_strategy(
            rows,
            ["ground/supported"] * len(rows),
        ),
        "best_static_pair": {
            "pair": best_static_pair,
            **summarize_strategy(rows, [best_static_pair] * len(rows)),
        },
        "deterministic_gate": summarize_strategy(
            rows,
            gate_pairs,
            reasons=gate_reasons,
        ),
    }

    model_provided = predictions_path is not None
    if predictions_path is not None:
        model_pairs = load_model_predictions(predictions_path, rows)
        hybrid_pairs: list[str] = []
        hybrid_reasons: list[str] = []
        for model_pair, gate_pair in zip(model_pairs, gate_pairs, strict=True):
            output_pair, reason = hybrid_pair(model_pair, gate_pair)
            hybrid_pairs.append(output_pair)
            hybrid_reasons.append(reason)
        strategies["frozen_model"] = summarize_strategy(rows, model_pairs)
        strategies["runtime_hybrid"] = summarize_strategy(
            rows,
            hybrid_pairs,
            reasons=hybrid_reasons,
        )

    gate_exact = strategies["deterministic_gate"]["exact"]
    report = {
        "dataset": DATASET,
        "evaluation_scope": (
            "public_development_real_query_plus_synthetic_tool_state_perturbations"
        ),
        "rows": len(rows),
        "input_artifacts": {
            "routing_rows": {
                "path": display_path(rows_path),
                "sha256": sha256_file(rows_path),
                "records": len(rows),
            },
            "experiment_manifest": {
                "path": display_path(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "frozen_predictions": (
                {
                    "path": f"private_input::{Path(predictions_path).name}",
                    "sha256": sha256_file(predictions_path),
                    "committed": False,
                }
                if predictions_path is not None
                else None
            ),
        },
        "model_predictions_provided": model_provided,
        "strategies": strategies,
        "decision": {
            "deterministic_gate_exact": gate_exact == len(rows),
            "deterministic_gate_beats_best_static": (
                gate_exact > best_static_exact
            ),
            "frozen_model_beats_best_static": (
                strategies["frozen_model"]["exact"] > best_static_exact
                if model_provided
                else None
            ),
            "runtime_hybrid_unsafe_ground_zero": (
                strategies["runtime_hybrid"]["unsafe_ground_supported"] == 0
                if model_provided
                else None
            ),
            "ready_for_frozen_model_scoring": not model_provided,
            "ready_for_dpo_rlvr": False,
            "ready_for_hugging_face_publication": False,
        },
        "scientific_boundary": {
            "development_only": True,
            "independent_test_claimed": False,
            "actual_query_identifier_values_visible": True,
            "live_tool_execution_evaluated": False,
            "synthetic_tool_result_state_used": True,
            "llm_judge_used": False,
            "completed_sealed_rows_used": False,
        },
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    strategies = report["strategies"]
    lines = [
        "# Stage A Prospective Real-Query Runtime-Hybrid Evaluation",
        "",
        "Scope: public development cases with actual model-visible query IDs and",
        "synthetic tool-result perturbations. This is not a new sealed test or",
        "a live-tool execution result.",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Exact | Macro pair accuracy | Unsafe ground | Coverage | Selective risk |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in strategies.items():
        if not isinstance(summary, Mapping) or "rows" not in summary:
            continue
        lines.append(
            "| `{name}` | {exact}/{rows} | {macro:.3f} | {unsafe} | "
            "{coverage:.3f} | {risk:.3f} |".format(
                name=name,
                exact=summary["exact"],
                rows=summary["rows"],
                macro=summary["macro_pair_accuracy"],
                unsafe=summary["unsafe_ground_supported"],
                coverage=summary["decisive_coverage"],
                risk=summary["selective_risk"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Actual drug/condition identifier values are model-visible.",
            "- Tool-result states and perturbations are synthetic and deterministic.",
            "- Completed sealed rows are not loaded, rescored, or used for selection.",
            "- DPO, RLVR, and Hugging Face publication remain closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        default=(
            "post_training/"
            "stage_a_prospective_real_query_routing_perturbations_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "post_training/stage_a_prospective_real_query_experiment_manifest.json"
        ),
    )
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = load_jsonl(args.rows)
    manifest = load_json(args.manifest)
    report = build_report(
        rows=rows,
        rows_path=args.rows,
        manifest=manifest,
        manifest_path=args.manifest,
        predictions_path=args.predictions,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
