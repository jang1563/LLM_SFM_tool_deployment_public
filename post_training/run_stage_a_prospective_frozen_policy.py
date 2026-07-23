#!/usr/bin/env python3
"""Score the frozen Stage A routing policy on prospective public development rows.

This runner loads the previously frozen trainable state without retraining.
Model-heavy execution is intended for Cayuga/Expanse. Candidate scores and
row-level predictions remain private; aggregate analysis is produced offline
by evaluate_stage_a_prospective_runtime_hybrid.py.
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

from post_training.build_stage_a_prospective_real_query_slice import (  # noqa: E402
    CANDIDATE_PAIRS,
    MANIFEST_DATASET,
    ROUTING_DATASET,
)
from post_training.build_stage_a_sealed_extension import (  # noqa: E402
    require_external_private_path,
)
from post_training.evaluate_stage_a_prospective_runtime_hybrid import (  # noqa: E402
    deterministic_gate,
    load_json,
    prospective_features,
    sha256_file,
)
from post_training.run_stage_a_saved_output_evidence_candidate_routing_smoke import (  # noqa: E402
    score_candidate_outputs,
)
from post_training.run_stage_a_sealed_candidate_routing_eval import (  # noqa: E402
    load_frozen_model,
)
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    load_jsonl,
)


DATASET = "negbiodb_ct_stage_a_prospective_frozen_policy_run_v1"
PREDICTION_DATASET = (
    "negbiodb_ct_stage_a_prospective_frozen_policy_private_predictions_v1"
)
FREEZE_DATASET = "negbiodb_ct_stage_a_candidate_routing_policy_freeze_v1"


def private_output_path(path: str | Path) -> Path:
    output = Path(path).resolve()
    try:
        relative = output.relative_to(ROOT.resolve())
    except ValueError:
        return output
    allowed_root = Path("post_training/runs")
    if relative != allowed_root and allowed_root not in relative.parents:
        raise ValueError(
            "private output inside repository must stay under post_training/runs"
        )
    return output


def validate_inputs(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_path: str | Path,
    manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    trainable_state_path: str | Path,
) -> list[str]:
    issues: list[str] = []
    require_external_private_path(
        trainable_state_path,
        role="frozen trainable state",
    )
    if manifest.get("dataset") != MANIFEST_DATASET:
        issues.append("unexpected_experiment_manifest_dataset")
    expected_rows = manifest.get("artifacts", {}).get("routing_rows", {})
    if expected_rows.get("sha256") != sha256_file(rows_path):
        issues.append("routing_rows_sha256_mismatch")
    if expected_rows.get("records") != len(rows):
        issues.append("routing_rows_record_count_mismatch")
    if manifest.get("ready_for_frozen_model_scoring") is not True:
        issues.append("experiment_manifest_not_ready_for_frozen_model")
    if freeze.get("dataset") != FREEZE_DATASET:
        issues.append("unexpected_policy_freeze_dataset")
    frozen_state = freeze.get("frozen_artifacts", {}).get("trainable_state", {})
    if frozen_state.get("sha256") != sha256_file(trainable_state_path):
        issues.append("trainable_state_sha256_mismatch")
    policy = freeze.get("policy", {})
    if policy.get("candidate_pairs") != list(CANDIDATE_PAIRS):
        issues.append("frozen_candidate_pairs_mismatch")
    if policy.get("saved_state_load_only") is not True:
        issues.append("freeze_does_not_require_saved_state_load")

    seen_ids: set[str] = set()
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in seen_ids:
            issues.append(f"{row_id}:duplicate_row_id")
        seen_ids.add(row_id)
        if row.get("dataset") != ROUTING_DATASET:
            issues.append(f"{row_id}:unexpected_row_dataset")
        if row.get("target_pair") not in CANDIDATE_PAIRS:
            issues.append(f"{row_id}:invalid_target_pair")
        candidates = row.get("candidate_outputs")
        candidate_pairs = [
            item.get("pair")
            for item in candidates
            if isinstance(item, Mapping)
        ] if isinstance(candidates, list) else []
        if candidate_pairs != list(CANDIDATE_PAIRS):
            issues.append(f"{row_id}:candidate_pair_order_mismatch")
        gate_output, _ = deterministic_gate(prospective_features(row))
        if gate_output["selected_pair"] != row.get("target_pair"):
            issues.append(f"{row_id}:deterministic_gate_target_mismatch")
        visible = json.dumps(row.get("model_visible_task"), sort_keys=True)
        for forbidden in (
            "hidden_eval_metadata",
            '"perturbation"',
            '"target_pair"',
            '"target_output"',
        ):
            if forbidden in visible:
                issues.append(f"{row_id}:prompt_leak:{forbidden}")
    return sorted(set(issues))


def private_prediction_row(
    *,
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    index: int,
    model_id: str,
) -> dict[str, Any]:
    return {
        "id": f"stage_a_prospective_prediction::{index:06d}",
        "dataset": PREDICTION_DATASET,
        "source_row_id": row["id"],
        "model": model_id,
        "prediction": dict(result["prediction"]),
        "candidate_scores": list(result["candidate_scores"]),
    }


def write_private_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    output = private_output_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    output.chmod(0o600)


def compact_prediction_summary(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = Counter(
        str(row.get("prediction", {}).get("selected_pair"))
        for row in predictions
    )
    return {
        "rows": len(predictions),
        "by_predicted_pair": dict(sorted(pairs.items())),
    }


def dry_run_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_path: str | Path,
    manifest_path: str | Path,
    freeze: Mapping[str, Any],
    trainable_state_path: str | Path,
    issues: Sequence[str],
) -> dict[str, Any]:
    policy = freeze.get("policy", {})
    return {
        "dataset": DATASET,
        "dry_run": True,
        "rows": len(rows),
        "routing_rows_sha256": sha256_file(rows_path),
        "experiment_manifest_sha256": sha256_file(manifest_path),
        "freeze_id": freeze.get("freeze_id"),
        "model_id": policy.get("model_id"),
        "model_revision": policy.get("model_revision"),
        "trainable_state": {
            "path": f"private_input::{Path(trainable_state_path).name}",
            "sha256": sha256_file(trainable_state_path),
        },
        "issues": list(issues),
        "ready_for_full_mode": not issues,
        "training_performed": False,
        "completed_sealed_rows_used": False,
    }


def run_full(
    *,
    rows: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    trainable_state_path: str | Path,
    predictions_out: str | Path,
    device: str,
    allow_download: bool,
) -> dict[str, Any]:
    policy = freeze["policy"]
    model_id = str(policy["model_id"])
    model, tokenizer, selected_device, trainable_params = load_frozen_model(
        model_id=model_id,
        trainable_state_path=trainable_state_path,
        train_last_layers=int(policy["train_last_layers"]),
        device=device,
        allow_download=allow_download,
    )
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        result = score_candidate_outputs(
            model,
            tokenizer,
            row,
            device=selected_device,
            max_length=int(policy["max_length"]),
        )
        predictions.append(
            private_prediction_row(
                row=row,
                result=result,
                index=index,
                model_id=model_id,
            )
        )
        print(f"[prospective {index + 1}/{len(rows)}] scored", flush=True)
    write_private_jsonl(predictions_out, predictions)
    return {
        "dataset": DATASET,
        "dry_run": False,
        "rows": len(rows),
        "freeze_id": freeze.get("freeze_id"),
        "model_id": model_id,
        "model_revision": policy.get("model_revision"),
        "device_class": (
            "cuda" if selected_device.startswith("cuda") else selected_device
        ),
        "trainable_params_loaded": trainable_params,
        "training_performed": False,
        "completed_sealed_rows_used": False,
        "private_predictions": {
            "path": f"private_output::{Path(predictions_out).name}",
            "sha256": sha256_file(predictions_out),
            "committed": False,
        },
        "prediction_summary": compact_prediction_summary(predictions),
    }


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
    parser.add_argument(
        "--policy-freeze",
        default="post_training/stage_a_candidate_routing_policy_freeze_2026-07-23.json",
    )
    parser.add_argument("--trainable-state", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = load_jsonl(args.rows)
    manifest = load_json(args.manifest)
    freeze = load_json(args.policy_freeze)
    issues = validate_inputs(
        rows=rows,
        rows_path=args.rows,
        manifest=manifest,
        freeze=freeze,
        trainable_state_path=args.trainable_state,
    )
    if issues:
        raise SystemExit("Prospective frozen-policy validation failed:\n- " + "\n- ".join(issues))
    if args.dry_run:
        report = dry_run_report(
            rows=rows,
            rows_path=args.rows,
            manifest_path=args.manifest,
            freeze=freeze,
            trainable_state_path=args.trainable_state,
            issues=issues,
        )
    else:
        if not args.allow_model_load:
            raise SystemExit(
                "Full mode requires --allow-model-load; run --dry-run first."
            )
        report = run_full(
            rows=rows,
            freeze=freeze,
            trainable_state_path=args.trainable_state,
            predictions_out=args.predictions_out,
            device=args.device,
            allow_download=args.allow_download,
        )
    if args.report_out:
        report_path = private_output_path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report_path.chmod(0o600)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
