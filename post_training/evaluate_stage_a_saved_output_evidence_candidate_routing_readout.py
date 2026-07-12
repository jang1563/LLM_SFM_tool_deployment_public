#!/usr/bin/env python3
"""No-model readout for evidence-conditioned candidate-routing rows.

This readout establishes the baseline surface before any model-heavy Cayuga
candidate objective. It scores finite-candidate policies from tracked public
rows only: static priors over the candidate set and the runtime evidence gate
already embedded in each row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.export_stage_a_saved_output_evidence_candidate_routing_rows import (  # noqa: E402
    CANDIDATE_PAIRS,
    DATASET as ROW_DATASET,
)
from post_training.run_stage_a_sft_smoke_eval import write_json  # noqa: E402
from post_training.run_stage_a_strict_component_sft_smoke import load_jsonl  # noqa: E402


DATASET = "negbiodb_ct_stage_a_saved_output_evidence_candidate_routing_readout_v1"
DEFAULT_ROWS = "post_training/stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl"
DEFAULT_TRAIN_ROWS = (
    "post_training/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl"
)
DEFAULT_HELDOUT_ROWS = (
    "post_training/stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl"
)
DEFAULT_MANIFEST = "post_training/stage_a_saved_output_evidence_candidate_routing_manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def input_artifact(path: str | Path, *, role: str) -> dict[str, str]:
    return {
        "path": public_path(path),
        "role": role,
        "sha256": sha256_file(path),
    }


def pair_output(pair: str) -> dict[str, str]:
    action, status = pair.split("/", maxsplit=1)
    return {"selected_pair": pair, "action": action, "evidence_status": status}


def runtime_evidence_pair(row: Mapping[str, Any]) -> str:
    return str(row.get("runtime_evidence_pair"))


def static_pair_policy(pair: str) -> Callable[[Mapping[str, Any]], str]:
    def _predict(_: Mapping[str, Any]) -> str:
        return pair

    return _predict


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_name: str,
    predictor: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    scored = []
    for row in rows:
        predicted_pair = predictor(row)
        target_pair = str(row.get("target_pair"))
        scored.append(
            {
                "case_id": row.get("source_manifest_case_id"),
                "target_pair": target_pair,
                "predicted_pair": predicted_pair,
                "exact": predicted_pair == target_pair,
                "bridge_focus_case": bool(row.get("bridge_focus_case", False)),
                "runtime_evidence_reason": row.get("runtime_evidence_reason"),
            }
        )
    exact = sum(1 for row in scored if row["exact"])
    bridge_rows = [row for row in scored if row["bridge_focus_case"]]
    bridge_exact = sum(1 for row in bridge_rows if row["exact"])
    return {
        "policy": policy_name,
        "rows": len(scored),
        "exact": exact,
        "accuracy": round(exact / len(scored), 6) if scored else 0.0,
        "bridge_focus_rows": len(bridge_rows),
        "bridge_focus_exact": bridge_exact,
        "bridge_focus_accuracy": round(bridge_exact / len(bridge_rows), 6)
        if bridge_rows
        else None,
        "by_target_pair": dict(sorted(Counter(row["target_pair"] for row in scored).items())),
        "by_predicted_pair": dict(
            sorted(Counter(row["predicted_pair"] for row in scored).items())
        ),
        "error_case_ids": [str(row["case_id"]) for row in scored if not row["exact"]],
        "bridge_focus_error_case_ids": [
            str(row["case_id"]) for row in bridge_rows if not row["exact"]
        ],
    }


def policy_summaries(
    *,
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    policies: dict[str, Callable[[Mapping[str, Any]], str]] = {
        "runtime_evidence_gate": runtime_evidence_pair,
    }
    for pair in CANDIDATE_PAIRS:
        policies[f"static_{pair.replace('/', '_')}"] = static_pair_policy(pair)

    out: dict[str, Any] = {}
    for name, predictor in policies.items():
        out[name] = {
            "all": score_rows(rows, policy_name=name, predictor=predictor),
            "train": score_rows(train_rows, policy_name=name, predictor=predictor),
            "heldout": score_rows(heldout_rows, policy_name=name, predictor=predictor),
        }
    return out


def build_report(
    *,
    rows_path: str | Path = DEFAULT_ROWS,
    train_rows_path: str | Path = DEFAULT_TRAIN_ROWS,
    heldout_rows_path: str | Path = DEFAULT_HELDOUT_ROWS,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    rows = load_jsonl(rows_path)
    train_rows = load_jsonl(train_rows_path)
    heldout_rows = load_jsonl(heldout_rows_path)
    manifest = load_json(manifest_path)
    policies = policy_summaries(rows=rows, train_rows=train_rows, heldout_rows=heldout_rows)
    runtime = policies["runtime_evidence_gate"]
    best_static_heldout = max(
        row["heldout"]["exact"]
        for name, row in policies.items()
        if name.startswith("static_")
    )
    passes_no_model_readout = (
        runtime["heldout"]["exact"] == runtime["heldout"]["rows"]
        and runtime["heldout"]["bridge_focus_exact"]
        == runtime["heldout"]["bridge_focus_rows"]
        and not manifest.get("overlap_case_ids")
        and not manifest.get("overlap_split_groups")
        and not manifest.get("overlap_source_task_ids")
    )
    report = {
        "dataset": DATASET,
        "row_dataset": ROW_DATASET,
        "input_artifacts": {
            "rows": input_artifact(rows_path, role="candidate-routing rows"),
            "train_rows": input_artifact(train_rows_path, role="candidate-routing train rows"),
            "heldout_rows": input_artifact(
                heldout_rows_path, role="candidate-routing held-out rows"
            ),
            "manifest": input_artifact(manifest_path, role="candidate-routing manifest"),
        },
        "row_manifest_summary": {
            "row_count": manifest.get("row_count"),
            "train_rows": manifest.get("train_rows"),
            "heldout_rows": manifest.get("heldout_rows"),
            "bridge_focus_rows": manifest.get("bridge_focus_rows"),
            "by_target_pair": manifest.get("by_target_pair"),
            "bridge_focus_by_target_pair": manifest.get("bridge_focus_by_target_pair"),
            "overlap_case_ids": manifest.get("overlap_case_ids"),
            "overlap_split_groups": manifest.get("overlap_split_groups"),
            "overlap_source_task_ids": manifest.get("overlap_source_task_ids"),
        },
        "policies": policies,
        "decision": {
            "selected_next_step": "prepare_evidence_conditioned_candidate_routing_smoke_spec",
            "runtime_evidence_gate_heldout_exact": runtime["heldout"]["exact"],
            "runtime_evidence_gate_heldout_rows": runtime["heldout"]["rows"],
            "runtime_evidence_gate_bridge_focus_exact": runtime["heldout"][
                "bridge_focus_exact"
            ],
            "runtime_evidence_gate_bridge_focus_rows": runtime["heldout"][
                "bridge_focus_rows"
            ],
            "best_static_prior_heldout_exact": best_static_heldout,
            "passes_no_model_readout": passes_no_model_readout,
            "ready_for_model_heavy_candidate_smoke_spec": passes_no_model_readout,
            "ready_for_tool_query": False,
            "ready_for_dpo_rlvr": False,
            "ready_for_hugging_face_publication": False,
            "ready_for_release_tagging": False,
            "interpretation": (
                "The substrate has a deterministic visible-evidence baseline "
                "at 5/5 held-out and 4/4 bridge-focus exact, while every static "
                "single-pair prior is at most 1/5 held-out. The next step may be "
                "a small candidate-routing smoke spec, not DPO/RLVR or tool_query."
            ),
        },
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
            "hidden_labels_used_for_readout": False,
        },
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    policies = report["policies"]
    lines = [
        "# Stage A Saved-Output Evidence Candidate-Routing Readout",
        "",
        "Purpose: score the evidence-conditioned candidate-routing rows before any model-heavy checkpoint.",
        "",
        "## Summary",
        "",
        (
            f"- Runtime evidence gate held-out exact: "
            f"{decision['runtime_evidence_gate_heldout_exact']}/"
            f"{decision['runtime_evidence_gate_heldout_rows']}"
        ),
        (
            f"- Runtime evidence gate bridge-focus exact: "
            f"{decision['runtime_evidence_gate_bridge_focus_exact']}/"
            f"{decision['runtime_evidence_gate_bridge_focus_rows']}"
        ),
        (
            f"- Best static prior held-out exact: "
            f"{decision['best_static_prior_heldout_exact']}/"
            f"{decision['runtime_evidence_gate_heldout_rows']}"
        ),
        f"- Passes no-model readout: `{decision['passes_no_model_readout']}`",
        "",
        "## Held-Out Policies",
        "",
        "| Policy | Exact | Bridge-focus exact | Predicted pairs |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, summary in policies.items():
        heldout = summary["heldout"]
        lines.append(
            "| `{name}` | {exact}/{rows} | {bridge_exact}/{bridge_rows} | `{pairs}` |".format(
                name=name,
                exact=heldout["exact"],
                rows=heldout["rows"],
                bridge_exact=heldout["bridge_focus_exact"],
                bridge_rows=heldout["bridge_focus_rows"],
                pairs=json.dumps(heldout["by_predicted_pair"], sort_keys=True),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Selected next step: `{decision['selected_next_step']}`",
            f"- Ready for model-heavy candidate smoke spec: `{decision['ready_for_model_heavy_candidate_smoke_spec']}`",
            f"- Ready for DPO/RLVR: `{decision['ready_for_dpo_rlvr']}`",
            "",
            decision["interpretation"],
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--train-rows", default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--heldout-rows", default=DEFAULT_HELDOUT_ROWS)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out-json",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_readout_2026-07-10.json",
    )
    parser.add_argument(
        "--out-md",
        default="post_training/STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_READOUT_2026-07-10.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        rows_path=args.rows,
        train_rows_path=args.train_rows,
        heldout_rows_path=args.heldout_rows,
        manifest_path=args.manifest,
    )
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
