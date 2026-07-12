#!/usr/bin/env python3
"""Build SFT rows that train the model to generate the evidence rationale."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.evidence_rationale import evidence_action, evidence_decision, evidence_rationale  # noqa: E402
from post_training.split_sft_data import load_jsonl, write_jsonl  # noqa: E402


DEFAULT_STRATEGY = "generative_evidence_rationale_v1"
PROMPT_HEADER = """You are learning to explain the evidence boundary and then output the final decision for a clinical-trial tool-use trajectory.
Given the claim and already-returned tool observations, output a BOUNDARY_RATIONALE line followed by FINAL_SUBMIT_DECISION_JSON."""
PROMPT_SUFFIX = "BOUNDARY_RATIONALE_AND_FINAL_DECISION:"


def final_json_for_row(row: Mapping[str, Any]) -> str:
    decision = evidence_decision(row)
    action = str(decision["action"])
    payload: dict[str, str] = {"action": action}
    if action in {"ground", "flag"} and decision.get("cited_nct"):
        payload["nct"] = str(decision["cited_nct"])
    return json.dumps(payload, sort_keys=True)


def strip_existing_boundary_prompt(messages: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    out = []
    for message in messages:
        if (
            message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and str(message["content"]).startswith("BOUNDARY_RATIONALE:")
        ):
            continue
        out.append(dict(message))
    return out


def build_target(row: Mapping[str, Any]) -> str:
    action = evidence_action(row)
    return (
        f"BOUNDARY_RATIONALE: {evidence_rationale(row)} Evidence-derived final action: {action}.\n"
        f"FINAL_SUBMIT_DECISION_JSON:\n{final_json_for_row(row)}"
    )


def generative_rationale_copy(
    row: Mapping[str, Any],
    *,
    dataset: str,
    pair_index: int,
    strategy: str,
) -> dict[str, Any]:
    messages = list(row["messages"])
    if messages[-1].get("tool_call", {}).get("name") != "submit_decision":
        raise ValueError(f"Expected final submit_decision for {row['id']}")
    action = evidence_action(row)
    action_class = row.get("action_class")
    out = dict(row)
    out["id"] = f"{row['id']}::generative_rationale::{pair_index}"
    out["dataset"] = dataset
    out["source_example_id"] = row.get("source_example_id", row["id"])
    out["boundary_strategy"] = strategy
    out["boundary_pair_role"] = "generative_evidence_rationale"
    out["boundary_pair_index"] = pair_index
    out["boundary_rationale"] = evidence_rationale(row)
    out["evidence_derived_action"] = action
    out["evidence_matches_action_class"] = None if action_class is None else action == str(action_class)
    out["sft_prompt_header"] = PROMPT_HEADER
    out["sft_prompt_suffix"] = PROMPT_SUFFIX
    out["sft_target_text"] = build_target(row)
    out["messages"] = strip_existing_boundary_prompt(messages)
    return out


def counts(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def build_rows(
    source_rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    strategy: str,
) -> list[dict[str, Any]]:
    return [
        generative_rationale_copy(row, dataset=dataset, pair_index=pair_index, strategy=strategy)
        for pair_index, row in enumerate(source_rows)
    ]


def manifest_for_rows(
    *,
    source: str | Path,
    out: str | Path,
    dataset: str,
    strategy: str,
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    matches = [row.get("evidence_matches_action_class") for row in rows]
    return {
        "source": str(source),
        "out": str(out),
        "dataset": dataset,
        "strategy": strategy,
        "examples": len(rows),
        "by_action_class": counts(rows, "action_class"),
        "by_evidence_action": counts(rows, "evidence_derived_action"),
        "by_role": counts(rows, "boundary_pair_role"),
        "evidence_action_matches": sum(match is True for match in matches),
        "evidence_action_mismatches": sum(match is False for match in matches),
        "evidence_action_unlabeled": sum(match is None for match in matches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="post_training/negbiodb_ct_oracle_sft_v1.jsonl")
    parser.add_argument("--out", default="post_training/negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl")
    parser.add_argument(
        "--manifest-out",
        default="post_training/negbiodb_ct_oracle_sft_generative_rationale_manifest.json",
    )
    parser.add_argument("--dataset", default="negbiodb_ct_oracle_sft_generative_rationale_v1")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    args = parser.parse_args()

    source_rows = load_jsonl(args.sft)
    rows = build_rows(source_rows, dataset=args.dataset, strategy=args.strategy)
    write_jsonl(args.out, rows)
    manifest = manifest_for_rows(
        source=args.sft,
        out=args.out,
        dataset=args.dataset,
        strategy=args.strategy,
        rows=rows,
    )
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
