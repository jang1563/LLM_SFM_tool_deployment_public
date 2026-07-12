#!/usr/bin/env python3
"""Export larger deterministic-oracle NegBioDB-CT SFT data."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from negbiodb_ct import load_task_records
from negbiodb_ct.baselines import score_decision
from negbiodb_ct.export_post_training_data import (
    display_path,
    score_model_output,
    supervised_example,
    tool_results_for_record,
    write_jsonl,
)
from negbiodb_ct.run_agent import CT_DB


SOURCE_RUNNER = "deterministic_oracle_policy"


def oracle_model_output(record: Mapping[str, Any]) -> dict[str, Any]:
    action = str(record["scoring_key"]["gold_action"])
    output: dict[str, Any] = {
        "action": action,
        "called": ["search_failures", "check_other_indications"],
    }
    gold_nct = record["scoring_key"].get("gold_nct")
    if action in {"ground", "flag"} and gold_nct:
        output["cited_source_ids"] = [gold_nct]
    return output


def decision_from_model_output(output: Mapping[str, Any]) -> dict[str, str | None]:
    action = str(output["action"])
    citations = output.get("cited_source_ids") or []
    return {
        "action": action,
        "cited_nct": str(citations[0]) if action in {"ground", "flag"} and citations else None,
    }


def oracle_runner_row(record: Mapping[str, Any]) -> dict[str, Any]:
    model_output = oracle_model_output(record)
    decision = decision_from_model_output(model_output)
    score = score_decision(decision, record["scoring_key"])
    generic_score = score_model_output(record, model_output)
    return {
        "packet_id": record["packet_id"],
        "class": record["action_class"],
        "gold": record["scoring_key"]["gold_action"],
        "pred": decision,
        "called": list(model_output["called"]),
        "model_output": model_output,
        "correct": score["correct"],
        "reward": score["reward"],
        "generic_score": generic_score["score"],
        "generic_violations": list(generic_score["violations"]),
    }


def build_oracle_sft_rows(
    records: list[Mapping[str, Any]],
    con: sqlite3.Connection,
    *,
    allow_failures: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    skipped = []
    for record in records:
        runner_row = oracle_runner_row(record)
        if not allow_failures and (not runner_row["correct"] or runner_row["generic_violations"]):
            skipped.append({
                "packet_id": record["packet_id"],
                "correct": runner_row["correct"],
                "generic_violations": runner_row["generic_violations"],
            })
            continue
        observations = tool_results_for_record(record, con)
        example = supervised_example(
            record,
            runner_row,
            observations,
            source_runner=SOURCE_RUNNER,
        )
        example["dataset"] = "negbiodb_ct_oracle_sft_v1"
        example["oracle_target"] = True
        rows.append(example)
    return rows, skipped


def manifest_for_oracle_sft(
    *,
    tasks: str | Path,
    out: str | Path,
    rows: list[Mapping[str, Any]],
    skipped: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset": "negbiodb_ct_oracle_sft_v1",
        "source_runner": SOURCE_RUNNER,
        "target_policy": "gold_scoring_key_plus_native_ct_tool_observations",
        "boundary": "Deterministic oracle-policy SFT data; not live runner behavior.",
        "tasks": display_path(tasks),
        "sft": display_path(out),
        "sft_examples": len(rows),
        "by_class": dict(sorted(Counter(row["action_class"] for row in rows).items())),
        "skipped": list(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument("--out", default="post_training/negbiodb_ct_oracle_sft_v1.jsonl")
    parser.add_argument("--manifest-out", default="post_training/negbiodb_ct_oracle_sft_manifest.json")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    records = load_task_records(args.tasks)
    con = sqlite3.connect(CT_DB)
    rows, skipped = build_oracle_sft_rows(records, con, allow_failures=args.allow_failures)
    write_jsonl(args.out, rows)
    manifest = manifest_for_oracle_sft(tasks=args.tasks, out=args.out, rows=rows, skipped=skipped)
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if skipped and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
