"""Run no-API Stage A manifest baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from negbiodb_ct.stage_a_manifest import (
    failure_trajectories_for_stage_a_row,
    ideal_trajectory_from_stage_a_row,
    load_stage_a_manifest,
    score_stage_a_trajectory,
    validate_stage_a_manifest,
)


BASELINES = ("oracle", "self_answer", "wrong_tool", "partial_query")


def trajectory_for_baseline(row: dict[str, Any], baseline: str):
    if baseline == "oracle":
        return ideal_trajectory_from_stage_a_row(row)
    variants = failure_trajectories_for_stage_a_row(row)
    if baseline == "self_answer":
        return variants["self_answering_without_tools"]
    if baseline == "wrong_tool":
        return variants["wrong_tool"]
    if baseline == "partial_query":
        return variants["partial_query"]
    raise ValueError(f"Unknown baseline: {baseline}")


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for baseline in BASELINES:
            result = score_stage_a_trajectory(row, trajectory_for_baseline(row, baseline))
            by_baseline[baseline].append(
                {
                    "case_id": row["case_id"],
                    "score": round(result.score, 3),
                    "passed": result.passed,
                    "violations": list(result.violations),
                }
            )

    summary = {}
    for baseline, results in by_baseline.items():
        summary[baseline] = {
            "cases": len(results),
            "passed": sum(1 for result in results if result["passed"]),
            "mean_score": round(sum(result["score"] for result in results) / len(results), 3),
        }
    return {"summary": summary, "rows": by_baseline}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = load_stage_a_manifest(args.manifest)
    issues = validate_stage_a_manifest(rows, min_rows=1)
    if issues:
        raise SystemExit("Stage A manifest validation failed:\n- " + "\n- ".join(issues))

    report = evaluate(rows)
    if args.json:
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return

    print("baseline       cases  passed  mean_score")
    print("-------------  -----  ------  ----------")
    for baseline in BASELINES:
        item = report["summary"][baseline]
        print(f"{baseline:<13}  {item['cases']:<5}  {item['passed']:<6}  {item['mean_score']:<10.3f}")


if __name__ == "__main__":
    main()
