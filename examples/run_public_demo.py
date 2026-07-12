"""Run the public-safe synthetic trajectory demo.

The JSONL records in demo/public_trajectory_cases.jsonl are synthetic. They are
designed to exercise the evaluator without private databases, API calls, or real
clinical-trial records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import (
    EvidencePacket,
    TaskSpec,
    ToolStep,
    Trajectory,
    TrajectoryEvaluator,
)


def _task_from_json(payload: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        input_id=payload["input_id"],
        claim=payload["claim"],
        required_tools=tuple(payload.get("required_tools", ())),
        gold_evidence_status=payload.get("gold_evidence_status", "unknown"),
        expected_terminal_action=payload.get("expected_terminal_action"),
        gold_source_ids=tuple(payload.get("gold_source_ids", ())),
        requires_attribution=bool(payload.get("requires_attribution", False)),
        requires_external_tool=bool(payload.get("requires_external_tool", True)),
        web_zero=bool(payload.get("web_zero", False)),
    )


def _trajectory_from_json(payload: dict[str, Any]) -> Trajectory:
    packet = EvidencePacket(**payload["evidence_packet"])
    steps = tuple(
        ToolStep(step["name"], step.get("arguments", {}), step.get("observation", {}))
        for step in payload.get("steps", ())
    )
    return Trajectory(
        input_id=packet.input_id,
        steps=steps,
        evidence_packet=packet,
        terminal_action=payload["terminal_action"],
        cited_source_ids=tuple(payload.get("cited_source_ids", ())),
        predicted_evidence_status=payload.get("predicted_evidence_status"),
        rationale=payload.get("rationale"),
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(ROOT / "demo" / "public_trajectory_cases.jsonl"))
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a compact table.",
    )
    args = parser.parse_args()

    evaluator = TrajectoryEvaluator()
    rows = []
    for record in load_cases(Path(args.cases)):
        result = evaluator.evaluate(
            _task_from_json(record["task"]),
            _trajectory_from_json(record["trajectory"]),
        )
        rows.append(
            {
                "label": record["label"],
                "score": round(result.score, 3),
                "passed": result.passed,
                "violations": list(result.violations),
            }
        )

    if args.json:
        print(json.dumps({"cases": rows}, indent=2))
        return

    print("label                         score  passed  violations")
    print("----------------------------  -----  ------  ----------")
    for row in rows:
        violations = ",".join(row["violations"]) if row["violations"] else "-"
        print(f"{row['label']:<28}  {row['score']:<5.3f}  {str(row['passed']):<6}  {violations}")


if __name__ == "__main__":
    main()
