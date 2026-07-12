"""Run tiny Gate-1-style baselines on NegBioDB-CT pilot tasks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import TrajectoryEvaluator
from negbiodb_ct import (
    CT_REQUIRED_TOOL_LOOP,
    ideal_trajectory_from_record,
    load_task_records,
    prediction_trajectory_from_record,
    task_spec_from_record,
)


def main() -> None:
    records = load_task_records(ROOT / "negbiodb_ct/tasks_pilot.jsonl")
    evaluator = TrajectoryEvaluator()

    policies = {
        "oracle": lambda record: ideal_trajectory_from_record(record),
        "self_answer_no_tool": lambda record: prediction_trajectory_from_record(
            record, "self_answer"
        ),
        "constant_ground_full_loop": lambda record: prediction_trajectory_from_record(
            record, "ground", tool_names=CT_REQUIRED_TOOL_LOOP
        ),
        "constant_reject_full_loop": lambda record: prediction_trajectory_from_record(
            record, "reject", tool_names=CT_REQUIRED_TOOL_LOOP
        ),
        "constant_defer_full_loop": lambda record: prediction_trajectory_from_record(
            record, "defer", tool_names=CT_REQUIRED_TOOL_LOOP
        ),
        "constant_verify_full_loop": lambda record: prediction_trajectory_from_record(
            record, "verify", tool_names=CT_REQUIRED_TOOL_LOOP
        ),
        "constant_flag_full_loop": lambda record: prediction_trajectory_from_record(
            record, "flag", tool_names=CT_REQUIRED_TOOL_LOOP
        ),
    }

    print(f"records={len(records)}")
    for name, factory in policies.items():
        scores = []
        passed = 0
        for record in records:
            result = evaluator.evaluate(task_spec_from_record(record), factory(record))
            scores.append(result.score)
            passed += int(result.passed)
        print(
            f"{name}\tmean_score={sum(scores) / len(scores):.3f}"
            f"\tpassed={passed}/{len(records)}"
        )


if __name__ == "__main__":
    main()
