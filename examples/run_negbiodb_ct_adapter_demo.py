"""Evaluate NegBioDB-CT pilot records through the generic trajectory scorer."""

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import TrajectoryEvaluator
from negbiodb_ct import (
    ideal_trajectory_from_record,
    load_task_records,
    prediction_trajectory_from_record,
    task_spec_from_record,
)


def main() -> None:
    records = load_task_records(ROOT / "negbiodb_ct/tasks_pilot.jsonl", limit=50)
    evaluator = TrajectoryEvaluator()

    oracle_scores = []
    self_answer_scores = []
    self_answer_violations: Counter[str] = Counter()

    for record in records:
        task = task_spec_from_record(record)

        oracle_result = evaluator.evaluate(task, ideal_trajectory_from_record(record))
        oracle_scores.append(oracle_result.score)

        self_answer = prediction_trajectory_from_record(record, "self_answer")
        self_answer_result = evaluator.evaluate(task, self_answer)
        self_answer_scores.append(self_answer_result.score)
        self_answer_violations.update(self_answer_result.violations)

    print(f"records={len(records)}")
    print(f"oracle_mean_score={sum(oracle_scores) / len(oracle_scores):.3f}")
    print(f"self_answer_mean_score={sum(self_answer_scores) / len(self_answer_scores):.3f}")
    print("self_answer_top_violations=")
    for violation, count in self_answer_violations.most_common(5):
        print(f"  {violation}: {count}")


if __name__ == "__main__":
    main()
