"""Score prompt-style model JSON outputs with the shared evaluator."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import TrajectoryEvaluator
from negbiodb_ct import (
    CT_ACTION_TO_STATUS,
    CT_REQUIRED_TOOL_LOOP,
    load_task_records,
    required_model_output_schema,
    task_spec_from_record,
    trajectory_from_model_output,
)


def main() -> None:
    records = load_task_records(ROOT / "negbiodb_ct/tasks_pilot.jsonl", limit=1)
    record = records[0]
    action = record["scoring_key"]["gold_action"]
    gold_nct = record["scoring_key"]["gold_nct"]
    output = {
        "action": action,
        "evidence_status": CT_ACTION_TO_STATUS[action].value,
        "tool_calls": list(CT_REQUIRED_TOOL_LOOP),
        "cited_source_ids": [gold_nct] if gold_nct else [],
        "rationale": "Toy prompt-output bridge example.",
    }

    trajectory = trajectory_from_model_output(record, output)
    result = TrajectoryEvaluator().evaluate(task_spec_from_record(record), trajectory)

    print("required_schema=", required_model_output_schema())
    print(f"record={record['packet_id']}")
    print(f"score={result.score:.3f} passed={result.passed}")
    if result.violations:
        print("violations=", ", ".join(result.violations))


if __name__ == "__main__":
    main()
