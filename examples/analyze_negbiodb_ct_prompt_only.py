"""Analyze prompt-only NegBioDB-CT baseline artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import TrajectoryEvaluator
from negbiodb_ct import load_task_records, task_spec_from_record, trajectory_from_model_output


def classify_prompt_only_failure(row: Mapping[str, object], violations: tuple[str, ...]) -> str:
    pred = row.get("pred", {})
    pred_action = pred.get("action") if isinstance(pred, dict) else None
    gold = str(row["gold"])
    violation_set = set(violations)

    if pred_action is None:
        return "schema_or_parse_failure"
    if row.get("correct") and violation_set == {"missing_required_tool_sequence"}:
        return "correct_action_but_no_tool_trace"
    if pred_action == "defer" and gold != "defer":
        return "conservative_defer_wrong_without_tools"
    if pred_action == "verify" and gold == "defer":
        return "oververify_without_evidence"
    if gold in {"ground", "flag"} and pred_action in {"verify", "defer"}:
        return "missed_positive_or_invalid_evidence"
    if gold == "reject" and pred_action in {"defer", "verify"}:
        return "missed_mixed_endpoint_contradiction"
    if pred_action in {"reject", "flag"} and gold in {"defer", "verify"}:
        return "unsupported_contradiction_from_prior"
    return "other_prompt_only_failure"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "negbiodb_ct/agent_prompt_only_sonnet_n40.json"
    task_records = {
        record["packet_id"]: record
        for record in load_task_records(ROOT / "negbiodb_ct/tasks_pilot.jsonl")
    }
    data = json.loads(path.read_text())
    evaluator = TrajectoryEvaluator()

    buckets: Counter[str] = Counter()
    pred_actions: Counter[str | None] = Counter()
    by_class_pred: dict[str, Counter[str | None]] = defaultdict(Counter)
    violations: Counter[str] = Counter()
    scores = []

    for row in data["rows"]:
        record = task_records[row["packet_id"]]
        result = evaluator.evaluate(
            task_spec_from_record(record, tool_profile="native_ct"),
            trajectory_from_model_output(record, row["model_output"], tool_profile="native_ct"),
        )
        pred_action = row["pred"]["action"]
        pred_actions[pred_action] += 1
        by_class_pred[row["class"]][pred_action] += 1
        violations.update(result.violations)
        scores.append(result.score)
        if result.violations:
            buckets[classify_prompt_only_failure(row, tuple(result.violations))] += 1

    print(f"path={path}")
    print(f"rows={len(data['rows'])}")
    print(f"action_accuracy={data['summary']['action_accuracy']:.3f}")
    print(f"mean_reward={data['summary']['mean_reward']:.3f}")
    print(f"native_profile_mean_score={sum(scores) / len(scores):.3f}")
    print(f"tool_call_rate={data['summary']['tool_call_rate']:.3f}")
    print("pred_actions=")
    for action, count in pred_actions.most_common():
        print(f"  {action}: {count}")
    print("by_class_pred=")
    for cls, counter in sorted(by_class_pred.items()):
        print(f"  {cls}: {dict(counter)}")
    print("top_violations=")
    for violation, count in violations.most_common(8):
        print(f"  {violation}: {count}")
    print("prompt_only_failure_buckets=")
    for bucket, count in buckets.most_common():
        print(f"  {bucket}: {count}")


if __name__ == "__main__":
    main()
