"""Re-score saved NegBioDB-CT runner outputs under full and native profiles."""

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import TrajectoryEvaluator
from negbiodb_ct import (
    load_task_records,
    task_spec_from_record,
    trajectory_from_model_output,
)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "negbiodb_ct/agent_sonnet_n40.json"
    tasks = {
        record["packet_id"]: record
        for record in load_task_records(ROOT / "negbiodb_ct/tasks_pilot.jsonl")
    }
    data = json.loads(path.read_text())
    evaluator = TrajectoryEvaluator()

    full_scores = []
    native_scores = []
    native_violations: Counter[str] = Counter()
    full_violations: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    bucket_examples: dict[str, list[dict[str, object]]] = {}
    by_class: dict[str, list[float]] = {}
    missing_packet_ids = []

    for row in data["rows"]:
        record = tasks.get(row["packet_id"])
        if record is None:
            missing_packet_ids.append(row["packet_id"])
            continue
        output = row["model_output"]

        full = evaluator.evaluate(
            task_spec_from_record(record),
            trajectory_from_model_output(record, output),
        )
        native = evaluator.evaluate(
            task_spec_from_record(record, tool_profile="native_ct"),
            trajectory_from_model_output(record, output, tool_profile="native_ct"),
        )
        full_scores.append(full.score)
        native_scores.append(native.score)
        full_violations.update(full.violations)
        native_violations.update(native.violations)
        by_class.setdefault(row["class"], []).append(native.score)
        if native.violations:
            bucket = classify_native_failure(row, record, native.violations)
            buckets[bucket] += 1
            bucket_examples.setdefault(bucket, []).append({
                "packet_id": row["packet_id"],
                "class": row["class"],
                "gold_action": record["scoring_key"]["gold_action"],
                "gold_nct": record["scoring_key"].get("gold_nct"),
                "pred": row["pred"],
                "violations": list(native.violations),
            })

    print(f"path={path}")
    print(f"rows={len(data['rows'])}")
    print(f"original_generic_mean_score={data['summary'].get('generic_mean_score')}")
    print(f"matched_rows={len(native_scores)}")
    print(f"stale_or_missing_packet_ids={len(missing_packet_ids)}")
    if missing_packet_ids:
        print("first_missing_packet_ids=")
        for packet_id in missing_packet_ids[:5]:
            print(f"  {packet_id}")
    if not native_scores:
        print("No rows could be re-scored against the current task set.")
        return
    print(f"full_profile_mean_score={sum(full_scores) / len(full_scores):.3f}")
    print(f"native_profile_mean_score={sum(native_scores) / len(native_scores):.3f}")
    print("native_profile_by_class=")
    for cls, scores in sorted(by_class.items()):
        print(f"  {cls}: {sum(scores) / len(scores):.3f} ({len(scores)} rows)")
    print("full_profile_top_violations=")
    for violation, count in full_violations.most_common(5):
        print(f"  {violation}: {count}")
    print("native_profile_top_violations=")
    for violation, count in native_violations.most_common(5):
        print(f"  {violation}: {count}")
    print("native_failure_buckets=")
    for bucket, count in buckets.most_common():
        print(f"  {bucket}: {count}")
        for example in bucket_examples[bucket][:3]:
            print(f"    {json.dumps(example, sort_keys=True)}")


def classify_native_failure(row: Mapping[str, object], record: Mapping[str, object], violations: tuple[str, ...]) -> str:
    """Classify remaining native-profile failures into next-action buckets."""

    row_class = str(row["class"])
    pred = row.get("pred", {})
    pred_action = pred.get("action") if isinstance(pred, dict) else None
    violation_set = set(violations)

    if row_class == "flag" and pred_action == "flag" and violation_set == {"missing_required_attribution"}:
        return "prompt_schema_flag_needs_nct"

    if row_class == "ground" and pred_action in {"defer", "verify"}:
        return "ground_efficacy_record_not_recognized"

    if row_class == "reject" and pred_action == "ground":
        return "mixed_endpoint_reject_overridden_by_ground"

    if "missing_required_attribution" in violation_set:
        return "attribution_error"

    if {"evidence_status_mismatch", "terminal_action_mismatch"} & violation_set:
        return "action_or_evidence_status_error"

    return "other_native_failure"


if __name__ == "__main__":
    main()
