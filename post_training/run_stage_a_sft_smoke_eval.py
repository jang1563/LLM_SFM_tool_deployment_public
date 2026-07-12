#!/usr/bin/env python3
"""No-API Stage A SFT smoke/eval harness.

This script does not train a language model. It establishes the local evaluation
shape for the next Stage A SFT experiment by comparing deterministic policies on
the existing train/held-out SFT artifacts. Future model checkpoints should be
reported with the same split, gate, and violation fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import EvidencePacket, ToolStep, Trajectory
from negbiodb_ct.stage_a_manifest import score_stage_a_trajectory


DATASET = "negbiodb_ct_stage_a_sft_smoke_eval_v1"
DEFAULT_POLICIES = (
    "oracle_replay",
    "nearest_train_replay",
    "train_majority_replay",
    "self_answer",
)


@dataclass(frozen=True)
class PolicyPrediction:
    policy: str
    trajectory: Trajectory
    source_train_case_id: str | None = None
    similarity: float | None = None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(obj)
    return rows


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prompt_text(row: Mapping[str, Any]) -> str:
    messages = row.get("messages", [])
    if len(messages) < 2:
        return ""
    user_content = str(messages[1].get("content", ""))
    try:
        payload = json.loads(user_content)
    except json.JSONDecodeError:
        return user_content
    return str(payload.get("claim") or user_content)


def input_id(row: Mapping[str, Any]) -> str:
    return str(row.get("source_manifest_case_id") or row.get("task_id") or row.get("id"))


def trajectory_from_payload(
    payload: Mapping[str, Any],
    *,
    target_input_id: str | None = None,
) -> Trajectory:
    item_input_id = str(target_input_id or payload["input_id"])
    status = str(payload.get("predicted_evidence_status", "unknown"))
    return Trajectory(
        input_id=item_input_id,
        steps=tuple(
            ToolStep(
                name=str(step["name"]),
                arguments=dict(step.get("arguments", {})),
                observation=dict(step.get("observation", {})),
            )
            for step in payload.get("steps", ())
        ),
        evidence_packet=EvidencePacket(
            input_id=item_input_id,
            representation_type="drug_indication_claim",
            negative_evidence_status=status,
            claim_guard_status="checked" if payload.get("steps") else "unchecked",
        ),
        terminal_action=str(payload.get("terminal_action", "answer_self")),
        cited_source_ids=tuple(str(source) for source in payload.get("cited_source_ids", ())),
        predicted_evidence_status=status,
    )


def final_decision_from_trajectory(row: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = row["target_trajectory"]
    return {
        "terminal_action": str(trajectory.get("terminal_action")),
        "evidence_status": str(trajectory.get("predicted_evidence_status", "unknown")),
        "cited_source_ids": tuple(str(source) for source in trajectory.get("cited_source_ids", ())),
    }


def majority_template(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = [final_decision_from_trajectory(row) for row in train_rows]
    action = majority_value([item["terminal_action"] for item in decisions])
    evidence_status = majority_value([item["evidence_status"] for item in decisions])
    cited_source_ids = majority_value([item["cited_source_ids"] for item in decisions])
    step_signatures = [
        tuple(
            (
                str(step.get("name")),
                tuple(sorted((str(key), str(value)) for key, value in dict(step.get("arguments", {})).items())),
            )
            for step in row["target_trajectory"].get("steps", ())
        )
        for row in train_rows
    ]
    step_signature = majority_value(step_signatures)
    steps = [
        {
            "name": name,
            "arguments": dict(arguments),
            "observation": {"status": "completed"},
        }
        for name, arguments in step_signature
    ]
    return {
        "input_id": "<target>",
        "steps": steps,
        "terminal_action": action,
        "cited_source_ids": list(cited_source_ids),
        "predicted_evidence_status": evidence_status,
    }


def majority_value(values: Sequence[Any]) -> Any:
    counts = Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz import fuzz

        return float(fuzz.token_set_ratio(left, right))
    except Exception:
        from difflib import SequenceMatcher

        return 100.0 * SequenceMatcher(None, left, right).ratio()


def nearest_train_row(
    train_rows: Sequence[Mapping[str, Any]],
    eval_row: Mapping[str, Any],
) -> tuple[Mapping[str, Any], float]:
    query = prompt_text(eval_row)
    scored = [
        (similarity(query, prompt_text(candidate)), str(candidate.get("id")), candidate)
        for candidate in train_rows
    ]
    best_score, _, best = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
    return best, best_score


def predict(
    policy: str,
    *,
    train_rows: Sequence[Mapping[str, Any]],
    eval_row: Mapping[str, Any],
    majority_payload: Mapping[str, Any],
) -> PolicyPrediction:
    target_input_id = input_id(eval_row)
    if policy == "oracle_replay":
        return PolicyPrediction(
            policy=policy,
            trajectory=trajectory_from_payload(eval_row["target_trajectory"], target_input_id=target_input_id),
        )
    if policy == "nearest_train_replay":
        nearest, score = nearest_train_row(train_rows, eval_row)
        return PolicyPrediction(
            policy=policy,
            trajectory=trajectory_from_payload(nearest["target_trajectory"], target_input_id=target_input_id),
            source_train_case_id=str(nearest.get("source_manifest_case_id")),
            similarity=round(score, 3),
        )
    if policy == "train_majority_replay":
        return PolicyPrediction(
            policy=policy,
            trajectory=trajectory_from_payload(majority_payload, target_input_id=target_input_id),
        )
    if policy == "self_answer":
        return PolicyPrediction(
            policy=policy,
            trajectory=Trajectory(
                input_id=target_input_id,
                steps=(),
                evidence_packet=EvidencePacket(
                    input_id=target_input_id,
                    representation_type="drug_indication_claim",
                    negative_evidence_status="unknown",
                    claim_guard_status="unchecked",
                ),
                terminal_action="answer_self",
                cited_source_ids=(),
                predicted_evidence_status="unknown",
            ),
        )
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_split(
    *,
    split_name: str,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    manifest_rows_by_case: Mapping[str, Mapping[str, Any]],
    policies: Sequence[str],
) -> dict[str, Any]:
    majority_payload = majority_template(train_rows)
    out: dict[str, Any] = {}
    for policy in policies:
        row_reports = []
        for row in eval_rows:
            case_id = input_id(row)
            manifest_row = manifest_rows_by_case[case_id]
            prediction = predict(
                policy,
                train_rows=train_rows,
                eval_row=row,
                majority_payload=majority_payload,
            )
            result = score_stage_a_trajectory(manifest_row, prediction.trajectory)
            row_reports.append(
                {
                    "case_id": case_id,
                    "split": split_name,
                    "case_family": str(row.get("case_family")),
                    "gold_evidence_status": str(row.get("gold_evidence_status")),
                    "expected_terminal_action": str(row.get("expected_terminal_action")),
                    "predicted_evidence_status": str(prediction.trajectory.predicted_evidence_status),
                    "predicted_terminal_action": str(prediction.trajectory.terminal_action),
                    "source_train_case_id": prediction.source_train_case_id,
                    "similarity": prediction.similarity,
                    "score": round(result.score, 3),
                    "passed": result.passed,
                    "reward_breakdown": dict(result.reward_breakdown),
                    "violations": list(result.violations),
                }
            )
        out[policy] = {
            "summary": summarize_rows(row_reports),
            "rows": row_reports,
        }
    return out


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "passed": 0,
            "mean_score": 0.0,
            "gate_accuracy": {},
            "violations": {},
            "by_case_family": {},
        }
    reward_keys = sorted(
        {
            key
            for row in rows
            for key in dict(row.get("reward_breakdown", {}))
        }
    )
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("case_family"))].append(row)
    violations = Counter(
        violation
        for row in rows
        for violation in row.get("violations", ())
    )
    return {
        "cases": len(rows),
        "passed": sum(1 for row in rows if row.get("passed")),
        "mean_score": round(sum(float(row.get("score", 0.0)) for row in rows) / len(rows), 3),
        "gate_accuracy": {
            key: round(
                sum(float(row.get("reward_breakdown", {}).get(key, 0.0)) for row in rows) / len(rows),
                3,
            )
            for key in reward_keys
        },
        "violations": dict(sorted(violations.items())),
        "by_case_family": {
            family: {
                "cases": len(items),
                "passed": sum(1 for item in items if item.get("passed")),
                "mean_score": round(sum(float(item.get("score", 0.0)) for item in items) / len(items), 3),
            }
            for family, items in sorted(by_family.items())
        },
    }


def build_report(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    policies: Sequence[str] = DEFAULT_POLICIES,
) -> dict[str, Any]:
    manifest_rows_by_case = {str(row["case_id"]): row for row in manifest_rows}
    report = {
        "dataset": DATASET,
        "boundary": (
            "No-API Stage A SFT smoke/eval harness. Baselines are deterministic "
            "policy replays, not trained model results."
        ),
        "policies": list(policies),
        "train_cases": len(train_rows),
        "heldout_cases": len(heldout_rows),
        "splits": {
            "train": evaluate_split(
                split_name="train",
                train_rows=train_rows,
                eval_rows=train_rows,
                manifest_rows_by_case=manifest_rows_by_case,
                policies=policies,
            ),
            "heldout": evaluate_split(
                split_name="heldout",
                train_rows=train_rows,
                eval_rows=heldout_rows,
                manifest_rows_by_case=manifest_rows_by_case,
                policies=policies,
            ),
        },
    }
    report["summary"] = {
        split: {
            policy: payload["summary"]
            for policy, payload in policies_payload.items()
        }
        for split, policies_payload in report["splits"].items()
    }
    return report


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    from negbiodb_ct.stage_a_manifest import load_stage_a_manifest, validate_stage_a_manifest

    rows = load_stage_a_manifest(path)
    issues = validate_stage_a_manifest(rows, min_rows=1)
    if issues:
        raise SystemExit("Stage A manifest validation failed:\n- " + "\n- ".join(issues))
    return rows


def print_summary(report: Mapping[str, Any]) -> None:
    print("split    policy                cases  passed  mean_score")
    print("-------  --------------------  -----  ------  ----------")
    for split in ("train", "heldout"):
        for policy in report["policies"]:
            item = report["summary"][split][policy]
            print(
                f"{split:<7}  {policy:<20}  "
                f"{item['cases']:<5}  {item['passed']:<6}  {item['mean_score']:<10.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument("--train-sft", default="post_training/stage_a_sft_train_v1.jsonl")
    parser.add_argument("--heldout-sft", default="post_training/stage_a_sft_heldout_v1.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(
        manifest_rows=load_manifest_rows(args.manifest),
        train_rows=load_jsonl(args.train_sft),
        heldout_rows=load_jsonl(args.heldout_sft),
    )
    if args.out:
        write_json(args.out, report)
    if args.json:
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
    else:
        print_summary(report)


if __name__ == "__main__":
    main()
