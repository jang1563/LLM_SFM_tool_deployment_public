#!/usr/bin/env python3
"""Build a private Stage A sealed extension and a public-safe commitment.

The candidate task pool and selected manifest must stay outside the repository.
Only aggregate class counts, overlap results, and cryptographic hashes are
written to the public checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from negbiodb_ct.adapter import load_task_records  # noqa: E402
from negbiodb_ct.stage_a_manifest import (  # noqa: E402
    ACTION_CLASS_ORDER,
    load_stage_a_manifest,
    split_group_for_record,
    stage_a_row_from_task_record,
    validate_stage_a_manifest,
    write_stage_a_manifest,
)


DATASET = "negbiodb_ct_stage_a_sealed_extension_v1"
COMMITMENT_DATASET = "negbiodb_ct_stage_a_sealed_extension_commitment_v1"
DEFAULT_SEED = 20260710


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def is_inside_repo(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True


def require_external_private_path(path: str | Path, *, role: str) -> None:
    if is_inside_repo(path):
        raise ValueError(f"{role} must stay outside the public repository")


def public_reference(path: str | Path, *, external_prefix: str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"{external_prefix}::{path.name}"


def normalize_claim(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def task_claim(record: Mapping[str, Any]) -> str:
    observation = record.get("observation")
    if not isinstance(observation, Mapping):
        return ""
    return normalize_claim(observation.get("claim"))


def manifest_claim(row: Mapping[str, Any]) -> str:
    visible = row.get("model_visible_task")
    if not isinstance(visible, Mapping):
        return ""
    return normalize_claim(visible.get("claim"))


def exclusion_sets(
    task_records: Iterable[Mapping[str, Any]],
    manifest_rows: Iterable[Mapping[str, Any]],
) -> dict[str, set[str]]:
    source_task_ids: set[str] = set()
    split_groups: set[str] = set()
    claims: set[str] = set()

    for record in task_records:
        packet_id = str(record.get("packet_id") or "")
        if packet_id:
            source_task_ids.add(packet_id)
        claim = task_claim(record)
        if claim:
            claims.add(claim)
        try:
            split_groups.add(split_group_for_record(record))
        except (KeyError, TypeError, ValueError):
            continue

    for row in manifest_rows:
        hidden = row.get("hidden_eval_metadata")
        if isinstance(hidden, Mapping):
            source_task_id = str(hidden.get("source_task_id") or "")
            split_group = str(hidden.get("split_group") or "")
            if source_task_id:
                source_task_ids.add(source_task_id)
            if split_group:
                split_groups.add(split_group)
        claim = manifest_claim(row)
        if claim:
            claims.add(claim)

    return {
        "source_task_ids": source_task_ids,
        "split_groups": split_groups,
        "claims": claims,
    }


def select_sealed_records(
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    excluded_task_records: Sequence[Mapping[str, Any]],
    excluded_manifest_rows: Sequence[Mapping[str, Any]],
    per_action: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    if per_action <= 0:
        raise ValueError("per_action must be positive")

    excluded = exclusion_sets(excluded_task_records, excluded_manifest_rows)
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in sorted(candidate_records, key=lambda row: str(row.get("packet_id"))):
        action = str(record.get("action_class") or "")
        packet_id = str(record.get("packet_id") or "")
        claim = task_claim(record)
        if action not in ACTION_CLASS_ORDER or not packet_id or not claim:
            continue
        try:
            split_group = split_group_for_record(record)
        except (KeyError, TypeError, ValueError):
            continue
        if (
            packet_id in excluded["source_task_ids"]
            or split_group in excluded["split_groups"]
            or claim in excluded["claims"]
        ):
            continue
        buckets[action].append(record)

    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    selected_groups: set[str] = set()
    selected_claims: set[str] = set()
    for action in ACTION_CLASS_ORDER:
        candidates = list(buckets[action])
        random.Random(f"{seed}:{action}").shuffle(candidates)
        chosen = 0
        for record in candidates:
            packet_id = str(record["packet_id"])
            split_group = split_group_for_record(record)
            claim = task_claim(record)
            if (
                packet_id in selected_ids
                or split_group in selected_groups
                or claim in selected_claims
            ):
                continue
            selected.append(record)
            selected_ids.add(packet_id)
            selected_groups.add(split_group)
            selected_claims.add(claim)
            chosen += 1
            if chosen == per_action:
                break
        if chosen < per_action:
            raise ValueError(
                f"insufficient source-disjoint candidates for {action}: "
                f"{chosen}<{per_action}"
            )
    return selected


def build_private_manifest(
    selected_records: Sequence[Mapping[str, Any]],
    *,
    per_action: int,
    tool_profile: str = "nullatlas_full",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected_records):
        row = stage_a_row_from_task_record(
            record,
            case_index=index,
            tool_profile=tool_profile,
        )
        case_id = f"stage_a_sealed_v1::{index:06d}"
        row["case_id"] = case_id
        row["dataset"] = DATASET
        row["model_visible_task"]["input_id"] = case_id
        rows.append(row)

    issues = validate_stage_a_manifest(
        rows,
        min_rows=per_action * len(ACTION_CLASS_ORDER),
        min_status_count=per_action,
        require_unique_split_groups=True,
    )
    if issues:
        raise ValueError("sealed manifest validation failed: " + "; ".join(issues))
    return rows


def manifest_overlap_counts(
    rows: Sequence[Mapping[str, Any]],
    exclusions: Mapping[str, set[str]],
) -> dict[str, int]:
    source_task_ids: set[str] = set()
    split_groups: set[str] = set()
    claims: set[str] = set()
    for row in rows:
        hidden = row["hidden_eval_metadata"]
        source_task_ids.add(str(hidden["source_task_id"]))
        split_groups.add(str(hidden["split_group"]))
        claims.add(manifest_claim(row))
    return {
        "source_task_id_overlap": len(source_task_ids & exclusions["source_task_ids"]),
        "split_group_overlap": len(split_groups & exclusions["split_groups"]),
        "normalized_claim_overlap": len(claims & exclusions["claims"]),
    }


def artifact_entry(
    path: str | Path,
    *,
    role: str,
    records: int,
    external_prefix: str,
) -> dict[str, Any]:
    return {
        "path": public_reference(path, external_prefix=external_prefix),
        "role": role,
        "records": records,
        "sha256": sha256_file(path),
    }


def build_public_commitment(
    *,
    candidate_tasks_path: str | Path,
    private_manifest_path: str | Path,
    public_task_paths: Sequence[str | Path],
    public_manifest_paths: Sequence[str | Path],
    candidate_records: Sequence[Mapping[str, Any]],
    private_rows: Sequence[Mapping[str, Any]],
    excluded_task_records: Sequence[Mapping[str, Any]],
    excluded_manifest_rows: Sequence[Mapping[str, Any]],
    per_action: int,
    seed: int,
) -> dict[str, Any]:
    exclusions = exclusion_sets(excluded_task_records, excluded_manifest_rows)
    overlap = manifest_overlap_counts(private_rows, exclusions)
    action_counts = Counter(
        str(row["hidden_eval_metadata"]["source_task_id"]).split("::")[1]
        for row in private_rows
    )
    status_counts = Counter(
        str(row["hidden_eval_metadata"]["gold_evidence_status"])
        for row in private_rows
    )
    family_counts = Counter(
        str(row["hidden_eval_metadata"]["case_family"])
        for row in private_rows
    )
    selection_payload = {
        "candidate_pool_sha256": sha256_file(candidate_tasks_path),
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "public_task_sha256": [sha256_file(path) for path in public_task_paths],
        "public_manifest_sha256": [sha256_file(path) for path in public_manifest_paths],
        "per_action": per_action,
        "seed": seed,
    }
    selection_commitment = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True).encode()
    ).hexdigest()
    ready = (
        len(private_rows) == per_action * len(ACTION_CLASS_ORDER)
        and all(value == 0 for value in overlap.values())
        and all(action_counts[action] == per_action for action in ACTION_CLASS_ORDER)
    )
    public_exclusions = [
        artifact_entry(
            path,
            role="public exposed task exclusion",
            records=len(load_task_records(path)),
            external_prefix="external_public_exclusion",
        )
        for path in public_task_paths
    ] + [
        artifact_entry(
            path,
            role="public Stage A manifest exclusion",
            records=len(load_stage_a_manifest(path)),
            external_prefix="external_public_exclusion",
        )
        for path in public_manifest_paths
    ]
    return {
        "dataset": COMMITMENT_DATASET,
        "sealed_dataset": DATASET,
        "selection": {
            "seed": seed,
            "per_action": per_action,
            "rows": len(private_rows),
            "candidate_pool_rows": len(candidate_records),
            "selection_commitment_sha256": selection_commitment,
        },
        "aggregate_balance": {
            "action_class_counts": dict(sorted(action_counts.items())),
            "evidence_status_counts": dict(sorted(status_counts.items())),
            "case_family_counts": dict(sorted(family_counts.items())),
        },
        "overlap_checks": overlap,
        "input_artifacts": {
            "private_candidate_pool": artifact_entry(
                candidate_tasks_path,
                role="private source candidate pool",
                records=len(candidate_records),
                external_prefix="external_private_input",
            ),
            "private_sealed_manifest": artifact_entry(
                private_manifest_path,
                role="private sealed Stage A manifest",
                records=len(private_rows),
                external_prefix="external_private_input",
            ),
            "public_exclusions": public_exclusions,
        },
        "public_safety_contract": {
            "private_manifest_committed": False,
            "candidate_pool_committed": False,
            "row_level_case_ids_published": False,
            "row_level_hidden_labels_published": False,
            "row_level_source_ids_published": False,
            "private_paths_redacted": True,
        },
        "decision": {
            "ready_for_one_time_sealed_evaluation": ready,
            "ready_for_training_on_sealed_rows": False,
            "ready_for_dpo_rlvr": False,
            "ready_for_hugging_face_publication": False,
            "selected_next_step": (
                "complete_tool_query_diagnostic_then_run_one_time_sealed_evaluation"
                if ready
                else "repair_sealed_extension_overlap_or_balance"
            ),
            "interpretation": (
                "The sealed extension is source-disjoint from all declared public "
                "task and manifest exclusions. Keep its rows private and use it once "
                "for model evaluation after the missing tool_query diagnostic."
                if ready
                else "The sealed extension failed balance or overlap checks and must not be used."
            ),
        },
    }


def render_markdown(commitment: Mapping[str, Any]) -> str:
    selection = commitment["selection"]
    overlap = commitment["overlap_checks"]
    decision = commitment["decision"]
    lines = [
        "# Stage A Sealed Evaluation Extension Commitment",
        "",
        "Purpose: commit to a private source-disjoint evaluation manifest without publishing rows or hidden labels.",
        "",
        "## Summary",
        "",
        f"- Private sealed rows: {selection['rows']}",
        f"- Rows per action family: {selection['per_action']}",
        f"- Candidate pool rows: {selection['candidate_pool_rows']}",
        f"- Source-task overlap: {overlap['source_task_id_overlap']}",
        f"- Split-group overlap: {overlap['split_group_overlap']}",
        f"- Normalized-claim overlap: {overlap['normalized_claim_overlap']}",
        (
            "- Ready for one-time sealed evaluation: "
            f"`{decision['ready_for_one_time_sealed_evaluation']}`"
        ),
        f"- Selection commitment: `{selection['selection_commitment_sha256']}`",
        "",
        "The private manifest, row-level labels, source IDs, and local storage paths are not public artifacts.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-tasks", required=True)
    parser.add_argument("--private-manifest-out", required=True)
    parser.add_argument("--public-task-exclusion", action="append", default=[])
    parser.add_argument("--public-manifest-exclusion", action="append", default=[])
    parser.add_argument("--public-json-out", required=True)
    parser.add_argument("--public-md-out", required=True)
    parser.add_argument("--per-action", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tool-profile", default="nullatlas_full")
    args = parser.parse_args()

    candidate_path = Path(args.candidate_tasks)
    private_manifest_path = Path(args.private_manifest_out)
    require_external_private_path(candidate_path, role="candidate task pool")
    require_external_private_path(private_manifest_path, role="private sealed manifest")

    public_task_paths = [Path(path) for path in args.public_task_exclusion]
    public_manifest_paths = [Path(path) for path in args.public_manifest_exclusion]
    if not public_task_paths or not public_manifest_paths:
        raise ValueError("at least one public task and manifest exclusion is required")

    candidate_records = load_task_records(candidate_path)
    excluded_task_records = [
        row for path in public_task_paths for row in load_task_records(path)
    ]
    excluded_manifest_rows = [
        row for path in public_manifest_paths for row in load_stage_a_manifest(path)
    ]
    selected = select_sealed_records(
        candidate_records,
        excluded_task_records=excluded_task_records,
        excluded_manifest_rows=excluded_manifest_rows,
        per_action=args.per_action,
        seed=args.seed,
    )
    private_rows = build_private_manifest(
        selected,
        per_action=args.per_action,
        tool_profile=args.tool_profile,
    )
    private_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_stage_a_manifest(private_manifest_path, private_rows)
    commitment = build_public_commitment(
        candidate_tasks_path=candidate_path,
        private_manifest_path=private_manifest_path,
        public_task_paths=public_task_paths,
        public_manifest_paths=public_manifest_paths,
        candidate_records=candidate_records,
        private_rows=private_rows,
        excluded_task_records=excluded_task_records,
        excluded_manifest_rows=excluded_manifest_rows,
        per_action=args.per_action,
        seed=args.seed,
    )
    if not commitment["decision"]["ready_for_one_time_sealed_evaluation"]:
        raise ValueError("sealed extension failed public commitment checks")
    write_json(args.public_json_out, commitment)
    Path(args.public_md_out).write_text(render_markdown(commitment))
    print(json.dumps(commitment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
