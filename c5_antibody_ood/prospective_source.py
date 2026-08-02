"""SAbDab2 source adapter for the preregistered prospective C5 panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import load_c5_manifest, write_c5_manifest
from .prospective_panel import (
    PANEL_ROLES,
    SABDAB2_SPLIT_BYTES,
    SABDAB2_SPLIT_SHA256,
    build_panel_commitment,
    canonical_sha256,
    validate_preregistration,
    write_json,
)
from .source_pilot import sha256_file


SABDAB2_EXPECTED_ROWS = 15_641
SABDAB2_EXPECTED_COLUMNS = 48
SABDAB2_REQUIRED_COLUMNS = frozenset(
    {
        "INSTANCE",
        "PDB_ID",
        "SABDAB_ID",
        "PDBdepo",
        "method",
        "resolution",
        "type",
        "holo",
        "Hchain",
        "Lchain",
        "agchains",
        "agtypes",
        "ab_ag_cluster",
        "ab_ag_split",
    }
)
SABDAB2_PAIRED_TYPES = frozenset({"FAB", "FAB+FC", "FV"})
SABDAB2_SOURCE_ALLOWLIST = tuple(sorted(SABDAB2_REQUIRED_COLUMNS))
ROLE_ORDER = {role: index for index, role in enumerate(PANEL_ROLES)}


class ProspectiveSourceError(ValueError):
    """Raised when source intake or preregistered selection fails closed."""


@dataclass(frozen=True)
class ProspectiveSourceContract:
    expected_sha256: str = SABDAB2_SPLIT_SHA256
    expected_bytes: int = SABDAB2_SPLIT_BYTES
    expected_rows: int = SABDAB2_EXPECTED_ROWS
    expected_columns: int = SABDAB2_EXPECTED_COLUMNS


@dataclass(frozen=True)
class EligibleTarget:
    instance_id: str
    pdb_id: str
    sabdab_id: str
    source_split: str
    release_date: str
    experimental_method: str
    resolution_angstrom: float
    heavy_chain: str
    light_chain: str
    antigen_chains: tuple[str, ...]
    source_row_sha256: str
    source_cluster_sha256: str


def intake_sabdab2_split(
    path: str | Path,
    preregistration: Mapping[str, Any],
    *,
    contract: ProspectiveSourceContract = ProspectiveSourceContract(),
) -> tuple[list[EligibleTarget], dict[str, Any]]:
    """Read the pinned split while retaining only public-safe metadata."""

    preregistration_issues = validate_preregistration(preregistration)
    if preregistration_issues:
        raise ProspectiveSourceError(
            "preregistration_invalid:" + ",".join(preregistration_issues)
        )
    source = Path(path)
    if not source.is_file():
        raise ProspectiveSourceError("source_missing")
    actual_bytes = source.stat().st_size
    if actual_bytes != contract.expected_bytes:
        raise ProspectiveSourceError(
            f"source_byte_count:{actual_bytes}!={contract.expected_bytes}"
        )
    actual_sha256 = sha256_file(source)
    if actual_sha256 != contract.expected_sha256:
        raise ProspectiveSourceError("source_sha256_mismatch")

    csv.field_size_limit(sys.maxsize)
    eligible: list[EligibleTarget] = []
    exclusion_counts: Counter[str] = Counter()
    source_split_counts: Counter[str] = Counter()
    row_count = 0
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = SABDAB2_REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ProspectiveSourceError(
                "source_missing_columns:" + ",".join(sorted(missing))
            )
        if len(fieldnames) != contract.expected_columns:
            raise ProspectiveSourceError(
                f"source_column_count:{len(fieldnames)}"
                f"!={contract.expected_columns}"
            )
        for row_count, row in enumerate(reader, start=1):
            source_split = _required(row, "ab_ag_split")
            source_split_counts[source_split] += 1
            target, reason = _project_if_eligible(row, preregistration)
            if target is None:
                exclusion_counts[reason] += 1
            else:
                eligible.append(target)
    if row_count != contract.expected_rows:
        raise ProspectiveSourceError(
            f"source_row_count:{row_count}!={contract.expected_rows}"
        )
    if set(source_split_counts) != {"test", "train"}:
        raise ProspectiveSourceError("source_split_values_invalid")

    return eligible, {
        "source": {
            "rows": row_count,
            "columns": len(fieldnames),
            "bytes": actual_bytes,
            "sha256": actual_sha256,
            "split_counts": dict(sorted(source_split_counts.items())),
            "allowlisted_columns": list(SABDAB2_SOURCE_ALLOWLIST),
            "allowlisted_column_count": len(SABDAB2_SOURCE_ALLOWLIST),
            "excluded_column_count": (
                len(fieldnames) - len(SABDAB2_SOURCE_ALLOWLIST)
            ),
        },
        "eligibility": {
            "retained_rows": len(eligible),
            "retained_by_split": dict(
                sorted(Counter(row.source_split for row in eligible).items())
            ),
            "excluded_by_reason": dict(sorted(exclusion_counts.items())),
        },
        "privacy": {
            "raw_sequences_emitted": False,
            "raw_numbering_or_cdr_fields_emitted": False,
            "raw_structures_or_paths_emitted": False,
            "native_interface_labels_read": False,
            "dockq_values_read": False,
        },
    }


def select_public_panel(
    eligible: Sequence[EligibleTarget],
    preregistration: Mapping[str, Any],
    *,
    blocked_pdb_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen metadata-only selection and reserve policy."""

    selection = preregistration["protocol"]["target_selection"]
    public_seed = str(selection["public_seed"])
    blocked = {value.lower() for value in blocked_pdb_ids}
    by_split: dict[str, list[EligibleTarget]] = {"train": [], "test": []}
    overlap_rows = 0
    for target in eligible:
        if target.pdb_id in blocked:
            overlap_rows += 1
            continue
        by_split[target.source_split].append(target)

    selected: list[dict[str, Any]] = []
    used_pdb: set[str] = set()
    used_sabdab: set[str] = set()
    duplicate_pdb_skipped = 0
    duplicate_sabdab_skipped = 0
    conflict_clusters_skipped = 0
    cluster_first = selection.get("deduplicate_source_cluster") is True
    selection_definitions = (
        (
            "calibration",
            "calibration_reserve",
            selection["calibration"],
        ),
        (
            "evaluation",
            "evaluation_reserve",
            selection["evaluation"],
        ),
    )
    for primary_role, reserve_role, definition in selection_definitions:
        source_split = str(definition["source_split"])
        primary_count = int(definition["primary_targets"])
        reserve_count = int(definition["reserve_targets"])
        required = primary_count + reserve_count
        accepted: list[EligibleTarget] = []
        if cluster_first:
            grouped: defaultdict[str, list[EligibleTarget]] = defaultdict(list)
            for target in by_split[source_split]:
                grouped[target.source_cluster_sha256].append(target)
            ordered_clusters = sorted(
                grouped,
                key=lambda cluster: (
                    hashlib.sha256(
                        f"{public_seed}|cluster|{cluster}".encode()
                    ).hexdigest(),
                    cluster,
                ),
            )
            for cluster in ordered_clusters:
                ordered_targets = sorted(
                    grouped[cluster],
                    key=lambda target: (
                        hashlib.sha256(
                            (
                                f"{public_seed}|target|{cluster}|"
                                f"{target.instance_id}"
                            ).encode()
                        ).hexdigest(),
                        target.instance_id,
                    ),
                )
                selected_target: EligibleTarget | None = None
                for target in ordered_targets:
                    if target.pdb_id in used_pdb:
                        duplicate_pdb_skipped += 1
                        continue
                    if target.sabdab_id in used_sabdab:
                        duplicate_sabdab_skipped += 1
                        continue
                    selected_target = target
                    break
                if selected_target is None:
                    conflict_clusters_skipped += 1
                    continue
                accepted.append(selected_target)
                used_pdb.add(selected_target.pdb_id)
                used_sabdab.add(selected_target.sabdab_id)
                if len(accepted) == required:
                    break
        else:
            ordered = sorted(
                by_split[source_split],
                key=lambda target: (
                    hashlib.sha256(
                        (
                            public_seed
                            + "|"
                            + target.instance_id
                        ).encode()
                    ).hexdigest(),
                    target.instance_id,
                ),
            )
            for target in ordered:
                if target.pdb_id in used_pdb:
                    duplicate_pdb_skipped += 1
                    continue
                if target.sabdab_id in used_sabdab:
                    duplicate_sabdab_skipped += 1
                    continue
                accepted.append(target)
                used_pdb.add(target.pdb_id)
                used_sabdab.add(target.sabdab_id)
                if len(accepted) == required:
                    break
        if len(accepted) != required:
            unit = "clusters" if cluster_first else "targets"
            raise ProspectiveSourceError(
                f"insufficient_{source_split}_{unit}:"
                f"{len(accepted)}<{required}"
            )
        for index, target in enumerate(accepted):
            role = primary_role if index < primary_count else reserve_role
            rank = index + 1 if index < primary_count else index - primary_count + 1
            selected.append(_public_row(target, role=role, rank=rank))

    selected.sort(
        key=lambda row: (
            ROLE_ORDER[str(row["panel_role"])],
            int(row["selection_rank"]),
        )
    )
    audit = {
        "eligible_rows_before_overlap_exclusion": len(eligible),
        "blocked_pdb_ids": len(blocked),
        "blocked_overlap_rows_excluded": overlap_rows,
        "duplicate_pdb_rows_skipped": duplicate_pdb_skipped,
        "duplicate_sabdab_rows_skipped": duplicate_sabdab_skipped,
        "selected_rows": len(selected),
        "selected_by_role": dict(
            sorted(Counter(row["panel_role"] for row in selected).items())
        ),
        "selected_unique_pdb_ids": len(
            {row["pdb_id"] for row in selected}
        ),
        "selected_unique_sabdab_ids": len(
            {row["sabdab_id"] for row in selected}
        ),
        "selected_source_cluster_overlap": len(
            {
                row["source_cluster_sha256"]
                for row in selected
                if row["source_split"]
                == selection["calibration"]["source_split"]
            }
            & {
                row["source_cluster_sha256"]
                for row in selected
                if row["source_split"]
                == selection["evaluation"]["source_split"]
            }
        ),
    }
    if cluster_first:
        audit.update(
            {
                "selection_unit": "official_ab_ag_cluster",
                "selected_unique_source_clusters": len(
                    {row["source_cluster_sha256"] for row in selected}
                ),
                "conflict_clusters_skipped": conflict_clusters_skipped,
            }
        )
    return selected, audit


def build_prospective_panel(
    source_path: str | Path,
    preregistration: Mapping[str, Any],
    *,
    blocked_pdb_ids: set[str],
    contract: ProspectiveSourceContract = ProspectiveSourceContract(),
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build the panel, audit, and fail-closed prediction commitment."""

    eligible, source_audit = intake_sabdab2_split(
        source_path,
        preregistration,
        contract=contract,
    )
    rows, selection_audit = select_public_panel(
        eligible,
        preregistration,
        blocked_pdb_ids=blocked_pdb_ids,
    )
    commitment = build_panel_commitment(
        rows,
        preregistration,
        blocked_pdb_ids=blocked_pdb_ids,
    )
    audit = {
        "schema_version": "c5_prospective_source_intake_audit_v1",
        "preregistration_id": preregistration["preregistration_id"],
        "protocol_sha256": preregistration["commitment"]["protocol_sha256"],
        **source_audit,
        "selection": selection_audit,
        "validation": commitment["validation"],
        "decision": commitment["decision"],
    }
    return rows, audit, commitment


def blocked_pdb_ids_from_manifests(
    fromm_manifest: str | Path,
    gray_manifest: str | Path,
) -> set[str]:
    """Return all prior public C5 PDB IDs without reading labels."""

    blocked: set[str] = set()
    for row in load_c5_manifest(fromm_manifest):
        blocked.add(
            str(row["model_visible_task"]["complex_id"]).lower()
        )
    for row in load_c5_manifest(gray_manifest):
        blocked.add(
            str(row["model_visible_task"]["complex_id"])
            .split("_", maxsplit=1)[0]
            .lower()
        )
    return blocked


def _project_if_eligible(
    row: Mapping[str, str | None],
    preregistration: Mapping[str, Any],
) -> tuple[EligibleTarget | None, str]:
    selection = preregistration["protocol"]["target_selection"]
    if _optional(row, "holo") != "True":
        return None, "not_antibody_antigen_complex"
    if _optional(row, "type") not in SABDAB2_PAIRED_TYPES:
        return None, "not_paired_chain_antibody"
    if _optional(row, "ab_ag_split") not in {
        selection["calibration"]["source_split"],
        selection["evaluation"]["source_split"],
    }:
        return None, "outside_preregistered_source_splits"
    release_date = _optional(row, "PDBdepo")
    if not release_date:
        return None, "candidate_metadata_missing"
    if release_date < selection["minimum_release_date"]:
        return None, "before_release_date_cutoff"
    method = _optional(row, "method")
    if not method:
        return None, "candidate_metadata_missing"
    if method not in selection["allowed_experimental_methods"]:
        return None, "experimental_method_excluded"
    try:
        resolution = float(_optional(row, "resolution"))
    except ValueError:
        return None, "resolution_missing_or_invalid"
    if (
        not math.isfinite(resolution)
        or resolution <= 0
        or resolution > selection["maximum_resolution_angstrom"]
    ):
        return None, "resolution_outside_limit"

    raw_antigen_types = _optional(row, "agtypes")
    if not raw_antigen_types:
        return None, "candidate_metadata_missing"
    antigen_types = _split_source_list(
        raw_antigen_types,
        require_unique=False,
    )
    if "PROTEIN" not in antigen_types:
        return None, "no_protein_antigen"
    if any(
        not _optional(row, key)
        for key in ("Hchain", "Lchain", "agchains")
    ):
        return None, "chain_mapping_invalid"
    try:
        heavy = _single_chain(row, "Hchain")
        light = _single_chain(row, "Lchain")
        all_antigen_chains = _split_source_list(_required(row, "agchains"))
    except ValueError:
        return None, "chain_mapping_invalid"
    if len(all_antigen_chains) != len(antigen_types):
        return None, "chain_mapping_invalid"
    antigens = tuple(
        chain
        for chain, antigen_type in zip(
            all_antigen_chains,
            antigen_types,
            strict=True,
        )
        if antigen_type == "PROTEIN"
    )
    if not antigens or "+" in antigens:
        return None, "chain_mapping_invalid"
    if len({heavy, light, *antigens}) != 2 + len(antigens):
        return None, "chain_mapping_invalid"

    if any(
        not _optional(row, key)
        for key in ("INSTANCE", "PDB_ID", "SABDAB_ID", "ab_ag_cluster")
    ):
        return None, "candidate_metadata_missing"
    instance_id = _required(row, "INSTANCE")
    pdb_id = _short_pdb_id(_required(row, "PDB_ID"))
    sabdab_id = _required(row, "SABDAB_ID")
    cluster = _required(row, "ab_ag_cluster")
    projected = {
        key: _required(row, key)
        for key in SABDAB2_SOURCE_ALLOWLIST
    }
    return (
        EligibleTarget(
            instance_id=instance_id,
            pdb_id=pdb_id,
            sabdab_id=sabdab_id,
            source_split=_required(row, "ab_ag_split"),
            release_date=release_date,
            experimental_method=method,
            resolution_angstrom=resolution,
            heavy_chain=heavy,
            light_chain=light,
            antigen_chains=antigens,
            source_row_sha256=canonical_sha256(projected),
            source_cluster_sha256=canonical_sha256(
                "sabdab2-ab-ag-cluster-v0.1.0|" + cluster
            ),
        ),
        "",
    )


def _public_row(
    target: EligibleTarget,
    *,
    role: str,
    rank: int,
) -> dict[str, Any]:
    return {
        "target_id": f"c5-sabdab2::{target.instance_id.lower()}",
        "pdb_id": target.pdb_id,
        "sabdab_id": target.sabdab_id,
        "instance_id": target.instance_id,
        "source_split": target.source_split,
        "panel_role": role,
        "selection_rank": rank,
        "release_date": target.release_date,
        "experimental_method": target.experimental_method,
        "resolution_angstrom": target.resolution_angstrom,
        "antibody_format": "paired_chain",
        "chain_role_mapping": [
            {
                "chain_id": target.heavy_chain,
                "role": "antibody_heavy",
            },
            {
                "chain_id": target.light_chain,
                "role": "antibody_light",
            },
            *[
                {"chain_id": chain, "role": "antigen"}
                for chain in target.antigen_chains
            ],
        ],
        "source_row_sha256": target.source_row_sha256,
        "source_cluster_sha256": target.source_cluster_sha256,
    }


def _required(row: Mapping[str, str | None], key: str) -> str:
    value = _optional(row, key)
    if not value:
        raise ProspectiveSourceError(f"source_value_missing:{key}")
    return value


def _optional(row: Mapping[str, str | None], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _split_source_list(
    value: str,
    *,
    require_unique: bool = True,
) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split("/") if part.strip())
    if not values or (require_unique and len(set(values)) != len(values)):
        raise ValueError("source_list_invalid")
    return values


def _single_chain(row: Mapping[str, str | None], key: str) -> str:
    values = _split_source_list(_required(row, key))
    if len(values) != 1 or values[0] == "+":
        raise ValueError("single_chain_required")
    return values[0]


def _short_pdb_id(value: str) -> str:
    normalized = value.lower()
    prefix = "pdb_0000"
    if not normalized.startswith(prefix):
        raise ProspectiveSourceError("pdb_id_prefix_invalid")
    short = normalized[len(prefix) :]
    if len(short) != 4 or not short.isalnum():
        raise ProspectiveSourceError("pdb_id_invalid")
    return short


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--fromm-manifest", type=Path, required=True)
    parser.add_argument("--gray-manifest", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--commitment-out", type=Path, required=True)
    args = parser.parse_args()

    preregistration = _load_json(args.preregistration)
    blocked = blocked_pdb_ids_from_manifests(
        args.fromm_manifest,
        args.gray_manifest,
    )
    rows, audit, commitment = build_prospective_panel(
        args.split_csv,
        preregistration,
        blocked_pdb_ids=blocked,
    )
    write_c5_manifest(args.manifest_out, rows)
    write_json(args.audit_out, audit)
    write_json(args.commitment_out, commitment)
    print(
        json.dumps(
            {
                "manifest_rows": len(rows),
                "protocol_sha256": preregistration["commitment"][
                    "protocol_sha256"
                ],
                "ready_for_prediction": commitment["decision"][
                    "ready_for_prediction"
                ],
                "validation_issues": commitment["validation"]["issues"],
            },
            sort_keys=True,
        )
    )
    return int(not commitment["decision"]["ready_for_prediction"])


if __name__ == "__main__":
    raise SystemExit(main())
