#!/usr/bin/env python3
"""Run independent Gray calibration and locked Fromm transfer replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .independent_calibration import build_independent_calibration
from .manifest import load_c5_manifest, write_c5_manifest
from .source_pilot import public_artifact_issues, sha256_file


def render_markdown(report: Mapping[str, Any]) -> str:
    selection = report["overlap_and_selection"]
    calibration = report["calibration"]
    source_policies = report["source_cohort_policies"]
    transfer = report["locked_fromm_evaluation"]
    decision = report["decision"]
    lines = [
        "# Stage B C5 Independent Calibration Replay",
        "",
        "This no-model replay calibrates AF3 ranking-score trust routing on the",
        "Hitawala-Gray post-cutoff Ab-Ag/Nb-Ag benchmark, after removing every",
        "PDB ID present in the Fromm panel. It then applies only a certified",
        "antibody gate to the already frozen Fromm evaluation rows.",
        "",
        "This is independent-source published-label evidence, not a blinded",
        "hidden benchmark or a new structure-prediction result.",
        "",
        "## Intake And Overlap",
        "",
        f"- Source rows: `{report['source']['rows']}`; complete bound rows: "
        f"`{report['source']['bound_rows_retained']}` across "
        f"`{report['source']['bound_targets_retained']}` targets.",
        f"- Source selection before overlap exclusion: "
        f"`{selection['selected_before_overlap']}` targets.",
        f"- Excluded overlap: `{selection['overlapping_pdb_ids_excluded']}` "
        f"PDB IDs / `{selection['overlapping_complexes_excluded']}` complex "
        "copies.",
        f"- Independent calibration cohort: "
        f"`{selection['selected_after_overlap']}` targets "
        f"(`{selection['selected_by_format']['antibody']}` antibody, "
        f"`{selection['selected_by_format']['nanobody']}` nanobody); residual "
        "PDB overlap `0`.",
        f"- Selection ties: `{selection['targets_with_top_ranking_ties']}` at "
        "ranking score, "
        f"`{selection['targets_with_post_iptm_ties']}` after ipTM-HA.",
        "",
        "## Independent Calibration",
        "",
        "| Format | Targets | Trust-all failures | Fixed 0.80 failures | "
        "Certified gate | Closest upper bound |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    primary_key = "alpha_0.30"
    for antibody_format in ("antibody", "nanobody"):
        policy = source_policies[antibody_format]
        certificate = calibration["format_specific_certificates"][
            antibody_format
        ][primary_key]
        closest = certificate["closest_candidate"]
        closest_upper = (
            f"{closest['risk_upper_bound']:.3f}" if closest else "n/a"
        )
        lines.append(
            f"| `{antibody_format}` | "
            f"{policy['trust_all']['targets']} | "
            f"{policy['trust_all']['failures_among_trusted']} | "
            f"{policy['fixed_ranking_score_0_80']['failures_among_trusted']} | "
            f"{'yes' if certificate['certified'] else 'no'} | "
            f"{closest_upper} |"
        )
    lines.extend(
        [
            "",
            "The finite candidate family is the pre-existing `0.50-0.99` grid",
            "with a uniform Hoeffding/union-bound correction (`delta=0.10`).",
            "Gray `ranking_score` and Fromm `ranking_confidence` are aligned by",
            "AF3 score name and range; exact model-version equivalence is not",
            "claimed.",
            "",
            "## Locked Fromm Replay",
            "",
            "| Policy | Trusted | Failures among trusted | Failure rate | Coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in (
        "trust_all",
        "fixed_ranking_score_0_80",
        "independent_calibration_gate",
    ):
        result = transfer[name]
        lines.append(
            f"| `{name}` | {result['trusted']}/{result['targets']} | "
            f"{result['failures_among_trusted']} | "
            f"{result['failure_rate_among_trusted']:.3f} | "
            f"{result['coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Independent source intake, overlap exclusion, privacy projection, "
            "and canonical trajectory validation: "
            f"`{'pass' if decision['independent_source_adapter_passed'] else 'fail'}`.",
            "- Antibody ranking-score trust certificate: "
            f"`{'pass' if decision['antibody_ranking_gate_certified'] else 'not certified'}`.",
            "- Nanobody ranking-score trust certificate: "
            f"`{'pass' if decision['nanobody_ranking_gate_certified'] else 'not certified'}`.",
            "- External trust remains disabled unless the independent antibody "
            "certificate passes.",
            "- Model training, DPO, and RLVR remain closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    manifest_path: str | Path,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    write_c5_manifest(manifest_path, rows)
    public_manifest_path = Path(manifest_path)
    report["derived_manifest"] = {
        "path": (
            public_manifest_path.as_posix()
            if not public_manifest_path.is_absolute()
            else public_manifest_path.name
        ),
        "sha256": sha256_file(manifest_path),
        "rows": len(rows),
    }
    privacy_issues = public_artifact_issues(report)
    if privacy_issues:
        raise ValueError(
            "compact report failed privacy validation: "
            + ",".join(privacy_issues)
        )
    Path(json_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    Path(markdown_path).write_text(render_markdown(report))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gray-scores",
        required=True,
        help="External pinned final_af3_rmsds.csv; never copied into repo.",
    )
    parser.add_argument(
        "--fromm-manifest",
        default="c5_antibody_ood/c5_source_backed_manifest_v1.jsonl",
    )
    parser.add_argument(
        "--out-manifest",
        default=(
            "c5_antibody_ood/"
            "c5_gray_independent_calibration_manifest_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--out-json",
        default=(
            "c5_antibody_ood/"
            "c5_gray_independent_calibration_result_2026-07-25.json"
        ),
    )
    parser.add_argument(
        "--out-md",
        default=(
            "c5_antibody_ood/"
            "C5_GRAY_INDEPENDENT_CALIBRATION_RESULT_2026-07-25.md"
        ),
    )
    args = parser.parse_args()

    fromm_rows = load_c5_manifest(args.fromm_manifest)
    rows, report = build_independent_calibration(
        args.gray_scores,
        fromm_rows,
        fromm_manifest_sha256=sha256_file(args.fromm_manifest),
    )
    write_outputs(
        rows,
        report,
        manifest_path=args.out_manifest,
        json_path=args.out_json,
        markdown_path=args.out_md,
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return (
        0
        if report["decision"]["independent_source_adapter_passed"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
