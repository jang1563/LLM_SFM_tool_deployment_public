#!/usr/bin/env python3
"""Run the source-backed C5 public-score pilot from an external AF3 CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .manifest import write_c5_manifest
from .source_pilot import build_source_pilot, public_artifact_issues, sha256_file


def render_markdown(report: Mapping[str, Any]) -> str:
    selection = report["selection"]
    split = report["split"]
    policies = report["policies"]
    decision = report["decision"]
    lines = [
        "# Stage B C5 Source-Backed Public-Score Pilot",
        "",
        "This no-model replay evaluates target-grouped trust routing over the",
        "published AlphaFold 3 antibody-antigen score table from Fromm et al.",
        "It is a published-label replay, not an independent hidden benchmark.",
        "",
        "## Intake And Split",
        "",
        f"- Source rows: `{report['source']['rows']}` across "
        f"`{report['source']['targets']}` targets.",
        f"- Selected targets: `{selection['selected_targets']}`; DockQ successes: "
        f"`{selection['selected_interface_successes']}` "
        f"(`{selection['selected_interface_success_rate']:.3f}`).",
        f"- Top-ranking-score ties: "
        f"`{selection['targets_with_top_score_ties']}` targets.",
        f"- Frozen target split: `{split['calibration_targets']}` calibration / "
        f"`{split['evaluation_targets']}` evaluation; overlap `0`.",
        "",
        "## Frozen Evaluation",
        "",
        "| Policy | Trusted | Failures among trusted | Failure rate | Coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in (
        "trust_all",
        "generic_fixed_iptm_0_80",
        "regime_specific_hoeffding",
        "fail_closed",
    ):
        result = policies[name]
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
            "- Source intake, privacy projection, split isolation, and canonical "
            f"trajectory validation: "
            f"`{'pass' if decision['source_backed_pilot_passed'] else 'fail'}`.",
            "- No regime-specific threshold is certified at the primary "
            f"`alpha={report['policies']['regime_specific_hoeffding']['alpha']:.2f}` "
            "gate."
            if not decision["regime_specific_trust_certified"]
            else "- The primary regime-specific threshold is certified.",
            "- The fixed `ipTM >= 0.80` baseline is not a calibrated general-PPI "
            "transfer gate.",
            "- Model training and DPO/RLVR remain closed.",
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
    Path(json_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    Path(markdown_path).write_text(render_markdown(report))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        required=True,
        help="External canonical scores_alphafold3.csv; never copied into repo.",
    )
    parser.add_argument(
        "--out-manifest",
        default="c5_antibody_ood/c5_source_backed_manifest_v1.jsonl",
    )
    parser.add_argument(
        "--out-json",
        default=(
            "c5_antibody_ood/"
            "c5_source_backed_pilot_result_2026-07-25.json"
        ),
    )
    parser.add_argument(
        "--out-md",
        default=(
            "c5_antibody_ood/"
            "C5_SOURCE_BACKED_PILOT_RESULT_2026-07-25.md"
        ),
    )
    args = parser.parse_args()

    rows, report = build_source_pilot(args.scores)
    write_outputs(
        rows,
        report,
        manifest_path=args.out_manifest,
        json_path=args.out_json,
        markdown_path=args.out_md,
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0 if report["decision"]["source_backed_pilot_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
