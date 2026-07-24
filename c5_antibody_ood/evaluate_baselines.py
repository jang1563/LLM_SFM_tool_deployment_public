#!/usr/bin/env python3
"""Evaluate the public C5 deterministic policy baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .manifest import load_c5_manifest
from .policies import build_c5_baseline_report


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact aggregate result without row-level hidden labels."""

    lines = [
        "# Stage B C5 No-API Policy Prototype",
        "",
        "This synthetic public fixture tests whether antibody-antigen specialist",
        "trust is blocked unless metric scope, regime-matched calibration, an",
        "RCPS threshold, and baseline arbitration are complete.",
        "",
        "## Results",
        "",
        "| Policy | Exact pass | Mean score | Unsafe trust |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, summary in report["policies"].items():
        lines.append(
            f"| `{name}` | {summary['exact_pass']}/{summary['rows']} | "
            f"{summary['mean_score']:.3f} | {summary['unsafe_trust']} |"
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                f"- Oracle trajectories: `{report['oracle']['exact_pass']}/"
                f"{report['oracle']['rows']}`."
            ),
            (
                "- No-API manifest and fail-closed prototype: "
                f"`{'pass' if decision['no_api_prototype_passed'] else 'fail'}`."
            ),
            "- This is a synthetic policy-test result, not calibration evidence.",
            "- A source-backed C5 pilot with frozen interface labels is required next.",
            "- Model training, DPO/RLVR, and independent-transfer claims remain closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    report: dict[str, Any],
    *,
    manifest_path: str | Path,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    report["input_artifact"] = {
        "path": str(Path(manifest_path)),
        "sha256": sha256_file(manifest_path),
    }
    Path(json_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    Path(markdown_path).write_text(render_markdown(report))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="c5_antibody_ood/c5_policy_test_manifest_v1.jsonl",
    )
    parser.add_argument(
        "--out-json",
        default="c5_antibody_ood/c5_policy_baseline_result_2026-07-24.json",
    )
    parser.add_argument(
        "--out-md",
        default="c5_antibody_ood/C5_POLICY_BASELINE_RESULT_2026-07-24.md",
    )
    args = parser.parse_args()
    rows = load_c5_manifest(args.manifest)
    report = build_c5_baseline_report(rows)
    write_outputs(
        report,
        manifest_path=args.manifest,
        json_path=args.out_json,
        markdown_path=args.out_md,
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0 if report["decision"]["no_api_prototype_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
