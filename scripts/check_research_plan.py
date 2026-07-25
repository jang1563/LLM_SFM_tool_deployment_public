#!/usr/bin/env python3
"""Validate the active research plan against drift-critical checkpoints."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research" / "2026-06-25_posttrain_tool_use_landscape" / "LONG_TERM_RESEARCH_PLAN_2026-07-04.md"
ROADMAP = ROOT / "ROADMAP.md"
README = ROOT / "README.md"
PUBLIC_STATUS = ROOT / "STATUS.md"
C5_SOURCE_REPORT = (
    ROOT
    / "c5_antibody_ood"
    / "c5_source_backed_pilot_result_2026-07-25.json"
)
C5_PROVENANCE = (
    ROOT / "c5_antibody_ood" / "SOURCE_BACKED_PILOT_PROVENANCE.md"
)


def read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise RuntimeError(f"Could not read {path.relative_to(ROOT)}: {exc}") from exc


def require_contains(issues: list[str], text: str, needle: str, label: str) -> None:
    if needle not in text:
        issues.append(f"missing {label}: {needle!r}")


def require_pattern(issues: list[str], text: str, pattern: str, label: str) -> None:
    if not re.search(pattern, text, flags=re.DOTALL):
        issues.append(f"missing {label}: /{pattern}/")


def main() -> int:
    issues: list[str] = []
    plan = read(PLAN)
    roadmap = read(ROADMAP)
    readme = read(README)
    public_status = read(PUBLIC_STATUS)
    c5_provenance = read(C5_PROVENANCE)
    c5_report = json.loads(read(C5_SOURCE_REPORT))

    require_contains(
        issues,
        plan,
        "## Research-First 6-8 Week Execution Board",
        "research-first execution board",
    )
    require_pattern(
        issues,
        plan,
        r"Week 1.*enum_action.*Week 2.*tool_query.*routing_after_loop",
        "component experiment order",
    )
    require_contains(
        issues,
        plan,
        "DPO/RLVR remains gated until all three slices have held-out reports.",
        "DPO/RLVR component gate",
    )
    require_contains(
        issues,
        plan,
        "--decode-mode enum_candidate_score",
        "enum candidate-scoring repair path",
    )
    require_pattern(
        issues,
        plan,
        r"Do not repeat or tune on the completed 25-row sealed set\. Keep DPO, RLVR, and\s+Hugging Face publication closed until a learned routing repair beats static\s+baselines, adds useful decisive coverage, and survives independent evaluation\.",
        "post-sealed method and publication gate",
    )
    for needle in (
        "12 rows are balanced across `trust`, `baseline`, `verify`, and `defer`",
        "oracle and fail-closed trajectories pass 12/12",
        "`trust_all` produces 9 unsafe trusts",
        "groups every sampled prediction by `complex_id`",
    ):
        require_contains(issues, plan, needle, "completed C5 prototype")
    for needle in (
        "22,000 AF3 samples over 110 targets",
        "frozen 55/55 target-group split",
        "trust-all has 28 failures",
        "certifies no trusted set",
        "independent calibration evidence",
    ):
        require_contains(issues, plan, needle, "source-backed C5 checkpoint")
    for needle in (
        "`complex_id`",
        "metric type, scope, and value",
        "calibration dataset ID",
        "hidden interface label status",
        "expected terminal action",
    ):
        require_contains(issues, plan, needle, "C5 required field")

    require_contains(
        issues,
        plan,
        "A repeatedly inspected held-out slice must be frozen as development data.",
        "sealed evaluation drift gate",
    )
    require_contains(
        issues,
        plan,
        "source-separated sealed extension with private row-level labels",
        "sealed extension privacy gate",
    )

    for needle in (
        "### 1. Stage A Component Smoke Results",
        "### 4. Audited RLVR Gate",
        "### 6. Release v0.1 And Hugging Face Package",
        "python scripts/check_research_plan.py",
    ):
        require_contains(issues, roadmap, needle, "roadmap research-first milestone")

    require_contains(
        issues,
        readme,
        "python scripts/check_research_plan.py",
        "README quickstart research-plan check",
    )
    require_contains(
        issues,
        public_status,
        "stage_b_c5_independent_calibration_evidence",
        "public STATUS C5 research decision",
    )
    require_pattern(
        issues,
        public_status,
        r"[Dd]o not tune on or rescore these\s+25 sealed rows",
        "public STATUS sealed-set reuse prohibition",
    )
    require_contains(
        issues,
        public_status,
        "stage_a_sealed_extension_commitment_2026-07-10.json",
        "public STATUS sealed commitment artifact",
    )
    require_contains(
        issues,
        roadmap,
        "aggregate balance/overlap counts plus cryptographic commitments",
        "roadmap sealed commitment boundary",
    )
    for needle in (
        "general-PPI transfer",
        "Ab-Ag-specific calibration",
        "no model training or new structure prediction",
    ):
        require_contains(issues, plan, needle, "source-backed C5 next gate")
    for needle in (
        "10.5281/zenodo.17978681",
        "CC-BY-4.0",
        "56259a84f1e8cc216e5ee91a96584f824ca46f062ef4f2c06aa4674472daf1c8",
        "132,000 absolute compute-path",
    ):
        require_contains(issues, c5_provenance, needle, "C5 provenance")

    expected_report_values = {
        "source.rows": c5_report["source"].get("rows") == 22_000,
        "source.targets": c5_report["source"].get("targets") == 110,
        "split.calibration": c5_report["split"].get("calibration_targets") == 55,
        "split.evaluation": c5_report["split"].get("evaluation_targets") == 55,
        "trust_all.failures": (
            c5_report["policies"]["trust_all"].get("failures_among_trusted")
            == 28
        ),
        "fixed_gate.failures": (
            c5_report["policies"]["generic_fixed_iptm_0_80"].get(
                "failures_among_trusted"
            )
            == 3
        ),
        "regime_gate.not_certified": (
            c5_report["decision"].get("regime_specific_trust_certified")
            is False
        ),
        "pilot.passed": (
            c5_report["decision"].get("source_backed_pilot_passed") is True
        ),
    }
    for label, passed in expected_report_values.items():
        if not passed:
            issues.append(f"C5 source report invariant failed: {label}")

    if issues:
        print(f"FAIL research plan check found {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("OK research plan check passed")
    print("- component order: enum_action -> tool_query -> routing_after_loop")
    print("- Stage A checkpoint: tool_query 0/5; sealed routing 5/25; runtime oracle 25/25")
    print("- prospective Stage A: routing 35/180; hybrid 115/180; compiler 25/25 clean")
    print("- C5 prototype: fail-closed 12/12; trust-all 9 unsafe trusts")
    print("- C5 source replay: trust-all 28/55 failures; fixed gate 3/20")
    print("- C5 certification: no trusted set at alpha <= 0.30")
    print("- C5 next gate: independent calibration evidence")
    print("- DPO/RLVR/HF gate: useful routing coverage plus independent evaluation required")
    print("- sealed evaluation gate: completed rows cannot be tuned on or rescored")
    print("- C5 gate: calibration metadata required before trust")
    return 0


if __name__ == "__main__":
    sys.exit(main())
