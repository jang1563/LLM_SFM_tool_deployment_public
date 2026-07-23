#!/usr/bin/env python3
"""Validate the active research plan against drift-critical checkpoints."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research" / "2026-06-25_posttrain_tool_use_landscape" / "LONG_TERM_RESEARCH_PLAN_2026-07-04.md"
ROADMAP = ROOT / "ROADMAP.md"
README = ROOT / "README.md"
PUBLIC_STATUS = ROOT / "STATUS.md"


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
        "one-time source-separated sealed routing result remains 5/25",
        "frozen routing is 35/180",
        "runtime hybrid routing is 115/180",
        "base and frozen placeholder-SFT tool-query policies are both 0/25 exact",
        "150/150 malformed inputs rejected",
    ):
        require_contains(issues, plan, needle, "completed Stage A result")
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
        "stage_b_c5_manifest_prototype_after_stage_a_runtime_split",
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
        "frozen routing is 35/180",
        "runtime hybrid routing is 115/180",
        "25/25 clean exact",
        "150/150 malformed inputs rejected",
        "Build the first Stage B C5",
    ):
        require_contains(issues, plan, needle, "prospective Stage A checkpoint")

    if issues:
        print(f"FAIL research plan check found {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("OK research plan check passed")
    print("- component order: enum_action -> tool_query -> routing_after_loop")
    print("- Stage A checkpoint: tool_query 0/5; sealed routing 5/25; runtime oracle 25/25")
    print("- prospective Stage A: routing 35/180; hybrid 115/180; compiler 25/25 clean")
    print("- DPO/RLVR/HF gate: useful routing coverage plus independent evaluation required")
    print("- sealed evaluation gate: completed rows cannot be tuned on or rescored")
    print("- C5 gate: calibration metadata required before trust")
    return 0


if __name__ == "__main__":
    sys.exit(main())
