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
        r"Do not start DPO, RLVR, v0\.1 tagging, or Hugging Face publication until at least\s+one component cluster result",
        "release and method gate",
    )
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
        "complete_tool_query_diagnostic_then_run_one_time_sealed_evaluation",
        "public STATUS sealed-extension decision",
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

    if issues:
        print(f"FAIL research plan check found {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("OK research plan check passed")
    print("- component order: enum_action -> tool_query -> routing_after_loop")
    print("- DPO/RLVR gate: held-out component reports required")
    print("- sealed evaluation gate: exposed held-out becomes dev; private labels stay sealed")
    print("- C5 gate: calibration metadata required before trust")
    return 0


if __name__ == "__main__":
    sys.exit(main())
