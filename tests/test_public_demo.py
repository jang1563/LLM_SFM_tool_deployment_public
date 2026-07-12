from pathlib import Path

from examples.run_public_demo import _task_from_json, _trajectory_from_json, load_cases
from llm_sfm_tool_deployment import TrajectoryEvaluator


ROOT = Path(__file__).resolve().parents[1]


def test_public_demo_cases_exercise_pass_and_fail_paths() -> None:
    cases = load_cases(ROOT / "demo" / "public_trajectory_cases.jsonl")
    results = [
        TrajectoryEvaluator().evaluate(
            _task_from_json(case["task"]),
            _trajectory_from_json(case["trajectory"]),
        )
        for case in cases
    ]

    assert len(results) == 5
    assert any(result.passed for result in results)
    assert any(not result.passed for result in results)
    assert any("external_tool_required" in result.violations for result in results)
    assert any(
        "uncalibrated_specialist_requires_verify_baseline_or_defer" in result.violations
        for result in results
    )
