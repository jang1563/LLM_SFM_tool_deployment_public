import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_research_plan_checker_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_research_plan.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK research plan check passed" in result.stdout
    assert "enum_action -> tool_query -> routing_after_loop" in result.stdout
    assert "sealed evaluation gate" in result.stdout
