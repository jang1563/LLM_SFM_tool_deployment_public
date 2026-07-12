from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_stage_a_manifest_eval_script_reports_oracle_and_bad_baselines() -> None:
    result = subprocess.run(
        [sys.executable, "examples/run_stage_a_manifest_eval.py", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["oracle"]["passed"] == 25
    assert summary["self_answer"]["passed"] == 0
    assert summary["wrong_tool"]["passed"] == 0
    assert summary["partial_query"]["passed"] == 0
