from pathlib import Path
import subprocess
import sys

from scripts.check_public_release import (
    SENSITIVE_PATTERNS,
    scan_scheduler_table_ids,
    validate_manifest,
    validate_public_paths,
)


ROOT = Path(__file__).resolve().parents[1]


def manifest_with(
    public_surfaces: list[dict] | None = None,
    public_artifacts: list[dict] | None = None,
) -> dict:
    return {
        "manifest_version": 1,
        "project": "unit",
        "release_profile": "unit",
        "status": "unit",
        "public_surfaces": public_surfaces if public_surfaces is not None else [],
        "public_artifacts": public_artifacts if public_artifacts is not None else [],
        "validation_commands": [],
        "do_not_publish": [],
    }


def test_public_release_checker_passes_current_repo_surface() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_public_release.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK public release check passed" in result.stdout


def test_public_release_checker_catches_common_release_secrets() -> None:
    patterns = {label: pattern for label, pattern in SENSITIVE_PATTERNS}
    fake_hf_token = "hf_" + "abcdefghijklmnopqrstuvwx"
    fake_private_key_marker = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"

    assert patterns["huggingface token"].search(fake_hf_token)
    assert patterns["private key marker"].search(fake_private_key_marker)


def test_public_release_checker_rejects_career_framing_and_scheduler_ids() -> None:
    patterns = {label: pattern for label, pattern in SENSITIVE_PATTERNS}
    career_samples = [
        "hiring " + "reviewers",
        "one-line " + "resume " + "framing",
        "job-" + "application review",
        "portfolio " + "summary",
        "Life Sciences " + "RS role",
    ]
    scheduler_field = '"' + "slurm_" + 'job_id": "' + "1234567" + '"'
    scheduler_prose = "Cayuga Slurm GPU " + "job: `" + "1234567" + "`"

    for sample in career_samples:
        assert patterns["career or application framing"].search(sample)
    assert patterns["numeric scheduler JSON field"].search(scheduler_field)
    assert patterns["numeric scheduler job reference"].search(scheduler_prose)


def test_public_release_checker_rejects_scheduler_id_table_cells() -> None:
    table = (
        "| Target | Slurm " + "job | Score |\n"
        "| --- | ---: | ---: |\n"
        "| full | " + "1234567" + " | 0.5 |\n"
    )

    issues = scan_scheduler_table_ids(ROOT / "example.md", table)

    assert len(issues) == 1
    assert "numeric scheduler table value" in issues[0]


def test_public_release_checker_allows_scheduler_placeholders_and_scientific_ids() -> None:
    patterns = {label: pattern for label, pattern in SENSITIVE_PATTERNS}
    allowed = [
        '"job_id": "omitted"',
        'RUN_ID="run_${JOB_ID}"',
        "MONDO:0019225",
        "PMID 14913280",
    ]

    for sample in allowed:
        assert not patterns["numeric scheduler JSON field"].search(sample)
        assert not patterns["numeric scheduler job reference"].search(sample)


def test_public_release_checker_rejects_generated_or_private_paths() -> None:
    bad_paths = [
        ROOT / "__pycache__" / "module.cpython-313.pyc",
        ROOT / ".pytest_cache" / "README.md",
        ROOT / ".env",
        ROOT / "private.sqlite3",
        ROOT / "run.log",
        ROOT / "secret.pem",
        ROOT / "post_training" / "runs" / "unit" / "report.json",
        ROOT / "post_training" / "runs" / "unit" / "trainable_state.pt",
        ROOT / "post_training" / "model.safetensors",
        ROOT / "PORTFOLIO_SUMMARY.md",
        ROOT / "SESSION_RECOVERY_2026-06-25.md",
        ROOT / "WORKTREE_TRIAGE_2026-06-25.md",
    ]

    issues = validate_public_paths(bad_paths)

    assert len(issues) == len(bad_paths)


def test_public_release_checker_rejects_manifest_path_escape_and_duplicates() -> None:
    manifest = manifest_with(
        public_surfaces=[
            {"path": "README.md", "role": "entrypoint", "required": False},
            {"path": "README.md", "role": "duplicate", "required": False},
        ],
        public_artifacts=[
            {
                "path": "../private/raw_predictions.jsonl",
                "kind": "raw_predictions",
                "required": False,
                "safe_to_publish": True,
                "sha256": "0" * 64,
            },
            {
                "path": "/tmp/raw_predictions.jsonl",
                "kind": "raw_predictions",
                "required": False,
                "safe_to_publish": True,
                "sha256": "0" * 64,
            },
        ],
    )

    issues = validate_manifest(manifest)

    assert "manifest duplicate path: README.md" in issues
    assert "manifest path must not contain parent traversal: ../private/raw_predictions.jsonl" in issues
    assert "manifest path must be repo-relative: /tmp/raw_predictions.jsonl" in issues


def test_public_release_checker_rejects_raw_artifacts_in_public_manifest() -> None:
    manifest = manifest_with(
        public_artifacts=[
            {
                "path": "post_training/runs/unit/report.json",
                "kind": "compact_report",
                "required": False,
                "safe_to_publish": True,
                "sha256": "0" * 64,
            },
            {
                "path": "post_training/raw_predictions.jsonl",
                "kind": "raw_predictions",
                "required": False,
                "safe_to_publish": True,
                "sha256": "0" * 64,
            },
            {
                "path": "post_training/candidate_scores.jsonl",
                "kind": "candidate_score_jsonl",
                "required": False,
                "safe_to_publish": True,
                "sha256": "0" * 64,
            },
            {
                "path": "post_training/model_state.pt",
                "kind": "model_state",
                "required": False,
                "safe_to_publish": True,
                "sha256": "0" * 64,
            },
        ],
    )

    issues = validate_manifest(manifest)

    assert "manifest public artifact uses prohibited ignored run directory: post_training/runs/unit/report.json" in issues
    assert "manifest public artifact uses prohibited raw saved predictions: post_training/raw_predictions.jsonl" in issues
    assert "manifest public artifact uses prohibited candidate-score JSONL: post_training/candidate_scores.jsonl" in issues
    assert "manifest public artifact uses prohibited model state or trainable state: post_training/model_state.pt" in issues


def test_public_release_checker_requires_artifact_kind_sha_and_safe_flag() -> None:
    manifest = manifest_with(
        public_artifacts=[
            {
                "path": "demo/public_trajectory_cases.jsonl",
                "kind": "",
                "required": False,
                "safe_to_publish": "true",
                "sha256": "not-a-sha",
            }
        ],
    )

    issues = validate_manifest(manifest)

    assert "manifest entry has empty kind: demo/public_trajectory_cases.jsonl" in issues
    assert "manifest public artifact missing non-empty kind: demo/public_trajectory_cases.jsonl" in issues
    assert "manifest public artifact is not marked safe_to_publish: demo/public_trajectory_cases.jsonl" in issues
    assert "manifest public artifact missing valid sha256: demo/public_trajectory_cases.jsonl" in issues
