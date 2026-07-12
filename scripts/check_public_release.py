#!/usr/bin/env python3
"""Validate the public-release surface before GitHub or Hugging Face promotion."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "release" / "public_release_manifest.json"
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".mmd",
    ".py",
    ".sbatch",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_TEXT_BYTES = 5_000_000


def _fragment(*parts: str) -> str:
    return "".join(parts)


SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("personal absolute path", re.compile(r"/Users/[A-Za-z0-9._-]+")),
    ("home-relative private path", re.compile(r"(?<![A-Za-z0-9_])~/(?:Dropbox|\\.config|\\.ssh|\\.cache)")),
    (
        "local key-file breadcrumb",
        re.compile(_fragment("anthropic", r"[_-]api[_-]key", r"\.txt"), re.IGNORECASE),
    ),
    ("private allocation id", re.compile(_fragment("NAIRR", r"\d+|crl\d+"), re.IGNORECASE)),
    (
        "private trust repo name",
        re.compile(_fragment("sfm", r"[-_]", "trust", r"[-_]", "private"), re.IGNORECASE),
    ),
    ("raw codex session path", re.compile(re.escape(".codex") + r"/sessions")),
    (
        "application-strategy folder",
        re.compile(_fragment("Application", r"[_/-]", "documents"), re.IGNORECASE),
    ),
    ("github token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}")),
    ("huggingface token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("openai-style token", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("private key marker", re.compile(r"BEGIN (?:OPENSSH|RSA|DSA|EC) PRIVATE KEY")),
    (
        "inline secret assignment",
        re.compile(r"(?i)(?:api|secret|token|password)[_-]?key\s*=\s*['\"][^'\"]{12,}['\"]"),
    ),
    (
        "career or application framing",
        re.compile(
            _fragment(
                r"(?:",
                "hiring", r"\s+", "reviewers",
                r"|(?:one-line\s+)?", "resume", r"\s+", "framing",
                r"|", "job", r"[-\s]+", "application", r"(?:\s+(?:packet|review))?",
                r"|", "portfolio", r"\s+(?:summary|one-pager|package|polish|framing)",
                r"|(?:best|local)\s+", "application", r"\s+(?:framing|materials?)",
                r"|", "application", r"\s+", "strategy", r"\s+(?:anchors?|files?)",
                r"|", "application", r"\s+", "plan",
                r"|", "life", r"\s+", "sciences", r"\s+", "rs", r"\s+", "role",
                r"|", "jd", r"[-\s]+", "alignment",
                r"|", "job-boards", r"\.", "greenhouse", r"\.io",
                r")",
            ),
            re.IGNORECASE,
        ),
    ),
    (
        "numeric scheduler JSON field",
        re.compile(
            _fragment(
                r"['\"](?:",
                "slurm", r"_)?", "job", r"_", "id",
                r"['\"]\s*:\s*['\"]?\d{6,12}\b",
            ),
            re.IGNORECASE,
        ),
    ),
    (
        "numeric scheduler job reference",
        re.compile(
            _fragment(
                r"\b(?:", "slurm", r"|", "cayuga", r")\b[^\n]{0,100}\b",
                "job", r"\b(?:\s+", "id", r")?\s*(?:[:#=]\s*)?`?\d{6,12}\b",
            ),
            re.IGNORECASE,
        ),
    ),
)

ALLOWED_INLINE_SECRET_MARKERS = {
    'api_key="EMPTY"',
    "api_key='EMPTY'",
}

PROHIBITED_PUBLIC_PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "internal release-planning document",
        re.compile(r"(^|/)(?:PORTFOLIO_SUMMARY|SESSION_RECOVERY_\d{4}-\d{2}-\d{2}|WORKTREE_TRIAGE_\d{4}-\d{2}-\d{2})\.md$"),
    ),
    ("ignored run directory", re.compile(r"(^|/)post_training/runs($|/)")),
    ("python bytecode/cache", re.compile(r"(^|/)(__pycache__|.*\.py[cod])($|/)")),
    ("pytest cache", re.compile(r"(^|/)\.pytest_cache($|/)")),
    ("virtual environment", re.compile(r"(^|/)(\.venv|venv)($|/)")),
    ("environment file", re.compile(r"(^|/)\.env(?:\.|$)")),
    ("local database", re.compile(r".*\.(?:db|sqlite|sqlite3)(?:-.+)?$")),
    ("private key file", re.compile(r".*\.(?:key|pem)$")),
    ("model checkpoint or weights", re.compile(r".*\.(?:bin|ckpt|pt|pth|safetensors)$")),
    ("log file", re.compile(r".*\.log$")),
)

PROHIBITED_PUBLIC_ARTIFACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignored run directory", re.compile(r"(^|/)post_training/runs($|/)")),
    (
        "raw saved predictions",
        re.compile(r"(^|[/_-])raw[/_-](?:saved[/_-])?predictions?($|[._/-])", re.IGNORECASE),
    ),
    (
        "raw model output",
        re.compile(r"(^|[/_-])raw[/_-](?:model[/_-])?(?:output|outputs|text)($|[._/-])", re.IGNORECASE),
    ),
    (
        "raw candidate scores",
        re.compile(r"(^|[/_-])raw[/_-]candidate[/_-]scores?($|[._/-])", re.IGNORECASE),
    ),
    (
        "candidate-score JSONL",
        re.compile(r"(^|[/_-])candidate[/_-]scores?(?:[/_-]jsonl|\.jsonl)($|[._/\s-])", re.IGNORECASE),
    ),
    (
        "scheduler or Slurm log",
        re.compile(r"(^|[/_-])(?:scheduler[/_-]logs?|slurm)($|[._/-])", re.IGNORECASE),
    ),
    (
        "model state or trainable state",
        re.compile(r"(^|[/_-])(?:model[/_-]state|trainable[/_-]state)($|[._/-])", re.IGNORECASE),
    ),
    ("model checkpoint or weights", re.compile(r".*\.(?:bin|ckpt|pt|pth|safetensors)($|\s)", re.IGNORECASE)),
)

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def repo_files() -> list[Path]:
    command = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode == 0:
        return sorted(ROOT / line for line in result.stdout.splitlines() if line.strip())
    return sorted(path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts)


def validate_public_paths(files: list[Path]) -> list[str]:
    issues: list[str] = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        for label, pattern in PROHIBITED_PUBLIC_PATH_PATTERNS:
            if pattern.match(rel):
                issues.append(f"{label}: tracked or unignored generated/private path: {rel}")
                break
    return issues


def read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ValueError(f"Missing manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    with MANIFEST_PATH.open() as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path.relative_to(ROOT)}:{line_no}: {exc}") from exc
            count += 1
    return count


def manifest_paths(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    entries.extend(manifest.get("public_surfaces", []))
    entries.extend(manifest.get("public_artifacts", []))
    return entries


def validate_manifest_path(rel: Any) -> list[str]:
    if not isinstance(rel, str) or not rel.strip():
        return [f"manifest entry path must be a non-empty string: {rel!r}"]
    if rel != rel.strip():
        return [f"manifest path has leading/trailing whitespace: {rel!r}"]
    if "\\" in rel:
        return [f"manifest path must use POSIX separators: {rel}"]
    parsed = PurePosixPath(rel)
    if parsed.is_absolute():
        return [f"manifest path must be repo-relative: {rel}"]
    if ".." in parsed.parts:
        return [f"manifest path must not contain parent traversal: {rel}"]
    return []


def validate_public_artifact_entry(entry: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    rel = entry.get("path")
    if not isinstance(rel, str):
        return issues

    kind = entry.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        issues.append(f"manifest public artifact missing non-empty kind: {rel}")
    if entry.get("safe_to_publish") is not True:
        issues.append(f"manifest public artifact is not marked safe_to_publish: {rel}")

    expected_sha = entry.get("sha256")
    if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
        issues.append(f"manifest public artifact missing valid sha256: {rel}")

    descriptor = f"{rel} {kind if isinstance(kind, str) else ''}"
    for label, pattern in PROHIBITED_PUBLIC_ARTIFACT_PATTERNS:
        if pattern.search(descriptor):
            issues.append(f"manifest public artifact uses prohibited {label}: {rel}")
            break
    return issues


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required_keys = {
        "manifest_version",
        "project",
        "release_profile",
        "status",
        "public_surfaces",
        "public_artifacts",
        "validation_commands",
        "do_not_publish",
    }
    for key in sorted(required_keys - set(manifest)):
        issues.append(f"manifest missing required key: {key}")

    seen_paths: set[str] = set()
    public_artifacts = manifest.get("public_artifacts", [])
    for entry in manifest_paths(manifest):
        rel = entry.get("path")
        path_issues = validate_manifest_path(rel)
        if path_issues:
            issues.extend(path_issues)
            continue
        assert isinstance(rel, str)
        if rel in seen_paths:
            issues.append(f"manifest duplicate path: {rel}")
        seen_paths.add(rel)

        if "kind" not in entry and "role" not in entry:
            issues.append(f"manifest entry missing kind/role: {rel}")
        elif "kind" in entry and (not isinstance(entry.get("kind"), str) or not entry.get("kind", "").strip()):
            issues.append(f"manifest entry has empty kind: {rel}")
        elif "role" in entry and (not isinstance(entry.get("role"), str) or not entry.get("role", "").strip()):
            issues.append(f"manifest entry has empty role: {rel}")

        if entry in public_artifacts:
            issues.extend(validate_public_artifact_entry(entry))

        path = ROOT / rel
        if entry.get("required", True) and not path.exists():
            issues.append(f"manifest path missing: {rel}")
            continue
        expected_records = entry.get("records")
        if expected_records is not None:
            try:
                actual_records = count_jsonl_records(path)
            except ValueError as exc:
                issues.append(str(exc))
            else:
                if actual_records != expected_records:
                    issues.append(
                        f"manifest record mismatch for {rel}: expected {expected_records}, got {actual_records}"
                    )
        expected_sha = entry.get("sha256")
        if expected_sha and path.exists():
            if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
                issues.append(f"manifest sha256 is not a 64-character lowercase hex digest for {rel}")
            else:
                actual_sha = sha256_file(path)
                if actual_sha != expected_sha:
                    issues.append(f"manifest sha256 mismatch for {rel}: expected {expected_sha}, got {actual_sha}")

    return issues


def validate_public_demo(path: Path) -> list[str]:
    issues: list[str] = []
    labels: set[str] = set()
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            label = record.get("label")
            if not label:
                issues.append(f"{path.relative_to(ROOT)}:{line_no} missing label")
            elif label in labels:
                issues.append(f"{path.relative_to(ROOT)}:{line_no} duplicate label: {label}")
            labels.add(label)
            text = json.dumps(record, sort_keys=True)
            if "Toy" not in text:
                issues.append(f"{path.relative_to(ROOT)}:{line_no} does not look synthetic/demo-scoped")
            source_ids = tuple(record.get("task", {}).get("gold_source_ids", ())) + tuple(
                record.get("trajectory", {}).get("cited_source_ids", ())
            )
            for source_id in source_ids:
                if not str(source_id).startswith("DEMO-SOURCE-"):
                    issues.append(
                        f"{path.relative_to(ROOT)}:{line_no} has non-demo source id: {source_id}"
                    )
            for blocked in ("sqlite", "database_path", "trial_id", "nct_id", "pmid"):
                if blocked in text.lower():
                    issues.append(f"{path.relative_to(ROOT)}:{line_no} contains non-demo field marker: {blocked}")
    if len(labels) < 2:
        issues.append(f"{path.relative_to(ROOT)} should contain multiple demo cases")
    return issues


def should_scan_text(path: Path) -> bool:
    if path.suffix not in TEXT_SUFFIXES:
        return False
    try:
        return path.stat().st_size <= MAX_TEXT_BYTES
    except OSError:
        return False


def scan_scheduler_table_ids(path: Path, text: str) -> list[str]:
    """Reject numeric scheduler IDs stored in Markdown table columns."""

    issues: list[str] = []
    job_column: int | None = None
    for line_no, line in enumerate(text.splitlines(), start=1):
        if path.suffix != ".md" or "|" not in line:
            job_column = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if job_column is None:
            for index, cell in enumerate(cells):
                normalized = cell.lower().replace("_", " ")
                if ("slurm" in normalized or "scheduler" in normalized) and "job" in normalized:
                    job_column = index
                    break
            continue
        if job_column >= len(cells):
            continue
        value = cells[job_column].strip("`")
        if re.fullmatch(r"\d{6,12}", value):
            issues.append(
                f"numeric scheduler table value: {path.relative_to(ROOT)}:{line_no} contains {value!r}"
            )
    return issues


def scan_sensitive_strings(files: list[Path]) -> tuple[list[str], int]:
    issues: list[str] = []
    scanned = 0
    for path in files:
        if not should_scan_text(path):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError as exc:
            issues.append(f"could not read {path.relative_to(ROOT)}: {exc}")
            continue
        scanned += 1
        for label, pattern in SENSITIVE_PATTERNS:
            for match in pattern.finditer(text):
                snippet = match.group(0)
                if snippet in ALLOWED_INLINE_SECRET_MARKERS:
                    continue
                issues.append(f"{label}: {path.relative_to(ROOT)} contains {snippet!r}")
        issues.extend(scan_scheduler_table_ids(path, text))
    return issues, scanned


def main() -> int:
    issues: list[str] = []
    try:
        manifest = read_manifest()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL public release check: {exc}")
        return 1

    issues.extend(validate_manifest(manifest))
    files = repo_files()
    issues.extend(validate_public_paths(files))

    demo_path = ROOT / "demo" / "public_trajectory_cases.jsonl"
    if demo_path.exists():
        issues.extend(validate_public_demo(demo_path))
    else:
        issues.append("missing public demo JSONL: demo/public_trajectory_cases.jsonl")

    scan_issues, scanned_count = scan_sensitive_strings(files)
    issues.extend(scan_issues)

    demo_records_checked = count_jsonl_records(demo_path) if demo_path.exists() else 0

    if issues:
        print(f"FAIL public release check found {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("OK public release check passed")
    print(f"- manifest entries checked: {len(manifest_paths(manifest))}")
    print(f"- text files scanned: {scanned_count}")
    print(f"- demo records checked: {demo_records_checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
