#!/usr/bin/env python3
"""Check whether the existing Git history is safe to make public.

This is stricter than scripts/check_public_release.py. The release checker
validates the current public surface; this script answers a different question:
can this exact private repository, with its existing refs and history, be made
public without exposing old local paths or private breadcrumbs?
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_public_release import ALLOWED_INLINE_SECRET_MARKERS, SENSITIVE_PATTERNS


def fragment(*parts: str) -> str:
    return "".join(parts)


HISTORY_GREP = "|".join(
    (
        r"/Users/[A-Za-z0-9._-]+",
        r"~/(Dropbox|\.config|\.ssh|\.cache)",
        fragment(r"[Aa]nthropic", r"[_-][Aa][Pp][Ii][_-][Kk]ey", r"\.txt"),
        r"NAIRR[0-9]+|crl[0-9]+",
        fragment("sfm", r"[-_]", "trust", r"[-_]", "private"),
        fragment(r"\.codex", r"/sessions"),
        fragment("Application", r"[_/-]", "documents"),
        r"gh[pousr]_[A-Za-z0-9_]{20,}",
        r"hf_[A-Za-z0-9]{20,}",
        r"sk-[A-Za-z0-9_-]{20,}",
        r"BEGIN (OPENSSH|RSA|DSA|EC) PRIVATE KEY",
        r"(api|secret|token|password)[_-]?key[[:space:]]*=[[:space:]]*['\"][^'\"]{12,}['\"]",
    )
)


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)


def history_lines(max_lines: int) -> tuple[list[str], bool]:
    revs = run_git(["rev-list", "--all"])
    if revs.returncode != 0:
        raise RuntimeError(revs.stderr.strip() or "git rev-list failed")

    rev_list = [line.strip() for line in revs.stdout.splitlines() if line.strip()]
    if not rev_list:
        return [], False

    pattern = HISTORY_GREP
    process = subprocess.Popen(
        ["git", "grep", "-I", "-n", "-E", pattern, *rev_list, "--", "."],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    lines: list[str] = []
    truncated = False
    try:
        for line in process.stdout:
            lines.append(line.rstrip("\n"))
            if len(lines) >= max_lines:
                truncated = True
                process.terminate()
                break
        _, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        _, stderr = process.communicate()
    if process.returncode not in (0, 1, -15, 143):
        raise RuntimeError(stderr.strip() or "git grep failed")
    return lines, truncated


def classify(line: str) -> tuple[str, str]:
    for label, pattern in SENSITIVE_PATTERNS:
        match = pattern.search(line)
        if match and match.group(0) not in ALLOWED_INLINE_SECRET_MARKERS:
            return label, match.group(0)
    return "sensitive pattern", line[:120]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-issues", type=int, default=50)
    args = parser.parse_args()

    try:
        matches, truncated = history_lines(args.max_issues + 1)
    except RuntimeError as exc:
        print(f"FAIL public history check: {exc}")
        return 1

    issues = []
    for line in matches:
        label, snippet = classify(line)
        issues.append((label, snippet, line))

    if issues:
        count_label = f"{len(issues)}+" if truncated else str(len(issues))
        print(f"FAIL public history check found {count_label} issue(s).")
        print("This repository should not be made public in place; use a clean public mirror or rewritten history.")
        for label, snippet, line in issues[: args.max_issues]:
            location = ":".join(line.split(":")[:4])
            print(f"- {label}: {location} contains {snippet!r}")
        if len(issues) > args.max_issues:
            print(f"- ... {len(issues) - args.max_issues} more issue(s) omitted")
        return 1

    print("OK public history check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
