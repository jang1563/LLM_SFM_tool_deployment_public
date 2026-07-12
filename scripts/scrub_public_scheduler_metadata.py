#!/usr/bin/env python3
"""Remove concrete scheduler identifiers from curated Markdown and JSON."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_KEYS = {"job_id", "slurm_job_id"}
CONTEXTUAL_JOB_ID = re.compile(
    r"(?i)(\b(?:slurm\s+job|cayuga(?:\s+slurm(?:\s+gpu)?)?\s+job|job)"
    r"\s*(?::|\|)?\s*`?)\d{6,12}(`?)"
)
NUMERIC_ID = re.compile(r"^`?\d{6,12}`?$")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / rel for rel in result.stdout.splitlines() if rel and (ROOT / rel).is_file()]


def strip_scheduler_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_scheduler_keys(item)
            for key, item in value.items()
            if key.lower() not in SCHEDULER_KEYS
        }
    if isinstance(value, list):
        return [strip_scheduler_keys(item) for item in value]
    return value


def scrub_json(path: Path) -> bool:
    original = json.loads(path.read_text())
    scrubbed = strip_scheduler_keys(original)
    if scrubbed == original:
        return False
    path.write_text(json.dumps(scrubbed, indent=2) + "\n")
    return True


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def scrub_markdown_tables(lines: list[str]) -> list[str]:
    job_column: int | None = None
    output: list[str] = []
    for line in lines:
        if "|" not in line:
            job_column = None
            output.append(line)
            continue
        cells = split_table_row(line)
        if job_column is None:
            for index, cell in enumerate(cells):
                normalized = cell.lower().replace("_", " ")
                if ("slurm" in normalized or "scheduler" in normalized) and "job" in normalized:
                    job_column = index
                    break
            output.append(line)
            continue
        if job_column < len(cells) and NUMERIC_ID.fullmatch(cells[job_column]):
            cells[job_column] = "[omitted]"
            output.append("| " + " | ".join(cells) + " |\n")
        else:
            output.append(line)
    return output


def scrub_markdown(path: Path) -> bool:
    original = path.read_text()
    scrubbed = CONTEXTUAL_JOB_ID.sub(r"\1[omitted]\2", original)
    scrubbed = "".join(scrub_markdown_tables(scrubbed.splitlines(keepends=True)))
    if scrubbed == original:
        return False
    path.write_text(scrubbed)
    return True


def main() -> int:
    changed: list[str] = []
    for path in tracked_files():
        if path.suffix == ".json" and scrub_json(path):
            changed.append(path.relative_to(ROOT).as_posix())
        elif path.suffix == ".md" and scrub_markdown(path):
            changed.append(path.relative_to(ROOT).as_posix())
    print(f"scrubbed scheduler metadata from {len(changed)} file(s)")
    for rel in changed:
        print(f"- {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
