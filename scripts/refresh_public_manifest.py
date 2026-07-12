#!/usr/bin/env python3
"""Refresh checksums and JSONL record counts in the public-release manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from check_public_release import MANIFEST_PATH, ROOT, count_jsonl_records, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        required=True,
        help="Explicit manifest update date in YYYY-MM-DD form.",
    )
    return parser.parse_args()


def validated_date(value: str) -> str:
    parsed = dt.date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("--date must use canonical YYYY-MM-DD form")
    return value


def main() -> int:
    args = parse_args()
    update_date = validated_date(args.date)
    manifest = json.loads(MANIFEST_PATH.read_text())
    refreshed = 0
    for entry in manifest.get("public_artifacts", []):
        rel = entry.get("path")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"invalid public artifact path: {rel!r}")
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        entry["sha256"] = sha256_file(path)
        if path.suffix == ".jsonl":
            entry["records"] = count_jsonl_records(path)
        refreshed += 1
    manifest["last_updated"] = update_date
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"refreshed {refreshed} public artifact entries for {update_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
