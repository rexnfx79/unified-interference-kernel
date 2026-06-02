#!/usr/bin/env python3
"""Verify knowledge/manifest.yaml SHA256 hashes against canonical repo files."""

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "knowledge" / "manifest.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_snapshots(text: str) -> list[tuple[str, str]]:
    """Parse canonical + sha256 pairs from manifest snapshots block."""
    entries = []
    current_canonical = None
    in_snapshots = False
    for line in text.splitlines():
        if line.strip() == "snapshots:":
            in_snapshots = True
            continue
        if in_snapshots and line.startswith("external_sources:"):
            break
        if not in_snapshots:
            continue
        m = re.match(r"\s+canonical:\s+(\S+)", line)
        if m:
            current_canonical = m.group(1)
        m = re.match(r"\s+sha256:\s+([a-f0-9]+)", line)
        if m and current_canonical:
            entries.append((current_canonical, m.group(1)))
            current_canonical = None
    return entries


def main() -> int:
    text = MANIFEST.read_text()
    entries = parse_snapshots(text)
    if not entries:
        print("No snapshots found in manifest")
        return 1
    errors = []
    for canonical, expected in entries:
        path = ROOT / canonical
        if not path.exists():
            errors.append(f"Missing canonical: {canonical}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(
                f"Hash mismatch {canonical}: expected {expected[:12]}… got {actual[:12]}…"
            )
    if errors:
        print("MANIFEST LINT FAILED")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"MANIFEST OK ({len(entries)} snapshots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
