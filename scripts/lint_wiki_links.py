#!/usr/bin/env python3
"""Verify [[wikilink]] targets exist under knowledge/wiki/."""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "knowledge" / "wiki"

WIKILINK_RE = re.compile(r"\[\[([^\]|#\n]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def is_wiki_slug(target: str) -> bool:
    """True for concept-style slugs; skip code paths misusing [[...]]."""
    if "`" in target or "/" in target or "*" in target:
        return False
    if any(target.endswith(ext) for ext in (".py", ".md", ".tex", ".yaml", ".txt")):
        return False
    if target.startswith("status:"):
        return False
    return bool(SLUG_RE.match(target))


def wikilinks_in_line(line: str) -> list[str]:
    parts = re.split(r"(`[^`\n]+`)", line)
    targets = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            continue
        for m in WIKILINK_RE.finditer(part):
            target = m.group(1).strip()
            if is_wiki_slug(target):
                targets.append(target)
    return targets


def find_wikilinks(path: Path) -> list[tuple[int, str]]:
    links = []
    in_fence = False
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for target in wikilinks_in_line(line):
            links.append((i, target))
    return links


def collect_slugs(wiki_root: Path) -> set[str]:
    return {p.stem for p in wiki_root.rglob("*.md")}


def inbound_link_counts(wiki_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {slug: 0 for slug in collect_slugs(wiki_root)}
    for md in wiki_root.rglob("*.md"):
        for _, target in find_wikilinks(md):
            if target in counts:
                counts[target] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint wiki [[wikilink]] targets")
    parser.add_argument("--orphans", action="store_true", help="Report pages with no inbound links")
    args = parser.parse_args()

    slugs = collect_slugs(WIKI)
    broken: list[str] = []

    for md in sorted(WIKI.rglob("*.md")):
        for line_no, target in find_wikilinks(md):
            if target not in slugs:
                rel = md.relative_to(ROOT)
                broken.append(f"{rel}:{line_no} → [[{target}]]")

    if broken:
        print("WIKI LINK LINT FAILED")
        for item in broken:
            print(f"  - {item}")
        return 1

    print(f"WIKI LINKS OK ({len(slugs)} pages)")

    if args.orphans:
        counts = inbound_link_counts(WIKI)
        orphans = sorted(slug for slug, n in counts.items() if n == 0)
        if orphans:
            print(f"\nOrphan pages ({len(orphans)}):")
            for slug in orphans:
                print(f"  - {slug}")
        else:
            print("\nNo orphan pages.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
