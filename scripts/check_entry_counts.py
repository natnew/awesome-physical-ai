#!/usr/bin/env python3
"""
check_entry_counts.py — verify that every category in README.md
is within the 15–25 entry target band.

Usage:
    python scripts/check_entry_counts.py

Exit codes:
    0 — all categories within [15, 25]
    1 — one or more categories outside the band
"""

import re
import sys
from pathlib import Path

TARGET_LOW = 15
TARGET_HIGH = 25

KNOWN_CATEGORIES = {
    "Simulators",
    "Datasets",
    "Benchmarks",
    "Evaluation Methodology",
    "Robotics Foundation Models",
    "World Models",
    "Manipulation",
    "Locomotion",
    "Sim-to-Real",
    "Safety & Robustness",
    "Governance & Policy",
    "Production Patterns / Reference Architectures",
    "Courses",
    "Companies",
}

# Headings that are structural (not content categories)
SKIP_HEADINGS = {
    "Contents",
    "Contributing",
    "License",
    "Awesome Physical AI",
}


def parse_categories(readme_path: Path) -> dict[str, int]:
    """Return {category_name: entry_count} for all known categories."""
    text = readme_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    counts: dict[str, int] = {}
    current_category: str | None = None
    current_count = 0

    heading_re = re.compile(r"^#{1,3}\s+(.+)$")
    bullet_re = re.compile(r"^\s*[-*]\s+\[")  # lines like `- [Name](url)`

    for line in lines:
        m = heading_re.match(line)
        if m:
            # Save previous category
            if current_category and current_category in KNOWN_CATEGORIES:
                counts[current_category] = current_count
            heading_text = m.group(1).strip()
            if heading_text in KNOWN_CATEGORIES:
                current_category = heading_text
                current_count = 0
            else:
                current_category = None
                current_count = 0
        elif current_category and bullet_re.match(line):
            current_count += 1

    # Flush last category
    if current_category and current_category in KNOWN_CATEGORIES:
        counts[current_category] = current_count

    return counts


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    readme = repo_root / "README.md"

    if not readme.exists():
        print(f"ERROR: README.md not found at {readme}")
        return 1

    counts = parse_categories(readme)

    # Check for missing categories
    missing = KNOWN_CATEGORIES - counts.keys()
    if missing:
        print(f"WARNING: categories not found in README: {', '.join(sorted(missing))}")

    col_w = 32
    print(f"\n{'Category':<{col_w}} {'Count':>5}  Status")
    print("-" * (col_w + 15))

    any_fail = False
    for category in sorted(KNOWN_CATEGORIES):
        count = counts.get(category, 0)
        if count < TARGET_LOW:
            status = "LOW ⚠"
            any_fail = True
        elif count > TARGET_HIGH:
            status = "HIGH ⚠"
            any_fail = True
        else:
            status = "OK"
        print(f"{category:<{col_w}} {count:>5}  {status}")

    print()
    if any_fail:
        print(f"FAIL — one or more categories outside [{TARGET_LOW}, {TARGET_HIGH}].")
        return 1

    print(f"PASS — all categories within [{TARGET_LOW}, {TARGET_HIGH}].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
