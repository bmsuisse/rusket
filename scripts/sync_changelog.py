#!/usr/bin/env python3
"""sync_changelog.py — Regenerate docs/changelog.md from git-cliff.

Run from the repository root:
    uv run python scripts/sync_changelog.py

The script calls `git cliff` (must be installed / on PATH) and writes its
output to `docs/changelog.md`, replacing whatever was there before.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS_CHANGELOG = ROOT / "docs" / "changelog.md"

HEADER = """\
# Changelog

All notable changes are documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

"""


def run_git_cliff() -> str:
    """Run git-cliff and return its stdout."""
    cmd = ["git", "cliff", "--output", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT)
        return result.stdout
    except FileNotFoundError:
        print("ERROR: git-cliff not found. Install it with: cargo install git-cliff", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git-cliff failed:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    print("Running git-cliff …")
    cliff_output = run_git_cliff()

    # git-cliff may already include a header, strip it and use our own
    # so MkDocs always sees a consistent top-level `# Changelog` heading.
    lines = cliff_output.splitlines(keepends=True)
    # Drop leading blank lines and the auto-generated header line if present
    trimmed_lines: list[str] = []
    skip_cliff_header = True
    for line in lines:
        if skip_cliff_header:
            stripped = line.strip()
            # Skip blank lines and the "# Changelog" heading git-cliff may emit
            if stripped == "" or stripped.lower() == "# changelog":
                continue
            skip_cliff_header = False
        trimmed_lines.append(line)

    body = "".join(trimmed_lines)
    content = HEADER + body

    DOCS_CHANGELOG.write_text(content, encoding="utf-8")
    print(f"✔ Wrote {DOCS_CHANGELOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
