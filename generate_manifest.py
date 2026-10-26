#!/usr/bin/env python3
"""Generate manifest.json listing all markdown content files.

Scans ai-daily-insight/content/ and cloud-daily-insight/content/
for .md files and writes a JSON manifest at the repo root.
The frontend (index.html) reads this instead of calling the GitHub
Contents API, avoiding rate-limit issues.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SECTIONS = {
    "ai": REPO_ROOT / "ai-daily-insight" / "content",
    "cloud": REPO_ROOT / "cloud-daily-insight" / "content",
}


def collect_md_files(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(
        (f.name for f in directory.iterdir() if f.is_file() and f.suffix == ".md"),
        reverse=True,
    )


def main() -> None:
    manifest = {
        bucket: collect_md_files(path)
        for bucket, path in SECTIONS.items()
    }
    out = REPO_ROOT / "manifest.json"
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total = sum(len(v) for v in manifest.values())
    print(
        f"manifest.json written: {total} files "
        f"({len(manifest['ai'])} ai, {len(manifest['cloud'])} cloud)"
    )


if __name__ == "__main__":
    main()
