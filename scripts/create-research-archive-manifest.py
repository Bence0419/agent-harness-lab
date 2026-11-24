#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for every file under results/raw/.

Read-only with respect to results/raw: nothing is copied or modified.
Output format (sorted by path, one file per line, sha256sum compatible):

    <sha256>  <path relative to results/raw>

Usage: create-research-archive-manifest.py <output-path>
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <output-path>", file=sys.stderr)
        return 64

    if not RAW.is_dir():
        print(f"Error: raw results directory not found: {RAW}", file=sys.stderr)
        return 1

    output = Path(argv[1]).resolve()
    if output.is_relative_to(RAW):
        print("Error: output must not be written inside results/raw", file=sys.stderr)
        return 1

    rel_paths = sorted(
        p.relative_to(RAW).as_posix()
        for p in RAW.rglob("*")
        if p.is_file() and not p.is_symlink()
    )

    lines = [f"{sha256_of(RAW / rel)}  {rel}\n" for rel in rel_paths]
    try:
        output.write_text("".join(lines), encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"Error: cannot write {output}: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(lines)} entries to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
