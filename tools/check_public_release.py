#!/usr/bin/env python3
"""Lightweight public-release scan of git-tracked files. Standard library only; no network.

Reports ``path:line  category`` and NEVER the matched text, so running it (or its CI log) cannot
leak a secret. Exit status 1 if anything is found.

Categories: forbidden-path, api-key, bearer-token, credential-assignment, private-key,
internal-hostname, private-ip, user-home-path, tracked-result-data.

Usage: check_public_release.py [--root DIR] [--history]   (DIR may be a plain directory)

Tune false positives by editing ALLOW_PATHS / the patterns below, deliberately and in review.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hostnames that must never appear in public files. Built from fragments so this file does not
# itself contain them.
_BANNED_HOSTS = [r"bme" + r"\.hu", r"tm" + r"it\.", r"xr" + r"lab"]

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api-key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{40,}")),
    ("api-key", re.compile(r"\b(?:ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{30,}")),
    ("api-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer-token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]{24,}")),
    (
        "credential-assignment",
        re.compile(
            r"""(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b["']?\s*[:=]\s*["'][A-Za-z0-9/+_.-]{24,}["']"""
        ),
    ),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("internal-hostname", re.compile("|".join(_BANNED_HOSTS), re.IGNORECASE)),
    (
        "private-ip",
        re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
    ),
    (
        "user-home-path",
        re.compile(
            r"(?:/home|/Users)/(?!user\b|u\b|runner\b|example\b|you\b|<)[A-Za-z0-9_.-]+|C:\\Users\\"
        ),
    ),
]

# Files that legitimately mention these patterns (the scanner and its tests).
ALLOW_PATHS = {"tools/check_public_release.py", "tests/test_public_release_scan.py"}
SKIP_SUFFIXES = {".lock", ".png", ".jpg", ".gif", ".ico", ".whl", ".gz", ".zip"}
SKIP_NAMES = {"uv.lock", "package-lock.json"}
# Raw run data and experiment outputs must never be tracked.
FORBIDDEN_TRACKED_PREFIXES = ("results/raw/", "results/experiments/")


def tracked_files(root: Path = ROOT) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True
    ).stdout
    return [p for p in out.decode().split("\0") if p]


# A "secret" whose own text says it is fake (test fixtures) is not a finding. Only secret-like
# categories use this; hostnames, paths and key blocks are always findings.
FAKE_MARKERS = (
    "secret",
    "fake",
    "dummy",
    "example",
    "placeholder",
    "not-a-real",
    "must-not",
    "test",
)
FAKE_TOLERANT = {"api-key", "bearer-token", "credential-assignment"}


def scan_text(path: str, text: str) -> list[tuple[str, int, str]]:
    findings = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for category, pattern in PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            if category in FAKE_TOLERANT and any(f in m.group(0).lower() for f in FAKE_MARKERS):
                continue
            findings.append((path, lineno, category))
    return findings


# Path components that must never exist in a public snapshot (checked for plain directories).
FORBIDDEN_COMPONENTS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".env"}


def plain_files(root: Path) -> list[str]:
    """Every file under ``root`` (hidden ones included), for a snapshot that is not a git repo."""
    return sorted(
        str(p.relative_to(root)).replace("\\", "/")
        for p in root.rglob("*")
        if p.is_file() or p.is_symlink()
    )


def scan_history(root: Path = ROOT) -> list[tuple[str, int, str]]:
    """Scan every blob reachable from any ref, once each. Reports the path it was seen at."""
    listing = subprocess.run(
        ["git", "-C", str(root), "rev-list", "--objects", "--all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    findings: set[tuple[str, int, str]] = set()
    for line in listing:
        sha, _, path = line.partition(" ")
        if not path or path in ALLOW_PATHS or Path(path).name in SKIP_NAMES:
            continue
        if Path(path).suffix in SKIP_SUFFIXES:
            continue
        kind = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-t", sha], capture_output=True, text=True
        ).stdout.strip()
        if kind != "blob":
            continue
        raw = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", sha], capture_output=True
        ).stdout
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        findings.update(scan_text(path, text))
    return sorted(findings)


def scan(root: Path = ROOT, files: list[str] | None = None) -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []
    for rel in files if files is not None else tracked_files(root):
        if rel in ALLOW_PATHS:
            continue
        if FORBIDDEN_COMPONENTS & set(Path(rel).parts):
            findings.append((rel, 0, "forbidden-path"))
            continue
        if rel.startswith(FORBIDDEN_TRACKED_PREFIXES):
            findings.append((rel, 0, "tracked-result-data"))
            continue
        path = root / rel
        if path.suffix in SKIP_SUFFIXES or path.name in SKIP_NAMES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: not text we can judge
        findings.extend(scan_text(rel, text))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="repository or plain directory")
    parser.add_argument("--history", action="store_true", help="also scan all reachable git blobs")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    is_git = (root / ".git").exists()
    findings = scan(root, None if is_git else plain_files(root))
    if args.history:
        if not is_git:
            print("--history needs a git repository", file=sys.stderr)
            return 2
        findings += scan_history(root)
    for path, lineno, category in findings:
        print(f"{path}:{lineno}  {category}")
    if findings:
        print(
            f"\n{len(findings)} finding(s). Values are intentionally not printed.", file=sys.stderr
        )
        return 1
    print("public-release scan: no findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
