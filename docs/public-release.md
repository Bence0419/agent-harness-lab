# Public release notes and audit

This document records how the repository was prepared for public release.

## Redaction of institutional endpoint information

The original research used an OpenAI-compatible model endpoint hosted by an institution. Three
tracked research configuration files contained its real host name:

| File | Contained |
|---|---|
| `configs/opencode-bme-qwen-v1.json` | the endpoint URL (`baseURL`) |
| `configs/pi-bme-qwen-v1-models.json` | the endpoint URL (`baseUrl`) |
| `configs/swebench-bme-only-squid-v1.conf` | the host in the Squid allowlist (`dstdomain`) |

**Status: applied (maintainer-approved).** The public repository contains a hostname-redacted
representation of the historical provider configuration. It replaces that host name with the placeholder `llm.example.org`, preserving the
file structure and the methodology. File names, provider ids (`bme-lab`) and environment variable
names are unchanged. The Bash runners and the parity tests read these files; with the placeholder
they describe the same structure but would reach a fictional host. The real values are not part
of the repository; the study's author retains the original private copies. No research result,
trace or evaluation was changed.

The same host name is also present in earlier git history. Redacting the working tree does not
remove it from history; see the maintainer checklist.

## What is scanned

`tools/check_public_release.py` (run by CI and by `tests/test_public_release_scan.py`) scans
git-tracked files for API-key shapes, bearer tokens, credential assignments, private-key blocks,
banned internal host names, private IPv4 ranges, user home paths and tracked result data. It
prints `path:line category` only, never the matched value. Fake test fixtures are recognized by
their own text.

## Maintainer checklist before pushing

1. License: Apache-2.0 (`LICENSE`), declared in `pyproject.toml`.
2. Git history: the research repository's history contains the original host name and old home
   paths and is **not** published. The public repository is a fresh one-commit snapshot of the
   tracked files (see below); the research history stays private provenance.
3. Run `uv run python tools/check_public_release.py` and the test suite.
4. Build with `uv build` and smoke-test the wheel in a clean environment.
5. Only then tag and publish.

## Creating the public snapshot

```bash
mkdir /path/to/agent-harness-lab-public && git archive HEAD | tar -x -C /path/to/agent-harness-lab-public
cd /path/to/agent-harness-lab-public && git init -b main && git add -A && git commit -m "Initial public release"
```

`git archive` includes only tracked files (no `.git`, ignored results, `node_modules` or virtualenvs).
