"""The public-release scan: it catches obvious secrets/internal values, avoids false positives
on fake test fixtures, and never prints a matched value."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "check_public_release.py"
spec = importlib.util.spec_from_file_location("check_public_release", SCRIPT)
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)

REAL_LOOKING_KEY = "sk-" + "Ab3dE9fG1hJ2kL4mN6pQ8rS0tU2vW4xY6zA8bC0d"  # 40+ chars, no fake marker


def cats(text, path="x.txt"):
    return {c for _, _, c in scanner.scan_text(path, text)}


def test_detects_api_keys_bearer_tokens_and_private_keys():
    assert "api-key" in cats(f"key = {REAL_LOOKING_KEY}")
    assert "api-key" in cats("token ghp_" + "A" * 36)
    assert "bearer-token" in cats("Authorization: Bearer " + "Zx9" * 12)
    assert "private-key" in cats("-----BEGIN " + "RSA PRIVATE KEY-----")
    assert "credential-assignment" in cats('password = "' + "Qw8Er7Ty6Ui5Op4As3Df2Gh1Jk" + '"')


def test_detects_banned_hosts_private_ips_and_home_paths():
    assert "internal-hostname" in cats("url: https://llm." + "bme" + ".hu/v1")
    assert "private-ip" in cats("host 192.168.1.20")
    assert "private-ip" in cats("host 10.1.2.3")
    assert "user-home-path" in cats("/home/alice/projects")
    assert "user-home-path" in cats("C:\\Users\\bob")


def test_does_not_flag_fake_fixtures_or_placeholders():
    assert not cats("SECRET = 'sk-experiment-secret-value-123456'")
    assert not cats("api_key = 'placeholder-not-a-real-key-value'")
    assert not cats("OPENAI_API_KEY=sk-...")
    assert not cats("export OPENAI_API_KEY=<your key>")
    assert not cats("base_url: https://llm.example.org/v1")
    assert not cats("/home/user/project and /home/<USER>/x and $HOME/.nvm")
    assert not cats("8.8.8.8 and 1.1.1.1")


def test_findings_never_contain_the_matched_value(capsys, tmp_path):
    (tmp_path / "leak.txt").write_text(f"key = {REAL_LOOKING_KEY}\nhttps://x." + "bme" + ".hu/\n")
    findings = scanner.scan(tmp_path, ["leak.txt"])
    assert {c for _, _, c in findings} == {"api-key", "internal-hostname"}
    for path, lineno, category in findings:
        assert REAL_LOOKING_KEY not in f"{path}:{lineno} {category}"


def test_tracked_result_data_is_flagged():
    found = scanner.scan(ROOT, ["results/raw/swebench/x/run.json", "results/experiments/e/m.json"])
    assert {c for _, _, c in found} == {"tracked-result-data"}


def test_scan_finds_a_secret_in_a_tracked_file_of_a_repo(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.txt").write_text(f"{REAL_LOOKING_KEY}\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    assert scanner.scan(repo)  # findings in a throwaway repo


def test_raw_results_and_experiments_are_gitignored():
    text = (ROOT / ".gitignore").read_text()
    assert "results/raw/" in text and "results/experiments/" in text


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs git")
def test_repository_has_no_public_release_findings():
    findings = scanner.scan(ROOT)
    assert findings == [], [(p, ln, c) for p, ln, c in findings]


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        check=True,
        capture_output=True,
    )


def test_plain_directory_scan_finds_leaks_and_forbidden_paths(tmp_path):
    (tmp_path / "ok.txt").write_text("nothing here\n")
    assert scanner.main(["--root", str(tmp_path)]) == 0
    (tmp_path / "leak.md").write_text("see /home/alice/x\n")
    assert scanner.main(["--root", str(tmp_path)]) == 1
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "m.js").write_text("x\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("x\n")
    found = scanner.scan(tmp_path, scanner.plain_files(tmp_path))
    assert {c for _, _, c in found} == {"user-home-path", "forbidden-path"}


def test_history_scan_finds_a_value_removed_from_the_tree(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "a.txt").write_text("host https://x." + "bme" + ".hu/\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "one")
    (repo / "a.txt").write_text("clean\n")
    _git(repo, "commit", "-qam", "two")
    assert scanner.scan(repo) == []  # the current tree is clean ...
    assert {c for _, _, c in scanner.scan_history(repo)} == {
        "internal-hostname"
    }  # ... history is not
    assert scanner.main(["--root", str(repo), "--history"]) == 1
    assert scanner.main(["--root", str(repo)]) == 0
