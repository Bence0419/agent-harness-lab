import pytest

from agent_harness_lab.doctor import Status, run_checks
from agent_harness_lab.doctor.checks import (
    check_docker_cli,
    check_docker_daemon,
    check_openai_key,
    check_opencode,
    check_pi,
    check_python,
    check_wsl,
    parse_version,
)


def by_name(report):
    return {r.name: r for r in report.results}


def test_parse_version():
    assert parse_version("git version 2.43.0") == "2.43.0"
    assert parse_version("v24.19.0") == "24.19.0"
    assert parse_version("opencode v2.2.0") == "2.2.0"
    assert parse_version("Docker version 27.1.1, build abc") == "27.1.1"
    assert parse_version("no digits here") is None
    assert parse_version("") is None


def test_healthy_environment_has_no_errors_or_warnings(healthy_probe):
    report = run_checks(healthy_probe)
    assert not report.has_errors
    assert report.count(Status.WARNING) == 0


def test_missing_optional_executable_is_warning(healthy_probe):
    del healthy_probe.executables["pi"]
    result = check_pi(healthy_probe)
    assert result.status is Status.WARNING
    assert "not installed" in result.detail
    assert not run_checks(healthy_probe).has_errors


def test_missing_optional_tools_do_not_fail_report(healthy_probe):
    for name in ("pi", "opencode", "node", "timeout"):
        del healthy_probe.executables[name]
    report = run_checks(healthy_probe)
    assert not report.has_errors
    assert report.count(Status.WARNING) == 4


def test_missing_api_key_is_warning(healthy_probe):
    healthy_probe.env.clear()
    result = check_openai_key(healthy_probe)
    assert result.status is Status.WARNING
    assert not run_checks(healthy_probe).has_errors


def test_empty_api_key_counts_as_not_set(healthy_probe):
    healthy_probe.env["OPENAI_API_KEY"] = ""
    assert check_openai_key(healthy_probe).status is Status.WARNING


def test_api_key_value_never_appears_in_results(healthy_probe):
    report = run_checks(healthy_probe)
    assert "sk-super-secret-value" not in report.model_dump_json()
    assert by_name(report)["OPENAI_API_KEY"].status is Status.OK


def test_unsupported_python_is_error(healthy_probe):
    healthy_probe.py = (3, 11, 9)
    result = check_python(healthy_probe)
    assert result.status is Status.ERROR
    assert run_checks(healthy_probe).has_errors


def test_docker_daemon_unreachable_is_error(healthy_probe):
    healthy_probe.commands[("/fake/docker", "info", "--format", "{{.ServerVersion}}")] = (
        1,
        "Cannot connect to the Docker daemon",
    )
    assert check_docker_daemon(healthy_probe).status is Status.ERROR
    assert run_checks(healthy_probe).has_errors


def test_docker_daemon_skipped_when_cli_missing(healthy_probe):
    del healthy_probe.executables["docker"]
    report = by_name(run_checks(healthy_probe))
    assert report["Docker CLI"].status is Status.ERROR
    assert report["Docker daemon"].status is Status.WARNING


def test_wsl_detection(healthy_probe):
    assert "not running" in check_wsl(healthy_probe).detail
    healthy_probe.env["WSL_DISTRO_NAME"] = "Ubuntu"
    healthy_probe.os_release = "6.18.33.2-microsoft-standard-WSL2"
    assert "WSL2" in check_wsl(healthy_probe).detail


def test_checks_only_use_injected_probe(healthy_probe):
    run_checks(healthy_probe)
    assert healthy_probe.calls  # probe was used; no real subprocess needed


def test_required_executable_that_fails_to_run_is_error(healthy_probe):
    healthy_probe.commands[("/fake/docker", "--version")] = (1, "shim: integration disabled")
    assert check_docker_cli(healthy_probe).status is Status.ERROR


def test_optional_executable_that_fails_to_run_is_warning(healthy_probe):
    healthy_probe.commands[("/fake/pi", "--version")] = (127, "")
    assert check_pi(healthy_probe).status is Status.WARNING
    assert not run_checks(healthy_probe).has_errors


# --- harness discovery (Pi / OpenCode) -------------------------------------

ROOT = "/fake/project"
PACKAGE_JSON = (
    '{"dependencies": {"@earendil-works/pi-coding-agent": "1.0.2",'
    ' "@opencode/cli": "2.2.0"}}'
)


def project_probe(healthy_probe):
    """Healthy probe with a project root, pins, and no harness on PATH."""
    del healthy_probe.executables["pi"]
    del healthy_probe.executables["opencode"]
    healthy_probe.root = ROOT
    healthy_probe.files[f"{ROOT}/package.json"] = PACKAGE_JSON
    return healthy_probe


def add_local(probe, name, output):
    path = f"{ROOT}/node_modules/.bin/{name}"
    probe.executable_files.add(path)
    probe.commands[(path, "--version")] = (0, output)
    return path


def test_pi_found_on_path(healthy_probe):
    result = check_pi(healthy_probe)
    assert result.status is Status.OK
    assert result.detail == "1.0.2 (PATH)"


def test_pi_found_project_local(healthy_probe):
    probe = project_probe(healthy_probe)
    add_local(probe, "pi", "1.0.2")
    result = check_pi(probe)
    assert result.status is Status.OK
    assert result.detail == "1.0.2 (project-local)"


def test_opencode_found_project_local(healthy_probe):
    probe = project_probe(healthy_probe)
    add_local(probe, "opencode", "opencode v2.2.0")
    result = check_opencode(probe)
    assert result.status is Status.OK
    assert result.detail == "2.2.0 (project-local)"


def test_path_takes_precedence_over_project_local(healthy_probe):
    probe = project_probe(healthy_probe)
    probe.executables["pi"] = "/fake/global/pi"
    probe.commands[("/fake/global/pi", "--version")] = (0, "1.0.2")
    local = add_local(probe, "pi", "9.9.9")
    result = check_pi(probe)
    assert result.detail == "1.0.2 (PATH)"
    assert [local, "--version"] not in probe.calls


def test_project_local_exists_but_version_fails(healthy_probe):
    probe = project_probe(healthy_probe)
    path = add_local(probe, "pi", "")
    probe.commands[(path, "--version")] = (1, "boom")
    result = check_pi(probe)
    assert result.status is Status.WARNING
    assert path in result.detail
    assert "project-local" in result.detail
    assert "failed" in result.detail


def test_project_local_not_executable_is_ignored(healthy_probe):
    probe = project_probe(healthy_probe)
    path = add_local(probe, "pi", "1.0.2")
    probe.executable_files.discard(path)
    assert check_pi(probe).status is Status.WARNING


def test_version_mismatch_is_warning_not_error(healthy_probe):
    probe = project_probe(healthy_probe)
    add_local(probe, "pi", "1.0.3")
    result = check_pi(probe)
    assert result.status is Status.WARNING
    assert "1.0.3" in result.detail and "1.0.2" in result.detail
    assert not run_checks(probe).has_errors


def test_path_version_mismatch_is_warning(healthy_probe):
    probe = project_probe(healthy_probe)
    probe.executables["opencode"] = "/fake/global/opencode"
    probe.commands[("/fake/global/opencode", "--version")] = (0, "opencode v2.1.0")
    result = check_opencode(probe)
    assert result.status is Status.WARNING
    assert "(PATH)" in result.detail and "2.2.0" in result.detail


def test_no_pin_available_reports_ok_with_version(healthy_probe):
    probe = project_probe(healthy_probe)
    del probe.files[f"{ROOT}/package.json"]
    add_local(probe, "pi", "7.7.7")
    assert check_pi(probe).status is Status.OK


def test_neither_installation_found_is_warning(healthy_probe):
    probe = project_probe(healthy_probe)
    for check in (check_pi, check_opencode):
        result = check(probe)
        assert result.status is Status.WARNING
        assert "not installed" in result.detail
    assert not run_checks(probe).has_errors


def test_no_project_root_still_uses_path_only(healthy_probe):
    healthy_probe.root = None
    del healthy_probe.executables["pi"]
    assert check_pi(healthy_probe).status is Status.WARNING


def test_harness_checks_run_only_through_probe(healthy_probe):
    probe = project_probe(healthy_probe)
    add_local(probe, "pi", "1.0.2")
    add_local(probe, "opencode", "opencode v2.2.0")
    run_checks(probe)
    # every executed argv was served by the fake; nothing real was needed
    assert [f"{ROOT}/node_modules/.bin/pi", "--version"] in probe.calls


# --- SWE-bench tasks repository (optional capability) ----------------------
def test_tasks_repo_configured_reports_commit(healthy_probe):
    from agent_harness_lab.doctor.checks import check_swebench_tasks_repo

    result = check_swebench_tasks_repo(healthy_probe)
    assert result.status is Status.OK
    assert "6e08cbcd763c869abd7aae4bcb9e1b3991180777" in result.detail


def test_tasks_repo_not_configured_is_warning_not_error(healthy_probe):
    from agent_harness_lab.doctor.checks import check_swebench_tasks_repo

    del healthy_probe.env["AHL_SWEBENCH_TASKS_REPO"]
    result = check_swebench_tasks_repo(healthy_probe)
    assert result.status is Status.WARNING and "not configured" in result.detail
    assert not run_checks(healthy_probe).has_errors


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda p: p.dirs.clear(), "missing directory"),
        (lambda p: p.dirs.discard("/fake/tasks-repo/tasks"), "does not look like"),
        (lambda p: p.heads.clear(), "git commit unreadable"),
    ],
)
def test_misconfigured_tasks_repo_is_warning_never_error(healthy_probe, mutate, fragment):
    from agent_harness_lab.doctor.checks import check_swebench_tasks_repo

    mutate(healthy_probe)
    result = check_swebench_tasks_repo(healthy_probe)
    assert result.status is Status.WARNING and fragment in result.detail
    assert not run_checks(healthy_probe).has_errors
