# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** to the maintainers (use the repository's
private security-advisory feature once it is public, or contact the maintainers directly). Do not
open a public issue with exploit details.

**Never paste API keys, tokens or credentials into issues, pull requests or logs.** If you
believe a key has leaked, revoke it with the provider immediately, then tell us which file or
commit exposed it (without the value).

## What AHL does, and what it does not protect

AHL runs coding agents. A coding agent executes **untrusted, model-generated commands**. AHL
runs them inside a Docker container created from the task image, and applies these mitigations:

* **Restricted-provider networking** (default): the container is on an internal Docker network
  and can reach only the model provider's host, through a pinned Squid proxy.
* **Secrets by name only:** API keys are forwarded with `docker create -e NAME`, so values never
  enter command lines, plans, run records or logs; the evaluator process receives no
  credentials.
* **No evaluator material in the agent's container:** the task directory (gold patch, tests) is
  never mounted or copied.
* **Read-only mounts:** the harness runtime is mounted read-only.
* **Cleanup:** containers, networks and proxies created for a run are removed, and only those.

These **reduce** risk. They do **not** make arbitrary code intrinsically safe, and the isolation
is not perfect:

* Docker is the security boundary. A container escape, an unsafe Docker daemon configuration or
  a privileged setup defeats every mitigation above.
* The host paths AHL mounts (the host Node.js runtime and the harness installation) are readable
  by the agent. Review `ahl run --preflight` output, which lists the exact mounts.
* The model provider receives the task text and the agent's tool output.
* `unrestricted` networking removes the egress limit; it is for experiments only.
* The official SWE-bench evaluator builds and runs task code in Docker and rebuilds images.

**Do not run AHL on a machine holding sensitive data or credentials you are not prepared to
expose to untrusted code.** Prefer a dedicated VM or a disposable WSL2 distribution, a
least-privilege provider key with a spending limit, and a Docker daemon you control.

## Execution gate

Real execution is disabled unless `AHL_ENABLE_REAL_EXECUTION=1` is set. This is a deliberate
pre-release safety behavior against accidental spend and execution.
