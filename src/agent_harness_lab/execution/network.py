"""Network policies, reproducing the frozen runners.

* ``unrestricted``: the container uses Docker's default bridge (the earliest Pi baseline runs).
* ``restricted-provider``: an ``--internal`` Docker network holding the agent container and a
  pinned Squid proxy (alias ``squid``). The proxy is also attached to ``bridge`` so it alone
  has egress, and its policy allows only the provider's host.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent_harness_lab.execution.docker import DockerClient
from agent_harness_lab.execution.models import (
    ContainerSpec,
    DockerError,
    NetworkRecord,
    Provision,
    StageError,
)
from agent_harness_lab.runplan.models import NetworkMode, NetworkPlan

# Pinned image used by every frozen restricted-network run (digest, not a tag).
SQUID_IMAGE = "ubuntu/squid@sha256:6a097f68bae708cedbabd6188d68c7e2e7a38cedd05a176e1cc0ba29e3bbe029"
PROXY_ALIAS = "squid"
PROXY_URL = f"http://{PROXY_ALIAS}:3128"
PROXY_SETTLE_SECONDS = 2.0  # the OpenCode runners slept 2 s after starting the proxy
SQUID_CONF_DESTINATION = "/etc/squid/squid.conf"


def render_squid_policy(allowed_hosts: list[str]) -> str:
    """Deterministic Squid policy: allow only ``allowed_hosts`` (same rules as the frozen
    conf files; only the ACL name is generic)."""
    if not allowed_hosts:
        raise ValueError("a restricted policy needs at least one allowed host")
    hosts = " ".join(sorted(set(allowed_hosts)))
    return (
        "http_port 3128\n"
        "\n"
        "acl SSL_ports port 443\n"
        "acl Safe_ports port 80\n"
        "acl Safe_ports port 443\n"
        "acl CONNECT method CONNECT\n"
        "\n"
        f"acl provider_hosts dstdomain {hosts}\n"
        "\n"
        "http_access deny !Safe_ports\n"
        "http_access deny CONNECT !SSL_ports\n"
        "\n"
        "http_access allow provider_hosts\n"
        "http_access deny all\n"
        "\n"
        "cache deny all\n"
    )


def proxy_env() -> dict[str, str]:
    """Container environment pointing at the proxy (as in the frozen runners)."""
    no_proxy = "localhost,127.0.0.1"
    return {
        "HTTP_PROXY": PROXY_URL, "HTTPS_PROXY": PROXY_URL,
        "http_proxy": PROXY_URL, "https_proxy": PROXY_URL,
        "NO_PROXY": no_proxy, "no_proxy": no_proxy,
    }  # fmt: skip


@dataclass
class NetworkResources:
    """Planned network, plus flags recording what has actually been created (for cleanup)."""

    record: NetworkRecord
    container_env: dict[str, str]
    policy: str | None = None  # rendered Squid policy (restricted mode only)
    network_created: bool = False
    proxy_created: bool = False

    @property
    def env(self) -> dict[str, str]:
        return self.container_env

    @property
    def network_name(self) -> str | None:
        return self.record.network_name if self.network_created else None

    @property
    def proxy_name(self) -> str | None:
        return self.record.proxy_name if self.proxy_created else None


def plan_network(plan: NetworkPlan, *, names: tuple[str, str]) -> NetworkResources:
    """Pure: derive the network setup from the plan (no Docker calls, no files)."""
    if plan.mode is NetworkMode.UNRESTRICTED:
        return NetworkResources(record=NetworkRecord(mode=plan.mode.value), container_env={})
    network, proxy = names
    policy = render_squid_policy(plan.allowed_hosts)
    return NetworkResources(
        record=NetworkRecord(
            mode=plan.mode.value,
            network_name=network,
            proxy_name=proxy,
            squid_image=SQUID_IMAGE,
            squid_policy_sha256=hashlib.sha256(policy.encode()).hexdigest(),
            allowed_hosts=sorted(set(plan.allowed_hosts)),
        ),
        container_env=proxy_env(),
        policy=policy,
    )


def start_network(
    docker: DockerClient,
    resources: NetworkResources,
    *,
    labels: dict[str, str],
    staging_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Create the internal network + pinned proxy. Mutates ``resources`` as things exist."""
    if resources.policy is None:  # unrestricted: ordinary Docker networking
        return
    record = resources.record
    assert record.network_name and record.proxy_name
    conf = staging_dir / "squid.conf"
    conf.write_text(resources.policy, encoding="utf-8")
    try:
        docker.image_inspect(SQUID_IMAGE)  # pinned image must already exist (no auto-pull)
        docker.network_create(record.network_name, internal=True, labels=labels)
        resources.network_created = True
        docker.run_proxy(
            name=record.proxy_name,
            network=record.network_name,
            alias=PROXY_ALIAS,
            image=SQUID_IMAGE,
            mounts=[
                Provision(
                    kind="bind-mount",
                    source=str(conf.resolve()),
                    destination=SQUID_CONF_DESTINATION,
                    read_only=True,
                    purpose="rendered Squid allowlist policy",
                    sha256=record.squid_policy_sha256,
                )
            ],
            labels=labels,
        )
        resources.proxy_created = True
        docker.network_connect("bridge", record.proxy_name)  # the proxy alone gets egress
        sleep(PROXY_SETTLE_SECONDS)
    except DockerError as exc:
        raise StageError("network", str(exc)) from exc


def agent_container_spec(
    *, name, image, resources: NetworkResources, env, pass_env, mounts, labels
) -> ContainerSpec:
    return ContainerSpec(
        name=name,
        image=image,
        network=resources.record.network_name if resources.policy else None,
        env={**resources.env, **env},
        pass_env=list(pass_env),
        mounts=list(mounts),
        labels=labels,
    )
