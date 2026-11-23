"""Add a `modal` sandbox backend to Harbor's miles agent server (harbor-miles-v0.20.0). Idempotent.

    python patch_harbor.py $PTX_ROOT/harbor

With HARBOR_ENV_TYPE=modal, trials run in Modal sandboxes (harbor.environments.modal) and, like
daytona, skip the local-docker paths: the Docker Hub login guard, the build-cache prune loop and
compose-project teardown. HARBOR_MODAL_TIMEOUT_SECS (hard sandbox lifetime) and
HARBOR_MODAL_IDLE_TIMEOUT_SECS bound what a sandbox orphaned by a crash can bill.
"""

import sys
from pathlib import Path

HARBOR = Path(sys.argv[1])

MODAL_CONFIG = """    if os.getenv("HARBOR_ENV_TYPE", "docker").lower() == "modal":
        return EnvironmentConfig(
            type="modal",
            delete=True,
            override_memory_mb=override_memory_mb,
            kwargs={
                "app_name": os.getenv("HARBOR_MODAL_APP", "ptx-harbor"),
                "sandbox_timeout_secs": int(os.getenv("HARBOR_MODAL_TIMEOUT_SECS", "3600")),
                "sandbox_idle_timeout_secs": int(os.getenv("HARBOR_MODAL_IDLE_TIMEOUT_SECS", "900")),
            },
        )
    return EnvironmentConfig(
        type="docker","""

PATCHES = [
    (
        "agent_server/trial_runner.py",
        """    return EnvironmentConfig(
        type="docker",""",
        MODAL_CONFIG,
    ),
    (
        "agent_server/trial_runner.py",
        'if os.getenv("HARBOR_ENV_TYPE", "docker").lower() != "daytona":\n            compose_project',
        'if os.getenv("HARBOR_ENV_TYPE", "docker").lower() not in ("daytona", "modal"):\n            compose_project',
    ),
    (
        "miles_agent_server.py",
        '_daytona = os.getenv("HARBOR_ENV_TYPE", "docker").lower() == "daytona"',
        '_daytona = os.getenv("HARBOR_ENV_TYPE", "docker").lower() in ("daytona", "modal")',
    ),
    (
        "miles_agent_server.py",
        'if os.getenv("HARBOR_ENV_TYPE", "docker").lower() != "daytona":\n        _check_docker_login()',
        'if os.getenv("HARBOR_ENV_TYPE", "docker").lower() not in ("daytona", "modal"):\n        _check_docker_login()',
    ),
]

for rel, old, new in PATCHES:
    path = HARBOR / rel
    text = path.read_text()
    if new in text:
        continue
    assert text.count(old) == 1, f"patch target not unique/found in {path}: {old!r}"
    path.write_text(text.replace(old, new))
    print(f"patched {path}")
