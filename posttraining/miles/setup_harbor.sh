#!/usr/bin/env bash
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PTX_ROOT

# harbor-miles-v0.20.0 carries miles_agent_server.py; pinned to the commit every run here used.
HARBOR_COMMIT=ac63e51
if [ ! -d "$PTX_ROOT/harbor/.git" ]; then
  git clone -q --branch harbor-miles-v0.20.0 https://github.com/harbor-framework/harbor.git "$PTX_ROOT/harbor"
fi
git -C "$PTX_ROOT/harbor" checkout -q "$HARBOR_COMMIT"
(cd "$PTX_ROOT/harbor" && uv sync -q --extra modal)
python3 "$HERE/patch_harbor.py" "$PTX_ROOT/harbor"
"$PTX_ROOT/harbor/.venv/bin/python" -c "import harbor, modal; print('harbor ok')"
