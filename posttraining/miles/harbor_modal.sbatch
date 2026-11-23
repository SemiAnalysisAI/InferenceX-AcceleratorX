#!/bin/bash
#SBATCH -J ptx-harbor-modal
#SBATCH -N1 --exclusive --gres=gpu:0
#SBATCH -t 7-00:00:00
#SBATCH --no-requeue

# terminus-2 runs inside this server process and drives the Modal sandbox remotely, so model calls go
# from here to the miles session server over the cluster network; nothing on the cluster is exposed.
set -eo pipefail
REPO_ROOT=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
source "$REPO_ROOT/inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PTX_ROOT PORT MAX_CONCURRENT AGENT_TIMEOUT HARBOR_MODAL_APP \
    HARBOR_MODAL_TIMEOUT_SECS HARBOR_MODAL_IDLE_TIMEOUT_SECS
[ -f "$HOME/.modal.toml" ] || check_env_vars MODAL_TOKEN_ID MODAL_TOKEN_SECRET

python3 "$REPO_ROOT/posttraining/miles/patch_harbor.py" "$PTX_ROOT/harbor"
NODE_IP=$(scontrol show node "$(hostname)" | grep -oP 'NodeAddr=\K\S+')
mkdir -p "$PTX_ROOT/logs" /tmp/ptx-harbor/trials
echo "$(hostname) $NODE_IP:$PORT" | tee "$PTX_ROOT/logs/harbor_endpoint.txt"

cd "$PTX_ROOT/harbor"
exec env HARBOR_ENV_TYPE=modal \
    HARBOR_MODAL_APP="$HARBOR_MODAL_APP" \
    HARBOR_MODAL_TIMEOUT_SECS="$HARBOR_MODAL_TIMEOUT_SECS" \
    HARBOR_MODAL_IDLE_TIMEOUT_SECS="$HARBOR_MODAL_IDLE_TIMEOUT_SECS" \
    HARBOR_TASKS_DIR="$PTX_ROOT/harbor_tasks" \
    DASHBOARD_LOG_PATH=/tmp/ptx-harbor/requests.jsonl \
    "$PTX_ROOT/harbor/.venv/bin/python" miles_agent_server.py \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dashboard-port 0 \
    --max-concurrent "$MAX_CONCURRENT" \
    --agent-timeout "$AGENT_TIMEOUT" \
    --trials-dir /tmp/ptx-harbor/trials
