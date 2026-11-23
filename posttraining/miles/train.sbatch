#!/bin/bash
#SBATCH -J ptx-miles-swe
#SBATCH --ntasks-per-node=1 --gres=gpu:4 --exclusive
#SBATCH -t 2-00:00:00
#SBATCH --no-requeue

set -eo pipefail
REPO_ROOT=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
source "$REPO_ROOT/inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PTX_ROOT MILES_IMAGE CONTAINER_MOUNTS RUN_ID
CODE_DIR=$REPO_ROOT/posttraining/miles
RUN_DIR=$PTX_ROOT/runs/$RUN_ID
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=$(scontrol show node "$HEAD_NODE" | grep -oP 'NodeAddr=\K\S+')

# AGENT_SERVER_URL is optional: harbor_modal.sbatch writes its endpoint, and it may have only just started
for _ in $(seq 60); do
    url=$AGENT_SERVER_URL
    [ -n "$url" ] || url=http://$(awk '{print $2}' "$PTX_ROOT/logs/harbor_endpoint.txt")
    curl -sf -m 5 "$url/health" >/dev/null && break
    sleep 5
done
AGENT_SERVER_URL=$url
curl -sf -m 5 "$AGENT_SERVER_URL/health" >/dev/null || { echo "harbor not healthy at $AGENT_SERVER_URL"; exit 1; }
set -- --agent-server-url "$AGENT_SERVER_URL" --session-server-external-host "$HEAD_IP" "$@"

mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR/done"  # a reused RUN_ID would release the workers at once
ARGS=${*//\\/\\\\}; ARGS=${ARGS//\"/\\\"}
cat > "$RUN_DIR/run.json" <<JSON
{"run_id": "$RUN_ID", "slurm_job_id": "$SLURM_JOB_ID", "nodes": "$SLURM_JOB_NODELIST",
 "num_nodes": $SLURM_JOB_NUM_NODES, "gpus_per_node": 4, "image": "$MILES_IMAGE",
 "source_sha": "$(git -C "$REPO_ROOT" rev-parse HEAD)",
 "agent_server_url": "$AGENT_SERVER_URL", "head_ip": "$HEAD_IP",
 "start_ts": $(date +%s), "launcher_args": "$ARGS"}
JSON
echo "run $RUN_ID -> $RUN_DIR; head $HEAD_NODE $HEAD_IP; harbor $AGENT_SERVER_URL"

# node -> cluster IP, resolved outside the container (hostname -i can return docker0's 172.17.0.1)
NODE_IPS=$(for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    echo -n "$n=$(scontrol show node "$n" | grep -oP 'NodeAddr=\K\S+') "; done)
export RUN_ID RUN_DIR CODE_DIR AGENT_SERVER_URL HEAD_IP NODE_IPS LAUNCHER_ARGS="$*"
srun --kill-on-bad-exit=1 \
    --container-image="$MILES_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS,/tmp:/tmp" \
    --no-container-remap-root \
    bash -c '
set -eo pipefail
NODE_IP=$(tr " " "\n" <<< "$NODE_IPS" | grep "^$SLURMD_NODENAME=" | cut -d= -f2)
export PYTHONUNBUFFERED=1 MILES_HOST_IP=$NODE_IP MASTER_ADDR=$HEAD_IP MILES_SCRIPT_EXTERNAL_RAY=1
# node-local JIT caches: many processes sharing one NFS triton cache hit "Stale file handle"
export TRITON_CACHE_DIR=/tmp/ptx-cache-$USER/triton TORCHINDUCTOR_CACHE_DIR=/tmp/ptx-cache-$USER/inductor
export TILELANG_CACHE_DIR=/tmp/ptx-cache-$USER/tilelang RAY_TMPDIR=/tmp/ray-$USER
# Ray kills workers at 95% of host RAM; 122B trainers hold ~140 GB each on the host after a checkpoint
# save, which made Ray kill its own liveness probe. Let the cgroup limit be the only limit.
export RAY_memory_monitor_refresh_ms=0
python3 $CODE_DIR/patch_miles.py /root/miles
python3 $CODE_DIR/patch_fla.py
python3 $CODE_DIR/patch_fa4.py
cp $CODE_DIR/models/*.py /root/miles/scripts/models/  # model args miles does not ship (122B)
if [ "$SLURM_PROCID" = 0 ]; then
    ray start --head --node-ip-address $HEAD_IP --port 6379 --num-gpus 4 --disable-usage-stats
    until [ "$(ray list nodes --filter state=ALIVE 2>/dev/null | grep -c ALIVE)" -ge "$SLURM_JOB_NUM_NODES" ]; do sleep 5; done
    echo "ray cluster up: $SLURM_JOB_NUM_NODES nodes"
    cd /root/miles
    set +e
    python3 $CODE_DIR/run_qwen35_swe.py \
        --num-nodes $SLURM_JOB_NUM_NODES \
        --run-dir $RUN_DIR \
        --output-dir $RUN_DIR/output \
        $LAUNCHER_ARGS
    rc=$?
    echo "launcher exited rc=$rc"
    python3 -c "import json,time;p=\"$RUN_DIR/run.json\";d=json.load(open(p));d.update(end_ts=int(time.time()),exit_code=$rc);json.dump(d,open(p,\"w\"),indent=1)"
    touch $RUN_DIR/done
    ray stop --force
    exit $rc
else
    # no scancel inside the container: workers leave when rank 0 writes the sentinel
    sleep 10
    ray start --address $HEAD_IP:6379 --node-ip-address $NODE_IP --num-gpus 4 --disable-usage-stats
    until [ -f $RUN_DIR/done ]; do sleep 15; done
    ray stop --force
fi
'
