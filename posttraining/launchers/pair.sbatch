#!/bin/bash
#SBATCH -J ptmini-pair -N 1 --gres=gpu:4 --exclusive -t 03:00:00 -o logs/%x-%j.out
set -eo pipefail
source "$PTMINI_DIR/../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PTMINI_DIR TAG REF STEPS LOCK_MHZ
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # nvidia-smi -i and CUDA_VISIBLE_DEVICES agree
cd "$PTMINI_DIR"; mkdir -p runs logs
UV=$(command -v uv)
MODEL=$(python3 -c "import json;print(json.load(open('$REF/meta.json'))['cfg']['model'])")
[ "runs/$TAG-full-s0" != "$REF" ] || { echo "TAG would overwrite REF"; exit 1; }
MB=${MICRO_BS:+--micro-bs $MICRO_BS}
run() { CUDA_VISIBLE_DEVICES=$1 $UV run --no-sync ptmini "${@:2}"; }
sudo nvidia-smi -rgc >/dev/null; trap "sudo nvidia-smi -rgc >/dev/null" EXIT
sudo nvidia-smi -i 2,3 -lgc "$LOCK_MHZ,$LOCK_MHZ"
L=$TAG-lock$LOCK_MHZ
# Projection and ground truth share one node so both see the same environment.
run 0 train --out runs/$TAG-full-s0 --model "$MODEL" --steps $STEPS --seed 0 $MB > runs/$TAG-full-s0.log 2>&1 &
run 1 replay --ref $REF --out runs/$TAG-replay --k 12 > runs/$TAG-replay.log 2>&1 &
run 2 train --out runs/$L-full-s0 --model "$MODEL" --steps $STEPS --seed 0 $MB > runs/$L-full-s0.log 2>&1 &
run 3 replay --ref $REF --out runs/$L-replay --k 12 > runs/$L-replay.log 2>&1 &
wait
