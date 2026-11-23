#!/bin/bash
#SBATCH -J ptmini-ref -N 1 --gres=gpu:4 --exclusive -t 04:00:00 -o logs/%x-%j.out
set -eo pipefail
source "$PTMINI_DIR/../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PTMINI_DIR TAG MODEL SEED0 STEPS
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # nvidia-smi -i and CUDA_VISIBLE_DEVICES agree
cd "$PTMINI_DIR"; mkdir -p runs logs
UV=$(command -v uv); [ -d .venv ] || $UV sync -q  # concurrent syncs on NFS race
for g in 0 1 2 3; do
  s=$((SEED0 + g))
  CUDA_VISIBLE_DEVICES=$g $UV run --no-sync ptmini train --out runs/$TAG-full-s$s --model "$MODEL" \
    --steps $STEPS --seed $s ${MICRO_BS:+--micro-bs $MICRO_BS} $EXTRA > runs/$TAG-full-s$s.log 2>&1 &
done
wait
