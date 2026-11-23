#!/bin/bash
#SBATCH -J ptmini-dose -N 1 --gres=gpu:4 --exclusive -t 04:00:00 -o logs/%x-%j.out
set -eo pipefail
source "$PTMINI_DIR/../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PTMINI_DIR TAG RUNS MODEL STEPS
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # nvidia-smi -i and CUDA_VISIBLE_DEVICES agree
cd "$PTMINI_DIR"; mkdir -p runs logs
UV=$(command -v uv)
g=0
for r in $RUNS; do
  n=${r%:*}; s=${r#*:}
  CUDA_VISIBLE_DEVICES=$g $UV run --no-sync ptmini train --out runs/$TAG-noise$n-s$s --model "$MODEL" \
    --steps $STEPS --seed $s --logit-noise $n ${MICRO_BS:+--micro-bs $MICRO_BS} > runs/$TAG-noise$n-s$s.log 2>&1 &
  g=$((g + 1))
done
wait
