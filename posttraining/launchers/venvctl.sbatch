#!/bin/bash
#SBATCH -J ptmini-venvctl -N 1 --gres=gpu:4 --exclusive -t 04:00:00 -o logs/%x-%j.out
set -eo pipefail
source "$PTMINI_DIR/../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PTMINI_DIR TAG SEED0 STEPS VENV
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$PTMINI_DIR"; mkdir -p runs logs
export PYTHONPATH=$PWD
for g in 0 1 2 3; do
  s=$((SEED0 + g))
  CUDA_VISIBLE_DEVICES=$g $VENV/bin/python -m ptmini.cli train --out runs/$TAG-full-s$s \
    --steps $STEPS --seed $s --micro-bs 16 > runs/$TAG-full-s$s.log 2>&1 &
done
wait
