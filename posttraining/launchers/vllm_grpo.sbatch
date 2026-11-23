#!/bin/bash
#SBATCH -J ptmini-vllm -N 1 --gres=gpu:4 --exclusive -t 04:00:00 -o logs/%x-%j.out
set -eo pipefail
source "$PTMINI_DIR/../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PTMINI_DIR TAG SEED0 STEPS VENV NGPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # nvidia-smi -i and CUDA_VISIBLE_DEVICES agree
cd "$PTMINI_DIR"; mkdir -p runs logs
export PYTHONPATH=$PWD VLLM_CACHE_ROOT=/tmp/vllm-cache TRITON_CACHE_DIR=/tmp/triton-cache
export FLASHINFER_WORKSPACE_BASE=/tmp/fi-cache CUDA_HOME=/usr/local/cuda PATH=$VENV/bin:/usr/local/cuda/bin:$PATH
for g in $(seq 0 $((NGPU - 1))); do
  s=$((SEED0 + g))
  CUDA_VISIBLE_DEVICES=$g python optimized/vllm_grpo.py --out runs/$TAG-s$s --seed $s \
    --steps $STEPS $EXTRA > runs/$TAG-s$s.log 2>&1 &
done
wait
