#!/usr/bin/env bash
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PT_DIR="$(cd "$HERE/.." && pwd)"
source "$PT_DIR/../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PT_OUT PT_TAG PT_MODEL PT_SEED0 PT_STEPS PT_GPUS HF_HOME

# The image's torch matches its accelerator; only ptmini's pure-Python deps are layered on top.
venv="$PT_OUT/.venv"
python3 -m venv --system-site-packages "$venv"
"$venv/bin/pip" install -q "transformers==5.23.0" "datasets==5.2.0" "accelerate==1.16.0"
"$venv/bin/pip" install -q --no-deps -e "$PT_DIR"
"$venv/bin/python" -c "from ptmini import task; task.load('test', 1)"  # one GSM8K download, not one per GPU

# ROCm honours CUDA_VISIBLE_DEVICES; also setting HIP_VISIBLE_DEVICES would index twice.
pids=()
for ((g = 0; g < PT_GPUS; g++)); do
  s=$((PT_SEED0 + g))
  CUDA_VISIBLE_DEVICES=$g "$venv/bin/ptmini" train \
    --out "$PT_OUT/$PT_TAG-full-s$s" --model "$PT_MODEL" --steps "$PT_STEPS" --seed "$s" --micro-bs 16 \
    > "$PT_OUT/$PT_TAG-full-s$s.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit "$rc"
