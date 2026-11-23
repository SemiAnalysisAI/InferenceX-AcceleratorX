#!/bin/bash
#SBATCH -J ptmini-chip -N 1 --gres=gpu:4 --exclusive -t 04:00:00 -o logs/%x-%j.out
set -eo pipefail
source "$PTMINI_DIR/../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PTMINI_DIR TAG REF STEPS LOCK_MHZ FP16
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # nvidia-smi -i and CUDA_VISIBLE_DEVICES agree
cd "$PTMINI_DIR"; mkdir -p runs logs
UV=$(command -v uv)
MODEL=$(python3 -c "import json;print(json.load(open('$REF/meta.json'))['cfg']['model'])")
MB=${MICRO_BS:+--micro-bs $MICRO_BS}
run() { CUDA_VISIBLE_DEVICES=$1 $UV run --no-sync ptmini "${@:2}"; }
sudo nvidia-smi -rgc >/dev/null; trap "sudo nvidia-smi -rgc >/dev/null" EXIT
sudo nvidia-smi -i 1,2 -lgc "$LOCK_MHZ,$LOCK_MHZ"
L=$TAG-lock$LOCK_MHZ
# GPU0 stock replay+parity, GPU1 locked full run (ground truth), GPU2 locked replay, GPU3 control.
{ run 0 replay --ref $REF --out runs/$TAG-replay --k 12 $MB && run 0 parity --ref $REF --out runs/$TAG-parity; } > runs/$TAG-replay.log 2>&1 &
run 1 train --out runs/$L-full-s0 --model "$MODEL" --steps $STEPS --seed 0 $MB > runs/$L-full-s0.log 2>&1 &
run 2 replay --ref $REF --out runs/$L-replay --k 12 $MB > runs/$L-replay.log 2>&1 &
if [ "$FP16" = 1 ]; then
  run 3 train --out runs/$TAG-fp16-full-s0 --model "$MODEL" --steps $STEPS --seed 0 --dtype fp16 $MB > runs/$TAG-fp16-full-s0.log 2>&1 &
else
  sudo nvidia-smi -i 3 -lgc "$LOCK_MHZ,$LOCK_MHZ"
  run 3 train --out runs/$L-full-s1 --model "$MODEL" --steps $STEPS --seed 1 $MB > runs/$L-full-s1.log 2>&1 &
fi
wait
