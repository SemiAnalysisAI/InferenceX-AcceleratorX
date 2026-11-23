#!/bin/bash
#SBATCH -J ptx-miles-convert
#SBATCH -N1 --gres=gpu:4 --exclusive
#SBATCH -t 2:00:00

set -eo pipefail
REPO_ROOT=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
source "$REPO_ROOT/inferencex-e2e/benchmarks/check_env.sh"
check_env_vars MILES_IMAGE CONTAINER_MOUNTS MODEL_TYPE HF_CHECKPOINT SAVE_DIR
CODE_DIR=$REPO_ROOT/posttraining/miles

srun --container-image="$MILES_IMAGE" --container-mounts="$CONTAINER_MOUNTS" --no-container-remap-root bash -c "
set -e
cd /root/miles
cp $CODE_DIR/models/*.py scripts/models/
MODEL_ARGS_LINE=\$(python3 miles/utils/external_utils/model_args_utils.py $MODEL_TYPE)
read -ra MODEL_ARGS <<< \"\$MODEL_ARGS_LINE\"
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/root/Megatron-LM \
    torchrun --nproc-per-node 4 tools/convert_hf_to_torch_dist.py \"\${MODEL_ARGS[@]}\" \
    --expert-model-parallel-size 4 \
    --hf-checkpoint $HF_CHECKPOINT \
    --save $SAVE_DIR
du -sh $SAVE_DIR
"
