#!/usr/bin/env bash
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
source "$REPO_ROOT/inferencex-e2e/benchmarks/check_env.sh"

usage() {
  cat <<'USAGE'
usage: posttraining/run.sh <sku> --out DIR --tag TAG --model HF_ID --seed0 N --steps N --time MIN
  Runs one ptmini GRPO seed per GPU on one node of <sku> (configs/platforms.json) inside the
  platform's image, writing <out>/<tag>-full-s<seed>/. Needs HF_HOME (model + GSM8K cache).
USAGE
}

SKU="$1"
case "$SKU" in ""|-*) usage >&2; exit 2 ;; esac
shift
while [ "$#" -gt 0 ]; do
  case "$1" in
    --out) PT_OUT="$2"; shift 2 ;;
    --tag) PT_TAG="$2"; shift 2 ;;
    --model) PT_MODEL="$2"; shift 2 ;;
    --seed0) PT_SEED0="$2"; shift 2 ;;
    --steps) PT_STEPS="$2"; shift 2 ;;
    --time) PT_TIME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
check_env_vars PT_OUT PT_TAG PT_MODEL PT_SEED0 PT_STEPS PT_TIME HF_HOME

cfg() { jq -er --arg s "$SKU" ".[\$s].$1" "$HERE/configs/platforms.json"; }
cfg runner_label >/dev/null || { echo "unknown sku: $SKU" >&2; exit 2; }
PT_GPUS="$(cfg gpus_per_node)"
image="$(cfg image)"
partition="$(cfg partition)"
account="$(jq -r --arg s "$SKU" '.[$s].account' "$HERE/configs/platforms.json")"
mapfile -t srun_extra < <(jq -r --arg s "$SKU" '.[$s].srun_extra[]' "$HERE/configs/platforms.json")

mkdir -p "$PT_OUT" "$HF_HOME"
PT_OUT="$(cd "$PT_OUT" && pwd)"
HF_HOME="$(cd "$HF_HOME" && pwd)"
git -C "$REPO_ROOT" rev-parse HEAD > "$PT_OUT/source-sha"

alloc=(--partition="$partition" --nodes=1 --gres=gpu:"$PT_GPUS" --exclusive --time="$PT_TIME")
[ -z "$account" ] || alloc+=(--account="$account")
export PT_OUT PT_TAG PT_MODEL PT_SEED0 PT_STEPS PT_GPUS HF_HOME
srun "${alloc[@]}" --ntasks=1 "${srun_extra[@]}" \
  --container-image="$image" --container-writable \
  --container-mounts="$REPO_ROOT:$REPO_ROOT,$PT_OUT:$PT_OUT,$HF_HOME:$HF_HOME" \
  --export=ALL bash "$HERE/launchers/seeds.sh"
