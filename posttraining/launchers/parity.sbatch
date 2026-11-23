#!/bin/bash
#SBATCH -J ptmini-parity -N 1 --gres=gpu:1 -t 00:30:00 -o logs/%x-%j.out
set -eo pipefail
source "$PTMINI_DIR/../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars PTMINI_DIR REF OUT
cd "$PTMINI_DIR"
uv run --no-sync ptmini parity --ref "$REF" --out "$OUT"
