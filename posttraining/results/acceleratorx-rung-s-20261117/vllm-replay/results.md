# vLLM forced-length replay of $PTMINI_RUNS/s05-full-s0

vLLM 0.33.0, torch 2.14.0+cu130, 1x NVIDIA GB300, bf16, defaults otherwise (CUDA graphs, prefix cache reset before each timed call). Engine load 216.4 s. Mean of 2 timed reps after 1 warmup step. Script: $PTMINI_VLLM_DIR/replay_vllm.py

| step | sum len | max len | HF t_gen (s) | vLLM per-request (s) | vLLM forced-max (s) |
|---|---|---|---|---|---|
| 0 | 27820 | 384 | 57.23 | 1.260 | 1.335 |
| 8 | 34462 | 384 | 18.71 | 1.195 | 1.502 |
| 17 | 28965 | 384 | 19.80 | 1.178 | 1.407 |
| 25 | 30647 | 384 | 18.59 | 1.221 | 1.540 |
| 34 | 25920 | 384 | 18.64 | 1.130 | 1.281 |
| 42 | 25585 | 334 | 16.49 | 1.116 | 1.475 |
| 51 | 29404 | 384 | 18.71 | 1.285 | 1.535 |
| 59 | 28946 | 384 | 18.83 | 1.259 | 1.293 |

Per-request fit t = 0.958 + 0.0085e-3 * sum_len, R^2 = 0.150

Notes:
- Install: `uv pip install vllm` (0.33.0 aarch64 wheel, pulls torch 2.14.0+cu130) worked on GB300 directly.
  FlashInfer (default attn backend, trtllm-gen decode, sm103) has no matching precompiled cubins, so it
  JIT-compiles at first start; needs env.sh (CUDA_HOME=/usr/local/cuda, venv bin on PATH for ninja).
  Cold first start 296 s; this run 216 s (most of it the first CUDA-graph capture), from Python start to LLM ready.
- The low R^2 is because run-to-run noise (up to +-0.18 s between the 2 reps) is as large as the sum_len
  effect: all steps but one decode 384 steps, so time is mostly per-decode-step overhead, not tokens.
  Per-request ~1.1-1.3 s vs forced-max ~1.3-1.5 s (per-request saves ~15%).
- HF step 0 (57 s) includes the reference run first-step warmup; steady-state HF is ~16.5-19.8 s, so
  vLLM per-request is ~15x faster on the same shapes.
- Raw numbers: replay.json. Launcher: $PTMINI_VLLM_DIR/replay.sbatch (1 GPU on a GB300 node).
