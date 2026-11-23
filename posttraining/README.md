# Post-training (experimental)

RL post-training benchmarks across accelerators. The method is in [DESIGN.md](DESIGN.md) and the
rung-S pilot numbers are in [RESULTS.md](RESULTS.md).

`ptmini` is the rung-S reference: GRPO on GSM8K with Qwen2.5, full fine-tune (bf16 weights, fp32
AdamW master copy). It uses plain PyTorch and HF
`generate`, so it runs on any backend that exposes a torch device (`cuda`, `mps`, `npu`, `xla`, …).

| Command | Tier | Does |
|---|---|---|
| `ptmini train` | T3 / T2 | Full GRPO run. Writes `steps.jsonl` (per-step phase times, reward, eval acc) and `trace.pt` (prompts, completions, advantages) |
| `ptmini replay --ref RUN` | T1 | Re-runs n trace steps (`--k`) with identical shapes and writes per-phase times |
| `ptmini parity --ref RUN` | T0 | Per-token logprobs of reference sequences in bf16, fp16, fp32, noised-logit and fp8/int4-weight variants |
| `python -m ptmini.analyze` | – | E1–E5 tables |
| `optimized/vllm_replay.py` | T1, optimized track | vLLM replay of a trace with per-request forced lengths |
| `optimized/vllm_grpo.py` | T3, optimized track | `ptmini train` with vLLM rollouts (s\* transfer) |

`ptmini train --sampler-fp8 {e4m3,e5m2,int4}` rounds the sampler's weights (trainer stays bf16) and
`--logit-noise σ` noises its logits: the controlled damage used to calibrate T0.

```bash
cd posttraining && uv sync
uv run ptmini train --out runs/ref-s0 --seed 0
uv run ptmini replay --ref runs/ref-s0 --out runs/mychip-replay --device npu
uv run ptmini parity --ref runs/ref-s0 --out runs/mychip-parity --device npu
```

## Comparing a chip against the reference

`run.sh <sku>` runs one GRPO seed per GPU on one node of a platform in `configs/platforms.json`,
inside that platform's vLLM image (its torch build matches the accelerator; ptmini's Python deps
are layered on top). The `Post-training Sweep` workflow dispatches it on the InferenceX Slurm
runners and writes the E7 table to the job summary: seeds that reach acc ≥ 0.45, steps to target,
mean eval accuracy over steps 5–30 against the GB300 full-FT reference seeds (90% Welch CI), and
s/step. Those seeds go in `results/rung-s-fullft/` once run; until then the report step says so.

```bash
HF_HOME=/shared/hf posttraining/run.sh mi355x --out runs-mi355x --tag mi355x \
  --model Qwen/Qwen2.5-0.5B-Instruct --seed0 0 --steps 60 --time 180
```

Research launchers in `launchers/` (GB300 Slurm, `PTMINI_DIR` = this folder with a synced `.venv`):

| Launcher | Runs | Env |
|---|---|---|
| `ref` | Seeds, one per GPU | `TAG MODEL SEED0 STEPS`, optional `MICRO_BS EXTRA` |
| `pair` | Full run and replay on a stock GPU and a clock-locked GPU, side by side | `TAG REF STEPS LOCK_MHZ` |
| `chip` | T0/T1 plus an fp16 arm (`FP16=0`: a second locked seed instead) | `TAG REF STEPS LOCK_MHZ FP16` |
| `dose` | The numerics dose-response | `TAG RUNS` (four `sigma:seed`) `MODEL STEPS` |
| `parity` | T0 only, one GPU | `REF OUT` |
| `vllm_grpo` | vLLM-rollout seeds | `TAG SEED0 STEPS VENV NGPU`, optional `EXTRA` |
| `venvctl` | HF-sampler control inside the vLLM venv | `TAG SEED0 STEPS VENV` |

Rung M (Qwen3.5 MoE full fine-tune with miles on SWE tasks): [`miles/`](miles/README.md).
