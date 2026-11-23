# miles: Qwen3.5 SWE RL (rung M)

Full-parameter GRPO of Qwen3.5-35B-A3B (or 122B-A10B) on SWE tasks with [miles](https://github.com/radixark/miles)
(Megatron trainer + SGLang rollout). terminus-2 agents run in Modal sandboxes through Harbor's miles agent server.
Pools are split: the first `--train-nodes` nodes train (TP2 + SP, experts over all trainer GPUs, distributed
optimizer, no offload), the rest sample.

## Environment

| Variable | Used by | Meaning |
|---|---|---|
| `PTX_ROOT` | all | Shared dir holding `runs/`, `logs/`, `datasets/`, `harbor/` (Harbor checkout + `.venv`), `harbor_tasks/`, `data/` |
| `MILES_IMAGE` | `train`, `convert` | miles container image (enroot squashfs) |
| `CONTAINER_MOUNTS` | `train`, `convert` | pyxis mounts that expose `PTX_ROOT`, the model dirs and this checkout, e.g. `/data:/data` |
| `RUN_ID` | `train` | Run name; output goes to `$PTX_ROOT/runs/$RUN_ID` |
| `AGENT_SERVER_URL` | `train` | Optional; otherwise read from `$PTX_ROOT/logs/harbor_endpoint.txt` |
| `MODEL_TYPE`, `HF_CHECKPOINT`, `SAVE_DIR` | `convert` | miles model type (`qwen3.5-35B-A3B`, `qwen3.5-122B-A10B`), HF weights, torch_dist output |
| `PORT`, `MAX_CONCURRENT`, `AGENT_TIMEOUT` | `harbor_modal` | Agent server port, concurrent trials, per-trial seconds |
| `HARBOR_MODAL_APP`, `HARBOR_MODAL_TIMEOUT_SECS`, `HARBOR_MODAL_IDLE_TIMEOUT_SECS` | `harbor_modal` | Modal app name and sandbox lifetime caps |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | `harbor_modal`, `prep_r2e_mlperf extract` | Modal credentials, unless `~/.modal.toml` exists |

Submit every sbatch from inside this checkout (the scripts find the repo from `SLURM_SUBMIT_DIR`), with
`-o $PTX_ROOT/logs/%x-%j.out`.

## Steps

0. **Harbor**: `setup_harbor.sh` clones Harbor at `ac63e51` into `$PTX_ROOT/harbor`, builds its venv with Modal, applies `patch_harbor.py`.
1. **Convert** HF → Megatron torch_dist (the `--ref-load`), once per model: `sbatch convert.sbatch`.
2. **Data.** Python scripts read `PTX_ROOT` and write Harbor task dirs to `harbor_tasks/` and miles JSONL to `datasets/`:
   - `prep_mimo.py --parquet <code.parquet>`: MiMo-V2.6-RL-oss code tasks → `mimo_train` / `mimo_eval32`, plus the ids
     of tasks whose tests touch R2E-Gym or SWE-bench Verified repos and each task's test language.
   - `prep_r2e_mlperf.py extract|build`: MLPerf v6.1's R2E-Gym easy split (700 train in curriculum order, 251 val).
     Needs MLPerf's converter and id lists in `data/mlperf_r2e`, HF `R2E-Gym/R2E-Gym-Subset` in `data/r2e-gym-subset`.
   - `prep_swebench_verified.py`: SWE-bench Verified (500) for before/after transfer evals.
   - `build_mimo_learnable.py`: from solve-rate scans (`--num-rollout 0 --eval-temperature 1.0`), keep the tasks
     the model solves `--lo`..`--hi` of the time, drop tasks that overlap R2E-Gym or SWE-bench Verified, split a rate-stratified eval set.
3. **Harbor** on a CPU node: `sbatch harbor_modal.sbatch` (writes `logs/harbor_endpoint.txt`).
4. **Train**: `RUN_ID=... sbatch -N4 train.sbatch <launcher args>`; `train.sbatch` waits up to 5 min for Harbor.
   A run dir that already holds checkpoints resumes.

## Recipes

| Recipe | Flags | What changes |
|---|---|---|
| Default | — | GRPO, PPO clip 0.2 / 0.28, lr 1e-6, temp 0.8, flat groups dropped and resampled |
| MLPerf | `--mlperf [--mlperf-seq-mask]` | MLPerf v6.1 `qwen35_397b_grpo`: lr 1e-6·√(GBS/256), clip 0.125·√(256/GBS), AdamW 0.9/0.999 wd 0, temp 1.0 (eval 0.1, top_p 0.95), fixed data order, flat groups kept, token TIS truncated at 1.002 |
| Score centering | `--sc [--sc-topk 20]` | `--loss-type score_centering` (arXiv:2609.20807) on sampler top-K logprobs, Dr.GRPO advantages, token-level loss, flat-group filter, MLPerf optimizer |

MLPerf sanity run:
`--prompt-data $PTX_ROOT/datasets/r2e_mlperf_train.jsonl --eval-data $PTX_ROOT/datasets/r2e_mlperf_val.jsonl --mlperf --mlperf-seq-mask --rollout-batch-size 16 --n-samples-per-prompt 16 --num-rollout 30`
plus `--hf-checkpoint`, `--ref-load`.
122B: `--megatron-model-type qwen3.5-122B-A10B --train-nodes 4 --rollout-gpus-per-engine 2` on 5 nodes.

## Results (GB300)

Qwen3.5-35B-A3B, MLPerf recipe on R2E-Gym easy, 4 nodes (2 train + 2 sample), val 251 tasks × 4:

| step | 0 | 5 | 10 | 15 | 25 | 30 |
|---|---|---|---|---|---|---|
| pass@4 (MLPerf target 0.69) | 0.673 | 0.705 | 0.709 | 0.701 | 0.749 | 0.729 |
| avg@4 | 0.485 | 0.510 | 0.502 | 0.499 | 0.550 | 0.537 |

- Crossed 0.69 at the step-5 eval, 1 h 39 min after training started: 26 GB300-hours.
- The base is 1.7 pt under the target, so on this data the target measures throughput more than convergence.
- On learnable MiMo SWE sets (tasks the base solves 10–70% or 25–75% of the time, eval 64 × 8 at temp 1.0),
  avg@8 stayed flat over 20 steps (0.318 → 0.299 on the 25–75% set).

Deviations from MLPerf: 35B instead of 397B; terminus-2 instead of OpenHands; synchronous rollouts; GRPO mean
baseline; grading in the agent's container.
