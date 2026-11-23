# Post-training benchmark design (draft)

Goal: compare accelerators (NVIDIA, AMD, TPU, Trainium, Ascend, PPU) on RL post-training. Results
should be explainable, priced, continuous, hard to game, and cheap enough in chip time that
immature stacks can take part. This is the chip axis of PostTrainingX. Hosted providers (Tinker, Fireworks)
and framework comparisons are separate tracks: frameworks appear here only as the optimized stack
a vendor runs.

## MLPerf post-training, and where we go further

MLPerf Training LLM post-training (Sept 2026) runs Qwen3.5-397B-A17B with GRPO on R2E-Gym agentic
coding until grouped pass@4 ≥ 0.69. It reports wall-clock time-to-train. The reference setup is
256 GB300 for ~150 min (≈640 GB300-h) per run, with 6 reference runs per batch size as convergence
points.

| MLPerf | Limitation | This design |
|---|---|---|
| Full time-to-train for every entry | Hundreds of chip-hours per entry; immature stacks stay out | Convergence measured once per config; a new chip pays for parity + replay + a short run |
| One end-to-end number | No attribution | Generation, training, weight sync and overlap reported separately, plus end-to-end |
| One model at 256 chips | Nothing for one node or one chip | Ladder from 1 chip to a rack |
| Power is a separate, optional track; no $ | Faster ≠ cheaper | $ and kWh to target from the InferenceX TCO and power model, for every result |
| Correctness = reached the target | A lossy stack can pass on seed luck | Calibrated numerics gate + early-dynamics check against reference seeds |
| Submitter-tuned, twice a year | Stale | Nightly CI with software versions recorded |

Kept from MLPerf: a held-out target metric, a fixed algorithm with a batch→LR rule, a cap on
trajectory age, eval time counted, and a chip-hour cap per entry.

## Method: measure convergence once per config, measure speed per chip

Time-to-target = s\* × step time. s\* is the number of steps to reach the target, which depends on
the algorithm, config and numerics. Step time depends on hardware and software for a fixed
per-step workload. MLPerf measures their product on every chip. We measure each factor where it is
cheap, and reuse s\* only when a chip has shown it reproduces the reference dynamics.

| Tier | Runs on | Cost | Pass condition / output |
|---|---|---|---|
| T0 numerics | every chip and stack | minutes | Per-token logprob error on reference sequences vs fp32 ≤ k × the reference chip's own bf16 error. Measured on the stack actually submitted, KV-cache decode path included. Also: sampler-vs-trainer importance-ratio spread no worse than reference. k comes from dose-response; rung S gives k ≈ 10 (tentative). |
| T1 replay | every chip × scale point | minutes | n steps of a hidden, rotating reference trace. Same prompts; each request's sampled decode is forced to its recorded length, so token content isn't known in advance and there is no prefill shortcut; the engine schedules freely. The engine returns sampled tokens and logprobs, and a sample is re-scored by the trainer and checked against T0 bounds. Output: per-phase times and a cost model. |
| T2 dynamics | every chip and stack | first ~10–30% of steps | Equivalence (TOST) of mean eval accuracy over a step window, plus reward/KL trajectories, against ≥ 16 reference seeds. Rung S: the window mean put int4 at −20 pt (90% CI −33…−8), fp8 within ±5 pt (4 seeds) and vLLM rollouts within ±2 pt (8 seeds). The trainer's software version is part of the arm. |
| T3 full run | reference chip (seeds); each chip family once per major software version | full | Distribution of s\* and of collapse rate. The chip-family run validates T1–T2 on that family. |

**Projection** = warmup + Σ predicted step time over the reference trace's shapes before s\* + one
cold eval + (evals − 1) warm evals. Replay measures all of these terms. Price and energy come from
multiplying by TCO and power.

**s\* is a distribution.** RL runs collapse at random. The headline is expected time-to-target,
published with the collapse rate and seed CV. T2 cannot detect a change in a rare collapse rate;
that is what the periodic T3 is for.

**The target must sit on the steep part of the curve.** s\* resolution is bounded by eval
cadence, eval standard error, and the curve's slope at the target. Rung S put the target (0.45) on
the plateau (0.44–0.50): steps-to-target became "when does eval noise cross the line", and doubling
the batch looked slower. Rule: choose the target from reference curves where accuracy still rises
steeply, verify it on ≥ 1,024 eval problems, and report window-mean accuracy alongside s\*.

**Measurement hygiene.** Every step logs timestamp, SM clock and power. Projection and ground truth
share a node and a time window, and throttled windows are re-run. Rung S hit a rack-wide GPU
slowdown that moved unpaired comparisons by up to 15%, and cold start (first eval) varied 92–400 s
on identical nodes, so cold start is measured and reported separately from steady state. Sampler
settings are declared and matched: HF `generate` silently applies a model's
`generation_config` (Qwen: `repetition_penalty=1.1`).

**Configs, not free tuning.** s\* is published for a fixed table: GBS ∈ {B, 2B, 4B} with
LR ∝ √GBS, and trajectory age ∈ {0, 1}. Vendors pick a row.

**Topology.** For disaggregated generation and training, step time is not the sum of the phases.
Submitters declare a schedule (sync or async-1, and the chip split). We compose phase measurements
under it, validate against a live steady-state step, and publish the composition error and
weight-sync bandwidth.

## Workload ladder and tracks

| Rung | Model | Task | Scale |
|---|---|---|---|
| S | Qwen2.5-0.5B (full FT) | GSM8K GRPO | 1 chip; method validation (pilot ran LoRA; full-FT reference pending) |
| M | Qwen3.5-35B-A3B (full FT, miles) | MiMo-V2.6 SWE tasks the model solves 10–70% of the time, deduplicated against R2E-Gym and SWE-bench Verified; MLPerf's R2E-Gym split only as a sanity check | 4 nodes; sanity pass@4 0.673 → 0.749 in 25 steps (RESULTS E8); MiMo target not set yet (`miles/`) |
| L | GLM-5.3-Flash 320B/18B | agentic coding | 16–64 chips; PostTrainingX miles, SkyRL, verl and prime-rl ports |

- **Reference implementation** (plain PyTorch): defines correctness and s\*. It is not a speed
  measurement: on rung S its generation is host-bound.
- **Optimized track** (vLLM + verl, miles, slime, SkyRL, prime-rl, vendor forks): any stack that
  passes T0 and T2. All published speed comes from here.

**Published per chip × rung × scale:**
- headline time-, $- and kWh-to-target, with validation error;
- phase metrics: generation tok/s, trainer tok/s and MFU (6 × active params × trained tokens ÷
  (time × chips × dense BF16 peak), as in the PostTrainingX MOPD article), weight-sync bandwidth, overlap;
- cost lines: accelerator $ at InferenceX TCO (one price per chip across all PostTrainingX docs), plus
  CPU/sandbox $ for agentic rungs, which our cost tables have left out so far;
- T0 and T2 verdicts;
- chip-hours spent on the measurement.

## Rung-S pilot (GB300, 1 day; RESULTS.md)

| ID | Question | Result |
|---|---|---|
| E1 | How noisy is s\*? | 16 seeds: steps-to-target 5–25, CV 52%, 4/16 collapse later. Most of it is eval noise: the target sits on the plateau (1,024-problem eval). Micro-batch 4 vs 16 (same math, 16 seeds each): s\* 15 vs 10 (p = 0.06), collapses 0 vs 4 (p = 0.10). |
| E2/E3 | Does replay predict step time on another chip? | Steady step time within 2–7% (median) from 4 replayed steps. Wall over s\* within −4…+5% on stock and on 300/600 MHz-locked GB300s (up to 1.29× slower). Same chip family, so this tests speed projection, not cross-vendor s\* transfer. |
| E4 | Does T0 track training damage? | k ≈ 10 set on random logit noise, then tested blind on systematic damage: fp8 e4m3 (3.8×) and e5m2 (6.8×) rollout weights pass and train like the reference; int4 (37×) fails, 0/4 reach target. fp16 is 8× *more* accurate than bf16. |
| E5 | Chip-time saving | New chip 1,000 chip-s vs 3,667–12,000 for the 7–23 real runs needed for ±10% (8- vs 16-seed CV): 3.7–12×. One real run is 524. The saving grows with s\*. |
| E6 | Reference vs optimized | vLLM generates the same trace 15× faster than the reference; forcing every request to the batch max overstates its cost by 15% |
| E7 | Does s\* transfer to an optimized stack? | GRPO with vLLM rollouts vs HF rollouts on the same software (8 seeds each): window-mean accuracy −0.1 pt (90% CI ±1.8), sampler–trainer logprob gap 0.014. A torch/transformers version change alone moved it +2 pt. Small model only. |

## Open questions

- **s\* transfer across vendors**, and across stacks at a target on the steep part of the curve:
  the first cross-vendor T3 decides whether one reference s\* is enough.
- **The k threshold** lies in ~9–37× at rung S. fp8 KV cache and biased kernels are untested.
- **Rare collapse rates:** 4/16 reference collapses; equivalence needs far more seeds than T2 has.
- **Agentic multi-turn replay:** recorded vs sampled tool latency.
- **A rung-M target that converges:** on learnable MiMo sets avg@8 stayed flat for 20 steps (two bands,
  two loss variants pending). Rung M needs a task set and batch where the 35B climbs before a target can sit on
  the steep part.
- **Cross-vendor runs:** `run.sh` and the Post-training Sweep workflow run rung S on gb300 / b300 / mi355x
  and compare against the GB300 seeds; none has run on a runner yet.
