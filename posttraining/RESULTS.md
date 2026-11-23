# Rung-S pilot results (GB300, 2026-11-17)

> **Superseded setup.** This pilot trained LoRA adapters. ptmini is now a full fine-tune (all weights,
> fp32 master copy, lr 1e-6), so these numbers validate the method, not the current reference. The
> full-FT reference seeds that the Post-training Sweep compares against are still to be run.

Setup: Qwen2.5-0.5B-Instruct, LoRA r64, GRPO on GSM8K. 16 prompts × 8 samples per step, ≤384 new
tokens. Eval is greedy accuracy on 256 GSM8K test problems every 5 steps unless noted. One run per
GB300 GPU on GB300 nodes. "Chip B" is a GB300 with SM clocks locked at 300 or
600 MHz. Run tags: `s05` = micro-batch 16 (reference), `s05m4` = micro-batch 4 (same math). Tables
come from:

    python -m ptmini.analyze --tag s05 --target 0.45 --pair p300 p600 --chip s05c300 \
        --arms s05m4 s05fp8e4m3:s05m4 s05fp8e5m2:s05m4 s05fp8int4:s05m4 s05g2:s05m4 \
        venvctl vllm vllm:venvctl

**Bottom line**
- Replay predicted wall time over s\* steps within −4% … +5% on stock and slowed chips, with
  projection and ground truth on the same node at the same time.
- The T0 gate, calibrated on random logit noise (k ≈ 10), called all three unseen systematic
  arms correctly: fp8 e4m3/e5m2 rollouts pass and stay within ±5 pt of the reference; int4
  rollouts fail, lose 20 pt and never reach target.
- Swapping the HF sampler for vLLM (optimized stack) leaves training unchanged within ±2 pt
  against a same-software control: small-model evidence that s\* transfers across stacks. A
  torch/transformers version change alone moved accuracy +2 pt (CI spans 0).
- Qualifying a new chip costs 1,000 chip-s. Real runs to ±10% cost 3,667 (8-seed CV) to ~12,000
  (16-seed CV) chip-s: 3.7–12×.
- The rung-S target sits on the accuracy plateau, so steps-to-target is mostly eval noise. Mean
  eval accuracy over a step window is the stable comparison. Rung M must fix the target.
- Not shown: s\* transfer across vendors.

## E1: one run per chip measures seed luck (16 seeds, target acc ≥ 0.45)

| seeds | steps-to-target | collapse | steps CV | wall CV |
|---|---|---|---|---|
| 0–7 | 10, 5, 10, 10, 10, 5, 10, 15 | 2/8 | 34% | 25% |
| 0–15 | 5 ×3, 10 ×8, 15 ×2, 25 ×2 | 4/16 | 52% | 47% |

- Collapse = eval accuracy later falls below half the base (seeds 1, 6, 9, 15: → 0.01–0.17).
- Steady step time CV is 1.5%. Cold start is not: the first eval took 92 s in one batch and
  290–400 s in another on identical nodes (NFS load contention). Wall CV over 16 seeds includes it.
- Micro-batch 4 vs 16 is the same math (gradient accumulation) but not obviously the same
  dynamics: 16 seeds each give s\* 15 vs 10 (permutation p = 0.06) and 0/16 vs 4/16 collapses
  (Fisher p = 0.10), while window-mean accuracy agrees (+1.8 pt, CI spans 0). Training time per
  step is 14.6 vs 6.2 s. A math-preserving implementation change may move s\*; that is what T2 is
  for, and 16 seeds can't settle it.
- Qwen2.5-3B (4 seeds) starts at 0.816 and stays flat: GSM8K is saturated for it. Not used.

**The target sits on the plateau.** One run with 1,024 eval problems every 2 steps (4 seeds,
`s05f`): accuracy goes 0.31 → 0.42 in 2 steps (mostly format learning), then wanders in 0.40–0.50.
Eval standard error at 0.45 is ±3.1 pt at 256 problems and ±1.6 pt at 1,024.

| target (1,024-problem eval) | steps-to-target, 4 seeds |
|---|---|
| 0.42 | 4, 2, 2, 6 |
| 0.44 | 10, 4, 6, 10 |
| 0.45 | 14, 6, 6, 10 |

A target near the plateau turns s\* into "when does eval noise cross the line". The target must sit
on the steep part of the curve and be checked at ≥1,024 eval problems.

## E2/E3: replay projection (`launchers/pair.sbatch`)

Each node ran four jobs side by side:
- a 20-step full run on a stock GPU;
- a 20-step full run on a locked GPU;
- a 12-step replay of reference seed 0's trace on each of the two.

Projection = warmup + per-phase cost model over the reference shapes before s\*=10 + cold eval +
2 warm evals.

| chip | slowdown | projected s (4 / 12 steps replayed) | measured s | err (4 / 12) |
|---|---|---|---|---|
| stock | 1.00× | 622 / 593 | 610 | +2.0% / −2.7% |
| 300 MHz | 1.29× (train 2.3×, gen 1.1×) | 699 / 674 | 668 | +4.6% / +0.9% |
| stock | 1.00× | 604 / 584 | 609 | −0.8% / −4.1% |
| 600 MHz | 1.10× | 643 / 622 | 642 | +0.0% / −3.1% |

- Steady step-time error with 4 replayed steps: 1.7–6.5% median, depending on the run.
- The cold eval costs 146 s and the warm eval 76 s.
- The first replay version charged the cold-eval cost to every eval and overestimated by +5.7%. A
  code review found this; it is fixed above.
- The clock lock changes speed, not chip family. Cross-vendor s\* transfer is untested.

**Environmental interference.** Before telemetry was added, every run on four nodes slowed 3–5× in
training during the same wall-clock window. Generation was unchanged, which points to a GPU-side
cause, probably rack power capping. Projections checked against runs from a different time window
were off by −7% to +15%. Per-step clock/power telemetry and paired measurement fix this.

## E4: numerics gate (T0), random and systematic damage

T0 ratio = mean per-token |logprob − fp32| on 32 reference sequences, divided by the reference bf16
error (0.0292). Damage is applied to the sampler only; the trainer stays bf16. Random: Gaussian
noise σ on the logits. Systematic: frozen weights rounded to fp8 (per-tensor scale) or int4
(per-channel scale) during generation, as a low-precision rollout engine would.

| arm | T0 ratio | never reach | collapse | below seed band @ step 5 / 10 |
|---|---|---|---|---|
| reference bf16 | 1.0 | 0/16 | 4/16 | – |
| fp16 | 0.13 | – | – | – |
| fp8 e4m3 weights | 3.8 | 0/4 | 0/4 | 0/4 / 0/4 |
| σ 0.25 | 4.4 | 0/2 | 0/2 | 0/2 / 0/2 |
| fp8 e5m2 weights | 6.8 | 0/4 | 1/4 | 0/4 / 0/4 |
| σ 0.5 | 8.7 | 0/2 | 0/2 | 0/2 / 1/2 |
| σ 1 | 18.6 | 0/4 | 1/4 | 1/4 / 1/4 |
| int4 weights | 37.0 | 4/4 | 3/4 | 2/4 / 3/4 |
| σ 2 | 45.8 | 3/4 | 2/4 | 2/4 / 2/4 |
| σ 4 | – | 3/4 | 3/4 | 3/4 / 3/4 |

- k ≈ 10 was set from the σ arms before the fp8/int4 runs. It then separated the systematic arms
  correctly: both fp8 formats pass and are within ±5 pt of the reference on mean accuracy (E7);
  int4 fails and loses 20 pt.
- fp8 e5m2 passes but dipped hard in 3/4 seeds (to 0.08–0.29) and recovered. n=4 can't separate
  this from the reference's 4/16 collapse rate.
- 18.6× is still indistinguishable from reference noise at n=4; the true threshold lies in
  ~9–37×.
- fp16 is 8× *more* accurate than bf16 (also at 3B), so the gate uses measured error, not dtype.

## E5: chip time to qualify a new chip

| item | chip-s |
|---|---|
| T0 parity (bf16, fp16, fp32) | 87 |
| T1 replay (cold + warm eval, warmup, 4 steps) | 389 |
| T2 short run (10 steps + evals) | 524 |
| **new chip** | **1,000** |
| one real run to s\* = 10 | 524 |
| real runs for ±10% on mean time-to-target, 8-seed CV 25%: n = 7 | 3,667 |
| same, 16-seed CV 47%: n = 23 | ~12,000 |

The 16-seed CV is inflated by the plateau target (E1), so the true saving at rung S lies between
3.7× and 12×. At rung S, T2 equals a whole run because s\* is tiny. The saving grows with s\*: T0
and T1 are fixed costs, and T2 is ~10% of s\*.

## E6: reference vs optimized generation (vLLM 0.33, 1 GB300, same trace)

| | per-step gen |
|---|---|
| HF reference | ~18.6 s |
| vLLM, per-request forced lengths | ~1.2 s (15× faster) |
| vLLM, every request forced to step max | +15% over per-request |

- The reference's generation is host-bound. At 300 MHz it slowed 1.1× while training slowed
  2.3×, so the reference can't measure the chip.
- vLLM engine load is 216–296 s, a one-time cost to count. Script: `optimized/vllm_replay.py`.

## E7: other configs and stacks (mean eval acc over steps 5–30)

Mean accuracy over a step window is far less noisy than steps-to-target here. Δ is against the
named reference seeds, with a 90% Welch t CI (unpaired; 4 seeds per arm gives ±3–5 pt).

| arm | vs | seeds | reach | collapse | mean acc | Δ [90% CI] | s/step |
|---|---|---|---|---|---|---|---|
| micro-batch 4 | s05 | 16 | 16/16 | 0/16 | 0.447 | +0.018 [−0.010, +0.045] | 34.7 |
| fp8 e4m3 rollouts | s05m4 | 4 | 4/4 | 0/4 | 0.438 | −0.009 [−0.028, +0.009] | 37.4 |
| fp8 e5m2 rollouts | s05m4 | 4 | 4/4 | 1/4 | 0.436 | −0.011 [−0.049, +0.027] | 34.2 |
| int4 rollouts | s05m4 | 4 | 0/4 | 3/4 | 0.243 | −0.204 [−0.329, −0.079] | 40.9 |
| GBS ×2, LR ×√2 | s05m4 | 4 | 3/4 | 0/4 | 0.442 | −0.005 [−0.023, +0.012] | 58.1 |
| HF sampler, vLLM venv | s05 | 8 | 8/8 | 0/8 | 0.452 | +0.023 [−0.004, +0.050] | 29.7 |
| vLLM rollouts | s05 | 8 | 8/8 | 0/8 | 0.451 | +0.022 [−0.008, +0.051] | 8.0 |
| vLLM rollouts | vLLM-venv HF | 8 | 8/8 | 0/8 | 0.451 | −0.001 [−0.019, +0.017] | 8.0 |

- **vLLM rollouts** (`optimized/vllm_grpo.py`): colocated vLLM 0.33 sampler, LoRA pushed each step,
  same HF trainer and eval. Generation 1.8 s/step vs 19 s (10×); adapter sync 1.3 s (in s/step). Mean
  |sampler − trainer logprob| on sampled tokens is 0.014 (4% of tokens > 0.1). HF `generate`
  silently applies Qwen's `repetition_penalty=1.1` from `generation_config`; the vLLM port matches
  it. Every stack must declare and match sampler settings.
- **Confound:** the vLLM venv has older torch/transformers (2.14/5.20 vs 2.15/5.23), and step-0
  greedy eval, identical for every ptmini run (0.332), reads 0.320 in that venv and 0.305 with the
  vLLM engine live in-process. Control (`launchers/venvctl.sbatch`, HF sampler in the vLLM venv,
  8 seeds): **vLLM vs control −0.001 [−0.019, +0.017]**, so the sampler swap is equivalent within
  ±2 pt. Control vs reference is +0.023 [−0.004, +0.050]: the earlier +2 pt came from the software
  version, not vLLM. T2 must hold the trainer stack fixed or treat its version as a variable.
- **GBS ×2** used 2× the samples per step and did not reach the target faster: on a plateau,
  steps-to-target can't resolve a batch-size effect.

## E8: rung M on MLPerf's recipe (Qwen3.5-35B-A3B, miles, 16 GB300)

MLPerf Training v6.1 `qwen35_397b_grpo` data and hyperparameters, full-parameter GRPO of the 35B sibling:
- data: R2E-Gym easy split, 700 train in curriculum order, 251 val × 4; 16 × 16 per step;
- optimizer: lr 1e-6, clip 0.125, AdamW 0.9/0.999 wd 0;
- sampling: temp 1.0, eval temp 0.1 / top_p 0.95;
- seq-mask TIS [0.999, 1.002]; 64K context; agent capped at 30 turns.

Layout: 2 train nodes and 2 SGLang nodes, with terminus-2 in Modal sandboxes.

| step | 0 | 5 | 10 | 15 | 25 |
|---|---|---|---|---|---|
| pass@4 (MLPerf target 0.69) | 0.673 | 0.705 | 0.709 | 0.701 | 0.749 |
| avg@4 | 0.485 | 0.510 | 0.502 | 0.499 | 0.550 |

- At step 25, +7.6 pt pass@4 and +6.5 pt avg@4, against a ±3 pt eval standard error. Train reward rose 0.46 → 0.75.
- Steps take 11–30 min; a 1,004-rollout eval takes ~22 min.
- The same model, same full-FT stack and similar budget on MiMo SWE tasks stayed flat for 30 steps
  (avg@4 0.24 → 0.26). There, over half of the groups were all-fail. The data, not the stack, decided whether it learned.
- MLPerf's 0.69 target sits only 1.7 pt above this model's base score. That is the plateau-target problem from
  E1 again: on rung M, "time to 0.69" would mostly measure eval noise. The target must be set per rung, on the
  steep part of the curve.
- MLPerf's own reference runs (mlcommons/training `llm_post_training/rcp_logs`, 6 seeds at each of GBS 256 / 512 / 1024 / 1152):
  all 24 stop at the *first* eval the rules allow, step ⌈2.5 + 3840/GBS⌉ = 18 / 10 / 7 / 6, with pass@4 0.70–0.78.
  The 397B is already past 0.69 when evals start, so MLPerf's time-to-train measures throughput over a fixed
  ~4.6K–7K rollouts, not convergence. That is the same plateau-target problem, seen in the reference data.
- Deviations from MLPerf:
  - 35B instead of 397B;
  - terminus-2 instead of OpenHands;
  - synchronous training;
  - grading in the agent's container.

  Code and the full deviation list: PostTrainingX `rl/qwen35_swe/`.

## Not yet shown

- Cross-vendor chips via InferenceX runners.
- s\* transfer at rung M across chips; a rung-M target on the steep part of the curve (E8: MLPerf's 0.69 is too close to the 35B base).
- Rare-collapse-rate equivalence: needs ≫ 16 seeds per arm.
- Rung M/L, and a replay red-team.
