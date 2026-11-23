## s05: target acc >= 0.45

### E1 seed spread (stock GB300)

| run | best acc | final acc | collapsed | steps-to-target | wall-to-target s |
|---|---|---|---|---|---|
| s05-full-s0 | 0.516 | 0.504 | no | 10 | 524 |
| s05-full-s1 | 0.484 | 0.012 | yes | 5 | 327 |
| s05-full-s2 | 0.457 | 0.449 | no | 10 | 525 |
| s05-full-s3 | 0.477 | 0.434 | no | 10 | 515 |
| s05-full-s4 | 0.480 | 0.457 | no | 10 | 510 |
| s05-full-s5 | 0.465 | 0.391 | no | 5 | 325 |
| s05-full-s6 | 0.480 | 0.430 | yes | 10 | 521 |
| s05-full-s7 | 0.473 | 0.422 | no | 15 | 722 |
| s05-full-s8 | 0.523 | 0.523 | no | 15 | 1153 |
| s05-full-s9 | 0.500 | 0.008 | yes | 10 | 977 |
| s05-full-s10 | 0.492 | 0.449 | no | 25 | 1423 |
| s05-full-s11 | 0.484 | 0.484 | no | 25 | 1522 |
| s05-full-s12 | 0.480 | 0.461 | no | 10 | 828 |
| s05-full-s13 | 0.516 | 0.500 | no | 10 | 840 |
| s05-full-s14 | 0.527 | 0.453 | no | 5 | 629 |
| s05-full-s15 | 0.461 | 0.168 | yes | 10 | 825 |

16/16 reach target; s* (lower median, misses = inf) = 10; steps-to-target CV 51.7%, wall-to-target CV 47.1%; eval cadence 5 steps

### E2/E3 step-time projection over s* steps, paired (same node, same time)

| run | chip | slowdown | n replayed | projected s | measured s | err | MHz full/replay |
|---|---|---|---|---|---|---|---|
| p300 | stock | 1.00x | 4 | 622 | 610 | +2.0% | 2070/2070 |
| p300 | stock | 1.00x | all | 593 | 610 | -2.7% | 2070/2070 |
| p300 | 300 MHz | 1.29x | 4 | 699 | 668 | +4.6% | 300/300 |
| p300 | 300 MHz | 1.29x | all | 674 | 668 | +0.9% | 300/300 |
| p600 | stock | 1.00x | 4 | 604 | 609 | -0.8% | 2070/2070 |
| p600 | stock | 1.00x | all | 584 | 609 | -4.1% | 2070/2070 |
| p600 | 600 MHz | 1.10x | 4 | 643 | 642 | +0.0% | 600/600 |
| p600 | 600 MHz | 1.10x | all | 622 | 642 | -3.1% | 600/600 |

### E2 steady step-time error vs replayed steps n (paired stock run, steps 1-19)

| n | median abs err | p90 abs err |
|---|---|---|
| 2 | 7.2% | 21.0% |
| 3 | 7.9% | 14.6% |
| 4 | 6.5% | 10.1% |
| 6 | 5.3% | 7.8% |
| 8 | 1.3% | 7.9% |
| 12 | 2.3% | 2.3% |

### E4 T0: per-token |logprob - fp32|, ratio to reference bf16 (0.0292)

| arm | T0 ratio | never reach | collapse | below band @5 | below band @10 |
|---|---|---|---|---|---|
| fp16 | 0.13 | | | | |
| sigma 0.25 | 4.4 | 0/2 | 0/2 | 0/2 | 0/2 |
| sigma 0.5 | 8.8 | 0/2 | 0/2 | 0/2 | 0/2 |
| sigma 1 | 18.7 | 0/4 | 1/4 | 1/4 | 1/4 |
| sigma 2 | 44.8 | 3/4 | 2/4 | 2/4 | 2/4 |
| sigma 4 | - | 3/4 | 3/4 | 3/4 | 3/4 |

T0 parity chip-s: 87

### E5 chip-s to qualify a new chip vs real runs to target

| item | chip-s |
|---|---|
| T0 parity | 87 |
| T1 replay (2 evals, warmup, 4 steps) | 389 |
| T2 short run (10 steps + evals) | 524 |
| **new chip** | **1000** |
| one real run to s*=10 | 524 |
| real runs for ±10% mean time-to-target, n=(CV/0.1)²=23 | 12048 |

### E7 other configs and stacks: target 0.45, and mean eval acc over steps
5-30 vs reference seeds (90% Welch t CI of the difference)

| arm | vs | seeds | reach | collapse | steps-to-target | mean acc | Δ vs ref | s/step |
|---|---|---|---|---|---|---|---|---|
| s05m4 | s05 | 16 | 16/16 | 0/16 | [5, 5, 10, 10, 10, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20, 25] | 0.447 | +0.018 [-0.010, +0.045] | 34.7 |
| s05fp8e4m3 | s05m4 | 4 | 4/4 | 0/4 | [5, 15, 15, 45] | 0.438 | -0.009 [-0.028, +0.009] | 37.4 |
| s05fp8e5m2 | s05m4 | 4 | 4/4 | 1/4 | [5, 10, 10, 20] | 0.436 | -0.011 [-0.049, +0.027] | 34.2 |
| s05fp8int4 | s05m4 | 4 | 0/4 | 3/4 | [inf, inf, inf, inf] | 0.243 | -0.204 [-0.329, -0.079] | 40.9 |
| s05g2 | s05m4 | 4 | 3/4 | 0/4 | [15, 20, 20, inf] | 0.442 | -0.005 [-0.023, +0.012] | 58.1 |
| venvctl | s05 | 8 | 8/8 | 0/8 | [5, 5, 5, 5, 10, 10, 20, 25] | 0.452 | +0.023 [-0.004, +0.050] | 29.7 |
| vllm | s05 | 8 | 8/8 | 0/8 | [5, 5, 5, 10, 10, 15, 35, 55] | 0.451 | +0.022 [-0.008, +0.051] | 8.0 |
| vllm | venvctl | 8 | 8/8 | 0/8 | [5, 5, 5, 10, 10, 15, 35, 55] | 0.451 | -0.001 [-0.019, +0.017] | 8.0 |
