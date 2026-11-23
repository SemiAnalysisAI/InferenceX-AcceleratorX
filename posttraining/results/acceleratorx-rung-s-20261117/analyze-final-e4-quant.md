### E4 T0: per-token |logprob - fp32|, ratio to reference bf16 (0.0292)

| arm | T0 ratio | never reach | collapse | below band @5 | below band @10 |
|---|---|---|---|---|---|
| fp16 | 0.13 | | | | |
| sampler fp8e4m3 | 3.8 | 0/4 | 0/4 | 0/4 | 0/4 |
| sampler fp8e5m2 | 6.8 | 0/4 | 1/4 | 0/4 | 0/4 |
| sampler fp8int4 | 37.0 | 4/4 | 3/4 | 2/4 | 3/4 |
| sigma 0.25 | 4.4 | 0/2 | 0/2 | 0/2 | 0/2 |
| sigma 0.5 | 8.7 | 0/2 | 0/2 | 0/2 | 0/2 |
| sigma 1 | 18.6 | 0/4 | 1/4 | 1/4 | 1/4 |
| sigma 2 | 45.8 | 3/4 | 2/4 | 2/4 | 2/4 |
| sigma 4 | - | 3/4 | 3/4 | 3/4 | 3/4 |

T0 parity chip-s: 304
