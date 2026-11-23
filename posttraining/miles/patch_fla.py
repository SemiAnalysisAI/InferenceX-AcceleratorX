"""Idempotent runtime patch to fla (flash-linear-attention 0.5.2) in the miles image.

FusedRMSNormGated, ShortConvolution and l2norm autotune with ``NB`` (number of blocks, which grows
with the packed token count) in the key, so every micro-batch with a new token total re-runs
autotuning (6-11 s per kernel, 6 kernels): a step with unseen lengths took ~226 s instead of ~60 s.
Dropping NB tunes each kernel once per feature size.
"""

import sys
from pathlib import Path

FLA = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/sglang/lib/python3.12/site-packages/fla")

PATCHES = [
    ("modules/fused_norm_gate.py", 'key=["D", "NB", ', 'key=["D", ', 2),
    ("modules/conv/triton/kernels.py", "key=['D', 'W', 'NB']", "key=['D', 'W']", 2),
    # l2norm_{fwd,bwd}_kernel: NB = cdiv(tokens * heads, 65536) ≈ 29-30, still varies per micro-batch
    ("modules/l2norm.py", 'key=["D", "NB"]', 'key=["D"]', 2),
]

for rel, old, new, count in PATCHES:
    path = FLA / rel
    text = path.read_text()
    if old not in text:
        assert text.count(new) >= count, f"patch target not found in {path}: {old!r}"
        continue
    assert text.count(old) == count, (
        f"expected {count} x {old!r} in {path}, found {text.count(old)}"
    )
    path.write_text(text.replace(old, new))
    print(f"patched {path}")
