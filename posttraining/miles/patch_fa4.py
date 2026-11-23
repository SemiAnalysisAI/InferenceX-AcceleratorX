"""Idempotent runtime patch to FlashAttention-4 (flash_attn.cute) in the miles image.

Transformer Engine calls FA4 with causal=True and window (-1, 0). `_resolve_causal_local_window`
only clears the window when both sides are negative, so (-1, 0) becomes a local window and the SM100
head_dim-256 backward (Qwen3.5 attention layers) asserts "does not support local attention". A
negative side means unbounded in flash-attn's convention: clear each side on its own.
"""

import sys
from pathlib import Path

PATH = (
    Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/opt/sglang/lib/python3.12/site-packages/flash_attn/cute"
    )
    / "interface.py"
)
OLD = """    if (window_size_left is not None and window_size_right is not None
            and window_size_left < 0 and window_size_right < 0):
        window_size_left = None
        window_size_right = None
"""
NEW = """    # ptx patch_fa4.py: a negative side is unbounded on its own, so TE's causal (-1, 0) stays causal
    if window_size_left is not None and window_size_left < 0:
        window_size_left = None
    if window_size_right is not None and window_size_right < 0:
        window_size_right = None
"""

text = PATH.read_text()
if "ptx patch_fa4.py" not in text:
    assert text.count(OLD) == 1, f"patch target not found in {PATH}"
    PATH.write_text(text.replace(OLD, NEW))
    print(f"patched {PATH}")
