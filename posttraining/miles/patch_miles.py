"""Idempotent runtime patch to /root/miles; train.sbatch runs it on every node before `ray start`.

An agentic sample that aborts before its first model call (the sandbox image failed to build) never
gets a reward. Training drops it via check_no_aborted, but eval keeps it and
_compute_zero_std_metrics crashes on round(None); treat a missing reward as 0 there, as
log_eval_rollout_data does.
"""

import sys
from pathlib import Path

MILES = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/miles")

PATCHES = [
    (
        "miles/ray/rollout/metrics.py",
        "interesting_rewards = [str(round(reward, 1)) for reward in uniform_rewards]",
        "interesting_rewards = [str(round(reward or 0.0, 1)) for reward in uniform_rewards]",
    ),
]

for rel, old, new in PATCHES:
    path = MILES / rel
    text = path.read_text()
    if new in text:
        continue
    assert old in text, f"patch target not found in {path}: {old!r}"
    path.write_text(text.replace(old, new, 1))
    print(f"patched {path}")
