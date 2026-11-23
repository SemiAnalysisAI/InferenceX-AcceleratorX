"""GSM8K prompts and the exact-match reward."""

import re

from datasets import load_dataset

SYSTEM = (
    "Solve the math problem step by step. "
    "End with the final numeric answer on its own line as '#### <number>'."
)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load(split: str, n: int | None = None) -> list[dict[str, str]]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = [
        {"q": r["question"], "a": r["answer"].split("####")[-1].strip().replace(",", "")}
        for r in ds
    ]
    return rows[:n] if n else rows


def reward(text: str, gold: str) -> float:
    """1 if the answer after '####' (else in \\boxed{}, else the last number) equals gold."""
    marked = next((m for m in ("####", "\\boxed{") if m in text), None)
    nums = _NUM.findall(text.rsplit(marked, 1)[-1] if marked else text)
    if not nums:
        return 0.0
    pick = nums[0] if marked else nums[-1]
    return float(abs(float(pick.replace(",", "")) - float(gold)) < 1e-6)
