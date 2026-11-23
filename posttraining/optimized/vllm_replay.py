"""Forced-length vLLM replay of recorded ptmini rollout steps (optimized-track t_gen).

Needs its own env with vllm plus this package: `uv pip install vllm -e ..`.

per-request: each request decodes exactly its recorded completion length L_i.
forced-max:  every request decodes the step's max length (what HF batched decode costs).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from ptmini.task import SYSTEM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--reps", type=int, default=2)
    a = p.parse_args()
    ref, out = Path(a.ref), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ref / "meta.json").read_text())["cfg"]
    steps = {s["step"]: s for s in map(json.loads, (ref / "steps.jsonl").open()) if "t_gen" in s}
    trace = torch.load(ref / "trace.pt")
    n = len(trace)
    pick = sorted({round(q * (n - 1) / (a.k - 1)) for q in range(a.k)})

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="train")]

    def encode(q: str) -> list[int]:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False)

    import subprocess

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    t0 = time.perf_counter()
    llm = LLM(model=cfg["model"], dtype="bfloat16", seed=cfg["seed"])
    t_load = time.perf_counter() - t0

    def run(step: int, mode: str) -> float:
        tr = trace[step]
        prompts = [
            TokensPrompt(prompt_token_ids=encode(qs[i]))
            for i in tr["idx"]
            for _ in range(cfg["group"])
        ]
        lens = [len(c) for c in tr["comps"]]
        tgt = lens if mode == "per-request" else [max(lens)] * len(lens)
        sps = [
            SamplingParams(
                max_tokens=L,
                min_tokens=L,
                ignore_eos=True,
                temperature=cfg["temperature"],
                top_p=1.0,
            )
            for L in tgt
        ]
        llm.reset_prefix_cache()
        t = time.perf_counter()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        dt = time.perf_counter() - t
        got = [len(o.outputs[0].token_ids) for o in outs]
        if got != tgt:
            raise RuntimeError(f"step {step} {mode}: lengths not forced")
        return dt

    for mode in ("per-request", "forced-max"):  # warmup (compile/graphs already captured at load)
        run(pick[0], mode)
    rows = []
    for step in pick:
        lens = [len(c) for c in trace[step]["comps"]]
        rec = {
            "step": step,
            "sum_len": sum(lens),
            "max_len": max(lens),
            "hf_t_gen": steps[step]["t_gen"],
        }
        for mode in ("per-request", "forced-max"):
            ts = [run(step, mode) for _ in range(a.reps)]
            rec[mode] = sum(ts) / len(ts)
            rec[mode + "_all"] = ts
        rows.append(rec)
        print(json.dumps(rec), flush=True)

    x = torch.tensor([[1.0, r["sum_len"]] for r in rows], dtype=torch.float64)
    y = torch.tensor([r["per-request"] for r in rows], dtype=torch.float64)
    sol = torch.linalg.lstsq(x, y[:, None]).solution[:, 0]
    r2 = 1 - ((y - x @ sol) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    meta = {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "gpu": gpu,
        "t_load": t_load,
        "picked": pick,
        "fit_a": float(sol[0]),
        "fit_b": float(sol[1]),
        "r2": float(r2),
    }
    (out / "replay.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    lines = [
        f"# vLLM forced-length replay of {ref}",
        "",
        f"vLLM {meta['vllm']}, torch {meta['torch']}, 1x {meta['gpu']}, bf16, defaults otherwise "
        f"(CUDA graphs, prefix cache reset before each timed call). Engine load {t_load:.1f} s. "
        f"Mean of {a.reps} timed reps after 1 warmup step. Script: posttraining/optimized/vllm_replay.py",
        "",
        "| step | sum len | max len | HF t_gen (s) | vLLM per-request (s) | vLLM forced-max (s) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['step']} | {r['sum_len']} | {r['max_len']} | {r['hf_t_gen']:.2f} | "
        f"{r['per-request']:.3f} | {r['forced-max']:.3f} |"
        for r in rows
    ]
    lines += [
        "",
        f"Per-request fit t = {sol[0]:.3f} + {sol[1] * 1e3:.4f}e-3 * sum_len, R^2 = {r2:.3f}",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
