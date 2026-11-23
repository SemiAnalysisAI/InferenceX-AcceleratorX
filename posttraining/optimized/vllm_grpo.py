"""ptmini train with only the sampler swapped: rollouts from a colocated vLLM engine.

Same prompts (random.Random(seed)), reward, group advantages, full-parameter HF trainer
(Policy.train) and greedy HF eval as `ptmini train`. Each step the trainer's weights are loaded
into the in-process vLLM model (t_sync). Tests whether reference s* transfers to a vLLM-sampled
stack (DESIGN.md).

Env: the ptmini-vllm venv, with PYTHONPATH pointing at posttraining/.
"""

import os

# The engine must live in this process so collective_rpc can hand it the trainer's tensors.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


import argparse
import json
import random
from pathlib import Path

import torch
import transformers
import vllm
from vllm import LLM, SamplingParams, TokensPrompt

from ptmini import task
from ptmini.cli import evaluate, write_meta
from ptmini.grpo import Cfg, Policy, group_advantages, pick_device, telemetry, timed


def lp_gap(pol: Policy, prompts: list, comps: list, vlps: list) -> dict[str, float]:
    """Trainer-vs-sampler logprob gap on the sampled tokens (both raw, before penalty/temp)."""
    pol.model.eval()
    diffs = []
    with torch.no_grad():
        for k in range(0, len(comps), 16):
            sl = slice(k, k + 16)
            seqs = [p + c for p, c in zip(prompts[sl], comps[sl], strict=True)]
            ids, attn, mask = pol._batch(seqs, [len(p) for p in prompts[sl]])  # noqa: SLF001
            hf = pol.logprobs(ids, attn)[mask.bool()].cpu()
            diffs.append(hf - torch.tensor([x for v in vlps[sl] for x in v]))
    d = torch.cat(diffs)
    return {"lp_mad": d.abs().mean().item(), "lp_bias": d.mean().item(),
            "lp_p99": d.abs().quantile(0.99).item(), "lp_gt01": (d.abs() > 0.1).float().mean().item()}  # fmt: skip


def push_weights(llm: LLM, pol: Policy) -> None:
    weights = list(pol.model.state_dict().items())
    llm.collective_rpc(lambda worker: worker.model_runner.model.load_weights(weights))
    llm.reset_prefix_cache()  # cached KV was computed with the old weights


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--micro-bs", type=int, default=16)  # s05 reference runs used 16
    p.add_argument("--gpu-mem", type=float, default=0.25)
    p.add_argument("--lp-every", type=int, default=10)  # logprob-gap probe cadence (0 = off)
    a = p.parse_args()
    cfg, out = Cfg(seed=a.seed, micro_bs=a.micro_bs), Path(a.out)
    llm = LLM(
        model=cfg.model,
        dtype="bfloat16",
        seed=cfg.seed,
        max_model_len=2048,
        gpu_memory_utilization=a.gpu_mem,
    )
    dev = pick_device(None)
    pol = Policy(cfg, dev)
    # HF generate merges the model's generation_config: Qwen2.5-Instruct sets repetition_penalty
    # (1.1), which the reference therefore applies. temperature/top_k/top_p are overridden.
    rep = pol.model.generation_config.repetition_penalty or 1.0
    sp = SamplingParams(
        temperature=cfg.temperature,
        top_p=1.0,
        top_k=0,
        repetition_penalty=rep,
        max_tokens=cfg.max_new,
        logprobs=0,
    )
    write_meta(
        out, dev, cfg, steps=a.steps, dtype="bf16", eval_every=a.eval_every, eval_n=a.eval_n,
        sampler="vllm", vllm=vllm.__version__, transformers=transformers.__version__,
        gpu_mem=a.gpu_mem, sampling=repr(sp),
    )  # fmt: skip
    train_rows, test_rows = task.load("train"), task.load("test", a.eval_n)
    rng = random.Random(cfg.seed)  # noqa: S311  (prompt sampling, not crypto)
    log = (out / "steps.jsonl").open("w")
    for step in range(a.steps + 1):
        rec: dict[str, object] = {"step": step}
        if step % a.eval_every == 0 or step == a.steps:
            rec["acc"], rec["t_eval"] = timed(dev, evaluate, pol, test_rows)
        if step < a.steps:
            idx = rng.sample(range(len(train_rows)), cfg.prompts_per_step)
            prompts = [pol.encode(train_rows[i]["q"]) for i in idx for _ in range(cfg.group)]
            golds = [train_rows[i]["a"] for i in idx for _ in range(cfg.group)]
            _, rec["t_sync"] = timed(dev, push_weights, llm, pol)
            tp = [TokensPrompt(prompt_token_ids=q) for q in prompts]
            outs, rec["t_gen"] = timed(dev, llm.generate, tp, sp, use_tqdm=False)
            gen = [o.outputs[0] for o in outs]
            comps = [list(g.token_ids) for g in gen]
            bad = [g for g in gen if g.finish_reason == "stop" and g.token_ids[-1] not in pol.eos]
            if bad:
                raise RuntimeError(f"stop without EOS token: {bad[0]}")
            rewards = [task.reward(pol.decode(c), g) for c, g in zip(comps, golds, strict=True)]
            adv = group_advantages(rewards, cfg.group)
            if a.lp_every and step % a.lp_every == 0:
                vlps = [[d[t].logprob for d, t in zip(g.logprobs, g.token_ids, strict=True)] for g in gen]  # fmt: skip
                rec |= lp_gap(pol, prompts, comps, vlps)
            gen_tel = telemetry(dev)
            rec["loss"], rec["t_train"] = timed(dev, pol.train, prompts, comps, adv)
            rec |= {f"gen_{k}": v for k, v in gen_tel.items()} | {
                f"train_{k}": v for k, v in telemetry(dev).items()
            }
            rec |= {
                "reward": sum(rewards) / len(rewards),
                "lmax": max(map(len, comps)),
                "n_tok": sum(map(len, comps)) + sum(map(len, prompts)),
                "trunc": sum(len(c) == cfg.max_new for c in comps) / len(comps),
            }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
