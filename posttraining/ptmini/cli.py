"""ptmini train | replay | parity. Tables: python -m ptmini.analyze. See DESIGN.md."""

import argparse
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn.functional as F

from ptmini import task
from ptmini.grpo import (
    Cfg,
    Policy,
    fp8_weights,
    group_advantages,
    pick_device,
    telemetry,
    timed,
)


def device_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    if dev.type == "mps":
        cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return f"{dev.type}:{platform.processor()}"


def cfg_from(a: argparse.Namespace) -> Cfg:
    return Cfg(
        model=a.model,
        prompts_per_step=a.prompts_per_step,
        group=a.group,
        max_new=a.max_new,
        micro_bs=a.micro_bs,
        seed=a.seed,
        logit_noise=a.logit_noise,
        sampler_fp8=a.sampler_fp8,
        lr=a.lr,
    )


def ref_cfg(meta: dict, **override: object) -> Cfg:
    """Cfg of a recorded run. Fields Cfg no longer has (lora_r, from the LoRA pilot) are dropped."""
    known = {f.name for f in fields(Cfg)}
    return Cfg(**{k: v for k, v in {**meta["cfg"], **override}.items() if k in known})


def write_meta(out: Path, dev: torch.device, cfg: Cfg, **extra: object) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = {"device": device_name(dev), "torch": torch.__version__, "cfg": asdict(cfg), **extra}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def evaluate(pol: Policy, rows: list[dict[str, str]], bs: int = 64) -> float:
    hits = 0.0
    for k in range(0, len(rows), bs):
        chunk = rows[k : k + bs]
        comps = pol.generate([pol.encode(r["q"]) for r in chunk], pol.cfg.max_new, greedy=True)
        hits += sum(task.reward(pol.decode(c), r["a"]) for c, r in zip(comps, chunk, strict=True))
    return hits / len(rows)


def cmd_train(a: argparse.Namespace) -> None:
    dev, cfg, out = pick_device(a.device), cfg_from(a), Path(a.out)
    write_meta(
        out, dev, cfg, steps=a.steps, dtype=a.dtype, eval_every=a.eval_every, eval_n=a.eval_n
    )
    pol = Policy(cfg, dev, dtype=getattr(torch, {"bf16": "bfloat16", "fp16": "float16"}[a.dtype]))
    train_rows, test_rows = task.load("train"), task.load("test", a.eval_n)
    rng = random.Random(cfg.seed)  # noqa: S311  (prompt sampling, not crypto)
    trace, log = [], (out / "steps.jsonl").open("w")
    for step in range(a.steps + 1):
        rec: dict[str, object] = {"step": step}
        if step % a.eval_every == 0 or step == a.steps:
            rec["acc"], rec["t_eval"] = timed(dev, evaluate, pol, test_rows)
        if step < a.steps:
            idx = rng.sample(range(len(train_rows)), cfg.prompts_per_step)
            prompts = [pol.encode(train_rows[i]["q"]) for i in idx for _ in range(cfg.group)]
            golds = [train_rows[i]["a"] for i in idx for _ in range(cfg.group)]
            comps, rec["t_gen"] = timed(dev, pol.generate, prompts, cfg.max_new)
            rewards = [task.reward(pol.decode(c), g) for c, g in zip(comps, golds, strict=True)]
            adv = group_advantages(rewards, cfg.group)
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
            trace.append({"idx": idx, "comps": comps, "adv": adv})
            if step % 10 == 9:
                torch.save(trace, out / "trace.pt")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
    torch.save(trace, out / "trace.pt")


def load_steps(ref: Path) -> list[dict]:
    return [json.loads(line) for line in (ref / "steps.jsonl").open()]


def cmd_replay(a: argparse.Namespace) -> None:
    """Re-run k reference steps with identical shapes: same prompts, decode forced to the step's
    longest completion (HF batches decode until the longest row ends), same training tokens."""
    dev, ref, out = pick_device(a.device), Path(a.ref), Path(a.out)
    meta = json.loads((ref / "meta.json").read_text())
    cfg = ref_cfg(meta, **({"micro_bs": a.micro_bs} if a.micro_bs else {}))
    dtype = getattr(torch, {"bf16": "bfloat16", "fp16": "float16"}[meta.get("dtype", "bf16")])
    steps = [s for s in load_steps(ref) if "t_gen" in s]
    trace = torch.load(ref / "trace.pt")
    by_len = sorted(steps, key=lambda s: s["lmax"])
    pick = [by_len[round(q * (len(by_len) - 1) / (a.k - 1))]["step"] for q in range(a.k)]
    write_meta(out, dev, cfg, ref=str(ref), picked=pick)
    pol = Policy(cfg, dev, dtype=dtype)
    rows = task.load("train")
    log = (out / "replay.jsonl").open("w")
    # Eval first, as in a real run: the cold one pays CUDA/allocator init, the warm one is the
    # per-eval cost. Both are counted in time-to-target.
    test_rows = task.load("test", meta.get("eval_n", 256))
    for kind in ("cold", "warm"):
        _, t_eval = timed(dev, evaluate, pol, test_rows)
        log.write(json.dumps({"eval": kind, "t_eval": t_eval, **telemetry(dev)}) + "\n")
    for n, step in enumerate([pick[0]] * a.warmup + pick):
        tr, s = trace[step], steps[step]
        prompts = [pol.encode(rows[i]["q"]) for i in tr["idx"] for _ in range(cfg.group)]
        _, t_gen = timed(dev, pol.generate, prompts, s["lmax"], force_len=True)
        gen_tel = telemetry(dev)
        _, t_train = timed(dev, pol.train, prompts, tr["comps"], tr["adv"])
        rec = {
            "step": step,
            "warmup": n < a.warmup,
            "lmax": s["lmax"],
            "n_tok": s["n_tok"],
            "t_gen": t_gen,
            "t_train": t_train,
        }
        rec |= {f"gen_{k}": v for k, v in gen_tel.items()} | {
            f"train_{k}": v for k, v in telemetry(dev).items()
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)


def cmd_parity(a: argparse.Namespace) -> None:
    """Per-token logprobs of n fixed reference sequences in bf16 and fp32 on this device."""
    dev, ref, out = pick_device(a.device), Path(a.ref), Path(a.out)
    cfg = ref_cfg(json.loads((ref / "meta.json").read_text()))
    tr, rows = torch.load(ref / "trace.pt")[0], task.load("train")
    write_meta(out, dev, cfg, ref=str(ref))
    t0 = time.perf_counter()
    res: dict[str, object] = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        pol = Policy(cfg, dev, dtype=dtype)
        pol.model.eval()
        prompts = [pol.encode(rows[i]["q"]) for i in tr["idx"] for _ in range(cfg.group)][: a.n]
        lps = []
        with torch.no_grad():
            for p, c in zip(prompts, tr["comps"][: a.n], strict=True):
                ids, attn, mask = pol._batch([p + c], [len(p)])  # noqa: SLF001
                lps.append(pol.logprobs(ids, attn)[mask.bool()].cpu())
        res[name] = torch.cat(lps)
        del pol
    for sigma in a.noise:
        g = torch.Generator().manual_seed(
            0
        )  # sampler-side logits noise on the bf16 path, same draw per token
        pol = Policy(cfg, dev, dtype=torch.bfloat16)
        pol.model.eval()
        lps = []
        with torch.no_grad():
            for p, c in zip(prompts, tr["comps"][: a.n], strict=True):
                ids, attn, mask = pol._batch([p + c], [len(p)])  # noqa: SLF001
                logits = pol.model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
                logits += sigma * torch.randn(logits.shape, generator=g).to(dev)
                lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
                lps.append(F.pad(lp, (1, 0))[mask.bool()].cpu())
        res[f"noise{sigma}"] = torch.cat(lps)
        del pol
    for fmt in a.fp8:  # sampler-side rounded weights on the bf16 path
        pol = Policy(cfg, dev, dtype=torch.bfloat16)
        pol.model.eval()
        lps = []
        with torch.no_grad(), fp8_weights(pol.model, fmt):
            for p, c in zip(prompts, tr["comps"][: a.n], strict=True):
                ids, attn, mask = pol._batch([p + c], [len(p)])  # noqa: SLF001
                lps.append(pol.logprobs(ids, attn)[mask.bool()].cpu())
        res[f"fp8{fmt}"] = torch.cat(lps)
        del pol
    res["seconds"] = time.perf_counter() - t0
    torch.save(res, out / "parity.pt")


def fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    x = torch.tensor([[1.0, v] for v in xs], dtype=torch.float64)
    sol = torch.linalg.lstsq(x, torch.tensor(ys, dtype=torch.float64)[:, None]).solution
    return float(sol[0]), float(sol[1])


def steps_to(steps: list[dict], target: float) -> int | None:
    return next((s["step"] for s in steps if s.get("acc", -1) >= target), None)


def main() -> None:
    p = argparse.ArgumentParser(prog="ptmini")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "replay", "parity"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--device")
        s.add_argument("--micro-bs", type=int, default=4 if name == "train" else None)
    tr = sub.choices["train"]
    tr.add_argument("--steps", type=int, default=60)
    tr.add_argument("--eval-every", type=int, default=5)
    tr.add_argument("--eval-n", type=int, default=256)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--prompts-per-step", type=int, default=16)
    tr.add_argument("--group", type=int, default=8)
    tr.add_argument("--max-new", type=int, default=384)
    tr.add_argument("--model", default=Cfg.model)
    tr.add_argument("--logit-noise", type=float, default=0.0)
    tr.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    tr.add_argument("--sampler-fp8", choices=["", "e4m3", "e5m2", "int4"], default="")
    tr.add_argument("--lr", type=float, default=Cfg.lr)
    for name in ("replay", "parity"):
        sub.choices[name].add_argument("--ref", required=True)
    sub.choices["replay"].add_argument("--k", type=int, default=6)
    sub.choices["replay"].add_argument("--warmup", type=int, default=1)
    sub.choices["parity"].add_argument("--n", type=int, default=32)
    sub.choices["parity"].add_argument(
        "--noise", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0]
    )
    sub.choices["parity"].add_argument("--fp8", nargs="*", default=["e4m3", "e5m2", "int4"])
    a = p.parse_args()
    {"train": cmd_train, "replay": cmd_replay, "parity": cmd_parity}[a.cmd](a)


if __name__ == "__main__":
    main()
