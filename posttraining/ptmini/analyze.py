"""Rung-S validation tables (DESIGN.md E1-E5) from a runs/ directory.

python -m ptmini.analyze --runs runs --tag s05 --target 0.45 --pair p300 p600 [--chip s05c300]
"""

import argparse
import json
import math
import random
import re
import statistics as st
from dataclasses import asdict
from pathlib import Path

import torch

from ptmini.cli import fit, load_steps
from ptmini.grpo import Cfg


def step_s(s: dict) -> float:
    return s["t_gen"] + s["t_train"] + s.get("t_sync", 0.0)  # t_sync: optimized-stack weight push


def collapsed(accs: list[float]) -> bool:
    """Eval accuracy fell below half the base at any point after training started."""
    return min(accs[1:], default=math.inf) < accs[0] / 2


def steps_to(steps: list[dict], target: float) -> float:
    return next((s["step"] for s in steps if s.get("acc", -1) >= target), math.inf)


def lower_median(xs: list[float]) -> float:
    return sorted(xs)[(len(xs) - 1) // 2]


def wall_to(steps: list[dict], s_star: float) -> float:
    """Timed phases to target: training steps before s* plus every eval at steps <= s*."""
    return sum(step_s(s) for s in steps if "t_gen" in s and s["step"] < s_star) + sum(
        s["t_eval"] for s in steps if "t_eval" in s and s["step"] <= s_star
    )


def read_replay(d: Path) -> tuple[list[dict], float, float]:
    """(step records, cold eval s, warm eval s). Older replays logged one eval: use it for both."""
    rows = list(map(json.loads, (d / "replay.jsonl").open()))
    evals = [r["t_eval"] for r in rows if "eval" in r]
    return [r for r in rows if "step" in r], evals[0], evals[-1]


def warmup_cost(rec: list[dict]) -> float:
    w = next(r for r in rec if r["warmup"])
    steady = next(r for r in rec if not r["warmup"] and r["step"] == w["step"])
    return step_s(w) - step_s(steady)


def project(rec: list[dict], shapes: list[dict], s_star: float) -> float:
    """Steady-state per-phase cost model summed over the reference shapes before s*."""
    g = fit([r["lmax"] for r in rec], [r["t_gen"] for r in rec])
    t = fit([r["n_tok"] for r in rec], [r["t_train"] for r in rec])
    return sum(
        g[0] + g[1] * s["lmax"] + t[0] + t[1] * s["n_tok"] for s in shapes if s["step"] < s_star
    )


def projected_wall(d: Path, shapes: list[dict], s_star: float, ev: int, n: int = 0) -> float:
    """Warmup + steps + cold eval + (evals - 1) warm evals; n > 0 uses only n replayed steps."""
    rec, cold, warm = read_replay(d)
    steady = [r for r in rec if not r["warmup"]][: n or None]
    n_evals = sum(1 for k in range(int(s_star) + 1) if k % ev == 0)
    return warmup_cost(rec) + project(steady, shapes, s_star) + cold + (n_evals - 1) * warm


def replay_cost(d: Path, n: int) -> float:
    rec, cold, warm = read_replay(d)
    steady = [r for r in rec if not r["warmup"]][:n]
    return cold + warm + sum(step_s(r) for r in rec if r["warmup"]) + sum(map(step_s, steady))


def arm_runs(root: Path, arm: str) -> list[list[dict]]:
    """Seed runs named <arm>-full-s<N> (ref.sbatch) or <arm>-s<N> (other launchers)."""
    pat = re.compile(rf"{re.escape(arm)}(-full)?-s\d+")
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and pat.fullmatch(d.name)]
    runs = [(load_steps(d), json.loads((d / "meta.json").read_text())["steps"]) for d in dirs]
    return [ss for ss, n in runs if ss and ss[-1]["step"] == n]  # finished runs only


def window_acc(runs: list[list[dict]], lo: int, hi: int) -> list[float]:
    return [st.mean(s["acc"] for s in ss if "acc" in s and lo <= s["step"] <= hi) for ss in runs]


def welch_ci(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """90% Welch t interval for mean(xs) - mean(ys); t quantile by Cornish-Fisher expansion."""
    vx, vy = st.variance(xs) / len(xs), st.variance(ys) / len(ys)
    df = (vx + vy) ** 2 / (vx**2 / (len(xs) - 1) + vy**2 / (len(ys) - 1))
    z = 1.6449
    t = z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    d, h = st.mean(xs) - st.mean(ys), t * math.sqrt(vx + vy)
    return d - h, d + h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--tag", required=True, help="prefix of the reference seed runs")
    p.add_argument("--target", type=float, required=True)
    p.add_argument("--pair", nargs="*", default=[], help="pair.sbatch prefixes, e.g. p300")
    p.add_argument("--chip", help="chip.sbatch prefix with parity results (default: --tag)")
    p.add_argument("--n-replay", type=int, default=4, help="replayed steps assumed for E5")
    p.add_argument("--arms", nargs="*", default=[], help="E7: run tags, as arm or arm:reference")
    p.add_argument("--window", type=int, nargs=2, default=[5, 30], help="E7 mean-acc step window")
    a = p.parse_args()
    root = Path(a.runs)
    seeds = sorted(
        (d for d in root.glob(f"{a.tag}-full-s*") if d.is_dir()),
        key=lambda d: int(d.name.rsplit("s", 1)[1]),
    )

    def cfg(d: Path) -> dict:  # older runs predate some Cfg fields: fill in defaults
        m = json.loads((d / "meta.json").read_text())
        run = {"steps": m["steps"], "dtype": "bf16", "eval_every": 5, "eval_n": 256}
        run |= {k: m[k] for k in run if k in m}
        return {**asdict(Cfg()), **m["cfg"], **run, "seed": 0}

    same = [d for d in seeds if cfg(d) == cfg(seeds[0])]
    if len(same) < len(seeds):
        print(
            f"(skipping {len(seeds) - len(same)} seed runs whose config differs from the first seed)"
        )
    seeds = same
    runs = {d.name: load_steps(d) for d in seeds}
    ref = runs[f"{a.tag}-full-s0"]
    ev = json.loads((seeds[0] / "meta.json").read_text()).get("eval_every", 5)
    shapes = [s for s in ref if "t_gen" in s]

    print(f"## {a.tag}: target acc >= {a.target}\n\n### E1 seed spread (stock GB300)\n")
    print("| run | best acc | final acc | collapsed | steps-to-target | wall-to-target s |")
    print("|---|---|---|---|---|---|")
    reach = {}
    for name, ss in runs.items():
        accs = [s["acc"] for s in ss if "acc" in s]
        reach[name] = steps_to(ss, a.target)
        w = f"{wall_to(ss, reach[name]):.0f}" if reach[name] < math.inf else "-"
        print(
            f"| {name} | {max(accs):.3f} | {accs[-1]:.3f} | {'yes' if collapsed(accs) else 'no'}"
            f" | {reach[name]} | {w} |"
        )
    s_star = lower_median(list(reach.values()))
    walls = [wall_to(runs[k], n) for k, n in reach.items() if n < math.inf]
    hits = [n for n in reach.values() if n < math.inf]
    cv, cv_steps = st.stdev(walls) / st.mean(walls), st.stdev(hits) / st.mean(hits)
    print(
        f"\n{len(walls)}/{len(reach)} reach target; s* (lower median, misses = inf) = {s_star}; "
        f"steps-to-target CV {cv_steps:.1%}, wall-to-target CV {cv:.1%}; eval cadence {ev} steps"
    )

    if a.pair:
        print("\n### E2/E3 step-time projection over s* steps, paired (same node, same time)\n")
        print(
            "| run | chip | slowdown | n replayed | projected s | measured s | err | MHz full/replay |"
        )
        print("|---|---|---|---|---|---|---|---|")
    for pre in a.pair:
        lock = pre.lstrip("p")
        base_mean = st.mean(
            step_s(x) for x in load_steps(root / f"{pre}-full-s0") if "t_gen" in x and x["step"] > 0
        )
        for chip, full, rep in (
            ("stock", root / f"{pre}-full-s0", root / f"{pre}-replay"),
            (f"{lock} MHz", root / f"{pre}-lock{lock}-full-s0", root / f"{pre}-lock{lock}-replay"),
        ):
            ss = load_steps(full)
            mean = st.mean(step_s(x) for x in ss if "t_gen" in x and x["step"] > 0)
            meas = wall_to(ss, s_star)
            mhz_f = min(x.get("train_mhz", 0) for x in ss if "t_gen" in x)
            mhz_r = min((x.get("train_mhz", 0) for x in read_replay(rep)[0]), default=0)
            for n in (a.n_replay, 0):
                proj = projected_wall(rep, shapes, s_star, ev, n)
                print(
                    f"| {pre} | {chip} | {mean / base_mean:.2f}x | {n or 'all'} | {proj:.0f} | "
                    f"{meas:.0f} | {(proj - meas) / meas:+.1%} | {mhz_f}/{mhz_r} |"
                )

    if a.pair:
        rec = [r for r in read_replay(root / f"{a.pair[0]}-replay")[0] if not r["warmup"]]
        full = load_steps(root / f"{a.pair[0]}-full-s0")
        truth = sum(step_s(s) for s in full if "t_gen" in s and 0 < s["step"] < 20)
        window = [s for s in shapes if 0 < s["step"] < 20]
        rng = random.Random(0)  # noqa: S311
        print(
            "\n### E2 steady step-time error vs replayed steps n (paired stock run, steps 1-19)\n"
        )
        print("| n | median abs err | p90 abs err |\n|---|---|---|")
        for n in (2, 3, 4, 6, 8, len(rec)):
            errs = sorted(
                abs(project(rng.sample(rec, n), window, 20) - truth) / truth for _ in range(50)
            )
            print(f"| {n} | {errs[25]:.1%} | {errs[45]:.1%} |")

    par = root / f"{a.chip or a.tag}-parity" / "parity.pt"
    if par.exists():
        d = torch.load(par)
        f32, ref_err = d["fp32"], (d["bf16"] - d["fp32"]).abs().mean()
        print(f"\n### E4 T0: per-token |logprob - fp32|, ratio to reference bf16 ({ref_err:.4f})\n")
        dirs: dict[float, list[Path]] = {}
        for x in root.glob(f"{a.tag}-noise*-s*"):
            if x.is_dir():
                dirs.setdefault(float(x.name.split("noise")[1].rsplit("-s", 1)[0]), []).append(x)
        band5, band10 = (
            min(s["acc"] for ss in runs.values() for s in ss if s["step"] == k and "acc" in s)
            for k in (5, 10)
        )
        print("| arm | T0 ratio | never reach | collapse | below band @5 | below band @10 |")
        print("|---|---|---|---|---|---|")
        print(f"| fp16 | {float((d['fp16'] - f32).abs().mean() / ref_err):.2f} | | | | |")
        for k in (k for k in d if k.startswith("fp8")):
            dose = [load_steps(x) for x in sorted(root.glob(f"{a.tag}{k}-full-s*")) if x.is_dir()]
            acc = [{s["step"]: s["acc"] for s in ss if "acc" in s} for ss in dose]
            nd = len(dose)
            miss = sum(steps_to(ss, a.target) == math.inf for ss in dose)
            col = sum(collapsed(list(x.values())) for x in acc)
            b5, b10 = sum(x[5] < band5 for x in acc), sum(x[10] < band10 for x in acc)
            r = float((d[k] - f32).abs().mean() / ref_err)
            print(f"| sampler {k} | {r:.1f} | {miss}/{nd} | {col}/{nd} | {b5}/{nd} | {b10}/{nd} |")
        ratios = {
            float(k[5:]): float((d[k] - f32).abs().mean() / ref_err)
            for k in d
            if k.startswith("noise")
        }
        for sig in sorted(set(dirs) | set(ratios)):
            dose = [load_steps(x) for x in sorted(dirs.get(sig, []))]
            acc = [{s["step"]: s["acc"] for s in ss if "acc" in s} for ss in dose]
            r = f"{ratios[sig]:.1f}" if sig in ratios else "-"
            nd = len(dose)
            miss = sum(steps_to(ss, a.target) == math.inf for ss in dose)
            col = sum(collapsed(list(x.values())) for x in acc)
            b5, b10 = sum(x[5] < band5 for x in acc), sum(x[10] < band10 for x in acc)
            print(f"| sigma {sig:g} | {r} | {miss}/{nd} | {col}/{nd} | {b5}/{nd} | {b10}/{nd} |")
        print(f"\nT0 parity chip-s: {d['seconds']:.0f}")

    if a.pair and par.exists():
        t0 = torch.load(par)["seconds"]
        t1 = replay_cost(root / f"{a.pair[0]}-replay", a.n_replay)
        t2 = wall_to(ref, 10)
        one = wall_to(ref, s_star)
        need = math.ceil((cv / 0.10) ** 2)
        print("\n### E5 chip-s to qualify a new chip vs real runs to target\n")
        print("| item | chip-s |\n|---|---|")
        print(
            f"| T0 parity | {t0:.0f} |\n| T1 replay (2 evals, warmup, {a.n_replay} steps) | {t1:.0f} |"
        )
        print(f"| T2 short run (10 steps + evals) | {t2:.0f} |")
        print(f"| **new chip** | **{t0 + t1 + t2:.0f}** |")
        print(f"| one real run to s*={s_star} | {one:.0f} |")
        print(f"| real runs for ±10% mean time-to-target, n=(CV/0.1)²={need} | {need * one:.0f} |")

    if a.arms:
        print(f"\n### E7 other configs and stacks: target {a.target}, and mean eval acc over steps")
        print(
            f"{a.window[0]}-{a.window[1]} vs reference seeds (90% Welch t CI of the difference)\n"
        )
        print(
            "| arm | vs | seeds | reach | collapse | steps-to-target | mean acc | Δ vs ref | s/step |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
    for spec in a.arms:
        arm, _, base = spec.partition(":")
        rs, rb = arm_runs(root, arm), arm_runs(root, base or a.tag)
        ma, mb = (window_acc(x, *a.window) for x in (rs, rb))
        lo, hi = welch_ci(ma, mb)
        to = sorted(steps_to(ss, a.target) for ss in rs)
        col = sum(collapsed([s["acc"] for s in ss if "acc" in s]) for ss in rs)
        sps = st.mean(step_s(s) for ss in rs for s in ss if "t_gen" in s and s["step"] > 0)
        print(
            f"| {arm} | {base or a.tag} | {len(rs)} | {sum(x < math.inf for x in to)}/{len(rs)} |"
            f" {col}/{len(rs)} | {to} | {st.mean(ma):.3f} | {st.mean(ma) - st.mean(mb):+.3f}"
            f" [{lo:+.3f}, {hi:+.3f}] | {sps:.1f} |"
        )


if __name__ == "__main__":
    main()
