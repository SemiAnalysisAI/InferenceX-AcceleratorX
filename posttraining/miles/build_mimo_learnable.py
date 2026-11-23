"""Build a learnable MiMo SWE task set for a model from measured solve rates.

Solve rates come from (a) eval-only scans (`--num-rollout 0`, n attempts per task at temp 1.0; each task's attempts
are contiguous in dump/trajectory/eval_0.jsonl) and (b) trained groups of earlier RL runs (dump/trajectory/<step>.jsonl,
grouped by group_index; these only hold mixed groups, so they can only add tasks inside the band).
Keeps tasks with lo <= rate <= hi, drops tasks whose hidden tests touch R2E-Gym / MLPerf or SWE-bench Verified repos, and splits a
held-out eval set stratified by rate. Train order is a fixed shuffle (seeded), like MLPerf's fixed data order.

python build_mimo_learnable.py --scan $PTX_ROOT/runs/<scan>/dump/trajectory/eval_0.jsonl --scan-n 8 \
    --runs $PTX_ROOT/runs/<run1> $PTX_ROOT/runs/<run2> --out $PTX_ROOT/datasets/mimo_learn
"""

import argparse
import collections
import glob
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path

P = Path(os.environ["PTX_ROOT"])  # datasets/, harbor_tasks/, harbor/, data/


def key(prompt: str | list | dict) -> str:
    return prompt if isinstance(prompt, str) else json.dumps(prompt)


def scan_rates(path: str, n: int) -> dict[str, list[int]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    out = collections.defaultdict(list)
    for r in rows:
        out[key(r["prompt"])].append(1 if (r.get("reward") or 0) > 0 else 0)
    short = {k for k, v in out.items() if len(v) < n}
    if short:
        print(f"warning: {len(short)} scan tasks have < {n} attempts; kept if >= {n // 2}")
    return {k: v for k, v in out.items() if len(v) >= n // 2}


def run_rates(run_dir: str) -> dict[str, list[int]]:
    out = collections.defaultdict(list)
    for f in glob.glob(f"{run_dir}/dump/trajectory/[0-9]*.jsonl"):
        groups, prompts = collections.defaultdict(list), {}
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            groups[r["group_index"]].append(1 if (r.get("reward") or 0) > 0 else 0)
            prompts[r["group_index"]] = key(r["prompt"])
        for g, v in groups.items():
            out[prompts[g]] += v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", nargs="*", default=[])
    ap.add_argument("--scan-n", type=int, default=8)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.7)
    ap.add_argument("--eval-tasks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        required=True,
        help="prefix: writes <out>_train.jsonl, <out>_eval.jsonl, <out>_stats.json",
    )
    a = ap.parse_args()

    rows = {
        key(json.loads(line)["prompt"]): json.loads(line)
        for line in (P / "datasets/mimo_train.jsonl").read_text().splitlines()
    }
    overlap = set(json.loads((P / "datasets/mimo_r2e_overlap_ids.json").read_text()))
    overlap |= set(json.loads((P / "datasets/mimo_swebench_overlap_ids.json").read_text()))
    held = {
        json.loads(line)["metadata"]["instance_id"]
        for line in (P / "datasets/mimo_eval32.jsonl").read_text().splitlines()
    }

    attempts: dict[str, list[int]] = collections.defaultdict(list)
    for path in a.scan:
        for k, v in scan_rates(path, a.scan_n).items():
            attempts[k] += v
    for run in a.runs:
        for k, v in run_rates(run).items():
            attempts[k] += v

    measured = {k: v for k, v in attempts.items() if k in rows}
    rate = {k: sum(v) / len(v) for k, v in measured.items()}
    keep = [
        k
        for k, r in rate.items()
        if a.lo <= r <= a.hi
        and rows[k]["metadata"]["instance_id"] not in overlap
        and rows[k]["metadata"]["instance_id"] not in held
    ]

    rng = random.Random(a.seed)
    keep.sort(key=lambda k: (rate[k], rows[k]["metadata"]["instance_id"]))
    stride = max(len(keep) // max(a.eval_tasks, 1), 1)
    eval_keys = set(
        keep[stride // 2 :: stride][: a.eval_tasks]
    )  # every stride-th by rate: stratified
    train_keys = [k for k in keep if k not in eval_keys]
    rng.shuffle(train_keys)

    def write(path: str, keys: Iterable[str], split: str) -> None:
        with open(path, "w") as f:
            for k in keys:
                r = dict(rows[k])
                r["metadata"] = {**r["metadata"], "split": split, "solve_rate": round(rate[k], 3)}
                f.write(json.dumps(r) + "\n")

    write(f"{a.out}_train.jsonl", train_keys, "mimo_learn_train")
    write(f"{a.out}_eval.jsonl", sorted(eval_keys), "mimo_learn_eval")

    def hist(keys: Iterable[str]) -> dict[float, int]:
        return dict(sorted(collections.Counter(round(rate[k], 1) for k in keys).items()))

    stats = {
        "measured": len(measured),
        "all_fail": sum(1 for r in rate.values() if r == 0),
        "all_pass": sum(1 for r in rate.values() if r == 1),
        "in_band": sum(1 for r in rate.values() if a.lo <= r <= a.hi),
        "dropped_eval_overlap": sum(
            1 for k in rate if rows[k]["metadata"]["instance_id"] in overlap
        ),
        "train": len(train_keys),
        "eval": len(eval_keys),
        "band": [a.lo, a.hi],
        "train_rate_hist": hist(train_keys),
        "eval_rate_hist": hist(eval_keys),
        "eval_mean_rate": round(sum(rate[k] for k in eval_keys) / max(len(eval_keys), 1), 3),
    }
    Path(f"{a.out}_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
