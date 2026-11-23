"""Harbor tasks + miles JSONL from the code subset of XiaomiMiMo/MiMo-V2.6-RL-oss.

  hf download --repo-type dataset XiaomiMiMo/MiMo-V2.6-RL-oss code.parquet --local-dir $PTX_ROOT/data/mimo
  python3 prep_mimo.py --parquet $PTX_ROOT/data/mimo/code.parquet

Reward follows mimoagent's opensource-code environment: reset every path the hidden test_patch touches to the
base commit (HEAD at image build, kept in /.ptx_base_ref), apply the patch, run test_command, 1 iff it exits 0.
Also writes the ids of tasks whose tests touch an R2E-Gym or SWE-bench Verified repo, so learnable sets built
from MiMo can exclude anything that overlaps those evals, and each task's dominant test-file language.
"""

import argparse
import collections
import json
import os
import random
import re
import shlex
from pathlib import Path

import pyarrow.parquet as pq

P = Path(os.environ["PTX_ROOT"])  # datasets/, harbor_tasks/, data/
IMAGE = "docker.io/xiaomimimo/mimo-v2.6-rl-oss:{tag}"
INSTRUCTION = (
    "The repository is at {cwd}. Resolve the issue below by changing the source code. "
    "Hidden tests will check your fix.\n\n{problem}\n"
)
# terminus-2 needs tmux >= 3.0 (new-session -e); the images lack it and older distros only package 2.x.
DOCKERFILE = """FROM alpine:3.20 AS tmux
RUN apk add --no-cache build-base bison curl libevent-dev libevent-static ncurses-dev ncurses-static && \\
    curl -fsSL https://github.com/tmux/tmux/releases/download/3.5a/tmux-3.5a.tar.gz | tar xz && \\
    cd tmux-3.5a && ./configure --enable-static --prefix=/usr/local && make -j"$(nproc)" && make install

FROM {image}
COPY --from=tmux /usr/local/bin/tmux /usr/local/bin/tmux
WORKDIR {cwd}
RUN /usr/local/bin/tmux new-session -d -s check -e CHECK=1 true
RUN git config --global --add safe.directory '*' && \\
    (git rev-parse --git-dir >/dev/null 2>&1 || \\
     (git init -q && git add -A && git -c user.name=ptx -c user.email=ptx@localhost commit -qm baseline --allow-empty)) && \\
    git rev-parse HEAD > /.ptx_base_ref
"""
TEST_SH = """#!/bin/bash
cd {cwd}
git() {{ command git -c core.hooksPath=/dev/null "$@"; }}
base=$(cat /.ptx_base_ref)
while IFS= read -r tf; do
  [ -z "$tf" ] && continue
  if git cat-file -e "$base:$tf" 2>/dev/null; then git checkout "$base" -- "$tf" 2>/dev/null || true
  else git rm -f --cached "$tf" >/dev/null 2>&1 || true; rm -f "$tf"; fi
done < /tests/touched.txt
if ! git apply --verbose /tests/test.patch; then echo "test patch did not apply"; echo 0 > /logs/verifier/reward.txt; exit 0; fi
timeout {timeout} bash -c {command}
rc=$?
echo "test_command rc=$rc"
[ $rc -eq 0 ] && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
"""
TASK_TOML = """version = "1.0"

[verifier]
timeout_sec = {timeout}.0

[agent]
timeout_sec = 1800.0

[environment]
build_timeout_sec = 1800.0
"""
# Test-path signatures of the repos behind each eval; MiMo ships no repo field, only the test patch.
R2E_REPOS = {
    "pillow": r"(^|/)(Tests/test_|src/PIL/|PIL/)",
    "numpy": r"(^|/)numpy/",
    "pandas": r"(^|/)pandas/",
    "tornado": r"(^|/)tornado/",
    "aiohttp": r"(^|/)(aiohttp/|tests/test_(client|web|http))",
    "scrapy": r"(^|/)scrapy/",
    "pyramid": r"(^|/)(pyramid/|src/pyramid/)",
}
SWEBENCH_REPOS = {
    "astropy": r"(^|/)astropy/",
    "django": r"(^|/)django/",
    "matplotlib": r"(^|/)(lib/)?(matplotlib|mpl_toolkits)/",
    "seaborn": r"(^|/)seaborn/",
    "flask": r"(^|/)(src/)?flask/",
    "requests": r"(^|/)(requests/|tests/test_requests\.py)",
    "xarray": r"(^|/)xarray/",
    "pylint": r"(^|/)pylint/",
    "pytest": r"(^|/)src/_pytest/",
    "scikit-learn": r"(^|/)sklearn/",
    "sphinx": r"(^|/)sphinx/",
    "sympy": r"(^|/)sympy/",
}


def touched(patch: str) -> list[str]:
    paths = re.findall(r"^diff --git a/(\S+) b/(\S+)$", patch, re.MULTILINE)
    return sorted({p for pair in paths for p in pair})


def hits(paths: list[str], sigs: dict[str, str]) -> list[str]:
    return [k for k, rx in sigs.items() if any(re.search(rx, p) for p in paths)]


def language(paths: list[str]) -> str:
    exts = collections.Counter(p.rsplit(".", 1)[-1] for p in paths if "." in p.rsplit("/", 1)[-1])
    return exts.most_common(1)[0][0] if exts else "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--agent-name", default="terminus-2")
    ap.add_argument(
        "--eval-size", type=int, default=32, help="held-out tasks for mimo_eval<N>.jsonl"
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows, r2e, swebench, lang = [], {}, {}, {}
    table = pq.read_table(a.parquet, columns=["extra_info"]).column("extra_info").to_pylist()
    for extra in table:
        inst = json.loads(extra["instance_json"])
        name, cwd = inst["instance_id"], inst["cwd"]
        timeout = int(inst["verifier_timeout_sec"])
        image = IMAGE.format(tag=inst["docker_image"].split(":")[0])
        paths = touched(inst["test_patch"])
        d = P / "harbor_tasks" / name
        (d / "environment").mkdir(parents=True, exist_ok=True)
        (d / "tests").mkdir(exist_ok=True)
        instruction = INSTRUCTION.format(cwd=cwd, problem=inst["problem_statement"])
        (d / "instruction.md").write_text(instruction)
        (d / "task.toml").write_text(TASK_TOML.format(timeout=timeout))
        (d / "environment" / "Dockerfile").write_text(DOCKERFILE.format(image=image, cwd=cwd))
        (d / "tests" / "test.patch").write_text(inst["test_patch"])
        (d / "tests" / "touched.txt").write_text("\n".join(paths) + "\n")
        test_sh = d / "tests" / "test.sh"
        command = shlex.quote(inst["test_command"])
        test_sh.write_text(TEST_SH.format(cwd=cwd, timeout=timeout, command=command))
        test_sh.chmod(0o755)
        rows.append(
            {"prompt": instruction, "metadata": {"instance_id": name, "agent_name": a.agent_name}}
        )
        if h := hits(paths, R2E_REPOS):
            r2e[name] = h
        if h := hits(paths, SWEBENCH_REPOS):
            swebench[name] = h
        lang[name] = language(paths)

    random.Random(a.seed).shuffle(rows)
    data = P / "datasets"
    data.mkdir(parents=True, exist_ok=True)
    splits = ((f"mimo_eval{a.eval_size}", rows[: a.eval_size]), ("mimo_train", rows[a.eval_size :]))
    for split, part in splits:
        with (data / f"{split}.jsonl").open("w") as f:
            for row in part:
                f.write(json.dumps({**row, "metadata": {**row["metadata"], "split": split}}) + "\n")
        print(f"{split}: {len(part)} tasks")
    (data / "mimo_r2e_overlap_ids.json").write_text(json.dumps(sorted(r2e)))
    (data / "mimo_swebench_overlap_ids.json").write_text(json.dumps(sorted(swebench)))
    (data / "mimo_lang.json").write_text(json.dumps(lang))
    for label, found in (("R2E-Gym", r2e), ("SWE-bench Verified", swebench)):
        per_repo = collections.Counter(r for h in found.values() for r in h)
        print(f"{label} overlap: {len(found)} tasks {dict(per_repo)}")
    print("dominant test language:", collections.Counter(lang.values()).most_common(8))


if __name__ == "__main__":
    main()
