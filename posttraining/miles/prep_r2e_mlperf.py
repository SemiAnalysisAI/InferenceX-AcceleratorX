"""Harbor tasks for MLPerf Training v6.1's post-training data: R2E-Gym easy, 700 train + 251 val.

  python prep_r2e_mlperf.py extract   # copy /r2e_tests + run_tests.sh + HEAD out of each image (Modal)
  python prep_r2e_mlperf.py build     # task dirs + r2e_mlperf_{train,val}.jsonl

Run on the cluster with the Harbor venv (pyarrow, jinja2, modal). MLPERF holds MLPerf's
create_r2e_gym_easy_subset_jsonl.py (NeMo-RL f987c05) and the two id lists; DATA is HF
R2E-Gym/R2E-Gym-Subset@e8b9fcb. The images' hidden tests are more than the dataset's test_file_codes
(conftest, helpers, fixtures, tornado's runner), so they are copied from the images, not rebuilt,
and shipped as tests/r2e_hidden.tar.gz (some file names hold ':', which /data rejects).

Prompt: MLPerf's OpenHands user_prompt.j2 (NeMo Gym 610a08a, sha256 checked) with workspace /testbed
and base_commit = image HEAD (the commit's parent). Grading replays nv-R2E-Gym 6823e64
run_local_evaluation.py in the agent's container: diff the agent's work against the start snapshot the
way OpenHands builds model_patch (git add -A, drop binaries), restore the snapshot, git apply
--whitespace=fix with R2E's excludes, put the hidden tests back, R2E's setup_env, run_tests.sh with a
60 s timeout, and require the parsed results to equal expected_output_json exactly. Empty patch = 0.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

P = Path(os.environ["PTX_ROOT"])  # datasets/, harbor_tasks/, harbor/, data/
MLPERF, DATA, HIDDEN = P / "data/mlperf_r2e", P / "data/r2e-gym-subset", P / "data/r2e_hidden"
PROMPT = Path(__file__).with_name("r2e_user_prompt.j2")
PROMPT_SHA = "0b0489b934d2c229c439b7af2f588fe404a6327be9da3c12d4c7c9efd3cf1c03"
COLS = [
    "repo_name",
    "docker_image",
    "commit_hash",
    "parsed_commit_content",
    "problem_statement",
    "expected_output_json",
    "num_non_test_lines",
]

DOCKERFILE = """FROM alpine:3.20 AS tmux
RUN apk add --no-cache build-base bison curl libevent-dev libevent-static ncurses-dev ncurses-static && \\
    curl -fsSL https://github.com/tmux/tmux/releases/download/3.5a/tmux-3.5a.tar.gz | tar xz && \\
    cd tmux-3.5a && ./configure --enable-static --prefix=/usr/local && make -j"$(nproc)" && make install

FROM {image}
COPY --from=tmux /usr/local/bin/tmux /usr/local/bin/tmux
WORKDIR /testbed
# Untracked files R2E excludes from the patch; hide the tests like NeMo Gym's agent phase; snapshot the start.
RUN git config --global --add safe.directory '*' && \\
    git ls-files --others --exclude-standard > /.r2e_untracked && \\
    for d in "" /root /testbed; do rm -rf $d/r2e_tests; if grep -qs r2e_tests $d/run_tests.sh; then rm -f $d/run_tests.sh; fi; done && \\
    cp .git/index /tmp/ptx.idx && GIT_INDEX_FILE=/tmp/ptx.idx git add -A && GIT_INDEX_FILE=/tmp/ptx.idx git write-tree > /.ptx_base_tree && rm -f /tmp/ptx.idx
"""

TASK_TOML = """version = "1.0"

[metadata]
source = "MLPerf Training v6.1 R2E-Gym easy ({split})"
docker_image = "{image}"

[verifier]
timeout_sec = 180.0  # tests 60 s (MLPerf) + 120 s, as NeMo Gym's eval container

[agent]
timeout_sec = 1800.0

[environment]
build_timeout_sec = 1800.0
"""

SOLVE = """#!/bin/bash
cd /testbed && git apply --whitespace=fix "$(dirname "$0")/gold.patch"
"""

TEST_SH = r"""#!/bin/bash
# nv-R2E-Gym 6823e64 run_local_evaluation.py, replayed in the agent's container (see prep_r2e_mlperf.py).
echo 0 > /logs/verifier/reward.txt
export PATH=/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /testbed || exit 0
git() { command git -c core.hooksPath=/dev/null -c core.pager= "$@"; }
base=$(cat /.ptx_base_tree)
git add -A >/dev/null 2>&1 || { echo "git add failed"; exit 0; }
git diff --no-color --cached "$base" | awk '
  /^diff --git /{ if (s != "" && !b) printf "%s", s; s = ""; b = 0 }
  /^Binary files .* differ$/ || /^GIT binary patch$/ { b = 1 }
  { s = s $0 "\n" }
  END { if (s != "" && !b) printf "%s", s }' > /tmp/model.patch
cp /tmp/model.patch /logs/verifier/model.patch
grep -q '[^[:space:]]' /tmp/model.patch || { echo "Empty patch."; exit 0; }
git read-tree -u --reset "$base" || { echo "restore failed"; exit 0; }
git apply --whitespace=fix $(sed 's/^/--exclude=/' /.r2e_untracked) /tmp/model.patch || { echo "Failed to apply patch."; exit 0; }
rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests /root/run_tests.sh /tmp/r2e && mkdir /tmp/r2e
tar xzf /tests/r2e_hidden.tar.gz -C /tmp/r2e && mv /tmp/r2e/r2e_tests / && mv /tmp/r2e/run_tests.sh /testbed/
{  # DockerRuntime.setup_env
  ln -s /testbed/.venv /root/.venv
  ln -s /testbed/.venv/bin/python /root/.local/bin/python
  ln -s /testbed/.venv/bin/python /root/.local/bin/python3
  find /testbed/.venv/bin -type f -executable -exec ln -sf {} /root/.local/bin/ \;
  timeout 120 uv pip install chardet
  find . -name '*.pyc' -delete; find . -name '__pycache__' -exec rm -rf {} +
  find /r2e_tests -name '*.pyc' -delete; find /r2e_tests -name '__pycache__' -exec rm -rf {} +
  mv /testbed/run_tests.sh /root/run_tests.sh; mv /testbed/r2e_tests /root/r2e_tests
  mv /r2e_tests /root/r2e_tests; ln -s /root/r2e_tests /testbed/r2e_tests
} >/dev/null 2>&1
timeout 60 bash /root/run_tests.sh > /logs/verifier/test_output.txt 2>&1
PY=/testbed/.venv/bin/python; [ -x $PY ] || PY=python3
$PY /tests/grade.py /logs/verifier/test_output.txt /tests/expected_output.json > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
"""

GRADE_PY = r"""# DockerRuntime._calculate_reward_r2e + parse_log_pytest (nv-R2E-Gym 6823e64). Python 3.5+.
import json, re, sys

def parse_log_pytest(log):
    test_status_map = {}
    if "short test summary info" not in log:
        return test_status_map
    for line in log.split("short test summary info")[1].strip().split("\n"):
        if "PASSED" in line:
            test_status_map[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            test_status_map[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            test_status_map[".".join(line.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return test_status_map

decolor = lambda d: {re.sub(r"\u001b\[\d+m", "", k): v for k, v in d.items()}
log = re.sub(r"\x1b\[[0-9;]*m|\r", "", open(sys.argv[1], encoding="utf-8", errors="replace").read())
parse = decolor(parse_log_pytest(log))
expected = decolor(json.load(open(sys.argv[2])))
parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse)}
expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected)}
ok = len(parse) == len(expected) and all(not k or (k in expected and parse[k] == expected[k]) for k in parse)
print(1 if ok else 0)
"""


def ids(name: str) -> list[str]:
    return [x.strip() for x in (MLPERF / name).read_text().splitlines() if x.strip()]


def load_rows(conv: ModuleType, want: list[str]) -> dict[str, dict]:
    import pyarrow.parquet as pq

    rows = {}
    for f in sorted((DATA / "data").glob("train-*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(columns=COLS, batch_size=64):
            for r in batch.to_pylist():
                iid = conv.build_instance_id(r["repo_name"], r["commit_hash"])
                if iid in want:
                    rows[iid] = r
    missing = set(want) - set(rows)
    assert not missing, f"not in dataset: {sorted(missing)[:5]}"
    return rows  # dataset order, as MLPerf's converter sees them


def _rng(r: dict) -> str:
    return f"{r['start']}" if r.get("length") is None else f"{r['start']},{r['length']}"


def gold_patch(
    parsed_commit: dict,
) -> str:  # ParsedCommit.get_patch() defaults: all .py files, tests included
    mark = {"context": " ", "added": "+", "deleted": "-", "note": "\\ "}
    out = []
    for fd in parsed_commit["file_diffs"]:
        path = fd["header"]["file"]["path"]
        if not path.endswith(".py"):
            continue
        out.append(f"diff --git a/{path} b/{path}\n")
        if fd["header"].get("misc_line"):
            out.append(fd["header"]["misc_line"] + "\n")
        if il := fd.get("index_line"):
            out.append(
                f"index {il['old_commit_hash']}..{il['new_commit_hash']}{' ' if il['mode'] else ''}{il['mode']}\n"
            )
        if fd.get("is_binary_file"):
            out.append(fd["binary_line"] + "\n")
        if fd.get("minus_file") and fd.get("plus_file"):
            out.append(f"--- {fd['minus_file']['path']}\n+++ {fd['plus_file']['path']}\n")
        for h in fd.get("hunks", []):
            d = h["descriptor"]
            out.append(
                f"@@ -{_rng(d['old_range'])} +{_rng(d['new_range'])} @@"
                + (f" {d['section']}" if d.get("section") else "")
                + "\n"
            )
            out += [mark[x["type"]] + x["content"] + "\n" for x in h["line_group"]["all_lines"]]
    return "".join(out)


def extract(rows: dict[str, dict], workers: int) -> None:
    import modal

    app = modal.App.lookup("ptx-r2e-extract", create_if_missing=True)
    HIDDEN.mkdir(parents=True, exist_ok=True)
    cmd = "cd /testbed && git rev-parse HEAD && tar czf - -C / r2e_tests -C /testbed run_tests.sh | base64 -w0"

    def one(item: tuple[str, str]) -> tuple[str, str]:
        iid, image = item
        for _ in range(3):
            try:
                sb = modal.Sandbox.create(
                    "sleep", "900", image=modal.Image.from_registry(image), app=app, timeout=1200
                )
                try:
                    p = sb.exec("bash", "-c", cmd)
                    head, blob = p.stdout.read().split("\n", 1)
                    p.wait()
                    assert p.returncode == 0 and len(head) == 40, p.stderr.read()[-500:]
                finally:
                    sb.terminate()
                (HIDDEN / f"{iid}.tar.gz").write_bytes(base64.b64decode(blob))
                (HIDDEN / f"{iid}.head").write_text(head + "\n")
                return iid, "ok"
            except Exception as e:  # noqa: BLE001 - Modal/registry errors vary; retried, then reported
                err = repr(e)[:300]
        return iid, f"FAILED {err}"

    todo = [(i, r["docker_image"]) for i, r in rows.items() if not (HIDDEN / f"{i}.head").exists()]
    print(f"extracting {len(todo)} of {len(rows)}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for n, (iid, status) in enumerate(ex.map(one, todo), 1):
            print(n, iid, status, flush=True)


def write_task(d: Path, iid: str, r: dict, split: str, prompt: str) -> None:
    if d.exists():
        shutil.rmtree(d)
    for sub in ("environment", "tests", "solution"):
        (d / sub).mkdir(parents=True)
    (d / "instruction.md").write_text(prompt)
    (d / "task.toml").write_text(TASK_TOML.format(split=split, image=r["docker_image"]))
    (d / "environment/Dockerfile").write_text(DOCKERFILE.format(image=r["docker_image"]))
    (d / "solution/gold.patch").write_text(gold_patch(json.loads(r["parsed_commit_content"])))
    (d / "solution/solve.sh").write_text(SOLVE)
    (d / "tests/test.sh").write_text(TEST_SH)
    (d / "tests/grade.py").write_text(GRADE_PY)
    (d / "tests/expected_output.json").write_text(r["expected_output_json"])
    shutil.copy(
        HIDDEN / f"{iid}.tar.gz", d / "tests/r2e_hidden.tar.gz"
    )  # tar: some test names hold ':'
    for f in ("solution/solve.sh", "tests/test.sh"):
        (d / f).chmod(0o755)


def build(
    conv: ModuleType,
    rows: dict[str, dict],
    train: list[str],
    val: list[str],
    tasks_dir: Path,
    out_dir: Path,
) -> None:
    import jinja2

    assert hashlib.sha256(PROMPT.read_bytes()).hexdigest() == PROMPT_SHA, "prompt template changed"
    template = jinja2.Environment(loader=jinja2.FileSystemLoader(PROMPT.parent)).get_template(
        PROMPT.name
    )
    ours = set(train) | set(val)
    clash = [
        d.name
        for d in tasks_dir.iterdir()
        if d.name in ours and not (d / "tests/grade.py").exists()
    ]
    assert not clash, f"would overwrite unrelated task dirs: {clash[:5]}"

    prompts = {}
    for iid, r in rows.items():
        head = (HIDDEN / f"{iid}.head").read_text().strip()
        instance = {"problem_statement": r["problem_statement"], "base_commit": head}
        prompts[iid] = template.render(instance=instance, workspace_path="/testbed")
        write_task(tasks_dir / iid, iid, r, "train" if iid in train else "val", prompts[iid])

    items = [
        conv.CurriculumItem(
            i, conv.repo_from_repo_name(r["repo_name"]), int(r["num_non_test_lines"] or 0), r
        )
        for i, r in rows.items()
        if i in train
    ]
    ordered, _ = conv.curriculum_order_train_items(items)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, order in (
        ("train", [it.identifier for it in ordered]),
        ("val", [i for i in rows if i in val]),
    ):
        with open(out_dir / f"r2e_mlperf_{split}.jsonl", "w") as f:
            for iid in order:
                meta = {
                    "instance_id": iid,
                    "agent_name": "terminus-2",
                    "split": f"r2e_mlperf_{split}",
                }
                f.write(json.dumps({"prompt": prompts[iid], "metadata": meta}) + "\n")
        print(f"{split}: {len(order)} rows -> {out_dir}/r2e_mlperf_{split}.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "build"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tasks-dir", type=Path, default=P / "harbor_tasks")
    ap.add_argument("--out-dir", type=Path, default=P / "datasets")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location(
        "conv", MLPERF / "create_r2e_gym_easy_subset_jsonl.py"
    )
    conv = importlib.util.module_from_spec(spec)
    sys.modules["conv"] = conv  # dataclasses need the module registered
    spec.loader.exec_module(conv)
    train, val = ids("train-instance-ids.txt"), ids("val-instance-ids.txt")
    assert len(train) == 700 and len(val) == 251 and not set(train) & set(val)
    rows = load_rows(conv, set(train) | set(val))
    if a.step == "extract":
        extract(rows, a.workers)
    else:
        build(conv, rows, set(train), set(val), a.tasks_dir, a.out_dir)


if __name__ == "__main__":
    main()
