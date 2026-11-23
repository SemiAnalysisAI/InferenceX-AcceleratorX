"""Harbor tasks for SWE-bench Verified (500) + miles eval JSONL, via Harbor's adapters/swebench (ac63e51).

  $PTX_ROOT/swebench_prep/.venv/bin/python prep_swebench_verified.py   # venv: swebench==4.1.0, datasets, jinja2

Kept from the adapter: official SWE-bench x86 images, its test.sh (reset + apply test_patch, repo test cmd,
swebench 4.0.3 log parser, resolved = FAIL_TO_PASS and PASS_TO_PASS all pass), solve.sh (gold patch).
Changed: image names (adapter references an undefined SWEGYM_DOCKER_NAMESPACE and silently maps nothing;
we use make_test_spec(namespace="swebench") as adapters/swebench_multilingual does); instruction = the R2E
MLPerf/OpenHands prompt with workspace /testbed; static tmux for terminus-2; the parser's deps preinstalled in
/opt/grader (offline grading); test.sh writes reward 0 first (early death scores 0, not an error) and unsets
PYTHONPATH (Modal sets /pkg/:/root/, which breaks pylint-7277's sys.path tests); agent and verifier timeouts
1800 s (MLPerf agent budget, SWE-bench harness test timeout); build 3600 s (cold Modal builds of big images).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import jinja2

P = Path(os.environ["PTX_ROOT"])  # datasets/, harbor_tasks/, harbor/, data/
sys.path.insert(0, str(P / "harbor/adapters/swebench/src"))
from prep_r2e_mlperf import DOCKERFILE as R2E_DOCKERFILE, PROMPT, PROMPT_SHA  # noqa: E402
from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: E402
from swebench_adapter import adapter as ad  # noqa: E402

TASKS, OUT = P / "harbor_tasks", P / "datasets/swebench_verified.jsonl"
TMUX = R2E_DOCKERFILE.split("FROM {image}")[0]
STOCK_TEST = (
    P / "harbor/adapters/swebench/src/swebench_adapter/task-template/tests/test.sh"
).read_text()
GRADER_DEPS = (
    '"swebench==4.0.3" "datasets==2.16.1" "fastcore<1.11"'  # test.sh parser.py's inline deps
)
assert (
    GRADER_DEPS.replace('" "', '", "') in STOCK_TEST and STOCK_TEST.count("uv run parser.py") == 1
)
DOCKERFILE = (
    TMUX
    + """FROM {image}
COPY --from=tmux /usr/local/bin/tmux /usr/local/bin/tmux
WORKDIR /testbed
# Grader deps preinstalled; stock test.sh would `uv run` them from PyPI on every grade.
RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh && mkdir -p /logs && \\
    UV_PYTHON_INSTALL_DIR=/opt/grader-python /root/.local/bin/uv venv -q --python 3.11 --python-preference only-managed /opt/grader && \\
    /root/.local/bin/uv pip install -q --python /opt/grader/bin/python {deps} && \\
    cd / && /opt/grader/bin/python -c "import swebench.harness.grading"  # not /testbed: psf/requests shadows requests
"""
)


def images(records: list[dict]) -> dict[str, str]:
    return {
        r["instance_id"]: make_test_spec(r, namespace="swebench").instance_image_key.replace(
            "arm64", "x86_64"
        )
        for r in records
    }


def main() -> None:
    assert hashlib.sha256(PROMPT.read_bytes()).hexdigest() == PROMPT_SHA, "prompt template changed"
    tpl = jinja2.Environment(loader=jinja2.FileSystemLoader(PROMPT.parent)).get_template(
        PROMPT.name
    )
    ad.get_image_names = images
    conv = ad.SWEBenchAdapter(output_dir=TASKS, overwrite=True, max_timeout_sec=1800.0)
    ids = conv.get_all_ids()
    assert len(ids) == 500 and len(conv.id_to_docker_image) == 500
    clash = [
        i for i in ids if (TASKS / i).exists() and not (TASKS / i / "tests/config.json").exists()
    ]
    assert not clash, f"would overwrite unrelated task dirs: {clash[:5]}"

    with open(OUT, "w") as f:
        for iid in ids:
            d = conv.generate_task(iid, iid, overwrite=True)
            rec = conv.loader.load(iid)
            prompt = tpl.render(
                instance={
                    "problem_statement": rec.problem_statement,
                    "base_commit": rec.base_commit,
                },
                workspace_path="/testbed",
            )
            (d / "instruction.md").write_text(prompt)
            (d / "environment/Dockerfile").write_text(
                DOCKERFILE.format(image=conv.id_to_docker_image[iid], deps=GRADER_DEPS)
            )
            c = (
                d / "task.toml"
            )  # cold Modal builds of the big images (matplotlib, seaborn) took >1800 s
            c.write_text(
                c.read_text().replace("build_timeout_sec = 1800.0", "build_timeout_sec = 3600.0")
            )
            t = d / "tests/test.sh"
            t.write_text(
                t.read_text()
                .replace(
                    "set -euo pipefail\n",
                    "set -euo pipefail\nmkdir -p /logs/verifier; "
                    "echo 0 > /logs/verifier/reward.txt\n"
                    "unset PYTHONPATH  # Modal injects /pkg/:/root/\n",
                    1,
                )
                .replace("uv run parser.py", "/opt/grader/bin/python parser.py")
            )
            meta = {"instance_id": iid, "agent_name": "terminus-2", "split": "swebench_verified"}
            f.write(json.dumps({"prompt": prompt, "metadata": meta}) + "\n")
    print(f"{len(ids)} tasks -> {TASKS}, {len(ids)} rows -> {OUT}")


if __name__ == "__main__":
    main()
