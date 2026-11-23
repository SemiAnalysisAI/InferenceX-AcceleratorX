"""Qwen3.5 MoE full-FT GRPO on SWE tasks with terminus-2 in Modal sandboxes via Harbor.

Model layout from miles' Qwen3.5-35B-A3B scripts: TP <= 2 (num_query_groups=2), experts over all
trainer GPUs, Megatron distributed optimizer. Split pools, not colocated: the first `train_nodes`
nodes train, the rest sample (SGLang engines), so nothing is offloaded.

Run inside the miles image on the ray head with MILES_SCRIPT_EXTERNAL_RAY=1 (train.sbatch).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import miles.utils.external_utils.command_utils as U
import typer

app = typer.Typer()

SCRIPT_DIR = Path(__file__).resolve().parent  # ptx_agent, ptx_rollout_log


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_nodes: int = 4
    num_gpus_per_node: int = 4
    train_nodes: int = 2
    # required; empty means unset (the dataclass CLI needs a default after the base class's)
    hf_checkpoint: str = ""
    ref_load: str = ""
    run_dir: str = ""
    prompt_data: str = ""
    eval_data: str = ""
    megatron_model_type: str = "qwen3.5-35B-A3B"
    megatron_path: str = "/root/Megatron-LM"

    num_rollout: int = 40
    rollout_batch_size: int = 8
    over_sampling_batch_size: int = 16
    n_samples_per_prompt: int = 16
    max_seq_len: int = 65536
    rollout_max_response_len: int = 8192
    max_tokens_per_gpu: int = 65536
    eval_interval: int = 5
    n_samples_per_eval_prompt: int = 4
    save_interval: int = 20

    lr: float = 1e-6
    tp: int = 2
    rollout_gpus_per_engine: int = 1  # 2 for 122B (250 GB of bf16 weights)
    # in-flight agent episodes across all engines; each is a subprocess on the rollout node, and miles'
    # default (512 per engine) let a 5,120-attempt eval start them all at once and OOM a 922 GB node.
    max_agents: int = 320
    filter_flat_groups: bool = True
    skip_eval_before_train: bool = False  # the base model's eval is already known
    # MLPerf Training v6.1 qwen35_397b_grpo recipe (training_policies/training_rules.adoc): fixed data order,
    # no dynamic sampling, temp 1.0 / eval 0.1 top_p 0.95, AdamW 0.9/0.999 wd 0, lr and clip scaled by batch,
    # token-level loss, token TIS truncated at 1.002 (sequence mask opt-in). Agent max 30 turns / 1800 s are set on the Harbor service.
    mlperf: bool = False
    mlperf_seq_mask: bool = False
    # Score centering (arXiv:2609.20807, miles --loss-type score_centering) + Dr.GRPO advantages,
    # token-level loss, flat-group filter, MLPerf optimizer. Under --mlperf, --skip-actor-forward-only
    # makes the PPO ratio exactly 1 (clipping never fires) and the sequence TIS mask drops 51-69% of
    # tokens without renormalising.
    sc: bool = False
    sc_topk: int = 20  # sampler top-K logprobs per position; 20 is the router's default limit (watch sc_*_head_mass >= 0.98)
    eval_temperature: float | None = (
        None  # override; 1.0 for solve-rate scans (num_rollout 0 = eval only)
    )
    load_dir: str = ""  # load a checkpoint dir from another run (e.g. eval-only of a trained model)
    seed: int | None = None  # Megatron --seed and --rollout-seed; None keeps miles' defaults

    agent_server_url: str = ""
    session_server_external_host: str = ""
    agent_trial_timeout: int = 2400
    extra_args: str = ""


def _train(args: ScriptArgs) -> None:
    for name in (
        "agent_server_url",
        "hf_checkpoint",
        "ref_load",
        "run_dir",
        "prompt_data",
        "eval_data",
    ):
        assert getattr(args, name), f"--{name.replace('_', '-')} is required"
    run_dir = Path(args.run_dir)
    train_gpus = args.train_nodes * args.num_gpus_per_node
    rollout_gpus = args.num_nodes * args.num_gpus_per_node - train_gpus
    assert rollout_gpus > 0

    gbs = args.rollout_batch_size * args.n_samples_per_prompt
    if args.mlperf or args.sc:
        args.lr = 1e-6 * (gbs / 256) ** 0.5
    if args.mlperf:
        args.filter_flat_groups = False
    if args.sc:
        args.filter_flat_groups = True

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ref_load} "
        f"--save {run_dir}/checkpoints --save-interval {args.save_interval} "
    )
    if args.load_dir:
        ckpt_args += f"--load {args.load_dir} "
    elif (run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").exists():
        ckpt_args += (
            f"--load {run_dir}/checkpoints "  # resume a run dir that already has checkpoints
        )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        f"{'--no-rollout-shuffle ' if args.mlperf or args.sc else '--rollout-shuffle '}"
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {gbs} "
        f"--rollout-temperature {1.0 if args.mlperf or args.sc else 0.8} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--num-steps-per-rollout 1 "
        "--balance-data "
    )
    if args.filter_flat_groups:
        rollout_args += (
            f"--over-sampling-batch-size {args.over_sampling_batch_size} "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )
    else:  # aborted groups are dropped either way (apply_preput_filters)
        rollout_args += "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "

    eval_args = (
        f"--eval-prompt-data swe {args.eval_data} "
        f"--eval-interval {args.eval_interval} "
        f"--n-samples-per-eval-prompt {args.n_samples_per_eval_prompt} "
        f"--eval-max-response-len {args.rollout_max_response_len} "
        f"{'--eval-temperature 0.1 --eval-top-p 0.95 ' if args.mlperf else '--eval-temperature 0.6 '}"
    )
    if args.eval_temperature is not None:  # later flags win in argparse
        eval_args += f"--eval-temperature {args.eval_temperature} --eval-top-p 1.0 "
    if args.skip_eval_before_train:
        eval_args += "--skip-eval-before-train "

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {train_gpus} --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--qkv-format thd --use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        # one optimizer step per rollout: old-policy log-probs = detached log-probs of the training forward
        f"{'--use-rollout-logprobs ' if args.sc else '--skip-actor-forward-only '}"
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 --use-distributed-optimizer "
    )
    if args.mlperf or args.sc:
        optimizer_args = (
            f"--optimizer adam --lr {args.lr:.4g} --lr-decay-style constant --weight-decay 0.0 "
            "--adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --use-distributed-optimizer "
            f"--clip-grad {0.125 * (256 / gbs) ** 0.5:.4g} "
        )
    if args.sc:
        grpo_args = (
            "--advantage-estimator grpo --disable-grpo-std-normalization --calculate-per-token-loss "
            "--kl-loss-coef 0.00 --entropy-coef 0.00 --observe-training-entropy "
            f"--loss-type score_centering --rollout-top-logprobs-num {args.sc_topk} "
            "--score-centering-is tis --score-centering-tis-clip 2.0 --rollout-top-p 1.0 --rollout-top-k -1 "
        )
    elif args.mlperf:
        grpo_args += (
            "--calculate-per-token-loss "
            "--use-tis --custom-tis-function-path "
            "examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
            "--tis-level token --tis-mode truncate --tis-lower-bound 0.999 --tis-upper-bound 1.002 "
        )
        # MLPerf also masks sequences whose geometric-mean ratio leaves [0.999, 1.002]. Between SGLang and Megatron
        # that dropped 69% of tokens at KL 3e-4, so the sequence mask is opt-in.
        if args.mlperf_seq_mask:
            grpo_args += (
                "--use-rs --rs-level geometric --rs-lower-bound 0.999 --rs-upper-bound 1.002 "
            )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_gpus_per_engine} "
        f"--rollout-num-gpus {rollout_gpus} "
        "--sglang-mem-fraction-static 0.8 "
        f"--sglang-server-concurrency {max(1, args.max_agents // (rollout_gpus // args.rollout_gpus_per_engine))} "
        f"--sglang-context-length {args.max_seq_len + args.rollout_max_response_len} "
        "--sglang-router-port 31000 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )

    external_host_arg = (
        f"--session-server-external-host {args.session_server_external_host} "
        if args.session_server_external_host
        else ""
    )
    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path ptx_agent.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--custom-rollout-log-function-path ptx_rollout_log.log_rollout "
        "--rollout-all-samples-process-path ptx_rollout_log.stash_all_samples "
        f"--max-seq-len {args.max_seq_len} "
        "--tito-model qwen35 "
        "--use-session-server "
        "--session-server-port 30000 "
        "--session-server-workers 32 "
        f"{external_host_arg}"
    )

    logging_args = f"--save-debug-event-data {run_dir}/events --dump-details {run_dir}/dump "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--accumulate-allreduce-grads-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024**3} "
        f"--actor-num-nodes {args.train_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
        "--distributed-timeout-minutes 60 "
    )

    if (
        args.num_rollout == 0
    ):  # eval only: Megatron's scheduler asserts lr_decay_steps > 0 when train_iters is 0
        misc_args += "--lr-decay-iters 1 "
    if args.seed is not None:
        misc_args += f"--seed {args.seed} --rollout-seed {args.seed} "

    train_args = (
        f"{ckpt_args} {rollout_args} {eval_args} {optimizer_args} {grpo_args} "
        f"{perf_args} {sglang_args} {agent_args} {logging_args} {misc_args} {args.extra_args} "
    )

    example_dir = f"{U.repo_base_dir}/examples/swe-agent-harbor-docker"
    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{example_dir}:{SCRIPT_DIR}:{U.repo_base_dir}",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_TRIAL_TIMEOUT": str(args.agent_trial_timeout),
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        # inductor's subprocess compile pool deadlocks under the torch_memory_saver LD_PRELOAD
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    for var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TILELANG_CACHE_DIR"):
        if os.environ.get(var):
            extra_env_vars[var] = os.environ[var]

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_args.txt").write_text(train_args.strip() + "\n")

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    _train(args)


if __name__ == "__main__":
    app()
