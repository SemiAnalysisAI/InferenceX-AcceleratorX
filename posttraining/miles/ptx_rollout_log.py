"""PTX per-rollout benchmark metrics, logged under ``ptx/`` next to miles' default rollout metrics.

Hooks (both run in the rollout executor process, see miles/ray/rollout/rollout_executor.py):
  --custom-rollout-log-function-path ptx_rollout_log.log_rollout
      Called by ``miles.ray.rollout.metrics.log_rollout_data(rollout_id, args, samples, extra, rollout_time)``
      before the default logging. ``samples`` are flattened, post-dynamic-filter, post-trim (= what trains);
      ``rollout_time`` is the same value miles logs as ``perf/rollout_time``. Returning a truthy value would
      skip miles' default logging, so this always returns False.
  --rollout-all-samples-process-path ptx_rollout_log.stash_all_samples   (optional, recommended)
      Called by ``generate_rollout_async`` with every group that finished generating, including groups the
      dynamic filter dropped. Without it, "all" == trained samples. Groups still in flight when oversampling
      aborts are never visible to either hook, so their tokens are not counted anywhere.

Definitions (one trajectory = one Harbor trial; grouped by miles' rollout key, so v1 merged samples and v2
per-turn siblings both reduce to one row):
  e2e_latency_s   agent_metrics.total_time (whole Harbor trial, incl. docker build/start and verifier).
  gen_time_s      sum over recorded model calls of (t1 - req_ts) from ``metadata["lifecycle"]``: session-server
                  arrival -> response recorded, i.e. the model-call latency the agent sees (falls back to
                  SGLang e2e_latency when req_ts is missing). Session v1 merges turns and concatenates their
                  segments, so every recorded turn is present, except: non-200 calls (never recorded), turns
                  cut by --max-seq-len truncation, and turns after an intermediate truncated turn (merge stops).
                  ``gen_turn_coverage`` = recorded segments / Harbor ``turns`` shows how complete it is.
  sandbox_time_s  e2e_latency_s - gen_time_s: everything that is not policy generation (env setup, tool
                  execution, agent overhead, verifier).
  output_len      loss-masked model tokens (sum of effective_response_length).
  input_len       total trajectory tokens - output_len (initial prompt + all env/tool observations).
  prompt_len      first-turn prompt tokens.
  *_tok_per_s_per_gpu  tokens / rollout_time / --rollout-num-gpus; "rollout_gen" counts loss-masked tokens of
                  all finished samples, "effective_gen"/"effective_seq" count loss-masked / full-sequence
                  tokens of the samples that enter the trainer.
  sandbox_concurrency  number of trials running at once from trial_start_ts/trial_end_ts (ptx_agent.run);
                  mean is time-weighted over [first start, last end] of the batch's trials.
"""

import logging
import re
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

SUCCESS_EXIT_STATUS = "Submitted"
_HARBOR_PHASES = ("env_setup_time", "agent_setup_time", "agent_run_time", "eval_time")

_all_samples: list | None = None


def stash_all_samples(args: Any, all_samples: list, data_source: Any = None) -> None:
    global _all_samples
    try:
        _all_samples = list(_flatten(all_samples))
    except Exception:
        logger.warning("ptx: stash_all_samples failed", exc_info=True)
        _all_samples = None


def log_rollout(
    rollout_id: int,
    args: Any,
    samples: list,
    rollout_extra_metrics: dict | None,
    rollout_time: float,
) -> bool:
    global _all_samples
    all_samples, _all_samples = _all_samples, None
    try:
        from miles.utils.metric_utils import compute_rollout_step, namespace_metrics
        from miles.utils.tracking_utils import tracking

        metrics = compute_ptx_metrics(args, samples, all_samples, rollout_time)
        logger.info("ptx %s: %s", rollout_id, metrics)
        log_dict, step_key = namespace_metrics(
            metrics,
            trainer_model_id=getattr(args, "trainer_model_id", None),
            step_name="rollout/step",
            step=compute_rollout_step(args, rollout_id),
        )
        tracking.log(args, log_dict, step_key=step_key)
    except Exception:
        logger.warning("ptx: rollout metrics for rollout %s failed", rollout_id, exc_info=True)
    return False


def compute_ptx_metrics(
    args: Any, samples: list, all_samples: list | None, rollout_time: float
) -> dict[str, float]:
    trained = [s for s in samples if not s.remove_sample]
    pool = all_samples or samples
    trajs = [_trajectory(group) for group in _group_by_trajectory(pool)]
    num_gpus = getattr(args, "rollout_num_gpus", None) or 0

    m: dict[str, float] = {
        "ptx/rollout_time_s": rollout_time,
        "ptx/all_samples_seen": float(bool(all_samples)),
        "ptx/num_trials": len(trajs),
        "ptx/num_trials_trained": len(_group_by_trajectory(trained)),
    }
    m["ptx/num_trials_dropped"] = m["ptx/num_trials"] - m["ptx/num_trials_trained"]

    for key in (
        "e2e_latency_s",
        "gen_time_s",
        "sandbox_time_s",
        *_HARBOR_PHASES,
        "input_len",
        "output_len",
        "prompt_len",
    ):
        m |= _stats(f"ptx/{key}", [t[key] for t in trajs])
    if covered := [t["gen_segments"] / t["turns"] for t in trajs if t["turns"]]:
        m["ptx/gen_turn_coverage"] = sum(covered) / len(covered)

    gen_all = sum(s.effective_response_length for s in pool)
    gen_trained = sum(s.effective_response_length for s in trained)
    seq_trained = sum(len(s.tokens) for s in trained)
    m |= {
        "ptx/gen_tokens_all": gen_all,
        "ptx/gen_tokens_trained": gen_trained,
        "ptx/seq_tokens_trained": seq_trained,
    }
    if rollout_time > 0 and num_gpus:
        m["ptx/rollout_gen_tok_per_s_per_gpu"] = gen_all / rollout_time / num_gpus
        m["ptx/effective_gen_tok_per_s_per_gpu"] = gen_trained / rollout_time / num_gpus
        m["ptx/effective_seq_tok_per_s_per_gpu"] = seq_trained / rollout_time / num_gpus

    m |= _concurrency(
        [(t["start"], t["end"]) for t in trajs if t["start"] is not None and t["end"] is not None]
    )

    if rewards := [t["reward"] for t in trajs if t["reward"] is not None]:
        m["ptx/reward_mean_all"] = sum(rewards) / len(rewards)
    if rewards := [r for s in trained if (r := _reward(s)) is not None]:
        m["ptx/reward_mean_trained"] = sum(rewards) / len(rewards)
    statuses = [t["exit_status"] or "none" for t in trajs]
    m["ptx/num_non_success"] = sum(st != SUCCESS_EXIT_STATUS for st in statuses)
    for st in set(statuses):
        m[f"ptx/exit_status/{re.sub(r'[^A-Za-z0-9_.-]', '_', st)}"] = statuses.count(st)
    return m


def _trajectory(group: list) -> dict[str, Any]:
    meta = next(
        (s.metadata for s in group if s.metadata and s.metadata.get("agent_metrics")),
        group[0].metadata or {},
    )
    am = meta.get("agent_metrics") or {}
    segments = [seg for s in group for seg in _segments(s)]
    latencies = [x for seg in segments if (x := _segment_latency(seg)) is not None]
    gen_time = sum(latencies) if latencies else None
    total_time = am.get("total_time")
    output_len = sum(s.effective_response_length for s in group)
    total_len = max(len(s.tokens) for s in group)
    return {
        "e2e_latency_s": total_time,
        "gen_time_s": gen_time,
        "sandbox_time_s": total_time - gen_time
        if total_time is not None and gen_time is not None
        else None,
        **{key: am.get(key) for key in _HARBOR_PHASES},
        "output_len": output_len,
        "input_len": total_len - output_len,
        "prompt_len": min(len(s.tokens) - s.response_length for s in group),
        "gen_segments": len(latencies),
        "turns": am.get("turns"),
        "start": am.get("trial_start_ts"),
        "end": am.get("trial_end_ts"),
        "reward": _reward(group[0]),
        "exit_status": meta.get("exit_status"),
    }


def _segments(sample: Any) -> list[dict]:
    value = (sample.metadata or {}).get("lifecycle")
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _segment_latency(seg: dict) -> float | None:
    t1 = seg.get("t1")
    t0 = seg.get("req_ts") if seg.get("req_ts") is not None else seg.get("t0")
    return t1 - t0 if t1 is not None and t0 is not None else None


def _reward(sample: Any) -> float | None:
    value = (sample.metadata or {}).get("reward", sample.reward)
    return float(value) if isinstance(value, (int, float)) else None


def _concurrency(intervals: list[tuple[float, float]]) -> dict[str, float]:
    if not intervals:
        return {}
    events = sorted(
        [(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals], key=lambda x: (x[0], x[1])
    )
    span = max(e for _, e in intervals) - min(s for s, _ in intervals)
    cur = peak = 0
    area = 0.0
    prev = events[0][0]
    for t, delta in events:
        area += cur * (t - prev)
        cur += delta
        peak = max(peak, cur)
        prev = t
    return {
        "ptx/sandbox_concurrency_max": peak,
        "ptx/sandbox_concurrency_mean": area / span if span > 0 else float(peak),
        "ptx/trial_span_s": span,
    }


def _stats(prefix: str, values: Iterable[float | None]) -> dict[str, float]:
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return {}
    return {
        f"{prefix}/p50": _percentile(xs, 50),
        f"{prefix}/mean": sum(xs) / len(xs),
        f"{prefix}/p90": _percentile(xs, 90),
        f"{prefix}/max": xs[-1],
    }


def _percentile(sorted_xs: list[float], q: float) -> float:
    pos = (len(sorted_xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def _group_by_trajectory(samples: list) -> list[list]:
    groups: dict[tuple, list] = {}
    for position, s in enumerate(samples):
        if s.rollout_id is not None:
            key = ("rollout", s.group_index, s.rollout_id)
        elif s.index is not None:
            key = ("sample", s.group_index, s.index)
        else:
            key = ("position", s.group_index, position)
        groups.setdefault(key, []).append(s)
    return list(groups.values())


def _flatten(items: Iterable) -> Iterable:
    for item in items:
        if isinstance(item, list):
            yield from _flatten(item)
        else:
            yield item
