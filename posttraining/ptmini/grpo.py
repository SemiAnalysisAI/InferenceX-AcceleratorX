"""Device-agnostic GRPO policy: HF generate for rollouts, full-parameter fwd/bwd for training.

Plain PyTorch only, so the same code runs on any backend with a torch device
(cuda, rocm, mps, npu, xla). It is the reference implementation; tuned stacks
(vLLM + verl, miles, ...) are compared against it, not substituted for it.
"""

import contextlib
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from ptmini.task import SYSTEM


@dataclass
class Cfg:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    prompts_per_step: int = 16
    group: int = 8
    max_new: int = 384
    lr: float = 1e-6
    micro_bs: int = 4
    temperature: float = 1.0
    seed: int = 0
    logit_noise: float = 0.0  # sampler-only Gaussian logit noise: random numerics damage
    sampler_fp8: str = ""  # "e4m3" | "e5m2" | "int4": sampler-only weight rounding


class _Noise(LogitsProcessor):
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, _input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return scores + self.sigma * torch.randn_like(scores)


@contextlib.contextmanager
def fp8_weights(model: torch.nn.Module, fmt: str) -> "Iterator[None]":
    """Temporarily round the decoder's linear weights, as a low-precision rollout engine would: fp8
    with a per-tensor scale, or symmetric int4 with a per-output-channel scale. Systematic damage.
    lm_head is left alone: Qwen2.5-0.5B ties it to the embeddings, which engines keep in bf16."""
    if not fmt:
        yield
        return
    linears = [
        m.weight
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and not name.endswith("lm_head")
    ]
    saved = [w.detach().clone() for w in linears]
    with torch.no_grad():
        for w in linears:
            w.copy_(_round(w.float(), fmt).to(w.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for w, orig in zip(linears, saved, strict=True):
                w.copy_(orig)


def _round(w: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "int4":
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7
        return (w / scale).round().clamp(-8, 7) * scale
    dt = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fmt]
    scale = w.abs().amax().clamp(min=1e-12) / torch.finfo(dt).max
    return (w / scale).to(dt).float() * scale


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(dev: torch.device) -> None:
    mod = getattr(torch, dev.type, None)
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize()


def telemetry(dev: torch.device) -> dict[str, float]:
    """Wall time plus SM clock and power where the backend exposes them (CUDA via NVML)."""
    out = {"wall": time.time()}
    if dev.type == "cuda":
        with contextlib.suppress(Exception):  # NVML missing or unsupported
            out |= {"mhz": torch.cuda.clock_rate(dev), "watts": torch.cuda.power_draw(dev) / 1000}
    return out


def timed[T](
    dev: torch.device, fn: "Callable[..., T]", *args: object, **kw: object
) -> tuple[T, float]:
    sync(dev)
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    sync(dev)
    return out, time.perf_counter() - t0


class Policy:
    def __init__(self, cfg: Cfg, dev: torch.device, *, dtype: torch.dtype = torch.bfloat16) -> None:
        torch.manual_seed(cfg.seed)
        self.cfg, self.dev = cfg, dev
        self.tok = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(dev)
        eos = self.model.generation_config.eos_token_id or self.tok.eos_token_id
        self.eos = {eos} if isinstance(eos, int) else set(eos)
        self.model.gradient_checkpointing_enable()
        # fp16 gradients of a token-mean loss underflow without a scale; bf16 and fp32 need none.
        self.loss_scale = 1024.0 if dtype == torch.float16 else 1.0
        self.master: list[torch.Tensor] = []
        self.opt: torch.optim.Optimizer | None = None

    def _optimizer(self) -> torch.optim.Optimizer:
        """fp32 master weights for AdamW: an lr-1e-6 update is below bf16 resolution on most weights.
        Built on first use so parity (no training) does not pay for them."""
        if self.opt is None:
            self.master = [p.detach().float().clone() for p in self.model.parameters()]
            self.opt = torch.optim.AdamW(self.master, lr=self.cfg.lr, weight_decay=0.0)
        return self.opt

    def encode(self, question: str) -> list[int]:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[int]],
        max_new: int,
        *,
        force_len: bool = False,
        greedy: bool = False,
    ) -> list[list[int]]:
        """Sample completions. force_len suppresses EOS so every row decodes exactly max_new tokens."""
        self.model.eval()
        batch = self.tok.pad({"input_ids": prompts}, return_tensors="pt").to(self.dev)
        with fp8_weights(self.model, "" if greedy else self.cfg.sampler_fp8):
            out = self._sample(batch, max_new, force_len=force_len, greedy=greedy)
        comps = []
        for row in out[:, batch["input_ids"].shape[1] :].tolist():
            end = next((i + 1 for i, t in enumerate(row) if t in self.eos), len(row))
            comps.append(row[:end])
        return comps

    def _sample(self, batch: dict, max_new: int, *, force_len: bool, greedy: bool) -> torch.Tensor:
        return self.model.generate(
            **batch,
            max_new_tokens=max_new,
            min_new_tokens=max_new if force_len else 0,
            do_sample=not greedy,
            temperature=self.cfg.temperature if not greedy else None,
            top_p=1.0 if not greedy else None,
            top_k=0 if not greedy else None,
            pad_token_id=self.tok.pad_token_id,
            logits_processor=LogitsProcessorList(
                [_Noise(self.cfg.logit_noise)] if self.cfg.logit_noise and not greedy else []
            ),
        )

    def _batch(self, seqs: list[list[int]], n_prompt: list[int]) -> tuple[torch.Tensor, ...]:
        width = max(map(len, seqs))
        pad = self.tok.pad_token_id
        ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=self.dev)
        attn = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=self.dev)
        mask = torch.zeros_like(ids, dtype=torch.float32)
        for i, (s, p) in enumerate(zip(seqs, n_prompt, strict=True)):
            mask[i, p : len(s)] = 1.0
        return ids, attn, mask

    def logprobs(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_ids=ids, attention_mask=attn).logits[:, :-1]
        tgt = ids[:, 1:]
        lp = -F.cross_entropy(logits.float().transpose(1, 2), tgt, reduction="none")
        return F.pad(lp, (1, 0))

    def train(self, prompts: list[list[int]], comps: list[list[int]], adv: list[float]) -> float:
        """One on-policy GRPO update: token-mean of -A * logp over the whole batch, one optimizer step."""
        self.model.train()
        n_tok = sum(map(len, comps))
        order = sorted(range(len(comps)), key=lambda i: len(prompts[i]) + len(comps[i]))
        total = 0.0
        for k in range(0, len(order), self.cfg.micro_bs):
            idx = order[k : k + self.cfg.micro_bs]
            ids, attn, mask = self._batch(
                [prompts[i] + comps[i] for i in idx], [len(prompts[i]) for i in idx]
            )
            a = torch.tensor([adv[i] for i in idx], device=self.dev)[:, None]
            loss = -(a * self.logprobs(ids, attn) * mask).sum() / n_tok
            (loss * self.loss_scale).backward()
            total += loss.item()
        opt = self._optimizer()
        for m, p in zip(self.master, self.model.parameters(), strict=True):
            m.grad = None if p.grad is None else p.grad.float() / self.loss_scale
            p.grad = None
        torch.nn.utils.clip_grad_norm_(self.master, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for m, p in zip(self.master, self.model.parameters(), strict=True):
                p.copy_(m)
        return total


def group_advantages(rewards: list[float], group: int) -> list[float]:
    out = []
    for g in range(0, len(rewards), group):
        r = torch.tensor(rewards[g : g + group])
        out += ((r - r.mean()) / (r.std() + 1e-4)).tolist()
    return out
