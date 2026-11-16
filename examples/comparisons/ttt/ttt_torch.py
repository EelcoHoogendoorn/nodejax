"""Reusable test-time training wrapper in PyTorch across stateless and recurrent modules.

PyTorch modules keep weights as mutable attributes and leave call signatures unconstrained (`forward(x)` vs. `forward(carry, x)`). Differentiating outer Adam steps through inner token-by-token weight updates forces `TTT` to evict weights out of `nn.Module` into plain dictionaries for `torch.func.functional_call`, compute inner updates with `torch.func.grad` because `torch.optim` mutates tensors in place and breaks higher-order autograd, mangle `.` into `/` inside `nn.ParameterDict` rate keys, branch on `hasattr(model, 'init_carry')`, and unroll time as a Python loop because `torch._higher_order_ops.scan` lacks `vmap` batching rules over inner `grad`.

Run directly with `python -m examples.comparisons.ttt.ttt_torch`.
"""

import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from examples.comparisons.ttt.ttt_common import (
    EVAL_TASKS, HIDDEN, META_LR, META_STEPS, QUERY0, TASKS, TTT_LR0, VOCAB,
    make_tasks, query_nll_numpy, report,
)


def xent(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -torch.log_softmax(logits, -1)[..., target]


def query_xent(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, -1)
    picked = torch.gather(logp, -1, target[..., None])[..., 0]
    return -picked[..., QUERY0:].mean()


class MLP(nn.Module):
    """Predicts next-token logits from a single token (`token -> logits`)."""

    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.embed = nn.Parameter(0.3 * torch.randn(vocab, hidden))
        self.w = nn.Parameter(0.3 * torch.randn(hidden, hidden) / (hidden ** 0.5))
        self.out = nn.Parameter(0.1 * torch.randn(hidden, vocab))

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.embed[token] @ self.w) @ self.out


class RNN(nn.Module):
    """Predicts next-token logits with explicit carry (`(carry, token) -> (carry, logits)`)."""

    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.embed = nn.Parameter(0.3 * torch.randn(vocab, hidden))
        self.wh = nn.Parameter(0.3 * torch.randn(hidden, hidden) / (hidden ** 0.5))
        self.out = nn.Parameter(0.1 * torch.randn(hidden, vocab))

    def init_carry(self) -> torch.Tensor:
        return torch.zeros(self.wh.shape[0], device=self.wh.device)

    def forward(self, carry: torch.Tensor,
                token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.embed[token] + self.wh @ carry)
        return h, h @ self.out


def _mangle(name: str) -> str:
    return name.replace('.', '/')


class TTT(nn.Module):
    """Wraps unary or carry-based `nn.Module`s in online gradient adaptation."""

    def __init__(self, model: nn.Module, loss_fn, lr0: float):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.has_carry = hasattr(model, 'init_carry')
        self.lr = nn.ParameterDict({
            _mangle(k): nn.Parameter(torch.full_like(v, lr0))
            for k, v in model.named_parameters()
        })

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        fast = dict(self.model.named_parameters())
        carry = self.model.init_carry() if self.has_carry else ()
        prev = tokens[0]
        preds = []
        for t in range(tokens.shape[0]):
            token = tokens[t]

            def selfsup(w):
                if self.has_carry:
                    next_carry, out = functional_call(self.model, w, (carry, prev))
                else:
                    next_carry, out = (), functional_call(self.model, w, (prev,))
                return self.loss_fn(out, token), (next_carry, out)

            grads, (carry, out) = grad(selfsup, has_aux=True)(fast)
            fast = {k: w - self.lr[_mangle(k)] * grads[k] for k, w in fast.items()}
            prev = token
            preds.append(out)
        return torch.stack(preds)


class Batched(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return vmap(self.module)(batch)


def run_variant(variant: str, inner: nn.Module, tokens: torch.Tensor,
                held_out: torch.Tensor) -> None:
    batched = Batched(TTT(inner, xent, TTT_LR0))
    with torch.no_grad():
        preds0 = batched(held_out).detach().cpu().numpy()

    opt = torch.optim.Adam(batched.parameters(), lr=META_LR)
    losses = []
    for step_idx in range(META_STEPS):
        batch = tokens[step_idx]
        loss = query_xent(batched(batch), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        preds1 = batched(held_out).detach().cpu().numpy()

    n_weights = sum(p.numel() for p in batched.parameters())
    report(
        'torch', variant, n_weights, bool(np.all(np.isfinite(losses))),
        query_nll_numpy(preds0, held_out.numpy()),
        query_nll_numpy(preds1, held_out.numpy()),
    )


def main() -> None:
    tokens = torch.as_tensor(
        make_tasks(np.random.RandomState(0), META_STEPS * TASKS).reshape(META_STEPS, TASKS, -1),
        dtype=torch.long,
    )
    held_out = torch.as_tensor(
        make_tasks(np.random.RandomState(99), EVAL_TASKS), dtype=torch.long
    )
    torch.manual_seed(0)
    run_variant('mlp', MLP(VOCAB, HIDDEN), tokens, held_out)
    torch.manual_seed(0)
    run_variant('rnn', RNN(VOCAB, HIDDEN), tokens, held_out)


if __name__ == '__main__':
    main()
