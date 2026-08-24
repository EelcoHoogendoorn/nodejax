"""A generic test-time-training wrapper in idiomatic PyTorch.

The eager-object exhibit, sibling to ttt_rnn_by_hand.py and
ttt_rnn_flax.py: the same task family, budget and scoring as
ttt_nodejax's ttt rows, with the same question: can TTT be
written once, over arbitrary recurrent modules, as nodejax's ttt
transform is?

The answer is yes, over a carry protocol this file declares:
init_carry() plus (carry, x) -> (carry, y), which is the node contract minus
the param slot, which torch modules keep as attributes. torch leaves
a module's call shape free, so the protocol is a local convention
rather than a library contract. TTT itself answers the
protocol. Its carry pairs the fast weights with the wrapped cell's
own, so the model composes as Batched(Scan(TTT(cell))), the torch
reading of nodejax's batch(scanned(train_step(...))).

The wrapper's interior is the boundary toll, torch's spelling of it:
outer gradients must flow through the inner updates, autograd flows
through values, and the module holds its weights as mutable
attributes, so the adapted weights leave the object and ride the
scan carry as plain name->tensor dicts. functional_call applies the
module at those values, torch.func.grad takes the per-sample inner
gradient (itself differentiable, so the outer backward sees through
it), and torch.func.vmap runs one drifting copy of the weights per
task over a module object that stores exactly one. The per-weight
rates are Parameters of the wrapper under mangled keys because registered
names reserve '.' for the module tree. The outer Adam reaches
them through the ordinary parameters() walk.

This value boundary is specific to the PyTorch formulation. The sibling NNX
implementation uses graph-aware transforms to adapt and carry a cloned Module
directly. Both are current framework idioms, and they place fast weights in
different representations.

torch, like flax, is a reference-exhibit dependency: the file runs
in any environment with torch installed.

Run directly:  python -m nodejax.examples.comparisons.ttt.ttt_rnn_torch
"""

import os

# torch and conda's mkl each ship a libomp on macOS; the flag lets one
# process host both copies so the import survives.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap

VOCAB, HIDDEN = 8, 16
STREAM, SUPPORT = 192, 128
TASKS, META_STEPS = 8, 300
TTT_LR0, META_LR = 0.05, 1e-3
CONCENTRATION = 2.0
QUERY0 = SUPPORT


def make_tasks(rs, n_tasks: int):
    logits = np.exp(CONCENTRATION * rs.standard_normal((n_tasks, VOCAB, VOCAB)))
    P = logits / logits.sum(-1, keepdims=True)
    tokens = np.zeros((n_tasks, STREAM), dtype=np.int64)
    state = rs.randint(VOCAB, size=n_tasks)
    rows = np.arange(n_tasks)
    for t in range(STREAM):
        tokens[:, t] = state
        u = rs.random(n_tasks)[:, None]
        state = (P[rows, state].cumsum(-1) > u).argmax(-1)
    return torch.tensor(tokens)


def pair(tokens: torch.Tensor):
    """The next-token pairing, spelled in the loader: the input column is
    the sequence delayed one step (primed with its own first token), the
    target column the sequence itself. This is the next_step register hand-rolled
    as data."""
    prev = torch.cat([tokens[..., :1], tokens[..., :-1]], dim=-1)
    return dict(input=prev, target=tokens)


def xent(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -torch.log_softmax(logits, -1)[..., target]


def query_xent(logits: torch.Tensor,
               target: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, -1)
    picked = torch.gather(logp, -1, target[..., None])[..., 0]
    return -picked[..., QUERY0:].mean()


class TanhCell(nn.Module):
    """ttt_nodejax's rnn predictor, torch-spelled: embed the previous
    token, tanh recurrence, the hidden state doubling as output."""

    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.embed = nn.Parameter(0.3 * torch.randn(vocab, hidden))
        self.wh = nn.Parameter(0.3 * torch.randn(hidden, hidden) / hidden ** 0.5)

    def init_carry(self):
        return torch.zeros(self.wh.shape[0])

    def forward(self, carry, token):
        h = torch.tanh(self.embed[token] + self.wh @ carry)
        return h, h


class Forecast(nn.Module):
    """Any protocol cell plus a logits head, still following the protocol."""

    def __init__(self, cell, hidden: int):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(hidden, VOCAB)

    def init_carry(self):
        return self.cell.init_carry()

    def forward(self, carry, token):
        carry, y = self.cell(carry, token)
        return carry, self.head(y)


def _mangle(name: str):
    """Cell parameter name -> ParameterDict key: registered names
    reserve '.' for the module tree, so the rates ride under '/'."""
    return name.replace('.', '/')


class TTT(nn.Module):
    """Test-time training over any carry-protocol module. It is itself
    a carry-protocol cell, nodejax's factoring: ttt is the per-sample
    step, scan is a separate transform around it. The carry pairs the
    fast weights with the wrapped cell's own carry (two memories at
    two speeds); the cell's Parameters are the meta-trained
    initialization, the per-weight rates are this wrapper's
    Parameters.

    One step consumes one sample (x, y) and returns the prequential
    prediction: it comes from the weights as they arrived, and the
    revealed target then funds one inner gradient step, taken with
    torch.func.grad over functional_call so it stays a value
    computation the outer backward differentiates."""

    def __init__(self, cell, loss_fn, lr0: float):
        super().__init__()
        self.cell = cell
        self.loss_fn = loss_fn
        self.lr = nn.ParameterDict(
            {_mangle(k): nn.Parameter(torch.full_like(v, lr0))
             for k, v in cell.named_parameters()})

    def init_carry(self):
        return dict(self.cell.named_parameters()), self.cell.init_carry()

    def forward(self, carry, sample):
        fast, h = carry
        x, y = sample

        def selfsup(w):
            h2, out = functional_call(self.cell, w, (h, x))
            return self.loss_fn(out, y), (h2, out)

        grads, (h2, out) = grad(selfsup, has_aux=True)(fast)
        # OUT OF FRAMEWORK: the inner rule is hand arithmetic over the fast
        # dict; torch.optim advances Parameters in place and cannot touch
        # these values, so no stock optimizer can be the inner loop. The
        # nodejax row hands the same rule in as an argument
        fast2 = {k: w - self.lr[_mangle(k)] * grads[k] for k, w in fast.items()}
        return (fast2, h2), out


class Scan(nn.Module):
    """nodejax's scan transform: a protocol cell applied down the time
    axis, the carry threaded by rebinding a python local. One sequence
    element per step. Here that is the (input, target) sample pair, TTT's
    sequence contract.

    torch ships a lax.scan-shaped prototype for compiled graph
    capture, torch._higher_order_ops.scan; measured against this
    cell, it differentiates through the inner grad steps on a single
    batch element and fails under torch.func.vmap (its traced body meets
    missing batching rules). The python loop is the spelling that
    composes with vmap, grad and backward in eager torch, where
    autograd records the unrolled tape either way."""

    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, input, target):
        carry = self.cell.init_carry()
        preds = []
        for t in range(input.shape[0]):
            carry, out = self.cell(carry, (input[t], target[t]))
            preds.append(out)
        return torch.stack(preds)


class Batched(nn.Module):
    """nodejax's batch transform, vmap-shaped: the task axis mapped,
    module Parameters broadcast: every batch element reads the same
    initialization and diverges only in the fast weights riding
    Scan's carry."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, target):
        return vmap(self.module)(input, target)


def train_step(model, opt, batch: dict[str, torch.Tensor]):
    """One outer update: prequential forecasts for a task batch,
    query-region cross-entropy, one Adam step. Backward runs through every
    inner torch.func.grad step of the meta batch, including the higher-order
    path, and opt.step() advances the Parameters in place."""
    preds = model(batch['input'], batch['target'])
    loss = query_xent(preds, batch['target'])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def run(name: str, model):
    batched = Batched(model)
    opt = torch.optim.Adam(batched.parameters(), lr=META_LR)

    train = pair(make_tasks(np.random.RandomState(0), META_STEPS * TASKS))
    fold = lambda a: a.reshape(META_STEPS, TASKS, *a.shape[1:])
    inp, tgt = fold(train['input']), fold(train['target'])
    losses = [train_step(batched, opt, dict(input=inp[s], target=tgt[s]))
              for s in range(META_STEPS)]

    tasks = pair(make_tasks(np.random.RandomState(99), TASKS))
    with torch.no_grad():
        preds = batched(tasks['input'], tasks['target'])
    n = sum(p.numel() for p in model.parameters())
    print(f'{name:16s} weights={n:5d} '
          f'finite={bool(np.all(np.isfinite(losses)))} '
          f'meta loss {np.mean(losses[:20]):.2f} -> {np.mean(losses[-20:]):.2f} '
          f'query xent {query_xent(preds, tasks["target"]):.2f}', flush=True)
    return losses


def main() -> None:
    torch.manual_seed(0)
    losses = run('ttt(TanhCell)', Scan(TTT(Forecast(TanhCell(VOCAB, HIDDEN), HIDDEN),
                                           xent, TTT_LR0)))
    assert np.mean(losses[-20:]) < np.mean(losses[:20]), 'meta loss fell'


if __name__ == '__main__':
    main()


# STATE CENSUS. State lives in three homes here: (1) module
# Parameters, including the cell's initialization and the wrapper's rates,
# advanced in place by opt.step(); (2) optimizer state, including Adam's
# moments, a dict keyed on those same Parameter objects, advanced by
# the same call; (3) the scan carry, including the fast-weight name->tensor
# dict paired with the cell's hidden state, threaded functionally by
# rebinding a python local down Scan's time loop, one drifting copy
# per task under vmap, alive for the duration of a forward pass.
# The by-hand file holds 3 as one explicit lax.scan carry and 1 and 2
# as the (theta, opt_state) value pair; nodejax's census of the same
# model is one state slot, with hidden state, fast weights and
# optimizer moments as fields of it.
