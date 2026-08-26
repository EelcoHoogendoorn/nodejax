"""Deeply nested composition, in torch: the same one-tree tower as
`tower_nodejax.py`, a residual stacked RNN committee adapted per task
inside the meta-training loop, eager modules with torch.func at the
gradient boundaries.

Same model, task, and budget. The module half is torch at its most
comfortable: Parameters as attributes, a ModuleList per axis of
structure, time and depth as python loops whose rebound locals ARE the
state routing, autograd recording the unrolled tape. The census: the
time loop rebinds (per-layer hiddens, norm stats) by hand; the depth
loop rebinds the signal; the committee is a ModuleList whose stacked
view is hand arithmetic, torch.stack over aligned named_parameters,
differentiable so the meta gradient reaches every member.

The meta level is where eager objects stop being enough. Adaptation
must produce NEW weights while gradients still flow to the originals,
and torch.optim advances Parameters in place, so no stock optimizer
can be the inner loop: the fast weights leave the object and ride as
name->tensor dicts, functional_call applies a member at those values,
torch.func.grad takes the differentiable inner step, and
torch.func.vmap runs the member and task axes over modules that store
exactly one net. The outer loop is idiomatic torch again, python steps
around loss.backward() and Adam.

THE 60 SECONDS ARE STRUCTURAL, and torch.compile does not buy them
back at this budget (measured, torch 2.13): the whole meta_loss
compiles fullgraph, torch.func stack included, and the per-step cost
drops 123 ms to 4 ms, competitive with the jax columns. But tracing
charges by the op, the python loops ARE the ops, and the unrolled
tape (time x depth x members x inner steps) takes 266 s to compile:
worse than just running eager for anything under ~2000 meta steps.
The jax files pay tracing per scan BODY, once, which is why their
whole run jits in seconds. torch does ship a lax.scan-shaped
prototype, torch._higher_order_ops.scan, measured here against this
tower's composition: forward eager runs, torch.func.grad through it
fails on compile-machinery internals, and vmap over it has no
batching rule, while the meta level here is grad inside vmap through
the scan. No usable scan, no small graph; the file stays eager, which
is the honest spelling of that.

torch, like flax, is a reference-exhibit dependency: the file runs in
any environment with torch installed.

Run directly:  python -m examples.comparisons.tower.tower_torch
"""

import os

# torch and conda's mkl each ship a libomp on macOS; the flag lets one
# process host both copies so the import survives.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import jax
import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from examples.comparisons.tower.tower_common import (
    HIDDEN, LAYERS, MEMBERS, META_STEPS, INNER_LR, OUTER_LR, MOMENTUM,
    make_tasks)
from nodejax.core.types import PyTree


def norm_step(stats: PyTree, x: jax.Array):
    m1, var = stats
    out = (x - m1) / torch.sqrt(var + 1e-5)
    new_m1 = (1 - MOMENTUM) * m1 + MOMENTUM * x
    new_var = (1 - MOMENTUM) * var + MOMENTUM * (x - m1) ** 2
    return (new_m1, new_var), out


class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.wx = nn.Parameter(0.5 * torch.randn(HIDDEN))
        self.wh = nn.Parameter(0.3 * torch.randn(HIDDEN, HIDDEN) / HIDDEN ** 0.5)
        self.b = nn.Parameter(torch.zeros(HIDDEN))

    def forward(self, h, x):
        h_new = torch.tanh(self.wx * x + self.wh @ h + self.b)
        return h_new, h_new


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_w = nn.Parameter(0.5 * torch.randn(HIDDEN))
        self.cells = nn.ModuleList(Cell() for _ in range(LAYERS))
        self.ro_w = nn.Parameter(0.1 * torch.randn(HIDDEN))
        self.ro_b = nn.Parameter(torch.zeros(()))

    def forward(self, xs):
        """The rollout: time and depth are python loops, the per-layer
        hiddens and the norm stats are rebound locals, autograd recording
        the unrolled tape."""
        hiddens = [torch.zeros(HIDDEN) for _ in self.cells]
        stats = (torch.zeros(HIDDEN), torch.ones(HIDDEN))
        preds = []
        for step in range(xs.shape[0]):
            stats, signal = norm_step(stats, self.up_w * xs[step])
            for depth, (cell, h) in enumerate(zip(self.cells, hiddens)):
                h_new, y = cell(h, signal)
                # the residual: one line in this loop, nothing marks it
                # as structure
                signal = signal + y
                hiddens[depth] = h_new
            preds.append(self.ro_w @ signal + self.ro_b)
        return torch.stack(preds)


class Committee(nn.Module):
    """MEMBERS independent Nets in a ModuleList; the mean of their
    rollouts is the committee's answer. `stacked` is the functional view
    the meta level consumes: torch.stack over aligned named_parameters,
    differentiable, so the meta gradient flows back into every member's
    Parameters, where the jax files spell the same committee as a vmapped
    constructor."""

    def __init__(self):
        super().__init__()
        self.members = nn.ModuleList(Net() for _ in range(MEMBERS))

    def stacked(self):
        names = [name for name, _ in self.members[0].named_parameters()]
        return {name: torch.stack([member.get_parameter(name)
                                   for member in self.members])
                for name in names}


def meta_loss(committee, sup_x, sup_y, qry_x, qry_y) -> jax.Array:
    """MAML over the committee, torch.func-spelled: the fast weights ride
    as name->tensor dicts because torch.optim advances Parameters in
    place and cannot be the inner loop; each inner step is a
    differentiable torch.func.grad the outer backward sees through."""
    template = committee.members[0]
    init = committee.stacked()

    def committee_pred(stacked, xs):
        # each member rolls the sequence; the mean is the committee's
        # answer, and the population DIES here, no aux stream existing
        # to sow it on
        population = vmap(
            lambda member: functional_call(template, member, (xs,)))(stacked)
        return population.mean(0)

    def per_task(sx, sy, qx, qy):
        fast = init
        for step in range(sx.shape[0]):
            def support_loss(stacked):
                return ((committee_pred(stacked, sx[step]) - sy[step]) ** 2).mean()
            grads = grad(support_loss)(fast)
            fast = {name: weight - INNER_LR * grads[name]
                    for name, weight in fast.items()}
        return ((committee_pred(fast, qx) - qy) ** 2).mean()

    return vmap(per_task)(sup_x, sup_y, qry_x, qry_y).mean()


def main() -> None:
    torch.manual_seed(0)
    committee = Committee()
    opt = torch.optim.Adam(committee.parameters(), lr=OUTER_LR)

    tasks = [torch.tensor(np.asarray(leaf))
             for leaf in make_tasks(jax.random.PRNGKey(2))]

    losses = []
    for _ in range(META_STEPS):
        # the meta loop is eager python: one tape per step, built and
        # freed, nothing compiled across steps
        loss = meta_loss(committee, *tasks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"tower loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.3 * losses[0]
    return losses[-1]


if __name__ == '__main__':
    main()
