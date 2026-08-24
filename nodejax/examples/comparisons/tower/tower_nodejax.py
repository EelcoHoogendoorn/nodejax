"""Deeply nested composition, in nodejax: a residual stacked RNN,
scanned over time, widened into a committee, adapted per task and
meta-trained over a task family, one tree. Each of those is an axis
whose state some transform must route, and this file exists to show,
beside `tower_flax.py`, `tower_equinox.py` and `tower_torch.py`,
where each framework makes you write that routing.

The task: predict an exponential moving average of a noisy scalar
sequence, chosen because the target is recurrent by construction. The
model: an input projection, a running Norm (its stats as
ordinary cyclic state — the change-cost measurement: one pipe term
here, hand-threaded carries in the flax file), a stack of RNN cells,
a linear Readout.
The tower: `residual` wraps each cell, `stack` scans over depth,
`scanned` internalizes the time axis, `ensemble` widens the rollout
into a committee whose mean head sows the population, `finetune`
makes the committee's whole adaptation the inner loop, `batch` maps it over
tasks, `train_step` meta-trains what the adaptation starts from, and
`trained` is the run over the task sequence. Eight transform
applications, one line each, no routing annotations anywhere: each
transform reads which tree is params and which is state off the
contract, and the meta gradient is second order, flowing through the
inner run.

Side by side with `tower_flax.py`, `tower_equinox.py` and
`tower_torch.py`: the same model, task, and budget, with the state
routing written the way each framework wants it written.

Run directly:  python -m nodejax.examples.comparisons.tower.tower_nodejax
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import (Node, node, Leaf, stack, residual, ensemble, reduce, scanned,
                     trained, batch, train_step, finetune, split_aux)
from nodejax.struct import Struct
from nodejax.examples.comparisons.tower.tower_common import (
    HIDDEN, LAYERS, MEMBERS, INNER_LR, OUTER_LR, META_STEPS, MOMENTUM,
    make_tasks)

@node
def Up(hidden: int) -> Node:
    def param(rng):
        return Struct(w=0.5 * jax.random.normal(rng.next(), (hidden,)))
    def apply(param, input):
        return param.w * input
    return Leaf(apply, param=param)


@node
def Norm(hidden: int, momentum: float) -> Node:
    """Streaming Norm: per-feature running mean and variance over TIME,
    ordinary cyclic state. Every transform in the tower routes it off
    the contract: scan carries it, batch gives each sequence its own
    stats, train_step holds it in model state, finetune leaves it to
    its own dynamics while the gradient step adapts params."""
    def init():
        return (jnp.zeros(hidden), jnp.ones(hidden))
    def apply(state, input):
        m1, var = state
        out = (input - m1) / jnp.sqrt(var + 1e-5)
        m1_new = (1 - momentum) * m1 + momentum * input
        var_new = (1 - momentum) * var + momentum * (input - m1) ** 2
        return (m1_new, var_new), out
    return Leaf(apply, init=init)


@node
def RNN(hidden: int) -> Node:
    def param(rng):
        return Struct(wx=0.5 * jax.random.normal(rng.next(), (hidden,)),
                      wh=0.3 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
                      b=jnp.zeros(hidden))
    def init(param):
        return jnp.zeros(param.wh.shape[0])
    def apply(param, state, input):
        h = jnp.tanh(param.wx * input + param.wh @ state + param.b)
        return h, h
    return Leaf(apply, init=init, param=param)


@node
def Readout(hidden: int) -> Node:
    def param(rng):
        return Struct(w=0.1 * jax.random.normal(rng.next(), (hidden,)), b=jnp.zeros(()))
    def apply(param, input):
        return param.w @ input + param.b
    return Leaf(apply, param=param)


def mse(out: jax.Array, target: jax.Array) -> jax.Array:
    """Scores the mean; the sown population passes through the split."""
    pred, _ = split_aux(out)
    return jnp.mean((pred - target) ** 2)


def tower() -> Node:
    """The whole program, one tree: residual cells stacked over depth, the
    member rolled over time, the rollouts widened into a committee with a
    mean head and its population sown, the committee's WHOLE ADAPTATION as
    the inner loop, batched over tasks, meta-trained over a sequence of task
    batches, and the run over all of it. Second-order gradients flow
    through every level, and no line here knows which level it serves.

    The other ordering, scanned(ensemble(member) >> reduce), commutes
    exactly (verified: same params from the same key, identical outputs);
    members stepping in lockstep and members rolling independently are the
    same computation when nothing couples them inside the step. This one
    reads as what the rivals spell: each member rolls, the committee
    averages rollouts, and the sown population arrives (MEMBERS, T)."""
    rnn = stack(residual(RNN(HIDDEN)), n=LAYERS)
    member = Up(HIDDEN) >> Norm(HIDDEN, MOMENTUM) >> rnn >> Readout(HIDDEN)
    committee = ensemble(scanned(member), n=MEMBERS) >> reduce(jnp.mean)
    adapt = finetune(train_step(committee, mse, optax.sgd(INNER_LR)))
    return trained(train_step(batch(adapt), mse, optax.adam(OUTER_LR)))


def main() -> None:
    run = tower().parameterize(rng=jax.random.PRNGKey(1))

    sup_x, sup_y, qry_x, qry_y = make_tasks(jax.random.PRNGKey(2))
    tasks = Struct(support=Struct(input=sup_x, target=sup_y), query=qry_x)
    tile = lambda leaf: jnp.broadcast_to(leaf, (META_STEPS, *leaf.shape))
    _, aux = run.apply(input=jax.tree.map(tile, tasks), target=tile(qry_y))

    print(f"tower loss {aux.loss[0]:.4f} -> {aux.loss[-1]:.4f}")
    assert aux.loss[-1] < 0.3 * aux.loss[0]
    return float(aux.loss[-1])


if __name__ == '__main__':
    main()
