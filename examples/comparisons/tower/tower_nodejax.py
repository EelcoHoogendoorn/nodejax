"""Eight-level nested composition in NodeJAX (`tower` benchmark).

Alongside `tower_nnx.py`, `tower_nnx_reusable.py`, `tower_equinox.py`, `tower_keras.py`, and `tower_torch.py`, this module builds a residual RNN cell, scans over depth (`stack`), unrolls over time (`scanned`) behind a stateful normalizer (`Norm`), widens into a committee (`ensemble` + `reduce`), adapts per task (`finetune(train_step(...))`), maps over task batches (`batch`), and meta-trains (`trained(train_step(...))`). Every transform reads `param` and `state` roles off the `Node` contract and returns another `Node`, keeping the entire meta-learner in five lines of composition.

Run directly with `python -m examples.comparisons.tower.tower_nodejax`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nodejax import (Node, node, Leaf, stack, residual, ensemble, reduce, scanned,
                     trained, batch, train_step, finetune)
from nodejax.struct import Struct
from examples.comparisons.tower.tower_common import (
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
    """Maintains per-feature running mean and variance over time as cyclic
    state. Every transform in the tower routes it off
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
    """Computes mean squared error against `target`."""
    return jnp.mean((out - target) ** 2)


def tower() -> Node:
    """Builds the whole program as one tree: residual cells stacked over depth,
    each member rolled over time, rollouts widened into a committee with a
    mean head and its population in aux, the committee's whole adaptation as
    the inner loop, batched over tasks, meta-trained over a sequence of task
    batches, and the run over all of it. Second-order gradients flow
    through every level, and no line here knows which level it serves.

    The other ordering, scanned(ensemble(member) >> reduce), commutes
    exactly (verified: same params from the same key, identical outputs);
    members stepping in lockstep and members rolling independently are the
    same computation when nothing couples them inside the step. This one
    reads as what the rivals spell: each member rolls, the committee
    averages rollouts, and the aux population arrives (MEMBERS, T)."""
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
    episodes = jax.tree.map(tile, tasks)
    _, aux = run.apply(
        support=episodes.support,
        query=episodes.query,
        target=tile(qry_y),
    )

    first, last = float(aux.loss[0]), float(aux.loss[-1])
    print(f"tower loss {first:.4f} -> {last:.4f}")
    assert np.isfinite(first) and np.isfinite(last)
    return last


if __name__ == '__main__':
    main()
