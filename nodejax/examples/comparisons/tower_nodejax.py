"""Deeply nested composition, in nodejax: a stacked RNN, scanned over
time, batched over sequences, trained over a stream, and, in a meta
variant, adapted per task inside the training loop. Each of those is
an axis whose state some transform must route, and this file exists to
show, beside `tower_flax.py` and `tower_equinox.py`, where each
framework makes you write that routing.

The task: predict an exponential moving average of a noisy scalar
sequence, chosen because the target is recurrent by construction. The
model: an input projection, a stack of RNN cells, a linear readout.
The tower: `stack` scans over depth, `scan` internalizes the time
axis, `batch` maps over sequences, `train_step` internalizes the
optimization scan. Four transform applications, one line each, no
routing annotations anywhere: each transform reads which tree is
params and which is state off the contract. The meta variant composes
`finetune` between rollout and trainer: an inner adaptation scan, a
fifth transform, second-order gradients, and a one-line difference
between the plain and the meta tower.

Side by side with `tower_flax.py` and `tower_equinox.py`: the same
model, task, and budget, with the state routing written the way each
framework wants it written.

Run directly:  python -m nodejax.examples.comparisons.tower_nodejax
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import node_def, stack, scan, batch, train_step, finetune
from nodejax.struct import Struct
from nodejax.examples.comparisons.tower_common import (
    HIDDEN, LAYERS, STEPS, INNER_LR, META_STEPS, mse, make_data, make_tasks)

def up(hidden):
    def param(rng):
        return Struct(w=0.5 * jax.random.normal(rng.next(), (hidden,)))
    def apply(param, input):
        return param.w * input
    return node_def(apply, param=param, name='up')


def rnn(hidden):
    def param(rng):
        return Struct(wx=0.5 * jax.random.normal(rng.next(), (hidden,)),
                      wh=0.3 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
                      b=jnp.zeros(hidden))
    def init(param):
        return jnp.zeros(param.wh.shape[0])
    def apply(param, state, input):
        h = jnp.tanh(param.wx * input + param.wh @ state + param.b)
        return h, h
    return node_def(apply, init=init, param=param, name='rnn')


def readout(hidden):
    def param(rng):
        return Struct(w=0.1 * jax.random.normal(rng.next(), (hidden,)), b=jnp.zeros(()))
    def apply(param, input):
        return param.w @ input + param.b
    return node_def(apply, param=param, name='readout')


def main():
    net = up(HIDDEN) >> stack(rnn(HIDDEN), n=LAYERS) >> readout(HIDDEN)
    rollout = scan(net)                       # time internalized
    trainer = train_step(batch(rollout), mse, optax.adam(3e-3))

    xs, ys = make_data(jax.random.PRNGKey(0))
    model = batch(rollout).parameterize(rng=jax.random.PRNGKey(1))
    tile = lambda a: jnp.broadcast_to(a, (STEPS, *a.shape))
    state, losses = jax.jit(trainer.scan)(trainer.init(model=model.param),
                                          Struct(input=tile(xs), target=tile(ys)))

    print(f"tower loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.05 * losses[0]
    return float(losses[-1])


# --- one level deeper: meta-learning across a task family ---

def main_meta():
    net = up(HIDDEN) >> stack(rnn(HIDDEN), n=LAYERS) >> readout(HIDDEN)
    rollout = scan(net)
    adapt = finetune(rollout, mse, optax.sgd(INNER_LR))     # the inner adaptation scan
    maml = train_step(batch(adapt), mse, optax.adam(3e-3))

    sup_x, sup_y, qry_x, qry_y = make_tasks(jax.random.PRNGKey(2))
    tasks = Struct(support=Struct(input=sup_x, target=sup_y), query=qry_x)
    model = batch(adapt).parameterize(rng=jax.random.PRNGKey(1))
    tile = lambda a: jnp.broadcast_to(a, (META_STEPS, *a.shape))
    stream = Struct(input=jax.tree.map(tile, tasks), target=tile(qry_y))
    state, losses = jax.jit(maml.scan)(maml.init(model=model.param), stream)

    print(f"meta  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.3 * losses[0]
    return float(losses[-1])


if __name__ == '__main__':
    main()
    main_meta()
