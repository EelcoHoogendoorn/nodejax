"""Qualitative comparison of recurrent cells in the meta-controller.

Runs the meta-controller setup (mandated PID bank with plain de/dt,
persisted RLS plant identifier feeding beliefs forward, Meta-SGD
inner tuning, domain-randomized plants) with a
range of recurrent cells, and writes one per-plant trajectory figure
per variant to plots/cells_<name>.png plus a summary table to stdout.

The min-cells (minGRU, diagonal LRU) are less expressive per layer —
no hidden-to-hidden mixing matrix — so each also runs at double depth
to compensate.

Run directly:  python -m examples.cell_comparison
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax.transforms.learning import learned_sgd
from nodejax import (
    KeyStream, Leaf, Node, PNode, batch, finetune, nn, node, scan, stack,
    train_step, trained,
)
from nodejax.struct import Struct
import examples.test_meta_controller as mc


@node(name='mingru')
def MinGRU(hidden: int, tanh_candidate: bool) -> Node:
    """minGRU: gate and candidate depend on the input alone (no
    hidden-to-hidden matrix), so the temporal gradient is a product of
    (1 - z) factors — stable by construction. The form has a
    linear candidate; the tanh variant bounds it."""
    def param(rng: KeyStream) -> Struct:
        def w() -> jax.Array:
            return 0.5 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden)
        return Struct(wz=w(), bz=jnp.zeros(hidden), wh=w(), bh=jnp.zeros(hidden))

    def init(node, param: Struct) -> jax.Array:
        return jnp.zeros_like(node.input)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        z = jax.nn.sigmoid(input @ param.wz + param.bz)
        hc = input @ param.wh + param.bh
        if tanh_candidate:
            hc = jnp.tanh(hc)
        h = (1 - z) * state + z * hc
        return h, h

    return Leaf(apply, init=init, param=param, apply_input_spec=jnp.zeros(mc.HIDDEN))


@node
def LRU(hidden: int) -> Node:
    """Diagonal leaky-integrator units: per-unit poles sigmoid-bounded
    in (0, 1) — gradient-stable by construction, mixing only through
    depth."""
    def param(rng: KeyStream) -> Struct:
        return Struct(p=jax.random.uniform(rng.next(), (hidden,), minval=1.0, maxval=3.0),
                      wx=0.5 * jax.random.normal(rng.next(), (hidden,)),
                      b=jnp.zeros(hidden))

    def init(node, param: Struct) -> jax.Array:
        return jnp.zeros_like(node.input)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        a = jax.nn.sigmoid(param.p)
        h = a * state + (1 - a) * jnp.tanh(param.wx * input + param.b)
        return h, h

    return Leaf(apply, init=init, param=param, apply_input_spec=jnp.zeros(mc.HIDDEN))


def task_with_cell(cell: Node, layers: int) -> Node:
    """The meta-controller task node with the given recurrent core."""
    from nodejax import externalize, observed_loop, at
    pipe = at(mc.Filters(mc.DT), 'error') >> mc.flat \
        >> mc.Up(3 + mc.ORDER + 1, mc.HIDDEN) \
        >> stack(cell, n=layers) >> mc.Readout(mc.HIDDEN) \
        >> mc.identified(mc.Motor(mc.DT), mc.ORDER)
    rollout = scan(observed_loop(pipe, belief_spec=jnp.zeros(mc.ORDER + 1)),
                   boundary='episode')
    return externalize(
        rollout, 'motor', at_init=mc.Motor(mc.DT).parameterize().param)


def run(name: str, cell: Node, layers: int) -> None:
    task = task_with_cell(cell, layers)
    adapt = finetune(train_step(task, mc.mse, learned_sgd(mc.INNER_LR0)))
    trainer = train_step(batch(adapt), mc.mse, optax.adam(mc.META_LR)).parameterize(
        rng=jax.random.PRNGKey(0))
    start = trainer.param.objective.model    # the un-meta-trained inits

    _, train_tasks, q_t = mc.make_tasks(np.random.RandomState(0),
                                        mc.META_STEPS * mc.TASKS, k=mc.K)

    def fold(x):
        return jax.tree.map(lambda a: a.reshape(mc.META_STEPS, mc.TASKS, *a.shape[1:]), x)

    folded = fold(train_tasks)
    final, aux = trained(trainer).apply(
        support=folded.support,
        query=folded.query,
        target=fold(q_t),
    )

    plant, tasks, q_t = mc.make_tasks(np.random.RandomState(99), mc.TASKS, k=mc.K)
    # three probes on FRESH tasks, isolating what meta-training bought:
    # 1) the method itself: meta-learned inits, adapted per task on its
    #    support episode, scored on the query
    _, meta_adapted = final.apply(bundle=tasks)
    # 2) the inits alone: the meta-learned weights predict the query with
    #    NO per-task adaptation (the recurrent cells still get their
    #    fresh start state)
    init_model = batch(task).with_input(tasks.query).bind(final.param.objective.model)
    _, init_only = init_model.initialize()(tasks.query)
    # 3) adaptation alone: the same inner loop from RANDOM inits, meta-
    #    training ablated
    adapt_only = batch(adapt).bind(start).apply(bundle=tasks)

    settled = mc.mse(meta_adapted[:, mc.T // 2:], q_t[:, mc.T // 2:])
    bias = jnp.mean(meta_adapted[:, -20:] - q_t[:, -20:], axis=1)
    n_weights = sum(x.size for x in jax.tree.leaves(final.param.objective.model))
    mc.plot_tuning(f'cells_{name}.png',
                   f'{name}: {layers} layers, {n_weights} weights | '
                   f'settled mse {settled:.5f}, worst bias {jnp.max(jnp.abs(bias)):.3f}',
                   plant, q_t, meta_adapted, init_only, adapt_only)
    print(f'{name:16s} L={layers} weights={n_weights:4d}: '
          f'finite={bool(jnp.all(jnp.isfinite(aux.loss)))} '
          f'adapted {mc.mse(meta_adapted, q_t):.4f} unadapted {mc.mse(init_only, q_t):.4f} '
          f'settled {settled:.5f} worst|bias| {jnp.max(jnp.abs(bias)):.4f}')


def main() -> None:
    run('elman-2', mc.RNN(mc.HIDDEN), 2)
    run('gru-2', nn.GRU(mc.HIDDEN), 2)
    run('mingru-tanh-2', MinGRU(mc.HIDDEN, tanh_candidate=True), 2)
    run('mingru-tanh-4', MinGRU(mc.HIDDEN, tanh_candidate=True), 4)
    run('mingru-lin-2', MinGRU(mc.HIDDEN, tanh_candidate=False), 2)
    run('mingru-lin-4', MinGRU(mc.HIDDEN, tanh_candidate=False), 4)
    run('lru-2', LRU(mc.HIDDEN), 2)
    run('lru-4', LRU(mc.HIDDEN), 4)


if __name__ == '__main__':
    main()
