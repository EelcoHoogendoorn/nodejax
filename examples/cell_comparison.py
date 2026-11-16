"""Compares recurrent cell families inside the indirect-adaptive meta-controller benchmark.

Runs the meta-controller setup (mandated PID bank with plain `de/dt`, persisted RLS plant identifier feeding beliefs across episodes, `learned_sgd` inner tuning, and domain-randomized plants) across Elman RNN, GRU, `MinGRU`, and diagonal `LRU` cores. Because `MinGRU` and `LRU` omit hidden-to-hidden mixing matrices, each also runs at double depth. Writes per-plant trajectory figures to `plots/cells_<name>.png` and prints settled tracking metrics to stdout.

Run directly via `python -m examples.cell_comparison`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax.transforms.learning import learned_sgd
from nodejax import (
    Leaf, Node, batch, finetune, nn, node, scan, stack,
    train_step, trained,
)
from nodejax.struct import Struct
import examples.test_meta_controller as mc


@node(name='mingru')
def MinGRU(hidden: int, tanh_candidate: bool) -> Node:
    """Advances a `MinGRU` cell whose gate and candidate depend on `input` alone, optionally bounding the candidate with `tanh`."""
    def param(rng) -> Struct:
        def weight() -> jax.Array:
            return 0.5 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden)
        return Struct(wz=weight(), bz=jnp.zeros(hidden), wh=weight(), bh=jnp.zeros(hidden))

    def init(node, param) -> jax.Array:
        return jnp.zeros_like(node.input)

    def apply(param, state, input) -> tuple[jax.Array, jax.Array]:
        gate = jax.nn.sigmoid(input @ param.wz + param.bz)
        candidate = input @ param.wh + param.bh
        if tanh_candidate:
            candidate = jnp.tanh(candidate)
        next_state = (1 - gate) * state + gate * candidate
        return next_state, next_state

    return Leaf(apply, init=init, param=param, apply_input_spec=jnp.zeros(mc.HIDDEN))


@node
def LRU(hidden: int) -> Node:
    """Advances diagonal leaky-integrator units with sigmoid-bounded poles in `(0, 1)`."""
    def param(rng) -> Struct:
        return Struct(p=jax.random.uniform(rng.next(), (hidden,), minval=1.0, maxval=3.0),
                      wx=0.5 * jax.random.normal(rng.next(), (hidden,)),
                      b=jnp.zeros(hidden))

    def init(node, param) -> jax.Array:
        return jnp.zeros_like(node.input)

    def apply(param, state, input) -> tuple[jax.Array, jax.Array]:
        pole = jax.nn.sigmoid(param.p)
        next_state = pole * state + (1 - pole) * jnp.tanh(param.wx * input + param.b)
        return next_state, next_state

    return Leaf(apply, init=init, param=param, apply_input_spec=jnp.zeros(mc.HIDDEN))


def task_with_cell(cell: Node, layers: int) -> Node:
    """Builds the meta-controller closed-loop task `Node` around `layers` stacked copies of `cell`."""
    from nodejax import externalize, observed_loop, at
    pipe = at(mc.Filters(mc.DT), 'error') >> mc.flat \
        >> mc.Up(3 + mc.ORDER + 1, mc.HIDDEN) \
        >> stack(cell, n=layers) >> mc.Readout(mc.HIDDEN) \
        >> mc.identified(mc.Motor(mc.DT), mc.ORDER)
    rollout = scan(observed_loop(pipe, belief_spec=jnp.zeros(mc.ORDER + 1)),
                   boundary='episode')
    return externalize(rollout, 'motor')


def run(name: str, cell: Node, layers: int) -> None:
    task = task_with_cell(cell, layers)
    adapt = finetune(train_step(task, mc.mse, learned_sgd(mc.INNER_LR0)))
    trainer = train_step(batch(adapt), mc.mse, optax.adam(mc.META_LR)).parameterize(
        rng=jax.random.PRNGKey(0))
    start = trainer.param.objective.model

    _, train_tasks, query_targets = mc.make_tasks(
        np.random.RandomState(0), mc.META_STEPS * mc.TASKS, k=mc.K)

    def fold(tree):
        return jax.tree.map(
            lambda array: array.reshape(mc.META_STEPS, mc.TASKS, *array.shape[1:]),
            tree,
        )

    folded = fold(train_tasks)
    final, aux = trained(trainer).apply(
        support=folded.support,
        query=folded.query,
        target=fold(query_targets),
    )

    plant, tasks, query_targets = mc.make_tasks(
        np.random.RandomState(99), mc.TASKS, k=mc.K)
    # Evaluate three probes on fresh tasks: meta-adapted, zero-shot meta-init, and inner adaptation from random init.
    _, meta_adapted = final.apply(bundle=tasks)
    init_model = batch(task).with_input(tasks.query).bind(final.param.objective.model)
    _, init_only = init_model.initialize()(tasks.query)
    adapt_only = batch(adapt).bind(start).apply(bundle=tasks)

    settled = mc.mse(meta_adapted[:, mc.T // 2:], query_targets[:, mc.T // 2:])
    bias = jnp.mean(meta_adapted[:, -20:] - query_targets[:, -20:], axis=1)
    n_weights = sum(leaf.size for leaf in jax.tree.leaves(final.param.objective.model))
    mc.plot_tuning(f'cells_{name}.png',
                   f'{name}: {layers} layers, {n_weights} weights | '
                   f'settled mse {settled:.5f}, worst bias {jnp.max(jnp.abs(bias)):.3f}',
                   plant, query_targets, meta_adapted, init_only, adapt_only)
    print(f'{name:16s} L={layers} weights={n_weights:4d}: '
          f'finite={bool(jnp.all(jnp.isfinite(aux.loss)))} '
          f'adapted {mc.mse(meta_adapted, query_targets):.4f} unadapted {mc.mse(init_only, query_targets):.4f} '
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
