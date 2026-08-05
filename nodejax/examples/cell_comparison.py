"""Qualitative comparison of recurrent cells in the meta-controller.

Runs the meta-controller setup (mandated PID bank with plain de/dt,
persisted RLS plant identifier feeding beliefs forward, Meta-SGD
inner tuning, domain-randomized plants) with a
range of recurrent cells, and writes one per-plant trajectory figure
per variant to plots/cells_<name>.png plus a summary table to stdout.

The min-cells (minGRU, diagonal LRU) are less expressive per layer —
no hidden-to-hidden mixing matrix — so each also runs at double depth
to compensate.

Run directly:  python -m nodejax.examples.cell_comparison
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import Node, NodeDef, node_def, stack, scan, batch, train_step, KeyStream
from nodejax.struct import Struct
import nodejax.examples.test_meta_controller as mc


def gru_def(hidden: int) -> NodeDef:
    """The classic gated cell: update and reset gates, candidate mixed
    through a reset-gated hidden state."""
    def param(rng: KeyStream) -> Struct:
        def w() -> jax.Array:
            return 0.4 * jax.random.normal(rng.next(), (2 * hidden, hidden)) / jnp.sqrt(2 * hidden)
        return Struct(wz=w(), bz=jnp.zeros(hidden),
                      wr=w(), br=jnp.zeros(hidden),
                      wh=w(), bh=jnp.zeros(hidden))

    def init(ndef, param: Struct) -> jax.Array:
        return jnp.zeros_like(ndef.input)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        xh = jnp.concatenate([input, state])
        z = jax.nn.sigmoid(xh @ param.wz + param.bz)
        r = jax.nn.sigmoid(xh @ param.wr + param.br)
        hc = jnp.tanh(jnp.concatenate([input, r * state]) @ param.wh + param.bh)
        h = (1 - z) * state + z * hc
        return h, h

    return node_def(apply, init=init, param=param, apply_input_spec=jnp.zeros(mc.HIDDEN), name='gru')


def mingru_def(hidden: int, tanh_candidate: bool) -> NodeDef:
    """minGRU: gate and candidate depend on the input alone (no
    hidden-to-hidden matrix), so the temporal gradient is a product of
    (1 - z) factors — stable by construction. The form has a
    linear candidate; the tanh variant bounds it."""
    def param(rng: KeyStream) -> Struct:
        def w() -> jax.Array:
            return 0.5 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden)
        return Struct(wz=w(), bz=jnp.zeros(hidden), wh=w(), bh=jnp.zeros(hidden))

    def init(ndef, param: Struct) -> jax.Array:
        return jnp.zeros_like(ndef.input)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        z = jax.nn.sigmoid(input @ param.wz + param.bz)
        hc = input @ param.wh + param.bh
        if tanh_candidate:
            hc = jnp.tanh(hc)
        h = (1 - z) * state + z * hc
        return h, h

    return node_def(apply, init=init, param=param, apply_input_spec=jnp.zeros(mc.HIDDEN), name='mingru')


def lru_def(hidden: int) -> NodeDef:
    """Diagonal leaky-integrator units: per-unit poles sigmoid-bounded
    in (0, 1) — gradient-stable by construction, mixing only through
    depth."""
    def param(rng: KeyStream) -> Struct:
        return Struct(p=jax.random.uniform(rng.next(), (hidden,), minval=1.0, maxval=3.0),
                      wx=0.5 * jax.random.normal(rng.next(), (hidden,)),
                      b=jnp.zeros(hidden))

    def init(ndef, param: Struct) -> jax.Array:
        return jnp.zeros_like(ndef.input)

    def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
        a = jax.nn.sigmoid(param.p)
        h = a * state + (1 - a) * jnp.tanh(param.wx * input + param.b)
        return h, h

    return node_def(apply, init=init, param=param, apply_input_spec=jnp.zeros(mc.HIDDEN), name='lru')


def task_with_cell(cell: NodeDef, layers: int) -> NodeDef:
    """The meta-controller task node with the given recurrent core."""
    from nodejax import externalize, observed_loop, at
    pipe = at(mc.filters_def(mc.DT), 'error') >> mc.flat \
        >> mc.up_def(3 + mc.ORDER + 1, mc.HIDDEN) \
        >> stack(cell, n=layers) >> mc.readout_def(mc.HIDDEN) \
        >> mc.identified(mc.motor_def(mc.DT), mc.ORDER)
    rollout = scan(observed_loop(pipe, belief0=jnp.zeros(mc.ORDER + 1)),
                   persist=('rls', 'belief'))
    return externalize(rollout, 'motor', at_init=mc.motor_def(mc.DT).build_param())


def run(name: str, cell: NodeDef, layers: int) -> None:
    from nodejax import metasgd
    task = task_with_cell(cell, layers)
    adapt = metasgd(task, mc.mse, mc.INNER_LR0)
    trainer = train_step(batch(adapt), mc.mse, optax.adam(mc.META_LR))
    model = batch(adapt).parameterize(rng=jax.random.PRNGKey(0))

    _, train_tasks, q_t = mc.make_tasks(np.random.RandomState(0),
                                        mc.META_STEPS * mc.TASKS, k=mc.K)

    def fold(x):
        return jax.tree.map(lambda a: a.reshape(mc.META_STEPS, mc.TASKS, *a.shape[1:]), x)

    final, losses = trainer.scan(trainer.init(model=model.param), Struct(input=fold(train_tasks), target=fold(q_t)))

    plant, tasks, q_t = mc.make_tasks(np.random.RandomState(99), mc.TASKS, k=mc.K)
    adapted = batch(adapt).apply(final.model, tasks)
    untuned = batch(task).bind(final.model.init)
    _, unadapted = untuned.apply(untuned.with_input(tasks.query).init(), tasks.query)
    random_adapted = batch(adapt).apply(model.param, tasks)

    settled = mc.mse(adapted[:, mc.T // 2:], q_t[:, mc.T // 2:])
    bias = jnp.mean(adapted[:, -20:] - q_t[:, -20:], axis=1)
    n_weights = sum(x.size for x in jax.tree.leaves(final.model.init))
    mc.plot_tuning(f'cells_{name}.png',
                   f'{name}: {layers} layers, {n_weights} weights | '
                   f'settled mse {settled:.5f}, worst bias {jnp.max(jnp.abs(bias)):.3f}',
                   plant, q_t, adapted, unadapted, random_adapted)
    print(f'{name:16s} L={layers} weights={n_weights:4d}: '
          f'finite={bool(jnp.all(jnp.isfinite(losses)))} '
          f'adapted {mc.mse(adapted, q_t):.4f} unadapted {mc.mse(unadapted, q_t):.4f} '
          f'settled {settled:.5f} worst|bias| {jnp.max(jnp.abs(bias)):.4f}')


def main() -> None:
    run('elman-2', mc.rnn_def(mc.HIDDEN), 2)
    run('gru-2', gru_def(mc.HIDDEN), 2)
    run('mingru-tanh-2', mingru_def(mc.HIDDEN, tanh_candidate=True), 2)
    run('mingru-tanh-4', mingru_def(mc.HIDDEN, tanh_candidate=True), 4)
    run('mingru-lin-2', mingru_def(mc.HIDDEN, tanh_candidate=False), 2)
    run('mingru-lin-4', mingru_def(mc.HIDDEN, tanh_candidate=False), 4)
    run('lru-2', lru_def(mc.HIDDEN), 2)
    run('lru-4', lru_def(mc.HIDDEN), 4)


if __name__ == '__main__':
    main()
