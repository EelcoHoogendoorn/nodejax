"""Bespoke Flax NNX implementation of the nested-transform `tower` benchmark.

`nnx.vmap` constructs layer and committee axes while `nnx.scan` runs depth, time, inner adaptation, and outer meta-training. Because NNX modules carry no shared step/state contract across those axes, every transform site specifies its own `nnx.StateAxes` policy, `nnx.split_rngs` decoration, and hand-threaded `(h_layers, stats)` carry tuple, scattering the architecture across module methods and bespoke loop bodies. Factoring those policies into reusable wrappers in `tower_nnx_reusable.py` nearly doubles the file size (340+ lines) and still drops out of the wrapper protocol at `maml_fit`.

Run directly with `python -m examples.comparisons.tower.tower_nnx`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from examples.comparisons.tower.tower_common import (
    HIDDEN, LAYERS, MEMBERS, META_STEPS, INNER_LR, OUTER_LR, MOMENTUM,
    make_tasks)
PyTree = object


# The running statistics are ordinary values in the rollout carry. NNX can
# also represent them as Variables, but this task resets them for each rollout,
# so keeping them beside the recurrent values makes that lifetime explicit.
def norm_step(stats: PyTree, x: jax.Array):
    m1, var = stats
    out = (x - m1) / jnp.sqrt(var + 1e-5)
    new_m1 = (1 - MOMENTUM) * m1 + MOMENTUM * x
    new_var = (1 - MOMENTUM) * var + MOMENTUM * (x - m1) ** 2
    return (new_m1, new_var), out


def norm_init():
    return (jnp.zeros(HIDDEN), jnp.ones(HIDDEN))


class Cell(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        k1, k2 = jax.random.split(rngs.params())
        self.wx = nnx.Param(0.5 * jax.random.normal(k1, (HIDDEN,)))
        self.wh = nnx.Param(0.3 * jax.random.normal(k2, (HIDDEN, HIDDEN)) / jnp.sqrt(HIDDEN))
        self.b = nnx.Param(jnp.zeros(HIDDEN))

    def __call__(self, h, x):
        h_new = jnp.tanh(self.wx[...] * x + self.wh[...] @ h + self.b[...])
        return h_new, h_new


class Net(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        k1, k2 = jax.random.split(rngs.params())
        self.up_w = nnx.Param(0.5 * jax.random.normal(k1, (HIDDEN,)))

        # stacked construction: a lifted vmap, rng split per layer
        @nnx.split_rngs(splits=LAYERS)
        @nnx.vmap(in_axes=0, out_axes=0)
        def make_cells(rngs: nnx.Rngs):
            return Cell(rngs)

        self.cells = make_cells(rngs)
        self.ro_w = nnx.Param(0.1 * jax.random.normal(k2, (HIDDEN,)))
        self.ro_b = nnx.Param(jnp.zeros(()))

    def step(self, carry, x):
        # the hand-threaded slot: every level of carry plumbing between
        # here and the time scan now names the stats explicitly
        h_layers, stats = carry
        stats, signal = norm_step(stats, self.up_w[...] * x)

        # depth scan: params per-layer (axis 0), the signal is the carry
        @nnx.scan(in_axes=(0, nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def depth(cells, signal, h):
            h_new, y = cells(h, signal)
            # the residual: one line in this body, nothing marks it as
            # structure
            return signal + y, h_new

        signal, h_new = depth(self.cells, signal, h_layers)
        return (h_new, stats), self.ro_w[...] @ signal + self.ro_b[...]

    def rollout(self, xs):
        # time scan: params broadcast, the (hidden block, norm stats)
        # tuple is the carry
        @nnx.scan(in_axes=(None, nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def over_time(net, carry, x):
            carry, y = net.step(carry, x)
            return carry, y

        _, ys = over_time(self, (jnp.zeros((LAYERS, HIDDEN)), norm_init()), xs)
        return ys


def make_committee():
    # the ensemble: the same lifted-vmap trick as the layer stack, one more
    # rng-split annotation, one more leading axis on every Param
    @nnx.split_rngs(splits=MEMBERS)
    @nnx.vmap(in_axes=0, out_axes=0)
    def make_members(rngs: nnx.Rngs):
        return Net(rngs)

    return make_members(nnx.Rngs(params=jax.random.PRNGKey(1)))


def main() -> None:
    sup_x, sup_y, qry_x, qry_y = make_tasks(jax.random.PRNGKey(2))
    committee = make_committee()
    optimizer = nnx.Optimizer(committee, optax.adam(OUTER_LR), wrt=nnx.Param)

    def rollout(model, input_sequence):
        population = nnx.vmap(
            lambda member, sequence: member.rollout(sequence),
            in_axes=(0, None),
        )(model, input_sequence)
        return jnp.mean(population, axis=0)

    adapt_axes = nnx.StateAxes({nnx.Param: nnx.Carry, ...: None})
    optimizer_axes = nnx.StateAxes({...: nnx.Carry})

    def meta_loss(committee):
        def per_task(initial, support_inputs, support_targets,
                     query_input, query_target):
            adapted = nnx.clone(initial)
            inner_optimizer = nnx.Optimizer(
                adapted, optax.sgd(INNER_LR), wrt=nnx.Param)

            @nnx.scan(
                in_axes=(adapt_axes, optimizer_axes, 0, 0),
                out_axes=0,
            )
            def adapt_step(adapted, inner_optimizer,
                           support_input, support_target):
                loss, grads = nnx.value_and_grad(
                    lambda model: jnp.mean(
                        (rollout(model, support_input) - support_target) ** 2)
                )(adapted)
                inner_optimizer.update(adapted, grads)
                return loss

            adapt_step(
                adapted, inner_optimizer, support_inputs, support_targets)
            return jnp.mean(
                (rollout(adapted, query_input) - query_target) ** 2)

        losses = nnx.vmap(
            per_task,
            in_axes=(None, 0, 0, 0, 0),
        )(committee, sup_x, sup_y, qry_x, qry_y)
        return jnp.mean(losses)

    train_axes = nnx.StateAxes({...: nnx.Carry})

    @nnx.scan(in_axes=(train_axes, train_axes, 0), out_axes=0)
    def train_loop(committee, optimizer, _):
        loss, grads = nnx.value_and_grad(meta_loss)(committee)
        optimizer.update(committee, grads)
        return loss

    losses = train_loop(committee, optimizer, jnp.arange(META_STEPS))
    first, last = float(losses[0]), float(losses[-1])
    print(f"tower loss {first:.4f} -> {last:.4f}")
    assert np.isfinite(first) and np.isfinite(last)
    return last


if __name__ == '__main__':
    main()
