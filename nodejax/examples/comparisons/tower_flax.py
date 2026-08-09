"""Deeply nested composition, in flax nnx: a stacked RNN, scanned over
time, batched, trained, and meta-adapted, with the state routing
stated at every lifted-transform site.

Same model, task, and budget as `tower_nodejax.py`. nnx modules are
mutable objects, and crossing each jax boundary means telling the
lifted transform how every kind of state in the subtree maps across
that axis. The census: constructing the layer stack is a lifted vmap
with its rng-split annotation; the depth scan is a lifted scan whose
in_axes say params are per-layer while the signal is the carry; the
time scan's annotations say the opposite, params broadcast and hidden
carried; the training loop steps a jitted update mutating model and
optimizer in place. The meta variant leaves the module system: the
inner adaptation differentiates through functional (graphdef, state)
splits, because a mutable module cannot be the carry of a scan whose
body takes its gradient.

The streaming norm measures the cost of ONE MORE KIND of state (its
running stats are neither params nor a hidden block): on the nodejax
side it is one pipe term, here it could not enter the module system at
all — see the note above `norm_step` for the three spellings tried —
so its stats thread the carries by hand, and every function between
the step and the time scan names them.

Run directly:  python -m nodejax.examples.comparisons.tower_flax
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nodejax.examples.comparisons.tower_common import (
    HIDDEN, LAYERS, STEPS, META_STEPS, INNER_LR, MOMENTUM, make_data, make_tasks)


# The streaming norm's stats are hand-threaded values, NOT module
# state. The module-system spellings were tried and measured (flax
# 0.12.8): a mutating Variable inside the broadcast time scan runs and
# SILENTLY freezes (each step reads the original stats, nothing chains,
# nothing writes back); declaring the stats a custom Variable kind
# carried via nnx.StateAxes({RunStats: nnx.Carry, ...: None}) fails
# with 'cannot extract graph node from different trace level', as does
# carrying the whole module. So the norm's state exits the module
# system entirely and rides the scan carry by hand, equinox-style.
def norm_step(stats, x):
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
            return y, h_new

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


def main():
    xs, ys = make_data(jax.random.PRNGKey(0))
    net = Net(nnx.Rngs(params=jax.random.PRNGKey(1)))
    optimizer = nnx.Optimizer(net, optax.adam(3e-3), wrt=nnx.Param)

    @nnx.jit
    def train_step(net, optimizer):
        def loss_fn(net):
            # the batch axis must ALSO be the lifted vmap: params broadcast
            pred = nnx.vmap(lambda n, x: n.rollout(x), in_axes=(None, 0))(net, xs)
            return jnp.mean((pred - ys) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(net)
        optimizer.update(net, grads)
        return loss

    losses = [float(train_step(net, optimizer)) for _ in range(STEPS)]
    print(f"tower loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.05 * losses[0]
    return losses[-1]


# --- one level deeper: meta-learning across a task family ---

def main_meta():
    """The exit from the module system: a mutable module cannot be the
    differentiated carry of the inner scan, so the meta level splits to
    (graphdef, state) once and runs MAML on plain values, the module
    resurrected by merge purely to compute rollouts."""
    sup_x, sup_y, qry_x, qry_y = make_tasks(jax.random.PRNGKey(2))
    net = Net(nnx.Rngs(params=jax.random.PRNGKey(1)))
    graphdef, params = nnx.split(net)

    def rollout(params, xs):
        return nnx.merge(graphdef, params).rollout(xs)

    def adapt(params, sx_all, sy_all):
        def istep(params, sup):
            sx, sy = sup
            g = jax.grad(lambda p: jnp.mean((rollout(p, sx) - sy) ** 2))(params)
            return jax.tree.map(lambda pv, gv: pv - INNER_LR * gv, params, g), None
        return jax.lax.scan(istep, params, (sx_all, sy_all))[0]

    def meta_loss(params):
        def per_task(sx, sy, qx, qy):
            return jnp.mean((rollout(adapt(params, sx, sy), qx) - qy) ** 2)
        return jnp.mean(jax.vmap(per_task)(sup_x, sup_y, qry_x, qry_y))

    opt = optax.adam(3e-3)

    @jax.jit
    def train(params, opt_state):
        def t_step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(meta_loss)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss
        return jax.lax.scan(t_step, (params, opt_state), None, length=META_STEPS)

    (params, _), losses = train(params, opt.init(params))
    print(f"meta  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.3 * losses[0]
    return float(losses[-1])


if __name__ == '__main__':
    main()
    main_meta()
