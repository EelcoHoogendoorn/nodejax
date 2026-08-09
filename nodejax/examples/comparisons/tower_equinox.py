"""Deeply nested composition, in equinox: a stacked RNN, scanned over
time, batched, trained, and meta-adapted, with every axis's state
threaded by hand.

Same model, task, and budget as `tower_nodejax.py`. Equinox's modules
handle the parameter half well; with no state contract, each of the
three nested scans below is written against raw `lax.scan`, and the
routing (which arrays stack per layer, what carries over time, what
carries over training steps) is hand-encoded in each scan's carry and
xs. The census: the depth scan threads (per-layer params, per-layer
hidden) explicitly; the time scan carries the (LAYERS, HIDDEN) state
block; the training scan carries (params, optimizer state) and closes
over the data; the batch axis rides a vmap. Inserting a member with a
new kind of state touches every level that encloses it.

The meta variant nests a fourth scan: per task, an inner loop of
gradient steps on support sequences (each step differentiating through
the time and depth scans), the adapted net scored on a query sequence,
and the outer scan differentiating through all of it, second-order.
Every level threads its own carry by hand.

Run directly:  python -m nodejax.examples.comparisons.tower_equinox
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from nodejax.examples.comparisons.tower_common import (
    HIDDEN, LAYERS, STEPS, META_STEPS, INNER_LR, MOMENTUM, make_data, make_tasks)


def norm_step(stats, x):
    m1, var = stats
    out = (x - m1) / jnp.sqrt(var + 1e-5)
    new_m1 = (1 - MOMENTUM) * m1 + MOMENTUM * x
    new_var = (1 - MOMENTUM) * var + MOMENTUM * (x - m1) ** 2
    return (new_m1, new_var), out


def norm_init():
    return (jnp.zeros(HIDDEN), jnp.ones(HIDDEN))

class Cell(eqx.Module):
    wx: jax.Array
    wh: jax.Array
    b: jax.Array

    def __call__(self, h, x):
        h_new = jnp.tanh(self.wx * x + self.wh @ h + self.b)
        return h_new, h_new


def make_cell(key):
    k1, k2 = jax.random.split(key)
    return Cell(wx=0.5 * jax.random.normal(k1, (HIDDEN,)),
                wh=0.3 * jax.random.normal(k2, (HIDDEN, HIDDEN)) / jnp.sqrt(HIDDEN),
                b=jnp.zeros(HIDDEN))


class Net(eqx.Module):
    up_w: jax.Array
    cells: Cell          # one Cell whose arrays carry a leading layer axis
    ro_w: jax.Array
    ro_b: jax.Array

    def step(self, carry, x):
        """One timestep: the streaming norm's stats are one more slot
        in the hand-threaded carry, then a scan over depth; lax.scan
        slices the stacked cell per layer, the layer's hidden rides
        alongside as xs."""
        h_layers, stats = carry
        stats, signal = norm_step(stats, self.up_w * x)

        def depth(signal, xs):
            cell, h = xs
            h_new, y = cell(h, signal)
            return y, h_new

        signal, h_new = jax.lax.scan(depth, signal, (self.cells, h_layers))
        return (h_new, stats), self.ro_w @ signal + self.ro_b

    def rollout(self, xs):
        """Scan over time, carrying the (LAYERS, HIDDEN) state block
        and the norm stats beside it."""
        def t_step(carry, x):
            return self.step(carry, x)

        _, ys = jax.lax.scan(t_step, (jnp.zeros((LAYERS, HIDDEN)), norm_init()), xs)
        return ys


def make_net(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return Net(
        up_w=0.5 * jax.random.normal(k1, (HIDDEN,)),
        cells=jax.vmap(make_cell)(jax.random.split(k2, LAYERS)),   # ctor lifted over layers
        ro_w=0.1 * jax.random.normal(k3, (HIDDEN,)),
        ro_b=jnp.zeros(()),
    )


def mse_loss(net, xs, ys):
    pred = jax.vmap(net.rollout)(xs)                    # the batch axis
    return jnp.mean((pred - ys) ** 2)


def main():
    xs, ys = make_data(jax.random.PRNGKey(0))
    net = make_net(jax.random.PRNGKey(1))
    opt = optax.adam(3e-3)

    @jax.jit
    def train(net, opt_state):
        """Scan over training steps, carrying (params, optimizer state)."""
        def t_step(carry, _):
            net, opt_state = carry
            loss, grads = eqx.filter_value_and_grad(mse_loss)(net, xs, ys)
            updates, opt_state = opt.update(grads, opt_state, net)
            net = eqx.apply_updates(net, updates)
            return (net, opt_state), loss

        return jax.lax.scan(t_step, (net, opt_state), None, length=STEPS)

    (net, _), losses = train(net, opt.init(eqx.filter(net, eqx.is_array)))
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.05 * losses[0]
    return float(losses[-1])


# --- one level deeper: meta-learning across a task family ---

def adapt(net, sup_x, sup_y):
    """The inner scan: one sgd step per support sequence, the net itself
    as the carry, each gradient flowing through rollout's two scans."""
    def istep(net, sup):
        sx, sy = sup
        grads = eqx.filter_grad(lambda n: jnp.mean((n.rollout(sx) - sy) ** 2))(net)
        return eqx.apply_updates(net, jax.tree.map(lambda g: -INNER_LR * g, grads)), None

    net, _ = jax.lax.scan(istep, net, (sup_x, sup_y))
    return net


def main_meta():
    sup_x, sup_y, qry_x, qry_y = make_tasks(jax.random.PRNGKey(2))
    net = make_net(jax.random.PRNGKey(1))
    opt = optax.adam(3e-3)

    def meta_loss(net):
        def per_task(sx, sy, qx, qy):
            return jnp.mean((adapt(net, sx, sy).rollout(qx) - qy) ** 2)
        return jnp.mean(jax.vmap(per_task)(sup_x, sup_y, qry_x, qry_y))

    @jax.jit
    def train(net, opt_state):
        def t_step(carry, _):
            net, opt_state = carry
            loss, grads = eqx.filter_value_and_grad(meta_loss)(net)
            updates, opt_state = opt.update(grads, opt_state, net)
            net = eqx.apply_updates(net, updates)
            return (net, opt_state), loss

        return jax.lax.scan(t_step, (net, opt_state), None, length=META_STEPS)

    (net, _), losses = train(net, opt.init(eqx.filter(net, eqx.is_array)))
    print(f"meta  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.3 * losses[0]
    return float(losses[-1])


if __name__ == '__main__':
    main()
    main_meta()
