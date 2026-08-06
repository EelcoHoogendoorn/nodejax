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
    HIDDEN, LAYERS, STEPS, META_STEPS, INNER_LR, make_data, make_tasks)

class Net(eqx.Module):
    up_w: jax.Array
    wx: jax.Array        # (LAYERS, HIDDEN): per-layer, stacked by hand
    wh: jax.Array        # (LAYERS, HIDDEN, HIDDEN)
    b: jax.Array         # (LAYERS, HIDDEN)
    ro_w: jax.Array
    ro_b: jax.Array

    def step(self, h_layers, x):
        """One timestep: scan over depth, threading each layer's hidden."""
        signal = self.up_w * x

        def depth(carry, xs):
            wx, wh, b, h = xs
            h_new = jnp.tanh(wx * carry + wh @ h + b)
            return h_new, h_new

        signal, h_new = jax.lax.scan(depth, signal, (self.wx, self.wh, self.b, h_layers))
        y = self.ro_w @ signal + self.ro_b
        return h_new, y

    def rollout(self, xs):
        """Scan over time, carrying the (LAYERS, HIDDEN) state block."""
        def t_step(h_layers, x):
            return self.step(h_layers, x)

        _, ys = jax.lax.scan(t_step, jnp.zeros((LAYERS, HIDDEN)), xs)
        return ys


def make_net(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return Net(
        up_w=0.5 * jax.random.normal(k1, (HIDDEN,)),
        wx=0.5 * jax.random.normal(k2, (LAYERS, HIDDEN)),
        wh=0.3 * jax.random.normal(k3, (LAYERS, HIDDEN, HIDDEN)) / jnp.sqrt(HIDDEN),
        b=jnp.zeros((LAYERS, HIDDEN)),
        ro_w=0.1 * jax.random.normal(k4, (HIDDEN,)),
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
