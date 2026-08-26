"""Deeply nested composition, in equinox: the same one-tree tower as
`tower_nodejax.py`, a residual stacked RNN committee adapted per task
inside the meta-training loop, with every axis's state threaded by
hand.

Same model, task, and budget. Equinox's modules handle the parameter
half well; with no state contract, every loop below is written against
raw `lax.scan`, and the routing (which arrays stack per layer, what
carries over time, what carries across adaptation and meta steps) is
hand-encoded in each scan's carry and xs. The census: the depth scan
threads (per-layer params, per-layer hidden) explicitly; the time scan
carries the (LAYERS, HIDDEN) state block beside the norm stats; the
inner adaptation scans the WHOLE COMMITTEE as its carry, one sgd step
per support sequence, each step differentiating through the member
vmap and rollout's two scans; the meta scan carries (params, optimizer
state) and differentiates through all of it, second order; the task
and member axes ride vmaps. Inserting a member with a new kind of
state touches every level that encloses it.

The committee is the layer trick a second time, the constructor
lifted over a member axis, its mean taken where the members meet; the
residual is a line in the depth body. The population the committee
computed dies at the reduction, there being no aux stream to sow it
on.

Run directly:  python -m examples.comparisons.tower.tower_equinox
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from examples.comparisons.tower.tower_common import (
    HIDDEN, LAYERS, MEMBERS, META_STEPS, INNER_LR, OUTER_LR, MOMENTUM,
    make_tasks)
from nodejax.core.types import PyTree


def norm_step(stats: PyTree, x: jax.Array):
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


def make_cell(key: jax.Array):
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
        """One timestep: the running norm's stats are one more slot
        in the hand-threaded carry, then a scan over depth; lax.scan
        slices the stacked cell per layer, the layer's hidden rides
        alongside as xs."""
        h_layers, stats = carry
        stats, signal = norm_step(stats, self.up_w * x)

        def depth(signal, xs):
            cell, h = xs
            h_new, y = cell(h, signal)
            # the residual: one line in this body, where nodejax wraps the
            # cell; nothing here marks it as structure
            return signal + y, h_new

        signal, h_new = jax.lax.scan(depth, signal, (self.cells, h_layers))
        return (h_new, stats), self.ro_w @ signal + self.ro_b

    def rollout(self, xs):
        """Scan over time, carrying the (LAYERS, HIDDEN) state block
        and the norm stats beside it."""
        def t_step(carry, x):
            return self.step(carry, x)

        _, ys = jax.lax.scan(t_step, (jnp.zeros((LAYERS, HIDDEN)), norm_init()), xs)
        return ys


def make_net(key: jax.Array):
    k1, k2, k3 = jax.random.split(key, 3)
    return Net(
        up_w=0.5 * jax.random.normal(k1, (HIDDEN,)),
        cells=jax.vmap(make_cell)(jax.random.split(k2, LAYERS)),   # ctor lifted over layers
        ro_w=0.1 * jax.random.normal(k3, (HIDDEN,)),
        ro_b=jnp.zeros(()),
    )


def make_committee(key: jax.Array):
    # the ensemble: the constructor lifted over members, exactly the layer
    # trick again, one more leading axis on every leaf
    return jax.vmap(make_net)(jax.random.split(key, MEMBERS))


def committee_rollout(committee, xs: jax.Array):
    # each member rolls the sequence; the mean is the committee's answer,
    # and the population DIES here, no aux stream existing to sow it on
    population = jax.vmap(lambda member: member.rollout(xs))(committee)
    return jnp.mean(population, axis=0)


def adapt(committee, sup_x, sup_y):
    """The inner scan: one sgd step per support sequence, the whole
    committee as the carry, each gradient flowing through the member vmap
    and rollout's two scans."""
    def istep(committee, sup):
        sx, sy = sup
        grads = eqx.filter_grad(
            lambda adapted: jnp.mean((committee_rollout(adapted, sx) - sy) ** 2))(committee)
        return eqx.apply_updates(committee, jax.tree.map(lambda g: -INNER_LR * g, grads)), None

    committee, _ = jax.lax.scan(istep, committee, (sup_x, sup_y))
    return committee


def main() -> None:
    sup_x, sup_y, qry_x, qry_y = make_tasks(jax.random.PRNGKey(2))
    committee = make_committee(jax.random.PRNGKey(1))
    opt = optax.adam(OUTER_LR)

    def meta_loss(committee):
        def per_task(sx, sy, qx, qy):
            return jnp.mean((committee_rollout(adapt(committee, sx, sy), qx) - qy) ** 2)
        return jnp.mean(jax.vmap(per_task)(sup_x, sup_y, qry_x, qry_y))

    @jax.jit
    def train(committee, opt_state):
        def t_step(carry, _):
            committee, opt_state = carry
            loss, grads = eqx.filter_value_and_grad(meta_loss)(committee)
            updates, opt_state = opt.update(grads, opt_state, committee)
            committee = eqx.apply_updates(committee, updates)
            return (committee, opt_state), loss

        return jax.lax.scan(t_step, (committee, opt_state), None, length=META_STEPS)

    (committee, _), losses = train(
        committee, opt.init(eqx.filter(committee, eqx.is_array)))
    print(f"tower loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.3 * losses[0]
    return float(losses[-1])


if __name__ == '__main__':
    main()
