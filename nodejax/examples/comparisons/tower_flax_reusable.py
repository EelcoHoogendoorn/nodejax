"""The flax nnx tower, refactored for reusability: how far generic
combinators get you, and what remains stuck to the model.

`tower_flax.py` writes the axis routing inline. This file extracts
every reusable piece the framework permits: a generic stacked
constructor, a generic depth scan over any cell obeying a carry
protocol, a generic time unroll, a generic fit loop. The experiment's
finding, in two halves. The combinators work: each is written once,
against a fixed cell protocol `(carry, x) -> (carry, y)` plus the
Variable kinds standing in for role declarations, and the model code
below shrinks accordingly. And the combinators converge toward a
contract: the protocol IS an apply signature, the Variable kinds ARE
param/state roles, and what cannot be extracted is exactly what has no
declared home: the initial carry's shape (supplied per call site), the
adaptation loop's exit from the module system, and the fact that a
combinator's output is a function, which further lifted transforms
cannot route state through the way they route modules.

Same model, task, and budget as `tower_nodejax.py`; matched losses.
The meta level is packaged too, and `maml_fit` is the finding at its
sharpest: the combinator is generic and reusable, and its interior is
value-land, the module split to (graphdef, state) at the door, merged
back only to compute rollouts, with results written into the module at
the end. The module system is a guest inside its own combinator.

Run directly:  python -m nodejax.examples.comparisons.tower_flax_reusable
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from nodejax.examples.comparisons.tower_common import (
    HIDDEN, LAYERS, STEPS, META_STEPS, INNER_LR, make_data, make_tasks)


# --- the reusable half: written once, generic over the cell protocol.
# "Generic" carries a caveat worth stating: these combinators apply only
# to modules that obey the convention (carry, x) -> (carry, y), and
# nothing enforces it. A module without a carry, or with an extra
# Variable kind, needs different combinators or different annotations;
# a stack transform total over ALL models requires what a convention
# cannot give, every model having the same slots, trivially filled when
# unused. That uniformity is a contract, and it is the thing being
# partially reinvented here. ---

def stacked(make_module, n):
    """Construct n modules with independent draws, stacked on axis 0."""
    @nnx.split_rngs(splits=n)
    @nnx.vmap(in_axes=0, out_axes=0)
    def make(rngs):
        return make_module(rngs)
    return make


def scan_layers(cells, signal, carries):
    """Feed `signal` through stacked cells obeying (carry, x) -> (carry, y);
    params per-layer, the signal threading through."""
    @nnx.scan(in_axes=(0, nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def depth(cells, signal, carry):
        carry_new, y = cells(carry, signal)
        return y, carry_new

    return depth(cells, signal, carries)


def unroll(module, carry0, xs):
    """Run a (carry, x) -> (carry, y) step over a leading time axis;
    params broadcast, the carry threading through."""
    @nnx.scan(in_axes=(None, nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def over_time(module, carry, x):
        return module.step_carry(carry, x)

    return over_time(module, carry0, xs)


def fit(model, loss_of, opt, steps):
    """Generic training loop: jitted step, optimizer and model mutating
    in place, losses returned."""
    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer):
        loss, grads = nnx.value_and_grad(loss_of)(model)
        optimizer.update(model, grads)
        return loss

    return [float(train_step(model, optimizer)) for _ in range(steps)]


# --- the model half: what stays after extraction ---

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
        self.cells = stacked(Cell, LAYERS)(rngs)
        self.ro_w = nnx.Param(0.1 * jax.random.normal(k2, (HIDDEN,)))
        self.ro_b = nnx.Param(jnp.zeros(()))

    def step_carry(self, h_layers, x):
        signal, h_new = scan_layers(self.cells, self.up_w[...] * x, h_layers)
        return h_new, self.ro_w[...] @ signal + self.ro_b[...]

    def rollout(self, xs):
        # the initial carry's shape has no declared home: every call site
        # must know (LAYERS, HIDDEN)
        _, ys = unroll(self, jnp.zeros((LAYERS, HIDDEN)), xs)
        return ys


def maml_fit(model, call, inner_lr, meta_opt, meta_steps, tasks):
    """Generic MAML over any module and a `call: (module, xs) -> ys`
    convention: adapt per task on support sequences, meta-train the
    initialization on query losses, write the result into the module.
    The interior is plain-value MAML; the module is split at the door."""
    sup_x, sup_y, qry_x, qry_y = tasks
    graphdef, params = nnx.split(model)

    def apply(params, xs):
        return call(nnx.merge(graphdef, params), xs)

    def adapt(params, sx_all, sy_all):
        def istep(params, sup):
            sx, sy = sup
            g = jax.grad(lambda p: jnp.mean((apply(p, sx) - sy) ** 2))(params)
            return jax.tree.map(lambda pv, gv: pv - inner_lr * gv, params, g), None
        return jax.lax.scan(istep, params, (sx_all, sy_all))[0]

    def meta_loss(params):
        def per_task(sx, sy, qx, qy):
            return jnp.mean((apply(adapt(params, sx, sy), qx) - qy) ** 2)
        return jnp.mean(jax.vmap(per_task)(sup_x, sup_y, qry_x, qry_y))

    @jax.jit
    def train(params, opt_state):
        def t_step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(meta_loss)(params)
            updates, opt_state = meta_opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss
        return jax.lax.scan(t_step, (params, opt_state), None, length=meta_steps)

    (params, _), losses = train(params, meta_opt.init(params))
    nnx.update(model, params)
    return losses


def main():
    xs, ys = make_data(jax.random.PRNGKey(0))
    net = Net(nnx.Rngs(params=jax.random.PRNGKey(1)))

    def loss_of(net):
        pred = nnx.vmap(lambda n, x: n.rollout(x), in_axes=(None, 0))(net, xs)
        return jnp.mean((pred - ys) ** 2)

    losses = fit(net, loss_of, optax.adam(3e-3), STEPS)
    print(f"tower loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.05 * losses[0]
    return losses[-1]


def main_meta():
    net = Net(nnx.Rngs(params=jax.random.PRNGKey(1)))
    losses = maml_fit(net, lambda m, xs: m.rollout(xs), INNER_LR,
                      optax.adam(3e-3), META_STEPS,
                      make_tasks(jax.random.PRNGKey(2)))
    print(f"meta  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.3 * losses[0]
    return float(losses[-1])


if __name__ == '__main__':
    main()
    main_meta()
