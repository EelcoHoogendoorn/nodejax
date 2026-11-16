"""Reusable test-time training wrapper in Equinox across stateless and recurrent modules.

Equinox represents every `eqx.Module` as a PyTree, which makes carrying fast weights through `lax.scan` natural, yet `eqx.Module` defines no stateful call contract (`model(x)` vs. `model(carry, x)`), no step-vs-sequence wrapper algebra, and no built-in state slot for the previous-token register. Writing a reusable `TTT` wrapper therefore requires runtime `hasattr(model, 'init_carry')` inspection, `eqx.filter` partitioning to allocate per-leaf rates, and hand-rolled `Batched` and `MetaTrainer` loops around `lax.scan`.

Run directly with `python -m examples.comparisons.ttt.ttt_equinox`.
"""

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from examples.comparisons.ttt.ttt_common import (
    EVAL_TASKS, HIDDEN, META_LR, META_STEPS, QUERY0, TASKS, TTT_LR0, VOCAB,
    make_tasks, query_nll_numpy, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    return -jax.nn.log_softmax(logits)[target]


def query_xent(preds: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(preds)
    picked = jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked[..., QUERY0:])


class MLP(eqx.Module):
    """Predicts next-token logits from a single token (`token -> logits`)."""

    embed: jax.Array
    w: jax.Array
    out: jax.Array

    def __init__(self, vocab: int, hidden: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.embed = 0.3 * jax.random.normal(k1, (vocab, hidden))
        self.w = 0.3 * jax.random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden)
        self.out = 0.1 * jax.random.normal(k3, (hidden, vocab))

    def __call__(self, token: jax.Array) -> jax.Array:
        return jnp.tanh(self.embed[token] @ self.w) @ self.out


class RNN(eqx.Module):
    """Predicts next-token logits with explicit carry (`(carry, token) -> (carry, logits)`)."""

    embed: jax.Array
    wh: jax.Array
    out: jax.Array

    def __init__(self, vocab: int, hidden: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.embed = 0.3 * jax.random.normal(k1, (vocab, hidden))
        self.wh = 0.3 * jax.random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden)
        self.out = 0.1 * jax.random.normal(k3, (hidden, vocab))

    def init_carry(self) -> jax.Array:
        return jnp.zeros(self.wh.shape[0])

    def __call__(self, carry: jax.Array, token: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.tanh(self.embed[token] + self.wh @ carry)
        return h, h @ self.out


class TTT(eqx.Module):
    """Wraps unary or carry-based Equinox modules in online gradient adaptation."""

    model: eqx.Module
    rates: eqx.Module
    loss_fn: Callable = eqx.field(static=True)
    has_carry: bool = eqx.field(static=True)

    def __init__(self, model: eqx.Module, loss_fn: Callable, lr0: float):
        self.model = model
        self.loss_fn = loss_fn
        self.has_carry = hasattr(model, 'init_carry')
        params = eqx.filter(model, eqx.is_inexact_array)
        self.rates = jax.tree.map(lambda x: jnp.full_like(x, lr0), params)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        carry0 = self.model.init_carry() if self.has_carry else ()

        def step(state, token: jax.Array):
            fast, carry, prev = state

            def selfsup(m: eqx.Module):
                if self.has_carry:
                    next_carry, out = m(carry, prev)
                else:
                    next_carry, out = (), m(prev)
                return self.loss_fn(out, token), (next_carry, out)

            grads, (next_carry, out) = eqx.filter_grad(selfsup, has_aux=True)(fast)
            updates = jax.tree.map(
                lambda g, r: None if g is None else -r * g,
                grads, self.rates, is_leaf=lambda x: x is None,
            )
            return (eqx.apply_updates(fast, updates), next_carry, token), out

        _, preds = jax.lax.scan(step, (self.model, carry0, tokens[0]), tokens)
        return preds


class Batched(eqx.Module):
    inner: eqx.Module

    def __call__(self, batch: jax.Array) -> jax.Array:
        return jax.vmap(self.inner)(batch)


def fit_meta(model: Batched, loss_fn: Callable, tx, batches: jax.Array):
    opt_state0 = tx.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def run_scan(m: Batched, opt_s):
        def body(carry, batch):
            cur_m, cur_opt = carry
            loss, grads = eqx.filter_value_and_grad(
                lambda mod, b: loss_fn(mod(b), b)
            )(cur_m, batch)
            updates, next_opt = tx.update(
                grads, cur_opt, eqx.filter(cur_m, eqx.is_inexact_array)
            )
            return (eqx.apply_updates(cur_m, updates), next_opt), loss

        (done_m, _), losses = jax.lax.scan(body, (m, opt_s), batches)
        return done_m, losses

    return run_scan(model, opt_state0)


def run_variant(variant: str, inner: eqx.Module, tokens: jax.Array,
                held_out: jax.Array) -> None:
    batched0 = Batched(TTT(inner, xent, TTT_LR0))
    preds0 = batched0(held_out)
    batched1, losses = fit_meta(batched0, query_xent, optax.adam(META_LR), tokens)
    preds1 = batched1(held_out)
    params = eqx.filter(batched1, eqx.is_inexact_array)
    n_weights = sum(x.size for x in jax.tree.leaves(params))
    report(
        'equinox', variant, n_weights, bool(jnp.all(jnp.isfinite(losses))),
        query_nll_numpy(preds0, held_out), query_nll_numpy(preds1, held_out),
    )


def main() -> None:
    tokens = jnp.asarray(
        make_tasks(np.random.RandomState(0), META_STEPS * TASKS).reshape(META_STEPS, TASKS, -1)
    )
    held_out = jnp.asarray(make_tasks(np.random.RandomState(99), EVAL_TASKS))
    key = jax.random.PRNGKey(0)
    run_variant('mlp', MLP(VOCAB, HIDDEN, key), tokens, held_out)
    run_variant('rnn', RNN(VOCAB, HIDDEN, key), tokens, held_out)


if __name__ == '__main__':
    main()
