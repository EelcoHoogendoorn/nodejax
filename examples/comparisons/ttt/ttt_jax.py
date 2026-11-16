"""Bespoke raw-JAX baseline for the RNN test-time training benchmark.

Hand-rolling one fixed recurrent model in raw JAX keeps the `(w, h, prev)` carry tuple, per-leaf step-size dict, `vmap` over tasks, and outer `lax.scan` explicit in one file, at the cost of zero reusable wrapper structure: switching from `RNN` to `MLP`, adding layers, or changing the inner optimizer rewrites `forecast_sequence` directly.

Run directly with `python -m examples.comparisons.ttt.ttt_jax`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from examples.comparisons.ttt.ttt_common import (
    EVAL_TASKS, HIDDEN, META_LR, META_STEPS, QUERY0, TASKS, TTT_LR0, VOCAB,
    make_tasks, query_nll_numpy, report,
)


def rnn_init(key: jax.Array) -> dict[str, jax.Array]:
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(
        embed=0.3 * jax.random.normal(k1, (VOCAB, HIDDEN)),
        wh=0.3 * jax.random.normal(k2, (HIDDEN, HIDDEN)) / jnp.sqrt(HIDDEN),
        out=0.1 * jax.random.normal(k3, (HIDDEN, VOCAB)),
    )


def rnn_apply(w: dict[str, jax.Array], h: jax.Array,
              token: jax.Array) -> tuple[jax.Array, jax.Array]:
    h = jnp.tanh(w['embed'][token] + w['wh'] @ h)
    return h, h @ w['out']


def forecast_sequence(theta: dict, tokens: jax.Array) -> jax.Array:
    def cell(carry, token):
        w, h, prev = carry

        def selfsup(wf):
            h2, logits = rnn_apply(wf, h, prev)
            return -jax.nn.log_softmax(logits)[token], (h2, logits)

        grads, (h2, logits) = jax.grad(selfsup, has_aux=True)(w)
        w2 = jax.tree.map(lambda wi, g, lr: wi - lr * g, w, grads, theta['lr'])
        return (w2, h2, token), logits

    start = (theta['init'], jnp.zeros(HIDDEN), tokens[0])
    _, logits = jax.lax.scan(cell, start, tokens)
    return logits


def batched_forecast(theta: dict, batch: jax.Array) -> jax.Array:
    return jax.vmap(lambda seq: forecast_sequence(theta, seq))(batch)


def query_xent(theta: dict, batch: jax.Array) -> jax.Array:
    logits = batched_forecast(theta, batch)
    logp = jax.nn.log_softmax(logits)
    picked = jnp.take_along_axis(logp, batch[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked[..., QUERY0:])


def main() -> None:
    init = rnn_init(jax.random.PRNGKey(0))
    theta0 = dict(init=init, lr=jax.tree.map(lambda x: jnp.full_like(x, TTT_LR0), init))
    opt = optax.adam(META_LR)

    tokens = jnp.asarray(
        make_tasks(np.random.RandomState(0), META_STEPS * TASKS).reshape(META_STEPS, TASKS, -1)
    )
    held_out = jnp.asarray(make_tasks(np.random.RandomState(99), EVAL_TASKS))
    preds0 = batched_forecast(theta0, held_out)

    def meta_step(carry, batch):
        theta, opt_state = carry
        loss, grads = jax.value_and_grad(query_xent)(theta, batch)
        updates, opt_state = opt.update(grads, opt_state, theta)
        return (optax.apply_updates(theta, updates), opt_state), loss

    (theta, _), losses = jax.lax.scan(meta_step, (theta0, opt.init(theta0)), tokens)
    preds1 = batched_forecast(theta, held_out)
    n_weights = sum(x.size for x in jax.tree.leaves(theta))
    report(
        'raw_jax', 'rnn', n_weights, bool(jnp.all(jnp.isfinite(losses))),
        query_nll_numpy(preds0, held_out), query_nll_numpy(preds1, held_out),
    )


if __name__ == '__main__':
    main()
