"""Train and evaluation mode switching in Haiku.

Haiku threads an explicit `is_training: bool` argument through every forward signature containing a mode-aware layer alongside the `state` dictionary returned by `hk.transform_with_state`. Any intermediate block that omits or hardcodes `is_training` breaks mode propagation silently.

Run directly with `python -m examples.comparisons.mode.mode_haiku`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from examples.comparisons.mode.mode_common import (
    CLASSES, HIDDEN, LR, MOMENTUM, RATE, STEPS, make_data, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


def forward(x: jax.Array, is_training: bool) -> jax.Array:
    x = jax.nn.gelu(hk.Linear(HIDDEN)(x))
    if is_training:
        x = hk.dropout(hk.next_rng_key(), RATE, x)
    x = hk.BatchNorm(
        create_scale=True, create_offset=True, decay_rate=1 - MOMENTUM
    )(x, is_training)
    return hk.Linear(CLASSES)(x)


def main() -> None:
    xs_np, ys_np = make_data(np.random.RandomState(0))
    xs, ys = jnp.asarray(xs_np), jnp.asarray(ys_np)
    f = hk.transform_with_state(forward)
    params, state = f.init(jax.random.PRNGKey(0), xs, True)
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, st, opt_s, key):
        def loss_of(cur_p):
            logits, st2 = f.apply(cur_p, st, key, xs, True)
            return xent(logits, ys), st2

        (loss, st_next), grads = jax.value_and_grad(loss_of, has_aux=True)(p)
        updates, opt_next = opt.update(grads, opt_s, p)
        return optax.apply_updates(p, updates), st_next, opt_next, loss

    losses = []
    key = jax.random.PRNGKey(1)
    for _ in range(STEPS):
        key, sub = jax.random.split(key)
        params, state, opt_state, loss = step(params, state, opt_state, sub)
        losses.append(float(loss))

    key, k1, k2 = jax.random.split(key, 3)
    logits_a, _ = f.apply(params, state, k1, xs, True)
    logits_b, _ = f.apply(params, state, k2, xs, True)
    train_stochastic = not bool(jnp.allclose(logits_a, logits_b))

    eval_a, _ = f.apply(params, state, None, xs, False)
    eval_b, _ = f.apply(params, state, None, xs, False)
    solo, _ = f.apply(params, state, None, xs[:16], False)

    report(
        'haiku',
        train_stochastic,
        bool(jnp.allclose(eval_a, eval_b)),
        bool(jnp.allclose(solo, eval_a[:16], atol=1e-6)),
        losses[0],
        losses[-1],
    )


if __name__ == '__main__':
    main()
