"""Train and evaluation mode switching in Equinox.

Equinox splits train/eval behavior across three separate mechanisms: per-call keys on `eqx.nn.Dropout`, explicit `eqx.nn.State` threading on `eqx.nn.BatchNorm`, and `eqx.nn.inference_mode(model)` to rewrite boolean `inference` leaves. Even after `inference_mode`, every evaluation call must still pass `state` in and unpack `(out, state)` out alongside `key=None`.

Run directly with `python -m examples.comparisons.mode.mode_equinox`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from examples.comparisons.mode.mode_common import (
    CLASSES, DIM, HIDDEN, LR, MOMENTUM, RATE, STEPS, make_data, report,
)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


class Net(eqx.Module):
    lin: eqx.nn.Linear
    drop: eqx.nn.Dropout
    bn: eqx.nn.BatchNorm
    head: eqx.nn.Linear

    def __init__(self, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.lin = eqx.nn.Linear(DIM, HIDDEN, key=k1)
        self.drop = eqx.nn.Dropout(RATE)
        self.bn = eqx.nn.BatchNorm(
            HIDDEN, axis_name='batch', momentum=1 - MOMENTUM, mode='batch'
        )
        self.head = eqx.nn.Linear(HIDDEN, CLASSES, key=k2)

    def __call__(self, x: jax.Array, state: eqx.nn.State, *, key):
        x = jax.nn.gelu(self.lin(x))
        x = self.drop(x, key=key)
        x, state = self.bn(x, state)
        return self.head(x), state


def forward(model: Net, state: eqx.nn.State, xs: jax.Array, key):
    keys = (
        jax.random.split(key, xs.shape[0])
        if key is not None
        else jnp.zeros((xs.shape[0], 2), jnp.uint32)
    )
    batched = jax.vmap(
        lambda x, k: model(x, state, key=k if key is not None else None),
        in_axes=(0, 0),
        out_axes=(0, None),
        axis_name='batch',
    )
    return batched(xs, keys)


def main() -> None:
    xs_np, ys_np = make_data(np.random.RandomState(0))
    xs, ys = jnp.asarray(xs_np), jnp.asarray(ys_np)
    model, state = eqx.nn.make_with_state(Net)(jax.random.PRNGKey(0))
    opt = optax.adam(LR)
    arrays, static = eqx.partition(model, eqx.is_array)
    opt_state = opt.init(arrays)

    @jax.jit
    def step(arrs, st, opt_s, key):
        def loss_of(cur_arrs):
            out, st2 = forward(eqx.combine(cur_arrs, static), st, xs, key)
            return xent(out, ys), st2

        (loss, st_next), grads = jax.value_and_grad(loss_of, has_aux=True)(arrs)
        updates, opt_next = opt.update(grads, opt_s, arrs)
        return eqx.apply_updates(arrs, updates), st_next, opt_next, loss

    losses = []
    key = jax.random.PRNGKey(1)
    for _ in range(STEPS):
        key, sub = jax.random.split(key)
        arrays, state, opt_state, loss = step(arrays, state, opt_state, sub)
        losses.append(float(loss))
    model = eqx.combine(arrays, static)

    key, k1, k2 = jax.random.split(key, 3)
    logits_a, state = forward(model, state, xs, k1)
    logits_b, state = forward(model, state, xs, k2)
    train_stochastic = not bool(jnp.allclose(logits_a, logits_b))

    inference = eqx.nn.inference_mode(model)
    eval_a, state = forward(inference, state, xs, None)
    eval_b, state = forward(inference, state, xs, None)
    solo, state = forward(inference, state, xs[:16], None)

    report(
        'equinox',
        train_stochastic,
        bool(jnp.allclose(eval_a, eval_b)),
        bool(jnp.allclose(solo, eval_a[:16], atol=1e-6)),
        losses[0],
        losses[-1],
    )


if __name__ == '__main__':
    main()
