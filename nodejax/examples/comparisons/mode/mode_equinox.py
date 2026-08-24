"""The mode switch, the equinox side of the comparison.

Self-contained: no nodejax import.

Equinox splits the mode across three mechanisms. Dropout takes a KEY per
call and an inference flag; BatchNorm keeps its running stats in an
eqx.nn.State object the caller threads through every forward by hand;
and the switch itself is eqx.nn.inference_mode(model), a tree rewrite
that flips the inference flag leaves. So training threads keys and
state, eval threads the state it froze and no keys, and the caller owns
every strand: nothing is mutated behind your back, and in exchange the
mode lives in three places at once.

equinox is a reference-exhibit dependency: the file runs in any
environment with equinox installed.

Run directly:  python -m nodejax.examples.comparisons.mode.mode_equinox
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

DIM, HIDDEN, CLASSES = 8, 16, 3
N, STEPS = 128, 200
RATE, MOMENTUM, LR = 0.3, 0.1, 0.02


def make_data(rs):
    centers = 2.0 * rs.standard_normal((CLASSES, DIM))
    labels = rs.randint(CLASSES, size=N)
    xs = centers[labels] + rs.standard_normal((N, DIM))
    return jnp.asarray(xs, jnp.float32), jnp.asarray(labels, jnp.int32)


def xent(logits: jax.Array, target: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(logp, target[..., None], axis=-1))


class Net(eqx.Module):
    lin: eqx.nn.Linear
    drop: eqx.nn.Dropout
    bn: eqx.nn.BatchNorm
    head: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.lin = eqx.nn.Linear(DIM, HIDDEN, key=k1)
        self.drop = eqx.nn.Dropout(RATE)
        self.bn = eqx.nn.BatchNorm(
            HIDDEN, axis_name='batch', momentum=1 - MOMENTUM, mode='batch')
        self.head = eqx.nn.Linear(HIDDEN, CLASSES, key=k2)

    def __call__(self, x, state, *, key):
        x = jax.nn.gelu(self.lin(x))
        x = self.drop(x, key=key)          # a key per call, or inference
        x, state = self.bn(x, state)       # the stats, threaded by hand
        return self.head(x), state


def forward(model, state, xs: jax.Array, key: jax.Array):
    """One batched pass: per-sample keys for dropout, the norm state
    broadcast in and one updated copy out, the batch axis named for the
    norm's collectives."""
    keys = (jax.random.split(key, xs.shape[0]) if key is not None
            else jnp.zeros((xs.shape[0], 2), jnp.uint32))
    batched = jax.vmap(lambda x, k: model(x, state, key=k if key is not None else None),
                       in_axes=(0, 0), out_axes=(0, None), axis_name='batch')
    return batched(xs, keys)


def main() -> None:
    xs, ys = make_data(np.random.RandomState(0))
    model, state = eqx.nn.make_with_state(Net)(jax.random.PRNGKey(0))
    opt = optax.adam(LR)
    arrays, static = eqx.partition(model, eqx.is_array)
    opt_state = opt.init(arrays)

    @jax.jit
    def step(arrays, state, opt_state, key):
        def loss_of(arrays):
            out, state2 = forward(eqx.combine(arrays, static), state, xs, key)
            return xent(out, ys), state2

        (loss, state), grads = jax.value_and_grad(loss_of, has_aux=True)(arrays)
        updates, opt_state = opt.update(grads, opt_state, arrays)
        return eqx.apply_updates(arrays, updates), state, opt_state, loss

    losses = []
    key = jax.random.PRNGKey(1)
    for _ in range(STEPS):
        key, sub = jax.random.split(key)
        arrays, state, opt_state, loss = step(arrays, state, opt_state, sub)
        losses.append(float(loss))
    model = eqx.combine(arrays, static)

    # the State is LINEAR: every use consumes it and returns the next,
    # so even a read-only check threads it on
    key, k1, k2 = jax.random.split(key, 3)
    logits_a, state = forward(model, state, xs, k1)   # train mode: keys drawn
    logits_b, state = forward(model, state, xs, k2)
    train_stochastic = not bool(jnp.allclose(logits_a, logits_b))

    # THE MODE SWITCH: a tree rewrite flips the inference leaves, and the
    # caller stops passing keys; the state still threads, now unchanging
    inference = eqx.nn.inference_mode(model)
    eval_a, state = forward(inference, state, xs, None)
    eval_b, state = forward(inference, state, xs, None)
    solo, state = forward(inference, state, xs[:16], None)

    print(f'{"equinox":10s} train_stochastic={train_stochastic} '
          f'eval_deterministic={bool(jnp.allclose(eval_a, eval_b))} '
          f'eval_isolated={bool(jnp.allclose(solo, eval_a[:16], atol=1e-6))} '
          f'loss {losses[0]:.2f} -> {losses[-1]:.2f}', flush=True)


if __name__ == '__main__':
    main()
