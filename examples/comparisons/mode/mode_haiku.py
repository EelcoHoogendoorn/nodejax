"""The mode switch, the haiku side of the comparison.

Self-contained: no nodejax import.

Haiku threads the mode as an ARGUMENT: is_training rides through every
call signature that contains a mode-aware layer, hk.BatchNorm takes it
explicitly, and dropout is a plain `if` around hk.dropout with a key
drawn only on the training branch. The running stats live in the state
dict hk.transform_with_state collects, threaded through every apply.
Nothing is mutated and nothing is global; in exchange the mode is a
parameter of every function between the caller and the layer, and a
middle layer that forgets to pass it down cuts the wire silently.

haiku is a reference-exhibit dependency: the file runs in any
environment with haiku installed.

Run directly:  python -m examples.comparisons.mode.mode_haiku
"""

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
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


def forward(x: jax.Array, is_training: bool):
    """The mode as a parameter of the forward, passed to whoever needs it."""
    x = jax.nn.gelu(hk.Linear(HIDDEN)(x))
    if is_training:
        x = hk.dropout(hk.next_rng_key(), RATE, x)
    x = hk.BatchNorm(create_scale=True, create_offset=True,
                     decay_rate=1 - MOMENTUM)(x, is_training)
    return hk.Linear(CLASSES)(x)


def main() -> None:
    xs, ys = make_data(np.random.RandomState(0))
    f = hk.transform_with_state(forward)
    params, state = f.init(jax.random.PRNGKey(0), xs, True)
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, state, opt_state, key):
        def loss_of(params):
            logits, state2 = f.apply(params, state, key, xs, True)
            return xent(logits, ys), state2

        (loss, state), grads = jax.value_and_grad(loss_of, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), state, opt_state, loss

    losses = []
    key = jax.random.PRNGKey(1)
    for _ in range(STEPS):
        key, sub = jax.random.split(key)
        params, state, opt_state, loss = step(params, state, opt_state, sub)
        losses.append(float(loss))

    key, k1, k2 = jax.random.split(key, 3)
    logits_a, _ = f.apply(params, state, k1, xs, True)   # train mode draws
    logits_b, _ = f.apply(params, state, k2, xs, True)
    train_stochastic = not bool(jnp.allclose(logits_a, logits_b))

    # THE MODE SWITCH: the argument flips, no key owed, and the returned
    # state is discarded rather than threaded on
    eval_a, _ = f.apply(params, state, None, xs, False)
    eval_b, _ = f.apply(params, state, None, xs, False)
    solo, _ = f.apply(params, state, None, xs[:16], False)

    print(f'{"haiku":10s} train_stochastic={train_stochastic} '
          f'eval_deterministic={bool(jnp.allclose(eval_a, eval_b))} '
          f'eval_isolated={bool(jnp.allclose(solo, eval_a[:16], atol=1e-6))} '
          f'loss {losses[0]:.2f} -> {losses[-1]:.2f}', flush=True)


if __name__ == '__main__':
    main()
