"""NodeJAX's stock residual under the transparent-wrapper parity contract.

The whole transform is six lines in `nodejax/transforms/residual.py`, and the
point of this column is what those lines do not say. They do not mention
params, state, entropy or aux. `self.body(input)` is a member call, so every
fact the member declares is forwarded by the machinery rather than restated
by the wrapper.

Run directly:
    python -m nodejax.examples.comparisons.residual.residual_nodejax
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax import Aux, Leaf, node, residual, split_aux
from nodejax.struct import Struct
from nodejax.examples.comparisons.residual.residual_common import (
    DECAY,
    DROPOUT_SEED,
    DROP_RATE,
    Evidence,
    OTHER_DROPOUT_SEED,
    PARAM_SEED,
    WIDTH,
    verify,
)


@node
def NoisyStatefulBody():
    """One member carrying every fact a wrapper has to forward: a parameter,
    running state, apply-time entropy, and an auxiliary output."""
    def param(rng):
        return Struct(
            weight=0.25 + jax.random.normal(rng.next(), (WIDTH,)) / WIDTH)

    def init():
        return Struct(mean=jnp.zeros(WIDTH))

    def apply(param, state, input, rng):
        mean = DECAY * state.mean + (1 - DECAY) * input
        keep = jax.random.bernoulli(
            rng.next(), 1 - DROP_RATE, shape=input.shape)
        output = jnp.tanh(param.weight * input + mean) * keep / (1 - DROP_RATE)
        return Struct(mean=mean), (output, Aux(mean=mean))

    return Leaf(apply, param=param, init=init, name='noisy_stateful')


@node
def BareBody():
    """A deterministic member with no state and no aux."""
    def param(rng):
        return Struct(
            weight=0.9 + 0.01 * jax.random.normal(rng.next(), (WIDTH,)))

    def apply(param, input):
        return param.weight * input

    return Leaf(apply, param=param, name='bare')


def _build(body, key):
    return residual(body).parameterize(rng=key).initialize()


def _nests(key, apply_key) -> bool:
    """A residual around a residual, deterministic member and stochastic
    member alike: the product is a member like any other."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    try:
        residual(residual(BareBody())).parameterize(rng=key).apply(inputs)
        _build(residual(NoisyStatefulBody()), key).apply(
            input=inputs, rng=apply_key)
    except Exception:
        return False
    return True


def _state_gets_gradient(param_key, apply_key) -> bool:
    """Whether a gradient reaches the member\'s running statistic.

    It cannot. ``param`` and ``state`` are separate trees, so a gradient
    taken against the parameters has nowhere to put one for state.
    """
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    wrapped = _build(NoisyStatefulBody(), param_key)

    def loss(param):
        _, result = wrapped.bind(param).apply(input=inputs, rng=apply_key)
        output, _ = split_aux(result)
        return jnp.sum(output ** 2)

    gradient = jax.grad(loss)(wrapped.param)
    return len(jax.tree.leaves(gradient)) > len(jax.tree.leaves(wrapped.param))


def evidence() -> Evidence:
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    param_key = jax.random.PRNGKey(PARAM_SEED)
    apply_key = jax.random.PRNGKey(DROPOUT_SEED)
    other_key = jax.random.PRNGKey(OTHER_DROPOUT_SEED)

    first = _build(NoisyStatefulBody(), param_key)
    replay = _build(NoisyStatefulBody(), param_key)
    other = _build(NoisyStatefulBody(), param_key)

    before = first.state.mean
    advanced, result = first.apply(input=inputs, rng=apply_key)
    output, aux = split_aux(result)
    _, replay_result = replay.apply(input=inputs, rng=apply_key)
    replay_output, _ = split_aux(replay_result)
    _, other_result = other.apply(input=inputs, rng=other_key)
    other_output, _ = split_aux(other_result)

    bare = residual(BareBody()).parameterize(rng=param_key)

    return Evidence(
        output_shape=output.shape,
        parameter_shape=bare.param.weight.shape,
        state_advanced=not bool(jnp.allclose(before, advanced.state.mean)),
        replayed=bool(jnp.allclose(output, replay_output)),
        different_draw=not bool(jnp.allclose(output, other_output)),
        aux_escaped=(aux is not None
                     and bool(jnp.allclose(aux.mean, advanced.state.mean))),
        nests=_nests(param_key, apply_key),
        state_gets_gradient=_state_gets_gradient(param_key, apply_key),
    )


def main() -> Evidence:
    return verify('nodejax', evidence())


if __name__ == '__main__':
    main()
