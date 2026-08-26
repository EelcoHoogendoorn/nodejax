"""NodeJAX's stock residual under the unary-signal wrapper contract.

The whole transform is six lines in ``nodejax/transforms/wiring/residual.py``. Those
lines do not mention params, state construction, priming, entropy, aux, or
member methods. ``self.body(input)`` is a member call, so the common Def
contract carries those facts through the wrapper. Deferred construction also
survives, so the result can still be specialized before it is bound.

The apply form itself is an honest limit of this particular transform. Its
authored apply declares one field named ``input``. A body with another runtime
argument works directly but that argument is not published by ``residual``.
The framework can represent and route such calls, but this six-line residual
does not claim semantics for which argument is the signal and which ones are
merely forwarded. The probes below report that boundary rather than hiding it.

Within the declared unary form there is no second state or initialization
protocol to adapt. Every Node presents the same roles to the wrapper, whether
it was authored as a leaf, a composition, or another transform.

Run directly:
    python -m examples.comparisons.residual.residual_nodejax
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax import Aux, Leaf, node, residual, split_aux
from nodejax.struct import Struct
from examples.comparisons.residual.residual_common import (
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


def _preserves_priming() -> bool:
    """Whether a body can still initialize state from its first real input."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)

    def init(input):
        return input

    def apply(state, input):
        return input, state

    wrapped = residual(Leaf(apply, init=init, name='primed')).with_input(inputs)
    state = wrapped.init(input=inputs)
    return bool(jnp.allclose(state, inputs))


def _preserves_methods() -> bool:
    """Whether methods declared by the body remain available on the result."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)

    def doubled(input):
        return 2 * input

    body = Leaf(lambda input: input, methods={'doubled': doubled}, name='method_body')
    return bool(jnp.allclose(residual(body).doubled(inputs), 2 * inputs))


def _forwards_extra_arguments() -> bool:
    """Whether this unary transform republishes additional body arguments."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    scale = jnp.asarray(2.0)
    body = Leaf(lambda input, scale: scale * input, name='scaled')
    expected = body.apply(input=inputs, scale=scale)
    try:
        output = residual(body).apply(input=inputs, scale=scale)
    except TypeError:
        return False
    return bool(jnp.allclose(output, expected + inputs))


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
    result = verify('nodejax', evidence())
    print(
        'nodejax      '
        f'priming={"yes" if _preserves_priming() else "NO"} '
        f'methods={"yes" if _preserves_methods() else "NO"} '
        f'extra-arguments={"yes" if _forwards_extra_arguments() else "NO"}',
        flush=True,
    )
    return result


if __name__ == '__main__':
    main()
