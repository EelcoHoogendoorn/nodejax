"""A reusable residual wrapper in idiomatic Equinox.

Same trivial transform, and a substrate that shares nothing with the NNX
column: an ``eqx.Module`` is a pytree, so the wrapper is a dataclass holding
its member, and a stateful member returns a new copy of itself rather than
mutating a variable.

Equinox pays one price the NNX column does not, and it is visible in the
protocol below. Since a member may carry state and there is no mutable place
to put it, every member returns ``(successor, output)`` whether or not it has
anything to carry. That is a declared convention, and a wider one than NNX
needed. The entropy fact is still discovered rather than declared, and it is
still the one that decides whether the wrapper composes with itself.

Run directly:
    python -m nodejax.examples.comparisons.residual.residual_equinox
"""

from __future__ import annotations

import inspect
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

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


class WithAux(NamedTuple):
    value: Any
    aux: Any


class BodyAux(NamedTuple):
    mean: jax.Array


def emit(value, aux):
    return WithAux(value, aux)


def _split_output(output):
    return output if type(output) is WithAux else (output, None)


class Residual(eqx.Module):
    """x + f(x) around a member satisfying this file's call protocol."""

    inner: eqx.Module
    takes_key: bool = eqx.field(static=True)

    def __init__(self, inner: eqx.Module):
        self.inner = inner
        self.takes_key = 'key' in inspect.signature(
            type(inner).__call__).parameters

    def __call__(self, value, *, key=None):
        if self.takes_key and key is None:
            raise TypeError('stochastic residual requires key=')
        if not self.takes_key and key is not None:
            raise TypeError('deterministic residual does not accept key=')
        successor, output = (self.inner(value, key=key) if self.takes_key
                             else self.inner(value))
        clean, aux = _split_output(output)
        advanced = eqx.tree_at(lambda wrapper: wrapper.inner, self, successor)
        added = value + clean
        return advanced, (emit(added, aux) if aux is not None else added)


class NoisyStatefulBody(eqx.Module):
    weight: jax.Array
    mean: jax.Array

    def __init__(self, key: jax.Array):
        self.weight = 0.25 + jax.random.normal(key, (WIDTH,)) / WIDTH
        self.mean = jnp.zeros(WIDTH)

    def __call__(self, value, *, key):
        mean = DECAY * self.mean + (1 - DECAY) * value
        keep = jax.random.bernoulli(key, 1 - DROP_RATE, shape=value.shape)
        output = jnp.tanh(self.weight * value + mean) * keep / (1 - DROP_RATE)
        successor = eqx.tree_at(lambda body: body.mean, self, mean)
        return successor, emit(output, BodyAux(mean))


class BareBody(eqx.Module):
    weight: jax.Array

    def __init__(self, key: jax.Array):
        self.weight = 0.9 + 0.01 * jax.random.normal(key, (WIDTH,))

    def __call__(self, value):
        return self, self.weight * value


def _nests() -> bool:
    """Whether the wrapper accepts its own product, deterministic member and
    stochastic member alike."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    param_key = jax.random.PRNGKey(PARAM_SEED)
    try:
        Residual(Residual(BareBody(param_key)))(inputs)
        Residual(Residual(NoisyStatefulBody(param_key)))(
            inputs, key=jax.random.PRNGKey(DROPOUT_SEED))
    except Exception:
        return False
    return True


def _state_gets_gradient(param_key, apply_key) -> bool:
    """Whether a gradient reaches the member\'s running statistic.

    It does. ``weight`` and ``mean`` are both inexact arrays in one pytree,
    and ``eqx.is_inexact_array`` is what selects what to differentiate, so
    nothing separates a parameter from a running statistic except the field
    name. The wrapper forwards this rather than causing it: with no mutable
    variable, a returned successor is where state has to live.
    """
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    wrapped = Residual(NoisyStatefulBody(param_key))

    def loss(member):
        _, result = member(inputs, key=apply_key)
        output, _ = _split_output(result)
        return jnp.sum(output ** 2)

    gradient = eqx.filter_grad(loss)(wrapped)
    return len(jax.tree.leaves(gradient)) > 1


def evidence() -> Evidence:
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    param_key = jax.random.PRNGKey(PARAM_SEED)
    apply_key = jax.random.PRNGKey(DROPOUT_SEED)
    other_key = jax.random.PRNGKey(OTHER_DROPOUT_SEED)

    first = Residual(NoisyStatefulBody(param_key))
    before = first.inner.mean
    advanced, result = first(inputs, key=apply_key)
    output, aux = _split_output(result)

    _, replay_result = Residual(NoisyStatefulBody(param_key))(
        inputs, key=apply_key)
    replay_output, _ = _split_output(replay_result)
    _, other_result = Residual(NoisyStatefulBody(param_key))(
        inputs, key=other_key)
    other_output, _ = _split_output(other_result)

    bare = Residual(BareBody(param_key))

    return Evidence(
        output_shape=output.shape,
        parameter_shape=bare.inner.weight.shape,
        state_advanced=not bool(jnp.allclose(before, advanced.inner.mean)),
        replayed=bool(jnp.allclose(output, replay_output)),
        different_draw=not bool(jnp.allclose(output, other_output)),
        aux_escaped=(aux is not None
                     and bool(jnp.allclose(aux.mean, advanced.inner.mean))),
        nests=_nests(),
        state_gets_gradient=_state_gets_gradient(param_key, apply_key),
    )


def main() -> Evidence:
    return verify('equinox', evidence())


if __name__ == '__main__':
    main()
