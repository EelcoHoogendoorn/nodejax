"""An attempt at a reusable residual wrapper in Equinox.

Equinox supplies a useful common layer convention. Ordinary layers return an
output. ``eqx.nn.StatefulLayer`` marks a layer that instead receives an
external ``eqx.nn.State`` and returns ``(output, state)``. The residual can
delegate that declaration, forward every call argument mechanically, and add
only to the output. Native state initialization through ``make_with_state``,
RNG, ordinary modules, aux marked by this file, and self-composition all work.

That is genuine composition within the Sequential-compatible layer protocol.
It is not a set-and-forget wrapper over every ``eqx.Module``. A Module may use
any call and return structure. A body that carries state by returning a
successor Module is not a ``StatefulLayer``, so its pair is indistinguishable
from an ordinary structured output. RNN cells use their own positional hidden
state. Other Modules may publish methods or attributes that the wrapper does
not reproduce. Aux also needs the local ``WithAux`` marker used below. That
marker survives another residual which knows about it, but an ordinary next
layer receives the marker itself rather than the clean output.

Native state updates are functional, but ``State.set`` invalidates the old
State. A member can therefore consume it before the residual discovers that its
output cannot be added. A failed wrapper call then has no successor to return,
and the caller's old State is no longer valid.

Construction remains outside the value. Wrapping one configured Module is
ordinary Equinox, but a later stack or ensemble that needs fresh copies must
receive a factory which constructs the member and residual together. The
order of those operations is therefore not uniformly Module to Module.

The comparison next door makes the split concrete. Its Equinox layer stack
uses returned successor Modules because native ``StateIndex`` values do not
provide independent per-layer slots when constructed under ``filter_vmap``.
This residual uses native ``State`` because it is the sound choice here. Each
transform works within its chosen state protocol, but the two protocols do
not compose with each other.

A general Equinox residual therefore needs a common description of call
binding, state, output projection, aux, construction, and member interfaces,
or separate adapters for each protocol. The probes below report the boundary:
native State and unary self-composition work; returned-successor state does
not.

Run directly:
    python -m examples.comparisons.residual.residual_equinox
"""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

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


class WithAux(NamedTuple):
    value: Any
    aux: Any


class BodyAux(NamedTuple):
    mean: jax.Array


def emit(value, aux):
    return WithAux(value, aux)


def _split_output(output):
    return output if type(output) is WithAux else (output, None)


class Residual(eqx.nn.StatefulLayer):
    """x + f(x) for a member following Equinox's Sequential layer protocol."""

    inner: eqx.Module

    def __init__(self, inner: eqx.Module):
        self.inner = inner

    def is_stateful(self) -> bool:
        return isinstance(self.inner, eqx.nn.StatefulLayer) and self.inner.is_stateful()

    def __call__(self, value, *args, **kwargs):
        stateful = self.is_stateful()
        result = self.inner(value, *args, **kwargs)
        if stateful:
            output, state = result
        else:
            output = result
        clean, aux = _split_output(output)
        added = value + clean
        result = emit(added, aux) if aux is not None else added
        return (result, state) if stateful else result


class NoisyStatefulBody(eqx.nn.StatefulLayer):
    weight: jax.Array
    mean: eqx.nn.StateIndex

    def __init__(self, key: jax.Array):
        self.weight = 0.25 + jax.random.normal(key, (WIDTH,)) / WIDTH
        self.mean = eqx.nn.StateIndex(jnp.zeros(WIDTH))

    def __call__(self, value, state, *, key):
        mean = DECAY * state.get(self.mean) + (1 - DECAY) * value
        keep = jax.random.bernoulli(key, 1 - DROP_RATE, shape=value.shape)
        output = jnp.tanh(self.weight * value + mean) * keep / (1 - DROP_RATE)
        return emit(output, BodyAux(mean)), state.set(self.mean, mean)


class BareBody(eqx.Module):
    weight: jax.Array

    def __init__(self, key: jax.Array):
        self.weight = 0.9 + 0.01 * jax.random.normal(key, (WIDTH,))

    def __call__(self, value, *, key=None):
        return self.weight * value


class SuccessorBody(eqx.Module):
    """The alternative returned-successor state convention used by the stack comparison."""

    weight: jax.Array
    mean: jax.Array

    def __init__(self, key: jax.Array):
        self.weight = 0.9 + 0.01 * jax.random.normal(key, (WIDTH,))
        self.mean = jnp.zeros(WIDTH)

    def __call__(self, value):
        mean = DECAY * self.mean + (1 - DECAY) * value
        successor = eqx.tree_at(lambda body: body.mean, self, mean)
        return successor, self.weight * value + mean


def _build(key: jax.Array):
    return eqx.nn.make_with_state(lambda: Residual(NoisyStatefulBody(key)))()


def _nests() -> bool:
    """Whether the wrapper accepts its own deterministic and stochastic products."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    param_key = jax.random.PRNGKey(PARAM_SEED)
    try:
        Residual(Residual(BareBody(param_key)))(inputs)
        stochastic, state = eqx.nn.make_with_state(
            lambda: Residual(Residual(NoisyStatefulBody(param_key))))()
        stochastic(inputs, state, key=jax.random.PRNGKey(DROPOUT_SEED))
    except Exception:
        return False
    return True


def _preserves_native_state() -> bool:
    """Whether native Equinox state still works through the wrapper and Sequential."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    param_key = jax.random.PRNGKey(PARAM_SEED)
    apply_key = jax.random.PRNGKey(DROPOUT_SEED)
    model, state = eqx.nn.make_with_state(
        lambda: eqx.nn.Sequential([Residual(NoisyStatefulBody(param_key))]))()
    before = state.get(model.layers[0].inner.mean)
    output, state = model(inputs, state, key=apply_key)
    after = state.get(model.layers[0].inner.mean)
    clean, aux = _split_output(output)
    return bool(
        clean.shape == inputs.shape
        and aux is not None
        and not jnp.allclose(before, after)
        and jnp.allclose(aux.mean, after)
    )


def _preserves_successor_state() -> bool:
    """Whether the same wrapper understands the returned-successor convention."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    param_key = jax.random.PRNGKey(PARAM_SEED)
    successor, expected = SuccessorBody(param_key)(inputs)
    assert not bool(jnp.allclose(successor.mean, jnp.zeros(WIDTH)))

    try:
        advanced, output = Residual(SuccessorBody(param_key))(inputs)
        actual_mean = advanced.inner.mean
    except Exception:
        return False
    return bool(
        jnp.allclose(actual_mean, successor.mean)
        and jnp.allclose(output, inputs + expected)
    )


def _state_gets_gradient(param_key, apply_key) -> bool:
    """Whether a gradient reaches the member\'s running statistic.

    It does not. ``make_with_state`` removes the running statistic from the
    Module pytree and puts it in a separate ``eqx.nn.State`` value, so
    differentiating the Module reaches only the weight.
    """
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    wrapped, state = _build(param_key)

    def loss(member):
        result, _ = member(inputs, state, key=apply_key)
        output, _ = _split_output(result)
        return jnp.sum(output ** 2)

    gradient = eqx.filter_grad(loss)(wrapped)
    return len(jax.tree.leaves(gradient)) > 1


def evidence() -> Evidence:
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    param_key = jax.random.PRNGKey(PARAM_SEED)
    apply_key = jax.random.PRNGKey(DROPOUT_SEED)
    other_key = jax.random.PRNGKey(OTHER_DROPOUT_SEED)

    first, first_state = _build(param_key)
    replay, replay_state = _build(param_key)
    other, other_state = _build(param_key)

    before = first_state.get(first.inner.mean)
    result, advanced_state = first(inputs, first_state, key=apply_key)
    output, aux = _split_output(result)
    after = advanced_state.get(first.inner.mean)

    replay_result, _ = replay(inputs, replay_state, key=apply_key)
    replay_output, _ = _split_output(replay_result)
    other_result, _ = other(inputs, other_state, key=other_key)
    other_output, _ = _split_output(other_result)

    bare = Residual(BareBody(param_key))

    return Evidence(
        output_shape=output.shape,
        parameter_shape=bare.inner.weight.shape,
        state_advanced=not bool(jnp.allclose(before, after)),
        replayed=bool(jnp.allclose(output, replay_output)),
        different_draw=not bool(jnp.allclose(output, other_output)),
        aux_escaped=aux is not None and bool(jnp.allclose(aux.mean, after)),
        nests=_nests(),
        state_gets_gradient=_state_gets_gradient(param_key, apply_key),
    )


def main() -> Evidence:
    result = verify('equinox', evidence())
    print(
        'equinox      '
        f'native-state={"yes" if _preserves_native_state() else "NO"} '
        f'successor-state={"yes" if _preserves_successor_state() else "NO"}',
        flush=True,
    )
    return result


if __name__ == '__main__':
    main()
