"""A reusable residual wrapper in idiomatic Flax NNX.

The transform is trivial: add the input to the member's output. It owns no
axis, no parameters, no state and no entropy of its own. Everything below the
addition is forwarding.

NNX supplies a native answer for aux. The body ``sow``s its intermediate and
``nnx.capture`` collects it outside the residual call, so the wrapper sees
only the value it must add and needs no private return convention.

RNG forwarding remains different. The wrapper must inspect whether its member
accepts ``rngs=`` because deterministic and stochastic calls have different
spellings. Its own signature must then expose ``rngs=`` even when its member is
deterministic. A second residual reads that forwarding slot as an RNG
requirement, so stochastic self-composition works and deterministic
self-composition does not.

Run directly:
    python -m nodejax.examples.comparisons.residual.residual_nnx
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
from flax import nnx

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


_MISSING = object()


class Residual(nnx.Module):
    """x + f(x) around a member satisfying this file's call protocol."""

    def __init__(self, inner: nnx.Module):
        self.inner = inner
        self.takes_rng = 'rngs' in inspect.signature(
            inner.__call__).parameters

    def __call__(self, value, *, rngs=_MISSING):
        if self.takes_rng and rngs is _MISSING:
            raise TypeError('stochastic residual requires rngs=')
        if not self.takes_rng and rngs is not _MISSING:
            raise TypeError('deterministic residual does not accept rngs=')
        output = (self.inner(value, rngs=rngs) if self.takes_rng
                  else self.inner(value))
        return value + output


class NoisyStatefulBody(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            0.25 + jax.random.normal(rngs.params(), (WIDTH,)) / WIDTH)
        self.mean = nnx.Variable(jnp.zeros(WIDTH))

    def __call__(self, value, *, rngs):
        mean = DECAY * self.mean[...] + (1 - DECAY) * value
        self.mean[...] = mean
        keep = jax.random.bernoulli(
            rngs.dropout(), 1 - DROP_RATE, shape=value.shape)
        output = jnp.tanh(
            self.weight[...] * value + mean) * keep / (1 - DROP_RATE)
        self.sow(nnx.Intermediate, 'mean', mean)
        return output


class BareBody(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            0.9 + 0.01 * jax.random.normal(rngs.params(), (WIDTH,)))

    def __call__(self, value):
        return self.weight[...] * value


def _construction_rngs() -> nnx.Rngs:
    return nnx.Rngs(params=PARAM_SEED)


def _apply_rngs(seed: int) -> nnx.Rngs:
    return nnx.Rngs(dropout=seed)


@nnx.capture(nnx.Intermediate)
def _captured_call(member, value, *, rngs):
    return member(value, rngs=rngs)


def _nests() -> bool:
    """Whether the wrapper accepts its own product, deterministic member and
    stochastic member alike.

    The stochastic case passes and the deterministic one does not, which is
    the sharper half of the finding: the inner wrapper declares ``rngs=``
    because it may forward one, so the outer reads it as a member that draws
    and demands a key ``BareBody`` never consumes.
    """
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    try:
        Residual(Residual(BareBody(_construction_rngs())))(inputs)
    except Exception:
        deterministic = False
    else:
        deterministic = True

    try:
        _captured_call(
            Residual(Residual(NoisyStatefulBody(_construction_rngs()))),
            inputs,
            rngs=_apply_rngs(DROPOUT_SEED),
        )
    except Exception:
        stochastic = False
    else:
        stochastic = True

    return deterministic and stochastic


def _state_gets_gradient() -> bool:
    """Whether a gradient reaches the member\'s running statistic.

    It does not. ``nnx.grad`` differentiates ``nnx.Param`` by default, and
    the running statistic is a plain ``nnx.Variable``, so the Variable
    subclass is what keeps the two apart.
    """
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    wrapped = Residual(NoisyStatefulBody(_construction_rngs()))

    def loss(member):
        output, _ = _captured_call(
            member, inputs, rngs=_apply_rngs(DROPOUT_SEED))
        return jnp.sum(output ** 2)

    gradient = nnx.grad(loss)(wrapped)
    return len(jax.tree.leaves(gradient)) > 1


def evidence() -> Evidence:
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    first = Residual(NoisyStatefulBody(_construction_rngs()))
    replay = Residual(NoisyStatefulBody(_construction_rngs()))
    other = Residual(NoisyStatefulBody(_construction_rngs()))

    before = first.inner.mean[...]
    output, captured = _captured_call(
        first, inputs, rngs=_apply_rngs(DROPOUT_SEED))
    replay_output, _ = _captured_call(
        replay, inputs, rngs=_apply_rngs(DROPOUT_SEED))
    other_output, _ = _captured_call(
        other, inputs, rngs=_apply_rngs(OTHER_DROPOUT_SEED))
    after = first.inner.mean[...]
    aux_mean = captured['inner']['mean'].get_value()[-1]

    bare = Residual(BareBody(_construction_rngs()))

    return Evidence(
        output_shape=output.shape,
        parameter_shape=bare.inner.weight[...].shape,
        state_advanced=not bool(jnp.allclose(before, after)),
        replayed=bool(jnp.allclose(output, replay_output)),
        different_draw=not bool(jnp.allclose(output, other_output)),
        aux_escaped=bool(jnp.allclose(aux_mean, after)),
        nests=_nests(),
        state_gets_gradient=_state_gets_gradient(),
    )


def main() -> Evidence:
    return verify('flax nnx', evidence())


if __name__ == '__main__':
    main()
