"""An attempt at a reusable residual wrapper in Flax NNX.

For a module shaped like ``value -> output``, NNX makes this direct. Parameters
and graph-resident Variables live on the member, and graph-aware transforms,
Param filtering, mutation, and ``sow``/``capture`` traverse the wrapper. Extra
call arguments can pass through without inspecting whether one happens to be
``rngs``, so deterministic and stochastic residuals both nest.

That is genuine composition within the unary-value protocol chosen here. It is
not a set-and-forget wrapper over every ``nnx.Module``. NNX does not declare
which argument is the signal, where the output lives, or whether state is a
mutated Variable or an explicit functional carry. Stock RNN cells use
``initialize_carry`` and ``(carry, input) -> (carry, output)``. This wrapper
neither preserves the cell interface nor knows to add only to the returned
output. Cache-bearing modules use another lifecycle method, ``init_cache``.

Construction has the same boundary. Wrapping one configured module is ordinary
NNX. A later stack or ensemble that needs independently initialized copies must
instead receive a factory that constructs the member and residual together.
The configured module carries its values, not the construction call that made
them.

Later transforms must also describe the two state forms separately. ``StateAxes``
maps or carries graph Variables, while a distinct call argument is marked as
functional ``Carry``. Neither choice records when that state resets. Finally,
the residual has no output contract with which to check addition before the
member runs, so an eager member may mutate graph state before an incompatible
output makes the addition fail.

A general NNX residual therefore needs separate adapters for unary modules,
RNN cells, and other call protocols, or one common description of construction,
initialization, call binding, functional state, and output projection. The
executable probes below report the boundary: graph Variable state, RNG, aux,
gradients, and unary self-composition work; functional state and the cell
interface do not.

Run directly:
    python -m examples.comparisons.residual.residual_nnx
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

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


class Residual(nnx.Module):
    """x + f(x) for a member whose first argument and complete result are the signal."""

    def __init__(self, inner: nnx.Module):
        self.inner = inner

    def __call__(self, value, *args, **kwargs):
        return value + self.inner(value, *args, **kwargs)


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


class StatefulCell(nnx.SimpleCell):
    """An NNX cell using functional carry and graph-resident state together."""

    def __init__(self, rngs: nnx.Rngs):
        super().__init__(WIDTH, WIDTH, rngs=rngs)
        self.mean = nnx.Variable(jnp.zeros(WIDTH))

    def __call__(self, carry, value):
        carry, output = super().__call__(carry, value)
        self.mean[...] = DECAY * self.mean[...] + (1 - DECAY) * value
        return carry, output


def _construction_rngs() -> nnx.Rngs:
    return nnx.Rngs(params=PARAM_SEED)


def _apply_rngs(seed: int) -> nnx.Rngs:
    return nnx.Rngs(dropout=seed)


@nnx.capture(nnx.Intermediate)
def _captured_call(member, value, *, rngs):
    return member(value, rngs=rngs)


def _nests() -> bool:
    """Whether the unary wrapper accepts its own deterministic and stochastic products."""
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


def _forwards_functional_state() -> bool:
    """Whether the unary wrapper also understands NNX's explicit carry convention."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    carry = jnp.ones(WIDTH)
    cell = StatefulCell(_construction_rngs())
    expected_carry, expected_output = cell(carry, inputs)
    assert expected_output.shape == inputs.shape
    assert not bool(jnp.allclose(cell.mean[...], jnp.zeros(WIDTH)))

    try:
        next_carry, output = Residual(StatefulCell(_construction_rngs()))(carry, inputs)
    except Exception:
        return False
    return bool(
        jnp.allclose(next_carry, expected_carry)
        and jnp.allclose(output, inputs + expected_output)
    )


def _preserves_cell_interface() -> bool:
    """Whether an NNX API accepting a cell also accepts the wrapped cell."""
    inputs = jnp.stack([jnp.linspace(-1.0, 1.0, WIDTH)] * 2)
    cell = StatefulCell(_construction_rngs())
    carry, outputs = nnx.RNN(cell, return_carry=True)(inputs)
    assert carry.shape == (WIDTH,)
    assert outputs.shape == inputs.shape
    assert not bool(jnp.allclose(cell.mean[...], jnp.zeros(WIDTH)))

    try:
        nnx.RNN(
            Residual(StatefulCell(_construction_rngs())),
            return_carry=True,
        )(inputs)
    except Exception:
        return False
    return True


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
    result = verify('flax nnx', evidence())
    print(
        'flax nnx    '
        f'functional-state={"yes" if _forwards_functional_state() else "NO"} '
        f'cell-interface={"yes" if _preserves_cell_interface() else "NO"}',
        flush=True,
    )
    return result


if __name__ == '__main__':
    main()
