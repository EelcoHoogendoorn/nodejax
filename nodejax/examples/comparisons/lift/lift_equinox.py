"""An attempt at a reusable lifted layer stack in idiomatic Equinox.

The substrate is as different from Flax NNX as the JAX ecosystem offers. An
``eqx.Module`` IS a pytree of its parameters, so building the stack is
``eqx.filter_vmap`` over the constructor and running it is a plain
``lax.scan`` over the stacked leaves. Nothing is lifted and no graph is split.

State is where the difference shows. Equinox has no mutable variable, so a
layer that carries something returns a NEW SELF, and the scan stacks those
successors like any other output. That is uniform and it costs nothing at the
scan. What it costs is that parameters and state are leaves of one pytree,
told apart only by which field you name: this file reads
``stacked.weight`` and ``stacked.mean`` off the same value, and no filter can
recover which was which.

Two limits below are structural rather than a matter of spelling, and they
lock together. The mechanism that separates state from parameters is the one
that cannot be built per-layer, and the mechanism that can be built per-layer
is the one that cannot separate them.

``eqx.nn.State`` is the official place to keep mutable state, and it is
addressed by ``eqx.nn.StateIndex``, a Python object created when the
constructor runs. ``eqx.filter_vmap`` traces a constructor ONCE, so N layers
receive one index between them rather than one each. The state leaf still
comes out with a leading axis and looks right until it is scanned: inside the
scan the weights are sliced to a single row, while a read of the state returns
the whole stacked array, because the index addresses a model-global slot and
identity is not something a scan can index per iteration. The carry types then
disagree and the scan is rejected. Building the layers in a Python loop does
give one slot each, and that is unrolling, which is what a lifted stack exists
to avoid. ``_state_slots_when_stacked`` measures both counts.

So this file uses the other convention, returned successors, which is what the
protocol below is built around. The price is that a returned successor puts
state and parameters in one pytree as ordinary arrays. Nothing distinguishes
them but the field name, and a transform does not know the field names of a
layer someone else wrote. ``eqx.filter_grad`` differentiates the running
statistic along with the weights, so an optimizer handed that gradient updates
the statistic as if it were a parameter. ``_state_gets_gradient`` measures it.

The layer protocol is this file's own, exactly as the NNX column's is:
``make_layer`` is a closure over construction arguments taking one key;
``__call__`` returns ``(successor, output)`` so a stateless layer answers
``(self, value)``; it may declare ``key=``; and it uses this file's
``WithAux`` when it emits aux. Only the entropy fact is discovered rather
than declared, by inspecting the signature, and that is the one that decides
whether the transform composes with itself.

Run directly:
    python -m nodejax.examples.comparisons.lift.lift_equinox
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from nodejax.examples.comparisons.lift.lift_common import (
    CONTEXTS,
    DECAY,
    DEPTH,
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


class LayerAux(NamedTuple):
    mean: jax.Array
    energy: jax.Array


def emit(value, aux):
    """Mark auxiliary output without constraining its pytree type."""
    return WithAux(value, aux)


def _split_output(output):
    return output if type(output) is WithAux else (output, None)


class LayerStack(eqx.Module):
    """Stack layers satisfying this comparison's constructor/call protocol."""

    layers: eqx.Module
    depth: int = eqx.field(static=True)
    takes_key: bool = eqx.field(static=True)

    def __init__(self, make_layer: Callable, depth: int, key: jax.Array):
        if type(depth) is not int or depth < 1:
            raise TypeError('layer_stack depth must be a positive int')
        self.layers = eqx.filter_vmap(make_layer)(jax.random.split(key, depth))
        self.depth = depth
        self.takes_key = 'key' in inspect.signature(
            type(self.layers).__call__).parameters

    def __call__(self, value, *, key=None):
        if self.takes_key and key is None:
            raise TypeError('stochastic layer_stack requires key=')
        if not self.takes_key and key is not None:
            raise TypeError('deterministic layer_stack does not accept key=')

        rows, static = eqx.partition(self.layers, eqx.is_inexact_array)
        keys = (jax.random.split(key, self.depth) if self.takes_key
                else jnp.zeros(self.depth))

        def step(carry, item):
            row, row_key = item
            layer = eqx.combine(row, static)
            successor, output = (layer(carry, key=row_key) if self.takes_key
                                 else layer(carry))
            clean, aux = _split_output(output)
            advanced, _ = eqx.partition(successor, eqx.is_inexact_array)
            return clean, (advanced, aux)

        out, (advanced, aux) = jax.lax.scan(step, value, (rows, keys))
        return eqx.combine(advanced, static), out, aux


def layer_stack(make_layer: Callable, depth: int,
                key: jax.Array) -> LayerStack:
    """Construct a sequential transform over the documented layer protocol."""
    return LayerStack(make_layer, depth, key)


class NoisyStatefulLayer(eqx.Module):
    weight: jax.Array
    mean: jax.Array

    def __init__(self, key: jax.Array):
        self.weight = 0.25 + jax.random.normal(key, (WIDTH,)) / WIDTH
        self.mean = jnp.zeros(WIDTH)

    def __call__(self, value, *, key):
        mean = DECAY * self.mean + (1 - DECAY) * value
        keep = jax.random.bernoulli(key, 1 - DROP_RATE, shape=value.shape)
        output = jnp.tanh(self.weight * value + mean) * keep / (1 - DROP_RATE)
        successor = eqx.tree_at(lambda layer: layer.mean, self, mean)
        return successor, emit(
            output, LayerAux(mean, jnp.mean(output ** 2)))


class BareLayer(eqx.Module):
    """A deterministic layer proving that state and aux are not mandatory."""

    weight: jax.Array

    def __init__(self, key: jax.Array):
        self.weight = 0.9 + 0.01 * jax.random.normal(key, (WIDTH,))

    def __call__(self, value):
        return self, self.weight * value


def _raises(call) -> bool:
    try:
        call()
    except TypeError:
        return True
    return False


def _nests() -> bool:
    """Whether layer_stack accepts its own product as a layer."""
    try:
        tower = layer_stack(
            lambda key: layer_stack(BareLayer, 2, key),
            DEPTH, jax.random.PRNGKey(PARAM_SEED))
        tower(jnp.linspace(-1.0, 1.0, WIDTH))
    except Exception:
        return False
    return True


class NoisyLayer(eqx.Module):
    """Draws at apply, carries nothing."""

    weight: jax.Array

    def __init__(self, key: jax.Array):
        self.weight = 0.9 + 0.01 * jax.random.normal(key, (WIDTH,))

    def __call__(self, value, *, key):
        keep = jax.random.bernoulli(key, 1 - DROP_RATE, shape=value.shape)
        return self, self.weight * value * keep / (1 - DROP_RATE)


class StatefulLayer(eqx.Module):
    """Carries state, draws nothing."""

    weight: jax.Array
    mean: jax.Array

    def __init__(self, key: jax.Array):
        self.weight = 0.9 + 0.01 * jax.random.normal(key, (WIDTH,))
        self.mean = jnp.zeros(WIDTH)

    def __call__(self, value):
        mean = DECAY * self.mean + (1 - DECAY) * value
        successor = eqx.tree_at(lambda layer: layer.mean, self, mean)
        return successor, self.weight * value + mean


class OtherStateLayer(eqx.Module):
    """State of Equinox\'s other kind.

    Everywhere else in this file a stateful layer returns a new self, which
    is a convention this column invented. Equinox also ships a real
    mechanism: ``eqx.nn.StateIndex`` marks a slot and ``eqx.nn.State``
    carries the values, threaded through the call as a second argument and
    returned alongside the output. It is the mechanism ``eqx.nn.BatchNorm``
    uses.
    """

    weight: jax.Array
    index: eqx.nn.StateIndex

    def __init__(self, key: jax.Array):
        self.weight = 0.9 + 0.01 * jax.random.normal(key, (WIDTH,))
        self.index = eqx.nn.StateIndex(jnp.zeros(WIDTH))

    def __call__(self, value, state):
        running = DECAY * state.get(self.index) + (1 - DECAY) * value
        return state.set(self.index, running), self.weight * value + running


#: Each context and the module that presents it. Every module owns
#: parameters, since a layer axis is what ``layer_stack`` ranges over.
_CONTEXTS = (
    ('plain', BareLayer, False),
    ('rng', NoisyLayer, True),
    ('state', StatefulLayer, False),
    ('state+rng', NoisyStatefulLayer, True),
    ('other-state', OtherStateLayer, False),
)


def _contexts(param_key, apply_key) -> tuple[str, ...]:
    """Which member contexts ``layer_stack`` carries."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    carried = []
    for name, make_layer, draws in _CONTEXTS:
        try:
            stacked = layer_stack(make_layer, DEPTH, param_key)
            _, output, _ = (stacked(inputs, key=apply_key) if draws
                            else stacked(inputs))
        except Exception:
            continue
        if output.shape == (WIDTH,):
            carried.append(name)
    assert set(carried) <= set(CONTEXTS)
    return tuple(carried)


def _state_gets_gradient(param_key) -> bool:
    """Whether a gradient reaches the running statistic.

    It does. ``weight`` and ``mean`` are both inexact arrays in one pytree,
    and ``eqx.is_inexact_array`` is what selects what to differentiate, so
    nothing separates a parameter from a running statistic except the field
    name. An optimizer handed this gradient updates the statistic too.
    """
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    stacked = layer_stack(StatefulLayer, DEPTH, param_key)

    def loss(member):
        _, output, _ = member(inputs)
        return jnp.sum(output ** 2)

    gradient = eqx.filter_grad(loss)(stacked)
    return len(jax.tree.leaves(gradient)) > 1


def _state_slots_when_stacked() -> tuple[int, int]:
    """How many distinct state slots ``eqx.nn.State`` gives DEPTH layers,
    built separately and built under ``eqx.filter_vmap``.

    The answer is DEPTH and 1. ``eqx.nn.StateIndex`` is addressed by Python
    object identity, created when the constructor runs, and
    ``eqx.filter_vmap`` traces the constructor once. So the layers cannot
    receive one slot each, and inside a scan there is no way to reach the
    row belonging to this iteration: see the module docstring.
    """
    keys = jax.random.split(jax.random.PRNGKey(PARAM_SEED), DEPTH)
    built = [OtherStateLayer(key) for key in keys]
    separate = len({id(layer.index.marker) for layer in built})
    _, state = eqx.nn.make_with_state(
        lambda rows: eqx.filter_vmap(OtherStateLayer)(rows))(keys)
    return separate, len(jax.tree.leaves(state))


def evidence() -> Evidence:
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    param_key = jax.random.PRNGKey(PARAM_SEED)
    apply_key = jax.random.PRNGKey(DROPOUT_SEED)
    other_key = jax.random.PRNGKey(OTHER_DROPOUT_SEED)

    first = layer_stack(NoisyStatefulLayer, DEPTH, param_key)
    replay = layer_stack(NoisyStatefulLayer, DEPTH, param_key)
    other = layer_stack(NoisyStatefulLayer, DEPTH, param_key)

    before = first.layers.mean
    advanced, output, aux = first(inputs, key=apply_key)
    _, replay_output, replay_aux = replay(inputs, key=apply_key)
    _, other_output, _ = other(inputs, key=other_key)

    bare = layer_stack(BareLayer, DEPTH, param_key)
    _, bare_output, bare_aux = bare(inputs)

    return Evidence(
        parameter_shape=first.layers.weight.shape,
        state_shape=advanced.mean.shape,
        aux_state_shape=aux.mean.shape,
        aux_energy_shape=aux.energy.shape,
        same_parameters=bool(jnp.allclose(
            first.layers.weight, other.layers.weight)),
        replayed=bool(
            jnp.allclose(output, replay_output)
            and jnp.allclose(aux.mean, replay_aux.mean)),
        different_draw=not bool(jnp.allclose(output, other_output)),
        missing_rng_rejected=_raises(
            lambda: layer_stack(
                NoisyStatefulLayer, DEPTH, param_key)(inputs)),
        surplus_rng_rejected=_raises(lambda: bare(inputs, key=apply_key)),
        state_advanced=not bool(jnp.allclose(before, advanced.mean)),
        aux_matches_state=bool(jnp.allclose(aux.mean, advanced.mean)),
        bare_output_supported=(
            bare_aux is None and bare_output.shape == (WIDTH,)),
        nests=_nests(),
        contexts=_contexts(param_key, apply_key),
        state_gets_gradient=_state_gets_gradient(param_key),
    )


def main() -> Evidence:
    result = verify('equinox', evidence())
    separate, stacked = _state_slots_when_stacked()
    print(f'{"":12s} eqx.nn.State slots for {DEPTH} layers: '
          f'{separate} built separately, {stacked} built under filter_vmap',
          flush=True)
    return result


if __name__ == '__main__':
    main()
