"""An attempt at a reusable lifted layer stack in idiomatic Flax NNX.

This is deliberately more than ``vmap`` over parameters. ``layer_stack``
attempts to reach true feature parity with nodejax stack. It
constructs independent layers, stores their complete NNX graphs on a leading
axis, and scans that axis when called. Parameters and mutable Variables move
together. A layer returns its clean value and may publish auxiliary data with
NNX's native ``Module.sow``. A caller that wants those values wraps the stack
with ``nnx.capture``; capture crosses the scan and stacks the sown leaves over
depth without putting them in the sequential carry.

It is reusable over a convention, not over every NNX module: ``make_layer`` is
a closure over construction arguments and accepts one ``Rngs`` value; the
resulting layer consumes one value and may request ``rngs=`` at apply. Aux
needs no local return convention: it uses ``nnx.Intermediate`` and the
framework's capture mechanism. NodeJAX's standard contract supplies the
remaining construction and entropy facts without a per-transform convention;
that difference is part of the comparison.

The convention is wide enough for stock modules: ``layer_stack`` accepts
``nnx.Linear`` unchanged, because a simple module satisfies all four
requirements by construction. It is NOT wide enough for its own product.
``layer_stack`` of a ``layer_stack`` raises, and the cause is the signature
inspection below: ``LayerStack.__call__`` must declare ``rngs=`` to serve both
the stochastic and deterministic cases from one Python signature, so the outer
transform reads it as stochastic and demands a key that nothing downstream
consumes. Recovering a member's contract by inspecting it means a transform
cannot present that contract when it is itself the member, so the family does
not nest. That is the criterion ``nests`` reports.

The comparison omits input-value priming. NNX modules are constructed eagerly,
so a layer that needs its input shape receives it as an ordinary constructor
argument.

Run directly:
    python -m nodejax.examples.comparisons.lift.lift_nnx
"""

from __future__ import annotations

from collections.abc import Callable
import inspect

import jax
import jax.numpy as jnp
from flax import nnx

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


_MISSING = object()


class LayerStack(nnx.Module):
    """Stack layers satisfying this comparison's factory/call protocol."""

    def __init__(self, make_layer: Callable[[nnx.Rngs], nnx.Module],
                 depth: int, rngs: nnx.Rngs):
        if type(depth) is not int or depth < 1:
            raise TypeError('layer_stack depth must be a positive int')

        @nnx.split_rngs(splits=depth)
        @nnx.vmap(in_axes=0, out_axes=0)
        def build(rows):
            return make_layer(rows)

        self.layers = build(rngs)
        self.depth = depth
        self.takes_rng = 'rngs' in inspect.signature(
            self.layers.__call__).parameters

    def __call__(self, value, *, rngs=_MISSING):
        # ``sow`` must run under capture: otherwise it adds an Intermediate
        # Variable while ``scan`` is tracing and changes the graph structure.
        # Preserve an enclosing capture so it can collect the leaf emissions;
        # install and discard a local one only for an ordinary clean call.
        def captured(stack, input, call_rngs):
            return stack._call(input, rngs=call_rngs)

        output, _ = nnx.capture(captured, nnx.Intermediate)(
            self, value, rngs)
        return output

    def _call(self, value, *, rngs):
        if self.takes_rng:
            if rngs is _MISSING:
                raise TypeError('stochastic layer_stack requires rngs=')
            rows = rngs.split(self.depth)

            @nnx.scan(in_axes=(0, 0, nnx.Carry),
                      out_axes=(nnx.Carry, 0))
            def run(layer, row_rngs, carry):
                clean = layer(carry, rngs=row_rngs)
                return clean, None

            return run(self.layers, rows, value)[0]

        if rngs is not _MISSING:
            raise TypeError('deterministic layer_stack does not accept rngs=')

        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=(nnx.Carry, 0))
        def run(layer, carry):
            clean = layer(carry)
            return clean, None

        return run(self.layers, value)[0]


def layer_stack(make_layer: Callable[[nnx.Rngs], nnx.Module],
                depth: int, rngs: nnx.Rngs) -> LayerStack:
    """Construct a sequential transform over the documented layer protocol."""
    return LayerStack(make_layer, depth, rngs)


class NoisyStatefulLayer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            0.25 + jax.random.normal(rngs.params(), (WIDTH,)) / WIDTH)
        self.mean = nnx.Variable(jnp.zeros(WIDTH))
        self.dropout = nnx.Dropout(DROP_RATE, deterministic=False)

    def __call__(self, value, *, rngs):
        mean = DECAY * self.mean[...] + (1 - DECAY) * value
        self.mean[...] = mean
        output = self.dropout(
            jnp.tanh(self.weight[...] * value + mean), rngs=rngs)
        self.sow(
            nnx.Intermediate, 'mean', mean,
            init_fn=lambda: None,
            reduce_fn=lambda _, current: current,
        )
        self.sow(
            nnx.Intermediate, 'energy', jnp.mean(output ** 2),
            init_fn=lambda: None,
            reduce_fn=lambda _, current: current,
        )
        return output


class BareLayer(nnx.Module):
    """A deterministic layer proving that aux is not mandatory."""

    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            0.9 + 0.01 * jax.random.normal(rngs.params(), (WIDTH,)))

    def __call__(self, value):
        return self.weight[...] * value


def _construction_rngs() -> nnx.Rngs:
    return nnx.Rngs(params=PARAM_SEED)


def _apply_rngs(dropout_seed: int) -> nnx.Rngs:
    return nnx.Rngs(dropout=dropout_seed)


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
            lambda rows: layer_stack(BareLayer, 2, rows),
            DEPTH, _construction_rngs())
        tower(jnp.linspace(-1.0, 1.0, WIDTH))
    except Exception:
        return False
    return True


class NoisyLayer(nnx.Module):
    """Draws at apply, carries nothing."""

    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            0.9 + 0.01 * jax.random.normal(rngs.params(), (WIDTH,)))

    def __call__(self, value, *, rngs):
        keep = jax.random.bernoulli(
            rngs.dropout(), 1 - DROP_RATE, shape=value.shape)
        return self.weight[...] * value * keep / (1 - DROP_RATE)


class StatefulLayer(nnx.Module):
    """Carries state, draws nothing."""

    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            0.9 + 0.01 * jax.random.normal(rngs.params(), (WIDTH,)))
        self.mean = nnx.Variable(jnp.zeros(WIDTH))

    def __call__(self, value):
        mean = DECAY * self.mean[...] + (1 - DECAY) * value
        self.mean[...] = mean
        return self.weight[...] * value + mean


class OtherStateLayer(nnx.Module):
    """State of NNX\'s other kind.

    ``nnx.BatchStat`` rather than a plain ``nnx.Variable``. The distinction
    is a Variable subclass, which is what ``nnx.StateAxes`` keys on when a
    lifted transform gives one collection a different axis from another.
    """

    def __init__(self, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            0.9 + 0.01 * jax.random.normal(rngs.params(), (WIDTH,)))
        self.running = nnx.BatchStat(jnp.zeros(WIDTH))

    def __call__(self, value):
        running = DECAY * self.running[...] + (1 - DECAY) * value
        self.running[...] = running
        return self.weight[...] * value + running


#: Each context and the module that presents it. Every module owns
#: parameters, since a layer axis is what ``layer_stack`` ranges over.
_CONTEXTS = (
    ('plain', BareLayer, False),
    ('rng', NoisyLayer, True),
    ('state', StatefulLayer, False),
    ('state+rng', NoisyStatefulLayer, True),
    ('other-state', OtherStateLayer, False),
)


def _contexts() -> tuple[str, ...]:
    """Which member contexts ``layer_stack`` carries."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    carried = []
    for name, make_layer, draws in _CONTEXTS:
        try:
            stacked = layer_stack(make_layer, DEPTH, _construction_rngs())
            output = (stacked(inputs, rngs=_apply_rngs(DROPOUT_SEED))
                      if draws else stacked(inputs))
        except Exception:
            continue
        if output.shape == (WIDTH,):
            carried.append(name)
    assert set(carried) <= set(CONTEXTS)
    return tuple(carried)


def _state_gets_gradient() -> bool:
    """Whether a gradient reaches the running statistic.

    It does not. ``nnx.grad`` differentiates ``nnx.Param`` by default, and
    the running statistic is a plain ``nnx.Variable``, so the Variable
    subclass is what keeps the two apart.
    """
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    stacked = layer_stack(StatefulLayer, DEPTH, _construction_rngs())

    def loss(member):
        output = member(inputs)
        return jnp.sum(output ** 2)

    gradient = nnx.grad(loss)(stacked)
    return len(jax.tree.leaves(gradient)) > 1


def evidence() -> Evidence:
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    first = layer_stack(
        NoisyStatefulLayer, DEPTH, _construction_rngs())
    replay = layer_stack(
        NoisyStatefulLayer, DEPTH, _construction_rngs())
    other = layer_stack(
        NoisyStatefulLayer, DEPTH, _construction_rngs())

    before = first.layers.mean[...]
    def captured_call(stack, input, rngs):
        return stack._call(input, rngs=rngs)

    capture = nnx.capture(captured_call, nnx.Intermediate)
    output, aux = capture(first, inputs, _apply_rngs(DROPOUT_SEED))
    replay_output, replay_aux = capture(
        replay, inputs, _apply_rngs(DROPOUT_SEED))
    other_output, _ = capture(
        other, inputs, _apply_rngs(OTHER_DROPOUT_SEED))
    after = first.layers.mean[...]
    aux = nnx.to_pure_dict(aux)['layers']
    replay_aux = nnx.to_pure_dict(replay_aux)['layers']

    bare = layer_stack(
        BareLayer, DEPTH, nnx.Rngs(params=PARAM_SEED))
    def captured_bare_call(stack, input):
        return stack._call(input, rngs=_MISSING)

    bare_output, bare_aux = nnx.capture(
        captured_bare_call, nnx.Intermediate)(bare, inputs)

    return Evidence(
        parameter_shape=first.layers.weight[...].shape,
        state_shape=after.shape,
        aux_state_shape=aux['mean'].shape,
        aux_energy_shape=aux['energy'].shape,
        same_parameters=bool(jnp.allclose(
            first.layers.weight[...], other.layers.weight[...])),
        replayed=bool(
            jnp.allclose(output, replay_output)
            and jnp.allclose(aux['mean'], replay_aux['mean'])),
        different_draw=not bool(jnp.allclose(output, other_output)),
        missing_rng_rejected=_raises(lambda: layer_stack(
            NoisyStatefulLayer, DEPTH, _construction_rngs())(inputs)),
        surplus_rng_rejected=_raises(
            lambda: bare(inputs, rngs=_apply_rngs(DROPOUT_SEED))),
        state_advanced=not bool(jnp.allclose(before, after)),
        aux_matches_state=bool(jnp.allclose(aux['mean'], after)),
        bare_output_supported=(
            not jax.tree.leaves(bare_aux)
            and bare_output.shape == (WIDTH,)),
        nests=_nests(),
        contexts=_contexts(),
        state_gets_gradient=_state_gets_gradient(),
    )


def main() -> Evidence:
    return verify('flax nnx', evidence())


if __name__ == '__main__':
    main()
