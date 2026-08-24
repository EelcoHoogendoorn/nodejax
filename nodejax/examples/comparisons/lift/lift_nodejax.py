"""NodeJAX's native stack under the lifted-stack parity contract.

The layer owns parameters, mutable state, apply-time randomness, and aux.
``stack`` constructs one independent row per layer and scans those rows while
feeding each clean output to the next layer.

Run directly:
    python -m nodejax.examples.comparisons.lift.lift_nodejax
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nodejax import Aux, Leaf, node, split_aux, stack
from nodejax.struct import Struct
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


@node
def NoisyStatefulLayer():
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
        return Struct(mean=mean), (
            output, Aux(mean=mean, energy=jnp.mean(output ** 2)))

    return Leaf(apply, param=param, init=init, name='noisy_stateful')


@node
def BareLayer():
    def param(rng):
        return Struct(
            weight=0.9 + 0.01 * jax.random.normal(rng.next(), (WIDTH,)))

    def apply(param, input):
        return param.weight * input

    return Leaf(apply, param=param, name='bare')


def _build(layer, key):
    return stack(layer, DEPTH).parameterize(rng=key).initialize()


def _raises(call) -> bool:
    try:
        call()
    except TypeError:
        return True
    return False


def _nests(key, apply_key) -> bool:
    """A stack of stacks: the product of a transform is a layer like any
    other, so the family composes with itself."""
    try:
        tower = _build(stack(NoisyStatefulLayer(), 2), key)
        tower.apply(input=jnp.linspace(-1.0, 1.0, WIDTH), rng=apply_key)
    except Exception:
        return False
    return True


@node
def NoisyLayer():
    """Draws at apply, carries nothing."""
    def param(rng):
        return Struct(
            weight=0.9 + 0.01 * jax.random.normal(rng.next(), (WIDTH,)))

    def apply(param, input, rng):
        keep = jax.random.bernoulli(
            rng.next(), 1 - DROP_RATE, shape=input.shape)
        return param.weight * input * keep / (1 - DROP_RATE)

    return Leaf(apply, param=param, name='noisy')


@node
def StatefulLayer():
    """Carries state, draws nothing."""
    def param(rng):
        return Struct(
            weight=0.9 + 0.01 * jax.random.normal(rng.next(), (WIDTH,)))

    def init():
        return Struct(mean=jnp.zeros(WIDTH))

    def apply(param, state, input):
        mean = DECAY * state.mean + (1 - DECAY) * input
        return Struct(mean=mean), param.weight * input + mean

    return Leaf(apply, param=param, init=init, name='stateful')


@node
def OtherStateLayer():
    """State of the library\'s other kind.

    NodeJAX has one state role, so the second kind is a tag on the same
    role rather than a second type. ``single_batch_state`` marks state that
    must not gain an axis when the node is mapped, which is what running
    statistics need and what ``nn.BatchNorm`` uses.
    """
    def param(rng):
        return Struct(
            weight=0.9 + 0.01 * jax.random.normal(rng.next(), (WIDTH,)))

    def init():
        return Struct(running=jnp.zeros(WIDTH))

    def apply(param, state, input):
        running = DECAY * state.running + (1 - DECAY) * input
        return Struct(running=running), param.weight * input + running

    return Leaf(apply, param=param, init=init, name='other_state',
                tags={'single_batch_state'})


#: Each context, the layer that presents it, and what that layer needs at
#: apply. Every layer owns parameters, since a layer axis is what ``stack``
#: ranges over.
_CONTEXTS = (
    ('plain', BareLayer, False, False),
    ('rng', NoisyLayer, False, True),
    ('state', StatefulLayer, True, False),
    ('state+rng', NoisyStatefulLayer, True, True),
    ('other-state', OtherStateLayer, True, False),
)


def _contexts(param_key, apply_key) -> tuple[str, ...]:
    """Which member contexts ``stack`` carries."""
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    carried = []
    for name, build, stateful, draws in _CONTEXTS:
        try:
            stacked = stack(build(), DEPTH).parameterize(rng=param_key)
            if stateful:
                stacked = stacked.initialize()
            result = (stacked.apply(input=inputs, rng=apply_key) if draws
                      else stacked.apply(inputs))
            output, _ = split_aux(result[1] if stateful else result)
        except Exception:
            continue
        if output.shape == (WIDTH,):
            carried.append(name)
    assert set(carried) <= set(CONTEXTS)
    return tuple(carried)


def _state_gets_gradient(param_key) -> bool:
    """Whether a gradient reaches the running statistic.

    It cannot. ``param`` and ``state`` are separate trees, so a gradient
    taken against the parameters has nowhere to put one for state.
    """
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    stacked = _build(StatefulLayer(), param_key)

    def loss(param):
        _, output = stacked.bind(param).apply(inputs)
        return jnp.sum(output ** 2)

    gradient = jax.grad(loss)(stacked.param)
    return len(jax.tree.leaves(gradient)) > len(
        jax.tree.leaves(stacked.param))


def evidence() -> Evidence:
    inputs = jnp.linspace(-1.0, 1.0, WIDTH)
    param_key = jax.random.PRNGKey(PARAM_SEED)
    apply_key = jax.random.PRNGKey(DROPOUT_SEED)
    other_key = jax.random.PRNGKey(OTHER_DROPOUT_SEED)
    first = _build(NoisyStatefulLayer(), param_key)
    replay = _build(NoisyStatefulLayer(), param_key)
    other = _build(NoisyStatefulLayer(), param_key)

    before = first.state.mean
    state, result = first.apply(input=inputs, rng=apply_key)
    output, aux = split_aux(result)
    replay_state, replay_result = replay.apply(input=inputs, rng=apply_key)
    replay_output, replay_aux = split_aux(replay_result)
    _, other_result = other.apply(input=inputs, rng=other_key)
    other_output, _ = split_aux(other_result)

    bare = stack(BareLayer(), DEPTH).parameterize(rng=param_key)
    bare_output = bare.apply(inputs)

    return Evidence(
        parameter_shape=first.param.weight.shape,
        state_shape=state.state.mean.shape,
        aux_state_shape=aux.mean.shape,
        aux_energy_shape=aux.energy.shape,
        same_parameters=bool(jnp.allclose(
            first.param.weight, other.param.weight)),
        replayed=bool(
            jnp.allclose(output, replay_output)
            and jnp.allclose(state.state.mean, replay_state.state.mean)
            and jnp.allclose(aux.mean, replay_aux.mean)),
        different_draw=not bool(jnp.allclose(output, other_output)),
        missing_rng_rejected=_raises(
            lambda: first.apply(input=inputs)),
        surplus_rng_rejected=_raises(
            lambda: bare.apply(inputs, rng=apply_key)),
        state_advanced=not bool(jnp.allclose(before, state.state.mean)),
        aux_matches_state=bool(jnp.allclose(aux.mean, state.state.mean)),
        bare_output_supported=bare_output.shape == (WIDTH,),
        nests=_nests(param_key, apply_key),
        contexts=_contexts(param_key, apply_key),
        state_gets_gradient=_state_gets_gradient(param_key),
    )


def main() -> Evidence:
    return verify('nodejax', evidence())


if __name__ == '__main__':
    main()
