"""RNG at the public and canonical contract boundaries.

Users supply raw keys to all three public operations. The public boundary
turns that key into an invocation-local stream; canonical implementations see
the stream as a separate channel and their ordinary data
bundle never contains the reserved key.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import Leaf, Node, Struct, serial, train_step
from nodejax.core.binding import REQUIRED
from nodejax.control import Gain
from nodejax.core.contract import (
    ApplyCall, CallForm, ContractCalls, InitCall, ParamCall,
)
from nodejax.core.definition import Def
from nodejax.core.rng import KeyStream, MaybeKeyStream


KEYS = tuple(jax.random.split(jax.random.PRNGKey(0), 3))


def _compiled_spy(*, stochastic: bool = True):
    """A complete three-role definition exposing each canonical call."""
    seen = {}
    def param_impl(node, param_input, rng):
        seen['param'] = rng, param_input
        return Struct(scale=jnp.asarray(param_input.scale))

    def init_impl(node, param, state_input, rng):
        seen['init'] = rng, state_input
        return jnp.asarray(state_input.offset)

    def apply_impl(node, param, state, input, rng):
        seen['apply'] = rng, input
        advanced = state + param.scale * input.input
        return advanced, advanced

    node = Node(Def(
        name='compiled_spy',
        calls=ContractCalls(
            param=ParamCall(
                impl=param_impl,
                form=CallForm.from_values(Struct(scale=REQUIRED)),
                takes_rng=stochastic,
            ),
            init=InitCall(
                impl=init_impl,
                form=CallForm.from_values(Struct(offset=0.0)),
                takes_rng=stochastic,
            ),
            apply=ApplyCall(
                impl=apply_impl,
                form=CallForm.from_values(Struct(input=REQUIRED)),
                input_spec=Struct(input=REQUIRED),
                takes_rng=stochastic,
            ),
        ),
    ))
    return node, seen


def test_raw_keys_bind_uniformly_and_canonical_data_is_rng_free():
    node, seen = _compiled_spy()

    model = node.parameterize(scale=2.0, rng=KEYS[0])
    state = model.init(offset=3.0, rng=KEYS[1])
    next_state, output = model.apply(
        state, input=jnp.asarray(4.0), rng=KEYS[2])

    assert jnp.allclose(next_state, 11.0)
    assert jnp.allclose(output, 11.0)
    assert seen['param'][1].__keys__ == ('scale',)
    assert seen['init'][1].__keys__ == ('offset',)
    assert seen['apply'][1].__keys__ == ('input',)
    assert all(type(frame) is MaybeKeyStream for frame, _ in seen.values())
    assert all('rng' not in data for _, data in seen.values())


def test_authored_functions_receive_keystreams_from_all_three_frames():
    def stochastic():
        def param(rng):
            assert type(rng) is KeyStream
            return Struct(draw=jax.random.normal(rng.next(), ()))

        def init(rng):
            assert type(rng) is KeyStream
            return Struct(draw=jax.random.normal(rng.next(), ()))

        def apply(param, state, input, rng):
            assert type(rng) is KeyStream
            draw = jax.random.normal(rng.next(), ())
            return state, param.draw + state.draw + input + draw

        return Leaf(apply, param=param, init=init, name='stochastic')

    first = stochastic().parameterize(rng=KEYS[0])
    repeated = stochastic().parameterize(rng=KEYS[0])
    other = stochastic().parameterize(rng=KEYS[1])
    assert jnp.array_equal(first.param.draw, repeated.param.draw)
    assert not jnp.array_equal(first.param.draw, other.param.draw)

    first_state = first.init(rng=KEYS[1])
    repeated_state = first.init(rng=KEYS[1])
    other_state = first.init(rng=KEYS[2])
    assert jnp.array_equal(first_state.draw, repeated_state.draw)
    assert not jnp.array_equal(first_state.draw, other_state.draw)

    _, first_output = first.apply(first_state, 1.0, rng=KEYS[2])
    _, repeated_output = first.apply(first_state, 1.0, rng=KEYS[2])
    _, other_output = first.apply(first_state, 1.0, rng=KEYS[0])
    assert jnp.array_equal(first_output, repeated_output)
    assert not jnp.array_equal(first_output, other_output)


@pytest.mark.parametrize('stochastic', [True, False], ids=['missing', 'surplus'])
def test_definition_decides_rng_acceptance_before_canonical_execution(
        stochastic):
    node, seen = _compiled_spy(stochastic=stochastic)
    supplied = {} if stochastic else {'rng': KEYS[0]}
    message = 'requires' if stochastic else 'does not accept'

    with pytest.raises(TypeError, match=message):
        node.parameterize(scale=2.0, **supplied)
    assert seen == {}

    model = node.bind(Struct(scale=jnp.asarray(2.0)))
    with pytest.raises(TypeError, match=message):
        model.init(offset=0.0, **supplied)
    assert seen == {}

    with pytest.raises(TypeError, match=message):
        model.apply(jnp.asarray(0.0), input=1.0, **supplied)
    assert seen == {}


def test_transform_authors_pass_streams_to_the_contract_explicitly():
    node, seen = _compiled_spy()
    with pytest.raises(TypeError, match='expects a MaybeKeyStream'):
        node.contract.param(Struct(scale=2.0), KEYS[0])
    with pytest.raises(TypeError, match='expects a keyed MaybeKeyStream'):
        node.contract.param(Struct(scale=2.0), MaybeKeyStream())
    assert seen == {}

    param_frame = MaybeKeyStream(KEYS[0])
    init_frame = MaybeKeyStream(KEYS[1])
    apply_frame = MaybeKeyStream(KEYS[2])

    param = node.contract.param(Struct(scale=2.0), param_frame)
    state = node.contract.init(param, Struct(offset=3.0), init_frame)
    next_state, output = node.contract.apply(param, state, Struct(input=jnp.asarray(4.0)), apply_frame)

    assert jnp.allclose(next_state, 11.0)
    assert jnp.allclose(output, 11.0)
    assert seen['param'][0] is param_frame
    assert seen['init'][0] is init_frame
    assert seen['apply'][0] is apply_frame
    assert all('rng' not in data for _, data in seen.values())


def test_finished_parameter_capture_is_deterministic_not_optional():
    def random_scale():
        def param(rng):
            return Struct(scale=jax.random.uniform(rng.next(), ()))

        def apply(param, input):
            return param.scale * input

        return Leaf(apply, param=param, name='random_scale')

    stored = random_scale().parameterize(rng=KEYS[0])
    pipe = serial(random=stored, gain=Gain())

    from_store = pipe.parameterize(gain=Struct(scale=2.0))
    replaced = pipe.parameterize(
        random=Struct(scale=jnp.asarray(5.0)),
        gain=Struct(scale=2.0),
    )

    assert not pipe.contract.param_takes_rng
    assert jnp.allclose(from_store.apply(1.0), 2.0 * stored.param.scale)
    assert jnp.allclose(replaced.apply(1.0), 10.0)
    with pytest.raises(TypeError, match='does not accept rng'):
        pipe.parameterize(gain=Struct(scale=2.0), rng=KEYS[1])


def test_trainer_rng_has_one_required_boundary_source():
    def param(scale=1.0):
        return Struct(scale=jnp.asarray(scale))

    def apply(param, input, rng):
        return param.scale * input + jax.random.normal(rng.next(), ())

    model = Leaf(
        apply, param=param, name='drawing_model').parameterize()
    trainer = train_step(
        model,
        lambda prediction, target: (prediction - target) ** 2,
        optax.sgd(0.01),
    ).initialize()

    with pytest.raises(TypeError, match='apply requires rng'):
        trainer.apply(input=1.0, target=0.0)
    with pytest.raises(TypeError, match='apply requires rng'):
        trainer.apply(
            input=Struct(input=1.0, rng=KEYS[0]),
            target=0.0,
        )

    successor, output = trainer.apply(
        input=1.0, target=0.0, rng=KEYS[0])
    assert successor._def is trainer._def
    assert output is not None


def test_stored_rng_state_keeps_a_static_required_init_channel():
    def streamy():
        def init(rng):
            return Struct(rng=rng)

        def apply(state, input):
            return state, input

        return Leaf(apply, init=init, name='streamy')

    original = KEYS[0]
    stored = streamy().bind(state=Struct(rng=original))
    model = serial(stream=stored, gain=Gain()).parameterize(
        gain=Struct(scale=1.0))

    assert model.contract.init_takes_rng
    with pytest.raises(TypeError, match='requires'):
        model.init()

    initialized = model.init(rng=KEYS[1])
    assert not jnp.array_equal(initialized.stream.rng, original)

    replacement = Struct(rng=KEYS[2])
    initialized = model.init(stream=replacement, rng=KEYS[1])
    assert not jnp.array_equal(initialized.stream.rng, replacement.rng)
    with pytest.raises(TypeError, match='replacement state contains no rng'):
        model.init(stream=Struct(), rng=KEYS[1])
