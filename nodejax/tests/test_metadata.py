"""The rigid data-input metadata surface.

Input specs describe only the fields a caller supplies as data. RNG topology
is definition metadata exposed by the corresponding ``contract.*_plan``;
raw keys never masquerade as fields in these specs.
"""
import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import Node, Leaf, serial, parallel
from nodejax.binding import (REQUIRED)
from nodejax.control import Gain


def Plant(dt: float=0.01, spring_k: float=1.0, damping_c: float=0.1) -> Node:
    def init(node):
        return Struct(pos=jnp.zeros_like(node.input), vel=jnp.zeros_like(node.input))
    def apply(state, input):
        acc = input - spring_k * state.pos - damping_c * state.vel
        vel = state.vel + dt * acc
        pos = state.pos + dt * vel
        return Struct(pos=pos, vel=vel), pos
    return Leaf(apply, init=init, name='plant')


def _pi() -> Node:
    def param(kp, ki=0.0):
        return Struct(kp=jnp.asarray(kp), ki=jnp.asarray(ki))
    def apply(param, input):
        return param.kp * input
    return Leaf(apply, param=param, name='pi')


def test_param_input_spec_marks_required_and_carries_defaults():
    g = Gain().contract.param_input_spec
    assert list(g.__keys__) == ['scale'] and g.scale is REQUIRED

    p = _pi().contract.param_input_spec
    assert list(p.__keys__) == ['kp', 'ki']
    assert p.kp is REQUIRED
    assert p.ki == 0.0                       # optional field carries its default


def test_param_input_spec_drops_node_and_rng_channels():
    def dense():
        def param(node, rng):                # shape from node (encapsulated), draw from rng
            return Struct(w=jnp.zeros((node.input.shape[-1], 3)))
        def apply(param, input):
            return input @ param.w
        return Leaf(apply, param=param, name='dense')
    s = dense().contract.param_input_spec
    assert list(s.__keys__) == []
    assert dense().contract.param_takes_rng


def test_param_input_spec_composes_by_member():
    net = serial(a=Gain(), b=_pi())
    s = net.contract.param_input_spec
    assert list(s.__keys__) == ['a', 'b']
    assert s.a.scale is REQUIRED
    assert s.b.kp is REQUIRED and s.b.ki == 0.0

    # nesting recurses
    outer = serial(x=serial(a=Gain(), b=_pi()), y=Gain())
    assert outer.contract.param_input_spec.x.b.kp is REQUIRED
    assert outer.contract.param_input_spec.y.scale is REQUIRED

    # parallel composes the same way
    par = parallel(a=Gain(), b=Gain())
    assert par.contract.param_input_spec.a.scale is REQUIRED


def test_param_input_spec_nonparametric_is_absent():
    assert Plant(0.1, 1.0, 0.1).contract.param_input_spec is None


def test_state_input_spec_is_the_init_bundle():
    from nodejax.control import Integrator, Walker
    # RNG is a separate call channel, not a state-input field.
    walker = Walker()
    assert list(walker.contract.state_input_spec.__keys__) == []
    assert walker.contract.init_takes_rng

    # plain init(param): nothing to supply
    assert list(Integrator().contract.state_input_spec.__keys__) == []

    # explicit field carrying its default
    def declared():
        def param(k):
            return Struct(k=jnp.asarray(k))
        def init(param, i0=0.0):
            return jnp.asarray(i0)
        def apply(param, state, input):
            return state + param.k * input, state
        return Leaf(apply, param=param, init=init, name='declared')
    s = declared().contract.state_input_spec
    assert list(s.__keys__) == ['i0'] and s.i0 == 0.0


def test_state_input_spec_non_cyclic_is_absent():
    assert Gain().contract.state_input_spec is None


def test_state_input_spec_composes_separately_from_init_entropy():
    from nodejax.control import Walker
    net = serial(w=Walker(), g=Gain())
    s = net.contract.state_input_spec
    assert list(s.__keys__) == ['w', 'g']
    assert net.contract.init_takes_rng
    assert list(s.w.__keys__) == []
    assert type(s.g) is Struct and not s.g  # nested declaration has no fields


def test_param_entropy_composes_separately_from_input_specs():
    def noisy():
        def param(rng, width=3):
            return Struct(w=jax.random.normal(rng.next(), (width,)))
        def apply(param, input):
            return param.w * input
        return Leaf(apply, param=param, name='noisy')

    net = serial(n=noisy(), g=Gain())
    s = net.contract.param_input_spec
    assert list(s.__keys__) == ['n', 'g']
    assert net.contract.param_takes_rng
    assert list(s.n.__keys__) == ['width'] and s.n.width == 3
    assert s.g.scale is REQUIRED

    # Nested composition retains one outer RNG requirement without data fields.
    outer = serial(x=serial(n=noisy(), g=Gain()), y=Gain())
    o = outer.contract.param_input_spec
    assert list(o.__keys__) == ['x', 'y']
    assert outer.contract.param_takes_rng

    # A deterministic composite has an empty RNG plan.
    det = serial(a=Gain(), b=Gain())
    assert not det.contract.param_takes_rng


def test_meta_is_the_complete_six_spec_surface():
    from nodejax import meta, spec
    # a shaped linear leaf: IN specs are rigid; OUT specs derive by eval_shape
    def lin():
        def param(weight, bias):
            return Struct(weight=jnp.asarray(weight), bias=jnp.asarray(bias))
        def apply(param, input):
            return input @ param.weight + param.bias
        return Leaf(apply, param=param, name='lin', apply_input_spec=spec(4))
    node = lin().parameterize(weight=jnp.ones((4, 3)), bias=jnp.zeros(3))
    m = meta(node)
    # IN (what a caller supplies)
    assert m.param_input_spec.weight is REQUIRED and m.param_input_spec.bias is REQUIRED
    assert m.state_input_spec is None
    assert m.apply_input_spec.input.shape == (4,)
    # OUT (what the node produces)
    assert m.param_spec.weight.shape == (4, 3)
    assert m.state_spec == ()
    assert m.output_spec.shape == (3,)


def test_tree_binding_discards_captured_member_parameters():
    """Captures belong after tree binding and disappear when it is rerun."""
    from nodejax import map_members

    pipe = Gain() >> Gain().parameterize(scale=jnp.asarray(3.0))
    spec = pipe.contract.param_input_spec
    assert spec.gain.scale is REQUIRED               # open slot: still required
    assert jnp.allclose(spec.gain_2.scale, 3.0)      # bound slot: its param, as default

    node = pipe.parameterize(gain=Struct(scale=jnp.asarray(2.0)))
    assert node.apply(1.0) == 6.0                    # stored construction filled

    rebuilt = map_members(pipe, lambda d: d)
    rebuilt_spec = rebuilt.contract.param_input_spec
    assert rebuilt_spec.gain.scale is REQUIRED
    assert rebuilt_spec.gain_2.scale is REQUIRED
    node2 = rebuilt.parameterize(
        gain=Struct(scale=jnp.asarray(2.0)),
        gain_2=Struct(scale=jnp.asarray(5.0)),
    )
    assert node2.apply(1.0) == 10.0


def test_static_replay_discards_captured_member_parameters():
    pipe = Gain() >> Gain().parameterize(scale=jnp.asarray(3.0))

    replayed = pipe.specialize()

    assert replayed.contract.param_input_spec.gain.scale is REQUIRED
    assert replayed.contract.param_input_spec.gain_2.scale is REQUIRED
