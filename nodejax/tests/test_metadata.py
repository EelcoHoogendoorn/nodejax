"""The rigid IN metadata surface: the input bundles a caller must supply,
derived from the constructor signatures (not from a bound param).

param_input_spec is 'what must I supply to param_fn': a leaf reads its ctor
fields (required marked REQUIRED, optional carrying its default), rng kept as
a bundle field, ndef excluded (self-def inspection is encapsulated). A
composite is the member-keyed tree of its members'.
"""
import jax
import jax.numpy as jnp

from nodejax.struct import Struct
from nodejax import node_def, serial, parallel
from nodejax.core import REQUIRED
from nodejax.control import Gain


def Plant(dt=0.01, spring_k=1.0, damping_c=0.1):
    def init(ndef):
        return Struct(pos=jnp.zeros_like(ndef.input), vel=jnp.zeros_like(ndef.input))
    def apply(state, input):
        acc = input - spring_k * state.pos - damping_c * state.vel
        vel = state.vel + dt * acc
        pos = state.pos + dt * vel
        return Struct(pos=pos, vel=vel), pos
    return node_def(apply, init=init, name='plant')


def _pi():
    def param(kp, ki=0.0):
        return Struct(kp=jnp.asarray(kp), ki=jnp.asarray(ki))
    def apply(param, input):
        return param.kp * input
    return node_def(apply, param=param, name='pi')


def test_param_input_spec_marks_required_and_carries_defaults():
    g = Gain().param_input_spec
    assert list(g.__keys__) == ['scale'] and g.scale is REQUIRED

    p = _pi().param_input_spec
    assert list(p.__keys__) == ['kp', 'ki']
    assert p.kp is REQUIRED
    assert p.ki == 0.0                       # optional field carries its default


def test_param_input_spec_drops_ndef_keeps_rng():
    def dense():
        def param(rng, ndef):                # shape from ndef (encapsulated), draw from rng
            return Struct(w=jnp.zeros((ndef.input.shape[-1], 3)))
        def apply(param, input):
            return input @ param.w
        return node_def(apply, param=param, name='dense')
    s = dense().param_input_spec
    assert list(s.__keys__) == ['rng']       # ndef is not in the public contract
    assert s.rng is REQUIRED


def test_param_input_spec_composes_by_member():
    net = serial(a=Gain(), b=_pi())
    s = net.param_input_spec
    assert list(s.__keys__) == ['a', 'b']
    assert s.a.scale is REQUIRED
    assert s.b.kp is REQUIRED and s.b.ki == 0.0

    # nesting recurses
    outer = serial(x=serial(a=Gain(), b=_pi()), y=Gain())
    assert outer.param_input_spec.x.b.kp is REQUIRED
    assert outer.param_input_spec.y.scale is REQUIRED

    # parallel composes the same way
    par = parallel(a=Gain(), b=Gain())
    assert par.param_input_spec.a.scale is REQUIRED


def test_param_input_spec_nonparametric_is_empty():
    assert Plant(0.1, 1.0, 0.1).ndef.param_input_spec == ()


def test_state_input_spec_is_the_seed_bundle():
    from nodejax.control import Integrator, Walker
    # rng seed: init(param, rng)
    w = Walker().state_input_spec
    assert list(w.__keys__) == ['rng'] and w.rng is REQUIRED

    # plain init(param): nothing to seed
    assert list(Integrator().state_input_spec.__keys__) == []

    # explicit seed field carrying its default
    def seeded():
        def param(k):
            return Struct(k=jnp.asarray(k))
        def init(param, i0=0.0):
            return jnp.asarray(i0)
        def apply(param, state, input):
            return state + param.k * input, state
        return node_def(apply, param=param, init=init, name='seeded')
    s = seeded().state_input_spec
    assert list(s.__keys__) == ['i0'] and s.i0 == 0.0


def test_state_input_spec_non_cyclic_is_empty():
    assert Gain().state_input_spec == ()


def test_state_input_spec_composes_by_member_with_boundary_rng():
    from nodejax.control import Walker
    net = serial(w=Walker(), g=Gain())
    s = net.state_input_spec
    # rng is a BOUNDARY field: the caller passes one key, the composite splits
    # it toward members — so the member sub-bundles carry no rng of their own
    assert list(s.__keys__) == ['rng', 'w', 'g']
    assert s.rng is REQUIRED
    assert list(s.w.__keys__) == []     # walker's own rng filled from the split
    assert s.g == ()                    # non-cyclic member: no seed


def test_param_input_spec_hoists_rng_to_the_boundary():
    def noisy():
        def param(rng, width=3):
            return Struct(w=jax.random.normal(rng.next(), (width,)))
        def apply(param, input):
            return param.w * input
        return node_def(apply, param=param, name='noisy')

    net = serial(n=noisy(), g=Gain())
    s = net.param_input_spec
    assert list(s.__keys__) == ['rng', 'n', 'g']
    assert s.rng is REQUIRED
    assert list(s.n.__keys__) == ['width'] and s.n.width == 3   # rng stripped
    assert s.g.scale is REQUIRED

    # nested: the inner pipe's boundary rng is itself hoisted outward
    outer = serial(x=serial(n=noisy(), g=Gain()), y=Gain())
    o = outer.param_input_spec
    assert list(o.__keys__) == ['rng', 'x', 'y']
    assert 'rng' not in o.x             # filled by the outer split

    # a deterministic composite has NO rng field: a passed key is an
    # ordinary unknown-field rejection, not a special-cased guard
    det = serial(a=Gain(), b=Gain())
    assert 'rng' not in det.param_input_spec


def test_meta_is_the_complete_six_spec_surface():
    from nodejax import meta, spec
    # a shaped linear leaf: IN specs are rigid; OUT specs derive by eval_shape
    def lin():
        def param(weight, bias):
            return Struct(weight=jnp.asarray(weight), bias=jnp.asarray(bias))
        def apply(param, input):
            return input @ param.weight + param.bias
        return node_def(apply, param=param, name='lin', apply_input_spec=spec(4))
    node = lin().parameterize(weight=jnp.ones((4, 3)), bias=jnp.zeros(3))
    m = meta(node)
    # IN (what a caller supplies)
    assert m.param_input_spec.weight is REQUIRED and m.param_input_spec.bias is REQUIRED
    assert m.state_input_spec == ()
    assert m.apply_input_spec.shape == (4,)
    # OUT (what the node produces)
    assert m.param_spec.weight.shape == (4, 3)
    assert m.state_spec == ()
    assert m.output_spec.shape == (3,)


def test_given_slots_are_spec_defaults_and_survive_rebuild():
    """A pre-bound member's slot carries its stored param as the slot
    DEFAULT in the composite's bundle spec — the spec states what binding
    made optional — and structural rewrites carry the stored construction."""
    from nodejax import map_members

    pipe = Gain() >> Gain().parameterize(scale=jnp.asarray(3.0))
    spec = pipe.param_input_spec
    assert spec.gain.scale is REQUIRED               # open slot: still required
    assert jnp.allclose(spec.gain_2.scale, 3.0)      # bound slot: its param, as default

    node = pipe.parameterize(gain=Struct(scale=jnp.asarray(2.0)))
    assert node.apply(1.0) == 6.0                    # stored construction filled

    rebuilt = map_members(pipe, lambda d: d)         # identity rewrite
    node2 = rebuilt.parameterize(gain=Struct(scale=jnp.asarray(2.0)))
    assert node2.apply(1.0) == 6.0                   # given survived the rebuild
