"""Verifies compiled `Contract` operations, bundle validation, PRNG streams, and call sugar."""
import jax
import jax.numpy as jnp
import pytest

from nodejax.struct import Struct
from nodejax import Node, scan, Leaf, serial, nn
from nodejax.control import Gain, Integrator, Walker
from nodejax.core.rng import MaybeKeyStream


def _stream(key=None):
    return MaybeKeyStream() if key is None else MaybeKeyStream(key)


def _param(node, bundle=Struct(), key=None):
    return node.contract.param(bundle, _stream(key))


def _init(node, param, bundle=Struct(), key=None, **kwargs):
    rng = _stream(key)
    if 'input' in kwargs:
        return node.contract.prime(param, bundle, kwargs['input'], rng)
    return node.contract.init(param, bundle, rng)


def _apply(node, param, state, input, key=None):
    return node.contract.apply(
        param, state, input, _stream(key))


def Plant(dt: float=0.01, spring_k: float=1.0, damping_c: float=0.1) -> Node:
    def init(node):
        return Struct(pos=jnp.zeros_like(node.input), vel=jnp.zeros_like(node.input))
    def apply(state, input):
        acc = input - spring_k * state.pos - damping_c * state.vel
        vel = state.vel + dt * acc
        pos = state.pos + dt * vel
        return Struct(pos=pos, vel=vel), pos
    return Leaf(apply, init=init, name='plant')


def PI() -> Node:
    def param(kp, ki=0.0):
        return Struct(kp=jnp.asarray(kp), ki=jnp.asarray(ki))
    def apply(param, input):
        return param.kp * input
    return Leaf(apply, param=param, name='pi')


def Noisy() -> Node:
    def param(rng, width=3):
        return Struct(w=jax.random.normal(rng.next(), (width,)))
    def apply(param, input):
        return param.w * input
    return Leaf(apply, param=param, name='noisy')


def Derivative(dt: float) -> Node:
    def init(input):
        return jnp.asarray(input)

    def apply(state, input):
        return input, (input - state) / dt

    return Leaf(apply, init=init)


def test_contract_param_leaf():
    p = _param(Gain(), Struct(scale=2.0))
    assert jnp.allclose(p.scale, 2.0)
    assert Gain().bind(p).apply(3.0) == 6.0

    q = _param(PI(), Struct(kp=1.0))
    assert jnp.allclose(q.ki, 0.0)


def test_parameterize_validates_the_bundle():
    with pytest.raises(TypeError, match='unknown bundle fields'):
        Gain().parameterize(Struct(scale=2.0, gain=1.0))
    with pytest.raises(TypeError, match='missing required'):
        Gain().parameterize(Struct())
    with pytest.raises(TypeError, match='parameterize requires rng'):
        Noisy().parameterize(Struct(width=4))
    with pytest.raises(TypeError, match='not parametric'):
        Plant(0.1, 1.0, 0.1).node.parameterize(Struct(x=1.0))
    assert _param(Plant(0.1, 1.0, 0.1).node) == ()


def test_contract_param_rng_is_a_separate_frame():
    key = jax.random.PRNGKey(0)
    p1 = _param(Noisy(), key=key)
    p2 = _param(Noisy(), key=key)
    p3 = _param(Noisy(), key=jax.random.PRNGKey(1))
    assert jnp.allclose(p1.w, p2.w)
    assert not jnp.allclose(p1.w, p3.w)
    assert p1.w.shape == (3,)


def test_contract_param_composite_boundary_rng():
    net = serial(n=Noisy(), g=Gain())
    p = _param(
        net, Struct(g=Struct(scale=2.0)), key=jax.random.PRNGKey(0))
    assert p.n.w.shape == (3,)
    assert jnp.allclose(p.g.scale, 2.0)
    with pytest.raises(TypeError, match='parameterize requires rng'):
        net.parameterize(Struct(g=Struct(scale=2.0)))

    det = serial(a=Gain(), b=PI())
    q = _param(det, Struct(a=Struct(scale=2.0), b=Struct(kp=1.0)))
    assert jnp.allclose(q.b.ki, 0.0)
    with pytest.raises(TypeError, match='does not accept rng'):
        det.parameterize(
            Struct(a=Struct(scale=2.0), b=Struct(kp=1.0)),
            rng=jax.random.PRNGKey(0))


def test_contract_init_leaf_and_state_input():
    node = Integrator().parameterize()
    assert _init(node.node, node.param) == 0.0

    w = Walker().parameterize(sigma=1.0)
    assert _init(w.node, w.param) == 0.0
    assert w.contract.apply_takes_rng

    seeded = Leaf(
        lambda state, input: (state, state + input),
        init=lambda rng: jax.random.normal(rng.next()),
        name='seeded',
    )
    s = _init(seeded.node, (), key=jax.random.PRNGKey(0))
    assert jnp.isfinite(s)
    with pytest.raises(TypeError, match='init requires rng='):
        seeded.init()


def test_public_init_non_cyclic_rejects_an_rng_channel():
    g = Gain().parameterize(scale=1.0)
    assert _init(g.node, g.param) == ()
    with pytest.raises(TypeError, match='init does not accept rng='):
        g.init(rng=jax.random.PRNGKey(0))


def test_contract_init_primes_from_the_input_channel():
    derivative = Derivative(0.1)
    # Priming `input` travels through `contract.prime` rather than the state-input bundle.
    state = _init(derivative.node, (), input=jnp.asarray(5.0))
    assert jnp.allclose(state, 5.0)
    assert derivative.contract.init_requires_input
    with pytest.raises(TypeError, match='unknown bundle fields'):
        derivative.init(Struct(input=jnp.asarray(5.0)))


def test_contract_init_composite_boundary_rng():
    seeded = Leaf(
        lambda state, input: (state, state + input),
        init=lambda rng: jax.random.normal(rng.next()),
        name='seeded',
    )
    net = serial(s=seeded, g=Gain()).parameterize(g=Struct(scale=2.0))
    s = _init(net.node, net.param, key=jax.random.PRNGKey(0))
    assert jnp.isfinite(s.s)
    assert s.g == ()


def test_composite_published_state_spec_is_a_canonical_init_input():
    net = (nn.relu >> Integrator()).parameterize()

    state = _init(net.node, net.param, net.contract.state_input_spec)

    assert state.integrator == 0.0
    assert state.relu == ()


def test_apply_fields_pack_into_the_bundle():
    def mix():
        def param(gain):
            return Struct(gain=jnp.asarray(gain))
        def apply(param, a, b):
            return Struct(y=param.gain * a + b)
        return Leaf(apply, param=param, name='mix')

    n = mix().parameterize(gain=2.0)
    assert n.apply(jnp.asarray(3.0), jnp.asarray(1.0)).y == 7.0
    assert n.apply(a=jnp.asarray(3.0), b=jnp.asarray(1.0)).y == 7.0
    assert n.apply(bundle=Struct(a=jnp.asarray(3.0), b=jnp.asarray(1.0))).y == 7.0
    with pytest.raises(TypeError, match='positionally and by keyword'):
        n.apply(jnp.asarray(3.0), a=jnp.asarray(5.0))


def test_cyclic_apply_fields_keep_state_positional():
    def acc():
        def init():
            return jnp.asarray(0.0)
        def apply(state, x):
            return state + x, state + x
        return Leaf(apply, init=init, name='acc')

    n = acc().parameterize()
    s = n.init()
    _, via_fields = n.apply(s, x=jnp.asarray(2.0))
    _, via_bundle = n.apply(s, bundle=Struct(x=jnp.asarray(2.0)))
    assert via_fields == via_bundle == 2.0
    with pytest.raises(TypeError, match='positional'):
        n.apply(x=jnp.asarray(2.0))


def test_unbound_surface_mirrors_the_bound_one():
    """Verifies unbound `Node.apply`, `Node.init`, and `scan` match their bound counterparts."""
    d = Gain()
    p = _param(d, Struct(scale=2.0))
    assert d.apply(p, 3.0) == d.bind(p).apply(3.0) == 6.0

    idef = Integrator()
    q = _param(idef)
    s = idef.init(q)
    assert s == idef.bind(q).init()
    s2, out = idef.apply(q, s, 5.0)
    assert out == 5.0
    _, traj = scan(idef).apply(q, s, jnp.ones(3))
    assert jnp.allclose(traj, jnp.array([1.0, 2.0, 3.0]))


def test_stateless_init_answers_the_empty_state():
    """A stateless node initializes to the empty state. A priming input is
    vacuous and accepted, so callers need not fork on cyclicity; a claimed
    state field stays a loud error."""
    d = Gain()
    p = _param(d, Struct(scale=2.0))
    bound = d.bind(p)

    assert bound.init() == ()
    assert bound.init(input=1.0) == ()
    assert d.init(p) == ()
    with pytest.raises(TypeError):
        bound.init(hidden=0.0)
