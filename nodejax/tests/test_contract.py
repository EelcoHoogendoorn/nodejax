"""The contract entries: build_param / build_state.

One bundle in, validated against the spec: unknown fields and missing
REQUIRED fields are loud; rng rides the bundle as a raw jax key; a
shape-reading constructor reaches its def through the def itself, never a
bundle field. These are the bundled signatures of the public node contract.

Above the contract sits the ONE boundary sugar, symmetric across all three
channels: parameterize / init / apply each take ONE bundle positionally, or
loose fields packed into one (state stays positional on a cyclic apply).
"""
import jax
import jax.numpy as jnp
import pytest

from nodejax.struct import Struct
from nodejax import node_def, serial
from nodejax.examples import gain_def, integrator_def, walker_def, plant_node


def pi():
    def param(kp, ki=0.0):
        return Struct(kp=jnp.asarray(kp), ki=jnp.asarray(ki))
    def apply(param, input):
        return param.kp * input
    return node_def(apply, param=param, name='pi')


def noisy():
    def param(rng, width=3):
        return Struct(w=jax.random.normal(rng.next(), (width,)))
    def apply(param, input):
        return param.w * input
    return node_def(apply, param=param, name='noisy')


# --- build_param ---

def test_build_param_leaf():
    p = gain_def().build_param(Struct(scale=2.0))
    assert jnp.allclose(p.scale, 2.0)
    assert gain_def().bind(p).apply(3.0) == 6.0

    q = pi().build_param(Struct(kp=1.0))       # ki falls to the ctor's default
    assert jnp.allclose(q.ki, 0.0)


def test_build_param_validates_the_bundle():
    with pytest.raises(TypeError, match='unknown bundle fields'):
        gain_def().build_param(Struct(scale=2.0, gain=1.0))
    with pytest.raises(TypeError, match='missing required'):
        gain_def().build_param(Struct())
    with pytest.raises(TypeError, match='missing required'):
        noisy().build_param(Struct(width=4))   # rng is REQUIRED in the spec
    with pytest.raises(TypeError, match='not parametric'):
        plant_node(0.1, 1.0, 0.1).ndef.build_param(Struct(x=1.0))
    assert plant_node(0.1, 1.0, 0.1).ndef.build_param(Struct()) == ()


def test_build_param_rng_rides_the_bundle():
    key = jax.random.PRNGKey(0)
    p1 = noisy().build_param(Struct(rng=key))
    p2 = noisy().build_param(Struct(rng=key))
    p3 = noisy().build_param(Struct(rng=jax.random.PRNGKey(1)))
    assert jnp.allclose(p1.w, p2.w)
    assert not jnp.allclose(p1.w, p3.w)
    assert p1.w.shape == (3,)                  # width falls to its default


def test_build_param_composite_boundary_rng():
    net = serial(n=noisy(), g=gain_def())
    p = net.build_param(Struct(rng=jax.random.PRNGKey(0), g=Struct(scale=2.0)))
    assert p.n.w.shape == (3,)
    assert jnp.allclose(p.g.scale, 2.0)
    with pytest.raises(TypeError, match='missing required'):
        net.build_param(Struct(g=Struct(scale=2.0)))   # boundary rng enforced

    det = serial(a=gain_def(), b=pi())
    q = det.build_param(Struct(a=Struct(scale=2.0), b=Struct(kp=1.0)))
    assert jnp.allclose(q.b.ki, 0.0)
    with pytest.raises(TypeError, match='unknown bundle fields'):
        det.build_param(Struct(a=Struct(scale=2.0), b=Struct(kp=1.0),
                               rng=jax.random.PRNGKey(0)))   # deterministic: no rng field


# --- build_state ---

def test_build_state_leaf_and_seed():
    node = integrator_def().parameterize(gain=1.0)
    assert node.ndef.build_state(node.param) == 0.0

    w = walker_def().parameterize(sigma=1.0)
    s = w.ndef.build_state(w.param, Struct(rng=jax.random.PRNGKey(0)))
    assert s.x == 0.0 and s.rng is not None
    with pytest.raises(TypeError, match='missing required'):
        w.ndef.build_state(w.param)


def test_build_state_non_cyclic_requires_empty_bundle():
    g = gain_def().parameterize(scale=1.0)
    assert g.ndef.build_state(g.param) == ()
    with pytest.raises(TypeError, match='not cyclic'):
        g.ndef.build_state(g.param, Struct(rng=jax.random.PRNGKey(0)))


def test_build_state_primes_from_the_input_channel():
    from nodejax.examples import derivative_node
    d = derivative_node(0.1)
    # input is its OWN channel, typed by apply_input_spec — never a bundle field
    s = d.ndef.build_state((), input=jnp.asarray(5.0))
    assert jnp.allclose(s, 5.0)                # primed from the real value
    assert d.ndef.init_requires_input          # the record, stored on the def
    with pytest.raises(TypeError, match='unknown bundle fields'):
        d.ndef.build_state((), Struct(input=jnp.asarray(5.0)))


def test_build_state_composite_boundary_rng():
    net = serial(w=walker_def(), g=gain_def()).parameterize(
        w=Struct(sigma=1.0), g=Struct(scale=2.0))
    s = net.ndef.build_state(net.param, Struct(rng=jax.random.PRNGKey(0)))
    assert s.w.x == 0.0                        # walker seeded from the split
    assert s.g == ()

# --- the boundary sugar: loose fields pack the same on all three channels ---

def test_apply_fields_pack_into_the_bundle():
    def mix():
        def param(gain):
            return Struct(gain=jnp.asarray(gain))
        def apply(param, a, b):
            return Struct(y=param.gain * a + b)
        return node_def(apply, param=param, name='mix')

    n = mix().parameterize(gain=2.0)
    bundle = Struct(a=jnp.asarray(3.0), b=jnp.asarray(1.0))
    assert n.apply(bundle).y == 7.0
    assert n.apply(a=jnp.asarray(3.0), b=jnp.asarray(1.0)).y == 7.0
    with pytest.raises(TypeError, match='not both'):
        n.apply(bundle, a=jnp.asarray(3.0))


def test_cyclic_apply_fields_keep_state_positional():
    def acc():
        def init():
            return jnp.asarray(0.0)
        def apply(state, x):
            return state + x, state + x
        return node_def(apply, init=init, name='acc')

    n = acc().parameterize()
    s = n.init()
    _, via_fields = n.apply(s, x=jnp.asarray(2.0))
    _, via_bundle = n.apply(s, Struct(x=jnp.asarray(2.0)))
    assert via_fields == via_bundle == 2.0
    with pytest.raises(TypeError, match='positional'):
        n.apply(x=jnp.asarray(2.0))              # state missing


def test_unbound_surface_mirrors_the_bound_one():
    """d.apply(param, ...) == d.bind(param).apply(...), sugar included;
    init and scan mirror the same way."""
    d = gain_def()
    p = d.build_param(Struct(scale=2.0))
    assert d.apply(p, 3.0) == d.bind(p).apply(3.0) == 6.0

    idef = integrator_def()
    q = idef.build_param(Struct(gain=1.0))
    s = idef.init(q)
    assert s == idef.bind(q).init()
    s2, out = idef.apply(q, s, 5.0)
    assert out == 5.0
    _, traj = idef.scan(q, s, jnp.ones(3))
    assert jnp.allclose(traj, jnp.array([1.0, 2.0, 3.0]))
