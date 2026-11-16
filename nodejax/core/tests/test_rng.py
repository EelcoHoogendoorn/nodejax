"""Verifies boundary PRNG routing across `parameterize`, `init`, and `apply`.

Public `parameterize`, `initialize`, and `apply` calls accept a raw boundary
key only when the compiled role's static PRNG plan requires one. A declared
authored `rng` parameter receives a scope-local `KeyStream` rather than a
field in the role's data bundle, and replicating transforms (`batch`,
`ensemble`, `stack`, `scan`) split keys across elements and steps.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import scan, Leaf, batch, ensemble
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator, Walker


def test_walker_draws_from_apply_rng():
    """Verifies successive keys draw distinct noise while identical keys and state replay."""
    node = Walker().parameterize(sigma=jnp.asarray(1.0))
    s0 = node.init()
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))

    s1, step1 = node.apply(s0, 0.0, rng=k1)
    s2, x2 = node.apply(s1, 0.0, rng=k2)
    assert not jnp.allclose(step1, x2 - step1)

    _, again = node.apply(s0, 0.0, rng=k1)
    assert jnp.allclose(step1, again)


def test_rng_determinism_across_keys():
    node = Walker().parameterize(sigma=jnp.asarray(1.0))
    xs = jnp.zeros(20)
    s0 = node.init()

    _, traj_a = scan(node)(s0, xs, rng=jax.random.PRNGKey(0))
    _, traj_a2 = scan(node)(s0, xs, rng=jax.random.PRNGKey(0))
    _, traj_b = scan(node)(s0, xs, rng=jax.random.PRNGKey(1))

    assert jnp.allclose(traj_a, traj_a2)
    assert not jnp.allclose(traj_a, traj_b)


def test_rng_in_param_constructor():
    """Verifies a `param` constructor declaring `rng` receives a `KeyStream` at `parameterize`."""
    def param(rng, n):
        return Struct(w=jax.random.normal(rng.next(), (n,)))

    def apply(param, input):
        return param.w @ input

    proj = Leaf(apply, param=param, name='proj')
    a = proj.parameterize(rng=jax.random.PRNGKey(0), n=3)
    b = proj.parameterize(rng=jax.random.PRNGKey(0), n=3)
    c = proj.parameterize(rng=jax.random.PRNGKey(1), n=3)

    assert jnp.allclose(a.param.w, b.param.w)
    assert not jnp.allclose(a.param.w, c.param.w)


def test_apply_rng_is_a_declared_requirement():
    """Verifies an authored `rng` parameter compiles into `apply_takes_rng`.

    The public key arrives alongside ordinary input fields, and the authoring
    adapter exposes the framework stream as a `KeyStream`.
    """
    def noise():
        def param(scale):
            return Struct(scale=jnp.asarray(scale))
        def apply(param, x, rng):
            return x + param.scale * jax.random.normal(rng.next())
        return Leaf(apply, param=param, name='noise')

    node = noise()
    assert bool(node.contract.apply_takes_rng)
    n = node.parameterize(scale=2.0)
    key = jax.random.PRNGKey(0)
    assert jnp.allclose(n.apply(x=0.0, rng=key),
                        n.apply(x=0.0, rng=key))
    assert not jnp.allclose(n.apply(x=0.0, rng=key),
                            n.apply(x=0.0, rng=jax.random.PRNGKey(1)))


def test_batch_splits_keys():
    """Verifies `batch(...).apply` splits `rng` per batch element."""
    b = batch(Walker(), n=3).parameterize(sigma=jnp.asarray(1.0))
    state = b.init()

    state, out = b.apply(state, jnp.zeros(3), rng=jax.random.PRNGKey(0))
    assert out.shape == (3,)
    assert jnp.unique(out).size == 3


def test_ensemble_splits_keys():
    """Verifies `ensemble(...).apply` splits a single boundary key across members."""
    e = ensemble(Walker(), n=2).parameterize(sigma=1.0)
    state = e.init()

    state, out = e.apply(state, 0.0, rng=jax.random.PRNGKey(0))
    assert out.shape == (2,)
    assert not jnp.allclose(out[0], out[1])


def test_rng_routing_by_inspection():
    """Verifies static `init` PRNG plans skip deterministic siblings without shifting key draws."""
    from nodejax import Composite

    def Noise():
        def init(rng):
            return jax.random.normal(rng.next())
        def apply(state, input):
            return state, input
        return Leaf(apply, init=init, name='noise')

    def Lag():
        def init():
            return jnp.zeros(())
        def apply(state, input):
            return input, state
        return Leaf(apply, init=init, name='lag')

    def build(**members):
        def apply(self, input):
            return input
        return Composite(**members)(apply, name='m')

    key = jax.random.PRNGKey(0)
    small = build(a=Noise(), z=Noise())
    grown = build(a=Noise(), extra=Lag(), more=Lag(), z=Noise())

    s_small = small.init(rng=key)
    s_grown = grown.init(rng=key)

    assert jnp.allclose(s_small.a, s_grown.a)
    assert jnp.allclose(s_small.z, s_grown.z)
    assert len(jax.tree.leaves(s_grown.extra)) <= 1


def test_rng_to_deterministic_node_is_an_error():
    """Verifies passing `rng=` to a deterministic role raises `TypeError` at the boundary."""
    with pytest.raises(TypeError, match='does not accept rng='):
        Gain().parameterize(scale=1.0, rng=jax.random.PRNGKey(0))

    with pytest.raises(TypeError, match='does not accept rng='):
        Integrator().parameterize().init(rng=jax.random.PRNGKey(0))


def test_a_leaked_keystream_raises_named():
    """Verifies returning an unconsumed `KeyStream` at any field raises `TypeError` naming the path."""
    def stash(rng) -> Struct:
        return Struct(stash=rng)
    leaky = Leaf(lambda state, input: (state, input), init=stash, name='leaky')
    with pytest.raises(TypeError, match=r'leaky\.init.*escaped at stash'):
        leaky.init(rng=jax.random.PRNGKey(0))

    def stash_rng(rng) -> Struct:
        return Struct(rng=rng)
    rng_leaky = Leaf(lambda state, input: (state, input), init=stash_rng, name='rng_leaky')
    with pytest.raises(TypeError, match=r'rng_leaky\.init.*escaped at rng'):
        rng_leaky.init(rng=jax.random.PRNGKey(0))


def test_a_stream_returned_from_apply_raises_named():
    def apply(x: jax.Array, rng):
        return rng
    chatty = Leaf(apply, name='chatty')
    with pytest.raises(TypeError, match=r'chatty\.apply.*escaped'):
        chatty.apply(x=jnp.zeros(3), rng=jax.random.PRNGKey(0))


def test_rng_requirement_flags_read_the_tree():
    """Verifies each role's static PRNG flag composes across members independently of data specs."""
    from nodejax import nn

    def Withkey():
        return Leaf(lambda state, input: (state, input),
                    init=lambda rng: jax.random.normal(rng.next()),
                    name='withkey')

    pipe = (nn.Linear(4) >> Withkey()).node
    assert bool(pipe.contract.init_takes_rng)
    assert bool(pipe.contract.param_takes_rng)
    assert not bool(nn.gelu.contract.init_takes_rng)
    assert not bool(nn.gelu.contract.param_takes_rng)
