"""rng in all three positions, user's choice per node.

- param constructor argument (parameterize(rng=...)): just data, no mechanism
- input pytree field: just data, no mechanism
- reserved 'rng' STATE field: auto-advanced by the authoring layer — each
  apply consumes a fresh key and stores its successor, so stochastic step
  nodes never split or thread keys by hand

The auto-advance lives inside the contract apply_fn, so it composes with
every transform unchanged. The one rule transforms must honor: key state
replicates by SPLITTING (batch/ensemble/stack inits), never by broadcast —
a copied key would give every element the same noise stream.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Node, node_def, batch, ensemble
from nodejax.struct import Struct
from nodejax.examples import walker_def, gain_def, integrator_def


def test_rng_state_advances():
    """Successive applies draw different noise; the same state redraws the
    same noise (purity)."""
    node = walker_def().parameterize(sigma=jnp.asarray(1.0))
    s0 = node.init(rng=jax.random.PRNGKey(0))

    s1, step1 = node.apply(s0, 0.0)
    s2, x2 = node.apply(s1, 0.0)
    assert not jnp.allclose(step1, x2 - step1)   # key advanced between steps

    _, again = node.apply(s0, 0.0)
    assert jnp.allclose(step1, again)            # frozen state -> same draw


def test_rng_determinism_across_seeds():
    node = walker_def().parameterize(sigma=jnp.asarray(1.0))
    xs = jnp.zeros(20)

    _, traj_a = node.scan(node.init(rng=jax.random.PRNGKey(0)), xs)
    _, traj_a2 = node.scan(node.init(rng=jax.random.PRNGKey(0)), xs)
    _, traj_b = node.scan(node.init(rng=jax.random.PRNGKey(1)), xs)

    assert jnp.allclose(traj_a, traj_a2)         # trajectory = f(init key)
    assert not jnp.allclose(traj_a, traj_b)


def test_rng_in_param_constructor():
    """Param-init keys don't come from state — a constructor declaring rng
    receives a KeyStream at parameterize(rng=key)."""
    def param(rng, n):
        return Struct(w=jax.random.normal(rng.next(), (n,)))

    def apply(param, input):
        return param.w @ input

    proj = node_def(apply, param=param, name='proj')
    a = proj.parameterize(rng=jax.random.PRNGKey(0), n=3)
    b = proj.parameterize(rng=jax.random.PRNGKey(0), n=3)
    c = proj.parameterize(rng=jax.random.PRNGKey(1), n=3)

    assert jnp.allclose(a.param.w, b.param.w)    # same key, same weights
    assert not jnp.allclose(a.param.w, c.param.w)


def test_apply_rng_is_a_declared_field():
    """apply-side entropy is DECLARED: a trailing rng field arrives as a
    KeyStream, shows in the apply input spec, and therefore composes —
    boundaries hoist it and split toward consumers. (A key buried inside an
    opaque whole-`input` is just data the framework cannot see or route.)"""
    def noise():
        def param(scale):
            return Struct(scale=jnp.asarray(scale))
        def apply(param, x, rng):
            return x + param.scale * jax.random.normal(rng.next())
        return node_def(apply, param=param, name='noise')

    nd = noise()
    assert 'rng' in nd.apply_input_spec              # spec-visible, so composable
    n = nd.parameterize(scale=2.0)
    key = jax.random.PRNGKey(0)
    assert jnp.allclose(n.apply(x=0.0, rng=key),
                        n.apply(x=0.0, rng=key))
    assert not jnp.allclose(n.apply(x=0.0, rng=key),
                            n.apply(x=0.0, rng=jax.random.PRNGKey(1)))


def test_batch_splits_keys():
    """batch(...).init tiles ordinary state but SPLITS the rng field: each
    batch element gets an independent noise stream."""
    b = batch(walker_def(), n=3).parameterize(sigma=jnp.asarray(1.0))
    state = b.init(rng=jax.random.PRNGKey(0))

    state, out = b.apply(state, jnp.zeros(3))
    assert out.shape == (3,)
    assert jnp.unique(out).size == 3             # three distinct draws


def test_ensemble_splits_keys():
    """ensemble(...).init splits a single key across members."""
    e = ensemble(walker_def()).parameterize(sigma=jnp.ones(2))
    state = e.init(rng=jax.random.PRNGKey(0))

    state, out = e.apply(state, 0.0)
    assert out.shape == (2,)
    assert not jnp.allclose(out[0], out[1])      # independent member streams


def test_rng_routing_by_inspection():
    """Entropy routing is derived by inspecting init signatures: only
    members that consume rng receive splits — so adding or removing a
    DETERMINISTIC member does not shift the streams of its stochastic
    siblings (seed stability under refactoring)."""
    from nodejax import node_def, composite
    from nodejax.struct import Struct

    def noise_def():
        def init(rng):
            return Struct(rng=rng)
        def apply(state, input):
            return state, input
        return node_def(apply, init=init, name='noise')

    def lag_def():
        def init():
            return jnp.zeros(())
        def apply(state, input):
            return input, state
        return node_def(apply, init=init, name='lag')

    def build(**members):
        def apply(self, input):
            return input
        return composite(apply, members=members, name='m')

    key = jax.random.PRNGKey(0)
    small = build(a=noise_def(), z=noise_def())
    grown = build(a=noise_def(), extra=lag_def(), more=lag_def(), z=noise_def())

    s_small = small.init(rng=key)
    s_grown = grown.init(rng=key)

    # the stochastic members' streams are untouched by the deterministic
    # additions between them
    assert jnp.all(s_small.a.rng == s_grown.a.rng)
    assert jnp.all(s_small.z.rng == s_grown.z.rng)
    assert len(jax.tree.leaves(s_grown.extra)) <= 1   # got no key


def test_rng_to_deterministic_node_is_an_error():
    """The spec is an iff: rng in the bundle spec means stochastic, absent
    means deterministic. A key passed to a deterministic node fails as an
    ordinary unknown bundle field — no special-cased guard, the same
    validation as any other field."""
    with pytest.raises(TypeError, match='unknown bundle fields'):
        gain_def().parameterize(scale=1.0, rng=jax.random.PRNGKey(0))

    with pytest.raises(TypeError, match='unknown bundle fields'):
        integrator_def().parameterize(gain=1.0).init(rng=jax.random.PRNGKey(0))


def test_keystreams_never_escape_the_lifts():
    """A returned KeyStream collapses to a raw key at EVERY lift exit —
    param, init and apply alike; the stream object never enters a pytree."""
    def hoarder():
        def param(rng):
            return Struct(key=rng)                 # tries to store the stream
        def init(rng):
            return Struct(rng=rng)
        def apply(param, state, x, rng):
            return Struct(rng=rng), x              # tries to emit it in state
        return node_def(apply, param=param, init=init, name='hoarder')

    n = hoarder().parameterize(rng=jax.random.PRNGKey(0))
    assert n.param.key.shape == (2,)               # a raw key, not a KeyStream
    s = n.init(rng=jax.random.PRNGKey(1))
    assert s.rng.shape == (2,)
    s2, out = n.apply(s, x=jnp.asarray(1.0), rng=jax.random.PRNGKey(2))
    assert s2.rng.shape == (2,)
