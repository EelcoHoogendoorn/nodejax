"""Methods and derivation: FOOP subclassing.

Methods are non-reserved callables on the def, param-first ('param plays
self'); the Node view binds param on attribute access. Derivation is
functional record update — the degenerate def->def transform: no
hierarchy, no MRO, and 'super' is an explicit call to Parent.apply_fn.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Node, NodeDef, node_def, derive, batch
from nodejax.struct import Struct
from nodejax.examples import gain_def, integrator_def


def gaussian_def():
    """A node carrying behavior beyond apply: whiten as the mapping,
    log_prob/sample as methods."""
    def param(mean, log_std):
        return Struct(mean=jnp.asarray(mean), log_std=jnp.asarray(log_std))

    def apply(param, input):
        return (input - param.mean) / jnp.exp(param.log_std)

    def log_prob(param, x):
        z = (x - param.mean) / jnp.exp(param.log_std)
        return -0.5 * z ** 2 - param.log_std - 0.5 * jnp.log(2 * jnp.pi)

    def sample(param, rng):
        return param.mean + jnp.exp(param.log_std) * jax.random.normal(rng)

    return node_def(apply, param=param, name='gaussian',
                    methods=dict(log_prob=log_prob, sample=sample))


def test_methods_bind_param():
    g = gaussian_def().parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))

    assert jnp.allclose(g.log_prob(0.0), -0.5 * jnp.log(2 * jnp.pi))
    key = jax.random.PRNGKey(0)
    assert jnp.allclose(g.sample(key), g.sample(key))  # pure, key-explicit

    # unbound access on the def: the raw param-first function
    raw = gaussian_def().log_prob
    assert jnp.allclose(raw(g.param, 0.0), g.log_prob(0.0))


def test_grad_through_method():
    """The pytree is the object, methods included: grad of a method w.r.t.
    the node flows into its params."""
    g = gaussian_def().parameterize(mean=jnp.asarray(1.0), log_std=jnp.asarray(0.0))
    grads = jax.grad(lambda n: n.log_prob(2.0))(g)
    assert isinstance(grads, Node)
    assert jnp.allclose(grads.param.mean, 1.0)  # d/dmean of -(x-mean)^2/2 at x=2


def test_missing_method_error_lists_methods():
    g = gaussian_def().parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))
    with pytest.raises(AttributeError, match='log_prob'):
        g.entropy()


def test_reserved_method_names_rejected():
    with pytest.raises(TypeError, match='apply'):
        node_def(lambda input: input, name='x', methods={'apply': lambda p: p})


def test_derive_override_apply_with_super():
    """Override apply, inherit init and param constructor; 'super' is an
    explicit call to the parent's contract fn."""
    Integrator = integrator_def()

    def apply(param, state, input):
        state, y = Integrator.apply_fn(param, state, input)
        return state, jnp.clip(y, -1.0, 1.0)

    Clipped = derive(Integrator, apply=apply, name='clipped')
    node = Clipped.parameterize(gain=jnp.asarray(1.0))

    final, outs = node.scan(None, jnp.ones(3))
    assert jnp.allclose(outs, jnp.array([1.0, 1.0, 1.0]))  # output clipped...
    assert jnp.allclose(final, 3.0)                        # ...state integrates on


def test_derive_can_add_state():
    """Flags recompute: deriving from a plain parent with a state-taking
    apply (plus init) yields a cyclic node — derivation moves through the
    lattice."""
    Gain = gain_def()

    def init(param):
        return jnp.asarray(0.0)

    def apply(param, state, input):
        y = Gain.apply(param, input)
        smoothed = 0.5 * state + 0.5 * y
        return smoothed, smoothed

    Smoothed = derive(Gain, apply=apply, init=init, name='smoothed')
    assert isinstance(Smoothed, NodeDef) and Smoothed.cyclic and Smoothed.parametric

    node = Smoothed.parameterize(scale=jnp.asarray(2.0))  # parent's param ctor inherited
    s = node.init()
    s, out = node.apply(s, 1.0)
    assert jnp.allclose(out, 1.0)   # 0.5*0 + 0.5*2
    s, out = node.apply(s, 1.0)
    assert jnp.allclose(out, 1.5)   # 0.5*1 + 0.5*2


def test_derive_merges_methods():
    G = gaussian_def()
    child = derive(G, name='gaussian2', methods=dict(
        log_prob=lambda param, x: jnp.asarray(42.0),          # override
        entropy=lambda param: param.log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e),  # add
    ))
    node = child.parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))

    assert jnp.allclose(node.log_prob(0.0), 42.0)                       # child wins
    assert jnp.allclose(node.entropy(), 0.5 * jnp.log(2 * jnp.pi * jnp.e))
    key = jax.random.PRNGKey(0)
    assert jnp.allclose(node.sample(key),                                # parent's kept
                        G.parameterize(mean=0.0, log_std=0.0).sample(key))


def test_derived_defs_stay_composable():
    """Derived defs are ordinary defs: they transform and compose."""
    Integrator = integrator_def()

    def apply(param, state, input):
        state, y = Integrator.apply_fn(param, state, input)
        return state, jnp.clip(y, -1.0, 1.0)

    Clipped = derive(Integrator, apply=apply, name='clipped')

    b = batch(Clipped, n=2).parameterize(gain=jnp.asarray(1.0))
    state = b.init()
    state, out = b.apply(state, jnp.array([0.4, 5.0]))
    assert jnp.allclose(out, jnp.array([0.4, 1.0]))

    pipe = (gain_def() >> Clipped).parameterize(
        gain=Struct(scale=3.0), clipped=Struct(gain=1.0))
    s = pipe.init()
    s, out = pipe.apply(s, 1.0)
    assert jnp.allclose(out, 1.0)  # 3.0 integrated once, clipped to 1
