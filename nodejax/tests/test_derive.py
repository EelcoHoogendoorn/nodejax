"""Methods and derivation: FOOP subclassing.

Methods are non-reserved callables on the def whose reserved parameter
names are CHANNELS (ndef, param, state, rng), injected by the view
that binds the method; every other parameter is a call argument.
Derivation is functional record update — the degenerate def->def
transform: no hierarchy, no MRO, and 'super' is an explicit call to
Parent.apply_fn.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Node, NodeDef, node_def, derive, batch
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def Gaussian():
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
        # rng is a channel, delivered as a KeyStream in every context
        return param.mean + jnp.exp(param.log_std) * jax.random.normal(rng.next())

    return node_def(apply, param=param, name='gaussian',
                    methods=dict(log_prob=log_prob, sample=sample))


def test_methods_bind_param():
    g = Gaussian().parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))

    assert jnp.allclose(g.log_prob(0.0), -0.5 * jnp.log(2 * jnp.pi))
    key = jax.random.PRNGKey(0)
    # rng is a channel: a bare node offers none, so the caller passes it
    # by keyword, explicitly
    assert jnp.allclose(g.sample(rng=key), g.sample(rng=key))

    # unbound access on the def: the raw function, channels and all
    raw = Gaussian().log_prob
    assert jnp.allclose(raw(g.param, 0.0), g.log_prob(0.0))


def test_grad_through_method():
    """The pytree is the object, methods included: grad of a method w.r.t.
    the node flows into its params."""
    g = Gaussian().parameterize(mean=jnp.asarray(1.0), log_std=jnp.asarray(0.0))
    grads = jax.grad(lambda n: n.log_prob(2.0))(g)
    assert isinstance(grads, Node)
    assert jnp.allclose(grads.param.mean, 1.0)  # d/dmean of -(x-mean)^2/2 at x=2


def test_missing_method_error_lists_methods():
    g = Gaussian().parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))
    with pytest.raises(AttributeError, match='log_prob'):
        g.entropy()


def test_reserved_method_names_rejected():
    with pytest.raises(TypeError, match='apply'):
        node_def(lambda input: input, name='x', methods={'apply': lambda p: p})


def test_derive_override_apply_with_super():
    """Override apply, inherit init and param constructor; 'super' is an
    explicit call to the parent's contract fn."""
    integrator = Integrator()

    def apply(param, state, input):
        state, y = integrator.apply_fn(param, state, input)
        return state, jnp.clip(y, -1.0, 1.0)

    Clipped = derive(integrator, apply=apply, name='clipped')
    node = Clipped.parameterize(gain=jnp.asarray(1.0))

    final, outs = node.scan(None, jnp.ones(3))
    assert jnp.allclose(outs, jnp.array([1.0, 1.0, 1.0]))  # output clipped...
    assert jnp.allclose(final, 3.0)                        # ...state integrates on


def test_derive_can_add_state():
    """Flags recompute: deriving from a plain parent with a state-taking
    apply (plus init) yields a cyclic node — derivation moves through the
    lattice."""
    gain = Gain()

    def init(param):
        return jnp.asarray(0.0)

    def apply(param, state, input):
        y = gain.apply(param, input)
        smoothed = 0.5 * state + 0.5 * y
        return smoothed, smoothed

    Smoothed = derive(gain, apply=apply, init=init, name='smoothed')
    assert isinstance(Smoothed, NodeDef) and Smoothed.cyclic and Smoothed.parametric

    node = Smoothed.parameterize(scale=jnp.asarray(2.0))  # parent's param ctor inherited
    s = node.init()
    s, out = node.apply(s, 1.0)
    assert jnp.allclose(out, 1.0)   # 0.5*0 + 0.5*2
    s, out = node.apply(s, 1.0)
    assert jnp.allclose(out, 1.5)   # 0.5*1 + 0.5*2


def test_derive_merges_methods():
    G = Gaussian()
    child = derive(G, name='gaussian2', methods=dict(
        log_prob=lambda param, x: jnp.asarray(42.0),          # override
        entropy=lambda param: param.log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e),  # add
    ))
    node = child.parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))

    assert jnp.allclose(node.log_prob(0.0), 42.0)                       # child wins
    assert jnp.allclose(node.entropy(), 0.5 * jnp.log(2 * jnp.pi * jnp.e))
    key = jax.random.PRNGKey(0)
    assert jnp.allclose(node.sample(rng=key),                            # parent's kept
                        G.parameterize(mean=0.0, log_std=0.0).sample(rng=key))


def test_derived_defs_stay_composable():
    """Derived defs are ordinary defs: they transform and compose."""
    integrator = Integrator()

    def apply(param, state, input):
        state, y = integrator.apply_fn(param, state, input)
        return state, jnp.clip(y, -1.0, 1.0)

    Clipped = derive(integrator, apply=apply, name='clipped')

    b = batch(Clipped, n=2).parameterize(gain=jnp.asarray(1.0))
    state = b.init()
    state, out = b.apply(state, jnp.array([0.4, 5.0]))
    assert jnp.allclose(out, jnp.array([0.4, 1.0]))

    pipe = (Gain() >> Clipped).parameterize(
        gain=Struct(scale=3.0), clipped=Struct(gain=1.0))
    s = pipe.init()
    s, out = pipe.apply(s, 1.0)
    assert jnp.allclose(out, 1.0)  # 3.0 integrated once, clipped to 1


def test_method_channels_by_name():
    """Reserved names in a method signature are channels, injected by
    name: a leading prefix in the contract order (ndef, param, state,
    rng), the call's own arguments after them, positional or
    keyword. state on a bare node is the caller's to pass by
    keyword."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def init():
        return jnp.asarray(0.0)

    def apply(param, state, input):
        return state + input, state + input

    def level(ndef, param, state, x):
        return (state + x) * param.scale, ndef.name

    acc = node_def(apply, param=param, init=init, name='acc',
                   methods=dict(level=level)).parameterize(scale=2.0)

    # the contract order is validated at definition: channels lead...
    with pytest.raises(TypeError, match='lead'):
        node_def(apply, param=param, init=init, name='acc',
                 methods=dict(bad=lambda x, param: x))
    # ...and keep their order among themselves
    with pytest.raises(TypeError, match='order'):
        node_def(apply, param=param, init=init, name='acc',
                 methods=dict(bad=lambda state, param: state))

    # bare node: param and ndef inject; state is passed explicitly
    val, nm = acc.level(1.0, state=jnp.asarray(3.0))
    assert val == 8.0 and nm == 'acc'

    # wired member: all channels live, state chained through the step
    from nodejax import composite

    def wapply(self, input):
        before = self.acc.level(0.0)[0]
        self.acc(input)
        after = self.acc.level(0.0)[0]
        return Struct(before=before, after=after)

    rig = composite(wapply, members=dict(acc=acc), name='rig').parameterize()
    s = rig.with_input(jnp.asarray(1.0)).init()
    _, out = rig.apply(s, jnp.asarray(1.0))
    assert out.before == 0.0 and out.after == 2.0


def test_method_rng_is_a_stream_everywhere():
    """The rng channel arrives as a KeyStream in every context: the
    boundary stream inside a wiring, a wrapped explicit key on a bare
    node — one drawing idiom."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def draw(param, rng):
        return jax.random.normal(rng.next()) * param.scale

    g = node_def(lambda param, input: input * param.scale, param=param,
                 name='g', methods=dict(draw=draw)).parameterize(scale=1.0)

    key = jax.random.PRNGKey(0)
    a, b = g.draw(rng=key), g.draw(rng=key)
    assert jnp.allclose(a, b)                      # explicit key: deterministic

    # in a wiring, the boundary stream feeds the method: the author
    # declares rng at the composite boundary, the method draws from it
    from nodejax import composite

    def wapply(self, x, rng):
        return self.g.draw() + x

    rig = composite(wapply, members=dict(g=g), name='rig').parameterize()
    o1 = rig.apply(Struct(x=jnp.asarray(0.0), rng=key))
    o2 = rig.apply(Struct(x=jnp.asarray(0.0), rng=key))
    o3 = rig.apply(Struct(x=jnp.asarray(0.0), rng=jax.random.PRNGKey(1)))
    assert jnp.allclose(o1, o2) and not jnp.allclose(o1, o3)
