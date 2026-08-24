"""Methods and derivation: FOOP subclassing.

Methods are non-reserved callables on the node whose reserved parameter
names are CHANNELS (node, param, state, rng), injected by the view
that binds the method; every other parameter is a call argument.
Derivation is functional record update — the degenerate def->def
transform: no hierarchy, no MRO, and 'super' is an explicit call to
the parent compiled apply call.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import scan, PNode, Node, Leaf, derive, batch
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator


def Gaussian() -> Node:
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

    return Leaf(apply, param=param, name='gaussian',
                    methods=dict(log_prob=log_prob, sample=sample))


def test_methods_bind_param():
    g = Gaussian().parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))

    assert jnp.allclose(g.log_prob(0.0), -0.5 * jnp.log(2 * jnp.pi))
    key = jax.random.PRNGKey(0)
    # rng is a channel: a bare node declares none, so the caller passes it
    # by keyword, explicitly
    assert jnp.allclose(g.sample(rng=key), g.sample(rng=key))

    # unbound access on the node: the raw function, channels and all
    raw = Gaussian().log_prob
    assert jnp.allclose(raw(g.param, 0.0), g.log_prob(0.0))


def test_grad_through_method():
    """The pytree is the object, methods included: grad of a method w.r.t.
    the node flows into its params."""
    g = Gaussian().parameterize(mean=jnp.asarray(1.0), log_std=jnp.asarray(0.0))
    grads = jax.grad(lambda n: n.log_prob(2.0))(g)
    assert type(grads) is PNode
    assert jnp.allclose(grads.param.mean, 1.0)  # d/dmean of -(x-mean)^2/2 at x=2


def test_missing_method_error_lists_methods():
    g = Gaussian().parameterize(mean=jnp.asarray(0.0), log_std=jnp.asarray(0.0))
    with pytest.raises(AttributeError, match='log_prob'):
        g.entropy()


def test_reserved_method_names_rejected():
    with pytest.raises(TypeError, match='apply'):
        Leaf(lambda input: input, name='x', methods={'apply': lambda p: p})


@pytest.mark.parametrize('name', [
    'contract', 'members', 'specialize', 'with_input',
])
def test_framework_authoring_names_are_reserved(name):
    with pytest.raises(TypeError, match=name):
        Leaf(lambda input: input, name='x', methods={name: lambda: None})


@pytest.mark.parametrize('name', [
    'param_fn', 'init_fn', 'apply_fn', 'feed', 'feed_bundle',
    'resolve_input', 'rebuild_members', 'preserve_input',
    'get_apply_input_spec',
])
def test_old_internal_operation_names_are_valid_methods(name):
    node = Leaf(
        lambda input: input,
        name='x',
        methods={name: lambda value: value + 1},
    )

    assert getattr(node, name)(2) == 3


def test_derive_override_apply_with_super():
    """Override apply, inherit init and param constructor; 'super' is an
    explicit call to the parent's contract fn."""
    integrator = Integrator()

    def apply(param, state, input):
        state, y = integrator.apply(param, state, input)
        return state, jnp.clip(y, -1.0, 1.0)

    Clipped = derive(integrator, apply=apply, name='clipped')
    node = Clipped.parameterize()

    final, outs = scan(node)(node.init(), jnp.ones(3))
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
    assert type(Smoothed) is Node and Smoothed.cyclic and Smoothed.parametric

    node = Smoothed.parameterize(scale=jnp.asarray(2.0))  # parent's param ctor inherited
    s = node.init()
    s, out = node.apply(s, 1.0)
    assert jnp.allclose(out, 1.0)   # 0.5*0 + 0.5*2
    s, out = node.apply(s, 1.0)
    assert jnp.allclose(out, 1.5)   # 0.5*1 + 0.5*2


def test_derive_struct_state_default_remains_complete():
    def init(initial=Struct(original=1.0)):
        return initial

    def apply(state, input):
        return state, input

    parent = Leaf(apply, init=init)
    child = derive(
        parent,
        state_input_spec=Struct(initial=Struct(replacement=2.0)),
    )

    default = child.initialize()
    assert default.state.__keys__ == ('replacement',)
    assert default.state.replacement == 2.0

    explicit = child.initialize(initial=Struct(other=3.0))
    assert explicit.state.__keys__ == ('other',)
    assert explicit.state.other == 3.0


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
    """Derived nodes are ordinary nodes: they transform and compose."""
    integrator = Integrator()

    def apply(param, state, input):
        state, y = integrator.apply(param, state, input)
        return state, jnp.clip(y, -1.0, 1.0)

    Clipped = derive(integrator, apply=apply, name='clipped')

    b = batch(Clipped, n=2).parameterize()
    state = b.init()
    state, out = b.apply(state, jnp.array([0.4, 5.0]))
    assert jnp.allclose(out, jnp.array([0.4, 1.0]))

    pipe = (Gain() >> Clipped).parameterize(
        gain=Struct(scale=3.0))
    s = pipe.init()
    s, out = pipe.apply(s, 1.0)
    assert jnp.allclose(out, 1.0)  # 3.0 integrated once, clipped to 1


def test_derived_apply_keeps_its_entropy_requirement():
    parent = Leaf(lambda input: input, name='identity')

    def apply(input, rng):
        return input + jax.random.normal(rng.next())

    child = derive(parent, apply=apply, name='noisy_identity')
    key = jax.random.PRNGKey(0)

    assert child.contract.apply_takes_rng
    assert child.contract.input_spec is None
    assert 'rng' not in child.contract.apply_fields
    assert jnp.allclose(child(input=0.0, rng=key), child(input=0.0, rng=key))


def test_derive_inherits_boundary_actions():
    def reset(carried, initialized, decided):
        return initialized

    parent = Leaf(lambda input: input, name='identity',
                  boundary={'episode': reset})
    child = derive(parent, name='derived_identity')

    assert child._def.boundaries == parent._def.boundaries
    assert child._def.boundaries['episode'] is reset


def test_method_channels_by_name():
    """Reserved names in a method signature are injected by name: the
    binding-stage prefix in binding order (self, node, param, state), the
    call's own arguments after them, positional or keyword. state on a bare
    node is the caller's to pass by keyword."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def init():
        return jnp.asarray(0.0)

    def apply(param, state, input):
        return state + input, state + input

    def level(node, param, state, x):
        return (state + x) * param.scale, node.name

    acc = Leaf(apply, param=param, init=init, name='acc',
                   methods=dict(level=level)).parameterize(scale=2.0)

    # binding order is validated at definition: bound names come first...
    with pytest.raises(TypeError, match='come first'):
        Leaf(apply, param=param, init=init, name='acc',
                 methods=dict(bad=lambda x, param: x))
    # ...and keep their order among themselves
    with pytest.raises(TypeError, match='order'):
        Leaf(apply, param=param, init=init, name='acc',
                 methods=dict(bad=lambda state, param: state))

    # bare node: param and node inject; state is passed explicitly
    val, nm = acc.level(1.0, state=jnp.asarray(3.0))
    assert val == 8.0 and nm == 'acc'

    # wired member: all channels live, state chained through the step
    from nodejax import Composite

    def wapply(self, input):
        before = self.acc.level(0.0)[0]
        self.acc(input)
        after = self.acc.level(0.0)[0]
        return Struct(before=before, after=after)

    rig = Composite(acc=acc)(wapply, name='rig').parameterize()
    s = rig.with_input(jnp.asarray(1.0)).bind(rig.param).init()
    _, out = rig.apply(s, jnp.asarray(1.0))
    assert out.before == 0.0 and out.after == 2.0


def test_method_rng_is_a_stream_everywhere():
    """The rng slot arrives as a KeyStream in every context: the
    boundary stream inside a wiring, a wrapped explicit key on a bare
    node — one drawing idiom."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def draw(param, rng):
        return jax.random.normal(rng.next()) * param.scale

    g = Leaf(lambda param, input: input * param.scale, param=param,
                 name='g', methods=dict(draw=draw)).parameterize(scale=1.0)

    key = jax.random.PRNGKey(0)
    a, b = g.draw(rng=key), g.draw(rng=key)
    assert jnp.allclose(a, b)                      # explicit key: deterministic

    # in a wiring, the boundary stream feeds the method: the author
    # declares rng at the composite boundary, the method draws from it
    from nodejax import Composite

    def wapply(self, x, rng):
        return self.g.draw() + x

    rig = Composite(g=g)(wapply, name='rig').parameterize()
    o1 = rig.apply(x=jnp.asarray(0.0), rng=key)
    o2 = rig.apply(x=jnp.asarray(0.0), rng=key)
    o3 = rig.apply(x=jnp.asarray(0.0), rng=jax.random.PRNGKey(1))
    assert jnp.allclose(o1, o2) and not jnp.allclose(o1, o3)
