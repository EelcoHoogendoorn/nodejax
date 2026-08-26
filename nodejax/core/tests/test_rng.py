"""The public RNG channel and model-owned RNG state are distinct.

- ``parameterize/initialize/apply(rng=key)`` accepts a raw boundary key only
  when the compiled role's static RNG plan requires one.
- A declared authored ``rng`` parameter receives a scope-local ``KeyStream``;
  it is never a field in the role's data bundle or input spec.
- A model STATE value named ``rng`` is modeled state. The authoring layer
  advances that stored key on apply, so stochastic step nodes need not split
  or thread their state key by hand.

The auto-advance lives inside the contract apply_fn, so it composes with
every transform unchanged. The one rule transforms must honor: key state
replicates by SPLITTING (batch/ensemble/stack inits), never by broadcast —
a copied key would give every element the same noise stream.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import scan, PNode, Leaf, batch, ensemble, KeyStream
from nodejax.struct import Struct
from nodejax.control import Gain, Integrator, Walker


def test_rng_state_advances():
    """Successive applies draw different noise; the same state redraws the
    same noise (purity)."""
    node = Walker().parameterize(sigma=jnp.asarray(1.0))
    s0 = node.init(rng=jax.random.PRNGKey(0))

    s1, step1 = node.apply(s0, 0.0)
    s2, x2 = node.apply(s1, 0.0)
    assert not jnp.allclose(step1, x2 - step1)   # key advanced between steps

    _, again = node.apply(s0, 0.0)
    assert jnp.allclose(step1, again)            # frozen state -> same draw


def test_state_rng_can_seed_a_local_keystream():
    """A Leaf can wrap its explicit RNG state for multiple local draws."""
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def init(rng):
        return Struct(rng=rng)

    def apply(param, state, input):
        rng = KeyStream(state.rng)
        first_key = rng.next()
        second_key = rng.next()
        output = Struct(
            value=input + param.scale * (
                jax.random.normal(first_key) + jax.random.normal(second_key)),
            first=first_key,
            second=second_key,
        )
        return state, output

    node = Leaf(apply, param=param, init=init).parameterize(scale=1.0)
    state = node.init(rng=jax.random.PRNGKey(42))
    successor, output = node.apply(state, 1.0)
    _, replay = node.apply(state, 1.0)
    assert type(successor.rng) is not KeyStream
    assert not jnp.array_equal(successor.rng, state.rng)
    assert not jnp.array_equal(output.first, output.second)
    assert jnp.array_equal(output.first, replay.first)
    assert jnp.array_equal(output.second, replay.second)


def test_rng_determinism_across_keys():
    node = Walker().parameterize(sigma=jnp.asarray(1.0))
    xs = jnp.zeros(20)

    _, traj_a = scan(node)(node.init(rng=jax.random.PRNGKey(0)), xs)
    _, traj_a2 = scan(node)(node.init(rng=jax.random.PRNGKey(0)), xs)
    _, traj_b = scan(node)(node.init(rng=jax.random.PRNGKey(1)), xs)

    assert jnp.allclose(traj_a, traj_a2)         # trajectory = f(init key)
    assert not jnp.allclose(traj_a, traj_b)


def test_rng_in_param_constructor():
    """Param-init keys don't come from state — a constructor declaring rng
    receives a KeyStream at parameterize(rng=key)."""
    def param(rng, n):
        return Struct(w=jax.random.normal(rng.next(), (n,)))

    def apply(param, input):
        return param.w @ input

    proj = Leaf(apply, param=param, name='proj')
    a = proj.parameterize(rng=jax.random.PRNGKey(0), n=3)
    b = proj.parameterize(rng=jax.random.PRNGKey(0), n=3)
    c = proj.parameterize(rng=jax.random.PRNGKey(1), n=3)

    assert jnp.allclose(a.param.w, b.param.w)    # same key, same weights
    assert not jnp.allclose(a.param.w, c.param.w)


def test_apply_rng_is_a_declared_requirement():
    """An authored trailing rng parameter compiles into the apply RNG plan.

    The public key arrives beside ordinary input fields, and the authoring
    adapter exposes the corresponding framework stream as a ``KeyStream``.
    A key buried inside opaque domain data is not the call channel.
    """
    def noise():
        def param(scale):
            return Struct(scale=jnp.asarray(scale))
        def apply(param, x, rng):
            return x + param.scale * jax.random.normal(rng.next())
        return Leaf(apply, param=param, name='noise')

    node = noise()
    assert bool(node.contract.apply_takes_rng)                        # flag-visible, so composable
    n = node.parameterize(scale=2.0)
    key = jax.random.PRNGKey(0)
    assert jnp.allclose(n.apply(x=0.0, rng=key),
                        n.apply(x=0.0, rng=key))
    assert not jnp.allclose(n.apply(x=0.0, rng=key),
                            n.apply(x=0.0, rng=jax.random.PRNGKey(1)))


def test_batch_splits_keys():
    """batch(...).init tiles ordinary state but SPLITS the rng field: each
    batch element gets an independent noise stream."""
    b = batch(Walker(), n=3).parameterize(sigma=jnp.asarray(1.0))
    state = b.init(rng=jax.random.PRNGKey(0))

    state, out = b.apply(state, jnp.zeros(3))
    assert out.shape == (3,)
    assert jnp.unique(out).size == 3             # three distinct draws


def test_ensemble_splits_keys():
    """ensemble(...).init splits a single key across members."""
    e = ensemble(Walker(), n=2).parameterize(sigma=1.0)
    state = e.init(rng=jax.random.PRNGKey(0))

    state, out = e.apply(state, 0.0)
    assert out.shape == (2,)
    assert not jnp.allclose(out[0], out[1])      # independent member streams


def test_rng_routing_by_inspection():
    """Init signatures compile into static plans before a call begins.

    Only members whose plans consume entropy receive streams, so adding or
    removing a deterministic member does not shift its stochastic siblings
    (key stability under refactoring).
    """
    from nodejax import Leaf, Composite
    from nodejax.struct import Struct

    def Noise():
        def init(rng):
            return Struct(rng=rng)
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

    # the stochastic members' streams are untouched by the deterministic
    # additions between them
    assert jnp.all(s_small.a.rng == s_grown.a.rng)
    assert jnp.all(s_small.z.rng == s_grown.z.rng)
    assert len(jax.tree.leaves(s_grown.extra)) <= 1   # got no key


def test_rng_to_deterministic_node_is_an_error():
    """The compiled RNG plan is exact in both directions.

    A deterministic role has an empty plan, so a surplus raw key fails at the
    public RNG boundary independently of ordinary bundle validation.
    """
    with pytest.raises(TypeError, match='does not accept rng='):
        Gain().parameterize(scale=1.0, rng=jax.random.PRNGKey(0))

    with pytest.raises(TypeError, match='does not accept rng='):
        Integrator().parameterize().init(rng=jax.random.PRNGKey(0))


def test_keystreams_never_escape_the_lifts():
    """The stream object never enters a pytree. A field named 'rng'
    stores it as its advanced raw key at every lift exit, param, init
    and apply alike; any OTHER field is a leak and raises, pinned in
    the leak tests below (recalibrated when the silent collapse of
    non-rng stores became a named error)."""
    def hoarder():
        def param(rng):
            return Struct(rng=rng)                 # the one sanctioned store
        def init(rng):
            return Struct(rng=rng)
        def apply(param, state, x, rng):
            return Struct(rng=rng), x
        return Leaf(apply, param=param, init=init, name='hoarder')

    n = hoarder().parameterize(rng=jax.random.PRNGKey(0))
    assert n.param.rng.shape == (2,)               # a raw key, not a KeyStream
    s = n.init(rng=jax.random.PRNGKey(1))
    assert s.rng.shape == (2,)
    s2, out = n.apply(s, x=jnp.asarray(1.0), rng=jax.random.PRNGKey(2))
    assert s2.rng.shape == (2,)


def test_a_leaked_keystream_raises_named():
    """The stream is scope-local: it leaves a node only through a field
    named 'rng', as its advanced key. Any other escape raises at the
    lift exit with the path, instead of surfacing later as an illegible
    jax error (KeyStream.__jax_array__ once auto-drew on coercion, and
    jax dropped that hook)."""
    def stash(rng) -> Struct:
        return Struct(stash=rng)
    leaky = Leaf(lambda state, input: (state, input), init=stash, name='leaky')
    with pytest.raises(TypeError, match=r'leaky\.init.*escaped at stash'):
        leaky.init(rng=jax.random.PRNGKey(0))


def test_the_rng_field_still_stores_the_stream():
    """The sanctioned store: init returns the stream under 'rng' and the
    lift collapses it to an advanced key, dropout's own idiom."""
    def start(rng) -> Struct:
        return Struct(rng=rng)
    keeper = Leaf(lambda state, input: (state, input), init=start, name='keeper')
    state = keeper.init(rng=jax.random.PRNGKey(0))
    assert state.rng.shape == (2,)


def test_a_stream_returned_from_apply_raises_named():
    def apply(x: jax.Array, rng):
        return rng
    chatty = Leaf(apply, name='chatty')
    with pytest.raises(TypeError, match=r'chatty\.apply.*escaped'):
        chatty.apply(x=jnp.zeros(3), rng=jax.random.PRNGKey(0))


def test_rng_requirement_flags_read_the_tree():
    """Each role's static RNG plan survives composition independently of its
    data-input spec. A role that does not exist consumes no key."""
    from nodejax import nn

    def Withkey():
        return Leaf(lambda state, input: (state, input),
                        init=lambda rng: Struct(rng=rng), name='withkey')

    pipe = (nn.Linear(4) >> Withkey()).node
    assert bool(pipe.contract.init_takes_rng)                 # the member's init wants a key
    assert bool(pipe.contract.param_takes_rng)                # linear's constructor draws
    assert not bool(nn.gelu.contract.init_takes_rng)     # acyclic: no init slot
    assert not bool(nn.gelu.contract.param_takes_rng)    # non-parametric: no ctor
