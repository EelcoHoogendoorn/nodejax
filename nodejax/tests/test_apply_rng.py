"""rng as an apply-input field, composed through composites.

A leaf whose apply names a trailing rng consumes entropy from its input
bundle. The requirement bubbles: a composite consumes apply-rng iff a member
does, takes ONE key in its own input bundle, and splits it toward each
consuming member — injected as that member's rng input field. Entropy never
rides the wire: an upstream member does not emit keys.
"""
import jax
import jax.numpy as jnp
import pytest

from nodejax.struct import Struct
from nodejax import node_def, serial, parallel


def scaler():
    def param(scale):
        return Struct(scale=jnp.asarray(scale))
    def apply(param, x):
        return Struct(x=param.scale * x)      # named output: the wire is a bundle
    return node_def(apply, param=param, name='scaler')


def jitter():
    def param(sigma):
        return Struct(sigma=jnp.asarray(sigma))
    def apply(param, x, rng):                  # trailing rng: entropy from the input
        return Struct(x=x + param.sigma * jax.random.normal(rng.next()))
    return node_def(apply, param=param, name='jitter')


def test_mid_pipe_member_draws_from_the_boundary_key():
    net = serial(a=scaler(), b=jitter()).parameterize(
        a=Struct(scale=2.0), b=Struct(sigma=1.0))
    key = jax.random.PRNGKey(0)
    out1 = net.apply(x=jnp.asarray(3.0), rng=key)
    out2 = net.apply(x=jnp.asarray(3.0), rng=key)
    out3 = net.apply(x=jnp.asarray(3.0), rng=jax.random.PRNGKey(1))
    assert jnp.allclose(out1.x, out2.x)        # same key -> same draw
    assert not jnp.allclose(out1.x, out3.x)    # different key -> different draw
    assert not jnp.allclose(out1.x, 6.0)       # noise actually applied to 2*3


def test_requirement_bubbles_through_nesting():
    inner = serial(a=scaler(), b=jitter())
    assert 'rng' in inner.apply_input_spec       # the spec IS the record
    outer = serial(core=inner, post=scaler())
    assert 'rng' in outer.apply_input_spec       # bubbled, not re-declared

    net = outer.parameterize(core=Struct(a=Struct(scale=2.0), b=Struct(sigma=1.0)),
                             post=Struct(scale=10.0))
    out = net.apply(x=jnp.asarray(3.0), rng=jax.random.PRNGKey(0))
    assert out.x.shape == ()                   # scaler wraps the jittered value


def test_missing_boundary_key_is_loud():
    net = serial(a=scaler(), b=jitter()).parameterize(
        a=Struct(scale=2.0), b=Struct(sigma=1.0))
    with pytest.raises(AttributeError, match='rng'):
        net.apply(x=jnp.asarray(3.0))     # input.rng fails naturally


def test_deterministic_pipe_consumes_no_apply_rng():
    net = serial(a=scaler(), b=scaler())
    assert 'rng' not in net.apply_input_spec


def test_parallel_splits_toward_the_stochastic_strand():
    block = parallel(n=jitter(), g=scaler()).parameterize(
        n=Struct(sigma=1.0), g=Struct(scale=2.0))
    key = jax.random.PRNGKey(0)
    inp = Struct(n=Struct(x=jnp.asarray(1.0)), g=Struct(x=jnp.asarray(3.0)), rng=key)
    out1 = block.apply(inp)
    out2 = block.apply(inp)
    assert jnp.allclose(out1.g.x, 6.0)         # deterministic strand untouched
    assert jnp.allclose(out1.n.x, out2.n.x)    # same key -> same draw
    with pytest.raises(AttributeError, match='rng'):
        block.apply(n=Struct(x=jnp.asarray(1.0)), g=Struct(x=jnp.asarray(3.0)))


def test_ensemble_splits_apply_rng_per_member():
    from nodejax import ensemble
    ens = ensemble(jitter(), n=3).parameterize(sigma=jnp.ones(3))
    key = jax.random.PRNGKey(7)
    out1 = ens.apply(x=jnp.asarray(0.0), rng=key)
    out2 = ens.apply(x=jnp.asarray(0.0), rng=key)
    assert out1.x.shape == (3,)
    assert jnp.allclose(out1.x, out2.x)              # same boundary key reproduces
    assert len(set(map(float, out1.x))) == 3         # members drew DIFFERENT noise


def test_batch_splits_apply_rng_per_element():
    from nodejax import batch
    b = batch(jitter()).parameterize(sigma=1.0)
    key = jax.random.PRNGKey(0)
    xs = Struct(x=jnp.zeros(4), rng=key)
    out = b.apply(xs)
    assert out.x.shape == (4,)
    assert len(set(map(float, out.x))) == 4          # per-element streams


def test_nonparametric_cyclic_ensemble():
    from nodejax import ensemble
    from nodejax.examples import walker_def
    import pytest as _pytest

    # walker minus params: a pure stochastic-state node
    def drifter():
        def init(rng):
            return Struct(x=jnp.asarray(0.0), rng=rng)
        def apply(state, input):
            new = state.x + input + 0.1 * jax.random.normal(state.rng)
            return state.replace(x=new), new
        return node_def(apply, init=init, name='drifter')

    ens = ensemble(drifter().ndef, n=3)
    node = ens.parameterize()
    s = node.init(rng=jax.random.PRNGKey(0))
    s2, out = node.apply(s, jnp.asarray(1.0))
    assert out.shape == (3,)
    assert len(set(map(float, out))) == 3            # independent streams

    with _pytest.raises(TypeError, match='needs n'):
        ensemble(drifter().ndef)


def test_wired_composite_splits_the_boundary_key():
    """A hand-wired composite obeys the same doctrine as every other
    composite: ONE boundary key on its input bundle, split toward each
    member call that consumes apply-rng — two calls, two draws. The wiring
    may still route a key explicitly (self.rng.next(), or rng= in the
    member's input), and explicit keys win over the injection."""
    from nodejax import composite

    def wired():
        def apply(self, input):
            once = self.j(x=input.x * 2.0)       # injected draw
            twice = self.j(x=input.x * 2.0)      # a DIFFERENT draw
            return Struct(a=once.x, b=twice.x)
        return composite(apply, members=dict(j=jitter()), name='wired',
                         apply_input_spec=Struct(x=jnp.asarray(0.0)))

    node = wired().parameterize(j=Struct(sigma=1.0))
    key = jax.random.PRNGKey(0)
    out1 = node.apply(x=jnp.asarray(3.0), rng=key)
    out2 = node.apply(x=jnp.asarray(3.0), rng=key)
    assert jnp.allclose(out1.a, out2.a)              # same key -> same draws
    assert not jnp.allclose(out1.a, out1.b)          # two calls, two draws
    assert 'rng' in wired().apply_input_spec         # bubbled onto the declared spec
