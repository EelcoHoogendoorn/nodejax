"""Verifies `apply` PRNG plans across `serial`, `parallel`, `batch`, `ensemble`, and authored `Composite` graphs."""
import jax
import jax.numpy as jnp
import pytest

from nodejax.struct import Struct
from nodejax import Node, Leaf, Composite, serial, parallel, nn


def Scaler() -> Node:
    def param(scale):
        return Struct(scale=jnp.asarray(scale))
    def apply(param, x):
        return param.scale * x
    return Leaf(apply, param=param, name='scaler')


def Jitter() -> Node:
    def param(sigma):
        return Struct(sigma=jnp.asarray(sigma))
    def apply(param, x, rng):                  # Declares an `apply` PRNG stream.
        return x + param.sigma * jax.random.normal(rng.next())
    return Leaf(apply, param=param, name='jitter')


def _apply_draw() -> Node:
    def apply(input, rng):
        return input + jax.random.normal(rng.next(), ())
    return Leaf(apply, name='apply_draw').node


def _primed() -> Node:
    def init(input):
        return input
    def apply(state, input):
        return state, input
    return Leaf(apply, init=init, name='primed').node


def _shape_state() -> Node:
    def init(node):
        return jnp.zeros_like(node.input)
    def apply(state, input):
        return state, input
    return Leaf(apply, init=init, name='shape_state').node


def _init_and_apply_key() -> Node:
    def init(rng):
        return rng.next()
    def apply(state, input, rng):
        return state, rng.next()
    return Leaf(apply, init=init, name='two_role_draw').node


def test_empty_input_rng_source_drives_serial_parameter_shape_walk():
    def apply(rng):
        return jax.random.normal(rng.next(), (4,))

    source = Leaf(apply, name='random_source')
    model = (source >> nn.Linear(2)).parameterize(
        rng=jax.random.PRNGKey(0))

    assert model.param.linear.w.shape == (4, 2)
    assert model.apply(rng=jax.random.PRNGKey(1)).shape == (2,)


def _priming_graph(kind: str, source: Node, sink: Node):
    if kind == 'serial':
        built = serial(source=source, sink=sink)
    else:
        def apply(self, input):
            return self.sink(self.source(input))
        built = Composite(source=source, sink=sink)(apply, name='wired_priming')
    return built if built.bound else built.bind(Struct(source=(), sink=()))


@pytest.mark.parametrize('kind', ('serial', 'wired'))
def test_real_init_priming_uses_caller_apply_entropy(kind):
    node = _priming_graph(kind, _apply_draw(), _primed())
    assert node.contract.init_takes_rng

    input = jnp.asarray(2.0)
    key = jax.random.PRNGKey(3)
    first = node.init(rng=key, input=input)
    again = node.init(rng=key, input=input)
    other = node.init(rng=jax.random.PRNGKey(4), input=input)
    assert jnp.array_equal(first.sink, again.sink)
    assert not jnp.array_equal(first.sink, other.sink)
    with pytest.raises(TypeError, match='init requires rng='):
        node.init(input=input)


@pytest.mark.parametrize('kind', ('serial', 'wired'))
def test_shape_only_init_never_adopts_apply_entropy(kind):
    node = _priming_graph(kind, _apply_draw(), _shape_state()).with_input(
        jnp.asarray(2.0))
    assert not node.contract.init_takes_rng

    from_spec = node.init()
    from_sample_shape = node.init(input=jnp.asarray(2.0))

    assert from_spec.sink.shape == ()
    assert jnp.array_equal(from_spec.sink, from_sample_shape.sink)
    with pytest.raises(TypeError, match='does not accept rng'):
        node.init(rng=jax.random.PRNGKey(3), input=jnp.asarray(2.0))


@pytest.mark.parametrize('kind', ('serial', 'wired'))
def test_init_and_apply_roles_receive_distinct_caller_streams(kind):
    node = _priming_graph(kind, _init_and_apply_key(), _primed())
    key = jax.random.PRNGKey(5)
    first = node.init(rng=key, input=jnp.asarray(0.0))
    again = node.init(rng=key, input=jnp.asarray(0.0))
    other = node.init(rng=jax.random.PRNGKey(6), input=jnp.asarray(0.0))

    assert jnp.array_equal(first.source, again.source)
    assert jnp.array_equal(first.sink, again.sink)
    assert not jnp.array_equal(first.source, first.sink)
    assert not jnp.array_equal(first.source, other.source)
    assert not jnp.array_equal(first.sink, other.sink)


def test_mid_pipe_member_draws_from_the_boundary_key():
    net = serial(a=Scaler(), b=Jitter()).parameterize(
        a=Struct(scale=2.0), b=Struct(sigma=1.0))
    key = jax.random.PRNGKey(0)
    out1 = net.apply(x=jnp.asarray(3.0), rng=key)
    out2 = net.apply(x=jnp.asarray(3.0), rng=key)
    out3 = net.apply(x=jnp.asarray(3.0), rng=jax.random.PRNGKey(1))
    assert jnp.allclose(out1, out2)
    assert not jnp.allclose(out1, out3)
    assert not jnp.allclose(out1, 6.0)


def test_requirement_bubbles_through_nesting():
    inner = serial(a=Scaler(), b=Jitter())
    assert bool(inner.contract.apply_takes_rng)
    outer = serial(core=inner, post=Scaler())
    assert bool(outer.contract.apply_takes_rng)

    net = outer.parameterize(core=Struct(a=Struct(scale=2.0), b=Struct(sigma=1.0)),
                             post=Struct(scale=10.0))
    out = net.apply(x=jnp.asarray(3.0), rng=jax.random.PRNGKey(0))
    assert out.shape == ()


def test_missing_boundary_key_is_loud():
    net = serial(a=Scaler(), b=Jitter()).parameterize(
        a=Struct(scale=2.0), b=Struct(sigma=1.0))
    with pytest.raises(TypeError, match='apply requires rng='):
        net.apply(x=jnp.asarray(3.0))


def test_deterministic_pipe_consumes_no_apply_rng():
    net = serial(a=Scaler(), b=Scaler())
    assert 'rng' not in net.contract.apply_fields


def test_parallel_splits_toward_the_stochastic_strand():
    block = parallel(n=Jitter(), g=Scaler()).parameterize(
        n=Struct(sigma=1.0), g=Struct(scale=2.0))
    key = jax.random.PRNGKey(0)
    out1 = block.apply(
        n=jnp.asarray(1.0), g=jnp.asarray(3.0), rng=key)
    out2 = block.apply(
        n=jnp.asarray(1.0), g=jnp.asarray(3.0), rng=key)
    assert jnp.allclose(out1.g, 6.0)
    assert jnp.allclose(out1.n, out2.n)
    with pytest.raises(TypeError, match='apply requires rng='):
        block.apply(n=jnp.asarray(1.0), g=jnp.asarray(3.0))


def test_ensemble_splits_apply_rng_per_member():
    from nodejax import ensemble
    ens = ensemble(Jitter(), n=3).parameterize(sigma=1.0)
    key = jax.random.PRNGKey(7)
    out1 = ens.apply(x=jnp.asarray(0.0), rng=key)
    out2 = ens.apply(x=jnp.asarray(0.0), rng=key)
    assert out1.shape == (3,)
    assert jnp.allclose(out1, out2)
    assert len(set(map(float, out1))) == 3


def test_batch_splits_apply_rng_per_element():
    from nodejax import batch
    b = batch(Jitter()).parameterize(sigma=1.0)
    key = jax.random.PRNGKey(0)
    out = b.apply(x=jnp.zeros(4), rng=key)
    assert out.shape == (4,)
    assert len(set(map(float, out))) == 4


def test_nonparametric_cyclic_ensemble():
    from nodejax import ensemble
    from nodejax.control import Walker
    import pytest as _pytest

    # Builds a parameterless stochastic-state node.
    def drifter():
        def init(rng):
            return Struct(x=jnp.asarray(0.0), rng=rng)
        def apply(state, input):
            new = state.x + input + 0.1 * jax.random.normal(state.rng)
            return state.replace(x=new), new
        return Leaf(apply, init=init, name='drifter')

    ens = ensemble(drifter().node, n=3)
    node = ens.parameterize()
    s = node.init(rng=jax.random.PRNGKey(0))
    s2, out = node.apply(s, jnp.asarray(1.0))
    assert out.shape == (3,)
    assert len(set(map(float, out))) == 3

    # Leaving `n` unbound yields a generic ensemble.
    g = ensemble(drifter().node)
    assert g.generic


def test_wired_composite_splits_the_boundary_key():
    """Verifies an authored `Composite` routes distinct child PRNG frames to repeated calls of one stochastic member."""
    from nodejax import Composite

    def wired():
        def apply(self, x):
            once = self.j(x=x * 2.0)
            twice = self.j(x=x * 2.0)
            return Struct(a=once, b=twice)
        return Composite(j=Jitter())(apply, name='wired')

    node = wired().parameterize(j=Struct(sigma=1.0))
    key = jax.random.PRNGKey(0)
    out1 = node.apply(x=jnp.asarray(3.0), rng=key)
    out2 = node.apply(x=jnp.asarray(3.0), rng=key)
    assert jnp.allclose(out1.a, out2.a)
    assert not jnp.allclose(out1.a, out1.b)
    assert bool(wired().contract.apply_takes_rng)
