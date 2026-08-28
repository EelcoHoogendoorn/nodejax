"""ensemble over a shape-resolved pipe.

Regression: the non-cyclic init path handed the pipe's init walk the
STACKED member params, which broadcast against an unstacked carry. The
(empty) state builds through one member's row, exactly as stack does.
"""
import jax
import jax.numpy as jnp
import pytest

from nodejax import Composite, Leaf, ensemble, nn
from nodejax.struct import Struct


def test_ensemble_of_resolved_pipe():
    X = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
    net = (nn.Linear(8) >> nn.relu >> nn.Linear(1)).with_input(X)

    population = ensemble(net, n=4).parameterize(rng=jax.random.PRNGKey(1))
    assert population.param.linear.w.shape == (4, 4, 8)
    assert population.apply(X).shape == (4, 32, 1)


def test_ensemble_of_composite_with_nonparametric_first_member():
    identity = Leaf(lambda input: input).node
    members = Composite(
        identity=identity,
        linear=nn.Linear(2),
    )

    def apply(self, input):
        return self.linear(self.identity(input))

    member = members(apply)
    population = ensemble(member, n=3).with_input(
        jnp.zeros(4),
    ).parameterize(rng=jax.random.PRNGKey(0))

    assert population.apply(jnp.ones(4)).shape == (3, 2)


def test_the_declared_size_is_the_size():
    """n is the only thing that says how many members there are, so a bound
    param tree of another height is an error rather than a second opinion.

    It used to be a second opinion, and it won: apply counted the param rows
    while init built n states, so an n=3 ensemble bound to four rows ran four
    members against three states, silently."""
    d = ensemble(nn.Linear(2).node, n=3).with_input(jnp.zeros(4))
    three = d.parameterize(rng=jax.random.PRNGKey(0)).param
    assert d.bind(three).apply(jnp.ones(4)).shape == (3, 2)

    four = jax.tree.map(lambda a: jnp.concatenate([a, a[:1]]), three)
    with pytest.raises(TypeError, match='has 4 rows; expected 3'):
        d.bind(four).apply(jnp.ones(4))


def test_ensemble_broadcasts_real_priming_input():
    def init(input):
        return input

    def apply(state, input):
        return state, input

    state = ensemble(Leaf(apply, init=init).node, n=3).bind(()).init(
        input=jnp.array([2.0, 4.0]))
    assert jnp.allclose(state, jnp.array([[2.0, 4.0]] * 3))


def test_ensemble_primes_each_member_from_its_param_row():
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def init(param, input):
        return param.scale * input

    def apply(param, state, input):
        return state, param.scale * input

    member = Leaf(apply, param=param, init=init).node
    model = ensemble(member, n=3).bind(
        Struct(scale=jnp.array([1.0, 2.0, 4.0])))
    assert jnp.allclose(
        model.init(input=3.0), jnp.array([3.0, 6.0, 12.0]))


def test_ensemble_splits_init_rng_while_broadcasting_input():
    def init(input, rng):
        return input + jax.random.normal(rng.next())

    def apply(state, input):
        return state, input

    definition = ensemble(Leaf(apply, init=init).node, n=3)
    assert definition.contract.init_takes_rng
    state = definition.bind(()).init(
        input=1.0, rng=jax.random.PRNGKey(0))
    assert jnp.unique(state).size == 3
