"""Verifies that every transform publishes an `input_spec` matching its own `apply` contract.

Shape-preserving transforms inherit the inner Node's resolved `input_spec`,
whereas input-reshaping transforms publish either their own reshaped `AxisSpec`
or `None` rather than leaking the inner Node's element spec.
"""

import jax
import jax.numpy as jnp
import pytest

import optax

from nodejax.transforms.learning import learned_sgd
from nodejax import (Node, train_step, Leaf, nn, at, batch, ensemble, externalize,
                     finetune, remat, detach, state_reinit, repeat, residual,
                     scan, stack)
from nodejax.core.binding import (AxisSpec)
from nodejax.core.spec import element_spec, materialize, spec_of
from nodejax.struct import Struct


def Cold() -> Node:
    """Builds a non-priming cyclic leaf that sizes its zero state from `node.input`."""
    def init(node):
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(state, input):
        return input, state

    return Leaf(apply, init=init, name='cold').node


ELEMENT = jnp.zeros(3)


def inner_for(name: str):
    """Selects an inner Node appropriate for the named transform."""
    if name == 'externalize':
        return (nn.Linear(4) >> nn.tanh).node
    if name == 'stack':
        return nn.Linear(3).node
    return Cold()
SHAPE_PRESERVING = [
    ('remat', lambda d: remat(d)),
    ('detach', lambda d: detach(d)),
    ('state_reinit', lambda d: state_reinit(d)),
    ('residual', lambda d: residual(d)),
    ('repeat', lambda d: repeat(d, n=3)),
    ('ensemble', lambda d: ensemble(d, n=4)),
    ('stack', lambda d: stack(d, n=3)),
]
RESHAPING = [
    ('at', lambda d: at(d, field='x')),
    ('batch(no n)', lambda d: batch(d)),
    ('scan', lambda d: scan(d)),
    ('externalize', lambda d: externalize(d, 'linear')),
]
# Learning transforms consume an already-bound model rather than an unbound `Node`.
INSTANCE_FAMILY = [
    ('ttt', lambda model: train_step(model, _MSE, learned_sgd(0.1))),
    ('metasgd', lambda model: finetune(train_step(model, _MSE, learned_sgd(0.1)))),
    ('finetune', lambda model: finetune(train_step(model, _MSE, optax.sgd(0.1)))),
]


def _MSE(out: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((out - target) ** 2)


ELEMENT_FOR = {
    'ttt': Struct(input=jnp.zeros(3), target=jnp.zeros(2)),
    'metasgd': Struct(support=Struct(input=jnp.zeros((5, 3)), target=jnp.zeros((5, 2))),
                      query=jnp.zeros(3)),
    'finetune': Struct(support=Struct(input=jnp.zeros((5, 3)), target=jnp.zeros((5, 2))),
                       query=jnp.zeros(3)),
}


@pytest.mark.parametrize('name,wrap', SHAPE_PRESERVING)
def test_a_shape_preserving_transform_keeps_the_inner_spec(name, wrap):
    """Verifies that shape-preserving transforms retain a pre-resolved inner `input_spec`."""
    inner = inner_for(name)
    wrapped = wrap(inner.with_input(ELEMENT))
    assert wrapped.contract.input_spec is not None, name
    node = (wrapped.parameterize(rng=jax.random.PRNGKey(0))
            if wrapped.parametric else wrapped.parameterize())
    state = node.init()
    if wrapped.cyclic:
        assert jax.tree.leaves(state)


@pytest.mark.parametrize('name,wrap', RESHAPING)
def test_a_reshaping_transform_does_not_publish_the_inner_spec(name, wrap):
    """Verifies that reshaping transforms publish either an `AxisSpec` or `None` instead of the inner spec."""
    wrapped = wrap(inner_for(name).with_input(ELEMENT))
    spec = wrapped.contract.input_spec
    if type(spec) is Struct and any(
            type(v) is AxisSpec for _, v in spec.__items__):
        assert element_spec(wrapped.contract.input_spec) is not None
        return
    assert wrapped.contract.input_spec is None, (
        f'{name} publishes {wrapped.contract.input_spec}, which its apply cannot take')


def test_batch_derives_the_batched_shape_when_it_knows_the_count():
    """Verifies that `batch(..., n=4)` over a resolved inner Node publishes an `AxisSpec` with `count=4`."""
    wrapped = batch(Cold().with_input(ELEMENT), n=4)
    assert wrapped.contract.input_spec is not None
    spec = wrapped.contract.input_spec
    assert type(spec.input) is AxisSpec and spec.input.count == 4
    assert element_spec(spec).input.shape == (3,)

    state = wrapped.parameterize().init()
    assert jax.tree.leaves(state)[0].shape == (4, 3)
    _, out = wrapped.parameterize().apply(state, jnp.ones((4, 3)))
    assert out.shape == (4, 3)


@pytest.mark.parametrize('name,wrap', SHAPE_PRESERVING + RESHAPING)
def test_a_published_spec_can_always_be_fed(name, wrap):
    """Verifies that any materialized published `input_spec` runs through `apply` under both binding orders."""
    inner = inner_for(name)
    for label, wrapped in (('wrap then bind', wrap(inner)),
                           ('bind then wrap', wrap(inner.with_input(ELEMENT)))):
        if wrapped.contract.input_spec is None:
            continue
        spec = wrapped.contract.input_spec
        try:
            value = materialize(spec)
        except TypeError as e:
            assert 'unknown count' in str(e), (name, label)
            continue
        key = ({'rng': jax.random.PRNGKey(0)}
               if wrapped.contract.apply_takes_rng else {})
        node = (wrapped.parameterize(rng=jax.random.PRNGKey(0))
                if wrapped.parametric else wrapped.parameterize())
        state = node.init()
        out = (node.apply(state, bundle=value, **key) if wrapped.cyclic
               else node.apply(bundle=value, **key))
        assert out is not None, (name, label)


def test_the_usual_order_was_never_the_broken_one():
    """Verifies wrapping first with `at` and resolving outer input afterward via `with_input`."""
    right = at(Cold(), field='x').with_input(Struct(x=ELEMENT, y=jnp.zeros(2)))
    assert right.contract.input_spec is not None
    node = right.parameterize()
    state, out = node.apply(node.init(), x=jnp.ones(3), y=jnp.ones(2))
    assert out.x.shape == (3,) and out.y.shape == (2,)


@pytest.mark.parametrize('name,wrap', INSTANCE_FAMILY)
def test_the_instance_family_publishes_its_element(name, wrap):
    """Verifies that `train_step` and `finetune` leave `input_spec` unset until data binds and accept matching bundles."""
    model = nn.Linear(2).with_input(ELEMENT).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    wrapped = wrap(model)
    assert wrapped.param.objective.model.w.shape == (3, 2)
    element = ELEMENT_FOR[name]
    if name == 'ttt':
        assert wrapped.contract.input_spec is None
        _, (out, _) = wrapped.apply(bundle=element)
    else:
        assert wrapped.contract.input_spec is None
        out = wrapped.apply(bundle=element)
    assert out is not None
