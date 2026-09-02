"""A node's published input spec is a promise, and a transform can break it.

Transparent transform construction inherits the inner's input evidence, which is
right for the many transforms that leave the signal's shape alone and wrong
for the few that do not. A transform that feeds its inner something OTHER than
what it was handed has an input shape of its own, and inheriting publishes a
shape its own apply cannot accept.

Nothing catches that until someone wraps an ALREADY-RESOLVED def, because an
unresolved inner has nothing to inherit and the usual order (wrap, then
with_input) overwrites the inherited value anyway. Both orders are legal, so
both are tested here.

The rule, and the only thing these tests check: if a node says it is resolved,
feeding it a value of that shape must work. An axis spec (core.AxisSpec)
claims only its ELEMENT; while its count is unknown it refuses to
materialize, loudly, instead of naming a shape it cannot know.
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
    """A leaf that reads its SHAPE at init and primes from nothing, so it is
    legal on a shape-only walk and it notices what it was told it would see."""
    def init(node):
        return jax.tree.map(jnp.zeros_like, node.input)

    def apply(state, input):
        return input, state

    return Leaf(apply, init=init, name='cold').node


ELEMENT = jnp.zeros(3)


def inner_for(name: str):
    """Most transforms take the shape-reading leaf. Two need params of their
    own: externalize projects a member out of a pipe, and stack reads its
    depth off a stacked param tree, so neither has anything to do with a
    nonparametric node."""
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
    ('ensemble', lambda d: ensemble(d, n=4)),      # broadcasts: input_axis=None
    ('stack', lambda d: stack(d, n=3)),            # depth, not a new axis
]
RESHAPING = [
    ('at', lambda d: at(d, field='x')),            # inner gets ONE field
    ('batch(no n)', lambda d: batch(d)),           # a new leading axis, size unknown
    ('scan', lambda d: scan(d)),                   # a new leading axis, length unknown
    ('externalize', lambda d: externalize(d, 'linear')),
]
# the inner-loop family reshapes too (an episode or a sequence element in,
# where the model takes one sample), but it consumes a BOUND model, so it
# has its own test with the one construction order that exists for it
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
    """These feed their inner exactly what they were handed, so inheriting is
    not merely safe, it is the useful answer: wrapping a resolved node leaves
    it resolved and it still inits with no further help."""
    inner = inner_for(name)
    wrapped = wrap(inner.with_input(ELEMENT))
    assert wrapped.contract.input_spec is not None, name
    node = (wrapped.parameterize(rng=jax.random.PRNGKey(0))
            if wrapped.parametric else wrapped.parameterize())
    state = node.init()                    # no further with_input needed
    if wrapped.cyclic:                     # a stateless wrap inits to (), and does
        assert jax.tree.leaves(state)


@pytest.mark.parametrize('name,wrap', RESHAPING)
def test_a_reshaping_transform_does_not_publish_the_inner_spec(name, wrap):
    """The bug this file exists for. `at` and `externalize` inherited a spec
    describing what their INNER eats, and both then failed inside apply with
    'ShapeDtypeStruct is not subscriptable' rather than at the call. A shape
    that cannot be fed must not be published."""
    wrapped = wrap(inner_for(name).with_input(ELEMENT))
    spec = wrapped.contract.input_spec
    if type(spec) is Struct and any(
            type(v) is AxisSpec for _, v in spec.__items__):
        # the reshaping is DECLARED: the inner's element under the map is
        # exactly not the inner's spec republished as this node's own
        assert element_spec(wrapped.contract.input_spec) is not None
        return
    assert wrapped.contract.input_spec is None, (
        f'{name} publishes {wrapped.contract.input_spec}, which its apply cannot take')


def test_batch_derives_the_batched_shape_when_it_knows_the_count():
    """Not saying the wrong thing leaves the option of saying the right one.
    batch reshapes by adding an axis, and with n it knows exactly which."""
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
    """The invariant, stated once over every transform above: `resolved` is a
    claim about what apply accepts, so a resolved node must accept a value of
    the shape it named. Neither order of construction may break it."""
    inner = inner_for(name)
    for label, wrapped in (('wrap then bind', wrap(inner)),
                           ('bind then wrap', wrap(inner.with_input(ELEMENT)))):
        if wrapped.contract.input_spec is None:
            continue
        spec = wrapped.contract.input_spec
        try:
            value = materialize(spec)      # the spec IS data now
        except TypeError as e:             # an unknown count refuses, loudly
            assert 'unknown count' in str(e), (name, label)
            continue
        key = ({'rng': jax.random.PRNGKey(0)}
               if wrapped.contract.apply_takes_rng else {})       # a key is real or absent
        node = (wrapped.parameterize(rng=jax.random.PRNGKey(0))
                if wrapped.parametric else wrapped.parameterize())
        state = node.init()
        out = (node.apply(state, bundle=value, **key) if wrapped.cyclic
               else node.apply(bundle=value, **key))
        assert out is not None, (name, label)


def test_the_usual_order_was_never_the_broken_one():
    """Why the suite stayed green through both bugs: wrapping FIRST and
    binding after overwrites whatever was inherited, and that is the order
    every call site in the library happens to use."""
    right = at(Cold(), field='x').with_input(Struct(x=ELEMENT, y=jnp.zeros(2)))
    assert right.contract.input_spec is not None
    node = right.parameterize()
    state, out = node.apply(node.init(), x=jnp.ones(3), y=jnp.ones(2))
    assert out.x.shape == (3,) and out.y.shape == (2,)


@pytest.mark.parametrize('name,wrap', INSTANCE_FAMILY)
def test_the_instance_family_publishes_its_element(name, wrap):
    """The inner-loop family consumes a BOUND model, so exactly one
    construction order exists: the ladder's, the model resolved by its
    own real input before the family sees it (the param-time projection
    walks this test once covered are deleted with that ruling). What
    remains to hold: the product's def publishes the element its OWN
    apply takes (the trainer a declared (input, target) pair whose
    shapes only data supplies; the episode node a support axis beside
    a query), and a value of that shape feeds it."""
    model = nn.Linear(2).with_input(ELEMENT).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    wrapped = wrap(model)
    assert wrapped.param.objective.model.w.shape == (3, 2)
    element = ELEMENT_FOR[name]
    if name == 'ttt':
        assert wrapped.contract.input_spec is None  # declared fields, data-sized
        _, (out, _) = wrapped.apply(bundle=element)
    else:
        assert wrapped.contract.input_spec is None  # episode shapes come from data
        out = wrapped.apply(bundle=element)
    assert out is not None
