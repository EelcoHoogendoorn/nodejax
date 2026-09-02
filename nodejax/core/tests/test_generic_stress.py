"""Generics under stress: the target behavior as tests.

THE SPEC (ruled): generics are first-class, and the definition is
degree of binding. One annotation, @node; one capture mechanism, its
construction record; one spectrum: a @node factory called with ALL
statics bound yields a NODE, called with some unbound it yields a
GENERIC: the same description, its unbound static arguments staying
open. specialize supplies the remainder, and unbound statics ride
through any assembly (doors, pipes, transforms, at any depth): a filled tree behaves
identically to one built concrete. No @generic annotation exists.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax.struct import Struct
from nodejax import (Node, nn, node, Leaf, batch, scan, ensemble, train_step,
                     trained, Composite, Wrapper)

@node
def Gain(scale: float) -> Node:
    def param(factor=1.0):
        return Struct(factor=jnp.asarray(factor * scale))

    def apply(param, input):
        return param.factor * input

    return Leaf(apply, param=param)


@node
def Cell(width: int) -> Node:
    def init(node):
        return jnp.zeros(width)

    def apply(state, input):
        new = jnp.tanh(state + input)
        return new, new

    return Leaf(apply, init=init)


def residual_of(body: Node) -> Node:
    """The wrapper tree, in its own scope: x + body(x)."""
    wrapped = Wrapper(body=body)

    def apply(self, input):
        return input + self.body(input)

    return wrapped(apply, name='res_g')


def mixed_block(g: Node) -> Node:
    """The mixed composite: one member generic, one bound."""
    members = Composite(g=g, lin=nn.Linear(3))

    def apply(self, input):
        return self.lin(self.g(input))

    return members(apply, name='mixed')


def test_fully_bound_is_a_node():
    """The bound end of the spectrum: all statics supplied, an ordinary
    node comes out. The regression anchor for everything below."""
    model = Gain(scale=2.0).parameterize(factor=3.0)
    assert jnp.allclose(model.apply(jnp.ones(2)), 6.0)


def test_a_partial_call_is_a_generic():
    """The unbound end: the same factory, statics missing, and the
    product is a generic that specialize completes."""
    g = Gain()                                   # scale unbound: a generic
    model = g.specialize(scale=2.0).parameterize(factor=3.0)
    assert jnp.allclose(model.apply(jnp.ones(2)), 6.0)


def test_specialization_can_leave_a_generic_partially_bound():
    calls = []

    @node
    def Affine(scale, offset):
        calls.append((scale, offset))
        return Leaf(lambda input: scale * input + offset)

    pending = Affine()
    scaled = pending.specialize(scale=2.0)

    assert type(scaled) is type(pending)
    assert scaled.unbound() == ('offset',)
    assert scaled.statics.scale == 2.0
    assert calls == []

    complete = scaled.specialize(offset=3.0)
    assert calls == [(2.0, 3.0)]
    assert jnp.allclose(complete.parameterize().apply(4.0), 11.0)


def test_generic_must_complete_before_input_binding():
    """Input evidence is a later stage and cannot bind to an open record."""
    input = jnp.zeros(4)
    pending = nn.Linear()
    with pytest.raises(TypeError, match='statics unbound'):
        pending.with_input(input)
    built = pending.specialize(n_out=3).with_input(input)

    assert built.contract.input_spec is not None
    assert built.parameterize(
        rng=jax.random.PRNGKey(0)).param.w.shape == (4, 3)


def test_transformed_generic_binds_input_after_static_completion():
    input = jnp.zeros((5, 4))
    generic = batch(nn.Linear(), n=5)
    built = generic.specialize(
        **{'sample.n_out': 3}).with_input(input)

    assert built.contract.input_spec is not None
    model = built.parameterize(rng=jax.random.PRNGKey(0))
    assert model.apply(input).shape == (5, 3)


def test_wrapper_door_carries_the_generic():
    """The Wrapper door assembles around a generic; the build defers as a
    whole and specialize fills through it."""
    wrapped = residual_of(Gain())
    model = wrapped.specialize(**{'body.scale': 2.0}).parameterize(factor=3.0)
    assert jnp.allclose(model.apply(jnp.ones(2)), 7.0)    # 1 + 6


def test_transforms_carry_the_generic():
    """A door-less transform over a generic defers the same way."""
    b = batch(Cell(), n=3)
    model = b.specialize(**{'sample.width': 4}).parameterize()
    state = model.init()
    state, out = model.apply(state, jnp.ones((3, 4)))
    assert out.shape == (3, 4)


def test_depth_two_transforms_carry_the_generic():
    """batch(scan(generic)): the unbound static rides two wrappers deep."""
    tower = batch(scan(Cell()), n=2)
    model = tower.specialize(**{'sample.step.width': 3}).parameterize()
    state = model.init()
    state, ys = model.apply(state, jnp.ones((2, 5, 3)))
    assert ys.shape == (2, 5, 3)


def test_composite_door_holds_a_generic_beside_concrete_members():
    """Mixed trees: one member generic, the rest bound; its unbound
    static argument is path-addressed at the top."""
    block = mixed_block(Gain())
    model = block.specialize(g={'scale': 2.0}).with_input(
        jnp.zeros(4)).parameterize(rng=jax.random.PRNGKey(0),
                                   g=Struct(factor=1.0))
    assert model.apply(jnp.ones(4)).shape == (3,)


def test_generic_under_train_step():
    """specialize binds the static FIRST, then the ladder binds, then the
    trainer trains: later rungs refuse to bind on a generic, and
    train_step consumes the model fully bound."""
    model = Gain().specialize(scale=1.0).parameterize(factor=0.0).initialize()
    trainer = train_step(
        model,
        lambda prediction, target: jnp.mean((prediction - target) ** 2),
        optax.sgd(0.1),
    )
    done, aux = trained(trainer).apply(input=jnp.ones((20, 2)),
                                       target=jnp.full((20, 2), 3.0))
    assert aux.loss[-1] < aux.loss[0]


def test_filled_equals_concrete():
    """The law: a filled tree is indistinguishable from one built
    concrete, in outputs and in param layout alike."""
    filled = ensemble(Gain(), n=2).specialize(**{'member.scale': 2.0})
    concrete = ensemble(Gain(scale=2.0), n=2)
    a = filled.parameterize(factor=1.0)
    b = concrete.parameterize(factor=1.0)
    x = jnp.ones(3)
    assert jnp.allclose(a.apply(x), b.apply(x))
    assert jax.tree.structure(a.param) == jax.tree.structure(b.param)


def test_respecialize_through_the_stack():
    """specialize on the FILLED tree re-binds the static through the
    transform stack: the record re-runs top down."""
    model = batch(Gain(), n=2).specialize(**{'sample.scale': 2.0})
    wider = model.specialize(**{'*.scale': 5.0}).parameterize(factor=1.0)
    assert jnp.allclose(wider.apply(jnp.ones((2, 3))), 5.0)
