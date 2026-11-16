"""Verifies `@node` generics across partial specialization, wrappers, transforms, composites, and `train_step`."""

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
    """Wraps `body` in a residual `Wrapper` returning `input + body(input)`."""
    wrapped = Wrapper(body=body)

    def apply(self, input):
        return input + self.body(input)

    return wrapped(apply, name='res_g')


def mixed_block(g: Node) -> Node:
    """Wires `g` before a concrete `nn.Linear(3)` inside a `Composite`."""
    members = Composite(g=g, lin=nn.Linear(3))

    def apply(self, input):
        return self.lin(self.g(input))

    return members(apply, name='mixed')


def test_fully_bound_is_a_node():
    """Verifies calling a `@node` factory with all static arguments produces a complete `Node`."""
    model = Gain(scale=2.0).parameterize(factor=3.0)
    assert jnp.allclose(model.apply(jnp.ones(2)), 6.0)


def test_a_partial_call_is_a_generic():
    """Verifies omitting required static arguments produces a generic completed by `specialize`."""
    g = Gain()
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
    """Verifies `Wrapper` defers construction over a generic member until `specialize`."""
    wrapped = residual_of(Gain())
    model = wrapped.specialize(**{'body.scale': 2.0}).parameterize(factor=3.0)
    assert jnp.allclose(model.apply(jnp.ones(2)), 7.0)


def test_transforms_carry_the_generic():
    """Verifies `batch` defers construction over a generic inner node until `specialize`."""
    b = batch(Cell(), n=3)
    model = b.specialize(**{'sample.width': 4}).parameterize()
    state = model.init()
    state, out = model.apply(state, jnp.ones((3, 4)))
    assert out.shape == (3, 4)


def test_depth_two_transforms_carry_the_generic():
    """Verifies nested `batch(scan(...))` carries unbound static arguments across two transforms."""
    tower = batch(scan(Cell()), n=2)
    model = tower.specialize(**{'sample.step.width': 3}).parameterize()
    state = model.init()
    state, ys = model.apply(state, jnp.ones((2, 5, 3)))
    assert ys.shape == (2, 5, 3)


def test_composite_door_holds_a_generic_beside_concrete_members():
    """Verifies `Composite` holds a generic member alongside a concrete member and specializes by path."""
    block = mixed_block(Gain())
    model = block.specialize(g={'scale': 2.0}).with_input(
        jnp.zeros(4)).parameterize(rng=jax.random.PRNGKey(0),
                                   g=Struct(factor=1.0))
    assert model.apply(jnp.ones(4)).shape == (3,)


def test_generic_under_train_step():
    """Verifies a generic node completed via `specialize` binds and trains under `train_step`."""
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
    """Verifies a specialized generic matches a directly constructed node in output and parameter layout."""
    filled = ensemble(Gain(), n=2).specialize(**{'member.scale': 2.0})
    concrete = ensemble(Gain(scale=2.0), n=2)
    a = filled.parameterize(factor=1.0)
    b = concrete.parameterize(factor=1.0)
    x = jnp.ones(3)
    assert jnp.allclose(a.apply(x), b.apply(x))
    assert jax.tree.structure(a.param) == jax.tree.structure(b.param)


def test_respecialize_through_the_stack():
    """Verifies calling `specialize` on an already specialized tree replays construction with the new value."""
    model = batch(Gain(), n=2).specialize(**{'sample.scale': 2.0})
    wider = model.specialize(**{'*.scale': 5.0}).parameterize(factor=1.0)
    assert jnp.allclose(wider.apply(jnp.ones((2, 3))), 5.0)
