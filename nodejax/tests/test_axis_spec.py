"""AxisSpec: the element under the map survives an unknown count.

A mapping transform reshapes its input, so it cannot publish its inner's
spec as its own. It used to publish the stacked shape when it knew the
count and nothing at all when it did not, erasing a fully known element.
Now it publishes the element under the map: params build against it,
bindings validate against it and adopt the count, and only tiling ever
demands a number.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Node, batch, nn
from nodejax.binding import (AxisSpec)
from nodejax.control import Integrator
from nodejax.spec import element_spec, materialize


def net() -> Node:
    return (nn.Linear(5) >> nn.gelu).with_input(jnp.zeros(4))


def test_batch_publishes_the_element():
    b = batch(net())
    spec = b.contract.input_spec
    assert type(spec.input) is AxisSpec          # the map lives per field
    assert spec.input.count is None
    assert element_spec(spec).input.shape == (4,)


def test_params_build_before_any_batch_binds():
    """The capability the erasure used to cost: params never depend on
    the count, so an element-resolved batch parameterizes without a
    batch in sight."""
    node = batch(net()).parameterize(rng=jax.random.PRNGKey(0))
    assert node.param.linear.w.shape == (4, 5)
    out = node.apply(jnp.ones((8, 4)))
    assert out.shape == (8, 5)


def test_binding_adopts_the_count():
    b = batch(Integrator().with_input(jnp.zeros(()))).with_input(jnp.zeros(3))
    spec = b.contract.input_spec
    assert type(spec.input) is AxisSpec and spec.input.count == 3
    node = b.parameterize()
    state = node.init()
    assert state.shape == (3,)                 # tiled from the adopted count


def test_axis_declarations_support_functional_replacement():
    declared = AxisSpec(jax.ShapeDtypeStruct((4,), jnp.float32))
    replaced = declared.replace(count=3)

    assert declared.count is None
    assert replaced.count == 3
    assert replaced is not declared


def test_axis_declarations_are_immutable_and_validate_metadata():
    declared = AxisSpec(jax.ShapeDtypeStruct((4,), jnp.float32))

    with pytest.raises(AttributeError):
        declared.count = 3
    with pytest.raises(TypeError, match='int or None'):
        AxisSpec(declared.element, count=3.0)
    with pytest.raises(ValueError, match='negative'):
        AxisSpec(declared.element, count=-1)
    with pytest.raises(TypeError, match='bool'):
        AxisSpec(declared.element, fixed=1)

    assert AxisSpec(declared.element, count=0).count == 0


def test_scan_extent_is_variable_across_calls():
    from nodejax import scanned

    model = scanned(Integrator().with_input(jnp.zeros(()))).parameterize()
    first = model.apply(jnp.ones(3))
    second = model.apply(jnp.ones(7))

    assert first.shape == (3,)
    assert second.shape == (7,)
    assert model.contract.input_spec.input.fixed is False
    assert model.contract.input_spec.input.count is None


def test_wrong_element_fails_named():
    with pytest.raises(TypeError, match='element'):
        batch(net()).with_input(jnp.zeros((8, 3)))


def test_an_unbatched_input_fails_named():
    """The old ndim-sniffing case, principled: a scalar leaf has no axis
    to map."""
    with pytest.raises(TypeError, match='no axis to map|scalar leaf'):
        batch(Integrator().with_input(jnp.zeros(())), n=4).with_input(jnp.zeros(()))


def test_unknown_count_refuses_materialize():
    with pytest.raises(TypeError, match='unknown count'):
        materialize(batch(net()).contract.input_spec)
