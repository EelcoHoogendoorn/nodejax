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

from nodejax import Leaf, Node, Struct, batch, nn, scan
from nodejax.core.binding import (AxisSpec)
from nodejax.control import Integrator
from nodejax.core.spec import element_spec, materialize


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


def test_nested_variable_axes_bind_and_remain_variable() -> None:
    values = jnp.ones((2, 3))
    nested = scan(
        scan(Integrator().with_input(jnp.zeros(()))),
    ).with_input(values)

    outer = nested.contract.input_spec.input
    assert type(outer) is AxisSpec
    assert outer.fixed is False
    assert outer.count is None
    assert type(outer.element) is AxisSpec
    assert outer.element.fixed is False
    assert outer.element.count is None

    model = nested.parameterize().initialize(input=values)
    output = model.apply(values)[1]
    assert output.shape == (2, 3)

    rebound = nested.with_input(jnp.ones((5, 7)))
    assert rebound.contract.input_spec.input.count is None
    assert rebound.contract.input_spec.input.element.count is None


def test_variable_axis_binds_a_fixed_axis_beneath_it() -> None:
    values = jnp.ones((2, 3))
    nested = scan(
        batch(Integrator().with_input(jnp.zeros(()))),
    ).with_input(values)
    outer = nested.contract.input_spec.input

    assert outer.fixed is False
    assert outer.count is None
    assert outer.element.fixed is True
    assert outer.element.count == 3

    model = nested.parameterize().initialize(input=values)
    assert model.state.shape == (3,)
    assert model.apply(values)[1].shape == (2, 3)

    nested.with_input(jnp.ones((5, 3)))
    with pytest.raises(TypeError, match='axis of 4.*declared count 3'):
        nested.with_input(jnp.ones((5, 4)))


def test_nested_axis_element_conflict_is_named() -> None:
    nested = scan(scan(Integrator().with_input(jnp.zeros(()))))
    with pytest.raises(TypeError, match='element'):
        nested.with_input(jnp.ones((2, 3, 4)))


def test_scan_can_declare_and_enforce_a_fixed_extent() -> None:
    fixed = scan(
        Integrator().with_input(jnp.zeros(())),
        n=3,
    )
    declared = fixed.contract.input_spec.input
    assert declared.fixed is True
    assert declared.count == 3
    assert materialize(fixed.contract.input_spec).input.shape == (3,)

    nested = scan(fixed).with_input(jnp.ones((2, 3)))
    assert nested.contract.input_spec.input.element.count == 3
    with pytest.raises(TypeError, match='axis of 4.*declared count 3'):
        nested.with_input(jnp.ones((2, 4)))

    values = jnp.ones(3)
    model = fixed.parameterize().initialize(input=values)
    output = model.apply(values)[1]
    assert output.shape == (3,)

    with pytest.raises(TypeError, match='axis of 4.*declared count 3'):
        fixed.with_input(jnp.ones(4))

    deferred = scan(Integrator(), n=3)
    with pytest.raises(TypeError, match='axis of 4.*declared count 3'):
        deferred.with_input(jnp.ones(4))
    deferred = deferred.with_input(values)
    assert deferred.contract.input_spec.input.count == 3
    deferred = deferred.parameterize().initialize(input=values)
    with pytest.raises(TypeError, match='expected n=3'):
        deferred.apply(jnp.ones(4))


def test_nested_fixed_scans_bind_a_deferred_multi_field_call() -> None:
    def Step() -> Node:
        def apply(state, left, right):
            successor = state + left + right
            return successor, successor

        return Leaf(apply, init=lambda: jnp.zeros(())).node

    values = Struct(
        left=jnp.ones((2, 3)),
        right=2.0 * jnp.ones((2, 3)),
    )
    nested = scan(scan(Step(), n=3), n=2).with_input(bundle=values)

    left = nested.contract.input_spec.left
    right = nested.contract.input_spec.right
    assert left.count == right.count == 2
    assert left.element.count == right.element.count == 3

    model = nested.parameterize().initialize(input=values)
    assert model.apply(bundle=values)[1].shape == (2, 3)


@pytest.mark.parametrize('count', (0, -1, 2.0))
def test_scan_fixed_extent_must_be_a_positive_int(
    count: int | float,
) -> None:
    with pytest.raises(TypeError, match='positive int'):
        scan(Integrator(), n=count)


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
