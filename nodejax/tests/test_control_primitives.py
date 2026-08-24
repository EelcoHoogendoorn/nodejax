"""Behavior and transform coverage for stock control primitives."""

import jax
import jax.numpy as jnp
import pytest

from nodejax import PNode, PyTree, scanned
from nodejax.control import (
    Deadband,
    FirstOrder,
    MovingAverage,
    Quantize,
    StateSpace,
)
from nodejax.struct import Struct


def test_quantize_and_deadband_compose_over_pytrees() -> None:
    input = Struct(
        primary=jnp.array([-0.74, -0.2, 0.1, 0.76]),
        secondary=jnp.array([1.24]),
    )
    pipeline = Deadband()(threshold=0.25) >> Quantize(0.5)

    output = jax.jit(pipeline.apply)(input)

    assert jnp.allclose(output.primary, jnp.array([-0.5, 0.0, 0.0, 0.5]))
    assert jnp.allclose(output.secondary, jnp.array([1.0]))


@pytest.mark.parametrize('resolution', [0.0, -0.1])
def test_quantize_requires_positive_resolution(resolution: float) -> None:
    with pytest.raises(ValueError, match='resolution must be positive'):
        Quantize(resolution)


@pytest.mark.parametrize('window', [0, -1, 1.5, True])
def test_moving_average_requires_positive_integer_window(
        window: int | float | bool) -> None:
    with pytest.raises(ValueError, match='window must be a positive integer'):
        MovingAverage(window)


def test_moving_average_cold_and_warm_initialization() -> None:
    inputs = jnp.array([3.0, 6.0, 9.0])
    moving = MovingAverage(3)
    cold = moving.with_input(0.0).bind(moving.param).initialize()

    cold, output = cold.scan(inputs)

    assert jnp.allclose(output, jnp.array([1.0, 3.0, 6.0]))
    assert jnp.allclose(cold.state, inputs)

    warm = MovingAverage(3, warm=True).initialize(input=2.0)
    warm, output = warm(5.0)

    assert jnp.allclose(output, 3.0)
    assert jnp.allclose(warm.state, jnp.array([2.0, 2.0, 5.0]))


def test_first_order_response_and_warm_start() -> None:
    inputs = jnp.ones(3)
    plant = FirstOrder(0.5)(tau=0.5, gain=2.0)
    cold = plant.with_input(0.0).bind(plant.param).initialize()

    cold, output = cold.scan(inputs)

    assert jnp.allclose(output, jnp.array([1.0, 1.5, 1.75]))
    assert jnp.allclose(cold.state, 1.75)

    warm = FirstOrder(0.5, warm=True)(tau=0.5, gain=2.0).initialize(input=3.0)
    assert jnp.allclose(warm.state, 6.0)


def test_state_space_follows_discrete_equations() -> None:
    system = StateSpace()(
        A=jnp.array([[1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
    ).initialize()
    inputs = jnp.array([[1.0], [2.0], [3.0]])

    system, output = system.scan(inputs)

    assert jnp.allclose(output[:, 0], jnp.array([0.0, 1.0, 3.0]))
    assert jnp.allclose(system.state, jnp.array([6.0]))


def test_state_space_direct_term_scans_and_differentiates() -> None:
    model = StateSpace()(
        A=jnp.array([[0.5]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[2.0]]),
        D=jnp.array([[0.25]]),
    )
    inputs = jnp.ones((4, 1))

    output = scanned(model).apply(inputs)
    assert jnp.allclose(output[:, 0], jnp.array([0.25, 2.25, 3.25, 3.75]))

    def loss(candidate: PNode) -> jax.Array:
        return jnp.sum(scanned(candidate).apply(inputs))

    gradients = jax.grad(loss)(model)
    assert all(jnp.all(jnp.isfinite(value))
               for value in jax.tree.leaves(gradients.param))


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('A', jnp.ones((2, 3)), 'A must be a square matrix'),
        ('B', jnp.ones((3, 1)), 'B must have one row per state'),
        ('C', jnp.ones((1, 3)), 'C must have one column per state'),
        ('D', jnp.ones((2, 1)), 'D must have shape'),
    ],
)
def test_state_space_validates_matrix_shapes(
        field: str, value: PyTree, message: str) -> None:
    matrices: dict[str, PyTree] = {
        'A': jnp.eye(2),
        'B': jnp.ones((2, 1)),
        'C': jnp.ones((1, 2)),
        'D': jnp.zeros((1, 1)),
    }
    matrices[field] = value

    with pytest.raises(ValueError, match=message):
        StateSpace()(**matrices)
