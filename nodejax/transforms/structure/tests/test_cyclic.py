"""cyclic: a step's first field promoted to the state of a system."""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Leaf, Node, PNode, PSNode, batch, cyclic, node, scan
from nodejax.struct import Struct


@node
def Kick() -> Node:
    """The step: the state plus a push."""
    def apply(x, push):
        return x + push

    return Leaf(apply)


@node
def Geared() -> Node:
    """A parametric step: the state plus a gain times the push."""
    def param(gain):
        return jnp.asarray(gain)

    def apply(param, x, push):
        return x + param * push

    return Leaf(apply, param=param)


def test_the_first_field_is_the_state_and_the_rest_are_the_call():
    world = cyclic(Kick()).parameterize()
    assert world.cyclic and world.contract.apply_fields == ('push',)

    started = world.initialize(x=jnp.asarray(1.0))
    assert started.state == 1.0
    started, out = started.apply(push=jnp.asarray(2.0))
    assert out == 3.0 and started.state == 3.0

    _, out = world.bind(state=jnp.asarray(10.0)).apply(push=jnp.asarray(1.0))
    assert out == 11.0


def test_a_promoted_step_scans_over_its_other_fields():
    """The sequence-driven run: the state carried, the pushes read in turn."""
    run = scan(cyclic(Kick()), n=3).parameterize().bind(state=jnp.asarray(0.0))
    run, outputs = run.apply(push=jnp.asarray([1.0, 2.0, 3.0]))
    assert jnp.allclose(outputs, jnp.asarray([1.0, 3.0, 6.0]))
    assert run.state == 6.0


def test_the_parameters_stay_the_steps():
    world = cyclic(Geared()).parameterize(gain=2.0)
    assert world.param == 2.0
    assert world.step.param == 2.0
    started, out = world.initialize(x=jnp.asarray(1.0)).apply(push=jnp.asarray(1.0))
    assert out == 3.0
    assert type(started) is PSNode and type(world) is PNode


def test_a_record_state_is_the_record():
    @node
    def Drift() -> Node:
        def apply(body, dt):
            return body.replace(position=body.position + dt * body.velocity)

        return Leaf(apply)

    body = Struct(position=jnp.zeros(2), velocity=jnp.ones(2))
    world = cyclic(Drift()).parameterize().bind(state=body)
    assert jnp.allclose(world.state.velocity, 1.0)
    world, out = world.apply(dt=jnp.asarray(0.5))
    assert jnp.allclose(out.position, 0.5)
    assert jnp.allclose(world.state.position, 0.5)


def test_a_single_field_step_ticks_on_its_own():
    doubling = cyclic(Leaf(lambda x: 2.0 * x, name='doubling')).parameterize()
    started = doubling.initialize(x=jnp.asarray(1.0))
    started, out = started.apply()
    assert out == 2.0 and started.state == 2.0


def test_batched_promoted_steps_carry_a_state_per_element():
    worlds = batch(cyclic(Kick()), n=3).parameterize().bind(state=jnp.arange(3.0))
    worlds, out = worlds.apply(push=jnp.ones(3))
    assert jnp.allclose(out, jnp.asarray([1.0, 2.0, 3.0]))
    assert jnp.allclose(worlds.state, out)


def test_a_self_ticking_system_scans_for_a_declared_length():
    """The time axis of a system with no inputs: scan with n, no sequence."""
    doubling = cyclic(Leaf(lambda x: 2.0 * x, name='doubling'))
    run = scan(doubling, n=3).parameterize().bind(state=jnp.asarray(1.0))
    run, trajectory = run.apply()
    assert jnp.allclose(trajectory, jnp.asarray([2.0, 4.0, 8.0]))
    assert run.state == 8.0


def test_an_already_cyclic_node_is_rejected_at_construction() -> None:
    world = cyclic(Kick()).node

    with pytest.raises(TypeError, match='requires an acyclic step.*already cyclic'):
        cyclic(world)
