"""Inside an authored apply, ``self.member`` is the member's view as of the
read, with a bound view's verbs. ``self`` is the composite's state, the one
mutable thing in the step: calls, binds, and resets through a member view
store their result in the member's slot; the view itself does not move.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Aux, Composite, Leaf, Node, Wrapper, node
from nodejax.struct import Struct


def Counter() -> Node:
    """State counts up by the input; the output is the count before."""
    return Leaf(
        lambda state, input: (state + input, state),
        init=lambda: jnp.zeros(()),
        name='counter',
    ).node


def held(apply) -> Node:
    return Composite(count=Counter())(apply).parameterize().initialize()


def test_a_member_view_is_the_slot_as_of_the_read():
    def apply(self, input):
        before = self.count
        self.count(input)
        after = self.count
        return Struct(before=before.state, after=after.state, live=self.count.state)

    program, out = held(apply).apply(jnp.asarray(2.0))
    assert out.before == 0.0 and out.after == 2.0 and out.live == 2.0
    assert program.state.count == 2.0


def test_repeated_calls_through_self_chain():
    def apply(self, input):
        self.count(input)
        return self.count(input)

    program, out = held(apply).apply(jnp.asarray(1.0))
    assert out == 1.0
    assert program.state.count == 2.0


def test_a_call_through_a_kept_view_runs_from_its_state():
    def apply(self, input):
        before = self.count
        self.count(input)
        return before(input)

    program, out = held(apply).apply(jnp.asarray(1.0))
    assert out == 0.0
    assert program.state.count == 1.0


def test_bind_stores_the_state_and_returns_the_rebound_view():
    def apply(self, input):
        rebound = self.count.bind(state=jnp.asarray(10.0))
        return Struct(bound=rebound.state, stored=self.count.state, out=self.count(input))

    program, out = held(apply).apply(jnp.asarray(1.0))
    assert out.bound == 10.0 and out.stored == 10.0 and out.out == 10.0
    assert program.state.count == 11.0


def test_reset_stores_fresh_state():
    def apply(self, input):
        self.count(input)
        self.count.reset()
        return self.count(input)

    program, out = held(apply).apply(jnp.asarray(3.0))
    assert out == 0.0
    assert program.state.count == 3.0


def test_scan_through_a_view_runs_the_member_over_a_sequence():
    def apply(self, input):
        return self.count.scan(input)

    program, out = held(apply).apply(jnp.ones(3))
    assert jnp.allclose(out, jnp.asarray([0.0, 1.0, 2.0]))
    assert program.state.count == 3.0


def test_param_reads_the_member_parameters():
    from nodejax import nn

    def apply(self, input):
        # The construction walk learns the layer's shape from the call, so the
        # parameters are read after it.
        out = self.linear(input)
        return Struct(weight=self.linear.param.w, out=out)

    features, width = 3, 2
    program = Composite(linear=nn.Linear(width))(apply).with_input(
        jnp.zeros(features)).parameterize(rng=jax.random.PRNGKey(0))
    out = program.apply(jnp.ones(features))
    assert out.weight.shape == (features, width)
    assert jnp.allclose(out.weight, program.param.linear.w)


def test_binding_a_member_of_a_member_writes_through_the_owner():
    @node
    def Pair() -> Node:
        members = Composite(count=Counter())

        def apply(self, input):
            return self.count(input)

        return members(apply)

    def apply(self, input):
        self.pair.count.bind(state=jnp.asarray(5.0))
        return self.pair(input)

    program = Composite(pair=Pair())(apply).parameterize().initialize()
    program, out = program.apply(jnp.asarray(1.0))
    assert out == 5.0
    assert program.state.pair.count == 6.0


def test_a_transparent_wrapper_member_shares_the_wrapper_slot():
    def apply(self, input):
        return self.count(input)

    doubled = Wrapper(count=Counter())(apply)

    def outer(self, input):
        self.doubled.count.bind(state=jnp.asarray(4.0))
        return self.doubled(input)

    program = Composite(doubled=doubled)(outer).parameterize().initialize()
    program, out = program.apply(jnp.asarray(1.0))
    assert out == 4.0
    assert program.state.doubled == 5.0


def test_binds_and_scans_survive_the_construction_walks():
    """Parameterization and initialization run the apply to discover
    shapes and states; the view verbs must not disturb either walk."""
    from nodejax import nn

    def apply(self, start, sequence):
        return self.memory.bind(state=start).scan(sequence)

    features, hidden, steps = 3, 4, 5
    program = Composite(memory=nn.GRU(hidden))(apply).with_input(bundle=Struct(
        start=jnp.zeros(hidden), sequence=jnp.zeros((steps, features)),
    )).parameterize(rng=jax.random.PRNGKey(0)).initialize()
    start = jnp.ones(hidden)
    program, out = program.apply(start=start, sequence=jnp.zeros((steps, features)))
    assert out.shape == (steps, hidden)
    reference = nn.GRU(hidden).with_input(jnp.zeros(features)).bind(
        program.param.memory, state=start).scan(jnp.zeros((steps, features)))[1]
    assert jnp.allclose(out, reference)


def switched(apply) -> Node:
    return Composite(a=Counter(), b=Counter())(apply).parameterize().initialize()


def test_a_member_stepped_inside_a_jax_transformation_is_refused():
    """Both branches of a lax.cond are traced, and each returns only its
    output: a successor stored from inside one would be another trace's
    value, stored whichever branch runs."""
    def apply(self, flag, input):
        return jax.lax.cond(flag, lambda: self.a(input), lambda: self.b(input))

    with pytest.raises(TypeError, match="member 'a' stored its state inside a JAX transformation"):
        switched(apply).apply(flag=jnp.asarray(True), input=jnp.asarray(1.0))


def test_a_successor_returned_from_a_transformation_and_bound_is_stored():
    def apply(self, flag, input):
        a, b = self.a, self.b

        def left():
            out = a(input)
            return self.a.state, b.state, out

        def right():
            out = b(input)
            return a.state, self.b.state, out

        state_a, state_b, out = jax.lax.cond(flag, left, right)
        self.a.bind(state=state_a)
        self.b.bind(state=state_b)
        return out

    step = jax.jit(lambda program, flag, input: program(flag=flag, input=input))
    program, _ = step(switched(apply), jnp.asarray(True), jnp.asarray(2.0))
    program, _ = step(program, jnp.asarray(False), jnp.asarray(3.0))
    assert program.state.a == 2.0 and program.state.b == 3.0


def test_a_stateless_member_may_run_inside_a_jax_transformation():
    def apply(self, flag, input):
        return jax.lax.cond(flag, lambda: self.double(input), lambda: input)

    double = Leaf(lambda input: 2.0 * input, name='double')
    program = Composite(double=double)(apply).parameterize()
    assert program.apply(flag=jnp.asarray(True), input=jnp.asarray(3.0)) == 6.0
    assert program.apply(flag=jnp.asarray(False), input=jnp.asarray(3.0)) == 3.0


def test_aux_produced_inside_a_jax_transformation_is_refused():
    def apply(self, flag, input):
        return jax.lax.cond(flag, lambda: self.probe(input), lambda: input)

    probe = Leaf(lambda input: (input, Aux(seen=input)), name='probe')
    program = Composite(probe=probe)(apply).parameterize()
    with pytest.raises(TypeError, match="aux 'probe' was produced inside a JAX transformation"):
        program.apply(flag=jnp.asarray(False), input=jnp.asarray(1.0))


def test_a_member_primed_inside_a_jax_transformation_is_refused():
    """Initialization runs the apply to prime members from what they are
    fed; a member first fed inside a branch would take a branch's value."""
    def apply(self, flag, input):
        return jax.lax.cond(flag, lambda: self.a(input), lambda: self.b(input))

    primed = Leaf(lambda state, input: (state + input, state),
                  init=lambda input: jnp.zeros_like(input), name='primed')
    program = Composite(a=primed, b=primed)(apply).parameterize()
    with pytest.raises(TypeError, match="member 'a' took its initial state inside"):
        program.initialize(input=Struct(flag=jnp.asarray(True), input=jnp.asarray(1.0)))
