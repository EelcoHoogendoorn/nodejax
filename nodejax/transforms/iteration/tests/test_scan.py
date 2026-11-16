"""Verifies sequence transforms (`scan`, `scanned`, and `carried`) over cyclic and acyclic steps."""

import jax
import jax.numpy as jnp
import pytest

from nodejax import (Node, Aux, Leaf, Composite, drop_aux, map_members,
                     scan, scanned, state_reinit)
from nodejax.struct import Struct
from nodejax.control import Integrator


def test_scan_transform():
    """Verifies that `scanned` internalizes step state and produces an acyclic sequence Node."""
    seq = scanned(Integrator())
    assert type(seq) is Node and not seq.cyclic
    node = seq.parameterize()
    outs = node.apply(jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(outs, jnp.array([1.0, 3.0, 6.0]))


def test_scanned_maps_an_acyclic_stochastic_step():
    """Verifies that `scanned` maps an acyclic stochastic step across a sequence with per-step PRNG draws."""
    def Draw():
        def apply(input, rng):
            return input + jax.random.normal(rng.next())

        return Leaf(apply, name='draw')

    sequence = scanned(Draw())
    assert not sequence.cyclic
    first = sequence.apply(
        jnp.zeros(5),
        rng=jax.random.PRNGKey(0),
    )
    second = sequence.apply(
        jnp.zeros(5),
        rng=jax.random.PRNGKey(0),
    )
    assert jnp.all(first == second)
    assert len({float(value) for value in first}) == 5


def test_scanned_maps_an_empty_acyclic_sequence():
    sequence = scanned(Leaf(lambda input: input + 1.0, name='increment'))
    output = sequence.apply(jnp.zeros((0,)))
    assert output.shape == (0,)


def test_scanned_apply_reads_the_current_step_member():
    """Verifies that `map_members` substituting the wrapped step updates `scanned` execution."""
    def Counter(scale):
        def init():
            return jnp.zeros(())

        def apply(state, input):
            successor = state + scale * input
            return successor, successor

        return Leaf(apply, init=init, name=f'counter_{scale}').node

    sequence = jnp.ones(3)
    original = scanned(Counter(1.0)).node
    substituted = map_members(
        original,
        lambda member: Counter(10.0)
        if member.name == 'counter_1.0' else member,
    ).with_input(sequence)
    assert substituted.contract.input_spec is not None
    assert jnp.allclose(
        substituted.parameterize().apply(sequence),
        jnp.array([10.0, 20.0, 30.0]))


def test_scan_contract_reads_the_substituted_step_member():
    """Verifies that `scan` delegates `param`, `init`, and `apply` to a substituted `step` member."""
    def Counter(scale):
        def param():
            return jnp.asarray(scale)

        def init(param):
            return jnp.asarray(scale)

        def apply(param, state, input):
            successor = state + scale * input
            return successor, successor

        return Leaf(apply, param=param, init=init,
                    name=f'counter_{scale}').node

    sequence = jnp.ones(3)
    original = scan(Counter(1.0)).node
    substituted = map_members(
        original,
        lambda member: Counter(10.0)
        if member.name == 'counter_1.0' else member,
    ).with_input(sequence)
    assert substituted.contract.input_spec is not None

    bound = substituted.parameterize()
    assert bound.param == 10.0
    assert bound.init() == 10.0
    _, outputs = bound.apply(bound.init(), sequence)
    assert jnp.allclose(outputs, jnp.array([20.0, 30.0, 40.0]))


def test_a_declared_member_resets_and_the_rest_carries():
    """Verifies that `scan(..., boundary=...)` reinitializes `state_reinit` members while carrying sibling state."""
    def Counter():
        def init():
            return jnp.zeros(())

        def apply(state, input):
            return state + 1.0, state + 1.0

        return Leaf(apply, init=init, name='count')

    def Register():
        def init():
            return jnp.zeros(())

        def apply(state, input):
            return state + input, state + input

        return Leaf(apply, init=init, name='register')

    def counted():
        def apply(self, input):
            ticks = self.count(input)
            self.register(input)
            return ticks

        return Composite(count=Counter(), register=state_reinit(Register()))(apply, name='counted')

    seq = scan(counted(), boundary='episode')
    assert seq.cyclic

    s1, _ = seq.apply(seq.init(), jnp.ones(5))
    assert s1.count == 5.0 and s1.register == 5.0

    s2, _ = seq.apply(s1, jnp.ones(5))
    assert s2.count == 10.0
    assert s2.register == 5.0


def test_a_boundary_nobody_claims_does_not_exist():
    """Verifies that an unclaimed `state_reinit` tag leaves `scanned` behavior unchanged."""
    def Counter():
        def init():
            return jnp.zeros(())

        def apply(state, input):
            return state + 1.0, state + 1.0

        return Leaf(apply, init=init, name='count')

    declared = scanned(state_reinit(Counter(), boundary='episode'))
    plain = scanned(Counter())
    assert jnp.allclose(declared.apply(jnp.ones(5)), plain.apply(jnp.ones(5)))


def test_apply_rng_nodes_are_scannable():
    """Verifies that `scanned` splits one boundary `apply` PRNG key into distinct per-step subkeys."""
    def Draw():
        def init():
            return jnp.zeros(())

        def apply(state, x, rng):
            v = jax.random.normal(rng.next()) + x
            return v, v

        return Leaf(apply, init=init, name='draw')

    rolled = scanned(Draw())
    assert rolled.contract.apply_takes_rng
    a = rolled.apply(rng=jax.random.PRNGKey(0), x=jnp.zeros(5))
    b = rolled.apply(rng=jax.random.PRNGKey(0), x=jnp.zeros(5))
    c = rolled.apply(rng=jax.random.PRNGKey(1), x=jnp.zeros(5))

    assert a.shape == (5,)
    assert jnp.all(a == b)
    assert not jnp.all(a == c)
    assert len({float(v) for v in a}) == 5


def test_both_homes_of_entropy_under_one_scan():
    """Verifies that `scanned` serves both `init` PRNG draws and per-step `apply` PRNG draws."""
    def Both():
        def init(rng):
            return Struct(bias=jax.random.normal(rng.next()), n=jnp.zeros(()))

        def apply(state, x, rng):
            drawn = jax.random.normal(rng.next())
            return state.replace(n=state.n + 1.0), state.bias + drawn + x

        return Leaf(apply, init=init, name='both')

    rolled = scanned(Both())
    assert rolled.contract.apply_takes_rng
    out = rolled.apply(rng=jax.random.PRNGKey(0), x=jnp.zeros(4))
    assert out.shape == (4,)
    assert len({float(v) for v in out}) == 4


def test_recording_does_not_change_what_the_node_is():
    """Verifies that `record=True` emits the state trajectory in `Aux` without altering primary outputs."""
    def Accum():
        def apply(state, input):
            new = state + input
            return new, new * 10

        return Leaf(apply, init=lambda: jnp.zeros(()), name='accum').node

    xs = jnp.ones(4)
    plain = scanned(Accum()).parameterize()
    watched = scanned(Accum(), record=True).parameterize()

    outs = plain.apply(xs)
    recorded, aux = watched.apply(xs)
    assert jnp.allclose(recorded, outs)
    assert jnp.allclose(aux.state, jnp.array([1.0, 2.0, 3.0, 4.0]))

    assert jnp.allclose(drop_aux(scanned(Accum(), record=True).parameterize().apply(xs)),
                        outs)


def test_recording_over_a_node_that_already_sows_is_refused():
    """Verifies that `record=True` raises `TypeError` when the step Node already emits `Aux(state=...)`."""
    def Sower():
        def apply(state, input):
            return state + input, (input, Aux(state=input * 2))

        return Leaf(apply, init=lambda: jnp.zeros(()), name='sower').node

    node = scanned(Sower(), record=True).parameterize()
    with pytest.raises(TypeError, match="already sows 'state'"):
        node.apply(jnp.ones(3))


def test_a_nested_scan_still_draws():
    """Verifies that nested `scanned(scan(...))` propagates `apply_takes_rng` and splits keys across all steps."""
    def Draws():
        def apply(state, x, rng):
            return state, jax.random.normal(rng.next(), ())

        return Leaf(apply, init=lambda: jnp.zeros(()), name='draws').node

    assert scan(Draws()).contract.apply_takes_rng
    assert scan(Draws()).contract.apply_fields == ('x',)

    outs = scanned(scan(Draws())).parameterize().apply(
        x=jnp.zeros((3, 4)), rng=jax.random.PRNGKey(0))

    flat = [float(v) for v in outs.ravel()]
    assert len(set(flat)) == 12
    assert outs.shape == (3, 4)


def test_a_resolved_step_transfers_under_the_map():
    """Verifies that `scan` over an input-resolved step publishes an `AxisSpec` with unknown sequence count."""
    from nodejax.core.binding import (AxisSpec)
    from nodejax.core.spec import element_spec

    def Step():
        def apply(state, input):
            return state + input, state

        return Leaf(apply, init=lambda: jnp.zeros(3), name='step').node

    step = Step().with_input(jnp.zeros(3))
    assert step.contract.input_spec is not None
    spec = scan(step).contract.input_spec
    assert type(spec.input) is AxisSpec and spec.input.count is None
    assert element_spec(spec).input.shape == (3,)


def test_scan_of_an_acyclic_step_is_its_sequence_map():
    """Verifies that `scan` maps an acyclic step across a sequence while rejecting `boundary=` and `carried`."""
    def Double() -> Node:
        return Leaf(lambda input: 2.0 * input, name='double').node

    sequence = jnp.arange(6.0).reshape(3, 2)
    mapped = scan(Double(), n=3).parameterize()
    assert not mapped.cyclic
    assert jnp.allclose(mapped.apply(sequence), 2.0 * sequence)

    with pytest.raises(TypeError):
        scan(Double(), boundary='episode')
    from nodejax import carried
    with pytest.raises(TypeError):
        carried(Double())


def test_carried_returns_the_step_bound_to_its_final_state():
    from nodejax import Composite, carried, iterated

    counter = Leaf(lambda state, input: (state + input, state), init=lambda: jnp.zeros(()),
                   name='counter')
    members = Composite(count=iterated(counter, n=2))

    def apply(self, input):
        return self.count(input)

    program = carried(members(apply)).parameterize()
    done = program.apply(jnp.ones(3))

    assert done.state.count == 6.0
    assert done.count.state == 6.0
    assert done.count.step.state == 6.0


def test_internalized_runs_start_from_a_bound_state():
    """Verifies that `scanned` and `carried` over a `PSNode` start from its bound state."""
    from nodejax import PNode, carried, nn

    features = 2
    cell = (nn.GRU(4) >> nn.GRU(3)).with_input(jnp.zeros(features)).parameterize(
        rng=jax.random.PRNGKey(0))
    blank = cell.initialize(input=jnp.zeros(features))
    started = blank.bind(state=jax.tree.map(jnp.ones_like, blank.state))
    sequence = jnp.linspace(0.0, 1.0, 5 * features).reshape(5, features)
    ended, expected = scan(started).apply(sequence)

    run = scanned(started)
    assert type(run) is PNode and not run.cyclic
    assert dict(run.statics_by_path()) == {
        'step.gru.hidden': 4,
        'step.gru_2.hidden': 3,
        'record': False,
        'n': None,
    }
    assert jnp.allclose(run.apply(sequence), expected)
    replayed = run.specialize().bind(run.param)
    assert jnp.allclose(replayed.apply(sequence), expected)
    assert not jnp.allclose(scanned(blank).apply(sequence), expected)

    done = carried(started).apply(sequence)
    assert jax.tree.all(jax.tree.map(jnp.allclose, done.state, ended.state))
    replayed_done = carried(started).specialize().bind(run.param).apply(sequence)
    assert jax.tree.all(jax.tree.map(
        jnp.allclose, replayed_done.state, ended.state))


def test_an_internalized_run_takes_the_step_state_inputs_beside_the_sequence():
    """Verifies that `scanned` accepts step state-initialization fields alongside the sequence input."""
    counter = Leaf(
        lambda state, input: (state + input, state),
        init=lambda initial: initial,
        name='counter',
    )
    run = scanned(counter).parameterize()
    outputs = run.apply(input=jnp.ones(3), initial=jnp.asarray(10.0))

    assert jnp.allclose(outputs, jnp.asarray([10.0, 11.0, 12.0]))


def test_internalized_runs_declare_and_enforce_their_length():
    run = scanned(Integrator(), n=3).parameterize()
    assert jnp.allclose(run.apply(jnp.ones(3)), jnp.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError, match='expected n=3'):
        run.apply(jnp.ones(4))
    from nodejax import carried
    done = carried(Integrator(), n=3).parameterize().apply(jnp.ones(3))
    assert done.state == 3.0
