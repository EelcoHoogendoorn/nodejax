"""scan: internalize the state loop, a stepper becomes a sequence function.

Each apply is an episode: state is built from init and runs the sequence.
With boundary=, scan CLAIMS a tag, and the nodes beneath that declared the
same one decide what to do with the episode boundary. Nothing here decides
for them.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import (Node, Aux, Leaf, Composite, drop_aux, map_members,
                     scan, scanned, state_reinit)
from nodejax.struct import Struct
from nodejax.control import Integrator


def test_scan_transform():
    """PCN -> PN: sequence-level node with internalized state."""
    seq = scanned(Integrator())
    assert type(seq) is Node and not seq.cyclic
    node = seq.parameterize()
    outs = node.apply(jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(outs, jnp.array([1.0, 3.0, 6.0]))


def test_scanned_apply_reads_the_current_step_member():
    """Replacing the wrapped step changes the sequence execution."""
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
    """A resolved scan does not reconstruct on a call, so this pins all three
    custom calls to the member seam: param construction, fresh state,
    and the step body must each consult the substituted step."""
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
    """A member declared state_reinit re-initializes at every episode start; an
    ordinary one carries. Two lifetimes means two nodes: the declaration is
    per node, so the slot that departs is its own member rather than a name
    spelled at the scan."""
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
    assert seq.cyclic                             # the carry lives outside, as ever

    s1, _ = seq.apply(seq.init(), jnp.ones(5))
    assert s1.count == 5.0 and s1.register == 5.0

    s2, _ = seq.apply(s1, jnp.ones(5))
    assert s2.count == 10.0                       # undeclared: kept counting
    assert s2.register == 5.0                     # declared state_reinit: fresh each episode


def test_a_boundary_nobody_claims_does_not_exist():
    """state_reinit declares a tag; if no enclosing scan claims it, no walk runs and
    the declaration does nothing. Here that is visible twice over, since
    scanned builds its state afresh every call regardless."""
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
    """A node that draws at APPLY is scannable: the one boundary key splits
    per STEP, the time-axis twin of a composite splitting toward members.
    Before this, scan handed the key to init only, and a node whose
    state carried no RNG key simply refused it."""
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
    assert jnp.all(a == b)                       # one key, one rollout
    assert not jnp.all(a == c)
    assert len({float(v) for v in a}) == 5        # and no two steps share a draw


def test_both_homes_of_entropy_under_one_scan():
    """A node carrying rng as state AND drawing at apply gets both served
    from the one key, without either taking the other's stream."""
    def Both():
        def init(rng):
            return Struct(rng=rng, n=jnp.zeros(()))

        def apply(state, x, rng):
            streamed = jax.random.normal(state.rng)
            drawn = jax.random.normal(rng.next())
            return state.replace(n=state.n + 1.0), streamed + drawn + x

        return Leaf(apply, init=init, name='both')

    rolled = scanned(Both())
    assert rolled.contract.apply_takes_rng
    out = rolled.apply(rng=jax.random.PRNGKey(0), x=jnp.zeros(4))
    assert out.shape == (4,)
    assert len({float(v) for v in out}) == 4      # the per-step draw varies


def test_recording_does_not_change_what_the_node_is():
    """record= sows the trajectory instead of wrapping the output, and this is
    what that buys: a recorded scan still composes.

    It used to return Struct(state=..., output=...), which made the recorded
    node a DIFFERENT node to everything downstream. Anything reading its
    output had to learn that it had been observed, which is the opposite of
    what an observation should cost."""
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
    assert jnp.allclose(recorded, outs)                  # the output is untouched
    assert jnp.allclose(aux.state, jnp.array([1.0, 2.0, 3.0, 4.0]))

    # and the trace comes back off, leaving exactly the unrecorded node
    assert jnp.allclose(drop_aux(scanned(Accum(), record=True)).parameterize().apply(xs),
                        outs)


def test_recording_over_a_node_that_already_sows_is_refused():
    """The trajectory joins what a node sows rather than replacing it, so a
    node already sowing `state` is a collision, and a named one."""
    def Sower():
        def apply(state, input):
            return state + input, (input, Aux(state=input * 2))

        return Leaf(apply, init=lambda: jnp.zeros(()), name='sower').node

    node = scanned(Sower(), record=True).parameterize()
    with pytest.raises(TypeError, match="already sows 'state'"):
        node.apply(jnp.ones(3))


def test_a_nested_scan_still_draws():
    """A scan carries its inner's rng declaration, and has to: an enclosing
    scan can only route a key to a node that says it wants one.

    scan used to set apply_input_spec=None whatever it wrapped, which threw
    the declaration away. One scan over a drawing node worked, because that
    scan reads the INNER's spec to route. Two scans did not: the outer read
    the inner scan's own spec, saw nothing, split nothing, and the run died on
    a field lookup rather than anything that named the problem.

    A marker spec transfers because it names fields and carries no shapes, and
    the sequence's fields are the step's fields; a resolved step transfers
    as its element under the map."""
    def Draws():
        def apply(state, x, rng):
            return state, jax.random.normal(rng.next(), ())

        return Leaf(apply, init=lambda: jnp.zeros(()), name='draws').node

    assert scan(Draws()).contract.apply_takes_rng                  # the declaration survives
    assert scan(Draws()).contract.apply_fields == ('x',)

    outs = scanned(scan(Draws())).parameterize().apply(
        x=jnp.zeros((3, 4)), rng=jax.random.PRNGKey(0))

    flat = [float(v) for v in outs.ravel()]
    assert len(set(flat)) == 12                          # no step shares a draw
    assert outs.shape == (3, 4)


def test_a_resolved_step_transfers_under_the_map():
    """The other half of the rule. A resolved spec describes ONE step and
    the sequence is many of them, so scan cannot claim it as its own flat
    shape; it publishes the step's element under the map instead, the
    length unknown until a binding supplies it. Recalibrated from
    'publishes nothing', which erased a fully known element."""
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
