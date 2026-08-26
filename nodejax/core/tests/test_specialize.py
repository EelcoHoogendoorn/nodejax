"""specialize: statics as first-class metadata on the node, re-bound on a
built tree.

A decorated factory stamps its record (factory, statics) on the node it
builds; specialize re-runs the factory under overrides and rebuilds the
tree around the changed leaves through the same recorded call that
map_members uses. Broadcasts ('*.field') reach every leaf whose record
declares the field and pass the rest untouched.
"""

import jax
import jax.numpy as jnp
from nodejax.core.ambient import node
import pytest

from nodejax import nn, Leaf, tree_freeze, tree_filter, scanned, map_members
from nodejax.core.binding import (split_aux)
from nodejax.control import Integrator


def test_leaf_respecializes_and_keeps_its_kind():
    """The record is the argument set the factory ran with; the eval
    rebuild is the identity. Dropout draws at APPLY, so BOTH builds are
    stateless and the flip moves only the key requirement: contract
    honesty, with the state tree invariant across the mode switch
    (recalibrated when dropout left the state-rng home). And the road
    back exists."""
    drop = nn.Dropout(0.3).node
    assert drop.statics_by_path()['rate'] == 0.3
    ident = drop.specialize(train=False)
    assert not drop.cyclic and not ident.cyclic
    assert drop.contract.apply_takes_rng
    assert not ident.contract.apply_takes_rng
    assert ident.specialize(train=True).contract.apply_takes_rng


def test_static_replay_discards_later_input_specialization():
    """Re-entering statics rebuilds before the input-evidence stage."""
    input = jnp.zeros(4)
    before = nn.Linear(3).with_input(input).specialize(n_out=5)
    after = nn.Linear(3).specialize(n_out=5).with_input(input)

    assert before.contract.input_spec is None
    assert after.contract.input_spec is not None
    left = before.with_input(input).parameterize(rng=jax.random.PRNGKey(0))
    right = after.parameterize(rng=jax.random.PRNGKey(0))
    assert left.param.w.shape == right.param.w.shape == (4, 5)


def test_wildcards_and_transfer_through_the_tree():
    """'*.train' reaches the leaves that declare it, through the pipe.
    The train build's state bundle demands the dropout key; the eval
    build's does not. Params transfer across the flip (the constraint a
    mode static must keep), the stats freeze, and the evaluator is a
    plain deterministic function."""
    pipe = (nn.Linear(16) >> nn.gelu >> nn.Dropout(0.3)
            >> nn.BatchNorm(0.1) >> nn.Linear(3)).with_input(jnp.zeros(4))
    evalp = pipe.specialize(**{'*.train': False})
    assert pipe.contract.apply_takes_rng                  # the train build owes a key
    assert not evalp.contract.apply_takes_rng             # the eval build owes nothing

    node = pipe.parameterize(rng=jax.random.PRNGKey(0))
    state = node.init()
    frozen = tree_freeze(evalp, tree_filter(state, 'batch_norm')).bind(node.param)
    assert not frozen.cyclic
    x = jnp.ones(4)
    assert jnp.allclose(frozen.apply(x), frozen.apply(x))


def test_dot_path_addresses_one_member():
    """Two dropouts, one addressed by member path: the other keeps its
    record and its draw."""
    pipe = (nn.Dropout(0.3) >> nn.Dropout(0.5)).with_input(jnp.zeros(4))
    tuned = pipe.specialize(**{'drop.train': False})
    assert not tuned.members.drop.contract.apply_takes_rng
    assert tuned.members.drop_2.contract.apply_takes_rng


def test_unrecorded_leaf_refuses():
    plain = Leaf(lambda input: 2.0 * input, name='double')
    with pytest.raises(TypeError, match='no construction record'):
        plain.specialize(gain=1.0)


def test_unknown_member_refuses():
    pipe = (nn.Linear(4) >> nn.gelu).with_input(jnp.zeros(3))
    with pytest.raises(TypeError, match='not members'):
        pipe.specialize(nope={'a': 1})


def test_transform_records_respecialize_and_rebuild():
    """Static specialization and structural replacement remain independent."""
    roll = scanned(Integrator()).parameterize()
    ys = roll.apply(jnp.ones(3))

    recording = roll.specialize(record=True).parameterize()
    clean, aux = split_aux(recording.apply(jnp.ones(3)))
    assert jnp.allclose(clean, ys)               # same run
    assert aux.state.shape == (3,)               # the trajectory now sows

    rebuilt = map_members(recording.node, lambda d: d)
    again, _ = split_aux(rebuilt.bind(recording.param).apply(jnp.ones(3)))
    assert jnp.allclose(again, ys)


def test_static_replay_rebuilds_from_canonical_members():
    """Member surgery is a concrete variant, not persistent static data."""
    def Counter(amount):
        def init():
            return jnp.zeros(())

        def apply(state, input):
            state = state + amount * input
            return state, state

        return Leaf(apply, init=init, name=f'counter_{amount}').node

    roll = scanned(Counter(1.0)).node
    rewritten = map_members(
        roll,
        lambda member: Counter(10.0)
        if member.name == 'counter_1.0' else member,
    )

    assert rewritten.members.step.name == 'counter_10.0'
    canonical = rewritten.specialize(record=True)
    assert canonical.members.step.name == 'counter_1.0'


def test_node_names_the_def_from_the_factory():
    """A node the factory left unnamed takes the factory's own name,
    lowercased; an explicit node name wins untouched."""
    from nodejax import node

    @node
    def Blip():
        return Leaf(lambda input: input + 1.0)

    assert Blip().name == 'blip'
    assert nn.Dropout(0.3).name == 'drop'    # explicit, untouched


def test_statics_of_lists_the_whole_argument_graph():
    """statics_by_path walks the RECORD, dot-joined, bound and unbound alike:
    the canonical listing, and the same paths are the addresses:
    specialize binds a bare key in the node's own record or refuses, so
    reaching a depth means spelling the path this listing prints."""
    from nodejax import batch, scan, statics_by_path
    from nodejax.core.binding import REQUIRED

    @node
    def Cell(width, leak=0.1):
        def init(node):
            return jnp.zeros(width)

        def apply(state, input):
            new = jnp.tanh((1 - leak) * state + input)
            return new, new

        return Leaf(apply, init=init)

    tree = batch(scan(Cell()), n=4)
    listed = statics_by_path(tree)
    assert listed['sample.step.width'] is REQUIRED       # the unbound argument, at home
    assert listed['sample.step.leak'] == 0.1             # defaults are statics now
    assert listed['n'] == 4                              # explicit args list deferred too

    filled = tree.specialize(**{'sample.step.width': 8})  # the listed address
    assert not filled.generic
    listed = statics_by_path(filled)
    assert listed['sample.step.width'] == 8              # listed canonically
    assert listed['sample.step.leak'] == 0.1             # the default, materialized
    assert listed['sample.boundary'] is None              # scan owns its level


def test_describe_writes_the_trees_out():
    """describe is one entry over every tree the library holds: a def's
    member hierarchy with kind flags and its own statics inline, a
    generic's deferred record standing in for the members it does not
    have yet, a bound node's binding trees with a
    leaf tally, a plain pytree alone. Pins are substrings on purpose:
    the format may evolve, the CONTENT may not silently vanish."""
    from nodejax import describe, batch, scan

    @node
    def Cell(width):
        def init(node):
            return jnp.zeros(width)

        def apply(state, input):
            new = jnp.tanh(state + input)
            return new, new

        return Leaf(apply, init=init)

    deferred = describe(batch(scan(Cell()), n=4))
    assert 'width=<unbound>' in deferred                 # the unbound argument, visible
    assert '[generic]' in deferred                       # the kind, spelled out
    assert 'n=4' in deferred                             # own statics inline

    model = nn.Linear(3).with_input(jnp.zeros(2)).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    bound = describe(model)
    assert 'param:' in bound and 'state:' in bound
    assert 'w = float32(2, 3)' in bound                  # the binding rows
    assert 'leaves' in bound                             # the tally

    assert 'w = float32(2, 3)' in describe(model.param)  # a pytree alone
