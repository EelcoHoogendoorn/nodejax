"""A boundary is where declared state is re-inited. Everything else carries.

A carry carries: that is what scan does between calls, and claiming a boundary
does not change it. What a boundary does is give the nodes that want to depart
from carrying a place to say so, and a name to say it against.

Everything else follows. WHICH state re-inits is declared by the node that
owns it, since a scan is layers away and could only point by name. WHEN is the
start of an enclosing scan call carrying the same tag. Hence a name both sides
use.

The default was once the other way round, and the asymmetry it created is
worth remembering: an unclaimed scan carried everything while a claimed one
rebuilt everything undeclared, so the same tree meant opposite things
depending on an argument that read like a label. Now the two agree and a claim
only ever adds departures.
"""

import jax
import jax.numpy as jnp
import pytest

from nodejax import Node, Leaf, Composite, scan, scanned, state_reinit, batch, nn
from nodejax.struct import Struct


def Accumulator() -> Node:
    def init():
        return jnp.zeros(())

    def apply(state, input):
        return state + input, state + input

    return Leaf(apply, init=init, name='acc')


def test_carrying_needs_no_declaration():
    """The default, and the reason most nodes say nothing at all. A scan
    threads its carry, and a claimed boundary threads it too."""
    kept = scan(Accumulator())
    assert kept.cyclic

    s, _ = kept.apply(kept.init(), jnp.ones(4))
    assert s == 4.0
    s, _ = kept.apply(s, jnp.ones(4))
    assert s == 8.0                              # carried, with nothing declared

    # scanned is the other thing: the loop goes INSIDE, no slot to thread
    plain = scanned(Accumulator())
    assert not plain.cyclic
    assert jnp.allclose(plain.apply(jnp.ones(4)), plain.apply(jnp.ones(4)))


def test_reset_starts_over_where_it_would_otherwise_carry():
    """state_reinit is how a node departs from carrying, and it needs a boundary to
    depart AT: the tag is what says when."""
    kept = scan(state_reinit(Accumulator()), boundary='episode')
    assert kept.cyclic

    s, _ = kept.apply(kept.init(), jnp.ones(4))
    assert s == 4.0
    s, _ = kept.apply(s, jnp.ones(4))
    assert s == 4.0                              # rebuilt: each run starts over


def test_the_tag_has_to_match():
    """Both sides name the same boundary, and naming is what lets a node under
    two scans say which one it answers to."""
    matched = scan(state_reinit(Accumulator(), boundary='outer'), boundary='outer')
    s, _ = matched.apply(matched.init(), jnp.ones(4))
    s, _ = matched.apply(s, jnp.ones(4))
    assert s == 4.0                              # the state_reinit fired


def test_a_claim_that_fires_nothing_is_an_error():
    """One rule, whatever else the tree declares. Both of these are the same
    mistake from the caller's side, "I named a tag that fires nothing", and a
    guard that raised on the first and not the second drew a line on
    information the caller does not have.

    The mistake is worth refusing because it reads like working code: a state_reinit
    spelled with another name never runs, and its state carries where it was
    meant to state_reinit."""
    with pytest.raises(TypeError, match='fires nothing'):
        scan(state_reinit(Accumulator(), boundary='outer'), boundary='inner')

    with pytest.raises(TypeError, match='nothing beneath it declares'):
        scan(Accumulator(), boundary='episode')


def test_reset_costs_no_structure():
    """A Wrapper's state IS the wrapped node's, so declaring changes nothing
    about the shape of anything: no level, no key, no param."""
    bare, kept = Accumulator(), state_reinit(Accumulator())

    assert jax.tree.structure(bare.init()) == jax.tree.structure(kept.init())
    assert jax.tree.structure(bare.param) == jax.tree.structure(kept.param)
    assert jax.tree.structure(scan(kept, boundary='episode').init()) == \
           jax.tree.structure(bare.init())


def test_a_hand_wired_leaf_can_declare_for_itself():
    """state_reinit is the library's spelling of the common case, not
    the mechanism. A leaf holding slots of different lifetimes declares
    directly, over its OWN slot names, which never leave it. The action reads
    three trees of that layout: what carried, what a fresh init gives, and
    what the subtree already decided."""
    def Mixed():
        def init():
            return Struct(learned=jnp.zeros(()), scratch=jnp.zeros(()))

        def apply(state, input):
            new = Struct(learned=state.learned + input, scratch=state.scratch + input)
            return new, new.learned

        return Leaf(apply, init=init, name='mixed',
                        boundary={'episode': lambda carried, init, decided:
                                  decided.replace(scratch=init.scratch)})

    seq = scan(Mixed(), boundary='episode')
    s, _ = seq.apply(seq.init(), jnp.ones(3))
    s, _ = seq.apply(s, jnp.ones(3))
    assert s.learned == 6.0                      # undeclared: carried
    assert s.scratch == 3.0                      # declared: rebuilt


# --- the handbook's worked example, held to its claim ---

SEQ = jnp.sin(jnp.arange(48.0).reshape(12, 4))
KEY = jax.random.PRNGKey(0)


def test_chunking_a_sequence_disagrees_with_running_it_whole():
    """The motivating failure, and the reason a boundary exists at all. A plain
    scan internalizes the loop, so there is no state to thread between calls
    and every chunk starts from a fresh carry."""
    model = scanned(nn.RNN(4)).with_input(SEQ).parameterize(rng=KEY)
    assert not model.cyclic                       # no slot to thread, by construction

    whole = model.apply(SEQ)
    chunks = [model.apply(SEQ[i:i + 4]) for i in range(0, 12, 4)]
    assert not jnp.allclose(jnp.concatenate(chunks), whole)


def test_scan_makes_the_chunks_agree_exactly():
    """scan threads the carry, so whatever the caller hands back continues and
    the chunked run IS the whole run. No declaration involved: threading is
    what scan does, and a boundary is for saying some state should NOT thread."""
    whole = scanned(nn.RNN(4)).with_input(SEQ).parameterize(rng=KEY).apply(SEQ)

    model = scan(nn.RNN(4)).with_input(SEQ[:4]).parameterize(rng=KEY)
    assert model.cyclic

    state, chunks = model.init(), []
    for i in range(0, 12, 4):
        state, out = model.apply(state, SEQ[i:i + 4])
        chunks.append(out)

    assert jnp.allclose(jnp.concatenate(chunks), whole, atol=1e-5)


def test_streams_chunked_under_batch_each_carry_their_own():
    """How stateful sequence training actually runs: several sequences in
    parallel, each chunked, each carrying its own state across chunks. batch
    maps state per element, and no boundary appears at all, because carrying
    across chunks is what a carry does."""
    seq = jnp.sin(jnp.arange(96.0).reshape(2, 12, 4))          # two sequences
    whole = batch(scanned(nn.RNN(4))).with_input(seq).parameterize(rng=KEY).apply(seq)

    model = batch(scan(nn.RNN(4))).with_input(seq[:, :4]).parameterize(rng=KEY)

    state, chunks = model.init(), []
    for i in range(0, 12, 4):
        state, out = model.apply(state, seq[:, i:i + 4])
        chunks.append(out)

    assert state.shape == (2, 4)                       # one carry per sequence, not one shared
    assert jnp.allclose(jnp.concatenate(chunks, axis=1), whole, atol=1e-5)


def test_two_boundaries_and_a_node_that_answers_to_one():
    """What naming is FOR. Two nested scans, two names, and state that answers
    to one, the other, or both.

    A recording is fed in chunks. The recurrent carry continues across chunks
    and re-inits per recording, because a new recording is a new signal. The
    statistics accumulate across everything, because they describe the sensor
    rather than the recording. Only the first has anything to declare, and the
    inner scan needs no boundary at all: continuing across chunks is what both
    of them do by default."""
    def Counter():
        def init():
            return jnp.zeros(())

        def apply(state, input):
            return state + 1.0, state + 1.0

        return Leaf(apply, init=init, name='count')

    per_chunk = state_reinit(Counter(), boundary='recording')     # rebuilt each recording
    always = Counter()                                     # carries: nothing to say

    assert sorted(per_chunk.contract.boundary_names) == ['recording']
    assert sorted(always.contract.boundary_names) == []

    def both():
        def apply(self, input):
            self.always(input)
            return self.per_chunk(input)

        return Composite(per_chunk=per_chunk, always=always)(apply, name='both')

    model = scan(scan(both()), boundary='recording')
    recording = jnp.ones((2, 3))                       # two chunks of three steps

    state, _ = model.apply(model.init(), recording)
    assert state.per_chunk == 6.0 and state.always == 6.0

    state, _ = model.apply(state, recording)           # a second recording
    assert state.per_chunk == 6.0                      # rebuilt: it declared so
    assert state.always == 12.0                        # carried, saying nothing


def test_the_name_picks_which_enclosing_scan_a_reset_answers():
    """What is left of "two boundaries" once carrying is free, and it is the
    part that always mattered. One declaration, two nested scans, and the NAME
    is the only thing saying which of them the state_reinit belongs to.

    The chunk comparison rests on this: its recurrent carry crosses chunks and
    re-inits per recording, and nothing about the node says so. Only the tag
    does, and the identical node under the other tag gives another answer."""
    def Counter():
        def apply(state, input):
            return state + 1.0, state + 1.0
        return Leaf(apply, init=lambda: jnp.zeros(()), name='c').node

    outer_tag = scan(scan(state_reinit(Counter(), boundary='recording')),
                     boundary='recording').parameterize()
    inner_tag = scan(scan(state_reinit(Counter(), boundary='chunk'), boundary='chunk')
                     ).parameterize()

    xs = jnp.ones((2, 3))                         # two chunks of three steps
    a, _ = outer_tag.apply(outer_tag.init(), xs)
    b, _ = inner_tag.apply(inner_tag.init(), xs)
    assert a == 6.0                               # rebuilt once, at the recording
    assert b == 3.0                               # rebuilt per chunk: only the last
