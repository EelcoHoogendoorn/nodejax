"""Verifies named `boundary=` state reinitialization across `scan` and `state_reinit`.

`scan` carries all undeclared state across calls by default. Nodes wrapped with
`state_reinit` (or declaring a custom `boundary` hook) reinitialize their
state at the start of an enclosing `scan` call that claims the matching tag.
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
    """Verifies that `scan` carries state across calls by default whereas `scanned` internalizes state."""
    kept = scan(Accumulator())
    assert kept.cyclic

    s, _ = kept.apply(kept.init(), jnp.ones(4))
    assert s == 4.0
    s, _ = kept.apply(s, jnp.ones(4))
    assert s == 8.0

    plain = scanned(Accumulator())
    assert not plain.cyclic
    assert jnp.allclose(plain.apply(jnp.ones(4)), plain.apply(jnp.ones(4)))


def test_reset_starts_over_where_it_would_otherwise_carry():
    """Verifies that `state_reinit` rebuilds state at each enclosing `scan` call claiming its boundary."""
    kept = scan(state_reinit(Accumulator()), boundary='episode')
    assert kept.cyclic

    s, _ = kept.apply(kept.init(), jnp.ones(4))
    assert s == 4.0
    s, _ = kept.apply(s, jnp.ones(4))
    assert s == 4.0


def test_the_tag_has_to_match():
    """Verifies that `state_reinit` fires when its boundary name matches the enclosing `scan` claim."""
    matched = scan(state_reinit(Accumulator(), boundary='outer'), boundary='outer')
    s, _ = matched.apply(matched.init(), jnp.ones(4))
    s, _ = matched.apply(s, jnp.ones(4))
    assert s == 4.0


def test_a_claim_that_fires_nothing_is_an_error():
    """Verifies that `scan(..., boundary=...)` raises `TypeError` when no member beneath declares that tag."""
    with pytest.raises(TypeError, match='fires nothing'):
        scan(state_reinit(Accumulator(), boundary='outer'), boundary='inner')

    with pytest.raises(TypeError, match='nothing beneath it declares'):
        scan(Accumulator(), boundary='episode')


def test_reset_costs_no_structure():
    """Verifies that `state_reinit` preserves the wrapped Node's `param` and `state` PyTree structures."""
    bare, kept = Accumulator(), state_reinit(Accumulator())

    assert jax.tree.structure(bare.init()) == jax.tree.structure(kept.init())
    assert jax.tree.structure(bare.param) == jax.tree.structure(kept.param)
    assert jax.tree.structure(scan(kept, boundary='episode').init()) == \
           jax.tree.structure(bare.init())


def test_a_hand_wired_leaf_can_declare_for_itself():
    """Verifies that a `Leaf` declaring a custom `boundary` hook selectively resets individual state fields."""
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
    assert s.learned == 6.0
    assert s.scratch == 3.0


SEQ = jnp.sin(jnp.arange(48.0).reshape(12, 4))
KEY = jax.random.PRNGKey(0)


def test_chunking_a_sequence_disagrees_with_running_it_whole():
    """Verifies that applying `scanned` to separate chunks resets state per chunk rather than continuing."""
    model = scanned(nn.RNN(4)).with_input(SEQ).parameterize(rng=KEY)
    assert not model.cyclic

    whole = model.apply(SEQ)
    chunks = [model.apply(SEQ[i:i + 4]) for i in range(0, 12, 4)]
    assert not jnp.allclose(jnp.concatenate(chunks), whole)


def test_scan_makes_the_chunks_agree_exactly():
    """Verifies that threading external state across `scan` chunks matches one full `scanned` run."""
    whole = scanned(nn.RNN(4)).with_input(SEQ).parameterize(rng=KEY).apply(SEQ)

    model = scan(nn.RNN(4)).with_input(SEQ[:4]).parameterize(rng=KEY)
    assert model.cyclic

    state, chunks = model.init(), []
    for i in range(0, 12, 4):
        state, out = model.apply(state, SEQ[i:i + 4])
        chunks.append(out)

    assert jnp.allclose(jnp.concatenate(chunks), whole, atol=1e-5)


def test_streams_chunked_under_batch_each_carry_their_own():
    """Verifies that `batch(scan(...))` carries an independent state stream per batch element across chunks."""
    seq = jnp.sin(jnp.arange(96.0).reshape(2, 12, 4))
    whole = batch(scanned(nn.RNN(4))).with_input(seq).parameterize(rng=KEY).apply(seq)

    model = batch(scan(nn.RNN(4))).with_input(seq[:, :4]).parameterize(rng=KEY)

    state, chunks = model.init(), []
    for i in range(0, 12, 4):
        state, out = model.apply(state, seq[:, i:i + 4])
        chunks.append(out)

    assert state.shape == (2, 4)
    assert jnp.allclose(jnp.concatenate(chunks, axis=1), whole, atol=1e-5)


def test_two_boundaries_and_a_node_that_answers_to_one():
    """Verifies that in nested scans, a tagged member resets per outer recording while an untagged member carries."""
    def Counter():
        def init():
            return jnp.zeros(())

        def apply(state, input):
            return state + 1.0, state + 1.0

        return Leaf(apply, init=init, name='count')

    per_chunk = state_reinit(Counter(), boundary='recording')
    always = Counter()

    assert sorted(per_chunk.contract.boundary_names) == ['recording']
    assert sorted(always.contract.boundary_names) == []

    def both():
        def apply(self, input):
            self.always(input)
            return self.per_chunk(input)

        return Composite(per_chunk=per_chunk, always=always)(apply, name='both')

    model = scan(scan(both()), boundary='recording')
    recording = jnp.ones((2, 3))

    state, _ = model.apply(model.init(), recording)
    assert state.per_chunk == 6.0 and state.always == 6.0

    state, _ = model.apply(state, recording)
    assert state.per_chunk == 6.0
    assert state.always == 12.0


def test_the_name_picks_which_enclosing_scan_a_reset_answers():
    """Verifies that the boundary name selects whether `state_reinit` fires on the inner or outer `scan`."""
    def Counter():
        def apply(state, input):
            return state + 1.0, state + 1.0
        return Leaf(apply, init=lambda: jnp.zeros(()), name='c').node

    outer_tag = scan(scan(state_reinit(Counter(), boundary='recording')),
                     boundary='recording').parameterize()
    inner_tag = scan(scan(state_reinit(Counter(), boundary='chunk'), boundary='chunk')
                     ).parameterize()

    xs = jnp.ones((2, 3))
    a, _ = outer_tag.apply(outer_tag.init(), xs)
    b, _ = inner_tag.apply(inner_tag.init(), xs)
    assert a == 6.0
    assert b == 3.0
