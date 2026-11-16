"""Verifies structural parameter sharing via `tie` across composite members.

`tie` stores one canonical parameter subtree on the source member and expands it
into every tied consumer before `init` and `apply`, so gradients accumulate by
the chain rule, PyTree operations preserve sharing, and member state remains
independent.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import (
    trained, Node, Leaf, PNode, PSNode, nn, scan, serial, train_step, tie,
)
from nodejax.core.types import PyTree
from nodejax.struct import Struct


def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)


def tile(tree: PyTree, n: int):
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), tree)


K = 3


def Encoder(k: int) -> Node:
    def param(rng):
        return Struct(weight=jax.random.normal(rng.next(), (k,)))
    def apply(param, input):
        return param.weight @ input
    return Leaf(apply, param=param, name='enc')


def Decoder(k: int) -> Node:
    def param(rng):
        return Struct(weight=jax.random.normal(rng.next(), (k,)))
    def apply(param, input):
        return param.weight * input
    return Leaf(apply, param=param, name='dec')


def StatefulScale(name: str) -> Node:
    def param(scale):
        return Struct(scale=jnp.asarray(scale))

    def init(param):
        return jnp.asarray(0.0)

    def apply(param, state, input):
        return state + input, param.scale * input

    return Leaf(apply, param=param, init=init, name=name)


def tied_autoencoder() -> Node:
    return tie(Encoder(K) >> Decoder(K), 'enc', 'dec')


def test_single_copy_survives_jax_boundaries():
    """Verifies that `tie` stores one weight leaf and preserves sharing across `tree_map` and `jit`."""
    node = tied_autoencoder().parameterize(rng=jax.random.PRNGKey(0))
    assert len(jax.tree.leaves(node.param)) == 1
    assert 'dec' not in node.param

    roundtrip = jax.tree.map(lambda x: x + 0.0, node)
    assert len(jax.tree.leaves(roundtrip.param)) == 1

    out = jax.jit(lambda n, x: n.apply(x))(node, jnp.ones(K))
    assert out.shape == (K,)


def test_gradients_accumulate_across_uses():
    """Verifies that the tied parameter gradient equals the sum of untied per-member gradients."""
    w = jnp.array([0.5, -1.0, 2.0])
    x = jnp.array([1.0, 2.0, 3.0])

    tied = tied_autoencoder().bind(Struct(enc=Struct(weight=w), dec=()))
    untied = (Encoder(K) >> Decoder(K)).bind(
        Struct(enc=Struct(weight=w), dec=Struct(weight=w)))

    f = lambda n: jnp.sum(n.apply(x))
    g_tied = jax.grad(f)(tied).param.enc.weight
    g_untied = jax.grad(f)(untied).param
    assert jnp.allclose(g_tied, g_untied.enc.weight + g_untied.dec.weight)


def test_tied_training_cannot_diverge():
    """Verifies that `train_step` updates the single shared parameter copy of a tied autoencoder."""
    auto = tied_autoencoder().parameterize(rng=jax.random.PRNGKey(1)).initialize()
    trainer = train_step(auto, mse, optax.adam(0.05))

    u = jnp.array([0.6, -0.8, 0.0])
    steps = 400
    final, aux = trained(trainer).apply(input=tile(u, steps), target=tile(u, steps))

    assert aux.loss[-1] < 1e-3
    assert 'dec' not in final.param


def test_alias_parameterization_rejected():
    with pytest.raises(TypeError, match='dec'):
        tied_autoencoder().parameterize(enc=Struct(weight=jnp.ones(K)),
                                        dec=Struct(weight=jnp.ones(K)))


def test_tie_preserves_bound_parameters_and_source_wins():
    pipe = serial(
        source=StatefulScale('source'),
        alias=StatefulScale('alias'),
        other=StatefulScale('other'),
    )
    model = pipe.bind(Struct(
        source=Struct(scale=jnp.asarray(2.0)),
        alias=Struct(scale=jnp.asarray(11.0)),
        other=Struct(scale=jnp.asarray(3.0)),
    ))

    tied = tie(model, 'source', 'alias')

    assert type(tied) is PNode
    assert tied.param.__keys__ == ('source', 'other')
    assert tied.param.source.scale == 2.0
    assert tied.param.other.scale == 3.0
    state = Struct(
        source=jnp.asarray(10.0),
        alias=jnp.asarray(20.0),
        other=jnp.asarray(30.0),
    )
    _, output = tied.apply(state, jnp.asarray(2.0))
    assert output == 24.0


def test_tie_preserves_complete_bound_state():
    pipe = serial(
        source=StatefulScale('source'),
        alias=StatefulScale('alias'),
        other=StatefulScale('other'),
    )
    model = pipe.bind(Struct(
        source=Struct(scale=jnp.asarray(2.0)),
        alias=Struct(scale=jnp.asarray(11.0)),
        other=Struct(scale=jnp.asarray(3.0)),
    ), state=Struct(
        source=jnp.asarray(10.0),
        alias=jnp.asarray(20.0),
        other=jnp.asarray(30.0),
    ))

    tied = tie(model, 'source', 'alias')

    assert type(tied) is PSNode
    assert tied.param.__keys__ == ('source', 'other')
    assert tied.state.__keys__ == ('source', 'alias', 'other')
    assert tied.state.source == 10.0
    assert tied.state.alias == 20.0
    assert tied.state.other == 30.0

    successor, output = tied(jnp.asarray(2.0))
    assert successor.state.source == 12.0
    assert successor.state.alias == 24.0
    assert successor.state.other == 38.0
    assert output == 24.0


def test_retying_an_unbound_tree_keeps_earlier_aliases_sparse():
    pipe = serial(
        source=StatefulScale('source'),
        first_alias=StatefulScale('first_alias'),
        second_alias=StatefulScale('second_alias'),
        other=StatefulScale('other'),
    ).with_input(jnp.asarray(1.0))
    first = tie(pipe, 'source', 'first_alias')
    second = tie(first, 'source', 'second_alias')

    model = second.parameterize(
        source=Struct(scale=jnp.asarray(2.0)),
        other=Struct(scale=jnp.asarray(3.0)),
    )

    assert model.param.__keys__ == ('source', 'other')
    state = Struct(
        source=jnp.asarray(0.0),
        first_alias=jnp.asarray(0.0),
        second_alias=jnp.asarray(0.0),
        other=jnp.asarray(0.0),
    )
    _, output = model.apply(state, jnp.asarray(1.0))
    assert output == 24.0


def test_retying_a_bound_tree_keeps_earlier_aliases_sparse():
    pipe = serial(
        source=StatefulScale('source'),
        first_alias=StatefulScale('first_alias'),
        second_alias=StatefulScale('second_alias'),
        other=StatefulScale('other'),
    ).with_input(jnp.asarray(1.0))
    first = tie(pipe, 'source', 'first_alias').parameterize(
        source=Struct(scale=jnp.asarray(2.0)),
        second_alias=Struct(scale=jnp.asarray(7.0)),
        other=Struct(scale=jnp.asarray(3.0)),
    )

    second = tie(first, 'source', 'second_alias')

    assert type(second) is PNode
    assert second.param.__keys__ == ('source', 'other')
    assert second.param.source.scale == 2.0
    assert second.param.other.scale == 3.0


def test_alias_before_captured_source_uses_source_during_spec_walk():
    source = StatefulScale('source').parameterize(scale=2.0)
    pipe = serial(
        alias=StatefulScale('alias'),
        source=source,
        other=StatefulScale('other'),
    ).with_input(jnp.asarray(1.0))

    model = tie(pipe, 'source', 'alias').parameterize(
        other=Struct(scale=jnp.asarray(3.0)))

    assert model.param.__keys__ == ('source', 'other')
    assert model.param.source.scale == 2.0
    state = Struct(
        alias=jnp.asarray(0.0),
        source=jnp.asarray(0.0),
        other=jnp.asarray(0.0),
    )
    _, output = model.apply(state, jnp.asarray(1.0))
    assert output == 12.0


VOCAB, DIM = 8, 4


def test_tied_embeddings_are_one_object():
    """Verifies that tying `nn.Embed` and `nn.Unembed` shares one embedding table across training."""
    lm = tie(nn.Embed(VOCAB, DIM) >> nn.Unembed(VOCAB, DIM), 'embed', 'unembed')
    ids = jnp.arange(VOCAB)

    node = lm.parameterize(rng=jax.random.PRNGKey(0))
    assert len(jax.tree.leaves(node.param)) == 1
    assert 'unembed' not in node.param

    def xent(logits, target):
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(VOCAB), target])

    trainer = train_step(node.initialize(), xent, optax.adam(0.1))
    final, aux = trained(trainer).apply(input=tile(ids, 200), target=tile(ids, 200))
    assert aux.loss[-1] < 0.1
    assert 'unembed' not in final.param


def test_tie_rejects_leaves_and_unknown_members():
    """Verifies that `tie` rejects memberless leaf Nodes and unknown member names."""
    from nodejax.control import Gain
    with pytest.raises(TypeError, match='has none'):
        tie(Gain(), 'enc', 'dec')
    pipe = serial(enc=Gain(), dec=Gain())
    with pytest.raises(TypeError, match='unknown members'):
        tie(pipe, 'enc', 'deocder')


def test_tie_resolves_shape_reading_members():
    """Verifies that `tie` propagates input shapes to interior shape-reading members like `nn.Linear`."""
    ids = jnp.arange(VOCAB)
    pipe = nn.Embed(VOCAB, DIM) >> nn.Linear(DIM) >> nn.Unembed(VOCAB, DIM)
    node = tie(pipe, 'embed', 'unembed').with_input(ids).parameterize(rng=jax.random.PRNGKey(0))
    assert node.param.linear.w.shape == (DIM, DIM)
    assert 'unembed' not in node.param
    assert node.apply(ids).shape == (VOCAB, VOCAB)
