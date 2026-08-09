"""Shared parameters, without reference semantics: SHARING IS
REPARAMETERIZATION. tie() precomposes a composite def with an expand map
that inserts one param subtree at every consumer's slot before init/apply.
The composite param carries a single copy, so:

- gradients from every use accumulate by the chain rule (verified exactly
  against the untied sum below),
- the optimizer sees one object; tied training cannot diverge,
- the sharing cannot be silently lost through a jax boundary — it is
  structural; there is no second copy to lose.

State stays per-member: share the object, not its state.

Two archetypes: a tied autoencoder, and tied embeddings, where the
embed and unembed views of a language model read one table: the same
object by definition, so training either view trains both.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import NodeDef, node_def, nn, scan, serial, train_step, tie
from nodejax.struct import Struct


# these are tie's foundational tests.


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


def tile(tree, n):
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), tree)


# === archetype 1: tied autoencoder (shared layer weights, summed grads) ===

K = 3


def Encoder(k):
    def param(rng):
        return Struct(weight=jax.random.normal(rng.next(), (k,)))
    def apply(param, input):
        return param.weight @ input          # k -> scalar code
    return node_def(apply, param=param, name='enc')


def Decoder(k):
    def param(rng):
        return Struct(weight=jax.random.normal(rng.next(), (k,)))
    def apply(param, input):
        return param.weight * input          # scalar -> k, SAME weights
    return node_def(apply, param=param, name='dec')


def tied_autoencoder():
    return tie(Encoder(K) >> Decoder(K), 'enc', 'dec')


def test_single_copy_survives_jax_boundaries():
    """One weight array in the whole composite param — and because sharing
    is structural, no jit/tree_map boundary can lose it."""
    node = tied_autoencoder().parameterize(rng=jax.random.PRNGKey(0))
    assert len(jax.tree.leaves(node.param)) == 1
    assert node.param.dec == ()

    roundtrip = jax.tree.map(lambda x: x + 0.0, node)
    assert len(jax.tree.leaves(roundtrip.param)) == 1

    out = jax.jit(lambda n, x: n.apply(x))(node, jnp.ones(K))
    assert out.shape == (K,)


def test_gradients_accumulate_across_uses():
    """The chain rule does the bookkeeping: the tied gradient equals the
    SUM of the untied per-use gradients at the same values."""
    w = jnp.array([0.5, -1.0, 2.0])
    x = jnp.array([1.0, 2.0, 3.0])

    # ready-made param pytrees enter via bind; bundles carry recipes
    tied = tied_autoencoder().bind(Struct(enc=Struct(weight=w), dec=()))
    untied = (Encoder(K) >> Decoder(K)).bind(
        Struct(enc=Struct(weight=w), dec=Struct(weight=w)))

    f = lambda n: jnp.sum(n.apply(x))
    g_tied = jax.grad(f)(tied).param.enc.weight
    g_untied = jax.grad(f)(untied).param
    assert jnp.allclose(g_tied, g_untied.enc.weight + g_untied.dec.weight)


def test_tied_training_cannot_diverge():
    """train_step on the tied def: the optimizer sees one copy, so the
    encoder/decoder weights are identical forever, by construction."""
    auto = tied_autoencoder()
    trainer = train_step(auto, mse, optax.adam(0.05))

    u = jnp.array([0.6, -0.8, 0.0])          # unit direction to reconstruct
    state = trainer.init(model=auto.parameterize(rng=jax.random.PRNGKey(1)).param)
    steps = 400
    final, losses = trainer.scan(state, Struct(input=tile(u, steps), target=tile(u, steps)))

    assert losses[-1] < 1e-3                  # rank-1 reconstruction learned
    assert final.model.dec == ()              # still exactly one copy


def test_alias_parameterization_rejected():
    with pytest.raises(TypeError, match='dec'):
        tied_autoencoder().parameterize(enc=Struct(weight=jnp.ones(K)),
                                        dec=Struct(weight=jnp.ones(K)))


# === archetype 2: tied embeddings (one table, two views) ===

VOCAB, DIM = 8, 4


def test_tied_embeddings_are_one_object():
    """One table serves both views; training the round trip toward
    identity moves embed and unembed together, because there is nothing
    else to move."""
    lm = tie(nn.Embed(VOCAB, DIM) >> nn.Unembed(VOCAB, DIM), 'embed', 'unembed')
    ids = jnp.arange(VOCAB)

    node = lm.parameterize(rng=jax.random.PRNGKey(0))
    assert len(jax.tree.leaves(node.param)) == 1
    assert node.param.unembed == ()

    def xent(logits, target):
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(VOCAB), target])

    trainer = train_step(lm, xent, optax.adam(0.1))
    final, losses = trainer.scan(trainer.init(model=node.param),
                                 Struct(input=tile(ids, 200), target=tile(ids, 200)))
    assert losses[-1] < 0.1                          # round trip learned
    assert final.model.unembed == ()                 # still one copy


def test_tie_rejects_leaves_and_unknown_members():
    """tie rewires member param slots, so a memberless def and a name that
    is no member both fail at construction."""
    from nodejax.control import Gain
    with pytest.raises(TypeError, match='has none'):
        tie(Gain(), 'enc', 'dec')
    pipe = serial(enc=Gain(), dec=Gain())
    with pytest.raises(TypeError, match='name no member'):
        tie(pipe, 'enc', 'deocder')
