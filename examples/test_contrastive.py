"""Multi-view InfoNCE contrastive learning over the named `'batch'` collective axis.

    encoder      one identity's views -> embeddings      per-identity
    infonce      those embeddings -> scalar loss         all_gather('batch')
    batch(...)   binds the 'batch' axis across identities

`Encoder` maps one identity's views to L2-normalized embeddings without referencing a batch axis.
`InfoNCE` gathers all candidate views across `'batch'` via `jax.lax.all_gather` and identifies each
member's own views with `jax.lax.axis_index('batch')` to compute the log-ratio of positive to
candidate similarities.

Because `train_step` evaluates `loss_fn` outside the `batch` `vmap`, cross-sample comparisons live
inside the batched `Node` pipeline (`batch(Encoder() >> InfoNCE())`) while `mean_score` averages
the per-identity losses without an external target array.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from nodejax import PNode, node, trained, Leaf, batch, unbatched, scan, train_step, nn, Node
from nodejax.struct import Struct

TRAIN_IDS, TEST_IDS = 64, 24
BATCH, VIEWS = 16, 4
LATENT, NUISANCE, HIDDEN, DIM = 6, 10, 32, 8
DIM_IN = LATENT + NUISANCE
SPREAD, TEMP, STEPS = 1.5, 0.1, 600

# Orthogonal mixing entangles latent identity and nuisance subspaces so unmixing generalizes to unseen identities.
MIX = jax.random.orthogonal(jax.random.PRNGKey(7), DIM_IN)


def identities(key: jax.Array, n: int) -> jax.Array:
    """Draws `n` standard normal latent identity vectors of width `LATENT`."""
    return jax.random.normal(key, (n, LATENT))


def views_of(key: jax.Array, codes: jax.Array) -> jax.Array:
    """Generates `VIEWS` noisy observations per identity code mixed through `MIX`."""
    shape = codes.shape[:-1] + (VIEWS,)
    nuisance = SPREAD * jax.random.normal(key, shape + (NUISANCE,))
    held = jnp.broadcast_to(codes[..., None, :], shape + (LATENT,))
    return jnp.concatenate([held, nuisance], axis=-1) @ MIX


def Encoder() -> Node:
    """Maps per-identity view arrays to L2-normalized embeddings of width `DIM`."""
    return nn.Linear(HIDDEN) >> nn.tanh >> nn.Linear(DIM) >> nn.L2Norm()


@node(name='infonce')
def InfoNCE(temp: float = TEMP) -> PNode:
    """Computes the multi-positive InfoNCE loss for one identity against all views gathered across `'batch'`.

    Assembles the batch candidate bank with `jax.lax.all_gather` and uses `jax.lax.axis_index('batch')`
    to mask self-matches while marking sibling views of the same identity as positives.
    """
    def apply(input) -> jax.Array:
        everyone = jax.lax.all_gather(input, 'batch')
        me = jax.lax.axis_index('batch')
        b, v = everyone.shape[0], everyone.shape[1]

        bank = everyone.reshape(b * v, -1)
        sim = input @ bank.T / temp
        rows = jnp.arange(b * v)
        myself = (me * v + jnp.arange(v))[:, None] == rows[None]
        my_identity = (rows // v == me)[None] & ~myself

        candidates = jnp.where(myself, -jnp.inf, sim)
        positives = jnp.where(my_identity, candidates, -jnp.inf)
        return jnp.mean(jax.nn.logsumexp(candidates, axis=-1)
                        - jax.nn.logsumexp(positives, axis=-1))

    return Leaf(apply)


def sequence_of(key: jax.Array, codes: jax.Array, steps: int) -> jax.Array:
    """Samples `steps` batches of identity codes and draws fresh noisy views for each batch."""
    pick, draw = jax.random.split(key)
    rows = jax.random.randint(pick, (steps, BATCH), 0, codes.shape[0])
    return views_of(draw, codes[rows])


def mean_score(per_identity: jax.Array) -> jax.Array:
    """Averages the per-identity InfoNCE losses emitted by the batched pipeline."""
    return jnp.mean(per_identity)


def retrieval_accuracy(param: Struct, codes: jax.Array, key: jax.Array) -> jax.Array:
    """Measures nearest-neighbor identity retrieval accuracy of `Encoder` on fresh views without `batch`."""
    enc = Encoder()
    v = views_of(key, codes)
    emb = enc.apply(param, v)
    query, gallery = emb[:, 0], emb[:, 1:].reshape(-1, DIM)
    owner = jnp.repeat(jnp.arange(codes.shape[0]), VIEWS - 1)
    nearest = jnp.argmin(jnp.linalg.norm(query[:, None] - gallery[None], axis=-1), 1)
    return jnp.mean(owner[nearest] == jnp.arange(codes.shape[0]))


def test_the_batch_axis_carries_the_objective() -> None:
    """Trains `batch(Encoder() >> InfoNCE())` and verifies non-collapsed loss and unseen-identity retrieval."""
    k = jax.random.split(jax.random.PRNGKey(0), 5)
    train_c, test_c = identities(k[0], TRAIN_IDS), identities(k[1], TEST_IDS)

    model = Encoder() >> InfoNCE()
    batched = batch(model).with_input(jnp.zeros((BATCH, VIEWS, DIM_IN)))
    start = batched.parameterize(rng=k[2]).param

    trainer = train_step(batched, mean_score, optax.adam(3e-3))
    trainer = trainer.bind(Struct(objective=Struct(model=start)))
    final, aux = trained(trainer).apply(input=sequence_of(k[3], train_c, STEPS))

    enc_before = start.without('infonce')
    enc_after = final.param.without('infonce')
    before = retrieval_accuracy(enc_before, test_c, k[4])
    after = retrieval_accuracy(enc_after, test_c, k[4])
    print(f"[contrastive] loss {aux.loss[0]:.3f} -> {aux.loss[-1]:.3f} | "
          f"retrieval on unseen identities {before:.2f} -> {after:.2f}")

    # A collapsed encoder makes all non-self candidates equally likely and scores `chance`.
    chance = jnp.log((BATCH * VIEWS - 1) / (VIEWS - 1))
    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.2 * chance, (aux.loss[-1], chance)

    # Unmixing transfers to held-out identities never seen during training.
    assert after > 0.9 and after > before + 0.5


def test_the_objective_is_vacuous_without_neighbours() -> None:
    """Verifies that `unbatched` satisfies `'batch'` over a single identity where InfoNCE evaluates to zero."""
    model = Encoder() >> InfoNCE()
    solo = unbatched(model).with_input(jnp.zeros((VIEWS, DIM_IN)))
    node = solo.parameterize(rng=jax.random.PRNGKey(1))

    one = views_of(jax.random.PRNGKey(2), identities(jax.random.PRNGKey(3), 1))[0]
    assert jnp.allclose(node.apply(one), 0.0)

    # Binding the same encoder parameters into a multi-identity batch produces a positive contrastive loss.
    batched = batch(model).with_input(jnp.zeros((BATCH, VIEWS, DIM_IN)))
    many = views_of(jax.random.PRNGKey(4), identities(jax.random.PRNGKey(3), BATCH))
    assert jnp.mean(batched.bind(node.param).apply(many)) > 0.0
