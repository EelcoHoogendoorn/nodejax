"""A contrastive objective: the batch axis as the SUBJECT, not a detail.

Every other named-axis example here is a normalizer, where the collective is
a mean and the interesting part is the state it accumulates. This one has no
state at all. The comparison IS the objective: each identity is scored against
every other identity in the batch, so the axis carries the thing being learned
rather than a running statistic of it.

    encoder      one identity's views -> embeddings      knows no batch
    infonce      those embeddings -> one scalar          all_gather('batch')
    batch(...)   binds the name, and only then is the objective defined

WHERE THE LOSS HAS TO LIVE. train_step's loss_fn runs OUTSIDE the vmap, on
whatever the batched node already returned, so it cannot see the axis and a
cross-sample objective cannot be written there. The comparison is a node
inside the batch, and the loss_fn is left with the averaging. That is not a
workaround: 'compare me against my neighbours' is a computation, and
computations are nodes. The trainer's target channel carries nothing here,
because the supervision is the batch's own composition.

Run directly:  python -m pytest examples/test_contrastive.py -s
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import node, trained, Leaf, batch, unbatched, scan, train_step, nn, Node
from nodejax.struct import Struct

TRAIN_IDS, TEST_IDS = 64, 24
BATCH, VIEWS = 16, 4
LATENT, NUISANCE, HIDDEN, DIM = 6, 10, 32, 8
DIM_IN = LATENT + NUISANCE
SPREAD, TEMP, STEPS = 1.5, 0.1, 600

# what makes the task LEARNABLE rather than memorizable: identity and nuisance
# live in different subspaces of the same observation, mixed by one fixed map.
# Undoing that mixing is a property of the map, not of any identity, so an
# encoder that finds it works on identities it has never seen.
MIX = jax.random.orthogonal(jax.random.PRNGKey(7), DIM_IN)


def identities(key: jax.Array, n: int):
    """n latent identity codes, one per identity: [n, LATENT]."""
    return jax.random.normal(key, (n, LATENT))


def views_of(key: jax.Array, codes: jax.Array):
    """VIEWS observations of each code: [..., VIEWS, DIM_IN].

    One view is its identity code and a fresh nuisance draw, side by side, run
    through the fixed mixing. SPREAD makes the nuisance the louder of the two,
    so the raw observation is dominated by what does NOT identify it and an
    untrained encoder retrieves poorly."""
    shape = codes.shape[:-1] + (VIEWS,)
    nuisance = SPREAD * jax.random.normal(key, shape + (NUISANCE,))
    held = jnp.broadcast_to(codes[..., None, :], shape + (LATENT,))
    return jnp.concatenate([held, nuisance], axis=-1) @ MIX


def Encoder() -> Node:
    """One identity's views in, its embeddings out. Written per-identity and
    with no idea a batch exists; the linears broadcast over the view axis."""
    return nn.Linear(HIDDEN) >> nn.tanh >> nn.Linear(DIM) >> nn.L2Norm()


@node(name='infonce')
def InfoNCE(temp: float=TEMP) -> Node:
    """The cross-sample half, and the only place the axis name appears.

    For each of my views: the positives are my other views, the candidates are
    every view in the batch, and the loss is the log-ratio between them. The
    axis is what supplies the candidates. all_gather hands this identity the
    whole batch, and axis_index says which rows are its own, which is the one
    thing a member cannot work out from its own input.

    A ratio and not a margin, because a margin is minimized by collapse: send
    every embedding to one point and the hinge sits at exactly the margin and
    stops. Here collapse makes every candidate equally likely, which is the
    WORST score rather than a cheap one, so the objective has to separate
    identities to make progress.
    """
    def apply(input):                       # [VIEWS, DIM], ONE identity
        everyone = jax.lax.all_gather(input, 'batch')            # [B, VIEWS, DIM]
        me = jax.lax.axis_index('batch')
        b, v = everyone.shape[0], everyone.shape[1]

        bank = everyone.reshape(b * v, -1)                       # every view, once
        sim = input @ bank.T / temp                              # [VIEWS, B*VIEWS]
        rows = jnp.arange(b * v)
        myself = (me * v + jnp.arange(v))[:, None] == rows[None]  # exactly me
        my_identity = (rows // v == me)[None] & ~myself           # my OTHER views

        candidates = jnp.where(myself, -jnp.inf, sim)             # never match self
        positives = jnp.where(my_identity, candidates, -jnp.inf)
        return jnp.mean(jax.nn.logsumexp(candidates, axis=-1)
                        - jax.nn.logsumexp(positives, axis=-1))

    return Leaf(apply)


def sequence_of(key: jax.Array, codes: jax.Array, steps: int):
    """One batch of identities per step, fresh views every time."""
    pick, draw = jax.random.split(key)
    rows = jax.random.randint(pick, (steps, BATCH), 0, codes.shape[0])
    return views_of(draw, codes[rows])                      # [steps, B, V, DIM_IN]


def mean_score(per_identity, target: jax.Array) -> jax.Array:
    """The objective is target-free: the batched node composed its own
    supervision and already scored itself, so the loss only averages and
    the target slot's placeholder is ignored."""
    return jnp.mean(per_identity)


def retrieval_accuracy(param, codes: jax.Array, key: jax.Array):
    """Fresh views of unseen identities: does view 0 land nearest a view of
    its own identity? The encoder runs alone here, with no batch anywhere: the
    axis belonged to the objective and never to the model."""
    enc = Encoder()
    v = views_of(key, codes)                                # [N, VIEWS, DIM_IN]
    emb = enc.apply(param, v)                               # [N, VIEWS, DIM]
    query, gallery = emb[:, 0], emb[:, 1:].reshape(-1, DIM)
    owner = jnp.repeat(jnp.arange(codes.shape[0]), VIEWS - 1)
    nearest = jnp.argmin(jnp.linalg.norm(query[:, None] - gallery[None], axis=-1), 1)
    return jnp.mean(owner[nearest] == jnp.arange(codes.shape[0]))


def test_the_batch_axis_carries_the_objective():
    """The whole example: a model that knows no batch, an objective that is
    nothing without one, and a retrieval score that says it learned."""
    k = jax.random.split(jax.random.PRNGKey(0), 5)
    train_c, test_c = identities(k[0], TRAIN_IDS), identities(k[1], TEST_IDS)

    model = Encoder() >> InfoNCE()
    batched = batch(model).with_input(jnp.zeros((BATCH, VIEWS, DIM_IN)))
    start = batched.parameterize(rng=k[2]).param

    trainer = train_step(batched, mean_score, optax.adam(3e-3))
    trainer = trainer.bind(Struct(model=start))
    final, aux = trained(trainer).apply(
        input=sequence_of(k[3], train_c, STEPS),
        target=jnp.zeros(STEPS))                      # a placeholder: see mean_score

    enc_before = start.without('infonce')
    enc_after = final.param.without('infonce')
    before = retrieval_accuracy(enc_before, test_c, k[4])
    after = retrieval_accuracy(enc_after, test_c, k[4])
    print(f"[contrastive] loss {aux.loss[0]:.3f} -> {aux.loss[-1]:.3f} | "
          f"retrieval on unseen identities {before:.2f} -> {after:.2f}")

    # chance is where a COLLAPSED encoder sits: if every embedding is the same
    # point every candidate is equally likely, so beating it by this much is
    # the assertion with teeth, and a ratio against the initial loss is not
    chance = jnp.log((BATCH * VIEWS - 1) / (VIEWS - 1))
    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.2 * chance, (aux.loss[-1], chance)

    # and it generalizes, which is the point of mixing rather than separating:
    # these identities were never in any batch, and what transferred is the
    # map, not them
    assert after > 0.9 and after > before + 0.5


def test_the_objective_is_vacuous_without_neighbours():
    """What `unbatched` says here, and why it is not a mode flag.

    The node declares an axis need; unbatched satisfies it over a batch of
    one rather than letting the node ask whether it is batched. What comes
    back is the honest answer for a batch of one: nobody else is present, so
    there are no negatives, every hinge is slack, and the objective is zero.
    A collective over one row is well defined, not a special case.
    """
    model = Encoder() >> InfoNCE()
    solo = unbatched(model).with_input(jnp.zeros((VIEWS, DIM_IN)))
    node = solo.parameterize(rng=jax.random.PRNGKey(1))

    one = views_of(jax.random.PRNGKey(2), identities(jax.random.PRNGKey(3), 1))[0]
    assert jnp.allclose(node.apply(one), 0.0)

    # and the same params in a real batch score something, since the
    # neighbours are what the objective was about
    batched = batch(model).with_input(jnp.zeros((BATCH, VIEWS, DIM_IN)))
    many = views_of(jax.random.PRNGKey(4), identities(jax.random.PRNGKey(3), BATCH))
    assert jnp.mean(batched.bind(node.param).apply(many)) > 0.0
