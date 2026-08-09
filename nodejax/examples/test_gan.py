"""A tiny GAN: two trainers, one scanned round, params spliced as data.

Adversarial training needs no framework support:

- the discriminator's trainer is train_step on the discriminator,
  fed real-vs-fake batches with labels;
- the generator's trainer is train_step on a def whose apply runs
  gen then critic, with the critic's params arriving on the INPUT
  channel: params are data, and the inner grad differentiates the
  model only, so the critic sits outside it with no detach anywhere —
  which also leaves the OUTER path open, so meta-gradients (see the
  last test) flow through the critic while inner training never
  moves it;
- each round feeds the live discriminator params into that input slot.
  Cross-network flow is input plumbing, not state surgery.

Generator noise enters as declared apply-rng: the generator's input
bundle is ONE key (hoisted onto the pipe boundary), drawn from inside.
The round IS a node: apply(state, rng) -> (state, losses) with both
trainers' states nested inside, entropy per round from declared
apply-rng, params drawn in init from its seed key. It scans, so the
host loop reads distribution stats between fused chunks (the
train_loop pattern).
"""

import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import node_def, derive, train_step, nn

BATCH, ZDIM, HIDDEN = 128, 4, 16
MU, SIGMA = 3.0, 0.5

def Noise():
    """The latent source: BATCH draws from the declared apply-rng key."""
    def apply(rng):
        return jax.random.normal(rng.next(), (BATCH, ZDIM))
    return node_def(apply, name='noise')


def mlp():
    """Both nets are the same stock sandwich; fan-ins from the shape walk."""
    return nn.Linear(HIDDEN) >> nn.tanh >> nn.Linear(1)


def Generator():
    """noise head >> mlp; resolving against the one-key input bundle lets
    the walk derive the linears' fan-ins from the noise head's output."""
    return (Noise() >> mlp()).with_input(Struct(rng=jax.random.PRNGKey(0)))


def Discriminator():
    """sample -> realness logit; fan-in from the sample shape."""
    return mlp().with_input(jnp.zeros((BATCH, 1)))


def bce(logits, labels):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


def GAN(d_opt, g_opt):
    """The adversarial round as ONE cyclic node, wired byol-style: both
    trainers' states nest in its state, the live critic crosses between
    them on the G step's INPUT channel, and the draws of a round come
    off the one apply-rng key."""
    gen, disc = Generator(), Discriminator()

    def fooled(param, critic, rng):
        # the generator's own params; the critic's params are INPUT data
        return disc.apply(critic, gen.apply(param, rng=rng.next()))

    fooled = derive(gen, apply=fooled, name='fooled')
    d_trainer = train_step(disc, bce, d_opt)
    g_trainer = train_step(fooled, bce, g_opt)
    real_labels = jnp.concatenate([jnp.ones((BATCH, 1)), jnp.zeros((BATCH, 1))])

    def init(rng):
        d_state = d_trainer.init(model=disc.parameterize(rng=rng.next()).param)
        g_state = g_trainer.init(model=gen.parameterize(rng=rng.next()).param)
        return Struct(g=g_state, d=d_state)

    def apply(state, rng):
        # D step: real vs the CURRENT generator's fakes
        real = MU + SIGMA * jax.random.normal(rng.next(), (BATCH, 1))
        fake = gen.apply(state.g.model, rng=rng.next())
        d_state, d_loss = d_trainer.apply(
            state.d, Struct(input=jnp.concatenate([real, fake]), target=real_labels))

        # G step: the live critic rides the input bundle
        g_state, g_loss = g_trainer.apply(
            state.g, Struct(input=Struct(critic=d_state.model, rng=rng.next()),
                            target=jnp.ones((BATCH, 1))))
        return Struct(g=g_state, d=d_state), Struct(d=d_loss, g=g_loss)

    return node_def(apply, init=init, name='gan'), gen


def test_gan_learns_a_gaussian():
    gan, gen = GAN(optax.adam(2e-3), optax.adam(5e-4))
    state = gan.init(rng=jax.random.PRNGKey(0))
    run_chunk = jax.jit(gan.scan)

    log = []
    for chunk in range(6):
        ks = jax.random.split(jax.random.PRNGKey(10 + chunk), 500)
        state, losses = run_chunk(state, Struct(rng=ks))

        samples = gen.apply(state.g.model, rng=jax.random.PRNGKey(99))
        log.append(Struct(mean=float(jnp.mean(samples)), std=float(jnp.std(samples)),
                          d=float(losses.d[-1]), g=float(losses.g[-1])))
        print(f"[gan] chunk {chunk}: mean {log[-1].mean:+.2f} std {log[-1].std:.2f} "
              f"(target {MU} / {SIGMA})  d {log[-1].d:.2f} g {log[-1].g:.2f}")

    # a GAN equilibrium oscillates; judge the orbit, not one checkpoint
    tail_mean = sum(s.mean for s in log[-3:]) / 3
    tail_std = sum(s.std for s in log[-3:]) / 3
    assert abs(tail_mean - MU) < 0.3                  # orbiting the mode
    assert 0.2 < tail_std < 1.0                       # spread, not collapse
    # the equilibrium is adversarial: the discriminator has not won
    assert log[-1].d > 0.3


def test_gan_population_by_ensemble():
    """What the round-as-node buys: transforms written for ANY node apply
    to the adversarial game verbatim. ensemble runs n independent GANs —
    split init keys, split per-round keys, stacked trainer states (adam
    moments and all) — under one vmap, knowing nothing about GANs."""
    from nodejax import ensemble

    gan, gen = GAN(optax.adam(2e-3), optax.adam(5e-4))
    population = ensemble(gan.ndef, n=4).parameterize()
    state = population.init(rng=jax.random.PRNGKey(0))
    run_chunk = jax.jit(population.scan)

    for chunk in range(4):
        ks = jax.random.split(jax.random.PRNGKey(50 + chunk), 500)
        state, losses = run_chunk(state, Struct(rng=ks))
    assert losses.d.shape == (500, 4)                 # per-round, per-member

    means = jax.vmap(lambda p: jnp.mean(gen.apply(p, rng=jax.random.PRNGKey(99))))(
        state.g.model)
    assert means.shape == (4,)
    assert jnp.unique(means).size == 4                # four independent games
    assert jnp.all(jnp.abs(means - MU) < 0.5)         # all found the mode


def test_meta_learn_the_learning_rates():
    """Meta-learning over the adversarial process: the two learning rates
    move to the PARAM channel (log-space), each meta-step replays a short
    game from scratch and scores the generator's output distribution, and
    the meta-gradient flows through every unrolled round — both adams
    included. The tower is stock: train_step over the game-scoring node."""
    from nodejax.util import tile
    ROUNDS, META_STEPS = 60, 100

    def Score():
        def param(g_lr, d_lr):
            return Struct(log_g=jnp.log(jnp.asarray(g_lr)),
                          log_d=jnp.log(jnp.asarray(d_lr)))

        def apply(param, rng):
            gan, gen = GAN(optax.adam(jnp.exp(param.log_d)),
                               optax.adam(jnp.exp(param.log_g)))
            state = gan.init(rng=rng.next())
            state, _ = gan.scan(state, Struct(rng=jax.random.split(rng.next(), ROUNDS)))
            samples = gen.apply(state.g.model, rng=rng.next())
            return Struct(mean=jnp.mean(samples), std=jnp.std(samples))

        return node_def(apply, param=param, name='score')

    def dist(out, target):
        return (out.mean - target.mean) ** 2 + (out.std - target.std) ** 2

    score = Score()
    lr0 = score.parameterize(g_lr=1e-4, d_lr=1e-4).param     # far too slow to converge
    meta = train_step(score, dist, optax.adam(0.2))

    stream = Struct(input=Struct(rng=jax.random.split(jax.random.PRNGKey(3), META_STEPS)),
                    target=tile(Struct(mean=jnp.asarray(MU), std=jnp.asarray(SIGMA)),
                                META_STEPS))
    final, meta_losses = jax.jit(meta.scan)(meta.init(model=lr0), stream)

    g_lr, d_lr = jnp.exp(final.model.log_g), jnp.exp(final.model.log_d)
    print(f"[meta] lrs 1e-4 -> g {g_lr:.2e}, d {d_lr:.2e} | "
          f"score {meta_losses[0]:.3f} -> {meta_losses[-1]:.3f}")
    assert g_lr > 5e-4 and d_lr > 5e-4                # both rates grew an order

    # the learned rates beat the initial ones on the SAME game (one key)
    probe = jax.random.PRNGKey(77)
    target = Struct(mean=jnp.asarray(MU), std=jnp.asarray(SIGMA))
    before = dist(score.apply(lr0, rng=probe), target)
    after = dist(score.apply(final.model, rng=probe), target)
    assert after < 0.3 * before
