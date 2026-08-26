"""A tiny GAN: two trainers, one scanned round, params spliced as data.

Adversarial training needs no framework support:

- the discriminator's trainer is train_step on the discriminator,
  fed real-vs-fake batches with labels;
- the generator's trainer is train_step on a node whose apply runs
  gen then critic, with the critic's params arriving on the INPUT
  channel: params are data, and the inner grad differentiates the
  model only, so the critic sits outside it with no detach anywhere —
  which also leaves the OUTER path open, so meta-gradients (see the
  last test) flow through the critic while inner training never
  moves it;
- each round feeds the live discriminator params into that input slot.
  Cross-network flow is input plumbing, not state surgery.

Generator noise enters through declared apply-rng: its data input is empty,
and the generator's boundary key is routed to the noise head separately.
The round IS a node: its parameter role draws both models, its state holds
both train steps, and apply(state, rng) -> (state, losses) performs one
adversarial update. It scans, so the host loop reads distribution stats
between fused chunks (the train_loop pattern).
"""

import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import (
    Composite, Leaf, Node, Wrapper, nn, node, trained, train_step,
)

BATCH, ZDIM, HIDDEN = 128, 4, 16
MU, SIGMA = 3.0, 0.5

@node
def Noise() -> Node:
    """The latent source: BATCH draws from the declared apply-rng key."""
    def apply(rng):
        return jax.random.normal(rng.next(), (BATCH, ZDIM))
    return Leaf(apply)


def mlp() -> Node:
    """Both nets are the same stock sandwich; fan-ins from the shape walk."""
    return nn.Linear(HIDDEN) >> nn.tanh >> nn.Linear(1)


def Generator() -> Node:
    """noise head >> mlp; the empty-input source already resolves the pipe,
    whose shape walk derives the linears' fan-ins from the noise output."""
    return Noise() >> mlp()


def Discriminator() -> Node:
    """sample -> realness logit; fan-in from the sample shape."""
    return mlp().with_input(jnp.zeros((BATCH, 1)))


def bce(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


@node
def GAN(d_opt, g_opt) -> Node:
    """The adversarial round as one T3-authored composite."""
    gen, disc = Generator(), Discriminator()

    def fooled(self, critic):
        # the generator's own params; the critic's params are input data
        return disc.apply(critic, self.gen())

    fooled = Wrapper(gen=gen)(fooled, name='fooled')
    d_step = train_step(disc, bce, d_opt)
    g_step = train_step(fooled, bce, g_opt)
    real_labels = jnp.concatenate([jnp.ones((BATCH, 1)), jnp.zeros((BATCH, 1))])

    def param_fn(contract, rng):
        d, g = contract.members.d, contract.members.g
        return Struct(
            d=d.param(Struct(), rng.child(d.param_takes_rng)),
            g=g.param(Struct(), rng.child(g.param_takes_rng)),
        )

    def init_fn(contract, param, state_input, rng):
        d, g = contract.members.d, contract.members.g
        return Struct(
            d=d.init(
                param.d, state_input.d,
                rng.child(d.init_takes_rng)),
            g=g.init(
                param.g, state_input.g,
                rng.child(g.init_takes_rng)),
        )

    def apply_fn(contract, param, state, input, rng):
        # tick is the length-giving sequence and nothing else: entropy
        # cannot ride the xs, so a zero per round says how many rounds.
        d, g = contract.members.d, contract.members.g
        generator = g.members.model.members.gen

        # D step: real vs the current generator's fakes.
        real = MU + SIGMA * jax.random.normal(rng.next(), (BATCH, 1))
        _, fake = generator.apply(
            state.g.opt.params, (), Struct(),
            rng.child(generator.apply_takes_rng))
        d_state, (_, d_aux) = d.apply(
            param.d, state.d,
            Struct(
                input=jnp.concatenate([real, fake]),
                target=real_labels),
            rng.child(d.apply_takes_rng),
        )

        # G step: the live critic rides the input bundle
        g_state, (_, g_aux) = g.apply(
            param.g, state.g,
            Struct(
                input=d_state.opt.params,
                target=jnp.ones((BATCH, 1))),
            rng.child(g.apply_takes_rng),
        )
        return Struct(g=g_state, d=d_state), Struct(d=d_aux.loss, g=g_aux.loss)

    game = Composite(d=d_step, g=g_step).roles(
        apply_fn,
        param=param_fn,
        init=init_fn,
        name='gan',
        param_input=Struct(),
        apply_fields=('tick',),
    )
    return game, gen


def test_gan_learns_a_gaussian():
    gan, gen = GAN(optax.adam(2e-3), optax.adam(5e-4))
    game = gan.parameterize(rng=jax.random.PRNGKey(0)).initialize()

    log = []
    for chunk in range(6):
        # ONE key per chunk, split per round inside .scan; the tick says
        # how many
        game, losses = game.scan(rng=jax.random.PRNGKey(10 + chunk),
                                 tick=jnp.zeros(500))

        samples = gen.apply(game.state.g.opt.params, rng=jax.random.PRNGKey(99))
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
    tournament = ensemble(gan, n=4).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()

    for chunk in range(4):
        tournament, losses = tournament.scan(rng=jax.random.PRNGKey(50 + chunk),
                                             tick=jnp.zeros(500))
    assert losses.d.shape == (500, 4)                 # per-round, per-member

    means = jax.vmap(lambda p: jnp.mean(gen.apply(p, rng=jax.random.PRNGKey(99))))(
        tournament.state.g.opt.params)
    assert means.shape == (4,)
    assert jnp.unique(means).size == 4                # four independent games
    assert jnp.all(jnp.abs(means - MU) < 0.5)         # all found the mode


def test_meta_learn_the_learning_rates():
    """Meta-learning over the adversarial process: the two learning rates
    move to the PARAM channel (log-space), each meta-step replays a short
    game from scratch and scores the generator's output distribution, and
    the meta-gradient flows through every unrolled round — both adams
    included. The tower is stock: train_step over the game-scoring node."""
    from nodejax import tile
    ROUNDS, META_STEPS = 60, 100

    def Score():
        def param(g_lr, d_lr):
            return Struct(log_g=jnp.log(jnp.asarray(g_lr)),
                          log_d=jnp.log(jnp.asarray(d_lr)))

        def apply(param, rng):
            gan, gen = GAN(optax.adam(jnp.exp(param.log_d)),
                               optax.adam(jnp.exp(param.log_g)))
            # ONE key at the boundary, split per round inside .scan; the
            # tick is what says how many rounds there are, since entropy
            # cannot ride the xs (see the rng doctrine in scan)
            game = gan.parameterize(rng=rng.next()).initialize()
            game, _ = game.scan(rng=rng.next(), tick=jnp.zeros(ROUNDS))
            samples = gen.apply(game.state.g.opt.params, rng=rng.next())
            return Struct(mean=jnp.mean(samples), std=jnp.std(samples))

        return Leaf(apply, param=param, name='score')

    def dist(out, target):
        return (out.mean - target.mean) ** 2 + (out.std - target.std) ** 2

    score = Score()
    lr0 = score.parameterize(g_lr=1e-4, d_lr=1e-4).param     # far too slow to converge
    meta = train_step(score.bind(lr0), dist, optax.adam(0.2))   # lr0 is where it starts

    # ONE key rides the sequence: scan splits it per meta-step and the
    # trainer hands the drawing model its share; nothing rides the input
    final, aux = jax.jit(trained(meta).apply)(
        input=Struct(), rng=jax.random.PRNGKey(3),
        target=tile(Struct(mean=jnp.asarray(MU), std=jnp.asarray(SIGMA)),
                    META_STEPS))
    meta_losses = aux.loss

    g_lr, d_lr = jnp.exp(final.param.log_g), jnp.exp(final.param.log_d)
    print(f"[meta] lrs 1e-4 -> g {g_lr:.2e}, d {d_lr:.2e} | "
          f"score {meta_losses[0]:.3f} -> {meta_losses[-1]:.3f}")
    assert g_lr > 5e-4 and d_lr > 5e-4                # both rates grew an order

    # the learned rates beat the initial ones on the SAME game (one key)
    probe = jax.random.PRNGKey(77)
    target = Struct(mean=jnp.asarray(MU), std=jnp.asarray(SIGMA))
    before = dist(score.apply(lr0, rng=probe), target)
    after = dist(score.apply(final.param, rng=probe), target)
    assert after < 0.3 * before
