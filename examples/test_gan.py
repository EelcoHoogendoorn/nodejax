"""Simultaneous adversarial training of a 1D Gaussian GAN built from two nested `train_step` trainers.

`GAN` is a two-member `Composite` pairing a discriminator trainer on real-versus-fake batches with
a generator trainer wrapping `fooled`, a `Wrapper` that evaluates the discriminator at critic
weights passed as an explicit `critic` input argument. Because `train_step` differentiates only
with respect to its wrapped model's parameters, passing `critic` as input leaves discriminator
weights untouched during the generator step without detaching gradients, keeping outer
meta-gradients open across unrolled rounds.

Latent samples enter `Generator` through `Noise`, which draws from the per-step `rng` stream
rather than taking a data array. Unrolling `GAN` with `.scan` advances both optimizers
simultaneously and collects per-round losses under `Aux` keyed by member name, composing directly
with `ensemble` for parallel games and `train_step` for meta-learning optimizer learning rates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import (
    Composite, Leaf, Node, PNode, PyTree, Wrapper, nn, node, split_aux, trained, train_step,
)

BATCH, ZDIM, HIDDEN = 128, 4, 16
MU, SIGMA = 3.0, 0.5


@node
def Noise() -> PNode:
    """Draws a `(BATCH, ZDIM)` standard normal latent batch from the `apply`-time `rng` stream."""
    def apply(rng) -> jax.Array:
        return jax.random.normal(rng.next(), (BATCH, ZDIM))

    return Leaf(apply)


def mlp() -> Node:
    """Builds a two-layer `tanh` MLP mapping inputs to scalar outputs with lazy fan-in inference."""
    return nn.Linear(HIDDEN) >> nn.tanh >> nn.Linear(1)


def Generator() -> Node:
    """Composes `Noise` with `mlp` so the latent output shape resolves downstream linear fan-ins."""
    return Noise() >> mlp()


def Discriminator() -> Node:
    """Builds an `mlp` critic sized to `(BATCH, 1)` scalar samples."""
    return mlp().with_input(jnp.zeros((BATCH, 1)))


def bce(logits: jax.Array, target: jax.Array) -> jax.Array:
    """Computes mean sigmoid binary cross-entropy loss."""
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, target))


@node
def GAN(discriminator_opt: optax.GradientTransformation,
        generator_opt: optax.GradientTransformation) -> Node:
    """Assembles one simultaneous adversarial training round over `Generator` and `Discriminator`.

    Each `apply` step updates both nested `train_step` members against each other's previous-round
    weights read from state, uses scalar `tick` inputs to set the `.scan` horizon, and emits
    per-round losses via `Aux`.
    """
    generator, discriminator = Generator(), Discriminator()

    def fooled(self, critic: PyTree) -> jax.Array:
        # Evaluating the critic at input-supplied weights restricts inner gradients to `self.generator`.
        return discriminator.apply(critic, self.generator())

    fooled = Wrapper(generator=generator)(fooled, name='fooled')
    members = Composite(
        discriminator=train_step(discriminator, bce, discriminator_opt),
        generator=train_step(fooled, bce, generator_opt),
    )
    real_labels = jnp.concatenate(
        [jnp.ones((BATCH, 1)), jnp.zeros((BATCH, 1))])

    def apply(self, tick: jax.Array, rng) -> None:
        # `self.discriminator` updates on real samples paired with fakes from the previous generator weights.
        real = MU + SIGMA * jax.random.normal(rng.next(), (BATCH, 1))
        fake = generator.apply(self.generator.state.opt.params.model,
                               rng=rng.next())
        self.discriminator(input=jnp.concatenate([real, fake]),
                           target=real_labels)
        # `self.generator` updates against the previous round's critic weights passed via `critic`.
        self.generator(critic=self.discriminator.state.opt.params.model,
                       target=jnp.ones((BATCH, 1)))

    return members(apply, name='gan')


def test_gan_learns_a_gaussian() -> None:
    """Trains `GAN` across scanned 500-round chunks and verifies convergence to the target Gaussian."""
    gan = GAN(optax.adam(2e-3), optax.adam(5e-4))
    generator = Generator()
    game = gan.parameterize(rng=jax.random.PRNGKey(0)).initialize()

    log = []
    for chunk in range(6):
        # `.scan` splits one chunk key across the 500 rounds specified by `tick`.
        game, output = game.scan(rng=jax.random.PRNGKey(10 + chunk),
                                 tick=jnp.zeros(500))
        samples_output, losses = split_aux(output)

        samples = generator.apply(game.state.generator.opt.params.model,
                                  rng=jax.random.PRNGKey(99))
        log.append(Struct(mean=float(jnp.mean(samples)), std=float(jnp.std(samples)),
                          discriminator=float(losses.discriminator.loss[-1]),
                          generator=float(losses.generator.loss[-1])))
        print(f"[gan] chunk {chunk}: mean {log[-1].mean:+.2f} std {log[-1].std:.2f} "
              f"(target {MU} / {SIGMA})  d {log[-1].discriminator:.2f} "
              f"g {log[-1].generator:.2f}")

    # Averaging the trailing three chunks scores the equilibrium orbit rather than one step.
    tail_mean = sum(s.mean for s in log[-3:]) / 3
    tail_std = sum(s.std for s in log[-3:]) / 3
    assert abs(tail_mean - MU) < 0.3
    assert 0.2 < tail_std < 1.0
    assert log[-1].discriminator > 0.3


def test_gan_population_by_ensemble() -> None:
    """Runs four independent `GAN` games in parallel under `ensemble` and verifies convergence."""
    from nodejax import ensemble

    gan = GAN(optax.adam(2e-3), optax.adam(5e-4))
    generator = Generator()
    tournament = ensemble(gan, n=4).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()

    for chunk in range(4):
        tournament, output = tournament.scan(rng=jax.random.PRNGKey(50 + chunk),
                                             tick=jnp.zeros(500))
    output, losses = split_aux(output)
    assert losses.discriminator.loss.shape == (500, 4)

    means = jax.vmap(
        lambda member: jnp.mean(generator.apply(
            member, rng=jax.random.PRNGKey(99))))(
        tournament.state.generator.opt.params.model)
    assert means.shape == (4,)
    assert jnp.unique(means).size == 4
    assert jnp.all(jnp.abs(means - MU) < 0.5)


def test_meta_learn_the_learning_rates() -> None:
    """Optimizes log-space generator and discriminator learning rates through unrolled `GAN` games."""
    from nodejax import tile
    ROUNDS, META_STEPS = 60, 100

    def Score() -> Node:
        def param(g_lr: float, d_lr: float) -> Struct:
            return Struct(log_g=jnp.log(jnp.asarray(g_lr)),
                          log_d=jnp.log(jnp.asarray(d_lr)))

        def apply(param, rng) -> Struct:
            gan = GAN(optax.adam(jnp.exp(param.log_d)),
                      optax.adam(jnp.exp(param.log_g)))
            generator = Generator()
            # `.scan` unrolls `ROUNDS` adversarial updates from fresh initialization before sampling.
            game = gan.parameterize(rng=rng.next()).initialize()
            game, _ = game.scan(rng=rng.next(), tick=jnp.zeros(ROUNDS))
            samples = generator.apply(game.state.generator.opt.params.model,
                                      rng=rng.next())
            return Struct(mean=jnp.mean(samples), std=jnp.std(samples))

        return Leaf(apply, param=param, name='score')

    def dist(out: Struct, target: Struct) -> jax.Array:
        return (out.mean - target.mean) ** 2 + (out.std - target.std) ** 2

    score = Score()
    lr0 = score.parameterize(g_lr=1e-4, d_lr=1e-4).param
    meta = train_step(score.bind(lr0), dist, optax.adam(0.2))

    # `trained(meta)` splits one sequence key across `META_STEPS` outer updates.
    final, aux = jax.jit(trained(meta).apply)(
        rng=jax.random.PRNGKey(3),
        target=tile(Struct(mean=jnp.asarray(MU), std=jnp.asarray(SIGMA)),
                    META_STEPS))
    meta_losses = aux.loss

    g_lr, d_lr = jnp.exp(final.param.log_g), jnp.exp(final.param.log_d)
    print(f"[meta] lrs 1e-4 -> g {g_lr:.2e}, d {d_lr:.2e} | "
          f"score {meta_losses[0]:.3f} -> {meta_losses[-1]:.3f}")
    assert g_lr > 5e-4 and d_lr > 5e-4

    # Shared probe keys compare the learned learning rates directly against the initial rates.
    probe = jax.random.PRNGKey(77)
    target = Struct(mean=jnp.asarray(MU), std=jnp.asarray(SIGMA))
    before = dist(score.apply(lr0, rng=probe), target)
    after = dist(score.apply(final.param, rng=probe), target)
    assert after < 0.3 * before
