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

BATCH, ZDIM, HIDDEN = 32, 4, 16
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
    """Runs `GAN` across scanned rounds and verifies finite losses and simultaneous weight updates."""
    gan = GAN(optax.adam(2e-3), optax.adam(5e-4))
    generator = Generator()
    initial = gan.parameterize(rng=jax.random.PRNGKey(0)).initialize()
    game = initial

    rounds = 3
    for chunk in range(2):
        # `.scan` splits one chunk key across the rounds specified by `tick`.
        game, output = game.scan(rng=jax.random.PRNGKey(10 + chunk),
                                 tick=jnp.zeros(rounds))
        _, losses = split_aux(output)
        assert losses.discriminator.loss.shape == (rounds,)
        assert losses.generator.loss.shape == (rounds,)
        assert jnp.all(jnp.isfinite(losses.discriminator.loss))
        assert jnp.all(jnp.isfinite(losses.generator.loss))

    assert not jax.tree.all(jax.tree.map(
        jnp.array_equal,
        game.state.generator.opt.params.model,
        initial.state.generator.opt.params.model,
    ))
    assert not jax.tree.all(jax.tree.map(
        jnp.array_equal,
        game.state.discriminator.opt.params.model,
        initial.state.discriminator.opt.params.model,
    ))

    samples = generator.apply(game.state.generator.opt.params.model,
                              rng=jax.random.PRNGKey(99))
    assert samples.shape == (BATCH, 1)
    assert jnp.all(jnp.isfinite(samples))


def test_gan_population_by_ensemble() -> None:
    """Runs four independent `GAN` games in parallel under `ensemble` and verifies distinct member trajectories."""
    from nodejax import ensemble

    gan = GAN(optax.adam(2e-3), optax.adam(5e-4))
    generator = Generator()
    tournament = ensemble(gan, n=4).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()

    rounds = 3
    tournament, output = tournament.scan(rng=jax.random.PRNGKey(50),
                                         tick=jnp.zeros(rounds))
    _, losses = split_aux(output)
    assert losses.discriminator.loss.shape == (rounds, 4)
    assert losses.generator.loss.shape == (rounds, 4)
    assert jnp.all(jnp.isfinite(losses.discriminator.loss))
    assert jnp.all(jnp.isfinite(losses.generator.loss))

    means = jax.vmap(
        lambda member: jnp.mean(generator.apply(
            member, rng=jax.random.PRNGKey(99))))(
        tournament.state.generator.opt.params.model)
    assert means.shape == (4,)
    assert jnp.all(jnp.isfinite(means))
    assert jnp.unique(means).size == 4


def test_meta_learn_the_learning_rates() -> None:
    """Differentiates log-space generator and discriminator learning rates through unrolled `GAN` games."""
    from nodejax import tile
    rounds, meta_steps = 3, 3

    def Score() -> Node:
        def param(g_lr: float, d_lr: float) -> Struct:
            return Struct(log_g=jnp.log(jnp.asarray(g_lr)),
                          log_d=jnp.log(jnp.asarray(d_lr)))

        def apply(param, rng) -> Struct:
            gan = GAN(optax.adam(jnp.exp(param.log_d)),
                      optax.adam(jnp.exp(param.log_g)))
            generator = Generator()
            # `.scan` unrolls `rounds` adversarial updates from fresh initialization before sampling.
            game = gan.parameterize(rng=rng.next()).initialize()
            game, _ = game.scan(rng=rng.next(), tick=jnp.zeros(rounds))
            samples = generator.apply(game.state.generator.opt.params.model,
                                      rng=rng.next())
            return Struct(mean=jnp.mean(samples), std=jnp.std(samples))

        return Leaf(apply, param=param, name='score')

    def dist(out: Struct, target: Struct) -> jax.Array:
        return (out.mean - target.mean) ** 2 + (out.std - target.std) ** 2

    score = Score()
    lr0 = score.parameterize(g_lr=1e-4, d_lr=1e-4).param
    meta = train_step(score.bind(lr0), dist, optax.adam(0.2))

    # `trained(meta)` splits one sequence key across `meta_steps` outer updates.
    final, aux = jax.jit(trained(meta).apply)(
        rng=jax.random.PRNGKey(3),
        target=tile(Struct(mean=jnp.asarray(MU), std=jnp.asarray(SIGMA)),
                    meta_steps))
    meta_losses = aux.loss

    assert meta_losses.shape == (meta_steps,)
    assert jnp.all(jnp.isfinite(meta_losses))
    assert not jnp.array_equal(final.param.log_g, lr0.log_g)
    assert not jnp.array_equal(final.param.log_d, lr0.log_d)

    probe = jax.random.PRNGKey(77)
    target = Struct(mean=jnp.asarray(MU), std=jnp.asarray(SIGMA))
    assert jnp.isfinite(dist(score.apply(final.param, rng=probe), target))
