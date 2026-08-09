"""metasgd: finetune with per-parameter step sizes as meta-params."""

import jax
import jax.numpy as jnp
import optax

from nodejax import node_def, metasgd, batch, train_step, KeyStream
from nodejax.struct import Struct


def Scale():
    def param(rng: KeyStream) -> Struct:
        return Struct(scale=jax.random.normal(rng.next(), ()))

    def apply(param, input):
        return param.scale * input

    return node_def(apply, param=param, name='scale')


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


def test_metasgd_learns_init_and_rates():
    """Task family y = a*x: the meta-loop learns an init AND per-param
    step sizes such that a few inner steps identify each task's a."""
    adapt = metasgd(Scale(), mse, lr0=0.1)
    trainer = train_step(batch(adapt), mse, optax.adam(0.05))
    model = batch(adapt).parameterize(rng=jax.random.PRNGKey(0))

    steps, tasks, k = 300, 4, 3
    rng = jax.random.PRNGKey(1)
    a = jax.random.uniform(rng, (steps, tasks), minval=-2.0, maxval=2.0)
    ones = jnp.ones((steps, tasks, k))
    stream = Struct(
        input=Struct(support=Struct(input=ones, target=a[:, :, None] * ones),
                     query=jnp.full((steps, tasks), 2.0)),
        target=2.0 * a)
    final, losses = trainer.scan(trainer.init(model=model.param), stream)

    assert jnp.all(jnp.isfinite(losses))
    assert jnp.mean(losses[-20:]) < 0.1 * jnp.mean(losses[:20])

    # held-out task: k support points identify an unseen a
    a_new = 1.3
    out = batch(adapt).apply(final.model,
        Struct(support=Struct(input=jnp.ones((1, k)), target=jnp.full((1, k), a_new)),
               query=jnp.full((1,), 2.0)))
    assert jnp.allclose(out, 2.0 * a_new, atol=0.1)

    # the step sizes are meta-params and moved off their seed
    assert not jnp.allclose(final.model.lr.scale, 0.1)