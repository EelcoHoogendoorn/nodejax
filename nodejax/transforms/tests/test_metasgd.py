"""metasgd: finetune with per-parameter step sizes as meta-params."""

import jax
import jax.numpy as jnp
import optax

from nodejax.transforms.train_step import learned_sgd
from nodejax import Node, finetune, trained, scan, Leaf, batch, train_step, KeyStream
from nodejax.struct import Struct


def Scale() -> Node:
    def param(rng: KeyStream) -> Struct:
        return Struct(scale=jax.random.normal(rng.next(), ()))

    def apply(param, input):
        return param.scale * input

    return Leaf(apply, param=param, name='scale')


def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)


def test_metasgd_learns_init_and_rates():
    """Task family y = a*x: the meta-loop learns an init AND per-param
    step sizes such that a few inner steps identify each task's a."""
    model = Scale().parameterize(rng=jax.random.PRNGKey(0)).initialize()
    adapt = finetune(train_step(model, mse, learned_sgd(0.1)))
    trainer = train_step(batch(adapt).initialize(), mse, optax.adam(0.05))

    steps, tasks, k = 300, 4, 3
    rng = jax.random.PRNGKey(1)
    a = jax.random.uniform(rng, (steps, tasks), minval=-2.0, maxval=2.0)
    ones = jnp.ones((steps, tasks, k))
    sequence = Struct(
        input=Struct(support=Struct(input=ones, target=a[:, :, None] * ones),
                     query=jnp.full((steps, tasks), 2.0)),
        target=2.0 * a)
    final, aux = trained(trainer).apply(bundle=sequence)

    assert jnp.all(jnp.isfinite(aux.loss))
    assert jnp.mean(aux.loss[-20:]) < 0.1 * jnp.mean(aux.loss[:20])

    # held-out task: k support points identify an unseen a
    a_new = 1.3
    out = batch(adapt).node.bind(final.param).apply(
        support=Struct(input=jnp.ones((1, k)), target=jnp.full((1, k), a_new)),
        query=jnp.full((1,), 2.0))
    assert jnp.allclose(out, 2.0 * a_new, atol=0.1)

    # the step sizes are meta-params and moved off their starting value
    assert not jnp.allclose(final.param.opt.scale, 0.1)
