"""``train_step(trainable=...)``: the optimizer holds and the gradient reaches
the selected parameters only; the rest are constants of the step."""

import jax
import jax.numpy as jnp
import optax

from nodejax import Composite, Leaf, Struct, node, train_step


@node
def Gain() -> Leaf:
    def param(scale=2.0):
        return jnp.asarray(scale)

    return Leaf(lambda param, input: param * input, param=param)


@node
def TwoGains() -> Composite:
    members = Composite(head=Gain(), tail=Gain())

    def apply(self, input):
        return self.tail(self.head(input))

    return members(apply)


def squared(output, target):
    return jnp.mean((output - target) ** 2)


def run(trainable, tx):
    bound = train_step(TwoGains().parameterize(), squared, tx, trainable=trainable).initialize()
    for _ in range(5):
        bound, _ = bound.apply(input=jnp.ones(()), target=jnp.asarray(1.0))
    return bound


def test_selected_member_trains_and_the_other_stays_put() -> None:
    bound = run('head', optax.adamw(0.1, weight_decay=0.5))
    held = bound.state.opt.params.model
    assert set(held.__keys__) == {'head'}
    assert not jnp.allclose(held.head, 2.0)
    trained = bound.trained()
    assert jnp.allclose(trained.param.tail, 2.0)
    assert jnp.allclose(trained.param.head, held.head)


def test_default_trains_everything() -> None:
    bound = run(None, optax.adam(0.1))
    held = bound.state.opt.params.model
    assert set(held.__keys__) == {'head', 'tail'}
    assert not jnp.allclose(held.head, 2.0)
    assert not jnp.allclose(held.tail, 2.0)


def test_predicate_selects_by_path() -> None:
    bound = run(lambda path: path.endswith('tail'), optax.adam(0.1))
    held = bound.state.opt.params.model
    assert set(held.__keys__) == {'tail'}
