"""Verifies that `ensemble(nn.Projection(), n)` builds a composable linear layer.

Because `ensemble` preserves the `Node` contract and input propagation resolves
fan-in through composition, an ensembled scalar projection acts numerically as a
dense layer and composes directly with `>>`, `batch`, `train_step`, and `trained`.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import nn, trained, ensemble, batch, train_step


def mse(prediction, target):
    return jnp.mean((prediction - target) ** 2)


def test_linear_is_an_ensemble_of_projections():
    """Verifies that `ensemble(nn.Projection(), n)` matches a dense linear map and infers fan-in from input."""
    linear = ensemble(nn.Projection(), n=3)
    node = linear.with_input(jnp.zeros(4)).parameterize(rng=jax.random.PRNGKey(0))

    assert node.param.w.shape == (3, 4) and node.param.b.shape == (3,)

    x = jnp.array([1.0, -2.0, 0.5, 3.0])
    y = node.apply(x)
    assert y.shape == (3,)
    assert jnp.allclose(y, node.param.w @ x + node.param.b)

    wide = linear.with_input(jnp.zeros(7)).parameterize(rng=jax.random.PRNGKey(0))
    assert wide.param.w.shape == (3, 7)


def test_the_ensemble_layer_trains_like_any_layer():
    """Verifies that a batched ensemble of projections trains under `train_step` to recover a target map."""
    w_true = jnp.array([[1.0, -2.0, 0.5, 3.0],
                       [0.0, 1.0, -1.0, 2.0],
                       [2.0, 0.0, 1.0, -1.0]])

    model = batch(ensemble(nn.Projection(), n=3)).with_input(jnp.zeros((16, 4))).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, mse, optax.adam(0.05))

    xs = jax.random.normal(jax.random.PRNGKey(1), (300, 16, 4))
    final, aux = trained(trainer).apply(input=xs, target=jnp.einsum('sbi,oi->sbo', xs, w_true))

    assert aux.loss[-1] < 1e-3 * aux.loss[0]
    from nodejax import PSNode
    assert isinstance(final, PSNode)
    assert jnp.allclose(final.param.w, w_true, atol=0.05)


def test_an_mlp_is_units_all_the_way_down():
    """Verifies that an MLP built solely from `nn.Projection` units and `ensemble` fits a nonlinear target."""
    mlp = ensemble(nn.Projection(), n=16) >> nn.gelu >> nn.Projection()
    model = mlp.with_input(jnp.zeros(3)).parameterize(rng=jax.random.PRNGKey(0))

    trainer = train_step(batch(model).initialize(), mse, optax.adam(0.01))
    xs = jax.random.normal(jax.random.PRNGKey(1), (400, 32, 3))
    ys = jnp.sin(2.0 * xs[..., 0]) + 0.5 * xs[..., 1] * xs[..., 2]
    final, aux = trained(trainer).apply(input=xs, target=ys)

    assert aux.loss[-1] < 0.05 * aux.loss[0]
