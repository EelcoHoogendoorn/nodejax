"""Dogfooding the composition claim.

In jax, manually adding batch or ensemble axes to components is
considered an antipattern, as it underutilizes vmap.
Yet, we never see a linear layer written as
    linear = ensemble(Hyperplane(), n)
The pattern is possible in any jax framework and attractive in none of
them: everyone can vmap a unit's constructor into stacked params (the
comparison tower files do), but the product is an artifact, not a
component — stacked-ness as an invisible convention every use site
must know, init-side and apply-side vmaps kept in agreement by hand.
Attractiveness needs two ingredients: the transform's product must
re-enter the system as a first-class component (generic transforms
over declared roles, so the ensembled layer pipes, batches, trains and
ensembles again), and sizes must flow through composition (input
propagation, so the units size themselves from the wiring). nodejax
has both, which is what these tests jointly demonstrate.

The library has reached its goal when the transform spelling of a
primitive is the PREFERRED spelling: a linear layer is an ensemble of
hyperplanes, the unit defined once and the output axis added by a
transform — not a matrix written by hand. These tests hold that claim
to account: the ensemble IS the matrix layer (numerically), it trains
inside a one-line tower like anything else, and an mlp needs no matrix
written anywhere.
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import Node, trained, scan, Leaf, ensemble, batch, train_step
from nodejax.struct import Struct
from nodejax.examples.util import mse


def Hyperplane() -> Node:
    """ONE unit: w @ x + b -> scalar, its width read from the wiring.
    The whole definition — shape-generic, so the ensemble scales with
    whatever input it is bound to."""
    def param(node, rng):
        d = node.input.shape[-1]
        return Struct(w=jax.random.normal(rng.next(), (d,)) / jnp.sqrt(d),
                      b=0.0)

    def apply(param, input):
        return param.w @ input + param.b

    return Leaf(apply, param=param, name='unit')


def test_linear_is_an_ensemble_of_hyperplanes():
    """ensemble(Hyperplane) is a linear layer: params one row per unit,
    input broadcast, outputs stacked — and numerically the matrix. The
    SAME def sizes itself to whatever width it is bound at."""
    linear = ensemble(Hyperplane(), n=3)
    node = linear.with_input(jnp.zeros(4)).parameterize(rng=jax.random.PRNGKey(0))

    assert node.param.w.shape == (3, 4) and node.param.b.shape == (3,)

    x = jnp.array([1.0, -2.0, 0.5, 3.0])
    y = node.apply(x)
    assert y.shape == (3,)
    assert jnp.allclose(y, node.param.w @ x + node.param.b)

    wide = linear.with_input(jnp.zeros(7)).parameterize(rng=jax.random.PRNGKey(0))
    assert wide.param.w.shape == (3, 7)


def test_the_ensemble_layer_trains_like_any_layer():
    """The transform-built layer rides the rest of the algebra: batched,
    trained, one line — recovering a target linear map."""
    w_true = jnp.array([[1.0, -2.0, 0.5, 3.0],
                       [0.0, 1.0, -1.0, 2.0],
                       [2.0, 0.0, 1.0, -1.0]])

    model = batch(ensemble(Hyperplane(), n=3)).with_input(jnp.zeros((16, 4))).parameterize(
        rng=jax.random.PRNGKey(0)).initialize()
    trainer = train_step(model, mse, optax.adam(0.05))

    xs = jax.random.normal(jax.random.PRNGKey(1), (300, 16, 4))
    final, aux = trained(trainer).apply(input=xs, target=jnp.einsum('sbi,oi->sbo', xs, w_true))

    assert aux.loss[-1] < 1e-3 * aux.loss[0]
    from nodejax import PSNode
    assert isinstance(final, PSNode)                        # the scaffolding is struck
    assert jnp.allclose(final.param.w, w_true, atol=0.05)


def test_an_mlp_is_units_all_the_way_down():
    """The same single unit builds the whole network: the hidden layer
    is ensemble(Hyperplane), the readout is ONE Hyperplane, the mlp is
    their pipe — no layer anywhere written as a matrix. Resolved and
    parameterized per-sample; batch() then shares the params."""
    from nodejax import nn

    mlp = ensemble(Hyperplane(), n=16) >> nn.gelu >> Hyperplane()
    model = mlp.with_input(jnp.zeros(3)).parameterize(rng=jax.random.PRNGKey(0))

    trainer = train_step(batch(model).initialize(), mse, optax.adam(0.01))
    xs = jax.random.normal(jax.random.PRNGKey(1), (400, 32, 3))
    ys = jnp.sin(2.0 * xs[..., 0]) + 0.5 * xs[..., 1] * xs[..., 2]
    final, aux = trained(trainer).apply(input=xs, target=ys)

    assert aux.loss[-1] < 0.05 * aux.loss[0]      # a nonlinear fit, units only
