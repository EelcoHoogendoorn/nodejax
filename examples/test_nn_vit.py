"""Convolutional-stem Vision Transformer on 8x8 digit images using `nodejax.nn`.

Every module constant specifies an architectural width, head count, depth, or
image dimension rather than hand-computed intermediate feature sizes. Calling
`with_input` with one flattened image propagates intermediate shapes through
`serial`, inferring convolutional kernel input dimensions, positional embedding
token counts, and the flattened classification head fan-in during
`parameterize`.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import Node, trained, stack, batch, train_step, serial, nn
from examples.test_conv_vit import data, xent, accuracy

IMAGE = 8
STEM, HIDDEN, HEADS, DEPTH = 16, 32, 4, 2
BATCH, EPOCHS = 125, 40


def build() -> Node:
    """Builds a convolutional-stem Vision Transformer classifier for 8x8 digit images."""
    return serial(
        image=nn.Reshape((IMAGE, IMAGE, 1)),
        conv1=nn.Conv(STEM),
        act1=nn.gelu,
        conv2=nn.Conv(HIDDEN, stride=2),
        act2=nn.gelu,
        tokens=nn.tokens(),
        pos=nn.PosEmbed(),
        blocks=stack(nn.TransformerBlock(HIDDEN, heads=HEADS, ratio=4), n=DEPTH),
        flat=nn.flat,
        head=nn.Linear(10),
    )


def test_nn_assembly() -> None:
    """Verifies shape propagation from one example input and re-parameterization of a single `Node` at distinct input widths."""
    pipe = build()
    model = pipe.with_input(jnp.zeros(IMAGE * IMAGE)).parameterize(rng=jax.random.PRNGKey(0))

    assert model.param.conv1.kernel.shape == (3, 3, 1, STEM)
    assert model.param.conv2.kernel.shape == (3, 3, STEM, HIDDEN)
    grid = IMAGE // 2
    assert model.param.pos.embed.shape == (grid * grid, HIDDEN)
    wqkv = model.param.blocks.attn.attention.wqkv
    assert wqkv.shape == (DEPTH, HIDDEN, 3 * HIDDEN)
    assert not jnp.allclose(wqkv[0], wqkv[1])
    assert model.param.head.w.shape == (grid * grid * HIDDEN, 10)

    assert model.apply(jnp.zeros(IMAGE * IMAGE)).shape == (10,)
    batched = batch(pipe).bind(model.param)
    assert batched.apply(jnp.zeros((5, IMAGE * IMAGE))).shape == (5, 10)

    # Re-parameterizing the same unbound `Node` with different example inputs yields distinct weight shapes.
    small = nn.Linear(4) >> nn.gelu >> nn.Linear(2)
    a = small.with_input(jnp.zeros(7)).parameterize(rng=jax.random.PRNGKey(0))
    b = small.with_input(jnp.zeros(3)).parameterize(rng=jax.random.PRNGKey(0))
    assert a.param.linear.w.shape == (7, 4) and b.param.linear.w.shape == (3, 4)


def test_nn_vit_trains_on_real_digits() -> None:
    """Verifies that the batched convolutional Vision Transformer reaches above 85% held-out accuracy on 8x8 digits."""
    X_train, y_train, X_test, y_test = data()
    # Binding a single-sample input spec before `batch` resolves all inner layer shapes.
    pipe = batch(build().with_input(jnp.zeros(IMAGE * IMAGE)))

    shuffle = np.random.RandomState(1)
    batch_indices = np.concatenate(
        [shuffle.permutation(len(X_train)) for _ in range(EPOCHS)]
    ).reshape(-1, BATCH)

    trainer = train_step(
        pipe.parameterize(rng=jax.random.PRNGKey(0)).initialize(),
        xent, optax.adam(1e-3))
    done, aux = trained(trainer).apply(input=X_train[batch_indices],
                                       target=y_train[batch_indices])

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < 0.3 * aux.loss[0]
    test_accuracy = accuracy(done(X_test)[1], y_test)
    assert test_accuracy > 0.85, test_accuracy
    print(f"\n[nn-vit] loss {aux.loss[0]:.3f} -> {aux.loss[-1]:.3f} | "
          f"TEST acc {test_accuracy:.3f}")
