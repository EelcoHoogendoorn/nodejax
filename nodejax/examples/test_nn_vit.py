"""The conv-transformer on digits, built from nodejax.nn.

The statics exhibit: every constant below is a DESIGN decision (stem
width, hidden width, heads, depth, data geometry) — no GRID, no
TOKENS, no conv-flatten arithmetic, no in-features anywhere. All
in-shapes derive at parameterize from one example input, threaded
member by member; the head's fan-in (the classic hand-computed
tokens * hidden) is inferred like every other.

Same architecture and assertions as test_conv_vit, whose docstring
names the hand-computed sizes as the measure of whether late-bound
shapes deserve building. This file is the answer.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax.struct import Struct
from nodejax import stack, batch, train_step, serial, node_def, nn
from nodejax.examples.test_conv_vit import data, xent, accuracy

IMAGE = 8                       # data geometry
STEM, HIDDEN, HEADS, DEPTH = 16, 32, 4, 2   # design decisions
BATCH, EPOCHS = 125, 40


def build():
    return serial(
        image=node_def(lambda input: input.reshape(IMAGE, IMAGE, 1), name='image'),
        conv1=nn.Conv(STEM),
        act1=nn.gelu,
        conv2=nn.Conv(HIDDEN, stride=2),
        act2=nn.gelu,
        tokens=nn.tokens(),
        pos=nn.PosEmbed(),
        blocks=stack(nn.Block(HIDDEN, heads=HEADS, ratio=4), n=DEPTH),
        flat=nn.flat,
        head=nn.Linear(10),
    )


def test_nn_assembly():
    """One key and one example: every in-shape inferred, including the
    conv-flatten head width; the same def re-parameterizes at other
    widths, shapes living in the params alone."""
    pipe = build()
    model = pipe.with_input(jnp.zeros(IMAGE * IMAGE)).parameterize(rng=jax.random.PRNGKey(0))

    assert model.param.conv1.kernel.shape == (3, 3, 1, STEM)
    assert model.param.conv2.kernel.shape == (3, 3, STEM, HIDDEN)
    grid = IMAGE // 2
    assert model.param.pos.embed.shape == (grid * grid, HIDDEN)
    wqkv = model.param.blocks.attn.attn.wqkv
    assert wqkv.shape == (DEPTH, HIDDEN, 3 * HIDDEN)
    assert not jnp.allclose(wqkv[0], wqkv[1])
    assert model.param.head.w.shape == (grid * grid * HIDDEN, 10)

    assert model.apply(jnp.zeros(IMAGE * IMAGE)).shape == (10,)
    batched = batch(pipe).bind(model.param)
    assert batched.apply(jnp.zeros((5, IMAGE * IMAGE))).shape == (5, 10)

    # def reuse across widths: same pipe, different example
    small = nn.Linear(4) >> nn.gelu >> nn.Linear(2)
    a = small.with_input(jnp.zeros(7)).parameterize(rng=jax.random.PRNGKey(0))
    b = small.with_input(jnp.zeros(3)).parameterize(rng=jax.random.PRNGKey(0))
    assert a.param.linear.w.shape == (7, 4) and b.param.linear.w.shape == (3, 4)


def test_nn_vit_trains_on_real_digits():
    X_train, y_train, X_test, y_test = data()
    pipe = batch(build())
    model = pipe.with_input(jnp.zeros(IMAGE * IMAGE)).parameterize(rng=jax.random.PRNGKey(0))

    shuffle = np.random.RandomState(1)
    batch_indices = np.concatenate(
        [shuffle.permutation(len(X_train)) for _ in range(EPOCHS)]
    ).reshape(-1, BATCH)

    trainer = train_step(pipe, xent, optax.adam(1e-3))
    final, losses = trainer.scan(
        trainer.init(model=model.param),
        Struct(input=X_train[batch_indices], target=y_train[batch_indices]))

    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < 0.3 * losses[0]
    test_accuracy = accuracy(pipe.apply(final.model, X_test), y_test)
    assert test_accuracy > 0.85, test_accuracy
    print(f"\n[nn-vit] loss {losses[0]:.3f} -> {losses[-1]:.3f} | "
          f"TEST acc {test_accuracy:.3f}")
