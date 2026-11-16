"""Recursive U-Net assembled with `Composite` member wiring.

Each `UNetLevel` encodes an `(H, W, C)` feature map, downsamples spatially,
delegates to a nested `inner` level or bottleneck, upsamples the returned map,
and concatenates it with the encoder activation held in a local Python variable
before decoding:

    UNetLevel(c_outer, c_inner, inner)
        enc:   c_outer -> c_inner
        inner: nested UNetLevel or bottleneck
        dec:   2 * c_inner -> c_outer
"""

import jax
import jax.numpy as jnp
import optax

from nodejax import node, trained, Node, Composite, batch, train_step, nn, tile


@node
def UNetLevel(c_outer: int, c_inner: int, inner: Node) -> Node:
    """Encodes, downsamples, applies `inner`, upsamples, and decodes concatenated `(H, W, C)` skip features."""
    members = Composite(enc=nn.Conv(c_inner, kernel=3) >> nn.relu,
                        down=nn.Downsample(),
                        inner=inner,
                        up=nn.Upsample(method='bilinear'),
                        dec=nn.Conv(c_outer, kernel=3) >> nn.relu)

    def apply(self, input: jax.Array) -> jax.Array:
        # Local `e` retains the full-resolution encoder map for the decoder skip connection.
        e = self.enc(input)
        d = self.inner(self.down(e))
        u = self.up(d)
        return self.dec(jnp.concatenate([e, u], axis=-1))

    return members(apply, name=f'level_{c_outer}_{c_inner}')


def UNet(channels: list[int]) -> Node:
    """Constructs a recursive U-Net from outer-to-inner image channel widths `channels`."""
    if len(channels) < 2:
        raise ValueError('U-Net needs at least 2 channel specs (input/output and bottleneck)')

    current = nn.Conv(channels[-1], kernel=3) >> nn.relu

    for i in reversed(range(len(channels) - 1)):
        c_outer = channels[i]
        c_inner = channels[i + 1]
        current = UNetLevel(c_outer, c_inner, current)

    return current


def test_unet_shapes_and_forward_pass() -> None:
    """Verifies parameter hierarchy, shape propagation, and output shape on a `(16, 16, 1)` image."""
    model = UNet([1, 8, 16]).with_input(jnp.zeros((16, 16, 1)))
    rng = jax.random.PRNGKey(0)

    bound = model.parameterize(rng=rng)

    assert 'enc' in bound.param
    assert 'inner' in bound.param
    assert 'dec' in bound.param

    x = jnp.ones((16, 16, 1))
    out = bound.apply(x)
    assert out.shape == (16, 16, 1)
    assert jnp.all(jnp.isfinite(out))


def test_batched_unet_training() -> None:
    """Verifies that a batched `UNet` updates parameters with finite losses under `train_step`."""
    model = UNet([1, 4, 8]).with_input(jnp.zeros((16, 16, 1)))
    batched_model = batch(model)

    def mse(pred, target):
        return jnp.mean((pred - target) ** 2)

    initial = batched_model.parameterize(rng=jax.random.PRNGKey(42)).initialize()
    trainer = train_step(initial, mse, optax.adam(0.01))

    images = jnp.ones((4, 16, 16, 1))
    targets = images * 2.0
    steps = 3

    done, aux = trained(trainer).apply(input=tile(images, steps),
                                       target=tile(targets, steps))

    assert aux.loss.shape == (steps,)
    assert jnp.all(jnp.isfinite(aux.loss))
    assert not jax.tree.all(jax.tree.map(jnp.array_equal, done.param, initial.param))
