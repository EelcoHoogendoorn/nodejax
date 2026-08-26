"""Recursive U-Net example using def-level composite wiring.

The classic U-Net architecture demonstrates how self-style composite wiring
simplifies multi-resolution skip connections:

- Downward path (encoder): conv, downsample, recurse into inner level
- Upward path (decoder): upsample inner output, concatenate encoder skip feature, conv
- The skip connection is simply a Python local (`e = self.enc(input)`)!

Structure:
  UNetLevel(c_outer, c_inner, inner):
    enc: c_outer -> c_inner
    inner: recursive level or bottleneck
    dec: 2 * c_inner -> c_outer
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import node, trained, Node, Composite, batch, train_step, nn
from nodejax.struct import Struct
from nodejax import tile

@node
def UNetLevel(c_outer: int, c_inner: int, inner: Node) -> Node:
    """One level of the recursive U-Net."""
    members = Composite(enc=nn.Conv(c_inner, kernel=3) >> nn.relu,
                        down=nn.Downsample(),
                        inner=inner,
                        up=nn.Upsample(method='bilinear'),
                        dec=nn.Conv(c_outer, kernel=3) >> nn.relu)

    def apply(self, input: jax.Array) -> jax.Array:
        e = self.enc(input)                                 # the skip connection: a Python local!
        d = self.inner(self.down(e))
        u = self.up(d)
        return self.dec(jnp.concatenate([e, u], axis=-1))

    return members(apply, name=f'level_{c_outer}_{c_inner}')


def UNet(channels: list[int]) -> Node:
    """Build a U-Net recursively from a list of channel depths.
    
    e.g. channels = [1, 16, 32] builds:
    - Bottleneck at channel depth 32
    - Outer level connecting 16 <-> 32
    - Top level connecting 1 <-> 16
    """
    if len(channels) < 2:
        raise ValueError('U-Net needs at least 2 channel specs (input/output and bottleneck)')
    
    # Bottleneck at innermost level
    current = nn.Conv(channels[-1], kernel=3) >> nn.relu
    
    # Wrap levels from inside out
    for i in reversed(range(len(channels) - 1)):
        c_outer = channels[i]
        c_inner = channels[i + 1]
        current = UNetLevel(c_outer, c_inner, current)
        
    return current


# --- Tests & Verification ---

def test_unet_shapes_and_forward_pass():
    """Verify recursive U-Net parameterization, shape propagation, and forward pass."""
    model = UNet([1, 8, 16]).with_input(jnp.zeros((16, 16, 1)))
    rng = jax.random.PRNGKey(0)
    
    # Parameterize model with random weights
    bound = model.parameterize(rng=rng)
    
    # Check structure of params
    assert 'enc' in bound.param
    assert 'inner' in bound.param
    assert 'dec' in bound.param
    
    # Single sample forward pass: 16x16 image with 1 channel -> output is 16x16 with 1 channel
    x = jnp.ones((16, 16, 1))
    out = bound.apply(x)
    assert out.shape == (16, 16, 1)
    assert jnp.all(jnp.isfinite(out))


def test_batched_unet_training():
    """Verify batched U-Net with train_step for image-to-image reconstruction."""
    model = UNet([1, 4, 8]).with_input(jnp.zeros((16, 16, 1)))
    batched_model = batch(model)
    
    def mse(pred, target):
        return jnp.mean((pred - target) ** 2)

    trainer = train_step(
        batched_model.parameterize(rng=jax.random.PRNGKey(42)).initialize(),
        mse, optax.adam(0.01))

    # a batch of 4 images, the same one every step
    images = jnp.ones((4, 16, 16, 1))
    targets = images * 2.0
    steps = 5

    _, aux = trained(trainer).apply(input=tile(images, steps),
                                    target=tile(targets, steps))

    assert jnp.all(jnp.isfinite(aux.loss))
    assert aux.loss[-1] < aux.loss[0]        # and it is actually training
