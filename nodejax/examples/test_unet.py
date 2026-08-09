"""Recursive U-Net example using def-level composite wiring.

The classic U-Net architecture demonstrates how self-style composite wiring
simplifies multi-resolution skip connections:

- Downward path (encoder): conv, pool, recurse into inner level
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

from nodejax import NodeDef, composite, batch, train_step, nn
from nodejax.struct import Struct


def conv(c_out: int, kernel: int = 3) -> NodeDef:
    """Convolution + ReLU block using stock nn.Conv."""
    return nn.Conv(c_out, kernel=kernel) >> nn.relu


def pool(x: jax.Array) -> jax.Array:
    """2x2 spatial downsampling via nearest resize."""
    h, w, c = x.shape
    return jax.image.resize(x, (h // 2, w // 2, c), method='nearest')


def upsample(x: jax.Array, target_hw: tuple[int, int]) -> jax.Array:
    """Bilinear upsampling back to target spatial shape."""
    h, w = target_hw
    return jax.image.resize(x, (h, w, x.shape[-1]), method='bilinear')


def UNetLevel(c_outer: int, c_inner: int, inner: NodeDef) -> NodeDef:
    """One level of the recursive U-Net."""
    members = dict(
        enc=conv(c_inner),
        inner=inner,
        dec=conv(c_outer)
    )

    def apply(self, input: jax.Array) -> jax.Array:
        e = self.enc(input)                                 # the skip connection: a Python local!
        d = self.inner(pool(e))
        u = upsample(d, e.shape[:2])
        return self.dec(jnp.concatenate([e, u], axis=-1))

    return composite(apply, members=members, name=f'level_{c_outer}_{c_inner}')


def UNet(channels: list[int]) -> NodeDef:
    """Build a U-Net recursively from a list of channel depths.
    
    e.g. channels = [1, 16, 32] builds:
    - Bottleneck at channel depth 32
    - Outer level connecting 16 <-> 32
    - Top level connecting 1 <-> 16
    """
    if len(channels) < 2:
        raise ValueError('U-Net needs at least 2 channel specs (input/output and bottleneck)')
    
    # Bottleneck at innermost level
    current = conv(channels[-1])
    
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

    trainer = train_step(batched_model, mse, optax.adam(0.01))
    bound = batched_model.parameterize(rng=jax.random.PRNGKey(42))
    state = trainer.init(model=bound.param)
    
    # Create dummy batch of 4 images (4, 16, 16, 1)
    dummy_input = jnp.ones((4, 16, 16, 1))
    dummy_target = jnp.ones((4, 16, 16, 1)) * 2.0
    
    # Single optimization step
    new_state, loss = trainer.apply(state, Struct(input=dummy_input, target=dummy_target))
    
    assert jnp.isfinite(loss)
    assert loss > 0.0
