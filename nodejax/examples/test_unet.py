"""Recursive U-Net example using def-level composite wiring.

The classic U-Net architecture demonstrates how self-style composite wiring
simplifies multi-resolution skip connections:

- Downward path (encoder): conv, pool, recurse into inner level
- Upward path (decoder): upsample inner output, concatenate encoder skip feature, conv
- The skip connection is simply a Python local (`e = self.enc(input)`)!

Structure:
  level_def(c_outer, c_inner, inner):
    enc: c_outer -> c_inner
    inner: recursive level or bottleneck
    dec: 2 * c_inner -> c_outer
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nodejax import NodeDef, node_def, composite, batch, train_step, KeyStream
from nodejax.struct import Struct


def conv_def(c_in: int, c_out: int, kernel: int = 3) -> NodeDef:
    """3x3 SAME convolution over ONE (H, W, C) feature map."""
    def param(rng: KeyStream) -> Struct:
        k = jax.random.normal(rng.next(), (kernel, kernel, c_in, c_out))
        return Struct(kernel=k / jnp.sqrt(kernel * kernel * c_in), bias=jnp.zeros(c_out))

    def apply(param: Struct, input: jax.Array) -> jax.Array:
        out = jax.lax.conv_general_dilated(
            input[None], param.kernel, window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))[0]
        return jax.nn.relu(out + param.bias)

    return node_def(apply, param=param, name=f'conv_{c_in}_{c_out}')


def pool(x: jax.Array) -> jax.Array:
    """2x2 spatial downsampling via nearest resize."""
    h, w, c = x.shape
    return jax.image.resize(x, (h // 2, w // 2, c), method='nearest')


def upsample(x: jax.Array, target_hw: tuple[int, int]) -> jax.Array:
    """Bilinear upsampling back to target spatial shape."""
    h, w = target_hw
    return jax.image.resize(x, (h, w, x.shape[-1]), method='bilinear')


def level_def(c_outer: int, c_inner: int, inner: NodeDef) -> NodeDef:
    """One level of the recursive U-Net."""
    members = dict(
        enc=conv_def(c_outer, c_inner),
        inner=inner,
        dec=conv_def(2 * c_inner, c_outer)
    )

    def apply(self, input: jax.Array) -> jax.Array:
        e = self.enc(input)                                 # the skip connection: a Python local!
        d = self.inner(pool(e))
        u = upsample(d, e.shape[:2])
        return self.dec(jnp.concatenate([e, u], axis=-1))

    return composite(apply, members=members, name=f'level_{c_outer}_{c_inner}')


def unet_def(channels: list[int]) -> NodeDef:
    """Build a U-Net recursively from a list of channel depths.
    
    e.g. channels = [1, 16, 32] builds:
    - Bottleneck at channel depth 32
    - Outer level connecting 16 <-> 32
    - Top level connecting 1 <-> 16
    """
    if len(channels) < 2:
        raise ValueError('UNet needs at least 2 channel specs (input/output and bottleneck)')
    
    # Bottleneck at innermost level
    current = conv_def(channels[-1], channels[-1])
    
    # Wrap levels from inside out
    for i in reversed(range(len(channels) - 1)):
        c_outer = channels[i]
        c_inner = channels[i + 1]
        current = level_def(c_outer, c_inner, current)
        
    return current


# --- Tests & Verification ---

def test_unet_shapes_and_forward_pass():
    """Verify recursive UNet parameterization, shape propagation, and forward pass."""
    model = unet_def([1, 8, 16])
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
    model_def = unet_def([1, 4, 8])
    batched_model = batch(model_def)
    
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
