"""Transformer primitives under ordinary NodeJAX binding and transforms."""

import jax
import jax.numpy as jnp

from nodejax import batch, nn, stack


KEY = jax.random.PRNGKey(0)


def test_rms_norm_infers_width_and_matches_definition() -> None:
    input = jnp.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    norm = nn.RMSNorm(eps=1e-6).with_input(input).parameterize()

    assert norm.param.scale.shape == (3,)
    expected = input * jax.lax.rsqrt(
        jnp.mean(jnp.square(input), axis=-1, keepdims=True) + 1e-6)
    assert jnp.allclose(norm.apply(input), expected)


def test_l2_norm_is_pure_and_normalizes_last_axis() -> None:
    input = jnp.array([[3.0, 4.0], [5.0, 12.0]])
    norm = nn.L2Norm()

    assert not norm.parametric
    assert not norm.cyclic
    assert jnp.allclose(jnp.linalg.norm(norm.apply(input), axis=-1), 1.0)


def test_causal_attention_blocks_future_tokens() -> None:
    input = jnp.arange(24.0, dtype=jnp.float32).reshape(4, 6) / 24.0
    changed = input.at[-1].set(jnp.full(6, 20.0))
    attention = nn.Attention(heads=2, causal=True).with_input(
        input).parameterize(rng=KEY)

    output = attention.apply(input)
    changed_output = attention.apply(changed)

    assert output.shape == input.shape
    assert jnp.allclose(output[:-1], changed_output[:-1])
    assert not jnp.allclose(output[-1], changed_output[-1])


def test_attention_batches_and_differentiates() -> None:
    input = jnp.arange(48.0, dtype=jnp.float32).reshape(2, 4, 6) / 48.0
    attention = batch(nn.Attention(heads=2, causal=True)).with_input(
        input).parameterize(rng=KEY)

    output = jax.jit(attention.apply)(input)
    assert output.shape == input.shape

    def loss(model) -> jax.Array:
        return jnp.sum(model.apply(input) ** 2)

    gradients = jax.grad(loss)(attention)
    assert all(jnp.all(jnp.isfinite(value))
               for value in jax.tree.leaves(gradients.param))


def test_swiglu_infers_fan_in_and_returns_requested_width() -> None:
    input = jnp.arange(15.0, dtype=jnp.float32).reshape(5, 3) / 15.0
    feed_forward = nn.SwiGLU(width=4, ratio=3).with_input(
        input).parameterize(rng=KEY)

    assert feed_forward.param.gate_weight.shape == (3, 12)
    assert feed_forward.param.value_weight.shape == (3, 12)
    assert feed_forward.param.output_weight.shape == (12, 4)
    assert feed_forward.apply(input).shape == (5, 4)


def test_causal_block_stacks_without_call_site_masking() -> None:
    input = jnp.ones((5, 8))
    model = stack(
        nn.TransformerBlock(8, heads=2, ratio=2, causal=True), n=3,
    ).with_input(input).parameterize(rng=KEY)

    output = jax.jit(model.apply)(input)
    assert output.shape == input.shape
