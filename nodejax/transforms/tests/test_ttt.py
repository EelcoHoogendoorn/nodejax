"""ttt: the wrapped node's params as gradient-adapted state."""

import jax.numpy as jnp

from nodejax import node_def, ttt, scan
from nodejax.struct import Struct
from nodejax.examples import mse, tile


def g_def():
    def param(scale=0.0) -> Struct:
        return Struct(scale=jnp.asarray(scale))

    def apply(param, input):
        return param.scale * input

    return node_def(apply, param=param, name='g')


def sample(x):
    """Reconstruction self-supervision, assembled as data: the target
    IS the input."""
    return Struct(input=jnp.asarray(x), target=jnp.asarray(x))


def test_ttt_adapts_toward_self_supervision():
    """Reconstruction drives the wrapped gain toward identity, one
    step per sample, weights carried as state."""
    node = ttt(g_def(), mse, 0.1).parameterize()
    assert node.param.init.scale == 0.0
    assert node.param.lr.scale == 0.1                    # per-leaf learned rates

    state = node.init()
    scales, outs = [], []
    for _ in range(50):
        state, out = node.apply(state, sample(1.0))
        scales.append(float(state.w.scale))
        outs.append(float(out))
    assert outs[0] == 0.0                                # first prediction: untrained
    assert scales[0] > 0.0                               # ...and the update landed
    assert all(b >= a for a, b in zip(scales, scales[1:]))
    assert abs(scales[-1] - 1.0) < 1e-2                  # converged to reconstruction


def test_ttt_predicts_then_updates():
    """Prequential order: the output comes from the incoming weights
    — a prediction those weights never trained on — and the update
    lands after, ready for the next step."""
    node = ttt(g_def(), mse, 0.5).parameterize()
    state, out = node.apply(node.init(), sample(1.0))
    assert jnp.allclose(out, 0.0)                        # predicted at scale 0
    assert jnp.allclose(state.w.scale, 1.0)              # -lr * d/ds (s-1)^2 at s=0


def test_ttt_under_scan_resets_per_sequence():
    """scan internalizes the adapted weights: each sequence starts
    from the meta-init — adaptation is per sequence, carried within."""
    rollout = scan(ttt(g_def(), mse, 0.1)).parameterize()
    stream = tile(sample(1.0), 20)
    ys1 = rollout.apply(stream)
    ys2 = rollout.apply(stream)
    assert jnp.allclose(ys1, ys2)                        # fresh start both times
    assert ys1[-1] > ys1[0]                              # adapted within the sequence
